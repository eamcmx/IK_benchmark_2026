function [q, info] = solve_ScrewTheory(robot, T_d, q0, opts) %#ok<INUSD> opts: closed form, no option applies
% SOLVE_SCREWTHEORY  Closed-form screw-theory IK of Chen, Zhu and Zhang (2015): product of exponentials,
%                    the new three-axis sub-problem for joints 1-3 and Paden-Kahan sub-problems 2 and 1
%                    for joints 4-6, with the eight solutions enumerated and one selected.
%
%   Source:
%     Qingcheng Chen, Shiqiang Zhu and Xuequn Zhang (2015). Improved Inverse Kinematics Algorithm Using
%     Screw Theory for a Six-DOF Robot Manipulator. International Journal of Advanced Robotic Systems,
%     12:140 (vol. 12, article 140, 9 pages; no issue number printed). DOI: 10.5772/60834
%   Page numbers are PDF pages 1-9 (PDF page = printed page).
%
%   Algorithm (Secs. 2-4, pp.2-6):
%     Model    Twists xi_i = [w_i; r_i x w_i] (Eq. 2, p.3) from the joint axes w_i and axis points r_i
%              in the base frame; the twist exponential of p.3 (unnumbered display);
%              g_st(theta) = e^(xi_1 theta_1) ... e^(xi_6 theta_6) g_st(0) (Eqs. 3, 17).
%     Step 1   (Eqs. 4-14, 18; pp.3-5) The wrist point q_a, where axes 4, 5, 6 meet, moves only with
%              joints 1-3: e^(xi_1 theta_1) e^(xi_2 theta_2) e^(xi_3 theta_3) q_a = q (Eq. 4).
%              q_c = [0; +/-sqrt(qx^2 + qy^2); qz] (Eq. 8, from Eqs. 5-7 with q1, q2, w2 of Eq. 6);
%              theta_1 = atan2(-(+/-rho) qx, (+/-rho) qy), rho = sqrt(qx^2 + qy^2) (Eq. 10, PK-1);
%              u = q_c - r2, v = q_a - r3, r23 = r2 - r3, r32 = r3 - r2 (Eq. 11);
%              theta_01 = atan2(w^T(u x r32), u^T r32), cos(phi_1) from Eq. 12, theta_2 (Eq. 13);
%              theta_02 = atan2(w^T(v x r23), v^T r23), phi_2, theta_3 (Eq. 14): the two intersections
%              of two circles in the plane normal to the parallel axes 2 and 3 (Fig. 3, p.4).
%     Step 2   (Eqs. 19-24, pp.5-6) g1 = e^(-xi_3 theta_3) e^(-xi_2 theta_2) e^(-xi_1 theta_1) g_st(theta)
%              g_st(0)^-1 (Eq. 19); q6 on axis 6, off axes 4 and 5 (Eq. 20); PK-2 (Eqs. 21-22):
%              u = q6 - q_w, v = g1 q6 - q_w,
%              alpha = ((w4'w5)(w5'u) - w4'v) / ((w4'w5)^5 - 1),
%              beta  = ((w4'w5)(w5'v) - w5'u) / ((w4'w5)^5 - 1),
%              gamma = +/-sqrt((|u|^2 - alpha^2 - beta^2 - 2 alpha beta (w4'w5)) / |w4 x w5|),
%              c = alpha w4 + beta w5 + gamma (w4 x w5) + q_w; then e^(xi_5 theta_5) q6 = c and
%              e^(xi_4 theta_4) c = g1 q6 (Eq. 23), each by PK-1, theta = atan2(w^T(u' x v'), u'^T v')
%              (Eq. 24).
%     Step 3   (Eqs. 25-28, p.6) g2 = e^(-xi_5 theta_5) e^(-xi_4 theta_4) g1 (Eq. 25); q_t off axis 6
%              (Eq. 26); e^(xi_6 theta_6) q_t = g2 q_t (Eq. 27); theta_6 by PK-1 (Eq. 28).
%     Solutions  2 (sign of q_c, Eq. 8) x 2 (circle intersections, Eqs. 13-14) x 2 (sign of gamma,
%              Eq. 22) = 8 (Table 1, p.7). A branch whose circles do not intersect has no solution (p.4).
%     Selection  "a series of position bounds for every joint need be set" (p.7): solutions outside
%              [q_min, q_max] are discarded. Among the rest, the minimum joint-displacement method
%              (p.7): the solution nearest q0.
%
%   Robot class (Sec. 4.1, p.4): six revolute joints; "the last three consecutive joints' axes
%   intersect at a point"; joints 1-3 satisfy the new sub-problem (Sec. 3, p.3: axes 2 and 3
%   parallel; Eqs. 6-8: axis 1 normal to them, and q_a in the plane through axis 1 normal to them).
%   Checked from the robot's own axes at the reference configuration (st_build):
%     Robot        n   Applicable  Reason
%     IRB120       6   yes         wrist axes meet at q_w; axes 2, 3 parallel; axis 1 normal to them;
%                                  q_w in the plane of axis 1 normal to them. In the base frame turned
%                                  by -90 deg about axis 1 (x = base -y, y = base x), the IRB120 at
%                                  q_home is the robot of Fig. 4(b) with a1 = 0, d1 = 0.290,
%                                  a2 = 0.270, a3 = 0.070, d4 = 0.302 m:
%                                  w = [z, -x, -x, y, -x, -z] against Eq. 15's [z, x, x, y, x, z]
%                                  (joints 2, 3, 5, 6 turn the other way), and the flange sits
%                                  0.072 m beyond q_w along axis 6.
%     UR5          6   no          axes 4, 5, 6 do not meet (offset wrist: axes 4 and 6 lie
%                                  d5 = 0.09465 m apart), so no wrist point q_a exists (Sec. 4.1).
%     Panda        7   no          seven joints; the method is for six-DOF robots (title; Sec. 4).
%     Synthetic9   9   no          as Panda.
%     Synthetic11  11  no          as Panda.
%     Synthetic13  13  no          as Panda.
%
%   Parameters:
%     None. The paper's only parameters are the structure parameters of its robot (Eq. 30, p.6);
%     here they come from the robot model (DH table) at the reference configuration. Reference points:
%     q1 = [0;0;0], q2 = [0;0;1] on axis 1 (Eq. 6); q6 = [0; a1 + d4; 0] (Eq. 20);
%     q_t = [0; a1; d1 + a2 + a3] (Eq. 26), in the frame of Fig. 4(b) (see Benchmark interface).
%     Numerical tolerances are not stated in the paper (checked pages 3-7). The benchmark uses 1e-9
%     (m and unit-vector components) for the geometric checks of the robot class, and treats an
%     arccos argument within 1e-12 of +/-1, or gamma^2 above -1e-12 |u|^2, as tangency (one solution,
%     p.4); beyond that the branch has no solution.
%
%   Benchmark interface:
%     Budget       No robot.fk and no robot.jac call: 0 FK evaluations per waypoint on every robot.
%                  The closed form uses no FK check (none in the paper). The PoE model (twists,
%                  g_st(0); Eqs. 15-16 in the paper) is built on every call from robot.dh at the
%                  reference configuration; it does not depend on T_d and is not counted as an FK
%                  evaluation. opts.max_iter and opts.budget play no role.
%     Reference    g_st(0) is the pose at robot.q_home and theta_i = q_i - q_home_i. On the IRB120,
%                  q_home = [0 0 0 0 90 0] deg has the layout of Fig. 4(b) (arm upright, forearm
%                  horizontal, axis 6 vertical). At its joint zero axes 4 and 6 coincide, so no point
%                  on axis 6 off axis 4 exists for Eq. 20.
%     Frame        The computation runs in the base frame turned about axis 1 (origin on axis 1,
%                  z along w1, x along +/-w2, q_w at y >= 0): the frame of Fig. 4(b), in which
%                  Eqs. 6, 8, 10, 18, 20 and 26 hold as printed. A rigid change of coordinates; the
%                  joint angles do not depend on it.
%     Units        Metres and radians; a closed form has no unit-dependent weighting.
%     Warm start   q0 enters through the selection only (nearest q0). If no solution is real and
%                  inside the limits, q0 is returned with info.converged = false.
%     Joint limits Each solution is mapped to joint values q = theta + q_home, wrapped to (-pi, pi]
%                  (the angle range of Sec. 5, p.6), and discarded if outside [q_min, q_max] (p.7).
%                  No clamping.
%     Output       info.solver, info.iters (1 closed-form pass; 0 if not applicable), info.time,
%                  info.converged (an admissible solution exists), info.fitness (joint displacement
%                  |q - q0|_2 of the returned solution, the selection criterion; NaN if none),
%                  info.not_applicable, info.reason, info.fk_calls = info.jac_calls = 0,
%                  info.Q_all (6 x 8 joint values, NaN where no solution), info.real,
%                  info.in_limits, info.branch (index of the returned solution, 0 if none).
%                  solve_ScrewTheory('ops') returns handles to the steps for unit tests.
%
%   Deviations from the paper:
%     - Robot class wording: Sec. 3 (p.3) describes the three axes of the new sub-problem as
%       "non-intersecting" and the two parallel axes as "non co-planar to the third". On the IRB120
%       axis 1 meets axes 2 and 3 (a1 = 0). Eqs. 4-18 contain a1 only as a coordinate and hold
%       unchanged for a1 = 0, so they are applied as printed; every branch reproduces the pose
%       (tested on random IRB120 poses).
%     - Reference configuration q_home and the frame of Fig. 4(b) (see Benchmark interface).
%     - Eq. 18 sets q to the position of g_st(theta), since the tool frame of Eq. 16 sits at q_a. The
%       IRB120 flange lies 0.072 m beyond q_a, so q = g_st(theta) g_st(0)^-1 q_a, the image of q_a
%       under Eq. 3, which equals Eq. 18's q when the tool frame sits at q_a.
%     - Eqs. 19 and 25 print g_st(theta) without g_st(0)^-1, which contradicts Eq. 17 unless
%       g_st(0) = I (Eq. 16 has a translation). g_st(theta) g_st(0)^-1 is used; it reproduces
%       Table 1, and the printed form does not (tested).
%     - Eq. 13 prints theta_2 = phi_1 +/- theta_01 and Eq. 14 theta_3 = theta_02 +/- phi_2. With
%       theta_01 and theta_02 as printed, the two circle intersections give
%       (theta_2, theta_3) = (phi_1 - theta_01, theta_02 + phi_2) and (-phi_1 - theta_01,
%       theta_02 - phi_2); phi_1 + theta_01 is the negative of the second theta_2 and fails Eq. 4
%       unless it is 0 or pi. Table 1 lists theta_2 = -1.95792331 deg for solutions 5-6, where
%       phi_1 + theta_01 = +1.958 deg (tested). Eq. 13 uses w = w2 and Eq. 14 w = w3.
%     - u' and v' of Eqs. 24 and 28 are not defined in the paper; they are the components of u and v
%       normal to w, the reading under which Eq. 23 holds. The printed second lines for theta_5
%       (Eq. 24) and theta_6 (Eq. 28) agree with it; the printed second line for theta_4 adds
%       (q_y6 - a1 - d4)^2 = alpha^2 to the cosine term (the unprojected u^T v) and violates Eq. 23
%       whenever alpha ~= 0 (tested), so the first line is used.
%     - Eq. 15 prints xi_5 = [1 0 0 a1+d4 0 0] and xi_6 = [0 0 1 0 a2+a3+d1 -a1-d4], which
%       contradict Eq. 2 with the w and r printed beneath Fig. 4 and do not reproduce Eq. 31; the
%       twists are built by Eq. 2.
%     - Eq. 22 prints the exponent 5 in (w4'w5)^5 - 1 and |w4 x w5| unsquared under the root; both
%       are applied as printed. On the IRB120 w4'w5 = 0 and |w4 x w5| = 1, so neither affects the
%       result.
%     - Selection: the paper names the minimum energy method or the minimum joint-displacement
%       method (p.7) and defines neither. Minimum joint displacement is used, as the Euclidean
%       distance to q0 (benchmark rule 4); ties go to the lower branch index.
%     - Joint 6 of the IRB120 has limits of +/-400 deg; only the paper's eight solutions in
%       (-pi, pi] are enumerated, not their 2*pi equivalents.
%     - Eq. 30 prints d4 = 594 mm for the paper's example, while Eq. 31 (p_z = 1400 mm) and
%       Table 1 correspond to d4 = 600 mm (tested). This concerns only the paper's example.
%
% ==========================================================================
% IK_benchmark_2026: A Comparative Benchmark of Inverse Kinematics Solvers for
%          Serial and Redundant Manipulators (6 to 13 DOF) under Warm-Started
%          Recursive Trajectory Tracking
%
% Authors   : Emmanuel A. Merchan-Cruz (Transport and Telecommunication
%             Institute, Riga, Latvia)
%             Gerardo Aragon-Camarasa (School of Computing Science,
%             University of Glasgow, Glasgow, U.K.)
%             Ekaterina Gromova (Transport and Telecommunication Institute,
%             Riga, Latvia)
% Year      : 2026
% License   : MIT (see LICENSE)
% Repository: https://github.com/eamcmx/IK_benchmark_2026
% ==========================================================================

    if ischar(robot) && strcmp(robot, 'ops')
        q = struct('model', @st_model, 'build', @st_build, 'twist', @st_twist, ...
            'expo', @st_exp, 'poe', @st_poe, 'eq8_10', @st_eq8_10, 'eq11_14', @st_eq11_14, ...
            'step2', @st_step2, 'step3', @st_step3, 'pk1', @st_pk1, 'solve', @st_solve, ...
            'joints', @st_joints, 'select', @st_select, 'wrap', @st_wrap, 'tol', st_tol());
        info = [];
        return
    end

    t_start = tic;
    q0 = q0(:);
    info = struct();
    info.solver    = 'Screw-Theory-PoE';
    info.fk_calls  = 0;
    info.jac_calls = 0;

    M = st_model(robot);
    if ~M.ok
        q = q0;
        info.iters          = 0;
        info.time           = toc(t_start);
        info.converged      = false;
        info.fitness        = NaN;
        info.not_applicable = true;
        info.reason         = M.reason;
        info.branch         = 0;
        return
    end

    [TH, re] = st_solve(M, T_d);                              % Steps 1-3, eight branches
    [Q, inl] = st_joints(M, TH, robot.q_min(:), robot.q_max(:));
    [q, k, dq] = st_select(Q, re & inl, q0);                  % p.7: bounds, minimum joint displacement

    info.iters          = 1;
    info.time           = toc(t_start);
    info.converged      = k > 0;
    info.fitness        = dq;
    info.not_applicable = false;
    info.reason         = '';
    info.Q_all          = Q;
    info.real           = re;
    info.in_limits      = inl;
    info.branch         = k;
end


% =====================================================================================================
% PoE model of the robot at its reference configuration (Eqs. 2, 15-16) and the robot class checks.
% =====================================================================================================
function M = st_model(robot)
    M = struct('ok', false, 'reason', '');
    if ~isfield(robot, 'dh')
        M.reason = 'no DH table: the joint axes cannot be read';
        return
    end
    n = robot.n;
    if n ~= 6
        M.reason = sprintf('%d joints: the method is for six-DOF robots (title; Sec. 4, p.4)', n);
        return
    end
    dh = robot.dh;
    toff = zeros(1, n);
    if isfield(dh, 'theta_offset'), toff = dh.theta_offset(:)'; end
    tool = eye(4);
    if isfield(robot, 'tool'), tool = robot.tool; end
    q_ref = robot.q_home(:);
    W = zeros(3, 6);  P = zeros(3, 6);  T = eye(4);
    for k = 1:6                                   % axis k = z_(k-1) through o_(k-1) (standard DH)
        W(:, k) = T(1:3, 3);
        P(:, k) = T(1:3, 4);
        T = T * dh_transform(dh.a(k), dh.alpha(k), dh.d(k), q_ref(k) + toff(k));
    end
    M = st_build(W, P, T * tool, q_ref);
end


function M = st_build(W, P, g0, q_ref)
% W, P: 3x6 joint axes and axis points, g0 = g_st(0), all in the base frame at the reference q_ref.
    tl = st_tol();
    M = struct('ok', false, 'reason', '', 'q_ref', q_ref(:));
    for k = 1:6, W(:, k) = W(:, k) / norm(W(:, k)); end

    % Sec. 4.1: axes 4, 5, 6 meet at the wrist point q_w = q_a
    [qw, res] = st_meet(W(:, 4:6), P(:, 4:6));
    M.wrist_residual = res;
    M.d46 = st_line_dist(W(:, 4), P(:, 4), W(:, 6), P(:, 6));
    if res > tl.geo || norm(cross(W(:, 4), W(:, 5))) < tl.geo || norm(cross(W(:, 5), W(:, 6))) < tl.geo
        M.reason = sprintf(['axes 4, 5, 6 do not meet in a point (Sec. 4.1, p.4): least-squares ' ...
            'residual %.4g m, axes 4 and 6 %.5g m apart'], res, M.d46);
        return
    end
    % Sec. 3 (p.3): axes 2 and 3 parallel and distinct; Eq. 6: axis 1 normal to them
    if norm(cross(W(:, 2), W(:, 3))) > tl.geo || st_line_dist(W(:, 2), P(:, 2), W(:, 3), P(:, 3)) < tl.geo
        M.reason = 'axes 2 and 3 are not two distinct parallel axes (Sec. 3, p.3)';
        return
    end
    if abs(W(:, 1)' * W(:, 2)) > tl.geo
        M.reason = 'axis 1 is not normal to axes 2 and 3 (Eq. 6: axis 1 = z, w2 = x)';
        return
    end
    % Eqs. 7-8: q_a lies in the plane through axis 1 normal to w2 (q_a,x = 0, as in Eq. 18)
    M.qa_offset = W(:, 2)' * (qw - P(:, 1));
    if abs(M.qa_offset) > tl.geo
        M.reason = sprintf(['the wrist point lies %.4g m off the plane through axis 1 normal to ' ...
            'axes 2 and 3 (Eq. 8 needs q_a,x = 0)'], M.qa_offset);
        return
    end

    % Frame of Fig. 4(b): origin on axis 1, z = w1, x = +/-w2, q_w at y >= 0
    ez = W(:, 1);
    ex = W(:, 2);
    ey = cross(ez, ex);
    if (qw - P(:, 1))' * ey < 0, ex = -ex; ey = -ey; end
    Rc = [ex'; ey'; ez'];
    M.A  = [Rc, -Rc * P(:, 1); 0 0 0 1];
    Wc = Rc * W;
    Pc = Rc * (P - P(:, 1));
    M.W   = Wc;
    M.XI  = zeros(6, 6);
    for k = 1:6, M.XI(:, k) = st_twist(Wc(:, k), Pc(:, k)); end      % Eq. 2
    M.g0  = M.A * g0;                                                  % Eq. 16
    M.qa  = Rc * (qw - P(:, 1));                                       % Eq. 18, q_a = q_w
    M.qa(1) = 0;                                                       % |q_a,x| < tol checked above
    M.qw  = M.qa;
    % points of axes 2 and 3 in the projection plane x = 0 (Fig. 3)
    M.r2 = Pc(:, 2) - (Pc(1, 2) / Wc(1, 2)) * Wc(:, 2);
    M.r3 = Pc(:, 3) - (Pc(1, 3) / Wc(1, 3)) * Wc(:, 3);
    % structure parameters in the notation of Eqs. 15-16 and Fig. 4(b)
    M.a1 = M.r2(2);  M.d1 = M.r2(3);  M.a2 = M.r3(3) - M.r2(3);
    M.a3 = M.qa(3) - M.r3(3);  M.d4 = M.qa(2) - M.a1;
    M.q6 = [0; M.a1 + M.d4; 0];                                        % Eq. 20
    M.qt = [0; M.a1; M.d1 + M.a2 + M.a3];                              % Eq. 26
    if st_point_line(M.q6, Wc(:, 6), M.qw) > tl.geo || st_point_line(M.q6, Wc(:, 4), M.qw) < tl.geo ...
            || st_point_line(M.q6, Wc(:, 5), M.qw) < tl.geo
        M.reason = 'the point of Eq. 20 is not on axis 6 or lies on axis 4 or 5';
        return
    end
    if st_point_line(M.qt, Wc(:, 6), M.qw) < tl.geo
        M.reason = 'the point of Eq. 26 lies on axis 6';
        return
    end
    M.ok = true;
end


% =====================================================================================================
% Steps 1-3 for all eight branches (Eqs. 4-28). TH: 6x8 PoE angles, re: 1x8 real flags.
% Branch order: k = 4*(i-1) + 2*(j-1) + l, i: q_c sign +/- (Eq. 8), j: circle intersection
% (Eqs. 13-14), l: gamma sign +/- (Eq. 22).
% =====================================================================================================
function [TH, re] = st_solve(M, T_d)
    G  = M.A * T_d;                                % target in the frame of Fig. 4(b)
    H  = G / M.g0;                                 % g_st(theta) g_st(0)^-1 (Eqs. 3, 17)
    q  = H(1:3, 1:3) * M.qa + H(1:3, 4);           % Eq. 4: image of q_a (Eq. 18)
    TH = nan(6, 8);
    re = false(1, 8);
    sg = [1, -1];
    for i = 1:2
        [t1, qc] = st_eq8_10(q, sg(i));
        [t2, t3, ok] = st_eq11_14(M, qc);
        for j = 1:2
            for l = 1:2
                k = 4 * (i - 1) + 2 * (j - 1) + l;
                if ~ok, continue; end
                g1 = st_exp(M.XI(:, 3), -t3(j)) * st_exp(M.XI(:, 2), -t2(j)) ...
                     * st_exp(M.XI(:, 1), -t1) * H;                         % Eq. 19
                [t4, t5, ok2] = st_step2(M, g1, sg(l));
                if ~ok2, continue; end
                g2 = st_exp(M.XI(:, 5), -t5) * st_exp(M.XI(:, 4), -t4) * g1; % Eq. 25
                t6 = st_step3(M, g2);
                TH(:, k) = [t1; t2(j); t3(j); t4; t5; t6];
                re(k) = true;
            end
        end
    end
end


function [t1, qc] = st_eq8_10(q, s)
% Eq. 8 (q_c, sign s) and Eq. 10 (theta_1 by PK-1 about axis 1 = z).
    rho = sqrt(q(1)^2 + q(2)^2);
    qc  = [0; s * rho; q(3)];
    t1  = atan2(-s * rho * q(1), s * rho * q(2));
end


function [t2, t3, ok] = st_eq11_14(M, qc)
% Eqs. 11-14: theta_2, theta_3 for the two intersections of the circles about r2 (radius |u|)
% and r3 (radius |v|) in the plane normal to axes 2 and 3 (Fig. 3).
    tl  = st_tol();
    w2  = M.XI(1:3, 2);  w3 = M.XI(1:3, 3);
    u   = qc - M.r2;  v = M.qa - M.r3;  r23 = M.r2 - M.r3;  r32 = M.r3 - M.r2;   % Eq. 11
    nu  = norm(u);  nv = norm(v);  nr = norm(r23);
    t01 = atan2(w2' * cross(u, r32), u' * r32);                                  % Eq. 12
    c1  = (nu^2 + nr^2 - nv^2) / (2 * nu * nr);
    t02 = atan2(w3' * cross(v, r23), v' * r23);                                  % Eq. 14
    c2  = (nv^2 + nr^2 - nu^2) / (2 * nv * nr);
    ok  = isfinite(c1) && isfinite(c2) && abs(c1) <= 1 + tl.tan && abs(c2) <= 1 + tl.tan;
    if ~ok
        t2 = [NaN, NaN];  t3 = [NaN, NaN];
        return
    end
    p1 = acos(max(-1, min(1, c1)));
    p2 = acos(max(-1, min(1, c2)));
    t2 = [p1 - t01, -p1 - t01];                                                  % Eq. 13
    t3 = [t02 + p2,  t02 - p2];                                                  % Eq. 14
end


function [t4, t5, ok] = st_step2(M, g1, sgam)
% Eqs. 20-24: PK-2 on q6 (Eq. 22, sign of gamma = sgam), then PK-1 for theta_5 and theta_4.
    tl = st_tol();
    w4 = M.XI(1:3, 4);  w5 = M.XI(1:3, 5);
    y6 = g1(1:3, 1:3) * M.q6 + g1(1:3, 4);                                       % Eq. 21
    u  = M.q6 - M.qw;  v = y6 - M.qw;
    k45 = w4' * w5;
    den = k45^5 - 1;
    al  = (k45 * (w5' * u) - w4' * v) / den;                                     % Eq. 22
    be  = (k45 * (w5' * v) - w5' * u) / den;
    g2  = (u' * u - al^2 - be^2 - 2 * al * be * k45) / norm(cross(w4, w5));
    ok  = g2 >= -tl.tan * (u' * u);
    if ~ok
        t4 = NaN;  t5 = NaN;
        return
    end
    ga = sgam * sqrt(max(g2, 0));
    c  = al * w4 + be * w5 + ga * cross(w4, w5) + M.qw;                          % Eq. 22
    t5 = st_pk1(w5, M.qw, M.q6, c);                                              % Eqs. 23-24
    t4 = st_pk1(w4, M.qw, c, y6);
end


function t6 = st_step3(M, g2)
% Eqs. 26-28: theta_6 by PK-1 on q_t.
    yt = g2(1:3, 1:3) * M.qt + g2(1:3, 4);                                       % Eq. 27
    t6 = st_pk1(M.XI(1:3, 6), M.qw, M.qt, yt);                                   % Eq. 28
end


function th = st_pk1(w, r, p, qq)
% PK-1 (Eqs. 10, 24, 28): rotation about the axis (w, r) taking p to qq; u', v' normal to w.
    u  = p - r;  v = qq - r;
    up = u - w * (w' * u);
    vp = v - w * (w' * v);
    th = atan2(w' * cross(up, vp), up' * vp);
end


% =====================================================================================================
% Joint values, limits and selection (Sec. 5, p.7).
% =====================================================================================================
function [Q, inl] = st_joints(M, TH, q_min, q_max)
    Q   = st_wrap(TH + M.q_ref);
    inl = all(Q >= q_min & Q <= q_max, 1);
    inl(any(isnan(Q), 1)) = false;
end


function [q, k, dq] = st_select(Q, adm, q0)
% Minimum joint-displacement solution among the admissible ones; q0 and k = 0 if there is none.
    q = q0;  k = 0;  dq = NaN;
    if ~any(adm), return; end
    d = sqrt(sum((Q - q0).^2, 1));
    d(~adm) = Inf;
    [dq, k] = min(d);
    q = Q(:, k);
end


function x = st_wrap(x)
% Wrap to (-pi, pi].
    x = pi - mod(pi - x, 2 * pi);
end


% =====================================================================================================
% Screw-theory primitives.
% =====================================================================================================
function xi = st_twist(w, r)
% Eq. 2: xi = [w; r x w] for a revolute joint.
    xi = [w; cross(r, w)];
end


function E = st_exp(xi, th)
% Twist exponential (p.3), w ~= 0: [e^(w^ th), (I - e^(w^ th))(w x v) + th w w^T v; 0 1].
    w = xi(1:3);  v = xi(4:6);
    K = [0, -w(3), w(2); w(3), 0, -w(1); -w(2), w(1), 0];
    R = eye(3) + K * sin(th) + K * K * (1 - cos(th));
    E = [R, (eye(3) - R) * cross(w, v) + th * (w * w') * v; 0 0 0 1];
end


function T = st_poe(XI, th, g0)
% Eq. 3: g_st(theta) = e^(xi_1 theta_1) ... e^(xi_n theta_n) g_st(0).
    T = eye(4);
    for k = 1:numel(th), T = T * st_exp(XI(:, k), th(k)); end
    T = T * g0;
end


function [p, res] = st_meet(W, P)
% Least-squares meeting point of the lines (W(:,k), P(:,k)) and the largest distance from it.
    A = zeros(3);  b = zeros(3, 1);
    for k = 1:size(W, 2)
        Pk = eye(3) - W(:, k) * W(:, k)';
        A = A + Pk;  b = b + Pk * P(:, k);
    end
    p = A \ b;
    res = 0;
    for k = 1:size(W, 2), res = max(res, st_point_line(p, W(:, k), P(:, k))); end
end


function d = st_point_line(x, w, p)
% Distance from point x to the line through p along unit w.
    e = x - p;
    d = norm(e - w * (w' * e));
end


function d = st_line_dist(w1, p1, w2, p2)
% Distance between two lines.
    c = cross(w1, w2);
    if norm(c) < 1e-12
        d = st_point_line(p2, w1, p1);
    else
        d = abs((p2 - p1)' * c) / norm(c);
    end
end


function tl = st_tol()
    tl = struct('geo', 1e-9, 'tan', 1e-12);
end
