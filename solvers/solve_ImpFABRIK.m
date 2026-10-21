function [q_best, info] = solve_ImpFABRIK(robot, T_d, q0, opts)
% SOLVE_IMPFABRIK  Improved FABRIK of Wang et al. (2024) with the end-link pose constraint: Algorithm 2
%                  without linear feed (lines 3-6), Eq. 11, Eqs. 5-9 and Eq. 10 / Algorithm 1.
%
%   Source:
%     Zhe Wang, Dean Hu, Detao Wan, Chang Liu (2024). An improved inverse kinematics solution method
%     for the hyper-redundant manipulator with end-link pose constraint. Journal of Field Robotics, 41:
%     1900-1921. DOI: 10.1002/rob.22362
%   No issue number is printed (p.1, p.22). Page numbers are PDF pages 1-22 (journal page = PDF
%   page + 1899).
%
%   The paper's manipulator (Sec. 2.1, p.3; Eqs. 1-4, pp.3-4; Table 1, p.4; Table 2, p.9):
%     n + 1 links joined by n joints, "Each joint has two rotation degrees of freedom" (p.3). The
%     first link L0 is fixed to the drive box; J0, the centre of the first joint, is the base point
%     (frame SJ1 coincides with SJ0, p.5). Frame SJ_i sits at J_i and is fixed to link L_i, its y-axis
%     along the link; theta_i turns SJ_i about the z-axis and phi_i about the x-axis of SJ_(i-1), so
%     link L_i runs from J_i to J_(i+1) (J_(n+1) = EP, the endpoint). All links lie along one line at
%     zero angles. Simulated arm: 12 joints, L_i = 200 mm, theta_i and phi_i in [-30, 30] deg.
%
%   Method (Algorithm 2, p.9; Fig. 7, p.8):
%     while errP > the specified value (errP = distance from EP to the target, p.7):
%       forward search with the end-link constraint (Sec. 4.1, pp.6-7; Fig. 5, p.7): EP is moved to
%         the target, then
%           J_n* = EP* - L_n [cos f sin q; cos f cos q; sin f]                              (Eq. 11)
%         with q the yaw and f the pitch of the end link in SJ0; then, i = n-1, ..., 1, J_i* lies on
%         the line from J_(i+1)* to J_i at distance L_i (Sec. 3.1, p.4; Fig. 3a, p.5):
%           J_i* = J_(i+1)* + L_i (J_i - J_(i+1)*) / ||J_i - J_(i+1)*||
%       backward search (Sec. 3.2, p.4; Fig. 4, p.5): the chain is "translated as a whole until the
%         base joint coincides with its initial position";
%       joint angles from the joint positions (pp.5-6), joint by joint, each frame SJ_i rebuilt from
%         the angles before it ("According to the recurrence relationship", p.6):
%           theta_1 = arctan(-x_J2 / y_J2),  phi_1 = arcsin(z_J2 / L_1)                     (Eq. 5)
%           P_J(i+2) in SJ_i = (T_J_i)^-1 P_J(i+2)                                          (Eq. 8)
%           theta_(i+1) = arctan(-x / (y - L_i)),  phi_(i+1) = arcsin(z / L_(i+1))           (Eq. 9)
%       joint limits (Eq. 10, Algorithm 1, p.6): an angle beyond its limit is set to the limit and
%         the joint coordinates are recalculated;
%       update the joint angles, EP and errP.
%     With the linear feed (Algorithm 2 lines 7-9, Eq. 12, p.8; enabled after k searches) the
%     backward search also moves the base along y.
%
%   Benchmark chains and applicability:
%     Synthetic9/11/13: n = 9, 11, 13 single-axis revolute joints, standard DH with a = 0, twists
%       +90 / -90 deg alternating (0 on joint n), d = 0 on the even joints, identity tool. Consecutive
%       axes meet at 90 deg, and the links run along the odd joints' axes between the joint centres
%         C_0 = O_1 (on the axis of joint 1, so it never moves), C_j = O_(2j+1) (j = 1..m-1),
%         C_m = EE origin,  m = (n-1)/2,  L_j = d_(2j+1) (0.333 m; the end link 0.20 m).
%       The base column O_0-O_1 (d_1) plays the paper's fixed link L0 and C_0 its J0. Joints 2j-1 and
%       2j act as the paper's 2-DOF joint J_j: they set the direction of link j (C_(j-1) to C_j).
%       Pair 1 is a yaw about the vertical axis of joint 1 and a pitch, the structure of Eq. 5. Pairs
%       j >= 2 are a roll of link j-1 about its own axis (joint 2j-1) and a pitch at C_(j-1) (joint
%       2j), not the yaw/pitch of Eq. 9. Joint n turns the end link about its own axis (EE roll),
%       which the paper's model does not have. The end link C_(m-1)-C_m lies along the EE z-axis
%       (alpha_n = 0, identity tool), so Eq. 11 constrains the EE approach axis.
%       SJ0: origin C_0, y along link 1 at zero angles (base x; the paper's links lie along y at zero
%       angles), z along the axis of joint 1 (base z), x = y cross z (base -y). Then theta_1 = q_1
%       and phi_1 = q_2.
%     UR5, Panda, IRB120: consecutive axes are offset or parallel (UR5 axes 2-3 and 3-4 parallel,
%       0.425 m and 0.392 m apart; Panda axes 3-4 and 4-5 0.0825 m apart and 6-7 0.088 m apart; IRB120
%       axes 2-3 parallel 0.270 m apart and 3-4 0.070 m apart), so no joint pair acts as a 2-DOF joint
%       at a point. Not applicable: info.not_applicable = true, q0 returned, no FK call. The structure
%       test reads robot.dh (n odd, a = 0, the twist pattern, d = 0 on the even joints, d > 0 on the
%       odd joints) and robot.tool.
%
%   Decode for single-axis joint pairs, derived from Eqs. 5-9; branch and singular case resolved by
%   the warm start:
%     v = R_(2j-2)^T (C_j - o_(2j-2)) - [0; 0; d_(2j-1)]
%       (the Eq. 8 form: frame 2j-2 of robot.fk with origin o and rotation R, whose z-axis is the axis
%       of joint 2j-1; d_(2j-1) = L_(j-1) is the "y - L_i" of Eq. 9, the base column for pair 1).
%     With t = q + theta_offset the pair maps e_z to v/|v| = [-sin t_b cos t_a; -sin t_b sin t_a;
%     cos t_b] (a = 2j-1, b = 2j), so
%       t_b = +/- acos(v_z / |v|)    (pitch; computed as atan2(+/- hypot(v_x, v_y), v_z)),
%       t_a = atan2(-v_y / sin t_b, -v_x / sin t_b)    (roll; the yaw for pair 1).
%     Pair 1 on the branch |q_2| <= 90 deg is Eq. 5 as printed in SJ0: q_1 = arctan(-x/y) (y > 0) and
%     q_2 = arcsin(z / L_1).
%     Branch: of the two solutions (t_a, t_b) and (t_a + pi, -t_b), the one nearest the previous
%       iterate (Euclidean norm of the two wrapped joint differences; the first search uses q0).
%       Each angle is the 2 pi representative nearest its previous value.
%     Singular case: hypot(v_x, v_y) <= 1e-12 |v| (link j in line with link j-1, or link 1 vertical):
%       the roll keeps its previous value and t_b = atan2(-(v_x cos t_a + v_y sin t_a), v_z).
%     Joint n (EE roll) keeps its previous value.
%     Frames: pair j needs frame 2j-2 built from the angles decoded before it; it is read from
%       robot.fk (one call per pair j >= 2; pair 1 uses the base frame).
%
%   Joint limits (Eq. 10, Algorithm 1): a decoded angle with no 2 pi representative in [q_min, q_max]
%     is set to the limit it exceeds (Eq. 10 on the representative nearest the previous value). The
%     joint coordinates are then recalculated with the relative angles beyond that joint unchanged
%     (Algorithm 1 line 4): the later pairs are decoded in the frames of the unlimited angles, and
%     the coordinates follow from robot.fk of the limited angles. On Synthetic9/11/13 (limits
%     [-pi, pi]) every angle has a representative inside the limits, so Eq. 10 never acts there.
%
%   Parameters:
%     Specified value of errP   1e-10 mm (Sec. 5.1, p.9), 1e-6 mm (Sec. 5.2, p.12; Sec. 5.3, p.14):
%                               accuracies of the paper's example tasks, replaced by opts.tol_pos
%                               (see Stop).
%     Iteration limit 30        Sec. 5.1 runs ("set as thirty", p.9; "set to 30", p.11); not used
%                               (see Budget). The Sec. 5.2 run takes 94 iterations (Fig. 16, p.13).
%     k = 20                    linear-feed threshold (Sec. 5.3, p.12); not used (no feed axis).
%     L_j                       the robot's, d_(2j+1) from robot.dh (Algorithm 2 takes the link
%                               lengths as input).
%     q, f                      from the target's z-axis u in SJ0: f = atan2(u_z, hypot(u_x, u_y)),
%                               q = atan2(u_x, u_y), so [cos f sin q; cos f cos q; sin f] = u. The
%                               paper sets them per task (Sec. 5.2, p.12; Sec. 5.3, p.14).
%     Joint ranges              the robot's [q_min, q_max] (+/-30 deg for the paper's arm, Tables 1-2).
%     Singular threshold 1e-12  not in the paper (within +/-30 deg its decode has no singular case);
%                               benchmark value, see Decode.
%
%   Benchmark interface:
%     Budget: one search costs m robot.fk calls (m - 1 frames for pairs 2..m, then one for the new
%       pose, joint positions and errP); q0 is evaluated once before the first search. A search starts
%       only if ikb_budget_left(opts) >= m. Per waypoint FK = 1 + m x searches <= 2000: at most 499,
%       399 and 333 searches (FK 1997, 1996, 1999) on Synthetic9, 11 and 13. No Jacobian call.
%       Outside the harness the solver counts its own FK calls against opts.budget. opts.max_iter and
%       the paper's iteration limit are not used; no quantity varies over the iteration count.
%     Stop: errP = ||p_d - p_EE|| < opts.tol_pos (m), the paper's test (Algorithm 2 line 1; p.7) with
%       its example-task accuracy replaced by the position part of the pose gate (1e-5 m when the
%       field is absent), or the budget. The end-link yaw and pitch (the EE z-axis) are imposed by
%       Eq. 11 in every forward search and hold to rounding at every iterate after the first search
%       while Eq. 10 does not act; they are not part of the paper's test. The EE roll is not
%       controlled (Eq. 11 has no roll; joint n is held), so opts.tol_rot is not read and the full
%       pose gate is met in Phase 1 only where the roll happens to agree. info.converged is the stop
%       test.
%     Warm start: the search starts at q0 (the previous waypoint's solution), clamped to the limits;
%       J0 is C_0 of q0; the branch rule starts from q0.
%     Joint limits: Eq. 10 / Algorithm 1 with the robot's [q_min, q_max] (see Joint limits); q0 is
%       clamped to them.
%     Units: metres and radians; the paper's millimetres enter only the stop accuracy, which is
%       replaced.
%     Output: the iterate that meets the stop test; if the budget ends first, the evaluated iterate
%       with the lowest errP (the paper's action criterion, p.2). info.fitness is that errP (m);
%       info.err_pos (m) and info.err_rot (rad) as in bench_run; info.axis_err is the angle between
%       the EE and target z-axes (rad). solve_ImpFABRIK('ops') returns the operators for unit tests.
%
%   Deviations from the paper:
%     - Joint model: the paper's 2-DOF joints are pairs of single-axis joints (yaw/pitch for pair 1,
%       roll/pitch for the others) on Synthetic9/11/13; not applicable to UR5, Panda, IRB120.
%     - Decode for single-axis joint pairs, derived from Eqs. 5-9; branch and singular case resolved
%       by the warm start (see Decode). arctan and arcsin in Eqs. 5 and 9 are single-valued within
%       the paper's +/-30 deg ranges; atan2 with the nearest-branch rule covers +/-180 deg ranges.
%     - Joint n (EE roll) held at its previous value; the paper's constraint covers the yaw q and
%       pitch f of the end link only, so the missing roll control is the paper's.
%     - Linear feed (Eq. 12, Algorithm 2 lines 7-9, k = 20) inactive: the benchmark robots have no
%       prismatic base axis, so every search is the search of lines 3-6, also after k searches.
%     - Stop at opts.tol_pos instead of 1e-10 / 1e-6 mm, and at the FK budget instead of 30
%       iterations; if the budget ends first, the best evaluated iterate is returned.
%     - Eq. 11 parameters q and f taken from the target's z-axis.
%     - Algorithm 1 line 4 read as recalculating the coordinates with the relative angles beyond the
%       limited joint unchanged (not stated how).
%     - Forward search where J_i coincides with J_(i+1)* (direction undefined, not stated): J_i* keeps
%       the current link vector, J_i* = J_(i+1)* + (J_i - J_(i+1)).
%     - Joint positions, frames and errP from robot.fk (standard DH) instead of Eqs. 1-4.
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
%             Samuel Moveh (Transport and Telecommunication Institute,
%             Riga, Latvia)
% Year      : 2026
% License   : MIT (see LICENSE)
% Repository: https://github.com/eamcmx/IK_benchmark_2026
% ==========================================================================

    if ischar(robot) && strcmp(robot, 'ops')
        q_best = struct('params', fab_params(), 'core', @fab_core, 'chain', @chain_model, ...
                        'centres', @centres, 'frame0', @frame_sj0, 'endlink', @endlink_qf, ...
                        'eq11', @eq11, 'forward', @forward_search, 'backward', @backward_search, ...
                        'pair', @decode_pair, 'decode', @decode_chain, 'limit', @eq10_limit, ...
                        'wrap', @wrap_pi);
        info = [];
        return
    end
    [q_best, info] = fab_core(robot, T_d, q0, opts, fab_params());
end

% ==================================================================== main loop
function [q_best, info] = fab_core(robot, T_d, q0, opts, P)
% Algorithm 2 lines 1-6 from q0 until errP < tol_pos or the FK budget ends.
    t_start = tic;
    q0 = q0(:);
    tp = 1e-5;
    if isfield(opts, 'tol_pos'), tp = opts.tol_pos; end
    S = chain_model(robot);

    info = struct();
    info.solver         = 'Improved-FABRIK';
    info.method         = 'Algorithm 2 (no linear feed) of Wang, Hu, Wan and Liu (2024)';
    info.not_applicable = ~S.ok;
    if ~S.ok                                                          % no 2-DOF joint pairs
        q_best = q0;
        info.iters = 0;  info.time = toc(t_start);  info.converged = false;  info.fitness = NaN;
        info.err_pos = NaN;  info.err_rot = NaN;  info.axis_err = NaN;  info.fk_calls = 0;
        info.jac_calls = 0;  info.stop = 'not_applicable';  info.gate = tp;
        info.history = zeros(1, 0);  info.params = P;  info.pairs = 0;
        return
    end

    q = min(max(q0, robot.q_min(:)), robot.q_max(:));                 % warm start
    q_best = q;  T_best = NaN(4);  f_best = NaN;
    p_d = T_d(1:3,4);  u_d = T_d(1:3,3);
    fk = 0;  it = 0;  conv = false;  stop = 'budget';  hist = zeros(1, 0);
    if budget_left(opts, fk) >= 1
        [Te, Ts] = robot.fk(q);
        fk = 1;
        C  = centres(Te, Ts, S);
        J0 = C(:,1);                                                  % initial position of the base joint
        R0 = frame_sj0(S);
        [qy, fp] = endlink_qf(R0.' * u_d);                            % yaw q, pitch f in SJ0
        Jn = J0 + R0 * eq11(R0.' * (p_d - J0), S.L(end), qy, fp);     % Eq. 11 (EP* = target)
        err = norm(p_d - C(:,end));                                   % errP
        hist(1) = err;  f_best = err;  T_best = Te;
        while true
            if err < tp                                               % Algorithm 2 line 1
                conv = true;  stop = 'gate';
                break
            end
            if budget_left(opts, fk) < S.m, break; end                % next search not evaluable
            Cf = forward_search(C, p_d, Jn, S.L);                     % line 4: Eq. 11 + Sec. 3.1
            Cb = backward_search(Cf, J0);                             % line 5: Sec. 3.2 translation
            [q, nf] = decode_chain(robot, S, Cb, q, P);               % Eqs. 5-9 form, Eq. 10 / Alg. 1
            fk = fk + nf;
            [Te, Ts] = robot.fk(q);                                   % line 6: angles, EP, errP
            fk = fk + 1;
            it = it + 1;
            C   = centres(Te, Ts, S);
            err = norm(p_d - C(:,end));
            hist(end+1) = err; %#ok<AGROW>
            if err < f_best
                f_best = err;  q_best = q;  T_best = Te;
            end
        end
    end

    ep = NaN;  er = NaN;  ax = NaN;
    if fk > 0
        [~, e_p, e_r] = pose_error(T_d, T_best);
        ep = norm(e_p);
        er = norm(e_r);
        ax = atan2(norm(cross(T_best(1:3,3), u_d)), dot(T_best(1:3,3), u_d));
    end

    info.iters     = it;                                              % searches (C_n)
    info.time      = toc(t_start);
    info.converged = conv;                                            % errP < tol_pos
    info.fitness   = f_best;                                          % errP of the returned iterate, m
    info.err_pos   = ep;                                              % m
    info.err_rot   = er;                                              % rad
    info.axis_err  = ax;                                              % EE z-axis vs target z-axis, rad
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.stop      = stop;                                            % 'gate', 'budget'
    info.gate      = tp;                                              % tol_pos (m)
    info.history   = hist;                                            % errP of every evaluated iterate
    info.params    = P;
    info.pairs     = S.m;
end

% ==================================================================== operators
function P = fab_params()
% k = 20: linear-feed threshold (Sec. 5.3, p.12), not used. sing: singular threshold of the pair decode.
    P = struct('k', 20, 'sing', 1e-12);
end

function S = chain_model(robot)
% Joint-pair structure of Synthetic9/11/13 read from robot.dh; S.ok false for any other chain.
    S = struct('ok', false, 'n', robot.n, 'm', 0, 'd', [], 'off', [], 'L', []);
    if ~isfield(robot, 'dh'), return; end
    n  = robot.n;
    dh = robot.dh;
    tool = eye(4);
    if isfield(robot, 'tool') && ~isempty(robot.tool), tool = robot.tool; end
    off = zeros(n, 1);
    if isfield(dh, 'theta_offset'), off = dh.theta_offset(:); end
    if n < 3 || mod(n, 2) ~= 1, return; end
    tw = zeros(n, 1);
    tw(1:2:n-2) = pi/2;
    tw(2:2:n-1) = -pi/2;
    d = dh.d(:);
    ok = all(abs(dh.a(:)) < 1e-12) && all(abs(dh.alpha(:) - tw) < 1e-12) ...
         && all(abs(d(2:2:n-1)) < 1e-12) && all(d(1:2:n) > 0) && max(max(abs(tool - eye(4)))) < 1e-12;
    if ~ok, return; end
    S.ok  = true;
    S.m   = (n - 1) / 2;
    S.d   = d;
    S.off = off;
    S.L   = d(3:2:n);                                                 % L_j = d_(2j+1), j = 1..m
end

function C = centres(Te, Ts, S)
% Joint centres C_0..C_m = O_1, O_3, ..., O_(n-2), EE origin (columns 1..m+1).
    C = [reshape(Ts(1:3, 4, 1:2:S.n-2), 3, []), Te(1:3,4)];
end

function R0 = frame_sj0(S)
% Axes of SJ0 in base coordinates: y along link 1 at zero angles, z along the axis of joint 1.
    z = [0; 0; 1];
    y = pair_dir(S.off(1), S.off(2));
    y = y - (y.' * z) * z;
    y = y / norm(y);
    R0 = [cross(y, z), y, z];
end

function u = pair_dir(ta, tb)
% Direction of the next link in the frame before the pair: Rz(ta) Rx(90) Rz(tb) Rx(-90) e_z.
    u = [-sin(tb) * cos(ta); -sin(tb) * sin(ta); cos(tb)];
end

function [q, f] = endlink_qf(u)
% Yaw q and pitch f of the end-link direction u (SJ0 coordinates), as used in Eq. 11.
    f = atan2(u(3), hypot(u(1), u(2)));
    q = atan2(u(1), u(2));
end

function Jn = eq11(pEP, Ln, q, f)
% Eq. 11: position of J_n from the endpoint, the end-link length and the yaw q / pitch f (SJ0).
    Jn = [pEP(1) - Ln * cos(f) * sin(q);
          pEP(2) - Ln * cos(f) * cos(q);
          pEP(3) - Ln * sin(f)];
end

function Cf = forward_search(C, p_d, Jn, L)
% Forward search with the end-link constraint (Sec. 4.1): EP* = target, J_n* by Eq. 11, then each
% joint on the line from the next new joint to its current position at the link length (Sec. 3.1).
    m  = size(C, 2) - 1;
    Cf = C;
    Cf(:, m+1) = p_d;
    Cf(:, m)   = Jn;
    for i = m-1:-1:1
        r  = C(:, i) - Cf(:, i+1);
        nr = norm(r);
        if nr > 0
            Cf(:, i) = Cf(:, i+1) + (L(i) / nr) * r;
        else
            Cf(:, i) = Cf(:, i+1) + (C(:, i) - C(:, i+1));            % direction undefined (not stated)
        end
    end
end

function Cb = backward_search(Cf, J0)
% Improved backward search (Sec. 3.2): rigid translation until the base joint is back at J0.
    Cb = Cf + (J0 - Cf(:, 1));
end

function [q, nf, qr] = decode_chain(robot, S, Cb, qp, P)
% Joint angles from the joint positions Cb, pair by pair (Eqs. 5-9 form), branch nearest the previous
% iterate qp, Eq. 10 / Algorithm 1; joint n kept. qr: the unlimited angles, whose frames serve the
% later pairs. nf: robot.fk calls made.
    q  = qp(:);
    qr = q;
    nf = 0;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    R  = eye(3);
    o  = zeros(3, 1);                                                 % frame 0 for pair 1
    for j = 1:S.m
        a = 2*j - 1;
        b = 2*j;
        if j > 1
            [~, Ts] = robot.fk(qr);                                   % frame 2j-2 (recurrence, p.6)
            nf = nf + 1;
            R  = Ts(1:3, 1:3, b-2);
            o  = Ts(1:3, 4, b-2);
        end
        v = R.' * (Cb(:, j+1) - o) - [0; 0; S.d(a)];                  % Eq. 8 form, minus L_(j-1)
        [ta, tb] = decode_pair(v, qp(a) + S.off(a), qp(b) + S.off(b), P.sing);
        qr(a) = qp(a) + wrap_pi(ta - S.off(a) - qp(a));
        qr(b) = qp(b) + wrap_pi(tb - S.off(b) - qp(b));
        q(a)  = eq10_limit(qr(a), qp(a), lo(a), hi(a));
        q(b)  = eq10_limit(qr(b), qp(b), lo(b), hi(b));
    end
end

function [ta, tb] = decode_pair(v, pa, pb, tol)
% Pair angles (DH thetas) that point the next link along v: tb = +/- acos(v_z/|v|), ta = atan2; the
% branch nearest the previous angles (pa, pb); roll kept when hypot(v_x, v_y) <= tol |v|.
    s = hypot(v(1), v(2));
    if s <= tol * norm(v)
        ta = pa;
        tb = atan2(-(v(1) * cos(ta) + v(2) * sin(ta)), v(3));
        return
    end
    ta1 = atan2(-v(2), -v(1));
    tb1 = atan2(s, v(3));                                             % sin(tb1) > 0
    ta2 = ta1 + pi;
    tb2 = -tb1;
    d1  = norm(wrap_pi([ta1 - pa; tb1 - pb]));
    d2  = norm(wrap_pi([ta2 - pa; tb2 - pb]));
    if d2 < d1
        ta = ta2;  tb = tb2;
    else
        ta = ta1;  tb = tb1;
    end
end

function y = eq10_limit(x, p, lo, hi)
% Eq. 10 / Algorithm 1: x if inside [lo, hi]; else its 2 pi representative inside the limits nearest
% p; with none inside, x set to the limit it exceeds.
    if x >= lo && x <= hi
        y = x;
        return
    end
    k = ceil((lo - x) / (2*pi)) : floor((hi - x) / (2*pi));
    if isempty(k)
        y = min(max(x, lo), hi);
    else
        c = x + 2*pi*k;
        [~, i] = min(abs(c - p));
        y = c(i);
    end
end

function y = wrap_pi(x)
% Angle difference wrapped to [-pi, pi].
    y = x - 2*pi * round(x / (2*pi));
end

function n = budget_left(opts, fk)
% FK calls still allowed on this waypoint: ikb_budget_left inside the harness, opts.budget minus the
% solver's own count outside it.
    n = ikb_budget_left(opts);
    if isinf(n) && isfield(opts, 'budget') && ~isempty(opts.budget)
        n = opts.budget - fk;
    end
end
