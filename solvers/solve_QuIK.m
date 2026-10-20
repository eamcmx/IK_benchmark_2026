function [q_best, info] = solve_QuIK(robot, T_d, q0, opts)
% SOLVE_QUIK  Quick Inverse Kinematics (QuIK, Halley's method) of Lloyd, Irani and Ahmadi (2022).
%
%   Source:
%     Steffan Lloyd, Rishad A. Irani, Mojtaba Ahmadi (2022). Fast and Robust Inverse Kinematics of
%     Serial Robots Using Halley's Method. IEEE Transactions on Robotics, 38(5), 2768-2780.
%     DOI: 10.1109/TRO.2022.3162954
%   Page numbers are PDF pages 1-13 (printed page = PDF page + 2767).
%   The method is the undamped QuIK of Sec. III (Eqs. 17-23, pp.4-5), the method the paper proposes
%   (abstract and Sec. V). DQuIK (Sec. III-A, Eqs. 25-28, p.5) is its damped extension and is not
%   used here; neither is NR / DNR.
%
%   Method:
%     Error e = [e_lin; w] of the end-effector pose T_n against the target T_d (Eqs. 10-11, p.4;
%     Appendix, Eqs. 34-38, p.12). w is the angle-axis vector of R_e = R_n R_d^T:
%       eps = [r32 - r23, r13 - r31, r21 - r12]^T of R_e                                  (Eq. 35)
%       w = atan2(||eps||, trace(R_e) - 1) ||eps||^-1 eps                                 (Eq. 36)
%       w = (3/4 - trace(R_e) / 12) eps     ||eps|| nearly zero, trace(R_e) near 3         (Eq. 37)
%       w = pi/2 [diag(R_e) + 1]            ||eps|| nearly zero, trace(R_e) near -1        (Eq. 38)
%     e_lin = p_n - p_d (see Readings), in m (see Units).
%     Error saturation, applied separately to e_lin (d = d_lin) and w (d = d_rot) (p.7):
%       e_sat = e if ||e|| < d, d e / ||e|| otherwise                                     (Eq. 31)
%     Newton-Raphson predictor:  dq_nr = -J(q_k) \ e_sat                                   (Eq. 18)
%     Kinematic Hessian H = dJ/dq, page H_i = dJ/dq_i (Eqs. 19-20, p.4), from the columns of J
%     alone, all joints revolute (p.5):
%       dj_vj/dq_i = j_wi x j_vj (i <= j),  dj_vi/dq_j (i > j);
%       dj_wj/dq_i = j_wi x j_wj (i < j),   0 otherwise                                   (Eq. 23)
%       [H dq_nr](:,j) = sum_i (dj_j/dq_i) dq_nr,i, the product of Eq. 4
%     QuIK (Halley) update:  q_{k+1} = q_k - [J(q_k) + 1/2 H(q_k) dq_nr] \ e_sat           (Eq. 17)
%     A \ b is the minimum-norm solution A^T (A A^T)^-1 b of Eq. 24 (p.5), with A A^T factorised by
%     Cholesky, the solver the paper reports for its implementation (p.11). J is the geometric
%     Jacobian of Eqs. 15-16 (p.4), robot.jac, in the world frame.
%
%   Parameters:
%     d_lin = 340 mm   p.7, the value tuned for QuIK on the paper's KUKA KR6 (Nelder-Mead over 1e5
%     d_rot = 1.00 rad samples with sigma = 1 rad). The paper reports no other QuIK values (pp.1-13
%                      checked), so these are used for every robot (the values the paper reports for
%                      its own IK runs); d_lin enters as 0.340 m. The NR values (140 mm, 0.86 rad) and
%                      the starting guess of its ref. [33] (half the average link length, pi/4) are
%                      not used.
%     lambda^2         Not part of QuIK; DQuIK only (1e-5 and 1e-7, p.7; 1e-7 in Table II, p.11).
%     Tolerance        Eq. 29 (p.6), ||e|| < epsilon with epsilon = 1e-8 rad (pp.7, 10) or 1e-12
%                      (p.9): the accuracies of the paper's example tasks. Not used (see Stop).
%     Iteration cap    200 (p.6). Not used (see Stop).
%     Nearly zero      ||eps|| in Eqs. 37-38 is not quantified (pp.1-13 checked). The thresholds of
%                      the authors' codebase (ref. [29], which the paper cites for implementation
%                      details, p.6) are used with the paper's trace cases (p.12): Eq. 37 if
%                      ||eps|| < 1e-3 and trace(R_e) > -0.99; Eq. 38 if ||eps|| <= 1e-10 and
%                      trace(R_e) <= -0.99; Eq. 36 otherwise.
%
%   Readings:
%     Linear error     Eq. 11 sets e = log(T_n T_d^-1) and Eq. 34 takes the translation of the
%                      argument, which composes to p_n - R_n R_d^T p_d. Eqs. 13 and 17 (p.4) set the
%                      error gradient equal to the geometric Jacobian J of Eq. 16, and the text of
%                      Eq. 11 describes v_e as the linear velocity between p_n and p_d. The two
%                      statements disagree: the gradient of p_n - R_n R_d^T p_d is
%                      j_vj - j_wj x (R_e p_d), not j_vj. The reading e_lin = p_n - p_d is used, the
%                      one under which J is the gradient of e_lin (Eqs. 13, 17) and H the derivative
%                      of J (Eq. 19); ref. [29] computes the same difference.
%     Product H dq     Eq. 4 contracts the third index of the tensor and H_i = dJ/dq_i is its i-th page
%                      (Eq. 20), so [H dq] = sum_i H_i dq_i, the derivative of J along dq.
%     e_sat            Eq. 31 (p.7) defines the saturated error without naming the equations it
%                      enters; it replaces e in both Eq. 18 and Eq. 17.
%     Singular A A^T   Where the Cholesky factorisation fails (A A^T not numerically positive
%                      definite) the inverse of Eq. 24 does not exist; the solve is then the
%                      Moore-Penrose pseudoinverse (SVD), the minimum-norm solution the paper names
%                      for an exactly singular J (p.5).
%
%   Benchmark interface:
%     Budget: an iteration evaluates the pose of q_k (1 robot.fk call), applies the stopping test
%       and, if the test fails and the budget can still evaluate q_{k+1}, calls robot.jac once (not
%       budgeted, counted by the harness) and applies Eqs. 31, 18, 23 and 17. An iteration starts
%       only if ikb_budget_left(opts) >= 1. Per waypoint: FK = updates + 1 <= 2000, Jacobian calls
%       = updates <= 1999; the Hessian product of Eq. 23 needs no further kinematics call. The pose
%       that robot.jac also returns is not used: the budget counts robot.fk calls, so every
%       evaluated iterate is charged one FK call. Outside the harness the solver counts its own FK
%       calls against opts.budget. opts.max_iter is not used. No quantity varies over the iteration
%       count, so ikb_generations is not used.
%     Stop: the benchmark pose gate, position error < opts.tol_pos (m) and rotation error <
%       opts.tol_rot (rad) of pose_error, the test of bench_run (1e-5 each when a field is absent),
%       or the budget. info.converged is the gate test.
%     Warm start: q_0 = q0 (the previous waypoint's solution), clamped to the limits.
%     Units: metres and radians. The KR6 table and d_lin are printed in mm (p.7), but the norm
%       ||e|| the paper reports is computed in metres: at sigma = pi/4 its median starts near 10^0
%       and falls to the machine-precision line near 2e-16 (Fig. 3, p.8), where millimetres would
%       give about 10^2.5 and a floor near 1e-13. e_lin, J and ||e|| are therefore in m (no
%       conversion of the benchmark's lengths), and d_lin = 340 mm enters as 0.340 m.
%     Joint limits: the paper states no handling (pp.1-13 checked). Every iterate is clamped to
%       [q_min, q_max].
%     Output: the iterate that meets the stopping test; if the budget ends first, the evaluated
%       iterate with the lowest ||e|| (Eq. 29's norm, m and rad, before saturation). info.fitness
%       is that norm; info.err_pos (m) and info.err_rot (rad) come from its pose, as in bench_run.
%       Per update, info.sat marks Eq. 31 active (linear, rotational) and info.n_pinv counts the
%       solves that took the pseudoinverse. solve_QuIK('ops') returns handles to the operators for
%       unit tests.
%
%   Deviations from the paper:
%     - Stopping test: the benchmark pose gate replaces Eq. 29 with its example-task accuracy, and
%       the FK budget replaces the 200-iteration cap; when the gate is not met, the evaluated
%       iterate with the lowest ||e|| is returned.
%     - Every iterate clamped to the joint limits (the paper states no handling).
%     - Pseudoinverse where the Cholesky factorisation of Eq. 24 fails (not stated in the paper).
%     - The nearly-zero thresholds of Eqs. 37-38 from ref. [29], and the reading of the linear
%       error and of the product H dq (see Readings).
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
        q_best = struct('params', quik_params(), 'core', @quik_core, 'step', @quik_step, ...
                        'error', @quik_error, 'logR', @rot_log, 'hess', @hess_product, ...
                        'sat', @saturate, 'lsolve', @min_norm_solve);
        info = [];
        return
    end
    [q_best, info] = quik_core(robot, T_d, q0, opts, quik_params());
end

% ==================================================================== main loop
function [q_best, info] = quik_core(robot, T_d, q0, opts, P)
% Eq. 12 with the QuIK update of Eq. 17, from q_0 = q0 until the benchmark pose gate is met or the
% FK budget ends.
    t_start = tic;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    q  = min(max(q0(:), lo), hi);

    [tp, tr] = pose_gate(opts);

    q_best = q;  f_best = NaN;  T_best = NaN(4);
    fk = 0;  jc = 0;  it = 0;  conv = false;  stop = 'budget';
    hist = zeros(1, 0);  sat = false(2, 0);  npv = zeros(1, 0);
    while budget_left(opts, fk) >= 1
        Tq = robot.fk(q);
        fk = fk + 1;
        e  = quik_error(Tq, T_d, P);                                  % Eqs. 11, 34-38 (m, rad)
        f  = norm(e);
        hist(fk) = f;
        if fk == 1 || f < f_best
            f_best = f;  q_best = q;  T_best = Tq;
        end
        [~, ep, er] = pose_error(T_d, Tq);
        if norm(ep) < tp && norm(er) < tr                             % benchmark pose gate
            f_best = f;  q_best = q;  T_best = Tq;
            conv = true;  stop = 'gate';
            break
        end
        if budget_left(opts, fk) < 1, break; end                      % q_{k+1} not evaluable
        J  = robot.jac(q);                                            % Eqs. 15-16
        jc = jc + 1;
        [dq, d] = quik_step(J, e, P);                                 % Eqs. 31, 18, 23, 17
        q  = min(max(q + dq, lo), hi);                                % Eq. 12; clamp
        it = it + 1;
        sat(:, it) = d.sat;
        npv(it) = d.n_pinv;
    end

    ep = NaN;  er = NaN;
    if fk > 0
        [~, e_p, e_r] = pose_error(T_d, T_best);
        ep = norm(e_p);
        er = norm(e_r);
    end

    info = struct();
    info.solver    = 'QuIK';
    info.method    = 'QuIK, Eqs. 17-18 with H dq_nr of Eq. 23, saturation Eq. 31, error Eqs. 34-38';
    info.iters     = it;                                              % updates applied
    info.time      = toc(t_start);
    info.converged = conv;                                            % pose gate met
    info.fitness   = f_best;                                          % ||e||, m and rad
    info.err_pos   = ep;                                              % m
    info.err_rot   = er;                                              % rad
    info.fk_calls  = fk;
    info.jac_calls = jc;
    info.stop      = stop;                                            % 'gate' or 'budget'
    info.gate      = [tp, tr];                                        % tol_pos (m), tol_rot (rad)
    info.history   = hist;                                            % ||e(q_k)||, k = 0..fk-1
    info.sat       = sat;                                             % Eq. 31 active: [linear; rotational]
    info.n_pinv    = npv;                                             % solves by pseudoinverse (0-2)
    info.params    = P;
end

% ==================================================================== operators
function P = quik_params()
% d_lin (340 mm, in m) and d_rot (rad) of Eq. 31 for QuIK (p.7); nearly-zero thresholds of
% Eqs. 37-38 (ref. [29]).
    P = struct('d_lin', 0.340, 'd_rot', 1.00, 'tol37', 1e-3, 'tol38', 1e-10, 'trace_split', -0.99);
end

function e = quik_error(T_n, T_d, P)
% Eqs. 10-11 and 34-38: e = [p_n - p_d (m); w(R_n R_d^T) (rad)].
    R_e = T_n(1:3, 1:3) * T_d(1:3, 1:3).';
    e   = [T_n(1:3, 4) - T_d(1:3, 4); rot_log(R_e, P)];
end

function [w, branch] = rot_log(R, P)
% Angle-axis vector of R: Eq. 35, then Eq. 36, 37 or 38 (p.12).
    ep = [R(3,2) - R(2,3); R(1,3) - R(3,1); R(2,1) - R(1,2)];        % Eq. 35
    ne = norm(ep);
    t  = R(1,1) + R(2,2) + R(3,3);
    if t > P.trace_split && ne < P.tol37
        w = (3/4 - t / 12) * ep;                                      % Eq. 37
        branch = 37;
    elseif t <= P.trace_split && ne <= P.tol38
        w = (pi / 2) * ([R(1,1); R(2,2); R(3,3)] + 1);                % Eq. 38
        branch = 38;
    else
        w = (atan2(ne, t - 1) / ne) * ep;                             % Eq. 36
        branch = 36;
    end
end

function [es, act] = saturate(e, P)
% Eq. 31 applied separately to the linear (d_lin, m) and rotational (d_rot, rad) parts.
    es  = e;
    act = false(2, 1);
    nl  = norm(e(1:3));
    if ~(nl < P.d_lin)
        es(1:3) = P.d_lin * e(1:3) / nl;
        act(1) = true;
    end
    nr  = norm(e(4:6));
    if ~(nr < P.d_rot)
        es(4:6) = P.d_rot * e(4:6) / nr;
        act(2) = true;
    end
end

function HD = hess_product(J, dq)
% [H dq] of Eq. 17 for revolute joints: the entries of Eq. 23 from the columns of J, contracted with
% dq as in Eq. 4 (page i of H weighted by dq_i). C*(i,j) is component * of j_wi x j_vj (linear) or
% of j_wi x j_wj (angular).
    Jv = J(1:3, :);
    Jw = J(4:6, :);
    HD = zeros(size(J));
    for c = 1:3
        a = mod(c, 3) + 1;                                            % cross product indices
        b = mod(c + 1, 3) + 1;
        Cv = Jw(a, :).' * Jv(b, :) - Jw(b, :).' * Jv(a, :);
        Cw = Jw(a, :).' * Jw(b, :) - Jw(b, :).' * Jw(a, :);
        Dv = triu(Cv) + triu(Cv, 1).';                                % i <= j: C(i,j); i > j: C(j,i)
        Dw = triu(Cw, 1);                                             % i < j only
        HD(c, :)     = dq.' * Dv;
        HD(c + 3, :) = dq.' * Dw;
    end
end

function [x, pv] = min_norm_solve(A, b)
% Eq. 24: x = A^T (A A^T)^-1 b with a Cholesky factorisation of A A^T; the Moore-Penrose
% pseudoinverse where that factorisation fails.
    [R, p] = chol(A * A.');
    if p == 0
        [y, ~] = linsolve(R.', b, struct('LT', true));                % R^T y = b
        [z, ~] = linsolve(R, y, struct('UT', true));                  % R z = y
        x  = A.' * z;
        pv = false;
    else
        x  = pinv(A) * b;
        pv = true;
    end
end

function [dq, d] = quik_step(J, e, P)
% One QuIK update from J and the error e: Eq. 31, Eq. 18, Eq. 23 and Eq. 17.
    [es, act] = saturate(e, P);                                       % Eq. 31
    [x1, p1]  = min_norm_solve(J, es);
    dq_nr     = -x1;                                                  % Eq. 18
    HD        = hess_product(J, dq_nr);                               % Eq. 23 with Eq. 4
    A         = J + 0.5 * HD;
    [x2, p2]  = min_norm_solve(A, es);
    dq        = -x2;                                                  % Eq. 17
    d = struct('e_sat', es, 'sat', act, 'dq_nr', dq_nr, 'HD', HD, 'A', A, 'n_pinv', p1 + p2);
end

function [tp, tr] = pose_gate(opts)
% The benchmark pose gate, as in bench_run: position < tol_pos (m) and rotation < tol_rot (rad).
    tp = 1e-5;  tr = 1e-5;
    if isfield(opts, 'tol_pos'), tp = opts.tol_pos; end
    if isfield(opts, 'tol_rot'), tr = opts.tol_rot; end
end

function n = budget_left(opts, fk)
% FK calls still allowed on this waypoint: ikb_budget_left inside the harness, opts.budget minus
% the solver's own count outside it.
    n = ikb_budget_left(opts);
    if isinf(n) && isfield(opts, 'budget') && ~isempty(opts.budget)
        n = opts.budget - fk;
    end
end
