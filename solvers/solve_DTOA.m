function [q_best, info] = solve_DTOA(robot, T_d, q0, opts)
% SOLVE_DTOA  Five-instant discretization (5IDF) Jacobian-pseudoinverse IK of Guo et al. (2022): DTRPP scheme, Eq. 16.
%
%   Source:
%     Dongsheng Guo, Aifen Li, Jianhuang Cai, Qingshan Feng and Yang Shi (2022). Inverse kinematics
%     of redundant manipulators with guaranteed performance. Robotica, 40, pp. 170-190 (no issue
%     number is printed). DOI: 10.1017/S026357472100045X
%   Page numbers are PDF pages 1-21 (printed page = PDF page + 169).
%   The benchmark name DTOA is kept for the file, the function and info.solver. The scheme run here is
%   the paper's discrete-time repetitive path planning (DTRPP) scheme, Eq. 16 (Sec. 3, p.5): its
%   JP-based IK method, Eq. 13 (Sec. 2.2, p.4), with the null-space choice z_k = gamma (theta_0 -
%   theta_k). The paper's discrete-time obstacle avoidance (DTOA) scheme, Eq. 18 (Sec. 4, p.7), is
%   Eq. 13 with an obstacle-avoidance term instead; it is not implemented (see Deviations).
%
%   Method (Secs. 2-3, pp.2-6):
%     Task phi(theta) = r_d (Eq. 1, p.1), r_d in R^m the end-effector desired path, J = d phi / d theta
%     (Eq. 2, p.2). Continuous JP-based IK method (Eq. 3, p.2), k > 0 the feedback gain:
%       theta_dot = J^+ (r_dot_d - k (phi(theta) - r_d)) + (I - J^+ J) z                     (Eq. 3)
%     5IDF (Lemma 1, p.3), sampling gap delta = t_{k+1} - t_k:
%       f_dot(t_k) = (18 f(t_{k+1}) + 5 f(t_k) - 18 f(t_{k-1}) - 9 f(t_{k-2}) + 4 f(t_{k-3}))
%                    / (42 delta) + O(delta^3)                                                 (Eq. 4)
%     Eq. 3 discretised with Eq. 4 (Eqs. 11-12, p.4), truncation error O(delta^4), step size
%     h = 7 delta k / 3 (p.4), k = 3, 4, ...:
%       theta_{k+1} = -(5/18) theta_k + theta_{k-1} + (1/2) theta_{k-2} - (2/9) theta_{k-3}
%                     + J_k^+ ((7/3) delta r_dot_dk - h (phi(theta_k) - r_dk))
%                     + (7/3) delta (I - J_k^+ J_k) z_k                                        (Eq. 13)
%     DTRPP (p.5): z_k = gamma (theta_0 - theta_k), the discrete form of the RPP choice
%     z = gamma (theta_0 - theta) of De Luca et al. (Eq. 15, p.5), theta_0 "the given initial joint
%     state", RPP parameter kappa = 7 delta gamma / 3:
%       theta_{k+1} = -(5/18) theta_k + theta_{k-1} + (1/2) theta_{k-2} - (2/9) theta_{k-3}
%                     + J_k^+ ((7/3) delta r_dot_dk - h (phi(theta_k) - r_dk))
%                     + kappa (I - J_k^+ J_k)(theta_0 - theta_k)                               (Eq. 16)
%     Start-up of Eq. 16 (p.6; Eq. 13 needs the four joint states theta_0 ... theta_3, p.4),
%     j = 0, 1, 2:
%       theta_{j+1} = theta_j + J_j^+ (delta r_dot_dj - h (phi(theta_j) - r_dj))
%                     + kappa (I - J_j^+ J_j)(theta_0 - theta_j)
%     Lemma 2 (p.4): zero stability, the roots of rho^4 + (5/18) rho^3 - rho^2 - (1/2) rho + 2/9 lie on
%     or in the unit circle. With kappa = 0, Eq. 16 is the DTMVN scheme, Eq. 19 (p.7).
%
%   Parameters:
%     h = 0.3        Step size of Eqs. 13 and 16. Every simulation of the paper uses h = 0.3 (Tables
%                    1-4, p.7-10; Figs. 1-12, p.8-19); Tables 5 and 6 (p.11, p.16) also list 0.2 and
%                    0.4. The feedback gain k of Eq. 3 is never given a value; h is set directly.
%     kappa = 0.3    RPP parameter of Eq. 16 (p.5) in every DTRPP simulation (Tables 2 and 4, p.8 and
%                    p.10; Figs. 2, 4, 5, 6, p.9-13); Table 5 (p.11) also lists 0.2 and 0.4. gamma of
%                    Eq. 15 is never given a value; kappa is set directly.
%     delta = 0.01   Sampling gap. The paper runs delta = 0.01 and 0.001 (Tables 1-4, Figs. 1-12) and
%                    also 0.1 (Tables 5-6), always over T = 10 s (p.7, p.14). Of these settings, 1000
%                    steps (delta = 0.01) is nearest the 2000-FK budget (10000 and 100 steps for the
%                    others). With h and kappa set directly, delta enters only the feed-forward terms
%                    (7/3) delta r_dot_dk and delta r_dot_dj, which are zero for a static target; it
%                    has no effect in the benchmark.
%     theta_0 = q0   The given initial joint state (p.5): the waypoint's warm start (see Warm start).
%     Task, m = 3    End-effector position only: "In the simulations, only the end-effector position
%                    is considered" and the UR "is viewed as a redundant one" (Sec. 5.1, p.7); the
%                    PA10 runs plot the errors epsilon_X, epsilon_Y, epsilon_Z only (pp.13-19). Here
%                    phi(theta) is the end-effector position, r_d the target position of T_d, and J
%                    the first three rows of robot.jac (the linear part of the geometric Jacobian).
%                    The paper's planning error epsilon_k = phi(theta_k) - r_dk (Lemma 3, p.5); its
%                    norm ||phi(theta) - r_d|| (m) is the fitness reported.
%     J^+            The pseudoinverse matrix of J (p.2), computed with MATLAB pinv (SVD, default
%                    tolerance); the paper does not say how it is computed.
%     Not used       The path duration T = 10 s (p.7, p.14); the OA thresholds d1 = 0.12 m and
%                    d2 = 0.05 m (p.14), the OA parameter varsigma = 7 delta lambda / 3 (-0.05, p.14;
%                    -0.02, p.17; -0.01 to -0.03, Table 6, p.16) and the maximum escape velocity v_m
%                    (p.6; value not stated, checked pages 1-19): Eq. 18 is not implemented.
%
%   Benchmark interface:
%     Static target: the paper tracks a moving path, one step of Eq. 16 per sampling instant
%       t_k = k delta (p.3), started once at t = 0 (pp.4, 6) and run over T = 10 s with no stopping
%       test (p.7, p.14). The benchmark gives one static pose per call, so r_dk = r_d and r_dot_dk = 0
%       for every k, and the recursion is started afresh at every waypoint (start-up of p.6 from
%       theta_0 = q0; no history is carried between waypoints). Eq. 16 then iterates towards one
%       target: linearised with J_k J_k^+ = I_3 and J_k (I - J_k^+ J_k) = 0, the start-up gives
%       e_j = (1 - h)^j e_0 and the position error then obeys
%       e_{k+1} = -(5/18 + h) e_k + e_{k-1} + (1/2) e_{k-2} - (2/9) e_{k-3}, the polynomial of Lemma 2
%       with h added, whose largest roots have modulus 0.907 at h = 0.3 (a sign-alternating complex
%       pair; real roots 0.879 and 0.307). The error therefore decays while oscillating, and the stop
%       fires at the first iterate below the tolerance. The O(delta^4) planning precision of Lemma 3
%       (p.5) is a path-tracking result and plays no role here.
%     Measured: on 6 robots x rectangle, circle and lissajous trajectories x 100 waypoints, the stop
%       is met on every waypoint: after 33 or 39 updates (FK 34 or 40) on every waypoint but the
%       first, and after 57-70 updates on the first; the budget is never reached.
%     Stop: the paper states no stopping test and its task is position-only, so the iteration stops at
%       the first evaluated iterate with ||phi(theta_k) - r_d|| < opts.tol_pos (m; 1e-5 when the field
%       is absent), the position part of the benchmark pose gate, or when the budget ends. Orientation
%       is not controlled: the rotation error of the result is whatever the position-only scheme leaves
%       (reported in info.err_rot), and opts.tol_rot is not read.
%     Budget: an iteration evaluates phi(theta_k) (1 robot.fk call), applies the stop and, if the
%       budget can still evaluate theta_{k+1}, calls robot.jac once (not budgeted, counted by the
%       harness) and applies a start-up step (the first three updates) or Eq. 16. An iteration starts
%       only if ikb_budget_left(opts) >= 1. Per waypoint: FK = updates + 1 <= 2000, Jacobian calls =
%       updates <= 1999, so at most 1999 updates (3 start-up steps + 1996 steps of Eq. 16). Outside
%       the harness the solver counts its own FK calls against opts.budget. opts.max_iter and the
%       paper's step count T / delta are not used. No quantity varies over the iteration count, so
%       ikb_generations is not used.
%     Warm start: theta_0 = q0 (the previous waypoint's solution), clamped to the limits; it is both
%       the first iterate and the theta_0 of the null-space term of Eq. 16, so the term pulls the
%       null-space motion back towards the warm start.
%     Units: metres and radians, as the paper (positions, d1 and d2 in m, pp.7-19); no conversion.
%     Joint limits: the paper does not treat them; extending Eq. 13 to "redundant manipulators with
%       joint constraint" is named as future work (Sec. 6, p.19). Every iterate is clamped to
%       [q_min, q_max], and the clamped value enters the history of Eq. 16.
%     Output: the iterate that meets the stop; if the budget ends first, the evaluated iterate with the
%       lowest ||phi(theta) - r_d||. info.fitness is that norm (m); info.err_pos (m) and info.err_rot
%       (rad) come from its pose, as in bench_run. solve_DTOA('ops') returns handles to the operators
%       for unit tests.
%
%   Deviations from the paper:
%     - Static target per call: r_dot_dk = 0, the recursion is restarted at every waypoint and iterated
%       to the stop (the paper tracks a moving path once over T = 10 s).
%     - Stop when the position error < opts.tol_pos or the FK budget ends (the paper has no stop).
%     - theta_0 of Eq. 16 is the waypoint's warm start q0, not the initial state of a closed path: a
%       solver call sees one waypoint.
%     - Every iterate clamped to the joint limits (not treated in the paper).
%     - The DTOA scheme, Eq. 18, is not implemented: the benchmark has no obstacles, so every escape
%       velocity v_Cij of p.6 would be 0 and Eq. 18 would reduce to the DTMVN scheme, Eq. 19 (p.7);
%       v_m is not stated. On its obstacle-free position task the paper studies the DTRPP scheme,
%       Eq. 16 (Sec. 5.1, p.7), with Eq. 19 only as a comparison; Eq. 16 is used here, and kappa = 0
%       in params gives Eq. 19.
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
        q_best = struct('params', dtrpp_params(), 'core', @dtrpp_core, 'startup', @startup_step, ...
                        'step', @idf5_step, 'error', @task_error, 'coeffs', idf5_coeffs());
        info = [];
        return
    end
    [q_best, info] = dtrpp_core(robot, T_d, q0, opts, dtrpp_params());
end

% ==================================================================== main loop
function [q_best, info] = dtrpp_core(robot, T_d, q0, opts, P)
% Start-up (p.6) and Eq. 16 from theta_0 = q0 on the static target r_d until the position error is
% below opts.tol_pos or the FK budget ends.
    t_start = tic;
    lo  = robot.q_min(:);
    hi  = robot.q_max(:);
    q   = min(max(q0(:), lo), hi);
    th0 = q;                                                          % theta_0 (p.5)
    rd  = T_d(1:3, 4);                                                % r_d, end-effector position (m = 3)
    rdd = zeros(3, 1);                                                % r_dot_d = 0 (static target)
    tp  = pos_gate(opts);
    H   = zeros(numel(q), 4);                                         % [theta_k, theta_{k-1}, theta_{k-2}, theta_{k-3}]
    H(:, 1) = q;

    q_best = q;  f_best = NaN;  T_best = NaN(4);
    fk = 0;  jc = 0;  it = 0;  conv = false;  stop = 'budget';
    hist = zeros(1, 0);
    while budget_left(opts, fk) >= 1
        Tq = robot.fk(q);
        fk = fk + 1;
        e  = task_error(rd, Tq);                                      % phi(theta_k) - r_dk
        f  = norm(e);
        hist(fk) = f;
        if fk == 1 || f < f_best
            f_best = f;  q_best = q;  T_best = Tq;
        end
        if f < tp                                                     % position part of the pose gate
            f_best = f;  q_best = q;  T_best = Tq;
            conv = true;  stop = 'gate';
            break
        end
        if budget_left(opts, fk) < 1, break; end                      % theta_{k+1} not evaluable
        J  = robot.jac(q);
        jc = jc + 1;
        Jv = J(1:3, :);                                               % d phi / d theta, position rows
        if it < 3
            qn = startup_step(q, Jv, e, rdd, th0, P);                 % theta_1, theta_2, theta_3 (p.6)
        else
            qn = idf5_step(H, Jv, e, rdd, th0, P);                    % Eq. 16
        end
        qn = min(max(qn, lo), hi);                                    % clamp (not treated in the paper)
        H  = [qn, H(:, 1:3)];                                         % shift the history
        q  = qn;
        it = it + 1;
    end

    ep = NaN;  er = NaN;
    if fk > 0
        [~, e_p, e_r] = pose_error(T_d, T_best);
        ep = norm(e_p);
        er = norm(e_r);
    end

    info = struct();
    info.solver    = 'DTOA';
    info.method    = 'DTRPP, Eq. 16 of Guo et al. (2022), end-effector position task';
    info.iters     = it;                                              % updates (start-up + Eq. 16)
    info.startup   = min(it, 3);                                      % start-up steps applied
    info.time      = toc(t_start);
    info.converged = conv;                                            % position error < tol_pos
    info.fitness   = f_best;                                          % ||phi(theta) - r_d||, m
    info.err_pos   = ep;                                              % m
    info.err_rot   = er;                                              % rad, not controlled
    info.fk_calls  = fk;
    info.jac_calls = jc;
    info.stop      = stop;                                            % 'gate' or 'budget'
    info.gate      = tp;                                              % tol_pos (m)
    info.history   = hist;                                            % ||phi(theta_k) - r_d||, one per FK call
    info.params    = P;
end

% ==================================================================== operators
function P = dtrpp_params()
% h and kappa of every DTRPP simulation (Tables 2 and 4, Figs. 2, 4-6); delta of the setting nearest
% the budget (no effect when r_dot_d = 0).
    P = struct('h', 0.3, 'kappa', 0.3, 'delta', 0.01);
end

function c = idf5_coeffs()
% Eq. 13 / Eq. 16 (p.4-5): theta_{k+1} = a(1) theta_k + a(2) theta_{k-1} + a(3) theta_{k-2}
% + a(4) theta_{k-3} + J_k^+ (b delta r_dot_dk - h e_k) + ...; Eq. 4 solved for f(t_{k+1}).
    c = struct('a', [-5/18, 1, 1/2, -2/9], 'b', 7/3);
end

function qn = startup_step(q, Jv, e, rdd, th0, P)
% Start-up of Eq. 16 (p.6): theta_{j+1} = theta_j + J_j^+ (delta r_dot_dj - h (phi(theta_j) - r_dj))
% + kappa (I - J_j^+ J_j)(theta_0 - theta_j).
    Jp = pinv(Jv);
    qn = q + Jp * (P.delta * rdd - P.h * e) + P.kappa * ((eye(numel(q)) - Jp * Jv) * (th0 - q));
end

function qn = idf5_step(H, Jv, e, rdd, th0, P)
% Eq. 16 (p.5) with H = [theta_k, theta_{k-1}, theta_{k-2}, theta_{k-3}] and e = phi(theta_k) - r_dk.
    c  = idf5_coeffs();
    q  = H(:, 1);
    Jp = pinv(Jv);
    qn = c.a(1) * H(:, 1) + c.a(2) * H(:, 2) + c.a(3) * H(:, 3) + c.a(4) * H(:, 4) ...
         + Jp * (c.b * P.delta * rdd - P.h * e) + P.kappa * ((eye(numel(q)) - Jp * Jv) * (th0 - q));
end

function e = task_error(rd, T)
% phi(theta) - r_d: end-effector position minus the desired position (m), Eqs. 3 and 16.
    e = T(1:3, 4) - rd;
end

function tp = pos_gate(opts)
% Position part of the benchmark pose gate, as in bench_run: position < tol_pos (m).
    tp = 1e-5;
    if isfield(opts, 'tol_pos'), tp = opts.tol_pos; end
end

function n = budget_left(opts, fk)
% FK calls still allowed on this waypoint: ikb_budget_left inside the harness, opts.budget minus
% the solver's own count outside it.
    n = ikb_budget_left(opts);
    if isinf(n) && isfield(opts, 'budget') && ~isempty(opts.budget)
        n = opts.budget - fk;
    end
end
