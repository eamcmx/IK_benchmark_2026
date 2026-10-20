function [q_best, info] = solve_DLS(robot, T_d, q0, opts)
% SOLVE_DLS  Modified Levenberg-Marquardt (mLM) IK method of Dulęba and Opałka (2013), Eq. 8.
%
%   Source:
%     Ignacy Dulęba, Michał Opałka (2013). A comparison of Jacobian-based methods of inverse
%     kinematics for serial robot manipulators. Int. J. Appl. Math. Comput. Sci., 23(2), 373-382.
%     DOI: 10.2478/amcs-2013-0028
%   Page numbers are PDF pages 1-10 (printed page = PDF page + 372).
%   The benchmark name DLS is kept for the file, the function and info.solver. The method is not
%   damped least squares: it has no damping term. The damped least-squares update
%   J^T (J J^T + lambda^2 I_m)^{-1} e (JD, Eq. 9, PDF p.3 of Adrià Colomé, Carme Torras (2015).
%   Closed-Loop Inverse Kinematics for Redundant Robots: Comparative Assessment and Two
%   Enhancements. IEEE/ASME Transactions on Mechatronics, 20(2), 944-955. DOI:
%   10.1109/TMECH.2014.2326304) is a different method and is not implemented here; that paper
%   does not treat mLM (its methods are listed in its Table II, PDF p.2).
%
%   Method (Sec. 2, pp.2-4):
%     Iteration from a known initial configuration q_0 (p.2) towards the goal x_f of the task
%     space, x = k(q) (Eq. 1), J = dk/dq (Eq. 2):
%       q_{i+1} = q_i + xi_i J^T(q_i) (diag(M(q_i)))^{-1} (x_f - k(q_i))            (Eq. 8)
%       M(q_i) = J(q_i) J^T(q_i)                                                    (Eq. 4)
%     diag(M) is the (m x m) matrix composed of the main diagonal elements of M (p.2). The
%     paper names the method the modified Levenberg-Marquardt one (mLM, p.3). It is Eq. 6,
%       q_{i+1} = q_i + xi_i J^T(q_i) (M(q_i) + lambda A)^{-1} (x_f - k(q_i))         (Eq. 6)
%     in the Levenberg-Marquardt version A = diag(M), with M omitted and lambda merged into
%     the gain: the xi_i of Eq. 8 is the xi_i of Eq. 6 times lambda (p.2). In the notation
%     of Eqs. 15-16 (p.4) the update is q_{i+1} = q_i + xi_i J^T B (x_f - k(q_i)) with the
%     weighting B = (diag(M(q_i)))^{-1}.
%     xi_i is "a given (or appropriately varied from one iteration to another) positive
%     coefficient" (p.2, under Eq. 3).
%
%   Parameters:
%     Step length 0.0005   Sec. 3.2 (p.8) restricts "the length of a single step motion" to 0.0005
%                          for all tasks and methods. The space in which the length is measured
%                          is not stated (checked pages 2-9). It is read as the joint-space length
%                          ||q_{i+1} - q_i|| (rad), with xi_i varied so that every step has it:
%                            xi_i = 0.0005 / ||J^T(q_i) B (x_f - k(q_i))||.
%                          A replay of Tasks 1-3 of p.8 (Eqs. 18 and 20, p.5; Task 3 with the
%                          first five of the six printed joint values, Eq. 20 having five
%                          joints) with this reading gives 9370, 7042, 1764 mLM updates;
%                          Table 6 (p.9) lists 9371, 7043, 1773 iterations, i.e. the updates
%                          plus the final evaluation of k (exact for Tasks 1 and 2, Theta in
%                          rad). A task-space step length of 0.0005 gives 28842, 22674, 1448
%                          updates and a constant xi_i = 0.0005 gives 50786, 123362, 4076.
%                          A cap ||q_{i+1} - q_i|| <= 0.0005 gives the same steps whenever the
%                          gain makes every uncapped step longer than 0.0005 before the goal is
%                          reached (as with xi_i = 1 in the replay); with a cap the result would
%                          depend on a constant xi_i the paper does not state. Near the
%                          benchmark's pose gate the two readings differ: a cap shortens the
%                          steps once ||xi_i J^T B e|| < 0.0005.
%     xi_i                 No value is stated (checked pages 2-4, 6, 8); set by the step length.
%     epsilon = 0.1        Sec. 3.2 (p.8) sets "the accuracy of reaching the goal x_f" to
%                          epsilon = 0.1 for all tasks: ||x_f - k(q_i)|| <= 0.1, || . || the
%                          Euclidean metrics (p.4), over all task coordinates (Tasks 1 and 2
%                          combine lengths and the angle Theta in one norm; the inequality is
%                          not printed, <= is used). It is the goal accuracy of the paper's
%                          three tasks, a setting of the paper's example tasks rather than part
%                          of the update law (Eqs. 7-8 with the 0.0005 step). The goal accuracy
%                          of the benchmark task is its pose gate, which is the stopping test
%                          here (see Stop). epsilon = 0.1 is kept for the replay of Table 6
%                          (params.eps_paper). Units: metres (the PUMA task, p.5 and p.8) and
%                          radians.
%     Task-space error     x_f - k(q). The paper places x in SE(3) (Eq. 1) but defines no
%                          orientation coordinates; its examples are (x, y, Theta), (x, z, phi)
%                          and positions (Eqs. 18-20, p.5). Here e = [p_d - p; w], w the rotation
%                          vector (axis x angle) of R_d R^T (pose_error.m), with the geometric
%                          Jacobian robot.jac (m = 6).
%
%   Benchmark interface:
%     Budget: an iteration evaluates k(q_i) (1 robot.fk call), applies the stopping test and,
%       if the test fails and the budget can still evaluate q_{i+1}, calls robot.jac once (not
%       budgeted, counted by the harness) and applies Eq. 8. An iteration starts only if
%       ikb_budget_left(opts) >= 1. Per waypoint: FK = updates + 1 <= 2000, Jacobian calls =
%       updates <= 1999, so at most 1999 updates and 0.9995 rad of joint travel. Outside the
%       harness the solver counts its own FK calls against opts.budget. opts.max_iter and the
%       iteration counts of Table 6 are not used. No quantity varies over the iteration count,
%       so ikb_generations is not used.
%     Stop: the benchmark pose gate, position error < opts.tol_pos (m) and rotation error <
%       opts.tol_rot (rad) of pose_error, the test of bench_run (1e-5 each when a field is
%       absent), or the budget. info.converged is the gate test. Through the operator handle
%       (solve_DLS('ops'), params.eps_stop = 0.1) the paper's test ||x_f - k(q_i)|| <= epsilon
%       ends the iteration instead (Table 6 setting); the solver reads no option for it.
%     Floor: every step is 0.0005 rad long, so near the target the iterate overshoots by about
%       the task-space size of one step (median 3.3e-4 m and 7.7e-4 rad per step on the UR5,
%       Synthetic13 and IRB120 rectangle waypoints 1-5) and does not settle below it. The gate
%       is met only when an iterate lands inside it (6.3 % of 1800 waypoints: 6 robots x
%       rectangle, circle, lissajous x 100). Otherwise the solver uses the full budget (2000 FK,
%       1999 Jacobian calls) and returns its best iterate (median 3.7e-5 m and 6.8e-5 rad on
%       those waypoints).
%     Warm start: q_0 = q0 (the previous waypoint's solution), clamped to the limits.
%     Units: metres and radians, as the paper's PUMA task (p.5, p.8); no conversion.
%     Joint limits: the paper restricts joint ranges only for its random configurations (p.5)
%       and states no handling during the iteration (checked pages 2-9); every iterate is
%       clamped to [q_min, q_max]. A clamped step is shorter than 0.0005.
%     Output: the iterate that meets the stopping test; if the budget ends first, the evaluated
%       iterate with the lowest ||x_f - k(q)||. info.fitness is that norm; info.err_pos (m) and
%       info.err_rot (rad) come from its pose, as in bench_run.
%       solve_DLS('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - 6-D error with the rotation vector of R_d R^T and the geometric Jacobian (the paper
%       defines no orientation coordinates in SE(3)).
%     - Joint-space reading of the 0.0005 step length (space not stated).
%     - Every iterate clamped to the joint limits (not stated).
%     - Stopping test: the benchmark pose gate instead of the paper's epsilon = 0.1, the goal
%       accuracy of its example tasks (p.8).
%     - Stop at the FK budget when the gate is not met; the evaluated iterate with the lowest
%       error is returned.
%     - The convergence check advised on p.6 for motion planning (one Jacobian transpose
%       iteration when the angle of Eq. 15 exceeds 90 deg) is not implemented: it lies outside
%       Eq. 8, and its trigger needs the pseudo-inverse direction M^{-1} (x_f - k(q)) of Eq. 14,
%       the inversion mLM avoids.
%     - A zero diagonal element of M (a zero row of J) contributes 0 to the update: the column
%       of J^T it multiplies is zero (IEEE arithmetic would give 0 x Inf = NaN).
%     - J^T B (x_f - k(q_i)) = 0 admits no step of length 0.0005 for any xi_i; the iteration
%       stops there (not stated).
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
        q_best = struct('params', mlm_params(), 'core', @mlm_core, 'step', @mlm_step, ...
                        'error', @mlm_error);
        info = [];
        return
    end
    [q_best, info] = mlm_core(robot, T_d, q0, opts, mlm_params());
end

% ==================================================================== main loop
function [q_best, info] = mlm_core(robot, T_d, q0, opts, P)
% Eq. 8 iterated from q_0 = q0 until the benchmark pose gate is met or the FK budget ends.
    t_start = tic;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    q  = min(max(q0(:), lo), hi);

    [tp, tr] = pose_gate(opts);
    if isempty(P.eps_stop), hit_name = 'gate'; else, hit_name = 'epsilon'; end

    q_best = q;  f_best = NaN;  T_best = NaN(4);
    fk = 0;  jc = 0;  it = 0;  conv = false;  stop = 'budget';
    hist = zeros(1, 0);  xis = zeros(1, 0);
    while budget_left(opts, fk) >= 1
        Tq = robot.fk(q);
        fk = fk + 1;
        e  = mlm_error(T_d, Tq);                                      % x_f - k(q_i)
        f  = norm(e);
        hist(fk) = f;
        if fk == 1 || f < f_best
            f_best = f;  q_best = q;  T_best = Tq;
        end
        if isempty(P.eps_stop)
            hit = norm(e(1:3)) < tp && norm(e(4:6)) < tr;            % benchmark pose gate
        else
            hit = f <= P.eps_stop;                                    % ||x_f - k(q)|| <= epsilon (p.8)
        end
        if hit                                                        % return the iterate that meets it
            f_best = f;  q_best = q;  T_best = Tq;
            conv = true;  stop = hit_name;
            break
        end
        if budget_left(opts, fk) < 1, break; end                      % q_{i+1} not evaluable
        J  = robot.jac(q);
        jc = jc + 1;
        [dq, xi] = mlm_step(J, e, P.step);                            % Eq. 8
        if ~isfinite(xi)
            stop = 'stationary';
            break
        end
        q  = min(max(q + dq, lo), hi);                                % clamp (not stated)
        it = it + 1;
        xis(it) = xi;
    end

    ep = NaN;  er = NaN;
    if fk > 0
        [~, e_p, e_r] = pose_error(T_d, T_best);
        ep = norm(e_p);
        er = norm(e_r);
    end

    info = struct();
    info.solver    = 'DLS';
    info.method    = 'mLM, Eq. 8 of Dulęba and Opałka (2013)';
    info.iters     = it;                                              % updates (Eq. 8 applied)
    info.time      = toc(t_start);
    info.converged = conv;                                            % stopping test met
    info.fitness   = f_best;                                          % ||x_f - k(q)||, m and rad
    info.err_pos   = ep;                                              % m
    info.err_rot   = er;                                              % rad
    info.fk_calls  = fk;
    info.jac_calls = jc;
    info.stop      = stop;                                            % 'gate', 'budget', 'stationary'
    info.gate      = [tp, tr];                                        % tol_pos (m), tol_rot (rad)
    info.history   = hist;                                            % ||x_f - k(q_i)||, i = 0..fk-1
    info.xi        = xis;                                             % xi_i of each update
    info.params    = P;
end

% ==================================================================== operators
function P = mlm_params()
% Sec. 3.2 (p.8): step length 0.0005; epsilon = 0.1, the goal accuracy of the paper's tasks (kept
% for the replay of Table 6). eps_stop empty: stop at the benchmark pose gate; a number (operator
% handle use only): stop at ||x_f - k(q)|| <= eps_stop.
    P = struct('step', 0.0005, 'eps_paper', 0.1, 'eps_stop', []);
end

function [dq, xi] = mlm_step(J, e, s)
% Eq. 8: dq = xi_i J^T (diag(M))^{-1} e with M = J J^T (Eq. 4) and xi_i = s / ||J^T (diag(M))^{-1} e||,
% so that ||dq|| = s. xi = Inf (and dq = 0) when J^T (diag(M))^{-1} e = 0.
    m = diag(J * J.');                                                % main diagonal of M
    w = e ./ m;                                                       % (diag(M))^{-1} e
    w(m == 0) = 0;                                                    % zero row of J
    d = J.' * w;
    nd = norm(d);
    if nd == 0
        xi = Inf;
        dq = zeros(size(d));
        return
    end
    xi = s / nd;
    dq = xi * d;
end

function e = mlm_error(T_d, T)
% x_f - k(q): position difference (m) and rotation vector of R_d R^T (rad), pose_error.m.
    e = pose_error(T_d, T);
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
