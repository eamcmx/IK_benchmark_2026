function [q_best, info] = solve_PINV(robot, T_d, q0, opts)
% SOLVE_PINV  Singular value filtering with selective damping (SVF + SD) of Colomé and Torras (2015).
%
%   Source:
%     Adrià Colomé, Carme Torras (2015). Closed-Loop Inverse Kinematics for Redundant Robots:
%     Comparative Assessment and Two Enhancements. IEEE/ASME Transactions on Mechatronics, 20(2),
%     944-955. DOI: 10.1109/TMECH.2014.2326304
%   Page numbers are PDF pages 1-12 (printed page = PDF page + 943).
%   The benchmark name PINV (label Lit:PINV) is kept for the file, the function and info.solver.
%   The plain Moore-Penrose iteration (JP, Eq. 4, p.3) is the baseline solve_pinv_orig. This file
%   is the paper's first enhancement, the SVF filter (Sec. IV, pp.4-5), combined with selective
%   damping (SD) as the paper evaluates it (Table V, row "SVF + SD", p.10) and recommends it
%   (conclusion 2, p.10: SD + SVF keeps the steps smooth, also for a singular goal; the 4R test
%   "SVF+SD", p.5 and Fig. 3, p.6).
%
%   Method:
%     Closed-loop update theta^{k+1} = theta^k + Delta theta^k (Eq. 1, p.1). The gain alpha of
%     Eq. 1 is omitted from every later update law (p.2). e is the pose error and J the geometric
%     Jacobian (notation table, p.2) of the m joints; n = 6 task coordinates.
%     SVF (Sec. IV, pp.4-5):
%       J = U Sigma V^T = sum_i sigma_i u_i v_i^T                                        (Eq. 15)
%       J_hat = sum_i h(sigma_i) u_i v_i^T                                                (Eq. 16)
%       h(sigma) = (sigma^3 + nu sigma^2 + 2 sigma + 2 sigma_0) / (sigma^2 + nu sigma + 2) (Eq. 17)
%       J_hat^+ = sum_{i=1..n} (1 / h(sigma_i)) v_i u_i^T, used as the pseudoinverse       (Eq. 18)
%     Every singular value is filtered (no threshold): h(0) = sigma_0, h(sigma) - sigma -> 0 as
%     sigma -> inf, h monotone for nu > sigma_0 and 2 > nu sigma_0 (properties 1-4, pp.4-5), so
%     the condition number of J_hat^+ stays below h(sigma_1) / sigma_0 (Eqs. 19-20, p.5).
%     SD (Sec. III-A, p.3; written as equations in Sec. VII, p.8) applied to the SVF inverse.
%     Table V lists SVF + SD among the methods not concerned with joint limits (p.10), so no
%     joint is activated (H = 0): the term (I_m - J^((I_m-H)(+)) J) H(-lambda theta) of Eq. 45 is
%     zero, and J^((I_m-H)(+)) is J^+ (Eq. 28 with every activation 1, p.7), here in its
%     "filtered version with SVF" (p.8), J_hat^+ of Eq. 18: in Eq. 37 hat sigma_s = h(sigma_s),
%     hat V = V, hat U = U. For s = 1..n:
%       M_s = hat sigma_s^{-1} sum_j |v_{j,s}| ||J^j||,  J^j the j-th column of J         (Eq. 41)
%       gamma_s = min(1, 1 / M_s) gamma_max                                              (Eq. 42)
%       w_s = hat sigma_s^{-1} v_s (u_s^T e)                                              (Eq. 43)
%       Delta q_s = w_s if ||w_s|| < gamma_s, else (w_s / ||w_s||) gamma_s               (Eq. 44)
%     Delta theta_hat = sum_s Delta q_s                                           (Eq. 45, H = 0)
%     Delta theta = Delta theta_hat if ||Delta theta_hat|| < gamma_max,
%                   else (Delta theta_hat / ||Delta theta_hat||) gamma_max               (Eq. 46)
%     Where neither bound is active the update is J_hat^+ e (Eqs. 18 and 38). The update has no
%     null-space term: it lies in the span of v_1..v_n.
%
%   Parameters:
%     nu = 10          p.5 (the value used throughout the paper) and Table V (p.10).
%     sigma_0 = 0.01   Table V (p.10), SVF row, the setting of the paper's WAM runs. Fig. 2 (p.5)
%                      uses 0.005 for a 4R illustration only.
%     gamma_max = 0.5  Table V (p.10), SD row. The SVF + SD row repeats no values; the SVF row's
%                      nu and sigma_0 and the SD row's gamma_max are used. Unit rad: gamma_max is
%                      the maximum joint variation (p.2) of the revolute WAM joints (Table IV, p.9).
%     alpha = 1        Not stated as a number (checked pages 1-12). Eq. 1 has a gain alpha that
%                      the paper omits from every later update law (p.2); the laws are applied as
%                      printed, with unit gain, and the step size is set by Eqs. 44 and 46.
%     Iterations       Not stated (checked pages 1-12); conclusion 1 (p.10) mentions a limited
%                      number of iterations for SD without a value. Not used (see Stop).
%     Stopping test    Not stated (checked pages 1-12); the benchmark pose gate (see Stop).
%     Norm ||.||       Eqs. 41, 44 and 46 do not name the norm (p.8). Read as the Euclidean norm,
%                      the norm the paper calls the common one (p.5, text of Eq. 21).
%     Eqs. 44 and 46   The first case is printed as "1" for a vector quantity (p.8), which is
%                      undefined. Read as the unbounded vector (w_s, Delta theta_hat): the
%                      nearest well-defined reading, equal to the second case at the bound.
%     Error e          The paper takes its orientation error from its ref. [36] without printing
%                      it and notes that the equivalence 2 rad = 1 m is often taken (p.2); its
%                      reported residual uses that position-orientation metric (p.10). Whether
%                      the weighting enters e is not stated (checked pages 1-12). Here
%                      e = [p_d - p; w], w the rotation vector (axis x angle) of R_d R^T
%                      (pose_error.m, with its branch for angles near pi), unweighted
%                      (1 rad = 1 m), with the geometric Jacobian robot.jac.
%
%   Benchmark interface:
%     Budget: an iteration evaluates the pose of q_k (1 robot.fk call), applies the stopping
%       test and, if the test fails and the budget can still evaluate q_{k+1}, calls robot.jac
%       once (not budgeted, counted by the harness) and applies Eqs. 15-18 and 41-46. An
%       iteration starts only if ikb_budget_left(opts) >= 1. Per waypoint: FK = updates + 1 <=
%       2000, Jacobian calls = updates <= 1999. The pose that robot.jac also returns is not used:
%       the budget counts robot.fk calls, so every evaluated iterate is charged one FK call, and
%       the Jacobian is computed only when an update follows. Outside the harness the solver
%       counts its own FK calls against opts.budget. opts.max_iter is not used. No quantity
%       varies over the iteration count, so ikb_generations is not used.
%     Stop: the benchmark pose gate, position error < opts.tol_pos (m) and rotation error <
%       opts.tol_rot (rad) of pose_error, the test of bench_run (1e-5 each when a field is
%       absent), or the budget. info.converged is the gate test.
%     Warm start: q_0 = q0 (the previous waypoint's solution), clamped to the limits.
%     Units: metres and radians, as the paper's WAM table (Table IV, p.9); no conversion.
%     Joint limits: SVF + SD is one of the methods not concerned with joint limits (Table V,
%       p.10), and the paper states no handling for them (its joint-limit methods, Sec. VI,
%       pp.7-8, are other methods). Every iterate is clamped to [q_min, q_max].
%     Output: the iterate that meets the stopping test; if the budget ends first, the evaluated
%       iterate with the lowest ||e||. info.fitness is that norm; info.err_pos (m) and
%       info.err_rot (rad) come from its pose, as in bench_run. Per update, info.n_sd counts the
%       directions bounded by Eq. 44 and info.bound46 marks Eq. 46 active.
%       solve_PINV('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Stopping test: the benchmark pose gate (the paper states none); stop at the FK budget
%       when the gate is not met, returning the evaluated iterate with the lowest ||e||.
%     - Error vector: unweighted rotation vector of R_d R^T through pose_error.m (the paper's
%       orientation error is not printed; the 2 rad = 1 m weighting is not stated to enter e).
%     - Every iterate clamped to the joint limits (the paper's SVF + SD does not consider them).
%     - Euclidean norm in Eqs. 41, 44 and 46, and the reading of the first case of Eqs. 44 and
%       46 (see Parameters).
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
        q_best = struct('params', svfsd_params(), 'core', @svfsd_core, 'step', @svfsd_step, ...
                        'filter', @svf_filter, 'error', @svfsd_error);
        info = [];
        return
    end
    [q_best, info] = svfsd_core(robot, T_d, q0, opts, svfsd_params());
end

% ==================================================================== main loop
function [q_best, info] = svfsd_core(robot, T_d, q0, opts, P)
% Eq. 1 with the SVF + SD update, from q_0 = q0 until the benchmark pose gate is met or the FK
% budget ends.
    t_start = tic;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    q  = min(max(q0(:), lo), hi);

    [tp, tr] = pose_gate(opts);

    q_best = q;  f_best = NaN;  T_best = NaN(4);
    fk = 0;  jc = 0;  it = 0;  conv = false;  stop = 'budget';
    hist = zeros(1, 0);  nsd = zeros(1, 0);  b46 = false(1, 0);
    while budget_left(opts, fk) >= 1
        Tq = robot.fk(q);
        fk = fk + 1;
        e  = svfsd_error(T_d, Tq);                                    % pose error at q_k
        f  = norm(e);
        hist(fk) = f;
        if fk == 1 || f < f_best
            f_best = f;  q_best = q;  T_best = Tq;
        end
        if norm(e(1:3)) < tp && norm(e(4:6)) < tr                     % benchmark pose gate
            f_best = f;  q_best = q;  T_best = Tq;
            conv = true;  stop = 'gate';
            break
        end
        if budget_left(opts, fk) < 1, break; end                      % q_{k+1} not evaluable
        J  = robot.jac(q);
        jc = jc + 1;
        [dq, d] = svfsd_step(J, e, P);                                % Eqs. 15-18, 41-46
        q  = min(max(q + dq, lo), hi);                                % Eq. 1, alpha = 1; clamp
        it = it + 1;
        nsd(it) = nnz(d.bounded);
        b46(it) = d.bound46;
    end

    ep = NaN;  er = NaN;
    if fk > 0
        [~, e_p, e_r] = pose_error(T_d, T_best);
        ep = norm(e_p);
        er = norm(e_r);
    end

    info = struct();
    info.solver    = 'PINV';
    info.method    = 'SVF + SD, Eqs. 15-18 and 41-46 with H = 0 (Table V, SVF + SD)';
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
    info.n_sd      = nsd;                                             % directions bounded by Eq. 44
    info.bound46   = b46;                                             % Eq. 46 active
    info.params    = P;
end

% ==================================================================== operators
function P = svfsd_params()
% nu (p.5, Table V), sigma_0 (Table V, SVF row), gamma_max (Table V, SD row), p.10.
    P = struct('nu', 10, 'sigma0', 0.01, 'gamma_max', 0.5);
end

function h = svf_filter(sigma, nu, sigma0)
% Eq. 17: h(sigma) = (sigma^3 + nu sigma^2 + 2 sigma + 2 sigma_0) / (sigma^2 + nu sigma + 2).
    h = (sigma.^3 + nu * sigma.^2 + 2 * sigma + 2 * sigma0) ./ (sigma.^2 + nu * sigma + 2);
end

function [dq, d] = svfsd_step(J, e, P)
% One update: SVD (Eq. 15), filtered singular values (Eqs. 17, 37), per-direction bound
% (Eqs. 41-44), sum (Eq. 45 with H = 0) and total bound (Eq. 46). Euclidean norms.
    [U, S, V] = svd(J, 'econ');                                       % Eq. 15
    sig = diag(S);
    sh  = svf_filter(sig, P.nu, P.sigma0);                            % hat sigma_s = h(sigma_s)
    cn  = sqrt(sum(J.^2, 1)).';                                       % ||J^j||, Eq. 41
    ns  = numel(sig);
    M   = zeros(ns, 1);  g = zeros(ns, 1);  nw = zeros(ns, 1);  bnd = false(ns, 1);
    dqh = zeros(size(J, 2), 1);
    for s = 1:ns
        M(s)  = (abs(V(:, s)).' * cn) / sh(s);                        % Eq. 41
        g(s)  = min(1, 1 / M(s)) * P.gamma_max;                       % Eq. 42
        w     = V(:, s) * ((U(:, s).' * e) / sh(s));                  % Eq. 43
        nw(s) = norm(w);
        if nw(s) >= g(s)                                              % Eq. 44
            w = (w / nw(s)) * g(s);
            bnd(s) = true;
        end
        dqh = dqh + w;                                                % Eq. 45, H = 0
    end
    nt  = norm(dqh);
    b46 = nt >= P.gamma_max;
    if b46
        dq = (dqh / nt) * P.gamma_max;                                % Eq. 46
    else
        dq = dqh;
    end
    d = struct('sigma', sig, 'h', sh, 'M', M, 'gamma', g, 'norm_w', nw, 'bounded', bnd, ...
               'dtheta_hat', dqh, 'bound46', b46);
end

function e = svfsd_error(T_d, T)
% Pose error: position difference (m) and rotation vector of R_d R^T (rad), pose_error.m.
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
