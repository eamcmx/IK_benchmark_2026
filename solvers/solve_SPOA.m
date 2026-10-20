function [q_best, info] = solve_SPOA(robot, T_d, q0, opts)
% SOLVE_SPOA  Stochastic Paint Optimizer IK solver of El Mrabet et al. (2025), Sec. III-C.
%
%   Source:
%     Aziz El Mrabet, Hicham Hihi, Mohammed Khalil Laghraib, Mbarek Chahboun, Mohcine Abouyaakoub,
%     Ali Ait Ali, Aymane Amalaoui (2025). Inverse Kinematics Optimization Using ACO, MOA, SPOA, and
%     ALO: A Benchmark Study on Industrial Robot Arms. Journal of Robotics and Control (JRC), 6(4),
%     1729-1745. DOI: 10.18196/jrc.v6i4.26809
%   Fitness weights only:
%     Aziz El Mrabet, Hicham Hihi, Mohammed Khalil Laghraib, Mbarek Chahboun, Aymane Amalaoui (2025).
%     Two-Stage Neural Network versus Ant Colony, Mayfly, Stochastic Paint, and Ant Lion Optimizers
%     for Robot Arm Inverse Kinematics. JRC, 6(6), 2715-2728. DOI: 10.18196/jrc.v6i6.27795
%
%   Algorithm (pp. 6-7, Eqs. 11-13; fitness p. 8, Eqs. 19-23):
%     Implemented from the text and Eqs. 11-13. The listing printed as Fig. 7 ("Pseudocode for the
%     SPO algorithm", p.6) is a Grey Wolf Optimizer listing (grey wolf population, a, A and C) and is
%     not used, apart from its definition of X_alpha, X_beta, X_delta as the best, second and third
%     best agents (l.4-6).
%     A population of N joint vectors (Eq. 11) is evaluated. After each generation the top k
%     individuals with minimum error are retained, and the population is replenished with N - k
%     candidates X_new = l1*X_alpha + l2*X_beta + l3*X_delta + delta (Eq. 13), l1 + l2 + l3 = 1,
%     l_i in [0, 1]. The next generation perturbs every solution, X <- X + eps*N(0, sigma^2)
%     (Eq. 12), and evaluates it. The run ends after T iterations or when the best solution's error
%     falls below eps_min (p.7).
%     Fitness: E = alpha_E*Ep + beta_E*Eo (Eq. 19), Ep = ||P_d - P_c|| (Eq. 21),
%     Eo = 2*acos(0.5*sqrt(n_x + o_y + a_z + 1)) with R_E = R_d*R_c' (Eqs. 22-23).
%
%   Parameters:
%     N      = 40    Table VI (p.7): population size 30-50; no value reported for the IK runs;
%                    midpoint.
%     k      = 8     Table VI: top-k selection 10-30% of N; midpoint 20% of 40.
%     lambda = three U(0,1) draws divided by their sum, drawn per new candidate. Table VI: "Random or
%                    fixed (sum = 1)"; the random option is used.
%     T      = 400   iterations. Table VI: 300-500; midpoint. Never reached within the FK budget.
%     eps = 0.1, sigma = 0.1. p.6 defines eps as "a learning or perturbation rate" and N(0, sigma^2)
%                    as Gaussian noise of variance sigma^2, with no values (pp. 6-7, Table VI
%                    checked). Benchmark uses 0.1 and 0.1: only the product matters, a standard
%                    deviation of 0.01 rad per joint, small against every joint range in the
%                    benchmark (the smallest is 2.79 rad, IRB 120 joint 3), as "controlled noise"
%                    requires.
%     delta  = eps*N(0, sigma^2) per joint. p.7: "a small random vector (mutation)", no magnitude;
%                    the text replenishes "via interpolation of top solutions and perturbation", and
%                    the perturbation the paper defines is Eq. 12's, so delta uses it.
%     eps_min = min(1000*opts.tol_pos, opts.tol_rot) in units of E. p.7 names the threshold without a
%                    value. With this value E < eps_min implies a position error below tol_pos (m)
%                    and a rotation error below tol_rot (rad), the benchmark's success test.
%     "clustering-based selection": not defined in the paper (pp. 6-7); the top k are taken by
%                    rank of E.
%     alpha_E = beta_E = 1 (Eq. 19). Not valued in this paper (p.8); the companion paper states
%                    E = Ep + EO (JRC 6(6), p.5, Eq. 9).
%
%   Benchmark interface:
%     Budget: initialisation and every generation need N FK calls; a generation starts only if
%       ikb_budget_left(opts) covers it. No final FK check. opts.max_iter is ignored.
%     Warm start: q0 (clamped to the limits) is member 1 of the initial population, which is
%       evaluated unperturbed and enters the first top-k selection.
%     Units: benchmark lengths are metres; the paper's robots are in mm (Tables I-III, pp. 3-4), so
%       Ep is computed in mm: E = 1000*||P_d - P_c||[m] + Eo[rad].
%     Joint limits: Eq. 20 bounds theta; the paper gives no handling rule, so every new position
%       (Eq. 12 and Eq. 13) is clamped to [q_min, q_max].
%     Output: the lowest-E solution ever evaluated. info.converged = best E < eps_min (p.7).
%       solve_SPOA('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Fig. 7 is not used (see above).
%     - The selection is applied after every generation including the initial one (generation 0),
%       so the first generation perturbs the selected and replenished population; the text's order
%       (perturb, select top k, replenish) is kept from then on.
%     - The colour-wheel, "beauty index" and colour-combination prose (p.6) define no operator;
%       nothing is implemented for them.
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
        q_best = struct('fitness', @spo_fitness, 'select', @spo_select, ...
                        'replenish', @spo_replenish, 'perturb', @spo_perturb, ...
                        'eps_min', @spo_eps_min, 'params', spo_params());
        info = [];
        return
    end

    t_start = tic;
    P = spo_params();
    N = P.N;

    n     = robot.n;
    q_min = robot.q_min(:);
    q_max = robot.q_max(:);
    q0    = min(max(q0(:), q_min), q_max);
    eps_min = spo_eps_min(opts);

    q_best = q0;  best_E = NaN;  best_ep = NaN;  best_eo = NaN;
    history = zeros(1, 0);  iters = 0;  fk = 0;  converged = false;

    if ikb_budget_left(opts) >= N
        % Eq. 11: initial population, q0 is member 1
        X = q_min + rand(n, N) .* (q_max - q_min);  X(:, 1) = q0;
        F = zeros(1, N);
        for i = 1:N
            [F(i), ep, eo] = spo_fitness(robot, T_d, X(:, i));  fk = fk + 1;
            [best_E, q_best, best_ep, best_eo] = keep_best(F(i), X(:, i), ep, eo, ...
                                                           best_E, q_best, best_ep, best_eo);
        end
        converged = best_E < eps_min;

        t = 0;
        while ~converged && t < P.T
            if ikb_budget_left(opts) < N, break; end
            t = t + 1;
            Xk = spo_select(X, F, P.k);                                  % top k (p.6-7)
            Xn = spo_replenish(Xk(:, 1:3), N - P.k, P.eps, P.sigma);     % Eq. 13
            X  = [Xk, min(max(Xn, q_min), q_max)];
            X  = min(max(spo_perturb(X, P.eps, P.sigma), q_min), q_max); % Eq. 12
            for i = 1:N
                [F(i), ep, eo] = spo_fitness(robot, T_d, X(:, i));  fk = fk + 1;
                [best_E, q_best, best_ep, best_eo] = keep_best(F(i), X(:, i), ep, eo, ...
                                                               best_E, q_best, best_ep, best_eo);
            end
            iters = t;
            history(t) = best_E;
            converged = best_E < eps_min;                                % p.7
        end
    end

    info = struct();
    info.solver    = 'SPOA';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = converged;
    info.fitness   = best_E;
    info.err_pos   = best_ep;                                % m
    info.err_rot   = best_eo;                                % rad
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.history   = history;
    info.params    = P;
end

% ------------------------------------------------------------------ operators
function P = spo_params()
    P = struct('N', 40, 'k', 8, 'eps', 0.1, 'sigma', 0.1, 'T', 400, ...
               'w_pos', 1, 'w_rot', 1, 'mm_per_m', 1000);
end

function [E, ep, eo] = spo_fitness(robot, T_d, q)
% Eqs. 19-23 with alpha = beta = 1; Ep in mm, Eo in rad. ep is returned in metres.
    T  = robot.fk(q);
    ep = norm(T_d(1:3, 4) - T(1:3, 4));
    RE = T_d(1:3, 1:3) * T(1:3, 1:3)';                       % Eq. 22
    eo = 2 * acos(min(1, 0.5 * sqrt(max(0, RE(1,1) + RE(2,2) + RE(3,3) + 1))));   % Eq. 23
    E  = 1000 * ep + eo;                                     % Eq. 19
end

function Xk = spo_select(X, F, k)
% Top k individuals with minimum error, best first (columns 1..3 = X_alpha, X_beta, X_delta).
    [~, o] = sort(F);
    Xk = X(:, o(1:k));
end

function Xn = spo_replenish(X3, m, eps, sigma)
% Eq. 13: X_new = l1*X_alpha + l2*X_beta + l3*X_delta + delta, l = U(0,1)^3 / sum, delta =
% eps*N(0, sigma^2) per joint.
    n  = size(X3, 1);
    Xn = zeros(n, m);
    for j = 1:m
        lam = rand(3, 1);  lam = lam / sum(lam);
        Xn(:, j) = X3 * lam + eps * sigma * randn(n, 1);
    end
end

function X = spo_perturb(X, eps, sigma)
% Eq. 12: X <- X + eps*N(0, sigma^2), every solution, every joint.
    X = X + eps * sigma * randn(size(X));
end

function e = spo_eps_min(opts)
% Threshold on E (mm + rad) that implies the benchmark tolerances tol_pos (m) and tol_rot (rad).
    tp = 1e-6;  tr = 1e-6;
    if isfield(opts, 'tol_pos'), tp = opts.tol_pos; end
    if isfield(opts, 'tol_rot'), tr = opts.tol_rot; end
    e = min(1000 * tp, tr);
end

function [best_E, q_best, best_ep, best_eo] = keep_best(E, q, ep, eo, best_E, q_best, best_ep, best_eo)
    if isnan(best_E) || E < best_E
        best_E = E;  q_best = q;  best_ep = ep;  best_eo = eo;
    end
end
