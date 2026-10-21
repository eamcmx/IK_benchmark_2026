function [q_best, info] = solve_ACO(robot, T_d, q0, opts)
% SOLVE_ACO  Discrete Ant Colony Optimization IK solver of El Mrabet et al. (2025), Sec. III-A.
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
%   Algorithm (pp. 4-5, Fig. 5, Eqs. 3-5; fitness p. 8, Eqs. 19-23):
%     Joint i takes values from a fixed discrete set Omega_i. In every iteration m ants each build a
%     complete joint vector: value j of joint i is chosen with probability
%       P_ij = tau_ij^alpha * eta_ij^beta / sum_l tau_il^alpha * eta_il^beta        (Eq. 3)
%     where eta_ij is the inverse of the estimated error of value j. Every trail is evaluated and Best
%     keeps the lowest-error trail (Fig. 5 l.8-13). Pheromone update (Eqs. 4-5):
%       tau_ij <- (1 - rho) * tau_ij + sum_k dtau_ij^k,   dtau_ij^k = Q / f(theta^k) if trail k used
%       value j for joint i, else 0.
%     Fitness: E = alpha_E*Ep + beta_E*Eo (Eq. 19), Ep = ||P_d - P_c|| (Eq. 21),
%     Eo = 2*acos(0.5*sqrt(n_x + o_y + a_z + 1)) with R_E = R_d*R_c' (Eqs. 22-23).
%
%   Parameters:
%     m       = 40   ants per iteration. Table IV (p.5): 30-50; no value reported for the IK runs;
%                    midpoint.
%     alpha   = 1.5  pheromone influence. Table IV: 1-2; midpoint.
%     beta    = 3.5  heuristic influence. Table IV: 2-5; midpoint.
%     rho     = 0.3  evaporation rate. Table IV: 0.1-0.5; midpoint.
%     Q       = 1    deposit constant. Table IV.
%     T       = 400  iterations. Table IV: 300-500; midpoint. Never reached within the FK budget.
%     tau_0   = 0    Fig. 5 l.4 ("pheromones of the components, initially zero").
%     alpha_E = beta_E = 1 (Eq. 19). Not valued in this paper (p.8); the companion paper states
%                    E = Ep + EO (JRC 6(6), p.5, Eq. 9).
%     |Omega_i| = 21 values per joint. Not stated in the paper (pp. 4-5 checked: Omega_i is only "the
%                    set of available discrete values for joint i"). Benchmark uses a uniform grid of
%                    spacing (q_max - q_min)/20 because one heuristic estimation costs
%                    sum_i(|Omega_i| - 1) <= 20n FK calls (120 on 6-DOF, 260 on 13-DOF), so even if
%                    Best improves every iteration the colony completes at least 12 iterations on
%                    6-DOF and 6 on 13-DOF robots within 2000 FK calls; 41 values would leave 3
%                    iterations on 13-DOF, and fewer values would make each step exceed 5% of the
%                    joint range.
%     eta_ij  = 1 / E(Best with joint i set to value j). The paper gives eta_ij(t) only as "the
%                    heuristic value (e.g., inverse of the estimated error)" (p.4, under Eq. 3) and does
%                    not say how the error is estimated. eta is indexed by joint, value and iteration
%                    but not by ant, so it is estimated once per iteration around the current Best:
%                    one FK call per (joint, value) pair other than Best's own value. It is
%                    re-estimated only when Best has changed (otherwise the values are identical).
%
%   Benchmark interface:
%     Budget: an iteration starts only if ikb_budget_left(opts) covers all its FK calls (m trails
%       plus the eta estimation when Best changed). The fitness of Best is stored when it is
%       evaluated; there is no final FK check. opts.max_iter is ignored.
%     Warm start: q0 (clamped to the limits) is trail 1 of the first iteration; Omega_i is the
%       uniform grid anchored at q0_i, so q0 is a valid trail. The first eta estimation is made
%       around q0 (Best after trail 1).
%     Units: benchmark lengths are metres; the paper's robots are in mm (Tables I-III, pp. 3-4), so
%       Ep is computed in mm: E = 1000*||P_d - P_c||[m] + Eo[rad].
%     Joint limits: Eq. 20 bounds theta to [theta_min, theta_max]; every Omega_i lies inside
%       [q_min, q_max], so no out-of-range value can be built.
%     Output: Best (the lowest-E trail). info.converged is the paper's stopping rule, Best ideal
%       (E = 0). solve_ACO('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Eq. 3 is 0/0 in the first iteration because tau_0 = 0. The first iteration uses the limit of
%       a uniform tau_0 -> 0, where tau cancels: P_ij = eta_ij^beta / sum_l eta_il^beta. From the
%       second iteration Eq. 3 is applied as printed (values no trail has used keep P = 0).
%     - eta_ij = Inf (estimated error exactly 0): the limit of Eq. 3, all probability shared equally
%       by those values.
%     - Fig. 5 l.6 prints "While Best is the ideal solution or we have run out of time"; read as
%       "until Best is ideal or T iterations are done".
%     - Fig. 5 l.9 "Optionally Hill-climb P_i" is not applied: optional, and the paper neither says it
%       was used nor defines it.
%     - Fig. 5 l.10 "Adjust male positions by (3)" is ignored: it belongs to the MOA listing.
%     - Fig. 5 l.11 compares Fitness(P_i) > Fitness(best) (a maximised fitness); the error E is
%       minimised, so Best is replaced when E(P_i) < E(Best).
%     - The "two colonies" of the prose (p.4) define no operator; nothing is implemented for them.
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
        q_best = struct('fitness', @aco_fitness, 'grid', @aco_grid, 'probs', @aco_probs, ...
                        'estimate', @aco_estimate, 'deposit', @aco_deposit, 'params', aco_params());
        info = [];
        return
    end

    t_start = tic;
    P = aco_params();
    m = P.m;

    n     = robot.n;
    q_min = robot.q_min(:);
    q_max = robot.q_max(:);
    q0    = min(max(q0(:), q_min), q_max);

    [V, valid, i0] = aco_grid(q0, q_min, q_max, P.K);       % Omega_i, row i of V
    eta_cost = sum(sum(valid, 2) - 1);                       % FK calls of one eta estimation
    tau      = zeros(size(V));                               % Fig. 5 l.4

    best_idx = i0;  best_q = q0;  best_E = NaN;  best_ep = NaN;  best_eo = NaN;
    Err = [];  eta_ref = [];                                 % estimated errors and their Best
    history = zeros(1, 0);  iters = 0;  converged = false;  fk = 0;

    for t = 1:P.T
        need_eta = isempty(eta_ref) || any(eta_ref ~= best_idx);
        if ikb_budget_left(opts) < m + need_eta * eta_cost, break; end

        idx = zeros(n, m);  f = zeros(1, m);
        if t == 1                                            % trail 1 = q0 (warm start)
            idx(:, 1) = i0;
            [f(1), ep, eo] = aco_fitness(robot, T_d, q0);  fk = fk + 1;
            best_E = f(1);  best_ep = ep;  best_eo = eo;
            k1 = 2;
        else
            k1 = 1;
        end

        if need_eta                                          % eta_ij(t) around Best
            Err = aco_estimate(robot, T_d, V, valid, best_idx, best_q, best_E);
            fk = fk + eta_cost;  eta_ref = best_idx;
        end

        Pij = aco_probs(tau, Err, valid, P.alpha, P.beta);  % Eq. 3, one row per joint
        C   = cumsum(Pij, 2);
        for k = k1:m                                         % construction, joint by joint
            r = rand(n, 1);
            for i = 1:n
                idx(i, k) = find(C(i, :) >= r(i) * C(i, end), 1, 'first');
            end
            q = V(sub2ind(size(V), (1:n)', idx(:, k)));
            [f(k), ep, eo] = aco_fitness(robot, T_d, q);  fk = fk + 1;
            if f(k) < best_E                                 % Fig. 5 l.11-12
                best_E = f(k);  best_idx = idx(:, k);  best_q = q;  best_ep = ep;  best_eo = eo;
            end
        end

        tau = aco_deposit(tau, idx, f, P.rho, P.Q);          % Eqs. 4-5
        iters = t;
        history(t) = best_E;
        if best_E == 0, converged = true; break; end        % Fig. 5 l.6: Best is ideal
    end

    q_best = best_q;
    info = struct();
    info.solver    = 'ACO';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = converged;
    info.fitness   = best_E;
    info.err_pos   = best_ep;                                % m
    info.err_rot   = best_eo;                                % rad
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.history   = history;
    info.values    = V;                                      % Omega (NaN-padded rows)
    info.tau       = tau;
    info.params    = P;
end

% ------------------------------------------------------------------ operators
function P = aco_params()
    P = struct('m', 40, 'alpha', 1.5, 'beta', 3.5, 'rho', 0.3, 'Q', 1, 'T', 400, 'K', 21, ...
               'w_pos', 1, 'w_rot', 1, 'mm_per_m', 1000);
end

function [E, ep, eo] = aco_fitness(robot, T_d, q)
% Eqs. 19-23 with alpha = beta = 1; Ep in mm, Eo in rad. ep is returned in metres.
    T  = robot.fk(q);
    ep = norm(T_d(1:3, 4) - T(1:3, 4));
    RE = T_d(1:3, 1:3) * T(1:3, 1:3)';                       % Eq. 22
    eo = 2 * acos(min(1, 0.5 * sqrt(max(0, RE(1,1) + RE(2,2) + RE(3,3) + 1))));   % Eq. 23
    E  = 1000 * ep + eo;                                     % Eq. 19
end

function [V, valid, i0] = aco_grid(q0, q_min, q_max, K)
% Omega_i: values q0_i + k*(q_max_i - q_min_i)/(K-1), k integer, inside [q_min_i, q_max_i].
% Row i of V holds Omega_i (NaN-padded); i0(i) is the index of q0_i.
    n = numel(q0);
    V = nan(n, K);  valid = false(n, K);  i0 = zeros(n, 1);
    for i = 1:n
        dq = (q_max(i) - q_min(i)) / (K - 1);
        a  = floor((q0(i) - q_min(i)) / dq + 1e-9);
        b  = floor((q_max(i) - q0(i)) / dq + 1e-9);
        v  = min(max(q0(i) + (-a:b) * dq, q_min(i)), q_max(i));
        v(a + 1) = q0(i);
        V(i, 1:numel(v)) = v;  valid(i, 1:numel(v)) = true;  i0(i) = a + 1;
    end
end

function Err = aco_estimate(robot, T_d, V, valid, best_idx, best_q, best_E)
% Estimated error of value j of joint i: E(Best with joint i set to V(i,j)). One FK per pair,
% except Best's own value, whose error is best_E.
    [n, K] = size(V);
    Err = inf(n, K);
    for i = 1:n
        for j = find(valid(i, :))
            if j == best_idx(i)
                Err(i, j) = best_E;
            else
                q = best_q;  q(i) = V(i, j);
                Err(i, j) = aco_fitness(robot, T_d, q);
            end
        end
    end
end

function Pij = aco_probs(tau, Err, valid, alpha, beta)
% Eq. 3 with eta = 1/Err. Rows whose tau is all zero (first iteration) use uniform tau (tau
% cancels). Err = 0 gives eta = Inf: all probability on those values (limit of Eq. 3).
    [n, K] = size(tau);
    Pij = zeros(n, K);
    for i = 1:n
        v = valid(i, :);
        logw = -inf(1, K);
        logw(v) = -beta * log(Err(i, v));                    % log eta^beta
        if any(tau(i, v) > 0)
            lt = alpha * log(tau(i, :));                     % log tau^alpha (-Inf where tau = 0)
            logw(v) = logw(v) + lt(v);
            logw(v & tau(i, :) == 0) = -inf;
        end
        if any(logw == inf)
            w = double(logw == inf);
        else
            w = exp(logw - max(logw));
        end
        Pij(i, :) = w / sum(w);
    end
end

function tau = aco_deposit(tau, idx, f, rho, Q)
% Eqs. 4-5: evaporation, then deposit Q/f(theta^k) on every component trail k used.
    tau = (1 - rho) * tau;
    n = size(idx, 1);
    for k = 1:size(idx, 2)
        lin = sub2ind(size(tau), (1:n)', idx(:, k));
        tau(lin) = tau(lin) + Q / f(k);
    end
end
