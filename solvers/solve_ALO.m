function [q_best, info] = solve_ALO(robot, T_d, q0, opts)
% SOLVE_ALO  Ant Lion Optimizer IK solver of El Mrabet et al. (2025), Sec. III-D.
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
%   Algorithm (pp. 7-8, Fig. 8, Eqs. 14-18; fitness p. 8, Eqs. 19-23):
%     n ants and n antlions are initialised at random inside the bounds and evaluated; the best
%     antlion is the elite E (Fig. 8 l.2-4). Each iteration, every ant selects an antlion A by
%     roulette wheel (l.7), performs a random walk near A (R_A) and one near E (R_E) (l.8-9), and
%     moves to Ant_i = (R_A + R_E)/2 (Eq. 15). A walk is x = [0, cum(2r(t_1) - 1), ...,
%     cum(2r(t_T) - 1)] with r = 1 if rand > 0.5 else 0 (Eqs. 17-18), min-max normalised into the
%     walk bounds [c, d]: R = (X_t - a)(d - c)/(b - a) + c, a = min, b = max of the walk (Eq. 16).
%     All ants are evaluated, ants and antlions are merged and sorted, the fittest n become the
%     antlions and the worst n the ants (l.12-14), and the elite is replaced if an antlion is better
%     (l.15).
%     Fitness: E = alpha_E*Ep + beta_E*Eo (Eq. 19), Ep = ||P_d - P_c|| (Eq. 21),
%     Eo = 2*acos(0.5*sqrt(n_x + o_y + a_z + 1)) with R_E = R_d*R_c' (Eqs. 22-23).
%
%   Parameters:
%     n_ants = n_antlions = 40. Table VII (p.8): population size 30-50, "Number of ants and
%                    antlions"; no value reported for the IK runs; midpoint. Read as the size of each
%                    of the two populations, because Fig. 8 l.13-14 selects "the fittest n agents as
%                    the new antlions and the worst n agents as the ants" from the merged set.
%     T      = 400   iterations. Table VII: 300-500; midpoint. Also the walk length in Eq. 17.
%                    Inside the benchmark T is the number of iterations the FK budget allows
%                    (ikb_generations: (2000 - 2n)/n = 48 for n = 40), so the walk and the
%                    shrinking bounds run their full course within the budget.
%     scaling factor I(t) = t. Table VII: "Dynamic", "Bounds shrink with each iteration"; no formula
%                    in the paper (pp. 7-8 checked). Benchmark uses I = t, the simplest factor that
%                    shrinks the bounds at every iteration without adding a constant.
%     walk bounds [c, d] = antlion -/+ (q_max - q_min)/(2*I(t)). The paper says only that c and d
%                    "define the lower and upper bounds of the random walk" and that the walk is
%                    performed "near the antlion" (p.7). Benchmark centres the interval on the
%                    antlion with width (q_max - q_min)/I(t), so it contains the antlion for every
%                    joint, including joints whose limits exclude 0.
%     roulette weights = 1/E. Fig. 8 l.7 and p.7 select the antlion "based on roulette wheel" using
%                    the antlions' fitness values, without the weighting; E is an error to minimise,
%                    so the weight is its inverse, as the same paper does for ACO (Eq. 3, Eq. 5).
%     Top-k selection 10-30% of N (Table VII): "retained for interpolation"; Sec. III-D and Fig. 8
%                    define no interpolation step, so it has no operator and is not used.
%     alpha_E = beta_E = 1 (Eq. 19). Not valued in this paper (p.8); the companion paper states
%                    E = Ep + EO (JRC 6(6), p.5, Eq. 9).
%
%   Benchmark interface:
%     Budget: initialisation needs 2n FK calls and each iteration n; a step starts only if
%       ikb_budget_left(opts) covers it. No final FK check. opts.max_iter is ignored.
%     Warm start: q0 (clamped to the limits) is antlion 1, so it takes part in the first elite
%       selection and roulette and is kept by the merge unless n better agents exist.
%     Units: benchmark lengths are metres; the paper's robots are in mm (Tables I-III, pp. 3-4), so
%       Ep is computed in mm: E = 1000*||P_d - P_c||[m] + Eo[rad].
%     Joint limits: Eq. 20 bounds theta; the paper gives no handling rule, so each new ant position
%       is clamped to [q_min, q_max].
%     Output: the lowest-E agent ever evaluated (the elite, unless an initial ant, which never
%       enters the merge, was better). The paper's only stopping rule is T (Fig. 8 l.5 "end
%       condition", Table VII "Max number of iterations"), so info.converged is always false.
%       solve_ALO('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Eq. 17 prints cum(2r(t_k)) - 1 with the -1 outside the cumulative sum: the first step is then
%       +/-1 and every later step 0 or +2, a walk that only rises and, normalised by Eq. 16, sweeps
%       from c towards d instead of moving "near the antlion". The -1 is applied to every step,
%       cum(2r(t_k) - 1), so all steps are +/-1.
%     - Eq. 16 mixes superscripts (d_i - c_i^t, b_i^t, + c_i); read as the min-max mapping of the
%       walk value at iteration t onto [c^t, d^t].
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
        q_best = struct('fitness', @alo_fitness, 'walk', @alo_walk, 'normalise', @alo_normalise, ...
                        'bounds', @alo_bounds, 'roulette', @alo_roulette, 'params', alo_params());
        info = [];
        return
    end

    t_start = tic;
    P = alo_params();
    N = P.N;  T_max = ikb_generations(opts, 2 * N, N, P.T);

    n     = robot.n;
    q_min = robot.q_min(:);
    q_max = robot.q_max(:);
    q0    = min(max(q0(:), q_min), q_max);

    q_best = q0;  best_E = NaN;  best_ep = NaN;  best_eo = NaN;
    history = zeros(1, 0);  iters = 0;  fk = 0;

    if ikb_budget_left(opts) >= 2 * N
        % Fig. 8 l.2-3: random ants and antlions inside the bounds; q0 is antlion 1
        ants     = q_min + rand(n, N) .* (q_max - q_min);
        antlions = q_min + rand(n, N) .* (q_max - q_min);
        antlions(:, 1) = q0;
        ants_E = zeros(1, N);  al_E = zeros(1, N);
        for i = 1:N
            [ants_E(i), ep, eo] = alo_fitness(robot, T_d, ants(:, i));  fk = fk + 1;
            [best_E, q_best, best_ep, best_eo] = keep_best(ants_E(i), ants(:, i), ep, eo, ...
                                                           best_E, q_best, best_ep, best_eo);
        end
        for i = 1:N
            [al_E(i), ep, eo] = alo_fitness(robot, T_d, antlions(:, i));  fk = fk + 1;
            [best_E, q_best, best_ep, best_eo] = keep_best(al_E(i), antlions(:, i), ep, eo, ...
                                                           best_E, q_best, best_ep, best_eo);
        end

        % Fig. 8 l.4: elite antlion
        [al_E, o] = sort(al_E);  antlions = antlions(:, o);
        elite = antlions(:, 1);  elite_E = al_E(1);

        for t = 1:T_max                                      % Fig. 8 l.5
            if ikb_budget_left(opts) < N, break; end
            for i = 1:N                                      % l.6-11
                A = antlions(:, alo_roulette(al_E));         % l.7
                [cA, dA] = alo_bounds(A, q_min, q_max, t);
                [cE, dE] = alo_bounds(elite, q_min, q_max, t);
                RA = alo_normalise(alo_walk(n, T_max), t, cA, dA);   % l.8, Eqs. 16-18
                RE = alo_normalise(alo_walk(n, T_max), t, cE, dE);   % l.9
                ants(:, i) = min(max((RA + RE) / 2, q_min), q_max);  % l.10, Eq. 15
            end
            for i = 1:N                                      % l.12
                [ants_E(i), ep, eo] = alo_fitness(robot, T_d, ants(:, i));  fk = fk + 1;
                [best_E, q_best, best_ep, best_eo] = keep_best(ants_E(i), ants(:, i), ep, eo, ...
                                                               best_E, q_best, best_ep, best_eo);
            end
            % l.13-14: merge, fittest N -> antlions, worst N -> ants
            allq = [antlions, ants];  allE = [al_E, ants_E];
            [allE, o] = sort(allE);  allq = allq(:, o);
            antlions = allq(:, 1:N);       al_E   = allE(1:N);
            ants     = allq(:, N+1:2*N);   ants_E = allE(N+1:2*N);
            if al_E(1) < elite_E                             % l.15
                elite = antlions(:, 1);  elite_E = al_E(1);
            end
            iters = t;
            history(t) = elite_E;
        end
    end

    info = struct();
    info.solver    = 'ALO';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = false;
    info.fitness   = best_E;
    info.err_pos   = best_ep;                                % m
    info.err_rot   = best_eo;                                % rad
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.history   = history;
    info.params    = P;
end

% ------------------------------------------------------------------ operators
function P = alo_params()
    P = struct('N', 40, 'T', 400, 'w_pos', 1, 'w_rot', 1, 'mm_per_m', 1000);
end

function [E, ep, eo] = alo_fitness(robot, T_d, q)
% Eqs. 19-23 with alpha = beta = 1; Ep in mm, Eo in rad. ep is returned in metres.
    T  = robot.fk(q);
    ep = norm(T_d(1:3, 4) - T(1:3, 4));
    RE = T_d(1:3, 1:3) * T(1:3, 1:3)';                       % Eq. 22
    eo = 2 * acos(min(1, 0.5 * sqrt(max(0, RE(1,1) + RE(2,2) + RE(3,3) + 1))));   % Eq. 23
    E  = 1000 * ep + eo;                                     % Eq. 19
end

function X = alo_walk(n, T)
% Eqs. 17-18: one walk per dimension, X(:, k+1) = cum(2r - 1) up to step k, X(:, 1) = 0.
    X = [zeros(n, 1), cumsum(2 * (rand(n, T) > 0.5) - 1, 2)];
end

function R = alo_normalise(X, t, c, d)
% Eq. 16: walk value at iteration t mapped from [a, b] = [min, max] of the walk onto [c, d].
    a = min(X, [], 2);  b = max(X, [], 2);
    R = (X(:, t + 1) - a) .* (d - c) ./ (b - a) + c;
end

function [c, d] = alo_bounds(L, q_min, q_max, t)
% Walk bounds around antlion L, shrunk by the scaling factor I = t.
    h = (q_max - q_min) / (2 * t);
    c = L - h;  d = L + h;
end

function j = alo_roulette(E)
% Roulette wheel on the antlions' fitness (weight 1/E); E = 0 takes all the weight.
    if any(E == 0)
        w = double(E == 0);
    else
        w = 1 ./ E;
    end
    j = find(cumsum(w) >= rand * sum(w), 1, 'first');
end

function [best_E, q_best, best_ep, best_eo] = keep_best(E, q, ep, eo, best_E, q_best, best_ep, best_eo)
    if isnan(best_E) || E < best_E
        best_E = E;  q_best = q;  best_ep = ep;  best_eo = eo;
    end
end
