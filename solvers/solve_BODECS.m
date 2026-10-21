function [q_best, info] = solve_BODECS(robot, T_d, q0, opts)
% SOLVE_BODECS  BODE-CS: hybrid bidirectional opposite differential evolution and improved cuckoo
%               search IK solver of Li, Luo & Qiao (2023), Secs. 2.2 and 3.
%
%   Source:
%     Minghao Li, Xiao Luo, Lijun Qiao (2023). Inverse Kinematics of Robot Manipulator Based on
%     BODE-CS Algorithm. Machines, 11, 648 (article 648, 33 pp.; no issue number printed).
%     DOI: 10.3390/machines11060648
%
%   Algorithm (fitness Sec. 2.2, Eqs. 6-12, pp.5-6; BODE Secs. 3.1-3.2, Eqs. 13-28, pp.6-9,
%   Algorithm 1, p.10; CS Secs. 3.3-3.5, Eqs. 29-41, pp.10-12, Algorithm 2, p.13; hybrid Sec. 3.6,
%   Eq. 42, p.13; Fig. 1, p.14; parameters Sec. 4.1, pp.16-17):
%     Initialisation (Alg. 1 l.3-8, Fig. 1): N individuals from the Halton sequence (Eqs. 18-20)
%     and their opposites x_max + x_min - x (Eq. 25) are evaluated; each individual keeps the
%     better of itself and its opposite.
%     Every iteration t = 0, 1, ... starts with the switch of Eq. 42 (Sec. 3.6, Fig. 1):
%       fit_det = ((fit_ave - fit_min)/fit_max)^2 over the swarm, r_DECS ~ U(0, 1);
%       r_DECS < fit_det: one improved-CS iteration (Alg. 2); otherwise one BODE generation (Alg. 1).
%     BODE generation (Alg. 1 l.10-18): for each individual j three random indices (Eq. 14) and one
%     r_DE ~ U(0, 1); r_DE against the thresholds of Eq. 27 selects one mutation of Eq. 28:
%       r_DE > r_base1             x_best + F(x_r1 - x_r2) + lambda(x_best - x_r3)  DE/best-best/1
%       r_co2 < r_DE <= r_base1    S_max + S_min - (1 - C_csor*C_rand) x_j          dynamical opposite
%       r_co1 < r_DE <= r_co2      x_j + F(x_r1 - x_r2) + lambda(x_best - x_r3)     DE/current-best/1
%       r_base2 < r_DE <= r_co1    x_best - F(x_r1 - x_r2) - lambda(x_best - x_r3)  bidirectional
%       r_DE <= r_base2            x_r1 + F(x_r2 - x_r3) + lambda(x_best - x_r1)    DE/rand-best/1
%     with r_base1 = 0.9 - 0.2 tau^2, r_base2 = 0.65 - 0.2 tau^2, r_co1 = r_base2 + (r_base1 -
%     r_base2)/3, r_co2 = r_base2 + 5(r_base1 - r_base2)/6, tau = t/Tmax (Eq. 27), S_max, S_min the
%     per-joint extremes of the swarm (Eq. 26), F = (F_max - F_min)(2 - exp(tau ln 2)) (Eq. 23);
%     then binomial crossover (Eq. 16) with Cr = C_min + (C_max - C_min) tau (Eq. 24) and one-to-one
%     greedy selection (Eq. 17).
%     Improved-CS iteration (Alg. 2 l.4-28); every new nest replaces its old nest only if better:
%       1. each nest j: Levy flight x_j + alpha_l s (x) H(Pa - eps) (x) (x_j - x_best) (Eq. 29), with
%          s = alpha_s u_m/|v_m|^(1/beta), alpha_s = (cos(2 pi r_css) + 1)/2 (Eqs. 39-40) and
%          Mantegna's u_m, v_m (Eqs. 33-34);
%       2. a fraction Pa = Pa_max - (Pa_max - Pa_min) tau (Eq. 41) of the worst nests is rebuilt by
%          the random walk of Eq. 35, x_j + rand(0, 1)(x_rcs1 - x_rcs2) where rand(0, 1) > Pa;
%       3. each nest j: if r_a < r_cso = C_0(1 - sqrt(tau)) (Eq. 36), the opposite of Eq. 26; then if
%          r_b < r_csbi = C_bi (fit_max - fit_ave)/(fit_max - fit_min) (Eq. 38), the linear global
%          best w_j x_j + w_gbest x_gbest (Eq. 37).
%     The run ends when t reaches Tmax or the position error of the best individual is at most
%     S_stop (Fig. 1; Alg. 1 l.19); the best individual is returned.
%     Fitness (Eqs. 6-8): f = w_P dP + w_R dR, dP = ||p - p_hat||, dR = sqrt(n_err + s_err + a_err),
%     the root of the summed squared differences of the n, s, a columns of the two rotation matrices.
%     Orientation weight (Eqs. 9-12): l_arm = sum(|d_i| + |a_i|), lambda_p = (pos_max - pos_min)/l_arm
%     with pos_max, pos_min the largest and smallest distance of the end effector from the base origin
%     over a Monte Carlo sample of the workspace (p.6, p.17), lambda_R = 1 - lambda_p,
%     w_R = (lambda_R c_rad / sum(theta_imax - theta_imin)) / lambda_p.
%
%   Parameters:
%     N = 30, Tmax = 100  The paper reports these IK settings: N = 30, Tmax = 100 for all algorithms
%                     (Sec. 4.1, p.16; repeated p.30, "with a swarm size of 30 and a maximum
%                     iteration of 100"), N x Tmax = 30 x 100 = 3000; swarm sizes 30, 50, 80, 100 in
%                     the swarm-size study (Fig. 9, p.20); N = 60 with Tmax = 100 for trajectory
%                     tracking (Sec. 4.3.2, p.24), 60 x 100 = 6000; N = 70, Tmax = 200 for the 7-DOF
%                     random points (p.23), 70 x 200 = 14000. The setting whose N x Tmax is nearest
%                     the 2000-FK budget is used: 30 x 100 = 3000. Inside the benchmark the horizon is
%                     T (Benchmark interface). N = 30 for every robot; not applied: "A swarm size is
%                     typically 5-10 times the population dimension" (p.16).
%     F_max = 0.9, F_min = 0.5, C_max = 0.8, C_min = 0.5, lambda = 0.1, C_csor = 0.05, alpha_l = 0.01,
%     beta = 3/2, Pa_max = 0.25, Pa_min = 0.05, C_0 = 0.2, C_bi = 0.15, w_j = 0.7, w_gbest = 0.3,
%     w_P = 1         p.16 (Sec. 4.1).
%     F               Eq. 23 as printed (p.8): 0.4 at tau = 0, 0 at tau = 1; F_max and F_min enter only
%                     through their difference. On the paper's own arm (Table 2, limits +/- pi/2) in
%                     its random-point setting (100 random reachable points, N = 30, Tmax = 100; q0
%                     random), this solver gives position errors 1.4e-5 / 4.9e-4 / 0.046 / 4.31 mm
%                     (min / median / mean / max; 5 of 100 above 0.0070 mm) and fitness 0.0063 /
%                     0.075 / 0.115 / 4.42; the paper reports 7.2e-6 / 5.9e-4 / 0.0070 mm and 0.0097 /
%                     0.067 / 0.112 (min / mean / max, Table 5, p.21). With an F_min term added
%                     (F = 0.9 -> 0.5) the same runs give 0.116 / 1.64 / 2.06 / 8.65 mm, all 100
%                     above 0.0070 mm.
%     S_stop = 1e-6 mm   p.16, on the position error ("P_error > S_stop", Fig. 1).
%     c_rad = 1 rad   p.6.  sigma_vm = 1, sigma_um of Eq. 34 (0.6966 for beta = 3/2).
%     lambda_p        per robot by the procedure of p.17 (Fig. 5): 30 independent estimates, each from
%                     300,000 random configurations, uniform in [q_min, q_max] (sampling distribution
%                     not stated, pp.6, 17); the mean of the 30 rounded to two decimals, as the paper
%                     sets 0.58 from estimates in 0.5725-0.5848 "to reduce the increase in
%                     computational time caused by high-precision decimals" (p.17). For the paper's
%                     own arm (Table 2, p.16) this procedure gives 0.5728-0.5830, mean 0.578 -> 0.58.
%                       robot         l_arm (m)  30 estimates     lambda_p  sum of ranges  w_R
%                       UR5           1.192509   0.8454-0.8479    0.85      24 pi rad      0.002341
%                       Panda         1.393      0.8408-0.8529    0.85      33.476 rad     0.005272
%                       Synthetic9    1.532      0.9889-0.9979    0.99      18 pi rad      0.000179
%                       Synthetic11   1.865      0.9794-0.9953    0.99      22 pi rad      0.000146
%                       Synthetic13   2.198      0.9608-0.9956    0.98      26 pi rad      0.000250
%                       IRB120        1.004      0.9325-0.9366    0.93      36.128 rad     0.002083
%                     The values are stored in the solver; another robot is estimated on first use.
%     Random numbers  one draw per individual: r_DE, C_rand (Eqs. 26, 28), r_css (Eq. 40), r_a, r_b
%                     (Alg. 2); one draw per joint: eps (Eq. 29), u_m, v_m (Eq. 33) and both
%                     rand(0, 1) of Eq. 35 (component-wise formulas); r_DECS one per iteration.
%
%   Reading of the switch, the abandonment and the swarm quantities:
%     Eq. 42: r_DECS in (0, 1) is generated "Before the iteration" and then compared with fit_det
%     (p.13); Fig. 1 makes the comparison once per pass of the loop. r_DECS is drawn at the start of
%     every iteration.
%     Abandonment: Alg. 2 l.9 abandons "a fraction (Pa) of the worst nests" and builds them with
%     Eqs. 35 and 41; Eq. 35 itself (Sec. 3.3, standard CS) moves joint k of a nest where
%     rand(0, 1) > Pa. The round(Pa*N) worst nests are rebuilt with Eq. 35 as printed.
%     x_best (Eqs. 28, 29), x_gbest (Eq. 37), S_k,max, S_k,min (Eqs. 26, 28) and fit_max, fit_ave,
%     fit_min (Eqs. 38, 42) are written at time t; each is taken from the swarm at the start of the
%     step that uses it (the BODE generation, step 1, step 2, step 3). Eq. 14 requires r1, r2, r3
%     distinct and does not exclude j; j is not excluded.
%
%   Benchmark interface:
%     Budget: initialisation needs 2N = 60 FK calls, a BODE generation N = 30, an improved-CS
%       iteration N + round(Pa*N) plus one per opposite and linear-global-best move (at most 3N +
%       round(Pa*N) = 98). Every batch (the initialisation, a BODE generation, step 1, step 2, each
%       move of step 3) starts only if ikb_budget_left(opts) covers it; otherwise the run stops and
%       returns. T = ikb_generations(opts, 60, 30, 100) = 64 (a 40-member population would give
%       ikb_generations(opts, 80, 40, 100) = 48); 1980 FK calls when every iteration is a BODE
%       generation. tau = t/T, t = 0 .. T-1; a run with CS iterations ends when the budget does,
%       before t reaches T. Outside the benchmark T = Tmax = 100. No final FK check. opts.max_iter
%       and every other option field are ignored.
%     Stop: the paper's rule; info.converged = position error of the result <= S_stop = 1e-6 mm.
%     Warm start: q0 (clamped to the limits) is individual 1 of the initial swarm, evaluated
%       unperturbed and compared with its opposite in the first selection; the swarm never loses
%       its best individual. Individuals 2..N are the Halton points H_D(2..N) (Eq. 20).
%     Units: Table 2 (p.16) and S_stop are in mm; benchmark lengths are metres, so dP is computed in
%       mm: f = 1000*||p - p_hat||[m] + w_R*dR. lambda_p is a length ratio, free of units.
%     Joint limits: no handling rule is stated (pp.6-13); every new candidate is clamped to
%       [q_min, q_max].
%     Output: the best individual (lowest f evaluated). info.err_pos (m), info.err_rot (rotation
%       angle, rad) and info.dR (Eq. 7) describe it; info.branch is 1 (BODE) or 2 (CS) per iteration.
%       solve_BODECS('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration horizon T = 64 instead of Tmax = 100 (benchmark rule); the schedules of Eqs. 23,
%       24, 27, 36 and 41 run over T.
%     - N = 30, the paper's setting for its 6-DOF arm, also on the 7- to 13-DOF robots.
%     - Eq. 19 is printed with the digit index from 1 and weights b^(-i-1), which would confine every
%       coordinate to [0, 1/b); the radical inverse it names is used, sum_i a_i b^(-i-1) over the
%       base-b digits a_0, a_1, ... of n.
%     - r_DECS drawn every iteration and abandonment of the round(Pa*N) worst nests by Eq. 35
%       (readings above).
%     - Random numbers per individual or per joint as listed under Parameters (not stated).
%     - Swarm quantities frozen per step; Eq. 14 indices may include j (readings above).
%     - lambda_p estimated per robot with uniform joint sampling and rounded as the paper rounds;
%       the paper's 0.58 belongs to its own arm.
%     - Position term in mm (Table 2 units).
%     - Candidates clamped to the joint limits; a candidate with fitness equal to its parent does
%       not replace it (tie rule not stated, Eq. 17 uses <).
%     - fit_det = 0 when fit_max = 0 and r_csbi = 0 when fit_max = fit_min (Eqs. 42 and 38 divide
%       by zero there).
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
        q_best = bodecs_ops(@bode_mutation);
        q_best.thresholds = @bode_thresholds;
        q_best.strategy   = @bode_strategy;
        info = [];
        return
    end
    [q_best, info] = bodecs_core(robot, T_d, q0, opts, @bode_mutation, 'BODE-CS');
end

% ==================================================================== mutation (Eqs. 27-28)
function [V, S] = bode_mutation(X, ~, ib, F, tau, P)
% Alg. 1 l.10-14: for each individual j three distinct indices (Eq. 14), one r_DE ~ U(0, 1), the
% strategy of Eq. 28 under the thresholds of Eq. 27, and its mutant. x_best = X(:, ib) and S_k,max,
% S_k,min are those of generation t.
    [n, N] = size(X);
    th   = bode_thresholds(tau);
    xb   = X(:, ib);
    Smax = max(X, [], 2);
    Smin = min(X, [], 2);
    V = zeros(n, N);
    S = zeros(1, N);
    for j = 1:N
        r = randperm(N, 3);
        S(j) = bode_strategy(rand, th);
        switch S(j)
            case 1      % DE/best-best/1 (Eq. 21)
                V(:, j) = xb + F * (X(:, r(1)) - X(:, r(2))) + P.lambda * (xb - X(:, r(3)));
            case 2      % dynamical opposite (Eq. 26)
                V(:, j) = bodecs_opposite(X(:, j), Smax, Smin, P);
            case 3      % DE/current-best/1
                V(:, j) = X(:, j) + F * (X(:, r(1)) - X(:, r(2))) + P.lambda * (xb - X(:, r(3)));
            case 4      % bidirectional (Eq. 22)
                V(:, j) = xb - F * (X(:, r(1)) - X(:, r(2))) - P.lambda * (xb - X(:, r(3)));
            otherwise   % DE/rand-best/1
                V(:, j) = X(:, r(1)) + F * (X(:, r(2)) - X(:, r(3))) + P.lambda * (xb - X(:, r(1)));
        end
    end
end

function th = bode_thresholds(tau)
% Eq. 27: [r_base1, r_base2, r_co1, r_co2] at tau = t/Tmax.
    b1 = 0.9 - 0.2 * tau^2;
    b2 = 0.65 - 0.2 * tau^2;
    th = [b1, b2, b2 + (b1 - b2) / 3, b2 + 5 * (b1 - b2) / 6];
end

function s = bode_strategy(rde, th)
% Eq. 28 selection rule, th = [r_base1, r_base2, r_co1, r_co2].
    if rde > th(1)
        s = 1;
    elseif rde > th(4)
        s = 2;
    elseif rde > th(3)
        s = 3;
    elseif rde > th(2)
        s = 4;
    else
        s = 5;
    end
end

% ==================================================================== shared BODE-CS code

function [q_best, info] = bodecs_core(robot, T_d, q0, opts, mutate, name)
% Fig. 1 (p.14): initialisation (Alg. 1 l.3-8), then per iteration the switch of Eq. 42 between a
% DE generation (Alg. 1 l.10-18, mutants from 'mutate') and an improved-CS iteration (Alg. 2
% l.4-28), until t = T or the position error of the best individual is at most S_stop.
    t_start = tic;
    P  = bodecs_params();
    N  = P.N;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    q0 = min(max(q0(:), lo), hi);
    [lp, wR] = bodecs_weights(robot, P);
    T   = ikb_generations(opts, 2 * N, N, P.Tmax);
    fit = @(q) bodecs_fitness(robot, T_d, q, wR, P);

    B = struct('f', NaN, 'q', q0, 'dP', NaN, 'dR', NaN, 'er', NaN, 'nfk', 0);
    hist = zeros(1, 0);  branch = zeros(1, 0);  fdet = zeros(1, 0);  rdecs = zeros(1, 0);
    nfk = zeros(1, 0);  strat = zeros(5, 0);  iters = 0;

    if ikb_budget_left(opts) >= 2 * N
        % Alg. 1 l.3-8: Halton individuals (Eq. 20), q0 is individual 1; opposites (Eq. 25);
        % each individual keeps the better of itself and its opposite.
        X  = bodecs_halton_pop(N, lo, hi);
        X(:, 1) = q0;
        Xo = lo + hi - X;
        Fx = zeros(1, N);
        Fo = zeros(1, N);
        for j = 1:N
            [Fx(j), B] = eval_keep(fit, X(:, j), B);
        end
        for j = 1:N
            [Fo(j), B] = eval_keep(fit, Xo(:, j), B);
        end
        s = Fo < Fx;
        X(:, s) = Xo(:, s);
        Fx(s)   = Fo(s);

        hist = zeros(1, T);  branch = zeros(1, T);  fdet = zeros(1, T);  rdecs = zeros(1, T);
        nfk  = zeros(1, T);  strat = zeros(5, T);
        t = 0;
        while t < T && B.dP > P.S_stop                               % Fig. 1, Alg. 1 l.19
            tau = t / T;
            fd  = bodecs_fitdet(Fx);                                 % Eq. 42
            r   = rand;
            n0  = B.nfk;
            if r < fd                                                % improved CS (Alg. 2)
                [X, Fx, B, done] = cs_iteration(X, Fx, B, fit, tau, P, lo, hi, opts);
                if ~done, break; end
                br = 2;  sc = zeros(5, 1);
            else                                                     % BODE (Alg. 1)
                if ikb_budget_left(opts) < N, break; end
                [X, Fx, B, S] = de_generation(X, Fx, B, fit, mutate, tau, P, lo, hi);
                br = 1;  sc = sum(S(:) == (1:5), 1).';
            end
            t = t + 1;
            iters = t;
            hist(t) = B.f;  branch(t) = br;  fdet(t) = fd;  rdecs(t) = r;
            nfk(t)  = B.nfk - n0;  strat(:, t) = sc;
        end
        hist = hist(1:iters);  branch = branch(1:iters);  fdet = fdet(1:iters);
        rdecs = rdecs(1:iters);  nfk = nfk(1:iters);  strat = strat(:, 1:iters);
    end

    q_best = B.q;
    info = struct();
    info.solver    = name;
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = B.dP <= P.S_stop;                               % p.16
    info.fitness   = B.f;                                            % Eq. 6 (mm + w_R*dR)
    info.err_pos   = B.dP / P.len_scale;                             % m
    info.err_rot   = B.er;                                           % rad, reporting only
    info.dR        = B.dR;                                           % Eq. 7
    info.fk_calls  = B.nfk;
    info.jac_calls = 0;
    info.N         = N;
    info.T         = T;
    info.lambda_p  = lp;
    info.omega_R   = wR;
    info.history   = hist;
    info.branch    = branch;
    info.fit_det   = fdet;
    info.r_DECS    = rdecs;
    info.fk_iter   = nfk;
    info.strategy  = strat;
    info.params    = P;
end

function [X, Fx, B, S] = de_generation(X, Fx, B, fit, mutate, tau, P, lo, hi)
% Alg. 1 l.10-18: mutants of generation t, binomial crossover (Eq. 16) with Cr of Eq. 24, every
% trial clamped and evaluated, then one-to-one greedy selection (Eq. 17).
    [n, N] = size(X);
    [~, ib] = min(Fx);
    F  = bodecs_F(tau, P);
    Cr = bodecs_Cr(tau, P);
    [V, S] = mutate(X, Fx, ib, F, tau, P);
    U = zeros(n, N);
    for j = 1:N
        U(:, j) = bodecs_crossover(X(:, j), V(:, j), Cr);
    end
    U  = min(max(U, lo), hi);
    Fu = zeros(1, N);
    for j = 1:N
        [Fu(j), B] = eval_keep(fit, U(:, j), B);
    end
    s = Fu < Fx;
    X(:, s) = U(:, s);
    Fx(s)   = Fu(s);
end

function [X, Fx, B, done] = cs_iteration(X, Fx, B, fit, tau, P, lo, hi, opts)
% Alg. 2 l.4-28 (Fig. 1): step 1 Levy flights, step 2 abandonment, step 3 opposite and linear global
% best moves; greedy replacement of each nest. done = false if the budget stopped the iteration.
    N = size(X, 2);
    done = false;
    Pa = bodecs_Pa(tau, P);                                          % Eq. 41

    % step 1 (l.4-8): Eq. 29 with Eqs. 33-34 and 39-40, x_best of the swarm at the start of the step
    if ikb_budget_left(opts) < N, return; end
    [~, ib] = min(Fx);
    xb = X(:, ib);
    for j = 1:N
        u = min(max(bodecs_levy_nest(X(:, j), xb, Pa, P), lo), hi);
        [fu, B] = eval_keep(fit, u, B);
        if fu < Fx(j), X(:, j) = u;  Fx(j) = fu; end
    end

    % step 2 (l.9-11): the round(Pa*N) worst nests rebuilt by Eq. 35 from the swarm at the start
    na = round(Pa * N);
    if ikb_budget_left(opts) < na, return; end
    [~, o] = sort(Fx, 'descend');
    X0 = X;
    for k = 1:na
        j = o(k);
        u = min(max(bodecs_abandon(X0(:, j), X0, Pa), lo), hi);
        [fu, B] = eval_keep(fit, u, B);
        if fu < Fx(j), X(:, j) = u;  Fx(j) = fu; end
    end

    % step 3 (l.12-28): opposite (Eqs. 36, 26) and linear global best (Eqs. 37-38) per nest, with
    % S_k,max, S_k,min, x_gbest and fit_max, fit_ave, fit_min of the swarm at the start of the step
    rcso  = bodecs_rcso(tau, P);
    rcsbi = bodecs_rcsbi(Fx, P);
    Smax  = max(X, [], 2);
    Smin  = min(X, [], 2);
    [~, ib] = min(Fx);
    xg = X(:, ib);
    for j = 1:N
        if rand < rcso                                               % l.13-20
            if ikb_budget_left(opts) < 1, return; end
            u = min(max(bodecs_opposite(X(:, j), Smax, Smin, P), lo), hi);
            [fu, B] = eval_keep(fit, u, B);
            if fu < Fx(j), X(:, j) = u;  Fx(j) = fu; end
        end
        if rand < rcsbi                                              % l.21-27
            if ikb_budget_left(opts) < 1, return; end
            u = min(max(bodecs_linear(X(:, j), xg, P), lo), hi);
            [fu, B] = eval_keep(fit, u, B);
            if fu < Fx(j), X(:, j) = u;  Fx(j) = fu; end
        end
    end
    done = true;
end

function [f, B] = eval_keep(fit, q, B)
% One counted FK evaluation; B keeps the lowest fitness evaluated and the evaluation count.
    [f, dP, dR, er] = fit(q);
    B.nfk = B.nfk + 1;
    if isnan(B.f) || f < B.f
        B.f = f;  B.q = q;  B.dP = dP;  B.dR = dR;  B.er = er;
    end
end

% -------------------------------------------------------------------- parameters and schedules
function P = bodecs_params()
% p.16 (Sec. 4.1), p.6 (c_rad), p.17 (Monte Carlo size).
    P = struct('N', 30, 'Tmax', 100, 'F_max', 0.9, 'F_min', 0.5, 'C_max', 0.8, 'C_min', 0.5, ...
               'lambda', 0.1, 'C_csor', 0.05, 'alpha_l', 0.01, 'beta', 1.5, 'Pa_max', 0.25, ...
               'Pa_min', 0.05, 'C_0', 0.2, 'C_bi', 0.15, 'w_j', 0.7, 'w_gbest', 0.3, 'w_P', 1, ...
               'c_rad', 1, 'S_stop', 1e-6, 'len_scale', 1000, 'mc_points', 300000, 'mc_runs', 30, ...
               'lp_digits', 2);
end

function F = bodecs_F(tau, P)
% Eq. 23 as printed: (F_max - F_min) at tau = 0, 0 at tau = 1.
    F = (P.F_max - P.F_min) * (2 - exp(tau * log(2)));
end

function Cr = bodecs_Cr(tau, P)
% Eq. 24.
    Cr = P.C_min + (P.C_max - P.C_min) * tau;
end

function Pa = bodecs_Pa(tau, P)
% Eq. 41.
    Pa = P.Pa_max - (P.Pa_max - P.Pa_min) * tau;
end

function r = bodecs_rcso(tau, P)
% Eq. 36.
    r = P.C_0 * (1 - sqrt(tau));
end

function r = bodecs_rcsbi(Fx, P)
% Eq. 38; 0 when all fitness values are equal.
    fmax = max(Fx);
    fmin = min(Fx);
    if fmax > fmin
        r = P.C_bi * (fmax - mean(Fx)) / (fmax - fmin);
    else
        r = 0;
    end
end

function d = bodecs_fitdet(Fx)
% Eq. 42; 0 when fit_max = 0.
    fmax = max(Fx);
    if fmax > 0
        d = ((mean(Fx) - min(Fx)) / fmax)^2;
    else
        d = 0;
    end
end

% -------------------------------------------------------------------- operators
function u = bodecs_crossover(x, v, Cr)
% Eq. 16: joint k from the mutant if rand(0, 1) < Cr or k = k_rand, otherwise from x.
    n = numel(x);
    m = rand(n, 1) < Cr;
    m(randi(n)) = true;
    u = x;
    u(m) = v(m);
end

function u = bodecs_opposite(x, Smax, Smin, P)
% Eq. 26: S_k,max + S_k,min - (1 - C_csor*C_rand) x_k, C_rand ~ U(0, 1) one draw per individual.
    u = Smax + Smin - (1 - P.C_csor * rand) * x;
end

function u = bodecs_levy_nest(x, xb, Pa, P)
% Eq. 29 with Eqs. 33-34 and 39-40: x + alpha_l s .* H(Pa - eps) .* (x - x_best),
% s = alpha_s u_m ./ |v_m|.^(1/beta), alpha_s = (cos(2 pi r_css) + 1)/2; r_css one draw per nest,
% u_m ~ N(0, sigma_um^2), v_m ~ N(0, 1) and eps ~ U(0, 1) one draw per joint.
    n  = numel(x);
    as = (cos(2 * pi * rand) + 1) / 2;
    um = bodecs_sigma(P.beta) * randn(n, 1);
    vm = randn(n, 1);
    s  = as * um ./ abs(vm).^(1 / P.beta);
    h  = rand(n, 1) < Pa;
    u  = x + P.alpha_l * s .* h .* (x - xb);
end

function u = bodecs_abandon(x, X, Pa)
% Eq. 35: joint k moves by rand(0, 1)*(x_rcs1 - x_rcs2) where rand(0, 1) > Pa; rcs1 ~= rcs2 random
% individuals; both random numbers one draw per joint.
    [n, N] = size(X);
    r  = randperm(N, 2);
    m  = rand(n, 1) > Pa;
    st = rand(n, 1);
    u  = x + m .* st .* (X(:, r(1)) - X(:, r(2)));
end

function u = bodecs_linear(x, xg, P)
% Eq. 37.
    u = P.w_j * x + P.w_gbest * xg;
end

function s = bodecs_sigma(beta)
% Eq. 34: sigma_um.
    s = (gamma(1 + beta) * sin(pi * beta / 2) ...
         / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
end

function X = bodecs_halton_pop(N, lo, hi)
% Eq. 20: individual j = x_min + (x_max - x_min) .* H_D(j), j = 1..N.
    X = lo + (hi - lo) .* bodecs_halton(N, numel(lo));
end

function H = bodecs_halton(N, D)
% Eqs. 18-19: column n = (phi_b1(n), ..., phi_bD(n)), n = 1..N, b_j the j-th prime, phi_b the
% radical inverse sum_i a_i b^(-i-1) of the base-b digits a_0, a_1, ... of n.
    b = primes(1000);
    b = b(1:D);
    H = zeros(D, N);
    for d = 1:D
        for k = 1:N
            m = k;  w = 1 / b(d);  v = 0;
            while m > 0
                v = v + w * mod(m, b(d));
                m = floor(m / b(d));
                w = w / b(d);
            end
            H(d, k) = v;
        end
    end
end

% -------------------------------------------------------------------- fitness (Eqs. 6-12)
function [f, dP, dR, er] = bodecs_fitness(robot, T_d, q, wR, P)
% Eqs. 6-8: f = w_P dP + w_R dR; dP Euclidean position error in mm (Table 2 units), dR the root of
% the summed squared n, s, a column differences (Eq. 7). er = rotation angle (rad), reporting only.
    T  = robot.fk(q);
    dP = P.len_scale * norm(T_d(1:3, 4) - T(1:3, 4));
    E  = T_d(1:3, 1:3) - T(1:3, 1:3);
    dR = sqrt(sum(E(:, 1).^2) + sum(E(:, 2).^2) + sum(E(:, 3).^2));
    f  = P.w_P * dP + wR * dR;
    er = acos(min(1, max(-1, (trace(T_d(1:3, 1:3).' * T(1:3, 1:3)) - 1) / 2)));
end

function [lp, wR] = bodecs_weights(robot, P)
% Eqs. 11-12: lambda_R = 1 - lambda_p, w_R = (lambda_R c_rad / sum(theta_max - theta_min)) / lambda_p.
    lp = bodecs_lambda_p(robot, P);
    wR = ((1 - lp) * P.c_rad / sum(robot.q_max(:) - robot.q_min(:))) / lp;
end

function lp = bodecs_lambda_p(robot, P)
% Eq. 10 by the procedure of p.17 (bodecs_mc): mean of the estimates rounded to P.lp_digits. Stored
% values for the benchmark robots (matched on name, l_arm, sum of joint ranges, n and tool); any
% other robot is estimated once per session.
    persistent cache
    tab = {'UR5',         1.192509, 24 * pi,       6,  0.85;
           'Panda',       1.393,    33.476,        7,  0.85;
           'Synthetic9',  1.532,    18 * pi,       9,  0.99;
           'Synthetic11', 1.865,    22 * pi,       11, 0.99;
           'Synthetic13', 2.198,    26 * pi,       13, 0.98;
           'IRB120',      1.004,    deg2rad(2070), 6,  0.93};
    la = sum(abs(robot.dh.d) + abs(robot.dh.a));
    sr = sum(robot.q_max(:) - robot.q_min(:));
    k  = find(strcmp(tab(:, 1), robot.name));
    if ~isempty(k) && abs(tab{k, 2} - la) < 1e-9 && abs(tab{k, 3} - sr) < 1e-9 ...
            && tab{k, 4} == robot.n && isequal(robot.tool, eye(4))
        lp = tab{k, 5};
        return
    end
    if isempty(cache), cache = containers.Map(); end
    key = sprintf('%s|%.15g|%.15g|%d|%s', robot.name, la, sr, robot.n, mat2str(robot.tool));
    if ~isKey(cache, key)
        mc = bodecs_mc(robot, P.mc_points, P.mc_runs);
        cache(key) = round(mean(mc.lp), P.lp_digits);
    end
    lp = cache(key);
end

function mc = bodecs_mc(robot, M, runs)
% p.17: 'runs' independent Monte Carlo estimates of lambda_p (Eq. 10), each from M configurations
% uniform in [q_min, q_max]; pos = distance of the end effector from the base origin. Own random
% stream per run (seed = run index), no counted FK call.
    dh = robot.dh;
    n  = robot.n;
    toff = zeros(1, n);
    if isfield(dh, 'theta_offset'), toff = dh.theta_offset; end
    la = sum(abs(dh.d) + abs(dh.a));
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    lp = zeros(1, runs);  pmax = zeros(1, runs);  pmin = zeros(1, runs);
    for k = 1:runs
        st = RandStream('mt19937ar', 'Seed', k);
        Q  = lo + rand(st, n, M) .* (hi - lo);
        R  = repmat(eye(3), 1, 1, M);
        p  = zeros(3, M);
        for i = 1:n
            th = Q(i, :) + toff(i);
            ct = cos(th);
            sn = sin(th);
            ca = cos(dh.alpha(i));
            sa = sin(dh.alpha(i));
            tl = [dh.a(i) * ct; dh.a(i) * sn; dh.d(i) * ones(1, M)];
            p  = p + reshape(pagemtimes(R, reshape(tl, 3, 1, M)), 3, M);
            A  = zeros(3, 3, M);
            A(1, 1, :) = ct;   A(1, 2, :) = -sn * ca;  A(1, 3, :) = sn * sa;
            A(2, 1, :) = sn;   A(2, 2, :) = ct * ca;   A(2, 3, :) = -ct * sa;
            A(3, 2, :) = sa;   A(3, 3, :) = ca;
            R  = pagemtimes(R, A);
        end
        p = p + reshape(pagemtimes(R, repmat(robot.tool(1:3, 4), 1, 1, M)), 3, M);
        d = sqrt(sum(p.^2, 1));
        pmax(k) = max(d);
        pmin(k) = min(d);
        lp(k)   = (pmax(k) - pmin(k)) / la;
    end
    mc = struct('lp', lp, 'pmax', pmax, 'pmin', pmin, 'l_arm', la);
end

% -------------------------------------------------------------------- operator handles for tests
function ops = bodecs_ops(mutate)
    ops = struct('params', bodecs_params(), 'core', @bodecs_core, 'mutate', mutate, ...
                 'de_generation', @de_generation, 'cs_iteration', @cs_iteration, ...
                 'fitness', @bodecs_fitness, 'weights', @bodecs_weights, ...
                 'lambda_p', @bodecs_lambda_p, 'mc', @bodecs_mc, 'halton', @bodecs_halton, ...
                 'halton_pop', @bodecs_halton_pop, 'F', @bodecs_F, 'Cr', @bodecs_Cr, 'Pa', @bodecs_Pa, ...
                 'rcso', @bodecs_rcso, 'rcsbi', @bodecs_rcsbi, 'fitdet', @bodecs_fitdet, ...
                 'sigma', @bodecs_sigma, 'crossover', @bodecs_crossover, ...
                 'opposite', @bodecs_opposite, 'levy', @bodecs_levy_nest, ...
                 'abandon', @bodecs_abandon, 'linear', @bodecs_linear);
end
