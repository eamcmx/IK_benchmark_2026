function [q_best, info] = solve_HBC(robot, T_d, q0, opts)
% SOLVE_HBC  Hybrid improved BRO algorithm (HBC) IK solver of Wu, Zhang, Song, Su and Gu (2022), Algorithm 3.
%
%   Source:
%     Honggang Wu, Xinming Zhang, Linsen Song, Chengzhi Su and Lidong Gu (2022). A hybrid improved BRO
%     algorithm and its application in inverse kinematics of 7R 6DOF robot. Advances in Mechanical
%     Engineering, 14(3), 1-20. DOI: 10.1177/16878132221085125
%   Page numbers are PDF pages 1-20 (PDF page = printed page).
%
%   Algorithm (Sec. "A new hybrid improved algorithm (HBC)": Eqs. 13-16, p.5; Steps 1-6 and Algorithm 3,
%   p.6; Fig. 1, p.7; with the BRO operators Eqs. 2-6 and Algorithm 1, pp.3-4, sigma^2 of the CSO Eq. 8,
%   p.4, and the IK fitness Eq. 23, p.16):
%     n players are placed at random in the search space and evaluated (Step 1). At t = 1 and whenever
%     mod(t, Gn) = 0 they are ranked by fitness and split into x_n = [x_a, x_b] (Eq. 13): the na best are
%     the elite players, the other nb the ordinary players. The level relationship gives each ordinary
%     player an elite drawn at random as its update target ("randomly assigning each elite player to the
%     ordinary player as its update target", p.5; the "follow-up relationship" of Step 2, p.6); subgroups
%     and relationship are re-established every Gn iterations (p.5, Alg. 3). With the loser move of
%     Step 3 used here (Reading of a contradiction) no operator reads the relationship. Iteration t
%     (Alg. 3):
%       elite players, i = 1..na (Eq. 14, sigma^2 of Eq. 8, k a random other elite):
%         x_i,a(t+1) = x_i,a * (1 + randn(0, sigma^2))        if p >= 0.5
%         x_i,a(t+1) = x_i,a + sigma^2 * r(x_best - x_i,a)    if p < 0.5
%         sigma^2 = 1 if f_i <= f_k, otherwise exp((f_k - f_i) / |f_i + eps|)                    (Eq. 8)
%       ordinary players, i = 1..nb: j is the ordinary player closest to i (Euclidean distance);
%         d = i, v = j, and d = j, v = i if f(x_i,b) < f(x_j,b). The winner moves toward x_best,
%         x_v,b = x_v,b + r(x_best - x_v,b) (Eq. 15), and its damage is set to 0. While the damage of the
%         loser is below thre the loser also moves toward x_best, x_d,b = x_d,b + r(x_best - x_d,b)
%         (Eq. 2, Step 3), and its damage rises by 1; otherwise it respawns in the current safe area,
%         x_d,b = r(ub_d - lb_d) + lb_d (Eq. 3), and its damage is set to 0.
%       safe area: if t >= Delta, lb_d = x_best - SD(x_d) and ub_d = x_best + SD(x_d) (Eqs. 4-5), bounds
%         beyond the solution space are set to the original bounds (Alg. 3), and Delta = Delta +
%         round(Delta/2) (Eq. 6).
%       all n players are re-evaluated and the best individual is recorded (Steps 4-5; Fig. 1).
%     The run ends after T iterations; the best individual recorded is returned (Alg. 3).
%     IK fitness (Eq. 23, p.16), position only: E_P = ||P - P_bar||, the Euclidean distance between the
%     desired position P and the actual position P_bar of the end effector.
%
%   Parameters:
%     n = 25, T = 100   the IK setting whose N x T is nearest the 2000-FK budget (benchmark rule for
%                     several reported IK settings). The paper's IK runs: PUMA 560, Table 9 (p.16; 50 runs
%                     each, p.11), HBC rows N x T = 25 x 100 = 2,500, 25 x 300 = 7,500, 50 x 100 = 5,000,
%                     50 x 300 = 15,000, 100 x 100 = 10,000, 100 x 300 = 30,000, 200 x 100 = 20,000,
%                     200 x 300 = 60,000, 300 x 100 = 30,000, 300 x 300 = 90,000, 300 x 500 = 150,000;
%                     7R 6DOF robot (ABB IRB5400, Tables 10-12, pp.17-18; 30 runs per test point) with
%                     parameters "the same as Section 4.2" (p.17), i.e. 100 x 300 = 30,000 (p.7).
%                     25 x 100 is the nearest to 2000 (next: 50 x 100). The benchmark-function runs
%                     (100 x 300, p.7) are not IK runs. Inside the benchmark T is the budget horizon
%                     (Benchmark interface).
%     na = 0.2n = 5, nb = n - 0.2n = 20   Table 2 (p.8); Eq. 13 (p.5).
%     thre = 3        Table 2 (p.8); p.4.
%     Gn = 10         Table 2 (p.8).
%     Delta           first value round(T / ceil(log10 T)) (Algorithm 1, p.4; Algorithm 3 and Step 1 set
%                     it among the "related parameters"), then Delta + round(Delta/2) (Eq. 6, p.4;
%                     Algorithm 3 prints the threshold of this line as "(a~)", read as Delta). For
%                     T = 100 Eqs. 4-5 act after iterations 50 and 75.
%     p               p in [0, 1] (p.5): one U(0, 1) draw per elite and iteration, the switch of a rule
%                     that "alternates between random exploring and directional update" (p.5); the
%                     distribution is not stated in the paper (checked pages 5-6).
%     r               "the coefficient (r in [0, 1])" (p.3): U(0, 1).
%     randn(0, sigma^2)   Gaussian with mean 0 and standard deviation sigma^2 (p.5; p.4).
%     Draw sizes      no draw size is printed (checked pages 3-6). randn(0, sigma^2) of Eq. 14 is "a
%                     random number" (p.5): one draw per elite, so the first branch scales the whole
%                     vector. The paper's benchmark results support this reading: Tables 3-4 (pp.12-13)
%                     report mean 0 and SD 0 over 50 runs on F8 and F10 (Rastrigin, Griewank; n = 100,
%                     T = 300, p.7) at D = 10 and 30, which this loop reaches with one draw per elite and
%                     misses in part of the runs with one draw per dimension. r, "the coefficient
%                     (r in [0, 1])" (p.3), is drawn once per dimension in Eqs. 2, 3, 14 and 15: Eqs. 3-5
%                     are written per dimension d, and the BRO that HBC extends (ref. 35 of the paper,
%                     Rahkar Farshi 2021, Algorithm 1) draws r inside its loop over dimensions. p is one
%                     draw per elite.
%     k (Eq. 8)       an elite other than i, uniform ("k in [1, Nr], k ~= i", p.4, with the elites in the
%                     place of the roosters); sigma^2 uses the fitness of the last evaluation.
%     eps             "a minimum constant in the computer" (p.4): realmin.
%     Elite assignment  each ordinary player draws its elite uniformly and independently (p.5; the
%                     distribution is not stated in the paper, checked pages 5-6); drawn as Step 2
%                     prints, not read by the Eq. 2 move.
%     x_best          the best individual recorded so far ("the global optimal position", p.5; "Record
%                     the best individual", Alg. 3), constant during an iteration.
%     SD(x_d)         standard deviation of the whole population in dimension d (p.4), after the moves of
%                     iteration t; sample standard deviation (normalised by n - 1): the normalisation is
%                     not stated in the paper (checked page 4).
%     Units           metres: Tables 8 and 10 (pp.16-17) give a_i and d_i in metres and Table 11 (p.17)
%                     the targets in metres, so Eq. 23 is in metres, the benchmark's unit. Eq. 8 depends
%                     on a ratio of fitness values only, so the unit does not enter sigma^2.
%     Initial population  "Randomly initialize the population" (Alg. 3); the distribution is not stated
%                     in the paper (checked pages 3-6); uniform in [q_min, q_max]. Damage levels start
%                     at 0 and change only by the battle lines of Algorithm 3.
%     Stop            T iterations. Step 6 (p.6) names a fitness termination condition but states no
%                     value (checked pages 5-7 and 16-18).
%
%   Benchmark interface:
%     Budget: the initialisation needs n = 25 FK calls and every iteration n = 25: all players are
%       re-evaluated once, after the moves (Step 4, Fig. 1; Algorithm 3 has no evaluation inside its
%       loops). T = ikb_generations(opts, 25, 25, 100) = 79, so 25 + 79 x 25 = 2000 FK calls per
%       waypoint; a 40-member population would give ikb_generations(opts, 40, 40, 100) = 49. With
%       T = 79 the first Delta is round(79/2) = 40: Eqs. 4-5 act after iterations 40 and 60; the
%       subgroups are re-selected at t = 1, 10, 20, ..., 70. The initialisation and every iteration start
%       only if ikb_budget_left(opts) covers n. Outside the benchmark T = 100 (2525 FK). No final FK
%       check. Every option field is ignored (the budget is read only through ikb_generations and
%       ikb_budget_left).
%     Stop: T iterations only; info.converged is always false (the paper states no criterion value).
%     Warm start: q0 (clamped to the limits) is player 1 of the initial population, evaluated
%       unperturbed; the other 24 are uniform in [q_min, q_max]. q0 takes part in the first ranking
%       (t = 1), and the recorded best starts from the initial population, so q0 is returned unless a
%       later evaluation has a strictly lower Eq. 23 error.
%     Robot and units: Eqs. 18-22 and Table 10 (the paper's 7R arm) are replaced by robot.fk with
%       robot.n joints; positions in metres, joint angles in radians.
%     Joint limits: the joint ranges are the solution space. Players start inside it and respawn inside
%       the current safe area, whose bounds are set to the original bounds where they leave it (Alg. 3).
%       Eqs. 2, 15 and the second branch of Eq. 14 move inside the box spanned by two points within the
%       range; the first branch of Eq. 14, x (1 + randn), can leave it. The paper states no handling of
%       out-of-range positions (checked pages 3-7; the "Boundary processing" box of Fig. 1 follows the
%       safe-area update and is read as its bound reset), so every new position is clamped to
%       [q_min, q_max]; info.clamped counts the clamped coordinates.
%     Output: the best individual recorded; info.fitness (Eq. 23, m), info.err_pos (m) and info.err_rot
%       (rotation angle, rad, reported only) from the pose stored at its evaluation. solve_HBC('ops')
%       returns handles to the operators for unit tests.
%
%   Relation to the BRO of Rahkar Farshi (2021) (solve_BRO): the paper's own operators are used where they
%   differ. The winner of a battle moves (Eq. 15; in BRO it stays); the loser moves toward x_best by
%   Eq. 2, as in BRO; only the nb ordinary players fight, among themselves; on equal fitness player i
%   is the loser (Alg. 3, "if f(x_i,b) < f(x_j,b)", as the paper's own BRO Algorithm 1; the
%   Algorithm 1 of Rahkar Farshi makes i the victor); all players are evaluated once per iteration
%   (Step 4; BRO evaluates the damaged soldier after each duel). The damage threshold, Eq. 3, the safe
%   area and the Delta schedule are those of BRO.
%
%   Reading of a contradiction in the paper (loser move):
%     Step 3 (p.6) updates the ordinary players "by equations (2), (3) and (15)": the loser moves toward
%     x_best by Eq. 2, x_i(t+1) = x_i + r(x_best - x_i) (p.3), which is used. Algorithm 3 (p.6), the text
%     of p.5 and the conclusion (p.18) print Eq. 16 instead, x_i,b(t+1) = x_i,b + r(x_k,a - x_i,b), toward
%     the elite assigned by the level relationship (the abstract, p.1: "elite random guidance"). The
%     reading of Step 3 reproduces the paper's 7R results (Tables 11-12, pp.17-18, 100 x 300: errors of
%     1e-16 to 9e-8 m): on Table 10's arm with Table 11's targets (points 1, 5, 8, 10; 5 runs each; joint
%     ranges not stated, [-pi, pi]) this loop reaches median errors of 1e-16 to 5e-15 m with Eq. 2 and
%     4e-6 to 2e-5 m with Eq. 16.
%
%   Deviations from the paper:
%     - Iteration horizon T = 79 instead of 100 (benchmark rule); the Delta schedule runs over T.
%     - q0 is a member of the initial population (warm start).
%     - New positions clamped to [q_min, q_max] (not stated in the paper).
%     - The first Delta is Algorithm 1's round(T / ceil(log10 T)), not the text's "The initial value
%       Delta is log10(Tmax)" (p.4), which for T = 100 would shrink the safe area after iterations 2, 3,
%       5, 8, 12, 18, 27, 41, 62 and 93.
%     - Not stated in the paper, chosen as given under Parameters: the draw sizes, the distributions of
%       p and of the elite assignment, in-place updates of the ordinary players (later battles of
%       iteration t see the moved positions, as in BRO; Eq. 15 is written in place),
%       battles judged on the fitness of the last evaluation, damage levels kept across re-selections,
%       the SD normalisation, ties in the ranking (index order) and in the nearest-player search (first
%       in rank order), the uniform initial population.
%     - robot.fk instead of Eqs. 18-22 (the paper's arm).
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
        q_best = struct('params', hbc_params(), 'core', @hbc_core, 'counts', @hbc_counts, ...
                        'groups', @hbc_groups, 'sigma2', @hbc_sigma2, 'elite', @hbc_elite, ...
                        'battles', @hbc_battles, 'nearest', @hbc_nearest, 'duel', @hbc_duel, ...
                        'winner', @hbc_winner, 'loser', @hbc_loser, 'respawn', @hbc_respawn, ...
                        'shrink', @hbc_shrink, 'delta0', @hbc_delta0, 'schedule', @hbc_schedule, ...
                        'fitness', @hbc_fitness);
        info = [];
        return
    end
    [q_best, info] = hbc_core(robot, T_d, q0, opts, hbc_params());
end

% ==================================================================== main loop (Algorithm 3)
function [q_best, info] = hbc_core(robot, T_d, q0, opts, P)
% Algorithm 3 (p.6): initial population (q0 first); per iteration the subgroups (t = 1 and
% mod(t, Gn) = 0), the elites by Eq. 14, the battles of the ordinary players (Eqs. 15, 2, 3), the
% safe area (Eqs. 4-6) when t >= Delta, the evaluation of all players (Eq. 23) and the record of the best.
    t_start = tic;
    N  = P.N;
    na = hbc_counts(P);
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    n  = numel(lo);
    q0 = min(max(q0(:), lo), hi);
    T  = ikb_generations(opts, N, N, P.T);
    pd = T_d(1:3, 4);

    B = struct('f', NaN, 'q', q0, 'T', NaN(4), 'nfk', 0);
    iters = 0;  hist = zeros(1, 0);  regroup = false(1, 0);  role = zeros(N, 0);  tgt = zeros(N, 0);
    branch = zeros(2, 0);  moves = zeros(3, 0);  clamped = zeros(2, 0);  shrinks = zeros(1, 0);
    fin = struct('X', zeros(n, 0), 'F', zeros(1, 0), 'damage', zeros(1, 0), 'lb', lo, 'ub', hi);

    if ikb_budget_left(opts) >= N
        X = [q0, lo + rand(n, N - 1) .* (hi - lo)];                     % Step 1, q0 first
        [F, B] = evaluate(robot, pd, X, B);
        D  = zeros(1, N);                                               % damage levels
        lb = lo;
        ub = hi;
        Delta = hbc_delta0(T);                                          % Algorithm 1
        hist = NaN(1, T);  regroup = false(1, T);  role = zeros(N, T);  tgt = zeros(N, T);
        branch = zeros(2, T);  moves = zeros(3, T);  clamped = zeros(2, T);
        G = struct();
        for t = 1:T
            if ikb_budget_left(opts) < N, break; end
            if t == 1 || mod(t, P.Gn) == 0                              % Eq. 13, level relationship
                G = hbc_groups(F, P);
                regroup(t) = true;
            end
            Xe = zeros(n, na);                                          % Eq. 14 with Eq. 8
            for m = 1:na
                i = G.ea(m);
                o = G.ea(G.ea ~= i);
                k = o(randi(na - 1));
                [Xe(:, m), b] = hbc_elite(X(:, i), B.q, hbc_sigma2(F(i), F(k), P.eps));
                branch(b, t) = branch(b, t) + 1;
            end
            [X, D, moves(:, t), clamped(2, t)] = hbc_battles(X, F, D, G, B.q, lb, ub, lo, hi, P);
            clamped(1, t) = nnz(Xe < lo | Xe > hi);
            X(:, G.ea) = min(max(Xe, lo), hi);                          % elites of iteration t+1
            if t >= Delta                                               % Eqs. 4-6
                [lb, ub] = hbc_shrink(X, B.q, lo, hi);
                Delta = Delta + round(Delta / 2);
                shrinks(end+1) = t; %#ok<AGROW>
            end
            [F, B] = evaluate(robot, pd, X, B);                          % Steps 4-5
            iters = t;
            hist(t) = B.f;
            role(:, t) = G.role';
            tgt(:, t)  = G.k';
        end
        hist = hist(1:iters);  regroup = regroup(1:iters);  role = role(:, 1:iters);
        tgt = tgt(:, 1:iters);  branch = branch(:, 1:iters);  moves = moves(:, 1:iters);
        clamped = clamped(:, 1:iters);
        fin = struct('X', X, 'F', F, 'damage', D, 'lb', lb, 'ub', ub);
    end

    ep = NaN;  er = NaN;
    if ~isnan(B.f)
        ep = norm(B.T(1:3, 4) - pd);
        er = acos(min(1, max(-1, (trace(T_d(1:3, 1:3).' * B.T(1:3, 1:3)) - 1) / 2)));
    end

    q_best = B.q;
    info = struct();
    info.solver    = 'HBC';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = false;                                             % no criterion value in the paper
    info.fitness   = B.f;                                               % Eq. 23, m
    info.err_pos   = ep;                                                % m
    info.err_rot   = er;                                                % rad, reporting only
    info.fk_calls  = B.nfk;
    info.jac_calls = 0;
    info.N         = N;
    info.T         = T;
    info.history   = hist;                                              % recorded best after iteration t
    info.regroup   = regroup;                                           % subgroups re-selected at t
    info.role      = role;                                              % 1 elite, 2 ordinary
    info.target    = tgt;                                               % elite assigned to each ordinary player
    info.branch    = branch;                                            % Eq. 14 first / second branch
    info.moves     = moves;                                             % Eq. 15 moves, Eq. 2 moves, Eq. 3 respawns
    info.clamped   = clamped;                                           % clamped coordinates: elites, ordinary
    info.shrinks   = shrinks;                                           % iterations after which Eqs. 4-5 acted
    info.final     = fin;                                               % last population, Eq. 23, damage, safe area
    info.params    = P;
end

function [F, B] = evaluate(robot, pd, X, B)
% Eq. 23 for every player (one FK call each); the recorded best is replaced only by a strictly lower
% error ("Record the best individual and its fitness value", Alg. 3).
    N = size(X, 2);
    F = zeros(1, N);
    for i = 1:N
        [F(i), Tq] = hbc_fitness(robot, pd, X(:, i));
        B.nfk = B.nfk + 1;
        if isnan(B.f) || F(i) < B.f
            B.f = F(i);
            B.q = X(:, i);
            B.T = Tq;
        end
    end
end

function [X, D, cnt, ncl] = hbc_battles(X, F, D, G, xb, lb, ub, lo, hi, P)
% Ordinary-player loop of Algorithm 3: player i battles its nearest ordinary player j; the winner moves
% by Eq. 15 (damage 0); the loser moves by Eq. 2 toward x_best (Step 3) while its damage is below thre
% (damage + 1), otherwise it respawns by Eq. 3 in the current safe area (damage 0). Positions change
% in place; F holds the fitness of the last evaluation. Elite columns are not changed.
% cnt = [Eq. 15 moves; Eq. 2 moves; Eq. 3 respawns]; ncl = clamped coordinates.
    cnt = zeros(3, 1);
    ncl = 0;
    for m = 1:numel(G.eb)
        i = G.eb(m);
        j = hbc_nearest(X, G.eb, i);
        [d, v] = hbc_duel(F, i, j);
        u = hbc_winner(X(:, v), xb);                                    % Eq. 15
        ncl = ncl + nnz(u < lo | u > hi);
        X(:, v) = min(max(u, lo), hi);
        D(v) = 0;
        cnt(1) = cnt(1) + 1;
        if D(d) < P.thre
            u = hbc_loser(X(:, d), xb);                                 % Eq. 2 (Step 3)
            D(d) = D(d) + 1;
            cnt(2) = cnt(2) + 1;
        else
            u = hbc_respawn(lb, ub);                                    % Eq. 3
            D(d) = 0;
            cnt(3) = cnt(3) + 1;
        end
        ncl = ncl + nnz(u < lo | u > hi);
        X(:, d) = min(max(u, lo), hi);
    end
end

% ==================================================================== operators
function P = hbc_params()
% Table 9 (p.16): n = 25, T = 100 (setting nearest the budget); Table 2 (p.8): na = 0.2n, thre = 3,
% Gn = 10; eps = realmin (p.4).
    P = struct('N', 25, 'T', 100, 'ra', 0.2, 'thre', 3, 'Gn', 10, 'eps', realmin);
end

function [na, nb] = hbc_counts(P)
% na = 0.2n, nb = n - 0.2n (Table 2; Eq. 13).
    na = round(P.ra * P.N);
    nb = P.N - na;
end

function G = hbc_groups(F, P)
% Eq. 13 and the level relationship (p.5; Alg. 3): rank by fitness (ascending, ties in index order); the
% na best are the elites (ea), the rest the ordinary players (eb), both in rank order. Each ordinary
% player, in rank order, draws its elite uniformly (G.k; 0 for elites).
    na = hbc_counts(P);
    N  = numel(F);
    [~, ord] = sort(F, 'ascend');
    G.ea = ord(1:na);
    G.eb = ord(na + 1:N);
    G.role = zeros(1, N);
    G.role(G.ea) = 1;
    G.role(G.eb) = 2;
    G.k = zeros(1, N);
    for m = 1:numel(G.eb)
        G.k(G.eb(m)) = G.ea(randi(na));
    end
end

function s2 = hbc_sigma2(fi, fk, ep)
% Eq. 8: 1 if f_i <= f_k, otherwise exp((f_k - f_i) / |f_i + eps|).
    if fi <= fk
        s2 = 1;
    else
        s2 = exp((fk - fi) / abs(fi + ep));
    end
end

function [u, b] = hbc_elite(x, xb, s2)
% Eq. 14: p = U(0, 1); p >= 0.5: x * (1 + randn(0, sigma^2)), one Gaussian number with standard
% deviation sigma^2 (b = 1); p < 0.5: x + sigma^2 r .* (x_best - x), r one U(0, 1) per dimension (b = 2).
    p = rand;
    if p >= 0.5
        u = x * (1 + s2 * randn);
        b = 1;
    else
        u = x + s2 * rand(numel(x), 1) .* (xb - x);
        b = 2;
    end
end

function j = hbc_nearest(X, idx, i)
% Ordinary player of idx closest to player i (Euclidean distance in joint space); first in idx order
% on ties.
    d2 = sum((X(:, idx) - X(:, i)).^2, 1);
    d2(idx == i) = Inf;
    [~, m] = min(d2);
    j = idx(m);
end

function [d, v] = hbc_duel(F, i, j)
% Algorithm 3: d = i, v = j; d = j, v = i if f(x_i) < f(x_j). On equal fitness i is the loser.
    d = i;
    v = j;
    if F(i) < F(j)
        d = j;
        v = i;
    end
end

function u = hbc_winner(x, xb)
% Eq. 15: x + r .* (x_best - x), r one U(0, 1) per dimension.
    u = x + rand(numel(x), 1) .* (xb - x);
end

function u = hbc_loser(x, xb)
% Eq. 2 (Step 3): x + r .* (x_best - x), r one U(0, 1) per dimension.
    u = x + rand(numel(x), 1) .* (xb - x);
end

function u = hbc_respawn(lb, ub)
% Eq. 3: r .* (ub_d - lb_d) + lb_d, r one U(0, 1) per dimension, current safe area.
    u = rand(numel(lb), 1) .* (ub - lb) + lb;
end

function [lb, ub] = hbc_shrink(X, xb, lo, hi)
% Eqs. 4-5: x_best -/+ SD(x_d) of the population (sample SD), set to the original bounds where they
% leave the solution space (Alg. 3).
    sd = std(X, 0, 2);
    lb = max(xb - sd, lo);
    ub = min(xb + sd, hi);
end

function D = hbc_delta0(T)
% Algorithm 1: Shrink = ceil(log10(Tmax)), Delta = round(Tmax/Shrink); Inf for T = 1.
    D = round(T / ceil(log10(T)));
end

function s = hbc_schedule(T)
% Iterations t = 1..T after which Eqs. 4-5 act (t >= Delta, then Delta = Delta + round(Delta/2), Eq. 6).
    s = zeros(1, 0);
    D = hbc_delta0(T);
    for t = 1:T
        if t >= D
            s(end+1) = t; %#ok<AGROW>
            D = D + round(D / 2);
        end
    end
end

% ==================================================================== IK fitness (Eq. 23)
function [f, T] = hbc_fitness(robot, pd, q)
% Eq. 23: E_P = ||P - P_bar||, desired position pd against the actual position (m); one FK call.
    T = robot.fk(q);
    f = sqrt((pd(1) - T(1, 4))^2 + (pd(2) - T(2, 4))^2 + (pd(3) - T(3, 4))^2);
end
