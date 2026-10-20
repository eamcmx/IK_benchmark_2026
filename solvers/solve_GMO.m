function [q_best, info] = solve_GMO(robot, T_d, q0, opts)
% SOLVE_GMO  Wild geese migration optimization (GMO) IK solver of Wu et al. (2022), Sec. 2 and Sec. 5.
%
%   Source:
%     Honggang Wu, Xinming Zhang, Linsen Song, Yufei Zhang, Lidong Gu and Xiaonan Zhao (2022). Wild Geese
%     Migration Optimization Algorithm: A New Meta-Heuristic Algorithm for Solving Inverse Kinematics of
%     Robot. Computational Intelligence and Neuroscience, Volume 2022, Article ID 5191758, 38 pages.
%     DOI: 10.1155/2022/5191758
%   Page numbers are PDF pages 1-38 (equal to the printed page numbers).
%
%   Algorithm (Sec. 2.2, pp.4-5; Fig. 4, p.6; Algorithm 1, p.7; IK fitness Eq. 17, p.33):
%     N geese, M head geese, b = N/M slots per migration group (p.4). The initial population is random
%     in the solution space and M head geese are selected at random (p.4; Fig. 4). In iteration t:
%       group formation (Eq. 1): slot i = b(j - 1) + 1 holds head j, x_i = x_j; every other slot of
%                 group j gets x_i = x_j - L + 2L rand(1,dim), a point within the radius L of head j.
%       one draw, rand > 0.5 (Algorithm 1): synchronized flight of every group, otherwise free foraging.
%       synchronized flight (Eqs. 2-3), members i of group j, equal steps within a group (p.3, p.4):
%                 x_i(t+1) = x_i + c1 (x_best - x_j) + c2 (x_k - x_j)                         (Eq. 2)
%                 c2 = exp((fit(j) - fit_ave) / (fit_worse - fit_best))           if fit(j) <= fit_ave
%                 c2 = exp((fit(j) + fit_ave - 2 fit_best) / (fit_worse - fit_best))  otherwise  (Eq. 3)
%                 fit_worse, fit_ave, fit_best: worst, average and best fitness of the head geese.
%       free foraging (Eq. 4):  x_i(t+1) = x_i + c3 (x_j - x_i + L) + c4 (x_best - x_j)
%       evaluation of all N geese; the best goose of each group becomes its head (Sec. 2.2.4, p.5;
%                 Algorithm 1); then the radius shrinks (Eq. 5): L = L (1 - 0.1 (t / T)).
%     Every right-hand side uses the positions of iteration t (superscripts t of Eqs. 1, 2, 4): the
%     heads x_j, x_k and x_best are held constant while the groups move. x_best is the global optimal
%     individual (p.4), the best goose evaluated so far ("Recording the optimal fitness value and its
%     individual location information data", Algorithm 1), which is also returned.
%     IK fitness (Eq. 17, p.33) with the columns n, o, alpha and the position p of the pose (Eq. 16):
%       f(x) = sum(|n - n^| + |o - o^| + |alpha - alpha^| + gamma |p - p^|)
%     with |.| elementwise and sum over the three rows: the L1 distance of the rotation columns plus
%     gamma times the L1 distance of the positions.
%
%   Parameters:
%     N = 100, T = 500  the only IK setting reported (Sec. 5.2, p.33): N x T = 100 x 500 = 50,000; the
%                     CEC2017 runs use the same N and T (Sec. 3.1, p.6). Inside the benchmark T is the
%                     budget horizon (Benchmark interface).
%     M = 20          Table 2 (p.8), the CEC2017 settings; the IK section (p.33) does not restate M, and
%                     Table 2 gives the only value. b = N/M = 5 (p.4).
%     L = (ud - ld)/N   initial radius (Table 2, p.8; p.4 prints "L = u d - l d/N" without the
%                     parentheses). The search range of the IK runs is not
%                     stated in the paper (checked pages 3-7 and 32-35; Table 23 gives the DH parameters
%                     only); benchmark uses ud = q_max and ld = q_min, so L is a per-joint vector.
%     gamma = 1       the adjustment factor of Eq. 17 (Sec. 5.2, p.33).
%     0.1, 0.5        the constants of Eq. 5 (p.5) and of the phase test (Algorithm 1, p.7).
%     c1              "The flight step size c1 in [0, 1]" (p.4); Table 2 lists no value for it, so it
%                     is read as a random number in [0, 1], like c3 and c4, uniform. One c1 per group
%                     and iteration: "the flight steps in the migration group members are set to be
%                     equal" (p.4), "individuals in each migration group update their positions in
%                     equal steps" (p.3).
%     k               "the randomly selected head goose" (p.4), one per group and iteration (equal
%                     steps), uniform among the other M - 1 heads: c2 controls "the proportion of other
%                     head geese's experience information" (p.4).
%     c3, c4          "random numbers between [0, 1]" (p.5), uniform, one scalar each per goose and
%                     iteration; the distribution and the size of the draw are not stated in the paper
%                     (checked pages 3-7).
%     rand(1,dim)     of Eq. 1: one U(0, 1) per joint for every goose that is not a head.
%     Eq. 1 loop      Algorithm 1 prints "For j = 1:M, For i = 1:N" around Eq. 1, which taken
%                     literally rebuilds every slot around every head in turn; the nearest well-defined
%                     reading, used here, is the slot ranges i = b(j - 1) + 1 : bj of the other two
%                     loops of Algorithm 1, so group j is rebuilt around head j only.
%     Initial heads   M distinct geese of the initial population, drawn uniformly ("M head goose
%                     individuals are randomly selected", Fig. 4, p.6), in the order drawn.
%     Initial population   random in the solution space (p.4; p.5 Sec. 2.3); the distribution is not
%                     stated in the paper (checked pages 3-7); uniform in [q_min, q_max]. All N are
%                     evaluated: Eq. 3 needs the fitness of the heads and Eqs. 2 and 4 need x_best at
%                     t = 1 (initialization O(N), Sec. 2.4, p.5).
%     Phase test      Algorithm 1: rand > 0.5 gives the synchronized flight. Fig. 4 (p.6) draws the
%                     branches the other way round; with probability 0.5 each the two are the same.
%     Eq. 3 at fit_worse = fit_best   all head fitness values are equal, the numerator fit(j) - fit_ave
%                     is 0 and the quotient 0/0 is undefined; c2 = exp(0) = 1, its value for any
%                     positive denominator.
%     Order           evaluation, new heads, then Eq. 5 (Fig. 4; Sec. 2.2.4: "After the head geese are
%                     all replaced, the migration group radius (L) is reduced by equation (5)"); Eqs. 1
%                     and 4 of iteration t use the radius before that update. Ties in a group: the
%                     lowest slot index.
%     Units           metres: Table 23 (p.33) gives a_i and d_i in metres, so the position term of
%                     Eq. 17 is in metres, the benchmark's unit; angles in rad.
%     Stop            T iterations (Algorithm 1, "For t = 1: T"); Fig. 4's "Termination criterion is
%                     met" is not defined, and no tolerance is stated in the paper (checked pages 3-7
%                     and 32-35).
%     Bounds          the handling of out-of-range values is not stated in the paper (checked pages
%                     3-7 and 32-35).
%     Random draws    per iteration: Eq. 1 (groups in order, per goose one U(0, 1) per joint), the
%                     phase draw, then per group k and c1 (flight) or per goose c3 and c4 (foraging).
%
%   Benchmark interface:
%     Budget: the initial population costs N = 100 FK calls and every iteration N = 100 (every goose
%       moves and is evaluated). T = ikb_generations(opts, 100, 100, 500) = 19 iterations, so
%       100 + 19 x 100 = 2000 FK per waypoint. A 40-member population would give
%       ikb_generations(opts, 40, 40, 500) = 49. The initial population and every iteration start
%       only if ikb_budget_left(opts) covers N. Outside the benchmark T = 500 (50,100 FK). No final FK
%       check. Every option field is ignored (the budget is read only through ikb_generations and
%       ikb_budget_left).
%     Schedule: Eq. 5 runs over this T, so the radius used in iteration t is
%       L0 prod_{s<t} (1 - 0.1 s / T): with T = 19 the last iteration uses 0.394 L0 (0.355 L0 after
%       the last update); with T = 500 the last iteration uses 6.1e-12 L0. L0 = (q_max - q_min)/100,
%       0.126 rad on a joint with limits of +/-2 pi.
%     Stop: T iterations. info.converged is always false: the paper has no convergence criterion.
%     Warm start: q0 (clamped to the limits) is goose 1 of the initial population and is evaluated;
%       it takes part in the random choice of the initial heads like every other goose, and as the
%       recorded best it is x_best of Eqs. 2 and 4 in the first iteration whenever no other goose of
%       the initial population is strictly better. q0 is returned unless a later evaluation has a
%       strictly lower Eq. 17 fitness.
%     Robot and units: Eqs. 14-15 and Table 23 (the paper's IRB5400) are replaced by robot.fk with
%       dim = robot.n joints; positions in metres, joint angles in radians.
%     Joint limits: every position made by Eq. 1 and every position made by Eqs. 2 and 4 is clamped
%       to [q_min, q_max] (not stated in the paper); info.clamped counts the clamped coordinates.
%     Output: the recorded best; info.fitness (Eq. 17), info.err_pos (m) and info.err_rot (rotation
%       angle, rad, reported only) from the pose stored at its evaluation.
%       solve_GMO('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration horizon T = 19 instead of 500 (benchmark rule); Eq. 5 runs over it.
%     - q0 is a member of the initial population (warm start).
%     - Positions clamped to [q_min, q_max] (bounds not stated).
%     - Not stated in the paper, chosen as given under Parameters: the search range (joint limits),
%       c1 random, the uniform distributions, the size of the draws of c1, c3, c4 and k, k ~= j, the
%       initial distribution, the evaluation of the whole initial population, x_best as the best so
%       far, Eq. 3 when all head fitness values are equal, ties within a group.
%     - Eq. 1 applied per group (the nearest reading of Algorithm 1's loop).
%     - robot.fk instead of Eqs. 14-15 (the paper's arm).
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
        q_best = struct('params', gmo_params(), 'core', @gmo_core, 'init_heads', @gmo_init_heads, ...
                        'formation', @gmo_formation, 'c2', @gmo_c2, 'flight', @gmo_flight, ...
                        'forage', @gmo_forage, 'heads', @gmo_heads, 'radius', @gmo_radius, ...
                        'fitness', @gmo_fitness, 'eq17', @gmo_eq17);
        info = [];
        return
    end
    [q_best, info] = gmo_core(robot, T_d, q0, opts, gmo_params());
end

% ==================================================================== main loop (Algorithm 1)
function [q_best, info] = gmo_core(robot, T_d, q0, opts, P)
% Initial population (q0 first), all evaluated, random initial heads; per iteration Eq. 1, the phase
% draw, Eqs. 2-3 or Eq. 4 from the positions of iteration t, clamp, evaluation (Eq. 17), record of
% the best, best of each group as its head, Eq. 5.
    t_start = tic;
    N  = P.N;
    M  = P.M;
    b  = N / M;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    n  = numel(lo);
    q0 = min(max(q0(:), lo), hi);
    T  = ikb_generations(opts, N, N, P.T);
    L  = (hi - lo) / N;                                                % Table 2: L = (ud - ld)/N
    L0 = L;

    bf = NaN;  bq = q0;  bT = NaN(4);  fk = 0;  iters = 0;
    X = zeros(n, 0);  F = zeros(1, 0);  H = zeros(n, 0);  FH = zeros(1, 0);  h0 = zeros(1, 0);
    hist = zeros(1, 0);  phase = zeros(1, 0);  Lh = zeros(n, 0);  hidx = zeros(M, 0);
    kk = zeros(M, 0);  cc = zeros(M, 0);  clamped = zeros(2, 0);  xbh = zeros(n, 0);

    if ikb_budget_left(opts) >= N
        X = [q0, lo + rand(n, N - 1) .* (hi - lo)];                    % p.4, q0 first
        [F, bf, bq, bT] = evaluate(robot, T_d, X, bf, bq, bT, P.gamma);
        fk = N;
        h0 = gmo_init_heads(N, M);                                     % Fig. 4: random heads
        H  = X(:, h0);
        FH = F(h0);
        hist = NaN(1, T);  phase = zeros(1, T);  Lh = zeros(n, T);  hidx = zeros(M, T);
        kk = zeros(M, T);  cc = NaN(M, T);  clamped = zeros(2, T);  xbh = zeros(n, T);
        for t = 1:T
            if ikb_budget_left(opts) < N, break; end
            xb = bq;                                                   % x_best of iteration t
            [Xt, clamped(1, t)] = gmo_formation(H, L, b, lo, hi);       % Eq. 1
            if rand > P.p                                              % Algorithm 1
                [Xn, kk(:, t), cc(:, t)] = gmo_flight(Xt, H, FH, xb, b);   % Eqs. 2-3
                phase(t) = 1;
            else
                Xn = gmo_forage(Xt, H, xb, L, b);                      % Eq. 4
                phase(t) = 2;
            end
            clamped(2, t) = nnz(Xn < lo | Xn > hi);
            X = min(max(Xn, lo), hi);                                  % joint limits
            [F, bf, bq, bT] = evaluate(robot, T_d, X, bf, bq, bT, P.gamma);
            fk = fk + N;
            [H, FH, hidx(:, t)] = gmo_heads(X, F, M, b);               % Sec. 2.2.4
            Lh(:, t) = L;
            xbh(:, t) = xb;
            L = gmo_radius(L, t, T, P.shrink);                         % Eq. 5
            iters = t;
            hist(t) = bf;
        end
        hist = hist(1:iters);  phase = phase(1:iters);  Lh = Lh(:, 1:iters);  hidx = hidx(:, 1:iters);
        kk = kk(:, 1:iters);  cc = cc(:, 1:iters);  clamped = clamped(:, 1:iters);  xbh = xbh(:, 1:iters);
    end

    ep = NaN;  er = NaN;
    if ~isnan(bf)
        ep = norm(bT(1:3, 4) - T_d(1:3, 4));
        er = acos(min(1, max(-1, (trace(T_d(1:3, 1:3).' * bT(1:3, 1:3)) - 1) / 2)));
    end

    q_best = bq;
    info = struct();
    info.solver    = 'GMO';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = false;                                           % no criterion in the paper
    info.fitness   = bf;                                              % Eq. 17
    info.err_pos   = ep;                                              % m
    info.err_rot   = er;                                              % rad, reporting only
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.N         = N;
    info.M         = M;
    info.b         = b;
    info.T         = T;
    info.L0        = L0;
    info.history   = hist;                                            % recorded best after iteration t
    info.phase     = phase;                                           % 1 synchronized flight, 2 free foraging
    info.L         = Lh;                                              % radius used in iteration t
    info.heads0    = h0;                                              % initial heads (indices)
    info.heads     = hidx;                                            % heads chosen after iteration t (indices)
    info.k         = kk;                                              % k of each group (flight), 0 otherwise
    info.c1        = cc;                                              % c1 of each group (flight), NaN otherwise
    info.xbest     = xbh;                                             % x_best used in iteration t
    info.clamped   = clamped;                                         % clamped coordinates after Eq. 1 / Eqs. 2, 4
    info.final     = struct('X', X, 'F', F, 'H', H, 'FH', FH, 'L', L);
    info.params    = P;
end

function [F, bf, bq, bT] = evaluate(robot, T_d, X, bf, bq, bT, gamma)
% Eq. 17 for every goose (one FK call each); the recorded best is replaced only by a strictly lower
% value (Algorithm 1, "Recording the optimal fitness value").
    N = size(X, 2);
    F = zeros(1, N);
    for i = 1:N
        [F(i), Tq] = gmo_fitness(robot, T_d, X(:, i), gamma);
        if isnan(bf) || F(i) < bf
            bf = F(i);
            bq = X(:, i);
            bT = Tq;
        end
    end
end

% ==================================================================== operators
function P = gmo_params()
% Sec. 5.2 (p.33): N = 100, T = 500, gamma = 1; Table 2 (p.8): M = 20; Eq. 5 (p.5): 0.1;
% Algorithm 1 (p.7): rand > 0.5.
    P = struct('N', 100, 'M', 20, 'T', 500, 'gamma', 1, 'shrink', 0.1, 'p', 0.5);
end

function h = gmo_init_heads(N, M)
% Fig. 4: M head geese drawn uniformly without replacement from the N geese, in the order drawn.
    pool = 1:N;
    h = zeros(1, M);
    for j = 1:M
        r = randi(numel(pool));
        h(j) = pool(r);
        pool(r) = [];
    end
end

function [X, ncl] = gmo_formation(H, L, b, lo, hi)
% Eq. 1: slot b(j - 1) + 1 = x_j; the other b - 1 slots of group j = x_j - L + 2L rand(1,dim); clamp.
    [n, M] = size(H);
    X = zeros(n, M * b);
    for j = 1:M
        i0 = b * (j - 1) + 1;
        X(:, i0) = H(:, j);
        X(:, i0 + 1:b * j) = H(:, j) - L + 2 * L .* rand(n, b - 1);
    end
    ncl = nnz(X < lo | X > hi);
    X = min(max(X, lo), hi);
end

function c2 = gmo_c2(fj, fa, fw, fb)
% Eq. 3; fw = fb (all head fitness values equal, numerator 0) gives exp(0) = 1.
    if fw == fb
        c2 = 1;
    elseif fj <= fa
        c2 = exp((fj - fa) / (fw - fb));
    else
        c2 = exp((fj + fa - 2 * fb) / (fw - fb));
    end
end

function [Xn, k, c1] = gmo_flight(Xt, H, FH, xb, b)
% Eqs. 2-3 for every group: one k (uniform among the other heads) and one c1 ~ U(0, 1) per group, so
% every member of group j takes the same step c1 (x_best - x_j) + c2 (x_k - x_j); heads, their fitness
% and x_best are those of iteration t.
    M  = size(H, 2);
    fw = max(FH);
    fa = mean(FH);
    fb = min(FH);
    Xn = Xt;
    k  = zeros(M, 1);
    c1 = zeros(M, 1);
    for j = 1:M
        o = [1:j - 1, j + 1:M];
        k(j)  = o(randi(M - 1));
        c1(j) = rand;
        c2 = gmo_c2(FH(j), fa, fw, fb);
        step = c1(j) * (xb - H(:, j)) + c2 * (H(:, k(j)) - H(:, j));
        idx = b * (j - 1) + 1:b * j;
        Xn(:, idx) = Xt(:, idx) + step;
    end
end

function Xn = gmo_forage(Xt, H, xb, L, b)
% Eq. 4 for every goose of group j: x_i + c3 (x_j - x_i + L) + c4 (x_best - x_j), one c3 and one c4
% ~ U(0, 1) per goose; heads and x_best are those of iteration t.
    M  = size(H, 2);
    Xn = Xt;
    for j = 1:M
        for i = b * (j - 1) + 1:b * j
            c3 = rand;
            c4 = rand;
            Xn(:, i) = Xt(:, i) + c3 * (H(:, j) - Xt(:, i) + L) + c4 * (xb - H(:, j));
        end
    end
end

function [H, FH, hi] = gmo_heads(X, F, M, b)
% Sec. 2.2.4, Algorithm 1: the best goose of each group (slots b(j - 1) + 1 : bj) is its new head;
% ties go to the lowest slot.
    H  = zeros(size(X, 1), M);
    FH = zeros(1, M);
    hi = zeros(M, 1);
    for j = 1:M
        idx = b * (j - 1) + 1:b * j;
        [FH(j), m] = min(F(idx));
        hi(j) = idx(m);
        H(:, j) = X(:, hi(j));
    end
end

function L = gmo_radius(L, t, T, s)
% Eq. 5: L = L (1 - 0.1 (t / T)).
    L = L * (1 - s * (t / T));
end

% ==================================================================== IK fitness (Eq. 17)
function [f, T] = gmo_fitness(robot, T_d, q, gamma)
% Eq. 17 of the pose robot.fk(q) against T_d; one FK call.
    T = robot.fk(q);
    f = gmo_eq17(T, T_d, gamma);
end

function f = gmo_eq17(T, T_d, gamma)
% Eq. 17: sum(|n - n^| + |o - o^| + |alpha - alpha^| + gamma |p - p^|), |.| elementwise, sum over
% the three rows; n, o, alpha, p = columns 1-4 of the pose (Eq. 16).
    v = abs(T(1:3, 1) - T_d(1:3, 1)) + abs(T(1:3, 2) - T_d(1:3, 2)) + abs(T(1:3, 3) - T_d(1:3, 3)) ...
        + gamma * abs(T(1:3, 4) - T_d(1:3, 4));
    f = v(1) + v(2) + v(3);
end
