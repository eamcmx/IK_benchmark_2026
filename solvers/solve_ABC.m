function [q_best, info] = solve_ABC(robot, T_d, q0, opts)
% SOLVE_ABC  Multi-strategy fusion artificial bee colony (MSABC) IK solver of Shi et al. (2024), Sec. 2.2.
%
%   Source:
%     SHI Jianping, XU Yongchi, GU Xun, CHEN Dongyun (2024). Improvement of Artificial Bee Colony Algorithm for
%     Inverse Kinematics of Redundant Manipulators. Journal of Mechanical Engineering, Vol. 60, No. 15, Aug. 2024,
%     pp. 60-70. DOI: 10.3901/JME.2024.15.060
%   The paper is in Chinese with an English title and abstract (p.1); the English title is quoted. Page numbers
%   below are PDF pages 1-11 (printed page = PDF page + 59; p.1 is the unnumbered printed page 60, p.11 is 70).
%
%   Algorithm (Sec. 2.2, pp.5-6; steps (1)-(5) of Sec. 2.2.5, p.6; IK fitness Eq. 15, p.4):
%     SN food sources; source i is the joint vector X_i = (x_i1 .. x_iD) (p.4). GB is the best food source of the
%     population (steps (2)-(4), p.6: "the current population best individual").
%     Step (1): all sources are initialised by
%       x_ij = l_j + (u_j - l_j) r                                                                         (Eq. 16)
%     Step (2), employed bees, i = 1..SN in turn, all D components (p.5):
%       v_j = (x_ij + gb_j) / 2 + F (x_k1j - x_k2j)   if r1 < C or j == n
%             x_ij                                    otherwise                                           (Eq. 18)
%       with k1 != k2 != i two random sources, r1 ~ U[0, 1], n a randomly selected dimension index (p.5).
%     Step (3), onlooker bees: each of the SN onlookers selects a source i by roulette and moves
%       V = w1 X_k3 + (1 - w1) X_i + w2 (X_k4 - X_k5) + w2 (r2 GB - X_i)                                  (Eq. 19)
%       w1 = (t / (t_max + 1))^2                                                                           (Eq. 20)
%       w2 = exp(-t / t_max)                                                                               (Eq. 21)
%       with X_k3, X_k4, X_k5 chosen with equal probability, k3 != k4 != k5 != i, r2 ~ U[0, 1] and t the
%       current iteration (p.5).
%     Step (4), scout bees: a source X_i that has not been updated for l consecutive times (Sec. 2.2.3, p.5)
%       V = X_i + (U - X_i) r3   if r4 < 0.5
%           L + (X_i - L) r3     if r4 >= 0.5                                                              (Eq. 22)
%       X_i = (V + GB) / 2                                                                                 (Eq. 23)
%       with r3, r4 ~ U[0, 1] and U, L the upper and lower boundary vectors (pp.5-6); by Eq. 22 the scout
%       "updates dimension by dimension" (p.6), then learns from GB by Eq. 23.
%     Out-of-range components (Sec. 2.2.4, p.6): each component of the intermediate candidate V of the three
%       phases that lies outside the search bounds is re-initialised by Eq. 16.
%     Selection (steps (2)-(4), p.6): each new individual is evaluated at once; it replaces its source in real
%       time if it is better than the source, and GB in real time if it is better than GB. Step (5): stop at the
%       maximum iteration count and output the best result.
%     IK fitness (Eq. 15, p.4), with n, o, a, p the columns of the pose (Eq. 2, p.3) and * the target:
%       f = min(lambda (|p*_x - p_x| + |p*_y - p_y| + |p*_z - p_z|) + (|n*_x - n_x| + |n*_y - n_y| + |n*_z - n_z|
%           + |o*_x - o_x| + |o*_y - o_y| + |o*_z - o_z| + |a*_x - a_x| + |a*_y - a_y| + |a*_z - a_z|))   (Eq. 15)
%     where min(.) is minimised over the joint vectors.
%
%   Parameters:
%     SN = 80, t_max = 200   the reported IK setting whose SN x t_max is nearest the 2000-FK budget. Sec. 3.1
%                     (p.6) runs the 7-DOF IK task 20 times per setting around the baseline SN = 80, t_max = 1000,
%                     F = 0.5, C = 0.9, l = 100, one parameter at a time: Table 1 (p.6) SN = 30..90 with
%                     t_max = 1000 (SN x t_max = 30,000..90,000); Table 2 (p.6) t_max = 200..1200 with SN = 80
%                     (16,000..96,000); Tables 3-5 (pp.6-7) and the final choice (p.7: "SN = 80, t_max = 1 000,
%                     F = 0.5, C = 0.9 and l = 100") 80 x 1000 = 80,000. Nearest: 80 x 200 = 16,000 (Table 2).
%                     SN is the number of employed bees and of onlooker bees (p.4). Inside the benchmark t_max is
%                     the budget horizon (Benchmark interface).
%     F = 0.5, C = 0.9, l = 100   p.7 (also the baseline of Sec. 3.1, p.6).
%     lambda = 0.01   Eq. 15, p.4: "according to the robot's D-H parameters, this paper sets the scale factor
%                     lambda = 0.01".
%     Roulette        the onlooker "selects an employed bee i at random by roulette" (p.5); the selection
%                     probability is not stated in the paper (checked pages 4-7); benchmark uses
%                     p_i = fit_i / sum(fit) with fit_i = 1 / (1 + f_i) (f >= 0 by Eq. 15), computed once per
%                     onlooker phase from the sources after the employed phase, and roulette-wheel selection with
%                     one U(0, 1) draw per onlooker.
%     k1, k2; k3, k4, k5   distinct sources other than i ("k1 != k2 != i", "k3 != k4 != k5 != i", p.5), drawn
%                     uniformly ("with equal probability", p.5) as an ordered pair or triple, once per move.
%     r1, n           r1: one U(0, 1) draw per component (Eq. 18 tests "r1 < C or j == n" per component j);
%                     n: one uniform index in 1..D per employed move.
%     r2              one U(0, 1) scalar per onlooker move (a random variable multiplying the vector GB, p.5).
%     r3, r4          one U(0, 1) draw each per component (Eq. 22 updates "dimension by dimension", p.6).
%     r of Eq. 16     one U(0, 1) draw per component: per joint of the initial population, per processed
%                     component of Sec. 2.2.4. A component exactly on a bound is not processed.
%     Trial counter   "not updated for l consecutive times" (p.5): set to 0 when a move replaces the source,
%                     increased by 1 when a move of the employed, onlooker or scout phase does not; a source with
%                     trial >= l is a scout in step (4), in index order. Which phases count, and the counter after
%                     a rejected scout move, are not stated in the paper (checked pages 4-6).
%     Ties            "better" (pp.5-6) is read as strictly lower f; GB of the initial population: the lowest
%                     index among equal values.
%     Units           the paper prints no length unit (checked pages 1-11). Its target (p.7) has
%                     p = (16.054 05, 66.859 50, 243.149 12) for the YuMi left arm (Sec. 1.1, p.2), i.e. 253 units
%                     from the base: millimetres (253 m is impossible for this arm). It is the target of Eq. 25 of
%                     Shi et al. (2020) for the same arm (Table 1 there in mm) rounded to 5 decimals. Eq. 15 with
%                     lambda = 0.01 applied to the Table 7 errors (p.8, position terms up to 3.77) gives the Table 6
%                     minima (p.7) of all seven algorithms. So the position terms are computed in mm (1000 x the
%                     distance in m) and lambda = 0.01 weights millimetres; n, o, a are unitless.
%     Stop            the maximum iteration count only (step (5), p.6); no accuracy stop. info.converged is always
%                     false: the paper defines no convergence criterion (checked pages 4-8).
%     Random draws    initial population: rand(D, SN - 1). Per employed move: k1, k2 (randperm over the ascending
%                     candidate list), n (randi), r1 (rand(D, 1)), then one r per out-of-range component in joint
%                     order. Per onlooker move: the roulette draw, k3, k4, k5 (randperm), r2, then the Eq. 16 draws.
%                     Per scout move: r3 (rand(D, 1)), r4 (rand(D, 1)), then the Eq. 16 draws on V.
%
%   Benchmark interface:
%     Budget: the initial population costs SN = 80 FK calls; every iteration SN employed + SN onlooker moves = 160
%       plus one per scout move (0 until a trial counter reaches l = 100). t_max = ikb_generations(opts, 80, 160,
%       200) = 12 iterations, so 80 + 12 x 160 = 2000 FK per waypoint when no scout fires. A 40-member population
%       would give ikb_generations(opts, 40, 80, 200) = 24. Each batch (the initial population; the employed and
%       onlooker phases of an iteration, 160; the scout phase) starts only if ikb_budget_left(opts) covers all its
%       evaluations; otherwise the run stops and returns, so a scout move inside the benchmark costs the last
%       iteration. Outside the benchmark t_max = 200: 32,080 FK plus the scout moves. No final FK check. Every
%       option field is ignored (the budget is read only through ikb_generations and ikb_budget_left).
%     Schedule: Eqs. 20-21 run over this t_max: with t_max = 12, w1 = 0.0059 and w2 = 0.9200 at t = 1, w1 = 0.8521
%       and w2 = 0.3679 at t = 12.
%     Stop: t_max iterations or the budget; no tolerance stop.
%     Warm start: q0 (clamped to the limits) is source 1 of the initial population and is evaluated; it is GB
%       unless another initial source is strictly better, and only a strictly better move replaces it.
%     Robot and units: Eqs. 1-14 (the paper's YuMi left arm) are replaced by robot.fk with D = robot.n joints;
%       lengths in m, converted to mm inside Eq. 15; angles in rad.
%     Joint limits: Sec. 2.2.4 (Eq. 16 re-initialisation of each out-of-range component), as the paper says; its
%       result lies inside [q_min, q_max], so no clamping is applied. info.border counts the processed components.
%     Output: GB; info.fitness (f, Eq. 15), info.f_pos (position sum in mm, before lambda), info.f_rot (n, o, a
%       sum), info.err_pos (m) and info.err_rot (rotation angle, rad, reported only), all from the pose stored at
%       the evaluation of GB. solve_ABC('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration horizon t_max = 12 instead of 200 (benchmark rules; the paper's final setting is 1000); Eqs.
%       20-21 run over it.
%     - q0 is source 1 of the initial population (warm start); the paper initialises every source by Eq. 16.
%     - Not stated in the paper, chosen as given under Parameters: the roulette probability, the size of the
%       draws of r1, r2, r3, r4, the draw of k1 .. k5, the trial-counter rule, ties, the order of the draws.
%     - robot.fk instead of Eqs. 1-14 (the paper's arm).
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
        q_best = struct('params', abc_params(), 'core', @abc_core, 'weights', @abc_weights, ...
                        'pick', @abc_pick, 'roulette', @abc_roulette, 'eq16', @abc_eq16, 'border', @abc_border, ...
                        'eq18', @abc_eq18, 'eq19', @abc_eq19, 'eq22', @abc_eq22, 'eq23', @abc_eq23, ...
                        'employed', @abc_employed, 'onlooker', @abc_onlooker, 'scout', @abc_scout, ...
                        'select', @abc_select, 'employed_phase', @abc_employed_phase, ...
                        'onlooker_phase', @abc_onlooker_phase, 'scout_phase', @abc_scout_phase, ...
                        'fitness', @abc_fitness, 'eq15', @abc_eq15);
        info = [];
        return
    end
    [q_best, info] = abc_core(robot, T_d, q0, opts, abc_params());
end

% ==================================================================== main loop (Sec. 2.2.5, steps (1)-(5))
function [q_best, info] = abc_core(robot, T_d, q0, opts, P)
% Step (1) (q0 first, all evaluated); per iteration: step (2) Eq. 18, step (3) Eqs. 19-21, step (4) Eqs. 22-23,
% each with Sec. 2.2.4 and real-time selection; stop at t_max or at the budget.
    t_start = tic;
    SN = P.SN;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    D  = numel(lo);
    q0 = min(max(q0(:), lo), hi);
    T  = ikb_generations(opts, SN, 2 * SN, P.tmax);
    fit = @(x) abc_fitness(robot, T_d, x, P);

    S = struct('X', zeros(D, 0), 'f', zeros(1, 0), 'trial', zeros(1, 0), 'g', q0, 'fg', NaN, 'Tg', NaN(4));
    fk = 0;  iters = 0;  cut = false;  f_init = NaN;  g_init = 0;  stop = 'budget';
    hist = zeros(1, 0);  W = zeros(2, 0);  acc = zeros(3, 0);  gup = zeros(3, 0);  nbd = zeros(3, 0);
    nsc = zeros(1, 0);

    if ikb_budget_left(opts) >= SN
        X = [q0, lo + rand(D, SN - 1) .* (hi - lo)];                     % step (1), Eq. 16, q0 first
        f = zeros(1, SN);  g = 1;  Tg = NaN(4);
        for i = 1:SN
            [f(i), Tq] = fit(X(:, i));
            fk = fk + 1;
            if i == 1 || f(i) < f(g), g = i;  Tg = Tq; end              % ties to the lowest index
        end
        S = struct('X', X, 'f', f, 'trial', zeros(1, SN), 'g', X(:, g), 'fg', f(g), 'Tg', Tg);
        f_init = S.fg;  g_init = g;
        hist = NaN(1, T);  W = zeros(2, T);  acc = zeros(3, T);  gup = zeros(3, T);  nbd = zeros(3, T);
        nsc = zeros(1, T);
        stop = 't_max';
        for t = 1:T
            if ikb_budget_left(opts) < 2 * SN, stop = 'budget'; break; end
            [S, ce] = abc_employed_phase(S, fit, P, lo, hi);              % step (2)
            [w1, w2] = abc_weights(t, T);                                 % Eqs. 20-21
            [S, co] = abc_onlooker_phase(S, fit, w1, w2, lo, hi);         % step (3)
            fk = fk + 2 * SN;
            W(:, t) = [w1; w2];
            acc(1:2, t) = [ce(1); co(1)];  gup(1:2, t) = [ce(2); co(2)];  nbd(1:2, t) = [ce(3); co(3)];
            hist(t) = S.fg;
            idx = find(S.trial >= P.l);                                   % Sec. 2.2.3
            if ikb_budget_left(opts) < numel(idx), cut = true;  stop = 'budget'; break; end
            [S, cs] = abc_scout_phase(S, fit, idx, lo, hi);               % step (4)
            fk = fk + numel(idx);
            nsc(t) = numel(idx);
            acc(3, t) = cs(1);  gup(3, t) = cs(2);  nbd(3, t) = cs(3);
            hist(t) = S.fg;
            iters = t;
        end
        k = iters + cut;
        hist = hist(1:k);  W = W(:, 1:k);  acc = acc(:, 1:k);  gup = gup(:, 1:k);  nbd = nbd(:, 1:k);
        nsc = nsc(1:k);
    end

    q_best = S.g;
    fp = NaN;  fr = NaN;  ep = NaN;  er = NaN;
    if ~isnan(S.fg)
        [~, fp, fr] = abc_eq15(S.Tg, T_d, P);
        ep = norm(S.Tg(1:3, 4) - T_d(1:3, 4));
        er = acos(min(1, max(-1, (trace(T_d(1:3, 1:3).' * S.Tg(1:3, 1:3)) - 1) / 2)));
    end

    info = struct();
    info.solver    = 'MSABC';
    info.iters     = iters;                                            % iterations completed (steps (2)-(4))
    info.time      = toc(t_start);
    info.converged = false;                                            % no criterion in the paper
    info.fitness   = S.fg;                                             % f, Eq. 15 (position in mm)
    info.fitness_init = f_init;                                        % best f of the initial population
    info.f_pos     = fp;                                               % position sum of Eq. 15, mm
    info.f_rot     = fr;                                               % n, o, a sum of Eq. 15
    info.err_pos   = ep;                                               % m
    info.err_rot   = er;                                               % rad, reported only
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.SN        = SN;
    info.T         = T;
    info.g_init    = g_init;                                           % source of GB after step (1)
    info.cut       = cut;                                              % scout phase of iteration iters + 1 not run
    info.stop      = stop;
    info.history   = hist;                                             % f of GB after iteration t
    info.w         = W;                                                % [w1; w2] of iteration t (Eqs. 20-21)
    info.accept    = acc;                                              % replacements: employed, onlooker, scout
    info.gbup      = gup;                                              % GB updates: employed, onlooker, scout
    info.border    = nbd;                                              % components processed by Sec. 2.2.4
    info.scouts    = nsc;                                              % scout moves in iteration t
    info.final     = struct('X', S.X, 'f', S.f, 'trial', S.trial, 'g', S.g, 'fg', S.fg);
    info.params    = P;
end

% ==================================================================== phases (steps (2)-(4), p.6)
function [S, c] = abc_employed_phase(S, fit, P, lo, hi)
% Step (2): sources i = 1..SN in turn; Eq. 18 with Sec. 2.2.4, evaluation and real-time selection.
% c = [replacements, GB updates, processed components].
    c = zeros(1, 3);
    for i = 1:size(S.X, 2)
        [v, ~, nb] = abc_employed(S.X, i, S.g, P.F, P.C, lo, hi);
        [S, a, u] = abc_select(S, i, v, fit);
        c = c + [a, u, nb];
    end
end

function [S, c] = abc_onlooker_phase(S, fit, w1, w2, lo, hi)
% Step (3): SN onlookers; roulette weights from the sources at the start of the phase; Eq. 19 with Sec. 2.2.4,
% evaluation and real-time selection.
    c = zeros(1, 3);
    cw = cumsum(1 ./ (1 + S.f));
    for m = 1:size(S.X, 2)
        i = abc_roulette(cw);
        [v, ~, nb] = abc_onlooker(S.X, i, S.g, w1, w2, lo, hi);
        [S, a, u] = abc_select(S, i, v, fit);
        c = c + [a, u, nb];
    end
end

function [S, c] = abc_scout_phase(S, fit, idx, lo, hi)
% Step (4): the sources idx (trial >= l) in index order; Eqs. 22-23 with Sec. 2.2.4 on V, evaluation and
% real-time (greedy) selection.
    c = zeros(1, 3);
    for i = idx(:).'
        [v, ~, nb] = abc_scout(S.X(:, i), S.g, lo, hi);
        [S, a, u] = abc_select(S, i, v, fit);
        c = c + [a, u, nb];
    end
end

function [S, a, u] = abc_select(S, i, v, fit)
% Steps (2)-(4): evaluate v; replace source i if strictly better (trial 0), otherwise trial + 1; replace GB if
% strictly better than GB.
    [f, Tq] = fit(v);
    a = f < S.f(i);
    u = false;
    if a
        S.X(:, i) = v;
        S.f(i) = f;
        S.trial(i) = 0;
        if f < S.fg
            S.g = v;  S.fg = f;  S.Tg = Tq;
            u = true;
        end
    else
        S.trial(i) = S.trial(i) + 1;
    end
end

% ==================================================================== moves (draws in the documented order)
function [v, d, nb] = abc_employed(X, i, g, F, C, lo, hi)
% Eq. 18 for source i, then Sec. 2.2.4. Draws: k1, k2; n; r1 (one per component); Eq. 16 draws.
    D  = size(X, 1);
    k  = abc_pick(size(X, 2), i, 2);
    n  = randi(D);
    r1 = rand(D, 1);
    v  = abc_eq18(X(:, i), g, X(:, k(1)), X(:, k(2)), F, C, r1, n);
    [v, nb] = abc_border(v, lo, hi);
    d = struct('k', k, 'n', n, 'r1', r1);
end

function [v, d, nb] = abc_onlooker(X, i, g, w1, w2, lo, hi)
% Eq. 19 for the followed source i, then Sec. 2.2.4. Draws: k3, k4, k5; r2; Eq. 16 draws.
    k  = abc_pick(size(X, 2), i, 3);
    r2 = rand;
    v  = abc_eq19(X(:, k(1)), X(:, i), X(:, k(2)), X(:, k(3)), g, w1, w2, r2);
    [v, nb] = abc_border(v, lo, hi);
    d = struct('k', k, 'r2', r2);
end

function [v, d, nb] = abc_scout(x, g, lo, hi)
% Eq. 22 for the abandoned source x, Sec. 2.2.4 on V, then Eq. 23. Draws: r3, r4 (one each per component);
% Eq. 16 draws.
    D  = numel(x);
    r3 = rand(D, 1);
    r4 = rand(D, 1);
    V  = abc_eq22(x, lo, hi, r3, r4);
    [V, nb] = abc_border(V, lo, hi);
    v  = abc_eq23(V, g);
    d = struct('r3', r3, 'r4', r4, 'V', V);
end

function k = abc_pick(SN, i, m)
% m distinct sources other than i, drawn uniformly as an ordered tuple ("k1 != k2 != i", "k3 != k4 != k5 != i").
    cand = 1:SN;
    cand(i) = [];
    k = cand(randperm(SN - 1, m));
end

function i = abc_roulette(cw)
% Roulette wheel over the cumulative weights cw (fit_i = 1 / (1 + f_i)): P(i) = fit_i / sum(fit); one draw.
    i = find(cw > rand * cw(end), 1, 'first');
end

% ==================================================================== equations
function P = abc_params()
% p.7 and Table 2 (p.6): SN = 80, t_max = 200, F = 0.5, C = 0.9, l = 100; Eq. 15 (p.4): lambda = 0.01;
% position terms in mm (len_scale = 1000 per m).
    P = struct('SN', 80, 'tmax', 200, 'F', 0.5, 'C', 0.9, 'l', 100, 'lambda', 0.01, 'len_scale', 1000);
end

function [w1, w2] = abc_weights(t, T)
% Eq. 20: w1 = (t / (t_max + 1))^2; Eq. 21: w2 = exp(-t / t_max).
    w1 = (t / (T + 1))^2;
    w2 = exp(-t / T);
end

function x = abc_eq16(l, u, r)
% Eq. 16: x_ij = l_j + (u_j - l_j) r.
    x = l + (u - l) .* r;
end

function [v, c] = abc_border(v, lo, hi)
% Sec. 2.2.4: each component outside [l_j, u_j] is re-initialised by Eq. 16, one draw each in joint order;
% c = their number.
    c = 0;
    for j = 1:numel(v)
        if v(j) > hi(j) || v(j) < lo(j)
            v(j) = abc_eq16(lo(j), hi(j), rand);
            c = c + 1;
        end
    end
end

function v = abc_eq18(x, g, x1, x2, F, C, r1, n)
% Eq. 18: v_j = (x_ij + gb_j) / 2 + F (x_k1j - x_k2j) if r1_j < C or j == n, x_ij otherwise.
    v = x;
    for j = 1:numel(x)
        if r1(j) < C || j == n
            v(j) = (x(j) + g(j)) / 2 + F * (x1(j) - x2(j));
        end
    end
end

function v = abc_eq19(xk3, xi, xk4, xk5, g, w1, w2, r2)
% Eq. 19: V = w1 X_k3 + (1 - w1) X_i + w2 (X_k4 - X_k5) + w2 (r2 GB - X_i).
    v = w1 * xk3 + (1 - w1) * xi + w2 * (xk4 - xk5) + w2 * (r2 * g - xi);
end

function V = abc_eq22(x, L, U, r3, r4)
% Eq. 22 per component: X_i + (U - X_i) r3 if r4 < 0.5; L + (X_i - L) r3 if r4 >= 0.5.
    V = x;
    for j = 1:numel(x)
        if r4(j) < 0.5
            V(j) = x(j) + (U(j) - x(j)) * r3(j);
        else
            V(j) = L(j) + (x(j) - L(j)) * r3(j);
        end
    end
end

function x = abc_eq23(V, g)
% Eq. 23: X_i = (V + GB) / 2.
    x = (V + g) / 2;
end

% ==================================================================== IK fitness (Eq. 15)
function [f, T] = abc_fitness(robot, T_d, q, P)
% Eq. 15 of the pose robot.fk(q) against T_d; one FK call.
    T = robot.fk(q);
    f = abc_eq15(T, T_d, P);
end

function [f, fp, fr] = abc_eq15(T, T_d, P)
% Eq. 15: fp = |p*_x - p_x| + |p*_y - p_y| + |p*_z - p_z| in mm; fr = sum of |n* - n|, |o* - o|, |a* - a| over
% x, y, z (columns 1-3, in the printed order); f = lambda fp + fr.
    fp = 0;
    for row = 1:3
        fp = fp + abs(T_d(row, 4) - T(row, 4));
    end
    fp = P.len_scale * fp;
    fr = 0;
    for col = 1:3
        for row = 1:3
            fr = fr + abs(T_d(row, col) - T(row, col));
        end
    end
    f = P.lambda * fp + fr;
end
