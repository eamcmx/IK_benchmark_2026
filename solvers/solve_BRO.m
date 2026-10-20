function [q_best, info] = solve_BRO(robot, T_d, q0, opts)
% SOLVE_BRO  Battle royale optimization (BRO) IK solver of Rahkar Farshi (2021), Secs. 2.2 and 4.
%
%   Source:
%     Taymaz Rahkar Farshi (2021). Battle royale optimization algorithm. Neural Computing and
%     Applications, 33, 1139-1157 (no issue number printed). DOI: 10.1007/s00521-020-05004-4
%   Page numbers are PDF pages 1-19 (printed page = PDF page + 1138).
%
%   Algorithm (Sec. 2.2, Eqs. 1-3, p.4; Fig. 2, p.5; Algorithm 1, p.6; IK application Sec. 4,
%   Eqs. 4-6, pp.13-16, Tables 9-10, p.18):
%     N soldiers are drawn uniformly in the problem space and evaluated (Alg. 1). Each iteration
%     visits the soldiers i = 1..N in turn: soldier i duels its nearest soldier j (Euclidean
%     distance in joint space, p.4); the soldier with the higher cost is the damaged one, dam, the
%     other the victor, vic (Alg. 1 is written for maximisation and p.9 converts it to minimisation;
%     on equal cost i is the victor, as in Alg. 1).
%       dam.damage < Threshold: dam moves by Eq. 1, x_dam,d = x_dam,d + r(x_best,d - x_dam,d),
%         toward x_best, "the best position found so far (elite player)" (p.4); dam.damage += 1.
%       otherwise: dam dies and respawns by Eq. 2, x_dam,d = r(ub_d - lb_d) + lb_d, inside the
%         current bounds; dam.damage = 0.
%     Then vic.damage = 0 and f(x_dam) is evaluated (Fig. 2). Positions and x_best change at once,
%     so later duels of the same iteration see them.
%     After the N duels, if iter >= Delta the bounds shrink around x_best by Eq. 3,
%     lb_d = x_best,d - SD(x_d), ub_d = x_best,d + SD(x_d), SD the standard deviation of the
%     population in dimension d, are clipped to the original bounds, and Delta = Delta +
%     round(Delta/2); the first Delta is round(T/ceil(log10 T)) (Alg. 1). A shrink moves no
%     soldier; only respawns use the new bounds.
%     The run ends after T iterations (MaxCicle, p.4); the best soldier is returned (Alg. 1).
%     Fitness (Sec. 4, p.14 and p.16): the Euclidean distance between the obtained and the target
%     end-effector positions, d = sqrt((x' - x)^2 + (y' - y)^2 + (z' - z)^2). The paper has no
%     orientation term, so orientation does not enter this solver.
%
%   Parameters:
%     N = 25, T = 100   Table 10 (p.18), first row. Table 10 reports eleven IK settings on the
%                     PUMA 560 (populations 25-300, 100-500 iterations, 50 runs each); by the
%                     benchmark rule for several reported IK settings, the one whose N x T is
%                     nearest the 2000-FK budget is used: 25 x 100 = 2500 evaluations (mean error
%                     0.00037). Not used: the other ten rows (the next nearest, 50 x 100, is 5000;
%                     the most accurate, 300 x 500, is 150,000) and N = 100 / T = 500 of the
%                     benchmark-function tests (Sec. 3, p.9). Inside the benchmark the horizon is T
%                     (Benchmark interface).
%     Threshold = 3   p.4 ("a value of threshold = 3 was appropriate"); Table 4 (p.10) tunes 2-7.
%     Delta           first value round(T/ceil(log10 T)), then Delta + round(Delta/2) (Alg. 1,
%                     p.6); for T = 100 Eq. 3 acts after iterations 50 and 75.
%     r               U(0, 1), one draw per dimension in Eqs. 1 and 2: Alg. 1 evaluates both
%                     inside "for d=1: Dimension"; p.4 calls r "a randomly generated number" and
%                     does not say whether it is redrawn per dimension. Eq. 1 then puts dam at a
%                     uniform point of the box spanned by x_dam and x_best.
%     SD              sample standard deviation (normalised by N - 1); the normalisation is not
%                     stated (p.4).
%
%   Benchmark interface:
%     Budget: the initialisation needs N = 25 FK calls and every iteration N (one evaluation per
%       duel, of the damaged soldier). T = ikb_generations(opts, 25, 25, 100) = 79, 2000 FK calls
%       per waypoint; a 40-member population would give ikb_generations(opts, 40, 40, 100) = 49.
%       With T = 79 the first Delta is round(79/ceil(log10 79)) = 40: Eq. 3 acts after iterations
%       40 and 60. The initialisation and every iteration start only if ikb_budget_left(opts)
%       covers them. Outside the benchmark T = 100. No final FK check.
%       opts.max_iter and every other option field are ignored.
%     Stop: T iterations only (Alg. 1; p.4 and Sec. 4 give no tolerance); info.converged is always
%       false, as the paper has no convergence criterion.
%     Warm start: q0 (clamped to the limits) is soldier 1 of the initial population, evaluated
%       unperturbed. It is x_best from the initialisation on and is replaced only by a strictly
%       lower cost; a soldier is damaged only by an opponent of lower or equal cost (on equal cost
%       soldier i wins, Alg. 1).
%     Robot and units: Eqs. 4-6 and Table 9 (PUMA 560) are replaced by robot.fk and the robot's
%       joint limits. Table 9 (p.18) is in metres, as the benchmark; f is in metres.
%     Joint limits: the joint ranges are the bounds of the problem space (p.14, Table 9). Soldiers
%       start and respawn inside the bounds, Eq. 3 bounds are clipped to the original bounds and
%       Eq. 1 moves inside the box of two points within them, so no operator leaves [q_min, q_max];
%       every new position is clamped to [q_min, q_max] as a guard against floating-point rounding.
%     Output: the lowest-cost soldier evaluated (x_best). info.fitness = info.err_pos (m);
%       info.err_rot (rotation angle, rad) is reported only. solve_BRO('ops') returns handles to
%       the operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration horizon T = 79 instead of 100 (benchmark rule); the Delta schedule runs over T.
%     - Algorithm 1 prints Eq. 1 as r(max(x_dam,d, x_best,d) - min(x_dam,d, x_best,d)) +
%       max(x_dam,d, x_best,d), which places dam beyond the larger coordinate: outside the segment
%       the text describes ("a point somewhere between the previous position and best position",
%       p.4) and, from a bound, outside the problem space. Eq. 1 as printed on p.4 is used.
%     - The first Delta is Algorithm 1's round(T/ceil(log10 T)), not the text's "The initial value
%       was Delta = log10(MaxCicle)" (p.4). Fig. 5 (p.17, iterations 1 to 500; 500 iterations in
%       Sec. 3, p.9) shows the population still spread over the whole domain at iteration 100 and
%       inside a shrunk box at iteration 200, as Algorithm 1 gives for T = 500 (first shrink after
%       iteration 167); the text's value would shrink after iterations 3, 4, 6, 9, ..., 94.
%     - Fig. 5 shows soldiers lying on the edges of the shrunk box, which a projection of soldiers
%       into the current bounds would produce; no such step is stated (pp.4-6) and none is applied.
%     - Where Fig. 2 and Algorithm 1 differ, Fig. 2 is followed: the damaged soldier's own damage
%       is incremented (Alg. 1 prints x_i.damage + 1), and vic.damage = 0 and the evaluation of
%       f(x_dam) follow both branches (Alg. 1 has the first only after Eq. 1, the second only after
%       Eq. 2). Where both agree they are followed against the text: dam dies once its damage has
%       reached the threshold (p.4 says "surpasses").
%     - Eq. 3 acts after the N duels of an iteration (Alg. 1); Fig. 2 draws it inside the loop.
%     - r per dimension and SD normalised by N - 1 (not stated; see Parameters).
%     - New positions clamped to [q_min, q_max] against floating-point rounding (Benchmark
%       interface).
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
        q_best = struct('params', bro_params(), 'core', @bro_core, 'sweep', @bro_sweep, ...
                        'nearest', @bro_nearest, 'duel', @bro_duel, 'move', @bro_move, ...
                        'respawn', @bro_respawn, 'shrink', @bro_shrink, 'delta0', @bro_delta0, ...
                        'schedule', @bro_schedule, 'fitness', @bro_fitness);
        info = [];
        return
    end
    [q_best, info] = bro_core(robot, T_d, q0, opts, bro_params());
end

% ==================================================================== main loop (Alg. 1, Fig. 2)
function [q_best, info] = bro_core(robot, T_d, q0, opts, P)
% Alg. 1 (p.6): initial population, then per iteration N duels (bro_sweep) and the Eq. 3 shrink
% when iter >= Delta. P holds N, T (MaxCicle) and the damage threshold.
    t_start = tic;
    N  = P.N;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    n  = numel(lo);
    q0 = min(max(q0(:), lo), hi);
    T  = ikb_generations(opts, N, N, P.T);
    fit = @(q) bro_fitness(robot, T_d, q);

    B = struct('f', NaN, 'q', q0, 'T', NaN(4), 'nfk', 0);
    hist = zeros(1, 0);  nmove = zeros(1, 0);  shrinks = zeros(1, 0);  iters = 0;
    fin = struct('X', [], 'F', [], 'damage', [], 'lb', lo, 'ub', hi);

    if ikb_budget_left(opts) >= N
        % uniform random population (p.4, Alg. 1); q0 is soldier 1
        X = lo + rand(n, N) .* (hi - lo);
        X(:, 1) = q0;
        F = zeros(1, N);
        for i = 1:N
            [F(i), B] = eval_keep(fit, X(:, i), B);
        end
        D  = zeros(1, N);                                    % damage levels
        lb = lo;
        ub = hi;
        Delta = bro_delta0(T);                               % Alg. 1
        hist = zeros(1, T);  nmove = zeros(1, T);
        t = 0;
        while t < T                                          % MaxCicle (p.4)
            if ikb_budget_left(opts) < N, break; end
            t = t + 1;
            [X, F, D, B, nmove(t)] = bro_sweep(X, F, D, B, lb, ub, lo, hi, fit, P);
            if t >= Delta                                    % Alg. 1: Eq. 3, Delta update
                [lb, ub] = bro_shrink(X, B.q, lo, hi);
                Delta = Delta + round(Delta / 2);
                shrinks(end+1) = t; %#ok<AGROW>
            end
            iters = t;
            hist(t) = B.f;
        end
        hist  = hist(1:iters);
        nmove = nmove(1:iters);
        fin = struct('X', X, 'F', F, 'damage', D, 'lb', lb, 'ub', ub);
    end

    er = NaN;
    if ~isnan(B.f)
        er = acos(min(1, max(-1, (trace(T_d(1:3, 1:3).' * B.T(1:3, 1:3)) - 1) / 2)));
    end

    q_best = B.q;
    info = struct();
    info.solver    = 'BRO';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = false;                                  % the paper stops on T only
    info.fitness   = B.f;                                    % m (Sec. 4)
    info.err_pos   = B.f;                                    % m
    info.err_rot   = er;                                     % rad, reporting only
    info.fk_calls  = B.nfk;
    info.jac_calls = 0;
    info.N         = N;
    info.T         = T;
    info.history   = hist;
    info.n_move    = nmove;                                  % Eq. 1 moves per iteration
    info.n_respawn = N * ones(1, iters) - nmove;             % Eq. 2 respawns per iteration
    info.shrinks   = shrinks;                                % iterations after which Eq. 3 acted
    info.final     = fin;
    info.params    = P;
end

function [X, F, D, B, nm] = bro_sweep(X, F, D, B, lb, ub, lo, hi, fit, P)
% One iteration of Alg. 1 / Fig. 2: soldier i = 1..N duels its nearest soldier j; the damaged
% soldier moves by Eq. 1 (damage < Threshold) or respawns by Eq. 2, and is re-evaluated.
    N  = size(X, 2);
    nm = 0;
    for i = 1:N
        j = bro_nearest(X, i);
        [dam, vic] = bro_duel(F, i, j);
        if D(dam) < P.threshold
            u = bro_move(X(:, dam), B.q);                    % Eq. 1, toward x_best
            D(dam) = D(dam) + 1;
            nm = nm + 1;
        else
            u = bro_respawn(lb, ub);                         % Eq. 2
            D(dam) = 0;
        end
        D(vic) = 0;
        X(:, dam) = min(max(u, lo), hi);                     % floating-point guard only
        [F(dam), B] = eval_keep(fit, X(:, dam), B);          % update f(x_dam) (Fig. 2)
    end
end

% ==================================================================== operators
function P = bro_params()
% Table 10 (p.18), first row: N = 25, T = 100; damage threshold 3 (p.4).
    P = struct('N', 25, 'T', 100, 'threshold', 3);
end

function j = bro_nearest(X, i)
% Nearest soldier to soldier i, Euclidean distance in joint space (p.4); the first index on ties.
    d = sum((X - X(:, i)).^2, 1);
    d(i) = Inf;
    [~, j] = min(d);
end

function [dam, vic] = bro_duel(F, i, j)
% Alg. 1 mirrored for a cost: j is damaged unless f(x_i) > f(x_j); on equal cost i is the victor.
    dam = j;
    vic = i;
    if F(i) > F(j)
        dam = i;
        vic = j;
    end
end

function u = bro_move(x, xb)
% Eq. 1: x_dam,d + r(x_best,d - x_dam,d), r ~ U(0, 1) one draw per dimension.
    u = x + rand(numel(x), 1) .* (xb - x);
end

function u = bro_respawn(lb, ub)
% Eq. 2: r(ub_d - lb_d) + lb_d, r ~ U(0, 1) one draw per dimension, current bounds.
    u = rand(numel(lb), 1) .* (ub - lb) + lb;
end

function [lb, ub] = bro_shrink(X, xb, lo, hi)
% Eq. 3: x_best,d -/+ SD(x_d) of the population, clipped to the original bounds (p.4, Alg. 1).
    sd = std(X, 0, 2);
    lb = max(xb - sd, lo);
    ub = min(xb + sd, hi);
end

function D = bro_delta0(T)
% Alg. 1: shrink = ceil(log10(MaxCicle)), Delta = round(MaxCicle/shrink); Inf for T = 1.
    D = round(T / ceil(log10(T)));
end

function s = bro_schedule(T)
% Iterations t = 1..T after which Eq. 3 acts (t >= Delta, then Delta = Delta + round(Delta/2)).
    s = zeros(1, 0);
    D = bro_delta0(T);
    for t = 1:T
        if t >= D
            s(end+1) = t; %#ok<AGROW>
            D = D + round(D / 2);
        end
    end
end

% ==================================================================== fitness (Sec. 4)
function [f, T] = bro_fitness(robot, T_d, q)
% p.16: d = sqrt((x' - x)^2 + (y' - y)^2 + (z' - z)^2), obtained vs target position, in metres.
    T = robot.fk(q);
    e = T(1:3, 4) - T_d(1:3, 4);
    f = sqrt(e(1)^2 + e(2)^2 + e(3)^2);
end

function [f, B] = eval_keep(fit, q, B)
% One counted FK evaluation; B keeps x_best (the lowest cost evaluated), its pose and the count.
    [f, T] = fit(q);
    B.nfk = B.nfk + 1;
    if isnan(B.f) || f < B.f
        B.f = f;
        B.q = q;
        B.T = T;
    end
end
