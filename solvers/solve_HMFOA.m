function [q_best, info] = solve_HMFOA(robot, T_d, q0, opts)
% SOLVE_HMFOA  Hybrid mutation fruit fly optimization algorithm (HMFOA) IK solver of Shi et al. (2020), Secs. 3 and 4.3.
%
%   Source:
%     Jianping Shi, Yuting Mao, Peishen Li, Guoping Liu, Peng Liu, Xianyong Yang and Dahai Wang (2020). Hybrid
%     Mutation Fruit Fly Optimization Algorithm for Solving the Inverse Kinematics of a Redundant Robot
%     Manipulator. Mathematical Problems in Engineering, Volume 2020, Article ID 6315675, 13 pages.
%     DOI: 10.1155/2020/6315675
%   No issue number is printed. Page numbers are PDF pages 1-13 (equal to the printed page numbers).
%
%   Algorithm (Secs. 3.1-3.4, pp.4-6; flow chart Fig. 4, p.6; IK fitness Sec. 4.3, p.7):
%     m flies; the position of fly i is its joint vector, X_i = [theta_i1 .. theta_iD] (S_i = X_i, p.4; p.7).
%     Pb_i is the best previous position of fly i, Pb_g the best previous position of the swarm and g its fly.
%     Step 1 (p.6): random initial positions in the search space, Pb_i = X_i, Pb_g = the best of them, t = 1.
%     Steps 2-5, for each fly i = 1..m in turn. Osphresis search (Eqs. 15-18, p.4):
%       X_i   = X'_i                          if rand1 <  1 - omega
%               X''_i                         if rand1 >= 1 - omega                            (Eq. 15)
%       X'_i  = Pb_g + omega (Pb_r1 - Pb_r2)          if rand2 <  0.5
%               Pb_g rand3 + omega (Pb_r1 - Pb_r2)    if rand2 >= 0.5                          (Eq. 16)
%       X''_i = Pb_i + omega (Pb_r1 - Pb_r2)          if rand4 <  0.5
%               L + rand5 (U - L)                     if rand4 >= 0.5                          (Eq. 17)
%       omega = (1 + cos(pi (t - 1) / t_max)) / 2                                              (Eq. 18)
%     r1 and r2 are "random and mutually different integers generated in the range [1, m]" and "different from
%     the fly's index i and g" (pp.4-5); L and U are the lower and upper boundary vectors of the search space
%     (p.5). One of the four mutation strategies is chosen per fly (p.5: "randomly select one of the four mutation
%     strategies according to probability as the current olfactory searching operator of the ith fly").
%     Cross-border processing (Eq. 21, Sec. 3.3, p.6), for every component j of X_i:
%       x_ij = u_j - min(x_ij - u_j, u_j - l_j) rand6   if x_ij > u_j
%       x_ij = l_j + min(l_j - x_ij, u_j - l_j) rand6   if x_ij < l_j
%     Then the fitness of X_i is computed and the vision search (Sec. 3.2, p.6) updates the search centres at
%     once ("the new position of the fly is evaluated immediately"):
%       Pb_g = X_i   if fitness(X_i) <= fitness(Pb_g)                                          (Eq. 19)
%       Pb_i = X_i   if fitness(X_i) <= fitness(Pb_i)                                          (Eq. 20)
%     so the next fly uses the updated Pb_g and Pb_i. Step 6 (p.6): after fly m, t = t + 1; the run ends at
%     t = t_max and outputs Pb_g (Fig. 4, "Output the best fitness").
%     IK fitness (Sec. 4.3, p.7), with n, o, a, p the columns of the pose (Eq. 2, p.2) and * the target:
%       f'  = |n*_x - n_x| + |n*_y - n_y| + |n*_z - n_z| + |o*_x - o_x| + |o*_y - o_y| + |o*_z - o_z|
%             + |a*_x - a_x| + |a*_y - a_y| + |a*_z - a_z|                                     (Eq. 22)
%       f'' = |p*_x - p_x| + |p*_y - p_y| + |p*_z - p_z|                                       (Eq. 23)
%       f   = min(f' + lambda f'')                                                             (Eq. 24)
%     where min(.) "is the minimum function" (p.7): f' + lambda f'' is minimised over the joint vectors.
%
%   Parameters:
%     m = 100, t_max = 1000   the IK setting: Sec. 4.3 (p.7) sets the population size of all algorithms to 100
%                     for IK, "and the rest of the parameters are the same as mentioned above", i.e. Sec. 4.1
%                     (p.7): "the maximum number of iterations is 1000". m x t_max = 100 x 1000 = 100,000, the
%                     only IK setting reported. The Sec. 4.1 setting 60 x 1000 (p.7) is the benchmark-function
%                     setting, replaced for IK by the paper itself. The dimension D (7 on p.7) is robot.n.
%                     Inside the benchmark t_max is the budget horizon (Benchmark interface).
%     lambda = 1e-2   weight factor of Eq. 24 (Sec. 4.3, p.7: "The weight factor lambda is set as 10^-2").
%     0.5             thresholds of rand2 (Eq. 16) and rand4 (Eq. 17), p.4.
%     rand1-rand4, rand6   "random variables uniformly distributed on the interval [0, 1]" (p.5; rand6 also
%                     p.6). rand1, and rand2 or rand4, are drawn once per fly and iteration (Eq. 15 and the
%                     strategy choice per fly, p.5). rand6: one draw per processed component (Eq. 21 is written
%                     per component).
%     rand3, rand5    one U(0, 1) draw per joint and use: Pb_g x rand3 and rand5 x (U - L) are products per joint,
%                     so L + rand5 (U - L) is a uniform point of the joint box. Basis: p.5 defines rand3 as a
%                     random variable uniformly distributed on [0, 1] without saying whether it is one value or
%                     one per component, and rand5 is not defined in the paper (absent from the p.5 list; checked
%                     pages 4-6). The other reading, also consistent with p.5, is one scalar per use (Pb_g scaled
%                     as a whole; restarts on the diagonal of the joint box). The per-joint reading is used
%                     because, with the warm start of every waypoint, it reaches the lower position error within
%                     the 2000-FK budget (median 4.6 mm against 25.1 mm over 120 benchmark waypoints). On the
%                     paper's own task (Table 1 arm, Eq. 25 target, m = 100, t_max = 1000, 50 runs) the per-joint
%                     reading gives SR = 84 % and the scalar reading SR = 94 %; Table 4 (p.10) reports 94 %.
%     r1, r2          an ordered pair of distinct indices drawn uniformly from {1..m} without i and g, one pair
%                     per fly whose strategy uses Pb_r1 - Pb_r2 (strategies 1-3).
%     g               the fly whose best position is Pb_g: the best initial fly (ties to the lowest index),
%                     then g = i whenever Eq. 19 accepts X_i.
%     L, U, l_j, u_j  "the upper and lower boundary vectors of the search space" (p.5) and "the lower and upper
%                     bounds for x_ij" (p.6): q_min and q_max (Table 1, p.5, lists the joint limits l_i, u_i as
%                     the bounds of the paper's arm).
%     Initial population   "randomly initialize the locations of all fruit flies in the search space" (Step 1,
%                     p.6); the distribution is not stated in the paper (checked pages 4-7); uniform in
%                     [q_min, q_max] per joint.
%     Units           Table 1 (p.5) gives d_i and a_i in mm, the target of Eq. 25 (p.7) has p_z = 243.149, and the
%                     best values of Table 4 (p.10) are Eq. 24 of the Table 6 errors (p.12) with f'' in mm (to
%                     10 digits for LGMS-FOA, AE-LGMS-FOA and HMFOA). So f'' is computed in mm (1000 x the
%                     distance in m) and lambda = 1e-2 weights millimetres; f' is unitless; angles in rad
%                     (Table 5, p.11).
%     Stop            t = t_max only (Step 6, p.6; Fig. 4, "t > t_max"); the paper has no accuracy stop.
%     Convergence     the success ratio SR (Eq. 26, p.8) counts runs that "converge with a specified accuracy (in
%                     this study it is assigned a value of 10^-9)"; info.converged = (f of Pb_g <= 1e-9), f in
%                     the paper's unit, reported only.
%     Random draws    per iteration, flies in index order: rand1; rand2 or rand4; r1 and r2 (randperm over the
%                     candidate list in ascending order) if the strategy uses them; rand3 or rand5 (one per
%                     joint, in joint order) if the strategy uses it; then one rand6 per out-of-range component,
%                     in joint order.
%
%   Benchmark interface:
%     Budget: the initial population costs m = 100 FK calls and every iteration m = 100 (every fly moves and is
%       evaluated once, Step 2). t_max = ikb_generations(opts, 100, 100, 1000) = 19 iterations, so
%       100 + 19 x 100 = 2000 FK per waypoint. A 40-member population would give
%       ikb_generations(opts, 40, 40, 1000) = 49. The initial population and every iteration start only if
%       ikb_budget_left(opts) covers m. Outside the benchmark t_max = 1000 (100,100 FK). No final FK check.
%       Every option field is ignored (the budget is read only through ikb_generations and ikb_budget_left).
%     Schedule: Eq. 18 runs over this t_max: with t_max = 19, omega = 1 at t = 1 (Eq. 17 only), 0.5413 at
%       t = 10, 0.4587 at t = 11 and 0.0068 at t = 19 (omega reaches 0 only at t = t_max + 1).
%     Stop: t_max iterations or the budget; no tolerance stop.
%     Warm start: q0 (clamped to the limits) is fly 1 of the initial population and is evaluated; it is Pb_1,
%       and Pb_g unless another initial fly is strictly better. Pb_1 = q0 is kept until fly 1 reaches a
%       fitness <= f(q0) (Eq. 20); q0 is returned unless a later fly reaches a fitness <= that of Pb_g (Eq. 19).
%     Robot and units: Eqs. 1-14 and Table 1 (the paper's YuMi left arm) are replaced by robot.fk with
%       D = robot.n joints; lengths in m, converted to mm inside Eq. 23; angles in rad.
%     Joint limits: Eq. 21 after every move, as the paper says; its result lies inside [q_min, q_max], so no
%       clamping is applied. info.border counts the processed components.
%     Output: Pb_g; info.fitness (f, Eq. 24), info.f1 (f', Eq. 22), info.f2 (f'' in mm, Eq. 23), info.err_pos
%       (m) and info.err_rot (rotation angle, rad, reported only), all from the pose stored at its evaluation.
%       solve_HMFOA('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration horizon t_max = 19 instead of 1000 (benchmark rule); Eq. 18 runs over it.
%     - q0 is fly 1 of the initial population (warm start).
%     - Not stated in the paper, chosen as given under Parameters: the initial distribution, the size of the
%       draw of rand3, the definition of rand5, the order of the draws, ties among the initial flies.
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
        q_best = struct('params', hmfoa_params(), 'core', @hmfoa_core, 'omega', @hmfoa_omega, ...
                        'pick', @hmfoa_pick, 'strategy', @hmfoa_strategy, 'osphresis', @hmfoa_osphresis, ...
                        'border', @hmfoa_border, 'vision', @hmfoa_vision, 'fitness', @hmfoa_fitness, ...
                        'eq24', @hmfoa_eq24);
        info = [];
        return
    end
    [q_best, info] = hmfoa_core(robot, T_d, q0, opts, hmfoa_params());
end

% ==================================================================== main loop (Fig. 4, Steps 1-6)
function [q_best, info] = hmfoa_core(robot, T_d, q0, opts, P)
% Step 1 (q0 first, all evaluated); per iteration and fly: Eqs. 15-18, Eq. 21, fitness (Eq. 24), Eqs. 19-20 at
% once; stop at t_max or at the budget.
    t_start = tic;
    m  = P.m;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    n  = numel(lo);
    q0 = min(max(q0(:), lo), hi);
    T  = ikb_generations(opts, m, m, P.tmax);

    X = zeros(n, 0);  Pb = zeros(n, 0);  fPb = zeros(1, 0);  g = 0;
    B = struct('f', NaN, 'f1', NaN, 'f2', NaN, 'T', NaN(4));
    fk = 0;  iters = 0;  f_init = NaN;  g_init = 0;
    hist = zeros(1, 0);  om = zeros(1, 0);  strat = zeros(4, 0);  nb = zeros(1, 0);
    acc = zeros(2, 0);  gh = zeros(1, 0);

    if ikb_budget_left(opts) >= m
        X = [q0, lo + rand(n, m - 1) .* (hi - lo)];                     % Step 1, q0 first
        fPb = zeros(1, m);
        for i = 1:m
            [fPb(i), Tq, f1, f2] = hmfoa_fitness(robot, T_d, X(:, i), P);
            fk = fk + 1;
            if i == 1 || fPb(i) < B.f                                   % best initial fly, ties to the lowest index
                g = i;
                B = struct('f', fPb(i), 'f1', f1, 'f2', f2, 'T', Tq);
            end
        end
        Pb = X;
        f_init = B.f;  g_init = g;
        hist = NaN(1, T);  om = zeros(1, T);  strat = zeros(4, T);  nb = zeros(1, T);
        acc = zeros(2, T);  gh = zeros(1, T);
        for t = 1:T
            if ikb_budget_left(opts) < m, break; end
            w = hmfoa_omega(t, T);                                      % Eq. 18
            for i = 1:m
                [x, s] = hmfoa_osphresis(Pb, i, g, w, lo, hi);          % Eqs. 15-17
                [x, c] = hmfoa_border(x, lo, hi);                       % Eq. 21
                X(:, i) = x;
                [f, Tq, f1, f2] = hmfoa_fitness(robot, T_d, x, P);
                fk = fk + 1;
                [ug, ui] = hmfoa_vision(f, B.f, fPb(i));                % Eqs. 19-20
                if ug
                    g = i;
                    B = struct('f', f, 'f1', f1, 'f2', f2, 'T', Tq);
                end
                if ui
                    Pb(:, i) = x;
                    fPb(i) = f;
                end
                strat(s, t) = strat(s, t) + 1;
                nb(t) = nb(t) + c;
                acc(:, t) = acc(:, t) + [ug; ui];
            end
            om(t) = w;
            iters = t;
            hist(t) = B.f;
            gh(t) = g;
        end
        hist = hist(1:iters);  om = om(1:iters);  strat = strat(:, 1:iters);  nb = nb(1:iters);
        acc = acc(:, 1:iters);  gh = gh(1:iters);
    end

    q_best = q0;  ep = NaN;  er = NaN;
    if g > 0
        q_best = Pb(:, g);                                              % Pb_g
        ep = norm(B.T(1:3, 4) - T_d(1:3, 4));
        er = acos(min(1, max(-1, (trace(T_d(1:3, 1:3).' * B.T(1:3, 1:3)) - 1) / 2)));
    end

    info = struct();
    info.solver    = 'HMFOA';
    info.iters     = iters;                                             % iterations completed
    info.time      = toc(t_start);
    info.converged = ~isnan(B.f) && B.f <= P.sr_acc;                    % SR accuracy (Eq. 26, p.8), reported only
    info.fitness   = B.f;                                               % f, Eq. 24 (f'' in mm)
    info.fitness_init = f_init;                                         % best f of the initial population
    info.f1        = B.f1;                                              % f', Eq. 22
    info.f2        = B.f2;                                              % f'', Eq. 23, mm
    info.err_pos   = ep;                                                % m
    info.err_rot   = er;                                                % rad, reported only
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.m         = m;
    info.T         = T;
    info.g_init    = g_init;                                            % fly of Pb_g after Step 1
    info.history   = hist;                                              % f of Pb_g after iteration t
    info.omega     = om;                                                % Eq. 18 in iteration t
    info.strategy  = strat;                                             % flies per strategy (Eq. 16 a/b, Eq. 17 a/b)
    info.border    = nb;                                                % components processed by Eq. 21
    info.accept    = acc;                                               % acceptances by Eq. 19 / Eq. 20
    info.g         = gh;                                                % fly of Pb_g after iteration t
    info.final     = struct('X', X, 'Pb', Pb, 'fPb', fPb, 'g', g);
    info.params    = P;
end

% ==================================================================== parameters and operators
function P = hmfoa_params()
% Sec. 4.3 (p.7): m = 100, lambda = 1e-2; Sec. 4.1 (p.7): t_max = 1000; Eq. 26 (p.8): SR accuracy 1e-9;
% f'' in mm (len_scale = 1000 per m; Table 1, p.5).
    P = struct('m', 100, 'tmax', 1000, 'lambda', 1e-2, 'sr_acc', 1e-9, 'len_scale', 1000);
end

function w = hmfoa_omega(t, T)
% Eq. 18: omega = (1 + cos(pi (t - 1) / t_max)) / 2.
    w = (1 + cos(pi * (t - 1) / T)) / 2;
end

function [x, s, r, c] = hmfoa_osphresis(Pb, i, g, w, lo, hi)
% Eqs. 15-17 for fly i: rand1 < 1 - omega selects Eq. 16, otherwise Eq. 17 (Eq. 15); rand2 (Eq. 16) or rand4
% (Eq. 17) < 0.5 selects the first branch. One strategy s per fly: 1, 2 = Eq. 16 first, second branch; 3, 4 =
% Eq. 17 first, second branch. r = [r1 r2] (zeros for s = 4); c = rand3 (s = 2) or rand5 (s = 4), one U(0, 1)
% per joint, NaN otherwise.
    if rand < 1 - w                                                     % Eq. 15, rand1
        if rand < 0.5, s = 1; else, s = 2; end                          % Eq. 16, rand2
    else
        if rand < 0.5, s = 3; else, s = 4; end                          % Eq. 17, rand4
    end
    r = [0 0];
    if s ~= 4, r = hmfoa_pick(size(Pb, 2), i, g); end
    c = NaN;
    if s == 2 || s == 4, c = rand(size(Pb, 1), 1); end                  % rand3 or rand5, per joint
    x = hmfoa_strategy(s, Pb, i, g, r, w, c, lo, hi);
end

function r = hmfoa_pick(m, i, g)
% r1 ~= r2, both different from i and g (pp.4-5): an ordered pair drawn uniformly from {1..m} without i and g.
    cand = 1:m;
    cand(cand == i | cand == g) = [];
    r = cand(randperm(numel(cand), 2));
end

function x = hmfoa_strategy(s, Pb, i, g, r, w, c, lo, hi)
% The four mutation strategies of Eqs. 16-17 as vector equations; Pb_g = Pb(:, g), c = rand3 or rand5 (one value
% per joint, products per joint).
    switch s
        case 1, x = Pb(:, g) + w * (Pb(:, r(1)) - Pb(:, r(2)));         % Eq. 16, rand2 <  0.5
        case 2, x = Pb(:, g) .* c + w * (Pb(:, r(1)) - Pb(:, r(2)));    % Eq. 16, rand2 >= 0.5
        case 3, x = Pb(:, i) + w * (Pb(:, r(1)) - Pb(:, r(2)));         % Eq. 17, rand4 <  0.5
        otherwise, x = lo + c .* (hi - lo);                             % Eq. 17, rand4 >= 0.5
    end
end

function [x, c] = hmfoa_border(x, lo, hi)
% Eq. 21 per component, in joint order: x_j > u_j -> u_j - min(x_j - u_j, u_j - l_j) rand6;
% x_j < l_j -> l_j + min(l_j - x_j, u_j - l_j) rand6; one rand6 per processed component; c = their number.
    c = 0;
    for j = 1:numel(x)
        if x(j) > hi(j)
            x(j) = hi(j) - min(x(j) - hi(j), hi(j) - lo(j)) * rand;
            c = c + 1;
        elseif x(j) < lo(j)
            x(j) = lo(j) + min(lo(j) - x(j), hi(j) - lo(j)) * rand;
            c = c + 1;
        end
    end
end

function [ug, ui] = hmfoa_vision(f, fg, fi)
% Eq. 19: Pb_g = X_i if fitness(X_i) <= fitness(Pb_g); Eq. 20: Pb_i = X_i if fitness(X_i) <= fitness(Pb_i).
    ug = f <= fg;
    ui = f <= fi;
end

% ==================================================================== IK fitness (Eqs. 22-24)
function [f, T, f1, f2] = hmfoa_fitness(robot, T_d, q, P)
% Eq. 24 of the pose robot.fk(q) against T_d; one FK call.
    T = robot.fk(q);
    [f, f1, f2] = hmfoa_eq24(T, T_d, P);
end

function [f, f1, f2] = hmfoa_eq24(T, T_d, P)
% Eq. 22: f' = sum of |n* - n|, |o* - o|, |a* - a| over x, y, z (columns 1-3, in the printed order);
% Eq. 23: f'' = |p*_x - p_x| + |p*_y - p_y| + |p*_z - p_z| in mm; Eq. 24: f = f' + lambda f''.
    f1 = 0;
    for col = 1:3
        for row = 1:3
            f1 = f1 + abs(T_d(row, col) - T(row, col));
        end
    end
    f2 = 0;
    for row = 1:3
        f2 = f2 + abs(T_d(row, 4) - T(row, 4));
    end
    f2 = P.len_scale * f2;
    f  = f1 + P.lambda * f2;
end
