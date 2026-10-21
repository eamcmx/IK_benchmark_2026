function [q_best, info] = solve_IPSO(robot, T_d, q0, opts)
% SOLVE_IPSO  Improved particle swarm optimizer (IPSO) IK solver of Li et al. (2026), pp.3-5 and p.10.
%
%   Source:
%     Zhentao Li, Panfeng Wang, Yun Feng, Guanjun Zhang, Zhiyang Guo, Minghui Zhang and Yi Zhang (2026).
%     Optimization of manipulator inverse kinematics using improved particle swarm algorithm for enhanced
%     manufacturing efficiency. Proc IMechE Part B: J Engineering Manufacture, 2026, Vol. 240(1-2) 188-202.
%     DOI: 10.1177/09544054241310337
%   Page 1 also prints "(c) IMechE 2025". Page numbers below are PDF pages 1-15 (printed page = PDF page + 187).
%
%   Algorithm (pp.3-5: Eqs. 1-6, Steps 1-4, Fig. 2, Table 1; IK fitness p.10, Eq. 9, Euler angles p.9):
%     N particles; the position X_i of particle i is its joint vector (p.10). Initial positions
%     X_i = x_min + rand (x_max - x_min), zero initial velocities, pbest_i = X_i (Table 1, p.5); gbest is the
%     best position of the swarm, "refined considering the historical positions of all particles" (p.3).
%     Each iteration t = 1..t_max (Table 1 and Fig. 2, p.5):
%       c1 = c1,max + (c1,min - c1,max) / t_max * t                                               (Eq. 4)
%       c2 = c2,min + (c2,max - c2,min) / t_max * t                                               (Eq. 5)
%       f_max,t and f_min,t = the largest and smallest fitness of the current positions (fitbox, Table 1).
%       For each particle i = 1..N in turn (Table 1):
%         w = w_min + (w_max - w_min) * (f_i,t - f_min,t) / (f_max,t - f_min,t)                  (Eq. 3)
%             if f_max,t != f_min,t, otherwise w = w_max (Table 1);
%         V_i^j(t + 1) = w V_i^j(t) + c1 r1 (pbest_i^j - X_i^j(t)) + c2 r2 (gbest^j - X_i^j(t))     (Eq. 1)
%         X_i^j(t + 1) = X_i^j(t) + V_i^j(t + 1)                                                    (Eq. 2)
%         then the boundary check of position and speed, the fitness of X_i, and the update of pbest_i and
%         gbest (Table 1), so the next particle moves with the updated gbest.
%       Step 1 (p.4): "Half of the particles in the population will be chosen as the parent particles" by the
%         roulette, particles "whose fitness values are close to that of the global best particle" preferred.
%       Step 2 (p.4): each parent produces three offspring, each "by randomly altering a specific value in the
%         position of the corresponding parent particle" (one location j, with bounds x_min, x_max):
%           X^child = X^parent - k (X^parent - x_min)   if r >= 0.5
%                     X^parent + k (x_max - X^parent)   if r < 0.5,   k = r (1 - t / t_max)^2     (Eq. 6)
%       Step 3 (p.5): the offspring with the best fitness replaces the parent if it is better, Table 1:
%         "if (f(X^childbest) < f(X^parent)) X^parent = X^childbest"; otherwise all offspring are discarded.
%       Step 4 (p.5): "The particles prior to mutation are sorted based on their fitness values, and the top 50%
%         of these particles are selected to combine with the offspring particles generated through mutation":
%         the next population is the best N/2 particles prior to mutation plus the N/2 parents after Step 3.
%       pbest and gbest are then updated with the new population (Table 1, first line inside "while"; Fig. 2,
%         "Get and updata pbest", "Get and update gbest").
%     IK fitness (Eq. 9, p.10), with P = [p_x, p_y, p_z, alpha, beta, gamma] of the pose and P_0 of the target:
%       f_error = [(p_x - p_x0)^2 + (p_y - p_y0)^2 + (p_z - p_z0)^2
%                  + m ((alpha - alpha_0)^2 + (beta - beta_0)^2 + (gamma - gamma_0)^2)]^(1/2)       (Eq. 9)
%     with the Euler angles of p.9 ("in the xyz coordinate system"), n, o, a the columns of the rotation (Eq. 8):
%       alpha = arctan(-a_y / a_z), beta = arctan(a_x / sqrt(n_x^2 + o_x^2)), gamma = arctan(-o_x / n_x).
%
%   Parameters:
%     N = 200, t_max = 500   the IK setting, p.11: "The population size and the maximum number of iterations for
%                     all algorithms are set to 200 and 500, respectively" (both IK scenarios). N x t_max =
%                     200 x 500 = 100,000, the only IK setting reported; the 100 x 1000 setting of p.6 is the
%                     benchmark-function setting. Inside the benchmark t_max is the budget horizon (Benchmark
%                     interface).
%     w_max = 0.729, w_min = 0.6   Table 1 (p.5, "wmin =0.6; wmax= 0.729"; "w = wmax (winitial = 0.729)") and
%                     Table 3 (p.6, "w = 0.729->0.6"). The p.4 text gives w_max, w_min as "0.6 and 0.729
%                     respectively", which would make the maximum smaller than the minimum; Table 1 is used.
%     c_i,max = 2, c_i,min = 1 (i = 1, 2)   p.4 and Table 1: c1 falls from 2 to 1, c2 rises from 1 to 2.
%     N/2 = 100 parents   Step 1 (p.4, "a mutation rate of 50%"); Table 3 (p.6, "Mutation ratio: 0.5").
%     3 offspring per parent   Step 2 (p.4); Table 3 (p.6, "The number of offspring:3").
%     m = 1000        Eq. 9 (p.10: "pose error is amplified by a constant m (m = 1000)").
%     r1, r2          U(0, 1) (p.3); one draw per particle and joint, as Eq. 1 is written per component j.
%     r, k (Eq. 6)    r ~ U(0, 1) (p.5); the printed equation uses the same symbol r in the branch test and in
%                     k = r (1 - t / t_max)^2, so one draw per offspring serves both.
%     Second branch of Eq. 6   printed "r0.5" (p.4): the comparison sign is absent from the page and from its
%                     text layer; r < 0.5, the complement of the first branch "r >= 0.5", is the nearest
%                     well-defined reading.
%     Location j (Eq. 6)   "a specific value" altered "randomly" (p.4); the choice of j is not stated in the paper
%                     (checked pages 4-5); j is uniform over the joints, one draw per offspring.
%     Roulette weights   not stated in the paper (checked pages 4-5); weight 1/f_i (the smaller the fitness, the
%                     closer to gbest, the larger the share), N/2 draws without replacement, so the parents are
%                     N/2 distinct particles. A particle with f_i = 0 has an infinite weight: such particles are
%                     drawn first, uniformly among themselves.
%     Speed limit     Table 1 checks "whether the position and speed of particles exceed the boundary limit"; the
%                     speed limit is not stated in the paper (checked pages 3-6 and 10-11). |V^j| is bounded by
%                     x_max^j - x_min^j, the width of the search range, the only bound the paper defines. A
%                     position outside [x_min, x_max] is clamped (handling not stated in the paper, checked pages
%                     5 and 10). Both after Eqs. 1-2, in the Table 1 order.
%     f_max,t, f_min,t   "the maximum and minimum fitness at the iteration t" (p.4): from the fitness of the
%                     positions at the start of iteration t (Table 1, fitbox) and used for all N particles of
%                     that iteration; Table 1's "Update the fmax and fmin" after each move is read as the refresh
%                     for the next iteration.
%     Mutation timing once per iteration, after all N particles have moved: Fig. 2 and Step 1 ("During each
%                     iteration"); Table 1 prints the selection, mutation and merge lines inside the particle loop.
%     Substituted particle   its velocity is not stated in the paper (checked pages 4-5); Table 1 replaces only its
%                     position ("X^parent = X^childbest"), so it keeps its velocity and pbest. In the new
%                     population the best N/2 particles prior to mutation come
%                     first (ascending fitness, ties to the lower index), then the N/2 parents in selection
%                     order. A parent that is also among the best N/2 appears twice (the paper does not exclude it).
%     Ties            strict "<" in the pbest update (Table 1), the gbest update and Step 3 (Table 1); the best
%                     offspring is the first of equal ones; the initial gbest is the first of equal ones.
%     Evaluations     every position produced by Eqs. 1-2 or Eq. 6 is evaluated once; the fitness of a stored
%                     position (Table 1's fitbox, Fig. 2's "Calculate and assess fitness" of the new population)
%                     is reused, not recomputed.
%     Arctan          the four-quadrant arctan: alpha = atan2(-a_y, a_z), beta = atan2(a_x, sqrt(n_x^2 + o_x^2)),
%                     gamma = atan2(-o_x, n_x). The IK target of p.11 has gamma = 2.606 > pi/2, outside the
%                     range of the one-argument arctan; its joint angles in Fig. 6(b) give gamma = 2.61 with
%                     atan2 on the Table 6 arm. Eq. 9 takes the plain differences of the angles (no wrapping),
%                     so two poses on opposite sides of the -pi/pi cut of alpha or gamma differ by about 2 pi.
%     Units           mm and rad. Table 6 (p.10) gives the link lengths in mm; the IK target of p.11, P =
%                     [-436.742, -637.332, 856.188, 0.0177, 0.842, 2.606], appears in Fig. 6(b) as x = -0.437,
%                     y = -0.637, z = 0.856 on axes in m; the targets of p.12 are P1 = [700, 200, 300, pi, 0, 0]
%                     and P2 = [500, 100, 500, pi, 0, 0], which the Table 6 arm reaches at the Table 8 angles in
%                     mm. So the position terms of Eq. 9 are computed in mm (1000 x the difference in m).
%     Stop            Table 1 runs "while the stop criterion is not met" with t up to the maximum number of
%                     iterations; the criterion is not stated in the paper (checked pages 4-5 and 11-12); p.12
%                     times the runs "to obtain the solution satisfying the accuracy" (Fig. 7(c), Table 7)
%                     without its value. This example-task accuracy is the benchmark pose gate: gbest's position
%                     error < opts.tol_pos (m) and rotation error < opts.tol_rot (rad) (pose_error), tested
%                     after the initial population and after every iteration; info.converged = the gate.
%     Random draws    initial positions of particles 2..N (rand(n, N - 1)); per iteration: for each particle in
%                     index order, r1 for all joints, then r2 for all joints; one rand per roulette draw; then per
%                     parent in selection order and per offspring, j (randi(n)) then r (rand).
%
%   Benchmark interface:
%     Budget: the initial population costs N = 200 FK calls and every iteration N moves + 3 x N/2 offspring =
%       500. t_max = ikb_generations(opts, 200, 500, 500) = 3 iterations, so 200 + 3 x 500 = 1700 FK per
%       waypoint; the remaining 300 do not cover an iteration. A 40-member population would give
%       ikb_generations(opts, 40, 100, 500) = 19. The initial population and every iteration start only if
%       ikb_budget_left(opts) covers them. Outside the benchmark t_max = 500 (250,200 FK). No final FK check.
%       Fields of opts other than tol_pos and tol_rot (max_iter, pop_size, ...) are ignored.
%     Schedules: Eqs. 4-6 run over this t_max. With t_max = 3: c1 = 1.667, 1.333, 1; c2 = 1.333, 1.667, 2;
%       (1 - t / t_max)^2 = 0.444, 0.111, 0. In the last iteration k = 0, so every offspring equals its parent
%       and Step 3 keeps every parent.
%     Stop: the pose gate above, or t_max iterations, or the budget.
%     Warm start: q0 (clamped to the limits) is particle 1 of the initial population and is evaluated; it enters
%       the first gbest selection (ties to the lowest index).
%     Robot and units: Eqs. 7-8 and Table 6 (the paper's arm) are replaced by robot.fk with D = robot.n joints;
%       lengths in m, converted to mm inside Eq. 9; angles in rad.
%     Joint limits: [x_min, x_max] = [robot.q_min, robot.q_max]; positions clamped, speeds bounded (above).
%     Output: gbest; info.fitness (Eq. 9, mm), info.err_pos (m) and info.err_rot (rad, pose_error) from the pose
%       stored at its evaluation. solve_IPSO('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration horizon t_max = 3 instead of 500 (benchmark rule); Eqs. 4-6 run over it.
%     - q0 is particle 1 of the initial population (warm start).
%     - Stop at the benchmark pose gate (the paper's accuracy is not stated).
%     - Not stated in the paper, chosen as given under Parameters: the location j, the roulette weights and
%       the f = 0 rule, the speed limit, the position clamp, the order of the merged population, the order of
%       the draws.
%     - Readings of printed inconsistencies, as given under Parameters: the second branch of Eq. 6, the w_max /
%       w_min values, the mutation once per iteration, the four-quadrant arctan.
%     - robot.fk instead of Eqs. 7-8 (the paper's arm).
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
        q_best = struct('params', ipso_params(), 'core', @ipso_core, 'factors', @ipso_factors, ...
                        'inertia', @ipso_inertia, 'move', @ipso_move, 'select', @ipso_select, ...
                        'child', @ipso_child, 'offspring', @ipso_offspring, 'merge', @ipso_merge, ...
                        'fitness', @ipso_fitness, 'eq9', @ipso_eq9, 'euler', @ipso_euler, 'gate', @ipso_gate);
        info = [];
        return
    end
    [q_best, info] = ipso_core(robot, T_d, q0, opts, ipso_params());
end

% ==================================================================== main loop (Table 1, Fig. 2)
function [q_best, info] = ipso_core(robot, T_d, q0, opts, P)
% Initial population (q0 first, all evaluated, zero velocity); per iteration Eqs. 4-5, Eq. 3, Eqs. 1-2 with the
% boundary check, pbest and gbest after each particle, Steps 1-4, pbest and gbest of the new population; stop at
% the pose gate, at t_max or at the budget.
    t_start = tic;
    N   = P.N;
    nP  = P.N * P.ratio;                                                % Step 1: N/2 parents
    nG  = N + P.n_off * nP;                                             % FK per iteration
    lo  = robot.q_min(:);
    hi  = robot.q_max(:);
    n   = numel(lo);
    vm  = hi - lo;                                                      % speed limit
    q0  = min(max(q0(:), lo), hi);
    T   = ikb_generations(opts, N, nG, P.tmax);
    [tp, tr] = ipso_tol(opts);

    S = struct('X', zeros(n, 0), 'V', zeros(n, 0), 'F', zeros(1, 0), 'XT', zeros(4, 4, 0), ...
               'Pb', zeros(n, 0), 'Fp', zeros(1, 0));
    B = struct('f', NaN, 'q', q0, 'T', NaN(4));
    fk = 0;  iters = 0;  f_init = NaN;  converged = false;
    hist = zeros(1, 0);  wr = zeros(2, 0);  par = zeros(nP, 0);  rep = zeros(1, 0);

    if ikb_budget_left(opts) >= N
        X = [q0, lo + rand(n, N - 1) .* (hi - lo)];                     % Table 1, q0 first
        S = struct('X', X, 'V', zeros(n, N), 'F', zeros(1, N), 'XT', zeros(4, 4, N), 'Pb', X, 'Fp', zeros(1, N));
        for i = 1:N
            [S.F(i), S.XT(:, :, i)] = ipso_fitness(robot, T_d, X(:, i), P);
            fk = fk + 1;
            if i == 1 || S.F(i) < B.f                                   % initial gbest, ties to the lowest index
                B = struct('f', S.F(i), 'q', X(:, i), 'T', S.XT(:, :, i));
            end
        end
        S.Fp = S.F;
        f_init = B.f;
        converged = ipso_gate(T_d, B.T, tp, tr);
        hist = NaN(1, T);  wr = NaN(2, T);  par = zeros(nP, T);  rep = zeros(1, T);
        for t = 1:T
            if converged || ikb_budget_left(opts) < nG, break; end
            [c1, c2] = ipso_factors(t, T, P);                           % Eqs. 4-5
            w = ipso_inertia(S.F, P);                                   % Eq. 3 on the fitbox of iteration t
            for i = 1:N                                                 % Table 1, particle loop
                [S.X(:, i), S.V(:, i)] = ipso_move(S.X(:, i), S.V(:, i), S.Pb(:, i), B.q, w(i), c1, c2, ...
                                                   lo, hi, vm);         % Eqs. 1-2, boundary check
                [S.F(i), S.XT(:, :, i)] = ipso_fitness(robot, T_d, S.X(:, i), P);
                fk = fk + 1;
                [S, B] = ipso_best(S, B, i);                            % pbest_i and gbest
            end
            idx = ipso_select(S.F, nP);                                 % Step 1
            Sp  = ipso_cols(S, idx);                                    % the parents
            for k = 1:nP                                                % Steps 2-3
                [xb, fb, Tb] = ipso_offspring(robot, T_d, Sp.X(:, k), t, T, lo, hi, P);
                fk = fk + P.n_off;
                if fb < Sp.F(k)
                    Sp.X(:, k) = xb;  Sp.F(k) = fb;  Sp.XT(:, :, k) = Tb;
                    rep(t) = rep(t) + 1;
                end
            end
            S = ipso_merge(S, Sp);                                      % Step 4
            for i = 1:N
                [S, B] = ipso_best(S, B, i);                            % pbest and gbest of the new population
            end
            iters = t;
            hist(t) = B.f;
            wr(:, t) = [min(w); max(w)];
            par(:, t) = idx(:);
            converged = ipso_gate(T_d, B.T, tp, tr);
        end
        hist = hist(1:iters);  wr = wr(:, 1:iters);  par = par(:, 1:iters);  rep = rep(1:iters);
    end

    q_best = q0;  ep = NaN;  er = NaN;
    if ~isnan(B.f)
        q_best = B.q;                                                   % gbest
        [~, e1, e2] = pose_error(T_d, B.T);
        ep = norm(e1);  er = norm(e2);
    end

    info = struct();
    info.solver    = 'IPSO';
    info.iters     = iters;                                             % iterations completed
    info.time      = toc(t_start);
    info.converged = converged;                                         % pose gate (Stop)
    info.fitness   = B.f;                                               % Eq. 9 of gbest, mm and rad
    info.fitness_init = f_init;                                         % best Eq. 9 of the initial population
    info.err_pos   = ep;                                                % m
    info.err_rot   = er;                                                % rad
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.N         = N;
    info.T         = T;
    info.history   = hist;                                              % Eq. 9 of gbest after iteration t
    info.w_range   = wr;                                                % smallest and largest Eq. 3 weight
    info.parents   = par;                                               % Step 1 parents of iteration t
    info.replaced  = rep;                                               % Step 3 replacements in iteration t
    info.final     = S;                                                 % population after the last iteration
    info.params    = P;
end

% ==================================================================== parameters and operators
function P = ipso_params()
% p.11: N = 200, t_max = 500; Table 1 (p.5): w_min = 0.6, w_max = 0.729, c_i,min = 1, c_i,max = 2; Table 3 (p.6):
% mutation ratio 0.5, 3 offspring; p.10: m = 1000; Eq. 9 positions in mm (len_scale = 1000 per m).
    P = struct('N', 200, 'tmax', 500, 'w_max', 0.729, 'w_min', 0.6, 'c_max', 2, 'c_min', 1, ...
               'ratio', 0.5, 'n_off', 3, 'm', 1000, 'len_scale', 1000);
end

function [c1, c2] = ipso_factors(t, T, P)
% Eq. 4: c1 = c1,max + (c1,min - c1,max) / t_max * t; Eq. 5: c2 = c2,min + (c2,max - c2,min) / t_max * t.
    c1 = P.c_max + (P.c_min - P.c_max) / T * t;
    c2 = P.c_min + (P.c_max - P.c_min) / T * t;
end

function w = ipso_inertia(F, P)
% Eq. 3 per particle with f_max,t and f_min,t of the fitness vector F; w_max for all if f_max,t = f_min,t (Table 1).
    fmax = max(F);
    fmin = min(F);
    if fmax ~= fmin
        w = P.w_min + (P.w_max - P.w_min) * (F - fmin) / (fmax - fmin);
    else
        w = P.w_max * ones(size(F));
    end
end

function [x, v] = ipso_move(x, v, pb, g, w, c1, c2, lo, hi, vm)
% Eq. 1 with r1, r2 ~ U(0, 1) per joint (r1 for all joints, then r2) and Eq. 2; then the boundary check of
% Table 1: the position clamped to [lo, hi], the speed bounded to [-vm, vm].
    r1 = rand(numel(x), 1);
    r2 = rand(numel(x), 1);
    v = w * v + c1 * r1 .* (pb - x) + c2 * r2 .* (g - x);                % Eq. 1
    x = x + v;                                                          % Eq. 2
    x = min(max(x, lo), hi);
    v = min(max(v, -vm), vm);
end

function [S, B] = ipso_best(S, B, i)
% pbest_i = X_i if f(X_i) < f(pbest_i) (Table 1); gbest = pbest_i if it is better than gbest.
    if S.F(i) < S.Fp(i)
        S.Pb(:, i) = S.X(:, i);
        S.Fp(i) = S.F(i);
        if S.F(i) < B.f
            B = struct('f', S.F(i), 'q', S.X(:, i), 'T', S.XT(:, :, i));
        end
    end
end

function idx = ipso_select(F, nP)
% Step 1: nP distinct parents by the roulette without replacement, weight 1/f (one rand per draw over the
% remaining particles in index order); particles with f = 0 first, uniformly among themselves.
    rem = 1:numel(F);
    idx = zeros(1, nP);
    for k = 1:nP
        u = rand;
        z = rem(F(rem) == 0);
        if ~isempty(z)
            s = z(max(1, ceil(u * numel(z))));
        else
            c = cumsum(1 ./ F(rem));
            s = rem(find(c >= u * c(end), 1));
        end
        idx(k) = s;
        rem(rem == s) = [];
    end
end

function [c, j, r] = ipso_child(x, t, T, lo, hi)
% Eq. 6 at one location j (uniform): k = r (1 - t / t_max)^2; r >= 0.5 -> toward x_min, r < 0.5 -> toward x_max.
    j = randi(numel(x));
    r = rand;
    k = r * (1 - t / T)^2;
    c = x;
    if r >= 0.5
        c(j) = x(j) - k * (x(j) - lo(j));
    else
        c(j) = x(j) + k * (hi(j) - x(j));
    end
end

function [xb, fb, Tb, C, Fc] = ipso_offspring(robot, T_d, x, t, T, lo, hi, P)
% Steps 2-3 for one parent: P.n_off offspring by Eq. 6, each evaluated once; the best is the first with the
% lowest fitness.
    C  = zeros(numel(x), P.n_off);
    Fc = zeros(1, P.n_off);
    for c = 1:P.n_off
        C(:, c) = ipso_child(x, t, T, lo, hi);
        [Fc(c), Tq] = ipso_fitness(robot, T_d, C(:, c), P);
        if c == 1 || Fc(c) < fb
            xb = C(:, c);  fb = Fc(c);  Tb = Tq;
        end
    end
end

function S = ipso_merge(S0, Sp)
% Step 4: the best size(S0) - size(Sp) particles prior to mutation (ascending fitness, ties to the lower index)
% followed by the parents after Step 3; every particle keeps its velocity and pbest.
    [~, ord] = sort(S0.F);
    top = ord(1:numel(S0.F) - numel(Sp.F));
    S = struct('X', [S0.X(:, top), Sp.X], 'V', [S0.V(:, top), Sp.V], 'F', [S0.F(top), Sp.F], ...
               'XT', cat(3, S0.XT(:, :, top), Sp.XT), 'Pb', [S0.Pb(:, top), Sp.Pb], 'Fp', [S0.Fp(top), Sp.Fp]);
end

function S = ipso_cols(S, idx)
% The particles idx of the population S.
    S = struct('X', S.X(:, idx), 'V', S.V(:, idx), 'F', S.F(idx), 'XT', S.XT(:, :, idx), ...
               'Pb', S.Pb(:, idx), 'Fp', S.Fp(idx));
end

function [tp, tr] = ipso_tol(opts)
% Pose gate tolerances (m, rad); default_opts values when absent.
    tp = 1e-6;
    tr = 1e-6;
    if isfield(opts, 'tol_pos'), tp = opts.tol_pos; end
    if isfield(opts, 'tol_rot'), tr = opts.tol_rot; end
end

function ok = ipso_gate(T_d, Tg, tp, tr)
% Benchmark pose gate on the pose Tg of gbest: position error < tp and rotation error < tr (pose_error).
    ok = false;
    if any(isnan(Tg(:))), return; end
    [~, e1, e2] = pose_error(T_d, Tg);
    ok = norm(e1) < tp && norm(e2) < tr;
end

% ==================================================================== IK fitness (Eq. 9, Euler angles p.9)
function [f, T] = ipso_fitness(robot, T_d, q, P)
% Eq. 9 of the pose robot.fk(q) against T_d; one FK call.
    T = robot.fk(q);
    f = ipso_eq9(T, T_d, P);
end

function [f, dp, de] = ipso_eq9(T, T_d, P)
% Eq. 9: position differences in mm, plain Euler-angle differences in rad, weight m on the angular terms.
    dp = P.len_scale * (T(1:3, 4) - T_d(1:3, 4));
    de = ipso_euler(T) - ipso_euler(T_d);
    f  = sqrt(dp(1)^2 + dp(2)^2 + dp(3)^2 + P.m * (de(1)^2 + de(2)^2 + de(3)^2));
end

function e = ipso_euler(T)
% p.9: alpha = arctan(-a_y / a_z), beta = arctan(a_x / sqrt(n_x^2 + o_x^2)), gamma = arctan(-o_x / n_x), with the
% four-quadrant arctan; n, o, a = columns 1-3 of the rotation.
    e = [atan2(-T(2, 3), T(3, 3));
         atan2(T(1, 3), sqrt(T(1, 1)^2 + T(1, 2)^2));
         atan2(-T(1, 2), T(1, 1))];
end
