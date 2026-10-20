function [q_best, info] = solve_PLPSO(robot, T_d, q0, opts)
% SOLVE_PLPSO  Parallel learning particle swarm optimizer (PLPSO) IK solver of Liu et al. (2021), Alg. 1 (p.11).
%
%   Source:
%     Fei Liu, Hailin Huang, Bing Li and Fengfeng Xi (2021). A parallel learning particle swarm optimizer for
%     inverse kinematics of robotic manipulator. Int J Intell Syst. 2021;36:6101-6132. DOI: 10.1002/int.22543
%   The printed citation (p.1 footer, p.32 "How to cite") has no issue number. Page numbers below are PDF pages
%   1-32 (printed page = PDF page + 6100).
%
%   Algorithm (Alg. 1 = Fig. 5, p.11; Fig. 1, p.6; Eqs. 1-10, pp.5-10; Table 1, p.9; IK fitness Eq. 19, p.24, with
%   the Euler angles of p.23):
%     s particles; the position x_i of particle i is its joint vector (p.24). Alg. 1 l.3 initialises positions and
%     velocities; l.4 divides the swarm once into two subpopulations Q_A and Q_B "with different sizes" (p.6),
%     "randomly divided into two independent particle subpopulations" (p.6); l.5-6 evaluate them and set pbest_i,
%     gbest_1 (best of Q_A) and gbest_2 (best of Q_B). Each iteration t = 1..t_max (Alg. 1 l.7-32):
%       w(t) = (w1 - w2) x (t_max - t) / t_max + w2                                                   (Eq. 3)
%       Ef_1 of Q_A and Ef_2 of Q_B (Alg. 1 l.9), each over its own subpopulation of s_k particles:
%         d_i = 1 / (s - 1) x sum_{j = 1, j != i}^{s} sqrt(sum_{k = 1}^{d} (x_i(t) - x_j(t))^2)       (Eq. 6)
%         Ef = (d_a - d_min) / (d_max - d_min)                                                         (Eq. 5)
%         d_a = "the average distance between the global best particle and all other particles" (p.8).
%       Trigger conditions (Sec. 2.2.3, Fig. 4, pp.9-10): C1: t % T = 0; C2: 0.25 <= Ef_1 <= 1; C3:
%         0.25 <= Ef_2 <= 1. Q_A learns from Q_B if C1 or C2 holds, Q_B learns from Q_A if C1 or C3 holds ("If the
%         C1 condition is satisfied, two swarms Q_A and Q_B will learn from each other", p.9).
%       Learning of Q_A from Q_B (Alg. 1 l.11-15; Q_B from Q_A, l.17-21, with the roles exchanged):
%         lambda = ceil((1 + r_lambda x Ef) x m / 4), m / 4 <= lambda <= m / 2                         (Eq. 10)
%         wQ_A = the lambda particles of Q_A farthest from the optimum, sorted "the farther the distance is, the
%         higher the sorting is" (wp_1..wp_lambda); bQ_B = the lambda particles of Q_B closest to it, sorted "the
%         closer the distance is, the higher the sorting is" (bp_1..bp_lambda) (p.7); the distance to the optimum
%         is the fitness (p.6); wp_i learns from bp_i:
%         wv_i^d(t) = bv_i^d(t) + s_a x c_a x r_1^d x (pbest_i^d(t - t x r_2^d) - wx_i^d(t))
%                               + s_b x c_b x r_3^d x (gbest^d(t - t x r_4^d) - wx_i^d(t))           (Eq. 7, l.14)
%         wx_i^d(t) = wx_i^d(t) + s_p x r_p^d x (bx_i^d(t) - wx_i^d(t))                               (Eq. 4, l.15)
%         with s_a, s_b, s_p of Table 1 (p.9) at the Ef of the learning subpopulation: Ef in [0, 0.25):
%         0, 0, Ef (exploration); [0.25, 0.5): Ef, 0, Ef (exploitation); [0.5, 0.75): 0, Ef, Ef (convergence);
%         [0.75, 1]: Ef, Ef, Ef (jumping out).
%       Every particle (Fig. 1, p.6), with gbest the gbest of its own subpopulation:
%         v_i^d(t + 1) = w x v_i^d(t) + c_a x r_a^d x (pbest_i^d - x_i^d(t)) + c_b x r_b^d x (gbest^d - x_i^d(t))  (Eq. 1)
%         x_i^d(t + 1) = x_i^d(t) + v_i^d(t + 1)                                                       (Eq. 2)
%       Correction (Alg. 1 l.26):
%         v_i^d = v_max^d - r_v x (v_max^d - v_min^d) if v_i^d > v_max^d; v_min^d + r_v x (v_max^d - v_min^d)
%                 if v_i^d < v_min^d                                                                    (Eq. 8)
%         x_i^d = x_max^d - r_x x (x_max^d - x_min^d) if x_i^d > x_max^d; x_min^d + r_x x (x_max^d - x_min^d)
%                 if x_i^d < x_min^d                                                                    (Eq. 9)
%       Fitness of Q_A and Q_B, update of pbest_i, gbest_1, gbest_2 (l.27-28); stop if a subpopulation's best meets
%       the accuracy err (l.29); return gBest = min(gbest_1, gbest_2) (l.33).
%     IK fitness (Eq. 19, p.24), with P = [alpha beta gamma p_x p_y p_z] of the pose and P_0 of the target:
%       f_error = [(p_xi - p_x0)^2 + (p_yi - p_y0)^2 + (p_zi - p_z0)^2
%                  + (alpha_i - alpha_0)^2 + (beta_i - beta_0)^2 + (gamma_i - gamma_0)^2]^(1/2)               (Eq. 19)
%     with the posture angles of p.23, n, o, a the columns of the rotation (Eq. 12):
%       alpha = actan(-a_y / a_z), beta = actan(a_x / sqrt(n_x^2 + o_x^2)), gamma = actan(-o_x / n_x).
%
%   Parameters:
%     s = 100, t_max = 1000   the IK setting: Table 7 (p.25) and Table 8 (p.27) list PLPSO with "Particle size"
%                     100; p.24: "The maximum number of iterations is set to 1000." N x t_max = 100 x 1000 =
%                     100,000, the only IK setting reported (scenario 2 of p.24 states no other iteration count);
%                     100 x 10,000 (p.14) is the CEC2013 setting and 5, 20, 100 x 10,000 (p.12) the structural-bias
%                     study. Inside the benchmark t_max is the budget horizon (Benchmark interface).
%     c_a = c_b = 1.49445   p.8: "Same as in Equation (1), c_a and c_b are equal to 1.49445 here".
%     T = 20          period of trigger C1, p.9: "In this study, let T = 20".
%     Ef bounds       0.25, 0.5, 0.75 of Table 1 (p.9); C2 and C3 test 0.25 <= Ef <= 1 (Alg. 1 l.10, l.16; Fig. 4);
%                     the p.10 text writes 0.25 < Ef, which differs only at Ef = 0.25 exactly.
%     w1 = 0.9, w2 = 0.4   not stated for PLPSO in the paper (checked pages 5-11, 14, 16 and 23-25). Eq. 3 is "the
%                     linearly decreased inertia weight proposed by Shi and Eberhart" (refs. 21, 47, p.6), and the
%                     paper's Table 3 (p.16) lists that method (LDWPSO, refs. 21, 47 on p.14) with w = 0.9 -> 0.4
%                     and c1 = c2 = 1.49445; the benchmark uses those two values.
%     Subpopulation sizes   "with different sizes" (p.6); the sizes are not stated in the paper (checked pages 5-12,
%                     14 and 23-28). Q_A = 49 and Q_B = 51 particles, the split of s = 100 closest to two halves
%                     that has different sizes. The members are random (one randperm, Alg. 1 l.4, "randomly
%                     divided", p.6), drawn once; each subpopulation keeps its own pbest set, gbest and Ef.
%     m in Eq. 10     printed as "the maximum number of iterations" (p.10). With the paper's own t_max = 1000 that
%                     gives lambda = 250 to 500 particles, more than the swarm (s = 100), so Eq. 10 is undefined as
%                     printed. Read as the number of particles of the learning subpopulation, so that between a
%                     quarter and a half of it learns ("m/4 <= lambda <= m/2", p.10): Q_A 13 to 25 of 49, Q_B 13
%                     to 26 of 51, always within the other subpopulation's size.
%     d_a in Eq. 5    from the gbest position of the subpopulation (p.8: "the global best particle"): the mean
%                     distance from it to the members that are not at that position, so d_a = d_b when gbest is the
%                     current position of member b. A gbest found earlier need not be a current position, so Eq. 5
%                     can leave [0, 1], where Table 1 and Eq. 10 are defined: Ef is clamped to [0, 1]; Ef = 0 when
%                     d_max = d_min (Eq. 5 undefined; not stated in the paper, checked pages 7-10).
%     Delays (Eq. 7)  t - t x r is not an integer and its rounding is not stated in the paper (checked pages 8 and
%                     10). The records are pbest and gbest at the start of iterations 1..t (record 1 = the
%                     initial values of Alg. 1 l.6, record t = the current ones); Eq. 7 reads record
%                     ceil(t - t x r) per joint, so each of the t records has probability 1/t. pbest_i is the
%                     learner's own personal best and gbest the gbest of the learner's subpopulation (p.8: "local and
%                     global delayed information of the particles in wQ"); bv_i and bx_i are the current velocity
%                     and position of the paired particle bp_i.
%     Speed range     [v_min, v_max] of Eq. 8 is not stated in the paper (checked pages 5-11, 14, 16 and 23-25);
%                     v_max = x_max - x_min and v_min = -v_max per joint, the width of the search range (Table 6,
%                     fifth column), the only range the paper defines.
%     Initialisation  Alg. 1 l.3, "Initialize Particles(SizeP, D): position&velocity", gives no rule (checked pages
%                     5-11 and 24): positions uniform in [x_min, x_max], velocities zero.
%     Random numbers  r^d in Eqs. 1, 4 and 7 are drawn per joint (superscript d); r_v and r_x of Eqs. 8-9 carry no
%                     superscript: one r_v and one r_x per particle and iteration, shared by its joints outside the
%                     range; one r_lambda per learning event.
%     Learning iteration   Alg. 1 runs Eqs. 1-2 only in its "else" branch (l.22-24) and tests C3 in an "else if"
%                     (l.16), so only one subpopulation could learn per iteration and nothing else would move;
%                     Fig. 1 (p.6) goes from "Two subpopulations learn from each other" to Eqs. 1-2 for all
%                     particles, p.6 says the subpopulations "learn from each other, and become updated. The velocity
%                     and position of the particles are updated according to (1) and (2)", p.9 says both learn under
%                     C1, and Eqs. 4 and 7 assign the state at t while Eqs. 1-2 produce t + 1. Read: each
%                     subpopulation tests its own trigger; learning (Eq. 7, then Eq. 4, the Alg. 1 l.14-15 order,
%                     so Eq. 7 uses the position before Eq. 4) changes the worse particles' state at t; then Eqs. 1-2
%                     move every particle, and Eqs. 8-9 correct it (l.26). p.7 lists position learning before the
%                     velocity update. Both directions read the state at the start of iteration t ("evolve in
%                     parallel", p.6). A learned position is not evaluated before Eqs. 1-2 move it (Fig. 1).
%     Ranking         by the fitness of the current positions at the start of the iteration (stable sort: of equal
%                     fitness the lower particle index ranks first); strict "<" in the pbest and gbest updates;
%                     gbest of a subpopulation = its first member (lowest index) with the lowest fitness.
%     Units           mm and rad. Table 6 (p.23) gives the UR5 in mm; Fig. 12 (p.25) starts near 2 on its log10
%                     scale and Fig. 16 (p.27) near 250, the fitness of random poses in mm; the IK solutions of
%                     Table 9 (p.28) reach the targets P1 = [200 200 200 pi 0 0] and P2 = [400 400 300 pi 0 0]
%                     (p.27) on the Table 6 arm in mm. The position terms of Eq. 19 are computed in mm (1000 x the
%                     difference in m).
%     Arctan          the four-quadrant arctan: alpha = atan2(-a_y, a_z), beta = atan2(a_x, sqrt(n_x^2 + o_x^2)),
%                     gamma = atan2(-o_x, n_x); P1 and P2 have alpha = pi, outside the range of the one-argument
%                     arctan. Eq. 19 takes the plain differences of the angles (no wrapping).
%     Stop            err (Alg. 1 l.1, l.29) is not stated in the paper (checked pages 6, 11, 14 and 24-25). It is
%                     the benchmark pose gate: position error < opts.tol_pos (m) and rotation error < opts.tol_rot
%                     (rad) (pose_error) of gbest_1 or gbest_2 (l.29 prints pbest_1, pbest_2; Fig. 1 prints gbest_1,
%                     gbest_2), tested after the initial population (Fig. 1 tests before the first move) and after
%                     every iteration (l.29). The returned gBest is min(gbest_1, gbest_2) by Eq. 19 (l.33), taken
%                     among the subpopulation bests that meet the gate when the stop fired; info.converged = the
%                     gate of the returned joints.
%     Random draws    initial positions of particles 2..s (rand(n, s - 1)), then randperm(s) for the split (Q_A =
%                     the first 49 entries). Per iteration: for Q_A, then Q_B, if it learns: r_lambda, then per pair
%                     r_1, r_2, r_3, r_4 (n draws each), then per pair r_p (n draws); then per particle in index
%                     order r_a, r_b (n draws each); then per particle in index order r_v if one of its speeds is
%                     out of range and r_x if one of its joints is out of range.
%
%   Benchmark interface:
%     Budget: the initial population costs s = 100 FK calls and every iteration s = 100 (Alg. 1 l.27). t_max =
%       ikb_generations(opts, 100, 100, 1000) = 19 iterations, so 100 + 19 x 100 = 2000 FK per waypoint. A
%       40-member population would give ikb_generations(opts, 40, 40, 1000) = 49. The initial population and every
%       iteration start only if ikb_budget_left(opts) covers them. Outside the benchmark t_max = 1000 (100,100 FK).
%       No final FK check. Fields of opts other than tol_pos and tol_rot (max_iter, pop_size, ...) are ignored.
%     Schedules: Eq. 3 runs over this t_max: w = 0.874 at t = 1, 0.4 at t = 19. The delays of Eq. 7 reach back over
%       the records of the current run. The period T = 20 of C1 is the paper's constant (p.9), not a schedule, so
%       with t_max = 19 C1 is never reached and learning is triggered by C2 and C3 only.
%     Stop: the pose gate above, or t_max iterations, or the budget.
%     Warm start: q0 (clamped to the limits) is particle 1 of the initial population; it is evaluated and enters the
%       gbest of the subpopulation the split assigns it to (ties to the lowest index).
%     Robot and units: Eqs. 12-18 and Table 6 (the paper's UR5) are replaced by robot.fk with D = robot.n joints;
%       lengths in m, converted to mm inside Eq. 19; angles in rad.
%     Joint limits: [x_min, x_max] = [robot.q_min, robot.q_max]; a joint outside it is reset by Eq. 9 (the paper's
%       rule, a random re-initialisation, not a clamp); Eq. 4 keeps a learner inside (an interpolation).
%     Output: min(gbest_1, gbest_2) as above; info.fitness (Eq. 19, mm and rad), info.err_pos (m) and info.err_rot
%       (rad, pose_error) from the pose stored at its evaluation. solve_PLPSO('ops') returns handles to the
%       operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration horizon t_max = 19 instead of 1000 (benchmark rule); Eq. 3 runs over it.
%     - q0 is particle 1 of the initial population (warm start).
%     - Stop at the benchmark pose gate (err is not stated).
%     - Not stated in the paper, chosen as given under Parameters: w1 and w2, the subpopulation sizes, the speed
%       range, the initial positions and velocities, the rounding of the delays, Ef when gbest lies outside
%       [d_min, d_max] or d_max = d_min, the ranking ties, the order of the draws.
%     - Readings of printed inconsistencies, as given under Parameters: m in Eq. 10, the control flow of a learning
%       iteration (independent triggers, Eqs. 1-2 for all particles), the order of Eqs. 7 and 4, the gbest and
%       pbest of Eq. 7, the four-quadrant arctan.
%     - robot.fk instead of Eqs. 12-18 (the paper's UR5).
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
        q_best = struct('params', plp_params(), 'core', @plp_core, 'inertia', @plp_inertia, 'ef', @plp_ef, ...
                        'table1', @plp_table1, 'trigger', @plp_trigger, 'lambda', @plp_lambda, 'rank', @plp_rank, ...
                        'delay', @plp_delay, 'learn', @plp_learn, 'move', @plp_move, 'correct', @plp_correct, ...
                        'fitness', @plp_fitness, 'eq19', @plp_eq19, 'euler', @plp_euler, 'gate', @plp_gate);
        info = [];
        return
    end
    [q_best, info] = plp_core(robot, T_d, q0, opts, plp_params());
end

% ==================================================================== main loop (Alg. 1, Fig. 1)
function [q_best, info] = plp_core(robot, T_d, q0, opts, P)
% Initial population (q0 first), split once into Q_A and Q_B, evaluation, pbest and the two gbest; per iteration
% Eq. 3, Eqs. 5-6 per subpopulation, triggers C1-C3, learning (Eqs. 10, 7, 4), Eqs. 1-2 for all particles, Eqs. 8-9,
% evaluation, pbest and gbest updates, records for the delays; stop at the pose gate, at t_max or at the budget.
    t_start = tic;
    s   = P.s;
    lo  = robot.q_min(:);
    hi  = robot.q_max(:);
    n   = numel(lo);
    vmx = hi - lo;                                                      % speed range [-vmx, vmx] (Eq. 8)
    vmn = -vmx;
    q0  = min(max(q0(:), lo), hi);
    T   = ikb_generations(opts, s, s, P.tmax);
    [tp, tr] = plp_tol(opts);

    S  = struct('X', zeros(n, 0), 'V', zeros(n, 0), 'F', zeros(1, 0), 'Pb', zeros(n, 0), 'Fp', zeros(1, 0));
    G  = {zeros(1, 0), zeros(1, 0)};
    Gq = repmat(q0, 1, 2);  Gf = [NaN NaN];  GT = NaN(4, 4, 2);
    fk = 0;  iters = 0;  f_init = NaN;  converged = false;
    hist = zeros(1, 0);  EF = zeros(2, 0);  ER = zeros(2, 0);  LR = false(2, 0);  LA = zeros(2, 0);  PR = cell(2, 0);

    if ikb_budget_left(opts) >= s
        X  = [q0, lo + rand(n, s - 1) .* (hi - lo)];                    % Alg. 1 l.3, q0 first
        pm = randperm(s);                                               % Alg. 1 l.4, once
        G  = {sort(pm(1:P.nA)), sort(pm(P.nA + 1:s))};
        sub = zeros(1, s);  sub(G{1}) = 1;  sub(G{2}) = 2;
        S  = struct('X', X, 'V', zeros(n, s), 'F', zeros(1, s), 'Pb', X, 'Fp', zeros(1, s));
        XT = zeros(4, 4, s);
        for i = 1:s                                                     % Alg. 1 l.5
            [S.F(i), XT(:, :, i)] = plp_fitness(robot, T_d, X(:, i), P);
            fk = fk + 1;
        end
        S.Fp = S.F;  PT = XT;
        for k = 1:2                                                     % Alg. 1 l.6
            [Gf(k), j] = min(S.F(G{k}));
            Gq(:, k) = X(:, G{k}(j));  GT(:, :, k) = XT(:, :, G{k}(j));
        end
        f_init = min(Gf);
        Hp = zeros(n, T + 1, s);  Hp(:, 1, :) = reshape(S.Pb, n, 1, s); % records for the delays of Eq. 7
        Hg = zeros(n, T + 1, 2);  Hg(:, 1, :) = reshape(Gq, n, 1, 2);
        converged = plp_gate(T_d, GT(:, :, 1), tp, tr) || plp_gate(T_d, GT(:, :, 2), tp, tr);
        hist = NaN(1, T);  EF = NaN(2, T);  ER = NaN(2, T);  LR = false(2, T);  LA = zeros(2, T);  PR = cell(2, T);
        for t = 1:T
            if converged || ikb_budget_left(opts) < s, break; end
            w = plp_inertia(t, T, P);                                   % Eq. 3 (l.8)
            [EF(1, t), ER(1, t)] = plp_ef(S.X(:, G{1}), Gq(:, 1));      % Eqs. 5-6 (l.9)
            [EF(2, t), ER(2, t)] = plp_ef(S.X(:, G{2}), Gq(:, 2));
            LR(:, t) = plp_trigger(t, EF(:, t), P);                     % C1-C3 (l.10, l.16)
            X0 = S.X;  V0 = S.V;
            for k = 1:2
                if LR(k, t)
                    o = 3 - k;
                    LA(k, t) = plp_lambda(EF(k, t), numel(G{k}));       % Eq. 10 (l.11, l.17)
                    [W, B] = plp_rank(S.F, G{k}, G{o}, LA(k, t));       % l.12-13, l.18-19
                    [S.X(:, W), S.V(:, W)] = plp_learn(X0, V0, W, B, t, EF(k, t), Hp, Hg(:, :, k), P);
                    PR{k, t} = [W; B];                                  % Eqs. 7 and 4 (l.14-15, l.20-21)
                end
            end
            [S.X, S.V] = plp_move(S.X, S.V, S.Pb, Gq(:, sub), w, P);    % Eqs. 1-2 (Fig. 1)
            [S.X, S.V] = plp_correct(S.X, S.V, lo, hi, vmn, vmx);       % Eqs. 8-9 (l.26)
            for i = 1:s                                                 % l.27-28
                [S.F(i), XT(:, :, i)] = plp_fitness(robot, T_d, S.X(:, i), P);
                fk = fk + 1;
                if S.F(i) < S.Fp(i)
                    S.Pb(:, i) = S.X(:, i);  S.Fp(i) = S.F(i);  PT(:, :, i) = XT(:, :, i);
                end
            end
            for k = 1:2
                [fm, j] = min(S.Fp(G{k}));
                if fm < Gf(k)
                    Gf(k) = fm;  Gq(:, k) = S.Pb(:, G{k}(j));  GT(:, :, k) = PT(:, :, G{k}(j));
                end
            end
            Hp(:, t + 1, :) = reshape(S.Pb, n, 1, s);
            Hg(:, t + 1, :) = reshape(Gq, n, 1, 2);
            iters = t;
            hist(t) = min(Gf);
            converged = plp_gate(T_d, GT(:, :, 1), tp, tr) || plp_gate(T_d, GT(:, :, 2), tp, tr);   % l.29
        end
        hist = hist(1:iters);  EF = EF(:, 1:iters);  ER = ER(:, 1:iters);  LR = LR(:, 1:iters);
        LA = LA(:, 1:iters);  PR = PR(:, 1:iters);
    end

    q_best = q0;  kb = 0;  fb = NaN;  ep = NaN;  er = NaN;
    if ~isnan(f_init)
        ok = [plp_gate(T_d, GT(:, :, 1), tp, tr), plp_gate(T_d, GT(:, :, 2), tp, tr)];
        cand = 1:2;
        if any(ok), cand = find(ok); end
        [fb, j] = min(Gf(cand));                                        % l.33, ties to Q_A
        kb = cand(j);
        q_best = Gq(:, kb);
        converged = ok(kb);
        [~, e1, e2] = pose_error(T_d, GT(:, :, kb));
        ep = norm(e1);  er = norm(e2);
    end

    info = struct();
    info.solver    = 'PLPSO';
    info.iters     = iters;                                             % iterations completed
    info.time      = toc(t_start);
    info.converged = converged;                                         % pose gate of the returned joints
    info.fitness   = fb;                                                % Eq. 19 of gBest, mm and rad
    info.fitness_init = f_init;                                         % min(gbest_1, gbest_2) after Alg. 1 l.6
    info.err_pos   = ep;                                                % m
    info.err_rot   = er;                                                % rad
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.N         = s;
    info.T         = T;
    info.groups    = G;                                                 % members of Q_A and Q_B
    info.best_sub  = kb;                                                % 1 = Q_A, 2 = Q_B
    info.gbest_fitness = Gf;                                            % Eq. 19 of gbest_1, gbest_2
    info.history   = hist;                                              % min(gbest_1, gbest_2) after iteration t
    info.Ef        = EF;                                                % Ef_1, Ef_2 per iteration (clamped)
    info.Ef_raw    = ER;                                                % Eq. 5 before clamping (NaN: d_max = d_min)
    info.learn     = LR;                                                % trigger of Q_A, Q_B per iteration
    info.lambda    = LA;                                                % Eq. 10 (0: no learning)
    info.pairs     = PR;                                                % [wp_1..wp_lambda; bp_1..bp_lambda]
    info.final     = S;                                                 % population after the last iteration
    info.params    = P;
end

% ==================================================================== parameters and operators
function P = plp_params()
% Table 7/8 and p.24: s = 100, t_max = 1000; p.8: c_a = c_b = 1.49445; p.9: T = 20; Table 3 (LDW of refs. 21, 47):
% w1 = 0.9, w2 = 0.4; sizes 49 / 51 of Q_A / Q_B; Table 1 / Fig. 4: lower trigger bound 0.25; Eq. 19 lengths in mm.
    P = struct('s', 100, 'nA', 49, 'tmax', 1000, 'Tp', 20, 'ca', 1.49445, 'cb', 1.49445, 'w1', 0.9, 'w2', 0.4, ...
               'ef_lo', 0.25, 'len_scale', 1000);
end

function w = plp_inertia(t, T, P)
% Eq. 3: w(t) = (w1 - w2) x (t_max - t) / t_max + w2.
    w = (P.w1 - P.w2) * (T - t) / T + P.w2;
end

function [Ef, raw, da, d] = plp_ef(Xk, g)
% Eq. 6 over the members Xk of one subpopulation, d_a from its gbest position g (mean distance to the members not
% at g), Eq. 5 clamped to [0, 1]; Ef = 0 (raw = NaN) when d_max = d_min. Sums in index order.
    [n, s] = size(Xk);
    d = zeros(1, s);
    for i = 1:s
        for j = 1:s
            if j ~= i
                a = 0;
                for k = 1:n, a = a + (Xk(k, i) - Xk(k, j))^2; end
                d(i) = d(i) + sqrt(a);
            end
        end
    end
    d = d / (s - 1);
    da = 0;  c = 0;
    for j = 1:s
        a = 0;
        for k = 1:n, a = a + (g(k) - Xk(k, j))^2; end
        if a > 0, da = da + sqrt(a);  c = c + 1; end
    end
    if c > 0, da = da / c; end
    dmin = min(d);  dmax = max(d);
    if dmax > dmin
        raw = (da - dmin) / (dmax - dmin);
        Ef  = min(1, max(0, raw));
    else
        raw = NaN;  Ef = 0;
    end
end

function [sa, sb, sp] = plp_table1(Ef)
% Table 1: delay coefficients s_a, s_b and learning coefficient s_p for the evolutionary factor Ef.
    if Ef < 0.25
        sa = 0;  sb = 0;
    elseif Ef < 0.5
        sa = Ef;  sb = 0;
    elseif Ef < 0.75
        sa = 0;  sb = Ef;
    else
        sa = Ef;  sb = Ef;
    end
    sp = Ef;
end

function lr = plp_trigger(t, Ef, P)
% C1: t % T = 0; C2: 0.25 <= Ef_1 <= 1 (Q_A learns); C3: 0.25 <= Ef_2 <= 1 (Q_B learns). Column [Q_A; Q_B].
    c1 = mod(t, P.Tp) == 0;
    lr = [c1 || (Ef(1) >= P.ef_lo && Ef(1) <= 1); c1 || (Ef(2) >= P.ef_lo && Ef(2) <= 1)];
end

function [lam, r] = plp_lambda(Ef, m)
% Eq. 10: lambda = ceil((1 + r_lambda x Ef) x m / 4), m = size of the learning subpopulation; one draw.
    r = rand;
    lam = ceil((1 + r * Ef) * m / 4);
end

function [W, B] = plp_rank(F, Gk, Go, lam)
% Alg. 1 l.12-13: the lam particles of Gk with the largest fitness, largest first (wp_1..wp_lam), and the lam
% particles of Go with the smallest fitness, smallest first (bp_1..bp_lam); stable sorts (ties to the lower index).
    [~, ow] = sort(F(Gk), 'descend');
    [~, ob] = sort(F(Go), 'ascend');
    W = Gk(ow(1:lam));
    B = Go(ob(1:lam));
end

function y = plp_delay(H, t, r)
% Delayed record of Eq. 7: joint d of record ceil(t - t x r_d) of H (n x records), record 1 = initial.
    n = size(H, 1);
    k = max(1, ceil(t - t * r));
    y = zeros(n, 1);
    for d = 1:n, y(d) = H(d, k(d)); end
end

function [Xw, Vw] = plp_learn(X0, V0, W, B, t, Ef, Hp, Hg, P)
% Alg. 1 l.14-15 for the pairs (W(i), B(i)): Eq. 7 for every pair (draws r_1, r_2, r_3, r_4), then Eq. 4 for every
% pair (draws r_p); X0, V0 = the state at the start of the iteration; Hp = pbest records (n x records x s), Hg =
% gbest records of the learner's subpopulation (n x records).
    [sa, sb, sp] = plp_table1(Ef);
    n   = size(X0, 1);
    lam = numel(W);
    Xw  = zeros(n, lam);
    Vw  = zeros(n, lam);
    for i = 1:lam                                                       % Eq. 7
        r1 = rand(n, 1);  r2 = rand(n, 1);  r3 = rand(n, 1);  r4 = rand(n, 1);
        pd = plp_delay(Hp(:, :, W(i)), t, r2);
        gd = plp_delay(Hg, t, r4);
        Vw(:, i) = V0(:, B(i)) + sa * P.ca * r1 .* (pd - X0(:, W(i))) + sb * P.cb * r3 .* (gd - X0(:, W(i)));
    end
    for i = 1:lam                                                       % Eq. 4
        rp = rand(n, 1);
        Xw(:, i) = X0(:, W(i)) + sp * rp .* (X0(:, B(i)) - X0(:, W(i)));
    end
end

function [X, V] = plp_move(X, V, Pb, Gs, w, P)
% Eqs. 1-2 for every particle in index order (r_a, then r_b, per joint); Gs(:, i) = gbest of particle i's
% subpopulation.
    [n, s] = size(X);
    for i = 1:s
        ra = rand(n, 1);  rb = rand(n, 1);
        V(:, i) = w * V(:, i) + P.ca * ra .* (Pb(:, i) - X(:, i)) + P.cb * rb .* (Gs(:, i) - X(:, i));   % Eq. 1
        X(:, i) = X(:, i) + V(:, i);                                                                     % Eq. 2
    end
end

function [X, V] = plp_correct(X, V, lo, hi, vmn, vmx)
% Eqs. 8-9 per particle in index order: one r_v for its speeds outside [vmn, vmx], one r_x for its joints outside
% [lo, hi], each drawn only when needed; both branches resample uniformly over the whole range.
    for i = 1:size(X, 2)
        up = V(:, i) > vmx;  dn = V(:, i) < vmn;
        if any(up | dn)
            rv = rand;
            V(up, i) = vmx(up) - rv * (vmx(up) - vmn(up));
            V(dn, i) = vmn(dn) + rv * (vmx(dn) - vmn(dn));
        end
        up = X(:, i) > hi;  dn = X(:, i) < lo;
        if any(up | dn)
            rx = rand;
            X(up, i) = hi(up) - rx * (hi(up) - lo(up));
            X(dn, i) = lo(dn) + rx * (hi(dn) - lo(dn));
        end
    end
end

function [tp, tr] = plp_tol(opts)
% Pose gate tolerances (m, rad); default_opts values when absent.
    tp = 1e-6;
    tr = 1e-6;
    if isfield(opts, 'tol_pos'), tp = opts.tol_pos; end
    if isfield(opts, 'tol_rot'), tr = opts.tol_rot; end
end

function ok = plp_gate(T_d, Tg, tp, tr)
% Benchmark pose gate on the pose Tg: position error < tp and rotation error < tr (pose_error).
    ok = false;
    if any(isnan(Tg(:))), return; end
    [~, e1, e2] = pose_error(T_d, Tg);
    ok = norm(e1) < tp && norm(e2) < tr;
end

% ==================================================================== IK fitness (Eq. 19, Euler angles p.23)
function [f, T] = plp_fitness(robot, T_d, q, P)
% Eq. 19 of the pose robot.fk(q) against T_d; one FK call.
    T = robot.fk(q);
    f = plp_eq19(T, T_d, P);
end

function [f, dp, de] = plp_eq19(T, T_d, P)
% Eq. 19: position differences in mm, plain Euler-angle differences in rad, unweighted.
    dp = P.len_scale * (T(1:3, 4) - T_d(1:3, 4));
    de = plp_euler(T) - plp_euler(T_d);
    f  = sqrt(dp(1)^2 + dp(2)^2 + dp(3)^2 + de(1)^2 + de(2)^2 + de(3)^2);
end

function e = plp_euler(T)
% p.23: alpha = actan(-a_y / a_z), beta = actan(a_x / sqrt(n_x^2 + o_x^2)), gamma = actan(-o_x / n_x), with the
% four-quadrant arctan; n, o, a = columns 1-3 of the rotation.
    e = [atan2(-T(2, 3), T(3, 3));
         atan2(T(1, 3), sqrt(T(1, 1)^2 + T(1, 2)^2));
         atan2(-T(1, 2), T(1, 1))];
end
