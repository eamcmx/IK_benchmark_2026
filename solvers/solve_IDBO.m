function [q_best, info] = solve_IDBO(robot, T_d, q0, opts)
% SOLVE_IDBO  Improved dung beetle optimization (I-DBO) IK solver of Li et al. (2024), Secs. II.B and III.
%
%   Source:
%     Jinjie Li, Wenjie Wang, Shuaixiu Wang, Shixi Zhang, Qi Yuan, Wei Lan and Pengtai Huang (2024).
%     Inverse kinematics of 7-DOF Redundant Manipulator Based on Improved Dung Beetle Optimization
%     Algorithm. 2024 10th International Conference on Mechanical and Electronics Engineering (ICMEE),
%     pp. 245-251. DOI: 10.1109/ICMEE63700.2024.11025244
%   No volume or issue is printed. Page numbers are PDF pages 1-7 (printed pages 245-251).
%
%   Algorithm (Sec. III, pp.3-4; flow chart Fig. 3, p.4; fitness Sec. II.B, p.3):
%     N beetles in four roles in the ratio 6 : 6 : 7 : 11 (p.4): rolling, brood (female), young (small)
%     and thief beetles. Initial population by the Tent map (Eq. 7, Sec. III.A, p.3): X_0 ~ U(0, 1) per
%     joint, X_{m+1} = X_m / omega if X_m < omega, (1 - X_m) / (1 - omega) otherwise; the iterates
%     X_1 .. X_{N-1} are scaled to the joint ranges, q = q_min + X (q_max - q_min) (Fig. 3: angles
%     "within the operating range of each joint"). In iteration t = 1..T, with x_worst the worst position
%     of the current population and X* = X^l = p^t the best position found so far:
%       rolling (Eq. 8), delta ~ U(0, 1), ST ~ U(0.5, 1):
%         delta <  ST:  x_i(t+1) = x_i(t) + alpha k x_i(t-1) + b |x_i(t) - x_worst|          (8.1)
%         delta >= ST:  x_i(t+1) = omega_t x_i(t) + r1 sin(r2) [r3 p^t - x_i(t)]              (8.2)
%         r1      = (omega_max - omega_min)/2 cos(pi t / T) + (omega_max + omega_min)/2      (Eq. 9)
%         omega_t = omega_max - (omega_max - omega_min) t / T                                 (Eq. 10)
%       brood (Eqs. 11-13), R = 1 - t/T:
%         Lb* = max(X* (1 - R), Lb),  Ub* = min(X* (1 + R), Ub)
%         B_i(t+1) = X* + b1 (B_i(t) - Lb*) + b2 (B_i(t) - Ub*)
%       young (Eqs. 14-16):
%         Lb^l = max(X^l (1 - R), Lb),  Ub^l = min(X^l (1 + R), Ub)
%         x_i(t+1) = X^l + c1 (x_i(t) - Lb^l) + c2 (x_i(t) - Ub^l)
%       thief (Eq. 17):
%         x_i(t+1) = X^l + S g (|x_i(t) - X*| + |x_i(t) - X^l|)
%     Products with vectors and |.| act per joint; max and min act per joint, so Lb* > Ub* in a joint
%     where X* < 0 (R > 0), as the equations are printed. Every beetle is then bounded to the joint
%     limits, all N are evaluated and every position is replaced (Fig. 3, positions "updated in
%     parallel"): there is no acceptance test. The best position evaluated is recorded, used as X*, X^l
%     and p^t, and returned (Fig. 3, "the optimal solution").
%     Fitness (Eqs. 4-6, p.3), minimised:
%       f  = eta ||p_d - p_c|| + beta ||a_d - a_c||       (Eq. 4; a = approach vector, column 3 of R)
%       f1 = sum_k |theta_k - theta_k^ks|                 (Eq. 5; theta^ks = initial joint angle)
%       f2 = f + gamma f1                                 (Eq. 6)
%
%   Parameters:
%     N = 50, T = 300   the IK setting of the Table II runs (p.5, "with 300 iterations and 50
%                     particles"): N x T = 50 x 300 = 15,000 is nearer the 2000-FK budget than the Sec.
%                     IV.A setting 50 x 500 = 25,000 (p.4). The dimension D (7 on p.4) is robot.n. Inside
%                     the benchmark T is the budget horizon (Benchmark interface).
%     Roles 10 : 10 : 12 : 18   the ratio 6 : 6 : 7 : 11 (p.4) scaled to N = 50 (10, 10, 11.67, 18.33)
%                     and rounded by largest remainder. The paper does not say which role takes which
%                     share (checked pages 3-4); the ratio is read in the order in which Sec. III.B and
%                     Fig. 3 present the roles: rolling, brood, young, thief. Beetles 1-10 roll, 11-20
%                     are brood balls, 21-32 young beetles and 33-50 thieves, for the whole run (one
%                     assignment, p.4).
%     omega = 0.499   Tent chaos parameter: not stated in the paper (checked pages 3-4; p.3 only says that
%                     omega does not take 0.5); benchmark uses 0.499, next to the midpoint 0.5 of (0, 1),
%                     which the paper excludes.
%     k = 0.1         deflection coefficient, "k in (0, 0.2)" (p.3): midpoint of the stated range, constant.
%     b = 0.5         "a constant value between (0, 1)" (p.3): midpoint of the stated range.
%     alpha           +1 or -1 with probability 1/2 each, one draw per rolling move: not stated in the
%                     paper (checked pages 3-4; p.3: alpha simulates "natural factors that cause dung
%                     beetles to stray from their intended course"); benchmark uses a random sign, which
%                     keeps or reverses the k term and leaves its size to k.
%     ST, delta       "ST is a random number between 0.5 and 1", delta "a random constant value between
%                     (0, 1)" (p.3): one U(0.5, 1) and one U(0, 1) per rolling beetle and iteration; how
%                     often they are drawn is not stated.
%     omega_max = 0.9, omega_min = 0.4   ends of Eqs. 9-10: not stated in the paper (checked pages 3-5);
%                     benchmark uses the upper and lower PSO inertia weights of two other benchmark
%                     sources, Abdor-Sierra et al. (2022), p.8 (wu = 0.9, wl = 0.4), and Orbegoso Moreno et
%                     al. (2023), p.3 (between 0.4 and 0.9); Eq. 10 decreases linearly between the two.
%     r1, r2, r3      r1 from Eq. 9, one value per iteration; r2 ~ U(0, 2 pi) and r3 ~ U(-2, 2) (p.3),
%                     one scalar each per move ("a random number").
%     p^t, X*, X^l    p^t is "the optimal individual position variable component in the t iteration"
%                     (p.3), X* "the current local optimum location" and X^l "the population's current
%                     local optimum position" (p.4, Eqs. 11-12 and 14-15). The paper does not say whether
%                     "current" means the best of the population at iteration t or the best found up to
%                     iteration t (checked pages 3-4). Benchmark uses the best position found so far for
%                     all three: Fig. 3 obtains "the optimal solution" after every evaluation, and with the
%                     warm start of every waypoint this keeps q0 as the attractor until a better position
%                     is found. The other reading, the best of the current population, is not used. As
%                     X* = X^l, Eq. 17 adds 2 |x_i(t) - X^l|.
%     x_worst         "the worst position in the current population" (p.3).
%     x_i(t-1)        the beetle's position before its last move; at t = 1 the initial position itself,
%                     the nearest well-defined reading (no earlier position exists).
%     R = 1 - t/T     (p.4), down to 0 at t = T.
%     b1, b2          "two independent random vectors of size 1 x D" (p.4); the distribution is not
%                     stated (checked page 4); benchmark uses U(0, 1) per joint, the interval of c2.
%     c1, c2          c1 ~ N(0, 1), one scalar ("the random number c1"); c2 "a random vector that is
%                     included in (0, 1)", U(0, 1) per joint (p.4).
%     S = 0.5         "a constant value" (p.4), value not stated in the paper (checked pages 4-5);
%                     benchmark uses 0.5, under which S (|x - X*| + |x - X^l|) = |x - X^l| (X* = X^l): each
%                     thief joint moves by a standard normal times its distance from X^l.
%     g               "a random vector of size 1 x D with a normal distribution" (p.4): N(0, 1) per joint.
%     eta = beta = 1  weights of Eq. 4: not stated in the paper (checked pages 3-5); benchmark uses equal
%                     unit weights (1 cm of position error counts as much as a unit of ||a_d - a_c||).
%     gamma = 0.1     weight of Eq. 6: not stated in the paper (checked pages 3-5). f1 selects, among the
%                     zeros of Eq. 4, the one nearest theta^ks (p.3: Eq. 6 is adopted "to find smooth and
%                     continuous motion trajectories"), and the paper reports Eq. 6 values of 2.3128e-14
%                     to 4.2311e-12 (Table II, p.5), so its runs end at zeros of Eq. 4 (f <= f2). A zero
%                     of Eq. 4 is a minimum of Eq. 6 only if gamma < beta / c, where c >= 1 is the joint
%                     motion (sum of |dtheta|) per radian of turn of a_c: turning a_c by phi needs at
%                     least phi rad of joint motion and changes ||a_d - a_c|| by at most phi. With
%                     gamma = beta the approach term never pays for its joint motion. Benchmark uses
%                     gamma = beta / 10 = 0.1; the value is a choice within gamma < beta, not a derived one.
%     theta^ks        "the initial joint angle of the k joint" (p.3): q0, the warm start, in the benchmark.
%     Units           Table I (p.2) gives a and d in cm, so the position term of Eq. 4 is in cm (100 x the
%                     distance in m); ||a_d - a_c|| is unitless (unit vectors, at most 2); joint angles in
%                     rad (Table I; Figs. 5 and 7).
%     Stop            T iterations (p.4: "T_max is the maximum number of iterations") or Fig. 3's
%                     "The output conditions are met", which the paper does not define (checked pages
%                     3-5): see Benchmark interface.
%     Bounds          Fig. 3 bounds each beetle after the moves, with Lb and Ub "the optimization
%                     problem's lower and upper bounds" (p.4): every beetle clamped to [q_min, q_max].
%     Random draws    per iteration, beetles in index order: rolling ST, delta, then alpha (8.1) or r2, r3
%                     (8.2); brood b1 then b2 (D each); young c1 then c2 (D); thief g (D).
%
%   Benchmark interface:
%     Budget: the initial population costs N = 50 FK calls and every iteration N = 50 (every beetle
%       moves and is evaluated). T = ikb_generations(opts, 50, 50, 300) = 39 iterations, so
%       50 + 39 x 50 = 2000 FK per waypoint. A 40-member population would give
%       ikb_generations(opts, 40, 40, 300) = 49. The initial population and every iteration start only if
%       ikb_budget_left(opts) covers N. Outside the benchmark T = 300 (15,050 FK). No final FK check.
%       Every option field other than tol_pos and tol_rot is ignored.
%     Schedules: Eqs. 9-10 and R run over this T: with T = 39, r1 = 0.8992 at t = 1 and 0.4 at t = T,
%       omega_t = 0.8872 at t = 1 and 0.4 at t = T, R = 0.9744 at t = 1 and 0 at t = T.
%     Stop: the paper's output condition is not defined, and its task is the position and the approach
%       vector (Eq. 4; the target of Sec. IV.A, p.4, is (100, 200, 300, 90 deg, 60 deg)). The run stops
%       when the recorded best meets the part of the benchmark pose gate that this task covers: position
%       error < opts.tol_pos (m) and angle between a_d and a_c < opts.tol_rot (rad), 1e-5 each when a
%       field is absent, tested after the initial population and after every iteration; otherwise after
%       T iterations or when the budget cannot pay for the next one. info.converged is that test. The
%       rotation about the approach axis is not part of Eq. 4, so the full pose gate is met only by
%       chance.
%     Warm start: q0 (clamped to the limits) is beetle 1 (a rolling beetle) of the initial population
%       and is evaluated; it is theta^ks of Eq. 5, so f1(q0) = 0. q0 is X* = X^l = p^t of the iterations
%       and the returned solution until an evaluation has a strictly lower f2.
%     Robot and units: Eqs. 1-3 and Table I (the paper's KUKA LBR Med 7) are replaced by robot.fk with
%       D = robot.n joints; lengths in m, converted to cm inside Eq. 4; angles in rad.
%     Joint limits: clamp after the moves (Fig. 3); info.clamped counts the clamped coordinates.
%     Output: the recorded best; info.fitness (f2, Eq. 6), info.f (Eq. 4), info.f1 (Eq. 5), info.err_pos
%       (m), info.err_app (angle between the approach vectors, rad) and info.err_rot (rotation angle of
%       pose_error, rad, reported only), all from the pose stored at its evaluation.
%       solve_IDBO('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration horizon T = 39 instead of 300 (benchmark rule); Eqs. 9-10 and R run over it.
%     - Stop at the task part of the pose gate (the paper's output condition is not defined).
%     - q0 is beetle 1 of the initial population and theta^ks of Eq. 5 (warm start).
%     - Not stated in the paper, chosen as given under Parameters: omega, alpha, omega_max, omega_min, S,
%       eta, beta, gamma, X* = X^l = p^t as the best position found so far, the order of the roles in
%       the ratio, the rounding of the role sizes, the distribution of b1 and b2, the number of draws of
%       ST, delta, r2 and r3, x_i(t-1) at t = 1.
%     - k and b at the midpoints of their stated ranges.
%     - robot.fk instead of Eqs. 1-3 (the paper's arm).
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
        q_best = struct('params', idbo_params(), 'core', @idbo_core, 'roles', @idbo_roles, ...
                        'tent', @idbo_tent, 'rolling', @idbo_rolling, 'r1', @idbo_r1, ...
                        'omega', @idbo_omega, 'region', @idbo_region, 'brood', @idbo_brood, ...
                        'young', @idbo_young, 'thief', @idbo_thief, 'fitness', @idbo_fitness, ...
                        'eq6', @idbo_eq6, 'gate', @idbo_gate, 'tol', @idbo_tol);
        info = [];
        return
    end
    q = min(max(q0(:), robot.q_min(:)), robot.q_max(:));
    [q_best, info] = idbo_core(robot, T_d, q0, opts, idbo_params(), q);
end

% ==================================================================== main loop (Fig. 3)
function [q_best, info] = idbo_core(robot, T_d, q0, opts, P, qks)
% Tent initial population (q0 first), evaluation (Eq. 6 with theta^ks = qks); per iteration the four
% role updates from the population of iteration t and the best position found so far (Eqs. 8-17),
% clamp, evaluation of all N, parallel replacement, record of the best; stop at T, at the budget, or at
% the task part of the pose gate.
    t_start = tic;
    N  = P.N;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    n  = numel(lo);
    q0 = min(max(q0(:), lo), hi);
    qks = qks(:);
    T  = ikb_generations(opts, N, N, P.T);
    [tp, tr] = idbo_tol(opts);
    [nr, id] = idbo_roles(N, P.ratio);

    B = struct('f2', NaN, 'f', NaN, 'f1', NaN, 'q', q0, 'T', NaN(4), 'iter', NaN, 'nfk', 0);
    iters = 0;  stop = 'budget';  hist = zeros(1, 0);  f_init = NaN;
    br = zeros(2, 0);  ncl = zeros(1, 0);  xs = zeros(n, 0);  iw = zeros(1, 0);
    r1h = zeros(1, 0);  wh = zeros(1, 0);  Rh = zeros(1, 0);
    X = zeros(n, 0);  F = zeros(1, 0);  Xp = zeros(n, 0);

    if ikb_budget_left(opts) >= N
        X  = [q0, lo + idbo_tent(n, N - 1, P.omega) .* (hi - lo)];    % Eq. 7, q0 first
        Xp = X;                                                        % x_i(t-1) at t = 1
        [F, B] = evaluate(robot, T_d, X, qks, B, P, 0);
        f_init = B.f2;
        stop = 'T';
        hist = NaN(1, T);  br = zeros(2, T);  ncl = zeros(1, T);  xs = zeros(n, T);  iw = zeros(1, T);
        r1h = zeros(1, T);  wh = zeros(1, T);  Rh = zeros(1, T);
        if idbo_gate(B.T, T_d, tp, tr)
            stop = 'gate';
        else
            for t = 1:T
                if ikb_budget_left(opts) < N, stop = 'budget'; break; end
                xb = B.q;                                              % X* = X^l = p^t: best so far
                [~, iw(t)] = max(F);                                   % x_worst: current population
                xw = X(:, iw(t));
                xs(:, t) = xb;
                R  = 1 - t / T;
                Xn = X;
                for i = id.roll                                        % Eq. 8
                    [Xn(:, i), c] = idbo_rolling(X(:, i), Xp(:, i), xw, xb, t, T, P);
                    br(c, t) = br(c, t) + 1;
                end
                [Lbs, Ubs] = idbo_region(xb, R, lo, hi);               % Eqs. 11-12
                for i = id.brood                                       % Eq. 13
                    Xn(:, i) = idbo_brood(X(:, i), xb, Lbs, Ubs);
                end
                [Lbl, Ubl] = idbo_region(xb, R, lo, hi);               % Eqs. 14-15
                for i = id.young                                       % Eq. 16
                    Xn(:, i) = idbo_young(X(:, i), xb, Lbl, Ubl);
                end
                for i = id.thief                                       % Eq. 17
                    Xn(:, i) = idbo_thief(X(:, i), xb, xb, P.S);
                end
                ncl(t) = nnz(Xn < lo | Xn > hi);
                Xp = X;
                X  = min(max(Xn, lo), hi);                             % bounds (Fig. 3)
                [F, B] = evaluate(robot, T_d, X, qks, B, P, t);        % parallel replacement
                r1h(t) = idbo_r1(t, T, P);  wh(t) = idbo_omega(t, T, P);  Rh(t) = R;
                iters = t;
                hist(t) = B.f2;
                if idbo_gate(B.T, T_d, tp, tr), stop = 'gate'; break; end
            end
        end
        hist = hist(1:iters);  br = br(:, 1:iters);  ncl = ncl(1:iters);  xs = xs(:, 1:iters);
        iw = iw(1:iters);  r1h = r1h(1:iters);  wh = wh(1:iters);  Rh = Rh(1:iters);
    end

    ep = NaN;  ea = NaN;  er = NaN;  met = false;
    if ~isnan(B.f2)
        [met, ep, ea] = idbo_gate(B.T, T_d, tp, tr);
        [~, ~, e_r] = pose_error(T_d, B.T);
        er = norm(e_r);
    end

    q_best = B.q;
    info = struct();
    info.solver    = 'I-DBO';
    info.iters     = iters;                                            % iterations completed
    info.time      = toc(t_start);
    info.converged = met;                                              % task part of the pose gate
    info.fitness   = B.f2;                                             % Eq. 6
    info.fitness_init = f_init;                                        % best f2 of the initial population
    info.f         = B.f;                                              % Eq. 4 (cm, unitless)
    info.f1        = B.f1;                                             % Eq. 5 (rad)
    info.err_pos   = ep;                                               % m
    info.err_app   = ea;                                               % rad, approach vectors
    info.err_rot   = er;                                               % rad, reported only
    info.fk_calls  = B.nfk;
    info.jac_calls = 0;
    info.N         = N;
    info.T         = T;
    info.roles     = nr;                                               % rolling, brood, young, thief
    info.stop      = stop;                                             % 'gate', 'T', 'budget'
    info.gate      = [tp, tr];
    info.best_iter = B.iter;                                           % iteration of the returned beetle (0 = initial)
    info.history   = hist;                                             % recorded best f2 after iteration t
    info.branch    = br;                                               % rolling moves by Eq. 8.1 / 8.2
    info.clamped   = ncl;                                              % coordinates clamped in iteration t
    info.xstar     = xs;                                               % X* = X^l = p^t used in iteration t
    info.iworst    = iw;                                               % index of x_worst in iteration t
    info.r1        = r1h;                                              % Eq. 9 in iteration t
    info.omega     = wh;                                               % Eq. 10 in iteration t
    info.R         = Rh;                                               % 1 - t/T in iteration t
    info.final     = struct('X', X, 'F', F, 'Xprev', Xp);
    info.params    = P;
end

function [F, B] = evaluate(robot, T_d, X, qks, B, P, t)
% Eq. 6 for every beetle (one FK call each); the record is replaced only by a strictly lower f2.
    N = size(X, 2);
    F = zeros(1, N);
    for i = 1:N
        [F(i), Tq, f, f1] = idbo_fitness(robot, T_d, X(:, i), qks, P);
        B.nfk = B.nfk + 1;
        if isnan(B.f2) || F(i) < B.f2
            B.f2 = F(i);  B.f = f;  B.f1 = f1;  B.q = X(:, i);  B.T = Tq;  B.iter = t;
        end
    end
end

% ==================================================================== parameters and roles
function P = idbo_params()
% p.5: N = 50, T = 300; p.4: ratio 6 : 6 : 7 : 11; p.3: k in (0, 0.2), b in (0, 1), ST in [0.5, 1],
% r2 in [0, 2 pi], r3 in [-2, 2]; not stated: omega, omega_max, omega_min, S, eta, beta, gamma;
% position of Eq. 4 in cm (len_scale = 100 per m).
    P = struct('N', 50, 'T', 300, 'ratio', [6 6 7 11], 'omega', 0.499, 'k', 0.1, 'b', 0.5, ...
               'ST_lo', 0.5, 'wmax', 0.9, 'wmin', 0.4, 'S', 0.5, 'eta', 1, 'beta', 1, 'gamma', 0.1, ...
               'len_scale', 100);
end

function [nr, id] = idbo_roles(N, ratio)
% The ratio scaled to N, rounded by largest remainder (ties to the earlier role); index blocks in the
% order rolling, brood, young, thief.
    qt = ratio * N / sum(ratio);
    nr = floor(qt);
    [~, o] = sort(qt - nr, 'descend');
    left = N - sum(nr);
    nr(o(1:left)) = nr(o(1:left)) + 1;
    e = cumsum(nr);
    id = struct('roll', 1:e(1), 'brood', e(1) + 1:e(2), 'young', e(2) + 1:e(3), 'thief', e(3) + 1:e(4));
end

% ==================================================================== operators
function X = idbo_tent(n, m, w)
% Eq. 7: X_0 ~ U(0, 1) per joint (n = 0), then the iterates X_1 .. X_m, one column each.
    x = rand(n, 1);
    X = zeros(n, m);
    for j = 1:m
        for d = 1:n
            if x(d) < w
                x(d) = x(d) / w;
            else
                x(d) = (1 - x(d)) / (1 - w);
            end
        end
        X(:, j) = x;
    end
end

function r1 = idbo_r1(t, T, P)
% Eq. 9.
    r1 = (P.wmax - P.wmin) / 2 * cos(pi * t / T) + (P.wmax + P.wmin) / 2;
end

function w = idbo_omega(t, T, P)
% Eq. 10.
    w = P.wmax - (P.wmax - P.wmin) * t / T;
end

function [x, c] = idbo_rolling(x, xp, xw, p, t, T, P)
% Eq. 8 for one rolling beetle: ST ~ U(0.5, 1), delta ~ U(0, 1); delta < ST: Eq. 8.1 with alpha = +1
% or -1 (one draw), k and b constant; otherwise Eq. 8.2 with r1 (Eq. 9), omega_t (Eq. 10),
% r2 ~ U(0, 2 pi), r3 ~ U(-2, 2) and the guide p. c = 1 or 2 (branch).
    ST    = P.ST_lo + (1 - P.ST_lo) * rand;
    delta = rand;
    if delta < ST
        if rand < 0.5, a = 1; else, a = -1; end
        x = x + a * P.k * xp + P.b * abs(x - xw);                     % Eq. 8.1
        c = 1;
    else
        r1 = idbo_r1(t, T, P);
        w  = idbo_omega(t, T, P);
        r2 = 2 * pi * rand;
        r3 = -2 + 4 * rand;
        x  = w * x + r1 * sin(r2) * (r3 * p - x);                      % Eq. 8.2
        c  = 2;
    end
end

function [Lb, Ub] = idbo_region(Xc, R, lo, hi)
% Eqs. 11-12 (centre X*) and 14-15 (centre X^l): Lb = max(Xc (1 - R), lo), Ub = min(Xc (1 + R), hi).
    Lb = max(Xc * (1 - R), lo);
    Ub = min(Xc * (1 + R), hi);
end

function B = idbo_brood(B, Xs, Lb, Ub)
% Eq. 13: X* + b1 (B - Lb*) + b2 (B - Ub*), b1 and b2 independent U(0, 1) vectors.
    b1 = rand(numel(B), 1);
    b2 = rand(numel(B), 1);
    B  = Xs + b1 .* (B - Lb) + b2 .* (B - Ub);
end

function x = idbo_young(x, Xl, Lb, Ub)
% Eq. 16: X^l + c1 (x - Lb^l) + c2 (x - Ub^l), c1 ~ N(0, 1) scalar, c2 a U(0, 1) vector.
    c1 = randn;
    c2 = rand(numel(x), 1);
    x  = Xl + c1 * (x - Lb) + c2 .* (x - Ub);
end

function x = idbo_thief(x, Xs, Xl, S)
% Eq. 17: X^l + S g (|x - X*| + |x - X^l|), g a N(0, 1) vector, products and |.| per joint.
    g = randn(numel(x), 1);
    x = Xl + S * g .* (abs(x - Xs) + abs(x - Xl));
end

% ==================================================================== fitness (Eqs. 4-6) and stop
function [f2, Tq, f, f1] = idbo_fitness(robot, T_d, q, qks, P)
% Eq. 6 of the pose robot.fk(q) against T_d with theta^ks = qks; one FK call.
    Tq = robot.fk(q);
    [f2, f, f1] = idbo_eq6(Tq, T_d, q, qks, P);
end

function [f2, f, f1] = idbo_eq6(Tq, T_d, q, qks, P)
% Eq. 4: f = eta ||p_d - p_c|| (cm) + beta ||a_d - a_c||; Eq. 5: f1 = sum |theta_k - theta_k^ks|;
% Eq. 6: f2 = f + gamma f1.
    dp = T_d(1:3, 4) - Tq(1:3, 4);
    da = T_d(1:3, 3) - Tq(1:3, 3);
    f  = P.eta * (P.len_scale * sqrt(dp(1)^2 + dp(2)^2 + dp(3)^2)) ...
         + P.beta * sqrt(da(1)^2 + da(2)^2 + da(3)^2);
    f1 = 0;
    for k = 1:numel(q)
        f1 = f1 + abs(q(k) - qks(k));
    end
    f2 = f + P.gamma * f1;
end

function [hit, ep, ea] = idbo_gate(Tq, T_d, tp, tr)
% Task part of the pose gate: position error (m) < tp and angle between the approach vectors (rad) < tr.
    dp = T_d(1:3, 4) - Tq(1:3, 4);
    da = T_d(1:3, 3) - Tq(1:3, 3);
    ep = sqrt(dp(1)^2 + dp(2)^2 + dp(3)^2);
    ea = 2 * asin(min(1, sqrt(da(1)^2 + da(2)^2 + da(3)^2) / 2));
    hit = ep < tp && ea < tr;
end

function [tp, tr] = idbo_tol(opts)
% The benchmark tolerances, as in bench_run: tol_pos (m) and tol_rot (rad), 1e-5 when absent.
    tp = 1e-5;  tr = 1e-5;
    if isfield(opts, 'tol_pos'), tp = opts.tol_pos; end
    if isfield(opts, 'tol_rot'), tr = opts.tol_rot; end
end
