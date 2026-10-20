function [q_best, info] = solve_ISAMDE(robot, T_d, q0, opts)
% SOLVE_ISAMDE  Improved self-adaptive mutation differential evolution (ISAMDE) IK solver of Zhang et al. (2021), pp.2-5.
%
%   Source:
%     Qianqian Zhang, Daqing Wang and Lifu Gao (2021). Research on the inverse kinematics of manipulator using an
%     improved self-adaptive mutation differential evolution algorithm. International Journal of Advanced Robotic
%     Systems, May-June 2021: 1-11. DOI: 10.1177/17298814211014413
%   No volume or issue number is printed on the PDF (p.1). The paper's sections are unnumbered; references below give
%   the page (PDF page = printed page) and the equation, figure or table.
%
%   Algorithm (pp.2-4: Eqs. 1-8, Figs. 1-4; IK settings and termination p.5):
%     NP individuals of dimension D = robot.n. Initial individuals (Eq. 1, p.2):
%       x_ij(0) = x_ij^L + r1 x (x_ij^U - x_ij^L),   r1 uniform in [0, 1].
%     Each generation t = 0, 1, ..., T - 1 (Fig. 2: "Initial population, t=0", ..., "t=t+1") builds, for every
%     individual i = 1..NP in turn, from the population x(t) (Fig. 2, pp.3-4):
%       AMO = 1 - (t/T)^N                                                                       (p.3, below Eq. 6)
%       h_i(t) = x_p1j(t) + F(x_p2j(t) - x_p3j(t)),   rand <= AMO     (DE/rand/1/bin, Eq. 2)
%                x_bj(t) + F(x_p1j(t) - x_p2j(t)),    otherwise       (DE/best/1/bin, Eq. 3)    (Eq. 6)
%       F = AF = eps(t)^2*2*10^-6 + 0.5,   eps(t) < E0
%                1,                         otherwise                                           (Eq. 7, Fig. 4)
%       boundary: "random initialization is used after the mutated individual crosses the boundary" (p.5)
%       v_ij(t + 1) = h_ij(t + 1), r1 <= C or j = j_rand;  x_ij(t), otherwise                  (Eq. 4)
%                     (j_rand "an integer randomly selected in [1, D]", p.3)
%       x_i(t + 1) = v_i(t + 1), f(v_i(t + 1)) < f(x_i(t));  x_i(t), f(v_i(t + 1)) >= f(x_i(t)) (Eq. 5)
%     p1, p2, p3 are "the serial number of different individuals randomly selected from the population" and x_bj(t)
%     "is the best individual of the current group" (p.2).
%     IK objective (p.4, unnumbered, and Eq. 8):
%       F_obj = omega_P * E_P + omega_O * E_O,   E_P = ||P_t - P_c||,   E_O = ||R_t - R_c||,   omega_P = 1,
%       omega_O = k sqrt(lambda * E_P)/pi,   E_P >= 0.1
%                 k sqrt(lambda * 0.1)/pi,   otherwise                                          (Eq. 8)
%     with k "the coefficient related to the rod length" and lambda "the sum of the length of the connecting rod"
%     (p.4, defined with the weight of Tabandeh et al., omega_o = k sqrt(lambda.P_d)/pi, that Eq. 8 modifies). The
%     lower branch keeps the orientation weight from vanishing, "to ensure that the two types of errors are always
%     at the same magnitude" (p.4).
%     Termination (p.5): "meet the given accuracy (position:10^-2 mm and orientation: 5 x 10^-4 rad) or reach the
%     maximum number of iterations".
%
%   Parameters:
%     NP = 70, T = 600   p.5, IK runs: "F0 = 0.5, C = 0.9, N = 0.9, P = 70, Nmax = 600"; the only IK setting
%                   reported, N x T = 70 x 600 = 42,000 (P = 140 in Table 4, p.6, is the benchmark-function
%                   setting). Inside the benchmark T is the budget horizon (Benchmark interface).
%     N = 0.9       power exponent of AMO, p.5 (N = 0.8 in Table 4 is the benchmark-function setting).
%     C = 0.9       crossover probability, p.5.
%     E0 = 500      Eq. 7, p.4: "In this article, E0 is 500."
%     F0 = 0.5      listed on p.5 as "the original scaling factor"; its role in ISAMDE is not stated in the paper
%                   (checked pages 2-5), where F is AF of Eq. 7 (lower value 0.5). F0 is not used.
%     k = 20        p.4: "In the simulation analysis of the 6 degree-of-freedom (DOF) manipulator, k is 20."; k for the
%                   7-DOF arm is not stated in the paper (checked pages 4-9); 20, the only value given, is used for
%                   every robot.
%     lambda        "the sum of the length of the connecting rod" (p.4); no value is given. Benchmark uses the sum of
%                   |a_i| + |d_i| over the robot's D-H table, in mm: UR5 1192.509, Panda 1393, Synthetic9 1532,
%                   Synthetic11 1865, Synthetic13 2198, IRB120 1004 (the paper's 6-DOF arm, Table 1, p.5: 1950).
%     eps(t)        "the error value under the current iteration number" (p.4), "the current generation error value"
%                   (p.9); which error is not stated in the paper (checked pages 3-5 and 9). ISAMDE with Eq. 7 also
%                   runs on the benchmark functions (Table 4; Fig. 8, axis "Error"), where the only error is the
%                   objective value; benchmark uses eps(t) = the lowest F_obj in x(t), the objective value of x_b(t).
%     E_O           "calculated by the Euclidean distance between R_t and R_c" (p.4): the Frobenius norm of
%                   R_t - R_c (the nine elements r11-r33 of Fig. 5(b) as one vector).
%     omega_O       per individual, from the E_P of the pose being evaluated (Eq. 8 is written in E_P).
%     Units         mm and rad: Tables 1-2 (p.5) in mm, E_P range "(-1500, 1500)" (p.4), accuracy 10^-2 mm (p.5),
%                   axes "ErrorP<mm>" (Figs. 9-12); p.6 gives metres only for the Newton-Raphson comparison ("the
%                   units of E_P and E_O are meter and radians"). E_P, lambda and the 0.1 of Eq. 8 are in mm,
%                   E_P = 1000 ||p - p_d|| with the benchmark's metres; eps(t) and E0 in the units of F_obj.
%     Indices       p1, p2, p3 mutually distinct, uniform over all NP individuals; exclusion of the target i is not
%                   stated in the paper (checked pages 2-3), so i is not excluded.
%     Boundary      whether the whole mutated individual or its crossing variables are re-initialised is not stated
%                   in the paper (checked pages 2 and 5); each variable j of h_i outside [x^L, x^U] is redrawn by
%                   Eq. 1 (written per variable x_ij), one r1 per variable, before the crossover.
%     Strategy      one rand per individual and generation (h_i(t), Eq. 6; Fig. 2 "Determine Mutation Strategy"
%                   inside the individual loop), compared with AMO by "rand <= AMO" as printed.
%     Generation    Eqs. 2-6 take x(t) and x_b(t); the trial vectors of generation t form x(t + 1) (Eq. 5), so x_b
%                   and eps(t) are constant during a generation and AF is one value per generation.
%     Ties          Eq. 5 keeps x_i(t) when f(v) >= f(x) (as printed); x_b is the first of equal best individuals.
%     Random draws  initial individuals 2..NP: rand(D, NP - 1); per individual i in index order: rand (strategy),
%                   randperm(NP, 3) (DE/rand/1) or randperm(NP, 2) (DE/best/1), one rand per crossing variable in
%                   index order, rand(D, 1) (r1 of Eq. 4), randi(D) (j_rand).
%     Stop          the p.5 accuracy belongs to the paper's example tasks and is replaced by the benchmark pose gate:
%                   x_b's position error < opts.tol_pos (m) and rotation angle < opts.tol_rot (rad) (pose_error),
%                   tested on x(0) and after every generation; info.converged = the gate.
%
%   Benchmark interface:
%     Budget: x(0) costs NP = 70 FK calls and every generation 70 (one per trial vector; eps(t), x_b(t) and the
%       gate use stored values). T = ikb_generations(opts, 70, 70, 600) = 27 generations, so 70 + 27 x 70 = 1960 FK
%       per waypoint unless the gate is met; the remaining 40 do not cover a generation. A 40-member population
%       would give ikb_generations(opts, 40, 40, 600) = 49. x(0) and every generation start only if
%       ikb_budget_left(opts) covers them. Outside the benchmark T = 600 (42,070 FK). No final FK check. Fields of
%       opts other than tol_pos and tol_rot (max_iter, pop_size, ...) are ignored.
%     Schedule: AMO runs over this T; generation g = 1..27 uses t = g - 1, so AMO falls from 1 (g = 1) to
%       1 - (26/27)^0.9 = 0.0334 (g = 27).
%     Warm start: q0 (clamped to the limits) is individual 1 of x(0) and is evaluated; Eq. 5 replaces it only by a
%       strictly better trial vector, so it enters the first selection.
%     Units: benchmark lengths in m, converted to mm inside F_obj; rotation by the Frobenius distance in F_obj and by
%       the geodesic angle in the gate and info.err_rot.
%     Joint limits: x^L = robot.q_min, x^U = robot.q_max; mutated variables outside are re-initialised (above).
%     Output: x_b of the last population (the lowest F_obj evaluated); info.fitness (F_obj, mm), info.err_pos (m),
%       info.err_rot (rad, pose_error) and info.E_O from its stored pose. solve_ISAMDE('ops') returns handles to the
%       operators for unit tests.
%
%   Deviations from the paper:
%     - Generation horizon T = 27 instead of 600 (benchmark rule); AMO runs over it.
%     - q0 is individual 1 of the initial population (warm start).
%     - Stop at the benchmark pose gate instead of the p.5 accuracy of the paper's example tasks.
%     - Not stated in the paper, chosen as given under Parameters: eps(t), lambda, k beyond 6 DOF, the matrix norm of
%       E_O, omega_O per individual, the index draws, the boundary granularity, the order of the random draws.
%     - robot.fk (standard D-H, metres) instead of the paper's arms (Tables 1-2, mm); lengths converted to mm.
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
        q_best = struct('params', isamde_params(), 'core', @isamde_core, 'amo', @isamde_amo, 'af', @isamde_af, ...
                        'mutant', @isamde_mutant, 'bound', @isamde_bound, 'cross', @isamde_cross, ...
                        'select', @isamde_select, 'omega', @isamde_omega, 'fobj', @isamde_fobj, ...
                        'fitness', @isamde_fitness, 'lambda', @isamde_lambda, 'gate', @isamde_gate);
        info = [];
        return
    end
    [q_best, info] = isamde_core(robot, T_d, q0, opts, isamde_params());
end

% ==================================================================== main loop (Fig. 2)
function [q_best, info] = isamde_core(robot, T_d, q0, opts, P)
% x(0) by Eq. 1 with q0 first; per generation AMO, Eq. 7, then per individual Eq. 6, the p.5 boundary rule, Eq. 4 and
% Eq. 5 into x(t + 1); stop at the pose gate, at T generations or at the budget.
    t_start = tic;
    NP  = P.NP;
    lo  = robot.q_min(:);
    hi  = robot.q_max(:);
    n   = numel(lo);
    q0  = min(max(q0(:), lo), hi);
    T   = ikb_generations(opts, NP, NP, P.T);
    lam = isamde_lambda(robot, P);
    [tp, tr] = isamde_tol(opts);

    X = zeros(n, 0);  F = zeros(1, 0);  XT = zeros(4, 4, 0);  b = 0;
    fk = 0;  iters = 0;  f_init = NaN;  converged = false;
    hist = zeros(1, 0);  amo = hist;  af = hist;  ep = hist;  nb = hist;  nr = hist;  nc = hist;

    if ikb_budget_left(opts) >= NP
        X  = [q0, lo + rand(n, NP - 1) .* (hi - lo)];                        % Eq. 1, q0 first
        F  = zeros(1, NP);
        XT = zeros(4, 4, NP);
        for i = 1:NP
            [F(i), XT(:, :, i)] = isamde_fitness(robot, T_d, X(:, i), lam, P);
            fk = fk + 1;
        end
        [f_init, b] = min(F);                                               % x_b(0), first of equal ones
        converged = isamde_gate(T_d, XT(:, :, b), tp, tr);
        hist = NaN(1, T);  amo = hist;  af = hist;  ep = hist;  nb = zeros(1, T);  nr = nb;  nc = nb;
        for g = 1:T
            if converged || ikb_budget_left(opts) < NP, break; end
            t   = g - 1;                                                    % Fig. 2: t = 0 for the first generation
            A   = isamde_amo(t, T, P);                                      % AMO
            e   = F(b);                                                     % eps(t)
            AF  = isamde_af(e, P);                                          % Eq. 7
            Xn  = X;  Fn = F;  XTn = XT;
            for i = 1:NP
                [h, s] = isamde_mutant(X, b, A, AF);                        % Eq. 6
                [h, c] = isamde_bound(h, lo, hi);                           % p.5
                v      = isamde_cross(X(:, i), h, P.C);                     % Eq. 4
                [fv, Tv] = isamde_fitness(robot, T_d, v, lam, P);
                fk = fk + 1;
                [Xn(:, i), Fn(i), XTn(:, :, i), r] = isamde_select(X(:, i), F(i), XT(:, :, i), v, fv, Tv);   % Eq. 5
                nb(g) = nb(g) + ~s;  nr(g) = nr(g) + r;  nc(g) = nc(g) + c;
            end
            X = Xn;  F = Fn;  XT = XTn;                                     % x(t + 1)
            [~, b] = min(F);
            iters = g;
            hist(g) = F(b);  amo(g) = A;  af(g) = AF;  ep(g) = e;
            converged = isamde_gate(T_d, XT(:, :, b), tp, tr);
        end
        hist = hist(1:iters);  amo = amo(1:iters);  af = af(1:iters);  ep = ep(1:iters);
        nb = nb(1:iters);  nr = nr(1:iters);  nc = nc(1:iters);
    end

    q_best = q0;  fb = NaN;  e1n = NaN;  e2n = NaN;  Eo = NaN;
    if b > 0
        q_best = X(:, b);
        fb = F(b);
        [~, ~, Eo] = isamde_fobj(XT(:, :, b), T_d, lam, P);
        [~, e1, e2] = pose_error(T_d, XT(:, :, b));
        e1n = norm(e1);  e2n = norm(e2);
    end

    info = struct();
    info.solver    = 'ISAMDE';
    info.iters     = iters;                                                 % generations completed
    info.time      = toc(t_start);
    info.converged = converged;                                             % pose gate (Stop)
    info.fitness   = fb;                                                    % F_obj of x_b, mm
    info.fitness_init = f_init;                                             % lowest F_obj of x(0)
    info.err_pos   = e1n;                                                   % m
    info.err_rot   = e2n;                                                   % rad
    info.E_O       = Eo;                                                    % ||R_t - R_c||_F of x_b
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.NP        = NP;
    info.T         = T;
    info.lambda    = lam;                                                   % mm
    info.history   = hist;                                                  % F_obj of x_b(t + 1)
    info.AMO       = amo;                                                   % AMO of generation g
    info.AF        = af;                                                    % Eq. 7 of generation g
    info.eps       = ep;                                                    % eps(t) of generation g
    info.n_best    = nb;                                                    % DE/best/1 mutants in generation g
    info.replaced  = nr;                                                    % Eq. 5 replacements in generation g
    info.reinit    = nc;                                                    % variables re-initialised (p.5)
    info.final     = struct('X', X, 'F', F);                                % last population
    info.params    = P;
end

% ==================================================================== parameters and operators
function P = isamde_params()
% p.5: NP = 70, Nmax = 600, N = 0.9, C = 0.9; Eq. 7 (p.4): E0 = 500, 2*10^-6, 0.5; p.4 and Eq. 8: omega_P = 1, k = 20,
% E_P threshold 0.1; positions in mm (len_scale = 1000 per m).
    P = struct('NP', 70, 'T', 600, 'N', 0.9, 'C', 0.9, 'E0', 500, 'a_af', 2e-6, 'b_af', 0.5, ...
               'wP', 1, 'k', 20, 'Ep0', 0.1, 'len_scale', 1000);
end

function A = isamde_amo(t, T, P)
% AMO = 1 - (t/T)^N (p.3).
    A = 1 - (t / T)^P.N;
end

function AF = isamde_af(e, P)
% Eq. 7: AF = eps^2 * 2*10^-6 + 0.5 if eps < E0, 1 otherwise.
    if e < P.E0
        AF = e^2 * P.a_af + P.b_af;
    else
        AF = 1;
    end
end

function [h, s, p] = isamde_mutant(X, b, A, F)
% Eq. 6: one rand against AMO; DE/rand/1 with three, DE/best/1 with two mutually distinct individuals of X.
    NP = size(X, 2);
    s  = rand <= A;
    if s
        p = randperm(NP, 3);
        h = X(:, p(1)) + F * (X(:, p(2)) - X(:, p(3)));
    else
        p = randperm(NP, 2);
        h = X(:, b) + F * (X(:, p(1)) - X(:, p(2)));
    end
end

function [h, c] = isamde_bound(h, lo, hi)
% p.5: a variable of the mutated individual outside [lo, hi] is re-initialised by Eq. 1 (one rand per variable,
% in index order); c = number of re-initialised variables.
    c = 0;
    for j = 1:numel(h)
        if h(j) < lo(j) || h(j) > hi(j)
            h(j) = lo(j) + rand * (hi(j) - lo(j));
            c = c + 1;
        end
    end
end

function [v, r, jr] = isamde_cross(x, h, C)
% Eq. 4: v_j = h_j if r1_j <= C or j = j_rand, x_j otherwise; r1 = rand(D, 1), then j_rand = randi(D).
    n  = numel(x);
    r  = rand(n, 1);
    jr = randi(n);
    take = r <= C;
    take(jr) = true;
    v = x;
    v(take) = h(take);
end

function [x, f, Tx, rep] = isamde_select(x, f, Tx, v, fv, Tv)
% Eq. 5: v replaces x if f(v) < f(x); x is kept if f(v) >= f(x).
    rep = fv < f;
    if rep
        x = v;  f = fv;  Tx = Tv;
    end
end

function w = isamde_omega(Ep, lam, P)
% Eq. 8: omega_O = k sqrt(lambda * E_P)/pi if E_P >= 0.1, k sqrt(lambda * 0.1)/pi otherwise (mm).
    if Ep >= P.Ep0
        w = P.k * sqrt(lam * Ep) / pi;
    else
        w = P.k * sqrt(lam * P.Ep0) / pi;
    end
end

function [f, Ep, Eo, w] = isamde_fobj(T, T_d, lam, P)
% F_obj = omega_P E_P + omega_O E_O (p.4): E_P = ||P_t - P_c|| in mm, E_O = ||R_t - R_c|| (Frobenius: the nine
% squared element differences summed in column order), omega_O by Eq. 8.
    dp = T_d(1:3, 4) - T(1:3, 4);
    Ep = P.len_scale * sqrt(dp(1)^2 + dp(2)^2 + dp(3)^2);
    dR = T_d(1:3, 1:3) - T(1:3, 1:3);
    s  = 0;
    for m = 1:9
        s = s + dR(m)^2;
    end
    Eo = sqrt(s);
    w  = isamde_omega(Ep, lam, P);
    f  = P.wP * Ep + w * Eo;
end

function [f, T] = isamde_fitness(robot, T_d, q, lam, P)
% F_obj of the pose of q; one FK call.
    T = robot.fk(q);
    f = isamde_fobj(T, T_d, lam, P);
end

function lam = isamde_lambda(robot, P)
% lambda (p.4) = sum over the D-H table of |a_i| + |d_i| (joint by joint), in mm.
    s = 0;
    for i = 1:numel(robot.dh.a)
        s = s + (abs(robot.dh.a(i)) + abs(robot.dh.d(i)));
    end
    lam = P.len_scale * s;
end

function [tp, tr] = isamde_tol(opts)
% Pose gate tolerances (m, rad); default_opts values when absent.
    tp = 1e-6;
    tr = 1e-6;
    if isfield(opts, 'tol_pos'), tp = opts.tol_pos; end
    if isfield(opts, 'tol_rot'), tr = opts.tol_rot; end
end

function ok = isamde_gate(T_d, Tb, tp, tr)
% Benchmark pose gate on the pose Tb of x_b: position error < tp and rotation angle < tr (pose_error).
    [~, e1, e2] = pose_error(T_d, Tb);
    ok = norm(e1) < tp && norm(e2) < tr;
end
