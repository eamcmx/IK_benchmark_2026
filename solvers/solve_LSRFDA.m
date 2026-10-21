function [q_best, info] = solve_LSRFDA(robot, T_d, q0, opts)
% SOLVE_LSRFDA  Improved flow direction algorithm (LSRFDA) IK solver of Wang et al. (2025), Secs. 2.2-3.2, pp.5-8.
%
%   Source:
%     Yaping Wang, Shengbiao Li, Di Xu, Dekang Hou and Yuqi Fan (2025). An inverse kinematics solution method based on
%     improved flow direction algorithm. International Journal on Interactive Design and Manufacturing (IJIDeM)
%     19: 7913-7923. DOI: 10.1007/s12008-025-02341-3
%   No issue number is printed on the PDF (p.1). The paper does not expand the name (p.3: "an improved FDA (LSRFDA)");
%   its two improvements are the Levy flight strategy in the global search and the self-updating method in the local
%   search (pp.1, 3). Page numbers below are PDF pages 1-11 (printed page = PDF page + 7912). Sections: 2.2 fitness
%   (p.5), 3.1 FDA (pp.5-6), 3.2 improvements (pp.6-7), 4 experiments (pp.7-10).
%
%   Algorithm (Eqs. 13-27, the LSRFDA steps 1-8 of p.7 and the flowchart Fig. 5 of p.8; fitness Eqs. 10-12, p.5):
%     N flows (water streams); the position Flow_X(i) of flow i is a joint vector.
%     Step 1 (Eq. 13): Flow_X(i) = lb + rand * (ub - lb); then the fitness of every flow (Fig. 5).
%     Each iteration iter = 1..Max_Iter, for every flow i, with Best_X "the historically optimal position" (p.6):
%       Step 3 (Eqs. 24 and 14): "Create locations of neighboring streams and update Lambda by formula (24)"
%         Lambda_new = (rand * Xrand - rand * Flow_X(i)) * ||Best_X - Flow_X(i)|| * s_L                      (Eq. 24)
%         Neighbor_X(j) = Flow_X(i) + randn * Lambda,   j = 1..n_nb                                          (Eq. 14)
%       Step 4: the fitness of every neighbour; "If the objective function value of the optimal adjacent flow is
%         better than the objective function value of the current flow, get into 5; if not, move to 6" (p.7).
%       Step 5 (Eqs. 18, 25, 26), j = the best neighbour:
%         S_o(i, j) = (Flow_fitness(i) - Neighbor_fitness(j)) / ||Flow_X(i) - Neighbor_X(j)||               (Eq. 18)
%         upsilon_new = s_L * s0                                                                             (Eq. 25)
%         Flow_newX(i) = Flow_X(i) + upsilon_new (Flow_X(i) - Neighbor_X(j)) / ||Flow_X(i) - Neighbor_X(j)|| (Eq. 26)
%       Step 6 (Eq. 27; Fig. 5: "Generate a random integer r"):
%         Flow_newX(i) = s_L * Best_X                                  if Flow_fitness(r) < Flow_fitness(i)
%                        Flow_X(i) + s_L * (Best_X - Flow_X(i))        otherwise                              (Eq. 27)
%       Fig. 5: the fitness of the new position; it replaces Flow_X(i) if it is better ("Is there a better flow of
%         water?"). Step 7: "If a better solution exists, update the global solution". Step 8: stop at iter = Max_Iter.
%     Levy step (Eqs. 22-23, p.6):
%       s_L = u / |epsilon|^(1/tau),   u ~ N(0, sigma_u^2),   epsilon ~ N(0, sigma_eps^2),   sigma_eps = 1     (Eq. 22)
%       sigma_u = { Gamma(1 + tau) sin(pi tau/2) / (Gamma[(1 + tau)/2] tau 2^(tau-1/2)) }^(1/tau)              (Eq. 23)
%     Fitness (Sec. 2.2, p.5):
%       f1 = ||P_P - P*_P||, "The norm of the position error is defined as f1"                                (Eq. 10)
%       f2 = sum_{i=1..6} z x [omega_i (theta_i(k) - theta*_i(k))]^2                                          (Eq. 11)
%       min eta = min(alpha f1 + beta f2)                                                                      (Eq. 12)
%     The base-FDA forms that LSRFDA replaces are not used: Lambda = M * N * W (Eq. 15) with the weight W of Eq. 16,
%     the velocity upsilon = randn * S_o (Eq. 17), the move of Eq. 19 and the random-runoff rule of Eq. 20. W enters
%     only Eq. 15, which step 3 replaces by Eq. 24 (W replaced by s_L), so no quantity of LSRFDA varies with iter.
%
%   Parameters:
%     N = 40          population size: not stated in the paper (checked pages 1-11; p.5 "generates an initial
%                     population", p.7 "1: Initialize population;"). Benchmark uses 40, its reference population size
%                     for papers that give none.
%     Max_Iter = 200  defined on p.6 without a value (checked pages 1-11); 200 is the end of the iteration axis of
%                     Fig. 6 (p.9) and of the graphical abstract (p.2). N x Max_Iter = 40 x 200 = 8000. The paper
%                     reports no IK setting, so there is no reported N x T to choose from. Used outside the benchmark
%                     only; inside, Max_Iter is the budget horizon (Benchmark interface).
%     n_nb = 2        neighbours per flow: not stated in the paper (checked pages 1-11; p.5: "Assuming that a number of
%                     neighborhood cells exist around each stream"). Fig. 5 computes "the fitness function for each
%                     neighboring stream to select the best stream" and step 4 compares "the optimal adjacent flow";
%                     2 is the smallest number for which that selection is a choice.
%     tau = 1.5       p.6: "1 < tau < 3"; no value is given. At the midpoint tau = 2 the factor sin(pi tau/2) of Eq. 23
%                     is 0, so sigma_u = 0 and every Levy step is 0; for 2 < tau < 3 the factor is negative and Eq. 23
%                     has no real value. Eq. 23 is well defined for 1 < tau < 2 only; benchmark uses its midpoint, 1.5.
%     sigma_u = 0.4925525544   Eq. 23 at tau = 1.5. The exponent of 2 is printed "tau-1/2" (a superscript without
%                     brackets; the same characters in the text layer) and is read as printed, tau - 1/2. The grouping
%                     (tau - 1)/2 would give sigma_u = 0.6965745026.
%     sigma_eps = 1   p.6, after Eq. 22.
%     alpha = 1, beta = 10^-10   p.5, after Eq. 12: "alpha = 1, beta = 10^-10 are set".
%     z = 0.001, omega = [3 2 1.5 1 1 1]   p.5, after Eq. 11. Eq. 11 sums the 6 joints of the paper's arm (i = 1..6);
%                     joints 7-13 of the redundant robots are weighted 1.
%     theta*(k)       p.5: "the transformation angle of each joint of the industrial robot is theta_i(k) - theta*_i(k)";
%                     Sec. 4 (p.7): the robot "moves from an initial joint angle q [0000000]" (printed so; 6 joints).
%                     theta* is the joint vector the robot moves from, here q0; theta(k) is the evaluated joint vector.
%     Units           metres and radians: Table 1 (p.4) gives d_i and a_(i-1) in m and Table 2 (p.8) "Position
%                     error/m". The joint-angle unit of Eq. 11 is not stated in the paper (checked pages 4-7; Table 1
%                     ranges in degrees, q_goal on p.7 without unit); radians are used.
%     rand, randn, s_L   rand is "the generation of random numbers between 0 and 1" (p.5, Eq. 13): one U(0, 1) number
%                     per joint, in Eqs. 13 and 24. randn is "a random number with mean 0, standard deviation 1"
%                     (p.5, Eq. 14): one N(0, 1) number per neighbour. s_L is "the length of the Levy flight" (p.6):
%                     one number per use, drawn afresh in Eq. 24, in Eq. 25 and in Eq. 27.
%     Xrand           not defined in the paper (checked pages 5-7); p.6 calls M = rand * Xrand - rand * Flow_X(i) of
%                     Eq. 15 "the random position of the i th stream approximation". Xrand is a random position drawn
%                     by Eq. 13, lb + rand * (ub - lb), afresh for every flow and iteration.
%     Norm of Eq. 18  printed ||Flow_x(i, d) - Neighbor_x(j, d)|| with S_o(i, j, d); read as the Euclidean distance
%                     between the two positions over their d coordinates, the norm of Eq. 26 (||Flow_X(i) -
%                     Neighbor_X(j)||) and of Eq. 15 ("N denotes the Euclidean distance", p.6): one slope per
%                     neighbour, "the slope vector between the neighboring streams and the current stream location"
%                     (p.6). Per coordinate, the slope would be infinite wherever flow and neighbour share a coordinate
%                     (both clamped to the same joint limit), and Eq. 26 undefined there.
%     s0 (Eq. 25)     the slope S_o of Eq. 18 to the best neighbour (Fig. 5: "select the best stream").
%     r (Eq. 27)      uniform over the flows other than i: "the rth stream" is compared with "the ith stream" (Fig. 5),
%                     a test that r = i can never pass.
%     First branch of Eq. 27   s_L * Best_X as printed (Fig. 5 labels this branch "Transferring the flow to the rth
%                     best inverse solution", a form Eq. 27 does not write).
%     Iteration       synchronous: steps 3-6 of every flow use Best_X and the fitness values at the start of the
%                     iteration (steps 3-7 each act on all streams); accepted positions and Best_X (step 7) are updated
%                     after all N flows.
%     Ties            strict "<" in step 4, in Eq. 27 and in the acceptance; the best neighbour and the initial Best_X
%                     are the first of equal ones; Best_X changes only to a strictly better flow (step 7).
%     Boundaries      handling not stated in the paper (checked pages 5-8; lb and ub appear in Eq. 13 only); every
%                     neighbour and every new position is clamped to [q_min, q_max] before its evaluation.
%     Stop            step 8 only (iter = Max_Iter); the paper states no accuracy test (checked pages 5-10), its task
%                     is position only, and the population runs to the budget. info.converged = false always (the paper
%                     has no convergence criterion).
%     Random draws    flows 2..N: rand(n, N - 1). Per iteration and flow in index order: Xrand (rand(n, 1)), the two
%                     rand of Eq. 24 (rand(n, 1) each), s_L of Eq. 24 (u = sigma_u randn, then epsilon = randn), one
%                     randn per neighbour; then for step 5 s_L (u, epsilon), or for step 6 r (randi(N - 1), shifted
%                     past i) and s_L (u, epsilon).
%
%   Benchmark interface:
%     Budget: the initial population costs N = 40 FK calls and every iteration N x (n_nb + 1) = 120 (two neighbours and
%       one new position per flow). Max_Iter = ikb_generations(opts, 40, 120, 200) = 16 iterations per waypoint, so
%       40 + 16 x 120 = 1960 FK per waypoint; the remaining 40 do not cover an iteration. The population is 40, so a
%       40-member population gives the same 16. The initial population and every iteration start only if
%       ikb_budget_left(opts) covers them. Outside the benchmark Max_Iter = 200 (40 + 200 x 120 = 24,040 FK). No final
%       FK check. No field of opts is read here (max_iter, pop_size, tol_pos, tol_rot, ...); the budget comes through
%       ikb_budget_left and ikb_generations.
%     Schedule: none (W of Eq. 16 is not used; no other quantity depends on iter or Max_Iter).
%     Warm start: q0 (clamped to the limits) is flow 1 of the initial population and is evaluated; a flow is replaced
%       only by a strictly better position, so q0 enters the first selection and is kept unless improved. q0 is also
%       theta* of Eq. 11.
%     Units: f1 in metres, f2 in radians (above).
%     Joint limits: lb = robot.q_min, ub = robot.q_max; clamp (above).
%     Output: Best_X; info.fitness (eta), info.err_pos (m) and info.err_rot (rad, pose_error) from the pose stored at
%       its evaluation. solve_LSRFDA('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Max_Iter = 16 inside the benchmark (budget horizon) instead of 200 (Fig. 6 axis).
%     - q0 is flow 1 of the initial population (warm start) and theta* of Eq. 11.
%     - Not stated in the paper, chosen as given under Parameters: N, n_nb, tau, Xrand, the joint-angle unit, the
%       boundary clamp, the order of the draws; omega_i = 1 beyond the sixth joint.
%     - Readings of printed ambiguities, as given under Parameters: the exponent of Eq. 23, the norm of Eq. 18, one
%       random number per joint (rand) or per use (randn, s_L), r other than i, synchronous iterations.
%     - robot.fk instead of Eqs. 1-9 and Table 1 (the paper's Fanuc r2000ic 165F).
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
        q_best = struct('params', lsr_params(), 'core', @lsr_core, 'sigma_u', @lsr_sigma_u, 'levy', @lsr_levy, ...
                        'xrand', @lsr_xrand, 'lambda', @lsr_lambda, 'neighbours', @lsr_neighbours, ...
                        'slope', @lsr_slope, 'move', @lsr_move, 'pick_r', @lsr_pick_r, 'selfupdate', @lsr_selfupdate, ...
                        'accept', @lsr_accept, 'f1', @lsr_f1, 'f2', @lsr_f2, 'eta', @lsr_eta, 'fitness', @lsr_fitness, ...
                        'omega', @lsr_omega, 'clamp', @lsr_clamp, 'vnorm', @lsr_vnorm);
        info = [];
        return
    end
    [q_best, info] = lsr_core(robot, T_d, q0, opts, lsr_params());
end

% ==================================================================== main loop (steps 1-8, Fig. 5)
function [q_best, info] = lsr_core(robot, T_d, q0, opts, P)
% Step 1 with q0 first; per iteration, for every flow: step 3 (Eqs. 24, 14), step 4, step 5 (Eqs. 18, 25, 26) or
% step 6 (Eq. 27), clamp, evaluation, acceptance into the next population; step 7 after all flows; stop at Max_Iter
% or at the budget.
    t_start = tic;
    N   = P.N;
    nb  = P.n_nb;
    lo  = robot.q_min(:);
    hi  = robot.q_max(:);
    n   = numel(lo);
    q0  = lsr_clamp(q0(:), lo, hi);
    th0 = q0;                                                           % theta* of Eq. 11
    w   = lsr_omega(n, P);
    su  = lsr_sigma_u(P.tau);                                           % Eq. 23
    npg = N * (nb + 1);                                                 % FK per iteration
    T   = ikb_generations(opts, N, npg, P.T);

    X = zeros(n, 0);  F = zeros(1, 0);  XT = zeros(4, 4, 0);  b = 0;
    fk = 0;  iters = 0;  f_init = NaN;
    hist = zeros(1, 0);  n5 = hist;  n27a = hist;  n27b = hist;  nacc = hist;

    if ikb_budget_left(opts) >= N
        X  = [q0, lo + rand(n, N - 1) .* (hi - lo)];                     % Eq. 13, q0 first
        F  = zeros(1, N);
        XT = zeros(4, 4, N);
        for i = 1:N
            [F(i), XT(:, :, i)] = lsr_fitness(robot, T_d, X(:, i), th0, w, P);
            fk = fk + 1;
        end
        [f_init, b] = min(F);                                           % Best_X, first of equal ones
        hist = NaN(1, T);  n5 = zeros(1, T);  n27a = n5;  n27b = n5;  nacc = n5;
        for g = 1:T
            if ikb_budget_left(opts) < npg, break; end
            B   = X(:, b);                                              % Best_X of this iteration
            Xn  = X;  Fn = F;  XTn = XT;
            for i = 1:N
                x   = X(:, i);
                lam = lsr_lambda(x, B, lo, hi, su, P);                  % step 3, Eq. 24
                Y   = lsr_neighbours(x, lam, nb, lo, hi);               % step 3, Eq. 14 (clamped)
                FY  = zeros(1, nb);
                for j = 1:nb
                    FY(j) = lsr_fitness(robot, T_d, Y(:, j), th0, w, P);
                    fk = fk + 1;
                end
                [fj, j] = min(FY);                                      % the best neighbour
                if fj < F(i)                                            % step 4
                    s0 = lsr_slope(F(i), fj, x, Y(:, j));               % Eq. 18
                    xn = lsr_move(x, Y(:, j), s0, su, P);               % Eqs. 25-26
                    n5(g) = n5(g) + 1;
                else
                    r  = lsr_pick_r(i, N);                              % Fig. 5, r other than i
                    [xn, first] = lsr_selfupdate(x, B, F(r) < F(i), su, P);   % Eq. 27
                    n27a(g) = n27a(g) + first;  n27b(g) = n27b(g) + ~first;
                end
                xn = lsr_clamp(xn, lo, hi);
                [fn, Tn] = lsr_fitness(robot, T_d, xn, th0, w, P);
                fk = fk + 1;
                [Xn(:, i), Fn(i), XTn(:, :, i), a] = lsr_accept(x, F(i), XT(:, :, i), xn, fn, Tn);
                nacc(g) = nacc(g) + a;
            end
            X = Xn;  F = Fn;  XT = XTn;
            [fm, bm] = min(F);
            if fm < F(b), b = bm; end                                   % step 7
            iters = g;
            hist(g) = F(b);
        end
        hist = hist(1:iters);  n5 = n5(1:iters);  n27a = n27a(1:iters);  n27b = n27b(1:iters);  nacc = nacc(1:iters);
    end

    q_best = q0;  fb = NaN;  e1n = NaN;  e2n = NaN;
    if b > 0
        q_best = X(:, b);
        fb = F(b);
        [~, e1, e2] = pose_error(T_d, XT(:, :, b));
        e1n = norm(e1);  e2n = norm(e2);
    end

    info = struct();
    info.solver       = 'LSRFDA';
    info.iters        = iters;                                          % iterations completed
    info.time         = toc(t_start);
    info.converged    = false;                                          % no convergence criterion in the paper
    info.fitness      = fb;                                             % eta of Best_X (Eq. 12)
    info.fitness_init = f_init;                                         % lowest eta of the initial population
    info.err_pos      = e1n;                                            % m
    info.err_rot      = e2n;                                            % rad
    info.fk_calls     = fk;
    info.jac_calls    = 0;
    info.N            = N;
    info.n_nb         = nb;
    info.T            = T;
    info.sigma_u      = su;
    info.history      = hist;                                           % eta of Best_X after each iteration
    info.n_step5      = n5;                                             % flows moved by Eqs. 25-26
    info.n_eq27_first = n27a;                                           % flows moved by s_L * Best_X
    info.n_eq27_else  = n27b;                                           % flows moved by Flow_X + s_L (Best_X - Flow_X)
    info.accepted     = nacc;                                           % new positions accepted
    info.final        = struct('X', X, 'F', F, 'b', b);                 % last population
    info.params       = P;
end

% ==================================================================== parameters and operators
function P = lsr_params()
% N, Max_Iter, n_nb, tau: not stated (header); alpha, beta (p.5, Eq. 12); z, omega (p.5, Eq. 11).
    P = struct('N', 40, 'T', 200, 'n_nb', 2, 'tau', 1.5, 'alpha', 1, 'beta', 1e-10, 'z', 0.001, ...
               'omega', [3 2 1.5 1 1 1]);
end

function su = lsr_sigma_u(tau)
% Eq. 23 with the exponent of 2 as printed, tau - 1/2.
    su = (gamma(1 + tau) * sin(pi * tau / 2) / (gamma((1 + tau) / 2) * tau * 2^(tau - 1/2)))^(1 / tau);
end

function [s, u, e] = lsr_levy(su, P)
% Eq. 22: s_L = u / |epsilon|^(1/tau), u = sigma_u randn, epsilon = randn (sigma_eps = 1); one draw of each.
    u = su * randn;
    e = randn;
    s = u / abs(e)^(1 / P.tau);
end

function xr = lsr_xrand(lo, hi)
% Xrand: a random position by Eq. 13, one rand per joint.
    xr = lo + rand(numel(lo), 1) .* (hi - lo);
end

function [lam, xr, r1, r2, s, nn] = lsr_lambda(x, B, lo, hi, su, P)
% Eq. 24: Lambda_new = (rand * Xrand - rand * Flow_X(i)) * ||Best_X - Flow_X(i)|| * s_L.
    n  = numel(x);
    xr = lsr_xrand(lo, hi);
    r1 = rand(n, 1);
    r2 = rand(n, 1);
    nn = lsr_vnorm(B - x);
    s  = lsr_levy(su, P);
    lam = (r1 .* xr - r2 .* x) * nn * s;
end

function [Y, c] = lsr_neighbours(x, lam, nb, lo, hi)
% Eq. 14: Neighbor_X(j) = Flow_X(i) + randn * Lambda, one randn per neighbour, clamped to [lo, hi].
    Y = zeros(numel(x), nb);
    c = zeros(1, nb);
    for j = 1:nb
        c(j) = randn;
        Y(:, j) = lsr_clamp(x + c(j) * lam, lo, hi);
    end
end

function s0 = lsr_slope(fi, fj, x, y)
% Eq. 18: S_o(i, j) = (Flow_fitness(i) - Neighbor_fitness(j)) / ||Flow_X(i) - Neighbor_X(j)||.
    s0 = (fi - fj) / lsr_vnorm(x - y);
end

function [xn, s, v] = lsr_move(x, y, s0, su, P)
% Eq. 25: upsilon_new = s_L * s0; Eq. 26: Flow_newX(i) = Flow_X(i) + upsilon_new (x - y) / ||x - y||.
    s  = lsr_levy(su, P);
    v  = s * s0;
    dd = lsr_vnorm(x - y);
    xn = x + v * ((x - y) / dd);
end

function r = lsr_pick_r(i, N)
% Fig. 5 "Generate a random integer r": uniform over {1..N} without i.
    r = randi(N - 1);
    if r >= i, r = r + 1; end
end

function [xn, first, s] = lsr_selfupdate(x, B, better, su, P)
% Eq. 27: s_L * Best_X if Flow_fitness(r) < Flow_fitness(i), Flow_X(i) + s_L * (Best_X - Flow_X(i)) otherwise.
    s = lsr_levy(su, P);
    first = better;
    if first
        xn = s * B;
    else
        xn = x + s * (B - x);
    end
end

function [x, f, Tx, a] = lsr_accept(x, f, Tx, xn, fn, Tn)
% Fig. 5 "Is there a better flow of water?": the new position replaces the flow if its eta is lower.
    a = fn < f;
    if a
        x = xn;  f = fn;  Tx = Tn;
    end
end

function y = lsr_f1(Tq, T_d)
% Eq. 10: ||P_P - P*_P||, the position error norm (m).
    dx = T_d(1, 4) - Tq(1, 4);
    dy = T_d(2, 4) - Tq(2, 4);
    dz = T_d(3, 4) - Tq(3, 4);
    y  = sqrt(dx^2 + dy^2 + dz^2);
end

function y = lsr_f2(q, th0, w, P)
% Eq. 11: sum_i z x [omega_i (theta_i - theta*_i)]^2 (rad).
    y = 0;
    for k = 1:numel(q)
        y = y + P.z * (w(k) * (q(k) - th0(k)))^2;
    end
end

function y = lsr_eta(f1, f2, P)
% Eq. 12: alpha f1 + beta f2.
    y = P.alpha * f1 + P.beta * f2;
end

function [f, Tq] = lsr_fitness(robot, T_d, q, th0, w, P)
% eta of the pose of q; one FK call.
    Tq = robot.fk(q);
    f  = lsr_eta(lsr_f1(Tq, T_d), lsr_f2(q, th0, w, P), P);
end

function w = lsr_omega(n, P)
% omega of Eq. 11 for joints 1-6; 1 beyond the sixth joint.
    w = ones(n, 1);
    m = min(n, numel(P.omega));
    w(1:m) = P.omega(1:m);
end

function x = lsr_clamp(x, lo, hi)
% Clamp to the joint limits.
    x = min(max(x, lo), hi);
end

function y = lsr_vnorm(v)
% Euclidean norm, summed in index order.
    s = 0;
    for k = 1:numel(v)
        s = s + v(k)^2;
    end
    y = sqrt(s);
end
