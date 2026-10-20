function [q_best, info] = solve_GA(robot, T_d, q0, opts)
% SOLVE_GA  Genetic algorithm (GA) IK solver as specified by Orbegoso Moreno et al. (2023), Sec. II.C.4.
%
%   Source:
%     Luis A. Orbegoso Moreno, D. Valverde Ramírez, M. Pasco Sánchez and L. Ruiz Rodríguez (2023).
%     Comparative Study of Iterative Methods for Inverse Kinematics of Redundant Serial Robots with
%     Increasing Degrees of Freedom. 2023 IEEE International Autumn Meeting on Power, Electronics and
%     Computing (ROPEC 2023). Ixtapa, Mexico. 6 pages; no volume, issue or page numbers printed.
%     DOI: 10.1109/ROPEC58757.2023.10409450
%   Page numbers are PDF pages 1-6. The paper compares five existing methods (PSO, QPSO, GA, DLS and
%   FABRIK; abstract, p.1); this file implements its GA. The paper bases its GA settings on its
%   Ref. [19] (C. Lee, J. Chang (2021), A Workspace-Analysis-Based genetic algorithm for solving
%   inverse kinematics of a Multi-Fingered anthropomorphic hand, Applied Sciences, 11(6), 2668,
%   doi 10.3390/app11062668; not consulted) and implements it with the DEAP Python library (p.3; the
%   paper lists no reference for DEAP: F.-A. Fortin, F.-M. De Rainville, M.-A. Gardner, M. Parizeau,
%   C. Gagné (2012), DEAP: Evolutionary Algorithms Made Easy, J. Mach. Learn. Res., 13, 2171-2175).
%   The operator definitions below follow the DEAP source code (github.com/DEAP/deap, master branch:
%   tools/crossover.py cxSimulatedBinary, tools/mutation.py mutGaussian, tools/selection.py
%   selTournament, algorithms.py varAnd and eaSimple); the DEAP version and the DEAP algorithm the
%   paper used are not stated.
%
%   Algorithm (Eq. 1 and Sec. II.C, p.3; GA settings Sec. II.C.4, p.3):
%     A population of N individuals, each a vector of n joint angles (genes = DoF, p.3), is
%     evaluated with the error E of Eq. 1. Each generation g = 1..T (DEAP eaSimple with varAnd):
%       selection    N tournaments of 3 ("Tournament-3", p.3): 3 aspirants drawn uniformly with
%                    replacement, the one with the lowest E wins (DEAP selTournament; tournament
%                    selection: D. E. Goldberg, K. Deb (1991), A comparative analysis of selection
%                    schemes used in genetic algorithms, Foundations of Genetic Algorithms 1, 69-93;
%                    not consulted).
%       crossover    the winners are paired in order, (1, 2), (3, 4), ...; each pair is crossed
%                    with probability 0.8 by simulated binary crossover (SBX, p.3) on every gene j:
%                      beta_j = (2 u_j)^(1/(eta_c+1))             if u_j <= 0.5
%                      beta_j = (1/(2(1 - u_j)))^(1/(eta_c+1))    otherwise,    u_j ~ U(0, 1)
%                      c1_j = 0.5((1 + beta_j) p1_j + (1 - beta_j) p2_j)
%                      c2_j = 0.5((1 - beta_j) p1_j + (1 + beta_j) p2_j)
%                    (DEAP cxSimulatedBinary; SBX: K. Deb, R. B. Agrawal (1995), Simulated binary
%                    crossover for continuous search space, Complex Systems, 9(2), 115-148; not
%                    consulted).
%       mutation     Gaussian mutation (p.3): every gene of every child is mutated with probability
%                    0.05, x_j <- x_j + N(mu, sigma^2) (DEAP mutGaussian).
%       replacement  the children replace the whole population and are all evaluated (DEAP eaSimple).
%     Fitness (Eq. 1, p.3):
%       E = sqrt(dx^2 + dy^2 + dz^2 + dO_{1,2}^2 + dO_{0,2}^2 + dO_{1,0}^2)
%     with (dx, dy, dz) = P_t - P_e (target minus end-effector position) and dO = O_t O_e^T (target
%     orientation times the transposed end-effector orientation), dO_{1,2} = O_{2,1} - O_{1,2},
%     dO_{0,2} = O_{2,0} - O_{0,2}, dO_{1,0} = O_{0,1} - O_{1,0}, indices from 0. The right-hand
%     entries carry no Delta on the page; they are read as entries of dO, the only reading under
%     which the three terms compare the two orientations. With D = R_d R^T and 1-based indices the
%     terms are D(3,2) - D(2,3), D(3,1) - D(1,3) and D(1,2) - D(2,1): 2 sin(theta) times the axis
%     of the relative rotation of angle theta, so they vanish at theta = 0 and at theta = pi, and E
%     does not distinguish the target orientation from its half-turns about any axis.
%
%   Parameters (p.3 unless stated):
%     N = 300          population size. The paper reports one GA setting for all its chains (7 to
%                      15 DoF; Table 2, p.5): N x T = 300 x 1000 = 300,000, the only candidate of
%                      the nearest-setting rule.
%     T = 1000         maximum number of iterations, common to the five methods (Sec. II.C, p.3).
%                      Inside the benchmark T is the budget horizon (Benchmark interface).
%     n genes          the DoF of the chain (p.3): robot.n.
%     cxpb = 0.8       cross rate (p.3), applied to each pair of consecutive winners (DEAP varAnd).
%     eta_c = 20       SBX distribution index: not stated in the paper (checked pages 1-6; Ref. [19]
%                      not available; DEAP cxSimulatedBinary has no default); benchmark uses 20
%                      because it is the value of DEAP's own SBX example (examples/ga/nsga2.py,
%                      eta = 20.0).
%     indpb = 0.05     the "mutated genes percentage" of 5% (p.3), read as the probability that a
%                      gene is mutated (DEAP mutGaussian indpb). Every child is passed to the
%                      mutation (DEAP mutpb = 1): not stated in the paper (checked pages 1-6);
%                      benchmark uses it because it is the reading under which the expected share
%                      of mutated genes is 5%.
%     mu = 0, sigma = 1 rad   mean and standard deviation of the Gaussian mutation: not stated in
%                      the paper (checked pages 1-6; Ref. [19] not available; DEAP mutGaussian has
%                      no default); benchmark uses mu = 0 and sigma = 1 because they are the values
%                      of the mutGaussian example in the DEAP documentation (Overview page, next to
%                      selTournament with tournsize = 3); genes are joint angles in rad (Table 1).
%     tournsize = 3    "Tournament-3" (p.3). Aspirants drawn with replacement (DEAP selRandom) and
%                      ties won by the first aspirant drawn: not stated in the paper (checked
%                      pages 1-6).
%     E_stop = 0.001   the paper's stop on Eq. 1 (Sec. II.C, p.3); replaced by the pose gate in the
%                      benchmark (Benchmark interface); used by the operator handle only.
%     Units            positions of Eq. 1 in cm: dx = 100 (x_t - x_e) with x in m. The paper's
%                      chains are given in cm (Table 1, p.2; Fig. 2, p.3), and its reported errors
%                      exclude metres: Table 2 (p.5) gives a mean final error of 3.47 for DLS on the
%                      7-DoF chain, while with metres E cannot exceed 2.08 on that chain (each
%                      orientation term vector has norm 2 sin(theta) <= 2; two points of its 0.28 m
%                      reach are at most 0.56 m apart).
%     Initial population   q0 (Warm start) and N - 1 individuals uniform within the joint limits:
%                      the paper draws its random configurations from "random angles contained in
%                      the admissible intervals of each joint" (Sec. II.B, p.2); how the GA
%                      population is formed is not stated (checked pages 1-6).
%     Replacement and evaluation   generational, no elitism, every child evaluated once per
%                      generation: not stated in the paper (checked pages 1-6); DEAP eaSimple
%                      replaces the population by the offspring and evaluates every offspring whose
%                      fitness the variation invalidated, which with mutpb = 1 is every offspring.
%
%   Benchmark interface:
%     Budget: the initial population needs N = 300 FK calls and every generation N = 300 (every
%       child evaluated once). T = ikb_generations(opts, 300, 300, 1000) = 5, so 300 + 5 x 300 =
%       1800 FK calls per waypoint; the 200 left cannot pay for a sixth generation. A 40-member
%       population would give ikb_generations(opts, 40, 40, 1000) = 49. The initial population and
%       every generation start only if ikb_budget_left(opts) covers N. No quantity of the GA varies
%       over T, so T only ends the loop. Outside the benchmark T = 1000. No final FK check.
%       opts.max_iter and opts.pop_size are ignored.
%     Consequence: 5 generations per waypoint, against 102.25 to 113.14 on average in the paper's
%       runs (Table 2, p.5) and up to 1000. Within the budget the GA scores 300 configurations and
%       varies them for 5 generations; the returned individual is often the warm start q0 itself
%       (41 % of the 2400 waypoints of a Phase-1 run of the grid: 6 robots x 4 trajectories x 100).
%     Stop: the benchmark pose gate, position error < opts.tol_pos (m) and rotation error <
%       opts.tol_rot (rad) of pose_error for the individual with the lowest E, the test of bench_run
%       (1e-5 each when a field is absent), checked after the initial population and after every
%       generation; otherwise T generations or the budget. info.converged is the gate test. The
%       paper's E < 0.001 is the accuracy of its own test runs, common to the five methods; it is
%       replaced by the gate (benchmark rule) and is looser than the gate in rotation (it admits a
%       relative rotation of up to 5e-4 rad). Through the operator handle (core with stop 'paper')
%       E < 0.001 ends the run instead and info.converged is that test.
%     Warm start: q0 (clamped to the limits) takes the place of the paper's zero-angle start (p.3)
%       as individual 1 of the initial population, evaluated unperturbed; it enters the first
%       tournaments, and the returned individual is the lowest E evaluated, so q0 is returned
%       unless a strictly lower E is found.
%     Units: lengths in m, converted to cm inside Eq. 1 (see Parameters); angles in rad.
%     Joint limits: the paper gives the limits of its chains (Table 1, p.2) but no handling for
%       the GA (checked pages 1-6; the limit enforcement on p.3 belongs to DLS). SBX and the
%       Gaussian mutation can leave the range, so every child is clamped to [q_min, q_max] after
%       the mutation; info.clamped counts the clamped genes per generation.
%     Robot: robot.fk with robot.n joints in place of the paper's screw-theory chains (Table 1).
%     Output: the individual with the lowest E evaluated; info.fitness is its E (cm and unitless
%       orientation terms); info.err_pos (m) and info.err_rot (rad) come from its stored pose, as
%       in bench_run. solve_GA('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Generation horizon T = 5 instead of up to 1000 (benchmark budget rule).
%     - Stop at the benchmark pose gate instead of E < 0.001 (benchmark rule for a paper's test
%       accuracy).
%     - q0 as individual 1 in place of the zero-angle start (warm start).
%     - Children clamped to the joint limits (handling not stated).
%     - Not stated in the paper, chosen as given under Parameters: eta_c, mu, sigma, mutpb = 1, the
%       uniform initial population, tournaments with replacement and their tie rule, pairing in
%       selection order, generational replacement without elitism, evaluation of every child.
%     - robot.fk instead of the paper's chains.
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
        q_best = struct('params', ga_params(), 'core', @ga_core, 'fitness', @ga_fitness, ...
                        'eq1', @ga_eq1, 'select', @ga_select, 'crossover', @ga_crossover, ...
                        'sbx', @ga_sbx, 'mutate', @ga_mutate, 'generation', @ga_generation, ...
                        'gate', @ga_gate);
        info = [];
        return
    end
    [q_best, info] = ga_core(robot, T_d, q0, opts, ga_params(), 'gate');
end

% ==================================================================== main loop (Sec. II.C.4)
function [q_best, info] = ga_core(robot, T_d, q0, opts, P, mode)
% Initial population (q0 first), then per generation selection, SBX, Gaussian mutation, clamp and
% evaluation of every child with Eq. 1. mode 'gate': stop at the benchmark pose gate; mode 'paper':
% stop at E < P.E_stop (Sec. II.C, p.3). Both also end after T generations or when the budget left
% cannot cover a generation.
    t_start = tic;
    N  = P.N;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    n  = numel(lo);
    q0 = min(max(q0(:), lo), hi);
    T  = ikb_generations(opts, N, N, P.T);
    [tp, tr] = ga_gate(opts);

    B = struct('f', NaN, 'q', q0, 'T', NaN(4), 'g', NaN, 'nfk', 0);
    iters = 0;  hist = zeros(1, 0);  f_init = NaN;  stop = 'budget';
    cnt = zeros(3, 0);

    if ikb_budget_left(opts) >= N
        X = [q0, lo + rand(n, N - 1) .* (hi - lo)];                        % q0 first
        [F, B] = evaluate(robot, T_d, X, B, P, 0);
        f_init = B.f;
        hist = NaN(1, T);  cnt = zeros(3, T);
        stop = 'T';
        if stop_test(B, T_d, mode, P, tp, tr)
            stop = mode;
        else
            for g = 1:T
                if ikb_budget_left(opts) < N, stop = 'budget'; break; end
                [X, cnt(:, g)] = ga_generation(X, F, lo, hi, P);
                [F, B] = evaluate(robot, T_d, X, B, P, g);
                iters = g;
                hist(g) = B.f;
                if stop_test(B, T_d, mode, P, tp, tr), stop = mode; break; end
            end
        end
        hist = hist(1:iters);
        cnt  = cnt(:, 1:iters);
    end

    ep = NaN;  er = NaN;  gate_met = false;
    if ~isnan(B.f)
        [~, e_p, e_r] = pose_error(T_d, B.T);
        ep = norm(e_p);
        er = norm(e_r);
        gate_met = ep < tp && er < tr;
    end
    paper_met = ~isnan(B.f) && B.f < P.E_stop;

    q_best = B.q;
    info = struct();
    info.solver    = 'GA';
    info.iters     = iters;                                             % generations completed
    info.time      = toc(t_start);
    if strcmp(mode, 'paper'), info.converged = paper_met; else, info.converged = gate_met; end
    info.fitness   = B.f;                                               % Eq. 1 (cm, unitless)
    info.fitness_init = f_init;                                         % lowest E of the initial population
    info.err_pos   = ep;                                                % m
    info.err_rot   = er;                                                % rad
    info.gate_met  = gate_met;
    info.paper_met = paper_met;                                         % E < 0.001, report only
    info.fk_calls  = B.nfk;
    info.jac_calls = 0;
    info.N         = N;
    info.T         = T;
    info.stop      = stop;                                              % 'gate', 'paper', 'T', 'budget'
    info.gate      = [tp, tr];
    info.best_gen  = B.g;                                               % generation of the returned individual (0 = initial)
    info.history   = hist;                                              % lowest E after generation g
    info.crossed   = cnt(1, :);                                         % pairs crossed by SBX
    info.mutated   = cnt(2, :);                                         % genes mutated
    info.clamped   = cnt(3, :);                                         % genes clamped to the limits
    info.params    = P;
end

function [F, B] = evaluate(robot, T_d, X, B, P, g)
% Eq. 1 for every individual (one FK call each); the best is replaced only by a strictly lower E.
    N = size(X, 2);
    F = zeros(1, N);
    for i = 1:N
        [F(i), Tq] = ga_fitness(robot, T_d, X(:, i), P);
        B.nfk = B.nfk + 1;
        if isnan(B.f) || F(i) < B.f
            B.f = F(i);
            B.q = X(:, i);
            B.T = Tq;
            B.g = g;
        end
    end
end

function hit = stop_test(B, T_d, mode, P, tp, tr)
% 'gate': position < tp and rotation < tr (pose_error, as bench_run); 'paper': E < E_stop (p.3).
    if strcmp(mode, 'paper')
        hit = B.f < P.E_stop;
    else
        [~, e_p, e_r] = pose_error(T_d, B.T);
        hit = norm(e_p) < tp && norm(e_r) < tr;
    end
end

% ==================================================================== operators
function P = ga_params()
% p.3: N = 300, T = 1000, cross rate 0.8, 5% mutated genes, Tournament-3, E < 0.001; not stated:
% eta_c = 20, mu = 0, sigma = 1 (rad); positions of Eq. 1 in cm (len_scale = 100 per m).
    P = struct('N', 300, 'T', 1000, 'cxpb', 0.8, 'indpb', 0.05, 'tournsize', 3, 'eta', 20, ...
               'mu', 0, 'sigma', 1, 'E_stop', 1e-3, 'len_scale', 100);
end

function [Y, c] = ga_generation(X, F, lo, hi, P)
% One generation of variation (DEAP varAnd after selTournament): N tournament winners, SBX on
% consecutive pairs with probability cxpb, Gaussian mutation of every child, clamp to the limits.
% c = [pairs crossed; genes mutated; genes clamped].
    N = size(X, 2);
    Y = X(:, ga_select(F, N, P.tournsize));
    [Y, nc] = ga_crossover(Y, P);
    nm = 0;
    for i = 1:N
        [Y(:, i), m] = ga_mutate(Y(:, i), P);
        nm = nm + m;
    end
    ncl = nnz(Y < lo | Y > hi);
    Y = min(max(Y, lo), hi);
    c = [nc; nm; ncl];
end

function S = ga_select(F, k, ts)
% Tournament-3 (p.3; DEAP selTournament): k tournaments, ts aspirants drawn uniformly with
% replacement, the lowest E wins (first aspirant drawn on ties). S = indices of the winners.
    N = numel(F);
    S = zeros(1, k);
    for i = 1:k
        a = randi(N, ts, 1);
        [~, m] = min(F(a));
        S(i) = a(m);
    end
end

function [Y, nc] = ga_crossover(Y, P)
% Pairs (1, 2), (3, 4), ... of the winners crossed by SBX with probability cxpb (DEAP varAnd).
    N  = size(Y, 2);
    nc = 0;
    for k = 2:2:N
        if rand < P.cxpb
            [Y(:, k - 1), Y(:, k)] = ga_sbx(Y(:, k - 1), Y(:, k), P.eta);
            nc = nc + 1;
        end
    end
end

function [c1, c2] = ga_sbx(p1, p2, eta)
% SBX on every gene (DEAP cxSimulatedBinary): u ~ U(0, 1), beta = (2u)^(1/(eta+1)) if u <= 0.5,
% (1/(2(1 - u)))^(1/(eta+1)) otherwise; c1 = 0.5((1 + beta) p1 + (1 - beta) p2),
% c2 = 0.5((1 - beta) p1 + (1 + beta) p2).
    n = numel(p1);
    u = rand(n, 1);
    beta = zeros(n, 1);
    for j = 1:n
        if u(j) <= 0.5
            b = 2 * u(j);
        else
            b = 1 / (2 * (1 - u(j)));
        end
        beta(j) = b ^ (1 / (eta + 1));
    end
    c1 = 0.5 * ((1 + beta) .* p1 + (1 - beta) .* p2);
    c2 = 0.5 * ((1 - beta) .* p1 + (1 + beta) .* p2);
end

function [x, m] = ga_mutate(x, P)
% Gaussian mutation (DEAP mutGaussian): each gene with probability indpb, x_j <- x_j + N(mu, sigma^2).
    hit = rand(numel(x), 1) < P.indpb;
    m = nnz(hit);
    x(hit) = x(hit) + (P.mu + P.sigma * randn(m, 1));
end

function [tp, tr] = ga_gate(opts)
% The benchmark pose gate, as in bench_run: position < tol_pos (m) and rotation < tol_rot (rad).
    tp = 1e-5;  tr = 1e-5;
    if isfield(opts, 'tol_pos'), tp = opts.tol_pos; end
    if isfield(opts, 'tol_rot'), tr = opts.tol_rot; end
end

% ==================================================================== fitness (Eq. 1)
function [E, Tq] = ga_fitness(robot, T_d, q, P)
% Eq. 1 of the pose robot.fk(q) against T_d; one FK call.
    Tq = robot.fk(q);
    E  = ga_eq1(Tq, T_d, P.len_scale);
end

function [E, dp, dO] = ga_eq1(Tq, T_d, s)
% Eq. 1: dp = s (P_t - P_e) (s = 100: m to cm); dO = O_t O_e^T with dO_{1,2} = O_{2,1} - O_{1,2},
% dO_{0,2} = O_{2,0} - O_{0,2}, dO_{1,0} = O_{0,1} - O_{1,0} (0-based indices).
    dp = s * (T_d(1:3, 4) - Tq(1:3, 4));
    D  = T_d(1:3, 1:3) * Tq(1:3, 1:3).';
    dO = [D(3, 2) - D(2, 3); D(3, 1) - D(1, 3); D(1, 2) - D(2, 1)];
    E  = sqrt(dp(1)^2 + dp(2)^2 + dp(3)^2 + dO(1)^2 + dO(2)^2 + dO(3)^2);
end
