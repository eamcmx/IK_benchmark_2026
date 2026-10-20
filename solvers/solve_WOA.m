function [q_best, info] = solve_WOA(robot, T_d, q0, opts)
% SOLVE_WOA  Whale Optimization Algorithm (WOA) of Mirjalili and Lewis (2016), Fig. 6, as an IK solver.
%
%   Source:
%     Seyedali Mirjalili and Andrew Lewis (2016). The Whale Optimization Algorithm. Advances in
%     Engineering Software, 95, 51-67. DOI: 10.1016/j.advengsoft.2016.01.008
%   Page numbers are the journal's printed pages 51-67 (PDF page = printed page - 50).
%
%   Algorithm (Sec. 2.2, Eqs. 2.1-2.8, pp.53-54; pseudocode Fig. 6, p.55):
%     N whales are placed at random in the search space and evaluated; X* is the best one (Fig. 6).
%     In iteration t = 1..T the parameter a decreases linearly from 2 to 0 (p.53), and each whale in
%     turn draws A = 2 a r - a and C = 2 r (Eqs. 2.3-2.4), l in [-1, 1] and p in [0, 1] (Fig. 6:
%     "Update a, A, C, l, and p" inside the loop over the agents) and moves:
%       p <  0.5, |A| <  1:  D  = |C X* - X|,       X <- X* - A D                 (Eqs. 2.1-2.2)
%       p <  0.5, |A| >= 1:  D  = |C X_rand - X|,   X <- X_rand - A D             (Eqs. 2.7-2.8)
%       p >= 0.5:            D' = |X* - X|,         X <- D' e^(b l) cos(2 pi l) + X*   (Eq. 2.5)
%     with | | the element-wise absolute value (p.53) and X_rand a whale chosen at random from the
%     current population (p.55). After the loop the whales outside the search space are amended, all
%     whales are evaluated, and X* is replaced by a better one (Fig. 6). The run ends after T
%     iterations (Fig. 6, "while (t < maximum number of iterations)"); X* is returned.
%
%   Parameters:
%     N = 20, T = 100   the reported setting whose N x T is nearest the 2000-FK budget (benchmark rule
%                     for several reported settings). The paper has no IK task; its runs use 30 x 500
%                     on the 29 test functions (p.55) and the 15-, 25- and 52-bar trusses (pp.62-64),
%                     10 x 500 on the tension/compression spring (Sec. 4.1, p.60), 20 x 500 on the
%                     welded beam (Sec. 4.2, p.60) and the pressure vessel (Sec. 4.3, p.61), and the
%                     grid n in {5, 10, 20, 40, 80, 100} x t in {50, 100, 500, 1000} of the parameter
%                     study on the 52-bar truss (p.65, Fig. 15). 20 x 100 and 40 x 50 both give exactly
%                     2000; 20 x 100 is used because n = 20 is also a population the paper runs on its
%                     own design problems. Inside the benchmark T is the budget horizon (Benchmark
%                     interface).
%     a                 linear from 2 to 0 over the iterations (p.53; the parameter study, p.65, finds
%                     2 -> 0 best): a = 2 (T - t) / T, t = 1..T, reaching 0 in the last iteration.
%     A, C              Eqs. 2.3-2.4 with r uniform in [0, 1] (p.53). "A is a random value in the
%                     interval [-a, a]" (p.54) and Fig. 6 tests |A| < 1, so A and C are scalars: one r
%                     for A and one for C per whale and iteration (with a vector r, as the arrows of
%                     Eqs. 2.3-2.4 suggest, the test |A| < 1 is undefined). The authors' code draws
%                     them the same way (see Relation to the authors' code).
%     p, l              p uniform in [0, 1], l uniform in [-1, 1] (p.54), one each per whale and iteration
%                     (Fig. 6).
%     b = 1             "a constant for defining the shape of the logarithmic spiral" (p.54); its value
%                     is not stated in the paper (checked pp.53-55 and 59-65); b = 1, the value of the
%                     authors' code (see Relation to the authors' code); e^(b l) lies in [1/e, e].
%     X_rand            uniform over all N whales of the current population, the moving whale included
%                     (the draw is not stated in the paper, checked pp.54-55).
%     Fitness           the paper defines no IK objective (its problems are analytic test functions,
%                     Sec. 3, and structural designs, Sec. 4); the benchmark's pose error is used:
%                     f = ||e_p|| + ||e_R||, metres plus radians, with e_p and e_R of pose_error
%                     (||e_R|| is the rotation angle). f < 1e-5 implies the pose gate.
%     Initial population  "Initialize the whales population" (Fig. 6), "a set of random solutions"
%                     (p.55); the distribution is not stated; uniform in [q_min, q_max].
%     Stop              T iterations (Fig. 6); the "termination criterion" of p.55 is not specified
%                     otherwise.
%
%   Benchmark interface:
%     Budget: the initialisation needs N = 20 FK calls and every iteration N = 20 (all whales are
%       evaluated once, after the moves, Fig. 6). T = ikb_generations(opts, 20, 20, 100) = 99, so
%       20 + 99 x 20 = 2000 FK calls per waypoint; a 40-member population would give
%       ikb_generations(opts, 40, 40, 100) = 49. The initialisation and every iteration start only if
%       ikb_budget_left(opts) covers N. Outside the benchmark T = 100 (2020 FK). No final FK check.
%       Every option field is ignored (the budget is read only through ikb_generations and
%       ikb_budget_left).
%     Stop: T iterations only; info.converged is always false (the paper has no accuracy test).
%     Warm start: q0 (clamped to the limits) is whale 1 of the initial population, evaluated
%       unperturbed; the other 19 are uniform in [q_min, q_max]. X* starts from the initial
%       population and is replaced only by a strictly lower f, so q0 is returned unless a later
%       evaluation is better.
%     Robot and units: robot.fk with robot.n joints; metres and radians (the paper has no unit).
%     Joint limits: the joint ranges are the search space. "Check if any search agent goes beyond the
%       search space and amend it" (Fig. 6) names no rule: after the loop over the whales every
%       coordinate outside [q_min, q_max] is clamped to the nearer bound. info.clamped counts the
%       clamped coordinates per iteration.
%     Output: X*; info.fitness (f), info.err_pos (m) and info.err_rot (rad) from the pose stored at its
%       evaluation. solve_WOA('ops') returns handles to the operators for unit tests.
%
%   Relation to the authors' code (WOA.m, "Whale Optimization Algorithm (WOA) source codes demo 1.0",
%   from the address given in the paper's abstract; consulted only for what the paper does not state):
%     It agrees with the reading above on b = 1, on scalar A and C (r1, r2 = rand() per whale), on one
%     p per whale, on clamping out-of-range coordinates to the bounds before each evaluation, on moving
%     the agents in place with X* held constant during the moves, on replacing X* only by a strictly lower
%     fitness, and on a uniform initial population. Where it departs from the paper, the paper is
%     followed: the code draws l from [a2, 1] with a2 decreasing from -1 to -2 (paper: l in [-1, 1],
%     p.54); it redraws X_rand for every coordinate (paper: "a random whale", p.55); and its a runs
%     from 2 at the first move to 2/T at the last, whose positions it does not evaluate (Fig. 6
%     evaluates after every iteration; here a reaches 0 in the last iteration).
%
%   Deviations from the paper:
%     - Iteration horizon T = 99 instead of 100 (benchmark rule); the schedule of a runs over T.
%     - q0 is a member of the initial population (warm start).
%     - Fitness: the benchmark's pose error (the paper defines no IK objective).
%     - Not stated in the paper, chosen as given under Parameters: b = 1 and scalar A and C (as in
%       the authors' code), the distributions of r, p, l, X_rand and of the initial population,
%       clamping as the amendment,
%       in-place moves (Fig. 6 moves the agents in turn, so a whale drawn as X_rand after it has moved
%       in the same iteration is at its moved, not yet amended position), ties (the earlier whale is
%       kept as X*).
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
        q_best = struct('params', woa_params(), 'core', @woa_core, 'a', @woa_a, 'coeffs', @woa_coeffs, ...
                        'move', @woa_move, 'encircle', @woa_encircle, 'search', @woa_search, ...
                        'spiral', @woa_spiral, 'fitness', @woa_fitness);
        info = [];
        return
    end
    [q_best, info] = woa_core(robot, T_d, q0, opts, woa_params());
end

% ==================================================================== main loop (Fig. 6)
function [q_best, info] = woa_core(robot, T_d, q0, opts, P)
% Fig. 6 (p.55): initial population (q0 first) and X*; per iteration a, then for each whale A, C, l, p
% and the move of Eq. 2.2, 2.8 or 2.5; then the amendment (clamp), the evaluation of all whales and
% the update of X*.
    t_start = tic;
    N  = P.N;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    n  = numel(lo);
    q0 = min(max(q0(:), lo), hi);
    T  = ikb_generations(opts, N, N, P.T);

    B = struct('f', NaN, 'q', q0, 'T', NaN(4), 'nfk', 0);
    iters = 0;  hist = zeros(1, 0);  branch = zeros(3, 0);  clamped = zeros(1, 0);  aval = zeros(1, 0);
    fin = struct('X', zeros(n, 0), 'F', zeros(1, 0));

    if ikb_budget_left(opts) >= N
        X = [q0, lo + rand(n, N - 1) .* (hi - lo)];                     % Fig. 6, q0 first
        [F, B] = evaluate(robot, T_d, X, B);
        hist = NaN(1, T);  branch = zeros(3, T);  clamped = zeros(1, T);  aval = NaN(1, T);
        for t = 1:T
            if ikb_budget_left(opts) < N, break; end
            a  = woa_a(t, T);
            xs = B.q;                                                   % X*, fixed within the iteration
            for i = 1:N
                [A, C, l, p] = woa_coeffs(a);
                xr = [];
                if p < 0.5 && abs(A) >= 1
                    xr = X(:, randi(N));                                % X_rand, current population
                end
                [X(:, i), br] = woa_move(X(:, i), xs, xr, A, C, l, p, P.b);
                branch(br, t) = branch(br, t) + 1;
            end
            clamped(t) = nnz(X < lo | X > hi);
            X = min(max(X, lo), hi);                                    % amendment (Fig. 6)
            [F, B] = evaluate(robot, T_d, X, B);                        % fitness, update X*
            iters = t;
            hist(t) = B.f;
            aval(t) = a;
        end
        hist = hist(1:iters);  branch = branch(:, 1:iters);  clamped = clamped(1:iters);  aval = aval(1:iters);
        fin = struct('X', X, 'F', F);
    end

    ep = NaN;  er = NaN;
    if ~isnan(B.f)
        [~, e_p, e_R] = pose_error(T_d, B.T);
        ep = norm(e_p);
        er = norm(e_R);
    end

    q_best = B.q;
    info = struct();
    info.solver    = 'WOA';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = false;                                             % no accuracy test in the paper
    info.fitness   = B.f;                                               % ||e_p|| + ||e_R|| (m + rad)
    info.err_pos   = ep;                                                % m
    info.err_rot   = er;                                                % rad
    info.fk_calls  = B.nfk;
    info.jac_calls = 0;
    info.N         = N;
    info.T         = T;
    info.history   = hist;                                              % X* fitness after iteration t
    info.a         = aval;                                              % a of iteration t
    info.branch    = branch;                                            % moves by Eq. 2.2 / 2.8 / 2.5
    info.clamped   = clamped;                                           % clamped coordinates
    info.final     = fin;                                               % last population and fitness
    info.params    = P;
end

function [F, B] = evaluate(robot, T_d, X, B)
% Fitness of every whale (one FK call each); X* is replaced only by a strictly lower fitness ("Update
% X* if there is a better solution", Fig. 6).
    N = size(X, 2);
    F = zeros(1, N);
    for i = 1:N
        [F(i), Tq] = woa_fitness(robot, T_d, X(:, i));
        B.nfk = B.nfk + 1;
        if isnan(B.f) || F(i) < B.f
            B.f = F(i);
            B.q = X(:, i);
            B.T = Tq;
        end
    end
end

% ==================================================================== operators
function P = woa_params()
% N = 20, T = 100 (setting nearest the budget, p.65 with pp.60-61); b = 1 (not stated, p.54).
    P = struct('N', 20, 'T', 100, 'b', 1);
end

function a = woa_a(t, T)
% a linear from 2 to 0 over the iterations (p.53): a = 2 (T - t) / T, t = 1..T.
    a = 2 * (T - t) / T;
end

function [A, C, l, p] = woa_coeffs(a)
% Eqs. 2.3-2.4 with scalar r ~ U(0, 1) for A and for C; l ~ U(-1, 1), p ~ U(0, 1) (p.54; Fig. 6).
    A = 2 * a * rand - a;
    C = 2 * rand;
    l = 2 * rand - 1;
    p = rand;
end

function [x, br] = woa_move(x, xs, xr, A, C, l, p, b)
% Fig. 6: p < 0.5 and |A| < 1 -> Eq. 2.2 (br = 1); p < 0.5 and |A| >= 1 -> Eq. 2.8 (br = 2);
% p >= 0.5 -> Eq. 2.5 (br = 3).
    if p < 0.5
        if abs(A) < 1
            x = woa_encircle(x, xs, A, C);
            br = 1;
        else
            x = woa_search(x, xr, A, C);
            br = 2;
        end
    else
        x = woa_spiral(x, xs, l, b);
        br = 3;
    end
end

function x = woa_encircle(x, xs, A, C)
% Eqs. 2.1-2.2: D = |C X* - X|, X(t+1) = X* - A D.
    x = xs - A * abs(C * xs - x);
end

function x = woa_search(x, xr, A, C)
% Eqs. 2.7-2.8: D = |C X_rand - X|, X(t+1) = X_rand - A D.
    x = xr - A * abs(C * xr - x);
end

function x = woa_spiral(x, xs, l, b)
% Eq. 2.5: D' = |X* - X|, X(t+1) = D' e^(b l) cos(2 pi l) + X*.
    x = abs(xs - x) * exp(b * l) * cos(2 * pi * l) + xs;
end

function [f, Tq] = woa_fitness(robot, T_d, q)
% Benchmark pose error (the paper defines no IK objective): f = ||e_p|| + ||e_R||, m + rad.
    Tq = robot.fk(q);
    [~, e_p, e_R] = pose_error(T_d, Tq);
    f = norm(e_p) + norm(e_R);
end
