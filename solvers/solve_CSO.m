function [q_best, info] = solve_CSO(robot, T_d, q0, opts)
% SOLVE_CSO  Improved Chicken Swarm Optimization (ICSO) IK solver of Wu, Song, Zhang and Su (2021), Sec. II.
%
%   Source:
%     Honggang Wu, Linsen Song, Xinming Zhang, Chengzhi Su (2021). Research on Inverse Kinematics of
%     Robot Based on Improved Chicken Swarm Algorithm. 2021 International Conference on Electronic
%     Information Engineering and Computer Science (EIECS), pp. 533-536.
%     DOI: 10.1109/EIECS53707.2021.9588154
%   Page numbers p.1-p.4 are PDF pages 1-4 (printed pages 533-536).
%   Where the source uses standard CSO without restating it (the partner k, the condition and eps of
%   Eq. 2, the spread of randn, FL), the CSO preliminaries of the same group are used:
%     Honggang Wu, Xinming Zhang, Linsen Song, Chengzhi Su and Lidong Gu (2022). A hybrid improved
%     BRO algorithm and its application in inverse kinematics of 7R 6DOF robot. Advances in
%     Mechanical Engineering, 14(3), 1-20. DOI: 10.1177/16878132221085125
%   cited below as W22 (PDF page = printed page; CSO Eqs. 7-12 and Algorithm 2, pp.4-5; Table 2, p.8).
%
%   Algorithm (Sec. II.A-B, pp.1-2; Table I, p.2; IK problem Sec. III.A, Eq. 13, p.3; settings p.4):
%     The N chickens are ranked by fitness: the best Nr are roosters (x_a), the next Nh hens (x_b),
%     the rest chicks (x_c). Each hen follows a rooster drawn at random from the roosters and each
%     chick a mother drawn at random from the hens (Sec. II.A.1, pp.1-2). The hierarchy is
%     re-established every G iterations (Table I). In iteration t every chicken moves, using the
%     positions and fitness values of iteration t (the superscripts of Eqs. 7-10), and is evaluated:
%       rooster:  x_i,a(t+1) = x_i,a + randn(0, sigma^2) (x_best - x_i,a)                     (Eq. 7)
%                 sigma^2 = 1 if f_i <= f_k, otherwise exp((f_k - f_i) / |f_i + eps|)        (Eq. 2)
%       hen:      e = rand(1,dim) (x_k1,a - x_k1,b)                                           (Eq. 8)
%                 x_i,b(t+1) = x_i,b + S1 rand (x_k1,a - x_i,b) + S2 rand (x_k2,b - x_i,b) + e
%                                                                                           (Eq. 9)
%                 S1 = exp((f_i - f_r1) / |abs(f_i) + eps|),  S2 = exp(f_r2 - f_i)           (Eqs. 4-5)
%       chick:    x_i,c(t+1) = FL x_i,c + rand(1,dim) (x_r3,b - x_r1,a) + x_r1,a             (Eq. 10)
%     These are the three ICSO improvements (abstract, p.1; Sec. II.B, p.2; conclusion, p.4): the
%     rooster learns the global best (Eq. 7 in place of the random walk of Eq. 1), the hen gets the
%     mutation factor e (Eqs. 8-9 in place of Eq. 3), and the chick follows rooster and hen (Eq. 10
%     in place of Eq. 6). x_best is the best position recorded so far ("Record the optimal value",
%     Table I), which is also returned ("Select the optimal position as the solution").
%     IK fitness (Eq. 13, p.3), position only, Euclidean distance between the expected position p'
%     and the actual position p:
%       Error = sqrt((p'_x - p_x)^2 + (p'_y - p_y)^2 + (p'_z - p_z)^2)
%
%   Parameters:
%     N = 25, T = 500  the IK setting nearest the budget. The IK runs of Table V (p.4) use N = 25,
%                     50, 100 and 150 with 500 iterations: N x T = 12,500, 25,000, 50,000 and
%                     75,000; 25 x 500 is the nearest to 2000. The benchmark-function runs (p.3)
%                     use N = 100, T = 500. Inside the benchmark T is the budget horizon.
%     Nr, Nh, Nc = 0.2N, 0.6N, 0.2N = 5, 15, 5   p.3 ("the ratios of rooster, hen and chick are
%                     0.2, 0.6 and 0.2"); p.4 does not restate them for the IK runs.
%     Nm              no mother-hen subset: the mother of a chick is drawn from all hens (p.2, "the
%                     followers of the chicks are randomly selected from the hens"). The Nm = 0.1N
%                     of W22 Table 2 (p.8) belongs to W22's CSO comparator.
%     G = 10          p.3.
%     FL = 0.7        not stated in the paper (checked pages 1-4); Eq. 10 uses the standard CSO
%                     coefficient. W22 defines FL in [0, 2] (p.5) and runs its CSO comparator with
%                     FL in [0.4, 1] (Table 2, p.8; benchmark functions, no CSO IK run). No single
%                     value is reported, so the midpoint of the range used in runs, [0.4, 1], is
%                     used (benchmark rule for a range).
%     k (Eq. 2)       a rooster other than i, uniform (W22 p.4: "k in [1, Nr], k ~= i"); the source
%                     uses k without defining it (p.2).
%     Eq. 2 condition f_i <= f_k (W22 Eq. 8, p.4); the source prints "f_k - f_i" with no operator.
%     eps             "a minimum constant in the computer" (W22 p.4): realmin. With Eq. 13 >= 0 it
%                     only guards the divisions of Eqs. 2 and 4 against zero.
%     randn(0, sigma^2)   one Gaussian draw with mean 0 and standard deviation sigma^2 (W22 p.4: "a
%                     standard deviation of sigma^2"); the source gives no definition.
%     Draw sizes      rand(1,dim) of Eqs. 8 and 10 is one U(0, 1) per joint. randn in Eq. 7 and
%                     rand in Eq. 9 are printed without a size, unlike rand(1,dim) in the same
%                     Table I lines, so each is one scalar draw per chicken and term.
%     Eq. 4           numerator printed "f_1 - f_r1", read as f_i - f_r1 (W22 Eq. 10, p.4); the
%                     denominator uses f_i.
%     Partners        Eq. 9: x_k1,a is the rooster the hen follows, drawn when the hierarchy is
%                     established ("the followers of the hens are randomly selected from the
%                     roosters", pp.1-2); the same x_k1,a enters e (Eq. 8). x_k1,b of Eq. 8 ("the
%                     randomly selected hen") is drawn from all hens in every update. x_k2,b is a
%                     hen other than i drawn in every update (p.2: "The hen competes with other
%                     individuals"; W22 Algorithm 2: "competes with another hen"). Eq. 10: x_r3,b
%                     is the chick's mother and x_r1,a the rooster the mother follows ("the
%                     corresponding rank relationship of rooster and hen", p.2).
%     Hierarchy at t = 1   Table I re-establishes it only when mod(t, G) = 0, which leaves the roles
%                     undefined for t < G; W22 Algorithm 2 (p.5) also establishes it at t = 1,
%                     which is used. Ties in the ranking keep the index order.
%     Rooster rule    Table I prints the Eq. 1 random walk on its rooster line; the text (Sec.
%                     II.B.1, p.2), the abstract (p.1) and the conclusion (p.4) give Eq. 7 for ICSO,
%                     which is used.
%     Units           metres: the paper's robot (Table IV, p.3) gives a_i and d_i in metres, so
%                     Eq. 13 is in metres, the benchmark's unit. S2 = exp(f_r2 - f_i) depends on it.
%     Initial population   "Initialize ICSO algorithm parameters; Evaluate the N chickens' fitness
%                     values" (Table I); the distribution is not stated in the paper (checked pages
%                     1-4); uniform in [q_min, q_max].
%     Stop            maxgen iterations (Table I); no tolerance is stated (checked pages 1-4).
%     Bounds          the search range and the handling of out-of-range values are not stated in
%                     the paper (checked pages 1-4).
%
%   Benchmark interface:
%     Budget: the initial population costs N = 25 FK calls and every iteration N = 25 (every chicken
%       moves and is evaluated). T = ikb_generations(opts, 25, 25, 500) = 79 iterations, so
%       25 + 79 x 25 = 2000 FK per waypoint. A 40-member population would give
%       ikb_generations(opts, 40, 40, 500) = 49. No quantity of the algorithm varies with T, so T
%       only sets the number of iterations. The initial population and every iteration start only
%       if ikb_budget_left(opts) covers N. Outside the benchmark T = 500 (12,525 FK). No final FK
%       check. Every option field is ignored (the budget is read only through ikb_generations and
%       ikb_budget_left).
%     Stop: T iterations. info.converged is always false: the paper has no convergence criterion.
%     Warm start: q0 (clamped to the limits) is chicken 1 of the initial population; the other 24
%       are uniform in [q_min, q_max]. q0 takes part in the first ranking (t = 1), and the recorded
%       best starts from the initial population, so q0 is returned unless a later evaluation has a
%       strictly lower Eq. 13 error.
%     Robot and units: Eqs. 11-12 and Table IV (the paper's PUMA 560) are replaced by robot.fk with
%       dim = robot.n joints; positions in metres, joint angles in radians.
%     Joint limits: every new position is clamped to [q_min, q_max] before its evaluation (not
%       stated in the paper). Eq. 10 adds FL x_c to a point between rooster and hen, so chick
%       coordinates often leave the range and are clamped; info.clamped counts them.
%     Output: the recorded best; info.fitness (Eq. 13, m), info.err_pos (m) and info.err_rot
%       (rotation angle, rad, reported only) from the pose stored at its evaluation.
%       solve_CSO('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration horizon T = 79 instead of 500 (benchmark rule).
%     - q0 is a member of the initial population (warm start).
%     - Positions clamped to [q_min, q_max] (bounds not stated).
%     - Hierarchy also established at t = 1 (Table I leaves the roles undefined for t < G).
%     - Eq. 7 used for the roosters where Table I prints Eq. 1 (see Rooster rule).
%     - Taken from W22 because the source does not state them: k ~= i and the f_i <= f_k condition
%       of Eq. 2, eps, the standard deviation sigma^2 of randn, and FL (midpoint of [0.4, 1]).
%     - Not stated in the paper: the initial distribution, the scalar draws of Eqs. 7 and 9, the
%       partner readings of Eqs. 8-10, the f_i of Eq. 4's numerator, ties in the ranking.
%     - robot.fk instead of Eqs. 11-12 (the paper's arm).
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
        q_best = struct('params', icso_params(), 'core', @icso_core, 'counts', @icso_counts, ...
                        'hierarchy', @icso_hierarchy, 'sigma2', @icso_sigma2, ...
                        'rooster', @icso_rooster, 'hen', @icso_hen, 'chick', @icso_chick, ...
                        'fitness', @icso_fitness);
        info = [];
        return
    end
    [q_best, info] = icso_core(robot, T_d, q0, opts, icso_params());
end

% ==================================================================== main loop (Table I)
function [q_best, info] = icso_core(robot, T_d, q0, opts, P)
% Initial population (q0 first), then per iteration: hierarchy (t = 1 and mod(t, G) = 0), moves of
% Eqs. 7-10 from the positions and fitness of iteration t, clamp, evaluation (Eq. 13), record of
% the best.
    t_start = tic;
    N  = P.N;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    n  = numel(lo);
    q0 = min(max(q0(:), lo), hi);
    T  = ikb_generations(opts, N, N, P.T);
    pd = T_d(1:3, 4);
    [Nr, Nh] = icso_counts(P);

    bf = NaN;  bq = q0;  bT = NaN(4);  fk = 0;  iters = 0;
    X = zeros(n, 0);  F = zeros(1, 0);
    hist = zeros(1, 0);  regroup = false(1, 0);  role = zeros(N, 0);  fol = zeros(N, 0);
    mot = zeros(N, 0);  clamped = zeros(3, 0);

    if ikb_budget_left(opts) >= N
        X = [q0, lo + rand(n, N - 1) .* (hi - lo)];                    % Table I, q0 first
        [F, bf, bq, bT] = evaluate(robot, pd, X, bf, bq, bT);
        fk = N;
        hist = NaN(1, T);  regroup = false(1, T);  role = zeros(N, T);  fol = zeros(N, T);
        mot = zeros(N, T);  clamped = zeros(3, T);
        H = struct();
        for t = 1:T
            if ikb_budget_left(opts) < N, break; end
            if t == 1 || mod(t, P.G) == 0                               % Table I; W22 Algorithm 2
                H = icso_hierarchy(F, P);
                regroup(t) = true;
            end
            Xn = X;
            for j = 1:Nr                                                % Eq. 7 with Eq. 2
                i = H.ra(j);
                o = H.ra(H.ra ~= i);
                k = o(randi(Nr - 1));
                Xn(:, i) = icso_rooster(X(:, i), bq, icso_sigma2(F(i), F(k), P.eps));
            end
            for j = 1:Nh                                                % Eqs. 8-9 with Eqs. 4-5
                i  = H.hb(j);
                r1 = H.k1(i);
                kb = H.hb(randi(Nh));
                o  = H.hb(H.hb ~= i);
                k2 = o(randi(Nh - 1));
                Xn(:, i) = icso_hen(X(:, i), X(:, r1), X(:, kb), X(:, k2), F(i), F(r1), F(k2), P.eps);
            end
            for j = 1:numel(H.ch)                                       % Eq. 10
                i = H.ch(j);
                Xn(:, i) = icso_chick(X(:, i), X(:, H.m(i)), X(:, H.k1(i)), P.FL);
            end
            out = Xn < lo | Xn > hi;
            for c = 1:3
                clamped(c, t) = nnz(out(:, H.role == c));
            end
            X = min(max(Xn, lo), hi);                                   % joint limits
            [F, bf, bq, bT] = evaluate(robot, pd, X, bf, bq, bT);
            fk = fk + N;
            iters = t;
            hist(t) = bf;
            role(:, t) = H.role';
            fol(:, t)  = H.k1';
            mot(:, t)  = H.m';
        end
        hist = hist(1:iters);  regroup = regroup(1:iters);  role = role(:, 1:iters);
        fol = fol(:, 1:iters);  mot = mot(:, 1:iters);  clamped = clamped(:, 1:iters);
    end

    ep = NaN;  er = NaN;
    if ~isnan(bf)
        ep = norm(bT(1:3, 4) - T_d(1:3, 4));
        er = acos(min(1, max(-1, (trace(T_d(1:3, 1:3).' * bT(1:3, 1:3)) - 1) / 2)));
    end

    q_best = bq;
    info = struct();
    info.solver    = 'CSO';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = false;                                           % no criterion in the paper
    info.fitness   = bf;                                              % Eq. 13, m
    info.err_pos   = ep;                                              % m
    info.err_rot   = er;                                              % rad, reporting only
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.N         = N;
    info.T         = T;
    info.history   = hist;                                            % recorded best after iteration t
    info.regroup   = regroup;                                         % hierarchy established at t
    info.role      = role;                                            % 1 rooster, 2 hen, 3 chick
    info.follow    = fol;                                             % rooster of a hen or chick (x_k1,a, x_r1,a)
    info.mother    = mot;                                             % mother of a chick (x_r3,b)
    info.clamped   = clamped;                                         % clamped coordinates per role
    info.final     = struct('X', X, 'F', F);                          % last population and Eq. 13
    info.params    = P;
end

function [F, bf, bq, bT] = evaluate(robot, pd, X, bf, bq, bT)
% Eq. 13 for every chicken (one FK call each); the recorded best is replaced only by a strictly
% lower error ("Record the optimal value", Table I).
    N = size(X, 2);
    F = zeros(1, N);
    for i = 1:N
        [F(i), Tq] = icso_fitness(robot, pd, X(:, i));
        if isnan(bf) || F(i) < bf
            bf = F(i);
            bq = X(:, i);
            bT = Tq;
        end
    end
end

% ==================================================================== operators
function P = icso_params()
% p.3: ratios 0.2 / 0.6 / 0.2, G = 10; p.4 (Table V): N = 25 (setting nearest the budget), 500
% iterations; FL = 0.7, midpoint of [0.4, 1] (W22 Table 2, p.8); eps = realmin (W22 p.4).
    P = struct('N', 25, 'T', 500, 'rr', 0.2, 'rh', 0.6, 'rc', 0.2, 'G', 10, 'FL', 0.7, ...
               'eps', realmin);
end

function [Nr, Nh, Nc] = icso_counts(P)
% Nr = 0.2N, Nh = 0.6N, Nc = N - Nr - Nh (p.3; W22 Table 2).
    Nr = round(P.rr * P.N);
    Nh = round(P.rh * P.N);
    Nc = P.N - Nr - Nh;
end

function H = icso_hierarchy(F, P)
% Sec. II.A.1 (pp.1-2), Table I: rank by fitness (ascending, ties in index order); the best Nr
% are roosters, the next Nh hens, the rest chicks. Each hen follows a rooster drawn uniformly from
% the roosters (in rank order of the hens), then each chick gets a mother drawn uniformly from the
% hens (in rank order of the chicks); the rooster of a chick is the rooster its mother follows.
    [Nr, Nh] = icso_counts(P);
    N = numel(F);
    [~, ord] = sort(F, 'ascend');
    H.ra = ord(1:Nr);
    H.hb = ord(Nr + 1:Nr + Nh);
    H.ch = ord(Nr + Nh + 1:N);
    H.role = zeros(1, N);
    H.role(H.ra) = 1;
    H.role(H.hb) = 2;
    H.role(H.ch) = 3;
    H.k1 = zeros(1, N);
    H.m  = zeros(1, N);
    for j = 1:Nh
        H.k1(H.hb(j)) = H.ra(randi(Nr));
    end
    for j = 1:numel(H.ch)
        i = H.ch(j);
        H.m(i)  = H.hb(randi(Nh));
        H.k1(i) = H.k1(H.m(i));
    end
end

function s2 = icso_sigma2(fi, fk, ep)
% Eq. 2 with the condition of W22 Eq. 8: 1 if f_i <= f_k, otherwise exp((f_k - f_i) / |f_i + eps|).
    if fi <= fk
        s2 = 1;
    else
        s2 = exp((fk - fi) / abs(fi + ep));
    end
end

function x = icso_rooster(x, xb, s2)
% Eq. 7: x + randn(0, sigma^2) (x_best - x), one Gaussian draw with standard deviation sigma^2
% (W22 p.4).
    x = x + (s2 * randn) * (xb - x);
end

function x = icso_hen(x, xr, xe, xh, fi, fr, fh, ep)
% Eq. 8: e = rand(1,dim) (x_k1,a - x_k1,b); Eqs. 4-5: S1 = exp((f_i - f_r1) / |abs(f_i) + eps|),
% S2 = exp(f_r2 - f_i); Eq. 9: x + S1 rand (x_k1,a - x) + S2 rand (x_k2,b - x) + e, with two scalar
% draws after the dim draws of e.
    e  = rand(numel(x), 1) .* (xr - xe);
    S1 = exp((fi - fr) / abs(abs(fi) + ep));
    S2 = exp(fh - fi);
    u1 = rand;
    u2 = rand;
    x  = x + S1 * u1 * (xr - x) + S2 * u2 * (xh - x) + e;
end

function x = icso_chick(x, xm, xr, FL)
% Eq. 10: FL x_c + rand(1,dim) (x_r3,b - x_r1,a) + x_r1,a, x_r3,b the mother, x_r1,a its rooster.
    x = FL * x + rand(numel(x), 1) .* (xm - xr) + xr;
end

% ==================================================================== IK fitness (Eq. 13)
function [f, T] = icso_fitness(robot, pd, q)
% Eq. 13: Euclidean distance between the expected position pd and the actual position (m); one FK
% call.
    T = robot.fk(q);
    f = sqrt((pd(1) - T(1, 4))^2 + (pd(2) - T(2, 4))^2 + (pd(3) - T(3, 4))^2);
end
