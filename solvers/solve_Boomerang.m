function [q_best, info] = solve_Boomerang(robot, T_d, q0, opts)
% SOLVE_BOOMERANG  Boomerang algorithm IK solver of Duymazlar and Engin (2023), Secs. 2.2-2.3.
%
%   Source:
%     Okan Duymazlar, Dilşad Engin (2023). Boomerang Algorithm based on Swarm Optimization for
%     Inverse Kinematics of 6 DOF Open Chain Manipulators. Turkish Journal of Electrical
%     Engineering and Computer Sciences, 31(2), 342-359 (Article 8). DOI: 10.55730/1300-0632.3988
%   Page numbers are PDF pages 1-19 (page 1 is the publisher's cover; printed page = PDF page + 340).
%
%   Algorithm (Sec. 2.2, p.7; Fig. 2a, p.8; Sec. 2.3, Eqs. 21-33, pp.7-9; settings Sec. 3.2, p.11):
%     A PSO variant whose total number of iterations T is divided into K throws ("Subiteration =
%     Total Iteration / Number of Throw", Fig. 2a). Each throw draws a new random angle set of N
%     searchers inside the joint limits and evaluates it. A throw "eliminates all joint values"
%     (p.7), unlike RPSO, which "keeps the best particle of former iteration" (p.7), so the personal
%     bests Lb and the global best Gb restart from the new set. Each iteration of the throw moves
%     every searcher and evaluates it (Fig. 2a: compute changes, update, exclude angles over the
%     limits, calculate position and orientation):
%       dtheta(i+1) = w dtheta(i) + rn(0:2) (Lb - theta) + rn(0:2) (Gb - theta)     (Eq. 27)
%       w = 1 - lambda/1000                                                          (Eq. 28)
%       dtheta(i+1) <= Dmin -> Dmin;  dtheta(i+1) >= Dmax -> Dmax                   (Eqs. 29-30)
%       Dmax = (theta_max - theta_min)/(5 xi),  Dmin = (theta_max - theta_min)/(-5 xi) (Eqs. 25-26)
%       theta(i+1) = theta(i) + dtheta(i+1)                                          (Eq. 31)
%       theta(i+1) >= theta_max -> theta_max rs(0:1);  theta(i+1) <= theta_min -> theta_min rs(0:1)
%                                                                                    (Eqs. 32-33)
%     Lb is each searcher's best position and Gb the best position of all searchers (p.8), both
%     counted from the start of the throw (p.7).
%     After every evaluation of the set the run stops if "Error <= Tolerance" (Fig. 2a); otherwise
%     it ends after the K throws.
%     Fitness (Eq. 24, p.7): fitness = || 60 dP + 40 dRPY ||_2, with dP the absolute differences of
%     x, y, z and dRPY those of Roll, Pitch, Yaw between the current and the targeted values ("the
%     absolute difference", p.4; "Error = |Current - Targeted|", Fig. 2a). Eqs. 21-23 (p.7), in
%     this order:
%       Roll = atan2(r21, r11),  Pitch = atan2(-r31, sqrt(r32^2 + r33^2)),  Yaw = atan2(r32, r33)
%
%   Parameters:
%     N = 10, K = 8, T = 5500   Sec. 3.2 (p.11): 10 searchers, 8 throws and 5500 total iterations,
%                     the only setting stated (N x T = 55,000). Fig. 6 (p.11) shows the tuning
%                     surfaces (one searcher with 1-10 throws and 1000-6000 total iterations; 1-10
%                     throws with 0-50 searchers) with optimum points marked but no values given, and
%                     Figs. 3-4 (p.9) a 2-DOF planar example without settings, so no other IK
%                     setting is reported. Inside the benchmark T is the budget horizon.
%     S_k             iterations of throw k. T/K = 687.5 (Fig. 2a), rounding not stated; the T
%                     iterations are split as equally as possible with the total kept:
%                     S_k = floor(k T/K) - floor((k-1) T/K) = 687, 688, 687, 688, ... (sum 5500).
%     xi = N = 10     p.7: "For a standard PSO algorithm xi variable corresponds to the number of
%                     particles"; Dmax = (theta_max - theta_min)/50 per joint, Dmin = -Dmax.
%     rn(0:2)         2 U(0, 1), drawn per searcher and joint, separately for the two terms. p.8
%                     gives the range only; distribution and granularity not stated.
%     lambda          the iteration count s = 1..S_k inside the current throw. p.8 calls lambda
%                     "total iteration number" and does not define it further (checked pages 7-9).
%                     Read as the constant total (w = 1 - 5500/1000 = -4.5) or as a count over
%                     all throws (w < 0 after iteration 1000), the median orientation error on
%                     the paper's IRB 120 targets (Table 1, p.13) exceeds ten times the paper's
%                     mean of 0.0761 degrees (Table 3, p.14); with the count inside the throw it
%                     is of that order.
%     w               Eq. 28 with lambda as above: from 0.999 to 0.313 (687 iterations) or 0.312
%                     (688) in every throw, positive throughout, so no lower floor is applied.
%     dtheta(0) = 0   initial change of a new angle set not stated in the paper (checked pages
%                     7-9); zero is used at the start of every throw.
%     Units of Eq. 24 not stated (checked pages 4, 7-8, 13). The paper's targets and errors are in
%                     mm and degrees (Table 1 and text, p.13; Tables 2-3, p.14), so dP is in mm and
%                     dRPY in degrees.
%     Tolerance       Fig. 2a stops at "Error <= Tolerance"; no value is stated (checked pages
%                     7-17; the 0.225 mm of Sec. 4.2, pp.15-16, is the fit of the assembled parts).
%                     Error is read as the fitness of Eq. 24 and the tolerance as
%                     eps = min(60000 tol_pos, 7200 tol_rot/(pi sqrt(3))): fitness <= eps implies
%                     |dP| <= fitness/60 mm <= tol_pos (m) and a rotation angle <= sqrt(3)
%                     fitness/40 degrees <= tol_rot (rad), the benchmark's success test (the
%                     rotation angle is at most the sum of the three angle differences).
%
%   Benchmark interface:
%     Budget: each throw evaluates its random set (N = 10 FK) and each iteration the moved set (N
%       FK). T = ikb_generations(opts, K N, N, 5500) = ikb_generations(opts, 80, 10, 5500) = 192
%       iterations, 24 per throw: 8 throws x (10 + 24 x 10) = 2000 FK per waypoint. A 40-member
%       population would give ikb_generations(opts, 320, 40, 5500) = 42. lambda runs on this
%       horizon, lambda = s 5500/T, so in every throw w falls from 0.971 to 0.3125 = 1 - 687.5/1000,
%       the course of a throw of the paper's run. A throw and an iteration start only if
%       ikb_budget_left(opts) covers N. Outside the benchmark T = 5500: 8 throws of 687/688 iterations, 55,080 FK. No final FK
%       check. Fields of opts other than tol_pos and tol_rot (max_iter, pop_size, ...) are ignored.
%     Stop: the tolerance above, tested after every evaluation of the set; info.converged = best
%       fitness <= eps.
%     Warm start: q0 (clamped to the limits) is searcher 1 of the first throw's random set,
%       evaluated unperturbed; the other searchers and all later sets are uniform in
%       [q_min, q_max] (distribution not stated; "no limitations for reinitialization of random
%       angle sets except physical joint limits", p.7). The best set evaluated over all throws is
%       kept apart from Gb and returned; it is replaced only by a strictly lower fitness, so q0 is
%       returned unless a searcher improves on it.
%     Robot and units: the OPW closed-form FK of Eqs. 8-20 (6-DOF ortho-parallel arms with a
%       spherical wrist, Sec. 2.1, p.5) is replaced by robot.fk, with n = robot.n joints. Both give
%       the same pose, so only the cost of an evaluation differs (a realisation change). The target
%       Roll, Pitch, Yaw come from T_d by Eqs. 21-23 (the paper takes them from Table 1).
%       Benchmark lengths are metres and angles radians; Eq. 24 uses dP x 1000 (mm) and
%       dRPY x 180/pi (degrees).
%     Joint limits: Eqs. 32-33 on the benchmark limits [q_min, q_max]. They assume theta_min < 0 <
%       theta_max: "a random value assignment is made within the physical joint limits" (p.8).
%       For a range that excludes 0 (Panda joint 4, [-3.0718, -0.0698]) theta_lim rs(0:1) can fall
%       outside it; the draw is then taken between the violated limit and c, the point of the range
%       nearest 0: theta = c + rs (theta_lim - c). This equals theta_lim rs when the range contains
%       0 (c = 0). For Panda joint 4 an upper violation gives theta_max and a lower one a uniform
%       value in the range. dtheta is kept: the paper reassigns only the joint value (pp.8-9).
%     Output: info.fitness (Eq. 24, mm and degrees), info.err_pos (m) and info.err_rot (rotation
%       angle, rad) of the returned set, from the pose stored at its evaluation.
%       solve_Boomerang('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration horizon T = 192 (24 per throw) instead of 5500 (benchmark rule); lambda is
%       scaled onto it.
%     - lambda read as the iteration count inside the throw (not defined; see Parameters).
%     - Eq. 24 in mm and degrees (units not stated; see Parameters).
%     - Angle differences taken modulo 2 pi, in [-pi, pi), before the absolute value. A plain
%       difference makes equal orientations differ by 2 pi when an angle sits at +/-pi (atan2 of a
%       signed zero), which contradicts "orientation error" (p.7).
%     - Error of Fig. 2a read as the fitness; tolerance value from the benchmark tolerances (not
%       stated; see Parameters).
%     - Eqs. 32-33 generalised to ranges that exclude 0 (see Joint limits).
%     - Not stated in the paper: rn(0:2) per searcher and joint, dtheta(0) = 0, uniform random
%       sets, the rounding of T/K.
%     - Each throw evaluates its random set once and then runs S_k moves, each followed by an
%       evaluation; Fig. 2a counts iterations at the "Iteration <= Subiteration" test without
%       saying where the count starts.
%     - Lb and Gb are updated after all N evaluations of an iteration (Fig. 2a order); a tie keeps
%       the earlier best (not stated).
%     - robot.fk instead of Eqs. 8-20 (same poses; see Robot and units).
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
        q_best = struct('params', boom_params(), 'core', @boom_core, 'split', @boom_split, ...
                        'inertia', @boom_inertia, 'dlimits', @boom_dlimits, ...
                        'velocity', @boom_velocity, 'clampv', @boom_clampv, ...
                        'limits', @boom_limits, 'rpy', @boom_rpy, 'fitness', @boom_fitness, ...
                        'tolerance', @boom_tolerance);
        info = [];
        return
    end
    [q_best, info] = boom_core(robot, T_d, q0, opts, boom_params());
end

% ==================================================================== main loop (Fig. 2a)
function [q_best, info] = boom_core(robot, T_d, q0, opts, P)
% K throws; each evaluates a random angle set and runs S_k iterations of Eqs. 27-33, with the
% tolerance test after every evaluation. P holds N, K, T and the Eq. 24 constants.
    t_start = tic;
    N  = P.N;
    K  = P.K;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    n  = numel(lo);
    q0 = min(max(q0(:), lo), hi);
    T  = ikb_generations(opts, K * N, N, P.T);
    S  = boom_split(T, K);
    [Dmin, Dmax] = boom_dlimits(lo, hi, N);                         % Eqs. 25-26, xi = N
    tol   = boom_tolerance(opts, P);
    rpy_d = boom_rpy(T_d(1:3, 1:3));                                % Eqs. 21-23, target

    B = struct('f', NaN, 'q', q0, 'T', NaN(4), 'nfk', 0);           % best set of all throws
    hist = zeros(1, T);  t_start_f = NaN(1, K);  t_end_f = NaN(1, K);
    iters = 0;  throws = 0;  converged = false;  out = false;
    fin = struct('X', [], 'V', [], 'Lb', [], 'Fl', [], 'Gb', [], 'Fg', NaN);

    for k = 1:K
        if ikb_budget_left(opts) < N, break; end
        X = lo + rand(n, N) .* (hi - lo);                           % random angle set (Fig. 2a)
        if k == 1, X(:, 1) = q0; end
        V = zeros(n, N);                                            % dtheta(0), not stated
        [F, B] = eval_set(robot, T_d, rpy_d, X, B, P);
        Lb = X;  Fl = F;                                            % Lb, Gb restart (p.7)
        [Fg, ig] = min(F);
        Gb = X(:, ig);
        throws = k;
        t_start_f(k) = Fg;
        converged = B.f <= tol;                                     % Fig. 2a tolerance test
        s = 0;
        while ~converged && s < S(k)
            if ikb_budget_left(opts) < N, out = true; break; end
            s = s + 1;
            w = boom_inertia(s, T, P);                              % Eq. 28
            V = boom_velocity(V, X, Lb, Gb, w);                     % Eq. 27
            V = boom_clampv(V, Dmin, Dmax);                         % Eqs. 29-30
            X = X + V;                                              % Eq. 31
            X = boom_limits(X, lo, hi);                             % Eqs. 32-33
            [F, B] = eval_set(robot, T_d, rpy_d, X, B, P);
            up = F < Fl;                                            % Lb (p.8)
            Lb(:, up) = X(:, up);
            Fl(up)    = F(up);
            [fm, im] = min(F);                                      % Gb (p.8)
            if fm < Fg
                Fg = fm;
                Gb = X(:, im);
            end
            iters = iters + 1;
            hist(iters) = B.f;
            converged = B.f <= tol;
        end
        t_end_f(k) = Fg;
        fin = struct('X', X, 'V', V, 'Lb', Lb, 'Fl', Fl, 'Gb', Gb, 'Fg', Fg);
        if converged || out, break; end
    end

    ep = NaN;  er = NaN;
    if ~isnan(B.f)
        ep = norm(B.T(1:3, 4) - T_d(1:3, 4));
        er = acos(min(1, max(-1, (trace(T_d(1:3, 1:3).' * B.T(1:3, 1:3)) - 1) / 2)));
    end

    q_best = B.q;
    info = struct();
    info.solver      = 'Boomerang';
    info.iters       = iters;
    info.time        = toc(t_start);
    info.converged   = converged;                                   % fitness <= eps (Fig. 2a)
    info.fitness     = B.f;                                         % Eq. 24, mm and degrees
    info.err_pos     = ep;                                          % m
    info.err_rot     = er;                                          % rad
    info.fk_calls    = B.nfk;
    info.jac_calls   = 0;
    info.N           = N;
    info.K           = K;
    info.T           = T;
    info.S           = S;
    info.throws      = throws;
    info.tol         = tol;
    info.history     = hist(1:iters);                               % best fitness per iteration
    info.throw_start = t_start_f;                                   % best of each random set
    info.throw_best  = t_end_f;                                     % Gb fitness at throw end
    info.final       = fin;
    info.params      = P;
end

function [F, B] = eval_set(robot, T_d, rpy_d, X, B, P)
% Calculate position and orientation and the error of every searcher (Fig. 2a); one FK call each.
% B keeps the best set of all throws, replaced only by a strictly lower fitness.
    N = size(X, 2);
    F = zeros(1, N);
    for i = 1:N
        [F(i), Tq] = boom_fitness(robot, T_d, rpy_d, X(:, i), P);
        B.nfk = B.nfk + 1;
        if isnan(B.f) || F(i) < B.f
            B.f = F(i);
            B.q = X(:, i);
            B.T = Tq;
        end
    end
end

% ==================================================================== operators
function P = boom_params()
% Sec. 3.2 (p.11): N = 10 searchers, K = 8 throws, T = 5500 total iterations; Eq. 24 (p.7):
% weights 60 and 40; dP in mm, dRPY in degrees (p.13).
    P = struct('N', 10, 'K', 8, 'T', 5500, 'w_pos', 60, 'w_rot', 40, ...
               'mm_per_m', 1000, 'deg_per_rad', 180 / pi);
end

function S = boom_split(T, K)
% Fig. 2a: Subiteration = Total Iteration / Number of Throw; T split over K throws, equal up to
% one iteration, sum T.
    S = floor((1:K) * T / K) - floor((0:K-1) * T / K);
end

function w = boom_inertia(s, T, P)
% Eq. 28: w = 1 - lambda/1000, lambda = iteration s of the throw on the paper's horizon
% (lambda = s P.T/T; lambda = s when T = P.T).
    w = 1 - (s * P.T / T) / 1000;
end

function [Dmin, Dmax] = boom_dlimits(lo, hi, xi)
% Eqs. 25-26 per joint, xi = number of particles (p.7).
    Dmax = (hi - lo) / (5 * xi);
    Dmin = (hi - lo) / (-5 * xi);
end

function V = boom_velocity(V, X, Lb, Gb, w)
% Eq. 27: w dtheta + rn(0:2) (Lb - theta) + rn(0:2) (Gb - theta); rn = 2 U(0, 1) per searcher
% and joint, one draw for each term.
    R1 = 2 * rand(size(X));
    R2 = 2 * rand(size(X));
    V = w * V + R1 .* (Lb - X) + R2 .* (Gb - X);
end

function V = boom_clampv(V, Dmin, Dmax)
% Eqs. 29-30: dtheta <= Dmin -> Dmin; dtheta >= Dmax -> Dmax (per joint).
    V = min(max(V, Dmin), Dmax);
end

function X = boom_limits(X, lo, hi)
% Eqs. 32-33: theta >= theta_max -> c + rs (theta_max - c); theta <= theta_min -> c + rs
% (theta_min - c); c is the point of [theta_min, theta_max] nearest 0 (c = 0 gives theta_lim rs).
% rs ~ U(0, 1), one draw per reassigned joint value, in column order.
    M  = size(X, 2);
    L  = repmat(lo, 1, M);
    H  = repmat(hi, 1, M);
    C  = repmat(min(max(0, lo), hi), 1, M);
    up = X >= H;
    m  = up | X <= L;
    if any(m(:))
        c   = C(m);
        rs  = reshape(rand(nnz(m), 1), size(c));
        lim = L;
        lim(up) = H(up);
        X(m) = c + rs .* (lim(m) - c);
    end
end

function r = boom_rpy(R)
% Eqs. 21-23: [Roll; Pitch; Yaw] = [atan2(r21, r11); atan2(-r31, sqrt(r32^2 + r33^2));
% atan2(r32, r33)], radians.
    r = [atan2(R(2, 1), R(1, 1)); ...
         atan2(-R(3, 1), sqrt(R(3, 2)^2 + R(3, 3)^2)); ...
         atan2(R(3, 2), R(3, 3))];
end

function [f, T] = boom_fitness(robot, T_d, rpy_d, q, P)
% Eq. 24: || 60 |dP| + 40 |dRPY| ||_2, dP in mm, dRPY in degrees (angle differences modulo 2 pi
% in [-pi, pi) before the absolute value); one FK call.
    T  = robot.fk(q);
    dP = P.mm_per_m * abs(T(1:3, 4) - T_d(1:3, 4));
    dA = boom_rpy(T(1:3, 1:3)) - rpy_d;
    dR = P.deg_per_rad * abs(mod(dA + pi, 2 * pi) - pi);
    f  = norm(P.w_pos * dP + P.w_rot * dR);
end

function e = boom_tolerance(opts, P)
% Fitness threshold that implies the benchmark tolerances: |dP| <= f/60 mm and a rotation angle
% <= sqrt(3) f/40 degrees.
    tp = 1e-6;
    tr = 1e-6;
    if isfield(opts, 'tol_pos'), tp = opts.tol_pos; end
    if isfield(opts, 'tol_rot'), tr = opts.tol_rot; end
    e = min(P.w_pos * P.mm_per_m * tp, P.w_rot * P.deg_per_rad * tr / sqrt(3));
end
