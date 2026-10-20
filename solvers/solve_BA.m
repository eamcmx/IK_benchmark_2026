function [q_best, info] = solve_BA(robot, T_d, q0, opts)
% SOLVE_BA  IK-BA: bat-algorithm IK solver with decoupled position and orientation of Slim et al.
%           (2023), Secs. 4.2-4.3.
%
%   Source:
%     Mohamed Slim, Nizar Rokbani, Bilel Neji, Mohamed Ali Terres, Taha Beyrouthy (2023). Inverse
%     Kinematic Solver Based on Bat Algorithm for Robotic Arm Path Planning. Robotics, 12, 38
%     (article 38, 26 pp.; no issue number printed). DOI: 10.3390/robotics12020038
%
%   Algorithm (Sec. 4.2, pp.6-7, Eqs. 5, 7, 9, 10, 13, 14; Algorithm 2, p.8; Sec. 4.3, pp.7-9,
%   Eqs. 15-17, Fig. 3; fitness p.7 with Eq. 1):
%     Decoupling (Sec. 4.3): the axes of the last three joints meet in one point, the wrist centre
%     O_(n-2). The swarm searches only theta_1..theta_(n-3), so that O_(n-2) reaches
%     P_w = P - d_n*R*[0; 0; 1] (Eq. 15), d_n the distance from O_(n-1) to O_n along z_(n-1).
%     The fitness is the position error of Eq. 1 between O_(n-2) and P_w (p.7). For the best
%     theta_1..theta_(n-3), 0R_(n-3) (Eq. 16) and (n-3)R_n = 0R_(n-3)' * 0R_n (Eq. 17) give
%     theta_(n-2), theta_(n-1), theta_n in closed form (Fig. 3).
%     IK-BA (Algorithm 2): one loudness A and one pulse rate r for all bats (p.6). Every
%     iteration first applies A <- alpha*A (Eq. 9) and r = r0*(1 - exp(-gamma*t)) (Eq. 10),
%     then for each bat i in turn:
%       f_i = f_min + (f_max - f_min)*beta,  beta ~ U(0, 1)          (Eq. 5)
%       V_i <- V_i + (X_i - (X_best + Theta_s)/2)*f_i                  (Eq. 13)
%       X_i <- X_i + V_i, evaluated                                    (Eq. 7)
%       X_new = r*X_best + (1 - r)*Theta_s + eps*A, evaluated          (Eq. 14)
%       if rand < A and fit(X_new) < fit(X_best): X_i <- X_new          (Alg. 2, acceptance)
%       if fit(X_i) < fit(X_best): X_best <- X_i                        (Alg. 2, rank and update)
%     Eq. 14 runs at every iteration: the (r_i < rand) condition of Algorithm 1 is removed (p.7).
%     Theta_s is the initial configuration (Alg. 2 input), here the warm start q0.
%
%   Reading of Algorithm 2 (p.8):
%     The global move (Eqs. 5, 13, 7) and the local solution (Eq. 14) are both listed for every
%     bat at every iteration, with no condition. The listing then says "evaluate the new solution"
%     once and ranks the bats by their fitness. Both the moved bat (Eq. 7) and X_new (Eq. 14) are
%     evaluated: two FK calls per bat and iteration. X_new does not depend on X_i, so evaluating
%     X_new alone would leave Eqs. 5, 13 and 7 without effect on any evaluated point, and the bats
%     moved by Eq. 7 could not be ranked by fitness.
%
%   Parameters:
%     N = 12        bats. The paper reports several IK settings: N = 50 with t_max = 200 for the
%                   path tracking test (Sec. 5.2, p.13; p.20 says the same test "is based on 20
%                   agents"), and N = 12, 15, 20 and 30 bats in the time assessment (Sec. 5.3.1,
%                   p.19, Table 7, p.20), where IK-BA suits real-time path planning "when the number
%                   of agents is about 12 or less" and "the most effective number of bats is the one
%                   with 12 agents" (p.20). Table 1 (p.11) gives no N. The setting whose N x t_max
%                   is nearest the 2000-FK budget is used: 12 x 200 = 2400.
%     t_max = 200   p.13 (repeated p.20). Outside the benchmark only (see Benchmark interface).
%                   The Table 7 runs stop at eps_p = 0.1 cm instead (p.20); the benchmark keeps the
%                   t_max rule of Sec. 5.2 and only reports the 0.1 cm criterion.
%     [f_min, f_max] = [0, 2], alpha = 0.97, gamma = 0.1, A0 = 1, r0 = 1: Table 1, p.11. With
%                   f_i >= 0, Eqs. 13 and 7 move a bat away from (X_best + Theta_s)/2; the
%                   equations are implemented as printed.
%     beta          U(0, 1) (p.6), one draw per bat and iteration: p.6 calls beta a vector but
%                   assigns each bat "a random frequency f_i".
%     eps           "random number in the range of [-1, 1] generated from a normal distribution"
%                   (p.6, Eq. 8, reused by Eq. 14): a standard normal truncated to [-1, 1] (drawn
%                   by redrawing values outside it), one draw per joint. The standard deviation
%                   is not stated (pp.5-8, Table 1 checked).
%     V(0) = 0      initial velocity not stated (pp.5-8, Table 1 checked).
%     Loop counter  t = 0, 1, ... ("while t < t_max", Alg. 2), so the k-th iteration uses
%                   A = alpha^k and r = r0*(1 - exp(-gamma*(k - 1))); the start value of t is not
%                   printed (pp.6-8 checked).
%     Stop and convergence: Sec. 5.2 runs t_max iterations with no error-based stop (pp.13, 20),
%                   so the run uses its whole iteration horizon. info.converged reports the
%                   paper's acceptable positional error eps_p < 0.1 cm (Sec. 5.3.1, p.20).
%     Wrist joints  Eqs. 16-17 give two solutions of (n-3)R_n (sign of theta_(n-1)) and their
%                   2*pi equivalents; the paper does not say which is used (pp.7-9). The one
%                   inside the joint limits with the least variation from Theta_s (Eq. 12, p.7)
%                   is returned. At a wrist singularity theta_(n-2) keeps its Theta_s value.
%
%   Benchmark interface:
%     Applicability: Sec. 4.3 requires a spherical wrist. From the DH model (no counted FK) the
%       solver checks at two configurations that the axes of joints n-2, n-1 and n meet in one
%       point within 1e-9 m, that this point is O_(n-2), and that Eq. 15 holds for the
%       end-effector (P - d_n*R*[0; 0; 1] = O_(n-2)). It runs on IRB120, Synthetic9,
%       Synthetic11 and Synthetic13. On UR5 (d5 = 0.09465 m) and Panda (a6 = 0.088 m) the wrist
%       axes miss a common point by 47 mm and 45 mm at the check configurations; there it sets
%       info.not_applicable = true and returns q0 without any FK call.
%     Budget: initialisation needs N FK calls, every iteration 2N; an iteration starts only if
%       ikb_budget_left(opts) covers it. T = ikb_generations(opts, 12, 24, 200) = 82 iterations,
%       1980 FK calls per waypoint (a 40-bat population would give T = 24). Each evaluation is one
%       robot.fk call; its frame transforms give O_(n-2) and 0R_(n-3). The wrist joints are
%       computed in closed form, with no FK call. No final FK check. opts.max_iter and
%       opts.pop_size are ignored. Eqs. 9-10 do not depend on t_max, so A and r follow Table 1's
%       alpha and gamma per iteration: after 82 iterations A = 0.97^82 = 0.082 (0.97^200 = 0.0023
%       after the paper's 200).
%     Warm start: Theta_s = q0 (clamped to the limits). Bat 1 starts at Theta_s(1:n-3) (p.7: "use
%       the initial configuration ... as initial position for at least one bat"), is evaluated
%       unperturbed and enters the first X_best selection; the other bats are uniform in the
%       joint limits (distribution not stated, Alg. 2).
%     Units: the paper's targets and errors are in cm (p.13 helix C, r, l; Table 3, p.17; p.20);
%       Table 2 (p.12) gives no numeric lengths. fit = 100*||O_(n-2) - P_w||[m], in cm. A single
%       position term carries no weighting, so the unit changes no comparison of the search.
%     Joint limits: Eq. 4 bounds Theta (p.5) without a handling rule; every X_i (Eq. 7) and X_new
%       (Eq. 14) is clamped to [q_min, q_max]; V is not changed by the clamp. A wrist joint with no
%       2*pi equivalent inside its limits is clamped (info.wrist_clamped).
%     Output: arm joints of the lowest fit ever evaluated, wrist joints from Eqs. 16-17.
%       info.fitness is that fit (cm); info.err_wrist the same error in m. solve_BA('ops') returns
%       handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration count = budget horizon T = 82 instead of t_max = 200 (benchmark rule).
%     - N = 12 (Table 7, p.20) combined with the t_max rule of the path tracking test (p.13), by the
%       benchmark's rule for papers with several IK settings.
%     - Two evaluations per bat and iteration (reading of Algorithm 2, see above).
%     - Eq. 13 writes X_i^t on its right-hand side, which Eq. 7 defines from V_i^t; the bat's
%       current position X_i^(t-1) is used.
%     - eps truncated to [-1, 1] and drawn per joint (a scalar added to a joint vector is
%       undefined); beta one draw per bat.
%     - X_best is the best bat position found so far ("update the best solution", Alg. 2); a bat
%       that moves away by Eq. 7 does not take X_best with it.
%     - "Select a solution among the best solutions X_best" (Alg. 2) defines no set; the single
%       X_best is used, as Eq. 14 writes X_gbest.
%     - Returned arm joints = the lowest fit ever evaluated (benchmark rule). Algorithm 2's X_best
%       can be worse: a better X_new is discarded when rand >= A, and an accepted X_new can
%       replace a better Eq. 7 position of the same bat. info.fitness_xbest reports X_best's fit.
%     - Wrist branch, 2*pi equivalent and singular case chosen as described under Parameters.
%     - Initial population uniform in the joint limits; V(0) = 0; loop counter from 0.
%     - Eq. 3 (orientation error, p.5) is not part of the search: in Sec. 4.3 the orientation is
%       set by Eqs. 16-17. It is provided as an operator for checks.
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
        q_best = struct('params', ba_params(), 'geometry', @ba_wrist_geometry, ...
                        'wrist_point', @ba_wrist_point, 'fitness', @ba_fitness, ...
                        'wrist', @ba_wrist, 'global_move', @ba_global, 'local', @ba_local, ...
                        'epsilon', @ba_epsilon, 'eq3', @ba_eq3);
        info = [];
        return
    end

    t_start = tic;
    P = ba_params();
    [applicable, W] = ba_wrist_geometry(robot);

    info = struct();
    info.solver = 'IK-BA';
    if ~applicable                                       % Sec. 4.3: spherical wrist required
        q_best = q0;
        info.iters          = 0;
        info.time           = toc(t_start);
        info.converged      = false;
        info.fitness        = NaN;
        info.not_applicable = true;
        info.fk_calls       = 0;
        info.jac_calls      = 0;
        info.wrist_geometry = W;
        return
    end

    n     = robot.n;
    m     = n - 3;                                       % searched joints (Sec. 4.3)
    N     = P.N;
    q_min = robot.q_min(:);
    q_max = robot.q_max(:);
    ths   = min(max(q0(:), q_min), q_max);               % Theta_s
    lo    = q_min(1:m);
    hi    = q_max(1:m);
    qw0   = ths(m+1:n);                                  % wrist joints during the search
    Pw    = ba_wrist_point(T_d, W.dn);                   % Eq. 15
    R_n   = T_d(1:3, 1:3) * W.tool(1:3, 1:3)';           % 0R_n
    T     = ikb_generations(opts, N, 2 * N, P.t_max);

    B = struct('f', NaN, 'x', ths(1:m), 'R3', [], 'ep', NaN);
    history = zeros(1, 0);  A_t = zeros(1, 0);  r_t = zeros(1, 0);
    iters = 0;  fk = 0;  n_accept = 0;  n_at_limit = 0;  n_move_best = 0;  Fb = NaN;

    if ikb_budget_left(opts) >= N
        X = lo + rand(m, N) .* (hi - lo);  X(:, 1) = ths(1:m);
        V = zeros(m, N);
        F = zeros(1, N);
        for i = 1:N
            [F(i), R3, ep] = ba_fitness(robot, X(:, i), qw0, Pw);  fk = fk + 1;
            B = keep_best(B, F(i), X(:, i), R3, ep);
        end
        [Fb, ib] = min(F);  Xb = X(:, ib);               % X_best

        A = P.A0;
        history = zeros(1, T);  A_t = zeros(1, T);  r_t = zeros(1, T);
        for t = 0:T-1                                    % while t < t_max (Alg. 2)
            if ikb_budget_left(opts) < 2 * N, break; end
            A = P.alpha * A;                             % Eq. 9
            r = P.r0 * (1 - exp(-P.gamma * t));          % Eq. 10
            for i = 1:N
                [X(:, i), V(:, i), hit] = ba_global(X(:, i), V(:, i), Xb, ths(1:m), P, lo, hi);
                n_at_limit = n_at_limit + hit;
                [F(i), R3, ep] = ba_fitness(robot, X(:, i), qw0, Pw);  fk = fk + 1;
                B = keep_best(B, F(i), X(:, i), R3, ep);

                xn = ba_local(Xb, ths(1:m), r, A, lo, hi);                   % Eq. 14
                [fn, R3, ep] = ba_fitness(robot, xn, qw0, Pw);  fk = fk + 1;
                B = keep_best(B, fn, xn, R3, ep);

                accepted = rand() < A && fn < Fb;       % acceptance (Alg. 2)
                if accepted
                    X(:, i) = xn;  F(i) = fn;  n_accept = n_accept + 1;
                end
                if F(i) < Fb                             % rank and update X_best (Alg. 2)
                    Fb = F(i);  Xb = X(:, i);  n_move_best = n_move_best + ~accepted;
                end
            end
            iters = t + 1;
            history(iters) = B.f;  A_t(iters) = A;  r_t(iters) = r;
        end
        history = history(1:iters);  A_t = A_t(1:iters);  r_t = r_t(1:iters);
    end

    clamped = false;
    if isempty(B.R3)
        q_best = ths;
    else
        [qw, clamped] = ba_wrist(B.R3, R_n, W, ths(m+1:n), q_min(m+1:n), q_max(m+1:n));
        q_best = [B.x; qw];                              % Eqs. 16-17
    end

    info.iters          = iters;
    info.time           = toc(t_start);
    info.converged      = B.f < P.eps_p_cm && ~clamped;  % eps_p < 0.1 cm (p.20)
    info.fitness        = B.f;                           % cm
    info.err_wrist      = B.ep;                          % m
    info.fitness_xbest  = Fb;                            % Alg. 2's X_best, cm
    info.not_applicable = false;
    info.wrist_clamped  = clamped;
    info.fk_calls       = fk;
    info.jac_calls      = 0;
    info.N              = N;
    info.T              = T;
    info.history        = history;
    info.A_t            = A_t;
    info.r_t            = r_t;
    info.n_accept       = n_accept;                      % accepted X_new (Eq. 14)
    info.n_at_limit     = n_at_limit;                    % Eq. 7 moves clamped at a limit
    info.n_move_best    = n_move_best;                   % Eq. 7 positions that became X_best
    info.wrist_geometry = W;
    info.params         = P;
end

% ------------------------------------------------------------------ operators
function P = ba_params()
    P = struct('N', 12, 't_max', 200, 'f_min', 0, 'f_max', 2, 'alpha', 0.97, 'gamma', 0.1, ...
               'A0', 1, 'r0', 1, 'eps_p_cm', 0.1, 'tol_geom', 1e-9);
end

function [ok, W] = ba_wrist_geometry(robot)
% Sec. 4.3 and Eq. 15 premises from the DH model (uncounted kernel FK, two configurations): the
% axes of joints n-2, n-1, n (z_(n-3), z_(n-2), z_(n-1)) meet in one point, the point is O_(n-2),
% and P - d_n*R*[0; 0; 1] equals it for the end-effector pose.
    P  = ba_params();
    W  = struct('dn', NaN, 'alpha', NaN(1, 3), 'offset', zeros(1, 3), 'tool', eye(4), ...
                'dist', Inf, 'dist_O', Inf, 'dist_eq15', Inf);
    ok = false;
    if ~isfield(robot, 'dh') || robot.n < 4, return; end
    n  = robot.n;  dh = robot.dh;
    if isfield(robot, 'tool') && ~isempty(robot.tool), W.tool = robot.tool; end
    toff = zeros(1, n);
    if isfield(dh, 'theta_offset'), toff = dh.theta_offset(:)'; end
    W.dn     = dh.d(n);
    W.alpha  = [dh.alpha(n-2), dh.alpha(n-1), dh.alpha(n)];
    W.offset = toff(n-2:n);

    q_lo = robot.q_min(:);  q_hi = robot.q_max(:);
    Q = [(q_lo + q_hi) / 2, q_lo + 0.37 * (q_hi - q_lo)];
    W.dist = 0;  W.dist_O = 0;  W.dist_eq15 = 0;
    for c = 1:2
        [Te, Ts] = fk_chain_kernel(Q(:, c), dh, W.tool);
        M = zeros(3);  b = zeros(3, 1);  Pk = zeros(3, 3, 3);  pk = zeros(3, 3);
        for k = 1:3                                  % joint n-3+k turns about z_(n-4+k)
            Tk = Ts(:, :, n - 4 + k);
            z  = Tk(1:3, 3);
            Pk(:, :, k) = eye(3) - z * z';
            pk(:, k) = Tk(1:3, 4);
            M = M + Pk(:, :, k);  b = b + Pk(:, :, k) * pk(:, k);
        end
        if rcond(M) < 1e-12, W.dist = Inf; return; end
        x = M \ b;                                   % closest point to the three axes
        for k = 1:3
            W.dist = max(W.dist, norm(Pk(:, :, k) * (x - pk(:, k))));
        end
        W.dist_O    = max(W.dist_O, norm(x - Ts(1:3, 4, n - 2)));
        W.dist_eq15 = max(W.dist_eq15, norm(Te(1:3, 4) - W.dn * Te(1:3, 3) - x));
    end
    ok = W.dist <= P.tol_geom && W.dist_O <= P.tol_geom && W.dist_eq15 <= P.tol_geom ...
         && abs(sin(W.alpha(1))) > 1e-9 && abs(sin(W.alpha(2))) > 1e-9;
end

function Pw = ba_wrist_point(T_d, dn)
% Eq. 15: P_w = P - d_n*[R]*[0; 0; 1].
    Pw = T_d(1:3, 4) - dn * T_d(1:3, 3);
end

function [f, R3, ep] = ba_fitness(robot, x, qw, Pw)
% p.7 with Eq. 1: position error between the wrist centre O_(n-2)(theta_1..theta_(n-3)) and P_w,
% in cm (ep in m). One robot.fk call; the wrist joints qw move neither O_(n-2) nor 0R_(n-3).
    m = numel(x);
    [~, Ts] = robot.fk([x; qw]);
    ep = norm(Pw - Ts(1:3, 4, m + 1));
    f  = 100 * ep;                                           % Eq. 1, cm
    R3 = Ts(1:3, 1:3, m);                                    % 0R_(n-3), Eq. 16
end

function [qw, clamped] = ba_wrist(R3, R_n, W, qs, lo, hi)
% Eqs. 16-17: (n-3)R_n = 0R_(n-3)' * 0R_n = Rz(p1)Rx(a1)Rz(p2)Rx(a2)Rz(p3)Rx(a3), p_k = theta + offset.
% Both solutions for p2, 2*pi equivalents inside the limits; the least variation from Theta_s
% (Eq. 12) wins, in-limit solutions first. At a wrist singularity p1 keeps its Theta_s value.
    a = W.alpha;  off = W.offset(:);
    G  = R3' * R_n * rotx(a(3))';                            % Eq. 17, Rx(a3) removed
    c1 = cos(a(1));  s1 = sin(a(1));  c2 = cos(a(2));  s2 = sin(a(2));
    cb = min(1, max(-1, (c1 * c2 - G(3, 3)) / (s1 * s2)));
    cand = zeros(3, 2);  inlim = false(1, 2);  dv = zeros(1, 2);
    p2s = [acos(cb), -acos(cb)];
    for b = 1:2
        p2 = p2s(b);
        if hypot(G(1, 3), G(2, 3)) < 1e-12                   % axes n-2 and n aligned
            p1 = qs(1) + off(1);
        else
            vx = s2 * sin(p2);  vy = -c1 * s2 * cos(p2) - s1 * c2;
            p1 = atan2(G(2, 3), G(1, 3)) - atan2(vy, vx);
        end
        H  = (rotz(p1) * rotx(a(1)) * rotz(p2) * rotx(a(2)))' * G;
        p3 = atan2(H(2, 1), H(1, 1));
        [cand(:, b), inlim(b)] = nearest_equivalent([p1; p2; p3] - off, qs, lo, hi);
        dv(b) = norm(cand(:, b) - qs);                       % Eq. 12
    end
    if any(inlim), dv(~inlim) = Inf; end
    [~, b] = min(dv);
    qw = cand(:, b);
    clamped = ~inlim(b);
end

function [q, ok] = nearest_equivalent(q, qs, lo, hi)
% Per joint, the 2*pi equivalent inside [lo, hi] closest to qs; clamped if none is inside.
    ok = true;
    for k = 1:numel(q)
        c = q(k) + 2 * pi * (-3:3);
        in = c >= lo(k) & c <= hi(k);
        if any(in)
            c = c(in);  [~, j] = min(abs(c - qs(k)));  q(k) = c(j);
        else
            [~, j] = min(max(lo(k) - c, c - hi(k)));
            q(k) = min(max(c(j), lo(k)), hi(k));  ok = false;
        end
    end
end

function [x, v, hit] = ba_global(x, v, xb, ths, P, lo, hi)
% Eq. 5 (beta ~ U(0, 1), one per bat), Eq. 13 with the current position, Eq. 7; clamped.
    f = P.f_min + (P.f_max - P.f_min) * rand();              % Eq. 5
    v = v + (x - (xb + ths) / 2) * f;                        % Eq. 13
    x = x + v;                                               % Eq. 7
    hit = any(x < lo | x > hi);
    x = min(max(x, lo), hi);
end

function x = ba_local(xb, ths, r, A, lo, hi)
% Eq. 14: X_new = r*X_best + (1 - r)*Theta_s + eps*A, clamped.
    x = r * xb + (1 - r) * ths + ba_epsilon(numel(xb)) * A;
    x = min(max(x, lo), hi);
end

function e = ba_epsilon(m)
% eps (p.6): standard normal truncated to [-1, 1], one value per joint; out-of-range draws redrawn.
    e = randn(m, 1);
    bad = abs(e) > 1;
    while any(bad)
        e(bad) = randn(nnz(bad), 1);
        bad = abs(e) > 1;
    end
end

function er = ba_eq3(R, R_d)
% Eq. 3: eps_r = ||R - R_d|| / ||R_d||, Euclidean (Frobenius) matrix norm.
    er = norm(R - R_d, 'fro') / norm(R_d, 'fro');
end

function B = keep_best(B, f, x, R3, ep)
    if isnan(B.f) || f < B.f
        B.f = f;  B.x = x;  B.R3 = R3;  B.ep = ep;
    end
end

function R = rotz(t)
    R = [cos(t) -sin(t) 0; sin(t) cos(t) 0; 0 0 1];
end

function R = rotx(t)
    R = [1 0 0; 0 cos(t) -sin(t); 0 sin(t) cos(t)];
end
