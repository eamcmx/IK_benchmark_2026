function [q_best, info] = solve_GWO(robot, T_d, q0, opts)
% SOLVE_GWO  Grey Wolf Optimizer IK solver as specified by Abdor-Sierra et al. (2022), Sec. 2.7.
%
%   Source:
%     Javier Alexis Abdor-Sierra, Emmanuel Alejandro Merchán-Cruz, Ricardo Gustavo
%     Rodríguez-Cañizo (2022). A comparative analysis of metaheuristic algorithms for solving the
%     inverse kinematics of robot manipulators. Results in Engineering, 16, 100597.
%     DOI: 10.1016/j.rineng.2022.100597
%   The paper attributes GWO to its Ref. [29]: S. Mirjalili, S.M. Mirjalili, A. Lewis (2014), Grey
%   wolf optimizer, Adv. Eng. Software 69, 46-61 (not consulted).
%
%   Algorithm (pp.3-4, Sec. 2.7, Eqs. 19-25; settings p.8; fitness pp.6-7, Eqs. 34-42 and 44;
%   stop rule p.8):
%     The pack of N wolves is evaluated and ranked: the best solution is alpha, the second best
%     beta, the third best delta, and the omega wolves follow these three (p.3). Each iteration
%     t = 1..T the omega wolves move by the hunting model (Eqs. 23-25), the encircling model of
%     Eqs. 19-20 with the prey position taken from each leader:
%       D_k = |C_k.*X_k - X|,  X'_k = X_k - A_k.*D_k  for X_k = X_alpha, X_beta, X_delta,
%       X(t+1) = (X'_1 + X'_2 + X'_3)/3,
%     with the coefficient vectors A = 2*a*r1 - a and C = 2*r2 (Eqs. 21-22), r1, r2 ~ U(0, 1), and a
%     decreasing linearly from 2 to 0 over the iterations (p.4). The moved wolves are evaluated and
%     the pack is ranked again. The run ends when the best fitness is below 1e-3 or after T
%     iterations (p.8).
%     Fitness (Eq. 44): f = Perror + Oerror*w, Perror = ||p_t - p_d|| (Eq. 34),
%     Oerror = |q0t - q0d| + ||q(1:3)t - q(1:3)d|| (Eqs. 40-42), with the unit quaternions of the
%     trialed and desired rotations from Eqs. 36-39 (sgn(0) = 1, Eq. 39).
%
%   Parameters:
%     N = 500, T = 2000   p.8: "GWO: Population size = 500, maximum iterations = 2000". Inside the
%                         benchmark T is the budget horizon (see Benchmark interface).
%     a = 2*(T - t)/T, t = 1..T. p.4: linear from 2 to 0; written in the (T - t)/T form of the
%                         paper's own schedules (Eqs. 7-9, 12), reaching 0 at the last iteration.
%     A, C: one r1 and one r2 per joint ("A and C are coefficient vectors", p.4), drawn
%                         independently for each leader (subscripts 1-3 of Eqs. 23-24).
%     w = 200 for 6-DOF robots, 300 for robots with more than 6 joints. p.7 sets w experimentally
%                         per robot: 200 for the UR5 (6-DOF), 300 for the SIA20D (7-DOF); mapping
%                         by DOF is the nearest reading for robots the paper did not use.
%     f_stop = 1e-3       p.8: a solution is optimal when "the error being evaluated is less than
%                         1e-03", one of the two stop criteria; applied to f of Eq. 44.
%
%   Benchmark interface:
%     Budget: initialisation needs N = 500 FK calls and each iteration N - 3 = 497 (the omega
%       wolves); an iteration starts only if ikb_budget_left(opts) covers it.
%       T = ikb_generations(opts, 500, 497, 2000) = 3 iterations with opts.budget = 2000 (1991 FK
%       calls), independent of the DOF (outside the benchmark T = 2000). With T = 3, a = 4/3, 2/3
%       and 0; at a = 0, A = 0 and every omega wolf moves to (X_alpha + X_beta + X_delta)/3
%       (Eqs. 24-25). No final FK check. opts.max_iter and opts.pop_size are ignored.
%     Warm start: q0 (clamped to the limits) is wolf 1 of the initial pack, evaluated unperturbed,
%       and enters the first ranking.
%     Units: benchmark lengths are metres; the paper's DH tables are in mm (p.5, Tables 2-3), so
%       Perror is computed in mm: f = 1000*||p_t - p_d||[m] + w*Oerror.
%     Joint limits: Eq. 44 bounds theta (p.7) without a handling rule, so every moved wolf is
%       clamped to [q_min, q_max].
%     Output: alpha (lowest f; the leaders do not move, so alpha is the best wolf evaluated).
%       info.converged = best f < 1e-3 (p.8). info.err_rot is the rotation angle of alpha (rad),
%       reported only. solve_GWO('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Only the omega wolves move, and alpha, beta and delta are the three best wolves of the
%       pack, ranked after every iteration: the paper states the update for the omega wolves
%       (p.4) and does not state when the leaders are updated (Ref. [29] not consulted).
%     - Iteration index t = 1..T in the a schedule (not stated; a reaches 0 at t = T).
%     - w mapped by DOF (see Parameters).
%     - The iteration count is the budget horizon T = 3 instead of 2000 (benchmark rule).
%     - max(0, .) inside the square roots of Eqs. 37-38 guards against round-off only.
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
        q_best = struct('fitness', @abdor_fitness, 'quat', @abdor_quat, 'oerror', @abdor_oerror, ...
                        'weight', @abdor_weight, 'a', @gwo_a, 'hierarchy', @gwo_hierarchy, ...
                        'move', @gwo_move, 'params', gwo_params());
        info = [];
        return
    end

    t_start = tic;
    P = gwo_params();
    N = P.N;

    n     = robot.n;
    q_min = robot.q_min(:);
    q_max = robot.q_max(:);
    q0    = min(max(q0(:), q_min), q_max);
    T     = ikb_generations(opts, N, N - 3, P.T);

    B = struct('f', NaN, 'q', q0, 'ep', NaN, 'oe', NaN, 'er', NaN);
    history = zeros(1, 0);  iters = 0;  fk = 0;  converged = false;

    if ikb_budget_left(opts) >= N
        % Initial pack, q0 is wolf 1
        X = q_min + rand(n, N) .* (q_max - q_min);  X(:, 1) = q0;
        F = zeros(1, N);
        for i = 1:N
            [F(i), ep, oe, er] = abdor_fitness(robot, T_d, X(:, i));  fk = fk + 1;
            B = keep_best(B, F(i), X(:, i), ep, oe, er);
        end
        converged = B.f < P.f_stop;                                         % p.8

        history = zeros(1, T);
        t = 0;
        while ~converged && t < T
            if ikb_budget_left(opts) < N - 3, break; end
            t = t + 1;
            [L, om] = gwo_hierarchy(X, F);                                  % p.3
            X(:, om) = min(max(gwo_move(X(:, om), L, gwo_a(t, T)), q_min), q_max);   % Eqs. 21-25
            for i = om
                [F(i), ep, oe, er] = abdor_fitness(robot, T_d, X(:, i));  fk = fk + 1;
                B = keep_best(B, F(i), X(:, i), ep, oe, er);
            end
            iters = t;
            history(t) = B.f;
            converged = B.f < P.f_stop;                                     % p.8
        end
        history = history(1:iters);
    end

    q_best = B.q;
    info = struct();
    info.solver    = 'GWO';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = converged;
    info.fitness   = B.f;
    info.err_pos   = B.ep;                                   % m
    info.err_rot   = B.er;                                   % rad
    info.O_error   = B.oe;                                   % Eq. 42
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.T         = T;
    info.history   = history;
    info.params    = P;
end

% ------------------------------------------------------------------ operators
function P = gwo_params()
    P = struct('N', 500, 'T', 2000, 'f_stop', 1e-3);
end

function a = gwo_a(t, T)
% p.4: a decreases linearly from 2 (t = 0) to 0 (t = T).
    a = 2 * (T - t) / T;
end

function [L, om] = gwo_hierarchy(X, F)
% p.3: best = alpha, second best = beta, third best = delta (columns of L); om = omega indices.
    [~, o] = sort(F);
    L  = X(:, o(1:3));
    om = o(4:end);
end

function Xn = gwo_move(X, L, a)
% Eqs. 21-25 for every column of X (omega wolves); L = [X_alpha, X_beta, X_delta].
    [n, m] = size(X);
    S = zeros(n, m);
    for k = 1:3
        A = 2 * a * rand(n, m) - a;                                         % Eq. 21
        C = 2 * rand(n, m);                                                 % Eq. 22
        D = abs(C .* L(:, k) - X);                                          % Eq. 23
        S = S + (L(:, k) - A .* D);                                         % Eq. 24
    end
    Xn = S / 3;                                                             % Eq. 25
end

function B = keep_best(B, f, q, ep, oe, er)
    if isnan(B.f) || f < B.f
        B.f = f;  B.q = q;  B.ep = ep;  B.oe = oe;  B.er = er;
    end
end

% ======================================================================== fitness
% Abdor-Sierra et al. (2022), Eqs. 34-44. Same code in solve_HS, solve_SCAPSO, solve_GWO, solve_PSO, solve_DE and solve_FA.
function [f, ep, oe, er] = abdor_fitness(robot, T_d, q)
% Eq. 44: f = Perror + Oerror*w with Perror (Eq. 34) in mm. ep = position error (m),
% oe = Oerror (Eq. 42), er = rotation angle between the two orientations (rad, reporting only).
    T  = robot.fk(q);
    ep = norm(T(1:3, 4) - T_d(1:3, 4));                                     % Eq. 34 (m)
    oe = abdor_oerror(T(1:3, 1:3), T_d(1:3, 1:3));                          % Eqs. 35-42
    f  = 1000 * ep + oe * abdor_weight(robot.n);                            % Eq. 44
    er = acos(min(1, max(-1, (trace(T_d(1:3, 1:3) * T(1:3, 1:3)') - 1) / 2)));
end

function oe = abdor_oerror(R_t, R_d)
% Eqs. 40-42: Oerror = |q0t - q0d| + ||q(1:3)t - q(1:3)d||.
    Qt = abdor_quat(R_t);
    Qd = abdor_quat(R_d);
    oe = abs(Qt(1) - Qd(1)) + norm(Qt(2:4) - Qd(2:4));
end

function Q = abdor_quat(R)
% Eqs. 36-39: unit quaternion [eta; eps_x; eps_y; eps_z] of R = [n s a]. max(0, .) guards the
% square roots against round-off.
    nx = R(1, 1);  ny = R(2, 1);  nz = R(3, 1);
    sx = R(1, 2);  sy = R(2, 2);  sz = R(3, 2);
    ax = R(1, 3);  ay = R(2, 3);  az = R(3, 3);
    Q = 0.5 * [sqrt(max(0, nx + sy + az + 1));                              % Eq. 37
               abdor_sgn(sz - ay) * sqrt(max(0, nx - sy - az + 1));         % Eq. 38
               abdor_sgn(ax - nz) * sqrt(max(0, sy - az - nx + 1));
               abdor_sgn(ny - sx) * sqrt(max(0, az - nx - sy + 1))];
end

function s = abdor_sgn(x)
% Eq. 39: sgn(x) = 1 for x >= 0, -1 for x < 0.
    if x >= 0, s = 1; else, s = -1; end
end

function w = abdor_weight(n)
% p.7: w = 200 for the UR5 (6-DOF), 300 for the SIA20D (7-DOF); 300 for every robot with n > 6.
    if n > 6, w = 300; else, w = 200; end
end
