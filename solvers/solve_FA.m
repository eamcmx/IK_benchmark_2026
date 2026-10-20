function [q_best, info] = solve_FA(robot, T_d, q0, opts)
% SOLVE_FA  Firefly Algorithm IK solver as specified by Abdor-Sierra et al. (2022), Sec. 2.6.
%
%   Source:
%     Javier Alexis Abdor-Sierra, Emmanuel Alejandro Merchán-Cruz, Ricardo Gustavo
%     Rodríguez-Cañizo (2022). A comparative analysis of metaheuristic algorithms for solving the
%     inverse kinematics of robot manipulators. Results in Engineering, 16, 100597.
%     DOI: 10.1016/j.rineng.2022.100597
%   The paper attributes FA to its Ref. [28]: X.-S. Yang (2009), Firefly algorithms for multimodal
%   optimization, International Symposium on Stochastic Algorithms, Springer, pp. 169-178 (not
%   consulted).
%
%   Algorithm (p.3, Sec. 2.6, Eqs. 17-18; settings p.9; fitness pp.6-7, Eqs. 34-42 and 44;
%   stop rule p.8):
%     A population of N fireflies is evaluated; the brightness of a firefly is given by the
%     objective function (p.3, third rule): the lower f, the brighter. Each iteration t = 1..T
%     moves every firefly i with the brightness and positions of iteration t. For every firefly j
%     brighter than i (p.3, second rule: "the less bright will move towards the brighter one"),
%     in index order, firefly i moves by Eq. 17,
%       x_i <- x_i + beta0*exp(-gamma*r_ij^2)*(x_j - x_i) + alpha*eps_i,  r_ij = ||x_i - x_j||
%     (Eq. 18), each move starting where the previous one ended. A firefly with no brighter one
%     moves randomly (p.3: "If there is no brighter one, then the firefly will move randomly"):
%       x_i <- x_i + alpha*eps_i,
%     the random term of Eq. 17. All moved fireflies are then evaluated. The run ends when the
%     best fitness is below 1e-3 or after T iterations (p.8).
%     Fitness (Eq. 44): f = Perror + Oerror*w, Perror = ||p_t - p_d|| (Eq. 34),
%     Oerror = |q0t - q0d| + ||q(1:3)t - q(1:3)d|| (Eqs. 40-42), with the unit quaternions of the
%     trialed and desired rotations from Eqs. 36-39 (sgn(0) = 1, Eq. 39).
%
%   Parameters (p.9 unless stated):
%     N = 30.
%     T = 500 for 6-DOF robots, 750 for robots with more than 6 joints. p.9: "maximum iterations =
%           500 for the UR5, 750 for the SIA20D"; mapping by DOF is the nearest reading for robots
%           the paper did not use (the same reading as w). Inside the benchmark T is the budget
%           horizon (see Benchmark interface).
%     gamma = 0.9.
%     beta0 = 0.9. p.9 lists "beta = 0.9"; the only beta of Sec. 2.6 is the attractiveness beta0
%           of Eq. 17 (p.3: "normally set to 1"), so the listed value is used as beta0.
%     alpha = 0.3, constant over the run (no schedule stated, pp.3, 9 checked).
%     eps_i: one draw from the standard normal distribution N(0, 1) per joint and move. p.3:
%           "epsilon in [0, 1] is a random number from the Gaussian distribution"; a Gaussian
%           variable cannot be confined to [0, 1], so the stated distribution is used, with zero
%           mean and unit standard deviation (neither stated).
%     alpha*eps_i is added as printed in Eq. 17, without a factor for the joint range: p.3 calls
%           alpha "a scale version of the problem" without a formula.
%     Joint unit: radians (the benchmark unit) for x, r_ij, gamma and alpha. The paper does not
%           state the unit of the decision variables (pp.3, 5, 9 checked); its DH tables give the
%           joint ranges in degrees (p.5, Tables 2-3).
%     w = 200 for 6-DOF robots, 300 for robots with more than 6 joints (the weight of Eq. 44).
%           p.7 sets w experimentally per robot: 200 for the UR5 (6-DOF), 300 for the SIA20D
%           (7-DOF); mapping by DOF is the nearest reading for robots the paper did not use.
%     f_stop = 1e-3. p.8: a solution is optimal when "the error being evaluated is less than
%           1e-03", one of the two stop criteria; applied to f of Eq. 44.
%
%   Benchmark interface:
%     Budget: initialisation and every iteration need N = 30 FK calls (every firefly moves); an
%       iteration starts only if ikb_budget_left(opts) covers it. T = ikb_generations(opts, 30, 30,
%       T_paper) = 65 iterations with opts.budget = 2000 (1980 FK calls), independent of the DOF
%       (outside the benchmark T = 500 or 750). No parameter varies over T, so T only bounds the
%       run. No final FK check. opts.max_iter and opts.pop_size are ignored.
%     Warm start: q0 (clamped to the limits) is firefly 1 of the initial population, evaluated
%       unperturbed; the returned solution is the best firefly evaluated, so q0 is returned unless
%       a better one is found. The other fireflies are drawn uniformly in [q_min, q_max]: the paper
%       loads a predefined initial population whose generation is not stated (p.7 Alg. 1, p.8).
%     Units: benchmark lengths are metres; the paper's DH tables are in mm (p.5, Tables 2-3), so
%       Perror is computed in mm: f = 1000*||p_t - p_d||[m] + w*Oerror.
%     Joint limits: Eq. 44 bounds theta (p.7) without a handling rule, so every position produced
%       by Eq. 17 or by the random move is clamped to [q_min, q_max].
%     Output: the best firefly evaluated (lowest f). The brightest firefly moves randomly, so this
%       can be a position the population has left. info.converged = best f < 1e-3 (p.8).
%       info.n_random counts random moves. info.err_rot is the rotation angle of the returned
%       firefly (rad), reported only. solve_FA('ops') returns handles to the operators for unit
%       tests.
%
%   Deviations from the paper:
%     - Order of the moves: once per brighter firefly, in index order, with the brightness and
%       x_j of iteration t (the x_j(t) of Eq. 17), all fireflies evaluated after the moves. The
%       paper gives no pseudocode (p.7 defers to the original papers; Ref. [28] not consulted).
%     - Random move of a firefly with no brighter one: the random term alpha*eps_i of Eq. 17
%       (rule stated without a formula, p.3).
%     - eps_i ~ N(0, 1) per joint (the stated range [0, 1] is incompatible with a Gaussian; mean
%       and standard deviation not stated).
%     - alpha*eps_i without a joint-range factor, and all joint quantities in radians (see
%       Parameters).
%     - The listed beta used as beta0 (see Parameters).
%     - Returned solution: the best firefly evaluated (not stated).
%     - Initial population uniform in the joint limits (generation not stated, see Warm start).
%     - T and w mapped by DOF (see Parameters).
%     - The iteration count is the budget horizon T = 65 instead of 500 or 750 (benchmark rule).
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
% Year      : 2026
% License   : MIT (see LICENSE)
% Repository: https://github.com/eamcmx/IK_benchmark_2026
% ==========================================================================

    if ischar(robot) && strcmp(robot, 'ops')
        q_best = struct('fitness', @abdor_fitness, 'quat', @abdor_quat, 'oerror', @abdor_oerror, ...
                        'weight', @abdor_weight, 'horizon', @fa_horizon, 'move', @fa_move, ...
                        'random', @fa_random, 'iteration', @fa_iteration, 'params', fa_params());
        info = [];
        return
    end

    t_start = tic;
    P = fa_params();
    N = P.N;

    n     = robot.n;
    q_min = robot.q_min(:);
    q_max = robot.q_max(:);
    q0    = min(max(q0(:), q_min), q_max);
    T     = ikb_generations(opts, N, N, fa_horizon(n, P));

    B = struct('f', NaN, 'q', q0, 'ep', NaN, 'oe', NaN, 'er', NaN);
    history = zeros(1, 0);  iters = 0;  fk = 0;  converged = false;  n_random = 0;

    if ikb_budget_left(opts) >= N
        % Initial population, q0 is firefly 1
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
            if ikb_budget_left(opts) < N, break; end
            t = t + 1;
            [X, nr] = fa_iteration(X, F, P, q_min, q_max);                  % Eqs. 17-18, p.3
            n_random = n_random + nr;
            for i = 1:N
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
    info.solver    = 'FA';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = converged;
    info.fitness   = B.f;
    info.err_pos   = B.ep;                                   % m
    info.err_rot   = B.er;                                   % rad
    info.O_error   = B.oe;                                   % Eq. 42
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.n_random  = n_random;
    info.T         = T;
    info.history   = history;
    info.params    = P;
end

% ------------------------------------------------------------------ operators
function P = fa_params()
    P = struct('N', 30, 'T_6dof', 500, 'T_redundant', 750, 'gamma', 0.9, 'beta0', 0.9, ...
               'alpha', 0.3, 'f_stop', 1e-3);
end

function T = fa_horizon(n, P)
% p.9: 500 iterations for the UR5 (6-DOF), 750 for the SIA20D (7-DOF); 750 for every n > 6.
    if n > 6, T = P.T_redundant; else, T = P.T_6dof; end
end

function [Xn, n_rand] = fa_iteration(X, F, P, q_min, q_max)
% One iteration of the rules of p.3 with the brightness F and positions X of iteration t.
    Xn = X;  n_rand = 0;
    for i = 1:size(X, 2)
        x = X(:, i);
        brighter = find(F < F(i));
        if isempty(brighter)
            x = min(max(fa_random(x, P), q_min), q_max);                    % no brighter one
            n_rand = n_rand + 1;
        else
            for j = brighter(:)'
                x = min(max(fa_move(x, X(:, j), P), q_min), q_max);         % Eqs. 17-18
            end
        end
        Xn(:, i) = x;
    end
end

function x = fa_move(x, xj, P)
% Eq. 17 toward the brighter firefly xj, r_ij from Eq. 18, eps_i ~ N(0, 1) per joint.
    r = norm(x - xj);                                                       % Eq. 18
    x = x + P.beta0 * exp(-P.gamma * r^2) * (xj - x) + P.alpha * randn(numel(x), 1);
end

function x = fa_random(x, P)
% Random move (p.3): the random term alpha*eps_i of Eq. 17, eps_i ~ N(0, 1) per joint.
    x = x + P.alpha * randn(numel(x), 1);
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
