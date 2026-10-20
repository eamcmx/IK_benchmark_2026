function [q_best, info] = solve_DE(robot, T_d, q0, opts)
% SOLVE_DE  Differential Evolution (DE/rand/1/bin) IK solver as specified by Abdor-Sierra et al.
%           (2022), Sec. 2.5.
%
%   Source:
%     Javier Alexis Abdor-Sierra, Emmanuel Alejandro Merchán-Cruz, Ricardo Gustavo
%     Rodríguez-Cañizo (2022). A comparative analysis of metaheuristic algorithms for solving the
%     inverse kinematics of robot manipulators. Results in Engineering, 16, 100597.
%     DOI: 10.1016/j.rineng.2022.100597
%   The paper attributes DE to its Ref. [25]: R. Storn, K. Price (1997), Differential evolution - a
%   simple and efficient heuristic for global optimization over continuous spaces, J. Global
%   Optim. 11(4), 341-359, and the variant names to its Ref. [27]: D. Simon (2013), Evolutionary
%   Optimization Algorithms, Wiley (neither consulted).
%
%   Algorithm (p.3, Sec. 2.5, Eqs. 15-16, Table 1; settings p.8; fitness pp.6-7, Eqs. 34-42 and
%   44; stop rule p.8):
%     A population of N individuals is evaluated. Each generation t = 1..T builds, for every
%     target x_i, a mutant and a trial vector from the population of generation t:
%       v_i  = x_r1 + F*(x_r2 - x_r3)                       (Eq. 15; i, r1, r2, r3 distinct, p.3)
%       u_ij = v_ij if r_cj < c or j = J_r, x_ij otherwise   (Eq. 16; r_cj ~ U(0, 1) per joint,
%                                                             J_r uniform in 1..n)
%     This is DE/rand/1/bin, "classic DE" (p.3; Table 1 row DE/rand/1/?). All trial vectors are
%     evaluated, and each trial vector replaces its target if its fitness is lower ("the best
%     solution is kept, and the worse is discarded for the next generation", p.3). The run ends
%     when the best fitness is below 1e-3 or after T generations (p.8).
%     Fitness (Eq. 44): f = Perror + Oerror*w, Perror = ||p_t - p_d|| (Eq. 34),
%     Oerror = |q0t - q0d| + ||q(1:3)t - q(1:3)d|| (Eqs. 40-42), with the unit quaternions of the
%     trialed and desired rotations from Eqs. 36-39 (sgn(0) = 1, Eq. 39).
%
%   Parameters:
%     N = 50, T = 2000   p.8: "DE/rand/1/bin, ...: Populations size = 50, maximum iterations =
%                        2000". Inside the benchmark T is the budget horizon (see Benchmark
%                        interface).
%     F = 0.65           p.3 gives only the range: "F is the scale factor in the range of
%                        [0.4, 0.9]"; no value is reported for the paper's runs (pp.3, 8-9
%                        checked), so the midpoint of the range is used.
%     c = 0.5            not stated in the paper (pp.3, 8-9 checked): p.3 names c the crossover
%                        factor and compares it with r_cj in [0, 1], so c acts only within [0, 1];
%                        the midpoint of that range is used.
%     w = 200 for 6-DOF robots, 300 for robots with more than 6 joints. p.7 sets w experimentally
%                        per robot: 200 for the UR5 (6-DOF), 300 for the SIA20D (7-DOF); mapping
%                        by DOF is the nearest reading for robots the paper did not use.
%     f_stop = 1e-3      p.8: a solution is optimal when "the error being evaluated is less than
%                        1e-03", one of the two stop criteria; applied to f of Eq. 44.
%
%   Benchmark interface:
%     Budget: initialisation and every generation need N = 50 FK calls; a generation starts only
%       if ikb_budget_left(opts) covers it. T = ikb_generations(opts, 50, 50, 2000) = 39
%       generations with opts.budget = 2000 (2000 FK calls), independent of the DOF (outside the
%       benchmark T = 2000). No final FK check. opts.max_iter and opts.pop_size are ignored.
%     Warm start: q0 (clamped to the limits) is individual 1 of the initial population, evaluated
%       unperturbed, and is the target of the first selection; the population never loses its
%       best individual. The other individuals are drawn uniformly in [q_min, q_max]: the paper
%       loads a predefined initial population whose generation is not stated (p.7 Alg. 1, p.8).
%     Units: benchmark lengths are metres; the paper's DH tables are in mm (p.5, Tables 2-3), so
%       Perror is computed in mm: f = 1000*||p_t - p_d||[m] + w*Oerror.
%     Joint limits: Eq. 44 bounds theta (p.7) without a handling rule, so every trial vector is
%       clamped to [q_min, q_max].
%     Output: the best individual (lowest f evaluated). info.converged = best f < 1e-3 (p.8).
%       info.err_rot is the rotation angle of the best individual (rad), reported only.
%       solve_DE('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - F = 0.65, the midpoint of the stated range (no value stated, see Parameters).
%     - c = 0.5 (no value stated, see Parameters).
%     - A trial vector with the same fitness as its target does not replace it (tie rule not
%       stated, p.3).
%     - Initial population uniform in the joint limits (generation not stated, see Warm start).
%     - w mapped by DOF (see Parameters).
%     - The generation count is the budget horizon T = 39 instead of 2000 (benchmark rule).
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
                        'weight', @abdor_weight, 'mutant', @de_mutant, ...
                        'crossover', @de_crossover, 'select', @de_select, 'params', de_params());
        info = [];
        return
    end

    t_start = tic;
    P = de_params();
    N = P.N;

    n     = robot.n;
    q_min = robot.q_min(:);
    q_max = robot.q_max(:);
    q0    = min(max(q0(:), q_min), q_max);
    T     = ikb_generations(opts, N, N, P.T);

    B = struct('f', NaN, 'q', q0, 'ep', NaN, 'oe', NaN, 'er', NaN);
    history = zeros(1, 0);  iters = 0;  fk = 0;  converged = false;

    if ikb_budget_left(opts) >= N
        % Initial population, q0 is individual 1
        X = q_min + rand(n, N) .* (q_max - q_min);  X(:, 1) = q0;
        F = zeros(1, N);
        for i = 1:N
            [F(i), ep, oe, er] = abdor_fitness(robot, T_d, X(:, i));  fk = fk + 1;
            B = keep_best(B, F(i), X(:, i), ep, oe, er);
        end
        converged = B.f < P.f_stop;                                         % p.8

        history = zeros(1, T);
        U  = zeros(n, N);
        Fu = zeros(1, N);
        t = 0;
        while ~converged && t < T
            if ikb_budget_left(opts) < N, break; end
            t = t + 1;
            for i = 1:N
                v = de_mutant(X, i, P.F);                                   % Eq. 15
                U(:, i) = de_crossover(X(:, i), v, P.c);                    % Eq. 16
            end
            U = min(max(U, q_min), q_max);
            for i = 1:N
                [Fu(i), ep, oe, er] = abdor_fitness(robot, T_d, U(:, i));  fk = fk + 1;
                B = keep_best(B, Fu(i), U(:, i), ep, oe, er);
            end
            [X, F] = de_select(X, F, U, Fu);                                % p.3
            iters = t;
            history(t) = B.f;
            converged = B.f < P.f_stop;                                     % p.8
        end
        history = history(1:iters);
    end

    q_best = B.q;
    info = struct();
    info.solver    = 'DE';
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
function P = de_params()
    P = struct('N', 50, 'T', 2000, 'F', 0.65, 'c', 0.5, 'f_stop', 1e-3);
end

function v = de_mutant(X, i, F)
% Eq. 15, DE/rand/1: r1, r2, r3 drawn without replacement from the individuals other than i.
    r = randperm(size(X, 2) - 1, 3);
    r = r + (r >= i);
    v = X(:, r(1)) + F * (X(:, r(2)) - X(:, r(3)));
end

function u = de_crossover(x, v, c)
% Eq. 16, binomial: joint j from the mutant if r_cj < c or j = J_r, else from the target.
    n  = numel(x);
    rc = rand(n, 1);
    Jr = randi(n);
    take = rc < c;
    take(Jr) = true;
    u = x;
    u(take) = v(take);
end

function [X, F] = de_select(X, F, U, Fu)
% p.3: the better of target and trial survives; a tie keeps the target.
    s = Fu < F;
    X(:, s) = U(:, s);
    F(s)    = Fu(s);
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
