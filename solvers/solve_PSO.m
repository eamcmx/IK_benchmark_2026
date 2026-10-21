function [q_best, info] = solve_PSO(robot, T_d, q0, opts)
% SOLVE_PSO  Particle Swarm Optimization IK solver as specified by Abdor-Sierra et al. (2022),
%            Sec. 2.1.
%
%   Source:
%     Javier Alexis Abdor-Sierra, Emmanuel Alejandro Merchán-Cruz, Ricardo Gustavo
%     Rodríguez-Cañizo (2022). A comparative analysis of metaheuristic algorithms for solving the
%     inverse kinematics of robot manipulators. Results in Engineering, 16, 100597.
%     DOI: 10.1016/j.rineng.2022.100597
%   The paper attributes PSO to its Ref. [18]: J. Kennedy, R. Eberhart (1995), Particle swarm
%   optimization, Proc. ICNN'95 International Conference on Neural Networks, vol. 4,
%   pp. 1942-1948 (not consulted).
%
%   Algorithm (p.2, Sec. 2.1, Eqs. 1-3; settings p.8; fitness pp.6-7, Eqs. 34-42 and 44;
%   stop rule p.8):
%     A swarm of N particles is evaluated. Each iteration t = 1..T moves every particle (Eqs. 1-3):
%       V <- w*V + r1*C1*(Pbest_i - X) + r2*C2*(Gbest - X),  r1, r2 ~ U(0, 1),
%       V bounded by maxvel = vel*(ub - lb) and minvel = -maxvel,  X <- X + V,
%     then evaluates all particles and updates Pbest and Gbest. The inertia w is reduced from wu
%     to wl over the iterations (p.2: "reducing the w parameter improved the algorithm"; end
%     values p.8). The run ends when the best fitness is below 1e-3 or after T iterations (p.8).
%     Fitness (Eq. 44): f = Perror + Oerror*w_f, Perror = ||p_t - p_d|| (Eq. 34),
%     Oerror = |q0t - q0d| + ||q(1:3)t - q(1:3)d|| (Eqs. 40-42), with the unit quaternions of the
%     trialed and desired rotations from Eqs. 36-39 (sgn(0) = 1, Eq. 39).
%
%   Parameters (p.8 unless stated):
%     N = 50 for 6-DOF robots, 100 for robots with more than 6 joints. p.8: "Populations size = 50
%           for the UR5, 100 for the SIA20D"; mapping by DOF is the nearest reading for robots the
%           paper did not use (the same reading as w_f).
%     T = 2000. Inside the benchmark T is the budget horizon (see Benchmark interface).
%     C1 = 2, C2 = 0.5.
%     vel = 0.009 (Eq. 3), applied per joint to q_max - q_min.
%     wu = 0.9, wl = 0.4: inertia w falls from wu to wl as w = wl + (T - t)/T*(wu - wl). The
%           schedule is not stated (pp.2, 8 checked); the linear decrease over T is used, written in
%           the (T - t)/T form of the paper's own schedules (Eqs. 8-9).
%     r1, r2 (Eq. 1): one draw per particle and iteration, as the equation is printed (scalars
%           multiplying vectors); per-joint draws are not stated.
%     w_f = 200 for 6-DOF robots, 300 for robots with more than 6 joints (the weight of Eq. 44).
%           p.7 sets it experimentally per robot: 200 for the UR5 (6-DOF), 300 for the SIA20D
%           (7-DOF); mapping by DOF is the nearest reading for robots the paper did not use.
%     f_stop = 1e-3. p.8: a solution is optimal when "the error being evaluated is less than
%           1e-03", one of the two stop criteria; applied to f of Eq. 44.
%     V(0) = 0: initial velocity not stated (pp.2, 7-8 checked).
%
%   Benchmark interface:
%     Budget: initialisation and every iteration need N FK calls; an iteration starts only if
%       ikb_budget_left(opts) covers it. With opts.budget = 2000,
%       T = ikb_generations(opts, 50, 50, 2000) = 39 iterations for 6-DOF robots (N = 50) and
%       T = ikb_generations(opts, 100, 100, 2000) = 19 for robots with more than 6 joints
%       (N = 100), 2000 FK calls in both cases (outside the benchmark T = 2000). No final FK check.
%       opts.max_iter and opts.pop_size are ignored.
%     Warm start: q0 (clamped to the limits) is particle 1 of the initial swarm, evaluated
%       unperturbed; Gbest is the best evaluated position, so q0 enters the first Gbest selection.
%       The other particles are drawn uniformly in [q_min, q_max]: the paper loads a predefined
%       initial population whose generation is not stated (p.7 Alg. 1, p.8).
%     Units: benchmark lengths are metres; the paper's DH tables are in mm (p.5, Tables 2-3), so
%       Perror is computed in mm: f = 1000*||p_t - p_d||[m] + w_f*Oerror.
%     Joint limits: Eq. 44 bounds theta (p.7) without a handling rule; Eq. 3 bounds the velocity.
%       Every new position is clamped to [q_min, q_max]; the velocity is not changed by the clamp.
%     Output: Gbest (lowest f evaluated). info.converged = best f < 1e-3 (p.8). info.err_rot is
%       the rotation angle of Gbest (rad), reported only. solve_PSO('ops') returns handles to the
%       operators for unit tests.
%
%   Deviations from the paper:
%     - N mapped by DOF (see Parameters).
%     - Inertia schedule linear from wu to wl (not stated, see Parameters).
%     - Eq. 1 is printed with a missing closing parenthesis; the evident reading is used.
%     - Updates are synchronous: all particles of iteration t move with the same Gbest, and Pbest
%       and Gbest are updated after all are evaluated (order not stated, p.2).
%     - Initial population uniform in the joint limits (generation not stated, see Warm start).
%     - w_f mapped by DOF (see Parameters).
%     - The iteration count is the budget horizon T = 39 (N = 50) or 19 (N = 100) instead of 2000
%       (benchmark rule).
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
                        'weight', @abdor_weight, 'popsize', @pso_popsize, ...
                        'inertia', @pso_inertia, 'move', @pso_move, 'pbest', @pso_pbest, ...
                        'params', pso_params());
        info = [];
        return
    end

    t_start = tic;
    P = pso_params();

    n     = robot.n;
    N     = pso_popsize(n, P);
    q_min = robot.q_min(:);
    q_max = robot.q_max(:);
    q0    = min(max(q0(:), q_min), q_max);
    vmax  = P.vel * (q_max - q_min);                                        % Eq. 3
    T     = ikb_generations(opts, N, N, P.T);

    B = struct('f', NaN, 'q', q0, 'ep', NaN, 'oe', NaN, 'er', NaN);
    history = zeros(1, 0);  iters = 0;  fk = 0;  converged = false;

    if ikb_budget_left(opts) >= N
        % Initial swarm, q0 is particle 1
        X = q_min + rand(n, N) .* (q_max - q_min);  X(:, 1) = q0;
        V = zeros(n, N);
        F = zeros(1, N);
        for i = 1:N
            [F(i), ep, oe, er] = abdor_fitness(robot, T_d, X(:, i));  fk = fk + 1;
            B = keep_best(B, F(i), X(:, i), ep, oe, er);
        end
        Pb = X;  Fp = F;
        [~, ig] = min(Fp);  G = Pb(:, ig);
        converged = B.f < P.f_stop;                                         % p.8

        history = zeros(1, T);
        t = 0;
        while ~converged && t < T
            if ikb_budget_left(opts) < N, break; end
            t = t + 1;
            w = pso_inertia(t, T, P);
            for i = 1:N
                [X(:, i), V(:, i)] = pso_move(X(:, i), V(:, i), Pb(:, i), G, ...
                                              w, P.C1, P.C2, vmax);         % Eqs. 1-3
            end
            X = min(max(X, q_min), q_max);
            for i = 1:N
                [F(i), ep, oe, er] = abdor_fitness(robot, T_d, X(:, i));  fk = fk + 1;
                B = keep_best(B, F(i), X(:, i), ep, oe, er);
            end
            [Pb, Fp] = pso_pbest(Pb, Fp, X, F);
            [~, ig] = min(Fp);  G = Pb(:, ig);                              % Gbest(t + 1)
            iters = t;
            history(t) = B.f;
            converged = B.f < P.f_stop;                                     % p.8
        end
        history = history(1:iters);
    end

    q_best = B.q;
    info = struct();
    info.solver    = 'PSO';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = converged;
    info.fitness   = B.f;
    info.err_pos   = B.ep;                                   % m
    info.err_rot   = B.er;                                   % rad
    info.O_error   = B.oe;                                   % Eq. 42
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.N         = N;
    info.T         = T;
    info.history   = history;
    info.params    = P;
end

% ------------------------------------------------------------------ operators
function P = pso_params()
    P = struct('N_6dof', 50, 'N_redundant', 100, 'T', 2000, 'C1', 2, 'C2', 0.5, ...
               'vel', 0.009, 'wu', 0.9, 'wl', 0.4, 'f_stop', 1e-3);
end

function N = pso_popsize(n, P)
% p.8: 50 for the UR5 (6-DOF), 100 for the SIA20D (7-DOF); 100 for every robot with n > 6.
    if n > 6, N = P.N_redundant; else, N = P.N_6dof; end
end

function w = pso_inertia(t, T, P)
% Inertia at iteration t of T: from wu (t = 0) to wl (t = T), linear, in the form of Eq. 8.
    w = P.wl + (T - t) / T * (P.wu - P.wl);
end

function [x, v] = pso_move(x, v, pb, g, w, C1, C2, vmax)
% Eq. 1 with r1, r2 ~ U(0, 1); Eq. 3 velocity bound; Eq. 2.
    r1 = rand();  r2 = rand();
    v = w * v + r1 * C1 * (pb - x) + r2 * C2 * (g - x);                     % Eq. 1
    v = min(max(v, -vmax), vmax);                                           % Eq. 3
    x = x + v;                                                              % Eq. 2
end

function [Pb, Fp] = pso_pbest(Pb, Fp, X, F)
% Pbest_i <- X_i if F_i < Fp_i.
    upd = F < Fp;
    Pb(:, upd) = X(:, upd);
    Fp(upd)    = F(upd);
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
