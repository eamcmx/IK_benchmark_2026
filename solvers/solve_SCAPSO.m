function [q_best, info] = solve_SCAPSO(robot, T_d, q0, opts)
% SOLVE_SCAPSO  Sine Cosine Particle Swarm Optimization IK solver as specified by Abdor-Sierra et
%               al. (2022), Sec. 2.3.
%
%   Source:
%     Javier Alexis Abdor-Sierra, Emmanuel Alejandro Merchán-Cruz, Ricardo Gustavo
%     Rodríguez-Cañizo (2022). A comparative analysis of metaheuristic algorithms for solving the
%     inverse kinematics of robot manipulators. Results in Engineering, 16, 100597.
%     DOI: 10.1016/j.rineng.2022.100597
%   The paper attributes SCAPSO to its Ref. [21]: F. Jin, J. Zhai (2020), Scapso-based inverse
%   kinematics method and its application to industrial robotic manipulator, 39th Chinese Control
%   Conference (CCC), pp. 5933-5938, doi 10.23919/CCC50068.2020.9188613; and the SCA to its
%   Ref. [22]: S. Mirjalili (2016), Knowl. Base Syst. 96, 120-133 (neither consulted).
%
%   Algorithm (pp.2-3: PSO Eqs. 1-3, Sec. 2.1; SCAPSO Eqs. 8-12, Sec. 2.3; settings p.8;
%   fitness pp.6-7, Eqs. 34-42 and 44; stop rule p.8):
%     A swarm of N particles is evaluated. Each iteration t = 1..T moves every particle, then
%     evaluates all and updates Pbest and Gbest. PSO move (Eqs. 1-3):
%       V <- w*V + r1*C1*(Pbest_i - X) + r2*C2*(Gbest - X),  V bounded by maxvel = vel*(ub - lb)
%       and minvel = -maxvel,  X <- X + V.
%     C1 = C1min + (T - t)/T*(C1max - C1min) falls and C2 = C2max - (T - t)/T*(C2max - C2min) rises
%     (Eqs. 8-9). SCA move (Eqs. 10-12):
%       X <- X + r1*sin(r2)*|r3*Gbest - X| if r4 < 0.5,  X <- X + r1*cos(r2)*|r3*Gbest - X| otherwise,
%       r1 = R*(T - t)/T,  r2 ~ U(0, 2*pi),  r3, r4 ~ U(0, 1).
%     Switch (p.3): Trial(i) tracks how many times particle i does not improve its Pbest, and
%     "Limit and EPS switches between SCA and PSO". Implemented reading: Trial(i) counts the
%     iterations since particle i last improved its Pbest by more than EPS (such an improvement
%     resets it to 0; every other iteration adds 1); particle i takes the SCA move while
%     Trial(i) >= Limit and the PSO move otherwise. Limit therefore switches a stagnant particle
%     from PSO to SCA and an improvement larger than EPS switches it back.
%     The run ends when the best fitness is below 1e-3 or after T iterations (p.8).
%     Fitness (Eq. 44): f = Perror + Oerror*w, Perror = ||p_t - p_d|| (Eq. 34),
%     Oerror = |q0t - q0d| + ||q(1:3)t - q(1:3)d|| (Eqs. 40-42), with the unit quaternions of the
%     trialed and desired rotations from Eqs. 36-39 (sgn(0) = 1, Eq. 39).
%
%   Parameters (p.8 unless stated):
%     N = 50, T = 2000. Inside the benchmark T is the budget horizon (see Benchmark interface).
%     C1max = 2, C1min = 1, C2max = 2, C2min = 1.
%     vel = 0.009 (Eq. 3), applied per joint to q_max - q_min.
%     wu = 0.9, wl = 0.4: inertia w falls from wu to wl as w = wl + (T - t)/T*(wu - wl). The
%           schedule is not stated (pp.2-3, 8 checked; p.2 only notes that reducing w improved the
%           algorithm); the linear form of Eq. 8 is used.
%     Limit = 50, EPS = 1e-3 (switch parameters, p.3; EPS in units of f).
%     R = 1. p.3 states only R > 0 and gives r1 the range [0, 1]; R = 1 is the largest value that
%           keeps r1 = R*(T - t)/T in [0, 1].
%     r1, r2 (Eq. 1) and r2, r3, r4 (Eqs. 10-11): one draw per particle and iteration, as the
%           equations are printed (scalars multiplying vectors); per-joint draws are not stated.
%     w_f = 200 for 6-DOF robots, 300 for robots with more than 6 joints (the weight of Eq. 44).
%           p.7 sets it experimentally per robot: 200 for the UR5 (6-DOF), 300 for the SIA20D
%           (7-DOF); mapping by DOF is the nearest reading for robots the paper did not use.
%     f_stop = 1e-3. p.8: a solution is optimal when "the error being evaluated is less than
%           1e-03", one of the two stop criteria; applied to f of Eq. 44. It is a separate setting
%           from the switch parameter EPS, which has the same value.
%     V(0) = 0: initial velocity not stated (pp.2-3, 8 checked).
%
%   Benchmark interface:
%     Budget: initialisation and every iteration need N = 50 FK calls; an iteration starts only if
%       ikb_budget_left(opts) covers it. T = ikb_generations(opts, 50, 50, 2000) = 39 iterations with
%       opts.budget = 2000, independent of the DOF (outside the benchmark T = 2000). With T = 39 <
%       Limit = 50, Trial(i) cannot reach Limit, so within the budget every move is a PSO move.
%       No final FK check. opts.max_iter and opts.pop_size are ignored.
%     Warm start: q0 (clamped to the limits) is particle 1 of the initial swarm, evaluated
%       unperturbed; Gbest is the best evaluated position, so q0 enters the first Gbest selection.
%     Units: benchmark lengths are metres; the paper's DH tables are in mm (p.5, Tables 2-3), so
%       Perror is computed in mm: f = 1000*||p_t - p_d||[m] + w_f*Oerror.
%     Joint limits: Eq. 44 bounds theta (p.7) without a handling rule; Eq. 3 bounds the velocity.
%       Every new position is clamped to [q_min, q_max]; the velocity is not changed by the clamp.
%     Output: Gbest (lowest f evaluated). info.converged = best f < 1e-3 (p.8). info.n_sca counts
%       SCA moves. info.err_rot is the rotation angle of Gbest (rad), reported only.
%       solve_SCAPSO('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Switch rule: the reading above of the p.3 sentence; the paper defines neither the
%       comparison nor the reset (Ref. [21] not consulted).
%     - Inertia schedule linear from wu to wl (not stated, see Parameters).
%     - R = 1 (only R > 0 stated, see Parameters).
%     - Eq. 1 is printed with a missing closing parenthesis; the evident reading is used.
%     - Updates are synchronous: all particles of iteration t move with Gbest(t) (the Gbest(t) of
%       Eq. 10), and Pbest, Trial and Gbest are updated after all are evaluated (order not stated).
%     - V is kept unchanged during SCA moves (not stated).
%     - w_f mapped by DOF (see Parameters).
%     - The iteration count is the budget horizon T = 39 instead of 2000 (benchmark rule).
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
                        'weight', @abdor_weight, 'schedule', @scapso_schedule, ...
                        'pso', @scapso_pso, 'sca', @scapso_sca, 'pbest', @scapso_pbest, ...
                        'params', scapso_params());
        info = [];
        return
    end

    t_start = tic;
    P = scapso_params();
    N = P.N;

    n     = robot.n;
    q_min = robot.q_min(:);
    q_max = robot.q_max(:);
    q0    = min(max(q0(:), q_min), q_max);
    vmax  = P.vel * (q_max - q_min);                                        % Eq. 3
    T     = ikb_generations(opts, N, N, P.T);

    B = struct('f', NaN, 'q', q0, 'ep', NaN, 'oe', NaN, 'er', NaN);
    history = zeros(1, 0);  iters = 0;  fk = 0;  converged = false;  n_sca = 0;

    if ikb_budget_left(opts) >= N
        % Initial swarm, q0 is particle 1
        X = q_min + rand(n, N) .* (q_max - q_min);  X(:, 1) = q0;
        V = zeros(n, N);
        F = zeros(1, N);
        for i = 1:N
            [F(i), ep, oe, er] = abdor_fitness(robot, T_d, X(:, i));  fk = fk + 1;
            B = keep_best(B, F(i), X(:, i), ep, oe, er);
        end
        Pb = X;  Fp = F;  trial = zeros(1, N);
        [~, ig] = min(Fp);  G = Pb(:, ig);
        converged = B.f < P.f_stop;                                         % p.8

        history = zeros(1, T);
        t = 0;
        while ~converged && t < T
            if ikb_budget_left(opts) < N, break; end
            t = t + 1;
            [w, C1, C2, r1] = scapso_schedule(t, T, P);                     % Eqs. 8, 9, 12
            for i = 1:N
                if trial(i) >= P.Limit                                      % switch (p.3)
                    X(:, i) = scapso_sca(X(:, i), G, r1);                   % Eqs. 10-11
                    n_sca = n_sca + 1;
                else
                    [X(:, i), V(:, i)] = scapso_pso(X(:, i), V(:, i), Pb(:, i), G, ...
                                                    w, C1, C2, vmax);       % Eqs. 1-3
                end
            end
            X = min(max(X, q_min), q_max);
            for i = 1:N
                [F(i), ep, oe, er] = abdor_fitness(robot, T_d, X(:, i));  fk = fk + 1;
                B = keep_best(B, F(i), X(:, i), ep, oe, er);
            end
            [Pb, Fp, trial] = scapso_pbest(Pb, Fp, trial, X, F, P.eps);
            [~, ig] = min(Fp);  G = Pb(:, ig);                              % Gbest(t + 1)
            iters = t;
            history(t) = B.f;
            converged = B.f < P.f_stop;                                     % p.8
        end
        history = history(1:iters);
    end

    q_best = B.q;
    info = struct();
    info.solver    = 'SCAPSO';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = converged;
    info.fitness   = B.f;
    info.err_pos   = B.ep;                                   % m
    info.err_rot   = B.er;                                   % rad
    info.O_error   = B.oe;                                   % Eq. 42
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.n_sca     = n_sca;
    info.T         = T;
    info.history   = history;
    info.params    = P;
end

% ------------------------------------------------------------------ operators
function P = scapso_params()
    P = struct('N', 50, 'T', 2000, 'C1min', 1, 'C1max', 2, 'C2min', 1, 'C2max', 2, ...
               'vel', 0.009, 'wu', 0.9, 'wl', 0.4, 'Limit', 50, 'eps', 1e-3, 'R', 1, ...
               'f_stop', 1e-3);
end

function [w, C1, C2, r1] = scapso_schedule(t, T, P)
% Eqs. 8, 9 and 12 at iteration t of T; inertia from wu to wl in the form of Eq. 8.
    s  = (T - t) / T;
    C1 = P.C1min + s * (P.C1max - P.C1min);                                 % Eq. 8
    C2 = P.C2max - s * (P.C2max - P.C2min);                                 % Eq. 9
    r1 = P.R * s;                                                           % Eq. 12
    w  = P.wl + s * (P.wu - P.wl);
end

function [x, v] = scapso_pso(x, v, pb, g, w, C1, C2, vmax)
% Eq. 1 with r1, r2 ~ U(0, 1); Eq. 3 velocity bound; Eq. 2.
    r1 = rand();  r2 = rand();
    v = w * v + r1 * C1 * (pb - x) + r2 * C2 * (g - x);                     % Eq. 1
    v = min(max(v, -vmax), vmax);                                           % Eq. 3
    x = x + v;                                                              % Eq. 2
end

function x = scapso_sca(x, g, r1)
% Eqs. 10-11 with r2 ~ U(0, 2*pi), r3, r4 ~ U(0, 1); r1 from Eq. 12.
    r2 = 2 * pi * rand();  r3 = rand();  r4 = rand();
    if r4 < 0.5
        x = x + r1 * sin(r2) * abs(r3 * g - x);                             % Eq. 10
    else
        x = x + r1 * cos(r2) * abs(r3 * g - x);                             % Eq. 11
    end
end

function [Pb, Fp, trial] = scapso_pbest(Pb, Fp, trial, X, F, eps_sw)
% Pbest_i <- X_i if F_i < Fp_i. Trial(i) (p.3): reset to 0 by an improvement larger than EPS,
% otherwise incremented.
    upd = F < Fp;
    imp = F < Fp - eps_sw;
    Pb(:, upd) = X(:, upd);
    Fp(upd)    = F(upd);
    trial(imp)  = 0;
    trial(~imp) = trial(~imp) + 1;
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
