function [q_best, info] = solve_BRPSO(robot, T_d, q0, opts)
% SOLVE_BRPSO  Boundary restricted particle swarm optimization (BRPSO) IK solver of Ghosh, Singh and
%              Ray (2022), Sec. 2.2.
%
%   Source:
%     Arabinda Ghosh, Omkar Singh, Anjan Kumar Ray (2022). Inverse Kinematic Solution of a 7 DOF
%     Robotic Manipulator using Boundary Restricted Particle Swarm Optimization. IFAC PapersOnLine,
%     55(1), 101-105 (printed "55-1"). DOI: 10.1016/j.ifacol.2022.04.017
%   Page numbers are PDF pages 1-5 (printed page = PDF page + 100).
%
%   Algorithm (Sec. 2.2, Eqs. 6-18, pp.3-4; settings Sec. 3, p.4):
%     A particle is a joint vector (Eqs. 6-7) whose initial values lie inside the joint ranges
%     (Eq. 8). The objective is the Euclidean distance between the current and the desired
%     end-effector positions, subject to the joint ranges (Eq. 9, p.3); it has no orientation term.
%     Each iteration t -> t+1 moves every particle i in every dimension n:
%       V(t+1) = omega V(t) + c1 r1 (p_i(b_i) - p(t)) + c2 r2 (p_u(w) - p(t))       (Eq. 14)
%       p(t+1) = p(t) + 1/2 rand (A - B)                                           (Eq. 15)
%       A = min[|V(t+1)|, |R_u - p(t)|] x [1 + sgn(V(t+1))]                         (Eq. 16)
%       B = min[|V(t+1)|, |R_l - p(t)|] x [1 - sgn(V(t+1))]                         (Eq. 17)
%       sgn(V) = -1 if V < 0, 0 if V = 0, 1 if V > 0                                (Eq. 18)
%     R_u and R_l are the upper and lower limits of joint n (p.4); A and B are step lengths, not
%     bounds. Eq. 15 replaces the usual p + V for every particle, dimension and iteration.
%     Closed form of Eqs. 15-18 for p(t) inside [R_l, R_u] (derived):
%       V > 0:  p(t+1) = p(t) + rand min(V, R_u - p(t))
%       V < 0:  p(t+1) = p(t) - rand min(|V|, p(t) - R_l)
%       V = 0:  p(t+1) = p(t)
%     Each joint moves a random fraction of its new velocity, cut at the limit it heads for, so it
%     stays inside [R_l, R_u]. Eqs. 15-18 leave V(t+1) unchanged; it enters the next Eq. 14.
%     After the moves all N particles are evaluated; p_i(b_i) is particle i's best position over
%     iterations 0..t+1 (Eqs. 10-11) and p_u(w) the best position of all particles over
%     iterations 0..t+1 (Eqs. 12-13). All particles of one iteration use the same p_u(w).
%
%   Parameters (Sec. 3, p.4 unless stated):
%     N = 100          "the number of particles considered are 100".
%     T = 500          "the maximum number of iterations is 500", the only setting reported;
%                      Fig. 2 (p.4) runs to iteration 500. Inside the benchmark T is the budget
%                      horizon (Benchmark interface).
%     c1 = c2 = 1.494, omega = 0.729   "In BRPSO, c1 = c2 = 1.494, and omega = 0.729 are
%                      considered" (the paper cites Kanoosh et al., 2019, for the values).
%     r1, r2           U(0, 1) ("uniformly distributed between (0,1)", p.3), one draw per particle,
%                      dimension and iteration: Eq. 14 is written per dimension n and r1, r2 carry
%                      no index; per-dimension draws are not stated (p.3).
%     rand (Eq. 15)    U(0, 1), one draw per particle, dimension and iteration. Eq. 15 is written
%                      per dimension n; the distribution of rand is not stated in the paper
%                      (checked pages 3-4); U(0, 1) is used, as for r1, r2.
%     V(0) = 0         initial velocity not stated in the paper (checked pages 1-5); zero is used.
%                      No velocity limit is stated (pp.3-4) and none is applied.
%
%   Benchmark interface:
%     Budget: the initialisation needs N = 100 FK calls and every iteration N (one evaluation per
%       particle). T = ikb_generations(opts, 100, 100, 500) = 19 iterations, 2000 FK calls per
%       waypoint; a 40-member population would give ikb_generations(opts, 40, 40, 500) = 49. The
%       initialisation and every iteration start only if ikb_budget_left(opts) covers N. Outside
%       the benchmark T = 500 (50,100 FK calls). No final FK check. No parameter depends on T, so
%       T sets only the number of iterations. opts.max_iter, opts.pop_size and every other option
%       field are ignored.
%     Stop: T iterations only. The paper states no tolerance or other stop rule (pp.3-4; Fig. 2
%       runs to iteration 500). info.converged is always false: the paper has no convergence
%       criterion.
%     Warm start: q0 (clamped to the limits, Eq. 8) is particle 1 of the initial swarm, evaluated
%       unperturbed. The other N - 1 particles are drawn uniformly in [q_min, q_max]: Eq. 8 gives
%       the ranges only, the distribution is not stated (p.3). p_u(w) is replaced only by a
%       strictly lower objective, so q0 stays the global best until a particle improves on it.
%     Robot and units: Eqs. 1-5 and Table 1 (the paper's 7-DOF arm) are replaced by robot.fk and
%       the robot's limits, R_u = q_max and R_l = q_min, with n = robot.n joints instead of 7.
%       Table 1 (p.2) gives lengths in metres, as the benchmark, so the objective is in metres. A
%       position-only Euclidean distance has the same minimiser in m and mm, and the objective
%       enters the algorithm only through comparisons, so the length unit does not change the
%       search (up to rounding). Eqs. 14-18 are linear in the joint coordinates and hold no
%       constant in angle units, so radians (benchmark) and degrees (Table 1, Eq. 8) give the
%       same search (up to rounding).
%     Joint limits: handled by Eqs. 15-18 alone. They keep every joint inside [q_min, q_max]
%       (closed form above; in floating point a crossing needs rand within 2^-52 of 1), so there is
%       no clamp, re-sampling or velocity reset.
%     Output: p_u(w), the lowest objective evaluated. info.fitness = info.err_pos = its objective
%       (m); info.err_rot (rotation angle, rad) is reported only. solve_BRPSO('ops') returns
%       handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration horizon T = 19 instead of 500 (benchmark rule); no parameter depends on T.
%     - The benchmark robot's closed limits [q_min, q_max]; Eqs. 8-9 write open intervals.
%     - Uniform initial swarm with q0 as particle 1 (distribution not stated, see Warm start).
%     - r1, r2 and rand drawn per dimension, rand ~ U(0, 1), V(0) = 0 (not stated, see
%       Parameters).
%     - Ties in Eqs. 11 and 13 (not stated): the earliest iteration, then the lowest particle
%       index.
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
        q_best = struct('params', brpso_params(), 'core', @brpso_core, ...
                        'velocity', @brpso_velocity, 'position', @brpso_position, ...
                        'sgn', @brpso_sgn, 'best', @brpso_best, 'fitness', @brpso_fitness);
        info = [];
        return
    end
    [q_best, info] = brpso_core(robot, T_d, q0, opts, brpso_params());
end

% ==================================================================== main loop (Sec. 2.2)
function [q_best, info] = brpso_core(robot, T_d, q0, opts, P)
% Initial swarm (Eq. 8), then per iteration Eq. 14, Eqs. 15-18, N evaluations (Eq. 9) and the
% best updates (Eqs. 10-13). P holds N, T, c1, c2 and omega.
    t_start = tic;
    N  = P.N;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    n  = numel(lo);
    q0 = min(max(q0(:), lo), hi);                                       % Eq. 8 ranges
    T  = ikb_generations(opts, N, N, P.T);

    fg = NaN;  G = q0;  Tg = NaN(4);  fk = 0;  iters = 0;  hist = zeros(1, 0);
    fin = struct('X', [], 'V', [], 'Pb', [], 'Fp', []);

    if ikb_budget_left(opts) >= N
        X = lo + rand(n, N) .* (hi - lo);                               % Eq. 8, uniform
        X(:, 1) = q0;
        V = zeros(n, N);                                                % V(0), not stated
        F = zeros(1, N);  Tx = zeros(4, 4, N);
        for i = 1:N
            [F(i), Tx(:, :, i)] = brpso_fitness(robot, T_d, X(:, i));
            fk = fk + 1;
        end
        Pb = X;  Fp = F;                                                % Eqs. 10-11, k = 0
        [fg, G, Tg] = brpso_best(fg, G, Tg, F, X, Tx);                  % Eqs. 12-13, k = 0

        hist = zeros(1, T);
        t = 0;
        while t < T
            if ikb_budget_left(opts) < N, break; end
            t = t + 1;
            V = brpso_velocity(V, X, Pb, G, P);                         % Eq. 14
            X = brpso_position(X, V, lo, hi);                           % Eqs. 15-18
            for i = 1:N
                [F(i), Tx(:, :, i)] = brpso_fitness(robot, T_d, X(:, i));
                fk = fk + 1;
            end
            upd = F < Fp;                                               % Eqs. 10-11
            Pb(:, upd) = X(:, upd);
            Fp(upd)    = F(upd);
            [fg, G, Tg] = brpso_best(fg, G, Tg, F, X, Tx);              % Eqs. 12-13
            iters = t;
            hist(t) = fg;
        end
        hist = hist(1:iters);
        fin = struct('X', X, 'V', V, 'Pb', Pb, 'Fp', Fp);
    end

    er = NaN;
    if ~isnan(fg)
        er = acos(min(1, max(-1, (trace(T_d(1:3, 1:3).' * Tg(1:3, 1:3)) - 1) / 2)));
    end

    q_best = G;
    info = struct();
    info.solver    = 'BRPSO';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = false;                                             % no criterion in the paper
    info.fitness   = fg;                                                % Eq. 9, m
    info.err_pos   = fg;                                                % m
    info.err_rot   = er;                                                % rad, reporting only
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.N         = N;
    info.T         = T;
    info.history   = hist;                                              % O^gb after each iteration
    info.final     = fin;
    info.params    = P;
end

% ==================================================================== operators
function P = brpso_params()
% Sec. 3, p.4: 100 particles, 500 iterations, c1 = c2 = 1.494, omega = 0.729.
    P = struct('N', 100, 'T', 500, 'c1', 1.494, 'c2', 1.494, 'omega', 0.729);
end

function V = brpso_velocity(V, X, Pb, G, P)
% Eq. 14: omega V + c1 r1 (p_i(b_i) - p) + c2 r2 (p_u(w) - p); r1, r2 ~ U(0, 1) per particle and
% dimension.
    R1 = rand(size(X));
    R2 = rand(size(X));
    V = P.omega * V + P.c1 * R1 .* (Pb - X) + P.c2 * R2 .* (G - X);
end

function X = brpso_position(X, V, lo, hi)
% Eqs. 15-18 with R_u = hi, R_l = lo; rand ~ U(0, 1) per particle and dimension; V is not changed.
    s = brpso_sgn(V);                                                   % Eq. 18
    A = min(abs(V), abs(hi - X)) .* (1 + s);                            % Eq. 16
    B = min(abs(V), abs(lo - X)) .* (1 - s);                            % Eq. 17
    X = X + 1/2 * rand(size(X)) .* (A - B);                             % Eq. 15
end

function s = brpso_sgn(V)
% Eq. 18: -1 if V < 0, 0 if V = 0, 1 if V > 0.
    s = double(V > 0) - double(V < 0);
end

function [fg, G, Tg] = brpso_best(fg, G, Tg, F, X, Tx)
% Eqs. 12-13: global best over all particles and iterations so far. Replaced only by a strictly
% lower objective; ties go to the earliest iteration, then to the lowest particle index.
    [fm, im] = min(F);
    if isnan(fg) || fm < fg
        fg = fm;
        G  = X(:, im);
        Tg = Tx(:, :, im);
    end
end

% ==================================================================== objective (Eq. 9)
function [f, T] = brpso_fitness(robot, T_d, q)
% Eq. 9: sqrt((Mx - Mx^d)^2 + (My - My^d)^2 + (Mz - Mz^d)^2), metres (Table 1); one FK call.
    T = robot.fk(q);
    f = sqrt((T(1, 4) - T_d(1, 4))^2 + (T(2, 4) - T_d(2, 4))^2 + (T(3, 4) - T_d(3, 4))^2);
end
