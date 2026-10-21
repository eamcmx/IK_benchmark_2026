function [q_best, info] = solve_HS(robot, T_d, q0, opts)
% SOLVE_HS  Harmony Search IK solver as benchmarked by Abdor-Sierra et al. (2022), Sec. 2.10.
%
%   Source:
%     Javier Alexis Abdor-Sierra, Emmanuel Alejandro Merchán-Cruz, Ricardo Gustavo
%     Rodríguez-Cañizo (2022). A comparative analysis of metaheuristic algorithms for solving the
%     inverse kinematics of robot manipulators. Results in Engineering, 16, 100597.
%     DOI: 10.1016/j.rineng.2022.100597
%   The paper describes HS in one paragraph without equations (p.4, Sec. 2.10) and attributes it to
%   its Ref. [34]: Z.W. Geem, J.H. Kim, G.V. Loganathan (2001), A new heuristic optimization
%   algorithm: harmony search, Simulation 76(2), 60-68 (not consulted).
%
%   Algorithm (p.4, Sec. 2.10; settings p.9; fitness pp.6-7, Eqs. 34-42 and 44; stop rule p.8):
%     A harmony memory (the population) of HMS joint vectors is evaluated. Each iteration improvises
%     one child joint by joint (p.4: the band is a solution, each musician an independent variable):
%     memory consideration with rate HMCR (p.4: "the crossover operator") copies that joint from a
%     uniformly chosen member; a memory-considered value is pitch-adjusted with rate PAR (p.4: "the
%     mutation operator") by Gaussian noise whose standard deviation is the distance bandwidth DB;
%     without memory consideration the value is drawn uniformly in [q_min, q_max]. If the child is
%     better than the worst member of the memory, the child replaces it (p.4). The run ends when the
%     best fitness is below 1e-3 or after T iterations (p.8).
%     Fitness (Eq. 44): f = Perror + Oerror*w, Perror = ||p_t - p_d|| (Eq. 34),
%     Oerror = |q0t - q0d| + ||q(1:3)t - q(1:3)d|| (Eqs. 40-42), with the unit quaternions of the
%     trialed and desired rotations from Eqs. 36-39 (sgn(0) = 1, Eq. 39).
%
%   Parameters:
%     HMS  = 500    p.9: "Population size = 500".
%     T    = 1000   p.9: "maximum iterations = 1000". Inside the benchmark T is the budget horizon
%                   (see Benchmark interface).
%     HMCR = 0.95   p.9.
%     PAR  = 0.1    p.9.
%     bw   = 20 deg = 0.3491 rad, every joint. p.9 gives "bw = 20" without a unit; p.4 defines the
%                   distance bandwidth DB as the standard deviation of the Gaussian mutation and does
%                   not say that bw and DB are the same symbol. The unit is inferred from the paper's
%                   DH tables, whose joint ranges are in degrees (p.5, Tables 2-3); 20 rad would exceed
%                   every joint range.
%     w    = 200 for 6-DOF robots, 300 for robots with more than 6 joints. p.7 sets w
%                   experimentally per robot: 200 for the UR5 (6-DOF), 300 for the SIA20D (7-DOF);
%                   mapping by DOF is the nearest reading for robots the paper did not use.
%     f_stop = 1e-3 p.8: a solution is optimal when "the error being evaluated is less than 1e-03",
%                   one of the two stop criteria; applied to f of Eq. 44.
%     Iteration = one improvisation (one child, one FK call). Not defined in the paper (pp.4, 8-9
%                   checked); paper defers to Geem et al. 2001 (not consulted).
%
%   Benchmark interface:
%     Budget: the initial memory needs HMS = 500 FK calls and each iteration 1. An iteration starts
%       only if ikb_budget_left(opts) covers it. T = ikb_generations(opts, 500, 1, 1000) = 1500
%       iterations with opts.budget = 2000, independent of the DOF (outside the benchmark T = 1000).
%       No final FK check. opts.max_iter and opts.pop_size are ignored.
%     Warm start: q0 (clamped to the limits) is member 1 of the initial memory, evaluated
%       unperturbed; the memory never loses its best member.
%     Units: benchmark lengths are metres; the paper's DH tables are in mm (p.5, Tables 2-3), so
%       Perror is computed in mm: f = 1000*||p_t - p_d||[m] + w*Oerror.
%     Joint limits: Eq. 44 bounds theta (p.7) without a handling rule, so every child is clamped to
%       [q_min, q_max].
%     Output: the best harmony (lowest f). info.converged = best f < 1e-3 (p.8). info.err_rot is the
%       rotation angle of the best harmony (rad), reported only. solve_HS('ops') returns handles to
%       the operators for unit tests.
%
%   Deviations from the paper:
%     - Memory consideration per joint from a uniformly chosen member, the uniform branch with
%       probability 1 - HMCR, pitch adjustment of memory-considered values only, and one child per
%       iteration: p.4 names HMCR, PAR and DB without formulas; paper defers to Geem et al. 2001
%       (not consulted); standard reading of the operator names.
%     - bw read as 20 degrees (unit not stated, see Parameters).
%     - w mapped by DOF (see Parameters).
%     - The iteration count is the budget horizon T = 1500 instead of 1000 (benchmark rule).
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
                        'weight', @abdor_weight, 'improvise', @hs_improvise, ...
                        'replace', @hs_replace, 'params', hs_params());
        info = [];
        return
    end

    t_start = tic;
    P   = hs_params();
    HMS = P.HMS;

    n     = robot.n;
    q_min = robot.q_min(:);
    q_max = robot.q_max(:);
    q0    = min(max(q0(:), q_min), q_max);
    bw    = P.bw * ones(n, 1);
    T     = ikb_generations(opts, HMS, 1, P.T);

    B = struct('f', NaN, 'q', q0, 'ep', NaN, 'oe', NaN, 'er', NaN);
    history = zeros(1, 0);  iters = 0;  fk = 0;  converged = false;

    if ikb_budget_left(opts) >= HMS
        % Initial harmony memory, q0 is member 1
        HM = q_min + rand(n, HMS) .* (q_max - q_min);  HM(:, 1) = q0;
        F  = zeros(1, HMS);
        for i = 1:HMS
            [F(i), ep, oe, er] = abdor_fitness(robot, T_d, HM(:, i));  fk = fk + 1;
            B = keep_best(B, F(i), HM(:, i), ep, oe, er);
        end
        converged = B.f < P.f_stop;                                         % p.8

        history = zeros(1, T);
        t = 0;
        while ~converged && t < T
            if ikb_budget_left(opts) < 1, break; end
            t = t + 1;
            x = min(max(hs_improvise(HM, q_min, q_max, P.HMCR, P.PAR, bw), q_min), q_max);
            [fx, ep, oe, er] = abdor_fitness(robot, T_d, x);  fk = fk + 1;
            [HM, F] = hs_replace(HM, F, x, fx);                             % p.4
            B = keep_best(B, fx, x, ep, oe, er);
            iters = t;
            history(t) = B.f;
            converged = B.f < P.f_stop;                                     % p.8
        end
        history = history(1:iters);
    end

    q_best = B.q;
    info = struct();
    info.solver    = 'HS';
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
function P = hs_params()
    P = struct('HMS', 500, 'T', 1000, 'HMCR', 0.95, 'PAR', 0.1, 'bw', 20 * pi / 180, ...
               'f_stop', 1e-3);
end

function x = hs_improvise(HM, q_min, q_max, HMCR, PAR, bw)
% One child, joint by joint: with probability HMCR the joint value of a uniformly chosen member,
% pitch-adjusted with probability PAR by bw*N(0, 1); otherwise U(q_min, q_max).
    [n, HMS] = size(HM);
    x = zeros(n, 1);
    for d = 1:n
        if rand() < HMCR
            x(d) = HM(d, randi(HMS));
            if rand() < PAR
                x(d) = x(d) + bw(d) * randn();
            end
        else
            x(d) = q_min(d) + rand() * (q_max(d) - q_min(d));
        end
    end
end

function [HM, F, j] = hs_replace(HM, F, x, fx)
% p.4: the child replaces the worst member if it is better. j = replaced index (0 if none).
    [fw, j] = max(F);
    if fx < fw
        HM(:, j) = x;  F(j) = fx;
    else
        j = 0;
    end
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
