function [q_best, info] = solve_CVS(robot, T_d, q0, opts)
% SOLVE_CVS  Chaos-based Vortex Search IK solver (CVS3, Gauss-Mouse map) of Toz (2020), Secs. 2 and 4.
%
%   Source:
%     Metin Toz (2020). Chaos-based Vortex Search algorithm for solving inverse kinematics problem of
%     serial robot manipulators with offset wrist. Applied Soft Computing Journal, 89, 106074.
%     DOI: 10.1016/j.asoc.2020.106074
%   Page numbers are PDF pages 1-22 (PDF page = printed page).
%
%   Algorithm (Sec. 2.1, Eqs. 1-9, pp.3-4; Sec. 2.3, Eqs. 10-14, pp.4-6; Table 1, p.5; IK problem
%   Sec. 4.1, Eqs. 19-21, p.18; settings Table 6, p.19):
%     Vortex Search keeps one centre mu and, in iteration t = 1..Maxt, samples k candidates around
%     it from a spherical Gaussian whose standard deviation is the radius r (Eqs. 2-4, Step 2),
%     evaluates them (Step 3) and then decreases the radius (Step 4):
%       s_i = mu + r N(0, I_n),  i = 1..k                                        (Eqs. 2-4)
%       sigma0 = (max(ub) - min(lb)) / 2                                          (Eq. 5)
%       r_0 = sigma0 (1/x) gammaincinv(x, a_0)       radius of the first iteration (Eq. 7)
%       a_t = a_0 - t/Maxt                                                        (Eq. 9)
%     CVS replaces the Eq. 8 radius by the chaotic radius of proposed method 1, computed in Step 4
%     of iteration t and used by iteration t + 1:
%       r_t = (sigma0 + |fb_ratio| C_m(t)) (1/x) gammaincinv(x, a_t)   if r < 0.5
%       r_t = (sigma0 - |fb_ratio| C_m(t)) (1/x) gammaincinv(x, a_t)   otherwise   (Eq. 10)
%       C_m(t) = (CM(t) - CM_min) N_ce(t) / (CM_max - CM_min)                     (Eq. 11)
%       N_ce(t) = N_cemax - (t/Maxt) (N_cemax - N_cemin)                          (Eq. 12)
%       fb_ratio = 1 if t = 1; fb_ratio = f_min/f_best and f_best = f_min if f_min < f_best;
%       then fb_ratio = 2r if |fb_ratio| > 2, fb_ratio = r if |fb_ratio| < 1      (Eq. 13)
%     and replaces the best update of Step 3 by proposed method 2:
%       if f_min < g_min:  g_min = f_min, mu = itrBest
%       elseif r < 0.5:    C_mu = C_m(t) itrBest;  if f(C_mu) < g_min: g_min = f(C_mu), mu = C_mu
%                                                                                 (Eq. 14)
%     f_min is the lowest candidate fitness of the iteration, itrBest the best solution so far
%     (the candidate that gave f_min when it improves g_min, otherwise the current centre), and r
%     a random number in [0, 1] (p.5). CM(t) is the Gauss-Mouse map of CVS3, the variant selected
%     for the IK problem (p.9), advanced once per iteration from CM(1) = x_0 = 0.7 (Table 1).
%     IK fitness (Eq. 21, p.18) of a candidate C_s for trajectory point j, position only (Eq. 19):
%       fitness = (err sum(|C_sj - F_s,j-1|) + ||P_desired,j - P_candidate,j||)^2
%     with F_s,j-1 the best solution of point j - 1 (F_s0 = zero vector, the zero position).
%
%   Parameters:
%     k = 50          candidate solutions per iteration (Table 6, p.19; also Sec. 3.1, p.6).
%     Maxt = 100      Table 6 and p.19 ("the Maxt to be only 100 for both of the algorithms"),
%                     the only IK setting reported (k x Maxt = 5000). The other runs, 10,000
%                     (Sec. 3.1, p.6), 1000 and 100 (Sec. 3.2, p.9) and 100,000 (VS, Sec. 3.3, p.9),
%                     are on the benchmark functions. Inside the benchmark Maxt is the budget
%                     horizon T.
%     N_cemax = 0.5, N_cemin = 1E-21   Table 6 (IK); Sec. 3.1 uses 0.1 and 1E-21 for the functions.
%                     In double precision 0.5 - 1E-21 = 0.5, so N_ce(Maxt) = 0 and N_cemin has no
%                     numerical effect.
%     a_0 = 1, x = 0.1                 Table 6; p.4.
%     err = 100                        Table 6.
%     CM              Gauss-Mouse map, Table 1 row 3 (p.5), CM(1) = x_0 = 0.7 (Table 1 caption).
%                     The printed definition "1/mod(x_i, 1) otherwise" leaves the stated interval
%                     (0, 1) at the first step (1/mod(0.7, 1) = 1.43). x_(i+1) = mod(1/x_i, 1) in
%                     double precision gives 0.7, 0.4286, 0.3333, 2.7e-15, 0.3125, 0.2, 1 - 4.4e-15,
%                     4.4e-15, 0.8125, 0.2308, ... and C_m(1), C_m(7), C_m(9) = 0.0693, 0.0930,
%                     0.0739 for N_cemax = 0.1 and Maxt = 100, the values plotted in Fig. 2a and 2b
%                     (p.6), so that form is used. In exact arithmetic the sequence would reach 0 at
%                     its fourth value (7/10, 3/7, 1/3, 0); Fig. 2a continues with the double-precision
%                     values, which is the map used here. The value for x_i = 0 is not printed; in
%                     double precision the sequence from 0.7 enters a cycle of period 6 at t = 11
%                     without reaching 0, so the case does not arise.
%     CM_min = 0, CM_max = 1   the interval (0, 1) of the Gauss-Mouse map (Table 1).
%     r               U(0, 1), a separate draw in each of Eqs. 10, 13 and 14 (p.5: "r is a random
%                     number in [0, 1] interval").
%     g_min at t = 1  not stated in the paper (checked pages 3-6, 18-19). Step 3 (p.4) updates the
%                     current best solution only if a candidate is better, and the initial best
%                     solution is the centre mu_0 (Step 1), so g_min starts at f(mu_0).
%     f_best at t = 1 Eq. 13 sets fb_ratio = 1 at t = 1; f_best ("the best fitness value obtained
%                     so far", p.5) is then f_min of iteration 1.
%     Radius timing   p.4: the radius of the first iteration is r_0 (Eq. 7), and the radius is
%                     decreased in Step 4, after the evaluation of Step 3. So the radius r_t of
%                     Eq. 10, built from C_m(t) and the fb_ratio of iteration t, is used by
%                     iteration t + 1; iteration t samples with a_(t-1) = 1 - (t - 1)/Maxt, from 1 down
%                     to 1/Maxt. CM(t) and C_m(t) carry the iteration index t, as in Fig. 2b
%                     (C_m(1) = 0.7 N_ce(1)).
%     Units           Positions in mm: the paper's arm and errors are in mm (Table 6, p.19;
%                     Fig. 12, p.20). The unit of the joint angles in C_s is not stated in the
%                     paper (checked pages 12-21); the angles of Table 5 (p.18, twists alpha) are in
%                     radians ("radyan"), so radians are used.
%     Stop            Maxt iterations ("both of the algorithms reached the maximum NOI", p.19;
%                     Step 5, p.4); no tolerance is stated.
%     Bounds          ub, lb of the IK search space and the handling of out-of-range values are
%                     not stated in the paper (checked pages 3-6, 17-19).
%
%   Benchmark interface:
%     Budget: f(mu_0) needs 1 FK call and an iteration at most k + 1 = 51 (50 candidates and at
%       most one C_mu). T = ikb_generations(opts, 1, 51, 100) = 39 iterations, so per waypoint
%       1 + 39 x 50 + n_prop FK calls, n_prop (0 to 39) the number of C_mu evaluations, at most
%       1990. A 40-member population would give ikb_generations(opts, 1, 41, 100) = 48. Eqs. 9
%       and 12 run on this horizon: a_t = 1 - t/T, N_ce(t) = 0.5 - (t/T)(0.5 - 1E-21). An
%       iteration starts only if ikb_budget_left(opts) covers 51. Outside the benchmark T = 100
%       (5001 + n_prop FK calls). No final FK check. Every option field is ignored (the budget is
%       read only through ikb_generations and ikb_budget_left).
%     Stop: T iterations. info.converged is always false: the paper has no convergence criterion.
%     Warm start: mu_0 = q0 (clamped to the limits), evaluated once; g_min = f(q0), so q0 is
%       returned unless a candidate or C_mu has a strictly lower fitness.
%     Anchor: F_s,j-1 = q0, the previous waypoint's solution passed by the harness, the same for
%       the whole call.
%     Robot and units: Eqs. 15-18 and Tables 5-6 (the paper's offset-wrist arm) are replaced by
%       robot.fk with n = robot.n joints; ub = q_max, lb = q_min, so sigma0 = (max(q_max) -
%       min(q_min))/2 (UR5: 2 pi). Positions are the benchmark's metres x 1000 (mm); joint angles
%       are radians.
%     Joint limits: every candidate and C_mu is clamped to [q_min, q_max] (not stated in the
%       paper). Eq. 14 scales the joint vector towards 0, so C_mu can leave a range that excludes 0
%       (Panda joint 4); the clamp applies there too.
%     Output: the centre mu with fitness g_min; info.fitness (Eq. 21), info.err_pos (m) and
%       info.err_rot (rotation angle, rad, reported only) from the pose stored at its evaluation.
%       solve_CVS('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - Iteration horizon T = 39 instead of Maxt = 100 (benchmark rule); a_t and N_ce(t) run on T.
%     - Initial centre q0 (warm start) instead of the midpoint of the search space (Eq. 1); g_min
%       starts at f(q0) (1 FK call).
%     - ub, lb = the robot's joint limits; candidates and C_mu clamped to them (not stated).
%     - F_s,j-1 = q0 as passed by the harness; for the first waypoint this is the robot's home
%       configuration, where the paper uses F_s0 = 0.
%     - Gauss-Mouse map as mod(1/x_i, 1) (see Parameters; the printed form leaves (0, 1)).
%     - Not stated in the paper: joint angles in radians in Eq. 21, CM_min = 0 and CM_max = 1,
%       separate U(0, 1) draws for the r of Eqs. 10, 13 and 14, g_min at t = 1, f_best at t = 1,
%       the iteration that uses r_t (see Radius timing).
%     - Eq. 13 applied as printed in every iteration, its last clause included; f_min in Eq. 13 is
%       the lowest candidate fitness, as in Eq. 14 (f(C_mu) not included).
%     - The radius of Step 4 is not computed in the last iteration (a_T = 0 lies outside a > 0 of
%       Eq. 6 and the radius would not be used).
%     - robot.fk instead of Eqs. 15-18 (the paper's arm).
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
        q_best = struct('params', cvs_params(), 'core', @cvs_core, 'map', @cvs_map, ...
                        'cm', @cvs_cm, 'fbratio', @cvs_fbratio, 'radius0', @cvs_radius0, ...
                        'radius', @cvs_radius, 'propose', @cvs_propose, 'fitness', @cvs_fitness);
        info = [];
        return
    end
    [q_best, info] = cvs_core(robot, T_d, q0, opts, cvs_params());
end

% ==================================================================== main loop (Steps 1-5)
function [q_best, info] = cvs_core(robot, T_d, q0, opts, P)
% Step 1 (centre q0, g_min = f(q0)), then per iteration: Step 2 (k candidates, Eqs. 2-4), Step 3
% (Eq. 14), Step 4 (Eq. 13, Eqs. 10-12 for the next radius). P holds the Table 6 values.
    t_start = tic;
    k  = P.k;
    lo = robot.q_min(:);
    hi = robot.q_max(:);
    n  = numel(lo);
    q0 = min(max(q0(:), lo), hi);
    T  = ikb_generations(opts, 1, k + 1, P.T);
    sigma0 = (max(hi) - min(lo)) / 2;                                 % Eq. 5
    pd = P.mm_per_m * T_d(1:3, 4);                                    % P_desired, mm (Eq. 19)
    F  = q0;                                                          % F_s,j-1 of Eq. 21

    mu = q0;  g = NaN;  Tmu = NaN(4);  fk = 0;  iters = 0;
    hist = zeros(1, 0);  rad = zeros(1, 0);  CMh = zeros(1, 0);  Cmh = zeros(1, 0);
    fbh = zeros(1, 0);  sgn = zeros(1, 0);  prop = false(1, 0);  acc = false(1, 0);

    if ikb_budget_left(opts) >= 1
        [g, Tmu] = cvs_fitness(robot, pd, mu, F, P);                  % g_min = f(mu_0)
        fk = 1;
        r  = cvs_radius0(sigma0, P);                                  % Eq. 7
        CM = P.x0;                                                    % CM(1) = x_0
        fb = NaN;  fbest = NaN;
        hist = NaN(1, T);  rad = NaN(1, T);  CMh = NaN(1, T);  Cmh = NaN(1, T);
        fbh = NaN(1, T);  sgn = zeros(1, T);  prop = false(1, T);  acc = false(1, T);
        for t = 1:T
            if ikb_budget_left(opts) < k + 1, break; end
            rad(t) = r;
            CMh(t) = CM;
            Cm = cvs_cm(CM, t, T, P);                                 % Eqs. 11-12
            Cmh(t) = Cm;
            C = min(max(mu + r * randn(n, k), lo), hi);               % Eqs. 2-4, clamp
            f = zeros(1, k);  Tc = zeros(4, 4, k);
            for i = 1:k
                [f(i), Tc(:, :, i)] = cvs_fitness(robot, pd, C(:, i), F, P);
            end
            fk = fk + k;
            [fmin, im] = min(f);
            if fmin < g                                               % Eq. 14
                g = fmin;  mu = C(:, im);  Tmu = Tc(:, :, im);
            elseif rand < 0.5
                Cmu = cvs_propose(Cm, mu, lo, hi);
                [fc, Tcm] = cvs_fitness(robot, pd, Cmu, F, P);
                fk = fk + 1;
                prop(t) = true;
                if fc < g
                    g = fc;  mu = Cmu;  Tmu = Tcm;  acc(t) = true;
                end
            end
            [fb, fbest] = cvs_fbratio(fb, fbest, fmin, t);            % Eq. 13
            fbh(t) = fb;
            if t < T                                                  % Step 4: radius of t + 1
                [r, sgn(t)] = cvs_radius(sigma0, fb, Cm, P.a0 - t / T, P);    % Eqs. 9-10
            end
            CM = cvs_map(CM);                                         % CM(t + 1)
            iters = t;
            hist(t) = g;
        end
        hist = hist(1:iters);  rad = rad(1:iters);  CMh = CMh(1:iters);  Cmh = Cmh(1:iters);
        fbh = fbh(1:iters);  sgn = sgn(1:iters);  prop = prop(1:iters);  acc = acc(1:iters);
    end

    ep = NaN;  er = NaN;
    if ~isnan(g)
        ep = norm(Tmu(1:3, 4) - T_d(1:3, 4));
        er = acos(min(1, max(-1, (trace(T_d(1:3, 1:3).' * Tmu(1:3, 1:3)) - 1) / 2)));
    end

    q_best = mu;
    info = struct();
    info.solver    = 'CVS';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = false;                                           % no criterion in the paper
    info.fitness   = g;                                               % Eq. 21, mm and rad
    info.err_pos   = ep;                                              % m
    info.err_rot   = er;                                              % rad, reporting only
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.k         = k;
    info.T         = T;
    info.sigma0    = sigma0;
    info.n_prop    = nnz(prop);                                       % C_mu evaluations (Eq. 14)
    info.history   = hist;                                            % g_min after each iteration
    info.radius    = rad;                                             % radius used in iteration t
    info.CM        = CMh;                                             % CM(t)
    info.Cm        = Cmh;                                             % C_m(t), Eq. 11
    info.fb        = fbh;                                             % fb_ratio after Eq. 13
    info.sign      = sgn;                                             % +1 / -1 of Eq. 10 (0: none)
    info.proposed  = prop;                                            % Eq. 14 C_mu evaluated
    info.accepted  = acc;                                             % C_mu accepted
    info.params    = P;
end

% ==================================================================== operators
function P = cvs_params()
% Table 6 (p.19): k = 50, Maxt = 100, N_cemax = 0.5, N_cemin = 1E-21, a_0 = 1, x = 0.1,
% err = 100; Table 1 (p.5): x_0 = 0.7, interval (0, 1); positions in mm.
    P = struct('k', 50, 'T', 100, 'Ncemax', 0.5, 'Ncemin', 1e-21, 'a0', 1, 'x', 0.1, ...
               'err', 100, 'x0', 0.7, 'CMmin', 0, 'CMmax', 1, 'mm_per_m', 1000);
end

function c = cvs_map(c)
% Gauss-Mouse map (Table 1 row 3, CVS3): x_(i+1) = mod(1/x_i, 1). The value for x_i = 0 is not
% printed; the sequence from x_0 = 0.7 never reaches it.
    if c == 0
        error('solve_CVS:map', 'Gauss-Mouse map at x = 0: value not stated in the paper.');
    end
    c = mod(1 / c, 1);
end

function Cm = cvs_cm(CM, t, T, P)
% Eq. 12: N_ce(t) = N_cemax - (t/T)(N_cemax - N_cemin); Eq. 11: C_m(t) = (CM(t) - CM_min) N_ce(t)
% / (CM_max - CM_min).
    Nce = P.Ncemax - (t / T) * (P.Ncemax - P.Ncemin);
    Cm  = (CM - P.CMmin) * Nce / (P.CMmax - P.CMmin);
end

function [fb, fbest] = cvs_fbratio(fb, fbest, fmin, t)
% Eq. 13, clauses in the printed order: t = 1 -> fb = 1 (f_best = f_min of iteration 1);
% f_min < f_best -> fb = f_min/f_best, f_best = f_min; |fb| > 2 -> 2r, |fb| < 1 -> r, r ~ U(0, 1).
    if t == 1
        fb = 1;
        fbest = fmin;
    elseif fmin < fbest
        fb = fmin / fbest;
        fbest = fmin;
    end
    if abs(fb) > 2
        fb = 2 * rand;
    elseif abs(fb) < 1
        fb = rand;
    end
end

function r = cvs_radius0(sigma0, P)
% Eq. 7: r_0 = sigma0 (1/x) gammaincinv(x, a_0).
    r = sigma0 * (1 / P.x) * gammaincinv(P.x, P.a0);
end

function [r, s] = cvs_radius(sigma0, fb, Cm, a, P)
% Eq. 10 with a = a_t of Eq. 9: (sigma0 + |fb| C_m) (1/x) gammaincinv(x, a) if r < 0.5, else
% (sigma0 - |fb| C_m) (1/x) gammaincinv(x, a); r ~ U(0, 1). s = +1 or -1, the sign used.
    if rand < 0.5
        s = 1;
    else
        s = -1;
    end
    r = (sigma0 + s * abs(fb) * Cm) * (1 / P.x) * gammaincinv(P.x, a);
end

function C = cvs_propose(Cm, best, lo, hi)
% Eq. 14: C_mu = C_m(t) itrBest, clamped to [lo, hi].
    C = min(max(Cm * best, lo), hi);
end

% ==================================================================== IK fitness (Eq. 21)
function [f, T] = cvs_fitness(robot, pd, q, F, P)
% Eq. 21: (err sum(|q - F|) + ||P_desired - P_candidate||)^2, positions in mm, joint angles in
% rad; one FK call.
    T  = robot.fk(q);
    pc = P.mm_per_m * T(1:3, 4);
    e  = sqrt((pd(1) - pc(1))^2 + (pd(2) - pc(2))^2 + (pd(3) - pc(3))^2);
    f  = (P.err * sum(abs(q - F)) + e)^2;
end
