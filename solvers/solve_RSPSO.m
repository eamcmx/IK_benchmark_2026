function [q_best, info] = solve_RSPSO(robot, T_d, q0, opts)
% SOLVE_RSPSO  RS-PSO: particle swarm optimisation under the Rough Sets Meta-Heuristic Schema (RSMS) of
%              Rokbani (2026), Secs. 2.2-2.8.
%
%   Source:
%     Nizar Rokbani (2026). Rough Sets Meta-Heuristic Schema for Inverse Kinematics and Path Planning
%     of Surgical Robotic Arms. Robotics, 15, 52 (article 52, 24 pp.; no issue number printed).
%     DOI: 10.3390/robotics15030052
%
%   Algorithm (Sec. 2, pp.3-6, Eqs. 3-12, Algorithm 1 p.6; RS-PSO Sec. 2.8, pp.9-10; fitness Sec. 3.2,
%   Eq. 17, pp.10-11):
%     N particles are initialised and evaluated (Alg. 1 c). Every iteration t = 0, 1, ...:
%       1. Information system I = (P_t, R), |R| = 3 (Def. 1, p.4; p.9): per particle the rank
%          attribute r1 (relative fitness rank), the diversity attribute r2 (local proximity to
%          neighbours) and the improvement attribute r3 (relative fitness change from t-1 to t).
%       2. Each attribute is discretised into L/M/H at population percentiles (Eq. 4); particles with
%          the same (r1, r2, r3) levels form one equivalence class of IND(R) (Eq. 5).
%       3. Target set T_t = the top alpha particles by fitness (Sec. 2.3). A class inside T_t is the
%          positive region (Eq. 6, ELITE), a class disjoint from T_t the negative region (Eq. 7,
%          POOR), a class that partly overlaps T_t the boundary region (Eq. 8); they partition the
%          swarm (Eq. 9).
%       4. Boundary ratio beta_t = |Bnd_R(T_t)|/N (Eq. 10).
%       5. Phase SEARCH / TRANSITION / EXPLOIT from beta_t, beta_lower, beta_upper with hysteresis
%          (Eq. 11, Sec. 2.4.2).
%       6. Each particle takes (w, c1, c2) = M_Alg(Phase_t, Class(X_i)) (Eq. 12) and makes the
%          inertia-weight PSO move (Sec. 2.8: w inertia, c1 cognitive, c2 social):
%            V <- w*V + c1*r1.*(Pbest_i - X) + c2*r2.*(Gbest - X),  X <- X + V.
%          All particles move with the Pbest/Gbest of P_t (step 6 forms P_(t+1)); each is then
%          evaluated once (Lemma 7, p.8) and Pbest, Gbest are updated.
%     The best solution found over all iterations is returned (Alg. 1 d).
%     Fitness (Eq. 17): J = wp*||P_t - P(q)||^2 + wo*d(R_target, R(q)).
%
%   Reading of the phase control (Sec. 2.4.2, p.5; Lemma 5, p.8):
%     Thresholds. p.5: "In a short preliminary run, a typical operating range [beta_min, beta_max]
%     is estimated, and beta_upper and beta_lower are set as problem-agnostic quantiles of this range
%     (e.g., upper and lower quartiles), ensuring beta_upper > beta_lower"; Lemma 5 then uses "fixed
%     thresholds". The values said to be reported (p.5-6) appear nowhere (pp.11-22 checked). Here the
%     first T_pre iterations of every run are the preliminary run: the phase is SEARCH and beta_t is
%     recorded. Then beta_min, beta_max = min and max of the recorded beta_t and
%       beta_lower = beta_min + (beta_max - beta_min)/4,  beta_upper = beta_min + 3*(beta_max - beta_min)/4
%     (the lower and upper quartile points of the range), fixed for the rest of the run, so
%     beta_upper > beta_lower whenever beta_max > beta_min. While all recorded beta_t are equal the
%     range has no width and the preliminary run continues.
%     Hysteresis. Eq. 11 (SEARCH if beta_t >= beta_upper, EXPLOIT if beta_t <= beta_lower,
%     TRANSITION in between) sets the phase, except that EXPLOIT is kept while beta_t is inside the
%     band: a return from EXPLOIT needs beta_t to cross beta_lower and reach beta_upper (p.5: "a
%     phase change requires beta_t to cross one threshold and move sufficiently beyond the other
%     before the reverse transition is allowed, in analogy to a Schmitt trigger"; Lemma 5 combines
%     the comparisons "with the previous phase label"). The forward progression SEARCH -> TRANSITION
%     -> EXPLOIT (p.5) and TRANSITION -> SEARCH follow Eq. 11 directly: a two-threshold band cannot
%     also hold SEARCH inside it without removing TRANSITION.
%
%   Parameters:
%     N = 40          particles. Not stated (pp.1-24 checked; p.14 "(n = 30) independent trials" is
%                     a trial count); the benchmark's reference population size.
%     t_max = 300     not stated in the text; Figs. 1(b),(c) (p.15) and 5 (p.22) plot 300 iterations
%                     (Table 15, p.22: mean 177.10 iterations to converge). Outside the benchmark only.
%     alpha = 1/2     |T_t|/N. "the top alpha agents" (p.4), "the top fraction rho" (Lemma 4, p.7);
%                     no value stated (pp.4-8, 11-22 checked); midpoint of the fraction's range
%                     (0, 1). A fraction equal to a level of Eq. 4 would make T_t a union of r1 bins:
%                     every class would lie wholly inside or outside T_t and beta_t = 0 always.
%     Level_bnd, Level_pos (Eq. 4) = the 1/3 and 2/3 population percentiles (three equal-mass bins),
%                     nearest rank: Level_p = v_(ceil(p*N)), the ceil(p*N)-th smallest value; values not
%                     stated (pp.4, 7 checked). L if v <= Level_bnd, M if Level_bnd < v <= Level_pos,
%                     H if v > Level_pos.
%     r1 = (1 + number of particles with lower J)/N; equal J share a rank.
%     r2 = Euclidean distance in joint space (rad) to the nearest other particle. No formula is given
%                     ("local proximity to neighbors", p.4); the nearest neighbour is the direct
%                     reading. It costs O(N*n) per particle where Lemma 2 (p.7) counts O(n).
%     r3 = (J_i(t-1) - J_i(t))/J_i(t-1) of the particle's current position (p.4: "relative fitness
%                     change from t - 1 to t"); 0 at t = 0, which has no t - 1, and where J_i(t-1) = 0.
%     T_pre = 10      iterations of the preliminary run (length not stated, p.5): ten beta_t samples,
%                     about one fifth of the 49 iterations the budget allows.
%     Quartile points 1/4 and 3/4 of [beta_min, beta_max] (p.5, "e.g., upper and lower quartiles").
%     Initial phase SEARCH: not stated; the first phase of the intended progression (p.5).
%     M_Alg (Eq. 12, p.6): the 3 x 3 lookup (phase x class) holding (w, c1, c2) for PSO is stated, its
%                     values are not: the paper prints no PSO coefficient, for RS-PSO or for its
%                     classical PSO (pp.6, 9-10, 13-22 checked). All 27 values are benchmark choices:
%                                    Pos (ELITE)      Bnd (BOUNDARY)   Neg (POOR)
%                       SEARCH       0.90, 1.0, 2.5   0.90, 2.0, 2.0   0.90, 2.5, 1.0
%                       TRANSITION   0.72, 1.5, 2.0   0.72, 1.5, 1.5   0.72, 2.0, 1.5
%                       EXPLOIT      0.40, 2.5, 1.0   0.40, 2.0, 2.0   0.40, 1.0, 2.5
%                     (entries w, c1, c2). w decreases with the phase and is the same for every class;
%                     0.90 and 0.40 are the end values of the inertia range of the benchmark's PSO
%                     (solve_PSO; Abdor-Sierra et al. 2022, p.8), 0.72 lies between them. Boundary
%                     particles get c1 = c2. In SEARCH an elite particle leans on the swarm (c2) and a
%                     poor one on its own memory (c1); in EXPLOIT the reverse, so poor particles are
%                     pulled toward Gbest and elite ones refine their Pbest; TRANSITION lies between.
%     r1, r2 of the move: U(0, 1), one draw per joint (the velocity equation is not printed,
%                     pp.3, 9-10). V(0) = 0 (not stated).
%     wp = wo = 1     not stated (p.11: "weighting coefficients of position and orientation errors used
%                     to balance the objectives"); equal weights.
%     d(R_target, R(q)) is "the orientation error" (p.11), not defined. p.10 says R "denotes a chosen
%                     orientation parametrization (e.g., Z-Y-X Euler or roll-pitch-yaw angles)", Eq. 15
%                     writes the pose as [x y z phi theta psi], Table 2 (p.12) gives the targets as RPY
%                     angles, and Eq. 24 (p.14) measures the orientation error as the Euclidean
%                     distance between orientation vectors. Reading used: d = Euclidean norm of the
%                     difference of the roll-pitch-yaw vectors [phi; theta; psi], R = Rz(psi) *
%                     Ry(theta) * Rx(phi), theta in [-pi/2, pi/2], each component difference wrapped to
%                     [-pi, pi), in rad; at theta = +/-pi/2 (gimbal lock) psi = 0. The geodesic error
%                     named on p.14 belongs to the validation statistics; it is reported in
%                     info.err_rot only.
%     eps_success = 1e-4 m: success threshold of the position error (Eq. 23, p.14; Table 15, p.22);
%                     info.converged only.
%
%   Benchmark interface:
%     Budget: initialisation and every iteration need N FK calls (one evaluation per particle per
%       iteration, Lemma 7); an iteration starts only if ikb_budget_left(opts) covers it.
%       T = ikb_generations(opts, 40, 40, 300) = 49 iterations, 2000 FK calls per waypoint (outside
%       the benchmark T = t_max = 300). No schedule depends on T. The attributes, regions and phase
%       need no FK call. No final FK check. opts.max_iter and the option fields pop_size, mu,
%       wp, wo, beta_lo, beta_hi, K_hyst, M_Alg and seed are ignored.
%     Stop: Algorithm 1 runs t = 0 .. t_max - 1 with no error-based stop (p.6), so the run uses its
%       whole horizon. info.converged = position error of the result < eps_success.
%     Warm start: q0 (clamped to the limits) is particle 1 of the initial swarm, evaluated
%       unperturbed; Gbest is the best evaluated position, so q0 enters the first selection. The
%       other particles are uniform in [q_min, q_max] (Alg. 1 c: "P_0 subset of X", distribution not
%       stated).
%     Units: Table 1 (p.11) gives the robot's lengths in mm (the targets, Table 2, and the errors,
%       Tables 5-15, are in m). The position term is computed in mm:
%       J = (1000*||P_t - P(q)||[m])^2 + d[rad].
%     Joint limits: a constraint of Eq. 17 (p.11), handling not stated; every new position is clamped
%       to [q_min, q_max]; the velocity is not changed by the clamp.
%     Output: Gbest, the lowest J evaluated (Alg. 1 d). info.beta, info.phase (1 SEARCH,
%       2 TRANSITION, 3 EXPLOIT), info.regions (|Pos|; |Bnd|; |Neg|) and info.n_classes per
%       iteration; info.beta_range = [beta_min, beta_max], info.thresholds = [beta_lower,
%       beta_upper], info.T_pre = iterations of the preliminary run. solve_RSPSO('ops') returns
%       handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - The preliminary run is the first T_pre = 10 iterations of each run (per waypoint), not a
%       separate run; its length is not stated.
%     - Thresholds = quartile points of the range [beta_min, beta_max] ("quantiles of this range");
%       the empirical quartiles of the recorded beta_t are the other reading.
%     - Hysteresis band applied as described above (its three-phase form is not specified).
%     - N, alpha, Level_bnd and Level_pos, the formulas of r1, r2, r3, T_pre, the initial phase, the
%       M_Alg values, wp, wo and d are not stated; chosen as under Parameters.
%     - Position term in mm (Table 1 units).
%     - Iteration count = budget horizon T = 49 instead of t_max = 300 (benchmark rule).
%     - Only the single-pose fitness of Eq. 17 is used: the multi-objective path fitness with the CJV
%       smoothness term (p.12; Eq. 25, p.14) and the joint-velocity bounds (p.11) are named but their
%       form and weights are not given.
%     - Initial population uniform in the joint limits; V(0) = 0; r1, r2 of the move drawn per joint.
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
        q_best = struct('params', rs_params(), 'fitness', @rs_fitness, 'rpy', @rs_rpy, ...
                        'orient', @rs_orient, 'attributes', @rs_attributes, ...
                        'discretise', @rs_discretise, 'classes', @rs_classes, 'target', @rs_target, ...
                        'regions', @rs_regions, 'rough', @rs_rough, 'thresholds', @rs_thresholds, ...
                        'phase', @rs_phase, 'malg', @rs_malg, 'theta', @rs_theta, 'move', @rs_move);
        info = [];
        return
    end

    t_start = tic;
    P = rs_params();

    n     = robot.n;
    N     = P.N;
    q_min = robot.q_min(:);
    q_max = robot.q_max(:);
    q0    = min(max(q0(:), q_min), q_max);
    T     = ikb_generations(opts, N, N, P.t_max);

    B = struct('f', NaN, 'q', q0, 'ep', NaN, 'd', NaN, 'er', NaN);
    history = zeros(1, 0);  beta = zeros(1, 0);  phases = zeros(1, 0);
    regions = zeros(3, 0);  ncls = zeros(1, 0);
    iters = 0;  fk = 0;  n_pre = 0;  thr = [NaN, NaN];  brange = [NaN, NaN];

    if ikb_budget_left(opts) >= N
        % Initial swarm (Alg. 1 c), q0 is particle 1
        X = q_min + rand(n, N) .* (q_max - q_min);  X(:, 1) = q0;
        V = zeros(n, N);
        F = zeros(1, N);
        for i = 1:N
            [F(i), ep, d, er] = rs_fitness(robot, T_d, X(:, i), P);  fk = fk + 1;
            B = keep_best(B, F(i), X(:, i), ep, d, er);
        end
        F_prev = F;                                      % r3 = 0 at t = 0
        Pb = X;  Fp = F;
        [~, ig] = min(Fp);  G = Pb(:, ig);

        phase = 1;                                       % SEARCH
        calibrated = false;  pre = zeros(1, 0);  lo_c = NaN;  hi_c = NaN;
        history = zeros(1, T);  beta = zeros(1, T);  phases = zeros(1, T);
        regions = zeros(3, T);  ncls = zeros(1, T);
        t = 0;
        while t < T
            if ikb_budget_left(opts) < N, break; end
            [reg, nb, S] = rs_rough(X, F, F_prev, P);    % steps 1-4, Eqs. 4-10 (beta_t = nb/N)
            if calibrated
                phase = rs_phase(nb, phase, lo_c, hi_c); % step 5, Eq. 11 with the band
            else                                         % preliminary run (Sec. 2.4.2)
                pre(end + 1) = nb;                       %#ok<AGROW>
                if numel(pre) >= P.T_pre && max(pre) > min(pre)
                    [lo_c, hi_c] = rs_thresholds(pre, P.quart);
                    calibrated = true;
                end
            end
            th = rs_theta(P.M, phase, reg);              % Eq. 12, columns (w; c1; c2)
            for i = 1:N
                [X(:, i), V(:, i)] = rs_move(X(:, i), V(:, i), Pb(:, i), G, th(:, i), q_min, q_max);
            end
            F_prev = F;
            for i = 1:N                                  % Lemma 7: one evaluation per particle
                [F(i), ep, d, er] = rs_fitness(robot, T_d, X(:, i), P);  fk = fk + 1;
                B = keep_best(B, F(i), X(:, i), ep, d, er);
            end
            upd = F < Fp;
            Pb(:, upd) = X(:, upd);  Fp(upd) = F(upd);
            [~, ig] = min(Fp);  G = Pb(:, ig);
            t = t + 1;  iters = t;
            history(t) = B.f;  beta(t) = nb / N;  phases(t) = phase;
            regions(:, t) = S.counts;  ncls(t) = S.n_classes;
        end
        history = history(1:iters);  beta = beta(1:iters);  phases = phases(1:iters);
        regions = regions(:, 1:iters);  ncls = ncls(1:iters);
        n_pre = numel(pre);
        if calibrated
            thr = [lo_c, hi_c] / N;  brange = [min(pre), max(pre)] / N;
        end
    end

    q_best = B.q;
    info = struct();
    info.solver     = 'RS-PSO';
    info.iters      = iters;
    info.time       = toc(t_start);
    info.converged  = B.ep < P.eps_success;                 % Eq. 23, p.14
    info.fitness    = B.f;                                  % Eq. 17 (mm^2 + rad)
    info.err_pos    = B.ep;                                 % m
    info.d          = B.d;                                  % RPY reading of d, rad
    info.err_rot    = B.er;                                 % geodesic angle, rad (reporting only)
    info.fk_calls   = fk;
    info.jac_calls  = 0;
    info.N          = N;
    info.T          = T;
    info.T_pre      = n_pre;
    info.history    = history;
    info.beta       = beta;
    info.phase      = phases;
    info.regions    = regions;
    info.n_classes  = ncls;
    info.beta_range = brange;
    info.thresholds = thr;
    info.params     = P;
end

% ------------------------------------------------------------------ parameters
function P = rs_params()
    P = struct('N', 40, 't_max', 300, 'alpha', 0.5, 'levels', [1/3, 2/3], 'T_pre', 10, ...
               'quart', [0.25, 0.75], 'wp', 1, 'wo', 1, 'len_scale', 1000, 'eps_success', 1e-4, ...
               'M', rs_malg());
end

function M = rs_malg()
% Eq. 12: M(phase, class, :) = [w, c1, c2]; phase 1 SEARCH, 2 TRANSITION, 3 EXPLOIT; class 1 Pos,
% 2 Bnd, 3 Neg. Values are benchmark choices (no values in the paper).
    M = zeros(3, 3, 3);
    M(:, :, 1) = [0.90 0.90 0.90; 0.72 0.72 0.72; 0.40 0.40 0.40];      % w
    M(:, :, 2) = [1.0  2.0  2.5 ; 1.5  1.5  2.0 ; 2.5  2.0  1.0 ];      % c1
    M(:, :, 3) = [2.5  2.0  1.0 ; 2.0  1.5  1.5 ; 1.0  2.0  2.5 ];      % c2
end

% ------------------------------------------------------------------ rough-set schema
function [reg, nb, S] = rs_rough(X, F, F_prev, P)
% Alg. 1 steps 1-4: attributes (Def. 1), Phi (Eq. 4), IND(R) (Eq. 5), T_t (Sec. 2.3), regions
% (Eqs. 6-8). reg: 1 Pos, 2 Bnd, 3 Neg per particle; nb = |Bnd_R(T_t)|.
    [r1, r2, r3] = rs_attributes(X, F, F_prev);
    C   = [rs_discretise(r1, P.levels); rs_discretise(r2, P.levels); rs_discretise(r3, P.levels)];
    id  = rs_classes(C);
    inT = rs_target(F, P.alpha);
    reg = rs_regions(id, inT);
    nb  = nnz(reg == 2);
    S   = struct('n_classes', max(id), 'counts', [nnz(reg == 1); nb; nnz(reg == 3)]);
end

function [r1, r2, r3] = rs_attributes(X, F, F_prev)
% Def. 1 (p.4): r1 relative fitness rank, r2 distance to the nearest other particle (joint space),
% r3 relative fitness change from t-1 to t.
    N  = numel(F);
    r1 = (1 + sum(F(:) < F(:).', 1)) / N;
    r2 = zeros(1, N);
    for i = 1:N
        dd = sqrt(sum((X - X(:, i)).^2, 1));  dd(i) = Inf;
        r2(i) = min(dd);
    end
    r3 = zeros(1, N);
    k  = F_prev > 0;
    r3(k) = (F_prev(k) - F(k)) ./ F_prev(k);
end

function c = rs_discretise(v, levels)
% Eq. 4: 1 (L) if v <= Level_bnd, 2 (M) if Level_bnd < v <= Level_pos, 3 (H) if v > Level_pos, with
% Level_p the nearest-rank population percentile v_(ceil(p*N)).
    s  = sort(v);  N = numel(v);
    Lb = s(ceil(levels(1) * N));
    Lp = s(ceil(levels(2) * N));
    c  = 1 + (v > Lb) + (v > Lp);
end

function id = rs_classes(C)
% Eq. 5: particles with identical discretised attribute vectors (columns of C) share a class.
    [~, ~, id] = unique(C.', 'rows');
    id = id(:).';
end

function inT = rs_target(F, alpha)
% Sec. 2.3: T_t = the round(alpha*N) particles of lowest fitness (ties by index).
    N = numel(F);
    [~, o] = sort(F);
    inT = false(1, N);
    inT(o(1:round(alpha * N))) = true;
end

function reg = rs_regions(id, inT)
% Eqs. 6-8: 1 = Pos ([X]_R subset of T_t), 3 = Neg ([X]_R disjoint from T_t), 2 = Bnd (partial).
    reg = zeros(1, numel(id));
    for k = unique(id)
        m = id == k;
        if all(inT(m))
            reg(m) = 1;
        elseif ~any(inT(m))
            reg(m) = 3;
        else
            reg(m) = 2;
        end
    end
end

% ------------------------------------------------------------------ phase control
function [lo, hi] = rs_thresholds(b, quart)
% Sec. 2.4.2: quartile points of the range [min(b), max(b)] of the preliminary-run samples.
    mn = min(b);  mx = max(b);
    lo = mn + quart(1) * (mx - mn);
    hi = mn + quart(2) * (mx - mn);
end

function ph = rs_phase(b, prev, lo, hi)
% Eq. 11 (1 SEARCH if b >= hi, 3 EXPLOIT if b <= lo, 2 TRANSITION between) with the hysteresis band of
% Sec. 2.4.2: inside the band EXPLOIT is kept, so leaving EXPLOIT needs b >= hi.
    if b >= hi
        ph = 1;
    elseif b <= lo
        ph = 3;
    elseif prev == 3
        ph = 3;
    else
        ph = 2;
    end
end

% ------------------------------------------------------------------ PSO move
function th = rs_theta(M, phase, reg)
% Eq. 12: theta_t(X_i) = M_Alg(Phase_t, Class(X_i)); columns (w; c1; c2).
    th = reshape(M(phase, reg, :), numel(reg), 3).';
end

function [x, v] = rs_move(x, v, pb, g, th, lo, hi)
% Sec. 2.8 inertia PSO with the particle's (w, c1, c2); r1, r2 ~ U(0, 1) per joint; position clamped.
    r1 = rand(numel(x), 1);  r2 = rand(numel(x), 1);
    v = th(1) * v + th(2) * r1 .* (pb - x) + th(3) * r2 .* (g - x);
    x = min(max(x + v, lo), hi);
end

% ------------------------------------------------------------------ fitness
function [J, ep, d, er] = rs_fitness(robot, T_d, q, P)
% Eq. 17: J = wp*||P_t - P(q)||^2 + wo*d(R_target, R(q)), position term in mm (Table 1 units), d the
% RPY reading (rad). ep = position error (m), er = geodesic angle (rad, reporting only).
    T  = robot.fk(q);
    ep = norm(T_d(1:3, 4) - T(1:3, 4));
    d  = rs_orient(T_d(1:3, 1:3), T(1:3, 1:3));
    J  = P.wp * (P.len_scale * ep)^2 + P.wo * d;
    er = acos(min(1, max(-1, (trace(T_d(1:3, 1:3).' * T(1:3, 1:3)) - 1) / 2)));
end

function d = rs_orient(Ra, Rb)
% Euclidean distance of the roll-pitch-yaw vectors, component differences wrapped to [-pi, pi).
    e = rs_rpy(Ra) - rs_rpy(Rb);
    e = mod(e + pi, 2 * pi) - pi;
    d = norm(e);
end

function a = rs_rpy(R)
% [phi; theta; psi] with R = Rz(psi)*Ry(theta)*Rx(phi), theta in [-pi/2, pi/2]; psi = 0 at
% theta = +/-pi/2.
    c = hypot(R(1, 1), R(2, 1));
    th = atan2(-R(3, 1), c);
    if c > 1e-12
        ph = atan2(R(3, 2), R(3, 3));
        ps = atan2(R(2, 1), R(1, 1));
    else
        ps = 0;
        ph = atan2(sign(-R(3, 1)) * R(1, 2), R(2, 2));
    end
    a = [ph; th; ps];
end

function B = keep_best(B, f, q, ep, d, er)
    if isnan(B.f) || f < B.f
        B.f = f;  B.q = q;  B.ep = ep;  B.d = d;  B.er = er;
    end
end
