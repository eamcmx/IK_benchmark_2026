function [q_best, info] = solve_MOA(robot, T_d, q0, opts)
% SOLVE_MOA  Mayfly Optimization Algorithm IK solver of El Mrabet et al. (2025), Sec. III-B.
%
%   Source:
%     Aziz El Mrabet, Hicham Hihi, Mohammed Khalil Laghraib, Mbarek Chahboun, Mohcine Abouyaakoub,
%     Ali Ait Ali, Aymane Amalaoui (2025). Inverse Kinematics Optimization Using ACO, MOA, SPOA, and
%     ALO: A Benchmark Study on Industrial Robot Arms. Journal of Robotics and Control (JRC), 6(4),
%     1729-1745. DOI: 10.18196/jrc.v6i4.26809
%   Fitness weights only:
%     Aziz El Mrabet, Hicham Hihi, Mohammed Khalil Laghraib, Mbarek Chahboun, Aymane Amalaoui (2025).
%     Two-Stage Neural Network versus Ant Colony, Mayfly, Stochastic Paint, and Ant Lion Optimizers
%     for Robot Arm Inverse Kinematics. JRC, 6(6), 2715-2728. DOI: 10.18196/jrc.v6i6.27795
%
%   Algorithm (pp. 5-6, Fig. 6, Eqs. 6-10; fitness p. 8, Eqs. 19-23):
%     M male and M female mayflies are initialised at random and evaluated; gbest is the best male
%     (Fig. 6 l.2-3; p.5). Each iteration (Fig. 6 l.4-25):
%       males:   v <- v + a1*exp(-beta*r_p)*(pbest - x) + a2*exp(-beta*r_g^2)*(gbest - x)  (Eq. 7),
%                x <- x + v (Eqs. 6, 8);
%       females: female i follows male i:
%                v <- g*v + alpha_f*exp(-beta*r_mf^2)*(x_i - y_i)  if f(y_i) > f(x_i),
%                v <- g*v + fl*r1                                  otherwise             (Eq. 9),
%                y <- y + v;
%       males and females are ranked; the k-th best male mates with the k-th best female:
%                off1 = a3*male + (1-a3)*female, off2 = (1-a3)*male + a3*female           (Eq. 10);
%       offspring are evaluated and assigned to the male or female group at random; in each group
%       the worst member is replaced by the best new one; pbest and gbest are updated (l.22-24).
%     Fitness: E = alpha_E*Ep + beta_E*Eo (Eq. 19), Ep = ||P_d - P_c|| (Eq. 21),
%     Eo = 2*acos(0.5*sqrt(n_x + o_y + a_z + 1)) with R_E = R_d*R_c' (Eqs. 22-23).
%
%   Parameters:
%     M = 20 males and 20 females. Table V (p.6): population size 30-50, "Number of male + female
%                    mayflies"; no value reported for the IK runs; midpoint 40. Fig. 6 runs both the
%                    male loop (l.6) and the female loop (l.12) over i = 1..M, so the groups are equal.
%     a1 = a2 = 1.5  Table V: 1.0-2.0; midpoint.
%     beta = 1.5     visibility coefficient. Table V: 1.0-2.0; midpoint.
%     g, fl: 0.9 -> 0.1. Table V "0.9-0.1"; p.6: weights "that decrease from their maximum to lowest
%                    values". Shape not stated; linear over T: g = fl = 0.9 - 0.8*(t-1)/(T-1).
%     alpha_f = 1.5  p.6: "alpha_f and beta are fixed parameters"; no value given (pp. 5-6 checked).
%                    Benchmark uses 1.5, the midpoint of Table V's range for the attraction
%                    coefficients a1, a2, the role alpha_f plays towards the paired male.
%     a3 ~ N(0.5, 0.1^2) restricted to [0.2, 0.8]. p.6: "a random number adhering to a Gaussian
%                    distribution"; Table V: crossover factor 0.2-0.8. The mean is the range
%                    midpoint, the range is the support (draws outside it are redrawn), and the
%                    standard deviation is not stated: 0.1 puts the range at +/-3 sd.
%     r1 in {-1, 0, 1}, uniform, drawn per joint. p.6: "A random integer sampled from the uniform
%                    distribution from -1 to 1"; Fig. 6 l.13-14 adjusts the female velocity per
%                    dimension j.
%     r_p, r_g, r_mf: Euclidean distances in joint space between x and pbest, x and gbest, male i
%                    and female i; the paper does not define them.
%     v(0) = 0. Initial velocities not stated.
%     T = 400        iterations. Table V: 300-500; midpoint. Inside the benchmark T is the number of
%                    iterations the FK budget allows (ikb_generations: (2000 - 2M)/(4M) = 24 for
%                    M = 20), so g and fl fall from 0.9 to 0.1 within the budget.
%     alpha_E = beta_E = 1 (Eq. 19). Not valued in this paper (p.8); the companion paper states
%                    E = Ep + EO (JRC 6(6), p.5, Eq. 9).
%
%   Benchmark interface:
%     Budget: initialisation needs 2M FK calls and each iteration 4M (M males, M females, 2M
%       offspring); a step starts only if ikb_budget_left(opts) covers it. No final FK check.
%       opts.max_iter is ignored.
%     Warm start: q0 (clamped to the limits) is male 1, so it is pbest of male 1 and gbest when it
%       is the best male.
%     Units: benchmark lengths are metres; the paper's robots are in mm (Tables I-III, pp. 3-4), so
%       Ep is computed in mm: E = 1000*||P_d - P_c||[m] + Eo[rad].
%     Joint limits: Eq. 20 bounds theta; the paper gives no handling rule, so every new position
%       (male, female, offspring) is clamped to [q_min, q_max]. Velocities are not limited.
%     Output: the lowest-E mayfly ever evaluated (gbest, unless a female or a female-assigned
%       offspring was better). The paper's only stopping rule is t < T (Fig. 6 l.4), so
%       info.converged is always false. solve_MOA('ops') returns handles to the operators for unit
%       tests.
%
%   Deviations from the paper:
%     - Eqs. 6 and 8 print x(t+1) = x(t) + v(t); Fig. 6 updates the velocity (l.8, l.14) before the
%       position (l.9, l.15), and the position step uses the velocity just computed.
%     - Fig. 6 l.3 takes gbest from "all solutions"; the definition under Eq. 7 (p.5) makes gbest
%       "the position of the best male mayfly", which is used throughout.
%     - Fig. 6 l.17 "Update pbest" in the female loop: no equation uses a female pbest; male pbests
%       are updated after the male evaluation.
%     - Fig. 6 l.24 "Replace the worst solution with the best new one" is applied as printed, once per
%       group (one replacement in the male group, one in the female group), without an
%       improvement test. A new male's pbest is its own position; new members start at v = 0.
%     - "Local search mechanisms" (p.5) are not defined; nothing is implemented for them.
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
        q_best = struct('fitness', @moa_fitness, 'male_velocity', @moa_male_velocity, ...
                        'female_velocity', @moa_female_velocity, 'crossover', @moa_crossover, ...
                        'alpha3', @moa_alpha3, 'r1', @moa_r1, 'weight', @moa_weight, ...
                        'split', @moa_split, 'replace', @moa_replace, 'params', moa_params());
        info = [];
        return
    end

    t_start = tic;
    P = moa_params();
    M = P.M;
    P.T = ikb_generations(opts, 2 * M, 4 * M, P.T);

    n     = robot.n;
    q_min = robot.q_min(:);
    q_max = robot.q_max(:);
    q0    = min(max(q0(:), q_min), q_max);

    q_best = q0;  best_E = NaN;  best_ep = NaN;  best_eo = NaN;
    history = zeros(1, 0);  iters = 0;  fk = 0;  mate_fx = [];  mate_fy = [];

    if ikb_budget_left(opts) >= 2 * M
        % Fig. 6 l.2-3: random males and females, q0 is male 1
        x = q_min + rand(n, M) .* (q_max - q_min);  x(:, 1) = q0;
        y = q_min + rand(n, M) .* (q_max - q_min);
        vx = zeros(n, M);  vy = zeros(n, M);
        fx = zeros(1, M);  fy = zeros(1, M);
        for i = 1:M
            [fx(i), ep, eo] = moa_fitness(robot, T_d, x(:, i));  fk = fk + 1;
            [best_E, q_best, best_ep, best_eo] = keep_best(fx(i), x(:, i), ep, eo, ...
                                                           best_E, q_best, best_ep, best_eo);
        end
        for i = 1:M
            [fy(i), ep, eo] = moa_fitness(robot, T_d, y(:, i));  fk = fk + 1;
            [best_E, q_best, best_ep, best_eo] = keep_best(fy(i), y(:, i), ep, eo, ...
                                                           best_E, q_best, best_ep, best_eo);
        end
        pb = x;  fpb = fx;                                   % male pbest
        [gb_E, ig] = min(fpb);  gb = pb(:, ig);              % gbest = best male

        for t = 1:P.T                                        % Fig. 6 l.4
            if ikb_budget_left(opts) < 4 * M, break; end
            g = moa_weight(t, P.T);  fl = g;

            % l.6-11: males (Eqs. 7-8)
            for i = 1:M
                vx(:, i) = moa_male_velocity(vx(:, i), x(:, i), pb(:, i), gb, P.a1, P.a2, P.beta);
                x(:, i)  = min(max(x(:, i) + vx(:, i), q_min), q_max);
            end
            for i = 1:M
                [fx(i), ep, eo] = moa_fitness(robot, T_d, x(:, i));  fk = fk + 1;
                [best_E, q_best, best_ep, best_eo] = keep_best(fx(i), x(:, i), ep, eo, ...
                                                               best_E, q_best, best_ep, best_eo);
                if fx(i) < fpb(i), pb(:, i) = x(:, i);  fpb(i) = fx(i); end
            end

            % l.12-18: females (Eq. 9), female i paired with male i
            for i = 1:M
                vy(:, i) = moa_female_velocity(vy(:, i), y(:, i), x(:, i), fy(i), fx(i), ...
                                               g, fl, P.alpha_f, P.beta);
                y(:, i)  = min(max(y(:, i) + vy(:, i), q_min), q_max);
            end
            for i = 1:M
                [fy(i), ep, eo] = moa_fitness(robot, T_d, y(:, i));  fk = fk + 1;
                [best_E, q_best, best_ep, best_eo] = keep_best(fy(i), y(:, i), ep, eo, ...
                                                               best_E, q_best, best_ep, best_eo);
            end

            % l.19-20: rank males and females
            [fx, o] = sort(fx);  x = x(:, o);  vx = vx(:, o);  pb = pb(:, o);  fpb = fpb(:, o);
            [fy, o] = sort(fy);  y = y(:, o);  vy = vy(:, o);
            mate_fx = fx;  mate_fy = fy;                     % parents' fitness, in pairing order

            % l.21-22: k-th best male with k-th best female (Eq. 10); evaluate offspring
            off = zeros(n, 2 * M);  foff = zeros(1, 2 * M);
            for k = 1:M
                [o1, o2] = moa_crossover(x(:, k), y(:, k), moa_alpha3());
                off(:, 2*k-1) = min(max(o1, q_min), q_max);
                off(:, 2*k)   = min(max(o2, q_min), q_max);
            end
            for k = 1:2 * M
                [foff(k), ep, eo] = moa_fitness(robot, T_d, off(:, k));  fk = fk + 1;
                [best_E, q_best, best_ep, best_eo] = keep_best(foff(k), off(:, k), ep, eo, ...
                                                               best_E, q_best, best_ep, best_eo);
            end

            % l.23-24: random split, worst member of each group <- best new one of that group
            to_male = moa_split(2 * M);
            [x, fx, jm] = moa_replace(x, fx, off(:, to_male), foff(to_male));
            if jm > 0
                vx(:, jm) = 0;  pb(:, jm) = x(:, jm);  fpb(jm) = fx(jm);
            end
            [y, fy, jf] = moa_replace(y, fy, off(:, ~to_male), foff(~to_male));
            if jf > 0, vy(:, jf) = 0; end

            % l.24: update gbest from the male pbests
            [m_E, im] = min(fpb);
            if m_E < gb_E, gb_E = m_E;  gb = pb(:, im); end

            iters = t;
            history(t) = best_E;
        end
    end

    info = struct();
    info.solver    = 'MOA';
    info.iters     = iters;
    info.time      = toc(t_start);
    info.converged = false;
    info.fitness   = best_E;
    info.err_pos   = best_ep;                                % m
    info.err_rot   = best_eo;                                % rad
    info.fk_calls  = fk;
    info.jac_calls = 0;
    info.history   = history;
    info.mate_fx   = mate_fx;                                % last mating: male fitness by pair
    info.mate_fy   = mate_fy;                                % last mating: female fitness by pair
    info.params    = P;
end

% ------------------------------------------------------------------ operators
function P = moa_params()
    P = struct('M', 20, 'a1', 1.5, 'a2', 1.5, 'beta', 1.5, 'alpha_f', 1.5, ...
               'g_max', 0.9, 'g_min', 0.1, 'a3_mean', 0.5, 'a3_sd', 0.1, 'a3_lo', 0.2, ...
               'a3_hi', 0.8, 'T', 400, 'w_pos', 1, 'w_rot', 1, 'mm_per_m', 1000);
end

function [E, ep, eo] = moa_fitness(robot, T_d, q)
% Eqs. 19-23 with alpha = beta = 1; Ep in mm, Eo in rad. ep is returned in metres.
    T  = robot.fk(q);
    ep = norm(T_d(1:3, 4) - T(1:3, 4));
    RE = T_d(1:3, 1:3) * T(1:3, 1:3)';                       % Eq. 22
    eo = 2 * acos(min(1, 0.5 * sqrt(max(0, RE(1,1) + RE(2,2) + RE(3,3) + 1))));   % Eq. 23
    E  = 1000 * ep + eo;                                     % Eq. 19
end

function v = moa_male_velocity(v, x, pbest, gbest, a1, a2, beta)
% Eq. 7 as printed: exp(-beta*r_p) on the cognitive term, exp(-beta*r_g^2) on the social term.
    rp = norm(pbest - x);
    rg = norm(gbest - x);
    v = v + a1 * exp(-beta * rp) * (pbest - x) + a2 * exp(-beta * rg^2) * (gbest - x);
end

function v = moa_female_velocity(v, y, x, fy, fx, g, fl, alpha_f, beta)
% Eq. 9: y = female, x = paired male, fy, fx their fitness values.
    if fy > fx
        v = g * v + alpha_f * exp(-beta * norm(x - y)^2) * (x - y);
    else
        v = g * v + fl * moa_r1(numel(v));
    end
end

function [off1, off2] = moa_crossover(male, female, a3)
% Eq. 10.
    off1 = a3 * male + (1 - a3) * female;
    off2 = (1 - a3) * male + a3 * female;
end

function a3 = moa_alpha3()
% Gaussian crossover factor (p.6), mean 0.5, sd 0.1, redrawn until inside [0.2, 0.8] (Table V).
    a3 = 0.5 + 0.1 * randn;
    while a3 < 0.2 || a3 > 0.8
        a3 = 0.5 + 0.1 * randn;
    end
end

function r = moa_r1(n)
% Random integer, uniform on {-1, 0, 1} (p.6), one per joint.
    r = randi([-1, 1], n, 1);
end

function w = moa_weight(t, T)
% g = fl, linear from 0.9 at t = 1 to 0.1 at t = T (Table V, p.6).
    w = 0.9 - 0.8 * (t - 1) / (T - 1);
end

function to_male = moa_split(K)
% Fig. 6 l.23: each offspring goes to the male or the female group at random.
    to_male = rand(1, K) < 0.5;
end

function [X, F, j] = moa_replace(X, F, new, fnew)
% Fig. 6 l.24: the worst member is replaced by the best new one (j = slot replaced, 0 if none).
    j = 0;
    if isempty(fnew), return; end
    [fb, b] = min(fnew);
    [~, j]  = max(F);
    X(:, j) = new(:, b);  F(j) = fb;
end

function [best_E, q_best, best_ep, best_eo] = keep_best(E, q, ep, eo, best_E, q_best, best_ep, best_eo)
    if isnan(best_E) || E < best_E
        best_E = E;  q_best = q;  best_ep = ep;  best_eo = eo;
    end
end
