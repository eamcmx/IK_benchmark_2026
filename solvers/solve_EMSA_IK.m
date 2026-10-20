function [q_best, info] = solve_EMSA_IK(robot, T_d, q0, opts)
% SOLVE_EMSA_IK  EMSA-IK: evolutionary multi-objective semi-analytical IK of Chen et al. (2025),
%                Algorithms 1-2 with the semi-analytical reduced chain of Eq. 3.
%
%   Source:
%     Zhaoyang Chen, Kang Min, Xinyang Fan, Baoxu Tu, Fenglei Ni and Hong Liu (2025). EMSA-IK: a
%     real-time evolutionary multi-objective semi-analytical inverse kinematics algorithm for
%     redundant manipulators. Industrial Robot: the international journal of robotics research and
%     application, 52(2), 249-257. DOI: 10.1108/IR-08-2024-0355
%   Page numbers are PDF pages 1-9 (PDF page k = printed page 248 + k).
%
%   Algorithm (Sec. 2.1, Eqs. 1-3, p.2; Secs. 3.1-3.5, Figs. 1-2, Eqs. 4-12, Algorithms 1-2, pp.3-5):
%     Parameterisation (Sec. 2.1, p.2): theta = [theta_p, theta_r]. The q = n - 6 parametric joints
%       theta_p are searched by NSGA-II in binary code; the reduced joints follow in closed form,
%       theta_r = FK^-1(x, theta_p) (Eq. 3). An individual (Eq. 4, p.3) holds theta_p (b bits per joint),
%       its n + 1 fitness values, Pareto rank, crowd distance and fitness sum.
%     Decoding (Eq. 5, p.3; Eq. 10, p.5): theta_p,j = Lb_j + sum_i(2^i*G_i)*(Ub_j - Lb_j)/(2^b - 1), code
%       [G_0, ..., G_(b-1)], [Lb, Ub] the search sub-space S_opt.
%     Feasibility (p.5): joint limits (Eq. 11) and accuracy ||P_target - P_t||_2 < gamma (Eq. 12), P_t by
%       forward kinematics. Infeasible individuals are extracted (discarded), never repaired.
%     Fitness (Sec. 3.4, Fig. 2, pp.3-4), per joint i = 1..n, theta_cur = current joint angles:
%       phi_c,i = 1 - exp(-(theta_i - theta_cur,i)^2/coeC^2)                                   (Eq. 6)
%       phi_l,i = 2/(1 + exp((theta_i - theta_min,i)/coeL)) + 2/(1 + exp(-(theta_i - theta_max,i)/coeL))
%                                                                                              (Eq. 7)
%       phi_d   = 1 - exp(-(sum_k sum_j phi_draw(k, j))^2/coeD^2), phi_draw from link-obstacle
%                 distances with thresholds d_in, d_out                                        (Eq. 8)
%       phi_m   = exp(-det(J*J')/coeM^2), J the geometric Jacobian (Eq. 9; sign, see Deviations)
%       Fitness_i = Ow_1*phi_c,i + Ow_2*phi_l,i + Ow_3*phi_d (i = 1..n), Fitness_(n+1) = phi_m;
%       fitness sum Fs = sum_i Fw_i*Fitness_i (Fig. 2). All n + 1 values are minimised (p.5).
%     ASSS (Algorithm 2, pp.4-5): sub-spaces [theta_cur - cnt*Sr, theta_cur + cnt*Sr] for cnt = 1, 2, ...;
%       each side of each parametric joint grows until it passes its limit (l.13-19), and the loop runs
%       while a side is still inside its limit (l.4). Each sub-space is scored by 5 rounds of N random
%       individuals (Eq. 10): SFitness = sum of the fitness sums of the feasible ones (l.5-12).
%       S_opt = the sub-space of minimum SFitness (l.22; Sec. 3.5, p.4); a sub-space without a feasible
%       individual is not eligible (see Deviations).
%     NSGA-II (Algorithm 1, p.4; Sec. 3.5, pp.4-5):
%       l.1   S_opt by Algorithm 2.
%       l.2   first-generation population: Eq. 10 individuals in S_opt, feasible ones kept until there
%             are N (p.5).
%       l.3   non-dominated sort and crowd distance (Deb et al. 2002) on the n + 1 fitness values.
%       l.5   binary tournament on rank, then crowd distance, selects N/2 parents ("half of the
%             individuals", p.5); multi-point crossover (probability pc) and bitwise mutation
%             (probability pm) give N/2 offspring.
%       l.6-7 feasible offspring are kept; new Eq. 10 individuals refill the offspring to N.
%       l.8-11 merge with the population, sort, crowd distance, keep N elitists: smaller Pareto rank,
%             ties by smaller fitness sum (p.5).
%       stop  the minimum fitness sum changes by less than Delta for Q generations (p.5).
%       l.13  theta_opt = the elitist with minimum Pareto rank and fitness sum.
%
%   Forms (p.6; Tables 1-2, pp.6-7): the paper publishes EMSA-IK (Algorithms 1-2) and its ablation
%     EMSA-IK (w/o ASSS), "EMSA-IK*" in Tables 1-2, "developed to conduct ablation experiments" (p.6).
%     The ablation is described only by what it leaves out. Algorithm 2 narrows "the whole search
%     space bounded by joint limits of each parametric joint" (Sec. 2.2, p.2), the "Raw Search Space"
%     of Fig. 1 (p.3), and p.6 gives the ablation a longer solution time "because of the large search
%     space". The ablation is therefore Algorithm 1 without l.1, with [Lb, Ub] of Eqs. 5 and 10 set to
%     [theta_min, theta_max] of the parametric joints; everything else is identical. Both forms keep
%     q0's theta_p as the first individual of the first-generation population (warm start).
%     Measured under the benchmark budget (Synthetic9/11/13 x 4 paths x waypoints 1-10 = 120
%     waypoints, warm start chained as in bench_run):
%       form                 pose gate  median error (m; rad)  FK per waypoint  generations
%       EMSA-IK (ASSS)       98/120     3.5e-16; 0             1994-2000        0
%       EMSA-IK (w/o ASSS)   91/120     3.0e-16; 0             1143-2000        0-15 (median 3.5)
%     Form used: EMSA-IK with ASSS, the best of the paper's two published forms under the budget with
%     warm start (more pose-gate successes; the benchmark's rule for this solver). The other form is
%     reachable for tests only, through solve_EMSA_IK('ops').run(robot, T_d, q0, opts, 'joint_range').
%
%   Eq. 3 on the benchmark robots:
%     theta_p = joints 1..m, m = n - 6: the proximal joints, as the paper's 7-DOF choice [theta_1] (p.6;
%     p.6 selects "the joints whose angles change caused the obvious change of the end-effector's
%     pose"). The paper takes the closed form of Eq. 3 from Xie et al. (2023) (p.2), names the Pieper
%     criterion as the condition for analytical IK of a non-redundant chain (p.2) and prints no closed
%     form for any arm. The closed form used here holds when links m+1..n have a = 0,
%     d_(m+1) = d_(m+3) = d_(m+5) = 0, d_(m+2) ~= 0, d_(m+4) ~= 0 and sin(alpha) ~= 0 for links m+1..m+5
%     (read from the DH table, no FK):
%       W = P - d_n*R*[0; sin(alpha_n); cos(alpha_n)] is the wrist centre, where the axes of joints
%         m+4, m+5, n meet; the axes of joints m+2 and m+3 meet at E = O_m + d_(m+2)*z_(m+1), with
%         |W - E| = |d_(m+4)|.
%       theta_(m+1): E on its circle about z_m at distance |d_(m+4)| from W (2 roots);
%       theta_(m+2), theta_(m+3): the direction of W - E (2 roots); theta_(m+4..n): the wrist rotation
%       R_(m+3)'*R (2 roots). Up to 8 branches. An arcsine or arccosine argument outside [-1, 1] is the
%       "imaginary number" of p.5: the angle is computed from the clipped argument and Eq. 12 rejects
%       the individual.
%     Robot        q  Applicable  Reason
%     UR5          0  no          6 joints, no redundant joint: the method is for redundant manipulators
%                                 (title and abstract, p.1; theta_p in R^q for q redundant DOF, p.2), and
%                                 non-redundant arms meeting the Pieper criterion have plain analytical
%                                 IK (p.2).
%     IRB120       0  no          as UR5.
%     Panda        1  no          with theta_p = [theta_1], no three consecutive axes of joints 2-7 meet
%                                 in a point or are parallel (a3 = 0.0825, a4 = -0.0825, a6 = 0.088 m):
%                                 the Pieper criterion fails, so Eq. 3 has no closed form.
%     Synthetic9   3  yes         joints 4-9: a = 0, d = [0 0.333 0 0.333 0 0.2] m, alpha = -/+90 deg.
%     Synthetic11  5  yes         joints 6-11, the same links.
%     Synthetic13  7  yes         joints 8-13, the same links.
%
%   Parameters (Sec. 4.2, p.6, unless stated):
%     N = 12, b = 16, Sr = 4 deg      the 7-DOF setting. The 9-DOF setting (N = 50, b = [11, 12, 10],
%                     Sr = [4 deg, 10 mm], Ow with four entries) belongs to a 7R + 2P arm with a prismatic
%                     parametric joint. The paper reports no iteration count (it stops by Delta and Q),
%                     so N x T cannot be formed; one ASSS sub-space costs 5N FK calls, which makes N = 12
%                     the setting nearest the 2000-FK budget (benchmark rule for several IK settings).
%     pc = 0.6, pm = 0.1   crossover and mutation probabilities; pm is applied per bit ("bitwise
%                     mutations", p.5).
%     Crossover points  "multi-point crossover" (p.5); the number of points is not stated in the paper
%                     (checked pages 4-6); two points, the smallest number that makes it multi-point.
%     Ow = [0.7, 0.2, 0.1]   continuity, joint limits, obstacle avoidance.
%     Fw              not stated for these robots: the paper prints 7 values for its 7-DOF arm and 9 for
%                     its 9-DOF arm (p.6), each one fewer than the n + 1 of its definition (p.4), and none
%                     for 10, 12 or 14 fitness values (checked pages 3-6). The benchmark uses Fw_i = 1, so
%                     the fitness sum is the plain sum of Fig. 2.
%     coeC = 2, coeL = 0.05, coeD = 2, coeM = 0.5; d_in = 0.01 m, d_out = 0.03 m.
%     Delta = 1e-6, Q = 10   termination (p.5, p.6).
%     ASSS rounds = 5 (Alg. 2 l.6).
%     gamma (Eq. 12)  "an extremely small number" (p.5); no value is stated (checked pages 5-6). The
%                     benchmark uses its pose gate, min(opts.tol_pos, opts.tol_rot) (1e-6 without
%                     them). The 2-norm of the 4x4 difference is the largest singular value.
%     Branch of Eq. 3  Eq. 3 is written as a single value (p.2); the choice among its branches is not
%                     stated (checked pages 2-5). Per joint the 2*pi equivalent nearest theta_cur is
%                     taken, inside the limits where one exists; the branch nearest theta_cur (Euclidean,
%                     reduced joints) among the real branches inside the limits is used.
%     Units           SI. The paper gives d_in, d_out and its pose errors in metres (p.6, Tables 1-2),
%                     so no millimetre conversion applies. The angle unit of Eqs. 6-7 is not
%                     stated (checked pages 4-6); radians are used.
%
%   Benchmark interface:
%     Applicability: on UR5, IRB120 and Panda (table above) the solver sets info.not_applicable = true and
%       returns q0 without any FK or Jacobian call.
%     Budget: every individual costs one robot.fk call (Eq. 12). Eq. 3 is evaluated from the DH table
%       (the m parametric link transforms and the closed-form angles) and calls no FK. robot.jac (Eq. 9,
%       feasible individuals only) is not budgeted and is counted in info.jac_calls. An ASSS sub-space
%       (5N = 60 FK) starts only if ikb_budget_left(opts) covers it; the ASSS loop then ends and the
%       first-generation population is formed one individual at a time while the budget lasts; a
%       generation starts only if the budget covers its minimum of N = 12 FK (N/2 offspring and at least
%       N/2 refills), and a generation that runs out of budget during its refill is dropped. The
%       algorithm runs as printed until the budget is spent, as for every solver: no budget is reserved
%       for the generations. With Sr = 4 deg the ASSS sweep needs 45 to 90 sub-spaces on joints with
%       +/-180 deg limits (2700 to 5400 FK). Inside the budget it scores 33 sub-spaces (1980 FK). The
%       sweep runs outward from theta_cur, so the scored ones are the nearest: theta_cur -/+ 4 deg up to
%       theta_cur -/+ 132 deg (33 x Sr), each side held where it passes its limit. The remaining 20 FK
%       go to the first-generation population (Alg. 1 l.2), and no generation fits: 0 generations on
%       every measured waypoint (Synthetic9/11/13, 4 paths x 10 waypoints). Outside the budget the full
%       algorithm costs 4,787 to 14,566 FK per waypoint (12 measured waypoints, 10 to 30 generations).
%       Outside the benchmark the run is unbounded except for a safety cap of 1e5 FK calls (the paper's
%       refill loops do not end when no feasible individual exists). No quantity varies with the
%       iteration count, so the budget horizon of the benchmark (ikb_generations) is not used.
%       opts.max_iter and opts.pop_size are ignored.
%     Warm start: theta_cur = q0 (the inputs theta_cur of Algorithms 1-2). The first individual of the
%       first-generation population is q0's theta_p, encoded to the nearest b-bit code of S_opt; it
%       enters the population if it is feasible (Eqs. 11-12), like every individual.
%     Joint limits: individuals outside [q_min, q_max] are infeasible (Eq. 11) and extracted (p.5); no
%       clamping. The last ASSS sub-space can pass a limit by less than Sr (Alg. 2 l.14-19).
%     Obstacles: the benchmark has none (n_o = 0), so the double sum of Eq. 8 is empty and phi_d = 0 for
%       every individual; the Ow_3 term is kept and contributes 0.
%     Output: the feasible individual with the smallest fitness sum among all individuals evaluated
%       (ASSS samples included). In any set it belongs to it has Pareto rank 1 (all Fw_i > 0), so it
%       is Algorithm 1's theta_opt whenever it is in the final population. If no feasible individual is
%       found, q0 is returned. info.converged = the returned joints satisfy Eqs. 11-12. info.fitness is
%       its fitness sum; info.fitness_alg1 that of Alg. 1 l.13 on the final population. Pose errors
%       come from the FK call of its Eq. 12 check (no extra FK). solve_EMSA_IK('ops') returns handles
%       to the operators for unit tests.
%
%   Deviations from the paper:
%     - Population migration (p.4: N*mr individuals of the last call's final population) is not used:
%       the benchmark passes only q0 between waypoints and mr is not stated (checked pages 4-6). q0's
%       theta_p stands in for the migrated individuals; the rest follow Eq. 10, as in the paper's own
%       first-sampling-time case (p.4).
%     - The 2000-FK budget stops the ASSS sweep after 33 of its 45 to 90 sub-spaces (up to theta_cur
%       -/+ 132 deg) and leaves 0 generations on every measured waypoint; the full algorithm needs 4,787
%       to 14,566 FK per waypoint (see Budget). No budget is reserved for the generations.
%     - Algorithm 2 l.5-10 and l.22 read literally give a sub-space with no feasible individual
%       SFitness = 0, the minimum, so it would be selected; the sum also falls as fewer individuals are
%       feasible. That contradicts the purpose of S_opt: the sub-space the evolutionary algorithm
%       searches (Sec. 3.5, p.4), whose first-generation population is created from feasible
%       individuals until it has N (p.5). Nearest well-defined reading: a sub-space with no feasible
%       individual is not eligible (SFitness = +Inf); otherwise the printed sum over the feasible
%       individuals is kept. If no scored sub-space has a feasible individual, every value is +Inf and
%       S_opt is the first scored sub-space (theta_cur -/+ Sr); the first-generation population is then
%       created and refilled there by Alg. 1 while budget lasts, and q0 is returned with
%       info.converged = false if no feasible individual is ever found.
%     - Eq. 5 and Eq. 10 print "+ Lb + theta_cur" and "- Lb + theta_cur", while Alg. 2 l.2-3 gives Lb and
%       Ub as absolute bounds (theta_cur -/+ cnt*Sr): self-contradictory. The decoded value spans
%       [Lb, Ub], the reading under which Eq. 5 adds theta_cur to bounds taken relative to theta_cur.
%     - Alg. 2 l.20 sets S_opt to the bounds after the l.13-19 update, one step past the sub-space just
%       scored, while Sec. 3.5 (p.4) selects "the sub-space with minimum SFitness": the scored
%       sub-space is used. The l.4 condition on vector bounds is read per joint and side ("any").
%     - Eq. 9 as printed, exp(+det(J*J')/coeM^2), is >= 1 and grows away from singularities, against the
%       stated range [0, 1] (p.4) and, under minimisation, against singularity avoidance (p.2, p.4).
%       exp(-det(J*J')/coeM^2) is used.
%     - Return rule (Output): the best individual ever evaluated rather than Alg. 1 l.13 alone.
%     - Crowd distance: an objective with equal values over a front adds 0 (division by zero in Deb et
%       al. 2002 otherwise).
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
        q_best = struct('params', emsa_params(), 'geometry', @emsa_geometry, ...
            'partial', @emsa_partial, 'eq3', @emsa_eq3, 'branch', @emsa_branch, ...
            'decode', @emsa_decode, 'encode', @emsa_encode, 'phi_c', @emsa_phi_c, ...
            'phi_l', @emsa_phi_l, 'phi_d', @emsa_phi_d, 'phi_m', @emsa_phi_m, ...
            'fitness', @emsa_fitness, 'eq12', @emsa_eq12, 'nd_sort', @emsa_nd_sort, ...
            'crowd', @emsa_crowd, 'tournament', @emsa_tournament, 'crossover', @emsa_crossover, ...
            'mutate', @emsa_mutate, 'elitist', @emsa_elitist, 'stall', @emsa_stall, ...
            'asss_boxes', @emsa_asss_boxes, 'asss_select', @emsa_asss_select, ...
            'context', @emsa_context, 'evaluate', @emsa_evaluate, 'run', @emsa_run);
        info = [];
        return
    end

    P0 = emsa_params();
    [q_best, info] = emsa_run(robot, T_d, q0, opts, P0.form);
end

function [q_best, info] = emsa_run(robot, T_d, q0, opts, form)
% One call of Algorithm 1. form 'asss': S_opt by Algorithm 2 (EMSA-IK); form 'joint_range': no
% Algorithm 2, the search space is the joint range of the parametric joints (EMSA-IK (w/o ASSS)).
    t_start = tic;
    P = emsa_params();
    [applicable, G] = emsa_geometry(robot);

    info = struct();
    info.solver = 'EMSA-IK';
    info.form   = form;
    if ~applicable
        q_best = q0;
        info.iters          = 0;
        info.time           = toc(t_start);
        info.converged      = false;
        info.fitness        = NaN;
        info.not_applicable = true;
        info.fk_calls       = 0;
        info.jac_calls      = 0;
        info.reason         = G.reason;
        return
    end

    C   = emsa_context(robot, T_d, q0, opts);
    n   = robot.n;
    m   = G.m;
    N   = P.N;
    L   = m * P.b;
    lo  = C.lo;
    hi  = C.hi;
    thc = C.thc;                                         % theta_cur (Algorithms 1-2)

    nfk  = 0;  njac = 0;
    best = struct('Fs', Inf, 'theta', thc, 'e12', NaN, 'T', [], 'src', 'none');

    % ------------------------------------------------ Alg. 1 l.1: S_opt by ASSS (Algorithm 2)
    if strcmp(form, 'asss')
        B = emsa_asss_boxes(thc(1:m), lo(1:m), hi(1:m), P.Sr);
    elseif strcmp(form, 'joint_range')                           % w/o ASSS: no l.1 (p.6)
        B = struct('Lb', zeros(m, 0), 'Ub', zeros(m, 0));
    else
        error('solve_EMSA_IK:form', 'Unknown form "%s".', form);
    end
    K  = size(B.Lb, 2);
    SF = zeros(1, 0);  NF = zeros(1, 0);
    for c = 1:K
        if emsa_left(opts, nfk, P.cap) < P.rounds * N, break; end
        sf = 0;  nf = 0;
        for i = 1:P.rounds                                       % Alg. 2 l.6
            R = randi([0 1], L, N);                              % l.7, Eq. 10
            for k = 1:N
                I = emsa_evaluate(C, R(:, k), B.Lb(:, c), B.Ub(:, c));
                nfk = nfk + 1;  njac = njac + I.feas;
                if I.feas                                        % l.8-10
                    sf = sf + I.Fs;  nf = nf + 1;
                    best = keep_best(best, I, 'asss');
                end
            end
        end
        SF(c) = sf;  NF(c) = nf;
    end
    if isempty(SF)
        c_opt = 1;
    else
        c_opt = emsa_asss_select(SF, NF);                        % l.22, empty sub-spaces not eligible
    end
    if strcmp(form, 'joint_range')                               % joint range (p.2, p.4, Fig. 1)
        Lb = lo(1:m);  Ub = hi(1:m);  c_opt = NaN;
    elseif K == 0
        Lb = thc(1:m) - P.Sr;  Ub = thc(1:m) + P.Sr;
    else
        Lb = B.Lb(:, c_opt);   Ub = B.Ub(:, c_opt);
    end
    fk_asss = nfk;

    % ------------------------------------------------ Alg. 1 l.2: first-generation population
    PB = zeros(L, 0);  PT = zeros(n, 0);  PF = zeros(n + 1, 0);  PS = zeros(1, 0);
    q0_feas = false;  q0_Fs = NaN;
    if emsa_left(opts, nfk, P.cap) >= 1                          % warm start: q0's theta_p
        I = emsa_evaluate(C, emsa_encode(thc(1:m), Lb, Ub, P.b), Lb, Ub);
        nfk = nfk + 1;  njac = njac + I.feas;
        q0_feas = I.feas;  q0_Fs = I.Fs;
        if I.feas
            [PB, PT, PF, PS] = add_ind(PB, PT, PF, PS, I);
            best = keep_best(best, I, 'q0');
        end
    end
    while size(PB, 2) < N && emsa_left(opts, nfk, P.cap) >= 1   % Eq. 10 until N feasible (p.5)
        I = emsa_evaluate(C, randi([0 1], L, 1), Lb, Ub);
        nfk = nfk + 1;  njac = njac + I.feas;
        if I.feas
            [PB, PT, PF, PS] = add_ind(PB, PT, PF, PS, I);
            best = keep_best(best, I, 'init');
        end
    end
    fk_init   = nfk - fk_asss;
    init_size = size(PB, 2);

    % ------------------------------------------------ Alg. 1 l.3-12: NSGA-II
    iters = 0;  hist = zeros(1, 0);  dq_stop = false;  dropped = false;  fk_dropped = 0;
    gen_fk = zeros(1, 0);  gen_ref = zeros(1, 0);  gen_offf = zeros(1, 0);
    rk = [];
    if size(PB, 2) == N
        rk = emsa_nd_sort(PF);                                   % l.3
        cr = emsa_crowd(PF, rk);
        hist = min(PS);
        while true                                               % l.4
            if emsa_left(opts, nfk, P.cap) < N, break; end
            fk0 = nfk;
            par = emsa_tournament(rk, cr, N / 2);                % l.5: half of the individuals (p.5)
            KB  = emsa_crossover(PB(:, par), P.pc, P.n_points);
            KB  = emsa_mutate(KB, P.pm);
            OB = zeros(L, 0);  OT = zeros(n, 0);  OF = zeros(n + 1, 0);  OS = zeros(1, 0);
            for k = 1:size(KB, 2)                                % l.6: feasible offspring kept
                if emsa_left(opts, nfk, P.cap) < 1, dropped = true; break; end
                I = emsa_evaluate(C, KB(:, k), Lb, Ub);
                nfk = nfk + 1;  njac = njac + I.feas;
                if I.feas
                    [OB, OT, OF, OS] = add_ind(OB, OT, OF, OS, I);
                    best = keep_best(best, I, 'gen');
                end
            end
            n_off = size(OB, 2);  nref = 0;
            while ~dropped && size(OB, 2) < N                    % l.7: refill to N by Eq. 10
                if emsa_left(opts, nfk, P.cap) < 1, dropped = true; break; end
                I = emsa_evaluate(C, randi([0 1], L, 1), Lb, Ub);
                nfk = nfk + 1;  njac = njac + I.feas;  nref = nref + 1;
                if I.feas
                    [OB, OT, OF, OS] = add_ind(OB, OT, OF, OS, I);
                    best = keep_best(best, I, 'gen');
                end
            end
            if dropped, fk_dropped = nfk - fk0; break; end
            MB = [PB, OB];  MT = [PT, OT];  MF = [PF, OF];  MS = [PS, OS];   % l.8
            rkM  = emsa_nd_sort(MF);                             % l.9
            crM  = emsa_crowd(MF, rkM);                          % l.10
            keep = emsa_elitist(rkM, MS, N);                     % l.11
            PB = MB(:, keep);  PT = MT(:, keep);  PF = MF(:, keep);  PS = MS(keep);
            rk = rkM(keep);    cr = crM(keep);
            iters = iters + 1;
            hist(end + 1)     = min(PS);                         %#ok<AGROW>
            gen_fk(end + 1)   = nfk - fk0;                       %#ok<AGROW>
            gen_ref(end + 1)  = nref;                            %#ok<AGROW>
            gen_offf(end + 1) = n_off;                           %#ok<AGROW>
            if emsa_stall(hist, P.Delta, P.Q), dq_stop = true; break; end   % termination (p.5)
        end
    end

    % ------------------------------------------------ Alg. 1 l.13 and output
    fit_alg1 = NaN;  th_alg1 = [];
    if ~isempty(PS)
        if isempty(rk), rk = emsa_nd_sort(PF); end
        j = emsa_elitist(rk, PS, 1);
        fit_alg1 = PS(j);  th_alg1 = PT(:, j);
    end
    if isfinite(best.Fs)
        q_best = best.theta;
        [~, ep, er] = pose_error(T_d, best.T);
        err_pos = norm(ep);  err_rot = norm(er);
    else
        q_best = q0;  err_pos = NaN;  err_rot = NaN;
    end

    info.iters              = iters;
    info.time               = toc(t_start);
    info.converged          = isfinite(best.Fs);                 % Eqs. 11-12 met by q_best
    info.fitness            = best.Fs;
    info.not_applicable     = false;
    info.fk_calls           = nfk;
    info.jac_calls          = njac;
    info.N                  = N;
    info.m                  = m;
    info.gamma              = C.gam;
    info.err_eq12           = best.e12;
    info.err_pos            = err_pos;
    info.err_rot            = err_rot;
    info.best_source        = best.src;
    info.asss_K             = K;                                 % sub-spaces Alg. 2 would score
    info.asss_scored        = numel(SF);
    info.asss_SFitness      = SF;                                % printed sum (0 if none feasible)
    info.asss_none_feasible = ~isempty(NF) && all(NF == 0);
    info.asss_n_feasible    = NF;
    info.asss_c_opt         = c_opt;
    info.asss_fk            = fk_asss;
    info.Sopt_Lb            = Lb;
    info.Sopt_Ub            = Ub;
    info.init_fk            = fk_init;
    info.init_size          = init_size;
    info.q0_member_feasible = q0_feas;
    info.q0_member_fitness  = q0_Fs;
    info.gen_fk             = gen_fk;
    info.gen_refill         = gen_ref;
    info.gen_offspring_feasible = gen_offf;
    info.history            = hist;                              % min fitness sum, generation 0..iters
    info.dq_stop            = dq_stop;
    info.generation_dropped = dropped;
    info.gen_dropped_fk     = fk_dropped;                        % FK of a generation cut by the budget
    info.fitness_alg1       = fit_alg1;
    info.theta_opt_alg1     = th_alg1;
    info.params             = P;
end

% ------------------------------------------------------------------ operators
function P = emsa_params()
    P = struct('N', 12, 'b', 16, 'Sr', 4 * pi / 180, 'pc', 0.6, 'pm', 0.1, 'n_points', 2, ...
               'Ow', [0.7, 0.2, 0.1], 'Fw', 1, 'coeC', 2, 'coeL', 0.05, 'coeD', 2, 'coeM', 0.5, ...
               'd_in', 0.01, 'd_out', 0.03, 'Delta', 1e-6, 'Q', 10, 'rounds', 5, ...
               'gamma_default', 1e-6, 'cap', 1e5, 'tol_dh', 1e-12, 'form', 'asss');
end

function [ok, G] = emsa_geometry(robot)
% Applicability of the closed form of Eq. 3 to joints m+1..n, from the DH table (no FK).
    P  = emsa_params();
    n  = robot.n;  m = n - 6;
    G  = struct('m', m, 'reason', '', 'a', [], 'd', [], 'alpha', [], 'off', [], 'tool', eye(4));
    ok = false;
    if isfield(robot, 'tool') && ~isempty(robot.tool), G.tool = robot.tool; end
    if m < 1
        G.reason = sprintf('%d joints: no redundant joint (q = n - 6 = %d)', n, m);
        return
    end
    dh = robot.dh;
    toff = zeros(1, n);
    if isfield(dh, 'theta_offset'), toff = dh.theta_offset(:)'; end
    k = m + 1:n;
    G.a = dh.a(k);  G.d = dh.d(k);  G.alpha = dh.alpha(k);  G.off = toff(k);
    t = P.tol_dh;  why = {};
    if any(abs(G.a) > t)
        why{end+1} = sprintf('a ~= 0 on links %s', mat2str(k(abs(G.a) > t)));
    end
    if any(abs(G.d([1 3 5])) > t)
        why{end+1} = sprintf('d ~= 0 on links %s', mat2str(k(find(abs(G.d([1 3 5])) > t) * 2 - 1)));
    end
    if abs(G.d(2)) <= t || abs(G.d(4)) <= t
        why{end+1} = 'd_(m+2) or d_(m+4) = 0';
    end
    if any(abs(sin(G.alpha(1:5))) <= 1e-9)
        why{end+1} = 'parallel consecutive axes among links m+1..m+5';
    end
    if ~isempty(why)
        G.reason = sprintf('theta_p = joints 1..%d; joints %d..%d: %s (no closed form for Eq. 3)', ...
                           m, m + 1, n, strjoin(why, '; '));
        return
    end
    ok = true;
end

function Tm = emsa_partial(dh, thp)
% Transform of the m parametric links, a term of Eq. 3 computed from the DH table (no FK call).
    m = numel(thp);
    toff = zeros(1, m);
    if isfield(dh, 'theta_offset'), toff = dh.theta_offset(1:m); end
    Tm = eye(4);
    for k = 1:m
        Tm = Tm * dh_transform(dh.a(k), dh.alpha(k), dh.d(k), thp(k) + toff(k));
    end
end

function [Th, isreal_b] = emsa_eq3(G, Tm, Tn, thc_r)
% Eq. 3: theta_r = FK^-1(x, theta_p) for joints m+1..n, all 8 branches (columns, joint values).
% isreal_b(j) is false when an arcsine or arccosine argument of branch j leaves [-1, 1] (p.5).
    d = G.d;  al = G.alpha;  off = G.off(:);
    Rm = Tm(1:3, 1:3);  Om = Tm(1:3, 4);
    Rn = Tn(1:3, 1:3);  Pn = Tn(1:3, 4);
    W  = Pn - d(6) * Rn * [0; sin(al(6)); cos(al(6))];      % wrist centre
    w  = Rm' * (W - Om);
    L1 = d(2);  L2 = d(4);
    Kc = (w' * w + L1^2 - L2^2) / (2 * L1);
    K1 = (Kc - cos(al(1)) * w(3)) / sin(al(1));
    rho = hypot(w(1), w(2));
    if rho < 1e-12
        re1 = abs(K1) < 1e-9;
        p1s = (thc_r(1) + off(1)) * [1, 1];
    else
        s = K1 / rho;
        re1 = abs(s) <= 1 + 1e-12;
        s = max(-1, min(1, s));
        be = atan2(w(2), w(1));
        p1s = [be + asin(s), be + pi - asin(s)];
    end
    Th = zeros(6, 8);  isreal_b = false(1, 8);  j = 0;
    for i1 = 1:2
        R1 = Rm * rotz(p1s(i1)) * rotx(al(1));
        E  = Om + L1 * R1(:, 3);
        h  = R1' * (W - E) / L2;
        [p2s, p3s, re2] = pan_tilt(h, al(2), al(3), thc_r(2) + off(2));
        for i2 = 1:2
            R3 = R1 * rotz(p2s(i2)) * rotx(al(2)) * rotz(p3s(i2)) * rotx(al(3));
            Gw = R3' * Rn * rotx(al(6))';
            [p4s, p5s, re3] = pan_tilt(Gw(:, 3), al(4), al(5), thc_r(4) + off(4));
            for i3 = 1:2
                H  = (rotz(p4s(i3)) * rotx(al(4)) * rotz(p5s(i3)) * rotx(al(5)))' * Gw;
                p6 = atan2(H(2, 1), H(1, 1));
                j  = j + 1;
                Th(:, j) = [p1s(i1); p2s(i2); p3s(i2); p4s(i3); p5s(i3); p6] - off;
                isreal_b(j) = re1 && re2 && re3;
            end
        end
    end
end

function [ps, rs, re] = pan_tilt(h, a, b, p_keep)
% Rz(p)*Rx(a)*Rz(r)*Rx(b)*e3 = h: two solutions (r = +/- arccos). p keeps p_keep when undefined.
    cr = (cos(a) * cos(b) - h(3)) / (sin(a) * sin(b));
    re = abs(cr) <= 1 + 1e-12;
    cr = max(-1, min(1, cr));
    rs = [acos(cr), -acos(cr)];
    ps = zeros(1, 2);
    for i = 1:2
        u1 = sin(b) * sin(rs(i));
        u2 = -cos(a) * sin(b) * cos(rs(i)) - sin(a) * cos(b);
        if hypot(h(1), h(2)) < 1e-12 && hypot(u1, u2) < 1e-12
            ps(i) = p_keep;
        else
            ps(i) = atan2(h(2), h(1)) - atan2(u2, u1);
        end
    end
end

function [th, j, inlim, D] = emsa_branch(Th, isreal_b, thc_r, lo_r, hi_r)
% Branch of Eq. 3 used for an individual: per joint the 2*pi equivalent nearest theta_cur (inside the
% limits where one exists); among real branches inside the limits the one nearest theta_cur.
    nb = size(Th, 2);  D = zeros(1, nb);  inlim = false(1, nb);  C = Th;
    for i = 1:nb
        [C(:, i), inlim(i)] = nearest_equivalent(Th(:, i), thc_r, lo_r, hi_r);
        D(i) = norm(C(:, i) - thc_r);
    end
    pool = find(isreal_b & inlim);
    if isempty(pool), pool = find(isreal_b); end
    if isempty(pool), pool = 1:nb; end
    [~, k] = min(D(pool));
    j  = pool(k);
    th = C(:, j);
end

function [q, ok] = nearest_equivalent(q, qs, lo, hi)
    ok = true;
    for k = 1:numel(q)
        c = q(k) + 2 * pi * (-3:3);
        in = c >= lo(k) & c <= hi(k);
        if any(in)
            c = c(in);
        else
            ok = false;
        end
        [~, j] = min(abs(c - qs(k)));
        q(k) = c(j);
    end
end

function th = emsa_decode(bits, Lb, Ub, b)
% Eq. 5 / Eq. 10 with absolute bounds: theta_j = Lb_j + sum_i(2^i*G_i)*(Ub_j - Lb_j)/(2^b - 1).
    q  = numel(Lb);  th = zeros(q, 1);  w = 2 .^ (0:b-1);
    for j = 1:q
        X = w * double(bits((j - 1) * b + (1:b)));
        th(j) = Lb(j) + X * (Ub(j) - Lb(j)) / (2^b - 1);
    end
end

function bits = emsa_encode(th, Lb, Ub, b)
% Nearest b-bit code [G_0..G_(b-1)] of th in [Lb, Ub] (inverse of Eq. 5).
    q = numel(th);  bits = zeros(q * b, 1);
    for j = 1:q
        X = round((th(j) - Lb(j)) / (Ub(j) - Lb(j)) * (2^b - 1));
        X = min(max(X, 0), 2^b - 1);
        bits((j - 1) * b + (1:b)) = bitget(X, 1:b)';
    end
end

function v = emsa_phi_c(th, thc, coeC)
    v = 1 - exp(-(th - thc).^2 / coeC^2);                                  % Eq. 6
end

function v = emsa_phi_l(th, lo, hi, coeL)
    v = 2 ./ (1 + exp((th - lo) / coeL)) + 2 ./ (1 + exp(-(th - hi) / coeL));   % Eq. 7
end

function v = emsa_phi_d(dist, P)
% Eq. 8 for a distance matrix dist(k, j), obstacle k and link j; an empty matrix (no obstacle) gives 0.
    D  = dist(:);
    pd = zeros(size(D));
    mid = D > P.d_in & D < P.d_out;
    pd(mid) = sin(pi * (P.d_out - D(mid)) / (2 * (P.d_out - P.d_in)));
    pd(D <= P.d_in) = 1;
    v = 1 - exp(-sum(pd)^2 / P.coeD^2);
end

function v = emsa_phi_m(J, coeM)
    v = exp(-det(J * J') / coeM^2);                                        % Eq. 9, sign see header
end

function [F, Fs] = emsa_fitness(th, thc, lo, hi, J, dist, P, Fw)
% Fig. 2: Fitness_i = Ow_1*phi_c,i + Ow_2*phi_l,i + Ow_3*phi_d, Fitness_(n+1) = phi_m, Fs = Fw'*F.
    F  = [P.Ow(1) * emsa_phi_c(th, thc, P.coeC) + P.Ow(2) * emsa_phi_l(th, lo, hi, P.coeL) ...
          + P.Ow(3) * emsa_phi_d(dist, P); emsa_phi_m(J, P.coeM)];
    Fs = sum(Fw(:) .* F);
end

function e = emsa_eq12(P_target, P_t)
    e = norm(P_target - P_t);                                              % Eq. 12, 2-norm
end

function C = emsa_context(robot, T_d, q0, opts)
% Data shared by the evaluations of one call: theta_cur = q0, limits, gamma, Fw, target.
    P = emsa_params();
    [~, G] = emsa_geometry(robot);
    gam = P.gamma_default;
    if isfield(opts, 'tol_pos') && isfield(opts, 'tol_rot')
        gam = min(opts.tol_pos, opts.tol_rot);
    end
    C = struct('robot', robot, 'G', G, 'P', P, 'T_d', T_d, 'T_n', T_d / G.tool, ...
               'thc', q0(:), 'lo', robot.q_min(:), 'hi', robot.q_max(:), 'gam', gam, ...
               'Fw', P.Fw * ones(robot.n + 1, 1));
end

function I = emsa_evaluate(C, bits, Lb, Ub)
% One individual: Eq. 5 decoding, Eq. 3, Eq. 11, Eq. 12 (one robot.fk call), fitness if feasible.
    rb = C.robot;  m = C.G.m;  n = rb.n;
    thp = emsa_decode(bits, Lb, Ub, C.P.b);
    Tm  = emsa_partial(rb.dh, thp);
    r   = m + 1:n;
    [Th, re] = emsa_eq3(C.G, Tm, C.T_n, C.thc(r));
    thr = emsa_branch(Th, re, C.thc(r), C.lo(r), C.hi(r));
    I.theta = [thp; thr];
    I.bits  = bits;
    I.T     = rb.fk(I.theta);                                              % P_t of Eq. 12
    I.e12   = emsa_eq12(C.T_d, I.T);
    I.lim   = all(I.theta >= C.lo & I.theta <= C.hi);                       % Eq. 11
    I.feas  = I.lim && I.e12 < C.gam;
    if I.feas
        [I.F, I.Fs] = emsa_fitness(I.theta, C.thc, C.lo, C.hi, rb.jac(I.theta), zeros(0, n), C.P, C.Fw);
    else
        I.F = [];  I.Fs = Inf;
    end
end

function rank = emsa_nd_sort(F)
% Non-dominated sorting (Deb et al. 2002) of the columns of F, all objectives minimised.
    N = size(F, 2);
    S = cell(1, N);  nd = zeros(1, N);  rank = zeros(1, N);
    for p = 1:N
        for r = 1:N
            if p == r, continue; end
            if all(F(:, p) <= F(:, r)) && any(F(:, p) < F(:, r))
                S{p}(end + 1) = r;
            elseif all(F(:, r) <= F(:, p)) && any(F(:, r) < F(:, p))
                nd(p) = nd(p) + 1;
            end
        end
    end
    front = find(nd == 0);  f = 1;
    while ~isempty(front)
        rank(front) = f;
        nxt = [];
        for p = front
            for r = S{p}
                nd(r) = nd(r) - 1;
                if nd(r) == 0, nxt(end + 1) = r; end            %#ok<AGROW>
            end
        end
        front = nxt;  f = f + 1;
    end
end

function cd = emsa_crowd(F, rank)
% Crowd distance per front (Deb et al. 2002); boundary individuals Inf; a constant objective adds 0.
    [M, N] = size(F);  cd = zeros(1, N);
    for f = unique(rank)
        idx = find(rank == f);
        if numel(idx) <= 2, cd(idx) = Inf; continue; end
        for k = 1:M
            [v, o] = sort(F(k, idx));
            cd(idx(o(1))) = Inf;  cd(idx(o(end))) = Inf;
            rngk = v(end) - v(1);
            if rngk <= 0, continue; end
            for i = 2:numel(idx) - 1
                cd(idx(o(i))) = cd(idx(o(i))) + (v(i + 1) - v(i - 1)) / rngk;
            end
        end
    end
end

function par = emsa_tournament(rank, cd, n_par)
% Binary tournament on Pareto rank, then crowd distance (p.5): n_par winners.
    N = numel(rank);  par = zeros(1, n_par);
    for k = 1:n_par
        ab = randi(N, 1, 2);
        a = ab(1);  c = ab(2);
        if rank(a) < rank(c) || (rank(a) == rank(c) && cd(a) >= cd(c))
            par(k) = a;
        else
            par(k) = c;
        end
    end
end

function K = emsa_crossover(Pb, pc, n_points)
% Multi-point crossover (p.5) of the pairs (1,2), (3,4), ... with probability pc: the bits after an
% odd number of the n_points cut positions are exchanged.
    [L, np] = size(Pb);  K = Pb;  idx = (1:L)';
    for k = 1:2:np - 1
        if rand() < pc
            cuts = sort(randperm(L - 1, n_points));
            sw = mod(sum(idx > cuts, 2), 2) == 1;
            K(sw, k) = Pb(sw, k + 1);
            K(sw, k + 1) = Pb(sw, k);
        end
    end
end

function K = emsa_mutate(K, pm)
    K = double(xor(K, rand(size(K)) < pm));                                % bitwise mutation (p.5)
end

function keep = emsa_elitist(rank, Fs, N)
% Elitist individuals (p.5): smaller Pareto rank first, ties by smaller fitness sum.
    [~, o] = sortrows([rank(:), Fs(:)]);
    keep = o(1:N)';
end

function tf = emsa_stall(hist, Delta, Q)
% Termination (p.5): the minimum fitness sum changed by less than Delta in each of the last Q generations.
    tf = numel(hist) >= Q + 1 && all(abs(diff(hist(end - Q:end))) < Delta);
end

function B = emsa_asss_boxes(thc, lo, hi, Sr)
% Algorithm 2 l.1-4 and l.13-19: the sub-spaces [Lb, Ub] that the while-loop scores, in order.
    cnt = 1;
    Lb = thc - cnt * Sr;  Ub = thc + cnt * Sr;
    B = struct('Lb', zeros(numel(thc), 0), 'Ub', zeros(numel(thc), 0));
    while any(Lb >= lo) || any(Ub <= hi)
        B.Lb(:, end + 1) = Lb;  B.Ub(:, end + 1) = Ub;
        cnt = cnt + 1;
        j = Lb >= lo;  Lb(j) = thc(j) - cnt * Sr;
        j = Ub <= hi;  Ub(j) = thc(j) + cnt * Sr;
    end
end

function c = emsa_asss_select(SF, NF)
% Alg. 2 l.22: minimum SFitness over the sub-spaces with at least one feasible individual (a sub-space
% with none has SFitness = +Inf); the first minimum, so sub-space 1 when none is eligible.
    v = SF;
    v(NF == 0) = Inf;
    [~, c] = min(v);
end

% ------------------------------------------------------------------ helpers
function v = emsa_left(opts, nfk, cap)
    v = ikb_budget_left(opts);
    if isinf(v), v = cap - nfk; end
end

function B = keep_best(B, I, src)
    if I.Fs < B.Fs
        B.Fs = I.Fs;  B.theta = I.theta;  B.e12 = I.e12;  B.T = I.T;  B.src = src;
    end
end

function [Bb, Bt, Bf, Bs] = add_ind(Bb, Bt, Bf, Bs, I)
    Bb(:, end + 1) = I.bits;  Bt(:, end + 1) = I.theta;  Bf(:, end + 1) = I.F;  Bs(end + 1) = I.Fs;
end

function R = rotz(t)
    R = [cos(t) -sin(t) 0; sin(t) cos(t) 0; 0 0 1];
end

function R = rotx(t)
    R = [1 0 0; 0 cos(t) -sin(t); 0 sin(t) cos(t)];
end
