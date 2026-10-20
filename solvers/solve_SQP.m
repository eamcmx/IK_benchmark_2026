function [q_best, info] = solve_SQP(robot, T_d, q0, opts)
% SOLVE_SQP  Combined FABRIK and SQP inverse kinematics of Xu et al. (2023): FABRIK with the wrist
%            target and wrist cone of Eqs. 6-7, the joint-position mapping of Eqs. 8-17 and the SQP
%            stage of Eq. 18 (Algorithm 1).
%
%   Source:
%     Zichun Xu, Yuntao Li, Xiaohang Yang, Zhiyuan Zhao, Jingdong Zhao, Hong Liu (2023). A Combined
%     Inverse Kinematics Algorithm Using FABRIK with Optimization. Journal of Intelligent & Robotic
%     Systems, 108:62 (volume 108, article 62, 17 pages; no issue number printed).
%     DOI: 10.1007/s10846-023-01895-2
%   Page numbers are PDF pages 1-17 (printed "Page x of 17").
%
%   Chain (Sec. 3, p.5; Fig. 5, p.7; Tables 5-7, p.15): n links and n_dof = 2n - 1 revolute joints,
%     a_i = 0, twists alternating -/+90 deg with alpha_(2n-1) = 0, d_(2h-1) = l_h > 0, d_(2h) = 0.
%     Joint 2h - 1 rolls about link h; joint 2h pitches at P_h, where the axes of joints 2h and 2h + 1
%     meet (the origin of DH frame 2h - 1). P_0 is the base, P_1 = P_1^int stays in place (link 1
%     lies on the axis of joint 1), P_n is the end effector.
%
%   Algorithm 1 (p.9):
%     l.1    P_t = P_des - R_des [0, 0, l_n]^T, the target of P_(n-1)                    (Eq. 6, p.5)
%     l.2    if P_t is reachable (see Readings):
%     l.3-6  while n <= n_l and (dist > eps_pos or P_(n-1)P_(n-2) outside the cone):
%              forward phase: P_(n-1) = P_t; P_(n-2) by Eq. 7 (cone of the last spherical wrist,
%              u = P_n P_(n-1) x P_(n-1) P_(n-2), p.5); P_(n-3) .. P_1 by Eq. 3 (p.4);
%              backward phase: P_1 = P_1^int; P_2 .. P_(n-1) by Eq. 5 (p.5).
%              A point P_i is placed from its anchor P_a (the point just placed) by
%                (1 - alpha_i) P_a + alpha_i P_i                          bend within the limit,
%                (1 - alpha_i R_u(dphi)) P_a + alpha_i R_u(dphi) P_i      otherwise,
%              alpha_i = l_i / d_i (d_i = |P_i - P_a|), dphi by Eq. 4, R_u(theta) = cos(theta) eye(3)
%              + sin(theta) [u]_x + (1 - cos(theta)) (u (x) u) (p.5), u = (link into P_a) x
%              (P_a P_i), the axis of the ball joint of Fig. 2b (p.5).
%              dist = ||P_(n-1) - P_t|| (Table 1, p.4; p.5).
%            after the iterations: P_n = P_(n-1) + P_t P_des (p.7, Fig. 6d)
%     l.7-8  dist <= eps_pos and the cone holds: Theta_des by Eqs. 8-16
%     l.9-11 n = n_l and dist > eps_pos: Theta_seed by Eqs. 8-17, Theta_des by Eq. 18
%   Mapping (pp.7-8; "reformulation of the method in [43]"):
%     |theta_i| = |pi - arccos((l_h^2 + l_(h+1)^2 - ||P_(h+1) - P_(h-1)||^2) / (2 l_h l_(h+1)))|,
%       i = 2, 4, .., 2n - 2, h = i/2                                                    (Eq. 8)
%     theta_1 from (A_1 A_2 A_3)_(1,4) = P_2x                                             (Eq. 9)
%     |theta_j| from (A_j A_(j+1) A_(j+2))_(1,4) = (A_(j-1)^-1 .. A_1^-1 [P_((j+3)/2); 1])_(1,1),
%       j = 3, 5, .., 2n - 5                                                              (Eq. 10)
%     |theta_k| from (A_k A_(k+1) A_(k+2))_(1,4) = (A_(k-1)^-1 .. A_1^-1 T_des)_(1,4), k = 2n - 3
%                                                                                         (Eq. 11)
%     |theta_l| = |arccos <x_(l-1), x_l>|, l = 2n - 1, x_(l-1) and x_l the first columns of
%       A_1 .. A_(l-1) and T_des                                                          (Eq. 12)
%     For this chain the left side of Eqs. 9-11 is c_x cos(theta_j), c_x = its value at theta_j = 0,
%     so |theta_j| = arccos(p_x / c_x) with p_x the right side; both roots of Eq. 9 are kept.
%     D = every sign combination of Eqs. 8-12 inside [Theta_min, Theta_max]               (Eq. 13)
%     eps_pos = ||P_temp - P_des||_2, eps_rot = arccos((tr(R_temp^-1 R_des) - 1) / 2), with
%       T_temp = A_1 .. A_(n_dof) the pose of each member of D                         (Eqs. 14-15)
%     S = {Theta in D : eps_pos <= eps_tol and eps_rot <= eps_tol}                         (Eq. 14)
%     Theta_des = argmin over S of ||Theta - Theta_init||_1                               (Eq. 16)
%     Theta_seed = argmin over D of (eps_pos + eps_rot)                                   (Eq. 17)
%   SQP stage: min phi_SS, the sum of squares of the twist p_err = [w^T v^T]^T, subject to
%     Theta_min <= Theta <= Theta_max, from Theta_seed (Eq. 18, p.9), by SLSQP [42] "updated by the
%     BFGS algorithm" (p.7) through the NLopt solver of TRAC-IK (p.9).
%
%   Parameters:
%     n_l = 15               switch index and FABRIK iteration limit (p.10; Table 2 also reports 5
%                            and 50, p.10; 15 is the value the paper selects).
%     eps_pos, eps_tol       10^-6 m and 10^-6 rad, the pose error of the paper's experiments (p.9);
%                            replaced by the benchmark pose gate (see Stop).
%     dphi_(n-1) of Eq. 7    "duly tuned" (p.5), no value given (checked pages 5-9); the form of
%                            Eq. 4 is used, which puts P_(n-2) on the cone surface.
%     SQP settings           iteration limit, tolerances, gradient and step rules: not stated in the
%                            paper (checked pages 3, 7-11). The benchmark uses fmincon 'sqp' with
%                            forward-difference gradients (FiniteDifferenceType 'forward', step
%                            sqrt(eps), the MATLAB defaults), MaxIterations = Inf and
%                            MaxFunctionEvaluations = the FK budget left (rule 3: the stage ends at
%                            the stopping test or the budget), OptimalityTolerance = StepTolerance =
%                            0 (so that fmincon's own tests do not end the stage before the pose gate),
%                            ConstraintTolerance 1e-6 (default; bounds only).
%     The 5 ms timeout of Sec. 4.3 (p.11) belongs to KDL, TRAC-IK and standalone SLSQP; the combined
%     algorithm has none, and no wall-clock limit is used.
%
%   Readings:
%     Eqs. 3-5 for n links   Eqs. 3-5 are printed for the planar 2-link arm of Fig. 2a; the 4-DOF arm
%                            of Fig. 2b replaces the joint axis by u = P_0P_1 x P_1P_2 (p.5), as Eq. 7
%                            does. On the chain of Sec. 3 every P_h is such a pair of joints, so u is
%                            the cross product of the two link directions, and the limit at P_h is
%                            that of the pitch joint 2h (the joint whose |theta| Eq. 8 measures).
%     Bend limit             phi is an unsigned angle in [0, pi]. Joint 2h realises a bend phi as
%                            theta = +phi or -phi (Eq. 13 admits both signs), so phi is within the
%                            limits if +phi or -phi, minus the DH offset, lies in [q_min, q_max]
%                            (mod 2 pi); otherwise dphi moves phi to the nearest such bend. For
%                            zero-offset limits inside [-pi, pi] this is Eq. 4 exactly; on
%                            Synthetic9/11/13 (limits +/-pi) no bend limit binds.
%     Eq. 11                 the printed right side is element (1,1) of A_(k-1)^-1 .. A_1^-1 T_des, a
%                            rotation entry equated with a coordinate (dimensionally inconsistent);
%                            element (1,4) is used, the x coordinate of P_des in frame k - 1, which is
%                            Eq. 10 with P_n = P_des.
%     Out-of-range arccos    an argument of Eqs. 8-12 outside [-1, 1] is clipped to [-1, 1] (the
%                            angle closest to satisfying the equation); Eqs. 14-17 then judge the
%                            candidate by its pose.
%     Degenerate c_x         c_x = 0 and p_x = 0 (straight chain at P_h) leave theta_j free;
%                            theta_j = 0 is taken.
%     Joint values           theta of Eqs. 8-12 is the DH angle; the joint value is theta minus the
%                            DH offset, taken as its representative in [q_min, q_min + 2 pi).
%     Reachability (l.2)     not defined (checked pages 5-9); the chain P_1 .. P_(n-1) reaches P_t
%                            if max(0, 2 max l - sum l) <= ||P_t - P_1|| <= sum l over l_2 .. l_(n-1).
%     While condition (l.3)  "n <= n_l, dist > eps_pos or ... not in cone" is read as n < n_l (at
%                            most n_l iterations, p.7: n_l "is also an iteration limit for FABRIK")
%                            and (dist > eps_pos or cone violated); dist and the cone are tested
%                            before the first iteration.
%     Twist (Eq. 18)         p_err = [w^T v^T]^T is not defined further (checked pages 8-9);
%                            v = P_des - P_temp and w = the rotation vector of R_des R_temp^-1, whose
%                            norms are eps_pos and eps_rot of Eq. 14, so phi_SS = eps_pos^2 + eps_rot^2
%                            (m^2 + rad^2). "p_err . p_err^T" is printed as an outer product; the text
%                            says "sum of squares".
%     eps_rot                arccos((tr(Re) - 1) / 2) is evaluated as atan2(||vee(Re - Re^T)|| / 2,
%                            (tr(Re) - 1) / 2), the same angle, without arccos's loss of precision
%                            below about 1e-8 rad.
%     Ties in Eq. 17         flipping a roll-pitch pair (theta_j + pi, -theta_(j+1)) leaves every
%                            P_h in place, so 2^(n-1) members of D share the minimum of Eq. 17 and
%                            differ only by rounding; values within 1e-12 (m + rad) of the minimum
%                            are equal, and the first in the order of Eq. 16 (L1 distance to
%                            Theta_init) is taken.
%     Empty S                Alg. 1 l.8 needs a member of S. If FABRIK converges but no member of D
%                            meets eps_tol, the seed of Eq. 17 and the SQP stage of l.10-11 are used
%                            (the switch of Fig. 1 for an unconverged FABRIK).
%     Order of Eq. 15        the members of D are evaluated in increasing ||Theta - Theta_init||_1;
%                            in l.8 the first member of S is Theta_des (the argmin of Eq. 16) and the
%                            rest of D is not evaluated; in l.10 all of D is evaluated.
%
%   Robots (DH tables of code/robots):
%     Robot        n_dof  Applicable  Reason
%     Synthetic9     9    yes         n = 5, l = [0.333 0.333 0.333 0.333 0.2] m, a = 0, twists
%                                     +/-90 deg alternating; |D| = 2^9 = 512.
%     Synthetic11   11    yes         n = 6, the same links; |D| = 2^11 = 2048.
%     Synthetic13   13    yes         n = 7, the same links; |D| = 2^13 = 8192.
%     UR5            6    no          even n_dof; a_2, a_3 ~= 0; d_4, d_6 ~= 0; parallel axes 2-4.
%     IRB120         6    no          even n_dof; a_2, a_3 ~= 0; d_4, d_6 ~= 0.
%     Panda          7    no          a_3, a_4, a_6 ~= 0 (joint offsets) and twists not alternating.
%     The paper covers the zero-offset chains of Fig. 5 and Tables 5-7 and names manipulators with
%     joint offsets as future work (p.15); Eqs. 8-12 need each P_h on two intersecting axes and the
%     roll axes along the links. On the robots outside the class the combined method does not exist,
%     and the solver sets info.not_applicable = true and returns q0 with no FK call.
%
%   Benchmark interface:
%     Budget: FK calls per waypoint = 1 (joint positions of the initial configuration) + 1 per member
%       of D whose pose Eq. 15 evaluates + 1 per fmincon objective call (forward-difference gradient
%       evaluations included). FABRIK iterations, Eq. 6, the reachability test and Eqs. 8-13 work on
%       positions and on the link transforms of the DH table and call no FK; robot.jac is never
%       called. An evaluation starts only if ikb_budget_left(opts) >= 1. fmincon starts only if the
%       budget left covers one iteration (2 n_dof + 2 FK), stops at an iterate when the budget left is
%       below n_dof + 1 (the next iteration no longer fits), and its objective stops it before an FK
%       call beyond the budget. On Synthetic11 and Synthetic13, D (2048 and 8192 members) exceeds the
%       budget: in l.10 the 1999 members nearest Theta_init are evaluated, the seed is the best of
%       them, and no budget is left for the SQP stage. No quantity varies with an iteration count,
%       so ikb_generations is not used; opts.max_iter is not used. Outside the harness the budget is
%       opts.budget minus the solver's own count; with no budget a safety cap of 1e5 FK calls applies.
%     Stop: the paper's tests with the benchmark pose gate for eps_pos and eps_tol (position <=
%       opts.tol_pos in m, rotation <= opts.tol_rot in rad, both required; 1e-5 each when a field is
%       absent): dist <= tol_pos ends FABRIK (l.3), S (Eq. 14) needs both errors; the SQP stage (no
%       stopping test stated) ends at the first iterate that meets the gate or at the budget.
%       info.converged = the returned joints meet the gate.
%     Warm start: Theta_init = q0 (the previous waypoint's solution), the initial configuration of
%       Fig. 1 and Alg. 1, and the reference of Eq. 16.
%     Units: m and rad (Tables 5-7 and eps_pos in m, p.9, p.15); no conversion.
%     Joint limits: as the paper: bend limits in Eqs. 3-5 and 7, members of D inside the limits
%       (Eq. 13), bounds of Eq. 18. The roll joints are limited only through Eq. 13.
%     Output: Theta_des of l.8, else the best SQP iterate (lowest phi_SS; the iterate that meets the
%       gate when one does), else Theta_seed; q0 when the target is unreachable or the budget allows no
%       evaluation of D. info.fitness = phi_SS of the returned joints (m^2 + rad^2), info.err_pos and
%       info.err_rot their eps_pos and eps_rot. info.stop is 'mapping' (l.8), 'seed' (Theta_seed meets
%       the gate), 'gate', 'budget', 'unreachable', 'fmincon <exitflag>' or 'not_applicable'.
%       solve_SQP('ops') returns handles to the operators for unit tests.
%
%   Deviations from the paper:
%     - SQP stage by MATLAB fmincon 'sqp', not SLSQP [42] through NLopt (p.9): its QP subproblem,
%       Hessian approximation, line search and merit function are MATLAB's.
%     - Gradients of phi_SS by forward differences (not stated in the paper), each an FK call.
%     - eps_pos and eps_tol (10^-6, the accuracy of the paper's experiments) replaced by the benchmark
%       pose gate; the SQP stage ends at the gate or the budget (no stopping test stated).
%     - The 2000-FK budget cuts Eq. 15 on D for Synthetic11 and Synthetic13 when FABRIK does not
%       converge within n_l: Theta_seed is the best of the 1999 members nearest Theta_init and the SQP
%       stage does not run. The full algorithm needs 1 + |D| FK before the SQP stage.
%     - Unreachable target (Alg. 1 l.2 has no output): q0 is returned.
%     - Readings above: Eq. 11 element (1,4); clipped arccos arguments; bend limits for offset joints;
%       the reachability test; empty S; ties in Eq. 17; coincident points (P_i = P_a) take the
%       direction of the link into P_a, and anti-parallel links take an axis u normal to that link;
%       eps_rot by atan2.
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
        q_best = struct('params', xu_params(), 'geometry', @xu_geometry, 'points', @xu_points, ...
            'target', @xu_target, 'reachable', @xu_reachable, 'bend_set', @xu_bend_set, ...
            'admissible', @xu_admissible, 'rot', @xu_rot, 'place', @xu_place, ...
            'forward', @xu_forward, 'backward', @xu_backward, 'cone', @xu_cone, ...
            'fabrik', @xu_fabrik, 'eq8', @xu_eq8, 'mapping', @xu_mapping, ...
            'err14', @xu_err14, 'core', @xu_core);
        info = [];
        return
    end
    [q_best, info] = xu_core(robot, T_d, q0, opts, xu_params());
end

% ==================================================================== Algorithm 1
function [q_best, info] = xu_core(robot, T_d, q0, opts, P)
    t_start = tic;
    q0 = q0(:);
    q_best = q0;
    [ok, G] = xu_geometry(robot);

    info = struct();
    info.solver         = 'FABRIK+SQP';
    info.not_applicable = ~ok;
    info.reason         = G.reason;
    info.iters          = 0;
    info.converged      = false;
    info.fitness        = NaN;
    info.err_pos        = NaN;
    info.err_rot        = NaN;
    info.fk_calls       = 0;
    info.jac_calls      = 0;
    info.stop           = 'not_applicable';
    info.branch         = 'none';
    info.fabrik_iters   = 0;
    info.fabrik_conv    = false;
    info.dist_history   = zeros(1, 0);
    info.n_D            = 0;
    info.n_eval15       = 0;
    info.n_tie          = 0;
    info.n_clip         = 0;
    info.n_guard        = 0;
    info.sqp_iters      = 0;
    info.sqp_fk         = 0;
    info.sqp_exitflag   = NaN;
    info.sqp_phi        = zeros(1, 0);
    info.sqp_nomatch    = 0;
    info.seed           = NaN(numel(q0), 1);
    info.params         = P;
    if ~ok
        info.time = toc(t_start);
        return
    end

    [tp, tr] = pose_gate(opts);
    info.gate = [tp, tr];
    fk = 0;
    info.stop = 'budget';
    if budget_left(opts, fk) < 1
        info.time = toc(t_start);
        return
    end

    % Initial configuration: joint positions P_0 .. P_n (Fig. 5), one FK call.
    [~, Ts] = robot.fk(q0);
    fk = fk + 1;
    X = xu_points(Ts, G);

    Pdes = T_d(1:3, 4);
    Pt   = xu_target(T_d, G.l(end));                                   % Eq. 6 (Alg. 1 l.1)
    if ~xu_reachable(X, Pt, G)                                         % Alg. 1 l.2
        info.stop = 'unreachable';  info.branch = 'unreachable';
        info.fk_calls = fk;  info.time = toc(t_start);
        return
    end

    [X, F] = xu_fabrik(X, Pt, Pdes, G, P.n_l, tp);                    % Alg. 1 l.3-6, shift (p.7)
    info.fabrik_iters = F.iters;
    info.fabrik_conv  = F.converged;
    info.dist_history = F.hist;
    info.n_guard      = F.n_guard;

    D = xu_mapping(X, T_d, G);                                         % Eqs. 8-13
    B = size(D.Q, 2);
    info.n_D    = B;
    info.n_clip = nnz(D.clip);
    [~, ord] = sort(sum(abs(D.Q - q0), 1));                            % ||Theta - Theta_init||_1 (Eq. 16)
    E = NaN(2, B);                                                     % eps_pos, eps_rot (Eqs. 14-15)
    found = 0;
    if F.converged                                                     % Alg. 1 l.7-8: Eqs. 8-16
        info.branch = 'mapping';
        for k = ord
            if budget_left(opts, fk) < 1, break; end
            T = robot.fk(D.Q(:, k));                                   % Eq. 15
            fk = fk + 1;
            [E(1, k), E(2, k)] = xu_err14(T, T_d);
            if E(1, k) <= tp && E(2, k) <= tr                          % Theta in S (Eq. 14)
                found = k;
                break
            end
        end
    else                                                               % Alg. 1 l.9-10: Eqs. 8-17
        info.branch = 'sqp';
        for k = ord
            if budget_left(opts, fk) < 1, break; end
            T = robot.fk(D.Q(:, k));                                   % Eq. 15
            fk = fk + 1;
            [E(1, k), E(2, k)] = xu_err14(T, T_d);
        end
    end
    ne  = nnz(~isnan(E(1, :)));
    seq = ord(1:ne);                                                   % evaluation order
    info.n_eval15 = ne;

    if found > 0                                                       % Eq. 16 (first member of S in L1 order)
        q_best = D.Q(:, found);
        info.converged = true;
        info.stop = 'mapping';
        info = finish(info, E(:, found), fk, t_start);
        return
    end
    if ne == 0
        info.fk_calls = fk;  info.time = toc(t_start);
        return
    end
    val = E(1, seq) + E(2, seq);                                       % Eq. 17; equal values (within
    i   = find(val <= min(val) + P.tie, 1, 'first');                  % P.tie) by the L1 order of Eq. 16
    ks  = seq(i);
    info.n_tie = nnz(val <= min(val) + P.tie);
    seed = D.Q(:, ks);
    info.seed = seed;
    q_best = seed;
    if E(1, ks) <= tp && E(2, ks) <= tr
        info.converged = true;
        info.stop = 'seed';
        info = finish(info, E(:, ks), fk, t_start);
        return
    end
    if F.converged
        info.branch = 'mapping+sqp';                                   % S empty: SQP from Eq. 17
    end
    n = G.N;
    if ne < B || budget_left(opts, fk) < 2 * n + 2                     % D cut by the budget, or no SQP iteration fits
        info = finish(info, E(:, ks), fk, t_start);
        return
    end

    S = xu_sqp(robot, T_d, seed, opts, fk, tp, tr);                    % Eq. 18
    fk = S.fk;
    info.sqp_iters    = S.iters;
    info.sqp_fk       = S.fk - S.fk0;
    info.sqp_exitflag = S.exitflag;
    info.sqp_phi      = S.phi;
    info.sqp_nomatch  = S.nomatch;
    info.stop         = S.stop;
    info.converged    = S.hit;
    if ~isempty(S.q)
        q_best = S.q;
        info = finish(info, S.e, fk, t_start);
    else
        info = finish(info, E(:, ks), fk, t_start);
    end
end

function info = finish(info, e, fk, t_start)
    info.err_pos  = e(1);
    info.err_rot  = e(2);
    info.fitness  = e(1)^2 + e(2)^2;                                   % phi_SS (Eq. 18)
    info.fk_calls = fk;
    info.iters    = info.fabrik_iters + info.sqp_iters;
    info.time     = toc(t_start);
end

% ==================================================================== parameters and geometry
function P = xu_params()
% n_l = 15 (p.10); tie = width of an equal value in Eq. 17 (m + rad).
    P = struct('n_l', 15, 'tie', 1e-12);
end

function [ok, G] = xu_geometry(robot)
% The chain class of Sec. 3 (n_dof = 2n - 1, Fig. 5, Tables 5-7), read from the DH table.
    G = struct('reason', '');
    if ~isfield(robot, 'dh')
        ok = false;  G.reason = 'no DH table';
        return
    end
    N  = robot.n;
    a  = robot.dh.a(:).';  al = robot.dh.alpha(:).';  d = robot.dh.d(:).';
    off = zeros(1, N);
    if isfield(robot.dh, 'theta_offset'), off = robot.dh.theta_offset(:).'; end
    why = {};
    if mod(N, 2) == 0 || N < 5
        why{end+1} = sprintf('%d joints, not n_dof = 2n - 1 with n >= 3', N);
    end
    ia = find(a ~= 0);
    if ~isempty(ia), why{end+1} = sprintf('a ~= 0 on links %s', mat2str(ia)); end
    ie = 2 * find(d(2:2:end) ~= 0);
    if ~isempty(ie), why{end+1} = sprintf('d ~= 0 on even links %s', mat2str(ie)); end
    io = 2 * find(d(1:2:end) <= 0) - 1;
    if ~isempty(io), why{end+1} = sprintf('d <= 0 on odd links %s', mat2str(io)); end
    tw = N >= 2 && all(abs(abs(al(1:N-1)) - pi/2) < 1e-12) && all(abs(al(2:N-1) + al(1:N-2)) < 1e-12) ...
         && abs(al(N)) < 1e-12;
    if ~tw, why{end+1} = 'twists not alternating -/+90 deg with alpha_n = 0'; end
    if isfield(robot, 'tool') && ~isequal(robot.tool, eye(4))
        why{end+1} = 'tool transform not identity';
    end
    ok = isempty(why);
    G.reason = strjoin(why, '; ');
    if ~ok, return; end
    n = (N + 1) / 2;
    G.N = N;  G.n = n;  G.a = a;  G.alpha = al;  G.d = d;  G.off = off;
    G.l = d(1:2:N);                                                    % l_h = d_(2h-1)
    G.qmin = robot.q_min(:);  G.qmax = robot.q_max(:);
    G.bend = cell(1, n - 1);                                           % joint 2h at P_h
    for h = 1:n-1
        m = 2 * h;
        G.bend{h} = xu_bend_set(G.qmin(m) + off(m), G.qmax(m) + off(m));
    end
end

function X = xu_points(Ts, G)
% P_0 = base origin, P_h = origin of frame 2h - 1 (axes of joints 2h and 2h + 1 meet there),
% P_n = origin of frame 2n - 1 (end effector).
    X = zeros(3, G.n + 1);
    for h = 1:G.n
        X(:, h + 1) = Ts(1:3, 4, 2 * h - 1);
    end
end

function Pt = xu_target(T_d, ln)
% Eq. 6: P_t = P_des - R_des [0, 0, l_n]^T.
    Pt = T_d(1:3, 4) - T_d(1:3, 1:3) * [0; 0; ln];
end

function ok = xu_reachable(X, Pt, G)
% Alg. 1 l.2 for the chain P_1 .. P_(n-1) with links l_2 .. l_(n-1), P_1 in place.
    L  = G.l(2:G.n-1);
    r  = norm(Pt - X(:, 2));
    ok = r <= sum(L) && r >= max(0, 2 * max(L) - sum(L));
end

function S = xu_bend_set(lo, hi)
% Bends phi in [0, pi] realisable by theta = +phi or -phi (mod 2 pi) inside [lo, hi] (DH angle).
    S = zeros(0, 2);
    for k = -2:2
        S = [S; max(0, lo - 2*pi*k), min(pi, hi - 2*pi*k); ...
                max(0, -(hi - 2*pi*k)), min(pi, -(lo - 2*pi*k))]; %#ok<AGROW>
    end
    S = S(S(:, 1) <= S(:, 2), :);
    S = sortrows(S);
    M = zeros(0, 2);
    for i = 1:size(S, 1)
        if ~isempty(M) && S(i, 1) <= M(end, 2)
            M(end, 2) = max(M(end, 2), S(i, 2));
        else
            M(end+1, :) = S(i, :); %#ok<AGROW>
        end
    end
    S = M;
end

function [ok, dphi] = xu_admissible(phi, S)
% phi inside the bend set; otherwise Delta phi to the nearest admissible bend (Eq. 4).
    ok = any(phi >= S(:, 1) & phi <= S(:, 2));
    dphi = 0;
    if ~ok
        c = S(:);
        [~, i] = min(abs(c - phi));
        dphi = c(i) - phi;
    end
end

function R = xu_rot(u, t)
% R_u(theta) = cos(theta) I + sin(theta) [u]_x + (1 - cos(theta)) (u (x) u)  (p.5).
    ux = [0, -u(3), u(2); u(3), 0, -u(1); -u(2), u(1), 0];
    R  = cos(t) * eye(3) + sin(t) * ux + (1 - cos(t)) * (u * u.');
end

% ==================================================================== FABRIK (Eqs. 3-7)
function [Xn, g, lim] = xu_place(A, B, a, L, S)
% New position of a point B on the link of length L from A (Eqs. 3, 5, 7):
%   (1 - alpha) A + alpha B                         bend within the limits,
%   (1 - alpha R_u(dphi)) A + alpha R_u(dphi) B     otherwise,
% alpha = L / |B - A|, phi = angle between a (the link into A) and B - A, u = a x (B - A).
    g = 0;  lim = false;
    b = B - A;
    if norm(b) == 0                                                    % coincident points
        B = A + a;  b = a;  g = 1;
    end
    al  = L / norm(b);
    phi = atan2(norm(cross(a, b)), dot(a, b));
    [ok, dphi] = xu_admissible(phi, S);
    if ok
        Xn = (1 - al) * A + al * B;
    else
        lim = true;
        u = cross(a, b);
        if norm(u) == 0                                                % a and b anti-parallel
            [~, k] = min(abs(a));  e = zeros(3, 1);  e(k) = 1;
            u = cross(a, e);  g = g + 1;
        end
        u  = u / norm(u);
        R  = xu_rot(u, dphi);
        Xn = (eye(3) - al * R) * A + al * R * B;
    end
end

function [X, g, nl] = xu_forward(X, Pt, Pdes, G)
% Forward phase: P_(n-1) = P_t; P_(n-2) by Eq. 7; P_(n-3) .. P_1 by Eq. 3 (column h + 1 = P_h).
    n = G.n;  g = 0;  nl = 0;
    X(:, n) = Pt;
    [X(:, n-1), gi, li] = xu_place(X(:, n), X(:, n-1), Pt - Pdes, G.l(n-1), G.bend{n-1});
    g = g + gi;  nl = nl + li;
    for i = n-3:-1:1
        [X(:, i+1), gi, li] = xu_place(X(:, i+2), X(:, i+1), X(:, i+2) - X(:, i+3), G.l(i+1), G.bend{i+1});
        g = g + gi;  nl = nl + li;
    end
end

function [X, g, nl] = xu_backward(X, P1, G)
% Backward phase, Eq. 5: P_1 = P_1^int; P_2 .. P_(n-1) outward.
    g = 0;  nl = 0;
    X(:, 2) = P1;
    for i = 2:G.n-1
        [X(:, i+1), gi, li] = xu_place(X(:, i), X(:, i+1), X(:, i) - X(:, i-1), G.l(i), G.bend{i-1});
        g = g + gi;  nl = nl + li;
    end
end

function ok = xu_cone(X, zd, G)
% P_(n-1) P_(n-2) in the cone of the last spherical wrist (joint 2n - 2).
    n = G.n;
    a = X(:, n) - X(:, n-1);
    phi = atan2(norm(cross(a, zd)), dot(a, zd));
    ok = xu_admissible(phi, G.bend{n-1});
end

function [X, F] = xu_fabrik(X, Pt, Pdes, G, n_l, tp)
% Alg. 1 l.3-6, then the last-link shift P_n = P_(n-1) + P_t P_des (p.7).
    n  = G.n;
    P1 = X(:, 2);
    zd = Pdes - Pt;
    it = 0;  g = 0;  nl = 0;
    dist = norm(X(:, n) - Pt);
    cone = xu_cone(X, zd, G);
    hist = dist;
    while it < n_l && (dist > tp || ~cone)
        [X, g1, l1] = xu_forward(X, Pt, Pdes, G);
        [X, g2, l2] = xu_backward(X, P1, G);
        it = it + 1;  g = g + g1 + g2;  nl = nl + l1 + l2;
        dist = norm(X(:, n) - Pt);
        cone = xu_cone(X, zd, G);
        hist(end+1) = dist; %#ok<AGROW>
    end
    X(:, n+1) = X(:, n) + (Pdes - Pt);
    F = struct('iters', it, 'converged', dist <= tp && cone, 'dist', dist, 'cone', cone, ...
               'hist', hist, 'n_guard', g, 'n_limited', nl);
end

% ==================================================================== mapping (Eqs. 8-13)
function b = xu_eq8(X, G)
% Eq. 8: |theta_2h| = |pi - arccos((l_h^2 + l_(h+1)^2 - ||P_(h+1) - P_(h-1)||^2) / (2 l_h l_(h+1)))|.
    n = G.n;  l = G.l;
    b = zeros(1, n - 1);
    for h = 1:n-1
        c = (l(h)^2 + l(h+1)^2 - sum((X(:, h+2) - X(:, h)).^2)) / (2 * l(h) * l(h+1));
        b(h) = abs(pi - acos(min(1, max(-1, c))));
    end
end

function D = xu_mapping(X, T_d, G)
% The set D of Eq. 13: every sign combination of Eqs. 8-12 inside the joint limits, in DH angles
% (TH) and joint values (Q = theta - offset, the representative in [q_min, q_min + 2 pi)).
    N = G.N;  n = G.n;
    b  = xu_eq8(X, G);
    TH = zeros(N, 1);
    M  = eye(4);
    CL = 0;
    clipb = @(x) min(1, max(-1, x));
    for j = 1:2:2*n-3
        h = (j + 1) / 2;
        if j <= 2*n - 5
            tgt = X(:, (j + 3) / 2 + 1);                               % P_((j+3)/2)
        else
            tgt = T_d(1:3, 4);                                         % Eq. 11
        end
        Bn = size(TH, 2);
        px = reshape(sum(M(1:3, 1, :) .* (tgt - M(1:3, 4, :)), 1), 1, Bn);
        THn = zeros(N, 4 * Bn);  CLn = zeros(1, 4 * Bn);  Mn = zeros(4, 4, 4 * Bn);
        c = 0;
        for se = [1, -1]
            te = se * b(h);
            C  = dh1(G, j, 0) * dh1(G, j+1, te) * dh1(G, j+2, 0);
            cx = C(1, 4);
            arg = px / cx;
            arg(isnan(arg)) = 1;
            cl  = abs(arg) > 1;
            ac  = acos(clipb(arg));
            for so = [1, -1]
                to  = so * ac;
                idx = c + (1:Bn);
                T2  = TH;  T2(j, :) = to;  T2(j+1, :) = te;
                THn(:, idx) = T2;
                CLn(idx)    = CL + cl;
                Mn(:, :, idx) = pagemtimes(pagemtimes(M, dh_pages(G, j, to)), dh_pages(G, j+1, te * ones(1, Bn)));
                c = c + Bn;
            end
        end
        ordr = reshape(reshape(1:4*Bn, Bn, 4).', 1, []);              % branch-major order
        THn = THn(:, ordr);  CLn = CLn(ordr);  Mn = Mn(:, :, ordr);
        keep = inlim(G, THn(j, :), j) & inlim(G, THn(j+1, :), j+1);   % Eq. 13
        TH = THn(:, keep);  CL = CLn(keep);  M = Mn(:, :, keep);
    end
    Bn  = size(TH, 2);                                                 % Eq. 12, l = 2n - 1
    arg = reshape(sum(M(1:3, 1, :) .* T_d(1:3, 1), 1), 1, Bn);
    cl  = abs(arg) > 1;
    ac  = acos(clipb(arg));
    THn = [TH, TH];  THn(N, :) = [ac, -ac];
    CLn = [CL + cl, CL + cl];
    ordr = reshape(reshape(1:2*Bn, Bn, 2).', 1, []);
    THn = THn(:, ordr);  CLn = CLn(ordr);
    keep = inlim(G, THn(N, :), N);
    D.TH = THn(:, keep);
    D.clip = CLn(keep);
    D.Q = rep(G, D.TH);
    D.bends = b;
end

function ok = inlim(G, th, m)
% Joint m inside [q_min, q_max] (Eq. 13), as the representative of theta - offset.
    q  = G.qmin(m) + mod(th - G.off(m) - G.qmin(m), 2*pi);
    ok = q <= G.qmax(m);
end

function Q = rep(G, TH)
    Q = G.qmin + mod(TH - G.off(:) - G.qmin, 2*pi);
end

function A = dh1(G, m, th)
    A = dh_transform(G.a(m), G.alpha(m), G.d(m), th);
end

function A = dh_pages(G, m, th)
% dh_transform for a row of angles, as 4 x 4 pages.
    B  = numel(th);
    ct = reshape(cos(th), 1, 1, B);  st = reshape(sin(th), 1, 1, B);
    ca = cos(G.alpha(m));  sa = sin(G.alpha(m));
    A  = zeros(4, 4, B);
    A(1, 1, :) = ct;   A(1, 2, :) = -st * ca;  A(1, 3, :) = st * sa;   A(1, 4, :) = G.a(m) * ct;
    A(2, 1, :) = st;   A(2, 2, :) = ct * ca;   A(2, 3, :) = -ct * sa;  A(2, 4, :) = G.a(m) * st;
    A(3, 2, :) = sa;   A(3, 3, :) = ca;        A(3, 4, :) = G.d(m);    A(4, 4, :) = 1;
end

% ==================================================================== errors (Eq. 14) and SQP (Eq. 18)
function [ep, er] = xu_err14(T, T_d)
% eps_pos = ||P_temp - P_des||_2 (m); eps_rot = arccos((tr(R_temp^-1 R_des) - 1) / 2) (rad),
% evaluated as atan2(||vee(Re - Re^T)|| / 2, (tr(Re) - 1) / 2), Re = R_temp^-1 R_des (same angle).
    ep = norm(T(1:3, 4) - T_d(1:3, 4));
    Re = T(1:3, 1:3).' * T_d(1:3, 1:3);
    v  = [Re(3, 2) - Re(2, 3); Re(1, 3) - Re(3, 1); Re(2, 1) - Re(1, 2)];
    er = atan2(norm(v) / 2, (Re(1, 1) + Re(2, 2) + Re(3, 3) - 1) / 2);
end

function S = xu_sqp(robot, T_d, seed, opts, fk0, tp, tr)
% Eq. 18: min phi_SS = ||w||^2 + ||v||^2 = eps_rot^2 + eps_pos^2 subject to the joint limits, by
% fmincon 'sqp' with forward-difference gradients, from the Eq. 17 seed; stops at the pose
% gate (any iterate) or when the next iteration no longer fits in the budget.
    nq = numel(seed);
    fk = fk0;
    cX = zeros(nq, 0);  cE = zeros(2, 0);
    iX = zeros(nq, 0);  iE = zeros(2, 0);
    its = 0;  hit = false;  aborted = false;  bstop = false;  nomatch = 0;
    left = budget_left(opts, fk);
    o = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off', ...
        'MaxFunctionEvaluations', left, 'MaxIterations', Inf, ...
        'OptimalityTolerance', 0, 'StepTolerance', 0, 'OutputFcn', @outf);
    flag = NaN;
    try
        [~, ~, flag] = fmincon(@obj, seed, [], [], [], [], robot.q_min(:), robot.q_max(:), [], o);
    catch err
        if ~aborted, rethrow(err); end
    end
    S = struct('fk0', fk0, 'fk', fk, 'iters', its, 'hit', hit, 'exitflag', flag, 'nomatch', nomatch, ...
               'phi', iE(1, :).^2 + iE(2, :).^2, 'q', [], 'e', [NaN; NaN]);
    if hit
        S.stop = 'gate';
    elseif aborted || bstop || flag == 0
        S.stop = 'budget';
    else
        S.stop = sprintf('fmincon %d', flag);
    end
    if ~isempty(iX)
        if hit
            k = size(iX, 2);
        else
            [~, k] = min(S.phi);
        end
        S.q = iX(:, k);
        S.e = iE(:, k);
    end

    function f = obj(x)
        if budget_left(opts, fk) < 1
            aborted = true;
            error('solve_SQP:budget', 'FK budget used up.');
        end
        T  = robot.fk(x);
        fk = fk + 1;
        [ep, er] = xu_err14(T, T_d);
        cX(:, end+1) = x;
        cE(:, end+1) = [ep; er];
        f = ep^2 + er^2;
    end

    function stop = outf(x, ~, state)
        stop = false;
        if ~any(strcmp(state, {'init', 'iter', 'done'})), return; end
        k = [];
        for c = size(cX, 2):-1:1                                       % the iterate is a recent evaluation
            if isequal(cX(:, c), x(:)), k = c; break; end
        end
        if isempty(k)
            nomatch = nomatch + 1;
            return
        end
        if strcmp(state, 'iter'), its = its + 1; end
        if isempty(iX) || ~isequal(iX(:, end), x(:))
            iX(:, end+1) = x(:);
            iE(:, end+1) = cE(:, k);
        end
        if cE(1, k) <= tp && cE(2, k) <= tr
            hit = true;
            stop = true;
        elseif strcmp(state, 'iter') && budget_left(opts, fk) < nq + 1
            bstop = true;
            stop = true;
        end
    end
end

% ==================================================================== interface helpers
function [tp, tr] = pose_gate(opts)
% The benchmark pose gate: tol_pos (m) and tol_rot (rad); 1e-5 each when a field is absent.
    tp = 1e-5;  tr = 1e-5;
    if isfield(opts, 'tol_pos'), tp = opts.tol_pos; end
    if isfield(opts, 'tol_rot'), tr = opts.tol_rot; end
end

function n = budget_left(opts, fk)
% FK calls still allowed on this waypoint: ikb_budget_left inside the harness, opts.budget minus
% the solver's own count outside it, and a safety cap of 1e5 FK calls without any budget.
    n = ikb_budget_left(opts);
    if isinf(n) && isfield(opts, 'budget') && ~isempty(opts.budget)
        n = opts.budget - fk;
    end
    if isinf(n)
        n = 1e5 - fk;
    end
end
