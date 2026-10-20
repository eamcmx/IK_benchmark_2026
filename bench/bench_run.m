function results = bench_run(robot, solver_fn, traj_Ts, q0, opts)
% BENCH_RUN  Run a single (robot, solver, trajectory) benchmark.
%
%   Solves IK at each waypoint of traj_Ts using the given solver function
%   handle, warm-starting each step from the previous step's solution
%   (the recursive-IK use case).
%
%   Measurement is done by the harness, not by the solver:
%     - time:  wall clock (tic/toc) around the whole solver call, so the
%              Phase-2 DLS polish is included;
%     - work:  every FK and Jacobian evaluation the solver makes is counted by
%              the evaluation counters inside fk_chain and jac_geometric
%              (reset before each waypoint, read right after the call);
%     - budget: the base solver may use at most opts.budget FK calls per
%              waypoint; ikb_budget_left(opts) tells it how many remain, and
%              any waypoint that goes over is flagged in .budget_exceeded;
%     - scoring: the pose error is recomputed from FK on the returned joint
%              vector with an UNcounted robot, so scoring is not charged to
%              the solver; both position and rotation must be within tolerance.
%   A solver that cannot handle a robot must set info.not_applicable = true;
%   such waypoints are flagged in .not_applicable and never scored as solved.
%
%   results fields:
%     .Q               : n x N joint trajectory
%     .errs_pos        : 1 x N final position error per step (m)
%     .errs_rot        : 1 x N final rotation error per step (rad)
%     .iters           : 1 x N iterations per step (solver-reported; NaN if absent)
%     .times           : 1 x N wall-clock solve time per step (s), measured by the harness
%     .fk_calls        : 1 x N FK calls per step, counted by the harness (base + polish)
%     .jac_calls       : 1 x N Jacobian calls per step, counted by the harness (base + polish)
%     .fk_base         : 1 x N FK calls of the base solver alone (= fk_calls in Phase 1)
%     .jac_base        : 1 x N Jacobian calls of the base solver alone
%     .budget_exceeded : 1 x N logical, fk_base > opts.budget
%     .not_applicable  : 1 x N logical, solver declared the robot outside its scope
%     .times_self      : 1 x N solver-reported time (info.time), kept for reference only
%     .fk_self         : 1 x N solver-reported FK count (info.fk_calls), reference only
%     .converged       : 1 x N logical
%     .manip           : 1 x N Yoshikawa manipulability
%     .lim_dist        : 1 x N normalised distance to nearest joint limit
%     .info            : cell array of per-step solver info structs

    if nargin < 5, opts = default_opts(); end
    n = robot.n;
    N = size(traj_Ts, 3);

    Q               = zeros(n, N);
    errs_pos        = zeros(1, N);
    errs_rot        = zeros(1, N);
    iters           = nan(1, N);
    times           = zeros(1, N);
    fk_calls        = zeros(1, N);
    jac_calls       = zeros(1, N);
    fk_base         = zeros(1, N);
    jac_base        = zeros(1, N);
    budget_exceeded = false(1, N);
    not_applicable  = false(1, N);
    times_self      = nan(1, N);
    fk_self         = nan(1, N);
    converged       = false(1, N);
    manip           = zeros(1, N);
    lim_dist        = zeros(1, N);
    info_cell       = cell(1, N);

    % Harness-side instrumentation: the counters live in fk_chain / jac_geometric.
    opts.ikb_counting = true;
    budget = Inf;
    if isfield(opts, 'budget') && ~isempty(opts.budget), budget = opts.budget; end

    q = q0(:);
    tp = 1e-5; tr = 1e-5;                    % pose-error tolerances used for scoring
    if isfield(opts,'tol_pos'), tp = opts.tol_pos; end
    if isfield(opts,'tol_rot'), tr = opts.tol_rot; end
    for k = 1:N
        T_d = traj_Ts(:,:,k);

        fk_chain('ikb_reset'); jac_geometric('ikb_reset');
        t0 = tic;
        [q_sol, info] = solver_fn(robot, T_d, q, opts);
        times(k) = toc(t0);
        fk_calls(k)  = fk_chain('ikb_count');
        jac_calls(k) = jac_geometric('ikb_count');

        q_sol        = q_sol(:);
        Q(:,k)       = q_sol;
        if isfield(info, 'ikb_fk_base')            % set by solve_polished (Phase 2)
            fk_base(k)  = info.ikb_fk_base;
            jac_base(k) = info.ikb_jac_base;
        else
            fk_base(k)  = fk_calls(k);
            jac_base(k) = jac_calls(k);
        end
        budget_exceeded(k) = fk_base(k) > budget;
        not_applicable(k)  = isfield(info, 'not_applicable') && isequal(info.not_applicable, true);

        % Solver-agnostic scoring: recompute the pose error from FK and
        % require BOTH position and orientation within tolerance. The
        % solver's self-reported converged flag is not used, since some
        % define convergence on position only and would score cells with
        % up to ~pi rad orientation error as "converged".
        T_act         = ikb_fk_uncounted(robot, q_sol);
        [~, ep, er]   = pose_error(T_d, T_act);
        errs_pos(k)   = norm(ep);
        errs_rot(k)   = norm(er);
        converged(k)  = ~not_applicable(k) && (errs_pos(k) < tp) && (errs_rot(k) < tr);
        if isfield(info, 'iters'),    iters(k)      = info.iters;    end
        if isfield(info, 'time'),     times_self(k) = info.time;     end
        if isfield(info, 'fk_calls'), fk_self(k)    = info.fk_calls; end
        info_cell{k}  = info;

        J = robot.jac(q_sol);
        manip(k) = manipulability(J);
        % Normalised distance to nearest joint limit (1 = at home, 0 = at limit).
        rng = robot.q_max - robot.q_min;
        d  = min(min(q_sol - robot.q_min, robot.q_max - q_sol) ./ max(rng, eps));
        lim_dist(k) = max(0, d);

        % Warm-start next step.
        q = q_sol;
    end

    results.Q               = Q;
    results.errs_pos        = errs_pos;
    results.errs_rot        = errs_rot;
    results.iters           = iters;
    results.times           = times;
    results.fk_calls        = fk_calls;
    results.jac_calls       = jac_calls;
    results.fk_base         = fk_base;
    results.jac_base        = jac_base;
    results.budget_exceeded = budget_exceeded;
    results.not_applicable  = not_applicable;
    results.times_self      = times_self;
    results.fk_self         = fk_self;
    results.converged       = converged;
    results.manip           = manip;
    results.lim_dist        = lim_dist;
    results.info            = info_cell;
    results.solver          = '';
    if isfield(info_cell{1}, 'solver'), results.solver = info_cell{1}.solver; end
    results.robot           = robot.name;
end
