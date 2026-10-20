function ok = test_harness()
% TEST_HARNESS  Checks of the benchmark harness measurement (bench_run, solve_polished,
% ikb_counter, ikb_budget_left, default_opts). Uses small fake solvers whose FK / Jacobian
% use and timing are known exactly, then cross-checks real solvers against an independent
% counter. Prints PASS/FAIL per check and returns true only if all pass.

    here = fileparts(mfilename('fullpath'));
    code = fileparts(here);
    addpath(fullfile(code,'solvers'), fullfile(code,'utils'), fullfile(code,'robots'), ...
            fullfile(code,'trajectories'), fullfile(code,'bench'));

    rb   = UR5_robot();
    Ts   = make_default_traj(rb, 'rectangle', 100);
    Ts3  = Ts(:,:,1:3);
    opts = default_opts('max_iter',200,'tol_pos',1e-5,'tol_rot',1e-5,'budget',2000);
    results = {};

    % 1. Exact counting; the harness's own scoring calls are not charged to the solver.
    r = bench_run(rb, @(R,T,q,o) fake_calls(R,T,q,o,7,3), Ts3, rb.q_home, opts);
    results{end+1} = check('counts FK and Jacobian calls exactly (7 / 3 per waypoint)', ...
        all(r.fk_calls == 7) && all(r.jac_calls == 3) && all(r.fk_base == 7));

    % 2. Time is measured by the harness, not taken from the solver.
    r = bench_run(rb, @fake_slow, Ts3, rb.q_home, opts);
    results{end+1} = check('times are harness wall clock (>= 20 ms) while the solver reports 0', ...
        all(r.times >= 0.020) && all(r.times_self == 0));

    % 3. Phase 2: base and polish counted separately, polish time included.
    pol = @(R,T,q,o) solve_polished(@(R2,T2,q2,o2) fake_calls(R2,T2,q2,o2,0,0), R, T, q, o);
    r = bench_run(rb, pol, Ts(:,:,2:3), rb.q_home, opts);
    results{end+1} = check('Phase 2: base counts 0, polish Jacobian calls counted, wrapper FK not counted', ...
        all(r.fk_base == 0) && all(r.jac_calls > 0) && all(r.fk_calls == 0));

    % 4. Budget flag.
    r1 = bench_run(rb, @(R,T,q,o) fake_calls(R,T,q,o,2001,0), Ts3, rb.q_home, opts);
    r2 = bench_run(rb, @(R,T,q,o) fake_calls(R,T,q,o,2000,0), Ts3, rb.q_home, opts);
    results{end+1} = check('budget_exceeded is set at 2001 FK calls and not at 2000', ...
        all(r1.budget_exceeded) && ~any(r2.budget_exceeded));

    % 5. A solver can stay inside the budget with ikb_budget_left.
    r = bench_run(rb, @fake_budgeted, Ts3, rb.q_home, opts);
    results{end+1} = check('a solver sizing generations with ikb_budget_left uses <= 2000 FK calls', ...
        all(r.fk_base <= 2000) && all(r.fk_base >= 1990) && ~any(r.budget_exceeded));

    % 6. Not-applicable solvers are flagged, never scored as solved, and not polished.
    Tq0 = rb.fk(rb.q_home); Tna = repmat(Tq0, 1, 1, 3);       % targets q0 already reaches
    r = bench_run(rb, @fake_na, Tna, rb.q_home, opts);
    rp = bench_run(rb, @(R,T,q,o) solve_polished(@fake_na, R, T, q, o), Ts3, rb.q_home, opts);
    results{end+1} = check('not_applicable flagged, not counted as converged, polish skipped', ...
        all(r.not_applicable) && ~any(r.converged) && all(rp.jac_calls == 0));

    % 7. The harness options carry no population size and no shared lambda.
    results{end+1} = check('default_opts has no pop_size and no lambda; dls_lambda = 1e-2', ...
        ~isfield(opts,'pop_size') && ~isfield(opts,'lambda') && opts.dls_lambda == 1e-2);

    % 8. Real solvers: harness count equals an independent global counter on the same call.
    for s = {'solve_HS','solve_PSO','solve_PINV','solve_DLS'}
        f = str2func(s{1});
        rng(7); r = bench_run(rb, f, Ts(:,:,1), rb.q_home, opts);
        rng(7); n_ind = independent_count(rb, f, Ts(:,:,1), rb.q_home, opts);
        results{end+1} = check(sprintf('%s: harness FK count %d equals independent count %d', ...
            s{1}, r.fk_calls, n_ind), r.fk_calls == n_ind); %#ok<AGROW>
    end

    % 9. The counted fk_chain and the uncounted kernel compute bit-identical poses.
    same = true;
    robs = {UR5_robot(), Panda_robot(), Synthetic13_robot()};
    for i = 1:numel(robs)
        R = robs{i};
        for j = 1:50
            q = R.q_min + rand(R.n,1).*(R.q_max - R.q_min);
            same = same && isequal(fk_chain(q, R.dh, R.tool), fk_chain_kernel(q, R.dh, R.tool));
        end
    end
    results{end+1} = check('fk_chain and fk_chain_kernel give bit-identical poses (150 random q)', same);

    ok = all(cellfun(@(x) x, results));
    fprintf('\n%d of %d checks passed\n', nnz(cellfun(@(x) x, results)), numel(results));
end

% ---------------------------------------------------------------- helpers
function passed = check(name, cond)
    passed = logical(cond);
    if passed, tag = 'PASS'; else, tag = 'FAIL'; end
    fprintf('[%s] %s\n', tag, name);
end

function [q, info] = fake_calls(robot, ~, q0, ~, n_fk, n_jac)
    for i = 1:n_fk,  robot.fk(q0);  end
    for i = 1:n_jac, robot.jac(q0); end
    q = q0; info = struct('iters', 0, 'time', 0, 'solver', 'fake_calls');
end

function [q, info] = fake_slow(~, ~, q0, ~)
    pause(0.02);
    q = q0; info = struct('iters', 0, 'time', 0, 'solver', 'fake_slow');
end

function [q, info] = fake_budgeted(robot, ~, q0, opts)
    P = 10;                                   % population of 10, one FK per member
    while ikb_budget_left(opts) >= P
        for i = 1:P, robot.fk(q0); end
    end
    q = q0; info = struct('iters', 0, 'time', 0, 'solver', 'fake_budgeted');
end

function [q, info] = fake_na(~, ~, q0, ~)
    q = q0; info = struct('iters', 0, 'time', 0, 'solver', 'fake_na', 'not_applicable', true);
end

function n = independent_count(rb, f, T_d, q0, opts)
% Count FK calls with a separate mechanism (a global counter).
% The solver is run under the same conditions bench_run creates (counting on, counter reset), so
% budget-sized solvers (ikb_budget_left, ikb_generations) take the same path as in the harness.
    global IKB_TEST_FK %#ok<GVMIS>
    IKB_TEST_FK = 0;
    rbc = rb;
    fk = rb.fk;
    rbc.fk = @(q) indep_fk(fk, q);
    opts.ikb_counting = true;
    fk_chain('ikb_reset'); jac_geometric('ikb_reset');
    f(rbc, T_d, q0, opts);
    n = IKB_TEST_FK;
end

function varargout = indep_fk(fk, q)
    global IKB_TEST_FK %#ok<GVMIS>
    IKB_TEST_FK = IKB_TEST_FK + 1;
    varargout = cell(1, max(nargout, 1));
    [varargout{:}] = fk(q);
end
