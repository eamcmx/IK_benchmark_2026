function [q, info] = solve_polished(base_solver, robot, T_d, q0, opts)
% SOLVE_POLISHED  Wrap any base IK solver with a final DLS polish stage.
%
%   [q, info] = solve_polished(base_solver, robot, T_d, q0, opts)
%
%   Runs base_solver(robot, T_d, q0, opts) to produce a coarse solution
%   q_raw, then refines it with dls_polish for up to 50 iterations against
%   the same tolerance gate (opts.tol_pos, opts.tol_rot). This standardised
%   wrapper is used in the paper's Phase-2 ("apples-to-apples") benchmark
%   to give every literature-faithful metaheuristic / heuristic a fair
%   shot at the strict 1e-5 m / 1e-5 rad precision target that pure
%   population methods cannot reach unaided.
%
%   The wrapper does not modify the base solver internally; each base
%   solver's source-paper implementation is left untouched. The polish is
%   a strict external add-on, identical across every solver, so the
%   comparison is fair.
%
%   Fields added to info:
%       info.polish_iters : number of DLS polish iterations consumed
%       info.q_raw        : the coarse solution before polish
%       info.err_pos_raw  : position error before polish
%       info.err_rot_raw  : rotation error before polish
%
%   Fields recomputed after polish:
%       info.q, info.err_pos, info.err_rot, info.converged, info.fk_calls
%
%   Usage in a bench harness:
%       fn = @(rb, Td, q0, o) solve_polished(@solve_PSO, rb, Td, q0, o);
%       res = bench_run(robot, fn, Ts, robot.q_home, opts);

    [q_raw, info] = base_solver(robot, T_d, q0, opts);
    q_raw = q_raw(:);

    % Inside the benchmark harness, record the base solver's own FK/Jacobian
    % count before anything else evaluates the kinematics. The wrapper's own
    % diagnostic FK evaluations do not touch the counter, so they are not
    % charged to the solver; the polish's Jacobian calls are counted.
    if isfield(opts, 'ikb_counting') && isequal(opts.ikb_counting, true)
        info.ikb_fk_base  = fk_chain('ikb_count');
        info.ikb_jac_base = jac_geometric('ikb_count');
    end
    fk_diag = @(qq) ikb_fk_uncounted(robot, qq);

    % A solver that declared the robot outside its scope is not rescued by the polish.
    if isfield(info, 'not_applicable') && isequal(info.not_applicable, true)
        q = q_raw;
        info.polish_iters = 0;
        info.solver = [info.solver, '+polish'];
        return
    end

    % Record the base solver's state before the polish.
    T_raw = fk_diag(q_raw);
    [~, e_pos_raw, e_rot_raw] = pose_error(T_d, T_raw);
    info.q_raw       = q_raw;
    info.err_pos_raw = norm(e_pos_raw);
    info.err_rot_raw = norm(e_rot_raw);

    % If already within tolerance, no polish work needed.
    if info.err_pos_raw < opts.tol_pos && info.err_rot_raw < opts.tol_rot
        q = q_raw;
        info.polish_iters = 0;
    else
        [q, polish_iters, polish_fk] = dls_polish(robot, T_d, q_raw, opts, 50);
        info.polish_iters = polish_iters;
        if isfield(info, 'fk_calls')
            info.fk_calls = info.fk_calls + polish_fk;
        else
            info.fk_calls = polish_fk;
        end
    end

    % Recompute final pose error and convergence flag.
    T_final = fk_diag(q);
    [~, e_pos, e_rot] = pose_error(T_d, T_final);
    info.err_pos   = norm(e_pos);
    info.err_rot   = norm(e_rot);
    info.converged = (info.err_pos < opts.tol_pos && info.err_rot < opts.tol_rot);
    info.solver    = [info.solver, '+polish'];
end
