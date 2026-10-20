function [q, polish_iters, polish_fk] = dls_polish(robot, T_d, q_init, opts, max_polish_iters)
% DLS_POLISH  Local refinement step for metaheuristic / heuristic solvers.
%
%   [q, iters, fk] = dls_polish(robot, T_d, q_init, opts, max_polish_iters)
%
%   Runs damped least-squares (right-pseudoinverse form) starting from
%   q_init, returning as soon as ||e_pos|| < opts.tol_pos and
%   ||e_rot|| < opts.tol_rot, or after max_polish_iters iterations.
%
%   This implements the standard "metaheuristic + gradient polish" hybrid
%   pattern from the IK literature (e.g., Abdor-Sierra et al. 2022,
%   Hartmann et al. 2018): population-based exploration produces a coarse
%   solution, then a few damped-Newton iterations polish to the strict
%   precision required by trajectory tracking. Without this step, pure
%   metaheuristic methods at modest budgets (~2k-10k fevals) cannot reach
%   sub-millimetre tolerances on 6D pose targets.
%
%   The polish does NOT alter q if it is already within tolerance — q_init
%   is returned with zero polish work.

    if nargin < 5 || isempty(max_polish_iters)
        max_polish_iters = 50;
    end
    q = q_init(:);
    polish_iters = 0;
    polish_fk    = 0;
    % Polish damping has its own option name so it cannot collide with a
    % solver's own lambda (ABC and BODECS read opts.lambda for paper values).
    dls_lambda = 1e-2;
    if isfield(opts, 'dls_lambda') && ~isempty(opts.dls_lambda), dls_lambda = opts.dls_lambda; end
    lambda2 = dls_lambda^2;

    for k = 1:max_polish_iters
        [J, T] = robot.jac(q);
        polish_fk = polish_fk + 1;
        [e, e_pos, e_rot] = pose_error(T_d, T);
        if norm(e_pos) < opts.tol_pos && norm(e_rot) < opts.tol_rot
            return
        end
        A = J * J' + lambda2 * eye(6);
        q = q + J' * (A \ e);
        q = min(max(q, robot.q_min), robot.q_max);  % joint-limit projection (unconditional)
        polish_iters = k;
    end
end
