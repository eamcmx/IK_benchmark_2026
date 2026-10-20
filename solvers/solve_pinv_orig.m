function [q, info] = solve_pinv_orig(robot, T_d, q0, opts)
% SOLVE_PINV_ORIG  Moore-Penrose pseudoinverse IK on the 6-vector pose error (baseline PINV-orig).
%
%   Source (method attribution):
%     Daniel E. Whitney (1969). Resolved motion rate control of manipulators and human
%     prostheses. IEEE Transactions on Man-Machine Systems, 10(2), 47-53.
%     DOI: 10.1109/TMMS.1969.299896
%   Citation and DOI as in references.bib (entry Whitney1969). The PDF is not on disk, so neither
%   the citation nor the method is checked against the article. Colomé and Torras (2015) cite the
%   same article as their ref. [21] (IEEE Trans. Man-Mach. Syst., vol. MMS-10, no. 2, pp. 47-53,
%   Jun. 1969). The file is the benchmark's own baseline, the textbook resolved-rate iteration.
%
%   Method (baseline rules):
%     q_{k+1} = q_k + alpha0 * J^+(q_k) * e(q_k), clamped to [q_min, q_max] after every update,
%     with e = pose_error(T_d, T(q_k)) (position in m, rotation vector of R_d R^T in rad), J the
%     geometric Jacobian and J^+ = pinv(J), MATLAB's SVD-based Moore-Penrose pseudoinverse. For
%     n > 6 joints at full rank, J^+ e is the minimum-norm joint update.
%     Options: opts.max_iter (number of passes), opts.tol_pos and opts.tol_rot (stop when the
%     position error < tol_pos and the rotation error < tol_rot), opts.alpha0 (step scale, 1.0 in
%     default_opts). The FK budget rule of the literature solvers does not apply to this baseline.
%
%   Evaluations: each pass calls robot.jac once, which returns J and the end-effector pose from
%     one chain evaluation; the pose gives e for the stopping test, so on the pass that meets the
%     tolerances the Jacobian is computed with the pose and not used (skipping it would need a
%     separate pose evaluation). After the loop one robot.fk call evaluates the returned q. Per
%     waypoint: 1 FK call and one Jacobian call per pass (updates + 1 when the tolerances are met,
%     opts.max_iter otherwise). info.fk_calls and info.jac_calls count the robot.fk and robot.jac
%     calls, as the harness does; info.iters is the number of passes.
%   Joint limits: every update is clamped to [q_min, q_max] (benchmark policy), and the clamped
%     iterate is returned as it is, so the returned joints are inside the limits whenever q0 is
%     (in the benchmark q0 is q_home or the previous waypoint's returned q).

    if nargin < 4, opts = default_opts(); end
    q = q0(:);
    fk_calls = 0; jac_calls = 0;
    t0 = tic;
    converged = false;
    final_iter = opts.max_iter;
    history = struct('err', []);

    for k = 1:opts.max_iter
        [J, T] = robot.jac(q); jac_calls = jac_calls + 1;           % J and the pose of q
        [e, e_pos, e_rot] = pose_error(T_d, T);

        history.err(end+1) = norm(e);

        if norm(e_pos) < opts.tol_pos && norm(e_rot) < opts.tol_rot
            converged = true; final_iter = k; break;
        end

        dq = pinv(J) * e;
        q = q + opts.alpha0 * dq;

        % Joint-limit projection after every update (benchmark policy).
        q = min(max(q, robot.q_min), robot.q_max);
    end

    T_final = robot.fk(q); fk_calls = fk_calls + 1;
    [~, e_pos, e_rot] = pose_error(T_d, T_final);

    info.solver    = 'PINV-orig';
    info.iters     = final_iter;
    info.time      = toc(t0);
    info.fk_calls  = fk_calls;
    info.jac_calls = jac_calls;
    info.converged = converged;
    info.err_pos   = norm(e_pos);
    info.err_rot   = norm(e_rot);
    info.history   = history;
end
