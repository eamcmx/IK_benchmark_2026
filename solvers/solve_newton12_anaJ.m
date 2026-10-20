function [q, info] = solve_newton12_anaJ(robot, T_d, q0, opts)
% SOLVE_NEWTON12_ANAJ  Newton-Raphson IK on the 12-element homogeneous-matrix residual with an
%   analytical Jacobian (baseline Newton12-anaJ).
%
%   Source: none. The file is the benchmark's own baseline, not a published method. It keeps the
%     12-element residual r = vec(T(1:3,1:4) - T_d(1:3,1:4)) (column-major: the 9 rotation entries,
%     then the 3 position entries) and differentiates it analytically from the geometric Jacobian:
%       d p / dq_k        = J_v(:, k)
%       d R(:, j) / dq_k  = [omega_k]_x R(:, j)
%     which gives a 12 x n matrix J12. Each update is q <- q + pinv(J12) (-r), clamped to
%     [q_min, q_max].
%
%   Method (baseline rules):
%     Options: opts.max_iter (number of passes), opts.tol_pos and opts.tol_rot (stop when the
%     position error < tol_pos and the rotation error < tol_rot, from pose_error), and a second stop
%     when every residual entry is below 1e-9. The FK budget rule of the literature solvers does not
%     apply to this baseline.
%
%   Evaluations: each pass calls robot.jac once, which returns J and the end-effector pose from one
%     chain evaluation. After the loop one robot.fk call evaluates the returned q. info.fk_calls and
%     info.jac_calls count the robot.fk and robot.jac calls, as the harness does.
%   Joint limits: every update is clamped to [q_min, q_max] (benchmark policy), and the clamped
%     iterate is returned as it is, so the returned joints are inside the limits whenever q0 is
%     (in the benchmark q0 is q_home or the previous waypoint's returned q).

    if nargin < 4, opts = default_opts(); end
    n = robot.n;
    q = q0(:);
    fk_calls = 0; jac_calls = 0;
    t0 = tic;
    converged = false;
    final_iter = opts.max_iter;
    history = struct('err', [], 'q', []);

    for k = 1:opts.max_iter
        [J, T] = robot.jac(q); jac_calls = jac_calls + 1;
        r = matrix_residual(T, T_d);     % 12-vec
        if all(abs(r) < 1e-9)
            converged = true; final_iter = k; break;
        end

        % Analytic 12 x n Jacobian: rows 1..9 = vec(R) derivative, rows 10..12 = position derivative.
        Jv = J(1:3, :);             % 3 x n
        Jw = J(4:6, :);             % 3 x n
        R  = T(1:3,1:3);
        J12 = zeros(12, n);
        for kk = 1:n
            wk = Jw(:, kk);
            wk_hat = [0,-wk(3), wk(2); wk(3),0,-wk(1); -wk(2),wk(1),0];
            dR = wk_hat * R;        % 3 x 3 derivative of R w.r.t. q_kk
            dRdq_vec = dR(:);       % 9-vec, column-major
            J12(1:9,  kk) = dRdq_vec;
            J12(10:12,kk) = Jv(:, kk);
        end
        % rows 1-9 and 10-12 follow the column-major layout of matrix_residual

        ds = pinv(J12) * (-r);
        q = q + ds;
        q = min(max(q, robot.q_min), robot.q_max);  % joint-limit projection (unconditional)

        history.err(end+1) = norm(r);
        history.q(:,end+1) = q;

        [~, e_pos, e_rot] = pose_error(T_d, T);
        if norm(e_pos) < opts.tol_pos && norm(e_rot) < opts.tol_rot
            converged = true; final_iter = k; break;
        end
    end

    T_final = robot.fk(q); fk_calls = fk_calls + 1;
    [~, e_pos, e_rot] = pose_error(T_d, T_final);

    info.solver    = 'Newton12-anaJ';
    info.iters     = final_iter;
    info.time      = toc(t0);
    info.fk_calls  = fk_calls;
    info.jac_calls = jac_calls;
    info.converged = converged;
    info.err_pos   = norm(e_pos);
    info.err_rot   = norm(e_rot);
    info.history   = history;
end

function r = matrix_residual(T, T_d)
% 12-vector matching the column-major vec of (T(1:3,1:4) - T_d(1:3,1:4)).
    D = T(1:3,1:4) - T_d(1:3,1:4);
    r = D(:);
end
