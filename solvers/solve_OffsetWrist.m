function [q_best, info] = solve_OffsetWrist(robot, T_d, q0, opts)
% SOLVE_OFFSETWRIST  Numerical compensation loop of Zhou et al. (2022) for 6-DOF arms with offset
%                    wrists: Sec. 2.4 steps (2)-(5), pseudocode l.5-15, Eqs. 11-20, from the warm start.
%
%   Source:
%     Xing Zhou, Yaoqi Xian, Yuanhao Chen, Tongshu Chen, Lin Yang, Simin Chen and Jian Huang (2022).
%     An improved inverse kinematics solution for 6-DOF robot manipulators with offset wrists.
%     Robotica 40, pp. 2275-2294. doi:10.1017/S0263574721001648
%   No issue number is printed (p.1, p.20). Page numbers are PDF pages 1-20 (journal page = PDF
%   page + 2274).
%
%   Method (Sec. 2, pp.3-10; Fig. 2, p.3):
%     Step (1) (p.9; pseudocode l.3-4, p.10): translate the desired frame {6} by -d5 along its Y axis
%       (Eq. 10) and solve the simplified, Pieper-compliant structure (d5 = 0) in closed form (Eqs.
%       3-8). The resulting theta_s are "the approximate solutions of the IK" of the actual structure
%       (p.8). Not used here (see Seed).
%     Steps (2)-(5) (p.9; pseudocode l.5-15, p.10), iterated on theta_s:
%       T_s  = FK(theta_s) of the actual structure                                      (Eq. 11)
%       dP   = T_d(1:3,4) - T_s(1:3,4)                                                  (Eq. 13)
%       R(K, phi) = T_s(1:3,1:3)^-1 T_d(1:3,1:3) = [r_ij]                               (Eq. 14)
%       phi  = arccos((r11 + r22 + r33 - 1)/2)                                          (Eq. 15)
%       K    = 1/(2 sin(phi)) [r32 - r23; r13 - r31; r21 - r12]                         (Eq. 16)
%       dAng = T_s(1:3,1:3) K phi                                                       (Eq. 17)
%       dX   = [dP; dAng] "expressed in world coordinate" (p.9); e = ||dP||_2 + ||dAng||_2 (Eq. 12)
%       if e is within the tolerance the iteration stops (step (4)); otherwise
%       d theta = J^-1 dX, pseudocode l.11 "J \ dX"                                     (Eq. 18)
%       J: Jacobian of the actual structure in the base frame,
%          [J_Pi; J_Oi] = [Z_i x (P_n - P_i); Z_i], i = 1, 2, 3, ..., 6, n = 6           (Eq. 19)
%       theta = theta + d theta                                                         (Eq. 20)
%     repeated "until the synthetical error is within the pre-set criteria or the number of iteration
%     reaches the maximum setting number" (step (5), p.9). Pseudocode l.9 prints "while e > epsilon or
%     within the maximum number of iteration"; the loop continues while both hold (step (5)).
%
%   Seed (pseudocode l.3-4): replaced by the warm start q0, as the benchmark assumes a warm start for
%     every solver; the paper derives the seed for the HSR-BR606 only. Eqs. 3-10 are written for its
%     modified-DH table (Table I, p.5; values p.4): axes 1-2 meet at 90 deg, axes 2-3 are parallel
%     (a2), axes 3-4 meet at 90 deg (a3 = 0; offsets d2, d3, d4 along the axes), axes 4-5 and 5-6 meet
%     at 90 deg, and the two wrist intersections lie d5 apart along axis 5. The HSR-BR606 is "used as
%     an example" (p.1); an extension to other arms is announced but not given (p.19). The inversion
%     of Eq. 6 for theta1-3 is not printed (p.7), and the only branch rule is that the angles are
%     "selected within the" ranges of Table II (p.4, p.6).
%
%   Applicability:
%     Robot        n   Applicable  Reason
%     UR5          6   yes         loop of l.5-15. No seed: axes 2-3-4 are parallel (axes 3-4 lie
%                                  0.39225 m apart), so Eqs. 6-7 do not describe it.
%     IRB120       6   yes         loop of l.5-15. No seed: the forearm offset a3 = 0.070 m (axes 3-4
%                                  apart) has no term in Eq. 6; with d5 = 0 Eq. 10 is the identity.
%     Panda        7   no          the method is 6-DOF: Eq. 18 inverts J and Eq. 19 is written for
%     Synthetic9   9   no          n = 6 (p.9); a 6 x n Jacobian with n > 6 has no inverse. The paper
%     Synthetic11 11   no          gives no update for n > 6 (checked pages 3-10, 19).
%     Synthetic13 13   no
%     On these the solver sets info.not_applicable = true and returns q0 without any FK or Jacobian
%     call.
%
%   Parameters:
%     epsilon      1e-6 in the accuracy test (Sec. 3.1, p.10) and in motion planning (Sec. 3.2, p.14);
%                  1e-7, 1e-6 and 1e-5 are compared in Fig. 9 (p.12). It is the accuracy of the
%                  paper's example tasks and is replaced by the benchmark pose gate (see Stop); kept as
%                  params.eps_paper for the operator handle.
%     Maximum iteration number   "the maximum setting number" (p.9); no value is stated (checked
%                  pages 3, 9, 10, 12, 15, 17). The 12, 9 and 5 of p.12 and the 4 of p.15 are the
%                  largest counts observed in the experiments. Not used (see Budget).
%     Damping, step control, joint-limit handling inside the loop: none stated (checked pages 8-10).
%     Units        Table I is in mm, while the experiments state lengths and deviations in metres
%                  (p.14: a rectangle "with 0.5 m sides"; Fig. 14(b), p.15, "Deviation (m)"; p.15:
%                  deviations of "8.989x10^-7 and 8.452x10^-7 m" under e < 1e-6). e therefore adds
%                  metres and radians, the benchmark's units; no conversion. J^-1 dX does not depend
%                  on the unit.
%
%   Benchmark interface:
%     Budget: an iteration evaluates T_s (1 robot.fk call) and the stopping test; if the test fails
%       and the budget can still evaluate the next iterate, it calls robot.jac once (not budgeted,
%       counted by the harness) and applies Eqs. 18 and 20. An iteration starts only if
%       ikb_budget_left(opts) >= 1. Per waypoint: FK = updates + 1 <= 2000, Jacobian calls = updates
%       <= 1999. Outside the harness the solver counts its own FK calls against opts.budget (2000
%       when the field is absent). opts.max_iter is not used and there is no iteration cap. No
%       quantity varies over the iteration count, so ikb_generations is not used.
%     Stop: the benchmark pose gate, ||dP|| < opts.tol_pos (m) and ||dAng|| < opts.tol_rot (rad), the
%       test of bench_run (1e-5 each when a field is absent), or the budget. info.converged is the
%       gate test. Through the operator handle (solve_OffsetWrist('ops'), params.eps_stop = 1e-6) the
%       paper's test e <= epsilon ends the iteration instead; the solver reads no option for it.
%     Warm start: theta_s starts at q0 (the previous waypoint's solution), clamped to the limits.
%     Units: metres and radians (see Parameters).
%     Joint limits: the paper takes the joint ranges as an input (pseudocode l.2) and selects the
%       closed-form angle group within them (p.4), but states no handling inside the loop (checked
%       pages 8-10); every iterate is clamped to [q_min, q_max].
%     Singular J: not treated in the paper (checked pages 2, 8-10). At an iterate where rcond(J) < eps
%       (singular to working precision, the test of MATLAB's J \ dX) the step is pinv(J) dX, the
%       Moore-Penrose solution; such steps are counted in info.pinv_steps.
%     Output: the iterate that meets the stopping test; if the budget ends first, the evaluated
%       iterate with the lowest e (Eq. 12). info.fitness is that e; info.err_pos (m) and info.err_rot
%       (rad) come from its pose, as in bench_run. solve_OffsetWrist('ops') returns handles to the
%       operators for unit tests.
%
%   Notes on the printed equations (not used by the loop; checked against Table I):
%     - Eq. 6, row 2, prints (d4 s23 + a2 c2) s1 + (d2 + d3) c1; Table I gives
%       (d4 s23 + a2 c2) s1 - (d2 + d3) c1.
%     - Eq. 7, element (1,2), prints -c6 s4 - c5 s6; Table I gives -c6 s4 - c4 c5 s6. Eq. 8 uses only
%       elements (1,3), (2,1), (2,2), (2,3) and (3,3), which agree with Table I.
%     - Eq. 10 translates along Y of the desired frame {6}; the result equals the simplified
%       structure's pose only at theta6 = 0, so the seed is approximate by design (p.8).
%
%   Deviations from the paper:
%     - Seed (pseudocode l.3-4, Sec. 2.4 step (1)) replaced by the warm start q0 (see Seed). The
%       paper's property that no initial estimate is needed (p.1, p.17) does not hold: the result and
%       the iteration count depend on q0.
%     - Stopping test: the benchmark pose gate instead of epsilon = 1e-6 (p.10, p.14).
%     - No iteration cap: the loop ends at the gate or at the FK budget (cap value not stated).
%     - Every iterate clamped to the joint limits (not stated).
%     - pinv(J) dX at an iterate where J is singular to working precision (not stated).
%     - Eq. 15: the arccos argument is clipped to [-1, 1] (rounding). Eq. 16 where it is 0/0: at
%       phi = 0, or where [r32 - r23; r13 - r31; r21 - r12] = 0, K phi is its limit
%       (1/2)[r32 - r23; r13 - r31; r21 - r12], which is 0 only when that vector is 0 (no dead band);
%       within 1e-6 rad of phi = pi (the threshold of the harness's pose_error for the same
%       singularity) K is the unit axis of R, the normalised column of R + I with the largest diagonal
%       entry, signed to agree with Eq. 16.
%     - If the budget ends first, the evaluated iterate with the lowest e is returned (the paper
%       returns the current joint variables: Fig. 2, p.3; pseudocode l.15).
%     - FK and Jacobian from the standard-DH chain (robot.fk, robot.jac) instead of the modified DH of
%       Eqs. 1-2: a modelling convention. The columns of Eq. 19 are unchanged, since modified-DH frame
%       i and standard-DH frame i-1 both lie on the axis of joint i.
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
        q_best = struct('params', ow_params(), 'core', @ow_core, 'step', @ow_step, ...
                        'error', @ow_error);
        info = [];
        return
    end
    [q_best, info] = ow_core(robot, T_d, q0, opts, ow_params());
end

% ==================================================================== main loop
function [q_best, info] = ow_core(robot, T_d, q0, opts, P)
% Pseudocode l.5-15 from theta_s = q0 until the stopping test is met or the FK budget ends.
    t_start = tic;
    info = struct();
    info.solver = 'Offset-Wrist';
    if robot.n ~= 6                                                   % 6-DOF method (Eqs. 18-19)
        q_best = q0;
        info.iters          = 0;
        info.time           = toc(t_start);
        info.converged      = false;
        info.fitness        = NaN;
        info.not_applicable = true;
        info.fk_calls       = 0;
        info.jac_calls      = 0;
        info.reason         = sprintf('%d joints: Eq. 18 inverts J and Eq. 19 is written for n = 6 (p.9)', ...
                                      robot.n);
        return
    end

    lo = robot.q_min(:);
    hi = robot.q_max(:);
    q  = min(max(q0(:), lo), hi);                                     % theta_s = q0 (warm start)

    [tp, tr] = pose_gate(opts);
    if isempty(P.eps_stop), hit_name = 'gate'; else, hit_name = 'epsilon'; end

    q_best = q;  f_best = NaN;  T_best = NaN(4);
    fk = 0;  jc = 0;  it = 0;  npv = 0;  conv = false;  stop = 'budget';
    hist = zeros(1, 0);
    while budget_left(opts, fk) >= 1
        Ts = robot.fk(q);                                             % l.5, Eq. 11
        fk = fk + 1;
        [dX, e] = ow_error(T_d, Ts);                                  % l.6-8, Eqs. 12-17
        hist(fk) = e;
        if fk == 1 || e < f_best
            f_best = e;  q_best = q;  T_best = Ts;
        end
        if isempty(P.eps_stop)
            hit = norm(dX(1:3)) < tp && norm(dX(4:6)) < tr;           % benchmark pose gate
        else
            hit = e <= P.eps_stop;                                    % l.9: while e > epsilon
        end
        if hit                                                        % return the iterate that meets it
            f_best = e;  q_best = q;  T_best = Ts;
            conv = true;  stop = hit_name;
            break
        end
        if budget_left(opts, fk) < 1, break; end                      % next iterate not evaluable
        J  = robot.jac(q);                                            % l.10, Eq. 19
        jc = jc + 1;
        [dq, pv] = ow_step(J, dX);                                    % l.11, Eq. 18
        q  = min(max(q + dq, lo), hi);                                % l.12, Eq. 20; clamp (not stated)
        it = it + 1;
        npv = npv + pv;
    end

    ep = NaN;  er = NaN;
    if fk > 0
        [~, e_p, e_r] = pose_error(T_d, T_best);
        ep = norm(e_p);
        er = norm(e_r);
    end

    info.method         = 'compensation loop of Zhou et al. (2022), pseudocode l.5-15, from the warm start';
    info.iters          = it;                                         % updates (Eqs. 18 and 20 applied)
    info.time           = toc(t_start);
    info.converged      = conv;                                       % stopping test met
    info.fitness        = f_best;                                     % e of Eq. 12, m + rad
    info.not_applicable = false;
    info.err_pos        = ep;                                         % m
    info.err_rot        = er;                                         % rad
    info.fk_calls       = fk;
    info.jac_calls      = jc;
    info.pinv_steps     = npv;                                        % updates with rcond(J) < eps
    info.stop           = stop;                                       % 'gate', 'budget', 'epsilon'
    info.gate           = [tp, tr];                                   % tol_pos (m), tol_rot (rad)
    info.history        = hist;                                       % e of every evaluated iterate
    info.params         = P;
end

% ==================================================================== operators
function P = ow_params()
% epsilon = 1e-6 (Sec. 3.1, p.10; Sec. 3.2, p.14), kept for the operator handle. eps_stop empty: stop at
% the benchmark pose gate; a number (operator handle use only): stop at e <= eps_stop (pseudocode l.9).
    P = struct('eps_paper', 1e-6, 'eps_stop', []);
end

function [dX, e, dP, dAng, phi, K] = ow_error(T_d, T_s)
% Eqs. 12-17: dP, the equivalent angle-axis (K, phi) of T_s(1:3,1:3)^-1 T_d(1:3,1:3), dAng = R_s K phi
% in the base frame, dX = [dP; dAng] and e = ||dP|| + ||dAng||. K = NaN where Eq. 16 is 0/0 and only
% the product K phi is defined (see Deviations).
    Rs  = T_s(1:3,1:3);
    dP  = T_d(1:3,4) - T_s(1:3,4);                                    % Eq. 13
    R   = Rs.' * T_d(1:3,1:3);                                        % Eq. 14 (inverse of a rotation)
    phi = acos(min(1, max(-1, (R(1,1) + R(2,2) + R(3,3) - 1) / 2)));  % Eq. 15
    v   = [R(3,2) - R(2,3); R(1,3) - R(3,1); R(2,1) - R(1,2)];
    if pi - phi < 1e-6                                                % Eq. 16 at phi = pi
        [~, k] = max(diag(R));
        K = R(:, k);
        K(k) = K(k) + 1;
        K = K / norm(K);
        if dot(K, v) < 0, K = -K; end
        dAng = Rs * K * phi;                                          % Eq. 17
    elseif phi == 0 || ~any(v)                                        % Eq. 16 at phi = 0: limit of K phi
        K = NaN(3, 1);
        dAng = Rs * (v / 2);
    else
        K = v / (2 * sin(phi));                                       % Eq. 16
        dAng = Rs * K * phi;                                          % Eq. 17
    end
    dX = [dP; dAng];
    e  = norm(dP) + norm(dAng);                                       % Eq. 12
end

function [dq, used_pinv] = ow_step(J, dX)
% Eq. 18, d theta = J^-1 dX, computed as J \ dX (pseudocode l.11). Where J is singular to working
% precision (rcond(J) < eps) the Moore-Penrose solution pinv(J) dX is used.
    if rcond(J) >= eps
        dq = J \ dX;
        used_pinv = false;
    else
        dq = pinv(J) * dX;
        used_pinv = true;
    end
end

function [tp, tr] = pose_gate(opts)
% The benchmark pose gate, as in bench_run: position < tol_pos (m) and rotation < tol_rot (rad).
    tp = 1e-5;  tr = 1e-5;
    if isfield(opts, 'tol_pos'), tp = opts.tol_pos; end
    if isfield(opts, 'tol_rot'), tr = opts.tol_rot; end
end

function n = budget_left(opts, fk)
% FK calls still allowed on this waypoint: ikb_budget_left inside the harness, opts.budget (2000 when
% absent) minus the solver's own count outside it.
    n = ikb_budget_left(opts);
    if isinf(n)
        b = 2000;
        if isfield(opts, 'budget') && ~isempty(opts.budget), b = opts.budget; end
        n = b - fk;
    end
end
