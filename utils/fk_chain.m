function [T_ee, Ts] = fk_chain(q, dh, tool)
% FK_CHAIN  Forward kinematics for a serial chain of revolute joints (standard DH).
%
%   [T_ee, Ts] = FK_CHAIN(q, dh) returns the 4x4 end-effector pose T_ee and
%   the cumulative transforms Ts(:,:,k) = T_0_k for k = 1..n.
%
%   [T_ee, Ts] = FK_CHAIN(q, dh, tool) post-multiplies T_ee by a fixed 4x4
%   tool transform (e.g., flange offset).
%
%   Inputs:
%     q    : n-vector of joint angles (rad). Joints assumed revolute.
%     dh   : struct with fields a(n), alpha(n), d(n), and optionally
%            theta_offset(n). Theta used in DH = q(k) + theta_offset(k).
%     tool : optional 4x4 tool transform (default eye(4)).
%
%   Evaluation counter (used by the benchmark harness to measure how many FK
%   evaluations a solver makes; one persistent increment per call, so the
%   measurement adds almost nothing to the cost of an evaluation):
%     n = FK_CHAIN('ikb_count')   number of calls since the last reset
%     FK_CHAIN('ikb_reset')       set the count to zero
%   The same computation without counting is FK_CHAIN_KERNEL (used by
%   JAC_GEOMETRIC and by the harness's own scoring).
%
%   Notes:
%     * Single FK evaluation per call: O(n) DH transforms, no redundant work.
%     * For prismatic joints add dh.type with 'P' / 'R' and adapt; this
%       benchmark uses revolute throughout so we keep it simple.

    persistent n_calls
    if isempty(n_calls), n_calls = 0; end
    if ischar(q)
        switch q
            case 'ikb_count', T_ee = n_calls;
            case 'ikb_reset', n_calls = 0; T_ee = 0;
            otherwise, error('fk_chain:command', 'Unknown command "%s".', q);
        end
        return
    end
    n_calls = n_calls + 1;

    % ---- same computation as fk_chain_kernel (kept inline: no extra call) ----
    if nargin < 3 || isempty(tool)
        tool = eye(4);
    end
    n = numel(q);
    if isfield(dh, 'theta_offset')
        toff = dh.theta_offset(:);
    else
        toff = zeros(n,1);
    end

    Ts = zeros(4,4,n);
    T  = eye(4);
    for k = 1:n
        Ak = dh_transform(dh.a(k), dh.alpha(k), dh.d(k), q(k) + toff(k));
        T  = T * Ak;
        Ts(:,:,k) = T;
    end
    T_ee = T * tool;
end
