function [J, T_ee] = jac_geometric(q, dh, tool)
% JAC_GEOMETRIC  Analytical geometric Jacobian for a revolute serial chain.
%
%   [J, T_ee] = JAC_GEOMETRIC(q, dh, tool) returns the 6xn body-fixed-frame
%   spatial Jacobian relating joint velocities to end-effector spatial
%   velocity:
%        [v; omega] = J * qdot
%   The top 3 rows are the linear Jacobian, the bottom 3 are the angular
%   Jacobian (both expressed in the base frame).
%
%   For revolute joint k with axis z_{k-1} (frame k-1's z-axis in base
%   frame) and origin o_{k-1}:
%        J_v_k     = z_{k-1} cross (o_n - o_{k-1})
%        J_omega_k = z_{k-1}
%
%   This is O(n) and exact (no finite differencing).
%
%   Evaluation counter (benchmark harness):
%     n = JAC_GEOMETRIC('ikb_count')   Jacobian calls since the last reset
%     JAC_GEOMETRIC('ikb_reset')       set the count to zero
%   The chain is built with FK_CHAIN_KERNEL, so a Jacobian call is not also
%   counted as an FK call.

    persistent n_calls
    if isempty(n_calls), n_calls = 0; end
    if ischar(q)
        switch q
            case 'ikb_count', J = n_calls;
            case 'ikb_reset', n_calls = 0; J = 0;
            otherwise, error('jac_geometric:command', 'Unknown command "%s".', q);
        end
        return
    end
    n_calls = n_calls + 1;

    if nargin < 3 || isempty(tool)
        tool = eye(4);
    end
    n = numel(q);
    [T_ee, Ts] = fk_chain_kernel(q, dh, tool);
    o_n = T_ee(1:3,4);

    J = zeros(6,n);
    % Frame 0 (base) z-axis and origin
    z_prev = [0;0;1];
    o_prev = [0;0;0];
    for k = 1:n
        J(1:3,k) = cross(z_prev, o_n - o_prev);
        J(4:6,k) = z_prev;
        z_prev = Ts(1:3,3,k);
        o_prev = Ts(1:3,4,k);
    end
end
