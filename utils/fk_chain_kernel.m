function [T_ee, Ts] = fk_chain_kernel(q, dh, tool)
% FK_CHAIN_KERNEL  Forward kinematics without the evaluation counter.
%
%   Identical computation to FK_CHAIN. Used where an FK evaluation must not be
%   charged to a solver: inside JAC_GEOMETRIC (a Jacobian is counted as one
%   Jacobian call, not also as an FK call) and in the harness's own scoring.

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
