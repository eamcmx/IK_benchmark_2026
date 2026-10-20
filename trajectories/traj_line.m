function Ts = traj_line(T_start, T_end, N)
% TRAJ_LINE  Cartesian straight-line trajectory between two poses.
%
%   Ts(:,:,k) is the homogeneous transform at parameter s in [0,1] sampled
%   uniformly over N steps. Position interpolated linearly; orientation
%   interpolated via SLERP on the equivalent rotation.
%
%   Inputs:
%     T_start, T_end : 4x4 homogeneous transforms.
%     N              : number of waypoints (>= 2).

    Ts = zeros(4,4,N);
    R0 = T_start(1:3,1:3); R1 = T_end(1:3,1:3);
    p0 = T_start(1:3,4);   p1 = T_end(1:3,4);

    % SLERP setup via axis-angle of R1 * R0'.
    R_rel = R1 * R0';
    c = max(-1, min(1, (trace(R_rel) - 1) / 2));
    theta = acos(c);
    if theta < 1e-9
        axis = [0;0;1]; theta = 0;
    else
        axis = (1/(2*sin(theta))) * [R_rel(3,2) - R_rel(2,3);
                                     R_rel(1,3) - R_rel(3,1);
                                     R_rel(2,1) - R_rel(1,2)];
    end

    for k = 1:N
        s = (k-1) / (N-1);
        p = (1 - s) * p0 + s * p1;
        ang = s * theta;
        R_inc = axisangle_to_R(axis, ang);
        R = R_inc * R0;
        Ts(:,:,k) = [R, p; 0 0 0 1];
    end
end

function R = axisangle_to_R(k, ang)
    k = k / max(norm(k), eps);
    K = [0,-k(3),k(2); k(3),0,-k(1); -k(2),k(1),0];
    R = eye(3) + sin(ang)*K + (1 - cos(ang))*(K*K);
end
