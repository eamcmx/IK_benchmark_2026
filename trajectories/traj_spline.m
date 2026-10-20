function Ts = traj_spline(waypoints, N)
% TRAJ_SPLINE  Cubic-spline interpolated 6D trajectory through waypoints.
%
%   waypoints : 4x4xK array of pose waypoints.
%   N         : number of output samples.
%
%   Position uses cubic spline; orientation uses successive SLERP between
%   adjacent waypoint rotations with arc-length parameterisation.

    K = size(waypoints, 3);
    s_w = linspace(0, 1, K);
    s_q = linspace(0, 1, N);

    P = squeeze(waypoints(1:3, 4, :));   % 3 x K
    Px = spline(s_w, P(1,:), s_q);
    Py = spline(s_w, P(2,:), s_q);
    Pz = spline(s_w, P(3,:), s_q);
    P_q = [Px; Py; Pz];

    Ts = zeros(4,4,N);
    for i = 1:N
        s = s_q(i);
        % Find segment.
        seg = max(1, min(K-1, sum(s >= s_w)));
        t  = (s - s_w(seg)) / (s_w(seg+1) - s_w(seg));
        Ra = waypoints(1:3,1:3,seg);
        Rb = waypoints(1:3,1:3,seg+1);
        R = slerp_R(Ra, Rb, t);
        Ts(:,:,i) = [R, P_q(:,i); 0 0 0 1];
    end
end

function R = slerp_R(R0, R1, t)
    Rrel = R1 * R0';
    c = max(-1, min(1, (trace(Rrel) - 1) / 2));
    th = acos(c);
    if th < 1e-9
        R = R0; return;
    end
    ax = (1/(2*sin(th))) * [Rrel(3,2)-Rrel(2,3); Rrel(1,3)-Rrel(3,1); Rrel(2,1)-Rrel(1,2)];
    K = [0 -ax(3) ax(2); ax(3) 0 -ax(1); -ax(2) ax(1) 0];
    Rinc = eye(3) + sin(t*th)*K + (1 - cos(t*th))*(K*K);
    R = Rinc * R0;
end
