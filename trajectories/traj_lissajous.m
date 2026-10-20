function Ts = traj_lissajous(T_centre, amplitudes, freqs, phases, N, rot_axis, rot_amp)
% TRAJ_LISSAJOUS  Continuous 3D Lissajous-knot trajectory with optional
%   coupled orientation tracking.
%
%   Position:
%     p(t) = p_centre + [A_x sin(f_x t + phi_x);
%                        A_y sin(f_y t + phi_y);
%                        A_z sin(f_z t + phi_z)]
%   for t uniformly sampled in [0, 2 pi]. With integer frequencies
%   (f_x, f_y, f_z) and an irrational-free phase set the curve closes
%   on itself; common ratios are 1:2:3 (a Lissajous knot of low
%   complexity) or 3:4:5 (richer self-intersection).
%
%   Orientation:
%     R(t) = Rotation about rot_axis by rot_amp * sin(t) applied to the
%     centre rotation R0. The orientation oscillates symmetrically along
%     the curve, returning to R0 at t = 0 and t = 2 pi.
%
%   Inputs:
%     T_centre   : 4x4 anchor pose (typically robot.fk(robot.q_home))
%     amplitudes : 3-vector [A_x, A_y, A_z] in metres
%     freqs      : 3-vector of integer frequencies [f_x, f_y, f_z]
%     phases     : 3-vector of phase offsets in radians
%     N          : number of waypoints
%     rot_axis   : 3-vector axis of orientation oscillation (default y)
%     rot_amp    : amplitude of the orientation oscillation (default deg2rad(20))

    if nargin < 6 || isempty(rot_axis), rot_axis = [0;1;0]; end
    if nargin < 7 || isempty(rot_amp),  rot_amp  = deg2rad(20); end

    R0 = T_centre(1:3,1:3);
    p0 = T_centre(1:3,4);
    Ts = zeros(4,4,N);
    for k = 1:N
        t = 2*pi * (k-1) / N;
        d = [amplitudes(1) * sin(freqs(1)*t + phases(1));
             amplitudes(2) * sin(freqs(2)*t + phases(2));
             amplitudes(3) * sin(freqs(3)*t + phases(3))];
        R = axang2R_local(rot_axis, rot_amp * sin(t)) * R0;
        Ts(:,:,k) = [R, p0 + d; 0 0 0 1];
    end
end

function R = axang2R_local(axis, ang)
    axis = axis / max(norm(axis), eps);
    K = [0,-axis(3),axis(2); axis(3),0,-axis(1); -axis(2),axis(1),0];
    R = eye(3) + sin(ang)*K + (1 - cos(ang))*(K*K);
end
