function [e, e_pos, e_rot] = pose_error(T_d, T)
% POSE_ERROR  6-vector pose error e = [pos_err; rot_err] from current pose T
% to desired pose T_d, both 4x4 homogeneous transforms. Both errors expressed
% in the base frame.
%
%   pos_err = T_d(1:3,4) - T(1:3,4)
%   rot_err = axis-angle vector of (T_d(1:3,1:3) * T(1:3,1:3)') — i.e., the
%            rotation that takes the current EE orientation to the desired
%            orientation, written as a 3-vector in axis*angle form.
%
%   e is the 6-vector [pos; rot]. This is the standard task-space error used
%   by Pseudoinverse / DLS / Jacobian-Transpose / Gauss-Newton.
%
%   Robust to angle = 0 (returns zero rotation error) and angle = pi
%   (uses a stable extraction from R rather than the trace formula).

    R_d = T_d(1:3,1:3);
    R   = T(1:3,1:3);
    p_d = T_d(1:3,4);
    p   = T(1:3,4);

    e_pos = p_d - p;
    Re = R_d * R';
    e_rot = rot2axisangle(Re);
    e = [e_pos; e_rot];
end

function w = rot2axisangle(R)
% Convert rotation matrix to axis*angle 3-vector.
    c = (trace(R) - 1) / 2;
    c = min(1, max(-1, c));
    theta = acos(c);
    if theta < 1e-9
        w = [0;0;0];
        return;
    end
    if abs(pi - theta) < 1e-6
        % Near-pi: use diagonal extraction (stable)
        d = diag(R);
        [~, k] = max(d);
        v = R(:,k);
        v(k) = v(k) + 1;
        v = v / norm(v);
        w = theta * v;
        return;
    end
    w = (theta / (2*sin(theta))) * [R(3,2) - R(2,3);
                                   R(1,3) - R(3,1);
                                   R(2,1) - R(1,2)];
end
