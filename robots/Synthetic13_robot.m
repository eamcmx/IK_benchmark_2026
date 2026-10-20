function r = Synthetic13_robot()
% SYNTHETIC13_ROBOT  13-DOF hyper-redundant synthetic manipulator.
%
%   Same regular construction as Synthetic9 / Synthetic11 with two extra
%   perpendicular-axis revolute pairs before the end segment.
%
%     k  theta_off  alpha   a   d
%     1     0       +90deg  0   0.333
%     2    -90deg     0     0   0
%     3     0       +90deg  0   0.333
%     4     0       -90deg  0   0
%     5     0       +90deg  0   0.333
%     6     0       -90deg  0   0
%     7     0       +90deg  0   0.333
%     8     0       -90deg  0   0
%     9     0       +90deg  0   0.333
%    10     0       -90deg  0   0
%    11     0       +90deg  0   0.333
%    12     0       -90deg  0   0
%    13     0         0     0   0.20
%
%   13 revolute DOF against a 6D pose task -> 7-dimensional self-motion
%   manifold.

    r.name = 'Synthetic13';
    r.n    = 13;

    %               1     2     3     4      5     6      7     8      9     10     11    12     13
    r.dh.a     = [0,    0,    0,    0,     0,    0,     0,    0,     0,    0,     0,    0,     0    ];
    r.dh.alpha = [pi/2, -pi/2,pi/2, -pi/2, pi/2, -pi/2, pi/2, -pi/2, pi/2, -pi/2, pi/2, -pi/2, 0    ];
    r.dh.d     = [0.333,0,    0.333,0,     0.333,0,     0.333,0,     0.333,0,     0.333,0,     0.20 ];
    r.dh.theta_offset = [0, -pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    r.tool = eye(4);

    r.q_min = -pi * ones(13,1);
    r.q_max =  pi * ones(13,1);
    % Bent, mid-workspace, max-manipulability ready pose (reach 1.23 m) so the
    % smooth trajectories are well-posed (q=0 would sit near the singular shell).
    r.q_home = [-0.0235; -0.4141; -0.6667; 0.3771; -0.2297; 0.5481; -0.1664; 0.8094; 1.101; 0.8732; -0.1776; 1.068; 0.0463];

    r.fk  = @(q) fk_chain(q, r.dh, r.tool);
    r.jac = @(q) jac_geometric(q, r.dh, r.tool);
end
