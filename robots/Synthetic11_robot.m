function r = Synthetic11_robot()
% SYNTHETIC11_ROBOT  11-DOF hyper-redundant synthetic manipulator.
%
%   Same regular construction as Synthetic9 (see Synthetic9_robot.m) with
%   one extra perpendicular-axis revolute pair (a 0.333 m link + a
%   re-orienting joint) inserted before the end segment.
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
%    11     0         0     0   0.20
%
%   11 revolute DOF against a 6D pose task -> 5-dimensional self-motion
%   manifold.

    r.name = 'Synthetic11';
    r.n    = 11;

    %               1     2     3     4      5     6      7     8      9     10     11
    r.dh.a     = [0,    0,    0,    0,     0,    0,     0,    0,     0,    0,     0    ];
    r.dh.alpha = [pi/2, -pi/2,pi/2, -pi/2, pi/2, -pi/2, pi/2, -pi/2, pi/2, -pi/2, 0    ];
    r.dh.d     = [0.333,0,    0.333,0,     0.333,0,     0.333,0,     0.333,0,     0.20 ];
    r.dh.theta_offset = [0, -pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    r.tool = eye(4);

    r.q_min = -pi * ones(11,1);
    r.q_max =  pi * ones(11,1);
    % Bent, mid-workspace, max-manipulability ready pose (reach 0.98 m) so the
    % smooth trajectories are well-posed (q=0 would sit near the singular shell).
    r.q_home = [0.5439; -0.0191; -0.707; 0.5576; -0.6847; 1.0475; 0.0252; 0.4842; -0.3109; 1.1388; -0.1617];

    r.fk  = @(q) fk_chain(q, r.dh, r.tool);
    r.jac = @(q) jac_geometric(q, r.dh, r.tool);
end
