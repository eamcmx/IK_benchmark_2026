function r = Synthetic9_robot()
% SYNTHETIC9_ROBOT  9-DOF truly-redundant synthetic manipulator.
%
%   A clean, regular redundant arm built from a repeating standard-DH unit,
%   so that canonical poses (e.g. a 45 deg elevation) can be obtained from a
%   single joint.
%
%   Construction (standard DH, T_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i),
%   theta_i = q_i + theta_offset_i):
%
%     k  theta_off  alpha   a   d
%     1     0       +90deg  0   0.333
%     2    -90deg   -90deg  0   0
%     3     0       +90deg  0   0.333
%     4     0       -90deg  0   0
%     5     0       +90deg  0   0.333
%     6     0       -90deg  0   0
%     7     0       +90deg  0   0.333
%     8     0       -90deg  0   0
%     9     0         0     0   0.20    (end segment / tool reach)
%
%   The chain is a series of perpendicular-axis revolute pairs with fully
%   alternating +/-90 deg twists. The odd joints (1,3,5,7) each carry a
%   0.333 m link along their z-axis; the even joints (2,4,6,8) are
%   zero-length re-orienting joints. The shoulder height is d1 = 0.333 m;
%   joint 2 (twist -90 deg, offset -90 deg) is the shoulder pitch, so
%   turning it lifts the arm (elevation tracks J2: 0deg->15, 45->54,
%   90->90). At q = 0 the arm extends horizontally to the EE at
%   [1.199, 0, 0.333] m. With 9 revolute DOF against a 6D pose task the
%   self-motion manifold is 3-dimensional -- a genuinely redundant arm
%   with a regular, interpretable structure.

    r.name = 'Synthetic9';
    r.n    = 9;

    %               1     2     3     4      5     6      7     8      9
    r.dh.a     = [0,    0,    0,    0,     0,    0,     0,    0,     0    ];
    r.dh.alpha = [pi/2, -pi/2,pi/2, -pi/2, pi/2, -pi/2, pi/2, -pi/2, 0    ];
    r.dh.d     = [0.333,0,    0.333,0,     0.333,0,     0.333,0,     0.20 ];
    r.dh.theta_offset = [0, -pi/2, 0, 0, 0, 0, 0, 0, 0];

    r.tool = eye(4);                    % EE at the end of the final link

    % Generous symmetric revolute limits (synthetic arm, no self-collision model).
    r.q_min = -pi * ones(9,1);
    r.q_max =  pi * ones(9,1);

    % Ready pose: a bent, mid-workspace, well-conditioned configuration
    % (max-manipulability home; EE at reach 0.92 m, ~60% of max). q = 0 would
    % extend the arm to 81% reach near the singular boundary, which floors
    % every solver on the smooth trajectories; this home keeps the smooth
    % cells well-posed while the cuspidal trajectory remains the hard test.
    r.q_home = [0.3649; -0.4874; -0.6279; 0.8007; 0.5422; 0.9917; 0.31; 1.1536; 0.1335];

    r.fk  = @(q) fk_chain(q, r.dh, r.tool);
    r.jac = @(q) jac_geometric(q, r.dh, r.tool);
end
