function r = Panda_robot()
% PANDA_ROBOT  Franka Emika Panda (7 DOF, all revolute).
%   Standard DH equivalents of the Panda's modified DH parameters
%   (metres, radians). Reference: Franka Control Interface (FCI)
%   documentation, "Robot and interface specifications" (2018+).
%
%   The flange-to-EE offset (0.107 m along z7) is applied via the tool
%   transform.

    r.name = 'Panda';
    r.n    = 7;

    % Standard-DH (Spong) parameters equivalent to the Panda's official
    % modified-DH (Craig) table. Conversion: shift the (a, alpha) columns one
    % row up (Tx and Rx about the common joint axis commute) and fold the
    % 0.107 m flange into d7. Verified to reproduce the official Franka FK to
    % machine precision -- at the ready pose the EE points straight down,
    % z = [0 0 -1], at position [0.307 0 0.590] m.
    r.dh.a     = [0,    0,    0.0825, -0.0825, 0,    0.088, 0    ];
    r.dh.alpha = [-pi/2, pi/2, pi/2,  -pi/2,   pi/2, pi/2,  0    ];
    r.dh.d     = [0.333,0,    0.316,  0,       0.384,0,     0.107];
    r.dh.theta_offset = zeros(1,7);

    r.tool = eye(4);   % flange already folded into d7

    % Official Panda joint limits (rad).
    r.q_min = [-2.8973; -1.7628; -2.8973; -3.0718; -2.8973; -0.0175; -2.8973];
    r.q_max = [ 2.8973;  1.7628;  2.8973; -0.0698;  2.8973;  3.7525;  2.8973];

    % Recommended "ready" home configuration (manufacturer suggested).
    r.q_home = [0; -pi/4; 0; -3*pi/4; 0; pi/2; pi/4];

    r.fk  = @(q) fk_chain(q, r.dh, r.tool);
    r.jac = @(q) jac_geometric(q, r.dh, r.tool);
end
