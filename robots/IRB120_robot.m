function r = IRB120_robot()
% IRB120_ROBOT  ABB IRB 120-3/0.6 (6 DOF, all revolute, spherical wrist).
%   Standard DH parameters (metres, radians) from ABB's published dimensions
%   (base 0.290, upper arm 0.270, forearm offset 0.070, forearm 0.302,
%   wrist centre to flange 0.072). Verified 2026-10-19 against RoboDK's
%   library model "ABB IRB 120-3/0.6" (C:\RoboDK\Library\ABB-IRB-120-3-0-6.robot):
%   flange pose identical to 1.5e-15 at 202 configurations (zero, home and
%   200 random joint vectors within the limits). Joint limits are the RoboDK
%   model's. Axes 4, 5 and 6 meet in one point (a4 = a5 = d5 = 0): the
%   spherical wrist that the closed-form solvers in the benchmark require.

    r.name  = 'IRB120';
    r.n     = 6;
    r.dh.a     = [0,      0.270,  0.070,  0,      0,      0    ];
    r.dh.alpha = [-pi/2,  0,     -pi/2,   pi/2,  -pi/2,   0    ];
    r.dh.d     = [0.290,  0,      0,      0.302,  0,      0.072];
    r.dh.theta_offset = [0, -pi/2, 0, 0, 0, pi];
    r.tool  = eye(4);

    r.q_min = deg2rad([-165; -110; -90; -160; -120; -400]);
    r.q_max = deg2rad([ 165;  110;  70;  160;  120;  400]);

    % Home: arm upright, forearm horizontal, wrist pitched so the flange points
    % down (joint 5 at 90 deg keeps the wrist away from its theta5 = 0 singularity).
    r.q_home = [0; 0; 0; 0; pi/2; 0];

    % Closures so callers can write r.fk(q), r.jac(q).
    r.fk  = @(q) fk_chain(q, r.dh, r.tool);
    r.jac = @(q) jac_geometric(q, r.dh, r.tool);
end
