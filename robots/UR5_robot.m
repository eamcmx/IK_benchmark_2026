function r = UR5_robot()
% UR5_ROBOT  Universal Robots UR5 (6 DOF, all revolute).
%   Standard DH parameters from the UR5 manual (metres, radians).
%   Reference: Universal Robots, "DH Parameters for calculations of
%   kinematics and dynamics" (2015).

    r.name  = 'UR5';
    r.n     = 6;
    r.dh.a     = [0,        -0.425,   -0.39225, 0,        0,        0      ];
    r.dh.alpha = [pi/2,      0,        0,       pi/2,    -pi/2,     0      ];
    r.dh.d     = [0.089159,  0,        0,       0.10915,  0.09465,  0.0823 ];
    r.dh.theta_offset = zeros(1,6);
    r.tool  = eye(4);

    % UR joints have effectively continuous travel (+/- 2*pi); use this as
    % a soft bound for solvers that respect joint limits.
    r.q_min = -2*pi*ones(6,1);
    r.q_max =  2*pi*ones(6,1);

    % A "neutral" home that is non-singular and away from joint limits.
    r.q_home = [0; -pi/2; pi/2; -pi/2; -pi/2; 0];

    % Closures so callers can write r.fk(q), r.jac(q).
    r.fk  = @(q) fk_chain(q, r.dh, r.tool);
    r.jac = @(q) jac_geometric(q, r.dh, r.tool);
end
