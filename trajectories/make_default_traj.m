function Ts = make_default_traj(robot, kind, N)
% MAKE_DEFAULT_TRAJ  Build a robot-appropriate default trajectory that
% exercises BOTH position AND orientation tracking.
%
%   Picks a Cartesian motion that lies inside each robot's reachable
%   workspace, anchored on the robot's home pose. `kind` is 'line',
%   'circle', or 'spline'. Orientation varies along every trajectory so
%   the rotational Jacobian is genuinely loaded, exercising full 6D pose IK.
%
%   Direction note: Panda's canonical "ready" q_home in this DH formulation
%   puts the EE at y ~ +0.47 m, near the +y workspace boundary. Trajectory
%   excursions in +y cross a singular subspace that warm-started PINV/DLS
%   cannot escape. All directions below move the EE in -y / mixed -y +z,
%   keeping every robot in well-conditioned workspace across the
%   benchmark robots.

    if nargin < 2, kind = 'line'; end
    if nargin < 3, N = 50; end

    T_home = robot.fk(robot.q_home);
    R0 = T_home(1:3,1:3);
    p0 = T_home(1:3,4);

    switch lower(kind)
        case 'line'
            % Position: 0.20 m along x, -0.10 m along y, -0.10 m along z.
            % Orientation: 30 deg about the world y-axis from start to end.
            d = [0.20; -0.10; -0.10];
            R_end = axang2R([0;1;0], deg2rad(30)) * R0;
            T_start = T_home;
            T_end   = [R_end, p0 + d; 0 0 0 1];
            Ts = traj_line(T_start, T_end, N);
        case 'circle'
            % xz-plane circle, radius 0.10 m, with a tilt that varies as
            % cos(theta) about the world x-axis (peak +/- 20 deg). Forces
            % the wrist to track a *moving* orientation along the sweep.
            Ts = zeros(4,4,N);
            for k = 1:N
                theta = 2*pi*(k-1)/N;
                d = [0.10*cos(theta); 0; 0.10*sin(theta)];
                tilt = deg2rad(20) * cos(theta);   % +- 20 deg about x
                R = axang2R([1;0;0], tilt) * R0;
                Ts(:,:,k) = [R, p0 + d; 0 0 0 1];
            end
        case 'spline'
            % Four waypoints, monotonic-y position, each with a distinct
            % orientation (rotations about different axes, growing in
            % magnitude) so the trajectory really tumbles the wrist.
            wp = zeros(4,4,4);
            wp(:,:,1) = T_home;
            wp(:,:,2) = build_pose(R0, p0 + [0;    -0.10; 0   ], [0;1;0],  deg2rad(15));
            wp(:,:,3) = build_pose(R0, p0 + [0.10; -0.10; 0   ], [1;0;0], -deg2rad(20));
            wp(:,:,4) = build_pose(R0, p0 + [0.10; -0.10; 0.05], [0;0;1],  deg2rad(25));
            Ts = traj_spline(wp, N);
        case 'lissajous'
            % Continuous 3D Lissajous knot (1:2:3 frequency ratio).
            % The anchor is offset from home in -y by 5 cm and the y
            % amplitude is small (0.04 m) so the entire y excursion stays
            % below the Panda home y (0.47 m), avoiding the +y workspace
            % boundary that breaks every other trajectory on Panda.
            % Orientation oscillates +/- 20 deg about world y, returning
            % to the centre orientation at t = 0 and t = 2 pi.
            T_centre = T_home;
            T_centre(1:3,4) = T_centre(1:3,4) + [0; -0.05; 0];
            amplitudes = [0.10; 0.04; 0.08];
            freqs      = [1; 2; 3];
            phases     = [pi/2; 0; 0];
            Ts = traj_lissajous(T_centre, amplitudes, freqs, phases, N, ...
                                [0;1;0], deg2rad(20));
        case 'rectangle'
            % Closed rectangle in the world xz-plane, sides 0.20 m x 0.10 m,
            % centred 0.05 m below home in y to keep the entire path away
            % from Panda's +y workspace boundary. Orientation slews
            % +-20 deg about world y sinusoidally over one full lap.
            % Four sharp 90 deg corners introduce C^0-discontinuous
            % tangent vectors, exercising abrupt direction reversals.
            p_centre = p0 + [0; -0.05; 0];
            Lx = 0.20; Lz = 0.10;
            corners = [ Lx/2,  Lz/2;
                       -Lx/2,  Lz/2;
                       -Lx/2, -Lz/2;
                        Lx/2, -Lz/2;
                        Lx/2,  Lz/2 ];   % closed
            % Distribute N waypoints around the 4 sides, proportional to side length.
            perim = 2*(Lx + Lz);
            side_frac = [Lx, Lz, Lx, Lz] / perim;
            n_side = max(2, round(side_frac * N));
            % adjust so that sum(n_side) == N (avoid overshoot)
            while sum(n_side) > N, [~,im] = max(n_side); n_side(im) = n_side(im)-1; end
            while sum(n_side) < N, [~,im] = min(n_side); n_side(im) = n_side(im)+1; end
            Ts = zeros(4,4,N);
            k = 0;
            for s = 1:4
                p_a = corners(s,:); p_b = corners(s+1,:);
                for jj = 0:(n_side(s)-1)
                    t = jj / n_side(s);
                    p_xz = (1-t)*p_a + t*p_b;
                    k = k + 1;
                    theta_lap = 2*pi*(k-1)/N;
                    tilt = deg2rad(20) * sin(theta_lap);
                    R = axang2R([0;1;0], tilt) * R0;
                    p = p_centre + [p_xz(1); 0; p_xz(2)];
                    Ts(:,:,k) = [R, p; 0 0 0 1];
                end
            end
        case {'cuspidal','boundary','workspace_boundary'}   % internal key 'cuspidal' == the workspace-boundary trajectory (kept for data-file compatibility)
            % "Pie"-shaped 3D workspace-boundary trajectory.
            %
            % Geometry, expressed relative to the world z-axis = base
            % rotation axis:
            %  Segment 1 (arc): the arm starts near-fully-extended at
            %    elevation 45 deg above the horizontal plane; the base is
            %    rotated through 90 deg, so the tip traces a circular arc
            %    of radius R*cos(45 deg) at height R*sin(45 deg).
            %  Segment 2 (radial line): the tip moves on a straight line
            %    from the end of the arc to a point on the world z-axis
            %    (the base rotation axis), descending at a constant slope
            %    of 20 deg below horizontal.
            %  Segment 3 (closing line): a straight line from the z-axis
            %    point back to the starting point of the arc, closing the
            %    pie wedge.
            % Per-robot R = 0.92 * (max reachable EE radius), estimated
            % from a deterministic random sample of the joint space. This
            % gives each robot its own pie perimeter scaled to actual reach
            % (DH offsets do not add linearly, so sum(link_lens) over-
            % estimates real workspace radius substantially on Panda-class
            % chains; using max ||p|| over a fixed seed keeps every cell
            % inside the chain's reachable workspace while still placing
            % the arc near the workspace-boundary shell).
            % Orientation is held at the home frame R0 along the path.
            R = 0.92 * robot_max_reach(robot);
            elev  = deg2rad(45);
            slope = deg2rad(20);
            az0 = 0;  az1 = deg2rad(90);

            rh = R * cos(elev);   % horizontal distance from base axis
            zh = R * sin(elev);   % height of the elevated arc
            z_C = zh - rh*tan(slope);   % z of the closing apex on base axis
            P_A = [ rh*cos(az0); rh*sin(az0); zh ];
            P_B = [ rh*cos(az1); rh*sin(az1); zh ];
            P_C = [ 0;            0;            z_C ];

            % Distribute N waypoints proportional to segment length.
            arc_len  = rh * (az1 - az0);
            line_len = norm(P_B - P_C);   % = rh / cos(slope) by construction
            total    = arc_len + 2*line_len;
            n1 = max(3, round(N * arc_len  / total));
            n2 = max(2, round(N * line_len / total));
            n3 = max(2, N - n1 - n2);

            seg1 = zeros(n1, 3);
            for jj = 1:n1
                t = (jj-1)/n1;                 % half-open [0,1)
                az = az0 + (az1 - az0)*t;
                seg1(jj,:) = [ rh*cos(az), rh*sin(az), zh ];
            end
            seg2 = zeros(n2, 3);
            for jj = 1:n2
                t = (jj-1)/n2;
                seg2(jj,:) = ((1-t)*P_B + t*P_C).';
            end
            seg3 = zeros(n3, 3);
            for jj = 1:n3
                t = (jj-1)/n3;
                seg3(jj,:) = ((1-t)*P_C + t*P_A).';
            end
            P_all = [seg1; seg2; seg3];

            % Resample to exactly N waypoints
            if size(P_all,1) ~= N
                idx = round(linspace(1, size(P_all,1), N));
                P_all = P_all(idx,:);
            end
            Ts = zeros(4,4,N);
            for k = 1:N
                Ts(:,:,k) = [R0, P_all(k,:).'; 0 0 0 1];
            end
        otherwise
            error('Unknown trajectory kind: %s', kind);
    end
end

function R = robot_max_reach(robot)
% Estimate max reachable EE radius from base origin via 500 deterministic
% random joint samples (fixed seed so the trajectory is reproducible).
    s = rng; rng(0);
    n = numel(robot.q_min);
    R = 0;
    for k = 1:500
        q = robot.q_min(:) + (robot.q_max(:) - robot.q_min(:)) .* rand(n,1);
        T = robot.fk(q);
        d = norm(T(1:3,4));
        if d > R, R = d; end
    end
    rng(s);
end

function L = robot_link_lengths(robot)
% Coarse link-length estimate from DH (|a_k| + |d_k| per joint).
% Used by 'cuspidal' to compute R_outer / R_inner.
    if isfield(robot,'dh') && isfield(robot.dh,'a') && isfield(robot.dh,'d')
        L = abs(robot.dh.a(:)) + abs(robot.dh.d(:));
    elseif isfield(robot,'a') && isfield(robot,'d')
        L = abs(robot.a(:)) + abs(robot.d(:));
    elseif isfield(robot,'DH')
        L = abs(robot.DH(:,1)) + abs(robot.DH(:,3));
    else
        % Fallback: hand-picked workspace radii by name.
        switch lower(robot.name)
            case 'ur5',      L = [0.425; 0.392; 0.10];
            case 'panda',    L = [0.333; 0.316; 0.384];
            otherwise,       L = [0.30; 0.30; 0.30];
        end
    end
end

function R = axang2R(axis, ang)
    axis = axis / max(norm(axis), eps);
    K = [0,-axis(3),axis(2); axis(3),0,-axis(1); -axis(2),axis(1),0];
    R = eye(3) + sin(ang)*K + (1 - cos(ang))*(K*K);
end

function T = build_pose(R0, p, axis, ang)
    R = axang2R(axis, ang) * R0;
    T = [R, p; 0 0 0 1];
end
