function plot_robot_renders()
% Uniform 3D stick-figure renders of every benchmark robot at its home pose,
% for fig_robot_<name>.png in the manuscript. Joint spheres + orange links +
% base box + tool frame triad.

this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir,'solvers'),fullfile(this_dir,'utils'), ...
        fullfile(this_dir,'robots'),fullfile(this_dir,'trajectories'),fullfile(this_dir,'bench'));
out_dir = fullfile(this_dir,'results','figures');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

robots = {UR5_robot(), Panda_robot(), Synthetic9_robot(), Synthetic11_robot(), Synthetic13_robot(), ...
          IRB120_robot()};
for rc = robots
    rb = rc{1};
    % UR5 and Panda are drawn at their real manufacturer home/ready q_home;
    % IRB120 at its q_home (arm upright, forearm horizontal, flange pointing down).
    % The synthetic arms use a clean reaching display pose instead of their
    % max-manipulability q_home (which folds awkwardly) -- purely for
    % illustration; the benchmark itself always uses q_home, unchanged.
    qshow = rb.q_home;
    switch rb.name
        case 'Synthetic9',  qshow = [0;0.8;0.30;0;0.30;0;0.30;0;0];
        case 'Synthetic11', qshow = [0;0.8;0.25;0;0.25;0;0.25;0;0.25;0;0];
        case 'Synthetic13', qshow = [0;0.8;0.22;0;0.22;0;0.22;0;0.22;0;0.22;0;0];
    end
    [Tee, Ts] = rb.fk(qshow);
    O = [zeros(3,1), squeeze(Ts(1:3,4,:))];     % base + joint origins (m)
    p = Tee(1:3,4); R = Tee(1:3,1:3);           % end-effector pose (incl. tool/flange)
    chain = [O, p];                             % connect links through to the EE flange

    f = figure('Visible','off','Position',[100 100 640 600],'Color','w');
    ax = axes(f); hold(ax,'on'); grid(ax,'on'); axis(ax,'equal'); view(ax,135,18);
    % base box
    plot3(ax, 0,0,0,'ks','MarkerFaceColor',[.3 .3 .3],'MarkerSize',16);
    % links (drawn through to the EE flange so the frame is attached) + joints
    plot3(ax, chain(1,:),chain(2,:),chain(3,:), '-','Color',[.85 .55 .1],'LineWidth',6);
    plot3(ax, O(1,2:end),O(2,2:end),O(3,2:end), 'o', ...
          'MarkerFaceColor',[.95 .78 .25],'MarkerEdgeColor',[.6 .4 .05],'MarkerSize',9);
    % EE coordinate frame: RGB triad (x=red, y=green, z=blue), no star marker.
    % Axis length scales with the robot's size so it is not oversized on the
    % small arms (UR5/Panda) nor tiny on the large synthetics.
    L = 0.15 * max(max(chain,[],2) - min(chain,[],2));
    plot3(ax, p(1),p(2),p(3),'k.','MarkerSize',9);   % frame origin
    cols = {[1 0 0],[0 .7 0],[0 0 1]};
    for a=1:3
        q=p+L*R(:,a);
        plot3(ax,[p(1) q(1)],[p(2) q(2)],[p(3) q(3)],'-','Color',cols{a},'LineWidth',3);
    end
    title(ax, sprintf('%s  (%d-DOF)', rb.name, rb.n), 'Interpreter','none','FontSize',13);
    xlabel(ax,'x (m)'); ylabel(ax,'y (m)'); zlabel(ax,'z (m)');
    % Tight cube centred on the arm so it fills the frame (base + joints + EE),
    % instead of a large symmetric box that leaves the robot small in whitespace.
    allp = [O, p];
    mn = min(allp,[],2); mx = max(allp,[],2);
    ctr = (mn+mx)/2; h = 0.58*max(mx-mn) + 0.04;
    xlim(ax, ctr(1)+[-h h]); ylim(ax, ctr(2)+[-h h]); zlim(ax, ctr(3)+[-h h]);
    out = fullfile(out_dir, sprintf('fig_robot_%s.png', lower(rb.name)));
    exportgraphics(f, out, 'BackgroundColor','white','Resolution',150);
    close(f);
    fprintf('wrote %s  (home reach %.2f m)\n', out, norm(p));
end
end
