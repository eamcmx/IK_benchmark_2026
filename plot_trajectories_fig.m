function plot_trajectories_fig()
% PLOT_TRAJECTORIES_FIG  Render the four benchmark trajectory shapes (the
% end-effector position paths) for a representative redundant arm, into
% fig_trajectories.png in the manuscript figures folder. The workspace-boundary panel
% is the closed 3D "pie wedge" (45 deg arc + two diagonals dipping 20 deg
% below horizontal) that drives the reachability stress test.

    this = fileparts(mfilename('fullpath'));
    addpath(fullfile(this,'robots'), fullfile(this,'trajectories'), ...
            fullfile(this,'utils'), fullfile(this,'bench'));
    rb = Synthetic9_robot();                 % representative redundant arm
    kinds     = {'rectangle','circle','lissajous','cuspidal'};            % generation keys
    dispkinds = {'rectangle','circle','lissajous','workspace-boundary'};  % display labels

    fig = figure('Visible','off','Position',[100 100 1320 360],'Color','w');
    tl  = tiledlayout(fig,1,4,'TileSpacing','compact','Padding','compact');
    for i = 1:4
        Ts = make_default_traj(rb, kinds{i}, 100);
        P  = squeeze(Ts(1:3,4,:));           % 3 x N end-effector positions
        ax = nexttile(tl); hold(ax,'on'); grid(ax,'on'); axis(ax,'equal'); view(ax,135,22);
        plot3(ax, P(1,:),P(2,:),P(3,:), 'b-','LineWidth',1.6);
        plot3(ax, P(1,1),P(2,1),P(3,1), 'go','MarkerFaceColor','g','MarkerSize',6);  % start
        title(ax, dispkinds{i}, 'Interpreter','none');
        xlabel(ax,'x (m)'); ylabel(ax,'y (m)'); zlabel(ax,'z (m)');
    end
    title(tl, sprintf('Benchmark trajectory shapes (end-effector path, %s)', rb.name), ...
          'Interpreter','none');

    out = fullfile(this,'results','figures','fig_trajectories.png');
    exportgraphics(fig, out, 'BackgroundColor','white','Resolution',150);
    close(fig);
    fprintf('wrote %s\n', out);
end
