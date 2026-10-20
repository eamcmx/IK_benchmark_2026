function regen_pareto_overlay(results_csv, out_png, robots, dof, traj)
% REGEN_PARETO_OVERLAY  Time-vs-accuracy Pareto fronts overlaid for several robots, one series per
%   robot: a faint dominated cloud and the highlighted lower-left frontier.
%
%   regen_pareto_overlay()      % results/polished.csv, redundant robots, rectangle,
%                               % -> results/figures/polished/fig_pareto_redundant_overlay.png
%   regen_pareto_overlay(results_csv, out_png, robots, dof, traj)
%
%   Cells where a solver declared the robot not applicable, and solvers in INVALID (rows not valid), are left out. Uses utils/pareto_min.m.

    this_dir = fileparts(mfilename('fullpath'));
    code_dir = fileparts(this_dir);
    addpath(fullfile(code_dir, 'utils'));
    if nargin < 1 || isempty(results_csv), results_csv = fullfile(code_dir, 'results', 'polished.csv'); end
    if nargin < 2 || isempty(out_png)
        out_png = fullfile(code_dir, 'results', 'figures', 'polished', 'fig_pareto_redundant_overlay.png');
    end
    if nargin < 3 || isempty(robots), robots = ["Panda", "Synthetic9", "Synthetic11", "Synthetic13"]; end
    if nargin < 4 || isempty(dof), dof = [7 9 11 13]; end
    if nargin < 5 || isempty(traj), traj = "rectangle"; end
    INVALID = strings(0, 1);             % solvers whose rows are not valid: none

    T = readtable(results_csv, 'TextType', 'string');
    if T.phase(1) == "polished", ptxt = 'Phase 2 (polished)'; else, ptxt = 'Phase 1 (as published)'; end
    cols = lines(numel(robots));

    fig = figure('Visible', 'off', 'Position', [100 100 950 650], 'Color', 'w');
    hold on;
    hleg = gobjects(1, numel(robots));
    for ri = 1:numel(robots)
        m = (T.robot == robots(ri)) & (T.traj == traj) & (T.crashed == 0) & (T.frac_not_applicable < 1) ...
            & ~ismember(T.label, INVALID) & ~isnan(T.med_t_ms) & T.med_t_ms > 0 ...
            & ~isnan(T.med_pos_err) & T.med_pos_err > 0;
        x = T.med_t_ms(m); y = T.med_pos_err(m);
        c = cols(ri, :);
        front = pareto_min(x, y);
        scatter(x(~front), y(~front), 16, c, 'o', ...
                'MarkerFaceColor', c, 'MarkerFaceAlpha', 0.18, 'MarkerEdgeAlpha', 0.18);
        [xf, si] = sort(x(front)); yfa = y(front); yf = yfa(si);
        hleg(ri) = plot(xf, yf, '-o', 'Color', c, 'LineWidth', 1.4, ...
                        'MarkerFaceColor', c, 'MarkerEdgeColor', 'k', 'MarkerSize', 7, ...
                        'DisplayName', sprintf('%s (%d-DOF)', robots(ri), dof(ri)));
        lab = erase(T.label(m), "Lit:"); labf = lab(front); labf = labf(si);
        fprintf('%s %s frontier: %s\n', robots(ri), traj, strjoin(cellstr(labf), ', '));
    end
    yline(1e-5, '--', 'position tolerance gate (10^{-5} m)', 'Color', [.3 .3 .3], ...
          'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom', 'FontSize', 8, ...
          'HandleVisibility', 'off');
    set(gca, 'XScale', 'log', 'YScale', 'log');
    xlabel('Median solve time per waypoint (ms)');
    ylabel('Median position error (m)');
    title(sprintf('%s: time vs accuracy Pareto fronts (%s)', ptxt, traj));
    grid on;
    legend(hleg, 'Location', 'east', 'FontSize', 9);

    out_dir = fileparts(out_png);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    exportgraphics(fig, out_png, 'BackgroundColor', 'white', 'Resolution', 150);
    close(fig);
    fprintf('Wrote %s\n', out_png);
end
