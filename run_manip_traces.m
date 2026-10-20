function traces = run_manip_traces(plot_only)
% RUN_MANIP_TRACES  Re-run a focused set of Phase 2 (solver, robot, circle) cells with the
% per-waypoint manipulability and joint trajectory kept, then regenerate fig_manipulability.png.
%
%   run_manip_traces()       runs the cells, saves results/manip_traces.mat, draws the figure
%   run_manip_traces(true)   redraws the figure from results/manip_traces.mat
%
% Layout:
%   Top row : PINV-orig on all six robots, circle trajectory.
%   Middle  : PINV-orig + 5 comparators on the Synthetic9 circle (posture coherence).
%   Bottom  : the same solvers on the Synthetic13 circle (highest redundancy).
%
% Every cell is run as run_benchmark runs it in Phase 2: the same options
% (default_opts('max_iter',200,'tol_pos',1e-5,'tol_rot',1e-5,'budget',2000) with harness counting
% on), the base solver wrapped by solve_polished, the 100-waypoint circle from make_default_traj,
% q_home as the first warm start, bench_run as the harness, and the same per-cell RNG seed. The
% success and median errors of each cell are printed next to its row of results/polished.csv.
%
% Yoshikawa manipulability w(q) = sqrt(det(J J')) is the per-waypoint .manip returned by bench_run.

if nargin < 1, plot_only = false; end

this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir,'solvers'), fullfile(this_dir,'utils'), ...
        fullfile(this_dir,'robots'), fullfile(this_dir,'trajectories'), ...
        fullfile(this_dir,'bench'));
mat_file = fullfile(this_dir, 'results', 'manip_traces.mat');
csv_file = fullfile(this_dir, 'results', 'polished.csv');
out_png  = fullfile(this_dir, 'results', 'figures', 'fig_manipulability.png');

if plot_only
    S = load(mat_file, 'traces');
    traces = S.traces;
    plot_manip_traces(traces, out_png);
    return
end

N_wp = 100;
opts = default_opts('max_iter',200,'tol_pos',1e-5,'tol_rot',1e-5,'budget',2000);
opts.ikb_counting = true;      % as bench_run sets it (harness-side FK/Jacobian counting)

robots = {UR5_robot(), IRB120_robot(), Panda_robot(), Synthetic9_robot(), ...
          Synthetic11_robot(), Synthetic13_robot()};
ref  = {'PINV-orig',     @solve_pinv_orig};
cmps = {'Newton12-anaJ', @solve_newton12_anaJ;
        'Lit:DLS',       @solve_DLS;
        'Lit:ImpFABRIK', @solve_ImpFABRIK;
        'Lit:GA',        @solve_GA;
        'Lit:PSO',       @solve_PSO};

traces = struct();
traces.traj = 'circle';
traces.N_wp = N_wp;
traces.opts = opts;

% Top row: PINV-orig on every robot
fprintf('=== Top row: PINV-orig, circle ===\n');
top = cell(1, numel(robots));
for r = 1:numel(robots)
    top{r} = run_cell(robots{r}, ref{1}, ref{2}, opts, N_wp);
end
traces.top = [top{:}];

% Middle and bottom rows: PINV-orig and the comparators on Synthetic9 and Synthetic13
fprintf('=== Middle row: Synthetic9 circle ===\n');
traces.mid = run_row(Synthetic9_robot(), [ref; cmps], opts, N_wp);
fprintf('=== Bottom row: Synthetic13 circle ===\n');
traces.bot = run_row(Synthetic13_robot(), [ref; cmps], opts, N_wp);

save(mat_file, 'traces');
fprintf('Saved traces to %s\n', mat_file);

check_against_csv([traces.top, traces.mid.cells, traces.bot.cells], csv_file);
print_mean_manip(traces.mid);
print_mean_manip(traces.bot);
plot_manip_traces(traces, out_png);
end

% -------------------------------------------------------------------------
function row = run_row(rb, solvers, opts, N_wp)
    c = cell(1, size(solvers, 1));
    for k = 1:size(solvers, 1)
        c{k} = run_cell(rb, solvers{k,1}, solvers{k,2}, opts, N_wp);
    end
    row.robot = rb.name;
    row.cells = [c{:}];
end

function c = run_cell(rb, label, base, opts, N_wp)
% One Phase 2 cell, set up exactly as in run_benchmark.
    rng(seed_of(label, rb.name, 'circle'), 'twister');
    Ts = make_default_traj(rb, 'circle', N_wp);
    fn = @(r_,Td,q_,o) solve_polished(base, r_, Td, q_, o);
    t0 = tic;
    res = bench_run(rb, fn, Ts, rb.q_home, opts);
    ok = ~res.not_applicable;
    c.label   = label;
    c.name    = erase(label, 'Lit:');
    c.robot   = rb.name;
    c.manip   = res.manip;
    c.Q       = res.Q;
    c.success = NaN;
    if any(ok), c.success = mean(res.converged(ok)); end
    c.med_pos_err = median(res.errs_pos);
    c.med_rot_err = median(res.errs_rot);
    c.frac_not_applicable = mean(res.not_applicable);
    c.mean_manip = mean(res.manip);
    fprintf('  %-14s %-12s succ=%.2f perr=%.2e rerr=%.2e mean w=%.6f (%.1f s)\n', ...
            label, rb.name, c.success, c.med_pos_err, c.med_rot_err, c.mean_manip, toc(t0));
end

function s = seed_of(label, robot, traj)
% Same per-cell seed as run_benchmark.
    str = [label '|' robot '|' traj];
    s = uint32(mod(sum(double(str).*(1:numel(str))), 2^31-1));
end

function check_against_csv(cells, csv_file)
% Prints each traced cell next to its row of the Phase 2 results.
    T = readtable(csv_file, 'TextType', 'string');
    fprintf('=== Traced cells vs %s ===\n', csv_file);
    for c = cells
        m = T.label == c.label & T.robot == c.robot & T.traj == "circle";
        if nnz(m) ~= 1
            fprintf('  %-14s %-12s no unique row in the CSV\n', c.label, c.robot);
            continue
        end
        fprintf('  %-14s %-12s succ %.4f / csv %.4f   perr %.4e / csv %.4e   rerr %.4e / csv %.4e\n', ...
                c.label, c.robot, c.success, T.success(m), c.med_pos_err, T.med_pos_err(m), ...
                c.med_rot_err, T.med_rot_err(m));
    end
end

function print_mean_manip(row)
    ref = row.cells(1).mean_manip;
    fprintf('=== Mean manipulability, %s circle (reference %s) ===\n', row.robot, row.cells(1).name);
    for c = row.cells
        fprintf('  %-14s mean w = %.6f   %+.3f %% vs %s\n', c.name, c.mean_manip, ...
                100 * (c.mean_manip - ref) / ref, row.cells(1).name);
    end
end

function plot_manip_traces(traces, out_png)
    nt = numel(traces.top);
    fig = figure('Visible','off','Position',[100 100 1700 950], 'Color','w');
    tl = tiledlayout(fig, 3, nt, 'TileSpacing','compact', 'Padding','compact');

    % Top row: PINV-orig on each robot
    for r = 1:nt
        ax = nexttile(tl, r);
        plot(ax, traces.top(r).manip, 'b-', 'LineWidth', 1.5);
        title(ax, sprintf('%s: PINV-orig', traces.top(r).robot), 'Interpreter','none');
        xlabel(ax, 'Waypoint'); ylabel(ax, 'w(q)'); grid(ax, 'on');
    end

    ax = nexttile(tl, nt + 1, [1 nt]); hold(ax, 'on');
    plot_row(ax, traces.mid, true);
    title(ax, sprintf('%s circle: posture coherence', traces.mid.robot), 'Interpreter','none');

    ax = nexttile(tl, 2*nt + 1, [1 nt]); hold(ax, 'on');
    plot_row(ax, traces.bot, false);
    title(ax, sprintf('%s circle', traces.bot.robot), 'Interpreter','none');

    title(tl, 'Yoshikawa manipulability w(q) along circle trajectories');

    out_dir = fileparts(out_png);
    if ~exist(out_dir,'dir'), mkdir(out_dir); end
    exportgraphics(fig, out_png, 'BackgroundColor','white', 'Resolution',150);
    close(fig);
    fprintf('Wrote %s\n', out_png);
end

function plot_row(ax, row, show_legend)
% PINV-orig is drawn first as a thick line; the comparators are drawn over it with distinct line
% styles, so traces that coincide with the reference stay visible instead of hiding each other.
    colors = {[0 0 1], [0 0 0], [0 .6 0], [1 0 1], [1 0 0], [0 .75 .75]};
    styles = {'-', '--', '-.', '-', '-', ':'};
    widths = [3.0, 1.6, 1.4, 1.2, 1.2, 2.0];
    for k = 1:numel(row.cells)
        j = 1 + mod(k-1, numel(colors));
        plot(ax, row.cells(k).manip, 'Color', colors{j}, 'LineStyle', styles{j}, ...
             'LineWidth', widths(j), 'DisplayName', row.cells(k).name);
    end
    if show_legend, legend(ax, 'Location','best', 'FontSize',7, 'Interpreter','none'); end
    xlabel(ax, 'Waypoint'); ylabel(ax, 'w(q)'); grid(ax, 'on');
end
