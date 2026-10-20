function gen_paper_tables(in_csv, out_dir)
% GEN_PAPER_TABLES  LaTeX tabular bodies from one results CSV (phase1.csv or polished.csv).
%
%   gen_paper_tables()                    % results/polished.csv -> results/tables/polished
%   gen_paper_tables(in_csv, out_dir)
%
% Writes, one row per solver (all solvers in the CSV, run_benchmark order: the two baselines,
% then the literature solvers alphabetically):
%   tab_time_body.tex         median wall time per waypoint (ms), one column per robot
%   tab_err_body.tex          median position error (m), one column per robot
%   tab_rot_body.tex          median rotation error (rad), one column per robot
%   tab_timeerr_body.tex      time columns then position-error columns
%   tab_cuspidal_success.tex  workspace-boundary (cuspidal) success, one column per robot
%   tab_polished_ranking.tex  per-solver ranking (mean success, median errors and time, polish iters)
% Robot columns: UR5, Panda, Synthetic9, Synthetic11, Synthetic13, IRB120. Time and error columns
% are the median over the smooth trajectories (rectangle, circle, lissajous) of the cell medians.
% A cell in which the solver declared the robot not applicable (frac_not_applicable = 1) prints
% "n/a" and is left out of every median. Solvers listed in INVALID (rows not valid)
% carry a dagger.

    code_dir = fileparts(fileparts(mfilename('fullpath')));
    if nargin < 1 || isempty(in_csv),  in_csv  = fullfile(code_dir, 'results', 'polished.csv'); end
    if nargin < 2 || isempty(out_dir), out_dir = fullfile(code_dir, 'results', 'tables', 'polished'); end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    INVALID = strings(0, 1);             % solvers whose rows are not valid: none

    T = readtable(in_csv, 'TextType', 'string');
    T = T(T.crashed == 0, :);
    T.na = T.frac_not_applicable >= 1;

    robots = ["UR5", "Panda", "Synthetic9", "Synthetic11", "Synthetic13", "IRB120"];
    smooth = ["rectangle", "circle", "lissajous"];
    solvers = solver_order(unique(T.label));

    write_rows(fullfile(out_dir, 'tab_time_body.tex'), solvers, ...
        @(s) per_robot(T, s, robots, smooth, 'med_t_ms', '%7.2f'), INVALID);
    write_rows(fullfile(out_dir, 'tab_err_body.tex'), solvers, ...
        @(s) per_robot(T, s, robots, smooth, 'med_pos_err', '%.1e'), INVALID);
    write_rows(fullfile(out_dir, 'tab_rot_body.tex'), solvers, ...
        @(s) per_robot(T, s, robots, smooth, 'med_rot_err', '%.1e'), INVALID);
    write_rows(fullfile(out_dir, 'tab_timeerr_body.tex'), solvers, ...
        @(s) [per_robot(T, s, robots, smooth, 'med_t_ms', '%7.2f'), ...
              per_robot(T, s, robots, smooth, 'med_pos_err', '%.1e')], INVALID);
    write_rows(fullfile(out_dir, 'tab_cuspidal_success.tex'), solvers, ...
        @(s) per_robot(T, s, robots, "cuspidal", 'success', '%4.2f'), INVALID);

    R = build_ranking(T, solvers);
    fid = fopen(fullfile(out_dir, 'tab_polished_ranking.tex'), 'w');
    for k = 1:numel(R)
        fprintf(fid, '%-24s & %.3f & %.1e & %.1e & %7.2f & %s & %d \\\\\n', ...
            tex_solver(R(k).solver, INVALID), R(k).mean_success, R(k).median_pos_err, ...
            R(k).median_rot_err, R(k).median_time_ms, num_or_dash(round(R(k).median_polish_iters), '%d'), ...
            R(k).n_cells);
    end
    fclose(fid);

    % A trailing blank line in an \input{} body becomes a \par inside a tabular and breaks
    % booktabs (Misplaced \noalign at \bottomrule), so every body ends without a newline.
    F = dir(fullfile(out_dir, 'tab_*.tex'));
    for i = 1:numel(F)
        pp = fullfile(out_dir, F(i).name);
        ss = deblank(strrep(fileread(pp), char(13), ''));
        fid = fopen(pp, 'w'); fwrite(fid, ss); fclose(fid);
    end
    fprintf('Wrote %d table bodies (%d solvers, %d robots) to %s\n', numel(F), numel(solvers), numel(robots), out_dir);
end

function write_rows(path, solvers, rowfun, INVALID)
    fid = fopen(path, 'w');
    for s = solvers(:)'
        cellsTxt = rowfun(s);
        fprintf(fid, '%-24s & %s \\\\\n', tex_solver(s, INVALID), strjoin(cellsTxt, ' & '));
    end
    fclose(fid);
end

function out = per_robot(T, solver, robots, trajs, col, fmt)
% One string per robot: the median over the given trajectories of the cell values, or "n/a" when
% the solver declared the robot not applicable on every one of those cells.
    out = strings(1, numel(robots));
    for i = 1:numel(robots)
        m = T.label == solver & T.robot == robots(i) & ismember(T.traj, trajs);
        if ~any(m)
            out(i) = "---";
        elseif all(T.na(m))
            out(i) = "n/a";
        else
            v = median(T.(col)(m & ~T.na), 'omitnan');
            out(i) = num_or_dash(v, fmt);
        end
    end
end

function R = build_ranking(T, solvers)
% Per solver, over the cells where it applies: mean success, medians of the cell medians.
    R = struct('solver', {}, 'mean_success', {}, 'median_pos_err', {}, 'median_rot_err', {}, ...
               'median_time_ms', {}, 'median_polish_iters', {}, 'n_cells', {});
    for k = 1:numel(solvers)
        m = T.label == solvers(k) & ~T.na;
        R(end+1).solver = solvers(k); %#ok<AGROW>
        R(end).mean_success = mean(T.success(m), 'omitnan');
        R(end).median_pos_err = median(T.med_pos_err(m), 'omitnan');
        R(end).median_rot_err = median(T.med_rot_err(m), 'omitnan');
        R(end).median_time_ms = median(T.med_t_ms(m), 'omitnan');
        R(end).median_polish_iters = median(T.med_polish_iters(m), 'omitnan');
        R(end).n_cells = nnz(m);
    end
    succ = [R.mean_success]'; succ(isnan(succ)) = -Inf;
    [~, ord] = sortrows([-succ, [R.median_time_ms]']);
    R = R(ord);
end

function s = solver_order(labels)
% run_benchmark order: Newton12-anaJ, PINV-orig, then Lit:* alphabetically (case-insensitive).
    base = ["Newton12-anaJ", "PINV-orig"];
    lit = labels(startsWith(labels, "Lit:"));
    [~, o] = sort(lower(lit));
    s = [base(ismember(base, labels)), lit(o)'];
    other = setdiff(labels, s, 'stable');
    s = [s, other(:)'];
end

function t = num_or_dash(v, fmt)
    if isnan(v), t = "---"; else, t = string(sprintf(fmt, v)); end
end

function s = tex_solver(name, INVALID)
    s = char(name);
    if any(name == INVALID), dag = '$^\dagger$'; else, dag = ''; end
    s = strrep(s, 'Lit:', '');          % the manuscripts use bare names (PINV, not Lit:PINV)
    s = [strrep(s, '_', '\_') dag];
end
