function gen_ranking_csv(in_csv, out_csv)
% GEN_RANKING_CSV  Per-solver ranking from one results CSV (per-cell rows).
%
%   gen_ranking_csv()           % results/polished.csv -> results/tables/polished/ranking.csv
%   gen_ranking_csv(in_csv, out_csv)
%
%   Columns: rank, solver, mean_success, median_pos_err, median_rot_err, median_time_ms,
%   median_polish_iters, n_cells (cells where the solver applies), valid (0 for solvers in INVALID,
%   whose rows are not valid). Statistics are over the cells where the solver applies
%   (frac_not_applicable < 1). Sorted by mean_success descending, ties by median_time_ms ascending.

    code_dir = fileparts(fileparts(mfilename('fullpath')));
    if nargin < 1 || isempty(in_csv),  in_csv  = fullfile(code_dir, 'results', 'polished.csv'); end
    if nargin < 2 || isempty(out_csv), out_csv = fullfile(code_dir, 'results', 'tables', 'polished', 'ranking.csv'); end
    INVALID = strings(0, 1);             % solvers whose rows are not valid: none

    T = readtable(in_csv, 'TextType', 'string');
    T = T(T.crashed == 0 & T.frac_not_applicable < 1, :);
    solvers = unique(T.label);
    R = struct('solver', {}, 'mean_success', {}, 'median_pos_err', {}, 'median_rot_err', {}, ...
               'median_time_ms', {}, 'median_polish_iters', {}, 'n_cells', {});
    for k = 1:numel(solvers)
        m = T.label == solvers(k);
        R(end+1).solver = char(solvers(k)); %#ok<AGROW>
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

    out_dir = fileparts(out_csv);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    fid = fopen(out_csv, 'w');
    fprintf(fid, 'rank,solver,mean_success,median_pos_err,median_rot_err,median_time_ms,median_polish_iters,n_cells,valid\n');
    for k = 1:numel(R)
        fprintf(fid, '%d,%s,%.4f,%.3e,%.3e,%.2f,%.1f,%d,%d\n', k, R(k).solver, R(k).mean_success, ...
            R(k).median_pos_err, R(k).median_rot_err, R(k).median_time_ms, R(k).median_polish_iters, ...
            R(k).n_cells, ~any(string(R(k).solver) == INVALID));
    end
    fclose(fid);
    fprintf('Wrote %s (%d solvers)\n', out_csv, numel(R));
end
