function gen_all_figs(results_csv, out_dir)
% GEN_ALL_FIGS  Result figures and the rank statistics from one results CSV.
%
%   gen_all_figs()                         % results/polished.csv -> results/figures/polished
%   gen_all_figs(csv_path, out_dir)
%
% Produces (and overwrites) in out_dir:
%   fig_cd.png                  Nemenyi critical-difference plot of the wall-time ranks
%   fig_pareto.png              median time vs mean success, every solver
%   fig_pareto_panda_rectangle.png  time vs position error, Panda rectangle
%   fig_success_grid.png        success per (cell, solver); not-applicable cells grey
%   fig_box_time.png            per-solver min/median/max of the cell-median time
%   fig_box_err.png             per-solver min/median/max of the cell-median position error
%   stats_report.txt            Friedman, Iman-Davenport and Nemenyi numbers
%
% Cells in which a solver declared the robot not applicable (frac_not_applicable = 1) are left out
% (NaN). Solvers in INVALID (rows not valid) are left out of every figure and test.
% The Friedman test needs complete blocks, so it ranks only the solvers that apply to every cell;
% the others are listed in the report. The Nemenyi critical value is q_alpha = q(alpha; k, Inf)/sqrt(2),
% computed numerically from the studentized range distribution for the actual k (Demsar 2006).

    code_dir = fileparts(fileparts(mfilename('fullpath')));
    if nargin < 1 || isempty(results_csv)
        results_csv = fullfile(code_dir, 'results', 'polished.csv');
    end
    if nargin < 2 || isempty(out_dir)
        out_dir = fullfile(code_dir, 'results', 'figures', 'polished');
    end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    addpath(fullfile(code_dir, 'utils'));

    INVALID = strings(0, 1);             % solvers whose rows are not valid: none
    alpha = 0.05;

    fprintf('Reading %s\n', results_csv);
    T = readtable(results_csv, 'TextType', 'string');
    T = T(T.crashed == 0 & ~ismember(T.label, INVALID), :);
    na = T.frac_not_applicable >= 1;
    T.med_t_ms(na) = NaN; T.success(na) = NaN; T.med_pos_err(na) = NaN;
    phase = T.phase(1);
    if phase == "polished", ptxt = 'Phase 2 (polished)'; else, ptxt = 'Phase 1 (as published)'; end

    solvers = unique(T.label);
    cells   = unique(strcat(T.robot, "|", T.traj));
    k = numel(solvers);
    n = numel(cells);
    names = erase(solvers, "Lit:");
    fprintf('Solvers: %d, cells: %d, rows: %d\n', k, n, height(T));

    [M_time, M_succ, M_perr] = build_matrices(T, solvers, cells);

    %% Nemenyi CD plot on the complete solvers
    complete = all(~isnan(M_time), 1);
    kc = nnz(complete);
    [chi2, Fid, p_chi, df, CD, q_a] = friedman_stats(M_time(:, complete), alpha);
    fprintf('Friedman over %d complete solvers: chi^2 = %.2f (df=%d), p = %.3e; F = %.2f; CD = %.2f (q_alpha = %.4f)\n', ...
            kc, chi2, df, p_chi, Fid, CD, q_a);
    fig = figure('Visible', 'off', 'Position', [100 100 900 760], 'Color', 'w');
    draw_nemenyi(M_time(:, complete), names(complete), CD);
    title(sprintf('%s: Nemenyi CD plot of time ranks (k=%d, N=%d, CD=%.2f)', ptxt, kc, n, CD));
    exportgraphics(fig, fullfile(out_dir, 'fig_cd.png'), 'BackgroundColor', 'white', 'Resolution', 150);
    close(fig);

    %% Pareto front: median time vs mean success
    mean_succ = mean(M_succ, 1, 'omitnan');
    med_time  = median(M_time, 1, 'omitnan');
    fig = figure('Visible', 'off', 'Position', [100 100 900 600], 'Color', 'w');
    semilogx(med_time(complete), mean_succ(complete), 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 7); hold on;
    semilogx(med_time(~complete), mean_succ(~complete), 'ks', 'MarkerFaceColor', [.9 .6 .1], 'MarkerSize', 7);
    xlabel('Median solve time per waypoint (ms)');
    ylabel('Mean success rate over the cells where the solver applies');
    title(sprintf('%s: speed vs success', ptxt));
    legend({'applies to every robot', 'applies to some robots'}, 'Location', 'southeast');
    grid on;
    [~, isucc] = sort(mean_succ, 'descend');
    [~, ispeed] = sort(med_time, 'ascend');
    for j = unique([isucc(1:min(6, k)), ispeed(1:min(6, k))])
        text(med_time(j) * 1.1, mean_succ(j), char(names(j)), 'FontSize', 8, 'Interpreter', 'none');
    end
    exportgraphics(fig, fullfile(out_dir, 'fig_pareto.png'), 'BackgroundColor', 'white', 'Resolution', 150);
    close(fig);

    %% Pareto front: Panda rectangle, time vs position error
    fig = figure('Visible', 'off', 'Position', [100 100 900 600], 'Color', 'w');
    cell_pr = find(cells == "Panda|rectangle", 1);
    if ~isempty(cell_pr)
        x = M_time(cell_pr, :); y = M_perr(cell_pr, :);
        ok = ~isnan(x) & ~isnan(y) & x > 0 & y > 0;
        x = x(ok).'; y = y(ok).'; nm = names(ok);
        front = pareto_min(x, y);
        loglog(x(~front), y(~front), 'o', 'MarkerEdgeColor', [.55 .55 .55], ...
               'MarkerFaceColor', [.82 .82 .82], 'MarkerSize', 6); hold on;
        yline(1e-5, '--', 'position tolerance gate (10^{-5} m)', 'Color', [.30 .30 .30], ...
              'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom', 'FontSize', 8);
        [xf, si] = sort(x(front)); yfa = y(front); nfa = nm(front); yf = yfa(si); nf = nfa(si);
        plot(xf, yf, '-', 'Color', 'k', 'LineWidth', 1.0);
        loglog(xf, yf, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [.85 .10 .10], 'MarkerSize', 8);
        for j = 1:numel(xf)
            text(xf(j) * 1.18, yf(j), char(nf(j)), 'FontSize', 8, 'Interpreter', 'none');
        end
        set(gca, 'XScale', 'log', 'YScale', 'log');
        xlabel('Median solve time per waypoint (ms)');
        ylabel('Median position error (m)');
        title(sprintf('%s: time vs accuracy, Panda rectangle', ptxt));
        grid on;
    end
    exportgraphics(fig, fullfile(out_dir, 'fig_pareto_panda_rectangle.png'), 'BackgroundColor', 'white', 'Resolution', 150);
    close(fig);

    %% Per-solver position-error min/median/max
    draw_range(log10(M_perr), names, ...
        'log_{10}(median position error per cell, m)', ...
        sprintf('%s: position error min/median/max over the cells where each solver applies', ptxt), ...
        fullfile(out_dir, 'fig_box_err.png'), log10(1e-5));

    %% Success per (cell, solver)
    fig = figure('Visible', 'off', 'Position', [100 100 1200 900], 'Color', 'w');
    im = imagesc(M_succ);
    set(im, 'AlphaData', ~isnan(M_succ));
    set(gca, 'Color', [.75 .75 .75]);
    colormap(parula); colorbar; clim([0 1]);
    xticks(1:k); xticklabels(names); xtickangle(60); set(gca, 'TickLabelInterpreter', 'none');
    yticks(1:n); yticklabels(strrep(cells, "cuspidal", "workspace-boundary"));
    title(sprintf('%s: success per (cell, solver), 10^{-5} m / 10^{-5} rad gate; grey = not applicable', ptxt));
    exportgraphics(fig, fullfile(out_dir, 'fig_success_grid.png'), 'BackgroundColor', 'white', 'Resolution', 150);
    close(fig);

    %% Per-solver wall-time min/median/max
    draw_range(log10(M_time), names, ...
        'log_{10}(median solve time per waypoint, ms)', ...
        sprintf('%s: solve time min/median/max over the cells where each solver applies', ptxt), ...
        fullfile(out_dir, 'fig_box_time.png'), []);

    %% Stats report
    fid = fopen(fullfile(out_dir, 'stats_report.txt'), 'w');
    fprintf(fid, '%s rank statistics (gen_all_figs)\n', ptxt);
    [~, src_name, src_ext] = fileparts(results_csv);
    fprintf(fid, 'Source: %s%s\n', src_name, src_ext);
    if ~isempty(INVALID), fprintf(fid, 'Left out (rows not valid): %s\n', strjoin(INVALID, ', ')); end
    fprintf(fid, 'Solvers: %d; cells (blocks): %d\n', k, n);
    fprintf(fid, 'Ranked (apply to every cell): %d\n', kc);
    fprintf(fid, 'Not ranked (not applicable on some cells): %s\n', strjoin(names(~complete), ', '));
    fprintf(fid, 'Measure ranked: median wall time per waypoint (ms), lower = better\n');
    fprintf(fid, 'Friedman chi^2 = %.4f (df=%d), p = %.6e\n', chi2, df, p_chi);
    fprintf(fid, 'Iman-Davenport F = %.4f, df = (%d, %d)\n', Fid, df, df * (n - 1));
    fprintf(fid, 'Nemenyi q_alpha (alpha=%.2f, k=%d) = %.4f; CD = %.4f\n', alpha, kc, q_a, CD);
    fclose(fid);
    fprintf('Wrote figures and stats_report.txt to %s\n', out_dir);
end

function [Mt, Ms, Mp] = build_matrices(T, solvers, cells)
    k = numel(solvers); n = numel(cells);
    Mt = nan(n, k); Ms = nan(n, k); Mp = nan(n, k);
    for c = 1:n
        parts = split(cells(c), "|");
        for s = 1:k
            mask = (T.robot == parts(1) & T.traj == parts(2) & T.label == solvers(s));
            if any(mask)
                Mt(c, s) = median(T.med_t_ms(mask));
                Ms(c, s) = median(T.success(mask));
                Mp(c, s) = median(T.med_pos_err(mask));
            end
        end
    end
end

function draw_range(L, names, ylab, ttl, path, gate)
    L(isinf(L)) = NaN;
    k = size(L, 2);
    mn = min(L, [], 1, 'omitnan'); md = median(L, 1, 'omitnan'); mx = max(L, [], 1, 'omitnan');
    [md_sorted, ord] = sort(md);
    fig = figure('Visible', 'off', 'Position', [100 100 1200 600], 'Color', 'w');
    x = 1:k;
    plot(x, md_sorted, 'b.-', 'MarkerSize', 12); hold on;
    plot(x, mn(ord), 'k_', 'MarkerSize', 10);
    plot(x, mx(ord), 'k_', 'MarkerSize', 10);
    for j = 1:k, plot([x(j) x(j)], [mn(ord(j)) mx(ord(j))], 'k-'); end
    if ~isempty(gate), yline(gate, 'r--', '10^{-5} m gate'); end
    xticks(1:k); xticklabels(cellstr(names(ord))); xtickangle(60); set(gca, 'TickLabelInterpreter', 'none');
    ylabel(ylab); title(ttl); grid on;
    exportgraphics(fig, path, 'BackgroundColor', 'white', 'Resolution', 150);
    close(fig);
end

function R = rank_rows(M)
% Ranks within each row (block), ties get the average rank.
    [n, k] = size(M);
    R = zeros(n, k);
    for i = 1:n
        x = M(i, :); [s, idx] = sort(x); r = zeros(size(x));
        j = 1;
        while j <= k
            jj = j;
            while jj < k && s(jj + 1) == s(j), jj = jj + 1; end
            r(idx(j:jj)) = (j + jj) / 2;
            j = jj + 1;
        end
        R(i, :) = r;
    end
end

function [chi2, F, p_chi, df, CD, q_a] = friedman_stats(M, alpha)
    [n, k] = size(M);
    Rj = sum(rank_rows(M), 1);
    chi2 = (12 / (n * k * (k + 1))) * sum(Rj .^ 2) - 3 * n * (k + 1);
    df = k - 1;
    F = ((n - 1) * chi2) / (n * (k - 1) - chi2);
    p_chi = gammainc(chi2 / 2, df / 2, 'upper');
    q_a = nemenyi_q(k, alpha);
    CD = q_a * sqrt(k * (k + 1) / (6 * n));
end

function q_a = nemenyi_q(k, alpha)
% q_alpha = q(alpha; k, Inf) / sqrt(2), with q the upper alpha point of the studentized range of
% k independent N(0,1) variables: P(Q <= q) = k * int phi(z) [Phi(z) - Phi(z - q)]^(k-1) dz.
    Phi = @(x) 0.5 * erfc(-x / sqrt(2));
    phi = @(x) exp(-x .^ 2 / 2) / sqrt(2 * pi);
    cdfQ = @(q) k * integral(@(z) phi(z) .* (Phi(z) - Phi(z - q)) .^ (k - 1), -Inf, Inf, ...
                             'AbsTol', 1e-12, 'RelTol', 1e-10);
    q = fzero(@(q) cdfQ(q) - (1 - alpha), [0.5 20]);
    q_a = q / sqrt(2);
end

function draw_nemenyi(M, names, CD)
    k = size(M, 2);
    Rbar = mean(rank_rows(M), 1);
    [Rsorted, ord] = sort(Rbar);
    plot(Rsorted, 1:k, 'ko', 'MarkerFaceColor', 'b'); hold on;
    for i = 1:k
        text(Rsorted(i) + 0.3, i, char(names(ord(i))), 'FontSize', 7, 'Interpreter', 'none');
    end
    yc = -1;
    plot([Rsorted(1) Rsorted(1) + CD], [yc yc], 'r-', 'LineWidth', 2);
    text(Rsorted(1) + CD / 2, yc - 0.6, sprintf('CD = %.2f', CD), 'HorizontalAlignment', 'center', 'FontSize', 9);
    xlabel('Mean rank (lower = faster)');
    ylabel('Solver (sorted)');
    grid on;
    xlim([0 k + 5]);
    ylim([-2.5 k + 2]);
end
