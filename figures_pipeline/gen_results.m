function gen_results()
% GEN_RESULTS  Regenerate every result table and figure from results/, for both phases.
%
%   Writes results/tables/<phase>/ (LaTeX table bodies, ranking.csv) and
%   results/figures/<phase>/ (PNG figures, stats_report.txt) for phase1 and polished.

    here = fileparts(mfilename('fullpath'));
    code_dir = fileparts(here);
    addpath(here, fullfile(code_dir, 'utils'));
    res = fullfile(code_dir, 'results');
    for ph = ["phase1", "polished"]
        csv  = fullfile(res, ph + ".csv");
        tdir = fullfile(res, 'tables', ph);
        fdir = fullfile(res, 'figures', ph);
        gen_paper_tables(csv, tdir);
        gen_ranking_csv(csv, fullfile(tdir, 'ranking.csv'));
        gen_all_figs(csv, fdir);
        regen_pareto_overlay(csv, fullfile(fdir, 'fig_pareto_redundant_overlay.png'), ...
            ["Panda", "Synthetic9", "Synthetic11", "Synthetic13"], [7 9 11 13], "rectangle");
        regen_pareto_overlay(csv, fullfile(fdir, 'fig_pareto_6dof_overlay.png'), ...
            ["UR5", "IRB120"], [6 6], "rectangle");
    end
end
