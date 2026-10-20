function T = ikb_generations(opts, n_init, n_per_gen, T_paper)
% IKB_GENERATIONS  Iteration horizon T for a population solver under the common FK budget.
%
%   T = ikb_generations(opts, n_init, n_per_gen, T_paper)
%
%   Inside the benchmark (opts.ikb_counting true) T is the number of generations the per-waypoint
%   budget allows: floor((opts.budget - n_init) / n_per_gen), where n_init is the number of FK
%   evaluations the initialisation needs and n_per_gen the number one generation needs. Every
%   schedule a paper defines over its iteration count (decreasing weights, shrinking radii, walk
%   lengths) runs over this T, so it completes within the budget.
%   Outside the benchmark the paper's own T_paper is returned.

    if isfield(opts, 'ikb_counting') && isequal(opts.ikb_counting, true) ...
            && isfield(opts, 'budget') && ~isempty(opts.budget) && isfinite(opts.budget)
        T = max(1, floor((opts.budget - n_init) / n_per_gen));
    else
        T = T_paper;
    end
end
