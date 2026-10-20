function n = ikb_budget_left(opts)
% IKB_BUDGET_LEFT  Forward-kinematics evaluations a solver may still make on the current waypoint.
%
%   n = ikb_budget_left(opts)
%
%   The benchmark gives every solver the same budget of opts.budget FK calls per waypoint and
%   counts them itself (fk_chain's evaluation counter, reset by bench_run before each waypoint).
%   Population solvers use this to stop before they exceed it: a generation of P evaluations
%   may start only if ikb_budget_left(opts) >= P. Outside the harness the budget is unlimited.

    if isfield(opts, 'ikb_counting') && isequal(opts.ikb_counting, true) ...
            && isfield(opts, 'budget') && ~isempty(opts.budget)
        n = opts.budget - fk_chain('ikb_count');
    else
        n = Inf;
    end
end
