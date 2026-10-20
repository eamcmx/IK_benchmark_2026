function mask = pareto_min(x, y)
% PARETO_MIN  Lower-left Pareto frontier of a 2-D point set (minimise BOTH).
%   mask = pareto_min(x, y) returns a logical column vector that is true for
%   the points NOT dominated by any other: point i is dominated iff some
%   other point j has x(j) <= x(i) AND y(j) <= y(i), strictly in at least one.
%   Lower-left = better on both axes (e.g. faster AND more accurate).
%
%   O(n^2); n is small (<= a few hundred solvers) so this is fine.

    x = x(:); y = y(:);
    nP = numel(x);
    mask = true(nP, 1);
    for i = 1:nP
        for j = 1:nP
            if j == i, continue; end
            if x(j) <= x(i) && y(j) <= y(i) && (x(j) < x(i) || y(j) < y(i))
                mask(i) = false;
                break;
            end
        end
    end
end
