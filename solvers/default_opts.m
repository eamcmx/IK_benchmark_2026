function opts = default_opts(varargin)
% DEFAULT_OPTS  Common harness options. Override fields via name/value pairs.
%   opts = default_opts() returns the defaults.
%   opts = default_opts('max_iter',200,'tol',1e-8) overrides specific fields.
%
%   Rules:
%     - Each solver uses its own paper's population size and coefficients; the harness does not
%       set a population size.
%     - Every solver gets the same budget of opts.budget forward-kinematics calls per waypoint,
%       counted by the harness (bench_run); population solvers size their generations with
%       ikb_budget_left(opts).
%     - The Phase-2 polish damping is opts.dls_lambda. There is deliberately no opts.lambda, so a
%       solver's own lambda is never overwritten by a harness value.

    opts.max_iter   = 200;        % iteration cap for iterative (Jacobian-type) solvers
    opts.tol_pos    = 1e-6;       % position tolerance (m)
    opts.tol_rot    = 1e-6;       % rotation tolerance (rad)
    opts.dls_lambda = 1e-2;       % damping of the universal Phase-2 DLS polish
    opts.alpha0     = 1.0;        % step scale (JT, GN line search start)
    opts.fd_step    = 1e-6;       % finite-difference step (finite-difference Newton)
    opts.respect_limits = true;   % clamp to joint limits each iter (uniform across the benchmark)
    opts.verbose    = false;
    opts.budget     = 2000;       % FK calls per waypoint for the base solver (counted by bench_run)

    for k = 1:2:numel(varargin)
        opts.(varargin{k}) = varargin{k+1};
    end
end
