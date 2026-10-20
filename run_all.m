function run_all(phases)
% RUN_ALL  Run the full benchmark grid into results, Phase 1 before Phase 2.
%
%   run_all()                 runs 'phase1' then 'polished'
%   run_all({'phase1'})       runs one phase
%
%   Calls run_benchmark chunk by chunk (one solver per chunk). run_benchmark skips cells already in
%   the CSV, so an interrupted run resumes where it stopped. Progress lines go to the console and to
%   results/run_all.log.

    if nargin < 1, phases = {'phase1', 'polished'}; end
    here = fileparts(mfilename('fullpath'));
    out_dir = fullfile(here, 'results');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    logf = fullfile(out_dir, 'run_all.log');
    n_solvers = 41;
    for p = 1:numel(phases)
        note(logf, sprintf('[%s] start %s', datestr(now, 31), phases{p}));
        for c = 1:n_solvers
            t0 = tic;
            try
                run_benchmark(phases{p}, c, 1);
                note(logf, sprintf('[%s] %s solver %d/%d done in %.0f s', datestr(now, 31), phases{p}, c, n_solvers, toc(t0)));
            catch ME
                note(logf, sprintf('[%s] %s solver %d/%d ERROR %s', datestr(now, 31), phases{p}, c, n_solvers, ME.message));
            end
        end
        note(logf, sprintf('[%s] end %s', datestr(now, 31), phases{p}));
    end
end

function note(logf, s)
    fprintf('%s\n', s);
    fid = fopen(logf, 'a'); fprintf(fid, '%s\n', s); fclose(fid);
end
