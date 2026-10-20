function run_benchmark(phase, chunk_idx, chunk_size)
% RUN_BENCHMARK  Driver of the benchmark grid.
%
%   run_benchmark(phase, chunk_idx, chunk_size)
%     phase      : 'phase1' (solvers as-published) or 'polished' (DLS-polish wrapper)
%     chunk_idx  : 1-based chunk index (default 1)
%     chunk_size : solvers per chunk (default 3)
%
%   Runs the full grid  6 robots x 4 trajectories x 41 solvers = 984 cells
%   per phase. Robots: UR5, Panda, Synthetic9, Synthetic11, Synthetic13,
%   IRB120 (ABB IRB 120, spherical wrist, so that the closed-form
%   spherical-wrist solvers are tested on a robot they apply to).
%
%   Solver list (41): 2 baselines explicitly + 39 literature solvers
%   discovered in solvers/.
%
%   Output: results/<phase1|polished>.csv, schema
%     label,phase,robot,traj,success,med_pos_err,med_rot_err,med_t_ms,
%     med_fk,max_fk,med_jac,frac_budget_exceeded,frac_not_applicable,
%     med_t_self_ms,med_polish_iters,med_fk_polish,crashed
%   med_t_ms is the harness wall clock around the whole call (polish included);
%   med_fk / max_fk / med_jac are harness-counted calls of the base solver.
%   success is over the waypoints the solver accepted (NaN if it declared the
%   robot not applicable on every waypoint).
%   Resume-safe (skips cells already in the CSV). A deterministic per-cell
%   RNG seed makes stochastic solvers reproducible.
%
%   Protocol: every solver gets the same budget of
%   2000 FK calls per waypoint and uses its own paper's population size; the
%   harness passes no population size.

if nargin < 1, phase = 'polished'; end
if nargin < 2, chunk_idx  = 1; end
if nargin < 3, chunk_size = 3; end
phase = lower(phase);
assert(any(strcmp(phase,{'phase1','polished'})), 'phase must be phase1 or polished');

this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir,'solvers'), fullfile(this_dir,'utils'), ...
        fullfile(this_dir,'robots'), fullfile(this_dir,'trajectories'), ...
        fullfile(this_dir,'bench'));

% ---- solver list: 2 baselines + 39 literature (scanned from solvers/) ----
solvers = {};
solvers(end+1,:) = {'Newton12-anaJ', @solve_newton12_anaJ};
solvers(end+1,:) = {'PINV-orig',     @solve_pinv_orig};
exclude = {'solve_newton12_anaJ','solve_pinv_orig'};   % the baselines, added above
B = dir(fullfile(this_dir,'solvers','solve_*.m'));
for i = 1:numel(B)
    [~, fn] = fileparts(B(i).name);
    if any(strcmp(fn, exclude)), continue; end
    solvers(end+1,:) = {['Lit:' strrep(fn,'solve_','')], str2func(fn)}; %#ok<AGROW>
end
n_total = size(solvers,1);
assert(n_total == 41, 'expected 41 solvers, got %d', n_total);

i_lo = (chunk_idx-1)*chunk_size + 1;
i_hi = min(chunk_idx*chunk_size, n_total);
if i_lo > n_total, fprintf('Chunk %d beyond end (%d solvers).\n', chunk_idx, n_total); return; end

% ---- grid + options ------------------------------------------------------
robots = {UR5_robot(), Panda_robot(), Synthetic9_robot(), ...
          Synthetic11_robot(), Synthetic13_robot(), IRB120_robot()};
traj_kinds = {'rectangle','circle','lissajous','cuspidal'};
N_wp = 100;
opts = default_opts('max_iter',200,'tol_pos',1e-5,'tol_rot',1e-5,'budget',2000);

out_csv = fullfile(this_dir,'results', ...
                   ternary(strcmp(phase,'phase1'),'phase1.csv','polished.csv'));
if ~exist(fileparts(out_csv),'dir'), mkdir(fileparts(out_csv)); end
if ~exist(out_csv,'file')
    fid = fopen(out_csv,'w');
    fprintf(fid,['label,phase,robot,traj,success,med_pos_err,med_rot_err,med_t_ms,' ...
                 'med_fk,max_fk,med_jac,frac_budget_exceeded,frac_not_applicable,' ...
                 'med_t_self_ms,med_polish_iters,med_fk_polish,crashed\n']);
    fclose(fid);
end

fprintf('[%s] chunk %d: solvers %d..%d of %d  (%d robots x %d trajs)\n', ...
        phase, chunk_idx, i_lo, i_hi, n_total, numel(robots), numel(traj_kinds));
chunk_start = tic;

for k = i_lo:i_hi
    label = solvers{k,1};
    base  = solvers{k,2};

    opts_k = opts;      % same options and budget for every solver

    for r = 1:numel(robots)
        rb = robots{r};
        for t = 1:numel(traj_kinds)
            if cell_done(out_csv, label, rb.name, traj_kinds{t}), continue; end

            % deterministic per-cell seed -> reproducible stochastic solvers
            rng(seed_of(label, rb.name, traj_kinds{t}), 'twister');

            Ts = make_default_traj(rb, traj_kinds{t}, N_wp);
            if strcmp(phase,'phase1')
                fn = base;                                 % as-published
            else
                fn = @(r_,Td,q_,o) solve_polished(base, r_, Td, q_, o);  % DLS polish
            end

            t_cell = tic; crashed = false;
            su=NaN; me=NaN; mr=NaN; mt=NaN; mfk=NaN; xfk=NaN; mjac=NaN;
            fbx=NaN; fna=NaN; mts=NaN; mpi=NaN; mfkp=NaN;
            try
                res = bench_run(rb, fn, Ts, rb.q_home, opts_k);
                ok  = ~res.not_applicable;
                if any(ok), su = mean(res.converged(ok)); end
                me   = median(res.errs_pos);
                mr   = median(res.errs_rot);
                mt   = 1000*median(res.times);
                mfk  = median(res.fk_base);
                xfk  = max(res.fk_base);
                mjac = median(res.jac_base);
                fbx  = mean(res.budget_exceeded);
                fna  = mean(res.not_applicable);
                mts  = 1000*median(res.times_self, 'omitnan');
                if strcmp(phase,'polished')
                    pv = zeros(1,numel(res.info));
                    for jj=1:numel(res.info)
                        if isfield(res.info{jj},'polish_iters'), pv(jj)=res.info{jj}.polish_iters; end
                    end
                    mpi  = median(pv);
                    mfkp = median(res.fk_calls - res.fk_base);
                end
            catch ME
                crashed = true;
                fprintf('  CRASH %s/%s/%s: %s\n', label, rb.name, traj_kinds{t}, ME.message);
            end
            fprintf('  %-22s %-12s %-9s succ=%-5s perr=%-9s ms=%-7s fk=%-6s over=%-5s (%.1fs)\n', ...
                label, rb.name, traj_kinds{t}, cs(su,'%.2f'), cs(me,'%.2e'), cs(mt,'%.1f'), ...
                cs(mfk,'%.0f'), cs(fbx,'%.2f'), toc(t_cell));

            fid = fopen(out_csv,'a');
            fprintf(fid, '%s,%s,%s,%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d\n', ...
                label, phase, rb.name, traj_kinds{t}, ...
                ifnan(su), ifnan(me), ifnan(mr), ifnan(mt), ifnan(mfk), ifnan(xfk), ifnan(mjac), ...
                ifnan(fbx), ifnan(fna), ifnan(mts), ifnan(mpi), ifnan(mfkp), double(crashed));
            fclose(fid);
        end
    end
end
fprintf('[%s] chunk %d done in %.1fs\n', phase, chunk_idx, toc(chunk_start));
end

% ---- helpers -------------------------------------------------------------
function out = ternary(c,a,b), if c, out=a; else, out=b; end, end

function s = seed_of(label, robot, traj)
    str = [label '|' robot '|' traj];
    s = uint32(mod(sum(double(str).*(1:numel(str))), 2^31-1));
end

function done = cell_done(out_csv, label, robot, traj)
    done = false;
    if ~exist(out_csv,'file'), return; end
    fid = fopen(out_csv,'r'); fgetl(fid);
    while true
        l = fgetl(fid); if ~ischar(l), break; end
        p = strsplit(l,',');
        if numel(p)>=4 && strcmp(p{1},label) && strcmp(p{3},robot) && strcmp(p{4},traj)
            done = true; break;
        end
    end
    fclose(fid);
end

function s = cs(x, fmt), if isnan(x), s='NaN'; else, s=sprintf(fmt,x); end, end
function v = ifnan(x), if isempty(x)||~isnumeric(x)||isnan(x), v=NaN; else, v=x; end, end
