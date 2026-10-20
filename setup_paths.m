function setup_paths()
% SETUP_PATHS  Add the benchmark folders to the MATLAB path. Run once per session from any folder.

    here = fileparts(mfilename('fullpath'));
    for d = {'utils', 'robots', 'solvers', 'trajectories', 'bench', 'figures_pipeline', 'tests'}
        if exist(fullfile(here, d{1}), 'dir'), addpath(fullfile(here, d{1})); end
    end
    fprintf('IK benchmark paths added.\n');
end
