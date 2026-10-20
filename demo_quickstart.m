% DEMO_QUICKSTART  Minimal end-to-end run of the benchmark harness (well under a minute).
%
%   Builds the UR5, takes the first 20 waypoints of the rectangle trajectory and runs, through the
%   benchmark harness (bench_run: FK budget, pose gate, error recomputed from FK):
%     - the two baselines, Newton12-anaJ and PINV-orig;
%     - two single-point literature solvers, PINV and QuIK, as published (Phase 1);
%     - the population solver WOA as published (Phase 1) and with the common DLS polish (Phase 2).
%   Prints one line per run and plots the position error along the path.

clear; close all;
setup_paths();

robot = UR5_robot();
Ts    = make_default_traj(robot, 'rectangle', 100);
Ts    = Ts(:, :, 1:20);
opts  = default_opts('max_iter', 200, 'tol_pos', 1e-5, 'tol_rot', 1e-5, 'budget', 2000);

runs = { ...
    'Newton12-anaJ',    @solve_newton12_anaJ; ...
    'PINV-orig',        @solve_pinv_orig; ...
    'PINV',             @solve_PINV; ...
    'QuIK',             @solve_QuIK; ...
    'WOA (Phase 1)',    @solve_WOA; ...
    'WOA (Phase 2)',    @(r, Td, q, o) solve_polished(@solve_WOA, r, Td, q, o)};

fprintf('\nDemo: UR5, rectangle trajectory, waypoints 1-20, gate 1e-5 m / 1e-5 rad, budget 2000 FK\n');
fprintf('%-16s | %-7s | %-9s | %-11s | %-8s\n', 'Solver', 'Success', 'Median ms', 'Median pos', 'Median FK');
fprintf('-----------------+---------+-----------+-------------+----------\n');
figure('Color', 'w', 'Position', [100, 100, 820, 420]);
for j = 1:size(runs, 1)
    rng(1, 'twister');
    res = bench_run(robot, runs{j, 2}, Ts, robot.q_home, opts);
    fprintf('%-16s | %7.2f | %9.2f | %11.2e | %8d\n', runs{j, 1}, mean(res.converged), ...
            1000 * median(res.times), median(res.errs_pos), round(median(res.fk_base)));
    semilogy(max(res.errs_pos, 1e-17), 'LineWidth', 1.4, 'DisplayName', runs{j, 1}); hold on;
end
yline(1e-5, 'r--', 'gate', 'HandleVisibility', 'off');
xlabel('Waypoint'); ylabel('Position error (m)'); grid on;
title('Quick start: UR5 rectangle, waypoints 1-20');
legend('Location', 'best');
fprintf('\n');
