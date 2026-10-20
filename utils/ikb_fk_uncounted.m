function T = ikb_fk_uncounted(robot, q)
% IKB_FK_UNCOUNTED  End-effector pose of robot at q without touching the FK evaluation counter.
%   Used by the harness (scoring) and by solve_polished (diagnostics), so their own FK
%   evaluations are never charged to the solver being measured.

    if isfield(robot, 'dh')
        tool = eye(4);
        if isfield(robot, 'tool'), tool = robot.tool; end
        T = fk_chain_kernel(q(:), robot.dh, tool);
    else
        T = robot.fk(q(:));
    end
end
