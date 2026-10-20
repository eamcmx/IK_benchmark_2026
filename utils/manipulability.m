function w = manipulability(J)
% MANIPULABILITY  Yoshikawa manipulability index w = sqrt(det(J*J')).
%   Falls to zero at singular configurations. Use as a redundancy quality
%   metric along trajectories.
    M = J * J';
    d = det(M);
    if d < 0
        d = 0;  % numerical guard
    end
    w = sqrt(d);
end
