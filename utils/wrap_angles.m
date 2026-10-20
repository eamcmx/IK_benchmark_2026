function q = wrap_angles(q)
% WRAP_ANGLES  Wrap each entry of q to (-pi, pi]. Used at solver exit.
    q = mod(q + pi, 2*pi) - pi;
end
