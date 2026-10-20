function T = dh_transform(a, alpha, d, theta)
% DH_TRANSFORM  Single Denavit-Hartenberg transformation matrix (standard DH).
%   T = DH_TRANSFORM(a, alpha, d, theta) returns the 4x4 homogeneous
%   transform from frame i-1 to frame i using the standard DH convention:
%       T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
%
%   Inputs are scalars. Vectorise externally if needed.

ct = cos(theta); st = sin(theta);
ca = cos(alpha); sa = sin(alpha);

T = [ ct, -st*ca,  st*sa, a*ct;
      st,  ct*ca, -ct*sa, a*st;
       0,     sa,     ca,    d;
       0,      0,      0,    1];
end
