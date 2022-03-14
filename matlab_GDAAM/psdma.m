B = randn(10,10);
C = randn(10,10);
A = randn(10,10);
I = eye(10,10);
G1 = [I - B*B', -A;A', I - C*C'];
%G2 = [B*B', A;-A', C*C']; 
G = [I - B, -A;A', I-C];
