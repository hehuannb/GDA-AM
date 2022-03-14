clear all; close all; clc
disp('Generate random matrix')
n = 100;


steps = -7:7;
[x1, x2] = meshgrid(steps, steps);
fx = x1.^2 + x2.^2;
contour(x1, x2, fx, 40);
colormap jet


xinit = randn(1);
yinit = randn(1);
fp0 = [xinit;yinit];
xtrue = 0;
ytrue = 0;
mMax = 10;
itmax = 10000;
atol = 1e-5;
lr = 1;
print = 1;
rega = 0;

disp('simAA');


