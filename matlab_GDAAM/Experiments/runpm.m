clear all; close all; clc
disp('Generate random matrix')
addpath('D:\GDA-AM\Gradient-Descent-Ascent-with-Anderson-Acceleration--GDA-AM-\matlab_GDAAM')
n = 100;
A = randn(n);
A = A/norm(A);
b = zeros(n,1); 
c = zeros(n,1);

xinit = randn(n,1);
yinit = randn(n,1);
fp0 = [xinit;yinit];
xtrue = -A'\c;
ytrue = -A\b;
mMax = 10;
itmax = 50000;
atol = 1e-5;
lr = 1;
print = 2000;
rega = 0;

disp('simEG-PM');
F_EG = @(x,nm) simEGPM(x,nm, n,lr, A, b, c);
[~,~,res_egpm, rest_egpm] = GDANM(F_EG,[xtrue;ytrue],fp0,itmax,atol,print);
semilogy(res_egpm(:,1),res_egpm(:,2),'--d','color','#00008B','linewidth',2,'MarkerSize',10,'MarkerIndices', 1:10:length(res_egpm))