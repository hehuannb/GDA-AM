clear all; close all; clc
disp('Generate random matrix')
addpath('D:\GDA-AM\Gradient-Descent-Ascent-with-Anderson-Acceleration--GDA-AM-\matlab_GDAAM')
n = 100;
A = randn(n);
A = A/norm(A);
b = randn(n,1); 
c = randn(n,1);
xinit = randn(n,1);
yinit = randn(n,1);
fp0 = [xinit;yinit];
xtrue = -A'\c;
ytrue = -A\b;
mMax = 10;  % used for Anderson Mixing, controls the memory table size
itmax = 50000; % Iteration Number
atol = 1e-5;   
lr = 1;
print = 100;
rega = 0;


disp('simAA');
F_sim = @(x) simGDA(x,n,0.001*lr, A, b, c);
[~,~,res_sim, rest_sim] = GDA(F_sim,[xtrue;ytrue],fp0,itmax,atol,print);
disp('altAA');
F_alt = @(x) altGDA(x, n, lr, A, b, c);
[~,~,res_alt, rest_alt] = GDA(F_alt,[xtrue;ytrue],fp0,itmax,atol,print);
disp('OG');
OG_gy = A'* xinit + b;
OG_gx = A * xinit + c;
oldg =[OG_gx;OG_gy];
F2 = @(x, oldg) OG_fp(x,oldg,A,b, c, n,lr/2);
[x1,iter1,res_og, rest_og] = OG(F2,[xtrue;ytrue], fp0,oldg,itmax,atol,print);
disp('simEG');
F_EG = @(x) simEG(x,n,lr, A, b, c);
[~,~,res_eg, rest_eg] = GDA(F_EG,[xtrue;ytrue],fp0,itmax,atol,print);
disp('simEG-NM'); % EG with Negative Momentum
F_EG = @(x,nm) simEGNM(x,nm, n,lr, A, b, c);
[~,~,res_egnm, rest_egnm] = GDANM(F_EG,[xtrue;ytrue],fp0,itmax,atol,print);
disp('simEG-PM'); % EG with Positive Momentum
F_EG = @(x,nm) simEGPM(x,nm, n,lr, A, b, c, 0.3);
[~,~,res_egpm, rest_egpm] = GDANM(F_EG,[xtrue;ytrue],fp0,itmax,atol,print);

disp('Sim-GDA-AM');
F_GDA = @(x) x - lr * simGDA(x,n,lr, A, b, c);
[~,~,res_sqr, rest_sqr] = walkerQR(F_GDA,[xtrue;ytrue], fp0,mMax,itmax,atol,0,0, print);
disp('Alt-GDA-AM');
F_GDA = @(x) altGDA(x,n,lr, A, b, c);
[~,~,res_aqr, rest_aqr] = walkerQR(F_GDA,[xtrue;ytrue], fp0,mMax,itmax,atol,0,0, print);

cyan        = [0.2 0.8 0.8];
brown       = [0.2 0 0];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];
interval = 200;
% create new figure

fig = figure; clf
semilogy(res_sim(:,1),res_sim(:,2),'-+', 'color',cyan,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sim))
hold on
semilogy(res_alt(:,1),res_alt(:,2),'--x','color',brown,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt))
semilogy(res_og(:,1),res_og(:,2),'-*','color',orange,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_og))
semilogy(res_eg(:,1),res_eg(:,2),'-.>','color',blue,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_eg))
semilogy(res_egnm(:,1),res_egnm(:,2),'-.d','color','#00008B','linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_egnm))
semilogy(res_egpm(:,1),res_egpm(:,2),'--v','color','#00008B','linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_egpm))
semilogy(res_sqr(:,1),res_sqr(:,2),'-o','color',red,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sqr))
semilogy(res_aqr(:,1),res_aqr(:,2),'-s','color',green,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_aqr))
h1=legend('SimGDA','AltGDA','OG','EG','EG-NM','EG-PM','SimGDA-AM','AltGDA-AM','Location', 'Best');
xlabel('Iteration')
ylabel('Distance norms (log-scale)')

set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',18)
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) 3*200, 2.5*200]); 
% print -depsc -r500 iterpm

fig = figure; clf
semilogy(rest_sim(:,1),res_sim(:,2),'-+', 'color',cyan,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sim))
hold on
semilogy(rest_alt(:,1),res_alt(:,2),'--x','color',brown,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt))
semilogy(rest_og(:,1),res_og(:,2),'-*','color',orange,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_og))
semilogy(rest_eg(:,1),res_eg(:,2),'-.>','color',blue,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_eg))
semilogy(rest_egnm(:,1),res_egnm(:,2),'-.d','color','#00008B','linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_egnm))
semilogy(rest_egpm(:,1),res_egpm(:,2),'--v','color','#00008B','linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_egpm))
semilogy(rest_sqr(:,1),res_sqr(:,2),'-o','color',red,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sqr))
semilogy(rest_aqr(:,1),res_aqr(:,2),'-s','color',green,'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_aqr))
h1=legend('SimGDA','AltGDA','OG','EG','EG-NM','EG-PM','SimGDA-AM','AltGDA-AM','Location', 'Best');
xlabel('Time (Seconds)')
ylabel('Distance norms (log-scale)')

set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',18)
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) 3*200, 2.5*200]); 
exportgraphics(gcf,'barchart.png','Resolution',300)