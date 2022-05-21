warning('off')
clear all; close all; clc
disp('Generate random matrix')
addpath('D:\GDA-AM\Gradient-Descent-Ascent-with-Anderson-Acceleration--GDA-AM-\matlab_GDAAM')
n = 100;
A = randn(n);
A = A/norm(A);
b = randn(n,1); 
c = randn(n,1);
xinit = rand(n,1);
yinit = rand(n,1);
fp0 = [xinit;yinit];
xtrue = -A'\c;
ytrue = -A\b;
itmax = 1000000;
atol = 1e-5;
print = 1000;
lrlist = [0.1, 0.5, 1];
mMax = 10;
cyan        = [0.2 0.8 0.8];
brown       = [0.2 0 0];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];

colors = [cyan;brown;red;orange;blue;green];
for i =1:size(lrlist,2)
    lr = lrlist(i);
    disp('Sim-AAQR');
    F_GDA = @(x) simGDA(x,n,lr, A, b, c);
    [~,~,res_sim{i}, rest_sim{i}] = walkerQR(F_GDA,[xtrue;ytrue], fp0,mMax,itmax,atol,0,0, print);
    F_EG = @(x,nm) simEGPM(x,nm, n,lr, A, b, c, 0.3);
    [~,~,res_egpm{i}, rest_egpm{i}] = GDANM(F_EG,[xtrue;ytrue],fp0,itmax,atol,print);
end
interval=50;
fig = figure;clf

semilogy(res_egpm{1}(:,1),res_egpm{1}(:,2),'-->', 'color',colors(1,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_egpm{1}));
hold on
semilogy(res_egpm{2}(:,1),res_egpm{2}(:,2),'--+', 'color',colors(2,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_egpm{2}));
semilogy(res_egpm{3}(:,1),res_egpm{3}(:,2),'--s', 'color',colors(3,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_egpm{3}));
% semilogy(res_sim{4}(:,1),res_sim{4}(:,2),'-o', 'color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sim{4}));
semilogy(res_sim{1}(:,1),res_sim{1}(:,2),'->','color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sim{1}));
semilogy(res_sim{2}(:,1),res_sim{2}(:,2),'-+','color',colors(5,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sim{2}));
semilogy(res_sim{3}(:,1),res_sim{3}(:,2),'-s','color',colors(6,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sim{3}));
% semilogy(res_alt{4}(:,1),res_alt{4}(:,2),'--o','color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{4}));
h1=legend('EG-PM: \eta=0.2', 'EG-PM: \eta=0.5','EG-PM: \eta=1','SimGDA-AM: \eta=0.2', 'SimGDA-AM: \eta=0.5'...
    ,'SimGDA-AM: \eta=1','Location', 'Best');
xlabel('Iteration')
ylabel('Distance norm (log-scale)')
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',18)
ylim([1e-6, 2e4]);
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) 3*200, 2.5*200]); %<- Set size
print -dpng -r500 pmlr2

lr = 1;
momlist = [0.1, 0.3, 0.5];
for i =1:size(momlist,2)
    mom = momlist(i);
    disp('Sim-AAQR');
    F_GDA = @(x) simGDA(x,n,lr, A, b, c);
    [~,~,res_sim{i}, rest_sim{i}] = walkerQR(F_GDA,[xtrue;ytrue], fp0,mMax,itmax,atol,0,0, print);
    F_EG = @(x,nm) simEGPM(x,nm, n,lr, A, b, c, mom);
    [~,~,res_egpm{i}, rest_egpm{i}] = GDANM(F_EG,[xtrue;ytrue],fp0,itmax,atol,print);
end
fig = figure;clf

semilogy(res_egpm{1}(:,1),res_egpm{1}(:,2),'-->', 'color',colors(2,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_egpm{1}));
hold on
semilogy(res_egpm{2}(:,1),res_egpm{2}(:,2),'--+', 'color',colors(3,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_egpm{2}));
semilogy(res_egpm{3}(:,1),res_egpm{3}(:,2),'--s', 'color',colors(1,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_egpm{3}));
% semilogy(res_sim{4}(:,1),res_sim{4}(:,2),'-o', 'color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sim{4}));
semilogy(res_sim{1}(:,1),res_sim{1}(:,2),'->','color',colors(6,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sim{1}));
% semilogy(res_alt{4}(:,1),res_alt{4}(:,2),'--o','color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{4}));
h1=legend('EG-PM: \beta=0.1', 'EG-PM: \beta=0.3','EG-PM: \beta=0.5','SimGDA-AM: \eta=1','Location', 'Best');
xlabel('Iteration')
ylabel('Distance norm (log-scale)')
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',18)
ylim([1e-6, 2e4]);
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) 3*200, 2.5*200]); %<- Set size
print -dpng -r500 pmmom2