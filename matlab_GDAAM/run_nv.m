warning('off')
clear all; close all; clc
disp('Generate random matrix')
n = 100;
A = randn(n);
A = A/norm(A);%
b = randn(n,1); 
c = randn(n,1);

xinit = randn(n,1);
yinit = randn(n,1);
fp0 = [xinit;yinit];
xtrue = -A'\c;
ytrue = -A\b;

itmax = 100000;
atol = 1e-5;
lr =0.01;
print = 1000;
rega=0;

klist = [3,50,100];

cyan        = [0.2 0.8 0.8];
brown       = [0.2 0 0];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];
res_alt_L = [];
res_alt_T = [];
res_sim_L = [];
res_sim_T = [];
colors = [cyan;brown;red;orange;green;blue];
for i =1:size(klist,2)
    mMax = klist(i);
    disp('AAQR');
    F_GDA = @(x) simGDA(x,n,lr, A, b, c);
    [~,~,res_sim{i}, rest_sim{i}] = walkerQR(F_GDA,[xtrue;ytrue], fp0,mMax,itmax,atol,0,0, print);
    disp('AAQR');
    F_GDA = @(x) altGDA(x,n,lr, A, b, c);
    [~,~,res_alt{i}, rest_alt{i}] = walkerQR(F_GDA,[xtrue;ytrue], fp0,mMax,itmax,atol,0,0, print);    
end
disp('OG');
OG_gy = A'* xinit + b;
OG_gx = A * xinit + c;
oldg =[OG_gx;OG_gy];
F2 = @(x, oldg) OG_fp(x,oldg,A,b, c, n,1);
[x1,iter1,res_og, rest_og] = OG(F2,[xtrue;ytrue], fp0,oldg,itmax,atol,print);
disp('simEG');
F_EG = @(x) simEG(x,n,1, A, b, c);
[~,~,res_eg, rest_eg] = GDA(F_EG,[xtrue;ytrue],fp0,itmax,atol,print);
interval = 100-1;
fig = figure;clf
semilogy(res_og(:,1),res_og(:,2),'--d', 'color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_og));
hold on
semilogy(res_eg(:,1),res_eg(:,2),'-->', 'color',colors(6,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_eg));
semilogy(res_sim{1}(:,1),res_sim{1}(:,2),'-s', 'color',colors(3,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(rest_sim{1}));
semilogy(res_alt{1}(:,1),res_alt{1}(:,2),'--o','color',colors(3,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{1}));
semilogy(res_sim{2}(:,1),res_sim{2}(:,2),'-s', 'color',colors(2,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(rest_sim{2}));
semilogy(res_alt{2}(:,1),res_alt{2}(:,2),'--o','color',colors(2,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{2}));
semilogy(res_sim{3}(:,1),res_sim{3}(:,2),'-s', 'color',colors(5,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(rest_sim{3}));
semilogy(res_alt{3}(:,1),res_alt{3}(:,2),'--o','color',colors(5,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{3}));
% semilogy(res_sim{4}(:,1),res_sim{4}(:,2),'--s', 'color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(rest_sim{4}));
% semilogy(res_alt{4}(:,1),res_alt{4}(:,2),'-o','color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{4}));
h1=legend('OG','EG','SimAM p=10','AltAM p=10',...
            'SimAM p=50','AltAM p=50','SimAM p=100','AltAM p=100','Location', 'Best');
xlabel('Iteration')
ylabel('Residual norm (log-scale)')
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',18)
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) 3*200, 2.5*200]); %<- Set size
print -depsc -r800 diffk

fig = figure;clf
semilogy(rest_og(:,1),rest_og(:,2),'--d', 'color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(rest_og));
hold on
semilogy(rest_eg(:,1),rest_eg(:,2),'-->', 'color',colors(6,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(rest_eg));
semilogy(rest_sim{1}(:,1),rest_sim{1}(:,2),'-s', 'color',colors(3,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(rest_sim{1}));
semilogy(rest_alt{1}(:,1),rest_alt{1}(:,2),'--o','color',colors(3,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{1}));
semilogy(rest_sim{2}(:,1),rest_sim{2}(:,2),'-s', 'color',colors(2,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(rest_sim{1}))
semilogy(rest_alt{2}(:,1),res_alt{2}(:,2),'--o','color',colors(2,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{2}));
semilogy(rest_sim{3}(:,1),res_sim{3}(:,2),'-s', 'color',colors(5,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(rest_sim{3}));
semilogy(rest_alt{3}(:,1),res_alt{3}(:,2),'--o','color',colors(5,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{3}));
% semilogy(rest_sim{4}(:,1),res_sim{4}(:,2),'--s', 'color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:10:length(rest_sim{4}));
% semilogy(rest_alt{4}(:,1),res_alt{4}(:,2),'-o','color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{4}));
h1=legend('OG','EG','SimAM p=10','AltAM p=10',...
            'SimAM p=50','AltAM p=50','SimAM p=100','AltAM p=100','Location', 'Best');
xlabel('Time (Seconds)')
ylabel('Residual norm (log-scale)')
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',18)
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) 3*200, 2.5*200]); %<- Set size
print -depsc -r500 suppdiffk2

