warning('off')
clear all; close all; clc
disp('Generate random matrix')
n = 100;
A = rand(n);
A = A/norm(A);
b = rand(n,1); 
c = rand(n,1);
xinit = rand(n,1);
yinit = rand(n,1);
fp0 = [xinit;yinit];
xtrue = -A'\c;
ytrue = -A\b;
itmax = 100000;
atol = 1e-5;
print = 1000;
lrlist = [0.1, 1, 2];
mMax = 10;
cyan        = [0.2 0.8 0.8];
brown       = [0.2 0 0];
orange      = [1 0.5 0];
blue        = [0 0.5 1];
green       = [0 0.6 0.3];
red         = [1 0.2 0.2];
disp('OG');
OG_gy = A'* xinit + b;
OG_gx = A * xinit + c;
oldg =[OG_gx;OG_gy];
F2 = @(x, oldg) OG_fp(x,oldg,A,b, c, n,1);
[x2,i2,res_og, rest_og] = OG(F2,[xtrue;ytrue], fp0,oldg,itmax,atol,print);
OG_gy = A'* xinit + b;
OG_gx = A * xinit + c;
oldg =[OG_gx;OG_gy];
F2 = @(x, oldg) OG_fp(x,oldg,A,b, c, n,2);
[x1,iter1,res_og2, rest_og2] = OG(F2,[xtrue;ytrue], fp0,oldg,itmax,atol,print);
colors = [cyan;brown;red;orange;blue];
for i =1:size(lrlist,2)
    lr = lrlist(i);
    disp('Sim-AAQR');
    F_GDA = @(x) simGDA(x,n,lr, A, b, c);
    [~,~,res_sim{i}, rest_sim{i}] = walkerQR(F_GDA,[xtrue;ytrue], fp0,mMax,itmax,atol,0,0, print);
    disp('Alt-AAQR');
    F_GDA = @(x) altGDA(x,n,lr, A, b, c);
    [~,~,res_alt{i}, rest_alt{i}] = walkerQR(F_GDA,[xtrue;ytrue], fp0,mMax,itmax,atol,0,0, print);    
end
interval=50;
fig = figure;clf
semilogy(res_og(:,1),res_og(:,2),'--*','color',colors(5,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_og))
hold on
semilogy(res_og2(:,1),res_og2(:,2),'-d','color',colors(5,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_og2))
semilogy(res_sim{1}(:,1),res_sim{1}(:,2),'->', 'color',colors(1,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sim{1}));
semilogy(res_sim{2}(:,1),res_sim{2}(:,2),'-+', 'color',colors(2,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sim{2}));
semilogy(res_sim{3}(:,1),res_sim{3}(:,2),'-s', 'color',colors(3,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sim{3}));
% semilogy(res_sim{4}(:,1),res_sim{4}(:,2),'-o', 'color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_sim{4}));
semilogy(res_alt{1}(:,1),res_alt{1}(:,2),'-->','color',colors(1,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{1}));
semilogy(res_alt{2}(:,1),res_alt{2}(:,2),'--+','color',colors(2,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{2}));
semilogy(res_alt{3}(:,1),res_alt{3}(:,2),'--s','color',colors(3,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{3}));
% semilogy(res_alt{4}(:,1),res_alt{4}(:,2),'--o','color',colors(4,:),'linewidth',2,'MarkerSize',10,'MarkerIndices', 1:interval:length(res_alt{4}));
h1=legend('OG: \eta=1', 'OG: \eta=2','SimGDA-AM: \eta=0.1','SimGDA-AM: \eta=1', 'SimGDA-AM: \eta=2'...
    ,'AltGDA-AM: \eta=0.1','AltGDA-AM: \eta=1', 'AltGDA-AM: \eta=2','Location', 'Best');
xlabel('Iteration')
ylabel('Distance norm (log-scale)')
set(gcf,'paperpositionmode','auto')
set(gca,'FontSize',18)
ylim([1e-6, 2e4]);
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) 3*200, 2.5*200]); %<- Set size
print -depsc -r500 largelr2


