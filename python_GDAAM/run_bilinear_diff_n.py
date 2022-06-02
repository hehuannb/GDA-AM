#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import argparse
import aa
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
import time
from AA import pytorchAA, numpyAA
from problems import quad2, quad1
from numpy import linalg as LA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams.update({'font.size': 14})
def_colors=(plt.rcParams['axes.prop_cycle'].by_key()['color'])
import seaborn as sns
plt.rcParams['figure.facecolor'] = 'white'


def altAA(problem, x, y, k, lr, maxiter, type2 = True, reg=1e-8):
    loss = problem.loss(x, y)
    lossAA=[]
    lossAA.append(loss)
    aa_wrk = numpyAA(2*n, k, type2=type2, reg=reg)
    print(aa_wrk.apply)
    fp = np.vstack((x, y))
    count = 0
    start = time.time()
    for i in range(maxiter):
        fpprev = np.copy(fp)      
        gx, _ = problem.grad(x, y)
        x_ = x - lr * gx
        _, gy = problem.grad(x_, y)
        y_ = y + lr * gy
        fp = np.vstack((x_, y_))
        fp = aa_wrk.apply(fpprev, fp) 
        x, y = fp[0:n],fp[n:] 
        lo = problem.loss(x, y)
        lossAA.append(lo)   
    end = time.time()
    return lossAA, end - start

def alt_pytorch(problem, x, y, k, lr, maxiter, type2 = True, gamma=1e-26, reg=1e-8, dev='cuda'):
    x = torch.squeeze(torch.tensor(x0, device=dev))
    y = torch.squeeze(torch.tensor(y0, device=dev))
    A = torch.tensor(problem.A, device=dev)
    B = torch.tensor(problem.B, device=dev)
    C = torch.tensor(problem.C, device=dev)
    xtrue = torch.squeeze(torch.tensor(problem.xtrue, device=dev))
    ytrue = torch.squeeze(torch.tensor(problem.ytrue, device=dev))
    def f(x, y):
        f = x.t() @ A @ y + B.t() @ x + C.t() @ y
        return f

    def distance(x, y):
        return torch.norm(x-xtrue)**2 + torch.norm(y-ytrue)**2
    fp = torch.cat((x, y))
    fp.requires_grad=False
    GDAAA_loss = []
    loss = distance(x, y)
    GDAAA_loss.append(loss)
    aa_wrk = pytorchAA(2 * n, k, type2=type2, reg=reg)

    start = time.time()
    for i in range(maxiter):
        fpprev = fp.detach().clone()
        x, y = fpprev[0:n].requires_grad_(True), fpprev[n:].requires_grad_(True)
        loss = f(x, y)
        loss.backward()
        with torch.no_grad():
            x_ = (1-gamma) * x - lr * x.grad
            x.grad = None
            y.grad = None
        loss = f(x_, y)
        loss.backward()
        with torch.no_grad():
            y_ = y + lr * y.grad
            y.grad = None
            x.grad = None
        fp = torch.cat((x_, y_))
        fp = aa_wrk.apply(fpprev, fp)
        dis = distance(x, y)
        GDAAA_loss.append(dis.item())   
        # print(dis.item())
    end = time.time()
    return GDAAA_loss, end-start

if __name__ == "__main__":
    maxiter = 1000
    cond = 20
    n = 100
    markevery = 50
    k = 50
    gammalist = [1e-2, 1e-10, 1e-15, 1e-25, 1e-30, 0]
    problem = quad1(n, cond,spd=True, bc=False)
    x0 = np.random.rand(n,1)
    y0 = np.random.rand(n,1)
    fig, ax = plt.subplots(figsize=(7,7))
    for i, gamma in enumerate(gammalist):
        
        losstorchAA, time2 = alt_pytorch(problem, x0, y0, k, 0.6, maxiter, True, gamma=gamma)
        ax.semilogy(losstorchAA, '-d',label='AltGDA-RAM, ' + r'$\alpha$' +'='+str(gamma),markevery=markevery)# ax.set_ylim([1e-15,1e4])
    ax.legend(loc=0, fontsize=15)
    # ax.set_ylim([1e-28,1e4])
    ax.set_xlabel('Iteration',fontsize=12)
    ax.set_ylabel('Loss',fontsize=12)
    fig.savefig('bipsupplr4.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
    
    
    
    fig, ax = plt.subplots(figsize=(7,7))
    # fig2, ax2 = plt.subplots(figsize=(7,7))
    lrlist =[0.5, 0.4,0.3, 0.3, 0.2]  
    nlist = [100, 500, 1000, 3000, 5000]
    k=100
    lossnp = []
    losstorch = []
    timenp = []
    timetorch = []
    for i, n in enumerate(nlist):
        cond = n/10
        lr = lrlist[i]
        problem = quad1(n, cond,spd=True, bc=False)
        x0 = np.random.rand(n,1)
        y0 = np.random.rand(n,1)
        k+=100
        lossAltAA, time1 = altAA(problem, x0, y0, k, lr, maxiter, True)
        losstorchAA, time2 = alt_pytorch(problem, x0, y0, k, lr, maxiter, True)
        lossnp.append(lossAltAA)
        losstorch.append(losstorchAA)
        timenp.append(time1)
        timetorch.append(time2)

    markevery = 300
    fig, ax = plt.subplots(figsize=(7,7))
    ax.semilogy(lossnp[1], '--',label='numpy, n=500',markevery=markevery)
    ax.semilogy(lossnp[2], '--',label='numpy, n=1000',markevery=markevery)
    ax.semilogy(lossnp[3], '--',label='numpy, n=3000',markevery=markevery)
    ax.semilogy(lossnp[4], '--',label='numpy, n=5000',markevery=markevery)
    ax.semilogy(losstorch[1], '--d',label='pytorch, n=500',markevery=markevery)
    ax.semilogy(losstorch[2], '--d',label='pytorch, n=1000',markevery=markevery)
    ax.semilogy(losstorch[3], '--d',label='pytorch, n=3000',markevery=markevery)
    ax.semilogy(losstorch[4], '--d',label='pytorch, n=5000',markevery=markevery)
    
    ax.legend(loc=0, fontsize=15)
    # ax.set_ylim([1e-4,1e4])
    ax.set_xlabel('Iteration',fontsize=12)
    ax.set_ylabel('Loss',fontsize=12)
    fig.savefig('bipsuppn1.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
    
    
    markevery = 300
    fig, ax = plt.subplots(figsize=(7,7))
    ax.semilogy(np.linspace(0, timenp[1], len(lossnp[1])), lossnp[1],'--',label='numpy, n=500',markevery=markevery)
    ax.semilogy(np.linspace(0, timenp[2], len(lossnp[2])), lossnp[2],'--',label='numpy, n=1000',markevery=markevery)
    ax.semilogy(np.linspace(0, timenp[3], len(lossnp[3])), lossnp[3],'--',label='numpy, n=3000',markevery=markevery)
    ax.semilogy(np.linspace(0, timenp[4], len(lossnp[4])), lossnp[4],'--',label='numpy, n=5000',markevery=markevery)
    ax.semilogy(np.linspace(0, timetorch[1], len(losstorch[1])), losstorch[1],'--d',label='pytorch, n=500',markevery=markevery)
    ax.semilogy(np.linspace(0, timetorch[2], len(losstorch[2])), losstorch[2],'--d',label='pytorch, n=1000',markevery=markevery)
    ax.semilogy(np.linspace(0, timetorch[3], len(losstorch[3])), losstorch[3],'--d',label='pytorch, n=3000',markevery=markevery)
    ax.semilogy(np.linspace(0, timetorch[4], len(losstorch[4])), losstorch[4],'--d',label='pytorch, n=5000',markevery=markevery)
    
    ax.legend(loc=0, fontsize=15)
    # ax.set_ylim([1e-4,1e4])
    ax.set_xlabel('Time(seconds)',fontsize=12)
    ax.set_ylabel('Loss',fontsize=12)
    fig.savefig('bipsuppn2.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
