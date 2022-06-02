#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import argparse
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
plt.rcParams['figure.facecolor'] = 'white'



def altAA(problem, x, y, k, lr, maxiter, type2 = False, reg=0):
    loss = problem.loss(x, y)
    lossAA=[]
    lossAA.append(loss)
    aa_wrk = numpyAA(2*n, k, type2=type2, reg=reg)
    print(aa_wrk.apply)
    fp = np.vstack((x, y))
    tol = 1e-10
    count = 0
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
    return lossAA, i


def simAA(problem, x, y, k, lr, maxiter, type2 = False, reg=1e-8):
    loss = problem.loss(x, y)
    lossAA=[]
    lossAA.append(loss)
    aa_wrk = numpyAA(2*n, k, type2=type2, reg=reg)
    print(aa_wrk.apply)
    fp = np.vstack((x, y))
    for i in range(maxiter):
        fpprev = np.copy(fp)      
        gx, gy = problem.grad(x, y)
        x_ =  x - lr * gx
        y_ = y + lr * gy
        fp = np.vstack((x_, y_))
        fp = aa_wrk.apply(fpprev, fp) 
        x, y = fp[0:n], fp[n:]
        lo = problem.loss(x, y)
        lossAA.append(lo)   
        if i %k ==0:
            aa_wrk = numpyAA(2*n, k, type2=type2, reg=reg)
    return lossAA

if __name__ == "__main__":
    maxiter = 2000
    n = 100
    markevery = 50
    fig, ax = plt.subplots(figsize=(7,7))
    lr = 0.6
    condlist = [5, 20, 40, 60, 80, 100]
    for cond in condlist:
        problem = quad1(n, cond,spd=True, bc=True)
        x0 = np.random.rand(n,1)
        y0 = np.random.rand(n,1)
        lossSimAA = simAA(problem, x0, y0, 100, lr, maxiter, True)
    
        ax.semilogy(lossSimAA, '-->',label='SimGDA-RAM, cond='+str(cond),markevery=markevery)
    ax.legend(loc=0, fontsize=15)
    # ax.set_ylim([1e-28,1e4])
    ax.set_xlabel('Iteration',fontsize=12)
    ax.set_ylabel('Loss',fontsize=12)
    fig.savefig('bipsuppcond2.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
    
    
    maxiter = 2000
    n=100
    markevery = 50
    fig, ax = plt.subplots(figsize=(7,7))
    condlist = [5, 20, 40, 60, 80, 100]
    for cond in condlist:
        problem = quad1(n, cond,spd=True, bc=True)
        x0 = np.random.rand(n,1)
        y0 = np.random.rand(n,1)
        lossaltAA,i = altAA(problem, x0, y0, 50, 0.6, maxiter, True)
        ax.semilogy(lossaltAA, '-->',label='AltGDA-RAM, cond='+str(cond),markevery=markevery)
    ax.legend(loc=0, fontsize=15)
    ax.set_ylim([1e-28,1e4])
    ax.set_xlabel('Iteration',fontsize=12)
    ax.set_ylabel('Loss',fontsize=12)
    fig.savefig('bipsuppcond1.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
    
    
    
    
    maxiter = 5000
    n = 1000
    markevery = 250
    fig, ax = plt.subplots(figsize=(7,7))
    condlist = [50, 100, 200, 300, 500, 700, 900]
    ilist = []
    lr=0.5
    k=50
    for i, cond in enumerate(condlist):
        problem = quad1(n, cond,spd=True, bc=True)
        x0 = np.random.rand(n,1)
        y0 = np.random.rand(n,1)
        # lr *=0.5
        if i>=1:
            # lr=0.7
            k += 50
        lossaltAA, i = altAA(problem, x0, y0, k, lr, maxiter, True)
        ilist.append(i)
        ax.semilogy(lossaltAA, '-->',label='cond='+str(cond)- ,markevery=markevery)
    ax.legend(loc=0, fontsize=15)
    # ax.set_ylim([1e-28,1e4])
    ax.set_xlabel('Iteration',fontsize=12)
    ax.set_ylabel('Loss',fontsize=12)
    fig.savefig('bipsuppcond3.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)