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


def simgd(problem, x0, y0, k, lr, maxiter, type2 = False):
    loss = problem.loss(x0, y0)
    lossalt=[]
    lossalt.append(loss)
    x, y = x0, y0
    for i in range( maxiter):
        gx, gy = problem.grad(x, y)
        x = x - lr * gx
        y = y + lr * gy
        lo = problem.loss(x, y)
        lossalt.append(lo)    
    return lossalt


def altgd(problem, x0, y0, k, lr, maxiter, type2 = False):
    loss = problem.loss(x0, y0)
    lossalt=[]
    lossalt.append(loss)
    x, y = x0, y0
    for i in range( maxiter):
        gx, _ = problem.grad(x, y)
        x = x - lr * gx
        _, gy = problem.grad(x, y)
        y = y + lr * gy
        lo = problem.loss(x, y)
        lossalt.append(lo)    
    return lossalt

def eg(problem, x0, y0, k, lr, maxiter, type2 = False):
    loss = problem.loss(x0, y0)
    losseg = []
    losseg.append(loss)
    x, y = x0, y0
    for i in range( maxiter):
        gx, gy = problem.grad(x, y)
        x_ = x - lr * gx
        y_ = y + lr * gy
        gx, gy = problem.grad(x_, y_)
        x = x - lr * gx
        y = y + lr * gy
        lo = problem.loss(x, y)
        losseg.append(lo)    
    return losseg


def omd(problem, x0, y0, k, lr, maxiter, type2 = False):
    loss = problem.loss(x0, y0)
    lossomd=[]
    lossomd.append(loss)
    x, y = x0, y0
    x_l, y_l = 0.5*x0, 0.5*y0
    g_xl, g_yl = problem.grad(x_l,y_l)
    for i in range(maxiter):
        g_x, g_y = problem.grad(x,y)
        x = x - 2 * lr * g_x + lr * g_xl
        y = y + 2 * lr * g_y - lr * g_yl
        g_xl, g_yl =  g_x, g_y
        lo = problem.loss(x, y)
        lossomd.append(lo)    
    return lossomd


def altAA(problem, x, y, k, lr, maxiter, type2 = False, reg=1e-9):
    loss = problem.loss(x, y)
    lossAA=[]
    lossAA.append(loss)
    aa_wrk = numpyAA(2*n, k, type2=type2, reg=reg)
    print(aa_wrk.apply)
    fp = np.vstack((x, y))
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
    return lossAA


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
    k = 50
    n = 100
    lr = 0.1
    cond = 5
    maxiter = 1000
    problem = quad1(n, cond,spd=True, bc=True)
    x0 = np.random.rand(n,1)
    y0 = np.random.rand(n,1)
    losseg = eg(problem, x0, y0, k, 0.3, maxiter, True)
    losseg2 = eg(problem, x0, y0, k, 0.6, maxiter, True)
    lossSimAA = simAA(problem, x0, y0, k, lr, maxiter, True)
    lossaltAA = altAA(problem, x0, y0, k, lr, maxiter, True)
    lr = 0.3
    lossSimAA1 = simAA(problem, x0, y0, k, lr, maxiter, True)
    lossaltAA1 = altAA(problem, x0, y0, k, lr, maxiter, True)
    lr = 0.6
    lossSimAA2 = simAA(problem, x0, y0, k, lr, maxiter, True)
    lossaltAA2 = altAA(problem, x0, y0, k, lr, maxiter, True)
    
    
    
    fig, ax = plt.subplots(figsize=(7,7))
    markevery = 50
    ax.semilogy(losseg, 'k-^', label='EG, lr=0.3',markevery=markevery)
    ax.semilogy(losseg2, 'k-o', label='EG, lr=0.6',markevery=markevery)
    ax.semilogy(lossSimAA, '-->',label='SimGDA-RAM, lr=0.1',markevery=markevery)
    ax.semilogy(lossSimAA1, '-->',label='SimGDA-RAM, lr=0.3',markevery=markevery)
    ax.semilogy(lossSimAA2, '-->',label='SimGDA-RAM, lr=0.6',markevery=markevery)
    ax.semilogy(lossaltAA, '-d',label='AltGDA-RAM, lr=0.1',markevery=markevery)
    ax.semilogy(lossaltAA1, '-d',label='AltGDA-RAM, lr=0.3',markevery=markevery)
    ax.semilogy(lossaltAA2, '-d',label='AltGDA-RAM, lr=0.6',markevery=markevery)
    
    ax.set_ylim([1e-28,1e5])
    ax.legend(loc=0, fontsize=15)
    ax.set_xlabel('Iteration',fontsize=12)
    ax.set_ylabel('Loss',fontsize=12)
    # fig.savefig('bipsupplr.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
    
    fig, ax = plt.subplots(figsize=(7,7))
    lrlist = [0.01, 0.05, 0.1, 0.3, 0.6, 0.8]
    for lr in lrlist:
        lossSimAA = simAA(problem, x0, y0, k, lr, maxiter, True)
        ax.semilogy(lossSimAA, '-->',label='SimGDA-RAM, lr='+str(lr),markevery=markevery)# ax.set_ylim([1e-15,1e4])
    ax.legend(loc=0, fontsize=15)
    ax.set_ylim([1e-28,1e4])
    ax.set_xlabel('Iteration',fontsize=12)
    ax.set_ylabel('Loss',fontsize=12)
    # fig.savefig('bipsupplr2.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
    
    
    fig, ax = plt.subplots(figsize=(7,7))
    lrlist = [0.01, 0.05, 0.1, 0.3, 0.6, 0.8]
    for lr in lrlist:
        lossaltAA = altAA(problem, x0, y0, k, lr, maxiter, True)
        ax.semilogy(lossaltAA, '-d',label='AltGDA-RAM, lr='+str(lr),markevery=markevery)# ax.set_ylim([1e-15,1e4])
    ax.legend(loc=0, fontsize=15)
    ax.set_ylim([1e-28,1e4])
    ax.set_xlabel('Iteration',fontsize=12)
    ax.set_ylabel('Loss',fontsize=12)
    # fig.savefig('bipsupplr3.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)