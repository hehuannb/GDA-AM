#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import math
import scipy 
from scipy import linalg
from numpy import linalg as LA
from pyblas.level1 import dnrm2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.DoubleTensor')

class pytorchAA:
    """Anderson acceleration as described by Walker and Ni in doi:10.2307/23074353."""

    def __init__(self, dimension, depth, type2=True, reg=1e-8):

        self._dimension = dimension
        self._depth = depth
        self.reg = reg * torch.eye(self._depth, device='cpu')
        self.Y = torch.zeros((self._dimension, self._depth), device = device) # changes in increments
        self.S = torch.zeros((self._dimension, self._depth), device = device) # changes in fixed point applications
        self.xTx = torch.zeros((self._depth, self._depth), device = device) 
        self.it =0
        if type2==False:
            self.apply = self.type1
        else:
            self.apply = self.type2
            
    def reset(self):
        self._Fk = torch.zeros((self._dimension, self._depth), requires_grad=False, device = device) # changes in increments
        self._Gk = torch.zeros((self._dimension, self._depth), requires_grad=False, device = device) # changes in fixed point applications
        self.it = 0
            
    def type2(self, x , fx ):
        mk = min(self.it, self._depth)
        g = x - fx
        if (mk > 0):
            col = (self.it -1) % self._depth
            y = g -self.gprev
            self.S[:,col] = x - self.xprev
            self.Y[:,col] = y
            self.xTx[col,:] = self.xTx[:,col] = y.matmul(self.Y)
            b = self.Y.t().mv(g)
            r = (torch.norm(self.Y) **2 + torch.norm(self.S) **2).cpu()
            lstsq_solution = linalg.lstsq(self.xTx.cpu()+ self.reg * r, b.cpu())
            gamma = torch.tensor(lstsq_solution[0]).cuda()
            xkp1 = fx - (self.S-self.Y) @ gamma
            self.it +=1
            self.xprev = x.detach().clone()
            self.gprev = x-fx
        else:
            xkp1 = fx
            self.it +=1
            self.xprev = x.detach().clone()
            self.gprev = x-fx
        return xkp1
    
    def type1(self, x , fx ):
        mk = min(self.it, self._depth)
        g = x - fx
        if (mk > 0):
            col = (self.it -1) % self._depth
            s = x - self.xprev
            y = g -self.gprev
            self.S[:,col] = s
            self.Y[:,col] = y
            self.xTx[:,col] = y.matmul(self.S).t()
            self.xTx[col, :] = s.matmul(self.Y)
            b = self.S.t().mv(g)
            r = (torch.norm(self.Y) **2 + torch.norm(self.S) **2).cpu()
            lstsq_solution = linalg.lstsq(self.xTx.cpu()+ self.reg * r, b.cpu())
            gamma = torch.tensor(lstsq_solution[0]).cuda()
            xkp1 = fx - (self.S-self.Y) @ gamma
            self.it +=1
            self.xprev = x.detach().clone()
            self.gprev = x-fx
        else:
            xkp1 = fx
            self.it +=1
            self.xprev = x.detach().clone()
            self.gprev = x-fx
        return xkp1
    
    

    
    

class numpyAA:
    def __init__(self, dimension, depth, type2=True,reg=1e-7):
        self._dimension = dimension
        self._depth = depth
        self.xTx = None
        self.reg = reg
        self.Y = np.zeros((self._dimension, self._depth)) # changes in increments
        self.S = np.zeros((self._dimension, self._depth)) # changes in fixed point applications
        self.xTx = np.zeros((self._depth, self._depth))
        self.it = 0
        if type2:
            self.apply = self.type2
        else:
            self.apply = self.type1
        
    def reset(self):
        self.Y = np.zeros((self._dimension, self._depth)) # changes in increments
        self.S = np.zeros((self._dimension, self._depth)) # changes in fixed point applications
        self.xTx = np.zeros((self._depth, self._depth))
        self.it = 0
        
        
    def type2(self, x : np.ndarray, fx : np.ndarray) -> np.ndarray:

        g = x[:,0] -fx[:,0]
        mk = min(self.it, self._depth)
        if (self.it > 0):
            # Build matrices of changes
            col = (self.it -1) % self._depth
            y = g - self.gprev
            self.S[:,col] = x[:,0] - self.xprev[:,0]
            self.Y[:,col] = y
            A = self.Y[:,0:mk].transpose() @ self.Y[:,0:mk]
            b = self.Y[:,0:mk].transpose()@ g
            normS = dnrm2(self._dimension, self.S[:,0:mk],1)
            normY = dnrm2(self._dimension, self.Y[:,0:mk],1)
            reg = normS**2 + normY**2
#             try:
#                 res =  scipy.linalg.lapack.dgesv(A + self.reg * reg * np.eye(mk), b)
#                 gamma_k = res[2]
#             except scipy.linalg.LinAlgError as e:
#                 if 'Singular matrix' in str(e):
            lstsq_solution = linalg.lstsq(A + self.reg  * reg * np.eye(mk), b)
            gamma_k = lstsq_solution[0]
            xkp1 = fx - np.dot(self.S[:,0:mk] - self.Y[:,0:mk], gamma_k)[:, np.newaxis]
            self.it +=1
            self.xprev = x.copy()
            self.gprev = x[:,0] - fx[:,0]

        else:
            xkp1 = fx
            self.it +=1
            self.xprev = x.copy()
            self.gprev = x[:,0] - fx[:,0]
        
        return xkp1
    
    def type1(self, x : np.ndarray, fx : np.ndarray) -> np.ndarray:
        
        
        g = x[:,0] -fx[:,0]
        mk = min(self.it, self._depth)
        if (self.it > 0):
            # Build matrices of changes
            col = (self.it -1) % self._depth
            s = x[:,0] - self.xprev[:,0]
            self.S[:,col] = s
            self.Y[:,col] = g - self.gprev
            A = self.S[:,0:mk].transpose() @ self.Y[:,0:mk]
            b = self.S[:,0:mk].transpose()@ g
            normS = dnrm2(self._dimension, self.S[:,0:mk],1)
            normY = dnrm2(self._dimension, self.Y[:,0:mk],1)
            reg = normS**2 + normY**2
            lstsq_solution = linalg.lstsq(A + self.reg * reg * np.eye(mk), b)
            gamma_k = lstsq_solution[0]
            xkp1 = fx - np.dot(self.S[:,0:mk] - self.Y[:,0:mk], gamma_k)[:, np.newaxis]
            self.it +=1
            self.xprev = x.copy()
            self.gprev = x[:,0] - fx[:,0]
            
        else:
            xkp1 = fx
            self.it +=1
            self.xprev = x.copy()
            self.gprev = x[:,0] - fx[:,0]
        
        return xkp1
    
    
