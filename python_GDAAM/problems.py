#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import autograd.numpy as np
import torch
import sklearn.datasets
from autograd import grad,jacobian
from numpy import linalg as LA
from scipy.linalg import pinv
import numpy.matlib as mt

# create a row vector of given size
def generate_sym(size):
    A = mt.rand(1,size)

# create a symmetric matrix size * size
    symmA = A.T * A
    return symmA

def gen_cond(n, cond):
    """
    Parameters
    ----------
    n : Matrix size
    cond : Condition number

    Returns
    -------
    P : Return a n by n SPD matrix given a condition number
    """
    cond_P = cond     # Condition number
    log_cond_P = np.log(cond_P)
    exp_vec = np.arange(-log_cond_P/4., log_cond_P * (n)/(4 * (n - 1)), log_cond_P/(2.*(n-1)))
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = LA.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = LA.qr((np.random.rand(n, n) - 5.) * 200)
    P = U.dot(S).dot(V.T)
    P = P.dot(P.T)
    return P


class Base(object):
    def __init__(self):
        self.xopt = None
        self.yopt = None
        self.xrange = None
        self.yrange = None
        self.f = None
        self.dfdx = grad(self.f)
        self.dfdy = grad(self.f, 1)   
        self.d2fdxdx = grad(self.dfdx)
        self.d2fdydy = grad(self.dfdy, 1)
        self.d2fdxdy = grad(self.dfdx, 1)
        self.d2fdydx = grad(self.dfdy)

    def fr(self, x, y):
        "this is used for the baseline model(follow the ridge)"
        yy = self.d2fdydy(x, y)
        yx = self.d2fdydx(x, y)
        if yy == 0:
            return 0
        return yx/yy
    
    def grad(self, x, y):
        derivs = np.array([self.dfdx(x,y), self.dfdy(x,y)])
        return derivs[0], derivs[1]
    
    def loss(self, x, y):
        return (x-self.xopt)**2 + (y-self.yopt)**2
        
    
class func1(Base):
    def __init__(self):
        super().__init__()
        self.xopt, self.yopt = 0, 0
        self.xrange = [-5, 5, .1]
        self.yrange = [-5, 5, .1]
        self.f =  lambda x, y:-3*x**2-y**2+4*x*y
        self.dfdx = grad(self.f)
        self.dfdy = grad(self.f, 1)   
        self.d2fdxdx = grad(self.dfdx)
        self.d2fdydy = grad(self.dfdy, 1)
        self.d2fdxdy = grad(self.dfdx, 1)
        self.d2fdydx = grad(self.dfdy)
    

    
class func2(Base):
    def __init__(self):
        super().__init__()
        self.xopt, self.yopt = 0.40278777, 0.59721223   
        self.xrange = [-5, 5, .1]
        self.yrange = [-5, 5, .1]
        self.f = lambda x, y : (x-1/2)*(y-1/2) + 1/3 * np.exp(-(x-1/4)**2-(y-3/4)**2)
        self.constraint=False   
        self.dfdx = grad(self.f)
        self.dfdy = grad(self.f, 1)   
        self.d2fdxdx = grad(self.dfdx)
        self.d2fdydy = grad(self.dfdy, 1)
        self.d2fdxdy = grad(self.dfdx, 1)
        self.d2fdydx = grad(self.dfdy)
    
class func3(Base):
    def __init__(self):
        super().__init__()
        print('Quad')
        self.xopt, self.yopt = 0, 0
        self.xrange = [-10, 10, .05]
        self.yrange = [-4, 5, .05]
        self.f = lambda x, y: (4*x**2 -(y-3*x+0.05*x**3)**2-0.1*y**4) * np.exp(-0.01 * (x**2+y**2))
        self.dfdx = grad(self.f)
        self.dfdy = grad(self.f, 1)   
        self.d2fdxdx = grad(self.dfdx)
        self.d2fdydy = grad(self.dfdy, 1)
        self.d2fdxdy = grad(self.dfdx, 1)
        self.d2fdydx = grad(self.dfdy)

class func4(Base):
    def __init__(self):
        super().__init__()
        self.xopt, self.yopt = -2-np.sqrt(2), 2 + np.sqrt(2)
        self.xrange = [-5, 5, .05]
        self.yrange = [-5, 5, .05]
        self.f = lambda x, y: 2*x**2+y**2 +4*x*y+4/3*y**3-1/4*y**4
        self.dfdx = grad(self.f)
        self.dfdy = grad(self.f, 1)   
        self.d2fdxdx = grad(self.dfdx)
        self.d2fdydy = grad(self.dfdy, 1)
        self.d2fdxdy = grad(self.dfdx, 1)
        self.d2fdydx = grad(self.dfdy)

class func5(Base):
    def __init__(self):
        super().__init__()
        self.xopt, self.yopt = -1, 2.5
        self.xrange = [-5, 5, .05]
        self.yrange = [-5, 5, .05]
        self.f = lambda x, y: 1/3*x**3+y**2+2*x*y-6*x-3*y+4
        self.dfdx = grad(self.f)
        self.dfdy = grad(self.f, 1)   
        self.d2fdxdx = grad(self.dfdx)
        self.d2fdydy = grad(self.dfdy, 1)
        self.d2fdxdy = grad(self.dfdx, 1)
        self.d2fdydx = grad(self.dfdy)

class func6(Base):
    def __init__(self):
        super().__init__()
        self.xopt, self.yopt = 0, 0
        self.xrange = [-5, 5, .05]
        self.yrange = [-5, 5, .05]
        self.f = lambda x, y: x**3 - y**3 - 2*x*y + 6
        self.dfdx = grad(self.f)
        self.dfdy = grad(self.f, 1)   
        self.d2fdxdx = grad(self.dfdx)
        self.d2fdydy = grad(self.dfdy, 1)
        self.d2fdxdy = grad(self.dfdx, 1)
        self.d2fdydx = grad(self.dfdy)


class quad2(object):
    def __init__(self, n, cond, spd=False):
        if spd:
            self.A= gen_cond(n, cond)
            self.B= gen_cond(n, cond)
            self.C= gen_cond(n, cond)
        else:
            self.A = np.random.randn(n,n)
            self.B = np.array(generate_sym(n))
            self.C =  np.array(generate_sym(n))
        self.b = np.random.randn(n,1)
        self.c = np.random.randn(n,1)
        ainvb = pinv(self.A) @ self.B
        self.xtrue = pinv(self.A.transpose() + self.C.transpose() @ ainvb) @ (-self.c-self.C@pinv(self.A)@self.b)
        self.ytrue = -pinv(self.A) @ (self.B @ self.xtrue+ self.b)
        f =  lambda x,y: x.transpose() @ self.A @ y + 1/2 * x.transpose() @ self.B @ x -1/2 *  y.transpose() @ self.C @ y + self.b.transpose() @ x + self.c.transpose() @ y
        self.dfdx = grad(f)
        self.dfdy = grad(f, 1)   
        self.hessian = jacobian(self.dfdx)
    
    def fr(self, x, y):
        yx = self.d2fdydx(x, y)
        return yx
    
    def grad(self, x, y):
        derivs = np.array([self.dfdx(x,y), self.dfdy(x,y)])
        return derivs[0], derivs[1]
    
    
    def loss(self, x, y):
        return LA.norm(x-self.xtrue)**2+ LA.norm(y-self.ytrue)**2
    


class quad1(object):
    def __init__(self, n, cond = 10, spd=False, bc=False):
        if spd:
            print('spd')
            self.A = gen_cond(n, cond)
        else:
            print('random')
            self.A = np.random.randn(n,n)
        if bc:
            print('bc')
            self.B = np.random.randn(n,1)
            self.C = np.random.randn(n,1)
            self.xtrue = LA.solve(self.A.transpose(), -self.C)
            self.ytrue = LA.solve(self.A, -self.B)    
        else:
            print('bc zeros')
            self.B = np.zeros((n,1))
            self.C = np.zeros((n,1))
            self.xtrue = LA.solve(self.A.transpose(), -self.C)
            self.ytrue = LA.solve(self.A, -self.B)   
        f =  lambda x,y:  x.transpose() @ self.A @ y + self.B.transpose() @ x + self.C.transpose() @ y 
        self.dfdx = grad(f)
        self.dfdy = grad(f, 1)   
        self.d2fdxdx = grad(self.dfdx)
        self.d2fdydy = grad(self.dfdy, 1)
        self.d2fdxdy = grad(self.dfdx, 1)
        self.d2fdydx = grad(self.dfdy)
    
    
    def fr(self, x, y):
        yx = self.d2fdydx(x, y)
        return yx
    
    def grad(self, x, y):
        derivs = np.array([self.dfdx(x,y), self.dfdy(x,y)])
        return derivs[0], derivs[1]
    
    
    def loss(self, x, y):
        return LA.norm(x-self.xtrue)**2+ LA.norm(y-self.ytrue)**2
    
    


    
    




    

     
    