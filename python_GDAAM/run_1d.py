
import matplotlib.patches as mpatches
import os
import autograd.numpy as np
import matplotlib.pylab as pylab
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from autograd import elementwise_grad, value_and_grad
from mpl_toolkits.mplot3d import Axes3D
from problems import func1, func2, func3, func4, func5, func6
from AA import numpyAA

def_colors=(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'font.size': 14})
params = {'mathtext.default': 'regular' } 
plt.rcParams.update(params)
gamma = 1e-26 # For 1d problems, you can also set gamma=0. 

def simgd(problem, x0, y0, iteration, lr, k=0):
    x, y = x0, y0
    xopt, yopt = problem.xopt, problem.yopt
    x_hist, y_hist = [x], [y]
    loss = [np.sqrt((x-xopt)**2 + (y-yopt)**2)]
    for i in range(iteration):
        g_x, g_y = problem.grad(x,y)
        x -= lr * g_x
        y += lr * g_y
        x_hist.append(x)
        y_hist.append(y)
        loss.append(problem.loss(x, y))
    return loss, x_hist, y_hist


def altgd(problem, x0, y0, iteration, lr, k=0):
    x, y = x0, y0
    xopt, yopt = problem.xopt, problem.yopt
    x_hist, y_hist = [x], [y]
    loss = [np.sqrt((x-xopt)**2 + (y-yopt)**2)]
    for i in range(iteration):
        g_x, _ = problem.grad(x,y)
        x -= lr * g_x
        _, g_y = problem.grad(x,y)
        y += lr * g_y
        x_hist.append(x)
        y_hist.append(y)
        loss.append(problem.loss(x, y))
    return loss, x_hist, y_hist


def adam(problem, x0, y0, iteration, lr, k=0):
    x, y = x0, y0
    LR = lr
    xopt, yopt = problem.xopt, problem.yopt
    x_hist, y_hist = [x], [y]
    loss = [np.sqrt((x-xopt)**2 + (y-yopt)**2)]
    BETA_1 = 0.5
    BETA_2 = 0.99
    EPSILON = 1e-8
    v_x, v_y = 0., 0.
    m_x, m_y = 0., 0.
    for i in range(iteration):
        g_x, g_y = problem.grad(x,y)
        m_x = BETA_1*m_x + (1-BETA_1)*g_x
        m_y = BETA_1*m_y + (1-BETA_1)*g_y 
        v_x = BETA_2*v_x + (1-BETA_2)*g_x**2
        v_y = BETA_2*v_y + (1-BETA_2)*g_y**2
        m_hat_x = m_x/(1-BETA_1**(i+1))
        m_hat_y = m_y/(1-BETA_1**(i+1))
        v_hat_x = v_x
        v_hat_y = v_y
        x = x - LR*m_hat_x/(np.sqrt(v_hat_x)+EPSILON)
        y = y + LR*m_hat_y/(np.sqrt(v_hat_y)+EPSILON)
        x_hist.append(x)
        y_hist.append(y)
        loss.append(problem.loss(x, y))
    return loss, x_hist, y_hist


def avg(problem, x0, y0, iteration, lr, k=0):
    x, y = x0, y0
    xopt, yopt = problem.xopt, problem.yopt
    loss = [np.sqrt((x-xopt)**2 + (y-yopt)**2)]
    xavg, yavg = x, y
    x_hist, y_hist = [xavg], [yavg]
    for i in range(iteration):
        x = x - lr/np.sqrt(i+1)*(y)
        y = y + lr/np.sqrt(i+1)*(x)        
        xavg = xavg*(i+1)/(i+2) + x/(i+2)
        yavg = yavg*(i+1)/(i+2) + y/(i+2)        
        x_hist.append(xavg)
        y_hist.append(yavg)
        loss.append(problem.loss(xavg, yavg))
    return loss, x_hist, y_hist

def omd(problem, x0, y0, iteration, lr, k=0):
    x, y = x0, y0
    x_l, y_l = 0.5*x0, 0.5*y0
    g_xl, g_yl = problem.grad(x_l,y_l)
    xopt, yopt = problem.xopt, problem.yopt
    x_hist, y_hist = [x], [y]
    loss = [np.sqrt((x-xopt)**2 + (y-yopt)**2)]
    for i in range(iteration):
        g_x, g_y = problem.grad(x,y)
        x = x - 2 * lr * g_x + lr * g_xl
        y = y + 2 * lr * g_y - lr * g_yl
        x_hist.append(x)
        y_hist.append(y)
        g_xl, g_yl =  g_x, g_y
        loss.append(problem.loss(x, y))
    return loss, x_hist, y_hist

def eg(problem, x0, y0, iteration, lr, k=0):
    x, y = x0, y0
    xopt, yopt = problem.xopt, problem.yopt
    x_hist, y_hist = [x], [y]
    loss = [np.sqrt((x-xopt)**2 + (y-yopt)**2)]
    for i in range(iteration):
        g_x, g_y = problem.grad(x,y)
        xe = x - lr * g_x
        ye = y + lr * g_y
        g_x, g_y = problem.grad(xe,ye)
        x -= lr * g_x
        y += lr * g_y
        x_hist.append(x)
        y_hist.append(y)
        loss.append(problem.loss(x, y))
    return loss, x_hist, y_hist



def altGDAAM(problem, x0, y0, iteration, lr, k, type2=True, reg=1e-10):
    '''
    Proposed Methods: alternating GDA with Anderson Acceleration with numpy
    '''
    x, y = x0, y0
    xopt, yopt = problem.xopt, problem.yopt
    x_hist, y_hist = [x], [y]
    loss = [np.sqrt((x-xopt)**2 + (y-yopt)**2)]
    fp = np.vstack((x, y))
    aa = numpyAA(2, k, type2=type2, reg=reg)
    for i in range(iteration):
        fpprev = np.copy(fp)
        g_x, _ = problem.grad(x,y)
        x_ = (1-gamma) * x - lr * g_x
        _, g_y = problem.grad(x_,y)
        y_ = y + lr * g_y
        fp = np.vstack((x_, y_))
        fp = aa.apply(fpprev, fp)
        x, y = fp[0],fp[1]
        lo = problem.loss(x, y)
        loss.append(lo)
        x_hist.append(x)
        y_hist.append(y)
    return loss, x_hist, y_hist


def simGDAAM(problem, x0, y0, iteration, lr, k, type2=True, reg=1e-8):
    '''
    Proposed Methods: alternating GDA with Anderson Acceleration with numpy
    '''
    x, y = x0, y0
    xopt, yopt = problem.xopt, problem.yopt
    x_hist, y_hist = [x], [y]
    loss = [np.sqrt((x-xopt)**2 + (y-yopt)**2)]
    fp = np.vstack((x, y))
    aa = numpyAA(2, k, type2=True, reg=reg)
    for i in range(iteration):
        fpprev = np.copy(fp)
        g_x, g_y = problem.grad(x,y)
        x_ =  (1-gamma) *x - lr * g_x
        y_ =  (1-gamma) *y + lr * g_y
        fp = np.vstack((x_, y_))
        fp = aa.apply(fpprev, fp)
        x, y = fp[0],fp[1]
        lo = problem.loss(x, y)
        loss.append(lo)
        x_hist.append(x)
        y_hist.append(y)
    return loss, x_hist, y_hist

def plot(loss, xpath, ypath, iteration, k, start):
    x0, y0 = start
    loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8 = loss
    xpath1, xpath2, xpath3, xpath4, xpath5, xpath6, xpath7, xpath8= xpath
    ypath1, ypath2, ypath3, ypath4, ypath5, ypath6, ypath7, ypath8 = ypath
    fig, axlist = plt.subplots(1, 2, figsize=(14,5))
    ax1 = axlist[0]
    ax2 = axlist[1]    
    ax1.contourf(x, y, z, 5, cmap=plt.cm.gray)
    ax1.quiver(x, y, x - dz_dx, y - dz_dy, alpha=.5)
    ax1.plot(xpath7, ypath7,  'm-', linewidth=2, label='SimGDA',markevery=markevery)
    ax1.plot(xpath1, ypath1, 'g--', linewidth=2, label='AltGDA',markevery=markevery)
    ax1.plot(xpath2, ypath2, '--',linewidth=2, label='Avg',markevery=markevery)
    ax1.plot(xpath3, ypath3, 'k-^', linewidth=2, label='EG',markevery=markevery)
    ax1.plot(xpath4, ypath4, 'c-*', linewidth=2, label='OMD',markevery=markevery)
    ax1.plot(xpath6, ypath6, 'b->', linewidth=2, label='SimGDA-RAM', markevery=markevery)
    ax1.plot(xpath5, ypath5, 'r-d', linewidth=2, label='AltGDA-RAM', markevery=markevery)
    x_init = ax1.scatter(x0, y0, marker='s', s=250, c='g',alpha=1,zorder=3, label='Start')
    x_sol = ax1.scatter(xsol, ysol, s=250, marker='*', color='violet', zorder=3, label='Optima')
    ax1.legend([x_init, x_sol],['Start','Optima'], markerscale=1, loc=4, fancybox=True, framealpha=1., fontsize=20)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')  
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylim([ymin,ymax])
    
    plot_interval =1
    ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss7[::plot_interval], 'm-', markevery=markevery, label='SimGDA')
    ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss1[::plot_interval], 'g--', markevery=markevery, label='AltGDA')
    ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss2[::plot_interval], '--', markevery=markevery, label='Averaging')
    ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss3[::plot_interval], 'k-^', markevery=markevery, label='EG')
    ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss4[::plot_interval],'c-*', markevery=markevery, label='OMD')
    ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss5[::plot_interval], 'r-d', markevery=markevery, label='SimGDA-RAM')
    ax2.semilogy(np.arange(0, iteration+plot_interval, plot_interval), loss6[::plot_interval], 'b->', markevery=markevery, label='AltGDA-RAM')
    ax2.set_xlabel('Iteration')
    ax2.set_ylim([1e-25,1e4])
    ax2.set_ylabel('Distance to optimal')
    axlist.flatten()[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True, framealpha=1., fontsize=20, markerscale=2)
    fig.savefig('figures/'+figname, dpi=300, bbox_inches = 'tight', pad_inches = 0)

def main(problem, iteration, x0, y0, lrset, k=5):
    allloss = [[] for _ in  range(8)]
    allxpath = [[] for _ in  range(8)]
    allypath = [[] for _ in  range(8)]
    allloss[0], allxpath[0], allypath[0] = altgd(problem, x0, y0, iteration, lr=lrset['altgd'])
    allloss[1], allxpath[1], allypath[1] = avg(problem, x0, y0, iteration, lr=lrset['avg']) 
    allloss[2], allxpath[2], allypath[2] = eg(problem, x0, y0, iteration, lr=lrset['eg'])
    allloss[3], allxpath[3], allypath[3] = omd(problem, x0, y0, iteration, lr=lrset['omd'])
    allloss[4], allxpath[4], allypath[4]= simGDAAM(problem, x0, y0, iteration, lr=lrset['AA'], k=k)   
    allloss[5], allxpath[5], allypath[5]= altGDAAM(problem, x0, y0, iteration, lr=lrset['AA'] ,k=k)   
    allloss[6], allxpath[6], allypath[6]= simgd(problem, x0, y0, iteration, lr=lrset['simgd'])   
    return allloss, allxpath, allypath


if __name__ == "__main__":
    iteration =100
    markevery= 10
    
    figname = 'path1.png'
    k = 20
    x0, y0 = 3.,3.
    problem = func2()
    xsol, ysol = problem.xopt, problem.yopt
    lr_set = {'simgd':0.05, 'altgd':0.1, 'avg':1, 'adam':0.01, 'eg':0.6,'omd':0.3, 'fr':0.05,'AA':0.5}
    f = problem.f
    type2=True
    loss_f3, xpath_f3, ypath_f3 = main(problem, iteration, x0, y0, lr_set, k=k)
    xmin, xmax, xstep = [-3.5, 5, .1]
    ymin, ymax, ystep = [-3.5, 5, .1]
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = f(x, y)
    dz_dx = elementwise_grad(f, argnum=0)(x, y)
    dz_dy = elementwise_grad(f, argnum=1)(x, y)
    plot(loss_f3, xpath_f3, ypath_f3, iteration, k, [x0, y0])
    
    figname = 'path2.png'
    k = 20
    x0, y0 = 3.,3.
    problem = func3()
    xsol, ysol = problem.xopt, problem.yopt
    lr_set = {'simgd':0.05, 'altgd':0.1, 'avg':1, 'adam':0.01, 'eg':0.1,'omd':0.05, 'fr':0.05,'AA':0.2}
    f = problem.f
    type2=True
    loss_f3, xpath_f3, ypath_f3 = main(problem, iteration, x0, y0, lr_set, k=k)
    xmin, xmax, xstep = [-3.5, 5, .1]
    ymin, ymax, ystep = [-3.5, 5, .1]
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = f(x, y)
    dz_dx = elementwise_grad(f, argnum=0)(x, y)
    dz_dy = elementwise_grad(f, argnum=1)(x, y)
    plot(loss_f3, xpath_f3, ypath_f3, iteration, k, [x0, y0])
    
    figname = 'path3.png'
    iteration =50
    markevery= 10
    k = 20
    x0, y0 = 3.,3.
    problem = func1()
    xsol, ysol = problem.xopt, problem.yopt
    lr_set = {'simgd':0.05, 'altgd':0.1, 'avg':1, 'adam':0.01, 'eg':0.1,'omd':0.05, 'fr':0.05,'AA':0.2}
    f = problem.f
    type2=True
    loss_f3, xpath_f3, ypath_f3 = main(problem, iteration, x0, y0, lr_set, k=k)
    xmin, xmax, xstep = [-3.5, 5, .1]
    ymin, ymax, ystep = [-3.5, 5, .1]
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = f(x, y)
    dz_dx = elementwise_grad(f, argnum=0)(x, y)
    dz_dy = elementwise_grad(f, argnum=1)(x, y)
    plot(loss_f3, xpath_f3, ypath_f3, iteration, k, [x0, y0])
    
    figname = 'path4.png'
    x0, y0 = 3., 3.
    k = 5
    iteration = 100
    problem = func4()
    xsol, ysol = problem.xopt, problem.yopt
    lr_set = {'simgd':0.05, 'altgd':0.1, 'avg':0.5, 'adam':0.001, 'eg':0.1,'omd':0.05, 'fr':0.05,'AA':0.2}
    f = problem.f
    type2=True
    loss_f3, xpath_f3, ypath_f3 = main(problem, iteration, x0, y0, lr_set, k=k)
    xmin, xmax, xstep = [-5, 4, .1]
    ymin, ymax, ystep = [-3.5, 7, .1]
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = f(x, y)
    dz_dx = elementwise_grad(f, argnum=0)(x, y)
    dz_dy = elementwise_grad(f, argnum=1)(x, y)
    plot(loss_f3, xpath_f3, ypath_f3, iteration, k, [x0, y0])
    
    
    figname = 'path5.png'
    k = 20
    iteration =200
    markevery= 10
    x0, y0 = 3.,3.
    problem = func5()
    xsol, ysol = problem.xopt, problem.yopt
    lr_set = {'simgd':0.05, 'altgd':0.05, 'avg':1, 'adam':0.01, 'eg':0.06,'omd':0.03, 'fr':0.05,'AA':0.5}
    f = problem.f
    type2=True
    loss_f3, xpath_f3, ypath_f3 = main(problem, iteration, x0, y0, lr_set, k=k)
    xmin, xmax, xstep = [-5, 10, .1]
    ymin, ymax, ystep = [-5, 10, .1]
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = f(x, y)
    dz_dx = elementwise_grad(f, argnum=0)(x, y)
    dz_dy = elementwise_grad(f, argnum=1)(x, y)
    plot(loss_f3, xpath_f3, ypath_f3, iteration, k, [x0, y0])
    
    
    figname = 'path6.png'
    k = 5
    iteration =200
    markevery= 10
    x0, y0 = 3.,3.
    problem = func6()
    xsol, ysol = problem.xopt, problem.yopt
    lr_set = {'simgd':0.05, 'altgd':0.05, 'avg':1, 'adam':0.01, 'eg':0.04,'omd':0.02, 'fr':0.05,'AA':0.1}
    f = problem.f
    type2=True
    loss_f3, xpath_f3, ypath_f3 = main(problem, iteration, x0, y0, lr_set, k=k)
    xmin, xmax, xstep = [-5, 5, .1]
    ymin, ymax, ystep = [-5, 5, .1]
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = f(x, y)
    dz_dx = elementwise_grad(f, argnum=0)(x, y)
    dz_dy = elementwise_grad(f, argnum=1)(x, y)
    plot(loss_f3, xpath_f3, ypath_f3, iteration, k, [x0, y0])