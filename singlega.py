# -*- coding: utf-8 -*-
"""
singlega.py
Copyright (c) 2021 Nobuo Namura
This code is released under the MIT License.

This Python code is for MBO-EPBII-SRVA and MBO-EPBII published in the following articles:
・N. Namura, "Surrogate-Assisted Reference Vector Adaptation to Various Pareto Front Shapes 
 for Many-Objective Bayesian Optimization," IEEE Congress on Evolutionary Computation, 
 Krakow, Poland, pp.901-908, 2021.
・N. Namura, K. Shimoyama, and S. Obayashi, "Expected Improvement of Penalty-based Boundary 
　Intersection for Expensive Multiobjective Optimization," IEEE Transactions on Evolutionary 
　Computation, vol. 21, no. 6, pp. 898-913, 2017.
Please cite the article(s) if you use the code.

This code was developed with Python 3.6.5.
This code except below is released under the MIT License, see LICENSE.txt.
The code in "EA_in_DEAP" is released under the GNU LESSER GENERAL PUBLIC LICENSE, see EA_in_DEAP/LICENSE.txt.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import functools
from pyDOE import lhs
import sys

import test_problem

#======================================================================
class SingleGA:
#======================================================================
    def __init__(self, func, xmin, xmax, npop=100, ngen=100, mingen=0, \
                 MIN=True, STOP=True, PRINT=False, INIT=False, \
                 istop=10, err=1.e-8, pcross=0.9, pmut=0.1, eta_c=10.0, eta_m=20.0, eps=1.e-14):
        if np.mod(npop,4)!=0:
            print("npop must be a multiple of 4")
            sys.exit()
        self.MIN = MIN
        self.STOP = STOP
        self.PRINT = PRINT
        self.INIT = INIT
        self.func = func
        self.xmin = xmin
        self.xmax = xmax
        self.nx = len(xmin)
        self.npop = npop
        self.ngen = ngen
        self.mingen = mingen
        self.istop = istop
        self.err = err
        self.pcross = pcross
        self.pmut = pmut
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.eps = eps

#======================================================================
    def optimize(self, x_init=0):
        igen = 1
        x, f = self.initialization(x_init)
        if self.MIN:
            fopt = np.min(f)
            xopt = x[np.argmin(f),:]
        else:
            fopt = np.max(f)
            xopt = x[np.argmax(f),:]
        if self.PRINT:
            print(igen,fopt)
        
        fopt0 = fopt
        icnv = 0
        error = 1.0e+20
        for igen in range(2,self.ngen+1):
            if self.STOP and igen > 2:
                if self.err > error:
                    icnv += 1
                    if igen > self.mingen and icnv >= self.istop:
                        break
                else:
                    icnv = 0
            fopt0 = fopt
            xnew = self.selection(x, f)
            xnew = self.mutation(xnew)
            fnew = np.apply_along_axis(self.func,1,xnew)
            x, f = self.bestN(x, xnew, f, fnew)
            if self.MIN:
                ipop = np.argmin(f)
                xopt = x[ipop,:]
                fopt = f[ipop]
                
            else:
                ipop = np.argmax(f)
                xopt = x[ipop,:]
                fopt = f[ipop]
            error = np.abs((fopt - fopt0)/fopt0)
            if self.PRINT:
                print(igen,fopt, error)
        if self.PRINT:
            print('Optimization finished')
            print(fopt, xopt)
    
        return fopt, xopt
    
#======================================================================
    def initialization(self, x_init):
        if self.INIT:
            lpop = self.npop-1
        else:
            lpop = self.npop
        x = lhs(self.nx, samples=lpop, criterion='cm')
        x = self.xmin + x*(self.xmax - self.xmin)
        if self.INIT:
            x = np.vstack([x, x_init])
        f = np.apply_along_axis(self.func,1,x)
        return x, f
    
#======================================================================
    def selection(self, x, f):
        xnew = x.copy()
        a1 = np.arange(self.npop)
        a2 = np.arange(self.npop)
        for i in range(self.npop):
            rand = np.random.randint(i,self.npop)
            temp = a1[rand]
            a1[rand] = a1[i]
            a1[i] = temp
            rand = np.random.randint(i,self.npop)
            temp = a2[rand]
            a2[rand] = a2[i]
            a2[i] = temp
        for i in range(0, self.npop, 4):
            parent1 = self.tournament(a1[i], a1[i+1], f)
            parent2 = self.tournament(a1[i+2], a1[i+3], f)
            xnew[i,:], xnew[i+1,:] = self.crossover(parent1, parent2, x.copy())
            parent1 = self.tournament(a2[i], a2[i+1], f)
            parent2 = self.tournament(a2[i+2], a2[i+3], f)
            xnew[i+2,:], xnew[i+3,:] = self.crossover(parent1, parent2, x.copy())
        return xnew
    
#======================================================================
    def tournament(self, a1, a2, f):
        if self.MIN:
            if f[a1] < f[a2]:
                return a1
            elif f[a1] > f[a2]:
                return a2
            else:
                if np.random.uniform() <= 0.5:
                    return a1
                else:
                    return a2
        else:
            if f[a1] < f[a2]:
                return a2
            elif f[a1] > f[a2]:
                return a1
            else:
                if np.random.uniform() <= 0.5:
                    return a2
                else:
                    return a1
    
#======================================================================
    def crossover(self, p1, p2, x):
        child1 = x[p1,:]
        child2 = x[p2,:]
        if np.random.uniform() <= self.pcross:
            for i in range(self.nx):
                if np.random.uniform() <= 0.5:
                    if np.abs(x[p1,i]-x[p2,i]>self.eps):
                        if x[p1,i] < x[p2,i]:
                            y1 = x[p1,i]
                            y2 = x[p2,i]
                        else:
                            y1 = x[p2,i]
                            y2 = x[p1,i]
                        yL = self.xmin[i]
                        yU = self.xmax[i]
                        rand = np.random.uniform()
                        beta = 1.0 + (2.0*(y1-yL)/(y2-y1))
                        alpha = 2.0 - pow(beta,-(self.eta_c+1.0))
                        if rand <= (1.0/alpha):
                            betaq = rand*alpha
                        else:
                            betaq = 1.0/(2.0 - rand*alpha)
                        betaq = np.sign(betaq)*np.abs(betaq)**(1.0/(self.eta_c+1.0))
                        c1 = 0.5*((y1+y2)-betaq*(y2-y1))
                        beta = 1.0 + (2.0*(yU-y2)/(y2-y1))
                        alpha = 2.0 - beta**(-(self.eta_c+1.0))
                        if rand <= (1.0/alpha):
                            betaq = rand*alpha
                        else:
                            betaq = 1.0/(2.0 - rand*alpha)
                        betaq = np.sign(betaq)*np.abs(betaq)**(1.0/(self.eta_c+1.0))
                        c2 = 0.5*((y1+y2)+betaq*(y2-y1))
                        if c1<yL:
                            c1=yL
                        if c2<yL:
                            c2=yL
                        if c1>yU:
                            c1=yU
                        if c2>yU:
                            c2=yU
                        if np.random.uniform() <= 0.5:
                            child1[i] = c2
                            child2[i] = c1
                        else:
                            child1[i] = c1
                            child2[i] = c2
        return child1, child2

#======================================================================
    def mutation(self, x):
        for i in range(self.npop):
            for j in range(self.nx):
                if np.random.uniform() <= self.pmut:
                    y = x[i,j]
                    yL = self.xmin[j]
                    yU = self.xmax[j]
                    delta1 = (y-yL)/(yU-yL)
                    delta2 = (yU-y)/(yU-yL)
                    rnd = np.random.uniform()
                    mut_pow = 1.0/(self.eta_m+1.0)
                    if (rnd <= 0.5):
                        xy = 1.0-delta1
                        val = 2.0*rnd+(1.0-2.0*rnd)*xy**(self.eta_m+1.0)
                        deltaq = val**mut_pow - 1.0
                    else:
                        xy = 1.0-delta2
                        val = 2.0*(1.0-rnd)+2.0*(rnd-0.5)*xy**(self.eta_m+1.0)
                        deltaq = 1.0 - val**mut_pow
                    y = y + deltaq*(yU-yL)
                    if y < yL:
                        y = yL
                    if y > yU:
                        y = yU
                    x[i,j] = y
        return x

#======================================================================
    def bestN(self, x, xnew, f, fnew):
        x2n = np.vstack([x,xnew])
        f2n = np.hstack([f,fnew])
        order = np.argsort(f2n)
        if not self.MIN:
            order = order[::-1]
        x2n = x2n[order]
        f2n = f2n[order]
        x = x2n[:self.npop,:]
        f = f2n[:self.npop]
        return x, f

#======================================================================
if __name__ == "__main__":
    func_name = 'sphere'
    nx = 2
    nf = 1
    xmin = np.full(nx, -0.5)
    xmax = np.full(nx, 0.5)
    func = functools.partial(eval('test_problem.'+func_name), nf=nf)
    SGA = SingleGA(func, xmin, xmax, npop=100, ngen=100, MIN=True, STOP=False, PRINT=True, pcross=0.9, pmut=1.0/len(xmin))
    fopt, xopt = SGA.optimize()
    
    if nx ==2:
        x = xmin[0]+np.arange(0., 1.01, 0.01)*(xmax[0]-xmin[0])
        y = xmin[1]+np.arange(0., 1.01, 0.01)*(xmax[1]-xmin[1])
        X, Y = np.meshgrid(x, y)
        F = X.copy()
        for i in range(len(X[:,0])):
            for j in range(len(X[0,:])):
                F[i,j] = func(np.array([X[i,j],Y[i,j]]))
        plt.figure('problem')
        plt.plot(xopt[0],xopt[1],'o',c='black')
        plt.pcolor(X,Y,F)
        plt.colorbar()
        plt.contour(X,Y,F,40,colors='black')
        plt.show()
        
        fig = plt.figure('problem-3D')
        ax = Axes3D(fig)
        ax.scatter3D(xopt[0],xopt[1],fopt,c='black')
        ax.plot_surface(X, Y, F, rstride=1, cstride=1, cmap=cm.coolwarm)
    