# -*- coding: utf-8 -*-
"""
moead_epbii.py
Copyright (c) 2021 Nobuo Namura
This code is released under the MIT License.

This Python code is for MBO-EPBII-SRVA and MBO-EPBII published in the following articles:
・N. Namura, "Surrogate-assisted Reference Vector Adaptation to Various Pareto Front Shapes 
　for Many-objective Bayesian Optimization," 2021 IEEE Congress on Evolutionary Computation, Krakow, 2021.
・N. Namura, K. Shimoyama, and S. Obayashi, "Expected Improvement of Penalty-based Boundary 
　Intersection for Expensive Multiobjective Optimization," IEEE Transactions on Evolutionary 
　Computation, vol. 21, no. 6, pp. 898-913, 2017.
Please cite the article(s) if you use the code.

This code was developed with Python 3.6.5.
This code except below is released under the MIT License, see LICENSE.txt.
The code in "EA_in_DEAP" is released under the GNU LESSER GENERAL PUBLIC LICENSE, see EA_in_DEAP/LICENSE.txt.
"""

import numpy as np
from pyDOE import lhs
from scipy.spatial.distance import cdist

#======================================================================
class MOEAD_EPBII:
#======================================================================
    def __init__(self, refvec, func, xmin, xmax, ngen=100, PRINT=False, HOT_START=True, nghbr=20, \
                 factor=1.0, pcross=1.0, pmut=0.1, eta_c=10.0, eta_m=20.0, eps=1.e-14):
        self.refvec = refvec
        self.func = func
        self.MIN = False
        self.PRINT = PRINT
        self.HOT_START = HOT_START
        self.xmin = xmin
        self.xmax = xmax
        self.nx = len(self.xmin)
        self.nref = len(refvec[:,0])
        self.npop = self.nref
        self.ngen = ngen
        self.nghbr = min(nghbr, self.nref)
        self.factor = factor
        self.pcross = pcross
        self.pmut = pmut
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.eps = eps

#======================================================================
    def optimize(self, x_init=0):
        igen = 1
        self.select_neighbor()
        x, f = self.initialization(x_init)
        
        if self.PRINT:
            print(igen, self.npop, f[0], f[int(self.nref/2)], f[-1])
        
        for igen in range(2,self.ngen+1):
            for ipop in range(self.npop):
                parent1 = self.neighbor[ipop,0]
                parent2 = self.neighbor[ipop,np.random.randint(1,self.nghbr)]
                x_child = self.crossover_moead(parent1, parent2, x.copy())
                x_child = self.mutation_moead(x_child)
                x, f = self.update_population(ipop, x_child, x, f)
            if self.PRINT:
                 print(igen, self.npop, f[0], f[int(self.nref/2)], f[-1])

        return f, x
    
#======================================================================
    def select_neighbor(self):
        dist = cdist(self.refvec, self.refvec)
        self.neighbor = np.argsort(dist, axis=1)[:,:self.nghbr]


#======================================================================
    def initialization(self, x_init):
        if self.HOT_START:
            x = x_init
        else:
            x = lhs(self.nx, samples=self.npop, criterion='cm')
            x = self.xmin + x*(self.xmax - self.xmin)
        f = np.array([self.func(x[i,:], i) for i in range(self.npop)])
        return x, f

#======================================================================
    def update_population(self, ipop, x_child, x, f):
        fnew = f.copy()
        xnew = x.copy()
        for iref in range(self.nref):
            f_child = self.func(x_child, iref)
            if f_child > f[iref]:
                fnew[iref] = f_child
                xnew[iref] = x_child
        return xnew, fnew

#======================================================================
    def crossover_moead(self, p1, p2, x):
        child = x[p1,:]
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
                            child[i] = c2
                        else:
                            child[i] = c1
        return child

#======================================================================
    def mutation_moead(self, x):
        for j in range(self.nx):
            if np.random.uniform() <= self.pmut:
                y = x[j]
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
                x[j] = y
        return x
