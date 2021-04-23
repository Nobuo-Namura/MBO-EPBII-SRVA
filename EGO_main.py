# -*- coding: utf-8 -*-
"""
EGO_main.py
Copyright (c) 2021 Nobuo Namura
This code is released under the MIT License.

This Python code is for MBO-EPBII-SRVA and MBO-EPBII published in the following article:
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
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import functools
import time
import shutil
from pyDOE import lhs
import os

from kriging import Kriging
import test_problem
from initial_sample import generate_initial_sample
import indicator



#======================================================================
if __name__ == "__main__":
    """Constraints are unavailable in this version"""
    
    """=== Edit from here ==========================================="""
    #test problem
    func_name = 'ZDT3'        # Test problem name in test_problem.py
    seed = 101                # Random seed for SGM function
    nx = 2                    # Number of design variables (>=1)
    nf = 2                    # Number of objective functions (>=2)
    k = 1                     # Position paramete k in WFG problems
    xmin = np.full(nx, 0.0)   # Lower bound of design sapce
    xmax = np.full(nx, 1.0)   # Upper bound of design sapce
    MIN = np.full(nf, True)   # True=Minimization, False=Maximization
    #EGO
    n_trial = 1               # Number of independent run with different initial samples (>=1)
    n_add = 5                 # Number of additional sample points at each iteration (>=1)
    ns_max = 50               # Number of maximum function evaluation
    CRITERIA = 'EPBII'        # EPBII or EIPBII
    NOISE = np.full(nf,False) # Use True if functions are noisy (Griewank, Rastrigin, DTLZ1, etc.)
    VER2021 = True            # True=2021 version, False=2017 version
    SRVA = True               # True=surrogate-assisted reference vector adaptation, False=two-layered simplex latice-design
    OPTIMIZER = 'NSGA3'       # NSGA3 or NSGA2 for ideal and nadir point determination (and reference vector adaptation if VER2021=True)
    #reference vector for EPBII
    n_randvec = 20            # Number of adaptive reference vector (>=0)
    nh = 0                    # If VER2021=False, division number for the outer layer of the two-layered simplex latice-design (>=0)
    nhin = 0                  # If VER2021=False, division number for the inner layer of the two-layered simplex latice-design (>=0)
    #NSGA3 if OPTIMIZER=NSGA3:
    n_randvec_nsga3 = 0       # Number of random reference vector (>=0)
    nh_nsga3 = 99             # Division number for the outer layer of the two-layered simplex latice-design (>=0)
    nhin_nsga3 = 0            # Division number for the inner layer of the two-layered simplex latice-design (>=0)
    ngen_nsga3 = 200          # Number of generation in NSGA3
    #NSGA2 if OPTIMIZER=NSGA2:
    npop_nsga2 = 100          # Number of population in NSGA2
    ngen_nsga2 = 200          # Number of generation in NSGA2
    #initial sample
    GENE = True               # True=Generate initial sample with LHS, False=Read files
    ns = 30                   # If GENE=True, number of initial sample points (>=2)
    #others
    hv_ref = np.array([1.1,1.1]) # reference point for hypervolume
    IGD_plus = True           # True=IGD+, False=IGD
    PLOT = True               # True=Plot the results
    RESTART = False           # True=Read sample*_out.csv if it exists, False=Read sample*.csv
    current_dir = '.'
    fname_design_space = 'design_space'
    fname_sample = 'sample'
    fname_indicator = 'indicators'
    path_IGD_ref = current_dir + '/IGD_ref'
    """=== Edit End ================================================="""
    
    #Initial sample
    if func_name == 'SGM':
        problem = functools.partial(eval('test_problem.'+func_name), nf=nf, seed=seed)
        if GENE:
            generate_initial_sample(func_name, nx, nf, ns, n_trial, xmin, xmax, current_dir, fname_design_space, fname_sample, seed=seed)
    else:
        problem = functools.partial(eval('test_problem.'+func_name), nf=nf)
        if GENE:
            generate_initial_sample(func_name, nx, nf, ns, n_trial, xmin, xmax, current_dir, fname_design_space, fname_sample, k=k)
    f_design_space = current_dir + '/' + fname_design_space + '.csv'
    if func_name == 'SGM':
        file = path_IGD_ref + '/' + func_name + str(seed) + 'x' + str(nx) +'f' + str(nf) + '.csv'
        if os.path.exists(file):
            IGD_FLAG = True
            igd_ref = np.loadtxt(path_IGD_ref + '/' + func_name + str(seed) + 'x' + str(nx) +'f' + str(nf) + '.csv', delimiter=',')
        else:
            IGD_FLAG = False
    else:
        file = path_IGD_ref + '/' + func_name + 'f' + str(nf) + '.csv'
        if os.path.exists(file):
            IGD_FLAG = True
            igd_ref = np.loadtxt(path_IGD_ref + '/' + func_name + 'f' + str(nf) + '.csv', delimiter=',')
        else:
            IGD_FLAG = False
    gp = Kriging(MIN, CRITERIA, n_add, n_randvec, nh, nhin, n_randvec_nsga3, nh_nsga3, nhin_nsga3, ngen_nsga3, \
                 npop_nsga2, ngen_nsga2, VER2021, SRVA, OPTIMIZER, pbi_theta=1.0)
    
    #Preprocess for RMSE
    if nx == 2:
        ndiv = 101
        x_rmse0 = np.zeros([ndiv**2, nx])
        for i in range(101):
            for j in range(101):
                x_rmse0[i*ndiv+j,0] = float(i)/float(ndiv-1)
                x_rmse0[i*ndiv+j,1] = float(j)/float(ndiv-1)
    else:
        x_rmse0 = np.random.uniform(size=[10000, nx])

    #Independent run
    print('EGO')
    for itrial in range(1,n_trial+1,1):
        #Preprocess
        print('trial '+ str(itrial))
        if RESTART:
            f_sample = current_dir + '/' + fname_sample + str(itrial) + '_out.csv'
            FILEIN = False
            if not os.path.exists(f_sample):
                f_sample = current_dir + '/' + fname_sample + str(itrial) + '.csv'
                FILEIN = True
        else:
            f_sample = current_dir + '/' + fname_sample + str(itrial) + '.csv'
            FILEIN = True
        gp.read_sample(f_sample)
        gp.normalize_x(f_design_space)
        x_rmse = gp.xmin + (gp.xmax-gp.xmin)*x_rmse0
        max_iter = int((ns_max + (n_add - 1) - gp.ns)/n_add)
        rmse = np.zeros([max_iter, gp.nf + gp.ng])
        igd = np.zeros(max_iter+1)
        hv = np.zeros(max_iter+1)
        times = []
        rank = gp.pareto_ranking(gp.f, gp.g)
        if not IGD_FLAG:
            igd[0] = np.nan
        else:
            igd[0] = indicator.igd_history(gp.f[rank==1.0], igd_ref, IGD_plus, MIN)
        hv[0] = indicator.hv_history(gp.f[rank==1.0], hv_ref, MIN)
        f_indicator = current_dir + '/' + fname_indicator + str(itrial) +'.csv'
        if FILEIN:
            with open(f_indicator, 'w') as file:
                data = ['iteration', 'samples', 'time', 'IGD', 'Hypervolume']
                for i in range(gp.nf + gp.ng):
                    data.append('RMSE'+str(i+1))
                data = np.array(data).reshape([1,len(data)])
                np.savetxt(file, data, delimiter=',', fmt = "%s")
            f_sample_out =  current_dir + '/' + fname_sample + str(itrial) + '_out.csv'
            shutil.copyfile(f_sample, f_sample_out)
        else:
            f_sample_out = f_sample
        
        #Main loop for EGO
        for itr in range(max_iter):
            try:
                times.append(time.time())
                print('=== Iteration = '+str(itr)+', Number of sample = '+str(gp.ns)+' ======================')
                
                #Kriging and infill criterion
                gp.kriging_training(theta0 = 3.0, npop = 500, ngen = 500, mingen=0, STOP=True, NOISE=NOISE)
                x_add, f_add_est = gp.kriging_infill(PLOT=True)
                times.append(time.time())

                #RMSE
                for ifg in range(gp.nf + gp.ng):
                    gp.nfg = ifg
                    rmse[itr, ifg] = indicator.rmse_history(x_rmse, problem, gp.kriging_f, ifg)

                #Add sample points
                for i_add in range(gp.n_add):
                    f_add = problem(x_add[i_add])
                    gp.add_sample(x_add[i_add],f_add)
                
                #Indicators and file output
                with open(f_indicator, 'a') as file:
                    data = np.hstack([itr, gp.ns-gp.n_add, times[-1]-times[-2], igd[itr], hv[itr], rmse[itr, :]])
                    np.savetxt(file, data.reshape([1,len(data)]), delimiter=',')
                with open(f_sample_out, 'a') as file:
                    data = np.hstack([gp.x[-gp.n_add:,:], gp.f[-gp.n_add:,:], gp.g[-gp.n_add:,:]])
                    np.savetxt(file, data, delimiter=',')
                rank = gp.pareto_ranking(gp.f, gp.g)
                if not IGD_FLAG:
                    igd[itr+1] = np.nan
                else:
                    igd[itr+1] = indicator.igd_history(gp.f[rank==1.0], igd_ref, IGD_plus, MIN)
                hv[itr+1] = indicator.hv_history(gp.f[rank==1.0], hv_ref, MIN)
                if itr == max_iter-1:
                    with open(f_indicator, 'a') as file:
                        data = np.array([itr+1, gp.ns, 0.0, igd[itr+1], hv[itr+1]])
                        np.savetxt(file, data.reshape([1,len(data)]), delimiter=',')
                
                #Visualization
                if PLOT:
                    f_pareto = gp.f[rank==1.0]
                    if nf == 2:
                        plt.figure('test 2D Objective-space '+func_name+' with '+str(gp.ns-gp.n_add)+'-samples')
                        plt.plot(gp.f[:-gp.n_add,0], gp.f[:-gp.n_add,1], '.', c='black', label='sample points')
                        plt.plot(gp.f[-gp.n_add:,0], gp.f[-gp.n_add:,1], '.', c='magenta', label='additional sample points')
                        plt.plot(gp.utopia[0], gp.utopia[1], '+', c='black', label='utopia point')
                        plt.plot(gp.nadir[0], gp.nadir[1], '+', c='black', label='nadir point')
                        plt.plot(gp.refpoint[:,0], gp.refpoint[:,1], '.', c='blue',marker='+', label='reference PBI points')
                        plt.scatter(gp.f_candidate[:,0],gp.f_candidate[:,1],c=gp.fitness_org,cmap='jet',marker='*', label='candidate points')
                        plt.scatter(f_add_est[:,0],f_add_est[:,1], facecolors='none', edgecolors='black', marker='o', label='selected candidate points')
                        plt.scatter(gp.f_ref[:,0],gp.f_ref[:,1],c='grey',s=1,marker='*', label='estimated PF')
                        plt.legend()
                        plt.show(block=False)
                        title = current_dir + '/2D_Objective_space_'+func_name+' with '+str(gp.ns-gp.n_add)+'-samples_in_'+str(itrial)+'-th_trial.png'
                        plt.savefig(title, dpi=300)
                        plt.close()
                        
                        plt.figure('solutions on 2D Objective-space '+func_name+' with '+str(gp.ns)+'-samples')
                        if not IGD_FLAG:
                            pass
                        else:
                            plt.scatter(igd_ref[:,0],igd_ref[:,1],c='green',s=1)
                        plt.scatter(f_pareto[:,0],f_pareto[:,1],c='blue',s=20,marker='o')
                        title = current_dir + '/Optimal_solutions_'+func_name+' with '+str(gp.ns)+'-samples_in_'+str(itrial)+'-th_trial.png'
                        plt.savefig(title)
                        plt.close()
                        
                    elif nf > 2:
                        fig = plt.figure('3D Objective-space '+func_name+' with '+str(gp.ns-gp.n_add)+'-samples')
                        ax = Axes3D(fig)
                        ax.scatter3D(gp.f[-gp.n_add:,0],gp.f[-gp.n_add:,1],gp.f[-gp.n_add:,-1],c='red',marker='^', label='additional sample points')
                        ax.scatter3D(f_pareto[:,0],f_pareto[:,1],f_pareto[:,-1],c='blue',marker='+', label='NDSs among sample points')
                        ax.scatter3D(gp.f_candidate[:,0],gp.f_candidate[:,1],gp.f_candidate[:,-1],c=gp.fitness_org,cmap='jet',marker='*', label='candidate points')
                        ax.scatter3D(f_add_est[:,0],f_add_est[:,1],f_add_est[:,-1], c='black', label='selected candidate points')
                        ax.scatter3D(gp.f_ref[:,0],gp.f_ref[:,1],gp.f_ref[:,-1],c='grey',s=1,marker='*', label='estimated PF')
                        plt.legend()
                        title = current_dir + '/3D_Objective_space_'+func_name+' with '+str(gp.ns-gp.n_add)+'-samples_in_'+str(itrial)+'-th_trial.png'
                        plt.savefig(title)
                        plt.close()
                        
                        fig2 = plt.figure('solutions on 3D Objective-space '+func_name+' with '+str(gp.ns)+'-samples')
                        ax2 = Axes3D(fig2)
                        if not IGD_FLAG:
                            pass
                        else:
                            ax2.scatter3D(igd_ref[:,0],igd_ref[:,1],igd_ref[:,-1],c='green',s=1)
                        ax2.scatter3D(f_pareto[:,0],f_pareto[:,1],f_pareto[:,-1],c='blue',s=20,marker='o')
                        title = current_dir + '/Optimal_solutions_'+func_name+' with '+str(gp.ns)+'-samples_in_'+str(itrial)+'-th_trial.png'
                        plt.savefig(title)
                        plt.close()
            except:
                break
    if n_trial > 1:
        dfs = []
        for i in range(n_trial):
            path = current_dir + '/' + fname_indicator + str(i+1) + '.csv'
            df = pd.read_csv(path)
            dfs.append(df.values)
        dfs = np.array(dfs)
        mean = np.mean(dfs, axis=0)
        std = np.std(dfs, axis=0)
        df_mean = pd.DataFrame(mean, columns=df.columns)
        df_std = pd.DataFrame(std, columns=df.columns)
        dfs = pd.concat([df_mean, df_std],axis=1)
        df_mean.to_csv(current_dir + '/' + fname_indicator + '_mean.csv', index=None)