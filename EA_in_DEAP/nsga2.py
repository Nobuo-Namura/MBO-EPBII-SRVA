#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

def NSGA2(func, xmin, xmax, nx, weights, npop, ngen, p_cross, eta_cross=10.0, eta_mut=20.0, PRINT=False, PLOT=False):
    import array
    import random
    import json
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    from math import sqrt
    
    from deap import algorithms
    from deap import base
    from deap import benchmarks
    from deap.benchmarks.tools import diversity, convergence, hypervolume
    from deap import creator
    from deap import tools
    
    from mpl_toolkits.mplot3d import Axes3D
    
    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    BOUND_LOW, BOUND_UP = xmin, xmax
    NDIM = nx
    
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", func)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=eta_cross)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=eta_mut, indpb=1.0/NDIM)
    toolbox.register("select", tools.selNSGA2)
    
    def main(seed=None):
        random.seed(seed)
    
        NGEN = ngen
        MU = npop
        CXPB = p_cross
    
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "min", "max"
        
        pop = toolbox.population(n=MU)
        pop_ini = pop[:]
        
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
        pop = toolbox.select(pop, len(pop))
        
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        if PRINT:
            print(logbook.stream)
    
        for gen in range(1, NGEN):
    
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2)
                
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
    
            pop = toolbox.select(pop + offspring, MU)
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            if PRINT:
                print(logbook.stream)
    
#        print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))
    
        return pop, pop_ini, logbook
            
#    if __name__ == "__main__":
    pop, pop_ini, stats = main()

    fitnesses_ini = np.array([list(pop_ini[i].fitness.values) for i in range(len(pop_ini))])
    fitnesses = np.array([list(pop[i].fitness.values) for i in range(len(pop))])

    if PLOT:
        if len(fitnesses[0,:])==3:
            fig = plt.figure('objective-3D-NSGA2')
            ax = Axes3D(fig)
#            ax.scatter3D(fitnesses_ini[:,0],fitnesses_ini[:,1],fitnesses_ini[:,2],c='blue')
            ax.scatter3D(fitnesses[:,0],fitnesses[:,1],fitnesses[:,2],c='red')
            plt.savefig("fitnesses.png", dpi=300)
            plt.close()
        elif len(fitnesses[0,:])==2:
            plt.figure('objective-2D-NSGA2')
            plt.plot(fitnesses_ini[:,0], fitnesses_ini[:,1], "b.", label="Initial")
            plt.plot(fitnesses[:,0], fitnesses[:,1], "r.", label="Optimized" )
            plt.legend(loc="upper right")
            plt.title("fitnesses")
            plt.xlabel("f1")
            plt.ylabel("f2")
            plt.grid(True)
            plt.savefig("fitnesses.png", dpi=300)
            plt.close()
            
    return fitnesses, np.array(pop)