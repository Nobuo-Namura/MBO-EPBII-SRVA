def NSGA3(func, xmin, xmax, nx, weights, ngen, nref, refvec, p_cross=1.0, eta_cross=10.0, eta_mut=20.0, PRINT=False, PLOT=False):
    from math import factorial
    import random
    
    import matplotlib.pyplot as plt
    import numpy
    
    from deap import algorithms
    from deap import base
    from deap.benchmarks.tools import igd
    from deap import creator
    from deap import tools
    
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial import distance
    
    # Problem definition
    NDIM = nx
    BOUND_LOW, BOUND_UP = xmin, xmax
    ##
    
    # Algorithm parameters
    if nref%4 > 0:
        MU = int(nref + (4 - nref % 4))
    else:
        MU = int(nref)
    MUTPB = 1.0
    
    # Create classes
    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMin)
    ##
    
    
    # Toolbox initialization
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", func)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=eta_cross)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=eta_mut, indpb=1.0/NDIM)
    toolbox.register("select", tools.selNSGA3, ref_points=refvec)
    ##
    
    
    def main(seed=None):
        random.seed(seed)
    
        # Initialize statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
#        stats.register("avg", numpy.mean, axis=0)
#        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)
    
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "min", "max"
    
        pop = toolbox.population(n=MU)
        pop_ini = pop[:]
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
        # Compile statistics about the population
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)
    
        # Begin the generational process
        for gen in range(1, ngen):
            offspring = algorithms.varAnd(pop, toolbox, p_cross, MUTPB)
    
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
    
            # Select the next generation population from parents and offspring
            pop = toolbox.select(pop + offspring, MU)
    
            # Compile statistics about the new population
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)
    
        return pop, pop_ini, logbook
    
    
#    if __name__ == "__main__":
    pop, pop_ini, stats = main()
    fitnesses_ini = numpy.array([list(pop_ini[i].fitness.values) for i in range(len(pop_ini))])
    fitnesses = numpy.array([list(pop[i].fitness.values) for i in range(len(pop))])
    
    if PLOT:
        if len(fitnesses[0,:])==3:
            fig = plt.figure('objective-3D-NSGA3')
            ax = Axes3D(fig)
#            ax.scatter3D(fitnesses_ini[:,0],fitnesses_ini[:,1],fitnesses_ini[:,2],c='blue')
            ax.scatter3D(fitnesses[:,0],fitnesses[:,1],fitnesses[:,2],c='red')
            plt.savefig("fitnesses.png", dpi=300)
            plt.close()
        elif len(fitnesses[0,:])==2:
            plt.figure('objective-2D-NSGA3')
            plt.plot(fitnesses_ini[:,0], fitnesses_ini[:,1], "b.", label="Initial")
            plt.plot(fitnesses[:,0], fitnesses[:,1], "r.", label="Optimized" )
            plt.legend(loc="upper right")
            plt.title("fitnesses")
            plt.xlabel("f1")
            plt.ylabel("f2")
            plt.grid(True)
            plt.savefig("fitnesses.png", dpi=300)
            plt.close()


#    fig = plt.figure('NSGA3',figsize=(7, 7))
#    ax = fig.add_subplot(111, projection="3d")
#
#    p = numpy.array([ind.fitness.values for ind in pop])
#    ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker="o", s=24, label="Final Population")
#
#    ref_points = tools.uniform_reference_points(NOBJ, P)
#
#    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], marker="o", s=24, label="Reference Points")
#
#    ax.view_init(elev=11, azim=-25)
#    ax.autoscale(tight=True)
#    plt.legend()
#    plt.tight_layout()
#    plt.savefig("nsga3.png")
            
    return fitnesses, numpy.array(pop)