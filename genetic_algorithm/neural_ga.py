# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:12:42 2017

@author: jasplund
"""

from deap import base,creator,tools,gp, algorithms
import random
import operator
import math
import numpy
import pandas as pd

from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
import sklearn.ensemble as ske
from sklearn import linear_model
from sklearn.model_selection import train_test_split

digits_train_df = pd.read_csv('optdigits.tra',header=None,prefix='x')

digits_test_df = pd.read_csv('optdigits.tes', header=None, prefix='x')
print(str(digits_test_df.head()))

train_array = digits_train_df.values
X_train = train_array[:,0:64]
y_train = train_array[:,64]

print(len(X_train))

test_array = digits_test_df.values
X_test = test_array[:,0:64]
y_test = test_array[:,64]

print(str(y_train), ' ', len(y_train))

def log_func(x):
    return 10.0/(1.0+math.exp(-x))

def fitness_opt(individual):
    func = toolbox.compile(expr=individual)
    total = 0
#    print(y_train[0])
#    print(str(individual), ' ',len(y_train))
    for i in range(len(y_train)):
        try:
            xt = X_train[i]
            total += (log_func(func(*xt))-y_train[i])**2
        except OverflowError:
            total = 1000000
    return total,
    
def protectedDiv(left,right):
   if right==0:
       return 1
   else:
      return left/right

#add a basic set of operators
#the "1" in the second argument of PrimitiveSet
#is the number of inputs.

pset = gp.PrimitiveSet("MAIN", 64)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub,2)
pset.addPrimitive(operator.mul,2)
pset.addPrimitive(protectedDiv,2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos,1)
pset.addPrimitive(math.sin,1)
#pset.addPrimitive(math.exp,1)
#pset.addPrimitive(math.sqrt,1)
#pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))	

# This is the protected logorithm definition
def plog(x):
    if x > 0:
        return math.log(x)
    if x < 0:
        return math.log(math.fabs(x))
    else:
        return 0
        
#pset.addPrimitive(plog,1)


#We need an individual with a genotype

creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

#Add some parameters from the toolbox

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_ = 2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", fitness_opt)
toolbox.register("select", tools.selTournament, tournsize = 3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_ = 0, max_ = 2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value = 30))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value = 30))

#now we set up statistics 

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg",numpy.mean)
mstats.register("std",numpy.std)
mstats.register("min",numpy.min)
mstats.register("max",numpy.max)

# number of candidate solutions
pop = toolbox.population(n=300)
hof = tools.HallOfFame(10)
#pop, log = algorithms.eaSimple(pop, toolbox, 0.5,0.1,40, stats=mstats, halloffame=hof,verbose=True)

#print(hof)
for x in hof:
    print(str(x) + '\n')
    
def miracle_func(t):
    x = X_train[t]
    return -(protectedDiv(x[63],(x[29]*x[56]-(math.sin(x[16])-(x[29]*x[63]*x[56]-(x[39]-x[28]))))))

print(sum([math.floor(log_func(miracle_func(t)))==y_train[t] for t in range(len(y_train))])/len(y_train))

    