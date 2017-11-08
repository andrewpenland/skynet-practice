import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation,tree,preprocessing,metrics
import sklearn.ensemble as ske
from sklearn.neural_network import MLPClassifier
from pickle import dump
from deap import base,creator,tools,gp, algorithms
import random
import operator
import math
import numpy

digits_train_df = pd.read_csv('optdigits.tra',header=None,prefix='x')
print(str(digits_train_df.head()))

digits_test_df = pd.read_csv('optdigits.tes',header=None,prefix='x')

test_vals = [0,4]

#Get the training X and y values
train_array = digits_train_df.values
array_to_keep = [list(x) for x in train_array if (x[64] == 0 or x[64] == 4)]
X_train = [x[0:64] for x in array_to_keep]
Y_train = [array_to_keep[k][64] for k in range(len(X_train))]
print(str(len(array_to_keep)))

#create protected division so that we don't crash program
#by dividing by zero

def protectedDiv(left,right):
   try:
      return left/right
   except ZeroDivisionError:
      return 1

def protectedLog(x):
   if x == 0:
     return 0
   else:
     return math.log(abs(x))

def pos_max(x):
   return max(x,0)

def mean(x,y):
    return 0.5*x + 0.5*y
    
def create_max_with_n_args(n):
   def func1(*args):
      return max(args)
   func1.__name__ = "max"+str(n)
   return func1

def create_min_with_n_args(n):
   def func1(*args):
      return min(args)
   func1.__name__ = "min"+str(n)
   return func1
   
#add a basic set of operators
#the "8" in the second argument of PrimitiveSet
#is the number of inputs.

pset = gp.PrimitiveSet("MAIN", 64)
# pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub,2)
# pset.addPrimitive(operator.mul,2)
# pset.addPrimitive(protectedDiv,2)
# pset.addPrimitive(pos_max,1)
# pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(math.cos,1)
# pset.addPrimitive(math.sin,1)
# pset.addPrimitive(math.exp,1)
pset.addPrimitive(max,2)
pset.addPrimitive(min,2)
#for k in range(3,7):
#   pset.addPrimitive(create_max_with_n_args(k),k)
#   pset.addPrimitive(create_min_with_n_args(k),k)
pset.addPrimitive(operator.gt,2)
pset.addPrimitive(operator.lt,2)
pset.addPrimitive(abs,1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-10,10))

#We need an individual with a genotype

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

#Add some parameters from the toolbox

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_ = 2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#set up the fitness function
#this gives the number of times that func perfectly predicts the output
#this is really the only fitness function that makes sense

def evalSymbReg(individual):
   #Transform the tree expression into a callable function
   func = toolbox.compile(expr=individual)
   hits = sum([4*func(*X_train[i]) == Y_train[i] for i in range(len(Y_train))])
   return hits,

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize = 3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_ = 0, max_ = 2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value = 17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value = 17))

#now we set up statistics

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg",numpy.mean)
mstats.register("std",numpy.std)
mstats.register("min",numpy.min)
mstats.register("max",numpy.max)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(10)
pop, log = algorithms.eaSimple(pop, toolbox, 0.5,0.1,20, stats=mstats, halloffame=hof,verbose=True)

#print hof
#print([str(x) for x in hof])
for x in hof:
   print(str(x) + "\n")
