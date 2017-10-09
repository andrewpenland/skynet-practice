from deap import tools, base, creator, gp, algorithms
import operator
import math
import numpy
import random

def protectedDiv(left,right):
   try:
      return left/right
   except ZeroDivisionError:
      return 1

def protectedSquareRoot(x):
   if x < 0:
      return 0
   else:
      return math.sqrt(x)

pset = gp.PrimitiveSet("MAIN",3)
pset.addPrimitive(operator.add,2)
pset.addPrimitive(operator.mul,2)
pset.addPrimitive(operator.sub,2)
pset.addPrimitive(operator.neg,1)
#pset.addPrimitive(math.cos,1)
#pset.addPrimitive(math.sin,1)
pset.addPrimitive(protectedDiv,2)
pset.addPrimitive(protectedSquareRoot,1)
#pset.addEphemeralConstant("rand101", lambda: 2**(random.randint(-10,10)) * random.randint(-1,1))

pset.renameArguments(ARG0='a',ARG1='b',ARG2='c')

#I still don't really understand what this is doing, but let's go with it.

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

#Now we add a bunch of things to the toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_ = 1, max_ = 2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#helper function to generate a bunch of polynomials
#we will assume monic ( a = 1)

def generatePolynomialSpace(numpoints):
   polylist = []
   for i in range(numpoints):
       a = 1
       b = random.randint(-10,10)
       c = random.uniform(0, 0.25 * b**2)
       polylist.append([a,b,c])
   return polylist

def evalSymbReg(individual,testnum):
   #Transform the tree expression into a callable function
   func = toolbox.compile(expr=individual)
   #Generate the set of points
   points = generatePolynomialSpace(10)
   #Evaluate the mean squared error, AND penalize for identically zero
   sqerrors = (math.fabs(func(x[0],x[1],x[2])) for x in points)
   is_constant = (func(1,1,random.uniform(2,3)) == 0)
   return (math.fsum(sqerrors)/testnum + 100 * is_constant),

#test, no longer needed: print(generatePolynomialSpace(3))

toolbox.register("evaluate", evalSymbReg, testnum = 10)
toolbox.register("select", tools.selTournament, tournsize = 3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_ = 0, max_ = 2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"),max_value = 17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"),max_value=17))

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)

mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg",numpy.mean)
mstats.register("std",numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

pop = toolbox.population(3000)
hof = tools.HallOfFame(10)
pop, log = algorithms.eaSimple(pop,toolbox, 0.5,0.1, 400, stats=mstats, halloffame=hof, verbose=True)

       
for x in hof:
   print(str(x) + "\n")


