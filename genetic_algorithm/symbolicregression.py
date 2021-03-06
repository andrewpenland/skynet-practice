from deap import base,creator,tools,gp, algorithms
import random
import operator
import math
import numpy

#create protected division so that we don't crash program 
#by dividing by zero

def protectedDiv(left,right):
   try:
      return left/right
   except ZeroDivisionError:
      return 1

#add a basic set of operators
#the "1" in the second argument of PrimitiveSet
#is the number of inputs.

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub,2)
pset.addPrimitive(operator.mul,2)
pset.addPrimitive(protectedDiv,2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos,1)
pset.addPrimitive(math.sin,1)
pset.addPrimitive(math.exp,1)
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
        
pset.addPrimitive(plog,1)

#Rename the variables

pset.renameArguments(ARG0='x')

#We need an individual with a genotype

creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

#Add some parameters from the toolbox

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_ = 2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#create a function that evaluates the individual at a number of points
#(essentially determines the fitness function. Notice that it returns
#a tuple (which I think is why there is a "," on the end), since fitness
#must be an iterable

#in addition to the mean root square error, I am going to
#penalize the function for each time that it returns 0

def evalSymbReg(individual,points):
   #Transform the tree expression into a callable function
   func = toolbox.compile(expr=individual)
   #Evaluate the mean squared error between the expression
   #and the real function: x**4 +  x**3 +  x**2 + x
   sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
   numzeros_bad_errors = 0
   for x in points:
       if math.fabs(func(x) - x**4 - x**3 - x**2 - x) >0.1:
           numzeros_bad_errors += 1000000
   #print(sqerrors)
   return (math.fsum(sqerrors)/len(points) + numzeros_bad_errors),

#this evaluates our symbolic expression at evenly spaced points in [-1,1]
#could just as easily make this be on a randomly generated set of points 
   
random.seed(318)

toolbox.register("evaluate", evalSymbReg, points = [x/10.0 for x in range(-10,10)])
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

# number of candidate solutions
pop = toolbox.population(n=300)
hof = tools.HallOfFame(10)
pop, log = algorithms.eaSimple(pop, toolbox, 0.5,0.1,40, stats=mstats, halloffame=hof,verbose=True)

#print(hof)
for x in hof:
    print(str(x) + '\n')

#expr = hof[0]
#nodes, edges, labels = gp.graph(expr)
#
#### Graphviz Section ###
#import pygraphviz as pgv
#
#g = pgv.AGraph()
#g.add_nodes_from(nodes)
#g.add_edges_from(edges)
#g.layout(prog="dot")
#
#for i in nodes:
#    n = g.get_node(i)
#    n.attr["label"] = labels[i]
#
#g.draw("tree.pdf")

