# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:43:24 2017

@author: jasplund
"""

#Practice competition from Kaggle House prices

#same header/import information as before.
import matplotlib.pyplot as plt
#matplotlib inline
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
from sklearn.neural_network import MLPClassifier
import sklearn.ensemble as ske
from scipy import stats
from statistics import mean
import random
from matplotlib import style
style.use('fivethirtyeight')

house_test_df = pd.read_csv('test.csv', index_col = None)
#print(str(house_test_df.head()))
house_train_df = pd.read_csv('train.csv', index_col = None)
print(str(house_train_df.head()))

# .mean() will average the data so that only the average in gravel and paved
# will show and nothing else. 
class_alley_grouping = house_train_df.groupby(['Alley']).mean()
print(str(class_alley_grouping))
class_alley_grouping['SalePrice'].plot.bar()
plt.show()

# do 2-sample t-test on the data to determine which is significant.
# We will do this for each variable. 
#group1 = house_train_df[house_train_df['Alley'] == 'Pave']
#group2 = house_train_df[house_train_df['Alley'] == 'Grvl']
#t, p = ttest_ind(group1['SalePrice'],group2['SalePrice'])

# This is the test statistic
#print(str(t))
# This is the p-value
#print(str(p))

#if (p < 0.05):
#    print("We reject the null hypothesis -- the two groups have different means.")

#else:
#    print("We can not reject the null hypothesis -- the groups may have the same mean.")

#house_train_df.boxplot('SalePrice', by='Alley', figsize=(12, 8)) 

#ctrl = house_train_df['SalePrice'][data.group == 'ctrl']
 
#grps = pd.unique(data.group.values)
#d_data = {grp:data['weight'][data.group == grp] for grp in grps}
 
#k = len(pd.unique(data.group))  # number of conditions
#N = len(data.values)  # conditions times participants
#n = data.groupby('group').size()[0] #Participants in each condition

#SSbetween = (sum(data.groupby('group').sum()['weight']**2)/n) - (data['weight'].sum()**2)/N
#sum_y_squared = sum([value**2 for value in data['weight'].values])
#SSwithin = sum_y_squared - sum(data.groupby('group').sum()['weight']**2)/n
#SStotal = sum_y_squared - (data['weight'].sum()**2)/N
#MSbetween = SSbetween/DFbetween
#F = MSbetween/MSwithin
#p = stats.f.sf(F, DFbetween, DFwithin)

#This is the effect size.
#eta_sqrd = SSbetween/SStotal 
#om_sqrd = (SSbetween - (DFbetween * MSwithin))/(SStotal + MSwithin)

# pass this hm="how many data points", variance, step=how far on step up
# the y-value per point, correlation
def create_dataset(hm, variance, step=2, correlation=False):
    val=1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys,dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / 
         ((mean(xs)**2)-mean(xs**2)))
    b= mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_reg = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 -(squared_error_reg/squared_error_y_mean)
    
def reg_line(xdata,ydata):
    #create best fit line
    xs = np.array(xdata)
    ys = np.array(ydata)
    regression_line = [(m*x)+b for x in xs]
    return regression_line

def reg_assumptions():
    # Test assumptions for regression analysis.
    # First test that the scatterplot is close to linear. 
    
    # Next, test that the residual plot shows a random pattern. The residual 
    # value is the observed value minus the predicted value.
    return
    
#xs, ys = create_dataset(40, 80, 2, correlation='pos')
xs, ys = create_dataset(40, 80, 2, correlation=False)

m,b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x)+b for x in xs]
predict_x = 8
predict_y = (m*predict_x)+b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,s=100, color='g')
plt.plot(xs,regression_line)
plt.show()