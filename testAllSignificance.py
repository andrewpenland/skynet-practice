# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:28:59 2017

@author: jasplund
"""

#same header/import information as before.
import matplotlib.pyplot as plt
#matplotlib inline
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
from sklearn.neural_network import MLPClassifier
import sklearn.ensemble as ske
from scipy.stats import ttest_ind
from scipy import stats
from matplotlib import style
style.use('fivethirtyeight')
 
def ttest(df,val,group1,group2,output,significance_level = 0.05):
    group1 = df[df[val] == group1]
    group2 = df[df[val] == group2]
    t, p = ttest_ind(group1[output],group2[output])
    if (p <= significance_level):
        return 1
        #print("We reject the null hypothesis -- the two groups have different means.")
    else:
        return 0
        #print("We can not reject the null hypothesis -- the groups may have the same mean.")

def ttest_test():
    print("We are testing the ttest function on the Ames housing data.")
    house_train_df = pd.read_csv('train.csv', index_col = None)
    pvalue_pass = ttest(house_train_df,'Alley','Pave','Grvl','SalePrice')
    print("The answer on the Ames Housing Data with Gravel vs. Paving is:")
    print(str(pvalue_pass))
    if pvalue_pass == 1:
        print("This is correct.")
    else:
        print("This is incorrect.")

#data.boxplot('weight', by='group', figsize=(12, 8))
 
#ctrl = data['weight'][data.group == 'ctrl']
 
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
#eta_sqrd = SSbetween/SStotal
#om_sqrd = (SSbetween - (DFbetween * MSwithin))/(SStotal + MSwithin)

def anova_assumptions():
    #test assumptions for ANOVA test here.
    
def anova_test():
    # If assumptions for ANOVA are satisfied use ANOVA test. This will
    # test if there is a significant difference between some pair of 
    # samples. We do not need to test to find which one it is.
    
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
    squared_error_y_mean = squared_error(ys_orig, ys_mean_line)
    return 1 -(squared_error_reg/squared_error_y_mean)
    
def reg_line(xdata,ydata):
    #create best fit line
    xs = np.array(xdata)
    ys = np.array(ydata)
    return regression_line = [(m*x)+b for x in xs]

def reg_assumptions():
    # Test assumptions for regression analysis.
    # First test that the scatterplot is close to linear. 
    
    # Next, test that the residual plot shows a random pattern. The residual 
    # value is the observed value minus the predicted value.
    
    
def reg_test(xdata,ydata):
    # run hypothesis test to determine whether there is a significant linear 
    # relationship between an independent variable X and a dependent 
    # variable Y
    reg_line(xdata,ydata)
    r_squared = coefficient_of_determination(ys, regression_line)
        
        
def sig_test():
    house_train_df = pd.read_csv('train.csv', index_col = None)
    for name in house_train_df:
        if name!='SalePrice':
            pvalue_pass = ttest(house_train_df,'Alley','Pave','Grvl','SalePrice') 
    

        
def main(): 
    #ttest_test()
    #The following code will help reshape a list of lists prediction in any
    # sklearn algorithm
    #example_measures = np.array([[prediction 1], [prediction 2]])
    #example_measures = example_measures.reshape(len(example_measures,-1))
    #prediction = clf.predict(example_measures)
    
if __name__ == "__main__":
    main()

