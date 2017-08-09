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
        
def main(): 
    ttest_test()
    
if __name__ == "__main__":
    main()

