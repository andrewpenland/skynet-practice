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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
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

#print(list(house_test_df))

print(list(house_train_df.dtypes.names))


def carefulnanchecklist(x):
    for elt in x:
        if type(elt) is str:
            pass
        else:
            if np.isnan(elt):
                return True
    return False

#if house_train_df.dtypes[0] == np.int64:
#    print("success")
#    if carefulnanchecklist(house_train_df.values[:,33]):
#        print("double success")
#        test_df = house_train_df.copy()
#        le = preprocessing.LabelEncoder()
#        print(str(test_df.values[:,33]))
#        test_df.values[:,33] = [str(x) for x in test_df.values[:,33]]
#        test_df.values[:,33] = le.fit_transform(test_df.values[:,33])
#        print(test_df)
    

# .mean() will average the data so that only the average in gravel and paved
# will show and nothing else. 
class_alley_grouping = house_train_df.groupby(['Alley']).mean()
print(str(class_alley_grouping))
class_alley_grouping['SalePrice'].plot.bar()
plt.show()

def preprocess_house_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    #replace sex and embarked with integers
    i=0
    for name in processed_df.dtypes:
        actual_name = list(processed_df)
        if type(name) is np.float64:
            pass
        if type(name) is np.int64:
            pass
        if type(name) is np.object:
            if carefulnanchecklist(processed_df.values[:,i]):
#                pass
                processed_df = processed_df.drop(name, 1)
                i=i-1
#                processed_df.values[:,i] = [str(x) for x in processed_df.values[:,i]]
#                processed_df.values[:,i] = le.fit_transform(processed_df.values[:,i])
            else: 
                if name == BsmtFinType2 or name == BsmtFinType1:
                    processed_df = processed_df.drop(name, 1)
                else:
                    processed_df.values[:,i] = le.fit_transform(processed_df.values[:,i]) 
                    i=i-1
#       
        i += 1
        
#    processed_df.MSZoning = le.fit_transform(processed_df.MSZoning)
#    processed_df.Street = le.fit_transform(processed_df.Street)
#    processed_df.Alley = [str(x) for x in processed_df.Alley]
#    processed_df.Alley = le.fit_transform(processed_df.Alley)
#    processed_df.LotShape= le.fit_transform(processed_df.LotShape)
#    processed_df.LandContour= le.fit_transform(processed_df.LandContour)
#    processed_df.Utilities= le.fit_transform(processed_df.Utilities)
#    processed_df.LotConfig= le.fit_transform(processed_df.LotConfig)
#    processed_df.LandSlope= le.fit_transform(processed_df.LandSlope)
#    processed_df.Neighborhood= le.fit_transform(processed_df.Neighborhood)
#    processed_df.Condition1= le.fit_transform(processed_df.Condition1)
#    processed_df.Condition2= le.fit_transform(processed_df.Condition2)
#    processed_df.BldgType= le.fit_transform(processed_df.BldgType)
#    processed_df.BldgType= le.fit_transform(processed_df.BldgType)
#    processed_df.HouseStyle= le.fit_transform(processed_df.HouseStyle)
#    processed_df.RoofStyle= le.fit_transform(processed_df.RoofStyle)
#    processed_df.RoofMatl= le.fit_transform(processed_df.RoofMatl)
#    processed_df.Exterior1st= le.fit_transform(processed_df.Exterior1st)
#    processed_df.Exterior2nd= le.fit_transform(processed_df.Exterior2nd)
#    processed_df.MasVnrType = [str(x) for x in processed_df.MasVnrType]
#    processed_df.MasVnrType = le.fit_transform(processed_df.MasVnrType)
#    processed_df.ExterQual= le.fit_transform(processed_df.ExterQual)
#    processed_df.ExterCond= le.fit_transform(processed_df.ExterCond)
#    processed_df.Foundation= le.fit_transform(processed_df.Foundation)
#    processed_df.BsmtQual = [str(x) for x in processed_df.BsmtQual]
#    processed_df.BsmtQual = le.fit_transform(processed_df.BsmtQual)
#    processed_df.BsmtCond = [str(x) for x in processed_df.BsmtCond]
#    processed_df.BsmtCond = le.fit_transform(processed_df.BsmtCond)
#    processed_df.BsmtExposure = [str(x) for x in processed_df.BsmtExposure]
#    processed_df.BsmtExposure = le.fit_transform(processed_df.BsmtExposure)
#    processed_df.BsmtFinType1 = [str(x) for x in processed_df.BsmtFinType1]
#    processed_df.BsmtFinType1 = le.fit_transform(processed_df.BsmtFinType1)
#    processed_df.BsmtFinType2 = [str(x) for x in processed_df.BsmtFinType2]
#    processed_df.BsmtFinType2 = le.fit_transform(processed_df.BsmtFinType2)
#    processed_df.PoolQC = [str(x) for x in processed_df.PoolQC]
#    processed_df.PoolQC = le.fit_transform(processed_df.PoolQC)
#    processed_df.Fence = [str(x) for x in processed_df.Fence]
#    processed_df.Fence = le.fit_transform(processed_df.Fence)
#    processed_df.MiscFeature = [str(x) for x in processed_df.MiscFeature]
#    processed_df.MiscFeature = le.fit_transform(processed_df.MiscFeature)
#    processed_df.SaleType= le.fit_transform(processed_df.SaleType)
#    processed_df.SaleCondition= le.fit_transform(processed_df.SaleCondition)
    #processed_df = processed_df.drop(['name'], axis=1)
    #drop these categories
#    processed_df = processed_df.drop(['home.dest'],axis=1)
    return processed_df

processed_df = preprocess_house_df(house_train_df)



#Univariate Selection
#
#
#Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
X = processed_df.values[:,0:37]
Y = processed_df.values[:,37]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X,Y)
#summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summareize selected features
print(features[0:5,:])

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
#def create_dataset(hm, variance, step=2, correlation=False):
#    val=1
#    ys = []
#    for i in range(hm):
#        y = val + random.randrange(-variance,variance)
#        ys.append(y)
#        if correlation and correlation=='pos':
#            val+=step
#        elif correlation and correlation == 'neg':
#            val -= step
#    xs = [i for i in range(len(ys))]
#    return np.array(xs, dtype=np.float64), np.array(ys,dtype=np.float64)
#
#def best_fit_slope_and_intercept(xs,ys):
#    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / 
#         ((mean(xs)**2)-mean(xs**2)))
#    b= mean(ys) - m*mean(xs)
#    return m, b
#
#def squared_error(ys_orig, ys_line):
#    return sum((ys_line-ys_orig)**2)
#
#def coefficient_of_determination(ys_orig,ys_line):
#    y_mean_line = [mean(ys_orig) for y in ys_orig]
#    squared_error_reg = squared_error(ys_orig, ys_line)
#    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
#    return 1 -(squared_error_reg/squared_error_y_mean)
#    
#def reg_line(xdata,ydata):
#    #create best fit line
#    xs = np.array(xdata)
#    ys = np.array(ydata)
#    regression_line = [(m*x)+b for x in xs]
#    return regression_line
#
#def reg_assumptions():
#    # Test assumptions for regression analysis.
#    # First test that the scatterplot is close to linear. 
#    
#    # Next, test that the residual plot shows a random pattern. The residual 
#    # value is the observed value minus the predicted value.
#    return
#    
##xs, ys = create_dataset(40, 80, 2, correlation='pos')
#xs, ys = create_dataset(40, 80, 2, correlation=False)
#
#m,b = best_fit_slope_and_intercept(xs,ys)
#regression_line = [(m*x)+b for x in xs]
#predict_x = 8
#predict_y = (m*predict_x)+b
#
#r_squared = coefficient_of_determination(ys, regression_line)
#print(r_squared)

#plt.scatter(xs,ys)
#plt.scatter(predict_x,predict_y,s=100, color='g')
#plt.plot(xs,regression_line)
#plt.show()