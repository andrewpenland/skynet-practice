# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:20:06 2017

@author: cecil
"""

#First we need to convert all xls files to csv.

#In this program, we take all the data scraped and put it together 
#into a single csv file where each row is a precinct and each column is a race
# with a particular category. 
import pandas as pd
import csv
import numpy as np
import os

##path =  'C:\\Users\\jasplund\\Dropbox\\research\\gerry\\voting_data\\data'
##filenames = os.listdir(path)
##print(filenames)

#Make a set of all the contests in the election
contests = []
with open('C:\\Users\\cecil\\Dropbox\\research\\gerry\\voting_data\\data\\2012\\z_election_data_all_2012.csv') as csvfile:
    readFile = csv.reader(csvfile, delimiter=',')
    for row in readFile:
        if not ''==row[1] and not 'Contest'==row[1] and not 'Registered Voters'==row[1]:
            contests.append(row[1])


#Set the path and grab the list of files from the path that contains
#all of the csv files from which we wish to form a data set.
path =  'C:\\Users\\cecil\\Dropbox\\research\\gerry\\voting_data\\data\\2012\\csv'
pre_files = os.listdir(path)

#This will remove all of the files that are not relevant such as the
#set of registered voters and the table of contents. They can be identified
#by the files that contain a space in their name.
filenames = []
for file in pre_files:
    if len(str.split(file))==1:
        filenames.append(file)

# This is just a temporary directory to hold the data frames.
p={}
for contest in contests:
    p[contest] = 0

#Here is the main function that will form the dataframe called all_data.
#Before we can make all_data, we have to put all of the dataframes into
#a temporary directory. Then we will complete the for loop and finally
#run pd.Panel to form the data frame.
for file in filenames[0:11]:
    data1 = []
    #This grabs the file we are currently working on and turns it into a list.
    with open('C:\\Users\\cecil\\Dropbox\\research\\gerry\\voting_data\\data\\2012\\csv\\'+file) as csvfile:
        readFile = csv.reader(csvfile, delimiter=',')
        for row in readFile:
            if not 'Totals:'==row[0]:
                data1.append(row)

    #There are some elections that did not appear in the original list of 
    #elections. To remove these elections, we must catch them. That is what
    #this 'if' statement does.
    if not [key for key, value in p.items() if data1[0][0].lower() in key.lower()]==[]:
        
        #We next turn that list into a dataframe ignoring the first two rows.
        #The first row contains the contest and the second row contains
        #the candidates.
        df_data1 = pd.DataFrame(data1[3:], columns = data1[2])
        df_data1 = df_data1.set_index('Precinct')
    
        #We will next form a set that contains nothing but the results of the 
        #election. That would be the total votes. There may be several columns 
        #that have the same name and the total votes always contains the total
        #votes for each election.
        totals1 = df_data1[['Total Votes']]
    
        #This new column set contains the columns we wish to totals1 to contain.
        new_col1 = ['Votes for '+name for name in data1[1] if not ''==name]
    
        #Rename the columns of totals1
#        totals1 = totals1.rename(columns = {'Total Votes': new_col1[0]})
        totals1.columns = new_col1
#        print(totals1)
    
        #We must now insert totals into its proper place in p. 
        #We break into the cases where either p already has a set of data at
        #the indicated index and the case where p does not have anything at 
        #the indicated index.
        if isinstance(p[data1[0][0]],int):
            #Just add totals1 to our dictionary p.
            p[data1[0][0]] = totals1
        else:
#            print(data1[0][0])
#            print(p[data1[0][0]])
#            print(totals1)
            p[data1[0][0]] = pd.concat([p[data1[0][0]],totals1], axis=0)
            
    #This is where we go if the election is not in our original list of 
    #elections.
    else:
        #later, we may want to see what these elections are that were not 
        #covered by the elections previously.
        pass
            
    

#We now form the entire dataframe
#print(p)
for k,v in p.items():
    if isinstance(v,int):
        p[k] = pd.DataFrame()
#print(p)
all_data = pd.Panel(data = p)
print(all_data[contests[0]]) 