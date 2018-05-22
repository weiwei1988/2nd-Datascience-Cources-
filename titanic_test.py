# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 00:22:16 2018

@author: zhaow
"""

import numpy as np
import pandas as pd
import os
import titanic

os.chdir("C:/Users/zhaow/Desktop/titanic")

data = pd.read_csv('test.csv')

sex_dummies = pd.get_dummies(data['Sex'])
sex_dummies.columns = ['Sex_dummy', 'Drop']
sex_dummies = sex_dummies.drop('Drop', axis = 1)

group = data.groupby('Embarked')['Embarked'].count()

emb_dummies = pd.get_dummies(data['Embarked'])
emb_dummies.columns = ['Embarked_Q', 'Embarked_C', 'Embarked_S']
#emb_dummies = emb_dummies.drop('Drop', axis = 1)

data = pd.concat([data, sex_dummies, emb_dummies], axis = 1)
del data['Sex']
del data['Embarked']

data.isnull().sum()
data.shape

del data['Cabin']

#data_1 = data.drop('PassengerId', axis = 1)
data_1 = data
#data_1 = data_1.dropna()

def func(x):
    if x == True:
        return 1
    else:
        return 0

title_list = ['Mrs.', 'Master.', 'Miss.']
for title in title_list:
    data_1[title] = data_1['Name'].str.contains(title).apply(func)

mr = []
for i in range(len(data)):
    if data_1.ix[i, 'Mrs.'] == 1:
        mr.append(0)
    elif data_1.ix[i, 'Master.'] == 1:
        mr.append(0)
    elif data_1.ix[i, 'Miss.'] == 1:
        mr.append(0)
    else:
        mr.append(1)

data_1['Mr.'] = pd.DataFrame(mr)

temp = data_1['Name'].str.split(',', expand = True)
data_1['Family Name'] = temp.iloc[:, 0]
name_list = list(temp.iloc[:, 0])

family_surive_rate = [0 for i in range(len(name_list))]
i = 0

for i in range(len(name_list)):
    temp_data = titanic.data
    family_surive_rate[i] = temp_data[temp_data['Family Name'] == name_list[i]]['Survived'].sum() / temp_data[temp_data['Family Name'] == name_list[i]]['Survived'].count()
    del temp_data

family_surive_rate = pd.DataFrame(family_surive_rate)
family_surive_rate.columns = ['Family survive rate']
data_1 = pd.concat([data_1, family_surive_rate], axis = 1)
data_1['Family survive rate'] = data_1['Family survive rate'].fillna(0)

#data_mr = data_1[data_1['Mr.'] == 1]
#data_mr['Age'] = data_mr['Age'].fillna(data_mr['Age'].mean())

#data_master = data_1[data_1['Master.'] == 1]
#data_master['Age'] = data_master['Age'].fillna(data_master['Age'].mean())

#data_mrs = data_1[data_1['Mrs.'] == 1]
#data_mrs['Age'] = data_mrs['Age'].fillna(data_mrs['Age'].mean())

#data_miss = data_1[data_1['Miss.'] == 1]
#data_miss['Age'] = data_miss['Age'].fillna(data_miss['Age'].mean())

#data_test = pd.concat([data_mr, data_master, data_mrs, data_miss])

data_test = data_1
data_test = data_test.drop('Name', axis = 1).drop('Family Name', axis = 1).drop('Ticket', axis = 1)
data_test['Fare'] = data_test['Fare'].fillna(0)
title_list = ['Mrs.', 'Master.', 'Miss.', 'Mr.']
data_test['Survived'] = pd.Series([0 for i in range(len(data_test))])

data_out = pd.DataFrame()
for title in title_list:
    d1 = titanic.knn_fillna(data_test, title)
    data_out = pd.concat([data_out, d1]).sort_values(by = 'PassengerId')

data_out.to_csv('titanic_test.csv', index = False)
