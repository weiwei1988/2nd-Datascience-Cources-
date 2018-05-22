# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 21:12:53 2017

@author: zhaow
"""

import numpy as np
import pandas as pd
import os
from sklearn import neighbors
from sklearn import svm, neural_network

os.chdir("C:/Users/zhaow/Desktop/titanic")

data = pd.read_csv('train.csv')
#data = data.dropna()
#data.pivot_table(values = ['Survived'], index = ['Sex'], columns = ['Embarked'], aggfunc = 'sum')

#性別ダミーデータの作成
sex_dummies = pd.get_dummies(data['Sex'])
sex_dummies.columns = ['Sex_dummy', 'Drop']
sex_dummies = sex_dummies.drop('Drop', axis = 1)

group = data.groupby('Embarked')['Embarked'].count()

#乗船地Emarkedダミーの作成
emb_dummies = pd.get_dummies(data['Embarked'])
emb_dummies.columns = ['Embarked_Q', 'Embarked_C', 'Embarked_S']

data = pd.concat([data, sex_dummies, emb_dummies], axis = 1)
del data['Sex']
del data['Embarked']

data.isnull().sum()
data.shape

del data['Cabin']

#data_1 = data.drop('PassengerId', axis = 1)
#data_1 = data
#data_1 = data_1.dropna()

def func(x):
    if x == True:
        return 1
    else:
        return 0

#敬称の取得、ダミーデータ化
title_list = ['Mrs.', 'Master.', 'Miss.']
for title in title_list:
    data[title] = data['Name'].str.contains(title).apply(func)
    
mr = []
for i in range(len(data)):
    if data.ix[i, 'Mrs.'] == 1:
        mr.append(0)
    elif data.ix[i, 'Master.'] == 1:
        mr.append(0)
    elif data.ix[i, 'Miss.'] == 1:
        mr.append(0)
    else:
        mr.append(1)

data['Mr.'] = pd.DataFrame(mr)

temp = data['Name'].str.split(',', expand = True)
data['Family Name'] = temp.iloc[:, 0]
name_list = list(temp.iloc[:, 0])

#ファミリーネームから家族の有無の判定、自分以外の家族の生存率を計算
family_surive_rate = [0 for i in range(len(name_list))]
i = 0

for i in range(len(name_list)):
    temp_data = data.drop(i)
    family_surive_rate[i] = temp_data[temp_data['Family Name'] == name_list[i]]['Survived'].sum() / temp_data[temp_data['Family Name'] == name_list[i]]['Survived'].count()
    del temp_data

family_surive_rate = pd.DataFrame(family_surive_rate)
family_surive_rate.columns = ['Family survive rate']
data = pd.concat([data, family_surive_rate], axis = 1)
data['Family survive rate'] = data['Family survive rate'].fillna(0)

#敬称の年齢平均値を使って年齢NaN値を補完
#data_mr = data_1[data_1['Mr.'] == 1]
#data_mr['Age'] = data_mr['Age'].fillna(data_mr['Age'].mean())
#data_master = data_1[data_1['Master.'] == 1]
#data_master['Age'] = data_master['Age'].fillna(data_master['Age'].mean())
#data_mrs = data_1[data_1['Mrs.'] == 1]
#data_mrs['Age'] = data_mrs['Age'].fillna(data_mrs['Age'].mean())
#data_miss = data_1[data_1['Miss.'] == 1]
#data_miss['Age'] = data_miss['Age'].fillna(data_miss['Age'].mean())
#data_train = pd.concat([data_mr, data_master, data_mrs, data_miss])
data_train = data
data_train = data_train.drop('Name', axis = 1).drop('Family Name', axis = 1).drop('Ticket', axis = 1)
#data_train = data_train.dropna()
#del data_train['Miss.']

#敬称ごとのデータでknnを利用して年齢予測、年齢NaN値を補完

def mean_fillna(data, title):
    data_mr = data[data[title] == 1]
    X_train = data_mr[data_mr['Age'].notnull()].drop('Survived', axis =1)
    X_predict = data_mr[data_mr['Age'].isnull()].drop('Survived', axis =1)
    X_predict['Age'] = X_predict['Age'].fillna(X_train['Age'].mean())
    return pd.concat([X_train, X_predict]).sort_values(by = 'PassengerId')

def knn_fillna(data, title):
    clf = neighbors.KNeighborsRegressor(n_neighbors = 3)
    data_mr = data[data[title] == 1]
    X_train = data_mr[data_mr['Age'].notnull()].drop('Survived', axis =1)
    X_predict = data_mr[data_mr['Age'].isnull()].drop('Survived', axis =1)
    clf.fit(X_train.drop('Age', axis = 1).drop('PassengerId', axis = 1), X_train['Age'])
    X_predict['Age'] = clf.predict(X_predict.drop('Age', axis = 1).drop('PassengerId', axis = 1))
    return pd.concat([X_train, X_predict]).sort_values(by = 'PassengerId')

def svm_fillna(data, title):
    clf = svm.SVR(kernel = 'linear')
    data_mr = data[data[title] == 1]
    X_train = data_mr[data_mr['Age'].notnull()].drop('Survived', axis =1)
    X_predict = data_mr[data_mr['Age'].isnull()].drop('Survived', axis =1)
    clf.fit(X_train.drop('Age', axis = 1).drop('PassengerId', axis = 1), X_train['Age'])
    X_predict['Age'] = clf.predict(X_predict.drop('Age', axis = 1).drop('PassengerId', axis = 1))
    return pd.concat([X_train, X_predict]).sort_values(by = 'PassengerId')

def neu_fillna(data, title):
    clf = neural_network.MLPRegressor()
    data_mr = data[data[title] == 1]
    X_train = data_mr[data_mr['Age'].notnull()].drop('Survived', axis =1)
    X_predict = data_mr[data_mr['Age'].isnull()].drop('Survived', axis =1)
    clf.fit(X_train.drop('Age', axis = 1).drop('PassengerId', axis = 1), X_train['Age'])
    X_predict['Age'] = clf.predict(X_predict.drop('Age', axis = 1).drop('PassengerId', axis = 1))
    return pd.concat([X_train, X_predict]).sort_values(by = 'PassengerId')

title_list = ['Mrs.', 'Master.', 'Miss.', 'Mr.']

data_out = pd.DataFrame()
fill_method = {'mean': mean_fillna, 'knn': knn_fillna, 'svm': svm_fillna, 'neural': neu_fillna}

select = 'svm'
func = fill_method[select]
survived = data_train['Survived']


for title in title_list:    
    d1 = func(data_train, title)
    data_out = pd.concat([data_out, d1]).sort_values(by = 'PassengerId')
    
data_out = pd.concat([data_out, survived], axis = 1)
    
data_out.to_csv('titanic_train.csv', index = False)
