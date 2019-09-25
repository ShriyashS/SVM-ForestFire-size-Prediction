# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:56:11 2019

@author: Shriyash Shende
"""
import numpy as np

import pandas as pd
import seaborn as sns

f = pd.read_csv('forestfires.csv')
f.columns
f1 = f.drop(['month','day'],axis = 1)
f1.info()
f1.columns

sns.pairplot(data=f1)

X = f1.drop(['size_category'], axis = 1)
Y = f1['size_category']


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)

from sklearn.svm import SVC
clf = SVC(kernel='linear', random_state = 0)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test, Y_pred)
per = cn[0,0] + cn[1,1]
p = per + cn[0,1] + cn[1,0]
per / p

clf1 = SVC(kernel='rbf', random_state = 0)
clf1.fit(X_train, Y_train)
Y_pred_1 = clf1.predict(X_test)
cn1 = confusion_matrix(Y_test, Y_pred_1)
per = cn1[0,0] + cn1[1,1]
p = per + cn1[0,1] + cn1[1,0]
per / p

clf3 = SVC(kernel='poly', random_state = 0)
clf3.fit(X_train, Y_train)
Y_pred_2 = clf3.predict(X_test)
cn3 = confusion_matrix(Y_test, Y_pred_2)
per = cn3[0,0] + cn3[1,1]
p = per + cn3[0,1] + cn3[1,0]
per / p


clf4 = SVC(kernel='sigmoid', random_state = 0)
clf4.fit(X_train, Y_train)
Y_pred_4 = clf4.predict(X_test)
cn4 = confusion_matrix(Y_test, Y_pred_4)
per = cn4[0,0] + cn4[1,1]
p = per + cn4[0,1] + cn4[1,0]
per / p
