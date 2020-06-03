# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:52:55 2020

@author: dell
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("mushroom.csv")
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
data=data.apply(le.fit_transform)
y=data["class"]
x=data.drop("class",axis=1)

corr = data.corr()
f, ax = plt.subplots(figsize=(16, 16))
sns.heatmap(corr,square=True,vmax=3,annot=True)
plt.show()
x=data.drop(["class","cap-color","bruises","odor","gill-spacing","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-below-ring","ring-number","ring-type"],axis=1)

import numpy as np

one=OneHotEncoder()
x=one.fit_transform(x).toarray()


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,stratify=y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
clf=dt.fit(x_train,y_train)
ypred_train=dt.predict(x_train)
ypred=dt.predict(x_test)
accuracy=accuracy_score(ypred,y_test)
print("accuracy score=",accuracy*100)
cm=confusion_matrix(y_test, ypred)
print(cm)
df=pd.DataFrame({'Actual':y_test,"predicted":ypred})
df1=df.head(25)
df1.plot(kind="bar",figsize=(16,10))
plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')
plt.grid(which='major',linestyle=':',linewidth='0.5',color='black')
plt.show()
