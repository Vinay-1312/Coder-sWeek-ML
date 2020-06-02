# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:03:43 2020

@author: dell
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("Social_networking_ads.csv")

y=data["Purchased"].values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
x=data[["Age","EstimatedSalary"]].values
corr = data.corr()
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corr,square=True,vmax=3,annot=True)
plt.show()
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
clf=dt.fit(x_train,y_train)
ypred_train=dt.predict(x_train)
ypred=dt.predict(x_test)
accuracy=accuracy_score(ypred,y_test)

print("accuracy socre",accuracy*100)
df=pd.DataFrame({'Actual':y_test,"predicted":ypred})
df1=df.head(25)
df1.plot(kind="bar",figsize=(16,10))
plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')
plt.grid(which='major',linestyle=':',linewidth='0.5',color='black')
plt.show()