# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:32:04 2020

@author: dell
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:42:15 2020

@author: dell
"""

import numpy as np 

import pandas as pd
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:03:43 2020

@author: dell
"""
import pickle
import pandas as pd

data=pd.read_csv("Social_networking_ads.csv")

y=data["Purchased"].values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
x=data[["Age","EstimatedSalary"]].values

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


pickle.dump(dt,open('model.pkl','wb'))

    

