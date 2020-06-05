# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:58:24 2020

@author: dell
"""

import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("Wholesale.csv")
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data=data.drop(["Channel","Region"],axis=1).values
data=sc.fit_transform(data)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
p=PCA(n_components=2,whiten=True)
x=p.fit_transform(data)
c=[]
#kmeans
#elblow mehtod

for i in range(1,11):
    
    k=KMeans(n_clusters=i,init="k-means++",random_state=0)
    k.fit(x)
    c.append(k.inertia_)
plt.title("elbow")
plt.xlabel("numbero of clusters")
plt.plot(range(1,11),c)
plt.show()
k=KMeans(n_clusters=5,init="k-means++",random_state=0)
pred=k.fit_predict(x)
plt.scatter(x[pred==0,0],x[pred==0,1],s=80,color="yellow",label="cluster1")
plt.scatter(x[pred==1,0],x[pred==1,1],s=80,color="blue",label="cluster2")
plt.scatter(x[pred==2,0],x[pred==2,1],s=80,color="red",label="cluster3")
plt.scatter(x[pred==3,0],x[pred==3,1],s=80,color="green",label="cluster4")
plt.scatter(x[pred==4,0],x[pred==4,1],s=80,color="black",label="cluster5")
plt.scatter(k.cluster_centers_[:,0],k.cluster_centers_[:,1],s=200,label="centroids")
plt.xlabel("x1")
plt.title("kmeans")
plt.ylabel("x2")
plt.show()
for i in range(5):
    print(pred[i])

#hierachichal
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method="ward"))
plt.title("dendrogram")
plt.xlabel("number of observation")
plt.ylabel("euclidian distance")
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3)
pred=ac.fit_predict(x)


plt.scatter(x[pred==0,0],x[pred==0,1],s=80,color="yellow",label="cluster1")
plt.scatter(x[pred==1,0],x[pred==1,1],s=80,color="blue",label="cluster2")
plt.scatter(x[pred==2,0],x[pred==2,1],s=80,color="red",label="cluster3")

plt.xlabel("x1")
plt.title("hierachical")
plt.ylabel("x2")
plt.show()
for i in range(5):
    print(pred[i])
