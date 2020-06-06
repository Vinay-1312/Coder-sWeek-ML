# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:42:15 2020

@author: dell
"""

from textblob import TextBlob
import numpy as np 
import re
import pandas as pd
data=pd.read_csv("tweet.tsv",sep="\t",quoting=3)
def clean(text):
    text=re.sub(r'@[A-Za-z0-9]+','',text)#remove @
    text=re.sub(r'#','',text)#remove #
    
    text=re.sub(r'RT[\s]+','',text)#remove RT(Re Tweet)
    text=re.sub(r'https?:\/\/\S+','',text)#remove links
    text=re.sub(r'[^A-Za-z]',' ',text)
    return text
def polarity(text):
    return TextBlob(text).sentiment.polarity
def rate(score):
    if score>=0:
        return 1 #non toxic
    else:
        return 0 #toxic
data["text"]=data["text"].apply(clean)
data['text'].replace('', np.nan, inplace=True)
data.dropna(inplace=True)
data["polarity"]=data["text"].apply(polarity)
data["review"]=data["polarity"].apply(rate)
x=data["text"].values
y=data["review"].values
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer #containing only root words
corpus=[]
for i in range(0,1326):
    text=x[i]
    text=text.lower()
    text=text.split()
    wl=WordNetLemmatizer()
    all_stopwords=stopwords.words("english")
    all_stopwords.remove("not")
    all_stopwords.remove("no")
    text=[wl.lemmatize(word) for word in text if not word in set(all_stopwords)] #steeming nd removing stop words
    text=' '.join(text)
    corpus.append(text)
#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3500)
X=cv.fit_transform(corpus).toarray()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=0,stratify=y,test_size=0.35)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
ypred=rf.predict(x_test)
accuracy=accuracy_score(ypred,y_test)
print("accuracy score=",accuracy*100)
print("accuracy=",confusion_matrix(y_test,ypred))

    

