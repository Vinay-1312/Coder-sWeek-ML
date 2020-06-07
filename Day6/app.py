# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:37:27 2020

@author: dell
"""

import pandas as pd
from textblob import TextBlob
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from flask import Flask,request,render_template
import pickle
app=Flask(__name__,template_folder='templates')#initialising flask
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template("index.html")#
@app.route('/predict',methods=['POST'])
def predict():
    x=[int (x) for x in request.form.values()]
    prediction=model.predict(np.array(x).reshape(1,-1))
    if prediction == 1:
          return render_template("index.html", prediction_text='he/she will buy this product')
        
    elif prediction == 0:
        return render_template("index.html", prediction_text='he/she will not buy this product')
  
if __name__=="__main__":
    app.run(debug=True)