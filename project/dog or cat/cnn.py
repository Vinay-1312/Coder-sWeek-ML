# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:19:01 2020

@author: dell
"""


import pandas as pd
import tensorflow as tf 
print(tf.__version__)
"""from keras.preprocessing.image import ImageDataGenerator
#preprocecing image
train_datagen = ImageDataGenerator(
        rescale=1./255,#pixel value between 0 and 1
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_set = train_datagen.flow_from_directory(
        'dataset\training_set',
        target_size=(64, 64), #size of images when fed into cnn
        batch_size=32, #images in each class
        class_mode='binary')
test_datagen=ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory("dataset/test_set",target_size=(64,64),batch_size=32,class_mode="binary")
"""