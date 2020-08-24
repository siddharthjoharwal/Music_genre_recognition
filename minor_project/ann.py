#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[3]:


data = pd.read_csv('data.csv')

data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:, -1]

encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)


# In[4]:


scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

from keras import models
from keras import layers

model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1]),kernel_initializer='he_uniform'))

model.add(layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))

model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))

model.add(layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)

test_loss, test_acc = model.evaluate(X_test,y_test)

