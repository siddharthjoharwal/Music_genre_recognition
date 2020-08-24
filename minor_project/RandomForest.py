#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:45:09 2020

@author: siddharth
"""

#importing libraries
import pandas as pd
import numpy as np

# Preprocessing
data = pd.read_csv('Data.csv')

#seperating dependent and independent variables
data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:, -1]

#converting categorical variable into numerical
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

#normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

#splitting data into training and testing 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Fitting Random Forest Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = classifier.score(X_test,y_test)

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator = classifier,X=X_train,y=y_train,cv=10,scoring='accuracy')
acc.mean()
acc.std()

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)