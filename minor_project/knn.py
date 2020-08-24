#importing libraries
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#importing dataset
data = pd.read_csv('data.csv')

#drop unncessary data 
data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:, -1]

#convert categorical variables to numerical
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

#convert normalization
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#applying classfier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

accuracy = model.score(X_test,y_test)

#applying cross_validation
from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=model,X=X_train,y=y_train,cv=10,scoring='accuracy')
acc.mean()
acc.std()

#applying confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)