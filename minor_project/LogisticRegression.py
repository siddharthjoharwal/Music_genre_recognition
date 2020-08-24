#importing libraries
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#importing data
data = pd.read_csv('data.csv')

#dropping unnecessary data
data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:, -1]

#categorical to numerical
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

#normalization
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#applying Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#doing cross validation
from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=model,X=X_train,y=y_train,cv=10,scoring='accuracy')
acc.mean()
acc.std()

#understanding results through confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)