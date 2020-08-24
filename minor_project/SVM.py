import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#import datset
data = pd.read_csv('data.csv')

#drop unnecessary coulumns
data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:, -1]

#categorical to numerical
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

#normalaization
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#applying classifier
from sklearn.svm  import SVC
model = SVC(kernel='rbf',random_state=0)
model.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=model,X=X_train,y=y_train,cv=10,scoring='accuracy')
acc.mean()
acc.std()

y_pred = model.predict(X_test)
accuracy=model.score(X_test,y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)