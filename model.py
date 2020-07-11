
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
import pickle

train_dataF = pd.read_csv('Final_Train.csv')
test_dataF = pd.read_csv('Final_Test.csv')
#test_dataF.shape
X = train_dataF.drop(columns = ['MULTIPLE_OFFENSE'],axis=1)
y = train_dataF[['MULTIPLE_OFFENSE']]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30, random_state=10)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

RanModel = RandomForestClassifier()
KnnModel = KNeighborsClassifier(n_neighbors=5)
RanModel.fit(X_train,y_train)
KnnModel.fit(X_train,y_train)

pickle.dump(RanModel,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

y_predictRan = RanModel.predict(X_test)
y_predictKnn = KnnModel.predict(X_test)

print('RandomF: ',accuracy_score(y_test,y_predictRan))
print('KNN:     ',accuracy_score(y_test,y_predictKnn))