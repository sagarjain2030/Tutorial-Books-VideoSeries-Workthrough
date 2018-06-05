import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import pickle
#Loading data from csv
dataset = pd.read_csv('iris.csv',index_col=False)
print(dataset.shape)

#Descriptive Ananlysis
#Printing first 5 rows
print(dataset.head().to_string())

#Datatypes of each attribute
print("printing Datatypes")
print(dataset.dtypes)

#Describe Dataset
print("printing Desctiption of ")
print(dataset.describe())

#Correlations
print("Printing Data Correlation")
print(dataset.corr())

#Histogram
dataset.hist
scatter_matrix(dataset)
allData =  plt.subplot(441)
allData.set_title('All Data Together')
#plt.show(4,4,0)

setosaData = pd.read_csv('iris_setosa.csv')
setosaData.hist
scatter_matrix(setosaData)
setosa = plt.subplot(442)
setosa.set_title('Setosa Data')
#plt.show(4,4,1)

versicolorData = pd.read_csv('iris_versicolor.csv')
versicolorData.hist
scatter_matrix(versicolorData)
versiColor = plt.subplot(443)
versiColor.set_title('VersiColor Data')
#plt.show(4,4,2)


verginicaData = pd.read_csv('iris_virginica.csv')
verginicaData.hist
scatter_matrix(verginicaData)
verginica = plt.subplot(444)
verginica.set_title('Verginica Data')
#plt.show(4,4,3)
plt.show()

#Creating Feature DataSet and Target DataSet
X = dataset[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = dataset['class']

Xtrain,Xtest,ytrain,ytest = X[:110],X[110:],y[:110],y[110:]

validation_size = 0.20
seed = 7
scoring = 'accuracy'

models = []
models.append(('Linear Regression', LogisticRegression()))
models.append(('K Neightbours', KNeighborsClassifier()))
models.append(('Decision Tree Classifiers', DecisionTreeClassifier()))
models.append(('Gaussian naive Bayes', GaussianNB()))
models.append(('Support Vector Machine', SVC()))

# evaluate each model in turn
results = []
names = []
model_Done = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, Xtrain,ytrain, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    model_Done.append(model)
    msg = "%s scoring:  mean: %f (Std Deviation: %f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


#My Results are
#Linear Regression scoring:  mean: 0.927273 (Std Deviation: 0.218182)
#K Neightbours scoring:  mean: 0.927273 (Std Deviation: 0.218182)
#Decision Tree Classifiers scoring:  mean: 0.918182 (Std Deviation: 0.216852)
#Gaussian naive Bayes scoring:  mean: 0.909091 (Std Deviation: 0.215130)
#Support Vector Machine scoring:  mean: 0.927273 (Std Deviation: 0.218182)

#Using GridSearchCV for SVM to find best parameter
#Since its last model

modelSVM = model_Done[-1]
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
paramGrid = {'C': Cs, 'gamma': gammas}
grid = GridSearchCV(estimator=model, param_grid=paramGrid)
grid.fit(X, y)
print(grid.best_score_)
print(grid.best_estimator_.C,grid.best_estimator_.gamma)

#Best Score: 0.9736842105263158
#Best Estimators are
#C = 1
#Gamma = 0.1

#Saving The Model
filename = 'FinalModel.sav'
pickle.dump(grid, open(filename, 'wb'))