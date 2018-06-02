# Spot Check on different algorithms

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Using Prim Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

kfold = KFold(n_splits=10, random_state=7)

#Comparing Logitic Regression, SVM and Naive Bayes
performance = []
Algorithm = []

#Logistic Regression
model2 = LogisticRegression()
scoring = 'accuracy'
results = cross_val_score(model2, X, Y, cv=kfold, scoring=scoring)
performance.append(results.mean()*100)
Algorithm.append("Logistic Regression")
print("Logistic Regression")
print(results.mean())

#Support Vector Machine
model3 = SVC()
scoring = 'accuracy'
results = cross_val_score(model3, X, Y, cv=kfold, scoring=scoring)
performance.append(results.mean()*100)
Algorithm.append("Support Vector Machine")
print("Support Vector Machine")
print(results.mean())

#Naive Bayes
model4 = GaussianNB()
scoring = 'accuracy'
results = cross_val_score(model4, X, Y, cv=kfold, scoring=scoring)
performance.append(results.mean()*100)
Algorithm.append("Naive Bayes")
print("Naive Bayes")
print(results.mean())

#Lesson 10
#Comparing peformances of different algorithms
print("Algorithm performing better is " + Algorithm[performance.index(max(performance))])