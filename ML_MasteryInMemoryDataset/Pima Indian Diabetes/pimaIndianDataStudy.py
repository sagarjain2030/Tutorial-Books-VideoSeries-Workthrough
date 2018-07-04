import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from keras import Sequential


import numpy as np
import sys
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import pickle
np.set_printoptions(threshold=sys.maxsize)


#Reading dataset
dataset = pd.read_csv('diabetes.csv',index_col=False)
print(dataset.shape)
#(768, 9)

print(dataset.head().to_string())
'''
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72             35        0  33.6                     0.627   50        1
1            1       85             66             29        0  26.6                     0.351   31        0
2            8      183             64              0        0  23.3                     0.672   32        1
3            1       89             66             23       94  28.1                     0.167   21        0
4            0      137             40             35      168  43.1                     2.288   33        1
'''

print("printing Datatypes")
print(dataset.dtypes)
#printing Datatypes
#Pregnancies                   int64
#Glucose                       int64
#BloodPressure                 int64
#SkinThickness                 int64
#Insulin                       int64
#BMI                         float64
#DiabetesPedigreeFunction    float64
#Age                           int64
#Outcome                       int64
#dtype: object

print("printing Desctiption of ")
print(dataset.describe().to_string())
'''
printing Desctiption of 
       Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin         BMI  DiabetesPedigreeFunction         Age     Outcome
count   768.000000  768.000000     768.000000     768.000000  768.000000  768.000000                768.000000  768.000000  768.000000
mean      3.845052  120.894531      69.105469      20.536458   79.799479   31.992578                  0.471876   33.240885    0.348958
std       3.369578   31.972618      19.355807      15.952218  115.244002    7.884160                  0.331329   11.760232    0.476951
min       0.000000    0.000000       0.000000       0.000000    0.000000    0.000000                  0.078000   21.000000    0.000000
25%       1.000000   99.000000      62.000000       0.000000    0.000000   27.300000                  0.243750   24.000000    0.000000
50%       3.000000  117.000000      72.000000      23.000000   30.500000   32.000000                  0.372500   29.000000    0.000000
75%       6.000000  140.250000      80.000000      32.000000  127.250000   36.600000                  0.626250   41.000000    1.000000
max      17.000000  199.000000     122.000000      99.000000  846.000000   67.100000                  2.420000   81.000000    1.000000
'''

#Correlations
print("Printing Data Correlation")
print(dataset.corr().to_string())
'''
Printing Data Correlation
                          Pregnancies   Glucose  BloodPressure  SkinThickness   Insulin       BMI  DiabetesPedigreeFunction       Age   Outcome
Pregnancies                  1.000000  0.129459       0.141282      -0.081672 -0.073535  0.017683                 -0.033523  0.544341  0.221898
Glucose                      0.129459  1.000000       0.152590       0.057328  0.331357  0.221071                  0.137337  0.263514  0.466581
BloodPressure                0.141282  0.152590       1.000000       0.207371  0.088933  0.281805                  0.041265  0.239528  0.065068
SkinThickness               -0.081672  0.057328       0.207371       1.000000  0.436783  0.392573                  0.183928 -0.113970  0.074752
Insulin                     -0.073535  0.331357       0.088933       0.436783  1.000000  0.197859                  0.185071 -0.042163  0.130548
BMI                          0.017683  0.221071       0.281805       0.392573  0.197859  1.000000                  0.140647  0.036242  0.292695
DiabetesPedigreeFunction    -0.033523  0.137337       0.041265       0.183928  0.185071  0.140647                  1.000000  0.033561  0.173844
Age                          0.544341  0.263514       0.239528      -0.113970 -0.042163  0.036242                  0.033561  1.000000  0.238356
Outcome                      0.221898  0.466581       0.065068       0.074752  0.130548  0.292695                  0.173844  0.238356  1.000000
'''

#Histogram
dataset.hist
scatter_matrix(dataset)
plt.show()

#Creating Feature DataSet and Target DataSet
X = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = dataset['Outcome']

Xtrain,Xtest,ytrain,ytest = X[:460],X[460:],y[:460],y[460:]
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