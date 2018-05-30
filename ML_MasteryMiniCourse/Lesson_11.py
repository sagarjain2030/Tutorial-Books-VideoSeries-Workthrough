# Algorithm Tuning
from pandas import read_csv
import numpy
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dataframe = read_csv(url, names=names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)

#Grid Search CV
model1 = Ridge()
grid = GridSearchCV(estimator=model1, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)

##Randomize Search CV
model2 = Ridge()
rdm = RandomizedSearchCV(estimator=model2,param_distributions=param_grid,n_iter=6)
rdm.fit(X,Y)
print(rdm.best_score_)
print(rdm.best_estimator_)
