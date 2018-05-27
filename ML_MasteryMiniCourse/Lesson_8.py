# Cross Validation Classification LogLoss
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

#Score using accuracy
kfold1 = KFold(n_splits=10, random_state=7)
model1 = LogisticRegression()
scoring = 'accuracy'
results1 = cross_val_score(model1, X, Y, cv=kfold1, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results1.mean(), results1.std()))

#Score using negative log loss
kfold2 = KFold(n_splits=10, random_state=7)
model2 = LogisticRegression()
scoring = 'neg_log_loss'
results2 = cross_val_score(model2, X, Y, cv=kfold2, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results2.mean(), results2.std()))