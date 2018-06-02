

#[height,weight,shoes]
X = [[181, 80, 44],
     [177, 70, 43],
     [160, 60, 38],
     [154, 54, 37],
     [166, 65, 40],
     [190, 90, 47],
     [175, 64, 39],
     [177, 70, 40],
     [159, 55, 37],
     [171, 75, 42],
     [181, 85, 43]
    ]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male',
     'female','male','female', 'male']

_X=[[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
_Y=['male','male','male','female','female','female','male','male']

#TODO : Compare 4 algorithms and Name best performing algorithm

#Decision Tree
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X,Y)
prediction  = clf_tree.predict([[190,70,43]])
print('Prediction using Decision Tree :', prediction)
print('score is ', clf_tree.score(_X,_Y)*100)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
clf_logisticRegression = LogisticRegression()
clf_logisticRegression.fit(X,Y)
print('Prediction using Logistic Regression :', clf_logisticRegression.predict([[190,70,43]]))
print('score is ', clf_logisticRegression.score(_X,_Y)*100)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf_GaussianNB = GaussianNB()
clf_GaussianNB.fit(X,Y)
print('Prediction using Naive Bayes :', clf_GaussianNB.predict([[190,70,43]]))
print('score is ', clf_GaussianNB.score(_X,_Y)*100)

#Support Vector Machine
from sklearn.svm import SVC
clf_svm = SVC()
clf_svm.fit(X,Y)
print('Prediction using SVM :', clf_svm.predict([[190,70,43]]))
print('score is ', clf_svm.score(_X,_Y)*100)

list_Algo = ['Decision Tree', 'Logistic Regression', 'Naive Bayes', 'Support Vector Machine']
list_score = [clf_tree.score(_X,_Y)*100, clf_logisticRegression.score(_X,_Y)*100,
              clf_GaussianNB.score(_X,_Y)*100, clf_svm.score(_X,_Y)*100]
maximum = max(list_score)
index_Algo = [i for i,j in enumerate(list_score) if j == maximum]

print("Better result availalbe with alogrithm:")
for x in index_Algo:
    print(list_Algo[x])