import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np



#Loading data from csv
dataset = pd.read_csv('iris.csv')
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

