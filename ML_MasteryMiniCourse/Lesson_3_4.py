#TODO: Load CSV using Pandas from URL
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
print(data.shape)

#First few rows of Data
print("Printing Rows:")
print(data.head())

#Datatypes of each attribute
print("printing Datatypes")
print(data.dtypes)

#Describe Dataset
print("printing Desctiption of ")
print(data.describe())

#Correlations
print("Printing Data Correlation")
print(data.corr())