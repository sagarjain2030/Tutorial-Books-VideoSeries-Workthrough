import numpy as np
import pandas as pd

#TODO: Example for pandas Dataframe
array = np.array([[1,2,3],[4,5,6]])

rowName = ['a','b']
columnName = ['one','two','three']

mydataFrames = pd.DataFrame(array,rowName,columnName)
print(mydataFrames)