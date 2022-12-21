#yjhyjhtest
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as mtr
from sklearn.model_selection import train_test_split
import seaborn as sns  
from sklearn.datasets import load_iris, load_wine
'''density=np.array([0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719]).reshape(-1,1)
sugar_rate=np.array([0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103]).reshape(-1,1)
C = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)
X = np.hstack((density,sugar_rate))
Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
data = np.array(pd.read_csv('aaa.csv').values.tolist())

#a=data[:, 0:2]
#x = data[:, 0:2].transpose()
#y = data[:, 2:3].transpose()
print(data)
print(type(data[0]),' ,',type(data[0][0]))'''
'''data = np.array(pd.read_csv('aaa.csv').values.tolist())
X = data[:, 0:2].transpose()
Y = data[:, 2:3].transpose()
xba = np.vstack((X, np.ones((1, X.shape[1]))))
dataset = np.loadtxt('aaa.csv', delimiter=",")#???
x = dataset[:,0:-1]
y = dataset[:,-1]
z=np.array([[2,3,4],[4,5,6]])
Z=(z==z.max(axis=0,keepdims=1))
df = pd.read_excel("iris_data.xlsx")
X = df.loc[:, 'sepal_length':'petal_width']
Y = df.loc[:, 'class1':'class3'].values
#m = np.array(pd.read_excel("winequality_data.xlsx").values.tolist()).astype(int)
#data = np.array(pd.read_csv('aaa.txt').values.tolist())
print(X)
L=[10,20,30,40,50,60,70,80]
l=L[0:len(L):2]
print(l)
'''
'''
column=np.array([1,1,2,4,3,2]).reshape(6,1)
C=np.array([[1,1],
            [2,4],
            [3,2],
            [1,2],
            [3,4],
            [5,6]]).reshape(6,2)
C = pd.DataFrame(C, columns=None, index=None)
sortcolumn = np.unique(np.sort(column, axis=0))
smaller_column = C.iloc[:,1] < 3
bigger_column = C.iloc[:,1] > 3
print(smaller_column)
print(C[smaller_column])
print(column[smaller_column])
'''
'''
data = load_iris()
x, y = data['data'], data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train = pd.DataFrame(x_train, featcolumnumns=data.feature_names, index=None)
print(x_train)'''
def add(x):
    return x+1
sum=0
length=3
M=map(lambda j: add(j), range(length))
print(len(M))


