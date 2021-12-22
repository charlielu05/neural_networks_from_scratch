# 2 matrices and perform dot products of all combination of rows from first matrix and the columns of the second matrix
import numpy as np 

a = [1,2,3]
print(np.array([a]))

print(np.expand_dims(np.array(a), axis=0).shape)

a = [1,2,3]
b = [2,3,4]

a = np.array([a])
b = np.array([b]).T

print(np.dot(a,b))