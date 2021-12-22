import numpy as np 

inputs = [[1., 2., 3., 2.5],
        [2., 5., -1., 2.],
        [-1.5, 2.7, 3.3, -.8]]

weights = [[.2, .8, -.5, 1.],
        [.5, -.91, .26, -.5],
        [-.26, -.27, .17, .87]]

biases = [2., 3., .5]

print(np.array(inputs).shape)

print(np.array(weights).shape)

print(np.dot(np.array(inputs), np.array(weights).T) + biases)

# the biases are added to each row vector output from the matrix product
# 3 neurons with a bias value for each neuron
