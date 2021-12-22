import numpy as np 

inputs = [1., 2., 3., 2.5]
weights = [.2, .8, -.5, 1.]
bias = 2.

outputs = np.dot(weights, inputs) + bias
print(outputs)

# numpy implementation
weights = [[.2, .8, -.5, 1.],
            [.5, -.91, .26, -.5],
            [-.26, -.27, .17, .87]]
biases = [2., 3., .5]

layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)

# input shape 3x4 dot with 4x1 = 3x1