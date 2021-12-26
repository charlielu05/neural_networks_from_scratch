import numpy as np 
import nnfs 
from nnfs.datasets import spiral_data

nnfs.init()

n_inputs = 2
n_neurons = 4

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer = Layer_Dense(n_inputs=n_inputs, n_neurons = n_neurons)
print(layer.weights)
print(layer.biases)

x, y = spiral_data(samples=100, classes=3)
print(x.shape)
dense1 = Layer_Dense(2,3)
dense1.forward(x)
print(dense1.output[:5])