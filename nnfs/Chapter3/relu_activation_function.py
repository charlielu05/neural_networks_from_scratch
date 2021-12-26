import numpy as np 
import nnfs 
from nnfs.datasets import spiral_data

from dense_layer_class import Layer_Dense

nnfs.init()

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]


output =  [max(0, i) for i in inputs]

print(output)

output = np.maximum(0, inputs)

print(output)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

x, y = spiral_data(100, 3)
dense1= Layer_Dense(2,3)

activation1 = Activation_ReLU()

dense1.forward(x)

# forward through activation function 
activation1.forward(dense1.output)
print(activation1.output)