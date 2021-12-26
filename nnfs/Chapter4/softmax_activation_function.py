


layer_outputs = [4.8, 1.21, 2.385]

# eulers constant
E = 2.71828182846

# for each value calculate its exponential value
exp_value = [E**i for i in layer_outputs]

softmax_output = [j / sum(exp_value) for j in exp_value]

import numpy as np 

exp_value = np.exp(layer_outputs)

norm_value = exp_value / sum(exp_value)

class Activation_Softmax:
    def forward(self, inputs):
        # subtracting the largest number means the maximum output value will be 1 since exp**0 is equal to 1.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

softmax = Activation_Softmax()
softmax.forward([[1,2,3]])
print(softmax.output)
# if we subtracted 3 from the list 
softmax.forward([[-2, -1, 0]])
print(softmax.output)