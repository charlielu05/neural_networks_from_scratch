import matplotlib.pyplot as plt 
import nnfs
from nnfs.datasets import vertical_data
import sys
import os 
import numpy as np 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from common import * 

nnfs.init()

X, y = vertical_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:,1], c=y, s=40, cmap='brg')
plt.show()

dense1= Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

# loss function 
loss_function = Loss_CategoricalCrossentropy()

# Helper variables 
lowest_loss = 999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

# randomly create weights and biases 
for iteration in range(10000):
    dense1.weights = 0.05 * np.random.randn(2,3)
    dense1.biases = 0.05 * np.random.randn(1,3)
    dense2.weights = 0.05 * np.random.randn(3,3)
    dense2.biases = 0.05 * np.random.randn(1,3)

    # perform a forward pass 
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # calculate loss 
    loss = loss_function.calculate(activation2.output, y)

    # calculate predictions 
    y_pred = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(y_pred == y)

    # if loss is smaller save
    if loss < lowest_loss:
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss 
        print(f"Lowest loss: {lowest_loss}, Accuracy: {accuracy}")