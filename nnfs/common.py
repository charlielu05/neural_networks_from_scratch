import numpy as np 
import nnfs
from nnfs.datasets import spiral, spiral_data

nnfs.init()

# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output=probabilities

class Loss:
    # calculate the data and regularization losses
    # given model output and ground truth values 
    def calculate(self, output, y):

        # calculate sample loss
        sample_losses = self.forward(output, y)

        # calculate mean loss
        data_loss = np.mean(sample_losses)

        # return loss
        return data_loss

# cross entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # forward pass
    def forward(self, y_pred, y_true):

        # number of samples in a batch
        samples = len(y_pred)

        # clip data to avoid division by 0 
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        print(y_pred.shape)
        # probabilities for target values
        # only if categorical labels
        if len(y_true.shape) == 1:
            # using numpy function to select alternate index from y_true
            # y_true needs to contain the correct index number
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # mask values - only for one hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        # Losses
        negative_log_likelihood = -np.log(correct_confidences)
        
        return negative_log_likelihood

if __name__ == "__main__":
    softmax_outputs = np.array([[.7, .1, .2],
                            [.1, .5, .4],
                            [.02, .9, .08]])
    class_targets = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0]])

    loss_function = Loss_CategoricalCrossentropy()
    loss = loss_function.calculate(softmax_outputs, class_targets)
    print(loss) # result should be 0.38506
    # --- 
    X, y = spiral_data(samples=100, classes=3)
    dense1 = Layer_Dense(2,3)
    activation1 = Activation_ReLU()
    # create second dense layer with 3 input features since the output from last layer is now 3 output values
    dense2 = Layer_Dense(3,3)
    activation2 = Activation_Softmax()
    # create loss function 
    loss_function = Loss_CategoricalCrossentropy()

    # perform a forward pass
    dense1.forward(X)
    # perform forward pass from dense1 output to activation1 layer
    activation1.forward(dense1.output)
    # perform forward pass from activation1 output to dense2 layer
    dense2.forward(activation1.output)
    # perform forward pass through activation function 
    activation2.forward(dense2.output)

    print(activation2.output[:5])

    # calcualte the loss 
    loss = loss_function.calculate(activation2.output, y)
    # print loss value 
    print(f"loss: {loss}")

