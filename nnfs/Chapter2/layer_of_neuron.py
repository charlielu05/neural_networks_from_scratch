# layer is a group of neruon that get the same input but can have different weights nad biases

def neuron(inputs, weights, bias):
    assert len(inputs) == len(weights)
    return (sum([inputs[i] * weights[i] for i, _ in enumerate(inputs)]) + bias)

if __name__ == "__main__":
    inputs = [1,2,3,2.5]

    weights1 = [0.2, 0.8, -0.5, 1]
    weights2 = [0.5, -0.91, 0.26, -0.5]
    weights3 = [-0.26, -0.27, 0.17, 0.87]
    weights = [weights1, weights2, weights3]
    
    bias1 = 2
    bias2 = 3
    bias3 = 0.5
    biases = [bias1,bias2,bias3]

    print([neuron(inputs, weights1, bias1),neuron(inputs, weights2, bias2), neuron(inputs, weights3, bias3)] )
    print([neuron(inputs, weight, bias) for weight,bias in zip(weights,biases)])