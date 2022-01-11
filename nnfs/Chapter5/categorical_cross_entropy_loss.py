# example calculation of cross entropy loss,
# target = [1, 0, 0]
# prediction = [.7, .1, .2]
# loss = -(1*log(0.7) + 0*log(0.1) + 0*log(0.2)) = 0.35667

import math 
import numpy as np 

softmax_output = [.7, .1, .2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0]+ 
        math.log(softmax_output[1])*target_output[1]+
        math.log(softmax_output[2])*target_output[2])
print(loss)  
# this reduces down to just the negative log of the result target class

loss = -(math.log(softmax_output[0])*target_output[0])
print(loss)

#---
# dynamic calculation for cross entropy loss
softmax_outputs = [[.7, .1, .2],
                    [.1, .5, .4],
                    [.02, .9, .08]]
class_targets = [0,1,1]

for target_idx, distribution in zip(class_targets, softmax_outputs):
    print(target_idx)
    print(distribution[target_idx])
# using numpy to simplify 
softmax_outputs = np.array([[.7, .1, .2],
                            [.1, .5, .4],
                            [.02, .9, .08]])
class_targets = [0,1,1]

# this gets a list of the prediction confidence for the target class
print(softmax_outputs[range(len(softmax_outputs)), class_targets])

# apply negative log and calculate the average for the loss
print(f"negative log: {-np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])}")

# calculate average loss
negative_log_loss = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
average_log_loss = np.mean(negative_log_loss)
print(f"average log loss: {average_log_loss}")

# ---
softmax_outputs = np.array([[.7, .1, .2],
                            [.1, .5, .4],
                            [.02, .9, .08]])
class_targets = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0]])
# probabilities for target values - 
# only if categorical labels
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[
        range(len(softmax_outputs)),class_targets
    ]
# mask values - only for one-hot encoded labels
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(
        softmax_outputs * class_targets,
        axis = 1
    )

# Losses
neg_log = -np.log(correct_confidences)

average_loss = np.mean(neg_log)
print(average_loss)