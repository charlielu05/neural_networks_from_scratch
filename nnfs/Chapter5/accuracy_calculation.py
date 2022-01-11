import numpy as np 

# probabilities of 3 samples
softmax_outpuputs = np.array([[.7, .2, .1],
                            [.5, .1, .4],
                            [.02, .9, .08]])
# target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])

# calcualte values along the second axis (axis of index 1)
predictions = np.argmax(softmax_outpuputs, axis=1)
# if target is one hot encoded convert them
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)
# true evaluates to 1; false to 0 
accuracy = np.mean(predictions == class_targets)

print(f"accuracy: {accuracy}")