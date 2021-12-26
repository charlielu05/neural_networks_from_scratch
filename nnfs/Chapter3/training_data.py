# %%
from nnfs.datasets import spiral_data
import numpy as np
import nnfs
import matplotlib.pyplot as plt

# %%
nnfs.init()
x, y = spiral_data(samples=100, classes=3)
plt.scatter(x[:, 0], x[:,1])
plt.show()
# %%
plt.scatter(x[:,0], x[:,1], c=y, cmap='brg')
plt.show()
# %%
