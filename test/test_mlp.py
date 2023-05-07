import numpy as np
import tqdm
import sys, os

sys.path.append(os.path.expanduser("../castagrad/"))
from castagrad.nn import MLP

x = np.array([1, 1, 1])
y_gt = np.array([1, 1])

input_size = len(x)
output_size = len(y_gt)

mlp = MLP([input_size, 10, 10, 10, 10, output_size])

learning_rate = 0.01
n_epochs = 100000

for i in tqdm.tqdm(range(n_epochs)):
    y = mlp(x)
    loss = mlp.mse(y, y_gt)
    mlp.backward(y_gt)
    mlp.update()
print(y)