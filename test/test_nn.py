import numpy as np
import tqdm
import sys, os

sys.path.append(os.path.expanduser("../castagrad/"))
from castagrad.nn import FullyConnectedLayer, MSE, Sigmoid

x = np.array([1, 1, 1])
y_gt = np.array([1, 1])

input_size = len(x)
output_size = len(y_gt)
print(f"input shape: {x.shape}, output shape: {y_gt.shape}")
print(f"input: {x}")
print(f"output: {y_gt}")
print("----------------------------------------------------------------")

fcl = FullyConnectedLayer(input_size, output_size)
print(f"Weights: {fcl.w}")
print(f"Biases: {fcl.b}")
print("----------------------------------------------------------------")

learning_rate = 0.001
n_epochs = 100000

for i in tqdm.tqdm(range(1, n_epochs)):
    # Forward
    out_fcl = fcl(x)

    sigmoid = Sigmoid()
    out_sigmoid = sigmoid(out_fcl)

    mse = MSE()
    out_mse = mse(out_sigmoid, y_gt)

    # Backward
    grad_loss = mse.backward(out_sigmoid, y_gt)
    grad_activation = sigmoid.backward(grad_loss, out_fcl)
    fcl.backward(grad_activation, x)
    fcl.update(learning_rate=learning_rate)

print(f"Out sigmoid: {out_sigmoid}")
print(f"out_mse: {out_mse}")
