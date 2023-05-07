import numpy as np


class FullyConnectedLayer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        np.random.seed(42)
        # Initialize weights and biases with Xavier initialization
        self.w = np.random.normal(loc=0, scale=1/np.sqrt(self.input_size), size=(self.input_size, self.output_size))
        self.b = np.random.normal(loc=0, scale=1/np.sqrt(self.input_size), size=(self.output_size))
    def __call__(self, x):
        y = np.matmul(x, self.w) + self.b
        return y
    def backward(self, grad, x):
        self.grad_w = np.outer(x, grad)
        self.grad_b = grad
        return self.grad_w, self.grad_b
    def update(self, learning_rate=0.01):
        self.w -= learning_rate * self.grad_w 
        self.b -= learning_rate * self.grad_b

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    def backward(self, grad, x):
        if len(grad.shape) > 1:
            grad = np.sum(grad, axis=1)
        return grad * (np.exp(-x)) / (1 + np.exp(-x))**2

class MSE():
    def __call__(self, y_pred, y_gt):
        return (y_gt - y_pred)**2
    def backward(self, y_pred, y_gt):
        return 2 * (y_pred - y_gt)

class Layer():
    def __init__(self, fcl):
        self.fcl = fcl

class MLP():
    def __init__(self, layers_sizes):
        self.layers_sizes = layers_sizes
        self.sigmoid = Sigmoid()
        self.mse = MSE()
        self.layers = []
        for i in range(len(layers_sizes) - 1):
            fcl = FullyConnectedLayer(layers_sizes[i], layers_sizes[i+1])
            self.layers.append(Layer(fcl))
    def __call__(self, x):
        for layer in self.layers:
            layer.input = x
            layer.fcl_out = layer.fcl(x)
            layer.sigmoid_out = self.sigmoid(layer.fcl_out)
            x = layer.sigmoid_out
        return x
    def backward(self, y_gt):
        grad = self.mse.backward(self.layers[-1].sigmoid_out, y_gt)
        for layer in reversed(self.layers):
            grad_sigmoid = self.sigmoid.backward(grad, layer.fcl_out)
            grad, _ = layer.fcl.backward(grad_sigmoid, layer.input)
    def update(self, learning_rate=0.01):
        for layer in self.layers:
            layer.fcl.update(learning_rate)