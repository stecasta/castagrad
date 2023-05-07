import numpy as np


class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        np.random.seed(42)
        # Initialize weights and biases with Xavier initialization
        self.w = np.random.normal(
            loc=0,
            scale=1 / np.sqrt(self.input_size),
            size=(self.input_size, self.output_size),
        )
        self.b = np.random.normal(
            loc=0, scale=1 / np.sqrt(self.input_size), size=(self.output_size)
        )

    def __call__(self, x):
        y = np.matmul(x, self.w) + self.b
        return y

    def backward(self, grad, x):
        self.grad_w = np.outer(x, grad)
        self.grad_b = grad

    def update(self, learning_rate=0.01):
        self.w -= learning_rate * self.grad_w
        self.b -= learning_rate * self.grad_b


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, grad, x):
        return grad * (np.exp(-x)) / (1 + np.exp(-x)) ** 2


class MSE:
    def __call__(self, y_pred, y_gt):
        return (y_gt - y_pred) ** 2

    def backward(self, y_pred, y_gt):
        return 2 * (y_pred - y_gt)
