import numpy as np
from core.tensor import Tensor
from losses import MSE

class LinearRegression:
    def __init__(self, lr = 1e-3, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        _, n_features = x.shape

        # init wandb as Tensors
        self.weights = Tensor(np.zeros((n_features, 1)), requires_grad = True)
        self.bias = Tensor(0.0, requires_grad = True)

        y = Tensor(y.reshape(-1, 1), requires_grad = False)
        x_tensor = Tensor(x, requires_grad = False)

        for _ in range(self.n_iters): 
            predictions = x_tensor @ self.weights + self.bias
            loss = MSE(y, predictions) # MSE loss

            # backwards
            self.weights.zero_grad()
            self.bias.zero_grad()
            loss.backward()

            # SGD
            self.weights.data -= self.lr * self.weights.grad
            self.bias.data -= self.lr * self.bias.grad

    def predict(self, x):
        x_tensor = Tensor(x, requires_grad = False)
        return (x_tensor @ self.weights + self.bias).data