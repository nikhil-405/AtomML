import numpy as np
from core.tensor import Tensor
from losses import ce_loss, bce_loss

class LogisticRegression:
    def __init__(self, lr = 1e-3, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.num_classes = None

    def fit(self, x, y):
        _, n_features = x.shape
        C = len(np.unique(y))
        self.num_classes = C

        # init wandb as Tensors
        self.weights = Tensor(np.ones((n_features, C)), requires_grad = True)
        self.bias = Tensor(np.zeros((1, C)), requires_grad = True)

        y = Tensor(y.reshape(-1, 1), requires_grad = False)
        x_tensor = Tensor(x, requires_grad = False)

        for _ in range(self.n_iters): 
            logits = x_tensor @ self.weights + self.bias
            # if binary classification, use sigmoid
            # if multi-class classification, use softmax
            if (C == 2):
                predictions = logits.sigmoid()
                loss = bce_loss(y, predictions)
            else:
                predictions = logits.softmax(dim = 1)
                loss = ce_loss(y, predictions)


            # backwards
            self.weights.zero_grad()
            self.bias.zero_grad()
            loss.backward()

            # SGD
            self.weights.data -= self.lr * self.weights.grad
            self.bias.data -= self.lr * self.bias.grad

    def predict(self, x):
        x_tensor = Tensor(x, requires_grad = False)
        logits = x_tensor @ self.weights + self.bias

        if self.num_classes == 2:
            probs = logits.sigmoid()
        else:
            probs = logits.softmax(dim = 1).data
        # print(probs.shape)
        return np.argmax(probs, axis = 1)
    
    def predict_proba(self, x):
        x_tensor = Tensor(x, requires_grad = False)
        logits = x_tensor @ self.weights + self.bias
        if self.num_classes == 2:
            logits = logits.sigmoid()
        else:
            probs = logits.softmax(dim = 1).data
        # print(probs.shape)
        return probs
