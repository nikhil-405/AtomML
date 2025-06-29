import numpy as np

# will be limiting the use of this class to unsupervised tasks, and hence do not require y
class KNN:
    def __init__(self, k = 1):
        self.k = k
        self.X_train = None
        # self.y_train = None

    def fit(self, X):
        self.X_train = X
        # self.y_train = y

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.X_train, axis = 2)
        return np.argsort(distances, axis = 1)[:, :self.k]

    def __repr__(self):
        return f"KNN(k={self.k})"