import numpy as np

class KNN:
    def __init__(self, k = 1):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.X_train, axis=2)
        nearest_indices = np.argsort(distances, axis = 1)[:, :self.k]
        nearest_labels = self.y_train[nearest_indices]
        return np.array([np.bincount(labels).argmax() for labels in nearest_labels])

    def __repr__(self):
        return f"KNN(k={self.k})"