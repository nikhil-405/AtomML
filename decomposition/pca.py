import numpy as np

class PCA:
    def __init__(self, n_comps):
        self.n_components = n_comps
        self.n_samples = None
        self.n_features = None
        self.data = None
        self.mean = None
        self.std = None
        self.cov = None
        self.eig = None

    def fit(self, x, pr = True):
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        
        
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.data = (x - self.mean) / self.std
        
        self.cov = np.cov(self.data, rowvar=False)
        
        self.eigen_values, self.eigen_vectors = np.linalg.eig(self.cov)
        
        idx = np.argsort(self.eigen_values)[::-1]
        self.eigen_values = self.eigen_values[idx]
        self.eigen_vectors = self.eigen_vectors[:, idx]
        
        if pr:
            print(f"PCA(n_components={self.n_components})")
        
    def fit_transform(self, x):
        self.fit(x, pr = False)
        return self.transform(x)
    
    def transform(self, x):
        if not hasattr(self, 'eigen_values') or self.eigen_values is None:
            raise ValueError("Call fit method before calling the transform method")
        
        x_std = (x - self.mean) / self.std
        return np.dot(x_std, self.eigen_vectors[:, :self.n_components])
    
    def get_covariance(self):
        if self.cov is not None:
            return self.cov

        raise ValueError("Call fit method first")