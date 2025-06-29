import numpy as np

class SVD: 
    def __init__(self, n_components):
        self.n_components = n_components
        self.n_features = None
        self.n_samples = None
        self.data = None
        self.mean = None
        self.U = None
        self.V = None
        self.sigma = None

    # iterative approach
    def qr_eigendecomposition(self, A, max_iterations = 100, tol = 1e-8):
        n = A.shape[0]
        Q_acc = np.eye(n)
        A_k = A.copy()
        
        for k in range(max_iterations):
            Q, R = np.linalg.qr(A_k)
            A_k_next = np.dot(R, Q)
            Q_acc = np.dot(Q_acc, Q)

            if np.max(np.abs(np.diag(A_k) - np.diag(A_k_next))) < tol:
                break
                
            A_k = A_k_next
        
        eigenvalues = np.diag(A_k)        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = Q_acc[:, idx]
        
        return eigenvectors, eigenvalues
    
    def fit(self, x):
        self.data = x # (n, m)
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        self.mean = np.mean(x, axis = 0)
        x_centered = x - self.mean

        # X_transpose * X
        if self.n_samples >= self.n_features: # n > m
            A = np.dot(x_centered.T, x_centered)
            V, sigma_squared = self.qr_eigendecomposition(A)
            
            self.V = V[:, :min(self.n_components, self.n_features)]
            
            self.sigma = np.sqrt(np.abs(sigma_squared[:min(self.n_components, self.n_features)]))
            
            self.U = np.zeros((self.n_samples, min(self.n_components, self.n_features)))
            for i in range(min(self.n_components, self.n_features)):
                if self.sigma[i] > 1e-10:
                    self.U[:, i] = np.dot(x_centered, self.V[:, i]) / self.sigma[i]
                else:
                    self.U[:, i] = np.zeros(self.n_samples)

        # X * X_transposed
        else:
            A = np.dot(x_centered, x_centered.T)
            U, sigma_squared = self.qr_eigendecomposition(A)
            
            self.U = U[:, :min(self.n_components, self.n_samples)]
            
            self.sigma = np.sqrt(np.abs(sigma_squared[:min(self.n_components, self.n_samples)]))
            
            self.V = np.zeros((self.n_features, min(self.n_components, self.n_samples)))
            for i in range(min(self.n_components, self.n_samples)):
                if self.sigma[i] > 1e-10:
                    self.V[:, i] = np.dot(x_centered.T, self.U[:, i]) / self.sigma[i]
                else:
                    self.V[:, i] = np.zeros(self.n_features)
        
        return self

    def transform(self, x):
        if self.V is None:
            raise ValueError("Call fit method first")
        x_centered = x - self.mean
        return np.dot(x_centered, self.V[:, :self.n_components])
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)