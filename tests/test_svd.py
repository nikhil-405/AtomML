import numpy as np
import pytest
from decomposition.svd import SVD

def test_svd_against_numpy():
    np.random.seed(42)
    X = np.random.randn(20, 5)
    
    svd_ours = SVD(n_components=3)
    X_transformed_ours = svd_ours.fit_transform(X)
    
    X_centered = X - np.mean(X, axis=0)
    U_np, sigma_np, Vt_np = np.linalg.svd(X_centered, full_matrices=False)
    
    X_transformed_np = np.dot(X_centered, Vt_np[:3].T)  # First 3 components
    
    assert X_transformed_ours.shape == X_transformed_np.shape
    assert np.allclose(svd_ours.sigma, sigma_np[:3], rtol=1e-2)
    assert np.allclose(np.abs(X_transformed_ours), np.abs(X_transformed_np), rtol=1e-1)

def test_svd_simple_numpy():
    X = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    
    svd_ours = SVD(n_components=2)
    svd_ours.fit(X)
    
    X_centered = X - np.mean(X, axis=0)
    U_np, sigma_np, Vt_np = np.linalg.svd(X_centered, full_matrices=False)
    
    assert np.allclose(svd_ours.sigma, sigma_np, rtol=1e-2)
    assert len(svd_ours.sigma) == 2