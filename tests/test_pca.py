import numpy as np
import pytest
from decomposition.pca import PCA
from sklearn.decomposition import PCA as SKPCA

def test_pca_fit_transform():
    np.random.seed(42)
    X = np.random.randn(2000, 5)
    pca = PCA(n_comps = 2)
    sklearn_pca = SKPCA(n_components = 2)
    
    X_transformed = pca.fit_transform(X)
    X_transformed_sk = sklearn_pca.fit_transform(X)

    assert X_transformed.shape == X_transformed_sk.shape
    assert np.allclose(np.abs(X_transformed), np.abs(X_transformed_sk), rtol = 1e-3)

def test_pca_dimensionality_reduction():
    np.random.seed(42)
    X = np.random.randn(20, 5)
    pca = PCA(n_comps = 2)
    
    pca.fit(X, pr = False)
    X_transformed = pca.transform(X)
    
    assert X_transformed.shape == (20, 2)