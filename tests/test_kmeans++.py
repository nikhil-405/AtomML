import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from clustering import KMeansPlusPlus

def test_kmeans_pp(random_state = 123):
    # synthetic data
    X, _ = __import__('sklearn.datasets', fromlist = ['make_blobs']).make_blobs(
        n_samples = 300, centers = 4, cluster_std = 0.60, random_state = random_state
    )

    # Custom Implementation
    km = KMeansPlusPlus(X, alpha=1)
    centers, _ = km.cluster(k = 4, max_iters = 100)

    # cluster assignment
    labels_pred = np.zeros(X.shape[0], dtype = int)
    for i, x in enumerate(X):
        dists = np.linalg.norm(x - np.vstack(centers), axis=1)
        labels_pred[i] = np.argmin(dists)

    # Sklearn Implementation
    skl = KMeans(
        n_clusters = 4,
        init = 'k-means++',
        n_init = 1,
        max_iter = 100,
        random_state = random_state
    )
    labels_sk = skl.fit_predict(X)

    # ARI based evaluation
    ari = adjusted_rand_score(labels_sk, labels_pred)
    assert ari > 0.9, f"Adjusted Rand Index too low ({ari:.3f}); clustering does not match sklearn’s"