import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from nearest_neighbors import KNN

def test_knn():
    np.random.seed(42)
    num_points = 200
    num_features = 5

    datapoints = np.random.randint(0, 150, size = (num_points, num_features))
    sample_point = datapoints[random.choice(datapoints)]

    knn = KNN(k = 5)
    knn.fit(datapoints)
    atom_pred = knn.predict(sample_point)

    skKNN = NearestNeighbors(n_neighbors = 5)
    skKNN.fit(datapoints)
    sk_pred = skKNN.kneighbors(sample_point)[1]

    # tests
    assert sk_pred.shape == atom_pred.shape
    assert np.allclose(sk_pred, atom_pred)
