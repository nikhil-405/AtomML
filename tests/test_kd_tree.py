import numpy as np
import random
from nearest_neighbors.approximate import KDTree

# it is difficult to perform implicit evaluation of this method due to its stochastic nature, hence this test just implements quick sanity checks
def test_kd_tree():
    num_points = 200
    num_features = 5
    k = 5

    datapoints = np.random.randint(0, 150, size = (num_points, num_features))
    kd_tree = KDTree()
    kd_tree.fit(datapoints)
    sample_point = datapoints[random.choice(range(num_points))]
    pred = kd_tree.predict(sample_point, k = k)

    assert pred.shape == (k, num_features)
    