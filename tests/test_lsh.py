import numpy as np
import random
from nearest_neighbors.approximate import LSH

# it is difficult to perform implicit evaluation of this method due to its stochastic nature, hence this test just implements quick sanity checks
def test_lsh():
    num_points = 1000
    num_features = 5
    k = 5

    datapoints = np.random.randint(0, 150, size = (num_points, num_features))
    lsh = LSH()
    lsh.fit(datapoints, k = k)
    sample_point = datapoints[random.choice(range(num_points))]
    pred = lsh.predict(sample_point)
    print("Printing here:", pred)
    assert pred.shape == (k, num_features)
