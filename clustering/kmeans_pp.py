import numpy as np

def closest_center_distances(datapoints, centers):
    distances = []
    for point in datapoints:
        dists = np.linalg.norm(point - centers, axis=1)
        distances.append(np.min(dists))
    return np.array(distances) ** 2

class KMeansPlusPlus:
    def __init__(self, datapoints, alpha = 1, centers = None, random_state = None):
        self.datapoints = np.array(datapoints) ** alpha
        self.alpha = alpha
        self.centers = [] if centers is None else centers
        self.k = 0
        self.rng = np.random if random_state is None else np.random.RandomState(random_state)

    def initialize(self, k = 10):
        self.k = k
        n_samples = self.datapoints.shape[0]
        centers = []

        # first center
        first_idx = self.rng.randint(n_samples)
        centers.append(self.datapoints[first_idx])

        # Initializing Remaining centeres
        for _ in range(1, k):
            centers_arr = np.array(centers)
            sq_dists = closest_center_distances(self.datapoints, centers_arr)
            epsilon = 1e-8
            sq_dists += epsilon # avoids zero division error
            probs = sq_dists / sq_dists.sum()
            new_center_idx = self.rng.choice(n_samples, p=probs)
            centers.append(self.datapoints[new_center_idx])

        self.centers = centers

    def cluster(self, k, max_iters=5):
        if not self.centers:
            self.initialize(k)
        else:
            self.k = k

        for it in range(max_iters):
            clusters = [[] for _ in range(self.k)]

            # Assignment
            for point in self.datapoints:
                distances = [np.linalg.norm(point - centroid, ord=2) for centroid in self.centers]
                closest_centroid = np.argmin(distances)

                if closest_centroid >= self.k:
                    raise IndexError(f"Out of range error!!")
                clusters[closest_centroid].append(point)

            # Centroid update
            new_centers = []
            for cluster in clusters:
                if len(cluster) > 0:
                    new_centers.append(np.mean(cluster, axis=0))
                else:
                    new_centers.append(self.datapoints[self.rng.randint(len(self.datapoints))])

            # Convergence
            if np.allclose(new_centers, self.centers):
                print(f"Converged at iteration {it}")
                self.centers = new_centers
                break
            self.centers = new_centers

        return self.centers, clusters