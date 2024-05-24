import numpy as np
from scipy.spatial.distance import pdist
def find_median_distance(X):
    # find median distance based on L1 distance between inducing points
    distances = pdist(X, 'minkowski', p=1)
    median_dist = np.median(distances)
    return median_dist