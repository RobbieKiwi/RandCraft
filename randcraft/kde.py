import numpy as np
from scipy.stats import gaussian_kde

def kde(observations: np.ndarray, bandwidth: float | None = None):


    observations = observations.flatten()
    observations = observations[:, np.newaxis]
    kde_model = gaussian_kde(dataset=observations.T, bw_method=bandwidth)
    a = 2


data = np.random.rand(5)
kde(observations=data)
