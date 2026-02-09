import logging
from collections.abc import Callable

import numpy as np
from scipy.stats import gaussian_kde

from randcraft.constructors import make_discrete
from randcraft.random_variable import RandomVariable
from randcraft.rvs.discrete import DiscreteRV
from randcraft.rvs.gaussian_kde import GaussianKdeRV

logger = logging.getLogger(__name__)


def make_gaussian_kde(
    observations: np.ndarray | RandomVariable,
    bw_method: str | float | Callable | None = None,
    weights: np.ndarray | None = None,
) -> RandomVariable:
    """
    Using a set of observations create a gaussian kernel density estimate (KDE) using scipy.stats.gaussian_kde

    Args:
        observations (np.ndarray): Either a 1D numpy array of observations or a RandomVariable containing a discrete distribution of observations.
        bw_method (str | float | Callable | None, optional): The bandwidth selector for the scipy gaussian_kde method. Defaults to None.
        weights (np.ndarray | None, optional): The weights to use for the KDE. Defaults to None.

    Returns:
        RandomVariable: A random variable with a continuous distribution representing the KDE.
    """
    if not isinstance(observations, (np.ndarray, RandomVariable)):
        raise TypeError("Observations must be either a 1D numpy array or a RandomVariable")

    if isinstance(observations, RandomVariable):
        discrete_rv = observations
        assert isinstance(discrete_rv._rv, DiscreteRV), "RandomVariable for observations must have a discrete distribution"
        np_observations = np.array(discrete_rv._rv.values)
    else:
        discrete_rv = make_discrete_rv_from_observations(observations=observations, weights=weights, reduce=True)
        np_observations = observations

    discerete_inner_rv = discrete_rv._rv
    assert isinstance(discerete_inner_rv, DiscreteRV)
    kde_model = gaussian_kde(dataset=np_observations, bw_method=bw_method, weights=weights)
    return RandomVariable(rv=GaussianKdeRV(discrete=discerete_inner_rv, kde=kde_model))


def reduce_observations(observations: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Reduce observations by grouping identical values and summing their weights."""
    assert len(observations) > 1, "Need multiple observations for KDE"
    assert observations.ndim == 1, "Input data must be a 1D numpy array"
    if weights is not None:
        unique, inverse = np.unique(observations, return_inverse=True)
        summed_weights = np.bincount(inverse, weights=weights)
        return unique, summed_weights
    else:
        unique = np.unique(observations)
        return unique, None


def make_discrete_rv_from_observations(observations: np.ndarray, weights: np.ndarray | None = None, reduce: bool = True) -> RandomVariable:
    if reduce:
        observations, weights = reduce_observations(observations=observations, weights=weights)
    return make_discrete(values=observations, probabilities=weights)
