import logging
from collections.abc import Callable
from typing import overload

import numpy as np
from scipy.stats import gaussian_kde

from randcraft.constructors import make_discrete, make_normal
from randcraft.pdfs.continuous import ContinuousDistributionFunction
from randcraft.pdfs.discrete import DiscreteDistributionFunction
from randcraft.random_variable import RandomVariable
from randcraft.utils import weighted_std

logger = logging.getLogger(__name__)


@overload
def make_kde(
    observations: np.ndarray,
    kernel: None = None,
    bw_method: str | float | Callable | None = None,
    weights: np.ndarray | None = None,
) -> RandomVariable: ...


@overload
def make_kde(
    observations: np.ndarray,
    kernel: RandomVariable,
    bw_method: None = None,
    weights: np.ndarray | None = None,
) -> RandomVariable: ...


def make_kde(
    observations: np.ndarray | RandomVariable,
    kernel: RandomVariable | None = None,
    bw_method: str | float | Callable | None = None,
    weights: np.ndarray | None = None,
) -> RandomVariable:
    """
    Using a set of observations and a kernel, create a kernel density estimate (KDE).
    Either
    -supply your own kernel
    -use scipy's gaussian_kde to create a kernel

    Args:
        observations (np.ndarray): Either a 1D numpy array of observations or a RandomVariable containing a discrete distribution of observations.
        kernel (RandomVariable | None, optional): The kernel to use for the KDE. Defaults to None.
        bw_method (str | float | Callable | None, optional): The bandwidth selector for the scipy gaussian_kde method. Defaults to None.
        weights (np.ndarray | None, optional): The weights to use for the KDE. Defaults to None.

    Returns:
        RandomVariable: A random variable with a continuous distribution representing the KDE.
    """
    if kernel is not None:
        assert isinstance(kernel, RandomVariable), "Kernel must be a RandomVariable"
        assert isinstance(kernel.pdf, ContinuousDistributionFunction), "Kernel must have a continuous distribution"
        assert bw_method is None, "Bw method must be none if kernel is provided"
        if kernel.get_mean() != 0.0:
            logger.warning("Kernel mean is not zero. This may lead to unexpected results.")

    if not isinstance(observations, (np.ndarray, RandomVariable)):
        raise TypeError("Observations must be either a 1D numpy array or a RandomVariable")

    if isinstance(observations, RandomVariable):
        discrete_rv = observations
        assert isinstance(discrete_rv.pdf, DiscreteDistributionFunction), (
            "RandomVariable for observations must have a discrete distribution"
        )
        np_observations = np.array(discrete_rv.pdf.values)
    else:
        discrete_rv = make_discrete_rv_from_observations(observations=observations, weights=weights, reduce=True)
        np_observations = observations

    if kernel is None:
        kernel = make_kernel_using_scipy(observations=np_observations, bw_method=bw_method, weights=weights)

    return kernel + discrete_rv


def make_kernel_using_scipy(
    observations: np.ndarray, bw_method: str | float | Callable | None = None, weights: np.ndarray | None = None
) -> RandomVariable:
    kde_model = gaussian_kde(dataset=observations, bw_method=bw_method, weights=weights)
    factor: float = kde_model.factor  # type: ignore
    if weights is None:
        std_dev = np.std(observations, ddof=1)  # type: ignore
    else:
        std_dev = weighted_std(x=observations, weights=weights, unbiased=True)
    bandwidth: float = factor * std_dev  # type: ignore
    return make_normal(mean=0, std_dev=bandwidth)


def reduce_observations(
    observations: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Reduce observations by grouping identical values and summing their weights."""
    assert len(observations) > 1, "Need multiple observations for KDE"
    assert observations.ndim == 1, "Input data must be a 1D numpy array"
    if weights is not None:
        unique, inverse = np.unique(observations, return_inverse=True)
        summed_weights = np.zeros_like(unique, dtype=float)
        for idx, w in zip(inverse, weights):
            summed_weights[idx] += w
        return unique, summed_weights
    else:
        unique = np.unique(observations)
        return unique, None


def make_discrete_rv_from_observations(
    observations: np.ndarray, weights: np.ndarray | None = None, reduce: bool = True
) -> RandomVariable:
    if reduce:
        observations, weights = reduce_observations(observations=observations, weights=weights)
    return make_discrete(values=observations, probabilities=weights)
