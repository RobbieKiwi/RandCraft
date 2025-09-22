from collections.abc import Callable

import numpy as np

from randcraft.observations import make_gaussian_kde
from randcraft.random_variable import RandomVariable
from randcraft.rvs.anonymous import AnonymousRV
from randcraft.rvs.continuous import ContinuousRV


def make_anonymous_continuous_rv_from_sampler(sampler: Callable[[int], np.ndarray], n_observations: int = 500) -> RandomVariable:
    """
    Creates a random variable using a continuous sampling function.
    Statistics, cdf, pdf etc are estimated by making a kde using random observations from the sampler.

    Args:
        sampler (Callable[[int], np.ndarray]): A function that takes an integer and returns a numpy array of observations.
        n_observations (int, optional): The number of observations to draw from the sampler. Defaults to 10000.

    Returns:
        RandomVariable: A random variable containing the sampler and statistics estimated using a kde.
    """
    observations = sampler(n_observations)
    assert len(np.unique(observations)) / n_observations > 0.9, "It appears that the provided sampler is not continuous"
    ref_rv: ContinuousRV = make_gaussian_kde(observations=observations)._rv  # type: ignore
    return RandomVariable(rv=AnonymousRV(sampler=sampler, ref_rv=ref_rv))
