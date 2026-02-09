import warnings
from collections.abc import Callable

import numpy as np

from randcraft.models import Sampler
from randcraft.observations import make_discrete_rv_from_observations, make_gaussian_kde
from randcraft.random_variable import RandomVariable
from randcraft.rvs.anonymous import AnonymousRV
from randcraft.rvs.continuous import ContinuousRV

__all__ = ["make_anonymous_continuous_rv_from_sampler", "make_discrete_rv_from_sampler"]


def make_rv_from_sampler(sampler: Sampler, is_discrete: bool, n_observations: int = 500, stat_sampler: Sampler | None = None, _warn: bool = True) -> RandomVariable:
    """
    Creates a random variable using a sampling function.
    Statistics, cdf, pdf etc are estimated by making a kde using random observations from the sampler.

    Args:
        sampler (Sampler): A function that takes an integer and returns a numpy array of observations.
        is_discrete (bool): Whether the sampler is discrete or continuous. This is used to determine how to estimate statistics, cdf, pdf etc.
        n_observations (int, optional): The number of observations to draw from the sampler for stat computation. Defaults to 500.
        stat_sampler (Sampler | None, optional): Optionally provide a separate sampler for statistics estimation. This can be used to provide more observations for better estimation without affecting the randomness of the original sampler. Defaults to None.
        _warn (bool, optional): Whether to warn if the sampler appears to be of the wrong type (e.g. discrete sampler provided but is_discrete=False). Defaults to True.

    Returns:
        RandomVariable: A random variable containing the sampler and estimated statistics
    """
    assert n_observations > 20, "Need at least 20 observations to create an anonymous continuous distribution from sampler"

    stat_sampler = stat_sampler or sampler
    observations = stat_sampler(n_observations)

    values, counts = np.unique(observations, return_counts=True)
    if is_discrete:
        if _warn and counts.max() == 1:
            warnings.warn("All observations are unique. Consider using make_anonymous_continuous_rv_from_sampler instead.")
        return make_discrete_rv_from_observations(observations=observations)
    else:
        if _warn and len(np.unique(observations)) / n_observations < 0.9:
            warnings.warn("It appears that the provided sampler is not continuous. Consider using make_discrete_rv_from_sampler instead.")
        ref_rv: ContinuousRV = make_gaussian_kde(observations=observations)._rv  # type: ignore
        return RandomVariable(rv=AnonymousRV(sampler=sampler, ref_rv=ref_rv))


def make_anonymous_continuous_rv_from_sampler(sampler: Sampler, n_observations: int = 500, stat_sampler: Sampler | None = None) -> RandomVariable:
    return make_rv_from_sampler(sampler=sampler, is_discrete=False, n_observations=n_observations, stat_sampler=stat_sampler)


def make_discrete_rv_from_sampler(sampler: Callable[[int], np.ndarray], n_observations: int = 500, stat_sampler: Sampler | None = None) -> RandomVariable:
    return make_rv_from_sampler(sampler=sampler, is_discrete=True, n_observations=n_observations, stat_sampler=stat_sampler)
