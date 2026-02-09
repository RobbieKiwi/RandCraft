from collections.abc import Callable

import numpy as np

from randcraft.random_variable import RandomVariable
from randcraft.rvs import DiracDeltaRV, MixtureRV
from randcraft.sampler import make_rv_from_sampler

__all__ = ["mix_rvs", "add_special_event_to_rv", "apply_func_to_continuous_rv", "apply_func_to_discrete_rv"]


def mix_rvs(rvs: list[RandomVariable], probabilities: list[float] | None = None, seed: int | None = None) -> RandomVariable:
    pdfs = [rv._rv for rv in rvs]
    return RandomVariable(rv=MixtureRV(pdfs=pdfs, probabilities=probabilities, seed=seed))  # type: ignore


def add_special_event_to_rv(rv: RandomVariable, value: float, chance: float, seed: int | None = None) -> RandomVariable:
    assert 0 <= chance <= 1.0, "Value must be between 0 and 1"
    dirac_rv = RandomVariable(DiracDeltaRV(value=value))
    return mix_rvs(rvs=[rv, dirac_rv], probabilities=[1.0 - chance, chance], seed=seed)


def apply_func_to_continuous_rv(rv: RandomVariable, func: Callable[[np.ndarray], np.ndarray], n_observations: int = 500) -> RandomVariable:
    def sampler(n: int) -> np.ndarray:
        return func(rv.sample_numpy(n=n))

    def stat_sampler(n: int) -> np.ndarray:
        return func(rv._sample_forked(n=n))

    return make_rv_from_sampler(sampler=sampler, is_discrete=False, n_observations=n_observations, stat_sampler=stat_sampler)


def apply_func_to_discrete_rv(rv: RandomVariable, func: Callable[[np.ndarray], np.ndarray], n_observations: int = 500) -> RandomVariable:
    # Note that this will convert the discrete RV to a continuous one
    def sampler(n: int) -> np.ndarray:
        return func(rv.sample_numpy(n=n))

    def stat_sampler(n: int) -> np.ndarray:
        return func(rv._sample_forked(n=n))

    return make_rv_from_sampler(sampler=sampler, is_discrete=False, n_observations=n_observations, stat_sampler=stat_sampler, _warn=False)
