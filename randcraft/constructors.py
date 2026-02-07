import numpy as np
from scipy.stats import beta, gamma, lognorm, norm, uniform
from scipy.stats._distn_infrastructure import rv_continuous

from randcraft.random_variable import RandomVariable
from randcraft.rvs import (
    DiracDeltaRV,
    DiscreteRV,
    SciRV,
)
from randcraft.utils.arrays import clean_1d_array

__all__ = ["make_discrete", "make_dirac", "make_coin_flip", "make_die_roll", "make_scipy", "make_normal", "make_uniform", "make_beta", "make_gamma", "make_log_normal"]


# Discrete
def make_discrete(values: list[float] | list[int] | np.ndarray, probabilities: list[float] | np.ndarray | None = None, seed: int | None = None) -> RandomVariable:
    # If probabilities are not provided, equal probabilities are assumed
    values = clean_1d_array(values)
    if probabilities is not None:
        probabilities = clean_1d_array(probabilities)
    return RandomVariable(rv=DiscreteRV(values=values, probabilities=probabilities, seed=seed))


def make_dirac(value: float | int) -> RandomVariable:
    value = float(value)
    return RandomVariable(rv=DiracDeltaRV(value=value))


def make_coin_flip(seed: int | None = None) -> RandomVariable:
    return make_discrete(values=[0, 1], probabilities=[0.5, 0.5], seed=seed)


def make_die_roll(sides: int = 6, seed: int | None = None) -> RandomVariable:
    values = list(range(1, sides + 1))
    probabilities = [1 / sides] * sides
    return make_discrete(values=values, probabilities=probabilities, seed=seed)


# Scipy
def make_scipy(scipy_rv: rv_continuous, *args, **kwargs) -> RandomVariable:
    return RandomVariable(rv=SciRV(scipy_rv, *args, **kwargs))


# Helpers for common scipy distributions
def make_normal(mean: float | int, std_dev: float | int, seed: int | None = None) -> RandomVariable:
    mean = float(mean)
    std_dev = float(std_dev)
    return make_scipy(scipy_rv=norm, seed=seed, loc=mean, scale=std_dev)


def make_uniform(low: float | int, high: float | int, seed: int | None = None) -> RandomVariable:
    low = float(low)
    high = float(high)
    return make_scipy(scipy_rv=uniform, seed=seed, loc=low, scale=high - low)


def make_beta(a: float | int, b: float | int, seed: int | None = None) -> RandomVariable:
    a = float(a)
    b = float(b)
    return make_scipy(scipy_rv=beta, seed=seed, a=a, b=b)


def make_gamma(a: float | int, scale: float | int, seed: int | None = None) -> RandomVariable:
    a = float(a)
    scale = float(scale)
    return make_scipy(scipy_rv=gamma, seed=seed, a=a, scale=scale)


def make_log_normal(mean: float | int, std_dev: float | int, seed: int | None = None) -> RandomVariable:
    mean = float(mean)
    std_dev = float(std_dev)
    return make_scipy(scipy_rv=lognorm, seed=seed, s=std_dev, scale=np.exp(mean))
