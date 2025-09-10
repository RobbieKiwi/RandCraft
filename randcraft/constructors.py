from collections.abc import Callable

import numpy as np

from randcraft.pdfs import (
    AnonymousDistributionFunction,
    DiracDeltaDistributionFunction,
    DiscreteDistributionFunction,
    NormalDistributionFunction,
    UniformDistributionFunction,
)
from randcraft.random_variable import RandomVariable

__all__ = [
    "make_discrete",
    "make_dirac",
    "make_anon",
    "make_normal",
    "make_uniform",
]


def make_discrete(values: list[float] | list[int], probabilities: list[float] | None = None) -> RandomVariable:
    # If probabilities are not provided, equal probabilities are assumed
    values = [float(v) for v in values]
    return RandomVariable(pdf=DiscreteDistributionFunction(values=values, probabilities=probabilities))


def make_dirac(value: float | int) -> RandomVariable:
    value = float(value)
    return RandomVariable(pdf=DiracDeltaDistributionFunction(value=value))


def make_anon(sampler: Callable[[int], np.ndarray]) -> RandomVariable:
    return RandomVariable(pdf=AnonymousDistributionFunction(sampler=sampler))


def make_normal(mean: float | int, std_dev: float | int) -> RandomVariable:
    mean = float(mean)
    std_dev = float(std_dev)
    return RandomVariable(pdf=NormalDistributionFunction(mean=mean, std_dev=std_dev))


def make_uniform(low: float | int, high: float | int) -> RandomVariable:
    low = float(low)
    high = float(high)
    return RandomVariable(pdf=UniformDistributionFunction(low=low, high=high))
