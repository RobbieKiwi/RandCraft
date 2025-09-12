from collections.abc import Callable

import numpy as np

from randcraft.pdfs import (
    AnonymousDistributionFunction,
    BetaDistributionFunction,
    DiracDeltaDistributionFunction,
    DiscreteDistributionFunction,
    NormalDistributionFunction,
    UniformDistributionFunction,
)
from randcraft.random_variable import RandomVariable

__all__ = [
    "make_normal",
    "make_uniform",
    "make_discrete",
    "make_dirac",
    "make_coin_flip",
    "make_dice_roll",
    "make_anon",
    "make_beta",
]


# Continuous
def make_normal(mean: float | int, std_dev: float | int) -> RandomVariable:
    mean = float(mean)
    std_dev = float(std_dev)
    return RandomVariable(pdf=NormalDistributionFunction(mean=mean, std_dev=std_dev))


def make_uniform(low: float | int, high: float | int) -> RandomVariable:
    low = float(low)
    high = float(high)
    return RandomVariable(pdf=UniformDistributionFunction(low=low, high=high))


def make_beta(a: float | int, b: float | int) -> RandomVariable:
    a = float(a)
    b = float(b)
    return RandomVariable(pdf=BetaDistributionFunction(a=a, b=b))


# Discrete
def make_discrete(values: list[float] | list[int], probabilities: list[float] | None = None) -> RandomVariable:
    # If probabilities are not provided, equal probabilities are assumed
    values = [float(v) for v in values]
    return RandomVariable(pdf=DiscreteDistributionFunction(values=values, probabilities=probabilities))


def make_dirac(value: float | int) -> RandomVariable:
    value = float(value)
    return RandomVariable(pdf=DiracDeltaDistributionFunction(value=value))


def make_coin_flip() -> RandomVariable:
    return make_discrete(values=[0, 1], probabilities=[0.5, 0.5])


def make_dice_roll(sides: int = 6) -> RandomVariable:
    values = list(range(1, sides + 1))
    probabilities = [1 / sides] * sides
    return make_discrete(values=values, probabilities=probabilities)


# Misc
def make_anon(sampler: Callable[[int], np.ndarray]) -> RandomVariable:
    return RandomVariable(pdf=AnonymousDistributionFunction(sampler=sampler))
