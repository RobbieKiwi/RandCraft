from randcraft.pdfs import (
    DiracDeltaDistributionFunction,
    MixtureDistributionFunction,
)
from randcraft.random_variable import RandomVariable


def mix_rvs(rvs: list[RandomVariable], probabilities: list[float] | None = None) -> RandomVariable:
    pdfs = [rv.pdf for rv in rvs]
    return RandomVariable(pdf=MixtureDistributionFunction(pdfs=pdfs, probabilities=probabilities))  # type: ignore


def add_special_event_to_rv(rv: RandomVariable, value: float, chance: float) -> RandomVariable:
    assert 0 <= chance <= 1.0, "Value must be between 0 and 1"
    dirac_rv = RandomVariable(DiracDeltaDistributionFunction(value=value))
    return mix_rvs(rvs=[rv, dirac_rv], probabilities=[1.0 - chance, chance])
