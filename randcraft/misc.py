from randcraft.random_variable import RandomVariable
from randcraft.rvs import (
    DiracDeltaRV,
    MixtureRV,
)


def mix_rvs(rvs: list[RandomVariable], probabilities: list[float] | None = None) -> RandomVariable:
    pdfs = [rv._rv for rv in rvs]
    return RandomVariable(rv=MixtureRV(pdfs=pdfs, probabilities=probabilities))  # type: ignore


def add_special_event_to_rv(rv: RandomVariable, value: float, chance: float) -> RandomVariable:
    assert 0 <= chance <= 1.0, "Value must be between 0 and 1"
    dirac_rv = RandomVariable(DiracDeltaRV(value=value))
    return mix_rvs(rvs=[rv, dirac_rv], probabilities=[1.0 - chance, chance])
