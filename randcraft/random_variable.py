from typing import Self, TypeVar, Union

import numpy as np

from randcraft.models import Statistics
from randcraft.pdf_convolver import PdfConvolver
from randcraft.pdfs import (
    AnonymousDistributionFunction,
    DiracDeltaDistributionFunction,
    MixtureDistributionFunction,
    ProbabilityDistributionFunction,
)

__all__ = ["RandomVariable"]

T = TypeVar("T")


class RandomVariable:
    def __init__(self, pdf: ProbabilityDistributionFunction) -> None:
        self._pdf = pdf

    def __str__(self) -> str:
        mean = float(np.format_float_positional(x=self.statistics.mean.value, precision=3, fractional=False))
        var = float(np.format_float_positional(x=self.statistics.variance.value, precision=3, fractional=False))
        name = self.pdf.get_short_name()
        return f"<{self.__class__.__name__}({name}): {mean=}, {var=}>"

    def __repr__(self) -> str:
        return str(self)

    @property
    def pdf(self) -> ProbabilityDistributionFunction:
        return self._pdf

    @property
    def statistics(self) -> Statistics:
        return self.pdf.statistics

    def get_mean(self, exact: bool = False) -> float:
        return self.pdf.statistics.get(name="mean", certain=exact)

    def get_variance(self, exact: bool = False) -> float:
        return self.pdf.statistics.get(name="variance", certain=exact)

    def get_min_value(self, exact: bool = False) -> float:
        return self.pdf.statistics.get(name="min_value", certain=exact)

    def get_max_value(self, exact: bool = False) -> float:
        return self.pdf.statistics.get(name="max_value", certain=exact)

    def get_chance_that_rv_is_le(self, value: float, exact: bool = False) -> float:
        if exact and isinstance(self.pdf, AnonymousDistributionFunction):
            # TODO Move this certainty logic inside the pdf subclasses
            name = AnonymousDistributionFunction.get_short_name()
            raise ValueError(f"Exact CDF calculation is not supported for {name} distributions.")
        return self.pdf.chance_that_rv_is_le(value=value)

    def get_value_that_is_at_le_chance(self, chance: float, exact: bool = False) -> float:
        if exact and isinstance(self.pdf, AnonymousDistributionFunction):
            # TODO Move this certainty logic inside the pdf subclasses
            name = AnonymousDistributionFunction.get_short_name()
            raise ValueError(f"Exact quantile calculation is not supported for {name} distributions.")
        assert 0.0 <= chance <= 1.0, "Chance must be between 0 and 1."
        return self.pdf.value_that_is_at_le_chance(chance=chance)

    def sample_numpy(self, n: int) -> np.ndarray:
        return self.pdf.sample_numpy(n=n)

    def sample_one(self) -> float:
        return self.sample_numpy(1).tolist()[0]

    def add_special_event(self, value: float, chance: float) -> Self:
        assert 0 <= chance <= 1.0, "Value must be between 0 and 1"
        dirac_rv = RandomVariable(DiracDeltaDistributionFunction(value=value))
        return self.mix_rvs(rvs=[self, dirac_rv], probabilities=[1.0 - chance, chance])

    def __add__(self, other: Union["RandomVariable", float]) -> Self:
        # Assumes pdfs are not correlated
        if not isinstance(other, RandomVariable):
            pdf = DiracDeltaDistributionFunction(value=float(other))
        else:
            pdf = other.pdf
        new_pdf = PdfConvolver.convolve_pdfs(pdfs=[self.pdf, pdf])
        return RandomVariable(pdf=new_pdf)

    def __sub__(self, other: Union["RandomVariable", float]) -> Self:
        # Assumes pdfs are not correlated
        assert isinstance(other, RandomVariable)
        return self + (other * -1)

    def __mul__(self, factor: float) -> Self:
        factor = float(factor)
        new_pdf = self.pdf.scale(x=factor)
        return RandomVariable(pdf=new_pdf)

    def __rmul__(self, factor: float) -> Self:
        return self.__mul__(factor)

    def __truediv__(self, factor: float) -> Self:
        return self.__mul__(1.0 / factor)

    @classmethod
    def mix_rvs(cls, rvs: list["RandomVariable"], probabilities: list[float] | None = None) -> Self:
        pdfs = [rv.pdf for rv in rvs]
        return RandomVariable(pdf=MixtureDistributionFunction(pdfs=pdfs, probabilities=probabilities))
