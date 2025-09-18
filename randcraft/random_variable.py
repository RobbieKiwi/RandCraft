from typing import TypeVar, Union

import numpy as np

from randcraft.models import Statistics
from randcraft.pdf_convolver import PdfConvolver
from randcraft.rvs import (
    RV,
    AnonymousRV,
    DiracDeltaRV,
)

__all__ = ["RandomVariable"]

T = TypeVar("T")


class RandomVariable:
    # A wrapper around the different random variable implementations (RV)
    def __init__(self, rv: RV) -> None:
        self._rv = rv

    def __str__(self) -> str:
        mean = float(np.format_float_positional(x=self.statistics.mean.value, precision=3, fractional=False))
        var = float(np.format_float_positional(x=self.statistics.variance.value, precision=3, fractional=False))
        name = self._rv.short_name
        return f"<{self.__class__.__name__}({name}): {mean=}, {var=}>"

    def __repr__(self) -> str:
        return str(self)

    @property
    def statistics(self) -> Statistics:
        return self._rv.statistics

    def get_mean(self, exact: bool = False) -> float:
        return self._rv.statistics.get(name="mean", certain=exact)

    def get_variance(self, exact: bool = False) -> float:
        return self._rv.statistics.get(name="variance", certain=exact)

    def get_min_value(self, exact: bool = False) -> float:
        return self._rv.statistics.get(name="min_value", certain=exact)

    def get_max_value(self, exact: bool = False) -> float:
        return self._rv.statistics.get(name="max_value", certain=exact)

    def get_chance_that_rv_is_le(self, value: float, exact: bool = False) -> float:
        if exact and isinstance(self._rv, AnonymousRV):
            # TODO Move this certainty logic inside the RV class
            name = AnonymousRV.short_name
            raise ValueError(f"Exact CDF calculation is not supported for {name} distributions.")
        return self._rv.cdf(x=np.array([value]))[0]

    def get_value_that_is_at_le_chance(self, chance: float, exact: bool = False) -> float:
        if exact and isinstance(self._rv, AnonymousRV):
            # TODO Move this certainty logic inside the RV class
            name = AnonymousRV.short_name
            raise ValueError(f"Exact quantile calculation is not supported for {name} distributions.")
        assert 0.0 <= chance <= 1.0, "Chance must be between 0 and 1."
        return self._rv.ppf(q=np.array([chance]))[0]

    def sample_numpy(self, n: int) -> np.ndarray:
        return self._rv.sample_numpy(n=n)

    def sample_one(self) -> float:
        return self.sample_numpy(1).tolist()[0]

    def __add__(self, other: Union["RandomVariable", float]) -> "RandomVariable":
        # Assumes pdfs are not correlated
        if not isinstance(other, RandomVariable):
            rv = DiracDeltaRV(value=float(other))
        else:
            rv = other._rv
        new_rv = PdfConvolver.convolve_pdfs(pdfs=[self._rv, rv])  # type: ignore
        return RandomVariable(rv=new_rv)

    def __radd__(self, other: Union["RandomVariable", float]) -> "RandomVariable":
        return self.__add__(other)

    def __sub__(self, other: Union["RandomVariable", float]) -> "RandomVariable":
        # Assumes pdfs are not correlated
        if not isinstance(other, RandomVariable):
            other = float(other)
        return self + (other * -1)

    def __mul__(self, factor: float) -> "RandomVariable":
        if factor == 0.0:
            return RandomVariable(DiracDeltaRV(value=0.0))
        factor = float(factor)
        new_pdf = self._rv.scale(x=factor)
        return RandomVariable(rv=new_pdf)

    def __rmul__(self, factor: float) -> "RandomVariable":
        return self.__mul__(factor)

    def __truediv__(self, factor: float) -> "RandomVariable":
        return self.__mul__(1.0 / factor)
