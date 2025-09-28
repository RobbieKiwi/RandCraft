from typing import TypeVar, Union, overload

import numpy as np

from randcraft.models import Statistics
from randcraft.pdf_convolver import PdfConvolver
from randcraft.rvs import (
    RV,
    DiracDeltaRV,
)
from randcraft.rvs.base import PdfPlotType

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
        if isinstance(other, RandomVariable):
            return self + other.scale(-1)
        else:
            return self + (float(other) * -1)

    def __truediv__(self, other: float) -> "RandomVariable":
        return self.scale(1 / float(other))

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

    @overload
    def cdf(self, value: float | int, exact: bool = False) -> float: ...

    @overload
    def cdf(self, value: np.ndarray, exact: bool = False) -> np.ndarray: ...

    def cdf(self, value: np.ndarray | float | int, exact: bool = False) -> float | np.ndarray:
        if isinstance(value, np.ndarray):
            return self._rv.cdf(x=value).get(name="cdf", certain=exact)
        return self.cdf(value=np.array([float(value)]))[0]

    @overload
    def ppf(self, value: float | int, exact: bool = False) -> float: ...

    @overload
    def ppf(self, value: np.ndarray, exact: bool = False) -> np.ndarray: ...

    def ppf(self, value: np.ndarray | float | int, exact: bool = False) -> float | np.ndarray:
        if isinstance(value, np.ndarray):
            return self._rv.ppf(q=value).get(name="ppf", certain=exact)
        return self.ppf(value=np.array([value]))[0]

    def sample_numpy(self, n: int) -> np.ndarray:
        return self._rv.sample_numpy(n=n)

    def sample_one(self) -> float:
        return self.sample_numpy(1).tolist()[0]

    def scale(self, factor: float) -> "RandomVariable":
        if factor == 0.0:
            return RandomVariable(DiracDeltaRV(value=0.0))
        factor = float(factor)
        new_pdf = self._rv.scale(x=factor)
        return RandomVariable(rv=new_pdf)

    def multi_sample(self, n: int) -> "RandomVariable":
        # Return an RV that is the sum of n independent samples of this RV
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if n == 1:
            return self
        new_rv = PdfConvolver.convolve_pdfs(pdfs=[self._rv] * n)  # type: ignore
        return RandomVariable(rv=new_rv)

    def plot(self, kind: PdfPlotType = "both") -> None:
        self._rv.plot(kind=kind)
