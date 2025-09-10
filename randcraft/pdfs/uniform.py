from typing import Self

from scipy.stats import uniform
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from randcraft.pdfs import DiracDeltaDistributionFunction
from randcraft.pdfs.scipy_pdf import ScipyDistributionFunction


class UniformDistributionFunction(ScipyDistributionFunction):
    def __init__(self, low: float, high: float) -> None:
        self._low = low
        self._high = high
        self._scipy_rv: rv_continuous_frozen = uniform(loc=low, scale=high - low)  # type: ignore

    @classmethod
    def get_short_name(cls) -> str:
        return "uniform"

    @property
    def scipy_rv(self) -> rv_continuous_frozen:
        return self._scipy_rv

    def scale(self, x: float) -> "UniformDistributionFunction | DiracDeltaDistributionFunction":
        x = float(x)
        if x == 0.0:
            return DiracDeltaDistributionFunction(value=0.0)
        return UniformDistributionFunction.from_unsorted(values=(self.min_value * x, self.max_value * x))

    def add_constant(self, x: float) -> "UniformDistributionFunction":
        return UniformDistributionFunction(low=self._low + float(x), high=self._high + float(x))

    @classmethod
    def from_unsorted(cls, values: tuple[float, float]) -> Self:
        low, high = sorted(values)
        return cls(low=low, high=high)
