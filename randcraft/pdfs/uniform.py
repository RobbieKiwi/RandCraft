from typing import Self

from scipy.stats import uniform

from randcraft.pdfs.scipy_pdf import ScipyDistributionFunction


class UniformDistributionFunction(ScipyDistributionFunction):
    def __init__(self, low: float, high: float) -> None:
        self._low = low
        self._high = high
        super().__init__(uniform(loc=low, scale=high - low))

    @property
    def short_name(self) -> str:
        return "uniform"

    def scale(self, x: float) -> "UniformDistributionFunction":
        x = float(x)
        return UniformDistributionFunction.from_unsorted(values=(self.min_value * x, self.max_value * x))

    def add_constant(self, x: float) -> "UniformDistributionFunction":
        return UniformDistributionFunction(low=self._low + float(x), high=self._high + float(x))

    def copy(self) -> "UniformDistributionFunction":
        return UniformDistributionFunction(low=self._low, high=self._high)

    @classmethod
    def from_unsorted(cls, values: tuple[float, float]) -> Self:
        low, high = sorted(values)
        return cls(low=low, high=high)
