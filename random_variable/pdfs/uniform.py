from functools import cached_property
from typing import Self

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from random_variable.models import Statistics, Uncertainty
from random_variable.pdfs import DiracDeltaDistributionFunction
from random_variable.pdfs.base import ProbabilityDistributionFunction


class UniformDistributionFunction(ProbabilityDistributionFunction):
    def __init__(self, low: float, high: float) -> None:
        assert high >= low, "High value must be greater than or equal to low value."
        self._low = low
        self._high = high

    @classmethod
    def get_short_name(cls) -> str:
        return "uniform"

    @cached_property
    def statistics(self) -> Statistics:
        return Statistics(
            mean=Uncertainty(value=(self._low + self._high) / 2, is_certain=True),
            variance=Uncertainty(value=((self._high - self._low) ** 2) / 12, is_certain=True),
            min_value=Uncertainty(value=self._low, is_certain=True),
            max_value=Uncertainty(value=self._high, is_certain=True),
        )

    def scale(self, x: float) -> Self | DiracDeltaDistributionFunction:
        x = float(x)
        if x == 0.0:
            return DiracDeltaDistributionFunction(value=0.0)
        return UniformDistributionFunction.from_unsorted(values=(self.min_value * x, self.max_value * x))

    def add_constant(self, x: float) -> Self:
        return UniformDistributionFunction(low=self._low + float(x), high=self._high + float(x))

    def sample_numpy(self, n: int) -> np.ndarray:
        return np.random.uniform(low=self._low, high=self._high, size=n)

    def chance_that_rv_is_le(self, value: float) -> float:
        if value <= self._low:
            return 0.0
        if value >= self._high:
            return 1.0
        return (value - self._low) / (self._high - self._low)

    def value_that_is_at_le_chance(self, chance: float) -> float:
        if chance < 0.0 or chance > 1.0:
            raise ValueError("Chance must be between 0 and 1.")
        return self._low + chance * (self._high - self._low)

    def _get_plot_range(self) -> tuple[float, float]:
        low_high_range = self._high - self._low
        start = self._low - 0.2 * low_high_range
        end = self._high + 0.2 * low_high_range
        return start, end

    def plot_pdf_on_axis(self, ax: Axes) -> None:
        low_high_range = self._high - self._low
        start, end = self._get_plot_range()
        data = pd.Series(index=[start, self._low, self._high, end], data=[0, 1 / low_high_range, 0, 0])
        data.plot(ax=ax, drawstyle='steps-post')

    def plot_cdf_on_axis(self, ax: Axes) -> None:
        start, end = self._get_plot_range()
        data = pd.Series(index=[start, self._low, self._high, end], data=[0, 0, 1, 1])
        data.plot(ax=ax)

    @classmethod
    def from_unsorted(cls, values: tuple[float, float]) -> Self:
        low, high = sorted(values)
        return cls(low=low, high=high)
