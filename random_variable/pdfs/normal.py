from functools import cached_property
from typing import Self

import numpy as np
from matplotlib.axes import Axes
from scipy.stats import norm

from random_variable.models import Statistics, Uncertainty
from random_variable.pdfs import DiracDeltaDistributionFunction
from random_variable.pdfs.base import ProbabilityDistributionFunction


class NormalDistributionFunction(ProbabilityDistributionFunction):
    def __init__(self, mean: float, std_dev: float) -> None:
        self._mean = mean
        self._std_dev = std_dev

    @classmethod
    def get_short_name(cls) -> str:
        return "normal"

    @cached_property
    def statistics(self) -> Statistics:
        return Statistics(
            mean=Uncertainty(value=self._mean, is_certain=True),
            variance=Uncertainty(value=self._std_dev**2, is_certain=True),
            min_value=Uncertainty(value=float('-inf'), is_certain=True),
            max_value=Uncertainty(value=float('inf'), is_certain=True),
        )

    def scale(self, x: float) -> Self | DiracDeltaDistributionFunction:
        x = float(x)
        if x == 0.0:
            return DiracDeltaDistributionFunction(value=0.0)
        return NormalDistributionFunction(mean=self._mean * x, std_dev=self._std_dev * abs(x))

    def add_constant(self, x: float) -> Self:
        return NormalDistributionFunction(mean=self._mean + float(x), std_dev=self._std_dev)

    def sample_numpy(self, n: int) -> np.ndarray:
        return np.random.normal(loc=self._mean, scale=self._std_dev, size=n)

    def chance_that_rv_is_le(self, value: float) -> float:
        return norm.cdf(x=value, loc=self._mean, scale=self._std_dev)

    def value_that_is_at_le_chance(self, chance: float) -> float:
        return norm.ppf(q=chance, loc=self._mean, scale=self._std_dev)

    def _get_plot_range(self) -> tuple[float, float]:
        start = self._mean - 4 * self._std_dev
        end = self._mean + 4 * self._std_dev
        return start, end

    def plot_pdf_on_axis(self, ax: Axes) -> None:
        start, end = self._get_plot_range()
        x = np.linspace(start, end, 1000)
        ax.plot(x, norm.pdf(x, loc=self._mean, scale=self._std_dev))

    def plot_cdf_on_axis(self, ax: Axes) -> None:
        start, end = self._get_plot_range()
        x = np.linspace(start, end, 1000)
        ax.plot(x, norm.cdf(x, loc=self._mean, scale=self._std_dev))
