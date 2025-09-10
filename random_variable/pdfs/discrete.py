from functools import cached_property
from typing import Self, Optional

import numpy as np
from matplotlib.axes import Axes

from random_variable.models import Statistics, Uncertainty
from random_variable.pdfs.base import ProbabilityDistributionFunction


class DiscreteDistributionFunction(ProbabilityDistributionFunction):
    def __init__(self, values: list[float], probabilities: Optional[list[float]] = None) -> None:
        if probabilities is None:
            probabilities = [1.0 / len(values)] * len(values)
        assert len(values) == len(probabilities), "Values and probabilities must have the same length."
        assert all(p >= 0 for p in probabilities), "Probabilities must be non-negative."
        assert abs(sum(probabilities) - 1.0) < 1e-6, "Probabilities must sum to 1."
        assert len(set(values)) == len(values), "Values must be unique."

        self._values = np.array(values)
        self._probabilities = np.array(probabilities)

    @classmethod
    def get_short_name(cls) -> str:
        return "discrete"

    @cached_property
    def statistics(self) -> Statistics:
        mean = np.sum(self._values * self._probabilities)
        variance = np.sum((self._values - mean) ** 2 * self._probabilities)
        return Statistics(
            mean=Uncertainty(value=mean, is_certain=True),
            variance=Uncertainty(value=variance, is_certain=True),
            min_value=Uncertainty(value=np.min(self._values), is_certain=True),
            max_value=Uncertainty(value=np.max(self._values), is_certain=True),
        )

    @cached_property
    def values(self) -> list[float]:
        return self._values.tolist()

    @cached_property
    def probabilities(self) -> list[float]:
        return self._probabilities.tolist()

    def scale(self, x: float) -> Self | "DiracDeltaDistributionFunction":
        x = float(x)
        if x == 0.0:
            return DiracDeltaDistributionFunction(value=0.0)
        return DiscreteDistributionFunction(
            values=[float(v * x) for v in self._values], probabilities=self._probabilities.tolist()
        )

    def add_constant(self, x: float) -> Self:
        x = float(x)
        return DiscreteDistributionFunction(
            values=[float(v + x) for v in self._values], probabilities=self._probabilities.tolist()
        )

    def sample_numpy(self, n: int) -> np.ndarray:
        return np.random.choice(self._values, size=n, p=self._probabilities)

    def chance_that_rv_is_le(self, value: float) -> float:
        return float(np.sum(self._probabilities[self._values <= value]))

    def value_that_is_at_le_chance(self, chance: float) -> float:
        cumulative_prob = np.cumsum(self._probabilities)
        return float(self._values[np.searchsorted(cumulative_prob, chance)])

    def _get_plot_range(self) -> tuple[float, float]:
        min_value = self.statistics.min_value.value
        max_value = self.statistics.max_value.value
        low_high_range = max_value - min_value
        buffer = low_high_range * 0.1
        if buffer == 0.0:
            buffer = max(1.0, abs(min_value))
        return min_value - buffer, max_value + buffer

    def plot_pdf_on_axis(self, ax: Axes) -> None:
        for x, p in zip(self._values, self._probabilities):
            ax.vlines(x, 0, p, colors='C0', linewidth=2)
            ax.scatter(x, p, color='C0', s=50, zorder=5)
        return

    def plot_cdf_on_axis(self, ax: Axes) -> None:
        cumulative_prob = np.cumsum(self._probabilities)
        min_plot, max_plot = self._get_plot_range()
        cumulative_prob = np.concatenate(([0.0], cumulative_prob, [1.0]))
        values = np.concatenate(([min_plot], self._values, [max_plot]))
        ax.step(values, cumulative_prob, where='post')


class DiracDeltaDistributionFunction(DiscreteDistributionFunction):
    def __init__(self, value: float) -> None:
        super().__init__(values=[value])

    @classmethod
    def get_short_name(cls) -> str:
        return "dirac"

    @cached_property
    def statistics(self) -> Statistics:
        return Statistics(
            mean=Uncertainty(value=self.value, is_certain=True),
            variance=Uncertainty(value=0.0, is_certain=True),
            min_value=Uncertainty(value=self.value, is_certain=True),
            max_value=Uncertainty(value=self.value, is_certain=True),
        )

    @cached_property
    def value(self) -> float:
        return float(self._values[0])

    def scale(self, x: float) -> Self:
        return DiracDeltaDistributionFunction(value=self.mean * x)

    def sample_numpy(self, n: int) -> np.ndarray:
        return np.ones(n) * self.value

    def chance_that_rv_is_le(self, value: float) -> float:
        return 1.0 if value >= self.value else 0.0

    def value_that_is_at_le_chance(self, chance: float) -> float:
        return self.value
