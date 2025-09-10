from collections.abc import Callable
from functools import cached_property

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from randcraft.models import Statistics, certainly, maybe, sort_uncertainties
from randcraft.pdfs.base import ProbabilityDistributionFunction
from randcraft.pdfs.discrete import DiracDeltaDistributionFunction


class AnonymousDistributionFunction(ProbabilityDistributionFunction):
    def __init__(
        self, sampler: Callable[[int], np.ndarray], n_samples: int = 10000, statistics: Statistics | None = None
    ) -> None:
        self._sampler = sampler
        self._n_samples = n_samples
        self._external_statistics = statistics

    @classmethod
    def get_short_name(cls) -> str:
        return "anon"

    @cached_property
    def statistics(self) -> Statistics:
        if self._external_statistics is not None:
            return self._external_statistics
        variance = maybe(float(np.var(self._sorted_samples)))
        mean = maybe(float(np.mean(self._sorted_samples)))
        second_moment = variance + mean.apply(lambda x: x**2)
        return Statistics(
            moments=[mean, second_moment],
            support=(maybe(float(self._sorted_samples[0])), maybe(float(self._sorted_samples[-1]))),
        )

    def scale(self, x: float) -> "AnonymousDistributionFunction | DiracDeltaDistributionFunction":
        x = float(x)
        if x == 0.0:
            return DiracDeltaDistributionFunction(value=0.0)

        new_min_max = (self.stats.min_value * x, self.stats.max_value * x)
        new_min, new_max = sort_uncertainties(new_min_max)

        statistics = Statistics(
            moments=[m * x ** (k + 1) for k, m in enumerate(self.stats.moments)], support=(new_min, new_max)
        )

        def sampler(n: int) -> np.ndarray:
            samples = self._sampler(n)
            return samples * x

        return AnonymousDistributionFunction(sampler=sampler, n_samples=self._n_samples, statistics=statistics)

    def add_constant(self, x: float) -> "AnonymousDistributionFunction":
        certain_x = certainly(float(x))
        first_moment = self.stats.moments[0] + certain_x
        second_moment = self.stats.moments[1] + self.stats.moments[0] * 2 + (certain_x**2)
        # TODO add higher moments

        new_support = (self.stats.support[0] + certain_x, self.stats.support[1] + certain_x)

        statistics = Statistics(moments=[first_moment, second_moment], support=new_support)

        def sampler(n: int) -> np.ndarray:
            samples = self._sampler(n)
            return samples + x

        return AnonymousDistributionFunction(sampler=sampler, n_samples=self._n_samples, statistics=statistics)

    def sample_numpy(self, n: int) -> np.ndarray:
        return self._sampler(n)

    def chance_that_rv_is_le(self, value: float) -> float:
        if value < self.min_value:
            return 0.0
        if value >= self.max_value:
            return 1.0
        # Use linear interpolation to find the cumulative probability
        x_values, cumulative_probs = self.cdf
        return float(np.interp(value, x_values, cumulative_probs))

    def value_that_is_at_le_chance(self, chance: float) -> float:
        x_values, cumulative_probs = self.cdf
        return float(np.interp(chance, cumulative_probs, x_values))

    @cached_property
    def _sorted_samples(self) -> np.ndarray:
        return np.sort(self._sampler(self._n_samples))

    @cached_property
    def cdf(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns two numpy arrays (x_values, cumulative_probabilities) representing the CDF.
        The chance that x < value can be found by interpolating cumulative_probabilities at value.
        """
        x_values = self._sorted_samples
        cumulative_probs = np.arange(1, len(x_values) + 1) / len(x_values)
        return x_values, cumulative_probs

    def _get_plot_range(self) -> tuple[float, float]:
        if np.isinf(self.min_value) or np.isinf(self.max_value):
            start = self.mean - self.std_dev * 3
            end = self.mean + self.std_dev * 3
            return start, end

        min_value = self.min_value
        max_value = self.max_value
        low_high_range = max_value - min_value
        start = min_value - 0.1 * low_high_range
        end = max_value + 0.1 * low_high_range
        return start, end

    def plot_pdf_on_axis(self, ax: Axes) -> None:
        start, end = self._get_plot_range()
        samples = self._sorted_samples
        unique_samples, counts = np.unique(samples, return_counts=True)

        if len(unique_samples) < 0.25 * len(samples):
            # Plot as discrete
            for x, c in zip(unique_samples, counts):
                p = c / len(samples)
                ax.vlines(x, 0, p, colors="C0", linewidth=2)
                ax.scatter(x, p, color="C0", s=50, zorder=5)
            return

        # Plot as continuous
        kde = gaussian_kde(samples)
        x_values = np.linspace(start, end, 1000)
        step_size = x_values[1] - x_values[0]
        ser = pd.Series(index=x_values, data=kde(x_values))
        ser.loc[: self.min_value] = 0.0
        ser.loc[self.max_value :] = 0.0
        ser = ser / (ser.sum() * step_size)
        ser.plot(ax=ax, linestyle="dashed")

    def plot_cdf_on_axis(self, ax: Axes) -> None:
        start, end = self._get_plot_range()
        x_values, cumulative_probs = self.cdf
        x_values = np.concatenate(([start], x_values, [end]))
        cumulative_probs = np.concatenate(([0], cumulative_probs, [1]))

        ax.step(x_values, cumulative_probs, where="post")
        ax.set_xlim(start, end)
