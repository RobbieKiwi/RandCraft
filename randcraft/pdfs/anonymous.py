from collections.abc import Callable
from functools import cached_property

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from randcraft.models import AlgebraicFunction, Statistics, maybe
from randcraft.pdfs.base import ProbabilityDistributionFunction


class AnonymousDistributionFunction(ProbabilityDistributionFunction):
    def __init__(
        self,
        sampler: Callable[[int], np.ndarray],
        n_samples: int = 10000,
        external_statistics: Statistics | None = None,
    ) -> None:
        self._sampler = sampler
        self._n_samples = n_samples
        self._external_statistics = external_statistics

    @property
    def short_name(self) -> str:
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

    def plot_pdf_on_axis(self, ax: Axes, af: AlgebraicFunction | None = None) -> None:
        if af is not None:
            start, end = af.apply_on_range(self._get_plot_range())
            samples = np.sort(af.apply(self._sorted_samples))
            min_value, max_value = af.apply_on_range((self.min_value, self.max_value))
        else:
            start, end = self._get_plot_range()
            samples = self._sorted_samples
            min_value, max_value = self.min_value, self.max_value

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
        ser.loc[:min_value] = 0.0
        ser.loc[max_value:] = 0.0
        ser = ser / (ser.sum() * step_size)
        ser.plot(ax=ax, linestyle="dashed")

    def plot_cdf_on_axis(self, ax: Axes, af: AlgebraicFunction | None = None) -> None:
        if af is not None:
            start, end = af.apply_on_range(self._get_plot_range())
            x_values = np.sort(af.apply(self._sorted_samples))
            cumulative_probs = np.arange(1, len(x_values) + 1) / len(x_values)
        else:
            start, end = self._get_plot_range()
            x_values, cumulative_probs = self.cdf

        x_values = np.concatenate(([start], x_values, [end]))
        cumulative_probs = np.concatenate(([0], cumulative_probs, [1]))

        ax.step(x_values, cumulative_probs, where="post")
        ax.set_xlim(start, end)

    def copy(self) -> "AnonymousDistributionFunction":
        return AnonymousDistributionFunction(
            sampler=self._sampler, n_samples=self._n_samples, external_statistics=self._external_statistics
        )
