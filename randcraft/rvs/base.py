from abc import ABC, abstractmethod
from functools import cached_property
from typing import Literal, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.integrate import cumulative_trapezoid

from randcraft.models import ProbabilityDensityFunction, ProbabilityMassFunction, Statistics, Uncertainty, maybe

type PdfPlotType = Literal["pdf", "cdf", "both"]


class RV(ABC):
    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed

    @property
    @abstractmethod
    def short_name(self) -> str: ...

    @property
    @abstractmethod
    def statistics(self) -> Statistics: ...

    @abstractmethod
    def sample_numpy(self, n: int, forked: bool = False) -> np.ndarray: ...

    @abstractmethod
    def scale(self, x: float) -> "RV": ...

    @abstractmethod
    def add_constant(self, x: float) -> "RV": ...

    @abstractmethod
    def calculate_pdf(self, x: np.ndarray) -> ProbabilityDensityFunction | None: ...

    @abstractmethod
    def calculate_pmf(self) -> ProbabilityMassFunction | None: ...

    @abstractmethod
    def cdf(self, x: np.ndarray) -> Uncertainty[np.ndarray]: ...

    @abstractmethod
    def ppf(self, q: np.ndarray) -> Uncertainty[np.ndarray]: ...

    @abstractmethod
    def _get_discrete_points(self) -> np.ndarray: ...

    @abstractmethod
    def copy(self) -> Self: ...

    def __str__(self) -> str:
        info = {"mean": self.mean, "variance": self.variance}
        info_text = ", ".join(f"{k}={v}" for k, v in info.items())
        if self.seeded:
            info_text += ", seeded"
        return f"<{self.__class__.__name__}({info_text})>"

    def __repr__(self) -> str:
        return str(self)

    @property
    def stats(self) -> Statistics:
        return self.statistics

    @property
    def mean(self) -> float:
        return self.stats.mean.value

    @property
    def variance(self) -> float:
        return self.stats.variance.value

    @property
    def std_dev(self) -> float:
        return self.stats.std_dev.value

    @property
    def min_value(self) -> float:
        return self.stats.min_value.value

    @property
    def max_value(self) -> float:
        return self.stats.max_value.value

    @cached_property
    def seeded(self) -> bool:
        return not any(seed is None for seed in self._get_all_seeds())

    def _get_all_seeds(self) -> list[int | None]:
        # Return a list of all random seeds
        return [self._seed]

    def plot(self, kind: PdfPlotType = "both") -> None:
        start, end = self._get_plot_range()
        discrete_points = self._get_discrete_points()
        x = np.linspace(start, end, 1000)
        if len(discrete_points) > 0:
            step = (x[1] - x[0]) / 100  # TODO Use double index instead of small step
            x = np.unique(np.concatenate([x, discrete_points - step, discrete_points]))
            x.sort()

        v_lines = [(self.mean, "red", "mean")]
        if self.variance > 0:
            v_lines.append((self.mean - self.std_dev, "orange", "-1 std_dev"))
            v_lines.append((self.mean + self.std_dev, "orange", "+1 std_dev"))

        def _plot(is_cumulative: bool, ax: Axes) -> None:
            if is_cumulative:
                self.plot_cdf_on_axis(ax=ax, x=x)
                ax.set_title("CDF")
                ax.set_ylim(0.0, 1.01)
            else:
                self.plot_pdf_on_axis(ax=ax, x=x)
                ax.set_title("PDF")
                ax.set_ylim(bottom=0)
            ax.set_xlabel("x")
            ax.set_xlim(self._get_plot_range())
            ax.set_ylabel("P(X<=x)" if is_cumulative else "P(X=x)")
            for item in v_lines:
                pos, color, label = item
                ax.axvline(pos, color=color, label=label, linestyle="--", linewidth=1)
            ax.legend()
            ax.grid(True)

        if kind == "both":
            fig, axs = plt.subplots(2, 1, sharex="all")
            _plot(is_cumulative=False, ax=axs[0])
            _plot(is_cumulative=True, ax=axs[1])
        elif kind == "pdf":
            fig, ax1 = plt.subplots()
            _plot(is_cumulative=False, ax=ax1)
        elif kind == "cdf":
            fig, ax1 = plt.subplots()
            _plot(is_cumulative=True, ax=ax1)
        else:
            raise ValueError(f"Invalid kind: {kind}. Choose 'pdf', 'cdf', or 'both'.")

        plt.tight_layout()
        fig.set_size_inches(10, 6)
        plt.show()

    def _get_plot_range(self) -> tuple[float, float]:
        if not np.isinf(self.min_value):
            start = self.min_value - 0.1 * self.std_dev
        else:
            start = self.mean - 4 * self.std_dev
        if not np.isinf(self.max_value):
            end = self.max_value + 0.1 * self.std_dev
        else:
            end = self.mean + 4 * self.std_dev
        buffer = (end - start) * 0.01
        if buffer == 0.0:
            buffer = max(1.0, abs(self.mean))
        return start - buffer, end + buffer

    def plot_pdf_on_axis(self, ax: Axes, x: np.ndarray) -> None:
        cont_pdf = self.calculate_pdf(x)
        if cont_pdf is not None:
            ax.plot(cont_pdf.x, cont_pdf.y)

        disc_pdf = self.calculate_pmf()
        if disc_pdf is not None:
            for x, p in zip(disc_pdf.x, disc_pdf.y):
                ax.vlines(x, 0, p, colors="C0", linewidth=2)
                ax.scatter(x, p, color="C0", s=50, zorder=5)
        return

    def plot_cdf_on_axis(self, ax: Axes, x: np.ndarray) -> None:
        y = self.cdf(x).value
        ax.plot(x, y)

    @cached_property
    def _rng(self) -> np.random.Generator:
        # The basis random generator. This is used for getting repeatable results with seeeded rvs
        return np.random.default_rng(self._seed)

    @cached_property
    def _fork_rng(self) -> np.random.Generator:
        # A second random generator (not seeded), used for any sampling needed for stats calculation or plotting etc
        return np.random.default_rng()


class CdfEstimator:
    def __init__(self, rv: RV) -> None:
        self.rv = rv
        self.x, self.p = self.calculate_cdf(rv)

    def cdf(self, x: np.ndarray) -> Uncertainty[np.ndarray]:
        return maybe(np.interp(x, self.x, self.p))

    def ppf(self, x: np.ndarray) -> Uncertainty[np.ndarray]:
        return maybe(np.interp(x, self.p, self.x))

    @staticmethod
    def calculate_cdf(rv: RV) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns two numpy arrays (x_values, cumulative_probabilities) representing the CDF.
        The chance that x < value can be found by interpolating cumulative_probabilities at value.
        """
        mean = rv.statistics.mean.value
        std_dev = rv.statistics.std_dev.value
        has_finite_lower_support = not np.isinf(rv.statistics.min_value.value)
        has_finite_upper_support = not np.isinf(rv.statistics.max_value.value)

        if has_finite_lower_support and has_finite_upper_support:
            lower = rv.statistics.min_value.value
            upper = rv.statistics.max_value.value
        elif has_finite_lower_support:
            lower = rv.statistics.min_value.value
            upper = lower + 6 * std_dev
        elif has_finite_upper_support:
            upper = rv.statistics.max_value.value
            lower = upper - 6 * std_dev
        else:
            lower = mean - 3 * std_dev
            upper = mean + 3 * std_dev

        x_values = np.linspace(lower, upper, 10000)
        pdf = rv.calculate_pdf(x_values)
        assert pdf is not None, "PDF must be defined to calculate CDF"
        pdf_vals = pdf.y

        cdf_vals = cumulative_trapezoid(pdf_vals, x_values, initial=0)
        if cdf_vals[-1] < 1:
            remainder = 1 - cdf_vals[-1]
            cdf_vals = cdf_vals + remainder / 2
            x_values = np.concatenate(([-np.inf], x_values, [np.inf]))
            cdf_vals = np.concatenate(([0.0], cdf_vals, [1.0]))
        else:
            cdf_vals /= cdf_vals[-1]  # Normalize
            cdf_vals[0] = 0.0
            cdf_vals[-1] = 1.0

        return x_values, cdf_vals
