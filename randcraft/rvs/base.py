from abc import ABC, abstractmethod
from typing import Literal, Self, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from randcraft.models import ProbabilityDensityFunction, ProbabilityMassFunction, Statistics

type PdfPlotType = Literal["pdf", "cdf", "both"]


class RV(ABC):
    @property
    @abstractmethod
    def short_name(self) -> str: ...

    @property
    @abstractmethod
    def statistics(self) -> Statistics: ...

    @abstractmethod
    def sample_numpy(self, n: int) -> np.ndarray: ...

    @abstractmethod
    def scale(self, x: float) -> "RV": ...

    @abstractmethod
    def add_constant(self, x: float) -> "RV": ...

    @abstractmethod
    def calculate_pdf(self, x: np.ndarray) -> ProbabilityDensityFunction | None: ...

    @abstractmethod
    def calculate_pmf(self) -> ProbabilityMassFunction | None: ...

    @abstractmethod
    def cdf(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def ppf(self, q: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _get_plot_range(self) -> tuple[float, float]: ...

    @abstractmethod
    def _get_discrete_points(self) -> np.ndarray: ...

    @abstractmethod
    def copy(self) -> Self: ...

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}(mean={self.mean}, variance={self.variance})>"

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
        y = self.cdf(x)
        ax.plot(x, y)


T_RV = TypeVar("T_RV", bound=RV)
