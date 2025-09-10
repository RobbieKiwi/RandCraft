from abc import ABC, abstractmethod
from typing import Literal, TypeVar, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from random_variable.models import Statistics

type PdfPlotType = Literal["pdf", "cdf", "both"]


class ProbabilityDistributionFunction(ABC):
    @classmethod
    @abstractmethod
    def get_short_name(cls) -> str: ...

    @property
    @abstractmethod
    def statistics(self) -> Statistics: ...

    @abstractmethod
    def sample_numpy(self, n: int) -> np.ndarray: ...

    @abstractmethod
    def chance_that_rv_is_le(self, value: float) -> float: ...

    @abstractmethod
    def value_that_is_at_le_chance(self, chance: float) -> float: ...

    @abstractmethod
    def plot_pdf_on_axis(self, ax: Axes) -> None: ...

    @abstractmethod
    def plot_cdf_on_axis(self, ax: Axes) -> None: ...

    @abstractmethod
    def scale(self, x: float) -> Self: ...

    @abstractmethod
    def add_constant(self, x: float) -> Self: ...

    @abstractmethod
    def _get_plot_range(self) -> tuple[float, float]: ...

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}(mean={self.mean}, variance={self.variance})>"

    def __repr__(self) -> str:
        return str(self)

    def plot(self, kind: PdfPlotType = "both") -> None:
        v_lines = [(self.mean, "red", "mean")]
        if self.variance > 0:
            v_lines.append((self.mean - self.std_dev, "orange", "-1 std_dev"))
            v_lines.append((self.mean + self.std_dev, "orange", "+1 std_dev"))

        def _plot(is_cumulative: bool, ax: Axes) -> None:
            if is_cumulative:
                self.plot_cdf_on_axis(ax)
                ax.set_title("CDF")
            else:
                self.plot_pdf_on_axis(ax)
                ax.set_title("PDF")
            ax.set_xlabel('x')
            ax.set_xlim(self._get_plot_range())
            ax.set_ylabel('P(X<=x)' if is_cumulative else 'P(X=x)')
            for item in v_lines:
                pos, color, label = item
                ax.axvline(pos, color=color, label=label, linestyle='--', linewidth=1)
            ax.legend()
            ax.grid(True)

        if kind == "both":
            fig, axs = plt.subplots(2, 1, sharex="all")
            _plot(is_cumulative=False, ax=axs[0])
            _plot(is_cumulative=True, ax=axs[1])
            axs[0].set_ylim(0, None)
            axs[1].set_ylim(0, 1.1)
        elif kind == "pdf":
            fig, ax1 = plt.subplots()
            _plot(is_cumulative=False, ax=ax1)
            ax1.set_ylim(bottom=0)
        elif kind == "cdf":
            fig, ax1 = plt.subplots()
            _plot(is_cumulative=True, ax=ax1)
            ax1.set_ylim(0, 1.1)
        else:
            raise ValueError(f"Invalid kind: {kind}. Choose 'pdf', 'cdf', or 'both'.")

        plt.tight_layout()
        fig.set_size_inches(10, 6)
        plt.show()

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


T_Pdf = TypeVar("T_Pdf", bound=ProbabilityDistributionFunction)
