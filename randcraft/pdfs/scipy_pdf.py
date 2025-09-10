from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from matplotlib.axes import Axes
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from randcraft.models import Statistics, certainly
from randcraft.pdfs.base import ProbabilityDistributionFunction


class ScipyDistributionFunction(ProbabilityDistributionFunction, ABC):
    @property
    @abstractmethod
    def scipy_rv(self) -> rv_continuous_frozen:
        pass

    @cached_property
    def statistics(self) -> Statistics:  # type: ignore
        support = self.scipy_rv.support()
        lower = float(support[0])
        upper = float(support[1])

        return Statistics(
            moments=[certainly(self.scipy_rv.moment(n)) for n in range(1, 5)],
            support=(certainly(lower), certainly(upper)),
        )

    def sample_numpy(self, n: int) -> np.ndarray:
        return self.scipy_rv.rvs(size=n)

    def chance_that_rv_is_le(self, value: float) -> float:
        return float(self.scipy_rv.cdf(x=value))

    def value_that_is_at_le_chance(self, chance: float) -> float:
        return float(self.scipy_rv.ppf(q=chance))

    def _get_plot_range(self) -> tuple[float, float]:
        start = self.mean - 4 * self.std_dev
        end = self.mean + 4 * self.std_dev
        return start, end

    def plot_pdf_on_axis(self, ax: Axes) -> None:
        start, end = self._get_plot_range()
        x = np.linspace(start, end, 1000)
        ax.plot(x, self.scipy_rv.pdf(x))

    def plot_cdf_on_axis(self, ax: Axes) -> None:
        start, end = self._get_plot_range()
        x = np.linspace(start, end, 1000)
        ax.plot(x, self.scipy_rv.cdf(x))
