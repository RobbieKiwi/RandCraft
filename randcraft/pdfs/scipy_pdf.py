from abc import ABC
from functools import cached_property

import numpy as np
from matplotlib.axes import Axes
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

from randcraft.models import AlgebraicFunction, Statistics, certainly
from randcraft.pdfs.base import ProbabilityDistributionFunction


class ScipyDistributionFunction(ProbabilityDistributionFunction, ABC):
    def __init__(self, scipy_rv: rv_continuous_frozen | rv_discrete_frozen) -> None:
        # It really should be continuous, but the typing here helps to interface with strange scipy typing
        assert isinstance(scipy_rv, rv_continuous_frozen)
        self._scipy_rv = scipy_rv

    @property
    def scipy_rv(self) -> rv_continuous_frozen:
        return self._scipy_rv

    @cached_property
    def statistics(self) -> Statistics:
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
        if not np.isinf(self.min_value):
            start = self.min_value
        else:
            start = self.mean - 4 * self.std_dev
        if not np.isinf(self.max_value):
            end = self.max_value
        else:
            end = self.mean + 4 * self.std_dev
        return start, end

    def plot_pdf_on_axis(self, ax: Axes, af: AlgebraicFunction | None = None) -> None:
        start, end = self._get_plot_range()
        x = np.linspace(start, end, 1000)
        y = self.scipy_rv.pdf(x)

        if af is not None:
            x = af.apply(x)
            y = y / abs(af.scale)
        ax.plot(x, y)

    def plot_cdf_on_axis(self, ax: Axes, af: AlgebraicFunction | None = None) -> None:
        start, end = self._get_plot_range()
        x = np.linspace(start, end, 1000)
        y = self.scipy_rv.cdf(x)

        if af is not None:
            x = af.apply(x)
            if af.scale < 0:
                y = 1 - y
        ax.plot(x, y)
