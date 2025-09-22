from functools import cached_property

import numpy as np
from scipy.stats import gaussian_kde

from randcraft.models import ProbabilityDensityFunction, Statistics, Uncertainty, certainly
from randcraft.rvs.base import CdfEstimator
from randcraft.rvs.continuous import ContinuousRV
from randcraft.rvs.discrete import DiscreteRV


class GaussianKdeRV(ContinuousRV):
    def __init__(self, discrete: DiscreteRV, kde: gaussian_kde) -> None:
        self._discrete = discrete
        self._kde = kde

    @property
    def short_name(self) -> str:
        return "gaussian_kde"

    @cached_property
    def statistics(self) -> Statistics:
        mean = self._discrete.statistics.mean.make_uncertain()
        scale_factor: float = (1.0 + self._kde.factor) ** 2  # type: ignore
        variance = self._discrete.statistics.variance * scale_factor
        second_moment = mean**2 + variance.make_uncertain()
        # TODO calculate more moments

        min_value = certainly(-1 * float("inf"))
        max_value = certainly(float("inf"))

        return Statistics(
            moments=[mean, second_moment],
            support=(min_value, max_value),
        )

    def calculate_pdf(self, x: np.ndarray) -> ProbabilityDensityFunction:
        y = self._kde.evaluate(x)
        return ProbabilityDensityFunction(x=x, y=y)

    @cached_property
    def _cdf_estimator(self) -> CdfEstimator:
        return CdfEstimator(rv=self)

    def cdf(self, x: np.ndarray) -> Uncertainty[np.ndarray]:
        return self._cdf_estimator.cdf(x)

    def ppf(self, x: np.ndarray) -> Uncertainty[np.ndarray]:
        return self._cdf_estimator.ppf(x)

    def sample_numpy(self, n: int) -> np.ndarray:
        return self._kde.resample(n)[0]

    def copy(self) -> "GaussianKdeRV":
        return GaussianKdeRV(discrete=self._discrete, kde=self._kde)
