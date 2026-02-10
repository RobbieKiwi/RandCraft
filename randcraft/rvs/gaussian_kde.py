from functools import cached_property

import numpy as np
from scipy.stats import gaussian_kde

from randcraft.models import ProbabilityDensityFunction, Statistics, Uncertainty, certainly
from randcraft.rvs.base import CdfEstimator
from randcraft.rvs.continuous import ContinuousRV
from randcraft.rvs.discrete import DiscreteRV


class GaussianKdeRV(ContinuousRV):
    def __init__(self, discrete: DiscreteRV, kde: gaussian_kde, seed: int | None = None) -> None:
        self._discrete = discrete
        self._kde = kde
        super().__init__(seed=seed)

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

    @cached_property
    def _cdf_estimator(self) -> CdfEstimator:
        return CdfEstimator(rv=self)

    def calculate_pdf(self, x: np.ndarray) -> ProbabilityDensityFunction:
        y = self._kde.evaluate(x)
        return ProbabilityDensityFunction(x=x, y=y)

    def cdf(self, x: np.ndarray) -> Uncertainty[np.ndarray]:
        return self._cdf_estimator.cdf(x)

    def ppf(self, q: np.ndarray) -> Uncertainty[np.ndarray]:
        return self._cdf_estimator.ppf(q)

    def sample_numpy(self, n: int, forked: bool = False) -> np.ndarray:
        rng = self._fork_rng if forked else self._rng
        return self._kde.resample(size=n, seed=rng)[0]

    def copy(self) -> "GaussianKdeRV":
        return GaussianKdeRV(discrete=self._discrete, kde=self._kde)

    def _get_all_seeds(self) -> list[int | None]:
        return [self._seed] + self._discrete._get_all_seeds()
