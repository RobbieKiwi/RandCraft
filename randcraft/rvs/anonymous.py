import logging
from collections.abc import Callable
from functools import cached_property

import numpy as np
from scipy.stats import gaussian_kde

from randcraft.models import ProbabilityDensityFunction, Statistics, maybe
from randcraft.rvs.continuous import ContinuousRV
from randcraft.rvs.discrete import DiscreteRV

logger = logging.getLogger(__name__)


class AnonymousRV(ContinuousRV):  # TODO Make this directly build a kde pdf internally? Check plotting for this. PDF and CDF should be consistent
    def __init__(
        self,
        sampler: Callable[[int], np.ndarray],
        n_samples: int = 10000,
        external_statistics: Statistics | None = None,
    ) -> None:
        assert n_samples > 1000, f"At least 1000 samples are recommended for {self.__class__.__name__}"
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

    @cached_property
    def _sorted_samples(self) -> np.ndarray:
        samples = self._sampler(self._n_samples)
        unique_factor = len(np.unique(samples)) / len(samples)
        if unique_factor < 0.99:
            logger.warning(
                f"It appears that the provided sampler is not continuous. {self.__class__.__name__} is designed to handle continuous distributions, consider using {DiscreteRV.__name__} instead"
            )
        return np.sort(samples)

    @cached_property
    def _cdf(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns two numpy arrays (x_values, cumulative_probabilities) representing the CDF.
        The chance that x < value can be found by interpolating cumulative_probabilities at value.
        """
        x_values = self._sorted_samples
        cumulative_probs = np.arange(1, len(x_values) + 1) / len(x_values)
        return x_values, cumulative_probs

    @cached_property
    def _pdf(self) -> ProbabilityDensityFunction:
        start = self._sorted_samples[0]
        end = self._sorted_samples[-1]
        x = np.linspace(start, end, 1000)
        return self.calculate_pdf(x)

    def sample_numpy(self, n: int) -> np.ndarray:
        return self._sampler(n)

    def ppf(self, x: np.ndarray) -> np.ndarray:
        x_values, cumulative_probs = self._cdf
        return np.interp(x, cumulative_probs, x_values)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x_values, cumulative_probs = self._cdf
        return np.interp(x, x_values, cumulative_probs)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf.pdf(x)

    def calculate_pdf(self, x: np.ndarray) -> ProbabilityDensityFunction:
        kde = gaussian_kde(self._sorted_samples)
        return ProbabilityDensityFunction(x=x, y=kde(x))

    def copy(self) -> "AnonymousRV":
        return AnonymousRV(sampler=self._sampler, n_samples=self._n_samples, external_statistics=self._external_statistics)
