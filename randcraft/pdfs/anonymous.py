import logging
from collections.abc import Callable
from functools import cached_property

import numpy as np
from scipy.stats import gaussian_kde

from randcraft.models import Statistics, maybe
from randcraft.pdfs.continuous import ContinuousDistributionFunction
from randcraft.pdfs.discrete import DiscreteDistributionFunction

logger = logging.getLogger(__name__)


class AnonymousDistributionFunction(ContinuousDistributionFunction):
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
                f"It appears that the provided sampler is not continuous. {self.__class__.__name__} is designed to handle continuous distributions, consider using {DiscreteDistributionFunction.__name__} instead"
            )
        return np.sort(samples)

    @cached_property
    def cdf(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns two numpy arrays (x_values, cumulative_probabilities) representing the CDF.
        The chance that x < value can be found by interpolating cumulative_probabilities at value.
        """
        x_values = self._sorted_samples
        cumulative_probs = np.arange(1, len(x_values) + 1) / len(x_values)
        return x_values, cumulative_probs

    def sample_numpy(self, n: int) -> np.ndarray:
        return self._sampler(n)

    def calculate_inverse_cdf(self, x: np.ndarray) -> np.ndarray:
        x_values, cumulative_probs = self.cdf
        return np.interp(x, cumulative_probs, x_values)

    def calculate_cdf(self, x: np.ndarray) -> np.ndarray:
        x_values, cumulative_probs = self.cdf
        return np.interp(x, x_values, cumulative_probs)

    def calculate_pdf(self, x: np.ndarray) -> np.ndarray:
        kde = gaussian_kde(self._sorted_samples)
        return kde(x)

    def copy(self) -> "AnonymousDistributionFunction":
        return AnonymousDistributionFunction(
            sampler=self._sampler, n_samples=self._n_samples, external_statistics=self._external_statistics
        )
