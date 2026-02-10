import logging

import numpy as np

from randcraft.models import ProbabilityDensityFunction, Sampler, Statistics, Uncertainty
from randcraft.rvs.continuous import ContinuousRV

logger = logging.getLogger(__name__)


class AnonymousRV(ContinuousRV):
    def __init__(
        self,
        sampler: Sampler,  # Function used for sampling
        ref_rv: ContinuousRV,  # Reference used for calculating statistics, pdf, cdf etc and forked samples
        external_statistics: Statistics | None = None,  # Optionally override statistics
    ) -> None:
        super().__init__(seed=None)
        self._sampler = sampler
        self._ref_rv = ref_rv
        self._statistics = (external_statistics or ref_rv.statistics).make_uncertain()

    @property
    def short_name(self) -> str:
        return "anon"

    @property
    def statistics(self) -> Statistics:
        return self._statistics

    def sample_numpy(self, n: int, forked: bool = False) -> np.ndarray:
        if forked:
            return self._ref_rv.sample_numpy(n=n, forked=True)
        return self._sampler(n)

    def ppf(self, q: np.ndarray) -> Uncertainty[np.ndarray]:
        return self._ref_rv.ppf(q).make_uncertain()

    def cdf(self, x: np.ndarray) -> Uncertainty[np.ndarray]:
        return self._ref_rv.cdf(x).make_uncertain()

    def calculate_pdf(self, x: np.ndarray) -> ProbabilityDensityFunction:
        return self._ref_rv.calculate_pdf(x)

    def copy(self) -> "AnonymousRV":
        return AnonymousRV(sampler=self._sampler, ref_rv=self._ref_rv)
