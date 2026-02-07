from collections.abc import Sequence
from functools import cached_property

import numpy as np

from randcraft.models import ProbabilityDensityFunction, ProbabilityMassFunction, Statistics, Uncertainty, join_uncertainties, sum_uncertain_floats
from randcraft.rvs.base import RV, CdfEstimator
from randcraft.rvs.continuous import ContinuousRV
from randcraft.rvs.discrete import DiscreteRV


class MixtureRV(RV):
    def __init__(
        self,
        pdfs: Sequence[ContinuousRV | DiscreteRV],
        probabilities: list[float] | None = None,
        seed: int | None = None,
    ) -> None:
        # TODO Add support for mixture of mixtures by flattening the input
        if probabilities is None:
            probabilities = [1.0 / len(pdfs)] * len(pdfs)
        self._pdfs = pdfs
        self._probabilities = probabilities
        self._validate()
        super().__init__(seed=seed)

    def _validate(self) -> None:
        pdfs = self._pdfs
        probabilities = self._probabilities
        assert len(pdfs) > 0, "At least one pdf is required"
        assert all(isinstance(pdf, ContinuousRV | DiscreteRV) for pdf in pdfs), f"All pdfs must be instances of {ContinuousRV | DiscreteRV}"
        assert len(pdfs) == len(probabilities), "Number of PDFs must match number of weights"
        assert all(w > 0 for w in probabilities), "All weights must be positive"
        total_weight = sum(probabilities)
        assert abs(total_weight - 1.0) < 1e-9, "Weights must sum to 1"

    @property
    def short_name(self) -> str:
        return "mixture"

    @cached_property
    def statistics(self) -> Statistics:
        mean = sum_uncertain_floats(pdf.statistics.mean * weight for pdf, weight in zip(self.pdfs, self.probabilities))
        second_moment = sum_uncertain_floats(pdf.statistics.moments[1] * weight for pdf, weight in zip(self.pdfs, self.probabilities))
        # TODO calculate more moments

        extreme_values = [pdf.statistics.min_value for pdf in self.pdfs] + [pdf.statistics.max_value for pdf in self.pdfs]
        min_value = min(extreme_values, key=lambda x: x.value)
        max_value = max(extreme_values, key=lambda x: x.value)

        return Statistics(
            moments=[mean, second_moment],
            support=(min_value, max_value),
        )

    @property
    def pdfs(self) -> Sequence[ContinuousRV | DiscreteRV]:
        return self._pdfs

    @property
    def probabilities(self) -> list[float]:
        return self._probabilities

    def _get_discrete_points(self) -> np.ndarray:
        return np.sort(np.unique(np.concatenate([pdf._get_discrete_points() for pdf in self.pdfs])))

    def scale(self, x: float) -> "MixtureRV":
        x = float(x)
        return MixtureRV(pdfs=[pdf.scale(x) for pdf in self.pdfs], probabilities=self.probabilities)

    def add_constant(self, x: float) -> "MixtureRV":
        return MixtureRV(pdfs=[pdf.add_constant(x) for pdf in self.pdfs], probabilities=self.probabilities)

    def sample_numpy(self, n: int) -> np.ndarray:
        rng = self._rng
        pdf_choices = rng.choice(len(self.pdfs), size=n, p=self.probabilities)

        samples = np.zeros(n)
        for i in range(len(self.pdfs)):
            # Get the number of samples to draw from this PDF
            num_samples = np.sum(pdf_choices == i)
            if num_samples > 0:
                # Draw samples from the PDF and add them to the result
                samples[pdf_choices == i] = self.pdfs[i].sample_numpy(num_samples)
        return samples

    @cached_property
    def _cdf_estimator(self) -> CdfEstimator:
        return CdfEstimator(rv=self)

    def cdf(self, x: np.ndarray) -> Uncertainty[np.ndarray]:
        def func(cdfs: list[np.ndarray]) -> np.ndarray:
            cdf = np.sum([cdf * p for cdf, p in zip(cdfs, self.probabilities)], axis=0)
            cdf[x > self.statistics.max_value.value] = 1.0
            cdf[x <= self.statistics.min_value.value] = 0.0
            return cdf

        cdfs = [pdf.cdf(x) for pdf in self.pdfs]
        return join_uncertainties(uncertainties=cdfs, func=func)  # type: ignore

    def ppf(self, x: np.ndarray) -> Uncertainty[np.ndarray]:
        return self._cdf_estimator.ppf(x)

    def calculate_pdf(self, x: np.ndarray) -> ProbabilityDensityFunction | None:
        pdf_prob_pairs = [(pdf, p) for pdf, p in zip(self.pdfs, self.probabilities) if isinstance(pdf, ContinuousRV)]
        if not len(pdf_prob_pairs):
            return None
        y = np.zeros_like(x)
        for pdf, p in pdf_prob_pairs:
            cont_pdf = pdf.calculate_pdf(x)
            assert cont_pdf is not None
            y += cont_pdf.y * p
        return ProbabilityDensityFunction(x=x, y=y)

    def calculate_pmf(self) -> ProbabilityMassFunction | None:
        pdf_prob_pairs = [(pdf, p) for pdf, p in zip(self.pdfs, self.probabilities) if isinstance(pdf, DiscreteRV)]
        if not len(pdf_prob_pairs):
            return None

        value_probs: dict[float, float] = {}
        for pdf, p in pdf_prob_pairs:
            disc_pdf = pdf.calculate_pmf()
            assert disc_pdf is not None
            for v, prob in zip(disc_pdf.x, disc_pdf.y):
                if v in value_probs:
                    value_probs[v] += prob * p
                else:
                    value_probs[v] = prob * p
        values, probs = zip(*sorted(value_probs.items()))
        y = np.array(probs)
        x = np.array(values)
        return ProbabilityMassFunction(x=x, y=y)

    def copy(self) -> "MixtureRV":
        return MixtureRV(pdfs=self.pdfs, probabilities=self.probabilities)

    def _get_all_seeds(self) -> list[int | None]:
        inner_seeds = [pdf._get_all_seeds() for pdf in self.pdfs]
        return [self._seed] + [s for sublist in inner_seeds for s in sublist]
