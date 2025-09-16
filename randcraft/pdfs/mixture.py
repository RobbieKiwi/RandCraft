from collections.abc import Sequence
from functools import cached_property

import numpy as np

from randcraft.models import ContinuousPdf, DiscretePdf, Statistics, sum_uncertain_floats
from randcraft.pdfs.anonymous import AnonymousDistributionFunction
from randcraft.pdfs.base import ProbabilityDistributionFunction
from randcraft.pdfs.continuous import ContinuousDistributionFunction
from randcraft.pdfs.discrete import DiscreteDistributionFunction


class MixtureDistributionFunction(ProbabilityDistributionFunction):
    def __init__(
        self,
        pdfs: Sequence[ContinuousDistributionFunction | DiscreteDistributionFunction],
        probabilities: list[float] | None = None,
    ) -> None:
        # TODO Add support for mixture of mixtures by flattening the input
        if probabilities is None:
            probabilities = [1.0 / len(pdfs)] * len(pdfs)
        self._pdfs = pdfs
        self._probabilities = probabilities
        self._validate()

    def _validate(self) -> None:
        pdfs = self._pdfs
        probabilities = self._probabilities
        assert len(pdfs) > 0, "At least one pdf is required"
        assert all(isinstance(pdf, ContinuousDistributionFunction | DiscreteDistributionFunction) for pdf in pdfs), (
            f"All pdfs must be instances of {ContinuousDistributionFunction | DiscreteDistributionFunction}"
        )
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
        second_moment = sum_uncertain_floats(
            pdf.statistics.moments[1] * weight for pdf, weight in zip(self.pdfs, self.probabilities)
        )
        # TODO calculate more moments

        extreme_values = [pdf.statistics.min_value for pdf in self.pdfs] + [
            pdf.statistics.max_value for pdf in self.pdfs
        ]
        min_value = min(extreme_values, key=lambda x: x.value)
        max_value = max(extreme_values, key=lambda x: x.value)

        return Statistics(
            moments=[mean, second_moment],
            support=(min_value, max_value),
        )

    @property
    def pdfs(self) -> Sequence[ContinuousDistributionFunction | DiscreteDistributionFunction]:
        return self._pdfs

    @property
    def probabilities(self) -> list[float]:
        return self._probabilities

    @cached_property
    def _anonymous_pdf(self) -> AnonymousDistributionFunction:
        return AnonymousDistributionFunction(
            sampler=self.sample_numpy, n_samples=10000, external_statistics=self.statistics
        )

    def _get_discrete_points(self) -> np.ndarray:
        return np.sort(np.unique(np.concatenate([pdf._get_discrete_points() for pdf in self.pdfs])))

    def scale(self, x: float) -> "MixtureDistributionFunction":
        x = float(x)
        return MixtureDistributionFunction(pdfs=[pdf.scale(x) for pdf in self.pdfs], probabilities=self.probabilities)

    def add_constant(self, x: float) -> "MixtureDistributionFunction":
        return MixtureDistributionFunction(
            pdfs=[pdf.add_constant(x) for pdf in self.pdfs], probabilities=self.probabilities
        )

    def sample_numpy(self, n: int) -> np.ndarray:
        rng = np.random.default_rng()
        pdf_choices = rng.choice(len(self.pdfs), size=n, p=self.probabilities)

        samples = np.zeros(n)
        for i in range(len(self.pdfs)):
            # Get the number of samples to draw from this PDF
            num_samples = np.sum(pdf_choices == i)
            if num_samples > 0:
                # Draw samples from the PDF and add them to the result
                samples[pdf_choices == i] = self.pdfs[i].sample_numpy(num_samples)
        return samples

    def chance_that_rv_is_le(self, value: float) -> float:
        return sum([pdf.chance_that_rv_is_le(value=value) * p for pdf, p in zip(self.pdfs, self.probabilities)])

    def value_that_is_at_le_chance(self, chance: float) -> float:
        # Use numerical approximation
        return self._anonymous_pdf.value_that_is_at_le_chance(chance=chance)

    def calculate_cdf(self, x: np.ndarray) -> np.ndarray:
        cdf = np.sum([pdf.calculate_cdf(x) * p for pdf, p in zip(self.pdfs, self.probabilities)], axis=0)
        cdf[x > self.statistics.max_value.value] = 1.0
        cdf[x <= self.statistics.min_value.value] = 0.0
        return cdf

    def calculate_continuous_pdf(self, x: np.ndarray) -> ContinuousPdf | None:
        pdf_prob_pairs = [
            (pdf, p) for pdf, p in zip(self.pdfs, self.probabilities) if isinstance(pdf, ContinuousDistributionFunction)
        ]
        if not len(pdf_prob_pairs):
            return None
        y = np.zeros_like(x)
        for pdf, p in pdf_prob_pairs:
            cont_pdf = pdf.calculate_continuous_pdf(x)
            assert cont_pdf is not None
            y += cont_pdf.y * p
        return ContinuousPdf(x=x, y=y)

    def calculate_discrete_pdf(self) -> DiscretePdf | None:
        pdf_prob_pairs = [
            (pdf, p) for pdf, p in zip(self.pdfs, self.probabilities) if isinstance(pdf, DiscreteDistributionFunction)
        ]
        if not len(pdf_prob_pairs):
            return None

        value_probs: dict[float, float] = {}
        for pdf, p in pdf_prob_pairs:
            disc_pdf = pdf.calculate_discrete_pdf()
            assert disc_pdf is not None
            for v, prob in zip(disc_pdf.x, disc_pdf.y):
                if v in value_probs:
                    value_probs[v] += prob * p
                else:
                    value_probs[v] = prob * p
        values, probs = zip(*sorted(value_probs.items()))
        y = np.array(probs)
        x = np.array(values)
        return DiscretePdf(x=x, y=y)

    def _get_plot_range(self) -> tuple[float, float]:
        lows: list[float] = []
        highs: list[float] = []
        for pdf in self.pdfs:
            low, high = pdf._get_plot_range()
            lows.append(low)
            highs.append(high)
        start = min(lows)
        end = max(highs)
        return start, end

    def copy(self) -> "MixtureDistributionFunction":
        return MixtureDistributionFunction(pdfs=self.pdfs, probabilities=self.probabilities)
