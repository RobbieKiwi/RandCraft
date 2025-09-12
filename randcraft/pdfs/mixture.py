from functools import cached_property

import numpy as np
from matplotlib.axes import Axes

from randcraft.models import Statistics, sum_uncertain_floats
from randcraft.pdfs.anonymous import AnonymousDistributionFunction
from randcraft.pdfs.base import ProbabilityDistributionFunction


class MixtureDistributionFunction(ProbabilityDistributionFunction):
    def __init__(
        self,
        pdfs: list[ProbabilityDistributionFunction],
        probabilities: list[float] | None = None,
    ) -> None:
        if probabilities is None:
            probabilities = [1.0 / len(pdfs)] * len(pdfs)
        self._pdfs = pdfs
        self._probabilities = probabilities
        self._validate()

    def _validate(self) -> None:
        pdfs = self._pdfs
        probabilities = self._probabilities
        assert len(pdfs) > 0, "At least one pdf is required"
        assert all(isinstance(pdf, ProbabilityDistributionFunction) for pdf in pdfs)
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
    def pdfs(self) -> list[ProbabilityDistributionFunction]:
        return self._pdfs

    @property
    def probabilities(self) -> list[float]:
        return self._probabilities

    @cached_property
    def _anonymous_pdf(self) -> AnonymousDistributionFunction:
        return AnonymousDistributionFunction(
            sampler=self.sample_numpy, n_samples=10000, external_statistics=self.statistics
        )

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

    def _get_plot_range(self) -> tuple[float, float]:
        if np.isinf(self.min_value) or np.isinf(self.max_value):
            start = self.mean - self.std_dev * 3
            end = self.mean + self.std_dev * 3
            return start, end

        min_value = self.min_value
        max_value = self.max_value
        low_high_range = max_value - min_value
        start = min_value - 0.1 * low_high_range
        end = max_value + 0.1 * low_high_range
        return start, end

    def plot_pdf_on_axis(self, ax: Axes) -> None:
        # Use numerical approximation
        return self._anonymous_pdf.plot_pdf_on_axis(ax=ax)

    def plot_cdf_on_axis(self, ax: Axes) -> None:
        start, end = self._get_plot_range()
        x_values = np.arange(start=start, stop=end, step=(end - start) / 1000)
        probabilities = [self.chance_that_rv_is_le(value=float(x)) for x in x_values]
        ax.plot(x_values, probabilities)
        ax.set_xlim(start, end)

    def copy(self) -> "MixtureDistributionFunction":
        return MixtureDistributionFunction(pdfs=self.pdfs, probabilities=self.probabilities)
