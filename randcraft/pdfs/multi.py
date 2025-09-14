from collections.abc import Sequence
from functools import cached_property

import numpy as np
from matplotlib.axes import Axes

from randcraft.models import Statistics, sum_uncertain_floats
from randcraft.pdfs.anonymous import AnonymousDistributionFunction
from randcraft.pdfs.continuous import ContinuousDistributionFunction
from randcraft.pdfs.discrete import DiracDeltaDistributionFunction, DiscreteDistributionFunction


class MultiDistributionFunction(ContinuousDistributionFunction):
    def __init__(
        self,
        continuous_pdfs: list[ContinuousDistributionFunction],
        discrete_pdf: DiscreteDistributionFunction | None = None,
    ) -> None:
        assert len(continuous_pdfs) > 0, "At least one continuous pdf is required"
        self._continuous_pdfs = continuous_pdfs
        self._discrete_pdf = discrete_pdf

    @property
    def short_name(self) -> str:
        return "multi"

    @cached_property
    def pdfs(self) -> Sequence[ContinuousDistributionFunction | DiscreteDistributionFunction]:
        return self.continuous_pdfs + ([self._discrete_pdf] if self._discrete_pdf is not None else [])

    @property
    def continuous_pdfs(self) -> list[ContinuousDistributionFunction]:
        return self._continuous_pdfs

    @property
    def discrete_pdf(self) -> DiscreteDistributionFunction | None:
        return self._discrete_pdf

    @property
    def has_discrete_pdf(self) -> bool:
        return self.discrete_pdf is not None

    @cached_property
    def statistics(self) -> Statistics:
        mean = sum_uncertain_floats(pdf.statistics.mean for pdf in self.pdfs)
        variance = sum_uncertain_floats(pdf.statistics.variance for pdf in self.pdfs)
        second_moment = mean**2 + variance
        # TODO calculate more moments

        min_value = sum_uncertain_floats([pdf.statistics.min_value for pdf in self.pdfs])
        max_value = sum_uncertain_floats([pdf.statistics.max_value for pdf in self.pdfs])

        return Statistics(
            moments=[mean, second_moment],
            support=(min_value, max_value),
        )

    @cached_property
    def _anonymous_pdf(self) -> AnonymousDistributionFunction:
        return AnonymousDistributionFunction(
            sampler=self.sample_numpy, n_samples=10000, external_statistics=self.statistics
        )

    def scale(self, x: float) -> "MultiDistributionFunction":
        x = float(x)
        continuous_pdfs = [pdf.scale(x) for pdf in self.continuous_pdfs]

        discrete_pdf = self.discrete_pdf
        if discrete_pdf is not None:
            discrete_pdf = discrete_pdf.scale(x)
        return MultiDistributionFunction(continuous_pdfs=continuous_pdfs, discrete_pdf=discrete_pdf)

    def add_constant(self, x: float) -> "MultiDistributionFunction":
        x = float(x)
        discrete_pdf = self.discrete_pdf or DiracDeltaDistributionFunction(value=0.0)
        discrete_pdf = discrete_pdf.add_constant(x)
        return MultiDistributionFunction(continuous_pdfs=self.continuous_pdfs, discrete_pdf=discrete_pdf)

    def sample_numpy(self, n: int) -> np.ndarray:
        return sum([pdf.sample_numpy(n) for pdf in self.pdfs])  # type: ignore

    def chance_that_rv_is_le(self, value: float) -> float:
        return self._anonymous_pdf.chance_that_rv_is_le(value=value)  # Use numerical approximation

    def value_that_is_at_le_chance(self, chance: float) -> float:
        return self._anonymous_pdf.value_that_is_at_le_chance(chance=chance)  # Use numerical approximation

    def plot_pdf_on_axis(self, ax: Axes) -> None:
        return self._anonymous_pdf.plot_pdf_on_axis(ax=ax)  # Use numerical approximation

    def plot_cdf_on_axis(self, ax: Axes) -> None:
        return self._anonymous_pdf.plot_cdf_on_axis(ax=ax)  # Use numerical approximation

    def _get_plot_range(self) -> tuple[float, float]:
        return self._anonymous_pdf._get_plot_range()

    def copy(self) -> "MultiDistributionFunction":
        return MultiDistributionFunction(continuous_pdfs=self.continuous_pdfs, discrete_pdf=self.discrete_pdf)
