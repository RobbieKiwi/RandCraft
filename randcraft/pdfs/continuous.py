from typing import Self

import numpy as np
from matplotlib.axes import Axes

from randcraft.models import AlgebraicFunction, Statistics
from randcraft.pdfs.base import ProbabilityDistributionFunction


class ContinuousDistributionFunction(ProbabilityDistributionFunction):
    def scale(self, x: float) -> Self | "ScaledDistributionFunction":
        # Override these if possible
        return ScaledDistributionFunction(inner=self.copy(), algebraic_function=AlgebraicFunction(scale=x))

    def add_constant(self, x: float) -> Self | "ScaledDistributionFunction":
        # Override these if possible
        return ScaledDistributionFunction(inner=self.copy(), algebraic_function=AlgebraicFunction(offset=x))


class ScaledDistributionFunction(ContinuousDistributionFunction):
    def __init__(self, inner: ContinuousDistributionFunction, algebraic_function: AlgebraicFunction) -> None:
        self._inner = inner
        self._af = algebraic_function
        assert self.algebraic_function.scale != 0.0, "Scale cannot be zero"

    @property
    def inner(self) -> ContinuousDistributionFunction:
        return self._inner

    @property
    def algebraic_function(self) -> AlgebraicFunction:
        return self._af

    @property
    def short_name(self) -> str:
        return self.inner.short_name + "*"

    @property
    def statistics(self) -> Statistics:
        return self.inner.statistics.apply_algebraic_function(self.algebraic_function)

    def scale(self, x: float) -> "ScaledDistributionFunction":
        return ScaledDistributionFunction(inner=self.inner, algebraic_function=self.algebraic_function * x)

    def add_constant(self, x: float) -> "ScaledDistributionFunction":
        return ScaledDistributionFunction(inner=self.inner, algebraic_function=self.algebraic_function + x)

    def sample_numpy(self, n: int) -> np.ndarray:
        return self.algebraic_function.apply(self.inner.sample_numpy(n))

    def chance_that_rv_is_le(self, value: float) -> float:
        scaled_v = self.algebraic_function.apply_inverse(value)
        return self.inner.chance_that_rv_is_le(scaled_v)

    def value_that_is_at_le_chance(self, chance: float) -> float:
        inner_value = self.inner.value_that_is_at_le_chance(chance)
        return self.algebraic_function.apply(inner_value)

    def plot_pdf_on_axis(self, ax: Axes, af: AlgebraicFunction | None = None) -> None:
        return self.inner.plot_pdf_on_axis(ax=ax, af=self.algebraic_function)

    def plot_cdf_on_axis(self, ax: Axes, af: AlgebraicFunction | None = None) -> None:
        return self.inner.plot_cdf_on_axis(ax=ax, af=self.algebraic_function)

    def _get_plot_range(self) -> tuple[float, float]:
        return self.algebraic_function.apply_on_range(self.inner._get_plot_range())

    def copy(self) -> "ScaledDistributionFunction":
        return ScaledDistributionFunction(inner=self.inner.copy(), algebraic_function=self.algebraic_function)
