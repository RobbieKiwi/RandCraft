from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from randcraft.models import AlgebraicFunction, ContinuousPdf, Statistics
from randcraft.pdfs.base import ProbabilityDistributionFunction


class ContinuousDistributionFunction(ProbabilityDistributionFunction, ABC):
    @abstractmethod
    def calculate_continuous_pdf(self, x: np.ndarray) -> ContinuousPdf: ...

    @abstractmethod
    def calculate_inverse_cdf(self, x: np.ndarray) -> np.ndarray: ...

    def calculate_discrete_pdf(self) -> None:
        return None

    def _get_discrete_points(self) -> np.ndarray:
        points: list[float] = []
        if not self.statistics.has_infinite_lower_support:
            points.append(self.statistics.min_value.value)
        if not self.statistics.has_infinite_upper_support:
            points.append(self.statistics.max_value.value)
        return np.array(points)

    def chance_that_rv_is_le(self, value: float) -> float:
        if value < self.min_value:
            return 0.0
        if value >= self.max_value:
            return 1.0
        return float(self.calculate_cdf(np.array([value]))[0])

    def value_that_is_at_le_chance(self, chance: float) -> float:
        if chance <= 0.0:
            return self.min_value
        if chance >= 1.0:
            return self.max_value
        return float(self.calculate_inverse_cdf(np.array([chance]))[0])

    def scale(self, x: float) -> Self | "ScaledDistributionFunction":
        # Override these if possible
        return ScaledDistributionFunction(inner=self.copy(), algebraic_function=AlgebraicFunction(scale=x))

    def add_constant(self, x: float) -> Self | "ScaledDistributionFunction":
        # Override these if possible
        return ScaledDistributionFunction(inner=self.copy(), algebraic_function=AlgebraicFunction(offset=x))

    def _get_plot_range(self) -> tuple[float, float]:
        if not np.isinf(self.min_value):
            start = self.min_value - 0.1 * self.std_dev
        else:
            start = self.mean - 4 * self.std_dev
        if not np.isinf(self.max_value):
            end = self.max_value + 0.1 * self.std_dev
        else:
            end = self.mean + 4 * self.std_dev
        return start, end


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

    def calculate_continuous_pdf(self, x: np.ndarray) -> ContinuousPdf:
        y = self.inner.calculate_continuous_pdf(self.algebraic_function.apply_inverse(x)).y / abs(
            self.algebraic_function.scale
        )
        return ContinuousPdf(x=x, y=y)

    def calculate_cdf(self, x: np.ndarray) -> np.ndarray:
        return self.inner.calculate_cdf(self.algebraic_function.apply_inverse(x))

    def calculate_inverse_cdf(self, x: np.ndarray) -> np.ndarray:
        return self.algebraic_function.apply(self.inner.calculate_inverse_cdf(x))

    def scale(self, x: float) -> "ScaledDistributionFunction":
        return ScaledDistributionFunction(inner=self.inner, algebraic_function=self.algebraic_function * x)

    def add_constant(self, x: float) -> "ScaledDistributionFunction":
        return ScaledDistributionFunction(inner=self.inner, algebraic_function=self.algebraic_function + x)

    def sample_numpy(self, n: int) -> np.ndarray:
        return self.algebraic_function.apply(self.inner.sample_numpy(n))

    def _get_plot_range(self) -> tuple[float, float]:
        return self.algebraic_function.apply_on_range(self.inner._get_plot_range())

    def copy(self) -> "ScaledDistributionFunction":
        return ScaledDistributionFunction(inner=self.inner.copy(), algebraic_function=self.algebraic_function)
