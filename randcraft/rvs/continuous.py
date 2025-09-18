from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from randcraft.models import AlgebraicFunction, ProbabilityDensityFunction, Statistics
from randcraft.rvs.base import RV


class ContinuousRV(RV, ABC):
    @abstractmethod
    def calculate_pdf(self, x: np.ndarray) -> ProbabilityDensityFunction: ...

    @abstractmethod
    def ppf(self, x: np.ndarray) -> np.ndarray: ...

    def calculate_pmf(self) -> None:
        return None

    def _get_discrete_points(self) -> np.ndarray:
        points: list[float] = []
        if not self.statistics.has_infinite_lower_support:
            points.append(self.statistics.min_value.value)
        if not self.statistics.has_infinite_upper_support:
            points.append(self.statistics.max_value.value)
        return np.array(points)

    def scale(self, x: float) -> Self | "ScaledRV":
        # Override these if possible
        return ScaledRV(inner=self.copy(), algebraic_function=AlgebraicFunction(scale=x))

    def add_constant(self, x: float) -> Self | "ScaledRV":
        # Override these if possible
        return ScaledRV(inner=self.copy(), algebraic_function=AlgebraicFunction(offset=x))

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


class ScaledRV(ContinuousRV):
    def __init__(self, inner: ContinuousRV, algebraic_function: AlgebraicFunction) -> None:
        self._inner = inner
        self._af = algebraic_function
        assert self.algebraic_function.scale != 0.0, "Scale cannot be zero"

    @property
    def inner(self) -> ContinuousRV:
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

    def calculate_pdf(self, x: np.ndarray) -> ProbabilityDensityFunction:
        y = self.inner.calculate_pdf(self.algebraic_function.apply_inverse(x)).y / abs(self.algebraic_function.scale)
        return ProbabilityDensityFunction(x=x, y=y)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self.inner.cdf(self.algebraic_function.apply_inverse(x))

    def ppf(self, x: np.ndarray) -> np.ndarray:
        return self.algebraic_function.apply(self.inner.ppf(x))

    def scale(self, x: float) -> "ScaledRV":
        return ScaledRV(inner=self.inner, algebraic_function=self.algebraic_function * x)

    def add_constant(self, x: float) -> "ScaledRV":
        return ScaledRV(inner=self.inner, algebraic_function=self.algebraic_function + x)

    def sample_numpy(self, n: int) -> np.ndarray:
        return self.algebraic_function.apply(self.inner.sample_numpy(n))

    def _get_plot_range(self) -> tuple[float, float]:
        return self.algebraic_function.apply_on_range(self.inner._get_plot_range())

    def copy(self) -> "ScaledRV":
        return ScaledRV(inner=self.inner.copy(), algebraic_function=self.algebraic_function)
