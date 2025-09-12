from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Protocol, TypeVar, runtime_checkable

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class Uncertainty[T]:
    value: T
    is_certain: bool

    def __post_init__(self) -> None:
        assert not isinstance(self.value, Uncertainty), "Nested Uncertainty instances are not allowed"

    def __str__(self) -> str:
        if self.is_certain:
            return f"{self.value} (certain)"
        return f"{self.value} (uncertain)"

    def __repr__(self) -> str:
        return str(self)

    def __mul__(self, other: "Uncertainty[T]" | T) -> "Uncertainty[T]":
        return self._combine(other=other, func=lambda x, y: x * y)

    def __add__(self, other: "Uncertainty[T]" | T) -> "Uncertainty[T]":
        return self._combine(other=other, func=lambda x, y: x + y)

    def __sub__(self, other: "Uncertainty[T]" | T) -> "Uncertainty[T]":
        return self._combine(other=other, func=lambda x, y: x - y)

    def __pow__(self, other: int | float) -> "Uncertainty[T]":
        assert not isinstance(other, Uncertainty), "Exponentiation with uncertain exponent is not supported"
        return self.apply(func=lambda x: x**other)

    def apply(self, func: Callable[[T], T]) -> "Uncertainty[T]":
        return Uncertainty(value=func(self.value), is_certain=self.is_certain)

    def get(self, name: str = "value", certain: bool = False) -> T:
        if certain and not self.is_certain:
            raise NotImplementedError(f"{name} is uncertain")
        return self.value

    def _combine(self, other: "Uncertainty[T]" | T, func: Callable[[T, T], T]) -> "Uncertainty[T]":
        if not isinstance(other, Uncertainty):
            other = Uncertainty(value=other, is_certain=True)
        return Uncertainty(value=func(self.value, other.value), is_certain=self.is_certain and other.is_certain)


def certainly[T](x: T) -> Uncertainty[T]:
    return Uncertainty(value=x, is_certain=True)


def maybe[T](x: T) -> Uncertainty[T]:
    return Uncertainty(value=x, is_certain=False)


def sum_uncertain_floats(uncertainties: Iterable[Uncertainty[float]]) -> Uncertainty[float]:
    return sum(uncertainties, start=Uncertainty(value=0.0, is_certain=True))


def sort_uncertainties(uncertainties: Iterable[Uncertainty[float]]) -> list[Uncertainty[float]]:
    return sorted(uncertainties, key=lambda x: x.value)


@dataclass(frozen=True)
class Statistics:
    moments: list[Uncertainty[float]]
    support: tuple[Uncertainty[float], Uncertainty[float]]

    def __post_init__(self) -> None:
        for m in self.moments:
            assert isinstance(m, Uncertainty), "All moments must be of type Uncertainty"
        for s in self.support:
            assert isinstance(s, Uncertainty), "All support values must be of type Uncertainty"
        assert len(self.moments) >= 2, f"At least 2 moments must be defined. Got {len(self.moments)}"

    @cached_property
    def central_moments(self) -> list[Uncertainty[float]]:
        def calculate_nth_central_moment(n: int) -> Uncertainty[float]:
            mean = self.mean
            non_central_moment = self.moments[n]
            return non_central_moment - mean ** (n + 1)

        return [calculate_nth_central_moment(n) for n, _ in enumerate(self.moments)]

    @property
    def mean(self) -> Uncertainty[float]:
        return self.moments[0]

    @property
    def variance(self) -> Uncertainty[float]:
        return self.central_moments[1]

    @cached_property
    def std_dev(self) -> Uncertainty[float]:
        return self.variance.apply(lambda x: x**0.5)

    @property
    def min_value(self) -> Uncertainty[float]:
        return self.support[0]

    @property
    def max_value(self) -> Uncertainty[float]:
        return self.support[1]

    def get(
        self, name: Literal["mean", "variance", "std_dev", "min_value", "max_value"], certain: bool = False
    ) -> float:
        uncertainty = {
            "mean": self.mean,
            "variance": self.variance,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }[name]
        return uncertainty.get(name=name, certain=certain)

    def apply_algebraic_function(self, af: "AlgebraicFunction") -> "Statistics":
        scale = af.scale
        offset = af.offset
        new_min_max = (self.min_value * scale + offset, self.max_value * scale + offset)
        new_min, new_max = sort_uncertainties(new_min_max)

        mean = self.mean * scale + offset
        variance = self.variance * (scale**2)
        second_moment = variance + mean**2
        moments = [mean, second_moment]
        # TODO calculate higher order moments

        return Statistics(moments=moments, support=(new_min, new_max))


@runtime_checkable
class SupportsFloatAlgebra(Protocol):
    def __add__(self: T, x: float, /) -> T: ...
    def __radd__(self: T, x: float, /) -> T: ...
    def __sub__(self: T, x: float, /) -> T: ...
    def __rsub__(self: T, x: float, /) -> T: ...
    def __mul__(self: T, x: float, /) -> T: ...
    def __rmul__(self: T, x: float, /) -> T: ...
    def __truediv__(self: T, x: float, /) -> T: ...


F = TypeVar("F", bound=SupportsFloatAlgebra)


@dataclass(frozen=True)
class AlgebraicFunction:
    scale: float = 1.0
    offset: float = 0.0

    @cached_property
    def is_active(self) -> bool:
        return self.scale != 1.0 or self.offset != 0.0

    def __mul__(self, factor: float) -> "AlgebraicFunction":
        factor = float(factor)
        return AlgebraicFunction(scale=self.scale * factor, offset=self.offset * factor)

    def __rmul__(self, factor: float) -> "AlgebraicFunction":
        return self.__mul__(factor)

    def __add__(self, addend: float) -> "AlgebraicFunction":
        addend = float(addend)
        return AlgebraicFunction(scale=self.scale, offset=self.offset + addend)

    def __radd__(self, addend: float) -> "AlgebraicFunction":
        return self.__add__(addend)

    def apply(self, x: F) -> F:
        return self.scale * x + self.offset

    def apply_inverse(self, y: F) -> F:
        return (y - self.offset) / self.scale

    def apply_on_range(self, x_range: tuple[float, float]) -> tuple[float, float]:
        processed = self.apply(np.array(x_range))
        return float(np.min(processed)), float(np.max(processed))
