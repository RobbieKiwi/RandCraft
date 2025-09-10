from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar, Callable, Literal, Iterable

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

    def __add__(self, other: "Uncertainty[T] " | T) -> "Uncertainty[T]":
        return self._combine(other=other, func=lambda x, y: x + y)

    def __sub__(self, other: "Uncertainty[T] | T") -> "Uncertainty[T]":
        return self._combine(other=other, func=lambda x, y: x - y)

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


def certainly(x: T) -> Uncertainty[T]:
    return Uncertainty(value=x, is_certain=True)


def maybe(x: T) -> Uncertainty[T]:
    return Uncertainty(value=x, is_certain=False)


def sum_uncertain_floats(uncertainties: Iterable[Uncertainty[float]]) -> Uncertainty[float]:
    return sum(uncertainties, start=Uncertainty(value=0.0, is_certain=True))


def sort_uncertainties(uncertainties: Iterable[Uncertainty[float]]) -> list[Uncertainty[float]]:
    return sorted(uncertainties, key=lambda x: x.value)


@dataclass(frozen=True)
class Statistics:
    mean: Uncertainty[float]
    variance: Uncertainty[float]
    min_value: Uncertainty[float]
    max_value: Uncertainty[float]

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

    @cached_property
    def std_dev(self) -> Uncertainty[float]:
        return Uncertainty(value=self.variance.value**0.5, is_certain=self.variance.is_certain)

    @cached_property
    def expectation_of_x_squared(self) -> Uncertainty[float]:
        return self.mean.apply(lambda x: x**2) + self.variance.apply(lambda x: x**2)
