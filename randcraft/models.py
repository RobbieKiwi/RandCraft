from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, TypeVar

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
