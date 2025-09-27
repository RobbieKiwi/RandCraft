from collections.abc import Sequence
from functools import cached_property

import numpy as np
from scipy.signal import fftconvolve

from randcraft.models import ProbabilityDensityFunction, Statistics, Uncertainty, sum_uncertain_floats
from randcraft.rvs.base import CdfEstimator
from randcraft.rvs.continuous import ContinuousRV
from randcraft.rvs.discrete import DiracDeltaRV, DiscreteRV


class MultiRV(ContinuousRV):
    def __init__(
        self,
        continuous_pdfs: list[ContinuousRV],
        discrete_pdf: DiscreteRV = DiracDeltaRV(value=0.0),
    ) -> None:
        assert len(continuous_pdfs) > 0, "At least one continuous pdf is required"
        assert isinstance(discrete_pdf, DiscreteRV), f"discrete_pdf must be a {DiscreteRV.__name__}, got {type(discrete_pdf)}"
        self._continuous_pdfs = continuous_pdfs
        self._discrete_pdf = discrete_pdf

    @property
    def short_name(self) -> str:
        return "multi"

    @cached_property
    def pdfs(self) -> Sequence[ContinuousRV | DiscreteRV]:
        return self.continuous_pdfs + ([self._discrete_pdf] if self._discrete_pdf is not None else [])

    @property
    def continuous_pdfs(self) -> list[ContinuousRV]:
        return self._continuous_pdfs

    @cached_property
    def discrete_pdf(self) -> DiscreteRV:
        return self._discrete_pdf

    @cached_property
    def has_discrete_pdf(self) -> bool:
        if isinstance(self.discrete_pdf, DiracDeltaRV) and self.discrete_pdf.value == 0.0:
            return False
        return True

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

    def calculate_pdf(self, x: np.ndarray) -> ProbabilityDensityFunction:
        if not self.has_discrete_pdf:
            return ProbabilityDensityFunction(x=x, y=self._calculate_continuous_pdf(x))

        result = np.zeros_like(x)

        shifted_xs: list[np.ndarray] = []
        for offset in self.discrete_pdf.values:
            shifted_x = x - offset
            shifted_xs.append(shifted_x)

        # Find all unique values across all shifted_xs
        all_shifted_values = np.concatenate(shifted_xs)
        unique_shifted_values = np.unique(all_shifted_values)
        # Calculate continuous PDF only once for all unique shifted values
        unique_continuous = self._calculate_continuous_pdf(unique_shifted_values)
        # Create a mapping from value to index in unique_shifted_values
        idx_map = {val: idx for idx, val in enumerate(unique_shifted_values)}
        for offset, scale in zip(self.discrete_pdf.values, self.discrete_pdf.probabilities):
            shifted_x = x - offset
            # Map each value in shifted_x to its index in unique_shifted_values
            indices = np.array([idx_map.get(val, -1) for val in shifted_x])
            valid = indices >= 0
            result[valid] += scale * unique_continuous[indices[valid]]

        return ProbabilityDensityFunction(x=x, y=result)

    def _calculate_continuous_pdf(self, x: np.ndarray) -> np.ndarray:
        C = len(self.continuous_pdfs)
        if C == 1:
            return self.continuous_pdfs[0].calculate_pdf(x).y
        assert x.ndim == 1, "Input numpy array must be 1D"
        return self._continuous_pdf_ref.evaluate(x=x)

    @cached_property
    def _continuous_pdf_ref(self) -> ProbabilityDensityFunction:
        C = len(self.continuous_pdfs)
        N = 10000

        if self.statistics.has_infinite_lower_support:
            low = self.mean - 5 * self.std_dev
        else:
            low = self.statistics.support[0].value
        low = min(low, 0.0)

        if self.statistics.has_infinite_upper_support:
            high = self.mean + 5 * self.std_dev
        else:
            high = self.statistics.support[1].value
        high = max(high, 0.0)

        spread = high - low
        spread = max(spread, 1.0)  # Ensure some minimum spread
        start = low - spread
        end = high + spread
        x2 = np.linspace(start, end, N)
        zero_pad = np.zeros(N)

        def pad(arr: np.ndarray) -> np.ndarray:
            return np.concatenate((zero_pad, arr, zero_pad))

        pdfs = [pad(pdf.calculate_pdf(x2).y) for pdf in self.continuous_pdfs]

        full_pdf = fftconvolve(pdfs[0], pdfs[1], mode="full")
        for pdf in pdfs[2:]:
            full_pdf = fftconvolve(full_pdf, pdf, mode="full")

        new_x = np.linspace(start * C, end * C, C * (N - 1) + 1)
        short_pdf = full_pdf[N * C : N * C + len(new_x)]
        short_pdf /= np.trapezoid(short_pdf, new_x)  # Normalize
        return ProbabilityDensityFunction(x=new_x, y=short_pdf)

    @cached_property
    def _cdf_estimator(self) -> CdfEstimator:
        return CdfEstimator(rv=self)

    def cdf(self, x: np.ndarray) -> Uncertainty[np.ndarray]:
        return self._cdf_estimator.cdf(x)

    def ppf(self, x: np.ndarray) -> Uncertainty[np.ndarray]:
        return self._cdf_estimator.ppf(x)

    def scale(self, x: float) -> "MultiRV":
        x = float(x)
        continuous_pdfs = [pdf.scale(x) for pdf in self.continuous_pdfs]
        discrete_pdf = self.discrete_pdf.scale(x)
        return MultiRV(continuous_pdfs=continuous_pdfs, discrete_pdf=discrete_pdf)

    def add_constant(self, x: float) -> "MultiRV":
        x = float(x)
        discrete_pdf = self.discrete_pdf.add_constant(x)
        return MultiRV(continuous_pdfs=self.continuous_pdfs, discrete_pdf=discrete_pdf)

    def sample_numpy(self, n: int) -> np.ndarray:
        return sum([pdf.sample_numpy(n) for pdf in self.pdfs])  # type: ignore

    def copy(self) -> "MultiRV":
        return MultiRV(continuous_pdfs=self.continuous_pdfs, discrete_pdf=self.discrete_pdf)
