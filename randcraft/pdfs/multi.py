from collections.abc import Sequence
from functools import cached_property

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import fftconvolve

from randcraft.models import Statistics, sum_uncertain_floats
from randcraft.pdfs.continuous import ContinuousDistributionFunction
from randcraft.pdfs.discrete import DiracDeltaDistributionFunction, DiscreteDistributionFunction


class MultiDistributionFunction(ContinuousDistributionFunction):
    def __init__(
        self,
        continuous_pdfs: list[ContinuousDistributionFunction],
        discrete_pdf: DiscreteDistributionFunction = DiracDeltaDistributionFunction(value=0.0),
    ) -> None:
        assert len(continuous_pdfs) > 0, "At least one continuous pdf is required"
        assert isinstance(discrete_pdf, DiscreteDistributionFunction), (
            f"discrete_pdf must be a {DiscreteDistributionFunction.__name__}, got {type(discrete_pdf)}"
        )
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

    @cached_property
    def discrete_pdf(self) -> DiscreteDistributionFunction:
        return self._discrete_pdf

    @cached_property
    def has_discrete_pdf(self) -> bool:
        if isinstance(self.discrete_pdf, DiracDeltaDistributionFunction) and self.discrete_pdf.value == 0.0:
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

    def calculate_pdf(self, x: np.ndarray) -> np.ndarray:
        if not self.has_discrete_pdf:
            return self._calculate_continuous_pdf(x)

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

        return result

    def _calculate_continuous_pdf(self, x: np.ndarray) -> np.ndarray:
        if len(self.continuous_pdfs) == 1:
            return self.continuous_pdfs[0].calculate_pdf(x)

        assert x.ndim == 1, "Input numpy array must be 1D"
        low = np.min(x)
        high = np.max(x)
        spread = high - low
        start = low - spread
        end = high + spread
        x2 = np.linspace(start, end, len(x) * 3)
        raw_pdfs = [pdf.calculate_pdf(x2) for pdf in self.continuous_pdfs]
        combined_pdf = fftconvolve(raw_pdfs[0], raw_pdfs[1], mode="same")
        for pdf in raw_pdfs[2:]:
            combined_pdf = fftconvolve(combined_pdf, pdf, mode="same")
        combined_pdf /= np.trapezoid(combined_pdf, x2)  # Normalize
        return np.interp(x, x2, combined_pdf)

    @cached_property
    def _cdf(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns two numpy arrays (x_values, cumulative_probabilities) representing the CDF.
        The chance that x < value can be found by interpolating cumulative_probabilities at value.
        """
        mean = self.statistics.mean.value
        std_dev = self.statistics.std_dev.value
        has_finite_lower_support = not np.isinf(self.statistics.min_value.value)
        has_finite_upper_support = not np.isinf(self.statistics.max_value.value)

        if has_finite_lower_support and has_finite_upper_support:
            lower = self.statistics.min_value.value
            upper = self.statistics.max_value.value
        elif has_finite_lower_support:
            lower = self.statistics.min_value.value
            upper = lower + 6 * std_dev
        elif has_finite_upper_support:
            upper = self.statistics.max_value.value
            lower = upper - 6 * std_dev
        else:
            lower = mean - 3 * std_dev
            upper = mean + 3 * std_dev

        x_values = np.linspace(lower, upper, 10000)
        pdf_vals = self.calculate_pdf(x_values)

        cdf_vals = cumulative_trapezoid(pdf_vals, x_values, initial=0)
        if cdf_vals[-1] < 1:
            remainder = 1 - cdf_vals[-1]
            cdf_vals = cdf_vals + remainder / 2
            x_values = np.concatenate(([-np.inf], x_values, [np.inf]))
            cdf_vals = np.concatenate(([0.0], cdf_vals, [1.0]))
        else:
            cdf_vals /= cdf_vals[-1]  # Normalize
            cdf_vals[0] = 0.0
            cdf_vals[-1] = 1.0

        return x_values, cdf_vals

    def calculate_cdf(self, x: np.ndarray) -> np.ndarray:
        x_values, cumulative_probs = self._cdf
        return np.interp(x, x_values, cumulative_probs)

    def calculate_inverse_cdf(self, x: np.ndarray) -> np.ndarray:
        x_values, cumulative_probs = self._cdf
        return np.interp(x, cumulative_probs, x_values)

    def scale(self, x: float) -> "MultiDistributionFunction":
        x = float(x)
        continuous_pdfs = [pdf.scale(x) for pdf in self.continuous_pdfs]
        discrete_pdf = self.discrete_pdf.scale(x)
        return MultiDistributionFunction(continuous_pdfs=continuous_pdfs, discrete_pdf=discrete_pdf)

    def add_constant(self, x: float) -> "MultiDistributionFunction":
        x = float(x)
        discrete_pdf = self.discrete_pdf.add_constant(x)
        return MultiDistributionFunction(continuous_pdfs=self.continuous_pdfs, discrete_pdf=discrete_pdf)

    def sample_numpy(self, n: int) -> np.ndarray:
        return sum([pdf.sample_numpy(n) for pdf in self.pdfs])  # type: ignore

    def copy(self) -> "MultiDistributionFunction":
        return MultiDistributionFunction(continuous_pdfs=self.continuous_pdfs, discrete_pdf=self.discrete_pdf)
