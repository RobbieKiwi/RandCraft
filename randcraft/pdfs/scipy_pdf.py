from functools import cached_property

import numpy as np
from matplotlib.axes import Axes
from scipy.stats._distn_infrastructure import rv_continuous, rv_continuous_frozen

from randcraft.models import AlgebraicFunction, Statistics, certainly
from randcraft.pdfs.base import ProbabilityDistributionFunction, ScaledDistributionFunction


class RescalingError(Exception):
    pass


class ScipyDistributionFunction(ProbabilityDistributionFunction):
    def __init__(self, scipy_rv_type: rv_continuous, *args, **kwargs) -> None:
        # It really should be continuous, but the typing here helps to interface with strange scipy typing
        scipy_rv = scipy_rv_type(*args, **kwargs)
        assert isinstance(scipy_rv, rv_continuous_frozen)
        self._scipy_rv_type = scipy_rv_type
        self._scipy_rv = scipy_rv

    @cached_property
    def short_name(self) -> str:
        try:
            name = self.scipy_rv.dist.name  # type: ignore
            return "scipy-" + name
        except AttributeError:
            return "scipy-unknown"

    @property
    def scipy_rv(self) -> rv_continuous_frozen:
        return self._scipy_rv

    @cached_property
    def statistics(self) -> Statistics:
        support = self.scipy_rv.support()
        lower = float(support[0])
        upper = float(support[1])

        return Statistics(
            moments=[certainly(self.scipy_rv.moment(n)) for n in range(1, 5)],
            support=(certainly(lower), certainly(upper)),
        )

    def scale(self, x: float) -> "ScipyDistributionFunction | ScaledDistributionFunction":
        return self._safe_rescale(AlgebraicFunction(scale=float(x), offset=0.0))

    def add_constant(self, x: float) -> "ScipyDistributionFunction | ScaledDistributionFunction":
        return self._safe_rescale(AlgebraicFunction(scale=1.0, offset=float(x)))

    def _safe_rescale(self, af: AlgebraicFunction) -> "ScipyDistributionFunction | ScaledDistributionFunction":
        def scale_with_scale_distribution() -> ScaledDistributionFunction:
            return ScaledDistributionFunction(inner=self, algebraic_function=af)

        has_infinite_lower_support = self.scipy_rv.support()[0] == -np.inf
        has_infinite_upper_support = self.scipy_rv.support()[1] == np.inf
        has_finite_support_on_one_side = not (has_infinite_lower_support and has_infinite_upper_support)

        if has_finite_support_on_one_side and af.scale < 0:
            return scale_with_scale_distribution()

        def scale_with_scipy() -> ScipyDistributionFunction:
            shapes: str | None = self._scipy_rv_type.shapes  # type: ignore
            if shapes is None:
                shape_params = []
            else:
                shape_params = shapes.split(", ")
            shape_args = self._scipy_rv.args
            shape_kwargs = {k: v for k, v in zip(shape_params, shape_args)}
            unit_distribution = self._scipy_rv_type(loc=0.0, scale=1.0, **shape_kwargs)
            current_scale = self.std_dev / unit_distribution.std()
            current_loc = self.mean - unit_distribution.mean() * current_scale
            new_loc = af.apply(current_loc)
            new_scale = current_scale * af.scale
            shape_kwargs["loc"] = new_loc
            shape_kwargs["scale"] = new_scale
            return ScipyDistributionFunction(self._scipy_rv_type, **shape_kwargs)

        for f in [scale_with_scipy, scale_with_scale_distribution]:
            result = f()
            expected_stats = self.statistics.apply_algebraic_function(af)
            result_stats = result.statistics

            all_close = True
            for m1, m2 in zip(expected_stats.moments, result_stats.moments):
                if not np.isclose(m1.value, m2.value):
                    all_close = False
                    break
            if not all_close:
                continue
            return result

        raise RescalingError(f"Could not rescale the {self._scipy_rv_type} correctly.")

    def sample_numpy(self, n: int) -> np.ndarray:
        return self.scipy_rv.rvs(size=n)

    def chance_that_rv_is_le(self, value: float) -> float:
        return float(self.scipy_rv.cdf(x=value))

    def value_that_is_at_le_chance(self, chance: float) -> float:
        return float(self.scipy_rv.ppf(q=chance))

    def _get_plot_range(self) -> tuple[float, float]:
        if not np.isinf(self.min_value):
            start = self.min_value
        else:
            start = self.mean - 4 * self.std_dev
        if not np.isinf(self.max_value):
            end = self.max_value
        else:
            end = self.mean + 4 * self.std_dev
        return start, end

    def plot_pdf_on_axis(self, ax: Axes, af: AlgebraicFunction | None = None) -> None:
        start, end = self._get_plot_range()
        x = np.linspace(start, end, 1000)
        y = self.scipy_rv.pdf(x)

        if af is not None:
            x = af.apply(x)
            y = y / abs(af.scale)
        ax.plot(x, y)

    def plot_cdf_on_axis(self, ax: Axes, af: AlgebraicFunction | None = None) -> None:
        start, end = self._get_plot_range()
        x = np.linspace(start, end, 1000)
        y = self.scipy_rv.cdf(x)

        if af is not None:
            x = af.apply(x)
            if af.scale < 0:
                y = 1 - y
        ax.plot(x, y)

    def copy(self) -> "ScipyDistributionFunction":
        return self.scale(1.0)  # type: ignore
