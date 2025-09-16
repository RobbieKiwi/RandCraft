from functools import cached_property

import numpy as np
from scipy.stats._distn_infrastructure import rv_continuous, rv_continuous_frozen

from randcraft.models import AlgebraicFunction, ContinuousPdf, Statistics, certainly
from randcraft.pdfs.continuous import ContinuousDistributionFunction, ScaledDistributionFunction


class RescalingError(Exception):
    pass


class ScipyDistributionFunction(ContinuousDistributionFunction):
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

        has_finite_support_on_one_side = not (
            self.statistics.has_infinite_lower_support and self.statistics.has_infinite_upper_support
        )

        if has_finite_support_on_one_side and af.scale < 0:
            return scale_with_scale_distribution()

        def scale_with_scipy() -> ScipyDistributionFunction:
            shape_args = self._scipy_rv.args
            shape_kwargs = {k: v for k, v in self._scipy_rv.kwds.items()}
            shape_kwargs["loc"] = 0.0
            shape_kwargs["scale"] = 1.0
            unit_distribution = self._scipy_rv_type(*shape_args, **shape_kwargs)  # type: ignore
            current_scale = self.std_dev / unit_distribution.std()
            current_loc = self.mean - unit_distribution.mean() * current_scale
            new_loc = af.apply(current_loc)
            new_scale = current_scale * af.scale
            shape_kwargs["loc"] = new_loc
            shape_kwargs["scale"] = new_scale
            return ScipyDistributionFunction(self._scipy_rv_type, *shape_args, **shape_kwargs)

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

    def calculate_continuous_pdf(self, x: np.ndarray) -> ContinuousPdf:
        return ContinuousPdf(x=x, y=self.scipy_rv.pdf(x))

    def calculate_cdf(self, x: np.ndarray) -> np.ndarray:
        return self.scipy_rv.cdf(x)

    def calculate_inverse_cdf(self, x: np.ndarray) -> np.ndarray:
        return self.scipy_rv.ppf(x)

    def copy(self) -> "ScipyDistributionFunction":
        return self.scale(1.0)  # type: ignore
