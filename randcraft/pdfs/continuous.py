from typing import Self

from randcraft.pdfs.base import ProbabilityDistributionFunction, ScaledDistributionFunction
from randcraft.pdfs.discrete import DiracDeltaDistributionFunction


class ContinuousDistributionFunction(ProbabilityDistributionFunction):
    def scale(self, x: float) -> Self | DiracDeltaDistributionFunction | ScaledDistributionFunction:
        x = float(x)
        if x == 0.0:
            return DiracDeltaDistributionFunction(value=0.0)
        return super().scale(x)  # type: ignore
