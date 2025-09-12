from scipy.stats import beta
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from randcraft.pdfs.scipy_pdf import ScipyDistributionFunction


class BetaDistributionFunction(ScipyDistributionFunction):
    def __init__(self, a: float, b: float) -> None:
        self._alpha = a
        self._beta = b
        self._scipy_rv: rv_continuous_frozen = beta(a, b)  # type: ignore

    @property
    def short_name(self) -> str:
        return "beta"

    @property
    def scipy_rv(self) -> rv_continuous_frozen:
        return self._scipy_rv

    def copy(self) -> "BetaDistributionFunction":
        return BetaDistributionFunction(a=self._alpha, b=self._beta)
