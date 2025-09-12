from scipy.stats import beta

from randcraft.pdfs.scipy_pdf import ScipyDistributionFunction


class BetaDistributionFunction(ScipyDistributionFunction):
    def __init__(self, a: float, b: float) -> None:
        self._alpha = a
        self._beta = b
        super().__init__(beta(a, b))

    @property
    def short_name(self) -> str:
        return "beta"

    def copy(self) -> "BetaDistributionFunction":
        return BetaDistributionFunction(a=self._alpha, b=self._beta)
