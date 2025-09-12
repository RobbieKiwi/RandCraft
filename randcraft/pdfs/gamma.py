from scipy.stats import gamma

from randcraft.pdfs.scipy_pdf import ScipyDistributionFunction


class GammaDistributionFunction(ScipyDistributionFunction):
    def __init__(self, a: float) -> None:
        self._alpha = a
        super().__init__(gamma(a=a))

    @property
    def short_name(self) -> str:
        return "gamma"

    def copy(self) -> "GammaDistributionFunction":
        return GammaDistributionFunction(a=self._alpha)
