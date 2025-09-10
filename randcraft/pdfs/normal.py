from scipy.stats import norm
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from randcraft.pdfs import DiracDeltaDistributionFunction
from randcraft.pdfs.scipy_pdf import ScipyDistributionFunction


class NormalDistributionFunction(ScipyDistributionFunction):
    def __init__(self, mean: float, std_dev: float) -> None:
        self._scipy_rv: rv_continuous_frozen = norm(loc=mean, scale=std_dev)  # type: ignore

    @classmethod
    def get_short_name(cls) -> str:
        return "normal"

    @property
    def scipy_rv(self) -> rv_continuous_frozen:
        return self._scipy_rv

    def scale(self, x: float) -> "NormalDistributionFunction | DiracDeltaDistributionFunction":
        x = float(x)
        if x == 0.0:
            return DiracDeltaDistributionFunction(value=0.0)
        return NormalDistributionFunction(mean=self.mean * x, std_dev=self.std_dev * abs(x))

    def add_constant(self, x: float) -> "NormalDistributionFunction":
        return NormalDistributionFunction(mean=self.mean + float(x), std_dev=self.std_dev)
