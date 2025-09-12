from scipy.stats import norm

from randcraft.pdfs.scipy_pdf import ScipyDistributionFunction


class NormalDistributionFunction(ScipyDistributionFunction):
    def __init__(self, mean: float, std_dev: float) -> None:
        super().__init__(norm(loc=mean, scale=std_dev))

    @property
    def short_name(self) -> str:
        return "normal"

    def scale(self, x: float) -> "NormalDistributionFunction":
        x = float(x)
        return NormalDistributionFunction(mean=self.mean * x, std_dev=self.std_dev * abs(x))

    def add_constant(self, x: float) -> "NormalDistributionFunction":
        return NormalDistributionFunction(mean=self.mean + float(x), std_dev=self.std_dev)

    def copy(self) -> "NormalDistributionFunction":
        return NormalDistributionFunction(mean=self.mean, std_dev=self.std_dev)
