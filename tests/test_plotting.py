from unittest import TestCase

from randcraft import make_discrete, make_normal, make_uniform
from randcraft.constructors import make_beta
from randcraft.pdfs.continuous import ScaledDistributionFunction
from randcraft.random_variable import RandomVariable


class TestPlotting(TestCase):
    def test_plotting_convolved(self) -> None:
        rv1 = make_normal(mean=0, std_dev=1)
        rv2 = make_uniform(low=-1, high=1)
        combined = rv1 + rv2

        rv1.pdf.plot()
        rv2.pdf.plot()
        combined.pdf.plot()

        discrete = make_discrete(values=[1, 2, 3])
        discrete.pdf.plot()

        mixed = RandomVariable.mix_rvs([rv1, rv2, combined, discrete])
        mixed.pdf.plot()

    def test_scaled_plotting(self) -> None:
        a = 2.0
        b = 5.0

        rv = make_beta(a=a, b=b)
        rv.pdf.plot()

        new_rv = rv * -1 - 2
        self.assertIsInstance(new_rv, RandomVariable)
        self.assertIsInstance(new_rv.pdf, ScaledDistributionFunction)

        new_rv.pdf.plot()
