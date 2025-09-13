from unittest import TestCase

from scipy.stats import beta, uniform

from randcraft.constructors import make_scipy
from randcraft.pdfs.base import ScaledDistributionFunction
from randcraft.pdfs.scipy_pdf import ScipyDistributionFunction


class TestScipyRvs(TestCase):
    def test_uniform_rv(self) -> None:
        rv = make_scipy(uniform, loc=1, scale=2)
        b = rv * 2
        b = rv * 1

        self.assertIsInstance(b.pdf, ScipyDistributionFunction)
        self.assertEqual(b.get_mean(), rv.get_mean())

    def test_beta_rv(self) -> None:
        rv = make_scipy(beta, 1, 2)
        b = rv * 1

        self.assertIsInstance(b.pdf, ScipyDistributionFunction)
        self.assertEqual(b.get_mean(), rv.get_mean())

        # A beta distribution can not be flipped to be negative so we need to use the scaled distribution function
        c = rv * -1
        self.assertEqual(c.get_mean(), rv.get_mean() * -1)
        self.assertIsInstance(c.pdf, ScaledDistributionFunction)
