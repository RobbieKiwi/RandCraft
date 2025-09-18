from scipy.stats import beta, uniform

from randcraft.constructors import make_scipy
from randcraft.rvs.scipy_pdf import SciRV
from tests.base_test_case import BaseTestCase


class TestScipyRvs(BaseTestCase):
    def test_uniform_rv(self) -> None:
        rv = make_scipy(uniform, loc=1, scale=2)
        b = rv.scale(2)
        b = rv.scale(1)

        self.assertIsInstance(b._rv, SciRV)
        self.assertEqual(b.get_mean(), rv.get_mean())

    def test_beta_rv(self) -> None:
        rv = make_scipy(beta, 1, 2)
        b = rv.scale(1)
        self.assert_stats_are_close(a=b.statistics, b=rv.statistics)

        c = rv.scale(-1)
        self.assertEqual(c.get_mean(), rv.get_mean() * -1)
        self.assertEqual(c.get_variance(), rv.get_variance())
