from randcraft import make_dirac, make_normal
from randcraft.misc import add_special_event_to_rv, mix_rvs
from randcraft.pdfs import (
    MixtureDistributionFunction,
)
from randcraft.random_variable import RandomVariable
from tests.base_test_case import BaseTestCase


class TestMisc(BaseTestCase):
    def test_mixture_rv(self) -> None:
        rv_a = make_dirac(1.0)
        rv_b = make_normal(mean=2.0, std_dev=2.0)
        mixture = mix_rvs([rv_a, rv_b])

        self.assertIsInstance(mixture, RandomVariable)
        self.assertIsInstance(mixture.pdf, MixtureDistributionFunction)
        self.assertAlmostEqual(mixture.get_mean(exact=True), (1.0 + 2.0) / 2)
        # TODO Test variance of mixture, test using different weights

    def test_add_special_event(self) -> None:
        rv = make_dirac(value=1.0)
        new_rv = add_special_event_to_rv(rv=rv, value=2.0, chance=0.5)

        self.assertIsInstance(new_rv, RandomVariable)
        self.assertAlmostEqual(new_rv.get_mean(), 1.5)
