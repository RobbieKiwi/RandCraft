import numpy as np

from randcraft import make_coin_flip, make_dirac, make_normal
from randcraft.misc import add_special_event_to_rv, apply_func_to_continuous_rv, apply_func_to_discrete_rv, mix_rvs
from randcraft.random_variable import RandomVariable
from randcraft.rvs import MixtureRV
from tests.base_test_case import BaseTestCase


class TestMisc(BaseTestCase):
    def test_mixture_rv(self) -> None:
        rv_a = make_dirac(1.0)
        rv_b = make_normal(mean=2.0, std_dev=2.0)
        mixture = mix_rvs([rv_a, rv_b])

        self.assertIsInstance(mixture, RandomVariable)
        self.assertIsInstance(mixture._rv, MixtureRV)
        self.assertAlmostEqual(mixture.get_mean(exact=True), (1.0 + 2.0) / 2)
        # TODO Test variance of mixture, test using different weights

    def test_add_special_event(self) -> None:
        rv = make_dirac(value=1.0)
        new_rv = add_special_event_to_rv(rv=rv, value=2.0, chance=0.5)

        self.assertIsInstance(new_rv, RandomVariable)
        self.assertAlmostEqual(new_rv.get_mean(), 1.5)

    def test_apply_continuous(self) -> None:
        rv0 = make_normal(mean=1.0, std_dev=1.0, seed=2)
        rva = make_normal(mean=1.0, std_dev=1.0, seed=2)

        def double(x: np.ndarray) -> np.ndarray:
            return x * 2.0

        rv2a = apply_func_to_continuous_rv(rv=rva, func=double)
        self.assertTrue(np.array_equal(rv0.sample(10) * 2, rv2a.sample(10)))

    def test_apply_discrete(self) -> None:
        rv0 = make_coin_flip(seed=2)
        rva = make_coin_flip(seed=2)

        def double(x: np.ndarray) -> np.ndarray:
            return x * 2.0

        rv2a = apply_func_to_discrete_rv(rv=rva, func=double)
        self.assertTrue(np.array_equal(rv0.sample(10) * 2, rv2a.sample(10)))
