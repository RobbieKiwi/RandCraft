import numpy as np

from randcraft.constructors import make_coin_flip, make_normal
from tests.base_test_case import BaseTestCase


class TestSeed(BaseTestCase):
    def test_basic(self) -> None:
        rv1 = make_coin_flip(seed=2)
        rv2 = make_coin_flip(seed=2)
        self.assertTrue(np.array_equal(rv1.sample_numpy(10), rv2.sample_numpy(10)))

    def test_complex(self) -> None:
        rv1 = make_coin_flip(seed=2)
        rv2a = make_normal(mean=0.0, std_dev=1.0, seed=3)
        rv2b = make_normal(mean=0.0, std_dev=1.0, seed=3)
        rv2_no_seed = make_normal(mean=0.0, std_dev=1.0)

        rv3 = rv1 + rv2a
        rv3_no_seed = rv1 + rv2_no_seed
        self.assertTrue(rv3._rv.seeded)
        self.assertFalse(rv3_no_seed._rv.seeded)
        self.assertTrue(np.array_equal(rv2a.sample_numpy(10), rv2b.sample_numpy(10)))
