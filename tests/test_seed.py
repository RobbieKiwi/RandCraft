import numpy as np

from randcraft.constructors import make_coin_flip, make_normal
from randcraft.misc import mix_rvs
from tests.base_test_case import BaseTestCase


class TestSeed(BaseTestCase):
    def test_basic(self) -> None:
        rv1 = make_coin_flip(seed=2)
        rv2 = make_coin_flip(seed=2)
        self.assertTrue(np.array_equal(rv1.sample_numpy(10), rv2.sample_numpy(10)))

    def test_scipy(self) -> None:
        rv1 = make_normal(mean=0.0, std_dev=1.0, seed=3)
        rv2 = make_normal(mean=0.0, std_dev=1.0, seed=3)

        r1 = rv1.sample_numpy(10)
        r2 = rv2.sample_numpy(10)
        self.assertTrue(np.array_equal(r1, r2))
        self.assertFalse(np.array_equal(rv1.sample_numpy(10), r1))

        rv3 = make_normal(mean=0.0, std_dev=1.0)
        rv4 = make_normal(mean=0.0, std_dev=1.0)
        self.assertFalse(np.array_equal(rv3.sample_numpy(100), rv4.sample_numpy(100)))

    def test_multi(self) -> None:
        rv1 = make_coin_flip(seed=2)
        rv2a = make_normal(mean=0.0, std_dev=1.0, seed=3)
        rv2b = make_normal(mean=0.0, std_dev=1.0, seed=3)
        rv2_no_seed = make_normal(mean=0.0, std_dev=1.0)

        rv3 = rv1 + rv2a
        rv3_no_seed = rv1 + rv2_no_seed
        self.assertTrue(rv3._rv.seeded)
        self.assertFalse(rv3_no_seed._rv.seeded)
        self.assertTrue(np.array_equal(rv2a.sample_numpy(10), rv2b.sample_numpy(10)))

    def test_mixture(self) -> None:
        rv1 = make_coin_flip(seed=2)
        rv2 = make_normal(mean=0.0, std_dev=1.0, seed=3)
        rv3 = make_normal(mean=0.0, std_dev=1.0, seed=3)
        mixed = mix_rvs(rvs=[rv1, rv2, rv3], seed=1)

        rv1b = make_coin_flip(seed=2)
        rv2b = make_normal(mean=0.0, std_dev=1.0, seed=3)
        rv3b = make_normal(mean=0.0, std_dev=1.0, seed=3)
        mixedb = mix_rvs(rvs=[rv1b, rv2b, rv3b], seed=1)

        self.assertTrue(mixed._rv.seeded)
        self.assertTrue(np.array_equal(mixed.sample_numpy(10), mixedb.sample_numpy(10)))
