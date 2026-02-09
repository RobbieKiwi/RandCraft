import numpy as np

from randcraft.sampler import make_anonymous_continuous_rv_from_sampler, make_discrete_rv_from_sampler
from tests.base_test_case import BaseTestCase


class TestAnon(BaseTestCase):
    def test_continuous_from_sampler(self) -> None:
        def sampler(n: int) -> np.ndarray:
            generator = np.random.default_rng(seed=3)
            return 20.0 + generator.random(n)

        rv = make_anonymous_continuous_rv_from_sampler(sampler=sampler)

        # Statistics are not exact
        print(rv.get_mean())
        self.assertTrue(20.0 <= rv.get_mean() <= 21.0)
        self.assertTrue(0.0 < rv.get_variance() <= 1.0)
        print("hi")
        self.assertLess(rv.cdf(value=20.0), 0.1)
        self.assertGreater(rv.cdf(value=21.0), 0.9)

        self.assertRaises(NotImplementedError, lambda: rv.get_mean(exact=True))
        self.assertRaises(NotImplementedError, lambda: rv.get_variance(exact=True))

    def test_discrete_from_sampler(self) -> None:
        def sampler(n: int) -> np.ndarray:
            generator = np.random.default_rng(seed=3)
            return generator.integers(low=0, high=5, size=n)

        rv = make_discrete_rv_from_sampler(sampler=sampler)

        self.assertTrue(0.0 <= rv.get_mean() <= 5.0)
        self.assertTrue(0.0 < rv.get_variance() <= 2.0)
