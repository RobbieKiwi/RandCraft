import numpy as np

from randcraft.anonymous import make_anonymous_continuous_rv_from_sampler
from tests.base_test_case import BaseTestCase


class TestAnon(BaseTestCase):
    def test_anonymous_rv(self) -> None:
        print("hello")

        def sampler(n: int) -> np.ndarray:
            generator = np.random.default_rng()
            return 20.0 + generator.random(n)

        rv = make_anonymous_continuous_rv_from_sampler(sampler=sampler)

        sample = rv.sample_one()

        self.assertTrue(20.0 <= sample <= 21.0)

        # Statistics are not exact
        print(rv.get_mean())
        self.assertTrue(20.0 <= rv.get_mean() <= 21.0)
        self.assertTrue(0.0 < rv.get_variance() <= 1.0)
        print("hi")
        self.assertLess(rv.cdf(value=20.0), 0.1)
        self.assertGreater(rv.cdf(value=21.0), 0.9)

        self.assertRaises(NotImplementedError, lambda: rv.get_mean(exact=True))
        self.assertRaises(NotImplementedError, lambda: rv.get_variance(exact=True))
