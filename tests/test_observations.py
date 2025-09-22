import numpy as np

from randcraft.observations import make_gaussian_kde
from tests.base_test_case import BaseTestCase


class TestObservations(BaseTestCase):
    def test_kde(self) -> None:
        observations_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 1
        rv_a = make_gaussian_kde(observations=observations_a, bw_method=0.5)

        observations_b = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        rv_b = make_gaussian_kde(observations=observations_b)
        rv_joined = rv_a + rv_b

        self.assertAlmostEqual(rv_a.get_mean(), 3.0)
        self.assertAlmostEqual(rv_b.get_mean(), 0.5)
        self.assertAlmostEqual(rv_joined.get_mean(), 3.0 + 0.5)

        self.assertAlmostEqual(rv_a.get_variance(), float(np.var(observations_a)) * 2.25)
