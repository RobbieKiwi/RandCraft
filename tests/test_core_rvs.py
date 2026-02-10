from unittest import TestCase

import numpy as np

from randcraft import make_dirac, make_discrete, make_gamma, make_normal, make_uniform
from randcraft.constructors import make_beta


class TestCoreRvs(TestCase):
    def test_normal_rv(self) -> None:
        mean = 23.4
        std_dev = 2.1

        rv = make_normal(mean=mean, std_dev=std_dev)

        sample = rv.sample()
        self.assertIsInstance(sample, float)

        self.assertEqual(rv.get_mean(exact=True), mean)
        self.assertAlmostEqual(rv.get_variance(exact=True), std_dev**2)
        self.assertAlmostEqual(rv.cdf(value=mean - std_dev, exact=True), 0.15865525393145707)
        self.assertAlmostEqual(rv.cdf(value=mean, exact=True), 0.5)
        self.assertAlmostEqual(rv.cdf(value=mean + std_dev, exact=True), 0.8413447460685429)

    def test_uniform_rv(self) -> None:
        low = 10.0
        high = 20.0

        rv = make_uniform(low=low, high=high)

        sample = rv.sample()
        self.assertIsInstance(sample, float)

        samples = rv.sample(n=100)
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.shape, (100,))
        self.assertGreaterEqual(min(samples), low)
        self.assertLessEqual(max(samples), high)

        self.assertEqual(rv.get_mean(exact=True), (low + high) / 2)
        self.assertAlmostEqual(rv.get_variance(exact=True), ((high - low) ** 2) / 12)
        self.assertAlmostEqual(rv.cdf(value=low, exact=True), 0.0)
        self.assertAlmostEqual(rv.cdf(value=15.0, exact=True), 0.5)
        self.assertAlmostEqual(rv.cdf(value=high, exact=True), 1.0)

    def test_beta_rv(self) -> None:
        a = 2.0
        b = 5.0

        rv = make_beta(a=a, b=b)

        sample = rv.sample()
        self.assertIsInstance(sample, float)
        self.assertGreaterEqual(sample, 0.0)
        self.assertLessEqual(sample, 1.0)

        samples = rv.sample(n=100)
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.shape, (100,))
        self.assertGreaterEqual(min(samples), 0.0)
        self.assertLessEqual(max(samples), 1.0)

        self.assertAlmostEqual(rv.get_mean(exact=True), a / (a + b))
        self.assertAlmostEqual(rv.get_variance(exact=True), (a * b) / ((a + b) ** 2 * (a + b + 1)))
        self.assertAlmostEqual(rv.cdf(value=0.0, exact=True), 0.0)
        self.assertAlmostEqual(rv.cdf(value=0.5, exact=True), 0.890625)
        self.assertAlmostEqual(rv.cdf(value=1.0, exact=True), 1.0)

    def test_gamma_rv(self) -> None:
        shape = 2.0
        scale = 3.0

        rv = make_gamma(a=shape, scale=scale)

        sample = rv.sample()
        self.assertIsInstance(sample, float)
        self.assertGreaterEqual(sample, 0.0)

        samples = rv.sample(n=100)
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.shape, (100,))
        self.assertGreaterEqual(min(samples), 0.0)

        self.assertAlmostEqual(rv.get_mean(exact=True), shape * scale)
        self.assertAlmostEqual(rv.get_variance(exact=True), shape * scale**2)
        self.assertAlmostEqual(rv.cdf(value=0.0, exact=True), 0.0)
        self.assertAlmostEqual(rv.cdf(value=shape * scale, exact=True), 0.5939941502901615)

    def test_dirac_delta_rv(self) -> None:
        value = 42.0

        rv = make_dirac(value=value)

        sample = rv.sample()
        self.assertEqual(sample, value)

        self.assertEqual(rv.get_mean(exact=True), value)
        self.assertAlmostEqual(rv.get_variance(exact=True), 0.0)
        self.assertEqual(rv.cdf(value=value - 1e-9, exact=True), 0.0)
        self.assertEqual(rv.cdf(value=value, exact=True), 1.0)
        self.assertEqual(rv.cdf(value=value + 1e-9, exact=True), 1.0)

    def test_discrete_rv(self) -> None:
        values = [1, 2, 3]
        probabilities = [0.2, 0.5, 0.3]

        rv = make_discrete(values=values, probabilities=probabilities)

        sample = rv.sample(n=100)
        self.assertIsInstance(sample, np.ndarray)
        self.assertEqual(sample.shape, (100,))
        self.assertTrue(np.all(np.isin(sample, values)))

        self.assertAlmostEqual(rv.get_mean(exact=True), sum(v * p for v, p in zip(values, probabilities)))
        self.assertAlmostEqual(
            rv.get_variance(exact=True),
            sum(p * (v - rv.get_mean(exact=True)) ** 2 for v, p in zip(values, probabilities)),
        )
