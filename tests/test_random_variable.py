from unittest import TestCase

import numpy as np

from randcraft import make_anon, make_dirac, make_discrete, make_normal, make_uniform, make_dice_roll, make_coin_flip
from randcraft.pdfs import (
    DiracDeltaDistributionFunction,
    DiscreteDistributionFunction,
    MixtureDistributionFunction,
)
from randcraft.random_variable import RandomVariable


class TestRandomVariable(TestCase):
    def test_normal_rv(self) -> None:
        mean = 23.4
        std_dev = 2.1

        rv = make_normal(mean=mean, std_dev=std_dev)

        sample = rv.sample_one()
        self.assertIsInstance(sample, float)

        self.assertEqual(rv.get_mean(exact=True), mean)
        self.assertAlmostEqual(rv.get_variance(exact=True), std_dev**2)
        self.assertAlmostEqual(rv.get_chance_that_rv_is_le(value=mean - std_dev, exact=True), 0.15865525393145707)
        self.assertAlmostEqual(rv.get_chance_that_rv_is_le(value=mean, exact=True), 0.5)
        self.assertAlmostEqual(rv.get_chance_that_rv_is_le(value=mean + std_dev, exact=True), 0.8413447460685429)

    def test_uniform_rv(self) -> None:
        low = 10.0
        high = 20.0

        rv = make_uniform(low=low, high=high)

        sample = rv.sample_one()
        self.assertIsInstance(sample, float)

        samples = rv.sample_numpy(n=100)
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.shape, (100,))
        self.assertGreaterEqual(min(samples), low)
        self.assertLessEqual(max(samples), high)

        self.assertEqual(rv.get_mean(exact=True), (low + high) / 2)
        self.assertAlmostEqual(rv.get_variance(exact=True), ((high - low) ** 2) / 12)
        self.assertAlmostEqual(rv.get_chance_that_rv_is_le(value=low, exact=True), 0.0)
        self.assertAlmostEqual(rv.get_chance_that_rv_is_le(value=15.0, exact=True), 0.5)
        self.assertAlmostEqual(rv.get_chance_that_rv_is_le(value=high, exact=True), 1.0)

    def test_dirac_delta_rv(self) -> None:
        value = 42.0

        rv = make_dirac(value=value)

        sample = rv.sample_one()
        self.assertEqual(sample, value)

        self.assertEqual(rv.get_mean(exact=True), value)
        self.assertAlmostEqual(rv.get_variance(exact=True), 0.0)
        self.assertEqual(rv.get_chance_that_rv_is_le(value=value - 1e-9, exact=True), 0.0)
        self.assertEqual(rv.get_chance_that_rv_is_le(value=value, exact=True), 1.0)
        self.assertEqual(rv.get_chance_that_rv_is_le(value=value + 1e-9, exact=True), 1.0)

    def test_discrete_rv(self) -> None:
        values = [1, 2, 3]
        probabilities = [0.2, 0.5, 0.3]

        rv = make_discrete(values=values, probabilities=probabilities)

        sample = rv.sample_numpy(n=100)
        self.assertIsInstance(sample, np.ndarray)
        self.assertEqual(sample.shape, (100,))
        self.assertTrue(np.all(np.isin(sample, values)))

        self.assertAlmostEqual(rv.get_mean(exact=True), sum(v * p for v, p in zip(values, probabilities)))
        self.assertAlmostEqual(
            rv.get_variance(exact=True),
            sum(p * (v - rv.get_mean(exact=True)) ** 2 for v, p in zip(values, probabilities)),
        )

    def test_empirical_rv(self) -> None:
        mean = 23.4

        rv = make_anon(sampler=lambda n: np.ones(n) * mean)

        sample = rv.sample_one()
        self.assertEqual(sample, mean)

        self.assertAlmostEqual(rv.get_mean(), mean)
        self.assertAlmostEqual(rv.get_variance(), 0.0)
        self.assertEqual(rv.get_chance_that_rv_is_le(value=mean - 1e-9), 0.0)
        self.assertEqual(rv.get_chance_that_rv_is_le(value=mean), 1.0)
        self.assertEqual(rv.get_chance_that_rv_is_le(value=mean + 1e-9), 1.0)

        self.assertRaises(NotImplementedError, lambda: rv.get_mean(exact=True))
        self.assertRaises(NotImplementedError, lambda: rv.get_variance(exact=True))

    def test_mixture_rv(self) -> None:
        rv_a = make_dirac(1.0)
        rv_b = make_normal(mean=2.0, std_dev=2.0)
        mixture = RandomVariable.mix_rvs([rv_a, rv_b])

        self.assertIsInstance(mixture, RandomVariable)
        self.assertIsInstance(mixture.pdf, MixtureDistributionFunction)
        self.assertAlmostEqual(mixture.get_mean(exact=True), (1.0 + 2.0) / 2)
        # TODO Test variance of mixture, test using different weights

    def test_combining_different_types_of_rvs(self) -> None:
        rv1 = make_normal(mean=10, std_dev=2)
        rv2 = make_uniform(low=80, high=120)
        rv3 = make_anon(sampler=lambda n: np.ones(n) * 15.0)

        partial_combined_rv = rv1 + rv2
        self.assertIsInstance(partial_combined_rv, RandomVariable)
        self.assertAlmostEqual(partial_combined_rv.get_mean(exact=True), 10 + 100)
        self.assertAlmostEqual(partial_combined_rv.get_variance(exact=True), rv1.get_variance() + rv2.get_variance())

        combined_rv = rv1 + rv2 + rv3

        self.assertIsInstance(combined_rv, RandomVariable)
        self.assertAlmostEqual(combined_rv.get_mean(), 10 + 100 + 15)
        self.assertAlmostEqual(combined_rv.get_variance(), rv1.get_variance() + rv2.get_variance())

        self.assertRaises(NotImplementedError, lambda: rv3.get_mean(exact=True))
        self.assertRaises(NotImplementedError, lambda: rv3.get_variance(exact=True))

    def test_pdf_combination(self) -> None:
        rv1 = make_uniform(low=0, high=10)
        rv2 = make_dirac(value=10)
        combined = rv1 + rv2
        self.assertEqual(combined.get_chance_that_rv_is_le(value=10), 0.0)
        self.assertEqual(combined.get_chance_that_rv_is_le(value=20), 1.0)

    def test_normal_known_convolutions(self) -> None:
        rv1 = make_normal(mean=1, std_dev=4)
        rv2 = make_normal(mean=2, std_dev=1)
        dirac1 = make_dirac(value=1)
        combined = rv1 + rv2

        self.assertIsInstance(combined, RandomVariable)
        self.assertAlmostEqual(combined.get_mean(exact=True), 3)
        self.assertAlmostEqual(combined.get_variance(exact=True), 17)

        offset = combined + dirac1
        self.assertIsInstance(offset, RandomVariable)
        self.assertAlmostEqual(offset.get_mean(exact=True), 4)
        self.assertAlmostEqual(offset.get_variance(exact=True), 17)

    def test_normal_arithmetic(self) -> None:
        rv1 = make_normal(mean=5, std_dev=1)
        rv2 = make_normal(mean=3, std_dev=1)

        # Test addition
        rv_add = rv1 + rv2
        self.assertIsInstance(rv_add, RandomVariable)
        self.assertAlmostEqual(rv_add.get_mean(), 5 + 3)
        self.assertAlmostEqual(rv_add.get_variance(), rv1.get_variance() + rv2.get_variance())

        # Test subtraction
        rv_sub = rv1 - rv2
        self.assertIsInstance(rv_sub, RandomVariable)
        self.assertAlmostEqual(rv_sub.get_mean(), 5 - 3)
        self.assertAlmostEqual(rv_sub.get_variance(), rv1.get_variance() + rv2.get_variance())

        # Test multiplication
        rv_neg = rv1 * -1
        self.assertIsInstance(rv_neg, RandomVariable)
        self.assertAlmostEqual(rv_neg.get_mean(), -5)
        self.assertAlmostEqual(rv_neg.get_variance(), rv1.get_variance())

        # Test division
        rv_div = rv1 / 2
        self.assertIsInstance(rv_div, RandomVariable)
        self.assertAlmostEqual(rv_div.get_mean(), 5 / 2)
        self.assertAlmostEqual(rv_div.get_variance(), rv1.get_variance() / 4)

        rv3 = make_anon(sampler=lambda n: np.ones(n) * 15.0)

        rv3_double = rv3 * 2
        self.assertIsInstance(rv3_double, RandomVariable)
        self.assertAlmostEqual(rv3_double.get_mean(), 30.0)
        self.assertAlmostEqual(rv3_double.get_variance(), 0.0)

        # Test scale by zero
        rv_zero_scale = rv1 * 0
        self.assertIsInstance(rv_zero_scale, RandomVariable)
        self.assertAlmostEqual(rv_zero_scale.get_mean(), 0.0)
        self.assertAlmostEqual(rv_zero_scale.get_variance(), 0.0)
        self.assertIsInstance(rv_zero_scale.pdf, DiracDeltaDistributionFunction)

    def test_uniform_known_convolutions(self) -> None:
        rv1 = make_uniform(low=0, high=10)
        dirac1 = make_dirac(value=1)
        offset = rv1 + dirac1

        self.assertIsInstance(offset, RandomVariable)
        self.assertAlmostEqual(offset.get_mean(exact=True), 6)
        self.assertAlmostEqual(offset.get_variance(exact=True), rv1.get_variance())

    def test_uniform_multiplications(self) -> None:
        rv1 = make_uniform(low=0, high=10)

        double_rv1 = rv1 * 2
        self.assertIsInstance(double_rv1, RandomVariable)
        self.assertAlmostEqual(double_rv1.get_mean(exact=True), 10)
        self.assertAlmostEqual(double_rv1.get_variance(exact=True), rv1.get_variance() * 4)

        negative_rv1 = rv1 * -1
        self.assertIsInstance(negative_rv1, RandomVariable)
        self.assertAlmostEqual(negative_rv1.get_mean(exact=True), -5)
        self.assertAlmostEqual(negative_rv1.get_variance(exact=True), rv1.get_variance())

    def test_discrete_convolution(self) -> None:
        dice = make_discrete(values=[1, 2, 3, 4, 5, 6])
        two_dice = dice + dice
        self.assertIsInstance(two_dice, RandomVariable)
        self.assertIsInstance(two_dice.pdf, DiscreteDistributionFunction)
        self.assertAlmostEqual(two_dice.get_mean(exact=True), 7.0)

        one = make_dirac(value=1)
        two_dice_and_one = dice * 2 + one
        self.assertIsInstance(two_dice_and_one, RandomVariable)
        self.assertIsInstance(two_dice_and_one.pdf, DiscreteDistributionFunction)
        self.assertAlmostEqual(two_dice_and_one.get_mean(exact=True), 8.0)

    def test_discrete_and_continuous_convolution(self) -> None:
        coin_flip = make_coin_flip()
        norm = make_normal(mean=0, std_dev=0.2)
        combined = coin_flip + norm
        print(combined)
        self.assertEqual(combined.get_variance(), coin_flip.get_variance() + norm.get_variance())
        self.assertEqual(combined.get_mean(), 0.5)

    def test_plotting(self) -> None:
        rv1 = make_normal(mean=0, std_dev=1)
        rv2 = make_uniform(low=-1, high=1)
        combined = rv1 + rv2

        rv1.pdf.plot()
        rv2.pdf.plot()
        combined.pdf.plot()

        discrete = make_discrete(values=[1, 2, 3])
        discrete.pdf.plot()

        mixed = RandomVariable.mix_rvs([rv1, rv2, combined, discrete])
        mixed.pdf.plot()

    def test_add_special_event(self) -> None:
        rv = make_dirac(value=1.0)
        new_rv = rv.add_special_event(value=2.0, chance=0.5)

        self.assertIsInstance(new_rv, RandomVariable)
        self.assertAlmostEqual(new_rv.get_mean(), 1.5)
