import numpy as np

from randcraft.constructors import make_beta, make_coin_flip, make_die_roll, make_dirac, make_discrete, make_normal, make_uniform
from randcraft.random_variable import RandomVariable
from randcraft.rvs import DiracDeltaRV, DiscreteRV
from randcraft.rvs.continuous import ContinuousRV
from tests.base_test_case import BaseTestCase


class TestCombiningRvs(BaseTestCase):
    def test_scaled_rv(self) -> None:
        a = 2.0
        b = 5.0

        rv = make_beta(a=a, b=b)

        scaled_rv = rv.scale(factor=1)
        self.assert_stats_are_close(a=scaled_rv.statistics, b=rv.statistics)

        offset_rv = scaled_rv - 1
        self.assertAlmostEqual(offset_rv.get_mean(), rv.get_mean() - 1)

        negative_rv = rv.scale(factor=-1)
        new_rv = negative_rv - 2
        self.assertAlmostEqual(new_rv.get_mean(), rv.get_mean() * -1 - 2)
        self.assertAlmostEqual(new_rv.get_variance(), rv.get_variance())

    def test_combining_different_types_of_rvs(self) -> None:
        rv1 = make_normal(mean=10, std_dev=2)
        rv2 = make_uniform(low=80, high=120)

        partial_combined_rv = rv1 + rv2
        self.assertIsInstance(partial_combined_rv, RandomVariable)
        self.assertAlmostEqual(partial_combined_rv.get_mean(exact=True), 10 + 100)
        self.assertAlmostEqual(partial_combined_rv.get_variance(exact=True), rv1.get_variance() + rv2.get_variance())

    def test_pdf_combination(self) -> None:
        rv1 = make_uniform(low=0, high=10)
        rv2 = make_dirac(value=10)
        combined = rv1 + rv2
        self.assertEqual(combined.cdf(value=10), 0.0)
        self.assertEqual(combined.cdf(value=20), 1.0)

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

        # Test scaling
        rv_neg = rv1.scale(-1)
        self.assertIsInstance(rv_neg, RandomVariable)
        self.assertAlmostEqual(rv_neg.get_mean(), -5)
        self.assertAlmostEqual(rv_neg.get_variance(), rv1.get_variance())

        rv_div = rv1.scale(1 / 2)
        self.assertIsInstance(rv_div, RandomVariable)
        self.assertAlmostEqual(rv_div.get_mean(), 5 / 2)
        self.assertAlmostEqual(rv_div.get_variance(), rv1.get_variance() / 4)

        # Test scale by zero
        rv_zero_scale = rv1.scale(0)
        self.assertIsInstance(rv_zero_scale, RandomVariable)
        self.assertAlmostEqual(rv_zero_scale.get_mean(), 0.0)
        self.assertAlmostEqual(rv_zero_scale.get_variance(), 0.0)
        self.assertIsInstance(rv_zero_scale._rv, DiracDeltaRV)

    def test_uniform_known_convolutions(self) -> None:
        rv1 = make_uniform(low=0, high=10)
        dirac1 = make_dirac(value=1)
        offset = rv1 + dirac1

        self.assertIsInstance(offset, RandomVariable)
        self.assertAlmostEqual(offset.get_mean(exact=True), 6)
        self.assertAlmostEqual(offset.get_variance(exact=True), rv1.get_variance())

    def test_uniform_convolution_double(self) -> None:
        rv1 = make_uniform(low=0, high=2)
        joined = rv1.multi_sample(2)
        inner_rv: ContinuousRV = joined._rv  # type: ignore
        values = np.array([0, 1, 2, 3, 4])
        pdf = inner_rv.calculate_pdf(x=values)
        expected_values = [0, 0.25, 0.5, 0.25, 0.0]
        for v, ev in zip(pdf.y, expected_values):
            self.assertAlmostEqual(v, ev, places=3)

    def test_uniform_convolution_triple(self) -> None:
        rv1 = make_uniform(low=0, high=2)
        joined = rv1.multi_sample(3)
        inner_rv: ContinuousRV = joined._rv  # type: ignore
        values = np.array([0, 1, 2, 3, 4, 5, 6])
        pdf = inner_rv.calculate_pdf(x=values)
        self.assertAlmostEqual(pdf.y[0], 0.0, places=3)
        self.assertAlmostEqual(pdf.y[-1], 0.0, places=3)
        self.assertEqual(np.argmax(pdf.y), 3)

    def test_uniform_multiplications(self) -> None:
        rv1 = make_uniform(low=0, high=10)

        double_rv1 = rv1.scale(2)
        self.assertIsInstance(double_rv1, RandomVariable)
        self.assertAlmostEqual(double_rv1.get_mean(exact=True), 10)
        self.assertAlmostEqual(double_rv1.get_variance(exact=True), rv1.get_variance() * 4)

        negative_rv1 = rv1.scale(-1)
        self.assertIsInstance(negative_rv1, RandomVariable)
        self.assertAlmostEqual(negative_rv1.get_mean(exact=True), -5)
        self.assertAlmostEqual(negative_rv1.get_variance(exact=True), rv1.get_variance())

    def test_discrete_convolution(self) -> None:
        dice = make_discrete(values=[1, 2, 3, 4, 5, 6])
        two_dice = dice + dice
        self.assertIsInstance(two_dice, RandomVariable)
        self.assertIsInstance(two_dice._rv, DiscreteRV)
        self.assertAlmostEqual(two_dice.get_mean(exact=True), 7.0)

        one = make_dirac(value=1)
        two_dice_and_one = two_dice + one
        self.assertIsInstance(two_dice_and_one, RandomVariable)
        self.assertIsInstance(two_dice_and_one._rv, DiscreteRV)
        self.assertAlmostEqual(two_dice_and_one.get_mean(exact=True), 8.0)

    def test_discrete_and_continuous_convolution(self) -> None:
        coin_flip = make_coin_flip()
        norm = make_normal(mean=0, std_dev=0.2)
        combined = coin_flip + norm
        print(combined)
        self.assertEqual(combined.get_variance(), coin_flip.get_variance() + norm.get_variance())
        self.assertEqual(combined.get_mean(), 0.5)

    def test_three_dice(self) -> None:
        dice = make_die_roll(sides=6)
        three_dice = dice + dice + dice
        result = three_dice.cdf(10.0)
        self.assertEqual(result, 0.5)

    def test_complex_convolution(self) -> None:
        rv_base = make_uniform(low=-1, high=1)
        rv = rv_base + rv_base
        discrete = make_discrete(values=[0, 6])
        new_rv = rv + discrete
        self.assertAlmostEqual(new_rv.cdf(-2.0), 0.0, places=3)
        self.assertAlmostEqual(new_rv.cdf(0.0), 0.25, places=3)
        self.assertAlmostEqual(new_rv.cdf(3.0), 0.5, places=3)
        self.assertAlmostEqual(new_rv.cdf(6.0), 0.75, places=3)
        self.assertAlmostEqual(new_rv.cdf(8.0), 1.0, places=3)
