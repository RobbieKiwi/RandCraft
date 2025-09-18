from unittest import TestCase

from randcraft import make_discrete, make_normal, make_uniform
from randcraft.constructors import make_beta
from randcraft.misc import mix_rvs


class TestPlotting(TestCase):
    def test_plotting_convolved_uniform(self) -> None:
        rv_base = make_uniform(low=-1, high=1)
        rv = rv_base + rv_base
        rv._rv.plot()

        discrete = make_discrete(values=[0, 6])
        new_rv = rv + discrete
        new_rv._rv.plot()

    def test_plotting_mixed(self) -> None:
        rv1 = make_normal(mean=0, std_dev=1)
        rv2 = make_uniform(low=-1, high=1)
        combined = rv1 + rv2
        discrete = make_discrete(values=[1, 2, 3])
        mixed = mix_rvs([rv1, rv2, combined, discrete])
        mixed._rv.plot()

    def test_scaled_plotting(self) -> None:
        a = 2.0
        b = 5.0

        rv = make_beta(a=a, b=b)
        rv._rv.plot()

        new_rv = rv * -1 - 2
        new_rv._rv.plot()
