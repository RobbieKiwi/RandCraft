from unittest import TestCase

from randcraft.models import Statistics


class BaseTestCase(TestCase):
    def assert_stats_are_close(self, a: Statistics, b: Statistics, places: int = 5) -> None:
        moments_to_check = 2
        for k in range(moments_to_check):
            self.assertAlmostEqual(a.moments[k].value, b.moments[k].value, places=places)
        self.assertAlmostEqual(a.min_value.value, b.min_value.value, places=places)
        self.assertAlmostEqual(a.max_value.value, b.max_value.value, places=places)
