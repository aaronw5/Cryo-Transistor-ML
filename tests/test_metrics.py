import unittest

import numpy as np

from cryoml.metrics import device_rrms


class PaperExactMetricTests(unittest.TestCase):
    def test_includes_zero_denominator_curve_as_zero_rrms(self):
        sims = [np.array([5.0, 5.0]), np.array([2.0, 4.0])]
        meas = [np.array([0.0, 0.0]), np.array([1.0, 3.0])]

        result = device_rrms(sims, meas)

        self.assertEqual(result["n_curves"], 2)
        self.assertAlmostEqual(result["rrms"], 0.25)


if __name__ == "__main__":
    unittest.main()
