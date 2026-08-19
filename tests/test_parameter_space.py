from __future__ import annotations

import unittest

import numpy as np

from cryoml.parameter_space import ParameterBox


class ParameterBoxTests(unittest.TestCase):
    def setUp(self):
        self.box = ParameterBox(
            names=("a", "b", "c"),
            published={"a": 2.0, "b": 5.0, "c": -1.0},
            bounds={"a": [1.0, 3.0], "b": [5.0, 5.0],
                    "c": [-2.0, 0.0]})

    def test_inactive_parameter_is_fixed_at_published_value(self):
        values = self.box.local_to_array(np.asarray([1.0, -1.0, -1.0]))
        np.testing.assert_allclose(values, [3.0, 5.0, -2.0])
        self.assertEqual(self.box.n_active, 2)

    def test_published_round_trip(self):
        params = self.box.local_to_params(self.box.local_published)
        for name, value in self.box.published.items():
            self.assertAlmostEqual(params[name], value)


if __name__ == "__main__":
    unittest.main()
