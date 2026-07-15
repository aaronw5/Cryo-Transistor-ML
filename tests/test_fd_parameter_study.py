from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fd_parameter_study import absolute_two_point_jacobian  # noqa: E402


class FiniteDifferenceStudyTests(unittest.TestCase):
    def test_absolute_jacobian_is_effective_at_zero(self):
        matrix = np.array([[2.0, -3.0], [0.5, 4.0]])
        jacobian = absolute_two_point_jacobian(
            lambda z: matrix @ z, np.zeros(2), step=2e-2)
        np.testing.assert_allclose(jacobian, matrix, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
