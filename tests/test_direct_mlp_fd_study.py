from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from direct_mlp_fd_study import family_means  # noqa: E402


class DirectMlpFiniteDifferenceStudyTests(unittest.TestCase):
    def test_family_means_preserve_reported_aggregation(self):
        rows = [
            {"dev_type": "nmos", "score": 0.1},
            {"dev_type": "nmos", "score": 0.3},
            {"dev_type": "pmos", "score": 0.5},
        ]
        means = family_means(rows, "score")
        self.assertAlmostEqual(means["nmos_mean"], 0.2)
        self.assertAlmostEqual(means["pmos_mean"], 0.5)
        self.assertAlmostEqual(means["all_device_mean"], 0.3)
        self.assertAlmostEqual(means["combined"], 0.35)


if __name__ == "__main__":
    unittest.main()
