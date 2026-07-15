from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pdk_foundation_emulator import (FoundationEmulator, geometry_features,  # noqa: E402
                                     geometry_ranges)
from cryoml.devices import PAPER_DEVICES  # noqa: E402


class FoundationEmulatorTests(unittest.TestCase):
    def test_geometry_features_are_three_bounded_conditioners(self):
        ranges = geometry_ranges()
        for device in PAPER_DEVICES:
            features = geometry_features(device, ranges)
            self.assertEqual(features.shape, (3,))
            self.assertTrue((features >= -1.0).all())
            self.assertTrue((features <= 1.0).all())

    def test_model_maps_ten_inputs_to_full_current_vector(self):
        model = FoundationEmulator(2046, (32, 16))
        output = model(torch.zeros((4, 10), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (4, 2046))


if __name__ == "__main__":
    unittest.main()
