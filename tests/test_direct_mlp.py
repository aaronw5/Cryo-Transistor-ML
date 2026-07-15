from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pdk_direct_mlp import (DirectParameterMLP, fit_minmax, make_features,  # noqa: E402
                            sampled_curve_indices, signed_log)


class DirectMlpTests(unittest.TestCase):
    def test_feature_layout_has_301_unique_points_across_every_curve(self):
        slices = np.array([(i * 186, (i + 1) * 186) for i in range(11)])
        indices = sampled_curve_indices(slices)
        self.assertEqual(indices.shape, (301,))
        self.assertEqual(len(np.unique(indices)), 301)
        for start, stop in slices:
            self.assertTrue(np.any((indices >= start) & (indices < stop)))

    def test_linear_and_signed_log_features_form_602_inputs(self):
        slices = np.array([(i * 186, (i + 1) * 186) for i in range(11)])
        indices = sampled_curve_indices(slices)
        currents = np.linspace(-1e-3, 1e-3, 4 * 2046).reshape(4, 2046)
        train = currents[:3, indices]
        linear_lo, linear_span = fit_minmax(train)
        log_lo, log_span = fit_minmax(signed_log(train))
        features = make_features(currents, indices, linear_lo, linear_span,
                                 log_lo, log_span)
        self.assertEqual(features.shape, (4, 602))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all((features >= 0.0) & (features <= 1.0)))

    def test_model_forward_pass_predicts_seven_parameters(self):
        model = DirectParameterMLP(602, (32, 16), dropout=0.0)
        output = model(torch.zeros((5, 602), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (5, 7))


if __name__ == "__main__":
    unittest.main()
