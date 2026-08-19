from __future__ import annotations

import unittest

import numpy as np
import torch

from cryoml.foundation import (CANONICAL_CURVE_TAGS, FoundationSurrogate,
                               InverseAutoencoder, canonical_point_order,
                               hybrid_forward_loss, segmented_hybrid_loss,
                               voltage_point_weights)


class ExpandedFoundationTests(unittest.TestCase):
    def test_curve_order_is_canonical_even_when_source_is_permuted(self):
        kinds = np.asarray(["idvg", "idvd"])
        fixeds = np.asarray([0.01, 0.37])
        # Supply only the first two canonical tags by temporarily verifying the
        # full routine with all tags in reverse order.
        source = list(reversed(CANONICAL_CURVE_TAGS))
        kinds = np.asarray([kind for kind, _bias in source])
        fixeds = np.asarray([bias for _kind, bias in source])
        slices = np.asarray([(2 * index, 2 * index + 2)
                             for index in range(len(source))])
        order, descriptions = canonical_point_order(kinds, fixeds, slices)
        first_source = len(source) - 1
        np.testing.assert_array_equal(
            order[:2], np.asarray([2 * first_source, 2 * first_source + 1]))
        self.assertEqual(descriptions[0]["tag"], "idvd@0.37")

    def test_models_support_arbitrary_parameter_count(self):
        surrogate = FoundationSurrogate(43, 22, (32, 16))
        prediction = surrogate(torch.zeros((4, 46)))
        self.assertEqual(tuple(prediction.shape), (4, 22))
        inverse = InverseAutoencoder(22, 43, (32, 16))
        local = inverse(torch.zeros((4, 22)), torch.zeros((4, 3)))
        self.assertEqual(tuple(local.shape), (4, 43))
        self.assertTrue(torch.all(local <= 1) and torch.all(local >= -1))
        dual_inverse = InverseAutoencoder(44, 43, (32, 16))
        dual_local = dual_inverse(
            torch.zeros((4, 44)), torch.zeros((4, 3)))
        self.assertEqual(tuple(dual_local.shape), (4, 43))

    def test_high_voltage_training_weights_are_larger(self):
        descriptions = []
        cursor = 0
        for kind, bias in CANONICAL_CURVE_TAGS:
            descriptions.append({"kind": kind, "bias": bias,
                                 "start": cursor, "stop": cursor + 2})
            cursor += 2
        weights = voltage_point_weights(
            descriptions, cursor, emphasize_high_voltage=True)
        ordinary = descriptions[0]
        strongest = next(row for row in descriptions if row["bias"] == 1.85)
        self.assertGreater(weights[strongest["start"]],
                           weights[ordinary["start"]])

    def test_controlled_comparison_is_uniform_by_default(self):
        descriptions = [{"kind": "idvd", "bias": 0.37,
                         "start": 0, "stop": 2},
                        {"kind": "idvd", "bias": 1.85,
                         "start": 2, "stop": 4}]
        np.testing.assert_array_equal(
            voltage_point_weights(descriptions, 4), np.ones(4))

    def test_hybrid_loss_stays_finite_for_extreme_early_prediction(self):
        prediction = torch.full((2, 4), 100.0)
        target = torch.zeros((2, 4))
        current = torch.zeros((2, 4))
        device_ids = torch.zeros(2, dtype=torch.long)
        mean = torch.zeros(4)
        std = torch.ones(4)
        scales = torch.full((1, 4), 1e-9)
        weights = torch.ones((1, 4))
        loss, components = hybrid_forward_loss(
            prediction, target, current, device_ids, mean, std,
            scales, weights)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(components["linear"]))

    def test_segmented_loss_reports_concentrated_sweep_error(self):
        prediction = torch.zeros((1, 10))
        prediction[:, :2] = 1.0
        target = torch.zeros((1, 10))
        current = torch.zeros((1, 10))
        device_ids = torch.zeros(1, dtype=torch.long)
        loss, components = segmented_hybrid_loss(
            prediction, target, current, device_ids,
            torch.zeros(10), torch.ones(10), torch.ones((1, 10)),
            [{"start": 0, "stop": 10}], linear_weight=0.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(components["worst_segment"]),
                           float(components["mean_segment"]))


if __name__ == "__main__":
    unittest.main()
