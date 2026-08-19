from __future__ import annotations

import unittest

import numpy as np

from cryoml.noise_aware import (balanced_regime_objective,
                                fit_curve_noise_model,
                                noise_aware_curve_loss,
                                reported_bias_rrms,
                                segmented_curve_loss,
                                segment_rrms_within_budget,
                                within_noise_aware_budget,
                                within_tradeoff_budget)


class NoiseAwareLossTests(unittest.TestCase):
    def test_below_detection_prediction_is_censored_not_floor_fitted(self):
        x = np.linspace(0.0, 1.85, 186)
        measured = np.where(x < 0.8, 5e-8, 1e-6 * (x - 0.8) ** 2)
        model = fit_curve_noise_model("idvg", x, measured)
        below = noise_aware_curve_loss(
            np.where(x < 0.8, 0.0, measured), model)
        above = noise_aware_curve_loss(
            np.where(x < 0.8, 5e-6, measured), model)
        self.assertLess(below["total"], above["total"])
        self.assertGreater(model.detection_limit, 0.0)

    def test_shape_term_detects_wrong_upper_sweep_slope(self):
        x = np.linspace(0.0, 1.85, 186)
        measured = 1e-3 * np.maximum(x - 0.5, 0.0) ** 2
        model = fit_curve_noise_model("idvg", x, measured)
        correct = noise_aware_curve_loss(measured, model, shape_weight=0.1)
        wrong = noise_aware_curve_loss(
            1e-3 * np.maximum(x - 0.5, 0.0), model, shape_weight=0.1)
        self.assertLess(correct["total"], wrong["total"])
        self.assertLess(correct["shape"], wrong["shape"])

    def test_regime_balance_uses_general_curve_labels(self):
        result = balanced_regime_objective({
            "idvd@0.74": 0.01, "idvd@1.85": 0.02,
            "idvg@0.74": 0.03, "idvg@1.85": 0.04,
        }, tail_weight=0.0)
        self.assertEqual(set(result["groups"]), {
            "idvd_ordinary", "idvd_high", "idvg_ordinary", "idvg_high",
        })
        self.assertTrue(np.isfinite(result["objective"]))

    @staticmethod
    def _paper_score(overall=0.10, ordinary=0.10, high=0.10):
        return {
            "rrms": overall,
            "high_voltage_rrms": high,
            "per_curve": {
                "idvd@0.74": {"included": True, "rrms": ordinary},
                "idvd@1.85": {"included": True, "rrms": high},
            },
        }

    def test_reported_bias_rrms_separates_ordinary_and_high(self):
        score = self._paper_score(ordinary=0.12, high=0.04)
        self.assertEqual(
            reported_bias_rrms(score["per_curve"], high=False), 0.12)
        self.assertEqual(
            reported_bias_rrms(score["per_curve"], high=True), 0.04)

    def test_tradeoff_budget_allows_small_balanced_sacrifice(self):
        reference = self._paper_score()
        candidate = self._paper_score(
            overall=0.104, ordinary=0.104, high=0.095)
        self.assertTrue(within_tradeoff_budget(candidate, reference))

    def test_tradeoff_budget_rejects_hidden_curve_collapse(self):
        reference = self._paper_score()
        candidate = self._paper_score(
            overall=0.104, ordinary=0.30, high=0.04)
        self.assertFalse(within_tradeoff_budget(candidate, reference))

    def test_segmented_curve_loss_exposes_early_sweep_error(self):
        x = np.linspace(0.0, 1.0, 101)
        measured = 1e-3 * x
        model = fit_curve_noise_model("idvd", x, measured)
        prediction = measured.copy()
        prediction[x < 0.2] += 2e-4
        result = segmented_curve_loss(prediction, model)
        self.assertGreater(result["segments"]["s0_0_0.2"]["reported_rrms"],
                           result["segments"]["s3_0.7_1"]["reported_rrms"])

    def test_segment_guard_uses_best_incumbent_per_segment(self):
        def segment(value):
            return {"curve": {"s0": {"reported_rrms": value}}}
        self.assertTrue(segment_rrms_within_budget(
            segment(0.11), [segment(0.10), segment(0.20)]))
        self.assertFalse(segment_rrms_within_budget(
            segment(0.14), [segment(0.10), segment(0.20)],
            absolute_budget=0.02))


if __name__ == "__main__":
    unittest.main()


class GuardFieldTests(unittest.TestCase):
    """The segment guard must veto on the quantity it is budgeted for."""

    def _segments(self, total, reported):
        return {"idvg@0.01": {"s0_0_0.2": {
            "total": total, "reported_rrms": reported}}}

    def test_raw_field_vetoes_a_censored_curve_regression(self):
        anchor = self._segments(0.0008, 0.09)
        candidate = self._segments(0.0009, 0.30)
        self.assertFalse(segment_rrms_within_budget(
            candidate, [anchor], relative_budget=0.25,
            absolute_budget=0.02, field="reported_rrms"))

    def test_noise_aware_field_admits_the_same_candidate(self):
        anchor = self._segments(0.0008, 0.09)
        candidate = self._segments(0.0009, 0.30)
        self.assertTrue(segment_rrms_within_budget(
            candidate, [anchor], relative_budget=0.25,
            absolute_budget=0.002, field="total"))

    def test_noise_aware_field_still_vetoes_a_real_regression(self):
        anchor = self._segments(0.0008, 0.09)
        candidate = self._segments(0.02, 0.30)
        self.assertFalse(segment_rrms_within_budget(
            candidate, [anchor], relative_budget=0.25,
            absolute_budget=0.002, field="total"))


class NoiseAwareBudgetTests(unittest.TestCase):
    def _score(self, objective, ordinary, high, curve_total):
        return {
            "objective": objective,
            "groups": {"idvd_ordinary": ordinary, "idvg_ordinary": ordinary,
                       "idvd_high": high, "idvg_high": high},
            "curve_components": {"idvg@0.01": {"total": curve_total}},
        }

    def test_admits_a_candidate_within_budget(self):
        self.assertTrue(within_noise_aware_budget(
            self._score(0.0241, 0.02, 0.03, 0.0009),
            self._score(0.0240, 0.02, 0.03, 0.0008)))

    def test_rejects_a_real_curve_collapse(self):
        self.assertFalse(within_noise_aware_budget(
            self._score(0.0241, 0.02, 0.03, 0.05),
            self._score(0.0240, 0.02, 0.03, 0.0008)))

    def test_rejects_an_aggregate_regression(self):
        self.assertFalse(within_noise_aware_budget(
            self._score(0.9, 0.02, 0.03, 0.0009),
            self._score(0.0240, 0.02, 0.03, 0.0008)))

    def test_missing_regime_group_is_an_error(self):
        broken = {"objective": 0.02, "groups": {"idvd_ordinary": 0.02},
                  "curve_components": {}}
        with self.assertRaises(ValueError):
            within_noise_aware_budget(broken, broken)
