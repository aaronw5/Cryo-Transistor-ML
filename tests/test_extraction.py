"""cryoml.extraction: configuration, range rule, start draws, budgets, start
selection.  No NGSpice run; stored I25-1 specs are used where present."""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryoml import extraction as ex  # noqa: E402
from cryoml.cold_start import MIN_HALF_WIDTH, next_round_spec  # noqa: E402
from cryoml.ngspice_fit import trf_pass_budget  # noqa: E402
from cryoml.parameter_sets import PARAMS7, PARAMS45  # noqa: E402
from cryoml.parameter_space import ParameterBox  # noqa: E402

I25 = ROOT / "out/research/explore/I25_1"


class FakeProblem:
    """``simulate`` returns one curve; the first ``n_fail`` calls have a NaN
    inside the scored part (index >= 1), later calls only before it."""

    def __init__(self, box: ParameterBox, n_fail: int = 0):
        self.tag, self.box, self.n_fail, self.n_sim = "fake", box, n_fail, 0
        self.layout = [{"index": 0, "start": 1}]

    def simulate(self, local):
        self.n_sim += 1
        if self.n_sim <= self.n_fail:
            return [np.array([1.0, np.nan, 2.0])]
        return [np.array([np.nan, 1.0, 2.0])]

    def objective(self, local):
        self.n_sim += 1
        x = np.asarray(local)[self.box.active]
        return float(np.sum((x - 0.3) ** 2))


def fake_box() -> ParameterBox:
    return ParameterBox(names=("a", "b", "c"), published={"a": 1.0, "b": 2.0, "c": 0.0},
                        bounds={"a": [0.5, 2.0], "b": [1.0, 4.0], "c": [0.0, 0.0]},
                        log_scale=(True, True, False))


def start_record(k: int, rrms: float) -> ex.StartRecord:
    stage = ex.StageRecord("round 3 + FD", rrms, rrms, rrms, {}, [], {}, 0, 0.0)
    return ex.StartRecord("fake", k, 0, 0, 1, [], [stage], [])


class ConfigTests(unittest.TestCase):
    def test_defaults_are_the_i25_1_values(self):
        c = ex.SEVEN
        self.assertEqual(c.parameters, ("vth0", "u0", "nfactor", "vsat", "delta", "rdsw", "eta0"))
        self.assertEqual((c.cmaes_evals, c.trf_nfev), (3000, 400))
        self.assertEqual((ex.FORTY_FIVE.cmaes_evals, ex.FORTY_FIVE.trf_nfev), (8000, 600))
        self.assertEqual(ex.FORTY_FIVE.parameters, PARAMS45)
        for c in (ex.SEVEN, ex.FORTY_FIVE):
            self.assertEqual((c.rounds, c.n_starts, c.max_start_index, c.max_draws), (3, 3, 9, 200))
            self.assertEqual((c.polish_nfev, c.refit_nfev), (400, 400))
            self.assertEqual((c.start_seed_base, c.start_seed_stride, c.cmaes_seed),
                             (20261027, 1000, 20261015))
            self.assertEqual((c.sigma0, c.popsize, c.fd_step, c.irls_passes), (0.3, None, 0.01, 2))
            self.assertEqual((c.range_shrink, c.range_edge, c.template), (0.5, 0.9, "mult1"))
        self.assertEqual(ex.CONFIGS, {7: ex.SEVEN, 45: ex.FORTY_FIVE})

    def test_params45_matches_the_stored_screen(self):
        path = I25 / "screen_mult1/summary.json"
        if not path.exists():
            self.skipTest("I25-1 screen summary not present")
        self.assertEqual(PARAMS45, tuple(json.loads(path.read_text())["selected_parameters"]))
        self.assertEqual(len(PARAMS45), 45)
        self.assertTrue(set(PARAMS7) <= set(PARAMS45))

    def test_popsize_rule(self):
        self.assertEqual(ex.SEVEN.popsize_for(7), 9)
        self.assertEqual(ex.SEVEN.popsize_for(40), 15)
        self.assertEqual(ex.SEVEN.popsize_for(2), 6)
        self.assertEqual(ex.ExtractionConfig(popsize=12).popsize_for(40), 12)

    def test_start_seed(self):
        self.assertEqual(ex.start_seed(ex.SEVEN, "nmos_L0p19_W7", 2),
                         20261027 + 2000 + ex.device_index("nmos_L0p19_W7"))


class BudgetTests(unittest.TestCase):
    def test_trf_pass_budget(self):
        self.assertEqual(trf_pass_budget(400, 2, 7), 22)     # 400 // 2 // 9
        self.assertEqual(trf_pass_budget(600, 2, 40), 7)     # 600 // 2 // 42
        self.assertEqual(trf_pass_budget(400, 2, 40), 4)
        self.assertEqual(trf_pass_budget(10, 2, 7), 2)       # floor of 2
        self.assertEqual(trf_pass_budget(400, 0, 7), 44)     # passes < 1 -> 1


class RangeTests(unittest.TestCase):
    def test_initial_range_widths_and_fixed_zero(self):
        names = ("vth0", "u0", "k1", "ub")
        r = ex.initial_range(names, {"vth0": 0.5, "u0": 0.02, "k1": 0.6, "ub": 0.0})
        self.assertAlmostEqual(r.half_width["vth0"], math.log(3))
        self.assertAlmostEqual(r.half_width["u0"], math.log(100))
        self.assertAlmostEqual(r.half_width["k1"], math.log(3))
        bounds = r.to_json()["bounds"]
        np.testing.assert_allclose(bounds["vth0"], [0.5 / 3, 1.5])
        np.testing.assert_allclose(bounds["u0"], [0.0002, 2.0])
        self.assertEqual(bounds["ub"], [0.0, 0.0])
        box = r.box()
        np.testing.assert_array_equal(box.active, [True, True, True, False])
        np.testing.assert_allclose(box.local_published, 0.0, atol=1e-12)

    def test_negative_center_keeps_order(self):
        r = ex.initial_range(("vsat",), {"vsat": -2.0e4})
        lo, hi = r.to_json()["bounds"]["vsat"]
        self.assertLess(lo, hi)
        self.assertAlmostEqual(lo, -2.0e6)

    def test_next_range_rule(self):
        r1 = ex.initial_range(("vth0", "u0", "k1"), {"vth0": 0.5, "u0": 0.02, "k1": 0.6})
        card = {"vth0": 0.45, "u0": 0.03, "k1": 0.7}
        r2 = ex.next_range(r1, card, [0.2, -0.95, 0.9])
        self.assertEqual(r2.round_index, 2)
        self.assertEqual(dict(r2.center), card)
        self.assertAlmostEqual(r2.half_width["vth0"], 0.5 * math.log(3))   # interior: halved
        self.assertAlmostEqual(r2.half_width["u0"], math.log(100))         # |u| >= 0.9: kept
        self.assertAlmostEqual(r2.half_width["k1"], math.log(3))
        tiny = ex.SearchRange(("k1",), {"k1": 0.6}, {"k1": MIN_HALF_WIDTH * 1.5})
        self.assertAlmostEqual(ex.next_range(tiny, {"k1": 0.6}, [0.0]).half_width["k1"],
                               MIN_HALF_WIDTH)
        # Same result as cold_start.next_round_spec on a multi-device spec.
        spec = next_round_spec(r1.spec() | {"half_width_by_device": {"x": dict(r1.half_width)}},
                               {"x": card}, {"x": {"vth0": 0.2, "u0": -0.95, "k1": 0.9}},
                               round_index=2)
        self.assertEqual(spec["half_width_by_device"]["x"], dict(r2.half_width))
        self.assertEqual(spec["bounds_by_device"]["x"], r2.to_json()["bounds"])

    def test_ranges_reproduce_stored_i25_1_specs(self):
        """initial_range on the stored centers gives the stored round-1 box;
        next_range on stored round-1/2 records gives the stored next boxes."""
        if not (I25 / "specs/seven_r1.json").exists():
            self.skipTest("I25-1 specs not present")
        for pre, names in (("seven", PARAMS7), ("full", PARAMS45)):
            spec1 = json.loads((I25 / f"specs/{pre}_r1.json").read_text())
            for tag in ("nmos_L1_W3", "pmos_L0p35_W1p6"):
                r = ex.initial_range(names, spec1["published_by_device"][tag])
                self.assertEqual(r.to_json()["bounds"], spec1["bounds_by_device"][tag])
                self.assertEqual(r.to_json()["log_scale"], spec1["log_scale_by_device"][tag])
                for rnd in (2, 3):
                    rec = json.loads((I25 / f"stepA/{pre}_s0_r{rnd - 1}.json").read_text())
                    rec = rec["devices"][tag]
                    stored = json.loads((I25 / f"specs/{pre}_s0_r{rnd}.json").read_text())
                    r = ex.next_range(r, rec["params"], rec["local"])
                    self.assertEqual(r.to_json()["bounds"], stored["bounds_by_device"][tag])
                    self.assertEqual(dict(r.half_width), stored["half_width_by_device"][tag])


class DrawTests(unittest.TestCase):
    def test_first_finite_draw_is_kept(self):
        problem = FakeProblem(fake_box(), n_fail=2)
        draw = ex.draw_start(problem, np.random.default_rng(5), max_draws=10)
        self.assertEqual((draw.redraws, draw.runs), (2, 3))
        rng = np.random.default_rng(5)
        for _ in range(3):
            expected = rng.uniform(-1.0, 1.0, size=2)
        np.testing.assert_array_equal(draw.local[:2], expected)
        self.assertEqual(draw.local[2], 0.0)                    # fixed coordinate

    def test_nan_before_the_trim_start_is_ignored(self):
        draw = ex.draw_start(FakeProblem(fake_box()), np.random.default_rng(0))
        self.assertEqual((draw.redraws, draw.runs), (0, 1))

    def test_max_draws(self):
        with self.assertRaises(RuntimeError):
            ex.draw_start(FakeProblem(fake_box(), n_fail=5), np.random.default_rng(0), max_draws=5)


class SearchTests(unittest.TestCase):
    def test_cmaes_search_on_a_quadratic(self):
        problem = FakeProblem(fake_box())
        res = ex.cmaes_search(problem, np.zeros(3), evals=300, sigma0=0.3, seed=1, popsize=6)
        np.testing.assert_allclose(res.local[:2], 0.3, atol=1e-2)
        self.assertEqual(res.local[2], 0.0)
        self.assertEqual(problem.n_sim, res.evals)
        self.assertLessEqual(res.evals, 300 + 6)


class SelectionTests(unittest.TestCase):
    def test_lowest_finite_among_primary(self):
        starts = [start_record(0, 0.3), start_record(1, 0.2), start_record(2, math.inf)]
        self.assertEqual(ex.select_start(starts, 3), 1)

    def test_tie_goes_to_lowest_k(self):
        starts = [start_record(2, 0.2), start_record(0, 0.2), start_record(1, 0.4)]
        self.assertEqual(ex.select_start(starts, 3), 0)

    def test_extra_start_only_when_no_primary_is_eligible(self):
        starts = [start_record(0, math.inf), start_record(1, math.nan), start_record(2, math.inf),
                  start_record(4, 0.1), start_record(3, 0.5)]
        self.assertEqual(ex.select_start(starts, 3), 3)
        starts[1] = start_record(1, 0.9)
        self.assertEqual(ex.select_start(starts, 3), 1)

    def test_no_eligible_start(self):
        self.assertIsNone(ex.select_start([start_record(0, math.inf)], 3))


if __name__ == "__main__":
    unittest.main()
