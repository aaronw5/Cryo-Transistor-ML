from __future__ import annotations

import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))

import cmaes_trace  # noqa: E402
import dev_compare  # noqa: E402
from hybrid_tr import best_at  # noqa: E402


class FakeNgspice:
    """Stand-in for ``hybrid_tr.NgspiceResiduals``: linear residuals, the
    same call counting and best/trace bookkeeping, no simulator."""

    def __init__(self, n: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.A = rng.normal(size=(3 * n, n))
        self.b = self.A @ rng.uniform(-0.5, 0.5, n)
        self.active = np.arange(n)
        self.jacobian_cost = n
        self.call_cap = math.inf
        self.n_calls = self.n_jacobians = 0
        self.centre_rrms: dict[bytes, float] = {}
        self.best = {"rrms": math.inf, "local": None, "calls": 0, "kind": None}
        self.trace: list[list] = []

    def _seen(self, z, kind):
        self.n_calls += 1
        r = self.A @ z - self.b
        rrms = float(np.sqrt(np.mean(r ** 2)))
        if rrms < self.best["rrms"]:
            self.best = {"rrms": rrms, "local": np.array(z, dtype=np.float64),
                         "calls": self.n_calls, "kind": kind}
            self.trace.append([self.n_calls, rrms])
        return r, rrms

    def residual(self, z):
        r, self.centre_rrms[z.tobytes()] = self._seen(z, "trial")
        return r

    def _simulate_many(self, points):
        return [(*self._seen(p, "jacobian_probe"), False) for p in points]

    def jacobian(self, z, r):
        self.n_jacobians += 1
        rows = self._simulate_many([z + 1e-3 * e for e in np.eye(len(z))])
        return np.stack([(row[0] - r) / 1e-3 for row in rows], axis=1)


class TraceStatistics(unittest.TestCase):
    trace = [[1, 0.5], [10, 0.2], [100, 0.05], [1000, 0.001]]

    def test_best_at(self):
        self.assertEqual(best_at(self.trace, 0), math.inf)
        self.assertEqual(best_at(self.trace, 1), 0.5)
        self.assertEqual(best_at(self.trace, 99), 0.2)
        self.assertEqual(best_at(self.trace, 100), 0.05)
        self.assertEqual(best_at(self.trace, 5000), 0.001)

    def test_auc(self):
        grid = dev_compare.log_grid(5000)
        self.assertEqual(len(grid), 40)
        self.assertAlmostEqual(grid[0], 10.0)
        self.assertAlmostEqual(grid[-1], 5000.0)
        flat = [[1, 0.1]]
        self.assertAlmostEqual(dev_compare.auc([flat], grid), -1.0)
        # The 1e-6 floor, and averaging over targets.
        exact = [[1, 0.0]]
        self.assertAlmostEqual(dev_compare.auc([exact], grid), -6.0)
        self.assertAlmostEqual(dev_compare.auc([flat, exact], grid), -3.5)
        expected = np.mean([math.log10(best_at(self.trace, g)) for g in grid])
        self.assertAlmostEqual(dev_compare.auc([self.trace], grid), expected)

    def test_calls_to_reach(self):
        self.assertEqual(dev_compare.calls_to_reach(self.trace, 0.05), 100)
        self.assertEqual(dev_compare.calls_to_reach(self.trace, 1e-4), math.inf)

    def test_bootstrap_and_paired(self):
        rng = np.random.default_rng(1)
        diff = rng.normal(0.3, 0.1, 36)
        lo, hi = dev_compare.bootstrap_ci(diff)
        self.assertLess(lo, diff.mean())
        self.assertGreater(hi, diff.mean())
        self.assertGreater(lo, 0.2)
        self.assertEqual(dev_compare.bootstrap_ci(diff), [lo, hi])  # seed 0
        self.assertEqual(dev_compare.bootstrap_ci(np.full(5, 2.0)), [2.0, 2.0])
        ref = {f"t{i}": 1.0 for i in range(6)}
        arm = {"t0": 0.5, "t1": 0.5, "t2": 0.995, "t3": 1.0, "t4": 2.0,
               "t5": 0.4}
        row = dev_compare.paired(arm, ref)
        self.assertEqual((row["wins"], row["ties"], row["losses"]), (3, 2, 1))
        self.assertEqual(row["n"], 6)
        same = dev_compare.paired(ref, ref)
        self.assertEqual(same["wilcoxon_p"], 1.0)
        self.assertEqual((same["wins"], same["ties"], same["losses"]), (0, 6, 0))

    def test_pooled_never_excludes(self):
        stats = dev_compare.pooled({"a": 0.001, "b": 0.0, "c": 88.0})
        self.assertAlmostEqual(stats["mean"], 88.001 / 3)
        self.assertEqual(stats["worst_target"], "c")
        self.assertAlmostEqual(stats["frac_below_1e-2"], 2 / 3)
        self.assertAlmostEqual(stats["frac_below_1e-3"], 1 / 3)


class BudgetGuard(unittest.TestCase):
    def test_refuses_before_exceeding(self):
        fake = FakeNgspice(n=4)
        guard = cmaes_trace.Budgeted(fake, 10)
        self.assertEqual(fake.call_cap, 10)
        guard.many([np.zeros(4), np.ones(4) * 0.1, np.ones(4) * 0.2], "cma")
        z = np.full(4, 0.3)
        guard.residual(z, "trf_trial")
        guard.residual(z, "trf_trial")         # cached: free
        self.assertEqual(fake.n_calls, 4)
        guard.jacobian(z, "trf_jacobian_probe")  # 4 probes
        self.assertEqual(fake.n_calls, 8)
        with self.assertRaises(cmaes_trace.BudgetExhausted):
            guard.jacobian(z, "trf_jacobian_probe")
        with self.assertRaises(cmaes_trace.BudgetExhausted):
            guard.many([np.zeros(4)] * 3, "cma")
        self.assertEqual(fake.n_calls, 8)        # nothing ran
        guard.residual(np.full(4, 0.4), "trf_trial")
        guard.residual(np.full(4, 0.5), "trf_trial")
        with self.assertRaises(cmaes_trace.BudgetExhausted):
            guard.residual(np.full(4, 0.6), "trf_trial")
        self.assertEqual(fake.n_calls, 10)

    def test_relabels_best_by_phase(self):
        fake = FakeNgspice(n=4)
        guard = cmaes_trace.Budgeted(fake, 100)
        guard.many([np.full(4, 0.9)], "cma")
        self.assertEqual(fake.best["kind"], "cma")
        z = np.clip(np.linalg.lstsq(fake.A, fake.b, rcond=None)[0], -1, 1)
        guard.residual(z, "trf_trial")
        self.assertEqual(fake.best["kind"], "trf_trial")

    def test_phases_stay_within_budget(self):
        options = {"sigma0": 0.3, "popsize": 0, "restarts": 2}
        for budget in (37, 120):
            fake = FakeNgspice(n=4)
            guard = cmaes_trace.Budgeted(fake, budget)
            cma_stats, trf_stats = {}, {}
            try:
                cmaes_trace.cma_phase(guard, np.zeros(4), options, 7,
                                      int(0.6 * budget), cma_stats)
                self.assertLessEqual(fake.n_calls, int(0.6 * budget))
                # The CMA best's residual is cached: TRF starts for free.
                self.assertIn(fake.best["local"][fake.active].tobytes(),
                              guard.cache)
                before = fake.n_calls
                cmaes_trace.trf_phase(guard, 1e-3, trf_stats)
            except cmaes_trace.BudgetExhausted:
                pass
            self.assertLessEqual(fake.n_calls, budget)
            self.assertEqual(fake.trace[0], [1, fake.trace[0][1]])
            self.assertAlmostEqual(  # x0 = 0 injected and scored first
                fake.trace[0][1], float(np.sqrt(np.mean(fake.b ** 2))))
            self.assertGreater(cma_stats["generations"], 0)
            self.assertEqual(trf_stats["calls"], fake.n_calls - before)
            self.assertEqual(trf_stats["calls"],
                             trf_stats["njev"] * 4 + trf_stats["nfev"] - 1)
        # A start near the box face: the exact x0 is scored first (pycma's
        # boundary transform would have moved an injected one).
        edge = FakeNgspice(n=4)
        x_edge = np.array([0.999, -0.97, 0.2, 0.95])
        guard = cmaes_trace.Budgeted(edge, 40)
        cmaes_trace.cma_phase(guard, x_edge, options, 7, 40, {})
        r = edge.A @ x_edge - edge.b
        self.assertEqual(edge.trace[0], [1, float(np.sqrt(np.mean(r ** 2)))])
        lq = FakeNgspice(n=4)
        guard = cmaes_trace.Budgeted(lq, 60)
        stats = {}
        try:
            cmaes_trace.es_phase(guard, np.zeros(4), options, 7, stats,
                                 "lq_cmaes")
        except cmaes_trace.BudgetExhausted:
            pass
        self.assertLessEqual(lq.n_calls, 60)
        self.assertGreater(stats["generations"], 0)
        # lq-CMA-ES scores the bank start first, like every other arm.
        self.assertEqual(lq.trace[0][0], 1)
        self.assertAlmostEqual(lq.trace[0][1], float(np.sqrt(np.mean(lq.b ** 2))))

    def test_lq_restarts_do_not_replay_samples(self):
        # A flat objective stops each lq run early (tolflatfitness), so the
        # IPOP restarts happen well inside the budget.  Restarts must use a
        # new seed: fmin_lq_surr2's own restarts reuse it and re-simulate the
        # first run's offspring.
        flat = FakeNgspice(n=4)
        flat.A[:] = 0.0
        seen = []
        original = flat._seen

        def recording(z, kind):
            seen.append(np.asarray(z, dtype=np.float64).tobytes())
            return original(z, kind)

        flat._seen = recording
        # A tiny residual cache, so a replayed point is charged (as it is
        # on NGSpice once the 64-entry cache has moved on).
        guard = cmaes_trace.Budgeted(flat, 400, cache_size=1)
        stats = {}
        options = {"sigma0": 0.3, "popsize": 0, "restarts": 3}
        try:
            cmaes_trace.es_phase(guard, np.zeros(4), options, 7, stats,
                                 "lq_cmaes")
        except cmaes_trace.BudgetExhausted:
            pass
        self.assertLessEqual(flat.n_calls, 400)
        runs = stats["runs"]
        self.assertGreater(len(runs), 1)
        self.assertEqual(len({r["seed"] for r in runs}), len(runs))
        self.assertEqual([r["popsize"] for r in runs[:2]],
                         [runs[0]["popsize"], 2 * runs[0]["popsize"]])
        # Nothing simulated twice (x0 = 0 is scored once, first).
        self.assertEqual(len(seen), len(set(seen)))


def write_run(directory: Path, method: str, budget: int, records: dict,
              target_override=True, summary: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for tag, record in records.items():
        (directory / f"ml_{tag}.json").write_text(json.dumps(
            {"device": tag, "methods": {method: record}}))
    if summary:
        (directory / "summary.json").write_text(json.dumps(
            {"target_override": target_override}))


def record(trace, budget, calls=None, oracle=0.3):
    return {"trace": trace, "rrms": best_at(trace, budget), "budget": budget,
            "n_ngspice_calls": calls if calls is not None else trace[-1][0],
            "oracle_lhc": {"paper_rrms": oracle}}


class CompareRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.oracle = dev_compare.OracleReference(
            {"bank": "b", "round_dir": "r", "targets_template": "t"},
            self.root / "oracle.json", enabled=False)

    def tearDown(self):
        self.tmp.cleanup()

    def evaluate(self, arm, sets=("set_0",)):
        checks = dev_compare.Checks(allow_partial=True)
        with redirect_stdout(io.StringIO()):
            out = dev_compare.evaluate_arm(arm, list(sets), checks, self.oracle)
        return out, checks

    def test_fixed_and_trace_modes(self):
        tags = ("nmos_a", "pmos_b")
        for b in (500, 5000):
            write_run(self.root / f"fixed/B{b}/set_0", "m", b, {
                t: record([[1, 0.2], [b, 0.01 * (i + 1) * 500 / b]], b)
                for i, t in enumerate(tags)})
        fixed, checks = self.evaluate({
            "label": "fixed", "mode": "fixed", "method_key": "m",
            "budgets": [500, 5000],
            "dir_template": str(self.root / "fixed/B{budget}/{set}")})
        self.assertFalse(checks.failed)
        self.assertAlmostEqual(fixed["budgets"][500]["set_0/pmos_b"], 0.02)
        self.assertAlmostEqual(fixed["budgets"][5000]["set_0/pmos_b"], 0.002)
        trace, checks = self.evaluate({
            "label": "trace", "mode": "trace", "method_key": "m",
            "budgets": [5000],
            "dir_template": str(self.root / "fixed/B{budget}/{set}")})
        self.assertFalse(checks.failed)
        # Read from the B5000 trace: at 500 calls only the first entry.
        self.assertAlmostEqual(trace["budgets"][500]["set_0/pmos_b"], 0.2)
        self.assertAlmostEqual(trace["budgets"][5000]["set_0/pmos_b"], 0.002)

    def test_refusals_and_hard_checks(self):
        write_run(self.root / "test/B500/set_0", "m", 500,
                  {"nmos_a": record([[1, 0.1]], 500)}, target_override=False)
        out, checks = self.evaluate({
            "label": "test_run", "mode": "fixed", "method_key": "m",
            "budgets": [500],
            "dir_template": str(self.root / "test/B{budget}/{set}")})
        self.assertIsNone(out)
        self.assertTrue(checks.failed)
        over = record([[1, 0.1], [600, 0.05]], 500, calls=600)
        over["rrms"] = 0.05
        write_run(self.root / "over/B500/set_0", "m", 500, {"nmos_a": over})
        _out, checks = self.evaluate({
            "label": "over", "mode": "fixed", "method_key": "m",
            "budgets": [500],
            "dir_template": str(self.root / "over/B{budget}/{set}")})
        failed = [r["check"] for r in checks.rows if r["level"] == "FAIL"]
        self.assertTrue(any("n_ngspice_calls" in c for c in failed))
        self.assertTrue(any("best_at" in c for c in failed))

    def test_other_set_is_refused(self):
        oracle = dev_compare.OracleReference(
            {"bank": "b", "round_dir": "r", "targets_template": "t"},
            self.root / "oracle.json", enabled=True)
        oracle.values = {oracle.key("set_0", "nmos_a"): 0.3}
        write_run(self.root / "s1/B500/set_0", "m", 500,
                  {"nmos_a": record([[1, 0.1]], 500, oracle=0.31)})
        checks = dev_compare.Checks(allow_partial=True)
        with redirect_stdout(io.StringIO()):
            out = dev_compare.evaluate_arm({
                "label": "s1", "mode": "fixed", "method_key": "m",
                "budgets": [500],
                "dir_template": str(self.root / "s1/B{budget}/{set}")},
                ["set_0"], checks, oracle)
        self.assertIsNone(out)
        self.assertTrue(checks.failed)


if __name__ == "__main__":
    unittest.main()
