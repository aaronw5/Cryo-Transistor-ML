from __future__ import annotations

import math
import unittest

import numpy as np

from cryoml.racing import (CallLedger, RaceScheduler, RaceSettings,
                           RestartPool, TaggedBackend, min_distance_to,
                           rms_distance, seed_run, select_diverse)
from cryoml.trust_region import TrustRegionRun, TrustRegionSettings


class Counter:
    """Shared expensive backend stand-in: only the call counter matters."""

    def __init__(self, jacobian_cost: int = 5):
        self.n_calls = 0
        self.jacobian_cost = jacobian_cost


class FakeRun:
    """Scripted run with TrustRegionRun's advance contract: start call (if
    new), Jacobian (if none at the centre), then trials, each scripted as
    (predicted, rho, accepted); stationary once the script runs out.
    ``box_prediction`` is what ``solve_subproblem`` returns at any delta."""

    def __init__(self, backend, z, F, script, label="", box_prediction=0.0,
                 seeded=False):
        self.backend = backend
        self.z_c = np.asarray(z, dtype=np.float64)
        self.F_start = F
        self.F_c = F if seeded else math.inf
        self.status = "active" if seeded else "new"
        self.J_c = None
        self.r_c = None
        self.model = None
        self.delta = 0.25
        self.log = []
        self.script = list(script)
        self.label = label
        self.box_prediction = box_prediction
        self.subproblem_deltas = []

    def objective(self, z, r):
        return 0.5 * float(r @ r)

    def solve_subproblem(self):
        self.subproblem_deltas.append(self.delta)
        self.model = "built"
        return self.z_c, self.box_prediction

    def advance(self, limit):
        b = self.backend
        if self.status == "new":
            if b.n_calls + 1 > limit:
                return self.status
            b.n_calls += 1
            self.F_c = self.F_start
            self.status = "active"
        while self.status == "active":
            if self.J_c is None:
                if b.n_calls + b.jacobian_cost > limit:
                    break
                b.n_calls += b.jacobian_cost
                self.J_c = "J"
            if b.n_calls + 1 > limit:
                break
            if not self.script:
                self.status = "stationary"
                self.log.append({"calls": b.n_calls, "predicted": 0.0,
                                 "stationary": True})
                break
            predicted, rho, accepted = self.script.pop(0)
            b.n_calls += 1
            self.log.append({"calls": b.n_calls, "predicted": predicted,
                             "rho": rho, "accepted": accepted})
            if accepted:
                self.F_c -= rho * predicted
                self.z_c = self.z_c + 0.01
                self.J_c = self.model = None
        return self.status


class Rosenbrock:
    jacobian_cost = 2

    def __init__(self):
        self.n_calls = 0

    def residual(self, z):
        self.n_calls += 1
        return np.array([10.0 * (z[1] - z[0] ** 2), 0.5 - z[0]])

    def jacobian(self, z, r):
        self.n_calls += self.jacobian_cost
        return np.array([[-20.0 * z[0], 10.0], [-1.0, 0.0]])


class DiversityTests(unittest.TestCase):
    def test_select_diverse_skips_close_candidates_and_fills(self):
        candidates = [np.zeros(4), np.full(4, 0.1), np.full(4, 0.5),
                      np.full(4, -0.6)]
        scores = [0.0, 1.0, 2.0, 3.0]
        self.assertEqual(select_diverse(candidates, scores, 3, 0.35), [0, 2, 3])
        # Excluded points block their neighbourhood; fill when short.
        picked = select_diverse(candidates, scores, 3, 0.35,
                                exclude=[np.full(4, 0.45)])
        self.assertEqual(picked[:2], [0, 3])
        self.assertEqual(len(picked), 3)
        self.assertEqual(select_diverse(candidates, scores, 3, 0.35,
                                        exclude=[np.full(4, 0.45)],
                                        fill=False), [0, 3])

    def test_distances(self):
        self.assertAlmostEqual(rms_distance(np.zeros(4), np.ones(4)), 1.0)
        self.assertEqual(min_distance_to(np.zeros(2), []), math.inf)


class SeedTests(unittest.TestCase):
    def test_seed_run_is_active_with_zero_calls_and_jacobian_comes_next(self):
        backend = Rosenbrock()
        z0 = np.array([-0.5, 0.5])
        r0 = np.array([10.0 * (0.5 - 0.25), 1.0])
        run = TrustRegionRun(z0, backend, None, 0.0, TrustRegionSettings())
        seed_run(run, r0)
        self.assertEqual(run.status, "active")
        self.assertEqual(backend.n_calls, 0)
        self.assertEqual(run.F_c, 0.5 * float(r0 @ r0))
        self.assertEqual(run.F_start, run.F_c)
        run.advance(backend.n_calls + Rosenbrock.jacobian_cost + 1)
        # The Jacobian and one trial, no repeated start call.
        self.assertEqual(backend.n_calls, Rosenbrock.jacobian_cost + 1)
        self.assertEqual(run.log[0]["calls"], Rosenbrock.jacobian_cost + 1)
        with self.assertRaises(ValueError):
            seed_run(run, r0)

    def test_ledger_charges_each_run(self):
        backend = Rosenbrock()
        ledger = CallLedger(backend)
        runs = [TrustRegionRun(np.array(z), TaggedBackend(backend, ledger, l),
                               None, 0.0, TrustRegionSettings(), label=l)
                for z, l in (([-0.8, 0.8], "a"), ([0.9, -0.9], "b"))]
        runs[0].advance(4)
        runs[1].advance(9)
        by_label = ledger.calls_by_label()
        self.assertEqual(sum(by_label.values()), backend.n_calls)
        self.assertEqual(by_label["a"], runs[0].calls)
        self.assertEqual(by_label["b"], runs[1].calls)
        self.assertEqual(ledger.label_of_call(1), "a")
        self.assertEqual(ledger.label_of_call(backend.n_calls), "b")


class RaceTests(unittest.TestCase):
    def test_argmax_of_expected_reduction_per_call_is_chosen(self):
        backend = Counter(jacobian_cost=43)
        a = FakeRun(backend, [0.0], 5.0, [], "a", seeded=True)
        b = FakeRun(backend, [1.0], 5.0, [], "b", seeded=True)
        c = FakeRun(backend, [-1.0], 5.0, [], "c", seeded=True)
        a.J_c = c.J_c = "J"                      # cost 1; b needs a Jacobian
        a.log = [{"predicted": 1.0, "rho": 0.5}]
        b.log = [{"predicted": 10.0, "rho": 0.5}]
        c.log = [{"predicted": 0.8, "rho": 1.0}, {"predicted": 0.8, "rho": 1.0}]
        race = RaceScheduler([a, b, c], backend, 1000)
        for lane in race.lanes:
            race._update(lane)
        e = {lane.label: race.expected_reduction(lane) for lane in race.lanes}
        self.assertAlmostEqual(e["a"], 0.5 * 1.0 / 1)
        self.assertAlmostEqual(e["b"], 0.5 * 10.0 / 44)
        self.assertAlmostEqual(e["c"], 0.875 * 0.8 / 1)
        self.assertEqual(race.pick().label, "c")
        # Tie on e_i: the lower centre objective wins.
        c.log = [{"predicted": 1.0, "rho": 0.5}]
        race.lanes[2].rho_ema = 0.5
        c.F_c = 4.0
        self.assertEqual(race.pick().label, "c")
        # A run that has not logged a trial yet has e = inf.
        d = FakeRun(backend, [2.0], 9.0, [], "d")
        race.add(d)
        self.assertEqual(race.pick().label, "d")

    def test_nan_prediction_never_wins_the_pick(self):
        backend = Counter()
        bad = FakeRun(backend, [0.0], 1.0, [], "bad", seeded=True)
        good = FakeRun(backend, [1.0], 2.0, [], "good", seeded=True)
        bad.J_c = good.J_c = "J"
        bad.log = [{"predicted": float("nan"), "rho": float("nan")}]
        good.log = [{"predicted": 1e-6, "rho": 0.5}]
        race = RaceScheduler([bad, good], backend, 100)
        self.assertEqual(race.pick().label, "good")

    def test_rho_bar_is_clipped_and_robust_to_outliers(self):
        backend = Counter()
        run = FakeRun(backend, [0.0], 1.0, [], "a", seeded=True)
        run.log = [{"predicted": 1.0, "rho": -1e6}]
        race = RaceScheduler([run], backend, 100)
        race._update(race.lanes[0])
        self.assertAlmostEqual(race.rho_bar(race.lanes[0]), 0.5 * 0.05 + 0.25)
        run.log.append({"predicted": 1.0, "rho": 3.0})
        race._update(race.lanes[0])
        self.assertLessEqual(race.rho_bar(race.lanes[0]), 1.0)
        self.assertGreaterEqual(race.rho_bar(race.lanes[0]), 0.05)

    def test_drop_rule_never_drops_the_best_run(self):
        backend = Counter(jacobian_cost=3)
        # The best run has a nonsense negative box prediction: its bound
        # exceeds the best objective, but it is the best run.
        best = FakeRun(backend, [0.0], 1.0, [(0.1, 1.0, True)] * 3, "best",
                       box_prediction=-100.0, seeded=True)
        hopeless = FakeRun(backend, [1.0], 10.0, [(0.1, 1.0, True)] * 3,
                           "hopeless", box_prediction=1.0, seeded=True)
        hopeful = FakeRun(backend, [-1.0], 3.0, [(0.1, 1.0, True)] * 3,
                          "hopeful", box_prediction=2.5, seeded=True)
        race = RaceScheduler([best, hopeless, hopeful], backend, 200)
        race.run()
        dropped = [d["label"] for d in race.drops]
        self.assertNotIn("best", dropped)
        self.assertIn("hopeless", dropped)
        self.assertNotIn("hopeful", dropped)
        # The drop check used the 0.5 box and restored the radius.
        self.assertIn(0.5, hopeless.subproblem_deltas)
        self.assertEqual(hopeless.delta, 0.25)
        # A dropped run got its Jacobian but no trial after the drop.
        self.assertEqual(len(hopeless.log), 0)
        for drop in race.drops:
            self.assertGreater(drop["bound"], drop["best_F"])

    def test_best_run_is_not_dropped_on_its_own_negative_prediction(self):
        backend = Counter(jacobian_cost=3)
        only = FakeRun(backend, [0.0], 1.0, [(0.1, 1.0, False)] * 5, "only",
                       box_prediction=-5.0, seeded=True)
        race = RaceScheduler([only], backend, 50)
        race.run()
        self.assertEqual(race.drops, [])

    def test_budget_is_never_exceeded(self):
        rng = np.random.default_rng(0)
        for trial in range(200):
            backend = Counter(jacobian_cost=int(rng.integers(1, 8)))
            budget = int(rng.integers(0, 120))

            def fake(index):
                script = [(float(rng.uniform(0.01, 1.0)),
                           float(rng.uniform(-2.0, 1.5)),
                           bool(rng.random() < 0.5))
                          for _ in range(int(rng.integers(0, 12)))]
                return FakeRun(backend, rng.uniform(-1, 1, 3),
                               float(rng.uniform(1, 10)), script,
                               f"r{index}", float(rng.uniform(-1, 5)),
                               seeded=bool(rng.random() < 0.5))

            pool = RestartPool([{"label": f"p{i}",
                                 "point": rng.uniform(-1, 1, 3),
                                 "make": (lambda i: lambda z: fake(100 + i))(i)}
                                for i in range(5)], min_distance=0.2)
            race = RaceScheduler([fake(i) for i in range(4)], backend, budget,
                                 RaceSettings(), restart_fn=pool)
            race.run()
            self.assertLessEqual(backend.n_calls, budget, trial)
            self.assertEqual(race.unspent_calls, budget - backend.n_calls)

    def test_budget_with_real_trust_region_runs(self):
        for budget in (0, 1, 7, 23, 60, 150):
            backend = Rosenbrock()
            runs = [TrustRegionRun(np.array(z), backend, None, 0.0,
                                   TrustRegionSettings(), label=str(z))
                    for z in ([-0.8, 0.8], [0.9, -0.9], [0.0, 0.0])]
            pool = RestartPool([{"label": "far", "point": np.array([-1.0, -1.0]),
                                 "make": lambda z: TrustRegionRun(
                                     z, backend, None, 0.0,
                                     TrustRegionSettings(), label="far")}])
            race = RaceScheduler(runs, backend, budget, restart_fn=pool)
            race.run()
            self.assertLessEqual(backend.n_calls, budget)
            if budget >= 60:
                best = min(race.runs, key=lambda run: run.F_c)
                self.assertLess(best.F_c, 1e-6)

    def test_restarts_respect_min_distance(self):
        backend = Counter(jacobian_cost=2)
        made = []

        def make(label):
            def build(z):
                run = FakeRun(backend, z, 5.0, [(0.1, 1.0, False)], label)
                made.append(run)
                return run
            return build

        points = {"near": [0.1, 0.0], "far": [1.0, 1.0], "near_far": [0.95, 1.0],
                  "other": [-1.0, 1.0]}
        pool = RestartPool([{"label": k, "point": np.array(v), "make": make(k)}
                            for k, v in points.items()], min_distance=0.35)
        start = FakeRun(backend, [0.0, 0.0], 5.0, [(0.1, 1.0, False)], "start",
                        seeded=True)
        race = RaceScheduler([start], backend, 1000, restart_fn=pool)
        centres_before = []
        original = pool.__call__

        def spy(centres, remaining):
            centres_before.append([c.copy() for c in centres])
            return original(centres, remaining)

        race.restart_fn = spy
        race.run()
        self.assertEqual([run.label for run in made], ["far", "other"])
        used = {row["label"]: row["used"] for row in pool.history}
        self.assertEqual(used, {"near": False, "far": True, "near_far": False,
                                "other": True})
        for run, centres in zip(made, centres_before):
            self.assertGreaterEqual(min_distance_to(run.z_c - 0.0, centres),
                                    0.35)
        self.assertEqual(race.stop_reason, "restart_pool_empty")
        self.assertEqual(len(race.restarts), 2)

    def test_lazy_restart_points_are_evaluated_once(self):
        calls = []

        def point():
            calls.append(1)
            return np.array([0.9, 0.9])

        pool = RestartPool([{"label": "lazy", "point": point,
                             "make": lambda z: ("run", z)}])
        out = pool([np.zeros(2)], 100)
        self.assertEqual(out[0], "run")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(pool([np.zeros(2)], 100))

    def test_no_restart_below_its_cost(self):
        backend = Counter(jacobian_cost=10)
        run = FakeRun(backend, [0.0], 1.0, [], "a", seeded=True)
        pool = RestartPool([{"label": "p", "point": np.array([5.0]),
                             "make": lambda z: self.fail("restart")}])
        race = RaceScheduler([run], backend, 20, restart_fn=pool)
        race.run()
        # 10 Jacobian calls, stationary, 10 left < 12.
        self.assertEqual(race.stop_reason, "budget_below_restart_cost")
        self.assertEqual(race.unspent_calls, 10)


if __name__ == "__main__":
    unittest.main()
