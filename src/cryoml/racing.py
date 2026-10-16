"""Budget-aware anytime allocation of expensive calls among trust-region runs.

Several resumable local runs (``cryoml.trust_region.TrustRegionRun`` or
anything with the same surface) share one budget of expensive calls.
``RaceScheduler`` repeatedly advances the run with the highest expected
objective reduction per expensive call,

    e_i = rho_bar_i * p_i / c_i,

where ``p_i`` is the predicted reduction of the run's last trial step
(``run.log[-1]['predicted']``; +inf before its first trial), ``rho_bar_i`` an
exponential moving average of the run's observed actual/predicted ratios
(``RaceSettings``; observations are clipped to ``rho_clip`` before averaging,
so one catastrophic trial with rho ~ -1e6 does not pin a run at the floor for
many iterations; the average is therefore always inside ``rho_clip``), and
``c_i`` the expensive cost of its next iteration (``run.next_iteration_cost()``
if defined, else 1 start call if the run is new, plus ``jacobian_cost`` if
it has no Jacobian at its centre, plus 1 trial).  Ties go to the lower
centre objective.

Drops: when a run has a model at a new centre (its Jacobian there has just
been computed, before the trial), the model's predicted reduction inside the
infinity-norm box of radius ``drop_radius`` (0.5) gives an optimistic bound
``F_c - pred_box`` (zero expensive calls).  If that bound is worse than the
best centre objective of any run, and the run is not itself the best, it is
dropped for good.  Iterations that need a Jacobian are therefore advanced in
two parts (model-building calls, drop check, trial), which spends exactly
the calls of one uninterrupted iteration.

Restarts: when no run can be advanced and at least ``jacobian_cost + 2``
calls remain, ``restart_fn(centres, remaining)`` may return a new run
(``RestartPool`` proposes them in a fixed order, skipping any proposal within
``min_distance`` RMS local distance of a centre already visited).

The scheduler never asks a run for more calls than remain: a run whose next
iteration costs more than what is left is not a candidate, and every
``advance`` limit is at most ``budget``.  (A backend may still spend extra
calls inside one Jacobian, e.g. cliff retries, but those are capped by the
backend itself.)

Run contract: ``advance(limit)`` (absolute limit on ``backend.n_calls``,
model-building calls first, the trial last, stopping before any call that
would pass the limit), ``status`` ('new', 'active' or a terminal value),
``F_c``, ``J_c``, ``z_c``, ``log`` (dicts with 'predicted' and, for trials,
'rho'), ``delta``, ``model`` and ``solve_subproblem() -> (z, predicted)``
(model only).  ``seed_run`` starts a run whose start point was already
simulated without a duplicate call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class RaceSettings:
    rho_ema: float = 0.5                    # EMA weight of the newest rho
    rho_init: float = 0.5
    rho_clip: tuple[float, float] = (0.05, 1.0)
    drop_radius: float = 0.5                # optimistic-bound box radius
    min_distance: float = 0.35              # restart / start diversity (RMS)
    check_drops: bool = True


def rms_distance(a: np.ndarray, b: np.ndarray, active=None) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if active is not None:
        a, b = a[active], b[active]
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def min_distance_to(point: np.ndarray, others: Sequence[np.ndarray],
                    active=None) -> float:
    """Smallest RMS distance from ``point`` to ``others`` (inf if none)."""
    return min((rms_distance(point, other, active) for other in others),
               default=math.inf)


def select_diverse(candidates: Sequence[np.ndarray], scores: Sequence[float],
                   n: int, min_distance: float = 0.35,
                   exclude: Sequence[np.ndarray] = (), active=None,
                   fill: bool = True) -> list[int]:
    """Indices of up to ``n`` candidates in ascending score order, each at
    RMS local distance >= ``min_distance`` (over ``active`` coordinates, all
    if None) from every earlier pick and every point in ``exclude``, as in
    ``fps.diverse_oracle_starts``.  If fewer qualify and ``fill``, the next
    best unpicked candidates fill the list regardless of distance."""
    order = [int(i) for i in np.argsort(np.asarray(scores, dtype=np.float64),
                                        kind="stable")]
    picked: list[int] = []
    for index in order:
        if len(picked) >= n:
            break
        others = [candidates[j] for j in picked] + list(exclude)
        if min_distance_to(candidates[index], others, active) >= min_distance:
            picked.append(index)
    if fill:
        for index in order:
            if len(picked) >= n:
                break
            if index not in picked:
                picked.append(index)
    return picked


def seed_run(run, r: np.ndarray, F: float | None = None):
    """Start ``run`` at its ``z_c`` from an already simulated residual: no
    expensive call is made; the next ``advance`` begins with the Jacobian."""
    if run.status != "new":
        raise ValueError(f"seed_run needs a new run, got {run.status!r}")
    run.r_c = np.asarray(r, dtype=np.float64).copy()
    if F is None:
        F = run.objective(run.z_c, run.r_c)
    run.F_c = run.F_start = float(F)
    run.status = "active"
    return run


class CallLedger:
    """Which label spent each expensive call: segments ``(start, stop,
    label)`` of the shared counter (call k, 1-based, is in the segment with
    start < k <= stop)."""

    def __init__(self, backend):
        self.backend = backend
        self.segments: list[tuple[int, int, str]] = []

    def charge(self, label: str, n0: int) -> int:
        n1 = int(self.backend.n_calls)
        if n1 > n0:
            self.segments.append((int(n0), n1, label))
        return n1 - int(n0)

    def calls_by_label(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for start, stop, label in self.segments:
            out[label] = out.get(label, 0) + stop - start
        return out

    def label_of_call(self, k: int) -> str | None:
        for start, stop, label in self.segments:
            if start < k <= stop:
                return label
        return None


class TaggedBackend:
    """One run's view of a shared expensive backend: the same counter, cost
    and calls, with every call it makes charged to ``label`` in a ledger."""

    def __init__(self, backend, ledger: CallLedger, label: str):
        self.backend = backend
        self.ledger = ledger
        self.label = label

    @property
    def n_calls(self) -> int:
        return self.backend.n_calls

    @property
    def jacobian_cost(self) -> int:
        return self.backend.jacobian_cost

    def residual(self, z):
        n0 = self.backend.n_calls
        try:
            return self.backend.residual(z)
        finally:
            self.ledger.charge(self.label, n0)

    def jacobian(self, z, r):
        n0 = self.backend.n_calls
        try:
            return self.backend.jacobian(z, r)
        finally:
            self.ledger.charge(self.label, n0)

    def __getattr__(self, name):
        return getattr(self.backend, name)


class RestartPool:
    """Ordered restart proposals for ``RaceScheduler``.

    Each proposal is a dict with ``label``, ``point`` (an array, or a
    zero-expensive-call callable returning one, evaluated lazily and only
    once) and ``make(point) -> run`` (which may spend expensive calls, e.g.
    to validate a point before seeding the run).  A call returns the run of
    the first unused proposal whose point is at RMS distance >=
    ``min_distance`` from every centre in ``centres``; proposals skipped on
    the way are used up."""

    def __init__(self, proposals: list[dict], min_distance: float = 0.35,
                 active=None):
        self.proposals = list(proposals)
        self.min_distance = float(min_distance)
        self.active = active
        self.position = 0
        self.history: list[dict] = []

    def __len__(self) -> int:
        return len(self.proposals) - self.position

    def __call__(self, centres: Sequence[np.ndarray], remaining: int):
        while self.position < len(self.proposals):
            proposal = self.proposals[self.position]
            self.position += 1
            point = proposal["point"]
            if callable(point):
                point = point()
            point = np.asarray(point, dtype=np.float64)
            distance = min_distance_to(point, centres, self.active)
            used = distance >= self.min_distance
            self.history.append({"label": proposal["label"],
                                 "distance_to_centres": distance,
                                 "used": bool(used)})
            if used:
                return proposal["make"](point)
        return None


@dataclass
class _Lane:
    run: object
    label: str
    rho_ema: float
    seen: int = 0
    dropped: bool = False
    stuck: bool = False
    checked: bytes | None = None
    centre: np.ndarray | None = None
    rho_history: list = field(default_factory=list)


class RaceScheduler:
    """Advance the run with the best predicted reduction per expensive call
    until the budget is spent (module docstring)."""

    def __init__(self, runs: Sequence, backend, budget: int,
                 settings: RaceSettings | None = None,
                 restart_fn: Callable | None = None):
        self.backend = backend
        self.budget = int(budget)
        self.settings = settings or RaceSettings()
        self.restart_fn = restart_fn
        self.lanes: list[_Lane] = []
        self.centres: list[np.ndarray] = []
        self.drops: list[dict] = []
        self.restarts: list[dict] = []
        self.n_drop_checks = 0
        self.n_iterations = 0
        self.stop_reason = None
        self.unspent_calls = None
        for run in runs:
            self.add(run)

    # bookkeeping -------------------------------------------------------
    def add(self, run) -> _Lane:
        lane = _Lane(run, getattr(run, "label", "") or f"run{len(self.lanes)}",
                     self.settings.rho_init)
        self.lanes.append(lane)
        self._note_centre(lane)
        return lane

    @property
    def runs(self) -> list:
        return [lane.run for lane in self.lanes]

    @property
    def remaining(self) -> int:
        return self.budget - int(self.backend.n_calls)

    def _note_centre(self, lane: _Lane) -> bool:
        z = np.asarray(lane.run.z_c, dtype=np.float64)
        if lane.centre is not None and np.array_equal(z, lane.centre):
            return False
        lane.centre = z.copy()
        self.centres.append(lane.centre)
        return True

    def rho_bar(self, lane: _Lane) -> float:
        lo, hi = self.settings.rho_clip
        return float(np.clip(lane.rho_ema, lo, hi))

    def _update(self, lane: _Lane) -> None:
        lo, hi = self.settings.rho_clip
        alpha = self.settings.rho_ema
        log = lane.run.log
        for entry in log[lane.seen:]:
            rho = entry.get("rho")
            if rho is None or math.isnan(rho):
                continue
            observed = float(np.clip(rho, lo, hi))
            lane.rho_ema = alpha * observed + (1.0 - alpha) * lane.rho_ema
            lane.rho_history.append([int(entry.get("calls", -1)),
                                     round(self.rho_bar(lane), 6)])
        lane.seen = len(log)

    def iteration_cost(self, run) -> int:
        cost = getattr(run, "next_iteration_cost", None)
        if callable(cost):
            return int(cost())
        return ((1 if run.status == "new" else 0)
                + (int(self.backend.jacobian_cost) if run.J_c is None else 0)
                + 1)

    def expected_reduction(self, lane: _Lane) -> float:
        run = lane.run
        predicted = run.log[-1].get("predicted", math.inf) if run.log else math.inf
        if predicted is None or math.isnan(predicted):
            # A NaN key would make max() in ``pick`` order-dependent.
            return -math.inf
        return self.rho_bar(lane) * predicted / max(1, self.iteration_cost(run))

    def is_candidate(self, lane: _Lane) -> bool:
        return (lane.run.status in ("new", "active") and not lane.dropped
                and not lane.stuck
                and self.iteration_cost(lane.run) <= self.remaining)

    def best_lane(self) -> _Lane | None:
        finite = [lane for lane in self.lanes if math.isfinite(lane.run.F_c)]
        return min(finite, key=lambda lane: lane.run.F_c) if finite else None

    # one iteration -------------------------------------------------------
    def _advance(self, lane: _Lane, limit: int) -> None:
        lane.run.advance(min(self.budget, int(limit)))
        self._note_centre(lane)

    def _drop_check(self, lane: _Lane) -> bool:
        run = lane.run
        lane.checked = np.asarray(run.z_c, dtype=np.float64).tobytes()
        best = self.best_lane()
        if best is None or best is lane:
            return False
        self.n_drop_checks += 1
        saved_delta, saved_model = run.delta, getattr(run, "model", None)
        try:
            run.delta = self.settings.drop_radius
            _z, predicted = run.solve_subproblem()
        finally:
            run.delta = saved_delta
            run.model = saved_model
        bound = run.F_c - float(predicted)
        if bound > best.run.F_c:
            lane.dropped = True
            self.drops.append({"label": lane.label,
                               "calls": int(self.backend.n_calls),
                               "F_c": float(run.F_c),
                               "pred_box": float(predicted),
                               "bound": float(bound),
                               "best_label": best.label,
                               "best_F": float(best.run.F_c)})
            return True
        return False

    def iterate(self, lane: _Lane) -> bool:
        """One iteration of ``lane``; False if it made no progress."""
        run = lane.run
        cost = self.iteration_cost(run)
        if cost > self.remaining:
            return False
        n0, status0 = int(self.backend.n_calls), run.status
        if self.settings.check_drops and run.J_c is None and cost > 1:
            self._advance(lane, n0 + cost - 1)
            if (run.status == "active" and run.J_c is not None
                    and np.asarray(run.z_c, dtype=np.float64).tobytes()
                    != lane.checked and self._drop_check(lane)):
                self._update(lane)
                self.n_iterations += 1
                return True
            if run.status in ("new", "active"):
                self._advance(lane, int(self.backend.n_calls) + 1)
        else:
            self._advance(lane, n0 + cost)
        self._update(lane)
        progressed = int(self.backend.n_calls) > n0 or run.status != status0
        if not progressed:
            lane.stuck = True
        else:
            self.n_iterations += 1
        return progressed

    # the race ------------------------------------------------------------
    def pick(self, candidates: Sequence[_Lane] | None = None) -> _Lane | None:
        """argmax of e_i over the candidates; ties go to the lower F_c."""
        if candidates is None:
            candidates = [lane for lane in self.lanes if self.is_candidate(lane)]
        if not candidates:
            return None
        return max(candidates, key=lambda lane: (
            self.expected_reduction(lane), -lane.run.F_c))

    def _restart(self) -> bool:
        n0 = int(self.backend.n_calls)
        run = self.restart_fn(self.centres, self.remaining)
        if run is None:
            return False
        lane = self.add(run)
        self.restarts.append({"label": lane.label, "calls_before": n0,
                              "calls_spent": int(self.backend.n_calls) - n0,
                              "status": run.status,
                              "F_c": float(run.F_c)})
        return True

    def run(self) -> str:
        """Initialization (one iteration per run, in order), then the race."""
        for lane in list(self.lanes):
            if self.is_candidate(lane):
                self.iterate(lane)
        while self.remaining > 0:
            candidates = [lane for lane in self.lanes if self.is_candidate(lane)]
            if not candidates:
                if self.restart_fn is None:
                    self.stop_reason = "no_candidate"
                    break
                if self.remaining < int(self.backend.jacobian_cost) + 2:
                    self.stop_reason = "budget_below_restart_cost"
                    break
                if not self._restart():
                    self.stop_reason = "restart_pool_empty"
                    break
                continue
            self.iterate(self.pick(candidates))
        if self.stop_reason is None:
            self.stop_reason = "budget"
        self.unspent_calls = self.remaining
        return self.stop_reason

    def summary(self) -> dict:
        return {"n_runs": len(self.lanes), "n_drops": len(self.drops),
                "n_drop_checks": self.n_drop_checks,
                "n_restarts": len(self.restarts),
                "n_iterations": self.n_iterations,
                "stop_reason": self.stop_reason,
                "unspent_calls": self.unspent_calls,
                "drops": self.drops, "restarts": self.restarts,
                "rho_bar_history": {lane.label: lane.rho_history
                                    for lane in self.lanes},
                "final_rho_bar": {lane.label: round(self.rho_bar(lane), 6)
                                  for lane in self.lanes},
                "dropped": [lane.label for lane in self.lanes if lane.dropped]}
