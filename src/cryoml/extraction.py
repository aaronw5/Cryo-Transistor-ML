"""Model-card extraction for the sky130 77 K devices (I25-1 main method).

Every parameter value is fitted against the measured I-V curves with NGSpice
in the loop; there is no learned correction and no surrogate.  The procedure
for one device is::

    for start k = 0, 1, 2:
        range = initial_range(names, template card)        # round-1 range
        x = draw_start(...)                                # random card in the range;
                                                           # redraw until every scored curve simulates
        for round = 1, 2, 3:
            x = cmaes_search(x, range)                     # loss: device RRMS
            x = trf_least_squares(x, range)                # loss: sum of squared point residuals,
                                                           # finite-difference Jacobian
            if round < 3: range = next_range(range, x)     # centred on x, half log width
        x = trf_least_squares(x, range)                    # FD polish in the round-3 range
    k* = select_start(...)                                 # lowest device RRMS among finite starts
    card = trf_least_squares(card of k*, range of k*)      # final FD refit

``fit_round`` is one round (start score, CMA-ES, TRF), ``fd_polish`` one
stand-alone TRF call from a card, ``fit_start`` one start (draw, rounds, FD
polish), ``fit_devices`` / ``fit_device`` the whole procedure.

The template is the released 77 K card files of Beall et al. with every
multiplier set to 1 (``CRYOML_PDK_TEMPLATE=mult1``; files in
``data/processed/pdk77k_mult1/``).  ``spice_pdk`` reads the template once at
import, so the variable has to be set before ``cryoml`` is imported;
``check_template`` raises if it was not.

The device RRMS is the paper metric with the published card's frozen curve set
(``card_eval.score_card``).  During the search a curve NGSpice cannot simulate
counts as RRMS 10 (``ngspice_fit.PENALTY``); the reported RRMS of such a card is
``inf``.  Run counts are the NGSpice runs of the optimizers
(``DeviceProblem.n_sim``); the one scoring run per stage is not counted.

Command line: ``scripts/extract_cards.py``.  Defaults reproduce I25-1 arm A
(``SEVEN``) and arm B (``FORTY_FIVE``); see ``docs/METHODS.md``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import spice_pdk
from .card_eval import device_by_tag, include_tags
from .cold_start import device_bins, half_width, next_round_spec, spec_from_centers
from .devices import PAPER_DEVICES
from .ngspice_fit import DeviceProblem, final_score, polish
from .parameter_sets import PARAMS7, PARAMS45
from .parameter_space import ParameterBox
from .utils import device_tag

# Key of the single device inside the one-device range specs passed to
# ``cold_start.spec_from_centers`` / ``next_round_spec``.
_ONE = "device"

STAGE_NAMES = ("round 1", "round 2", "round 3", "round 3 + FD")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractionConfig:
    """Every setting of the procedure.  The defaults are the I25-1 values;
    ``SEVEN`` and ``FORTY_FIVE`` set the parameter set and round budgets."""

    parameters: tuple[str, ...] = PARAMS7
    # Per round: CMA-ES evaluations, then the nominal NGSpice budget of TRF.
    cmaes_evals: int = 3000
    trf_nfev: int = 400
    rounds: int = 3
    # Nominal NGSpice budgets of the FD polish after the last round and of the
    # final FD refit of the kept start.
    polish_nfev: int = 400
    refit_nfev: int = 400
    # Starts 0 .. n_starts-1 always run.  If none of them ends finite, starts
    # n_starts, n_starts+1, ... run one at a time up to max_start_index.
    n_starts: int = 3
    max_start_index: int = 9
    # Start draw j of start k on device i comes from
    # default_rng(start_seed_base + start_seed_stride * k + i); at most
    # max_draws draws per start.
    start_seed_base: int = 20261027
    start_seed_stride: int = 1000
    max_draws: int = 200
    # CMA-ES (Python ``cma``) in local coordinates [-1, 1]; the same seed in
    # every round and start.  popsize None means 4 + floor(3 ln n_free).
    cmaes_seed: int = 20261015
    sigma0: float = 0.3
    popsize: int | None = None
    # TRF (SciPy least_squares, method "trf"): absolute forward-difference
    # step in local units (backward at the upper bound), IRLS passes.
    fd_step: float = 0.01
    irls_passes: int = 2
    # Range update between rounds: half log width times range_shrink, except
    # coordinates with |local| >= range_edge, which keep their width.
    range_shrink: float = 0.5
    range_edge: float = 0.9
    # Card template the parameters are written into (spice_pdk.PDK_TEMPLATES).
    template: str = "mult1"

    def popsize_for(self, n_free: int) -> int:
        """CMA-ES population size for ``n_free`` free parameters."""
        if self.popsize:
            return int(self.popsize)
        return 4 + int(3 * np.log(n_free))

    def to_json(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


SEVEN = ExtractionConfig(parameters=PARAMS7, cmaes_evals=3000, trf_nfev=400)
FORTY_FIVE = ExtractionConfig(parameters=PARAMS45, cmaes_evals=8000, trf_nfev=600)
CONFIGS = {7: SEVEN, 45: FORTY_FIVE}


def check_template(config: ExtractionConfig) -> None:
    """Raise if the card template loaded by ``spice_pdk`` is not
    ``config.template``."""
    if spice_pdk.PDK_TEMPLATE != config.template:
        raise RuntimeError(
            f"card template is {spice_pdk.PDK_TEMPLATE!r}, the configuration needs "
            f"{config.template!r}: set CRYOML_PDK_TEMPLATE={config.template} before "
            "importing cryoml")


# ---------------------------------------------------------------------------
# Search ranges
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchRange:
    """Log-scale range of each parameter of one device: ``center * exp(+-h)``
    with ``h = half_width[name]`` (natural log).  A parameter whose center is 0
    is fixed at 0."""

    names: tuple[str, ...]
    center: Mapping[str, float]
    half_width: Mapping[str, float]
    round_index: int = 1

    def spec(self) -> dict:
        """One-device range in the ``cold_start`` spec layout."""
        return spec_from_centers(self.names, {_ONE: dict(self.center)},
                                 {_ONE: dict(self.half_width)},
                                 extra={"round": self.round_index})

    def box(self) -> ParameterBox:
        """The range as a ``ParameterBox`` (local coordinates in [-1, 1],
        log-uniform between the bounds)."""
        spec = self.spec()
        flags = spec["log_scale_by_device"][_ONE]
        return ParameterBox(names=self.names, published=spec["published_by_device"][_ONE],
                            bounds=spec["bounds_by_device"][_ONE],
                            log_scale=tuple(bool(flags[n]) for n in self.names))

    def to_json(self) -> dict:
        spec = self.spec()
        return {"round": self.round_index, "center": dict(spec["published_by_device"][_ONE]),
                "half_width": dict(self.half_width),
                "bounds": spec["bounds_by_device"][_ONE],
                "log_scale": spec["log_scale_by_device"][_ONE]}


def initial_range(names: Sequence[str], center: Mapping[str, float]) -> SearchRange:
    """Round-1 range centred on the template card values ``center``.
    Half widths (``cold_start.half_width``): VTH0 ln 3, the other six paper
    parameters ln 100, every other parameter ln 3."""
    names = tuple(names)
    return SearchRange(names, {n: float(center[n]) for n in names},
                       {n: half_width(n) for n in names}, round_index=1)


def next_range(previous: SearchRange, card: Mapping[str, float], local: Sequence[float], *,
               shrink: float = 0.5, edge: float = 0.9) -> SearchRange:
    """Range of the next round (``cold_start.next_round_spec``): centred on
    ``card``; each half width is multiplied by ``shrink`` (not below
    ``cold_start.MIN_HALF_WIDTH``) unless the card's local coordinate has
    ``|local| >= edge``, in which case the half width is kept."""
    spec = next_round_spec(previous.spec(), {_ONE: dict(card)},
                           {_ONE: dict(zip(previous.names, (float(v) for v in local)))},
                           shrink=shrink, edge=edge, round_index=previous.round_index + 1)
    return SearchRange(previous.names, dict(spec["published_by_device"][_ONE]),
                       dict(spec["half_width_by_device"][_ONE]),
                       round_index=previous.round_index + 1)


def template_values(tag: str, names: Sequence[str]) -> dict[str, float]:
    """Values of ``names`` in the device's bin of the loaded template, read
    back from NGSpice (``showmod``; one NGSpice run)."""
    from .pdk_extract import read_bin_parameters
    device = device_by_tag(tag)
    return read_bin_parameters(device.dev_type, device.L_um, device.W_um,
                               device_bins()[tag], tuple(names))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class StageRecord:
    """Card after one stage and its scores.  ``rrms``: device RRMS (frozen
    curve set; inf if a scored curve does not simulate).  ``objective``: the
    optimizer's value (failed curve = 10).  ``n_sim``: NGSpice runs of the
    stage's optimizers."""

    name: str
    rrms: float
    official_dynamic_rrms: float
    objective: float
    params: dict[str, float]
    local: list[float]
    per_curve: dict[str, float]
    n_sim: int
    runtime_s: float
    start_objective: float | None = None
    cmaes_best: float | None = None
    cmaes_evals: int | None = None
    trf_history: list[float] = field(default_factory=list)

    @property
    def finite(self) -> bool:
        return bool(np.isfinite(self.rrms))

    def to_json(self) -> dict:
        return _jsonable(self.__dict__)


@dataclass
class StartRecord:
    """One start: the drawn start card, every stage, and every range."""

    tag: str
    k: int
    seed: int
    redraws: int                 # draws rejected before the accepted one
    draw_runs: int               # NGSpice runs of the draw check
    start_local: list[float]
    stages: list[StageRecord]    # round 1 .. round R, then the FD polish
    ranges: list[SearchRange]    # round 1 .. round R

    @property
    def result(self) -> StageRecord:
        return self.stages[-1]

    @property
    def eligible(self) -> bool:
        """Every scored curve of the start's final card simulates."""
        return self.result.finite

    @property
    def n_sim(self) -> int:
        return self.draw_runs + sum(s.n_sim for s in self.stages)

    def to_json(self) -> dict:
        return {"tag": self.tag, "k": self.k, "seed": self.seed, "redraws": self.redraws,
                "draw_runs": self.draw_runs, "start_local": list(self.start_local),
                "eligible": self.eligible, "n_sim": self.n_sim,
                "stages": [s.to_json() for s in self.stages],
                "ranges": [r.to_json() for r in self.ranges]}


@dataclass
class DeviceResult:
    """Final card of one device and the records that produced it."""

    tag: str
    parameters: tuple[str, ...]
    selected_start: int
    selected_eligible: bool
    final: StageRecord
    final_range: SearchRange
    starts: list[StartRecord]

    @property
    def card(self) -> dict[str, float]:
        return dict(self.final.params)

    @property
    def stage_rrms(self) -> dict[str, float]:
        """Device RRMS of the kept start after each stage, then the final refit."""
        start = self._selected()
        out = {name: s.rrms for name, s in zip(_stage_names(len(start.stages) - 1), start.stages)}
        out["final refit"] = self.final.rrms
        return out

    @property
    def ngspice_runs(self) -> dict[str, int]:
        """Optimizer NGSpice runs: all starts (with draw checks), the kept
        start (without its draw checks), the final refit, and the total."""
        start = self._selected()
        all_starts = sum(s.n_sim for s in self.starts)
        return {"all_starts": all_starts,
                "selected_start": sum(s.n_sim for s in start.stages),
                "final_refit": self.final.n_sim,
                "total": all_starts + self.final.n_sim}

    def _selected(self) -> StartRecord:
        return next(s for s in self.starts if s.k == self.selected_start)

    def to_json(self) -> dict:
        return {"tag": self.tag, "parameters": list(self.parameters),
                "rrms": self.final.rrms,
                "official_dynamic_rrms": self.final.official_dynamic_rrms,
                "card": self.card, "selected_start": self.selected_start,
                "selected_eligible": self.selected_eligible,
                "stage_rrms": self.stage_rrms,
                "start_rrms": {str(s.k): s.result.rrms for s in self.starts},
                "ngspice_runs": self.ngspice_runs,
                "final": self.final.to_json(), "final_range": self.final_range.to_json(),
                "starts": [s.to_json() for s in self.starts]}


def _stage_names(rounds: int) -> list[str]:
    return [f"round {r}" for r in range(1, rounds + 1)] + [f"round {rounds} + FD"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stage_record(name: str, problem: DeviceProblem, local: np.ndarray, objective: float,
                  t0: float, **extra) -> StageRecord:
    scored = final_score(problem, local)
    return StageRecord(name=name, rrms=float(scored["rrms"]),
                       official_dynamic_rrms=float(scored["official_dynamic_rrms"]),
                       objective=float(objective), params=dict(scored["params"]),
                       local=list(scored["local"]), per_curve=dict(scored["per_curve"]),
                       n_sim=int(problem.n_sim), runtime_s=round(time.time() - t0, 1), **extra)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def device_index(tag: str) -> int:
    """Index of the device in ``devices.PAPER_DEVICES``."""
    return [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES].index(tag)


def start_seed(config: ExtractionConfig, tag: str, k: int) -> int:
    """Seed of the start-draw stream of start ``k`` of device ``tag``."""
    return config.start_seed_base + config.start_seed_stride * k + device_index(tag)


def scored_curves_finite(problem, sims: Sequence[np.ndarray]) -> bool:
    """True if every scored (frozen-set) curve of ``sims`` is finite after
    the metric trim."""
    return all(np.all(np.isfinite(np.asarray(sims[row["index"]], dtype=np.float64)[row["start"]:]))
               for row in problem.layout)


@dataclass
class StartDraw:
    local: np.ndarray
    redraws: int
    runs: int


def draw_start(problem, rng: np.random.Generator, max_draws: int = 200) -> StartDraw:
    """Random start card: each free local coordinate uniform in [-1, 1]
    (log-uniform between the bounds); fixed coordinates at the range center.
    Draws continue from ``rng`` until NGSpice simulates every scored curve of
    the card (one run per draw).  Raises after ``max_draws`` draws."""
    active = np.flatnonzero(problem.box.active)
    runs0 = problem.n_sim
    for j in range(max_draws):
        local = np.asarray(problem.box.local_published, dtype=np.float64).copy()
        local[active] = rng.uniform(-1.0, 1.0, size=len(active))
        if scored_curves_finite(problem, problem.simulate(local)):
            return StartDraw(local, j, problem.n_sim - runs0)
    raise RuntimeError(f"{problem.tag}: no start card with every scored curve finite "
                       f"in {max_draws} draws")


@dataclass
class SearchResult:
    local: np.ndarray
    value: float
    evals: int = 0
    history: list[float] = field(default_factory=list)


def cmaes_search(problem, local0: np.ndarray, *, evals: int, sigma0: float, seed: int,
                 popsize: int) -> SearchResult:
    """CMA-ES (Python ``cma``) on the device RRMS (``problem.objective``) over
    the free local coordinates, bounds [-1, 1], mean started at ``local0``,
    step size ``sigma0``; stops after ``evals`` evaluations (checked after each
    generation) or at cma's own criteria.  Returns the lowest-loss card."""
    import cma
    local0 = np.asarray(local0, dtype=np.float64)
    active = np.flatnonzero(problem.box.active)

    def loss(x):
        trial = local0.copy()
        trial[active] = np.clip(x, -1, 1)
        return problem.objective(trial)

    es = cma.CMAEvolutionStrategy(local0[active], sigma0, {
        "bounds": [-1, 1], "maxfevals": evals, "seed": seed, "verbose": -9,
        "popsize": popsize})
    while not es.stop():
        xs = es.ask()
        es.tell(xs, [loss(x) for x in xs])
    local = local0.copy()
    local[active] = np.clip(es.result.xbest, -1, 1)
    return SearchResult(local, float(es.result.fbest), int(es.result.evaluations))


def trf_least_squares(problem, local0: np.ndarray, *, max_nfev: int, fd_step: float,
                      passes: int) -> SearchResult:
    """Bounded TRF least squares (``ngspice_fit.polish``): SciPy
    ``least_squares(method="trf")`` on the per-point residuals
    (I_sim - I_meas) / (mean |I_meas| of the curve * sqrt(points * curves)),
    Jacobian by forward differences of ``fd_step`` in local units, ``passes``
    IRLS passes (pass p > 1 weights each curve by 1/sqrt(curve RRMS)), SciPy
    budget ``ngspice_fit.trf_pass_budget(max_nfev, passes, n_free)`` per pass.
    Returns the lowest device RRMS card seen, including ``local0``."""
    out = polish(problem, local0, max_nfev=max_nfev, diff_step=fd_step, irls_rounds=passes)
    return SearchResult(np.asarray(out["local"], dtype=np.float64), float(out["objective"]),
                        history=[float(v) for v in out["history"]])


def fit_round(tag: str, search_range: SearchRange, config: ExtractionConfig,
              local0: np.ndarray | None = None, name: str = "round 1") -> StageRecord:
    """One round in ``search_range``: score the start card (``local0``, or the
    range center when None), CMA-ES from it, TRF from the CMA-ES card.  Each
    step's card replaces the current one only if its loss is lower."""
    t0 = time.time()
    problem = DeviceProblem(tag, search_range.box())
    local = np.asarray(problem.box.local_published if local0 is None else local0,
                       dtype=np.float64)
    value = start_value = problem.objective(local)
    extra: dict = {"start_objective": start_value}
    if config.cmaes_evals > 0:
        cm = cmaes_search(problem, local, evals=config.cmaes_evals, sigma0=config.sigma0,
                          seed=config.cmaes_seed,
                          popsize=config.popsize_for(len(np.flatnonzero(problem.box.active))))
        extra.update(cmaes_best=cm.value, cmaes_evals=cm.evals)
        if cm.value < value:
            local, value = cm.local, cm.value
    if config.trf_nfev > 0:
        trf = trf_least_squares(problem, local, max_nfev=config.trf_nfev,
                                fd_step=config.fd_step, passes=config.irls_passes)
        extra["trf_history"] = trf.history
        if trf.value < value:
            local, value = trf.local, trf.value
    return _stage_record(name, problem, local, value, t0, **extra)


def fd_polish(tag: str, search_range: SearchRange, card: Mapping[str, float],
              config: ExtractionConfig, max_nfev: int, name: str) -> StageRecord:
    """One TRF least-squares call from ``card`` in ``search_range`` (the FD
    polish after the last round, and the final FD refit)."""
    t0 = time.time()
    problem = DeviceProblem(tag, search_range.box())
    start = problem.box.params_to_local(
        {**search_range.center, **{k: v for k, v in card.items() if k in problem.box.names}})
    trf = trf_least_squares(problem, start, max_nfev=max_nfev, fd_step=config.fd_step,
                            passes=config.irls_passes)
    return _stage_record(name, problem, trf.local, trf.value, t0, trf_history=trf.history)


def fit_start(tag: str, k: int, config: ExtractionConfig, range1: SearchRange) -> StartRecord:
    """Start ``k``: draw the start card in ``range1``, run rounds
    1 .. ``config.rounds`` (round r > 1 starts at the previous card, the
    center of its range), then the FD polish in the last range."""
    check_template(config)
    seed = start_seed(config, tag, k)
    draw = draw_start(DeviceProblem(tag, range1.box()), np.random.default_rng(seed),
                      config.max_draws)
    names = _stage_names(config.rounds)
    ranges, stages = [range1], []
    for r in range(1, config.rounds + 1):
        if r > 1:
            prev = stages[-1]
            ranges.append(next_range(ranges[-1], prev.params, prev.local,
                                     shrink=config.range_shrink, edge=config.range_edge))
        stages.append(fit_round(tag, ranges[-1], config,
                                draw.local if r == 1 else None, name=names[r - 1]))
    stages.append(fd_polish(tag, ranges[-1], stages[-1].params, config,
                            config.polish_nfev, name=names[-1]))
    return StartRecord(tag, k, seed, draw.redraws, draw.runs, draw.local.tolist(), stages, ranges)


def select_start(starts: Iterable[StartRecord], n_primary: int) -> int | None:
    """Kept start: among eligible starts with k < ``n_primary``, the one with
    the lowest final device RRMS (lowest k on ties); if none, the lowest
    eligible k >= ``n_primary``; None if no start is eligible."""
    eligible = sorted((s for s in starts if s.eligible), key=lambda s: s.k)
    primary = [s for s in eligible if s.k < n_primary]
    if primary:
        return min(primary, key=lambda s: s.result.rrms).k
    return eligible[0].k if eligible else None


# ---------------------------------------------------------------------------
# Whole procedure
# ---------------------------------------------------------------------------
def _start_task(job) -> StartRecord:
    tag, k, config, range1 = job
    return fit_start(tag, k, config, range1)


def _refit_task(job) -> StageRecord:
    tag, search_range, card, config = job
    check_template(config)
    return fd_polish(tag, search_range, card, config, config.refit_nfev, name="final refit")


MapFn = Callable[[Callable, list], Iterable]


def fit_devices(tags: Sequence[str], config: ExtractionConfig,
                ranges: Mapping[str, SearchRange] | None = None,
                map_fn: MapFn = map, log: Callable[[str], None] | None = None
                ) -> dict[str, DeviceResult]:
    """Run the procedure on every device of ``tags``.  ``ranges``: round-1
    range per device (default ``initial_range`` around ``template_values``).
    ``map_fn(fn, jobs)`` evaluates independent jobs (e.g. ``Pool.map``); all
    (device, start) pairs form one batch, extra starts and the final refits
    one batch each."""
    check_template(config)
    tags = list(tags)
    if ranges is None:
        ranges = {t: initial_range(config.parameters, template_values(t, config.parameters))
                  for t in tags}
    starts: dict[str, list[StartRecord]] = {t: [] for t in tags}

    def run(pairs):
        for rec in map_fn(_start_task, [(t, k, config, ranges[t]) for t, k in pairs]):
            starts[rec.tag].append(rec)
            if log:
                log(f"{rec.tag} start {rec.k}: redraws {rec.redraws}, "
                    f"RRMS {rec.result.rrms:.6g}, runs {rec.n_sim}")

    run([(t, k) for t in tags for k in range(config.n_starts)])
    for k in range(config.n_starts, config.max_start_index + 1):
        need = [t for t in tags if select_start(starts[t], config.n_starts) is None]
        if not need:
            break
        run([(t, k) for t in need])
    chosen = {}
    for t in tags:
        k = select_start(starts[t], config.n_starts)
        chosen[t] = (min(s.k for s in starts[t]) if k is None else k, k is not None)
    jobs = []
    for t in tags:
        rec = next(s for s in starts[t] if s.k == chosen[t][0])
        jobs.append((t, rec.ranges[-1], rec.result.params, config))
    finals = list(map_fn(_refit_task, jobs))
    out = {}
    for (t, search_range, _, _), final in zip(jobs, finals):
        out[t] = DeviceResult(t, tuple(config.parameters), chosen[t][0], chosen[t][1], final,
                              search_range, sorted(starts[t], key=lambda s: s.k))
        if log:
            log(f"{t} final: start {chosen[t][0]}, RRMS {final.rrms:.6g}")
    return out


def fit_device(tag: str, config: ExtractionConfig, range1: SearchRange | None = None,
               map_fn: MapFn = map) -> DeviceResult:
    """``fit_devices`` for one device."""
    return fit_devices([tag], config, None if range1 is None else {tag: range1},
                       map_fn)[tag]


# ---------------------------------------------------------------------------
# Card files
# ---------------------------------------------------------------------------
LIBRARY_NAME = "sky130_77k_extracted.lib.spice"


def write_card_library(cards: Mapping[str, Mapping[str, float]], out_dir: Path) -> Path:
    """Write the template corner cards with each device's bin set to its
    fitted values, and a library file (section ``tt_77k``) that includes
    them.  The values are written after the same BSIM4 A1/A2 clamp
    (``spice_pdk.bsim4_clamp_params``) that ``simulate_pdk`` applies during the
    fit.  Devices not in ``cards`` keep the template values.  Returns the
    library path."""
    err = spice_pdk.ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = device_bins()
    library = spice_pdk.LIB_FILE.read_text()
    for dev_type in ("nmos", "pmos"):
        base = spice_pdk.PDK77K_DIR / spice_pdk._CORNER_BASENAME[dev_type]
        text = base.read_text(errors="replace")
        for tag in sorted(cards):
            if device_by_tag(tag).dev_type != dev_type:
                continue
            params, _ = spice_pdk.bsim4_clamp_params(dev_type, bins[tag], dict(cards[tag]))
            text = spice_pdk._patch_params_in_bin(text, dev_type, bins[tag], params)
        path = out_dir / spice_pdk._CORNER_BASENAME[dev_type]
        path.write_text(text)
        library = library.replace(base.as_posix(), path.resolve().as_posix())
    lib_path = out_dir / LIBRARY_NAME
    lib_path.write_text(library)
    return lib_path


def score_library(lib_path: Path, tag: str) -> float:
    """Device RRMS (frozen curve set) of ``tag`` simulated from the card
    library ``lib_path`` with no parameter injection."""
    from .data_io import load_device_curves
    from .metrics import score_device_new
    device = device_by_tag(tag)
    curves = load_device_curves(device)
    sims = spice_pdk.simulate_pdk(device.dev_type, device.L_um, device.W_um, curves,
                                  library_path=Path(lib_path))
    value = float(score_device_new(device.dev_type, device.L_um, device.W_um, curves, sims,
                                   include_tags=include_tags(tag))["rrms"])
    return value if math.isfinite(value) else float("inf")
