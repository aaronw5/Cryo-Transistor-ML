#!/usr/bin/env python3
"""DEV headline comparison: every DEV arm reports through this one tool.

DEV = the held-out simulated targets ``set_0`` and ``set_1``
(``CRYOML_TARGET_OVERRIDE``) on the full round-1 43-parameter box: 36 targets
(18 per set).  Method selection and tuning look only at this table; TEST (the
18 real devices and ``set_2``) is run once per frozen method elsewhere.

Input is a registry (``out/research/dev/arms.json``): ``sets`` and a list of
``arms``, each ``{label, dir_template ({set}, {budget}), method_key, mode,
budgets}`` plus reference entries (``mode: "reference"``).

* ``mode: "trace"``: read the largest-budget directory; the value at budget b
  is ``best_at(trace, b)`` (the in-loop best after b NGSpice calls).
* ``mode: "fixed"``: one directory per budget; the value is the record's
  ``rrms`` (one fresh ``fps.score_real`` of the final card).  The curve still
  comes from the largest-budget trace.
* References: the bank oracle (0 NGSpice calls: the best sample of the
  pre-paid round bank, shared by all arms), the round-1 paper pipeline (its
  NGSpice validation calls), and pipeline + FD polish when present.

Primary metric: all-device mean paper RRMS (frozen inclusion) pooled over
the 36 targets at each budget.  Secondary (DEV means are dominated by a few
stuck devices, so always shown; no device is ever excluded): per-set
all-device mean and combined (mean of nMOS / pMOS family means), median,
geometric mean (floor 1e-6), worst target, fraction below 1e-2 / 1e-3.  AUC =
per-target mean of log10(max(best, 1e-6)) over a 40-point log grid of calls
(10 to the largest budget), averaged over targets.  Paired tests against a
reference arm: wins/ties/losses (tie when |diff| < max(1e-6, 1% of the
larger value)), Wilcoxon signed-rank p, bootstrap 95% CI of the mean
difference (10k resamples over targets, seed 0).

Strict curve check (reported, never a selection): a target whose record has
a non-finite included (frozen-inclusion) metric curve has no valid paper
RRMS (``metrics.device_rrms_new`` averages the surviving curves; upstream
``rrmsCalc.py`` gives NaN).  Every arm/budget and reference reports the
number of failed targets, the pooled mean over valid targets only and the
penalized pooled mean (failed curves at RRMS 10, ``cryoml.strict_score``);
the primary values are left as stored.  Trace-mode budgets below the top
have no record of their card, so their status is not checked.

Hard checks (each printed; any failure exits 1 unless ``--allow-partial``):
18 records per set; every ``n_ngspice_calls`` within its budget; the fresh
``rrms`` equals ``best_at(trace, budget)`` (a gap up to 1e-4 relative on a
record whose in-loop best is a batched altermod probe is a WARN row, see
``BATCHED_PROBE_KINDS``); every run's ``summary.json`` says
``target_override`` true and every record's bank-oracle score equals the one
recomputed from that set's override targets (so TEST or real-device runs, or
a run on the other set, are refused); no cross-arm best-of.

  PYTHONPATH=src .venv/bin/python scripts/research/dev_compare.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from hybrid_tr import BUDGET_MARKS, best_at  # noqa: E402
from cryoml import strict_score as ss  # noqa: E402

DEV = ROOT / "out" / "research" / "dev"
N_DEVICES = 18
FLOOR = 1e-6
THRESHOLDS = (0.05, 0.01)
BANK_NOTE = "pre-paid bank 10k/geometry, shared by all arms"
# In-loop best kinds scored inside a multi-card altermod NGSpice process
# (hybrid_tr.NgspiceResiduals._simulate_many, cmaes_trace TRF probes).  A card
# that follows a non-converged card in the same process can differ from its
# own fresh single-process simulation (measured: pmos_L8_W0p84 on the sky130c
# targets, relative gap 4.4e-5).  Such a gap up to PROBE_DRIFT_REL is reported
# as a WARN row; the record's fresh ``rrms`` stays the primary value.
BATCHED_PROBE_KINDS = ("jacobian_probe", "trf_jacobian_probe")
PROBE_DRIFT_REL = 1e-4


# ----------------------------------------------------------------- statistics

def log_grid(top: int, n: int = 40, start: float = 10.0) -> np.ndarray:
    return np.geomspace(start, top, n)


def auc(traces: list[list], grid: np.ndarray) -> float:
    """Mean over targets of the mean over the grid of log10(max(best, 1e-6))."""
    return float(np.mean([np.mean([math.log10(max(best_at(t, g), FLOOR))
                                   for g in grid]) for t in traces]))


def calls_to_reach(trace: list, threshold: float) -> float:
    for n, rrms in trace:
        if rrms <= threshold:
            return float(n)
    return math.inf


def bootstrap_ci(diff, n_resamples: int = 10_000, seed: int = 0) -> list[float]:
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_resamples, len(diff)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired(arm: dict, ref: dict) -> dict:
    """``arm`` vs ``ref`` over their common targets; diff = arm - ref
    (negative = the arm is better)."""
    from scipy.stats import wilcoxon
    common = sorted(set(arm) & set(ref))
    a = np.array([arm[t] for t in common], dtype=np.float64)
    b = np.array([ref[t] for t in common], dtype=np.float64)
    diff = a - b
    tie = np.abs(diff) < np.maximum(1e-6, 0.01 * np.maximum(np.abs(a), np.abs(b)))
    if not np.any(diff[np.isfinite(diff)] != 0):
        p = 1.0                     # no nonzero difference: no evidence
    else:
        try:
            with np.errstate(all="ignore"):
                p = float(wilcoxon(a, b).pvalue)
        except ValueError:          # e.g. non-finite values
            p = float("nan")
    finite = np.isfinite(diff)
    return {"n": len(common), "wins": int(np.sum(~tie & (diff < 0))),
            "ties": int(np.sum(tie)), "losses": int(np.sum(~tie & (diff > 0))),
            "mean_diff": float(np.mean(diff)) if len(diff) else float("nan"),
            "ci95": (bootstrap_ci(diff[finite]) if finite.any()
                     else [float("nan")] * 2),
            "wilcoxon_p": p}


def pooled(values: dict) -> dict:
    keys = sorted(values)
    v = np.array([values[k] for k in keys], dtype=np.float64)
    worst = int(np.argmax(v))
    return {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
            "geomean": float(np.exp(np.mean(np.log(np.maximum(v, FLOOR))))),
            "worst": float(v[worst]), "worst_target": keys[worst],
            "frac_below_1e-2": float(np.mean(v < 1e-2)),
            "frac_below_1e-3": float(np.mean(v < 1e-3))}


def per_set(values: dict, sets: list[str]) -> dict:
    from cryoml.card_eval import aggregate
    out = {}
    for name in sets:
        rows = {k.split("/", 1)[1]: {"rrms": v, "official_dynamic_rrms": math.nan}
                for k, v in values.items() if k.startswith(name + "/")}
        if not rows:
            continue
        agg = aggregate(rows)
        if not all(math.isfinite(r["rrms"]) for r in rows.values()):
            # family_totals drops non-finite devices; never exclude one.
            agg["combined_rrms"] = agg["nmos_rrms"] = agg["pmos_rrms"] = math.inf
        out[name] = {k: agg[k] for k in ("all_device_mean_rrms", "combined_rrms",
                                          "nmos_rrms", "pmos_rrms", "n_devices")}
    return out


def target_status(record: dict, tag: str) -> dict | None:
    """Strict status of one DEV target record (None: device not in the
    frozen baseline, e.g. a test fixture)."""
    try:
        include = ss.include_tags(tag)
    except KeyError:
        return None
    return ss.record_status(record, tag, include)


def strict_summary(values: dict, status: dict, sets: list[str]) -> dict:
    """Failed-target count, valid-only and penalized pooled means.  Targets
    without a status (unchecked) are counted, never dropped silently."""
    known = {k: st for k, st in status.items() if st is not None and k in values}
    agg = ss.aggregate(known, {k: values[k] for k in known})
    unchecked = sorted(set(values) - set(known))
    return {"n_targets": len(values), "n_checked": len(known),
            "n_unchecked": len(unchecked),
            "n_failed": agg["n_failed_devices"],
            "failed_targets": {k: agg["n_failed_curves"][k]
                               for k in agg["failed_devices"]},
            "per_set_failed": {name: sum(1 for k in agg["failed_devices"]
                                         if k.startswith(name + "/"))
                               for name in sets},
            "valid_mean": agg["valid"]["all_device_mean_rrms"],
            "n_valid": agg["n_valid_devices"],
            "penalized_mean": (agg["penalized"]["all_device_mean_rrms"]
                               if not unchecked else math.nan)}


# --------------------------------------------------------------------- checks

class Checks:
    def __init__(self, allow_partial: bool):
        self.allow_partial = allow_partial
        self.rows: list[dict] = []

    def warn(self, name: str, detail: str = "") -> None:
        """A reported finding that does not fail the run."""
        self.rows.append({"check": name, "level": "WARN", "detail": detail})
        print(f"[WARN] {name}{': ' + detail if detail else ''}", flush=True)

    def __call__(self, name: str, ok: bool, detail: str = "",
                 partial_ok: bool = False) -> bool:
        level = "PASS" if ok else ("WARN" if partial_ok and self.allow_partial
                                   else "FAIL")
        self.rows.append({"check": name, "level": level, "detail": detail})
        print(f"[{level}] {name}{': ' + detail if detail else ''}", flush=True)
        return ok

    @property
    def failed(self) -> bool:
        return any(r["level"] == "FAIL" for r in self.rows)


class OracleReference:
    """The bank-oracle paper RRMS per (set, device), recomputed from the
    round bank and that set's override targets (no NGSpice), cached."""

    def __init__(self, registry: dict, path: Path, enabled: bool):
        self.enabled = enabled
        self.path = path
        self.bank_name = str(registry["bank"])
        self.bank = ROOT / registry["bank"]
        self.round_dir = ROOT / registry["round_dir"]
        self.targets = registry["targets_template"]
        self.values = (json.loads(path.read_text())
                       if enabled and path.exists() else {})
        self.seen: dict[tuple, float] = {}
        self._fingerprints: dict[str, str] = {}

    def key(self, set_name: str, tag: str) -> str:
        """Cache key: bank, a hash of that set's target npz, set and device,
        so a changed bank or regenerated targets never reuse stale values."""
        if set_name not in self._fingerprints:
            import hashlib
            target = ROOT / self.targets.format(set=set_name)
            digest = (hashlib.sha1(target.read_bytes()).hexdigest()[:16]
                      if target.exists() else "missing")
            self._fingerprints[set_name] = f"{self.bank_name}|{digest}"
        return f"{self._fingerprints[set_name]}|{set_name}/{tag}"

    def get(self, set_name: str, tag: str) -> float | None:
        if not self.enabled:
            return self.seen.get((set_name, tag))
        key = self.key(set_name, tag)
        if key not in self.values:
            import foundation_parameter_study as fps
            from cryoml.card_eval import device_by_tag
            from cryoml.data_io import TARGET_OVERRIDE_ENV
            metadata = json.loads(
                (self.round_dir / "cache" / "metadata.json").read_text())
            previous = os.environ.get(TARGET_OVERRIDE_ENV)
            os.environ[TARGET_OVERRIDE_ENV] = str(
                ROOT / self.targets.format(set=set_name))
            try:
                context = fps.build_device_context(
                    device_by_tag(tag), self.bank, metadata,
                    metadata["parameter_set"])
                self.values[key] = fps.diverse_oracle_starts(context, 1)[1][
                    "paper_rrms"]
            finally:
                if previous is None:
                    os.environ.pop(TARGET_OVERRIDE_ENV, None)
                else:
                    os.environ[TARGET_OVERRIDE_ENV] = previous
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic: other DEV compares may read this cache concurrently.
            partial = self.path.with_suffix(f".{os.getpid()}.tmp")
            partial.write_text(json.dumps(self.values, indent=1) + "\n")
            partial.replace(self.path)
        return self.values[key]

    def matches(self, set_name: str, tag: str, value: float) -> bool:
        expected = self.get(set_name, tag)
        if expected is None:            # disabled: first arm seen pins it
            self.seen[(set_name, tag)] = value
            return True
        return abs(value - expected) <= 1e-9 * max(1.0, abs(expected))


def load_run(directory: Path, method_key: str, budget: int | None,
             set_name: str, checks: Checks, oracle: OracleReference,
             expect: dict | None = None, needs_summary: bool = True,
             label: str = "") -> dict | None:
    """Records of one run directory, or None if the directory is refused."""
    where = f"{label} {set_name}" + (f" B{budget}" if budget else "")
    files = sorted(directory.glob("ml_*.json"))
    if not files:
        checks(f"{where}: records present", False, f"none in {directory}",
               partial_ok=True)
        return None
    try:
        records = {(row := json.loads(p.read_text()))["device"]:
                   row["methods"][method_key] for p in files}
    except KeyError:
        checks(f"{where}: method {method_key!r} in every record", False,
               str(directory))
        return None
    summary = directory / "summary.json"
    if not needs_summary:
        # The pipeline's summary.json carries no override flag; its identity
        # rests on the bank-oracle check below (real-device oracles differ).
        known = oracle.enabled or bool(oracle.seen)
        checks(f"{where}: identity via bank oracle only", known,
               "" if known else "no oracle reference to check against",
               partial_ok=True)
    elif summary.exists():
        flag = json.loads(summary.read_text()).get("target_override")
        if not checks(f"{where}: summary.json target_override", flag is True,
                      f"{flag!r}; directory refused" if flag is not True else ""):
            return None
    elif needs_summary:
        checks(f"{where}: summary.json present", False,
               f"partial run; DEV identity rests on the oracle check "
               f"({directory})", partial_ok=True)
        if not checks.allow_partial:
            return None
    bad = [t for t, r in records.items() if not oracle.matches(
        set_name, t, r["oracle_lhc"]["paper_rrms"])]
    if not checks(f"{where}: bank-oracle identity ({set_name} targets)",
                  not bad, f"mismatch {bad}; directory refused" if bad else ""):
        return None
    checks(f"{where}: {N_DEVICES} records", len(records) == N_DEVICES,
           f"{len(records)}", partial_ok=True)
    for key, value in (expect or {}).items():
        wrong = [t for t, r in records.items() if r.get(key) != value]
        checks(f"{where}: {key} == {value!r}", not wrong, f"{wrong}")
    if budget is not None:
        over = [t for t, r in records.items()
                if r["n_ngspice_calls"] > budget or r.get("budget") != budget]
        checks(f"{where}: n_ngspice_calls <= {budget}", not over, f"{over}")
        drift, probe_drift = [], []
        for t, r in records.items():
            gap = abs(r["rrms"] - best_at(r["trace"], budget))
            if gap <= 1e-6 * max(1.0, abs(r["rrms"])):
                continue
            if (r.get("best_kind") in BATCHED_PROBE_KINDS
                    and gap <= PROBE_DRIFT_REL * abs(r["rrms"])):
                probe_drift.append(f"{t} ({gap / abs(r['rrms']):.1e})")
            else:
                drift.append(t)
        checks(f"{where}: fresh rrms == best_at(trace, {budget})", not drift,
               f"{drift}")
        if probe_drift:
            checks.warn(f"{where}: batched-probe in-loop score != fresh "
                        f"single-process score (relative gap <= "
                        f"{PROBE_DRIFT_REL:g}; the fresh rrms is the value)",
                        f"{probe_drift}")
    return records


# ------------------------------------------------------------------- compare

def evaluate_arm(arm: dict, sets: list[str], checks: Checks,
                 oracle: OracleReference) -> dict | None:
    budgets = sorted(int(b) for b in arm["budgets"])
    top = budgets[-1]
    marks = budgets if arm["mode"] == "fixed" else [
        b for b in BUDGET_MARKS if b <= top]
    values = {b: {} for b in marks}
    calls = {b: {} for b in marks}
    status = {b: {} for b in marks}
    traces, dirs = {}, []
    for set_name in sets:
        runs = {}
        for b in (budgets if arm["mode"] == "fixed" else [top]):
            directory = ROOT / arm["dir_template"].format(set=set_name, budget=b)
            dirs.append(str(directory))
            records = load_run(directory, arm["method_key"], b, set_name,
                               checks, oracle, arm.get("expect"),
                               label=arm["label"])
            if records:
                runs[b] = records
        for tag, record in runs.get(top, {}).items():
            traces[f"{set_name}/{tag}"] = record["trace"]
        for b in marks:
            if arm["mode"] == "fixed":
                for tag, record in runs.get(b, {}).items():
                    values[b][f"{set_name}/{tag}"] = float(record["rrms"])
                    calls[b][f"{set_name}/{tag}"] = record["n_ngspice_calls"]
                    status[b][f"{set_name}/{tag}"] = target_status(record, tag)
            else:
                for tag, record in runs.get(top, {}).items():
                    values[b][f"{set_name}/{tag}"] = best_at(record["trace"], b)
                    calls[b][f"{set_name}/{tag}"] = min(
                        b, record["n_ngspice_calls"])
                    # Only the top budget's card is on record.
                    status[b][f"{set_name}/{tag}"] = (
                        target_status(record, tag) if b == top else None)
    values = {b: v for b, v in values.items() if v}
    if not values:
        return None
    return {"label": arm["label"], "mode": arm["mode"], "budgets": values,
            "calls": calls, "status": status, "traces": traces, "dirs": dirs,
            "top": top}


def load_references(registry: dict, sets: list[str], checks: Checks,
                    oracle: OracleReference, arms: dict) -> dict:
    out = {}
    for ref in registry["arms"]:
        if ref.get("mode") != "reference":
            continue
        kind, values, calls, status = ref["kind"], {}, {}, {}
        for set_name in sets:
            if kind == "bank_oracle":
                tags = sorted({k.split("/", 1)[1] for arm in arms.values()
                               for k in arm["traces"] if k.startswith(set_name + "/")})
                for tag in tags:
                    if oracle.enabled:
                        values[f"{set_name}/{tag}"] = oracle.get(set_name, tag)
                    else:
                        values[f"{set_name}/{tag}"] = oracle.seen[(set_name, tag)]
                    calls[f"{set_name}/{tag}"] = 0
                    status[f"{set_name}/{tag}"] = None   # bank currents, no card record
            elif kind == "pipeline":
                directory = ROOT / ref["dir_template"].format(set=set_name)
                records = load_run(directory, ref["method_key"], None, set_name,
                                   checks, oracle, needs_summary=False,
                                   label=ref["label"])
                for tag, record in (records or {}).items():
                    values[f"{set_name}/{tag}"] = float(record["rrms"])
                    calls[f"{set_name}/{tag}"] = int(record[ref["calls_key"]])
                    status[f"{set_name}/{tag}"] = target_status(record, tag)
            elif kind == "polished":
                path = ROOT / ref["summary_template"].format(set=set_name)
                if not path.exists():
                    continue
                summary = json.loads(path.read_text())
                base = next(r for r in registry["arms"]
                            if r["label"] == ref["base"])
                base_dir = base["dir_template"].format(set=set_name)
                # polish_cards stores --cards as given (relative to the repo
                # root, where the queue runs); resolve against ROOT, not cwd.
                ok = (summary.get("target_override") is True
                      and (ROOT / summary["cards"]).resolve()
                      == (ROOT / base_dir).resolve())
                if not checks(f"{ref['label']} {set_name}: target_override and "
                              f"cards == {base_dir}", ok, str(path)):
                    continue
                base_calls = out.get(ref["base"], {}).get("calls", {})
                for tag, row in summary["devices"].items():
                    key = f"{set_name}/{tag}"
                    values[key] = float(row["rrms"])
                    calls[key] = base_calls.get(key, 0) + int(row["n_sim"])
                    # polish summaries: failed_curves, else history[-1] vs rrms
                    status[key] = target_status(row, tag)
        if values:
            out[ref["label"]] = {"label": ref["label"], "kind": kind,
                                 "note": ref.get("note", ""), "values": values,
                                 "calls": calls, **pooled(values),
                                 "per_set": per_set(values, sets),
                                 "strict": strict_summary(values, status, sets),
                                 "mean_calls": float(np.mean(list(calls.values())))}
    return out


def summarize_arm(arm: dict, sets: list[str], grid: np.ndarray) -> dict:
    traces = [arm["traces"][k] for k in sorted(arm["traces"])]
    out = {"label": arm["label"], "mode": arm["mode"], "dirs": arm["dirs"],
           "budgets": {}}
    for b, values in arm["budgets"].items():
        out["budgets"][str(b)] = {
            **pooled(values), "per_set": per_set(values, sets),
            "strict": strict_summary(values, arm.get("status", {}).get(b, {}), sets),
            "mean_calls": float(np.mean(list(arm["calls"][b].values()))),
            "values": values}
    if traces:
        curves = np.array([[best_at(t, g) for g in grid] for t in traces])
        out["auc"] = auc(traces, grid)
        # The curve stops at the arm's own largest budget.
        shown = grid <= arm["top"] * (1 + 1e-9)
        out["curve"] = {"grid": grid[shown].tolist(),
                        "mean": curves.mean(0)[shown].tolist(),
                        "median": np.median(curves, 0)[shown].tolist(),
                        "n_targets": len(traces)}
        out["median_calls_to"] = {
            str(thr): float(np.median([calls_to_reach(t, thr) for t in traces]))
            for thr in THRESHOLDS}
    return out


def plot(arms: dict, refs: dict, path: Path, grid: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
              "#008300", "#6250d6", "#e34948"]
    ink, muted = "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for color, arm in zip(series, arms.values()):
        if "curve" not in arm:
            continue
        x = arm["curve"]["grid"]
        bad = max((s.get("strict", {}).get("n_failed", 0)
                   for s in arm["budgets"].values()), default=0)
        ax.plot(x, arm["curve"]["mean"], color=color, lw=2,
                label=f"{arm['label']} (mean, n={arm['curve']['n_targets']})"
                + (f" [{bad} failed target(s)]" if bad else ""))
        ax.plot(x, arm["curve"]["median"], color=color, lw=1.2, ls="--")
        if arm["mode"] == "fixed":
            bs = sorted(int(b) for b in arm["budgets"])
            ax.plot(bs, [arm["budgets"][str(b)]["mean"] for b in bs], "o",
                    color=color, ms=7, mec="#fcfcfb", mew=1.5)
    left = grid[0] / 2
    markers = {"bank_oracle": "D", "pipeline": "s", "polished": "^"}
    for ref in refs.values():
        x = left if ref["kind"] == "bank_oracle" else max(ref["mean_calls"], left)
        name = ref["label"] + (" (0 calls, " + BANK_NOTE + ")"
                               if ref["kind"] == "bank_oracle"
                               else f" ({ref['mean_calls']:.0f} calls)")
        if ref.get("strict", {}).get("n_failed"):
            name += f" [{ref['strict']['n_failed']} failed target(s)]"
        ax.plot([x], [ref["mean"]], markers.get(ref["kind"], "o"), color=ink,
                ms=8, mfc="none", mew=1.5, label=name)
        ax.plot([x], [ref["median"]], markers.get(ref["kind"], "o"),
                color=muted, ms=5, mew=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import (FuncFormatter, LogLocator,
                                   NullFormatter, NullLocator)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(left / 1.3, grid[-1] * 1.3)
    # The bank oracle costs 0 calls: its own "0" tick left of a break line.
    ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
             if t <= grid[-1] * 1.01]
    ax.set_xticks([left] + ticks, ["0"] + [f"{t:g}" for t in ticks])
    ax.axvline(left * 1.4, color="#b5b3ad", lw=0.8, ls=":")
    ax.set_xlabel("cumulative NGSpice calls per target (Jacobian probes included)",
                  color=ink)
    ax.set_ylabel("best-so-far paper RRMS (frozen inclusion)", color=ink)
    ax.set_title("DEV: all-device mean (solid), median (dashed); "
                 "dots = fixed-budget runs", fontsize=9, color=ink)
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b3ad")
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f"{v:.4f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"


def fmt_calls(v: float | None) -> str:
    return "–" if v is None else ("never" if math.isinf(v) else f"{v:.0f}")


def markdown(report: dict) -> str:
    arms, refs, sets = report["arms"], report["references"], report["sets"]
    marks = sorted({int(b) for a in arms.values() for b in a["budgets"]})
    lines = ["# DEV comparison", "",
             f"Targets: {', '.join(sets)} ({report['n_targets']} pooled). "
             "Primary: all-device mean paper RRMS (frozen inclusion) at fixed "
             "cumulative NGSpice budgets per target, every call counted. "
             + ("**PARTIAL (smoke) report.**" if report["partial"] else ""), "",
             "## Primary: pooled all-device mean", "",
             "| arm | mode | " + " | ".join(f"B{b}" for b in marks)
             + " | AUC log10 | median calls to 0.05 | to 0.01 | failed targets "
             "(per budget) |",
             "|---|---|" + "---:|" * (len(marks) + 4)]

    def n_bad(stats):
        st = (stats or {}).get("strict")
        return "–" if not st or not st["n_checked"] else str(st["n_failed"])

    for arm in arms.values():
        cells = [fmt(arm["budgets"].get(str(b), {}).get("mean")) for b in marks]
        reach = arm.get("median_calls_to", {})
        bad = "/".join(n_bad(arm["budgets"].get(str(b))) for b in marks)
        lines.append(f"| {arm['label']} | {arm['mode']} | " + " | ".join(cells)
                     + f" | {fmt(arm.get('auc'))} | {fmt_calls(reach.get('0.05'))}"
                     f" | {fmt_calls(reach.get('0.01'))} | {bad} |")
    lines += ["", "| reference | calls (mean) | pooled mean | median | failed targets "
              "| note |", "|---|---:|---:|---:|---:|---|"]
    for ref in refs.values():
        lines.append(f"| {ref['label']} | {ref['mean_calls']:.0f} | "
                     f"{fmt(ref['mean'])} | {fmt(ref['median'])} | {n_bad(ref)} | "
                     f"{ref['note']} |")
    lines += ["", "## Secondary columns (never exclude a device)", "",
              "| arm | budget | pooled mean | "
              + " | ".join(f"{s} mean | {s} combined" for s in sets)
              + " | median | geo mean | worst (target) | <1e-2 | <1e-3 |",
              "|---|---:|---:|" + "---:|" * (2 * len(sets) + 5)]
    rows = [(a["label"], b, a["budgets"][str(b)]) for a in arms.values()
            for b in marks if str(b) in a["budgets"]]
    rows += [(r["label"], "ref", r) for r in refs.values()]
    for label, b, s in rows:
        set_cells = " | ".join(
            f"{fmt(s['per_set'].get(n, {}).get('all_device_mean_rrms'))} | "
            f"{fmt(s['per_set'].get(n, {}).get('combined_rrms'))}" for n in sets)
        lines.append(f"| {label} | {b} | {fmt(s['mean'])} | {set_cells} | "
                     f"{fmt(s['median'])} | {fmt(s['geomean'])} | "
                     f"{fmt(s['worst'])} ({s['worst_target']}) | "
                     f"{s['frac_below_1e-2']:.2f} | {s['frac_below_1e-3']:.2f} |")
    lines += ["", "## Strict curve check (failed included curves)", "",
              "A failed target has a non-finite included metric curve; its stored "
              "RRMS averages only the surviving curves (not a valid paper RRMS). "
              "valid-only = pooled mean over valid targets; penalized = every "
              f"target, failed curves at RRMS {ss.PENALTY:g} (penalized convention, "
              "not the paper metric). Unchecked = no card record (trace-mode "
              "budgets below the top, bank oracle).", "",
              "| arm | budget | failed / checked | stored pooled mean | valid-only "
              "(n) | penalized | failed targets (failed curves) |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for label, b, s in rows:
        st = s.get("strict")
        if not st:
            continue
        lines.append(
            f"| {label} | {b} | {st['n_failed']} / {st['n_checked']} | "
            f"{fmt(s['mean'])} | {fmt(st['valid_mean'])} ({st['n_valid']}) | "
            f"{fmt(st['penalized_mean'])} | "
            + (", ".join(f"{k} ({n})" for k, n in st["failed_targets"].items())
               or "none") + " |")
    lines += ["", "## Paired tests (diff = arm - reference; negative = arm better)",
              "", "| arm | reference | budget | n | W/T/L | mean diff | 95% CI | "
              "Wilcoxon p |", "|---|---|---:|---:|---|---:|---|---:|"]
    for row in report["paired"]:
        lines.append(
            f"| {row['arm']} | {row['reference']} | {row['budget']} | {row['n']} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {fmt(row['mean_diff'])} "
            f"| [{fmt(row['ci95'][0])}, {fmt(row['ci95'][1])}] | "
            f"{fmt(row['wilcoxon_p'])} |")
    lines += ["", "## Hard checks", ""]
    lines += [f"- {c['level']}: {c['check']}" + (f" ({c['detail']})"
                                                  if c["detail"] else "")
              for c in report["checks"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEV / "arms.json"))
    parser.add_argument("--out-dir", default=None,
                        help="default: the registry's directory")
    parser.add_argument("--reference", default="hybrid_none",
                        help="paired-test reference arm (cmaes_trf is always "
                             "added when present)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="smoke tests: missing runs / devices are warnings")
    parser.add_argument("--no-oracle-check", action="store_true",
                        help="skip recomputing the bank oracle per DEV set "
                             "(arms are then only checked against each other)")
    args = parser.parse_args()
    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text())
    if isinstance(registry, list):
        registry = {"arms": registry}
    registry = {"sets": ["set_0", "set_1"], "bank": "data/processed/cold_loop/full_r1",
                "round_dir": "out/cold_loop/full/round_1",
                "targets_template": "out/research/sim_targets/{set}.npz",
                **registry}
    sets = registry["sets"]
    out_dir = Path(args.out_dir) if args.out_dir else registry_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = Checks(args.allow_partial)
    checks("DEV sets only", all(s in ("set_0", "set_1") for s in sets),
           f"{sets} (set_2 and real devices are TEST)")
    oracle = OracleReference(registry, DEV / "oracle_reference.json",
                             not args.no_oracle_check)
    raw = {}
    for arm in registry["arms"]:
        if arm.get("mode") == "reference":
            continue
        evaluated = evaluate_arm(arm, sets, checks, oracle)
        if evaluated:
            raw[arm["label"]] = evaluated
        else:
            checks(f"{arm['label']}: any run loaded", False, "", partial_ok=True)
    every_dir = [d for arm in raw.values() for d in arm["dirs"]]
    checks("no cross-arm best-of: each arm row reads only its own directories",
           len(every_dir) == len(set(every_dir)),
           "every value is one arm's own record; no min over arms is taken")
    if not raw:
        print("nothing to compare")
        return 1
    top = max(arm["top"] for arm in raw.values())
    grid = log_grid(top)
    arms = {label: summarize_arm(arm, sets, grid) for label, arm in raw.items()}
    refs = load_references(registry, sets, checks, oracle, raw)
    for label, arm in arms.items():
        for b, stats in arm["budgets"].items():
            st = stats["strict"]
            if st["n_failed"]:
                checks.warn(f"{label} B{b}: {st['n_failed']} target(s) with failed "
                            "included curves (stored mean drops them)",
                            ", ".join(f"{k} ({n})" for k, n in st["failed_targets"].items()))
    for label, ref in refs.items():
        st = ref["strict"]
        if st["n_failed"]:
            checks.warn(f"{label}: {st['n_failed']} target(s) with failed included "
                        "curves (stored mean drops them)",
                        ", ".join(f"{k} ({n})" for k, n in st["failed_targets"].items()))
    targets = sorted({k for arm in raw.values() for k in arm["traces"]})
    if not args.allow_partial:
        checks(f"{N_DEVICES * len(sets)} DEV targets",
               len(targets) == N_DEVICES * len(sets), f"{len(targets)}")
    paired_rows = []
    for ref_label in dict.fromkeys([args.reference, "cmaes_trf"]):
        if ref_label not in raw:
            continue
        for label, arm in raw.items():
            if label == ref_label:
                continue
            for b in sorted(set(arm["budgets"]) & set(raw[ref_label]["budgets"])):
                paired_rows.append({
                    "arm": label, "reference": ref_label, "budget": b,
                    **paired(arm["budgets"][b], raw[ref_label]["budgets"][b])})
    report = {"registry": str(registry_path), "sets": sets,
              "n_targets": len(targets), "partial": bool(args.allow_partial),
              "primary": "pooled all-device mean paper RRMS (frozen inclusion)",
              "grid": grid.tolist(), "arms": arms,
              "references": refs, "paired": paired_rows,
              "checks": checks.rows, "checks_failed": checks.failed}
    (out_dir / "compare.json").write_text(json.dumps(report, indent=1) + "\n")
    (out_dir / "compare.md").write_text(markdown(report))
    plot(arms, refs, out_dir / "compare_curves.png", grid)
    print(f"wrote {out_dir / 'compare.md'}, compare.json, compare_curves.png")
    for arm in arms.values():
        print(arm["label"], {b: round(s["mean"], 5)
                             for b, s in arm["budgets"].items()},
              "AUC", round(arm.get("auc", math.nan), 4))
    for ref in refs.values():
        print(ref["label"], round(ref["mean"], 5), f"{ref['mean_calls']:.0f} calls")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
