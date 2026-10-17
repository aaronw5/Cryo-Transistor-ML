#!/usr/bin/env python3
"""DEV statistics: geometry-clustered, seed-replicated, Holm-corrected
(research node n2c).  Reads result files only; no simulation.

Registries are read through ``dev_compare``'s own functions (``Checks``,
``OracleReference``, ``evaluate_arm``: the same hard checks, bank-oracle
identity, budgets and ``expect`` keys).  A registry argument is a path, or
``path#key`` for a sub-registry stored under ``key`` (e.g.
``registry.json#tier_c``); top-level fields are the
defaults of the sub-registry.

Per-target value of an arm at a budget: the mean over its seeds (seed 0 =
the base label, seed n = the label ``<arm>__seed<n>``).  A target whose final
card has a failed included curve (``dev_compare.target_status``) enters at
its penalized value (failed curves at RRMS 10, ``cryoml.strict_score``) and is
counted.  Endpoints: tier a, raw strict RRMS; tier c, excess over the floor
(``RRMS - floor``, unclipped, floors from ``--floors``); secondary on tier a,
log10 RRMS (mean over seeds of ``log10(max(RRMS, 1e-6))``).  A difference of
excesses equals the difference of raw RRMS at the same target; the floor
changes the per-arm levels and the performance profiles.

Per comparison A - B at a budget (36 targets = 18 geometries x 2 sets):

* mean difference over the 36 targets;
* geometry-clustered bootstrap 95% CI: 10,000 resamples (seed 0) of the 18
  geometries, a drawn geometry contributes both of its sets, and within a
  target an arm's seeds are resampled when it has 2 or more;
* i.i.d. bootstrap 95% CI over the 36 targets (``dev_compare.paired``, as in
  the iteration-1 report), next to the clustered CI;
* two-sided Wilcoxon signed-rank on the 18 geometry-mean differences (the
  pre-registered test), and the 36-target Wilcoxon of ``dev_compare.paired``
  as a secondary value;
* Holm adjustment over the family (a test that cannot be computed enters
  with p = 1);
* leave-one-geometry-out maximum |change of the mean difference|;
* win/tie/loss with ``dev_compare``'s tie rule (tie when |diff| <
  max(1e-6, 1% of the larger value));
* interquartile mean of the 36 differences.

Profiles over the arms of the family: Dolan-Moré performance profiles per
budget (ratio of an arm's value to the best arm's on each target, values
floored at 1e-6), and Moré-Wild data profiles from the largest-budget traces
(best-so-far RRMS after ``kappa`` calls; solved when ``<= f_L + tau (f_0 -
f_L)``, ``f_0`` = the first call's RRMS, ``f_L`` = the lowest value of any
arm at the common largest budget; invariant to the floor shift).

Families:

* ``--family iter1``: the 5 iteration-1 paired comparisons of report table
  03 at budgets 500/1000/2000/5000 on tier a (20 tests, Holm over the 20);
  a re-analysis of already-read data, not pre-registered.
* ``--prereg out/research/dev/prereg_iter2.json``: the pre-registered
  family.  The file must be committed and unmodified (``git status``); its
  last commit hash (``git log -1 --format=%H -- <file>``) is recorded.

``--lq-audit DIR``: the lq-CMA-ES configuration audit rule (no statistics):
reads ``DIR/sigma<s>_r<r>`` (``cmaes_trace.py --method lq_cmaes``, set_0,
B2000, 3 targets), writes ``DIR/decision.json``; ``--apply`` adds the
``lq_cmaes_audited`` arm (with its ``dev_registry.py`` command template) to
``out/research/dev/arms.d/baselines.json`` and its queue lines to
``queue_lines_baselines.txt`` when the rule triggers.

  PYTHONPATH=src .venv/bin/python scripts/research/dev_stats.py \\
      --registry out/research/dev/arms.json --family iter1 \\
      --out out/research/dev/stats_iter1
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import dev_compare  # noqa: E402
from cryoml import strict_score as ss  # noqa: E402

DEV = ROOT / "out" / "research" / "dev"
N_RESAMPLES = 10_000
BOOT_SEED = 0
FLOOR = 1e-6
SEED_LABEL = re.compile(r"^(?P<base>.+)__seed(?P<seed>\d+)$")
ITER1 = [("sketch_surrogate_k8", "sketch_none_k8"),
         ("hybrid_surrogate", "hybrid_none"),
         ("anytime_sur_presearch_race", "anytime_none_race"),
         ("sketch_surrogate_k8", "cmaes_trf"),
         ("sketch_none_k8", "hybrid_none")]
ITER1_BUDGETS = [500, 1000, 2000, 5000]
DEFAULTS = {"sets": ["set_0", "set_1"], "bank": "data/processed/cold_loop/full_r1",
            "round_dir": "out/cold_loop/full/round_1",
            "targets_template": "out/research/sim_targets/{set}.npz"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#6250d6", "#e34948"]
PERF_TAUS = np.geomspace(1.0, 1e3, 61)
DATA_TAUS = (1e-1, 1e-2, 1e-3)


# ------------------------------------------------------------------ inputs

def load_registry(spec: str) -> tuple[dict, str]:
    path, _, key = spec.partition("#")
    data = json.loads((ROOT / path if not Path(path).is_absolute()
                       else Path(path)).read_text())
    if isinstance(data, list):
        data = {"arms": data}
    if key:
        sub = data[key]
        data = {**{k: v for k, v in data.items() if k not in ("arms", key)},
                **sub}
    return {**DEFAULTS, **data}, spec


def load_floors(path: Path) -> tuple[dict[str, float], dict]:
    """``{"set_k/tag": floor}`` from n2a's floors.json
    (``sky130c_floor.py``: ``{"targets": {"set_k/tag": {"floor": v, ...}},
    "finalized": bool}``).  Also accepted: ``{"floors": X}`` or X, with X
    ``{"set/tag": v}`` or ``{set: {tag: v}}``, ``v`` a number or a dict with
    ``floor``/``floor_rrms``/``rrms``.  Returns the floors and provenance."""
    data = json.loads(path.read_text())
    meta = {"path": str(path), "kind": data.get("kind"),
            "finalized": data.get("finalized"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    table = data.get("targets") or data.get("floors") or data
    keys = ("floor", "floor_rrms", "rrms")

    def number(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if isinstance(v, dict):
            for k in keys:
                if isinstance(v.get(k), (int, float)) and not isinstance(v.get(k), bool):
                    return float(v[k])
        return None

    out = {}
    for k, v in table.items():
        if "/" in k and number(v) is not None:
            out[k] = number(v)
        elif isinstance(v, dict) and re.fullmatch(r"set_\d+", k):
            for tag, w in v.items():
                if number(w) is not None:
                    out[f"{k}/{tag}"] = number(w)
    if not out:
        raise SystemExit(f"{path}: no floors found (expected set/tag -> floor)")
    return out, meta


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=False).stdout.strip()


def prereg_identity(path: Path) -> dict:
    """Refuse a pre-registration that is not committed or is modified."""
    rel = str(path.resolve().relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                             cwd=ROOT, capture_output=True).returncode == 0
    status = git("status", "--porcelain", "--", rel)
    commit = git("log", "-1", "--format=%H", "--", rel)
    if not tracked or not commit:
        raise SystemExit(f"{rel}: not committed; commit the pre-registration "
                         "before any analysis")
    if status:
        raise SystemExit(f"{rel}: modified since its last commit ({status!r}); "
                         "refusing to run")
    return {"path": rel, "commit": commit,
            "committed_at": git("log", "-1", "--format=%cI", "--", rel),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def arm_entries(registry: dict, base: str) -> dict[int, dict]:
    """Seed -> registry entry of ``base`` (seed 0) and ``base__seed<n>``."""
    out = {}
    for arm in registry["arms"]:
        if arm.get("mode") == "reference":
            continue
        if arm["label"] == base:
            out[0] = arm
        else:
            match = SEED_LABEL.match(arm["label"])
            if match and match["base"] == base:
                out[int(match["seed"])] = arm
    return out


class Tier:
    """One DEV tier: its registry, loaded arms (per seed) and endpoint."""

    def __init__(self, name: str, spec: str, checks, oracle_enabled: bool,
                 floors: dict | None):
        self.name = name
        self.registry, self.spec = load_registry(spec)
        self.sets = self.registry["sets"]
        self.checks = checks
        self.oracle = dev_compare.OracleReference(
            self.registry, DEV / "oracle_reference.json", oracle_enabled)
        self.floors = floors
        self.arms: dict[str, dict[int, dict]] = {}
        self.dirs: dict[str, list[str]] = {}

    def load(self, base: str) -> dict[int, dict] | None:
        if base in self.arms:
            return self.arms[base]
        entries = arm_entries(self.registry, base)
        if not entries:
            self.checks(f"tier {self.name}: arm {base} in registry", False,
                        self.spec, partial_ok=True)
            self.arms[base] = None
            return None
        loaded = {}
        for seed, entry in sorted(entries.items()):
            result = dev_compare.evaluate_arm(entry, self.sets, self.checks,
                                              self.oracle)
            if result:
                loaded[seed] = result
                self.dirs.setdefault(base, []).extend(result["dirs"])
            else:
                self.checks(f"tier {self.name}: {entry['label']} loaded", False,
                            "", partial_ok=True)
        self.arms[base] = loaded or None
        return self.arms[base]

    def targets(self, base: str, budget: int) -> list[str]:
        arm = self.arms.get(base) or {}
        keys = set()
        for result in arm.values():
            keys |= set(result["budgets"].get(budget, {}))
        return sorted(keys)

    def matrix(self, base: str, budget: int, geometries: list[str]):
        """``(G, S, k)`` per-seed values (penalized for failed targets), the
        seeds used, failed-target count and missing targets."""
        arm = self.arms.get(base) or {}
        seeds = sorted(s for s, r in arm.items() if budget in r["budgets"])
        values = np.full((len(geometries), len(self.sets), len(seeds)), np.nan)
        missing, failed = [], set()
        for k, seed in enumerate(seeds):
            vals = arm[seed]["budgets"][budget]
            status = arm[seed]["status"].get(budget, {})
            for g, tag in enumerate(geometries):
                for s, set_name in enumerate(self.sets):
                    key = f"{set_name}/{tag}"
                    if key not in vals:
                        missing.append(f"seed{seed}:{key}")
                        continue
                    st = status.get(key)
                    v = float(vals[key])
                    if st is not None and st["status"] == ss.FAILED:
                        failed.add(key)
                        v = float(st["penalized_rrms"])
                    values[g, s, k] = v
        return values, seeds, sorted(failed), missing

    def endpoint(self, values: np.ndarray, geometries: list[str]) -> np.ndarray:
        if self.floors is None:
            return values
        floor = np.array([[self.floors.get(f"{s}/{tag}", np.nan)
                           for s in self.sets] for tag in geometries])
        return values - floor[:, :, None]


# --------------------------------------------------------------- statistics

def wilcoxon_p(values: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return math.nan
    if not np.any(values != 0):
        return 1.0
    with np.errstate(all="ignore"):
        return float(wilcoxon(values).pvalue)


def clustered_bootstrap(TA: np.ndarray, TB: np.ndarray,
                        n_resamples: int = N_RESAMPLES,
                        seed: int = BOOT_SEED) -> np.ndarray:
    """Resampled mean differences: geometries with replacement (both sets
    of a drawn geometry), then each arm's seeds within a target when it has
    2 or more.  Draw order: geometries, A's seeds, B's seeds."""
    rng = np.random.default_rng(seed)
    G, S = TA.shape[:2]
    gi = rng.integers(0, G, (n_resamples, G))

    def resample(T):
        k = T.shape[2]
        if k >= 2:
            si = rng.integers(0, k, (n_resamples, G, S, k))
            return T[gi[:, :, None, None], np.arange(S)[None, None, :, None],
                     si].mean(-1)
        return T[gi][..., 0]

    A = resample(TA)
    B = resample(TB)
    return (A - B).mean(axis=(1, 2))


def compare(TA: np.ndarray, TB: np.ndarray, geometries: list[str],
            sets: list[str]) -> dict:
    from scipy.stats import trim_mean
    A, B = TA.mean(-1), TB.mean(-1)
    d = A - B
    D = d.mean(1)
    boot = clustered_bootstrap(TA, TB)
    keys = [f"{s}/{tag}" for tag in geometries for s in sets]
    flat_a = {k: v for k, v in zip(keys, A.ravel())}
    flat_b = {k: v for k, v in zip(keys, B.ravel())}
    iid = dev_compare.paired(flat_a, flat_b)
    G = len(D)
    logo = (D.sum() - D) / (G - 1) - D.mean() if G > 1 else np.zeros(1)
    worst = int(np.argmax(np.abs(logo)))
    return {
        "n_targets": int(d.size), "n_geometries": G,
        "mean_a": float(A.mean()), "mean_b": float(B.mean()),
        "mean_diff": float(d.mean()),
        "ci95_clustered": [float(np.percentile(boot, 2.5)),
                           float(np.percentile(boot, 97.5))],
        "ci95_iid": iid["ci95"],
        "wilcoxon_p_geometry": wilcoxon_p(D),
        "wilcoxon_p_targets": iid["wilcoxon_p"],
        "wins": iid["wins"], "ties": iid["ties"], "losses": iid["losses"],
        "iqm_diff": float(trim_mean(d.ravel(), 0.25)),
        "logo_max_abs_change": float(np.abs(logo[worst])),
        "logo_worst_geometry": geometries[worst],
        "geometry_mean_diffs": {tag: float(v) for tag, v in zip(geometries, D)},
    }


def holm(pvalues: list[float]) -> list[float]:
    p = np.array([1.0 if (v is None or not math.isfinite(v)) else v
                  for v in pvalues], dtype=np.float64)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def performance_profile(values: dict[str, np.ndarray]) -> dict:
    labels = list(values)
    T = np.vstack([np.where(np.isfinite(values[k]),
                            np.maximum(values[k], FLOOR), np.inf)
                   for k in labels])
    best = T.min(axis=0)
    ratio = T / best
    return {"tau": PERF_TAUS.tolist(),
            "rho": {k: [float(np.mean(ratio[i] <= t)) for t in PERF_TAUS]
                    for i, k in enumerate(labels)},
            "n_problems": int(T.shape[1]),
            "n_values_at_floor": {k: int(np.sum(values[k] <= FLOOR))
                                  for k in labels}}


def data_profile(traces: dict[str, list[dict]], targets: list[str],
                 f0: dict[str, float], top: int) -> dict:
    """``traces[label]`` = per-seed ``{target: trace}``; value after
    ``kappa`` calls = mean over seeds of ``best_at(trace, kappa)``.  ``f0``
    = the shared start value per target (bank-oracle paper RRMS); ``top`` =
    the largest budget common to the arms."""
    best_at = dev_compare.best_at
    labels = list(traces)
    grid = dev_compare.log_grid(top)

    def value(label, target, calls):
        return float(np.mean([best_at(tr[target], calls)
                              for tr in traces[label]]))

    fL = {t: min(value(k, t, top) for k in labels) for t in targets}
    first = {k: [tr[t][0][1] for tr in seeds for t in targets if tr[t]]
             for k, seeds in traces.items()}
    out = {"calls": grid.tolist(), "top": int(top),
           "f0": "bank-oracle paper RRMS of each target (the pre-paid bank's "
                 "best sample, the start shared by the arms)",
           "first_call_minus_f0_max_abs": {
               k: float(np.max(np.abs(np.array(v) - np.array(
                   [f0[t] for tr in traces[k] for t in targets if tr[t]]))))
               for k, v in first.items()},
           "d": {}}
    for tau in DATA_TAUS:
        out["d"][str(tau)] = {
            k: [float(np.mean([value(k, t, c) <= fL[t] + tau * (f0[t] - fL[t])
                               for t in targets])) for c in grid]
            for k in labels}
    return out


# ------------------------------------------------------------------ plots

def style(ax, ink="#0b0b0b", muted="#52514e"):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b3ad")
    ax.tick_params(colors=muted, labelsize=7)


def plot_profiles(profiles: dict, tier: str, path: Path, endpoint: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    perf, data = profiles["performance"], profiles.get("data")
    budgets = list(perf)
    ncol = max(len(budgets), len(DATA_TAUS))
    nrow = 2 if data else 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.9 * nrow),
                             dpi=150, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    labels = profiles["arms"]
    color = {k: COLORS[i % len(COLORS)] for i, k in enumerate(labels)}
    for j in range(ncol):
        ax = axes[0, j]
        if j >= len(budgets):
            ax.axis("off")
            continue
        p = perf[budgets[j]]
        style(ax)
        for k, rho in p["rho"].items():
            ax.plot(p["tau"], rho, color=color[k], lw=1.6, label=k)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"tier {tier}, B{budgets[j]}: performance profile",
                     fontsize=8)
        ax.set_xlabel(f"ratio to the best arm ({endpoint})", fontsize=7)
    axes[0, 0].set_ylabel("fraction of targets", fontsize=7)
    if data:
        for j in range(ncol):
            ax = axes[1, j]
            if j >= len(DATA_TAUS):
                ax.axis("off")
                continue
            style(ax)
            for k, d in data["d"][str(DATA_TAUS[j])].items():
                ax.plot(data["calls"], d, color=color[k], lw=1.6, label=k)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"tier {tier}: data profile, tau = {DATA_TAUS[j]:g}",
                         fontsize=8)
            ax.set_xlabel("NGSpice calls per target", fontsize=7)
        axes[1, 0].set_ylabel("fraction of targets solved", fontsize=7)
    axes[0, 0].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


# ------------------------------------------------------------------ family

def iter1_family() -> dict:
    tests = [{"id": f"I{i + 1}", "a": a, "b": b, "tier": "a", "budget": bud}
             for i, (a, b) in enumerate(ITER1) for bud in ITER1_BUDGETS]
    return {"name": "iter1", "pre_registered": False,
            "description": "re-analysis of the 5 iteration-1 paired comparisons "
                           "of report table 03 (tier a, budgets 500/1000/2000/"
                           "5000); these data were read in iteration 1; not "
                           "pre-registered",
            "tests": tests, "alpha": 0.05, "secondary_log10_tier_a": True}


def prereg_family(prereg: dict) -> dict:
    tests = [{"id": f"{c['id']}_{tier}_B{bud}", "comparison": c["id"],
              "a": c["a"], "b": c["b"], "tier": tier, "budget": int(bud)}
             for c in prereg["comparisons"] for tier in prereg["tiers"]
             for bud in prereg["budgets"]]
    if len(tests) != int(prereg["n_tests"]):
        raise SystemExit(f"prereg lists {prereg['n_tests']} tests, "
                         f"{len(tests)} built")
    return {"name": prereg["id"], "pre_registered": True,
            "description": prereg.get("summary", ""), "tests": tests,
            "alpha": float(prereg["multiplicity"]["alpha"]),
            "secondary_log10_tier_a": True}


def run_family(family: dict, tiers: dict[str, Tier]) -> dict:
    rows = []
    for test in family["tests"]:
        row = dict(test)
        tier = tiers.get(test["tier"])
        if tier is None:
            row["missing"] = f"tier {test['tier']} registry not given"
            rows.append(row)
            continue
        for label in (test["a"], test["b"]):
            tier.load(label)
        if not tier.arms.get(test["a"]) or not tier.arms.get(test["b"]):
            row["missing"] = "arm not in registry or not loaded"
            rows.append(row)
            continue
        b = test["budget"]
        keys = set(tier.targets(test["a"], b)) | set(tier.targets(test["b"], b))
        geometries = sorted({k.split("/", 1)[1] for k in keys})
        VA, seeds_a, failed_a, miss_a = tier.matrix(test["a"], b, geometries)
        VB, seeds_b, failed_b, miss_b = tier.matrix(test["b"], b, geometries)
        row.update({"seeds_a": seeds_a, "seeds_b": seeds_b,
                    "failed_targets_a": failed_a, "failed_targets_b": failed_b})
        complete = (seeds_a and seeds_b and not miss_a and not miss_b
                    and len(geometries) == dev_compare.N_DEVICES)
        if not complete:
            row["missing"] = (f"incomplete: geometries {len(geometries)}, "
                              f"missing {(miss_a + miss_b)[:6]}"
                              + (" (no seed has this budget)"
                                 if not seeds_a or not seeds_b else ""))
            rows.append(row)
            continue
        TA, TB = tier.endpoint(VA, geometries), tier.endpoint(VB, geometries)
        if not (np.all(np.isfinite(TA)) and np.all(np.isfinite(TB))):
            row["missing"] = "non-finite endpoint (missing floor?)"
            rows.append(row)
            continue
        row["primary"] = compare(TA, TB, geometries, tier.sets)
        row["endpoint"] = ("excess over floor (RRMS - floor)" if tier.floors
                           else "strict RRMS")
        if tier.floors is not None:
            row["n_targets_below_floor"] = {
                "a": int(np.sum(TA.mean(-1) < 0)), "b": int(np.sum(TB.mean(-1) < 0))}
        if family["secondary_log10_tier_a"] and tier.floors is None:
            row["log10"] = compare(np.log10(np.maximum(VA, FLOOR)),
                                   np.log10(np.maximum(VB, FLOOR)),
                                   geometries, tier.sets)
        rows.append(row)
    adjusted = holm([r.get("primary", {}).get("wilcoxon_p_geometry")
                     for r in rows])
    for row, p in zip(rows, adjusted):
        row["holm_p"] = p
        row["holm_reject"] = bool(p <= family["alpha"]) if "primary" in row else False
    log10_rows = [r for r in rows if "log10" in r]
    for row, p in zip(log10_rows, holm([r["log10"]["wilcoxon_p_geometry"]
                                        for r in log10_rows])):
        row["log10"]["holm_p_within_secondary"] = p
    return {"tests": rows, "n_tests": len(rows),
            "n_computed": sum("primary" in r for r in rows),
            "n_holm_reject": sum(r["holm_reject"] for r in rows)}


def profiles_for(family: dict, tiers: dict[str, Tier]) -> dict:
    out = {}
    for name, tier in tiers.items():
        tests = [t for t in family["tests"] if t["tier"] == name]
        labels = list(dict.fromkeys(x for t in tests for x in (t["a"], t["b"])))
        labels = [k for k in labels if tier.arms.get(k)]
        if not labels:
            continue
        budgets = sorted({t["budget"] for t in tests})
        perf = {}
        for b in budgets:
            have = [k for k in labels if tier.targets(k, b)]
            keys = sorted(set.intersection(*[set(tier.targets(k, b)) for k in have])) if have else []
            geometries = sorted({k.split("/", 1)[1] for k in keys})
            if len(geometries) != dev_compare.N_DEVICES:
                continue
            values = {}
            for k in have:
                V, _, _, miss = tier.matrix(k, b, geometries)
                if not miss:
                    values[k] = tier.endpoint(V, geometries).mean(-1).ravel()
            if len(values) >= 2:
                perf[str(b)] = performance_profile(values)
        traces, tops = {}, []
        for k in labels:
            seeds = [r["traces"] for r in tier.arms[k].values() if r["traces"]]
            if seeds:
                traces[k] = seeds
                tops.append(min(r["top"] for r in tier.arms[k].values()))
        targets = sorted(set.intersection(*[set(tr) for seeds in traces.values()
                                            for tr in seeds])) if traces else []
        data = None
        if (len(traces) >= 2
                and len(targets) == len(tier.sets) * dev_compare.N_DEVICES):
            f0 = {t: tier.oracle.get(*t.split("/", 1)) for t in targets}
            if all(v is not None for v in f0.values()):
                data = data_profile(traces, targets, f0, min(tops))
        out[name] = {"arms": labels, "performance": perf, "data": data,
                     "endpoint": "excess over floor" if tier.floors else "strict RRMS"}
    return out


# ---------------------------------------------------------------- markdown

def fmt(v) -> str:
    return dev_compare.fmt(v) if v is not None else "–"


def markdown(report: dict) -> str:
    fam = report["family"]
    lines = [f"# DEV statistics: family {fam['name']}", "",
             fam["description"], "",
             f"Generated {report['written']} by `scripts/research/dev_stats.py` "
             "from result files only (no simulation).", ""]
    if report.get("prereg"):
        p = report["prereg"]
        lines += [f"Pre-registration: `{p['path']}`, commit `{p['commit']}` "
                  f"({p['committed_at']}), sha256 `{p['sha256'][:16]}`.", ""]
    lines += [f"Registries: " + "; ".join(f"tier {k}: `{v}`"
                                          for k, v in report["registries"].items()),
              "",
              f"Tests: {report['results']['n_tests']}, computed "
              f"{report['results']['n_computed']}, Holm-adjusted p <= "
              f"{fam['alpha']}: {report['results']['n_holm_reject']}. "
              "Test: two-sided Wilcoxon signed-rank on the 18 geometry-mean "
              "differences; Holm over all tests of the family (a test that "
              "cannot be computed enters with p = 1). Difference = A - B; "
              "negative = A lower.", "",
              "## Primary endpoint", "",
              "| test | tier | budget | A − B | mean A | mean B | mean diff | "
              "clustered 95% CI | i.i.d. 95% CI | p (18 geom.) | Holm p | Holm <= "
              f"{fam['alpha']} | W/T/L | IQM diff | LOGO max | seeds A/B | failed A/B |",
              "|---|---|---:|---|---:|---:|---:|---|---|---:|---:|---|---|---:|---:|---|---|"]
    for r in report["results"]["tests"]:
        head = f"| {r['id']} | {r['tier']} | {r['budget']} | {r['a']} − {r['b']} |"
        if "primary" not in r:
            lines.append(head + f" – | – | – | – | – | – | {fmt(r['holm_p'])} | "
                         f"not computed: {r.get('missing', '')} | – | – | – | – | – |")
            continue
        s = r["primary"]
        lines.append(
            head + f" {fmt(s['mean_a'])} | {fmt(s['mean_b'])} | {fmt(s['mean_diff'])} | "
            f"[{fmt(s['ci95_clustered'][0])}, {fmt(s['ci95_clustered'][1])}] | "
            f"[{fmt(s['ci95_iid'][0])}, {fmt(s['ci95_iid'][1])}] | "
            f"{fmt(s['wilcoxon_p_geometry'])} | {fmt(r['holm_p'])} | "
            f"{'yes' if r['holm_reject'] else 'no'} | "
            f"{s['wins']}/{s['ties']}/{s['losses']} | {fmt(s['iqm_diff'])} | "
            f"{fmt(s['logo_max_abs_change'])} ({s['logo_worst_geometry']}) | "
            f"{len(r['seeds_a'])}/{len(r['seeds_b'])} | "
            f"{len(r['failed_targets_a'])}/{len(r['failed_targets_b'])} |")
    sec = [r for r in report["results"]["tests"] if "log10" in r]
    if sec:
        lines += ["", "## Secondary endpoint: log10 RRMS (tier a)", "",
                  "Mean over seeds of log10(max(RRMS, 1e-6)); not in the Holm "
                  "family of the primary endpoint (Holm within this table shown "
                  "for reference).", "",
                  "| test | budget | A − B | mean diff (log10) | clustered 95% CI | "
                  "i.i.d. 95% CI | p (18 geom.) | Holm p (secondary) |",
                  "|---|---:|---|---:|---|---|---:|---:|"]
        for r in sec:
            s = r["log10"]
            lines.append(
                f"| {r['id']} | {r['budget']} | {r['a']} − {r['b']} | "
                f"{s['mean_diff']:.3f} | [{s['ci95_clustered'][0]:.3f}, "
                f"{s['ci95_clustered'][1]:.3f}] | [{s['ci95_iid'][0]:.3f}, "
                f"{s['ci95_iid'][1]:.3f}] | {fmt(s['wilcoxon_p_geometry'])} | "
                f"{fmt(s['holm_p_within_secondary'])} |")
    for tier, prof in report["profiles"].items():
        lines += ["", f"## Profiles, tier {tier} ({prof['endpoint']})", "",
                  "Performance profile: fraction of the 36 targets on which the "
                  "arm's value is within a factor tau of the best arm's.", "",
                  "| budget | arm | tau 1 | tau 1.5 | tau 2 | tau 10 | tau 100 |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        taus = np.array(PERF_TAUS)
        for b, p in prof["performance"].items():
            for k, rho in p["rho"].items():
                cells = [rho[int(np.argmin(np.abs(taus - t)))] for t in
                         (1.0, 1.5, 2.0, 10.0, 100.0)]
                lines.append(f"| {b} | {k} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")
        if prof["data"]:
            d = prof["data"]
            calls = np.array(d["calls"])
            marks = [c for c in (100, 500, 1000, 2000, 5000) if c <= d["top"]]
            lines += ["", "Data profile: fraction of targets with best-so-far "
                      "RRMS <= f_L + tau (f_0 - f_L) after the given number of "
                      f"calls (f_0 = {d['f0']}; f_L = lowest value of these arms "
                      f"at {d['top']} calls).",
                      "", "| tau | arm | " + " | ".join(f"{c} calls" for c in marks) + " |",
                      "|---|---|" + "---:|" * len(marks)]
            for tau, rows in d["d"].items():
                for k, frac in rows.items():
                    cells = [frac[int(np.argmin(np.abs(calls - c)))] for c in marks]
                    lines.append(f"| {tau} | {k} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")
    bad = [c for c in report["checks"] if c["level"] != "PASS"]
    lines += ["", "## Checks", "",
              f"{len(report['checks'])} checks from dev_compare's loaders; "
              f"non-PASS: {len(bad)}."]
    lines += [f"- {c['level']}: {c['check']}" + (f" ({c['detail']})" if c["detail"] else "")
              for c in bad]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- lq audit

LQ_CONFIGS = [(0.1, 0), (0.1, 9), (0.3, 0), (0.3, 9)]
LQ_BASELINE = (0.3, 9)
LQ_DEVICES = ["nmos_L1_W3", "pmos_L0p35_W1p6", "pmos_L8_W5"]
LQ_BUDGETS = [500, 1000, 2000, 5000]
BASELINES = DEV / "arms.d" / "baselines.json"
QUEUE_LINES = DEV / "queue_lines_baselines.txt"
TIER_TARGETS = {"a": ("out/research/sim_targets", "out/research/dev"),
                "c": ("out/research/sim_targets_sky130c", "out/research/dev_sky130c")}


def lq_dir_name(sigma: float, restarts: int) -> str:
    return f"sigma{sigma:g}_r{restarts}"


def lq_lines(sigma: float, restarts: int) -> list[str]:
    lines = []
    for tier, (targets, out_root) in TIER_TARGETS.items():
        for budget in LQ_BUDGETS:
            for k in (0, 1):
                lines.append(
                    f"dev_{tier}_lq_cmaes_audited_B{budget}_s{k} | "
                    f"CRYOML_TARGET_OVERRIDE={targets}/set_{k}.npz "
                    ".venv/bin/python scripts/research/cmaes_trace.py "
                    "--round-dir out/cold_loop/full/round_1 "
                    "--bank data/processed/cold_loop/full_r1 --set full "
                    f"--method lq_cmaes --sigma0 {sigma:g} --restarts {restarts} "
                    f"--budget {budget} --out {out_root}/lq_cmaes_audited/B{budget}/set_{k}")
    return lines


def lq_audit(directory: Path, apply: bool, oracle_enabled: bool) -> int:
    registry = dict(DEFAULTS)
    checks = dev_compare.Checks(allow_partial=False)
    oracle = dev_compare.OracleReference(registry, DEV / "oracle_reference.json",
                                         oracle_enabled)
    configs = {}
    for sigma, restarts in LQ_CONFIGS:
        sub = directory / lq_dir_name(sigma, restarts)
        rows = {}
        for path in sorted(sub.glob("ml_*.json")):
            row = json.loads(path.read_text())
            rows[row["device"]] = row["methods"]["lq_cmaes"]
        name = lq_dir_name(sigma, restarts)
        ok = sorted(rows) == sorted(LQ_DEVICES)
        checks(f"{name}: records for {LQ_DEVICES}", ok, f"{sorted(rows)}")
        for tag, r in rows.items():
            checks(f"{name} {tag}: sigma0/restarts/budget",
                   (r.get("sigma0") == sigma and r.get("restarts") == restarts
                    and r.get("budget") == 2000 and r["n_ngspice_calls"] <= 2000),
                   f"{r.get('sigma0')}, {r.get('restarts')}, {r.get('budget')}, "
                   f"{r['n_ngspice_calls']}")
            checks(f"{name} {tag}: set_0 bank-oracle identity",
                   oracle.matches("set_0", tag, r["oracle_lhc"]["paper_rrms"]))
        configs[name] = {
            "sigma0": sigma, "restarts": restarts,
            "rrms": {t: float(r["rrms"]) for t, r in rows.items()},
            "mean_rrms": (float(np.mean([r["rrms"] for r in rows.values()]))
                          if ok else math.nan)}
    base = configs[lq_dir_name(*LQ_BASELINE)]["mean_rrms"]
    finite = {k: v for k, v in configs.items() if math.isfinite(v["mean_rrms"])}
    best = min(finite, key=lambda k: finite[k]["mean_rrms"]) if finite else None
    trigger = bool(best is not None and math.isfinite(base) and not checks.failed
                   and best != lq_dir_name(*LQ_BASELINE)
                   and finite[best]["mean_rrms"] <= 0.75 * base)
    decision = {
        "rule": "if the best configuration's 3-target mean RRMS is <= 0.75 x the "
                "mean of the iteration-1 configuration (sigma0 0.3, restarts 9) "
                "from the same audit, lq_cmaes_audited runs with that "
                "configuration on both tiers at 4 budgets; otherwise lq_cmaes "
                "stays as is",
        "devices": LQ_DEVICES, "budget": 2000, "set": "set_0 (tier a)",
        "configs": configs, "baseline": lq_dir_name(*LQ_BASELINE),
        "baseline_mean_rrms": base, "best": best,
        "best_mean_rrms": finite[best]["mean_rrms"] if best else None,
        "ratio_best_to_baseline": (finite[best]["mean_rrms"] / base
                                   if best and base else None),
        "trigger": trigger, "checks": checks.rows,
        "written": datetime.datetime.now().isoformat(timespec="seconds")}
    if trigger:
        sigma, restarts = finite[best]["sigma0"], finite[best]["restarts"]
        decision["queue_lines"] = lq_lines(sigma, restarts)
        decision["registry_entry"] = {
            "label": "lq_cmaes_audited",
            "dir_template": "out/research/dev/lq_cmaes_audited/B{budget}/{set}",
            "method_key": "lq_cmaes", "mode": "fixed", "budgets": LQ_BUDGETS,
            "expect": {"sigma0": sigma, "restarts": restarts, "x0_evaluated": True},
            "note": f"n2c lq-CMA-ES audit: sigma0 {sigma:g}, restarts {restarts} "
                    f"(3-target B2000 mean {finite[best]['mean_rrms']:.4f} vs "
                    f"{base:.4f} for sigma0 0.3, restarts 9)"}
        if apply:
            reg = json.loads(BASELINES.read_text())
            entry = {**decision["registry_entry"], "command": (
                "{override} .venv/bin/python scripts/research/cmaes_trace.py "
                "--round-dir out/cold_loop/full/round_1 "
                "--bank data/processed/cold_loop/full_r1 --set full "
                f"--method lq_cmaes --sigma0 {sigma:g} --restarts {restarts} "
                "--budget {budget} --out {out}")}
            reg["arms"] = [a for a in reg["arms"] if a["label"] != entry["label"]] + [entry]
            BASELINES.write_text(json.dumps(reg, indent=1) + "\n")
            existing = QUEUE_LINES.read_text() if QUEUE_LINES.exists() else ""
            new = [l for l in decision["queue_lines"] if l not in existing]
            QUEUE_LINES.write_text(existing + "".join(l + "\n" for l in new))
            decision["applied"] = True
    (directory / "decision.json").write_text(json.dumps(decision, indent=1) + "\n")
    print(json.dumps({k: decision[k] for k in (
        "baseline_mean_rrms", "best", "best_mean_rrms", "trigger")}, indent=1))
    for line in decision.get("queue_lines", []):
        print(line)
    return 1 if checks.failed else 0


# ------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=None,
                        help="tier a registry (path or path#key)")
    parser.add_argument("--registry-c", default=None,
                        help="tier c registry (path or path#key)")
    parser.add_argument("--floors", default=None, help="tier c floors.json")
    parser.add_argument("--family", choices=("iter1",), default=None)
    parser.add_argument("--prereg", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-oracle-check", action="store_true")
    parser.add_argument("--lq-audit", default=None,
                        help="lq-CMA-ES audit directory: apply the audit rule")
    parser.add_argument("--apply", action="store_true",
                        help="with --lq-audit: add the audited arm to "
                             "arms.d/baselines.json and its queue lines")
    args = parser.parse_args()
    if args.lq_audit:
        return lq_audit(Path(args.lq_audit), args.apply, not args.no_oracle_check)
    if bool(args.family) == bool(args.prereg):
        parser.error("give exactly one of --family iter1 and --prereg")
    if not args.out or not args.registry:
        parser.error("--registry and --out are required")
    prereg = None
    if args.prereg:
        prereg_path = Path(args.prereg)
        if not prereg_path.is_absolute():
            prereg_path = ROOT / prereg_path
        identity = prereg_identity(prereg_path)
        prereg = json.loads(prereg_path.read_text())
        family = prereg_family(prereg)
    else:
        identity = None
        family = iter1_family()
    checks = dev_compare.Checks(args.allow_partial)
    floors_meta = None
    tiers = {"a": Tier("a", args.registry, checks, not args.no_oracle_check, None)}
    for tier in tiers.values():
        checks(f"tier {tier.name}: DEV sets only",
               all(s in ("set_0", "set_1") for s in tier.sets), f"{tier.sets}")
    if args.registry_c:
        if not args.floors:
            parser.error("--registry-c needs --floors")
        floors, floors_meta = load_floors(Path(args.floors))
        checks("tier c: floors.json finalized (floor = min(polish, reference))",
               floors_meta["finalized"] is True,
               f"finalized = {floors_meta['finalized']!r}", partial_ok=True)
        tiers["c"] = Tier("c", args.registry_c, checks,
                          not args.no_oracle_check, floors)
        checks("tier c: DEV sets only",
               all(s in ("set_0", "set_1") for s in tiers["c"].sets),
               f"{tiers['c'].sets}")
        checks("tier c: targets differ from tier a",
               tiers["c"].registry["targets_template"]
               != tiers["a"].registry["targets_template"],
               tiers["c"].registry["targets_template"])
    results = run_family(family, tiers)
    profiles = profiles_for(family, tiers)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "family": {k: v for k, v in family.items() if k != "tests"},
        "prereg": identity,
        "registries": {name: tier.spec for name, tier in tiers.items()},
        "floors": floors_meta,
        "settings": {"n_resamples": N_RESAMPLES, "bootstrap_seed": BOOT_SEED,
                     "value_floor": FLOOR, "data_taus": list(DATA_TAUS)},
        "dirs": {name: tier.dirs for name, tier in tiers.items()},
        "results": results, "profiles": profiles,
        "checks": checks.rows, "checks_failed": checks.failed,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "stats.json").write_text(json.dumps(report, indent=1) + "\n")
    (out_dir / "stats.md").write_text(markdown(report))
    for name, prof in profiles.items():
        if prof["performance"]:
            plot_profiles(prof, name, out_dir / f"profiles_tier_{name}.png",
                          prof["endpoint"])
    print(f"wrote {out_dir / 'stats.md'} and stats.json")
    for r in results["tests"]:
        s = r.get("primary")
        print(f"{r['id']:>14} {r['a']} - {r['b']} B{r['budget']} tier {r['tier']}: "
              + (f"diff {s['mean_diff']:+.4f} clustered [{s['ci95_clustered'][0]:+.4f}, "
                 f"{s['ci95_clustered'][1]:+.4f}] iid [{s['ci95_iid'][0]:+.4f}, "
                 f"{s['ci95_iid'][1]:+.4f}] p {s['wilcoxon_p_geometry']:.4f} "
                 f"holm {r['holm_p']:.4f}" if s else f"not computed ({r.get('missing')})"))
    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
