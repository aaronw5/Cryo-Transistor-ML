#!/usr/bin/env python3
"""DEV comparison against the measured refit floor, both tiers.

Tier c (``sky130c``, misspecified, ``--registry-c``) has a nonzero refit
floor per target (``--floors``, written by ``sky130c_floor.py``); tier a
(well-specified, ``--registry-a``) is read with floor 0.  Per-target values
come from ``dev_compare`` unchanged (``Checks``, ``OracleReference``,
``evaluate_arm``: the same hard checks, fixed-budget fresh RRMS, top-budget
traces); targets are paired across tiers by set/tag.

At every fixed budget:

* raw strict RRMS: all-device mean and combined (mean of the nMOS / pMOS
  family means) pooled over the targets and per set; a target with a failed
  included curve (``cryoml.strict_score``) has no paper RRMS, so the strict
  mean is NaN whenever one is present (the failed count, the valid-only mean
  and the penalized mean are shown next to it);
* excess = RRMS - floor: mean, combined, median;
* paired tables on excess (``dev_compare.paired``; diff = A - B, negative =
  A lower) for (sketch_surrogate_k8, sketch_none_k8), (hybrid_surrogate,
  hybrid_none), (anytime_sur_presearch_race, anytime_none_race) and every
  ``--pair A,B``;
* difference-in-differences per target, (A - B)_c - (A - B)_a, mean and
  bootstrap 95% CI (10k resamples over paired targets, seed 0).  The floor
  cancels inside each tier's A - B, so this equals the DiD of raw RRMS.

Over calls (top-budget traces, best-so-far paper RRMS at n calls):

* More-Wild data profiles: the fraction of targets with
  best(n) - floor <= tau (start - floor), tau in {0.1, 0.01}, start = the
  target's bank-oracle RRMS (0 NGSpice calls), on the 40-point log grid of
  calls from 10 to 5000;
* crossover budget B* per pair and tier: the smallest grid point at which
  the bootstrap 95% CI lower bound of mean(A - B) exceeds 0 (A higher), and
  the largest grid point at which the upper bound is below 0 (A lower).
  Trace values below the top budget come from the top-budget run (the
  ``dev_compare`` curve convention), not from a run at that budget.

  PYTHONPATH=src .venv/bin/python scripts/research/dev_compare_floor.py \
      --registry-c out/research/dev_sky130c/arms.json \
      --registry-a out/research/dev/arms_iter2.json \
      --floors out/research/sim_targets_sky130c/floors.json \
      --out out/research/dev_sky130c/compare_floor
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import dev_compare as dc  # noqa: E402
from hybrid_tr import best_at  # noqa: E402

DEFAULT_PAIRS = (("sketch_surrogate_k8", "sketch_none_k8"),
                 ("hybrid_surrogate", "hybrid_none"),
                 ("anytime_sur_presearch_race", "anytime_none_race"))
TAUS = (0.1, 0.01)
GRID_TOP = 5000


def normalize(registry: dict) -> dict:
    return {"sets": ["set_0", "set_1"], "bank": "data/processed/cold_loop/full_r1",
            "round_dir": "out/cold_loop/full/round_1",
            "targets_template": "out/research/sim_targets/{set}.npz", **registry}


def family(key: str) -> str:
    return "nmos" if key.split("/")[-1].startswith("nmos") else "pmos"


def means(values: dict[str, float]) -> dict:
    v = {k: float(x) for k, x in values.items()}
    if not v:
        return {"n": 0, "mean": math.nan, "combined": math.nan, "median": math.nan}
    fam = {f: [x for k, x in v.items() if family(k) == f] for f in ("nmos", "pmos")}
    nm = float(np.mean(fam["nmos"])) if fam["nmos"] else math.nan
    pm = float(np.mean(fam["pmos"])) if fam["pmos"] else math.nan
    return {"n": len(v), "mean": float(np.mean(list(v.values()))),
            "combined": (nm + pm) / 2, "nmos": nm, "pmos": pm,
            "median": float(np.median(list(v.values())))}


def by_set(values: dict[str, float], sets: list[str]) -> dict:
    return {s: means({k: x for k, x in values.items() if k.startswith(s + "/")})
            for s in sets}


class Tier:
    """One tier's evaluated arms (``dev_compare.evaluate_arm``), bank-oracle
    starts and floors."""

    def __init__(self, name: str, registry_path: Path, floors: dict | None,
                 allow_partial: bool, oracle_check: bool):
        self.name = name
        self.path = registry_path
        self.registry = normalize(json.loads(registry_path.read_text()))
        self.sets = self.registry["sets"]
        self.checks = dc.Checks(allow_partial)
        self.checks(f"[{name}] DEV sets only",
                    all(s in ("set_0", "set_1") for s in self.sets), f"{self.sets}")
        self.oracle = dc.OracleReference(self.registry, dc.DEV / "oracle_reference.json",
                                         oracle_check)
        self.arms = {}
        for arm in self.registry["arms"]:
            if arm.get("mode") == "reference":
                continue
            evaluated = dc.evaluate_arm(arm, self.sets, self.checks, self.oracle)
            if evaluated:
                self.arms[arm["label"]] = evaluated
            else:
                self.checks(f"[{name}] {arm['label']}: any run loaded", False, "",
                            partial_ok=True)
        self.targets = sorted({k for a in self.arms.values() for k in a["traces"]})
        self.floor = {}
        self.start = {}
        for key in self.targets:
            set_name, tag = key.split("/")
            if floors is None:
                self.floor[key] = 0.0
            else:
                row = floors["targets"].get(key)
                ok = row is not None and row.get("floor") is not None
                self.checks(f"[{name}] floor for {key}", ok, "", partial_ok=True)
                self.floor[key] = float(row["floor"]) if ok else math.nan
            value = (self.oracle.get(set_name, tag) if self.oracle.enabled
                     else self.oracle.seen.get((set_name, tag)))
            self.start[key] = float(value) if value is not None else math.nan

    def strict(self, label: str, budget: int) -> dict:
        """Per-target strict values: the stored RRMS when valid, NaN when the
        target has a failed included curve.  Targets without a card record
        (trace-mode budgets below the top) keep the stored value and are
        counted as unchecked."""
        arm = self.arms[label]
        values = arm["budgets"].get(budget, {})
        status = arm["status"].get(budget, {})
        out, failed, unchecked, penal = {}, [], [], {}
        for key, value in values.items():
            st = status.get(key)
            if st is not None and st["status"] == "failed":
                out[key] = math.nan
                failed.append(key)
                penal[key] = st["penalized_rrms"]
            else:
                if st is None or st["status"] != "valid":
                    unchecked.append(key)
                out[key] = float(value)
                penal[key] = float(value)
        return {"values": out, "failed": failed, "unchecked": unchecked,
                "penalized": penal}


def budget_rows(tier: Tier) -> list[dict]:
    rows = []
    for label, arm in tier.arms.items():
        for b in sorted(arm["budgets"]):
            st = tier.strict(label, b)
            values = st["values"]
            valid = {k: v for k, v in values.items() if math.isfinite(v)}
            excess = {k: v - tier.floor[k] for k, v in values.items()}
            rows.append({
                "tier": tier.name, "arm": label, "budget": int(b),
                "n_targets": len(values), "n_failed": len(st["failed"]),
                "failed_targets": st["failed"], "n_unchecked": len(st["unchecked"]),
                "raw": means(values), "raw_per_set": by_set(values, tier.sets),
                "raw_valid_only": means(valid),
                "raw_penalized": means(st["penalized"]),
                "excess": means(excess), "excess_per_set": by_set(excess, tier.sets),
                "excess_values": excess})
    return rows


def excess_values(tier: Tier, label: str, budget: int) -> dict:
    values = tier.strict(label, budget)["values"]
    return {k: v - tier.floor[k] for k, v in values.items()}


def paired_rows(tier: Tier, pairs: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for a, b in pairs:
        if a not in tier.arms or b not in tier.arms:
            continue
        for budget in sorted(set(tier.arms[a]["budgets"]) & set(tier.arms[b]["budgets"])):
            ea, eb = excess_values(tier, a, budget), excess_values(tier, b, budget)
            rows.append({"tier": tier.name, "a": a, "b": b, "budget": int(budget),
                         **dc.paired(ea, eb)})
    return rows


def did_rows(tc: Tier, ta: Tier, pairs: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for a, b in pairs:
        if not all(x in t.arms for x in (a, b) for t in (tc, ta)):
            continue
        budgets = (set(tc.arms[a]["budgets"]) & set(tc.arms[b]["budgets"])
                   & set(ta.arms[a]["budgets"]) & set(ta.arms[b]["budgets"]))
        for budget in sorted(budgets):
            ca, cb = (tc.strict(x, budget)["values"] for x in (a, b))
            aa, ab = (ta.strict(x, budget)["values"] for x in (a, b))
            keys = sorted(set(ca) & set(cb) & set(aa) & set(ab))
            d = np.array([(ca[k] - cb[k]) - (aa[k] - ab[k]) for k in keys])
            finite = np.isfinite(d)
            rows.append({
                "a": a, "b": b, "budget": int(budget), "n": len(keys),
                "n_nonfinite": int((~finite).sum()),
                "diff_c": float(np.mean([ca[k] - cb[k] for k in keys])) if keys else math.nan,
                "diff_a": float(np.mean([aa[k] - ab[k] for k in keys])) if keys else math.nan,
                "did_mean": float(np.mean(d)) if len(d) else math.nan,
                "did_ci95": dc.bootstrap_ci(d[finite]) if finite.any() else [math.nan] * 2,
                "per_target": {k: float(x) for k, x in zip(keys, d)}})
    return rows


def data_profiles(tier: Tier, grid: np.ndarray) -> dict:
    out = {}
    for label, arm in tier.arms.items():
        prof = {}
        for tau in TAUS:
            solved, degenerate = [], []
            for key in tier.targets:
                trace = arm["traces"].get(key)
                if trace is None:
                    continue
                start, floor = tier.start[key], tier.floor[key]
                gap = start - floor
                if not (math.isfinite(gap) and gap > 0):
                    degenerate.append(key)
                    solved.append(np.ones(len(grid), dtype=bool))
                    continue
                best = np.array([best_at(trace, g) for g in grid])
                solved.append(best - floor <= tau * gap)
            frac = np.mean(solved, axis=0) if solved else np.full(len(grid), np.nan)
            prof[str(tau)] = {"fraction": frac.tolist(), "n_targets": len(solved),
                              "degenerate_targets": degenerate}
        out[label] = prof
    return out


def crossovers(tier: Tier, pairs: list[tuple[str, str]], grid: np.ndarray) -> list[dict]:
    rows = []
    for a, b in pairs:
        if a not in tier.arms or b not in tier.arms:
            continue
        keys = sorted(set(tier.arms[a]["traces"]) & set(tier.arms[b]["traces"]))
        top = min(tier.arms[a]["top"], tier.arms[b]["top"])
        lower, upper, mean = [], [], []
        for g in grid:
            d = np.array([best_at(tier.arms[a]["traces"][k], g)
                          - best_at(tier.arms[b]["traces"][k], g) for k in keys])
            d = d[np.isfinite(d)]
            ci = dc.bootstrap_ci(d) if len(d) else [math.nan, math.nan]
            lower.append(ci[0])
            upper.append(ci[1])
            mean.append(float(np.mean(d)) if len(d) else math.nan)
        shown = [i for i, g in enumerate(grid) if g <= top * (1 + 1e-9)]
        b_star = next((float(grid[i]) for i in shown if lower[i] > 0), None)
        a_lower = [float(grid[i]) for i in shown if upper[i] < 0]
        rows.append({"tier": tier.name, "a": a, "b": b, "n": len(keys),
                     "top": int(top), "b_star_a_higher": b_star,
                     "last_a_lower": a_lower[-1] if a_lower else None,
                     "grid": [float(grid[i]) for i in shown],
                     "mean_diff": [mean[i] for i in shown],
                     "ci_lower": [lower[i] for i in shown],
                     "ci_upper": [upper[i] for i in shown]})
    return rows


# ----------------------------------------------------------------- markdown

def f4(v) -> str:
    return dc.fmt(v) if v is not None else "–"


def markdown(report: dict) -> str:
    lines = ["# DEV comparison against the refit floor (tier c sky130c, tier a)", "",
             f"Tier c registry: `{report['registry_c']}`; tier a registry: "
             f"`{report['registry_a']}`; floors: `{report['floors']}` "
             f"(finalized: {report['floors_finalized']}). Tier-a floor = 0. "
             "RRMS = paper RRMS with frozen inclusion; strict = NaN for a target "
             "with a failed included curve. "
             + ("**PARTIAL report (missing runs are warnings).**"
                if report["partial"] else ""), "",
             "## Floors (tier c)", "",
             "| set | n | start mean | polish mean | reference mean | floor mean | "
             "floor combined | noise-only mean |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for set_name, s in report["floor_summary"].items():
        cell = lambda f, k="all_device_mean": f4(s.get(f, {}).get(k))  # noqa: E731
        lines.append(f"| {set_name} | {s.get('floor', {}).get('n', '–')} | {cell('start')} | "
                     f"{cell('polish')} | {cell('reference')} | {cell('floor')} | "
                     f"{cell('floor', 'combined')} | {cell('noise_only')} |")
    lines += ["", "## Raw strict RRMS and excess over the floor", "",
              "| tier | arm | budget | n | failed | raw mean | raw combined | "
              "valid-only mean | excess mean | excess combined | excess median |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["budget_rows"]:
        lines.append(f"| {r['tier']} | {r['arm']} | {r['budget']} | {r['n_targets']} | "
                     f"{r['n_failed']} | {f4(r['raw']['mean'])} | "
                     f"{f4(r['raw']['combined'])} | {f4(r['raw_valid_only']['mean'])} | "
                     f"{f4(r['excess']['mean'])} | {f4(r['excess']['combined'])} | "
                     f"{f4(r['excess']['median'])} |")
    lines += ["", "## Paired excess (diff = A - B; negative = A lower)", "",
              "| tier | A | B | budget | n | W/T/L | mean diff | 95% CI | Wilcoxon p |",
              "|---|---|---|---:|---:|---|---:|---|---:|"]
    for r in report["paired"]:
        lines.append(f"| {r['tier']} | {r['a']} | {r['b']} | {r['budget']} | {r['n']} | "
                     f"{r['wins']}/{r['ties']}/{r['losses']} | {f4(r['mean_diff'])} | "
                     f"[{f4(r['ci95'][0])}, {f4(r['ci95'][1])}] | {f4(r['wilcoxon_p'])} |")
    lines += ["", "## Cross-tier difference-in-differences: (A - B)_c - (A - B)_a", "",
              "| A | B | budget | n | mean (A-B) tier c | mean (A-B) tier a | DiD mean | "
              "95% CI | non-finite |", "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for r in report["did"]:
        lines.append(f"| {r['a']} | {r['b']} | {r['budget']} | {r['n']} | "
                     f"{f4(r['diff_c'])} | {f4(r['diff_a'])} | {f4(r['did_mean'])} | "
                     f"[{f4(r['did_ci95'][0])}, {f4(r['did_ci95'][1])}] | "
                     f"{r['n_nonfinite']} |")
    lines += ["", "## Crossover budgets (trace values, 40-point log grid 10-5000 calls)", "",
              "B* = smallest grid point where the bootstrap 95% CI lower bound of "
              "mean(A - B) > 0 (A higher); last A-lower = largest grid point where the "
              "upper bound < 0.", "",
              "| tier | A | B | n | grid top | B* (A higher) | last A-lower |",
              "|---|---|---|---:|---:|---:|---:|"]
    def calls(v):
        return "none" if v is None else f"{v:.0f}"

    for r in report["crossover"]:
        lines.append(f"| {r['tier']} | {r['a']} | {r['b']} | {r['n']} | {r['top']} | "
                     f"{calls(r['b_star_a_higher'])} | {calls(r['last_a_lower'])} |")
    grid = report["grid"]
    log_grid = np.log(np.asarray(grid))
    marks = sorted({int(np.argmin(np.abs(log_grid - math.log(m))))
                    for m in (10, 100, 500, 1000, 2000, 5000)})
    for tau in (str(t) for t in TAUS):
        lines += ["", f"## Data profiles, tau = {tau}: fraction of targets with "
                  f"best - floor <= tau (start - floor)", "",
                  "| tier | arm | " + " | ".join(f"{grid[i]:.0f}" for i in marks) + " |",
                  "|---|---|" + "---:|" * len(marks)]
        for tier_name, profiles in report["profiles"].items():
            for label, prof in profiles.items():
                frac = prof[tau]["fraction"]
                lines.append(f"| {tier_name} | {label} | "
                             + " | ".join(f"{frac[i]:.2f}" for i in marks) + " |")
    lines += ["", "## Hard checks", ""]
    lines += [f"- {c['level']}: {c['check']}" + (f" ({c['detail']})" if c["detail"] else "")
              for c in report["checks"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-c", default="out/research/dev_sky130c/arms.json")
    parser.add_argument("--registry-a", default="out/research/dev/arms_iter2.json")
    parser.add_argument("--floors", default="out/research/sim_targets_sky130c/floors.json")
    parser.add_argument("--out", default="out/research/dev_sky130c/compare_floor")
    parser.add_argument("--pair", action="append", default=[],
                        help="extra pair A,B (repeatable)")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-oracle-check", action="store_true")
    args = parser.parse_args()
    resolve = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p  # noqa: E731
    floors_path = resolve(args.floors)
    floors = json.loads(floors_path.read_text())
    pairs = list(DEFAULT_PAIRS) + [tuple(p.split(",", 1)) for p in args.pair]
    tc = Tier("c", resolve(args.registry_c), floors, args.allow_partial,
              not args.no_oracle_check)
    ta = Tier("a", resolve(args.registry_a), None, args.allow_partial,
              not args.no_oracle_check)
    tc.checks("[c] floors finalized (floor = min(polish, reference))",
              bool(floors.get("finalized")), "", partial_ok=True)
    grid = dc.log_grid(GRID_TOP)
    report = {
        "registry_c": str(args.registry_c), "registry_a": str(args.registry_a),
        "floors": str(args.floors), "floors_finalized": bool(floors.get("finalized")),
        "partial": bool(args.allow_partial), "pairs": [list(p) for p in pairs],
        "taus": list(TAUS), "grid": grid.tolist(),
        "floor_summary": floors.get("per_set", {}),
        "budget_rows": budget_rows(tc) + budget_rows(ta),
        "paired": paired_rows(tc, pairs) + paired_rows(ta, pairs),
        "did": did_rows(tc, ta, pairs),
        "profiles": {"c": data_profiles(tc, grid), "a": data_profiles(ta, grid)},
        "crossover": crossovers(tc, pairs, grid) + crossovers(ta, pairs, grid),
        "targets": {"c": tc.targets, "a": ta.targets},
        "start_bank_oracle": {"c": tc.start, "a": ta.start},
        "floor_values": tc.floor}
    report["checks"] = tc.checks.rows + ta.checks.rows
    report["checks_failed"] = tc.checks.failed or ta.checks.failed
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare_floor.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "compare_floor.md").write_text(markdown(report))
    print(f"wrote {out / 'compare_floor.md'}, compare_floor.json")
    for r in report["crossover"]:
        print(r["tier"], r["a"], r["b"], "B*", r["b_star_a_higher"],
              "last A-lower", r["last_a_lower"])
    return 1 if report["checks_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
