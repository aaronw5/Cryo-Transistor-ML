#!/usr/bin/env python3
"""I24-1 decide: Part A (three-way gate, retrospective), Part C (TEST
prediction, proposed amendment only), Part B (g v2 on top of B1, IHP).

Called by ``i24_1_gate3.py --stage decide``; writes I24_1/decision.json.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "out/research/explore"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i22_1_baselines as d22  # noqa: E402
from cryoml.explore import i22_1_baselines as m22  # noqa: E402
from cryoml.explore import i24_1_gate3 as m  # noqa: E402
from cryoml.explore.i3_2_bias import paired_ratio  # noqa: E402

OUT = EXP / "I24_1"
DATASETS = m22.DATASETS
IHP_TEMPS = (233, 300, 343, 398)
ARM_NAME = {"C0_refit": "C0_refit", "g": "selected g v2", "B1": "B1"}


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=_default) + "\n")


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


def rel(p: Path) -> str:
    return str(Path(p).relative_to(ROOT))


def s_g_source(ds: str) -> tuple[float, dict, str]:
    """Frozen-rule S3_D (v1 recipe, state C) and per-record log ratios."""
    if ds == "sky130":
        p = EXP / "I19_1/s3/sky130_v1.json"
        x = jload(p)
        return float(x["S3_D"]), x["per_record"], rel(p)
    if ds in ("nist4", "nistrt"):
        p = EXP / f"I20_1/s3/{ds}_v1.json"
        x = jload(p)
        return float(x["S3_D"]), x["per_record"], rel(p)
    if ds.startswith("ihp"):
        p = EXP / f"I19_1/s3/{ds}_v1.json"
        x = jload(p)
        return float(x["S3_D"]), x["per_record"], rel(p)
    p = EXP / "I21_1/s3tud/s3tud.json"
    x = jload(p)["per_temperature"][ds[3:]]
    return float(x["S3_D_state_C"]), x["per_record"], rel(p) + f" per_temperature.{ds[3:]}.S3_D_state_C"


def clusters(ds: str, tags) -> dict | None:
    if ds.startswith("tud"):
        return {t: d22.m6.group_of(t) for t in tags}
    return None


def group_fn(ds: str):
    return d22.m6.group_of if ds.startswith("tud") else (lambda t: t)


def part_a() -> dict:
    data, per, valid_all = {}, {}, True
    for ds in DATASETS:
        L = jload(OUT / f"partA/logo_{ds}.json")
        tags = list(L["per_record"])
        st = d22.stored_rows(ds)
        A22 = jload(EXP / f"I22_1/partA_{ds}/results.json")["per_device"]
        rows = {"C0_refit": {t: float(st["C0_refit"][t]) for t in tags},
                "g": {t: float(st["selected_g_v2"][t]) for t in tags},
                "B1": {t: float(A22["B1"][t]) for t in tags},
                "B1_logo": {t: float(L["per_record"][t]["b1_logo_rrms"]) for t in tags}}
        s_g, lr_g, src = s_g_source(ds)
        lr_b = {t: L["per_record"][t]["log_ratio"] for t in tags}
        s_b = L["S_B1"] if L["valid"] else float("nan")
        prim = m.choose(s_g, s_b)
        two = m.choose_two_way(s_g)
        s1 = m.choose_raw(s_g, s_b)
        s2 = m.choose(s_g, L["S_B1_full_screen"] if L["S_B1_full_valid"] else float("nan"))["arm"]
        s3 = m.per_device_gate(tags, group_fn(ds), {t: float(lr_g[t]) for t in tags}, lr_b)
        mean_lr_g = float(np.nanmean([float(lr_g[t]) for t in tags]))
        cl = clusters(ds, tags)
        series = {"C0_refit": rows["C0_refit"], "g always": rows["g"], "B1 always": rows["B1"],
                  "B1 always (LOGO rows)": rows["B1_logo"],
                  "two-way frozen gate": rows[two], "three-way gate": rows[prim["arm"]],
                  "three-way gate (LOGO B1 rows)": rows["B1_logo" if prim["arm"] == "B1" else prim["arm"]],
                  "S1 raw minimum": rows[s1], "S2 full-screen S_B1": rows[s2],
                  "S3 leave-group-out gate": {t: rows[s3[t]["arm"]][t] for t in tags}}
        gmr = {k: paired_ratio(v, rows["C0_refit"], cl) for k, v in series.items() if k != "C0_refit"}
        best = min(("C0_refit", "g", "B1"), key=lambda a: 1.0 if a == "C0_refit" else gmr[
            "g always" if a == "g" else "B1 always"]["ratio"])
        series["oracle (post hoc)"] = rows[best]
        gmr["oracle (post hoc)"] = paired_ratio(rows[best], rows["C0_refit"], cl)
        va = {"VA1": bool(abs(s_g - m.S_G_FREEZE[ds]) < m.VA1_TOL), "VA1_S_g": s_g,
              "VA1_per_record_mean": mean_lr_g, "VA2": bool(L["VA2"]["ok"]), "VA3": bool(L["VA3"]["ok"]),
              "VA3_required": L["VA3"]["required"], "VA3_max": L["VA3"]["max_abs_diff"],
              "VA4": bool(L["VA4"]["ok"] and all(len(v) == len(tags) for v in rows.values())),
              "VA5": bool(L["VA5_dev_check"]["ok"]), "VA6": bool(L["unit_tests_passed"])}
        va["valid"] = all(va[k] for k in ("VA1", "VA2", "VA3", "VA4", "VA5", "VA6"))
        valid_all = valid_all and va["valid"]
        aggr = {k: m22.family_means(v, d22.is_pmos) for k, v in series.items()}
        per[ds] = {"name": m22.NAMES[ds], "n": len(tags), "unit": f"cluster ({len(set(cl.values()))} groups)" if cl
                   else "device", "S_g": s_g, "S_g_source": src, "yhat_g": m.yhat_g(s_g), "S_B1": s_b,
                   "S_B1_full_screen": L["S_B1_full_screen"], "S_B1_n_nonfinite": L["n_nonfinite"],
                   "logo": {k: L[k] for k in ("n_equal", "n_new_refits", "n_empty", "frac_equal", "n_groups")},
                   "choice": prim, "two_way": two, "S1": s1, "S2": s2,
                   "S3_counts": {a: sum(1 for t in tags if s3[t]["arm"] == a) for a in ("C0_refit", "g", "B1")},
                   "S3_per_device": s3, "oracle": best, "gmr_vs_C0_refit": gmr, "aggregates": aggr,
                   "validity": va, "rows": rows}
        data[ds] = {"tags": tags, "clusters": cl, "rows": series, "control": rows["C0_refit"]}
    names = [k for k in data[DATASETS[0]]["rows"]]
    boot = m.pooled_bootstrap(data, names)
    paired = {"three-way / two-way": m.paired_pooled(boot, "three-way gate", "two-way frozen gate"),
              "three-way / B1 always": m.paired_pooled(boot, "three-way gate", "B1 always"),
              "three-way / g always": m.paired_pooled(boot, "three-way gate", "g always"),
              "S3 / two-way": m.paired_pooled(boot, "S3 leave-group-out gate", "two-way frozen gate"),
              "three-way (LOGO B1 rows) / two-way": m.paired_pooled(boot, "three-way gate (LOGO B1 rows)",
                                                                    "two-way frozen gate")}
    pooled = m.strip_boot(boot)
    out = m.part_a_outcome({ds: per[ds]["gmr_vs_C0_refit"]["three-way gate"]["ratio"] for ds in DATASETS},
                           pooled["three-way gate"]["pooled"], pooled["B1 always"]["pooled"],
                           pooled["g always"]["pooled"], paired["three-way / two-way"], valid_all)
    return {"datasets": per, "pooled": pooled, "paired_pooled": paired, "outcome": out, "valid": valid_all,
            "not_buildable": [ds for ds in DATASETS if not np.isfinite(per[ds]["S_B1"])],
            "bootstrap": {"n_boot": m.N_BOOT, "seed": m.SEED, "decisions_fixed": True}}


def part_c(A: dict) -> dict:
    td = jload(EXP / "I19_1/test_gate_decisions.json")
    inv = {v: k for k, v in m22.FREEZE_KEYS.items()}
    rows, amend = [], []
    for r in td["decisions"]:
        ds = inv[r["name"]]
        a = A["datasets"][ds]
        committed = "g (correction)" if r["decision"] == "apply" else "C0_refit"
        three = a["choice"]["arm"]
        two = a["two_way"]
        c3 = {"g": "g (correction)", "C0_refit": "C0_refit", "B1": "B1"}[three]
        c2 = {"g": "g (correction)", "C0_refit": "C0_refit"}[two]
        row = {"ds": ds, "test_split": r["test_split"], "n_test": r["n_test"], "committed_a755e18": committed,
               "committed_arm": r["arm"], "committed_S3_D": r["S3_D"], "two_way_frozen": c2,
               "three_way": c3, "S_g": a["S_g"], "S_B1": a["S_B1"], "differs_from_a755e18": c3 != committed,
               "differs_from_two_way": c3 != c2}
        rows.append(row)
        if row["differs_from_a755e18"]:
            amend.append(f"{r['test_split']}: {committed} -> {c3} (S_g {a['S_g']:+.4f}, S_B1 {a['S_B1']:+.4f})")
    return {"rows": rows, "proposed_amendment": amend, "status": "proposed only; no TEST decision file written",
            "test_data_not_opened": True}


def part_b() -> dict:
    per, valid = {}, True
    for t in IHP_TEMPS:
        p = OUT / f"partB/ihp_{t}K/decision.json"
        if not p.exists():
            per[str(t)] = None
            valid = False
            continue
        x = jload(p)
        per[str(t)] = {k: x[k] for k in ("validity", "comparisons", "labels", "aggregates", "lambda_star_counts",
                                         "calls_mean", "timing")}
        per[str(t)]["rows"] = x["rows"]
        valid = valid and x["validity"]["valid"]
    if not all(per.values()):
        return {"per_temperature": per, "outcome": {"label": "incomplete"}, "valid": False}
    out = m.part_b_outcome({k: v["comparisons"]["B1+g v2 vs B1"] for k, v in per.items()}, valid)
    return {"per_temperature": per, "outcome": out, "valid": valid}


def predictions(A: dict, C: dict, B: dict) -> dict:
    exp = {"sky130": "g", "nist4": "g", "nistrt": "g", "tud4": "C0_refit", "tud295": "C0_refit",
           **{f"ihp{t}": "B1" for t in IHP_TEMPS}}
    P = {"P1": all(A["datasets"][ds]["choice"]["arm"] == a for ds, a in exp.items()),
         "P1_per_dataset": {ds: A["datasets"][ds]["choice"]["arm"] == a for ds, a in exp.items()},
         "P2": all(A["datasets"][ds]["logo"]["frac_equal"] >= 0.75 for ds in DATASETS if not ds.startswith("tud")),
         "P3": A["outcome"]["label"] == "three-way gate supported (retrospective)",
         "P4": A["pooled"]["three-way gate"]["pooled"] < 0.85,
         "P5": sorted(r["ds"] for r in C["rows"] if r["differs_from_a755e18"]) == [f"ihp{t}" for t in IHP_TEMPS]
         and all(r["three_way"] == "B1" for r in C["rows"] if r["differs_from_a755e18"])}
    if B.get("valid"):
        pt = B["per_temperature"]
        P["P6"] = bool(B["outcome"]["label"] == "g v2 gain over B1 not shown" and
                       sum(pt[k]["comparisons"]["B1+g v2 vs B1"]["ratio"] > 1 for k in pt) >= 3)
        P["P7"] = all(0.99 <= pt[k]["comparisons"]["B1_cont vs B1"]["ratio"] <= 1.01 for k in pt)
    return P


def stage_decide() -> dict:
    A = part_a()
    C = part_c(A)
    B = part_b()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"prereg": "out/research/explore/I24_1/prereg.json", "git_head": head, "part_A": A, "part_C": C,
           "part_B": B, "predictions": predictions(A, C, B)}
    jdump(out, OUT / "decision.json")
    print(json.dumps({"A": A["outcome"], "valid_A": A["valid"],
                      "choices": {ds: A["datasets"][ds]["choice"]["arm"] for ds in DATASETS},
                      "S_B1": {ds: round(A["datasets"][ds]["S_B1"], 4) for ds in DATASETS},
                      "pooled": {k: (round(v["pooled"], 4), round(v["ci_low"], 4), round(v["ci_high"], 4))
                                 for k, v in A["pooled"].items()},
                      "paired": {k: (round(v["ratio"], 4), round(v["ci_low"], 4), round(v["ci_high"], 4))
                                 for k, v in A["paired_pooled"].items()},
                      "C": C["proposed_amendment"], "B": B["outcome"], "pred": out["predictions"]},
                     indent=1, default=_default), flush=True)
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    stage_decide()
