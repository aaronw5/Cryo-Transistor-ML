#!/usr/bin/env python3
"""I23-1 decide stage; called by i23_1_budget.py --stage decide.

Reads the I23-1 outputs (partA_<ihp>/results.json, verify_<ds>.json,
fixl1_<ds>/results.json), the I22-1 rows and cost table, and the stored
lambda = 1 V and comparison rows.  No simulation, no training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "out/research/explore"
OUT = EXP / "I23_1"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

from cryoml.explore import i22_1_baselines as m22  # noqa: E402
from cryoml.explore import i23_1_budget as m  # noqa: E402

LAM = m.FIXED_LAMBDA


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def rel(p) -> str:
    return str(Path(p).relative_to(ROOT))


def is_pmos(t: str) -> bool:
    return "pmos" in t or t.startswith("pfet")


def temp(ds: str) -> int:
    return int(ds[3:])


def clusters(ds: str, tags) -> dict | None:
    if not ds.startswith("tud"):
        return None
    from cryoml.explore.i6_1_tudelft import group_of
    return {t: group_of(t) for t in tags}


def gmr(a: dict, b: dict, cl=None) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    keys = sorted(set(a) & set(b))
    return paired_ratio({k: a[k] for k in keys}, {k: b[k] for k in keys}, cl)


def stored(ds: str) -> dict:
    """selected g v2, C0_refit, M_bias v1 rows (I22-1 comparison rows)."""
    import plot_freeze_draft as pf
    d = pf.datasets()[m22.FREEZE_KEYS[ds]]
    return {"selected_g_v2": dict(d["sel"]), "C0_refit": dict(d["c0"]), "M_bias_v1": dict(d["v1"])}


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
def part_A() -> dict:
    out, rows_all, valid = {}, {}, {}
    for ds in m.IHP:
        r = jload(OUT / f"partA_{ds}/results.json")
        i22 = jload(EXP / f"I22_1/partA_{ds}/results.json")
        st = stored(ds)
        held = r["held_out"]
        R = {"C0_refit": {t: st["C0_refit"][t] for t in held}, "C0_call": r["per_device"]["C0_call"],
             "B1": {t: i22["per_device"]["B1"][t] for t in held}, "B1_cap": r["per_device"]["B1_cap"],
             "selected_g_v2": {t: st["selected_g_v2"][t] for t in held},
             "C0_refit_rerun": r["per_device"]["C0_refit_rerun"]}
        cmp = {"B1_vs_C0_call": gmr(R["B1"], R["C0_call"]), "C0_call_vs_C0_refit": gmr(R["C0_call"], R["C0_refit"]),
               "C0_call_vs_selected_g_v2": gmr(R["C0_call"], R["selected_g_v2"]),
               "B1_cap_vs_C0_refit": gmr(R["B1_cap"], R["C0_refit"]), "B1_vs_B1_cap": gmr(R["B1"], R["B1_cap"]),
               "B1_vs_C0_refit": gmr(R["B1"], R["C0_refit"]), "B1_cap_vs_C0_call": gmr(R["B1_cap"], R["C0_call"])}
        labels = {k: m.label_vs_control(v) for k, v in cmp.items()}
        summ = {a: m22.family_means(v, is_pmos) for a, v in R.items()}
        calls = {"C0_call": r["validity"]["VA3_detail"]["C0_call"], "B1_cap": r["validity"]["VA3_detail"]["B1_cap"],
                 "n_capped": r["n_capped"], "B1_refit_calls_per_device": float(np.mean(list(r["N_B1"].values()))),
                 "C0_refit_calls_per_device": float(np.mean(list(r["N_C0"].values()))),
                 "C0_call_calls_per_device": float(np.mean([v for v in r["n_sim"]["C0_call"].values()])),
                 "B1_cap_calls_per_device": float(np.mean([v for v in r["n_sim"]["B1_cap"].values()])),
                 "B1_screen_calls_per_device": (float(np.mean(list(r["screen_calls_per_device"].values())))
                                                if r.get("screen_calls_per_device") else None)}
        valid[ds] = bool(r["validity"]["valid"])
        out[ds] = {"validity": {k: v for k, v in r["validity"].items() if k != "VA4_dev_check"},
                   "summary": summ, "comparisons": cmp, "labels": labels, "calls": calls,
                   "runtime_s": r["runtime_s"]}
        rows_all[ds] = R
    rule = m.rule_partA({ds: out[ds]["comparisons"]["B1_vs_C0_call"] for ds in m.IHP}, valid)
    pooled = {k: m.pooled_gmr([out[ds]["comparisons"][k]["ratio"] for ds in m.IHP])
              for k in out[m.IHP[0]]["comparisons"]}
    return {"datasets": out, "rows": rows_all, "rule": rule, "pooled_over_4_temperatures": pooled, "valid": valid}


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
def fixed_rows(ds: str) -> tuple[dict, dict, dict, dict]:
    """fixed_l1 rows {tag: rrms}, n_sim {tag: n}, source {tag: path}, training CPU s."""
    rows, nsim, src, cpu = {}, {}, {}, 0.0

    def take(path, tags=None, row_key="heldout_rows"):
        nonlocal cpu
        d = jload(path)
        use = tags if tags is not None else list(d["per_device"])
        for t in use:
            rows[t] = float(d["per_device"][t])
            nsim[t] = d[row_key][t].get("n_sim")
            src[t] = rel(path)
        return d
    if ds in ("sky130", "ihp300", "tud4", "tud295"):
        p = {"sky130": EXP / "I15_2/lodo_sym_e1_l1", "ihp300": EXP / "I17_2/lodo_sym_e1_l1",
             "tud4": EXP / "I18_2/lodo_4K_sym_e1_l1", "tud295": EXP / "I18_2/lodo_295K_sym_e1_l1"}[ds]
        d = take(p / "results.json")
        cpu = float(sum(sum(v) for v in d["train_seconds_per_member"].values()))
    elif ds == "nist4":
        d = jload(EXP / "I20_1/lodo_nist4/results.json")
        reused = [t for t, lam in d["lambda_by_device"].items() if float(lam) == LAM]
        take(EXP / "I20_1/lodo_nist4/results.json", reused)
        cpu = float(sum(sum(d["train_seconds_per_member"][t]) for t in reused))
        n = take(OUT / "fixl1_nist4/results.json")
        cpu += float(sum(sum(v) for v in n["timing"]["train_seconds_per_member"].values()))
    else:
        n = take(OUT / f"fixl1_{ds}/results.json")
        cpu = float(sum(sum(v) for v in n["timing"]["train_seconds_per_member"].values()))
    return rows, nsim, src, {"train_cpu_s": round(cpu, 1)}


def part_B() -> dict:
    cost = jload(EXP / "I22_1/cost.json")["datasets"]
    out, rows_all, valid = {}, {}, {}
    for ds in m.DATASETS:
        st = stored(ds)
        F, nsim, src, cpu = fixed_rows(ds)
        tags = sorted(st["selected_g_v2"])
        R = {"fixed_l1": F, **{a: {t: st[a][t] for t in tags} for a in ("selected_g_v2", "C0_refit", "M_bias_v1")}}
        cl = clusters(ds, tags)
        cmp = {"fixed_vs_selected": gmr(F, R["selected_g_v2"], cl), "fixed_vs_C0_refit": gmr(F, R["C0_refit"], cl),
               "fixed_vs_M_bias_v1": gmr(F, R["M_bias_v1"], cl),
               "selected_vs_C0_refit": gmr(R["selected_g_v2"], R["C0_refit"], cl)}
        lab_sel = m22.label_vs_reference(cmp["fixed_vs_selected"], "fixed", "selected").replace(
            "fixed better than selected", "fixed better").replace("selected better than fixed", "selected better")
        # validity
        v = {}
        reused = ds in m.REUSED
        new = (OUT / f"fixl1_{ds}/results.json").exists()
        if reused:
            vb1 = jload(OUT / f"verify_{ds}.json")
            v["VB1"] = bool(vb1["VB1"])
            v["VB1_max_abs_diff"] = vb1["max_abs_diff"]
            v["VB1_samples"] = vb1["sample_folds"]
            v["VB5_unit_tests_verify"] = bool(vb1["unit_tests_passed"])
        if new:
            nv = jload(OUT / f"fixl1_{ds}/results.json")["validity"]
            v["VB2"] = bool(nv["VB2"])
            v["VB2_overlap_n"] = nv["VB2_overlap_n"]
            v["VB2_max_abs_diff"] = nv["VB2_max_abs_diff"]
            v["VB5_unit_tests_new"] = bool(nv["VB5_unit_tests"])
            v["VB4_new"] = bool(nv["VB4"])
        n_fail = int(sum(1 for t in tags if not np.isfinite(F.get(t, np.nan))))
        v["VB3_complete"] = set(F) == set(tags)
        v["VB3_n_failed"] = n_fail
        v["VB3"] = bool(v["VB3_complete"] and n_fail <= m.MAX_FAIL_FRACTION * len(tags))
        dev = dev_check(ds, tags)
        v["VB4"] = bool(dev["ok"])
        v["valid"] = bool(v.get("VB1", True) and v.get("VB2", True) and v["VB3"] and v["VB4"]
                          and v.get("VB5_unit_tests_verify", True) and v.get("VB5_unit_tests_new", True))
        valid[ds] = v["valid"]
        c = cost[ds]
        c0 = c["arms"]["C0_refit"]["refit_calls_mean"]
        refit_mean = float(np.mean([x for x in nsim.values() if x is not None]))
        n_folds = 7 if ds.startswith("tud") else len(tags)
        out[ds] = {"name": m22.NAMES[ds], "n": len(tags), "unit": "cluster (7)" if cl else "device",
                   "summary": {a: m22.family_means(r, is_pmos) for a, r in R.items()},
                   "comparisons": cmp, "label_vs_selected": lab_sel,
                   "non_inferior_to_selected": m.non_inferior(cmp["fixed_vs_selected"]),
                   "label_vs_C0_refit": m.label_vs_control(cmp["fixed_vs_C0_refit"]),
                   "label_selected_vs_C0_refit": m.label_vs_control(cmp["selected_vs_C0_refit"]),
                   "validity": v, "sources": sorted(set(src.values())),
                   "n_reused": int(sum(1 for s in src.values() if "I23_1" not in s)),
                   "n_new": int(sum(1 for s in src.values() if "I23_1" in s)),
                   "cost": {"fixed_l1_refit_calls_mean": refit_mean, "fixed_l1_pre_calls": 0,
                            "fixed_l1_trainings": 5 * n_folds, **cpu,
                            "fixed_l1_ratio_to_C0_refit": refit_mean / c0,
                            "C0_refit_refit_calls_mean": c0,
                            "selected_total_calls_per_device": c["arms"]["selected_g_v2"]["total_calls_per_device"],
                            "selected_ratio_to_C0_refit": c["arms"]["selected_g_v2"]["ratio_to_C0_refit_refit_calls"],
                            "B1_total_calls_per_device": c["arms"]["B1"]["total_calls_per_device"],
                            "starting_card_calls_per_device": c["starting_cards"]["per_device_mean"]}}
        rows_all[ds] = R
    rule = m.rule_partB({ds: out[ds]["comparisons"]["fixed_vs_selected"] for ds in m.DATASETS}, valid)
    pooled = {"fixed_l1_vs_C0_refit": m.pooled_gmr([out[ds]["comparisons"]["fixed_vs_C0_refit"]["ratio"]
                                                    for ds in m.DATASETS]),
              "selected_vs_C0_refit": m.pooled_gmr([out[ds]["comparisons"]["selected_vs_C0_refit"]["ratio"]
                                                    for ds in m.DATASETS]),
              "fixed_l1_vs_selected": m.pooled_gmr([out[ds]["comparisons"]["fixed_vs_selected"]["ratio"]
                                                    for ds in m.DATASETS])}
    i22 = jload(EXP / "I22_1/decision.json")["part_A"]["pooled_gmr_vs_C0_refit"]
    pooled["I22_1_pooled_vs_C0_refit"] = i22
    return {"datasets": out, "rows": rows_all, "rule": rule, "pooled": pooled, "valid": valid,
            "non_inferior_all_9": [ds for ds in m.DATASETS if out[ds]["non_inferior_to_selected"]]}


def dev_check(ds: str, tags) -> dict:
    import i22_1_baselines as d22
    return d22.dev_check(ds, tags)


def predictions(A: dict, B: dict) -> dict:
    a = A["datasets"]
    b = B["datasets"]
    return {"P1": A["rule"]["label"] == "B1 gain survives the call match",
            "P2": {ds: bool(a[ds]["comparisons"]["C0_call_vs_C0_refit"]["ratio"] < 1) for ds in m.IHP},
            "P3": int(sum(a[ds]["comparisons"]["B1_cap_vs_C0_refit"]["ratio"] > 1 for ds in m.IHP)) >= 3,
            "P4": B["rule"]["label"] == "fixed lambda 1 V non-inferior to the selected g v2",
            "P5": {**{ds: b[ds]["label_vs_C0_refit"] == "gain" for ds in m.GAIN_SET},
                   **{ds: b[ds]["label_vs_C0_refit"] != "gain" for ds in m.IHP}},
            "P6": {ds: bool(b[ds]["cost"]["fixed_l1_ratio_to_C0_refit"] <= 1.1) for ds in m.DATASETS}}


def stage_decide() -> dict:
    A = part_A()
    B = part_B()
    out = {"prereg": rel(OUT / "prereg.json"), "amendments": [], "part_A": A, "part_B": B,
           "predictions": predictions(A, B)}
    jdump(out, OUT / "decision.json")
    print(json.dumps({"A_rule": A["rule"], "B_rule": B["rule"], "pooled_B": B["pooled"],
                      "A": {ds: {k: round(v["ratio"], 3) for k, v in A["datasets"][ds]["comparisons"].items()}
                            for ds in m.IHP},
                      "B": {ds: {k: (round(v["ratio"], 3), round(v["ci_low"], 3), round(v["ci_high"], 3))
                                 for k, v in B["datasets"][ds]["comparisons"].items()} for ds in m.DATASETS},
                      "predictions": out["predictions"]}, indent=1, default=float), flush=True)
    return out


if __name__ == "__main__":
    stage_decide()
