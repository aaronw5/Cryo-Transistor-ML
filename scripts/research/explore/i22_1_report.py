#!/usr/bin/env python3
"""I22-1 cost (Part B) and decide stages; called by i22_1_baselines.py.

Reads the I22-1 outputs (screen_<ds>.json, partA_<ds>/results.json,
partC/{ls, cma_*, turbo_*}.json) and the stored files of the selected g v2,
M_bias v1 and C0_refit arms.  No simulation, no training.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "out/research/explore"
OUT = EXP / "I22_1"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

from cryoml.explore import i22_1_baselines as m  # noqa: E402

GRID_POINTS = {"sky130": 75 * 75, "nist": 73 * 73, "ihp": 75 * 75, "tud": 45 * 45}
ARMS_ALL = ("C0_refit", "M_bias_v1", "selected_g_v2", "B1", "B2", "B3_d3", "B4")


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def kind(ds: str) -> str:
    return "sky130" if ds == "sky130" else "nist" if ds.startswith("nist") else ds[:3]


def temp(ds: str) -> int:
    return int(ds[3:])


def rel(p: Path) -> str:
    return str(Path(p).relative_to(ROOT))


# ---------------------------------------------------------------------------
# stored calls
# ---------------------------------------------------------------------------
def starting_calls(ds: str, tags) -> dict:
    """Per-device NGSpice calls of the starting cards (stored)."""
    k = kind(ds)
    if k == "sky130":
        parts = {}
        for name, p in (("direct r1", "out/research/ngspice_direct/full_r1_direct_seeded/summary.json"),
                        ("direct r2", "out/research/ngspice_direct/full_r2_direct/summary.json"),
                        ("direct r3", "out/research/ngspice_direct/full_r3_direct/summary.json"),
                        ("FD polish", "out/research/polished/direct43_r3_fd/real/summary.json")):
            parts[name] = float(jload(ROOT / p)["mean_ngspice_calls_per_device"])
        return {"per_device_mean": float(sum(parts.values())), "parts": parts,
                "source": "out/research/ngspice_direct/full_r{1_seeded,2,3}_direct, polished/direct43_r3_fd"}
    if k == "nist":
        src = EXP / ("I8_1" if ds == "nist4" else "I9_1") / "start/summary.json"
        rows = jload(src)["rows"]
    elif k == "ihp":
        src = EXP / "I5_1/start/summary.json"
        rows = jload(src)["per_temperature"][str(temp(ds))]["rows"]
    else:
        src = EXP / "I6_1/start/summary.json"
        rows = jload(src)["per_temperature"][str(temp(ds))]["rows"]
    v = [rows[t]["n_sim"] for t in tags if t in rows and rows[t].get("n_sim") is not None]
    return {"per_device_mean": float(np.mean(v)) if v else float("nan"), "n": len(v), "source": rel(src)}


def nsim_of(rows: dict, tags, key=lambda t: t) -> list:
    out = []
    for t in tags:
        r = rows.get(key(t))
        if r is not None and r.get("n_sim") is not None:
            out.append(int(r["n_sim"]))
    return out


def v1_refit_calls(ds: str, tags) -> dict:
    k = kind(ds)
    if k == "sky130":
        src = EXP / "I13_1/all18/results.json"
        v = nsim_of(jload(src)["heldout_rows"]["M_bias"], tags)
    elif k == "nist":
        src = EXP / f"I20_1/base_{ds}/results.json"
        v = nsim_of(jload(src)["rows"], tags, lambda t: f"M_bias|{t}")
    elif k == "ihp":
        src = EXP / ("I17_2/base/results.json" if temp(ds) == 300 else f"I21_1/ihp_{temp(ds)}K/base/results.json")
        v = nsim_of(jload(src)["rows"], tags, lambda t: f"M_bias|{t}")
    else:
        src = EXP / f"I18_2/base_{temp(ds)}K/results.json"
        v = nsim_of(jload(src)["rows"], tags, lambda t: f"M_bias|{t}")
    return {"calls": v, "source": rel(src)}


def g2_refit_and_selection(ds: str, tags) -> dict:
    """Selected g v2: held-out refit calls per device, nested-selection refits
    (count; calls stored or estimated), member trainings and CPU seconds."""
    from cryoml.explore.i6_1_tudelft import group_of
    k = kind(ds)
    out = {"source": []}
    sel_calls, final_sec = [], []
    if k == "sky130":
        lam = jload(EXP / "I15_2/decision.json")["lambda_star"]
        for t in tags:
            L = lam[t]
            p = EXP / "I13_1/lodo/results.json" if L == 10.0 else EXP / f"I15_2/lodo_sym_e1_l{L:g}/results.json"
            d = jload(p)
            sel_calls += nsim_of(d["heldout_rows"], [t])
            key = f"M_bias_v2|{t}" if L == 10.0 else f"sym_e1_l{L:g}|{t}"
            final_sec += d["train_seconds_per_member"].get(key, [])
        inner = jload(EXP / "I15_2/inner/inner.json")
        out["source"] += ["I15_2/decision.json", "I15_2/lodo_sym_e1_l*/results.json", "I13_1/lodo/results.json",
                          "I15_2/inner/inner.json"]
    elif k == "nist":
        d = jload(EXP / f"I20_1/lodo_{ds}/results.json")
        sel_calls = nsim_of(d["heldout_rows"], tags)
        final_sec = [s for v in d["train_seconds_per_member"].values() for s in v]
        inner = jload(EXP / f"I20_1/select_{ds}/select.json")
        out["source"] += [f"I20_1/lodo_{ds}/results.json", f"I20_1/select_{ds}/select.json"]
    elif k == "ihp":
        if temp(ds) == 300:
            lam = jload(EXP / "I17_2/decision.json")["lambda_star"]
            for t in tags:
                d = jload(EXP / f"I17_2/lodo_sym_e1_l{lam[t]:g}/results.json")
                sel_calls += nsim_of(d["heldout_rows"], [t])
                final_sec += [s for kk, v in d["train_seconds_per_member"].items() if kk.endswith("|" + t) for s in v]
            inner = jload(EXP / "I17_2/inner/inner.json")
            out["source"] += ["I17_2/decision.json", "I17_2/lodo_sym_e1_l*/results.json", "I17_2/inner/inner.json"]
        else:
            d = jload(EXP / f"I21_1/ihp_{temp(ds)}K/sel/results.json")
            sel_calls = nsim_of(d["heldout_rows"], tags)
            final_sec = [s for v in d["train_seconds_per_member"].values() for s in v]
            inner = jload(EXP / f"I21_1/ihp_{temp(ds)}K/inner/inner.json")
            out["source"] += [f"I21_1/ihp_{temp(ds)}K/sel/results.json", f"I21_1/ihp_{temp(ds)}K/inner/inner.json"]
    else:
        T = temp(ds)
        lam = jload(EXP / "I18_2/decision.json")["per_temperature"][str(T)]["lambda_star"]
        seen = set()
        for t in tags:
            g = group_of(t)
            d = jload(EXP / f"I18_2/lodo_{T}K_sym_e1_l{lam[g]:g}/results.json")
            sel_calls += nsim_of(d["heldout_rows"], [t])
            if g not in seen:
                seen.add(g)
                final_sec += [s for kk, v in d["train_seconds_per_member"].items() if kk.endswith("|" + g) for s in v]
        inner = jload(EXP / f"I18_2/inner_{T}K/inner.json")
        out["source"] += ["I18_2/decision.json", f"I18_2/lodo_{T}K_sym_e1_l*/results.json", f"I18_2/inner_{T}K/inner.json"]
    # nested selection refits: per outer key, the inner held-out refits of the 5 lambdas
    n_eval, n_full = 0, 0
    for key, o in inner["outer"].items():
        n = sum(len(v) for v in o["inner_rrms"].values())
        if key == "full":
            n_full += n
        else:
            n_eval += n
    med = float(np.median(sel_calls)) if sel_calls else float("nan")
    tsec = inner.get("train_seconds", {})
    is_full = lambda kk: kk.startswith("full|") or "|full|" in kk  # noqa: E731
    full_sec = [v for kk, v in tsec.items() if is_full(kk)]
    eval_sec = [v for kk, v in tsec.items() if not is_full(kk)]
    if k == "nist":
        stored_total = int(inner["n_sim_refit"])
        n_all = n_eval + n_full
        calls_eval = stored_total * n_eval / n_all if n_all else float("nan")
        calls_full = stored_total * n_full / n_all if n_all else float("nan")
        calls_kind = "stored total n_sim (I20-1 select), split between blocks and full-data by refit count"
    else:
        calls_eval, calls_full = n_eval * med, n_full * med
        calls_kind = "estimate: inner refits x median held-out n_sim of the selected g v2 arm"
    if k == "tud":
        full = jload(EXP / f"I18_2/full_{temp(ds)}K/full.json")
        n_full = sum(len(v) for o in full["outer"].values() for v in o["inner_rrms"].values())
        calls_full = n_full * med
        full_sec = list(full.get("train_seconds", {}).values())
        out["source"].append(f"I18_2/full_{temp(ds)}K/full.json")
    out.update({"heldout_calls": sel_calls, "median_heldout_calls": med,
                "selection_refits_eval": n_eval, "selection_calls_eval": calls_eval,
                "selection_refits_full": n_full, "selection_calls_full": calls_full, "selection_calls_kind": calls_kind,
                "selection_trainings_eval": len(eval_sec), "selection_train_cpu_s_eval": float(sum(eval_sec)),
                "selection_trainings_full": len(full_sec), "selection_train_cpu_s_full": float(sum(full_sec)),
                "final_trainings": len(final_sec), "final_train_cpu_s": float(sum(final_sec)),
                "n_folds": len([k_ for k_ in inner["outer"] if k_ != "full"])})
    return out


def stage_cost() -> dict:
    out = {"unit": "one NGSpice call = one simulation of a device's measured bias set (polish n_sim)", "datasets": {}}
    for ds in m.DATASETS:
        p = OUT / f"partA_{ds}/results.json"
        if not p.exists():
            print(f"{ds}: partA missing, skipped", flush=True)
            continue
        A = jload(p)
        tags = A["held_out"]
        n = len(tags)
        scr = jload(OUT / f"screen_{ds}.json")
        g2 = g2_refit_and_selection(ds, tags)
        v1 = v1_refit_calls(ds, tags)
        start = starting_calls(ds, tags)
        nf = len(A["folds"])
        rows = {}
        for a in ("C0_refit", "B1", "B2", "B3_d3", "B4"):
            calls = [int(c) for c in A["n_sim"][a].values() if c is not None]
            rows[a] = {"refit_calls_mean": float(np.mean(calls)), "refit_calls_total": int(sum(calls)),
                       "pre_calls_total": 0, "trainings": 0, "train_cpu_s": 0.0,
                       "refit_cpu_s": float(A["refit_runtime_s_sum"][a])}
        rows["B1"]["pre_calls_total"] = int(scr["calls"]["total"])
        rows["B1"]["pre_note"] = ("edge screen, " + ("stored I5-1 screen, counted for 300 K only"
                                                     if kind(ds) == "ihp" else "this run"))
        if kind(ds) == "ihp" and ds != "ihp300":
            rows["B1"]["pre_calls_total"] = 0
        for a in ("B3_d3", "B4"):
            rows[a]["trainings"] = nf
            rows[a]["train_cpu_s"] = float(sum(A["train_seconds"][a].values()))
        rows["M_bias_v1"] = {"refit_calls_mean": float(np.mean(v1["calls"])) if v1["calls"] else float("nan"),
                             "refit_calls_total": int(sum(v1["calls"])), "pre_calls_total": 0,
                             "trainings": 5 * nf, "train_cpu_s": None, "source": v1["source"]}
        rows["selected_g_v2"] = {
            "refit_calls_mean": float(np.mean(g2["heldout_calls"])) if g2["heldout_calls"] else float("nan"),
            "refit_calls_total": int(sum(g2["heldout_calls"])),
            "pre_calls_total": float(g2["selection_calls_eval"]),
            "pre_note": g2["selection_calls_kind"],
            "selection_refits": g2["selection_refits_eval"],
            "penalty_grid_decks": n, "penalty_grid_points": GRID_POINTS[kind(ds)],
            "trainings": g2["selection_trainings_eval"] + g2["final_trainings"],
            "train_cpu_s": g2["selection_train_cpu_s_eval"] + g2["final_train_cpu_s"],
            "full_data_selection_refits": g2["selection_refits_full"],
            "full_data_selection_calls": g2["selection_calls_full"],
            "full_data_selection_trainings": g2["selection_trainings_full"],
            "source": g2["source"]}
        c0 = rows["C0_refit"]["refit_calls_total"]
        for a, r in rows.items():
            r["total_calls"] = float(r["pre_calls_total"]) + float(r["refit_calls_total"])
            r["total_calls_per_device"] = r["total_calls"] / n
            r["ratio_to_C0_refit_refit_calls"] = r["total_calls"] / c0 if c0 else float("nan")
        out["datasets"][ds] = {"name": m.NAMES[ds], "n": n, "n_folds": nf, "starting_cards": start, "arms": rows,
                               "s3_gate": {"ngspice_calls": 0, "trainings": nf,
                                           "note": "v1 recipe, seed 0, 500 steps, applied to starting records"},
                               "P7_selection_gt_heldout": bool(rows["selected_g_v2"]["pre_calls_total"] >
                                                               rows["selected_g_v2"]["refit_calls_total"])}
        print(f"{ds}: start {start['per_device_mean']:.0f}/dev; " + "; ".join(
            f"{a} {r['total_calls_per_device']:.0f}/dev" for a, r in rows.items()), flush=True)
    jdump(out, OUT / "cost.json")
    return out


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def is_pmos(t: str) -> bool:
    return "pmos" in t or t.startswith("pfet")


def part_a(ds: str) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    import i22_1_baselines as drv
    A = jload(OUT / f"partA_{ds}/results.json")
    st = drv.stored_rows(ds)
    tags = A["held_out"]
    cl = drv.clusters(ds)
    rows = {"C0_refit": {t: st["C0_refit"][t] for t in tags},
            "M_bias_v1": {t: st["M_bias_v1"][t] for t in tags},
            "selected_g_v2": {t: st["selected_g_v2"][t] for t in tags}}
    for x in m.BASELINES:
        rows[x] = {t: A["per_device"][x][t] for t in tags}
    rows["C0_refit_rerun"] = {t: A["per_device"]["C0_refit"][t] for t in tags}
    summ = {a: m.family_means(v, is_pmos) for a, v in rows.items()}
    cmp_c0, cmp_g, cmp_v1, lab_c0, lab_g, ni = {}, {}, {}, {}, {}, {}
    for x in m.BASELINES + ("selected_g_v2", "M_bias_v1"):
        cmp_c0[x] = paired_ratio(rows[x], rows["C0_refit"], cl)
        lab_c0[x] = m.label_vs_control(cmp_c0[x])
    for x in m.BASELINES:
        cmp_g[x] = paired_ratio(rows[x], rows["selected_g_v2"], cl)
        lab_g[x] = m.label_vs_reference(cmp_g[x], x)
        ni[x] = m.non_inferior(cmp_g[x])
        cmp_v1[x] = paired_ratio(rows[x], rows["M_bias_v1"], cl)
    return {"ds": ds, "name": m.NAMES[ds], "n": len(tags), "unit": "cluster (7 groups)" if cl else "device",
            "validity": A["validity"], "rows": rows, "summary": summ, "vs_C0_refit": cmp_c0, "label_vs_C0_refit": lab_c0,
            "vs_g_v2": cmp_g, "label_vs_g_v2": lab_g, "non_inferior_to_g_v2": ni, "vs_M_bias_v1": cmp_v1,
            "b1_selected": A["b1"]["selected"], "n_failed": {a: s["n_failed"] for a, s in summ.items()},
            "source": rel(OUT / f"partA_{ds}/results.json")}


def part_c() -> dict | None:
    from cryoml.explore.i3_2_bias import paired_ratio
    C = OUT / "partC"
    if not (C / "ls.json").exists():
        return None
    ls = jload(C / "ls.json")
    tags = ls["tags"]
    f0 = {t: ls["primary"][t]["trace"][0] for t in tags}
    lsv = {t: ls["primary"][t]["best_traced"] for t in tags}
    out = {"tags": tags, "budget": ls["budget"], "validity": {"VC1": ls["VC1"], "VC1_max_abs_diff": ls["VC1_max_abs_diff"],
                                                               "VC3_unit_tests": ls["unit_tests_passed"],
                                                               "resim_vs_traced_max": ls["resim_vs_traced_max"]},
           "LS": {"value": lsv, "f_start": f0, "vs_start": paired_ratio(lsv, f0),
                  "secondary_value_4B": {t: m.value_at(ls["secondary"][t]["trace"], 4 * ls["budget"][t]) for t in tags},
                  "secondary_n_sim": {t: ls["secondary"][t]["n_sim"] for t in tags}},
           "methods": {}}
    for meth in ("cma", "turbo"):
        for cfg in ("primary", "secondary"):
            p = C / f"{meth}_{cfg}.json"
            if not p.exists():
                continue
            R = jload(p)
            by = {}
            for r in R["rows"]:
                by.setdefault(r["tag"], []).append(r)
            val = {t: m.seed_geomean([r["value_at_budget"] for r in by[t]]) for t in tags}
            v4 = {t: m.seed_geomean([r["value_at_4budget"] for r in by[t]]) for t in tags} \
                if all(r.get("value_at_4budget") is not None for r in R["rows"]) else None
            cmpls = paired_ratio(val, lsv)
            d = {"value": val, "vs_LS": cmpls, "label": ("LS better" if np.isfinite(cmpls["ci_low"]) and cmpls["ci_low"] > 1
                                                         else "method better" if cmpls["ci_high"] < 1 else "inconclusive"),
                 "vs_start": paired_ratio(val, f0),
                 "per_seed_vs_LS": {s: paired_ratio({r["tag"]: r["value_at_budget"] for r in R["rows"] if r["seed"] == s},
                                                    lsv)["ratio"] for s in sorted({r["seed"] for r in R["rows"]})},
                 "VC2_budget_exact": R["VC2_budget_exact"], "VC2_resim_max": R["VC2_resim_max"],
                 "wall_s": R["wall_s"], "workers": R["workers"],
                 "run_seconds_mean": float(np.mean([r["seconds"] for r in R["rows"]])),
                 "budget_factor": R.get("budget_factor")}
            if v4 is not None:
                d["value_4B"] = v4
                d["vs_LS_4B_secondary"] = paired_ratio(v4, out["LS"]["secondary_value_4B"])
            out["methods"][f"{meth}_{cfg}"] = d
    out["validity"]["VC2"] = bool(all(d["VC2_budget_exact"] and d["VC2_resim_max"] <= 1e-12
                                      for d in out["methods"].values()))
    out["validity"]["valid"] = bool(out["validity"]["VC1"] and out["validity"]["VC2"] and out["validity"]["VC3_unit_tests"])
    return out


def stage_decide() -> dict:
    A = {ds: part_a(ds) for ds in m.DATASETS if (OUT / f"partA_{ds}/results.json").exists()}
    valid = {ds: bool(a["validity"]["valid"]) for ds, a in A.items()}
    vs_g = {ds: a["vs_g_v2"] for ds, a in A.items()}
    oc = m.outcome(vs_g, valid)
    pooled = {}
    for arm in m.BASELINES + ("selected_g_v2", "M_bias_v1"):
        pooled[arm] = m.pooled_gmr([A[ds]["vs_C0_refit"][arm]["ratio"] for ds in A if valid[ds]])
    outside = {ds: {"gain_over_C0_refit": [x for x in m.BASELINES if A[ds]["label_vs_C0_refit"][x] == "gain"],
                    "labels_vs_g_v2": A[ds]["label_vs_g_v2"]} for ds in A if ds not in m.GAIN_SET}
    C = part_c()
    cost = jload(OUT / "cost.json") if (OUT / "cost.json").exists() else None
    # predictions
    pred = {}
    if "sky130" in A:
        pred["P1"] = all(A["sky130"]["label_vs_g_v2"][x] == f"g v2 better than {x}" for x in m.BASELINES)
    if "nist4" in A and "nistrt" in A:
        pred["P2"] = all(all(A[d]["label_vs_g_v2"][x] == f"g v2 better than {x}" for x in ("B1", "B2", "B4"))
                         and A[d]["non_inferior_to_g_v2"]["B3_d3"] for d in ("nist4", "nistrt"))
    ihp = [d for d in ("ihp233", "ihp300", "ihp343", "ihp398") if d in A]
    if ihp:
        pred["P3"] = {d: bool(A[d]["label_vs_C0_refit"]["B1"] == "gain" and
                              A[d]["label_vs_g_v2"]["B1"] == "B1 better than g v2") for d in ihp}
    tud = [d for d in ("tud4", "tud295") if d in A]
    if tud:
        pred["P4"] = {d: bool(A[d]["vs_g_v2"]["B1"]["ratio"] > 1 and
                              all(v == "inconclusive" for v in A[d]["label_vs_g_v2"].values())) for d in tud}
    pred["P5"] = oc["label"] == "selected g v2 not matched by a correction-stage baseline"
    if C is not None:
        pred["P6"] = all(C["methods"].get(f"{x}_primary", {}).get("label") == "LS better" for x in ("cma", "turbo"))
    if cost is not None:
        pred["P7"] = {ds: v["P7_selection_gt_heldout"] for ds, v in cost["datasets"].items()}
    out = {"prereg": "out/research/explore/I22_1/prereg.md", "amendments": ["prereg_amendment_A1.md"],
           "part_A": {"validity": valid, "datasets": {ds: {k: v for k, v in a.items() if k != "rows"} for ds, a in A.items()},
                      "rows": {ds: a["rows"] for ds, a in A.items()},
                      "outcome": oc, "pooled_gmr_vs_C0_refit": pooled, "outside_gain_set": outside},
           "part_B": cost, "part_C": C, "predictions": pred}
    jdump(out, OUT / "decision.json")
    print(json.dumps({"validity": valid, "outcome": oc["label"], "pooled": pooled, "predictions": pred}, indent=1,
                     default=float), flush=True)
    for ds, a in A.items():
        print(f"{ds}: " + "; ".join(f"{x} vs g2 {a['vs_g_v2'][x]['ratio']:.3f} [{a['vs_g_v2'][x]['ci_low']:.3f}, "
                                     f"{a['vs_g_v2'][x]['ci_high']:.3f}] vs C0 {a['vs_C0_refit'][x]['ratio']:.3f}"
                                     for x in m.BASELINES), flush=True)
    if C:
        for k_, d in C["methods"].items():
            print(f"Part C {k_}: GMR vs LS {d['vs_LS']['ratio']:.3f} [{d['vs_LS']['ci_low']:.3f}, "
                  f"{d['vs_LS']['ci_high']:.3f}] {d['label']}", flush=True)
    return out
