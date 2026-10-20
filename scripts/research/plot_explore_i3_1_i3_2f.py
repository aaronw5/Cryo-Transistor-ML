#!/usr/bin/env python3
"""Report batch for exploration direction I3-1 (hybrid temperature law,
simulated tiers M, W, H) and the I3-2 follow-up analyses F1 (SPICE-native
Verilog-A validation) and F2 (robustness of M_bias).

Reads stored files only (no NGSpice, no training):

* I3-1: ``out/research/explore/I3_1/{prereg.json,prereg_addendum.json}``,
  ``runs/full/tier_{M,W,H}/{summary.json,eval/*.json,u0_switch_check.json}``,
  ``runs/full/decision.json``.
* I3-2 follow-up: ``out/research/explore/I3_2/followup_prereg.json`` and the
  amendment, ``followup/f1/{summary.json,curves.npz}``,
  ``followup/f2_real/results.json``, ``followup/f2_{gf180,ihp}_b/results.json``,
  ``followup/f2_summary.json``, the REAL_DEV control rows
  (``out/research/explore/I2_1/real_dev/results.json``), the I3-2 M_bias rows
  (``out/research/explore/I3_2/real_dev/results.json``) and the tier b floors
  (``out/research/pdk2/floors/2026-10-16_final/floors_{gf180,ihp}.json``).

Every stored GMR / paired ratio used in a table is recomputed from the stored
per-device values and checked against the stored value.

Writes PNGs, ``tables.md`` and ``README.md`` to
``out/research/report/<date>_i3_1_i3_2f/`` and adds the batch to
``out/research/report/index.json`` and ``INDEX.md``.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i3_1_i3_2f.py --date 2026-10-20
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA,  # noqa: E402
                                   VIOLET, RED, DPI, EXP, ROOT, fam, load, rel, style, yn)
from cryoml.explore.i3_hybrid import gmr as i31_gmr  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i3_1_i3_2f"
I31 = EXP / "I3_1"
I31_RUN = I31 / "runs/full"
I31_PREREG = I31 / "prereg.json"
I31_ADD = I31 / "prereg_addendum.json"
I31_DEC = I31_RUN / "decision.json"
TIERS = ("M", "W", "H")
TIER_LABEL = {"M": "tier M (logistic knee + bump + R(T); truth outside par3)",
              "W": "tier W (x = mu w^g; truth inside par3)",
              "H": "tier H (kink + freeze-out step; new in I3-1)"}

I32 = EXP / "I3_2"
FU = I32 / "followup"
FU_PREREG = I32 / "followup_prereg.json"
FU_AMEND = I32 / "followup_prereg_amendment_2026-10-20.json"
F1 = FU / "f1"
F2_REAL = FU / "f2_real/results.json"
F2_PDK = {"gf180": FU / "f2_gf180_b/results.json", "ihp": FU / "f2_ihp_b/results.json"}
F2_SUM = FU / "f2_summary.json"
I21_REAL = EXP / "I2_1/real_dev/results.json"
I32_REAL = I32 / "real_dev/results.json"
FLOORS = {"gf180": ROOT / "out/research/pdk2/floors/2026-10-16_final/floors_gf180.json",
          "ihp": ROOT / "out/research/pdk2/floors/2026-10-16_final/floors_ihp.json"}
PDK_LABEL = {"gf180": "GF180 (BSIM4) tier b", "ihp": "IHP SG13G2 (PSP103) tier b"}
R2_THRESHOLD = 0.08698

# comparisons reported per tier (a vs b), in display order
CMPS = [("hyb", "par3"), ("hyb", "par14"), ("hyb", "par3_pop"), ("hyb", "pt_interp"),
        ("hyb", "lrn"), ("hyb", "lrn_mean"), ("hyb", "hyb_mean"),
        ("lrn", "par3"), ("lrn", "lrn_mean"), ("lrn", "pt_interp"), ("hyb_mean", "par3")]
ARMS = ["par3", "par14", "par3_pop", "pt_interp", "lrn", "lrn_mean", "hyb", "hyb_mean"]
ARM_COLOR = {"par3": INK, "pt_interp": AQUA, "lrn": ORANGE, "lrn_mean": GREY, "hyb": BLUE,
             "hyb_mean": VIOLET, "par14": MUTED, "par3_pop": MUTED}
ARM_DESC = {"par3": "per-device parametric law c + (mu - c) w^g fitted on the O4 anchors",
            "par14": "par3 with 14 free exponents",
            "par3_pop": "par3 with population-pooled exponents",
            "pt_interp": "per-parameter interpolation between the anchor fits",
            "lrn": "E3b learned population-shared law (network)",
            "lrn_mean": "E3b population-mean shape (no network)",
            "hyb": "I3-1 primary: par3 + learned population residual",
            "hyb_mean": "par3 + population-mean residual (no network)"}
HELD = ("110", "190", "260")
ANCH = ("77", "150", "225", "300")


def f3(x, n=3) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def g_text(g: dict) -> str:
    return f"{g['gmr']:.3f} [{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}]"


def r_text(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def sci(x: float) -> str:
    return f"{x:.2g}" if x == 0 or 1e-3 <= abs(x) < 1e3 else f"{x:.1e}"


# ================================================================== paired ratio (I3-2)

def paired_ratio(method: dict, control: dict, clusters: dict | None = None,
                 n_boot: int = 10000, seed: int = 0) -> dict:
    """Copy of cryoml.explore.i3_2_bias.paired_ratio (commit 7f0469a), kept
    here so that this read-only report does not import a module that is being
    edited."""
    tags = sorted(set(method) & set(control))
    a = np.array([method[t] for t in tags], dtype=float)
    b = np.array([control[t] for t in tags], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    lr = np.log(a[ok] / b[ok])
    used = [t for t, k in zip(tags, ok) if k]
    out = {"n": len(tags), "n_used": int(ok.sum()), "wins": int(np.sum(a[ok] < b[ok]))}
    out["ratio"] = float(np.exp(lr.mean()))
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(lr), size=(n_boot, len(lr)))
        boots = lr[idx].mean(axis=1)
    else:
        names = sorted({clusters[t] for t in used})
        members = [np.array([i for i, t in enumerate(used) if clusters[t] == c]) for c in names]
        sums = np.array([lr[m].sum() for m in members])
        sizes = np.array([len(m) for m in members], dtype=float)
        pick = rng.integers(0, len(names), size=(n_boot, len(names)))
        boots = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["ci_low"], out["ci_high"] = float(np.exp(lo)), float(np.exp(hi))
    return out


def check_ratio(stored: dict, mine: dict, what: str) -> None:
    for k in ("ratio", "ci_low", "ci_high"):
        assert abs(stored[k] - mine[k]) < 1e-12, (what, k, stored[k], mine[k])
    assert stored["wins"] == mine["wins"], what


# ================================================================== I3-1 loading

def load_i31() -> dict:
    dec = load(I31_DEC)
    tiers = {}
    for t in TIERS:
        d = I31_RUN / f"tier_{t}"
        s = load(d / "summary.json")
        ev = {p.stem: load(p) for p in sorted((d / "eval").glob("*.json"))}
        assert len(ev) == 36, (t, len(ev))
        per = s["i3_1"]["per_device"]
        # per-device held-out RRMS from eval/*.json (mean over 110/190/260 K, NaN if any NaN)
        held = {}
        for key in s["table"]:
            if key.startswith("O4|") or key == "floor|floor":
                held[key] = {}
                for tag, e in ev.items():
                    v = [e[key][T] for T in HELD] if key in e else [np.nan]
                    v = [np.nan if x is None else x for x in v]
                    held[key][tag] = float(np.mean(v)) if all(np.isfinite(v)) else np.nan
        for arm in ARMS:
            a = held[f"O4|{arm}"]
            for tag, x in per[arm].items():
                x = np.nan if x is None else x
                assert (np.isnan(x) and np.isnan(a[tag])) or abs(x - a[tag]) < 1e-12, (t, arm, tag)
            m = np.nanmean(list(a.values()))
            assert abs(m - s["table"][f"O4|{arm}"]["heldout_all_device_mean"]) < 1e-12, (t, arm)
        tiers[t] = {"summary": s, "held": held, "eval": ev, "dir": d}
    # recompute every stored GMR and the decision
    for t in TIERS:
        s, held = tiers[t]["summary"], tiers[t]["held"]
        tags = sorted(tiers[t]["eval"])
        for a, b in CMPS:
            for eps, block in ((0.01, "gmr"), (0.001, "gmr_eps_0.001")):
                st = dec["per_tier"][t][block].get(f"{a}_vs_{b}")
                if st is None:
                    continue
                mine = i31_gmr([held[f"O4|{a}"][x] for x in tags], [held[f"O4|{b}"][x] for x in tags],
                               eps=eps)
                assert abs(mine["gmr"] - st["gmr"]) < 1e-12 and mine["wins"] == st["wins"], (t, a, b)
                assert all(abs(x - y) < 1e-12 for x, y in zip(mine["ci95"], st["ci95"])), (t, a, b)
                assert mine["n_devices"] == st["n_devices"]
        for k, sc in dec["per_tier"][t]["scaling"].items():
            mine = i31_gmr([held[sc["key"]][x] for x in tags], [held["O4|par3"][x] for x in tags])
            assert abs(mine["gmr"] - sc["gmr_vs_par3"]["gmr"]) < 1e-12, (t, k)
            assert mine["wins"] == sc["gmr_vs_par3"]["wins"]
            assert abs(np.nanmean(list(held[sc["key"]].values())) - sc["heldout_all_device_mean"]) < 1e-12
    return {"dec": dec, "tiers": tiers}


def i31_conditions(d: dict) -> list[tuple]:
    dec = d["dec"]
    g = {t: dec["per_tier"][t]["gmr"] for t in TIERS}
    ff = {t: dec["per_tier"][t]["failed_fraction"] for t in TIERS}
    rows = [
        ("M1", "tier M: GMR(hyb, par3) <= 0.80 and CI upper bound < 1", g_text(g["M"]["hyb_vs_par3"]),
         g["M"]["hyb_vs_par3"]["gmr"] <= 0.80 and g["M"]["hyb_vs_par3"]["ci95"][1] < 1),
        ("M2", "tier M: CI upper bound of GMR(hyb, lrn_mean) < 1", g_text(g["M"]["hyb_vs_lrn_mean"]),
         g["M"]["hyb_vs_lrn_mean"]["ci95"][1] < 1),
        ("M3", "tier M: GMR(hyb, pt_interp) < 1", g_text(g["M"]["hyb_vs_pt_interp"]),
         g["M"]["hyb_vs_pt_interp"]["gmr"] < 1),
        ("W1", "tier W: CI upper bound of GMR(hyb, par3) <= 1.25", g_text(g["W"]["hyb_vs_par3"]),
         g["W"]["hyb_vs_par3"]["ci95"][1] <= 1.25),
        ("H1", "tier H: CI upper bound of GMR(hyb, par3) < 1", g_text(g["H"]["hyb_vs_par3"]),
         g["H"]["hyb_vs_par3"]["ci95"][1] < 1),
        ("H2", "tier H: GMR(hyb, lrn_mean) < 1 and GMR(hyb, pt_interp) < 1",
         f"{g['H']['hyb_vs_lrn_mean']['gmr']:.3f}; {g['H']['hyb_vs_pt_interp']['gmr']:.3f}",
         g["H"]["hyb_vs_lrn_mean"]["gmr"] < 1 and g["H"]["hyb_vs_pt_interp"]["gmr"] < 1),
        ("F", "failed-cell fraction <= 10% for hyb, par3, lrn_mean, pt_interp (each tier)",
         "max " + ", ".join(f"{t} {max(ff[t][m] for m in ('hyb', 'par3', 'lrn_mean', 'pt_interp')):.3f}"
                            for t in TIERS),
         all(max(ff[t][m] for m in ("hyb", "par3", "lrn_mean", "pt_interp")) <= 0.10 for t in TIERS)),
    ]
    for key, _, _, ok in rows:
        assert ok == dec["continue_conditions"][key], key
    kill = [("K_M", "KILL condition: tier M GMR(hyb, par3) >= 0.95", f"{g['M']['hyb_vs_par3']['gmr']:.3f}",
             g["M"]["hyb_vs_par3"]["gmr"] >= 0.95, "M_gmr_ge_0.95"),
            ("K_W", "KILL condition: tier W GMR(hyb, par3) >= 1.50", f"{g['W']['hyb_vs_par3']['gmr']:.3f}",
             g["W"]["hyb_vs_par3"]["gmr"] >= 1.50, "W_gmr_ge_1.50"),
            ("K_H", "KILL condition: tier H GMR(hyb, par3) >= 1.00", f"{g['H']['hyb_vs_par3']['gmr']:.3f}",
             g["H"]["hyb_vs_par3"]["gmr"] >= 1.00, "H_gmr_ge_1.00")]
    for _, _, _, ok, key in kill:
        assert ok == dec["kill_conditions"][key], key
    cont = all(r[3] for r in rows)
    verdict = "CONTINUE" if cont else ("KILL" if any(k[3] for k in kill) else "INCONCLUSIVE")
    assert all(dec["valid"].values())
    assert verdict == dec["verdict_rule_output"], verdict
    return rows, kill, verdict


# ================================================================== I3-1 tables

def table_i31_decision(d: dict) -> rf.Table:
    rows_c, kill, verdict = i31_conditions(d)
    rows = [{"k": k, "c": c, "v": v, "ok": yn(ok)} for k, c, v, ok in rows_c]
    rows += [{"k": "", "c": c, "v": v, "ok": "hit" if ok else "not hit"} for _, c, v, ok, _ in kill]
    rows.append({"k": "", "c": "u0 unit-switch validity check (INVALID if failed)",
                 "v": "valid on M, W, H", "ok": "yes"})
    rows.append({"k": "", "c": "rule output (decision.json, I3_1_decide)", "v": "", "ok": verdict})
    cols = [{"key": "k", "head": "id", "md": "id", "kind": "text"},
            {"key": "c", "head": "pre-registered condition", "md": "condition", "kind": "text"},
            {"key": "v", "head": "value: GMR [95% CI]", "md": "value", "kind": "rtext"},
            {"key": "ok", "head": "met", "md": "met", "kind": "rtext"}]
    notes = [
        "Pre-registration out/research/explore/I3_1/prereg.json (commit c877413) and prereg_addendum.json "
        "(hyb inputs geo_t / geo_t_res and output penalty 0.01; written before any NGSpice run of the new "
        "arms). CONTINUE = M1-M3, W1, H1, H2 and F all met. KILL = not CONTINUE and any KILL condition hit. "
        "INCONCLUSIVE otherwise.",
        "GMR(a, b) = exp(mean_d log((a_d + 0.01) / (b_d + 0.01))) over the 36 test devices per tier "
        "(per-device held-out RRMS = mean over 110/190/260 K, setting O4); percentile bootstrap 95% CI, "
        "4000 resamples, numpy default_rng(0). Values recomputed from eval/*.json and checked against "
        "decision.json (|difference| < 1e-12).",
        "Simulated devices only (12 REAL_DEV geometries x 3 test draws per tier; 72 training devices per "
        "tier). No measured I-V enters the decision.",
    ]
    return rf.Table("01_table_i3_1_decision",
                    "I3-1 decision: pre-registered conditions, values and rule output", cols, rows, notes)


def table_i31_gmr(d: dict) -> rf.Table:
    dec = d["dec"]
    rows = []
    for a, b in CMPS:
        r = {"cmp": f"{a} vs {b}"}
        for t in TIERS:
            g = dec["per_tier"][t]["gmr"][f"{a}_vs_{b}"]
            r[f"g{t}"] = g_text(g)
            r[f"w{t}"] = f"{g['wins']}/{g['n_devices']}"
            e = dec["per_tier"][t]["gmr_eps_0.001"].get(f"{a}_vs_{b}")
            r[f"e{t}"] = f"{e['gmr']:.3f}" if e else "—"
        rows.append(r)
    cols = [{"key": "cmp", "head": "comparison (a vs b)", "md": "comparison", "kind": "text"}]
    for t in TIERS:
        cols += [{"key": f"g{t}", "head": "GMR [95% CI]", "md": f"tier {t} GMR [95% CI]",
                  "kind": "rtext", "group": f"tier {t}", "family": t == "W"},
                 {"key": f"w{t}", "head": "a < b", "md": f"tier {t} a < b", "kind": "rtext",
                  "group": f"tier {t}", "family": t == "W"},
                 {"key": f"e{t}", "head": "eps 0.001", "md": f"tier {t} GMR eps 0.001", "kind": "rtext",
                  "group": f"tier {t}", "family": t == "W"}]
    notes = [
        "GMR with eps = 0.01 (decision statistic) and eps = 0.001 (sensitivity, reported, not deciding; "
        "stored for the hyb comparisons only). a < b = devices with lower unshifted held-out RRMS for a. "
        "GMR < 1 = a lower than b.",
        "Arms (O4 setting, anchors 77/150/225/300 K): " + "; ".join(f"{k}: {v}" for k, v in ARM_DESC.items())
        + ".",
        "hyb_mean has 1 failed cell on tier M (35 devices in its comparisons); lrn has 1 failed cell on tier "
        "W (35 devices in its comparisons).",
    ]
    return rf.Table("05_table_i3_1_gmr", "I3-1 paired GMR of every reported comparison, per tier (O4)",
                    cols, rows, notes)


def table_i31_per_temp(d: dict) -> rf.Table:
    rows = []
    for t in TIERS:
        s = d["tiers"][t]["summary"]
        held = d["tiers"][t]["held"]
        for arm in ARMS + ["floor"]:
            key = "floor|floor" if arm == "floor" else f"O4|{arm}"
            e = s["table"][key]
            vals = [x for x in held[key].values() if np.isfinite(x)]
            r = {"tier": t, "arm": arm, "all": e["heldout_all_device_mean"], "med": float(np.median(vals)),
                 "fail": f"{e['n_failed_cells']}/{e['n_cells']}",
                 "anch": e.get("anchor_all_device_mean")}
            for T in HELD:
                r[T] = e["per_temp"].get(T)
            for T in ANCH:
                r[f"a{T}"] = e["per_temp"].get(T)
            rows.append(r)
    cols = [{"key": "tier", "head": "tier", "md": "tier", "kind": "text"},
            {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
            {"key": "all", "head": "mean", "md": "held-out all-device mean", "kind": "rrms",
             "group": "held-out 110/190/260 K"},
            {"key": "med", "head": "median", "md": "held-out median", "kind": "rrms",
             "group": "held-out 110/190/260 K"},
            {"key": "110", "head": "110 K", "md": "110 K", "kind": "rrms", "group": "held-out 110/190/260 K",
             "family": True},
            {"key": "190", "head": "190 K", "md": "190 K", "kind": "rrms", "group": "held-out 110/190/260 K",
             "family": True},
            {"key": "260", "head": "260 K", "md": "260 K", "kind": "rrms", "group": "held-out 110/190/260 K",
             "family": True},
            {"key": "fail", "head": "failed cells", "md": "failed cells", "kind": "rtext",
             "group": "held-out 110/190/260 K"},
            {"key": "a77", "head": "77 K", "md": "anchor 77 K", "kind": "rrms", "group": "anchor temperatures"},
            {"key": "a150", "head": "150 K", "md": "anchor 150 K", "kind": "rrms",
             "group": "anchor temperatures"},
            {"key": "a225", "head": "225 K", "md": "anchor 225 K", "kind": "rrms",
             "group": "anchor temperatures"},
            {"key": "a300", "head": "300 K", "md": "anchor 300 K", "kind": "rrms",
             "group": "anchor temperatures"},
            {"key": "anch", "head": "anchor mean", "md": "anchor mean", "kind": "rrms",
             "group": "anchor temperatures"}]
    notes = [
        "All-device mean of the strict frozen-inclusion paper RRMS of the NGSpice re-simulated card against "
        "the simulated truth, 36 test devices per tier, setting O4. Held-out mean = mean over devices of the "
        "per-device mean over 110/190/260 K (failed cells excluded). floor = per-temperature fit on the "
        "held-out data (reference, not a method).",
        "hyb applies the learned correction at every temperature, including the anchors (no gating, as "
        "pre-registered); its anchor RRMS is the error of the corrected card at the temperatures where par3 "
        "was fitted. lrn and lrn_mean return the anchor fits at the anchors.",
    ]
    return rf.Table("06_table_i3_1_per_temperature",
                    "I3-1 held-out RRMS per arm and temperature, per tier (O4)", cols, rows, notes)


def table_i31_lrn(d: dict) -> rf.Table:
    dec = d["dec"]
    rows = []
    for t in TIERS:
        s = d["tiers"][t]["summary"]
        g = dec["per_tier"][t]["gmr"]
        o4, o2 = s["decisions"]["O4"], s["decisions"]["O2"]
        red = o4["lrn_vs_control"]
        red2 = o2["lrn_vs_control"]
        rows.append({
            "tier": t,
            "lrn": s["table"]["O4|lrn"]["heldout_all_device_mean"],
            "par3": s["table"]["O4|par3"]["heldout_all_device_mean"],
            "pt": s["table"]["O4|pt_interp"]["heldout_all_device_mean"],
            "lm": s["table"]["O4|lrn_mean"]["heldout_all_device_mean"],
            "gp": f"{g_text(g['lrn_vs_par3'])} {g['lrn_vs_par3']['wins']}/{g['lrn_vs_par3']['n_devices']}",
            "gm": f"{g_text(g['lrn_vs_lrn_mean'])} {g['lrn_vs_lrn_mean']['wins']}/"
                  f"{g['lrn_vs_lrn_mean']['n_devices']}",
            "gi": f"{g_text(g['lrn_vs_pt_interp'])} {g['lrn_vs_pt_interp']['wins']}/"
                  f"{g['lrn_vs_pt_interp']['n_devices']}",
            "r4": f"{red['reduction']:.3f} [{red['ci95'][0]:.3f}, {red['ci95'][1]:.3f}]",
            "rpt": f"{o4['lrn_vs_pt_interp']['reduction']:.3f}",
            "r2": f"{red2['reduction']:.3f} [{red2['ci95'][0]:.3f}, {red2['ci95'][1]:.3f}]",
            "e3": s["e3_rule_output_reported"],
        })
    cols = [{"key": "tier", "head": "tier", "md": "tier", "kind": "text"},
            {"key": "lrn", "head": "lrn", "md": "lrn mean", "kind": "rrms", "group": "held-out mean (O4)"},
            {"key": "par3", "head": "par3", "md": "par3 mean", "kind": "rrms", "group": "held-out mean (O4)"},
            {"key": "pt", "head": "pt_interp", "md": "pt_interp mean", "kind": "rrms",
             "group": "held-out mean (O4)"},
            {"key": "lm", "head": "lrn_mean", "md": "lrn_mean mean", "kind": "rrms",
             "group": "held-out mean (O4)"},
            {"key": "gp", "head": "lrn vs par3", "md": "GMR lrn vs par3, wins", "kind": "rtext",
             "group": "GMR [95% CI], wins (O4)", "family": True},
            {"key": "gm", "head": "lrn vs lrn_mean", "md": "GMR lrn vs lrn_mean, wins", "kind": "rtext",
             "group": "GMR [95% CI], wins (O4)", "family": True},
            {"key": "gi", "head": "lrn vs pt_interp", "md": "GMR lrn vs pt_interp, wins", "kind": "rtext",
             "group": "GMR [95% CI], wins (O4)", "family": True},
            {"key": "r4", "head": "R_O4 [CI]", "md": "R_O4 [CI]", "kind": "rtext", "group": "E3 rule"},
            {"key": "rpt", "head": "vs pt_interp", "md": "O4 lrn vs pt_interp reduction", "kind": "rtext",
             "group": "E3 rule"},
            {"key": "r2", "head": "R_O2 [CI]", "md": "R_O2 [CI]", "kind": "rtext", "group": "E3 rule"},
            {"key": "e3", "head": "E3 rule output", "md": "E3 rule output", "kind": "rtext",
             "group": "E3 rule"}]
    notes = [
        "lrn is the E3b method (learned population-shared temperature law, unchanged code). The tier M and W "
        "lrn rows reproduce the E3b full runs (per-device max |difference| 0 on 252 cells each); tier H is "
        "new in I3-1 and was not simulated or scored before the I3-1 decision run.",
        "E3 rule (copied into E3b): CONTINUE if for at least one S in {O4, O2}: R_S >= 0.25, CI lower bound "
        "of R_S > 0, lrn >= 10% below pt_interp, failed cells <= 10%; KILL if R_S < 0.10 for both S or lrn "
        "worse than pt_interp for both S; INCONCLUSIVE otherwise. R_S = 1 - mean(lrn) / mean(control_S), "
        "control_O4 = par3, control_O2 = par3_pop. The E3 rule output is reported per tier by the run "
        "(summary.json e3_rule_output_reported); only tier M decided E3b.",
        "Tier H, lrn: GMR vs par3 below 1 with CI upper bound below 1; R_O4 0.145 is below the 0.25 "
        "threshold; lrn vs pt_interp GMR 1.083 with CI above 1 (lrn higher than pt_interp on 21/36 devices) "
        "and O4 reduction 0.049 below the 0.10 threshold; R_O2 is negative.",
    ]
    return rf.Table("09_table_i3_1_lrn_by_tier",
                    "I3-1: the E3b method lrn on tiers M, W and H", cols, rows, notes)


def table_i31_checks(d: dict) -> rf.Table:
    dec = d["dec"]
    rows = []
    for t in TIERS:
        s = d["tiers"][t]["summary"]
        u = s["u0_switch_check"]
        clipped = {k: v for k, v in u["n_evaluated_cards_u0_clipped"].items() if v}
        i = s["i3_1"]
        rows.append({
            "tier": t,
            "truth": f"{u['truths']['n_at_or_above']}/{u['truths']['n_checked']} "
                     f"({u['truths']['n_rejected_draws']} rejected, {u['truths']['n_regenerated']} regen.)",
            "fit": f"{u['fitted_cards']['n_at_or_above']}/{u['fitted_cards']['n_checked']}",
            "pred": f"{u['predicted_cards_before_clip']['n_at_or_above']}/"
                    f"{u['predicted_cards_before_clip']['n_checked']}",
            "eval": f"{u['evaluated_cards']['n_at_or_above']}/{u['evaluated_cards']['n_checked']}",
            "clip": ", ".join(f"{k.split('|')[1]} {v}" for k, v in clipped.items()) or "0",
            "steps": ", ".join(str(x) for x in i["hyb_best_steps"]),
            "feat": ", ".join(i["hyb_feature_sets"]),
            "zero": str(i["hyb_members_zero_state"]),
            "repro": ("max " + sci(max(v["max_abs_diff"] for v in dec["reproduction_vs_E3b"][t].values()))
                      if t in dec["reproduction_vs_E3b"] else "new tier"),
            "wall": f"{s['wall_s']:.0f}",
        })
    cols = [{"key": "tier", "head": "tier", "md": "tier", "kind": "text"},
            {"key": "truth", "head": "truths", "md": "truths at/above switch", "kind": "rtext",
             "group": "u0 switch: at or above / checked"},
            {"key": "fit", "head": "fitted", "md": "fitted cards", "kind": "rtext",
             "group": "u0 switch: at or above / checked"},
            {"key": "pred", "head": "predicted (pre-clip)", "md": "predicted before clip", "kind": "rtext",
             "group": "u0 switch: at or above / checked"},
            {"key": "eval", "head": "evaluated", "md": "evaluated cards", "kind": "rtext",
             "group": "u0 switch: at or above / checked"},
            {"key": "clip", "head": "clipped cards", "md": "clipped evaluated cards", "kind": "rtext"},
            {"key": "steps", "head": "hyb best steps", "md": "hyb member best steps", "kind": "rtext",
             "group": "hyb ensemble (5 members)"},
            {"key": "feat", "head": "chosen input set", "md": "hyb member input set", "kind": "rtext",
             "group": "hyb ensemble (5 members)"},
            {"key": "zero", "head": "zero state", "md": "members at zero-correction state", "kind": "rtext",
             "group": "hyb ensemble (5 members)"},
            {"key": "repro", "head": "vs E3b", "md": "reproduction vs E3b (max abs diff)", "kind": "rtext"},
            {"key": "wall", "head": "wall s", "md": "wall time s", "kind": "rtext"}]
    notes = [
        "Validity rule: 0 truths, 0 fitted cards and 0 evaluated cards at or above the BSIM4 u0 unit switch; "
        "predicted cards are counted before the x_u0 clip. All three tiers are valid.",
        "hyb members: each member trains the geo_t (geometry, T) and geo_t_res (geometry, par3 anchor "
        "residuals, T) networks and keeps the one with the lower validation criterion; best step of 3000 "
        "(step -1 = zero-correction state, chosen by 0 members on every tier).",
        "Reproduction vs E3b: existing arms O4 par3, pt_interp, lrn, lrn_mean and O2 par3_pop on tiers M and "
        "W, 252 cells each, against the E3b full runs.",
        "Queue jobs I3_1_full_M 2026-10-20 01:55-02:22, I3_1_full_W 02:22-02:40, I3_1_full_H 02:40-02:58, "
        "I3_1_decide 02:58 (out/research/done.txt).",
    ]
    return rf.Table("10_table_i3_1_checks", "I3-1 validity checks, hyb ensemble state and reproduction",
                    cols, rows, notes)


# ================================================================== I3-1 figures

def fig_forest(d: dict, t: str, path: Path) -> list[str]:
    dec = d["dec"]["per_tier"][t]
    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    n = len(CMPS)
    ys = np.arange(n)[::-1]
    for y, (a, b) in zip(ys, CMPS):
        g = dec["gmr"][f"{a}_vs_{b}"]
        col = BLUE if a == "hyb" else ORANGE if a == "lrn" else VIOLET
        ax.plot(g["ci95"], [y, y], color=col, lw=2.0, solid_capstyle="round", zorder=3)
        ax.scatter([g["gmr"]], [y], s=60, color=col, edgecolor=BG, lw=0.8, zorder=4,
                   marker="D" if a == "hyb" else "o" if a == "lrn" else "s")
        e = dec["gmr_eps_0.001"].get(f"{a}_vs_{b}")
        if e:
            ax.scatter([e["gmr"]], [y], s=46, facecolor="none", edgecolor=col, lw=1.2, zorder=4)
        ax.text(1.02, y, f"{g_text(g)}   {g['wins']}/{g['n_devices']}", transform=ax.get_yaxis_transform(),
                fontsize=8.6, color=INK, va="center")
    ax.axvline(1.0, color=INK, lw=0.9, ls="--", zorder=2)
    yh = ys[0]
    thr = {"M": [(0.80, "M1: <= 0.80"), (0.95, "KILL: >= 0.95")],
           "W": [(1.25, "W1: CI upper <= 1.25"), (1.50, "KILL: >= 1.50")],
           "H": [(1.00, "H1: CI upper < 1; KILL: >= 1.00")]}[t]
    for k, (x, lab) in enumerate(thr):
        ax.plot([x, x], [yh - 0.4, yh + 0.4], color=RED, lw=1.4, zorder=5)
        left = k == 0 and len(thr) > 1
        ax.text(x / 1.015 if left else x * 1.015, yh + 0.48, lab, color=RED, fontsize=7.8,
                ha="right" if left else "left", va="bottom")
    ax.set_xscale("log")
    lo = min(min(dec["gmr"][f"{a}_vs_{b}"]["ci95"][0] for a, b in CMPS), 0.7)
    hi = max(max(dec["gmr"][f"{a}_vs_{b}"]["ci95"][1] for a, b in CMPS), 1.6)
    ax.set_xlim(lo * 0.85, hi * 1.15)
    ticks = [x for x in (0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0) if lo * 0.85 <= x <= hi * 1.15]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:g}" for x in ticks])
    ax.minorticks_off()
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{a} vs {b}" for a, b in CMPS], fontsize=9.5)
    ax.set_ylim(-0.7, n - 0.1)
    ax.set_xlabel("GMR(a, b) = exp(mean log((a + 0.01) / (b + 0.01))), log scale; < 1: a lower than b",
                  fontsize=9.5)
    s = d["tiers"][t]["summary"]["table"]
    means = ", ".join(f"{a} {s['O4|' + a]['heldout_all_device_mean']:.4f}"
                      for a in ("hyb", "par3", "pt_interp", "lrn", "lrn_mean", "hyb_mean"))
    ax.set_title(f"I3-1 {TIER_LABEL[t]}: paired GMR with 95% CI, 36 test devices, O4\n"
                 f"held-out all-device mean: {means}", fontsize=10.5, color=INK, loc="left")
    handles = [Line2D([], [], color=BLUE, marker="D", lw=2, label="hyb vs b"),
               Line2D([], [], color=ORANGE, marker="o", lw=2, label="lrn (E3b method) vs b"),
               Line2D([], [], color=VIOLET, marker="s", lw=2, label="hyb_mean vs par3"),
               Line2D([], [], color=GREY, marker="o", lw=0, markerfacecolor="none",
                      label="hollow: GMR with eps 0.001 (sensitivity)"),
               Line2D([], [], color=RED, lw=1.4, label="pre-registered thresholds on hyb vs par3")]
    ax.legend(handles=handles, fontsize=8.2, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.26),
              ncol=3)
    ax.text(1.02, n - 0.35, "GMR [95% CI]   a < b", transform=ax.get_yaxis_transform(), fontsize=8.6,
            color=MUTED, va="center", fontweight="bold")
    fig.subplots_adjust(left=0.15, right=0.74, top=0.88, bottom=0.2)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I31_DEC), rel(I31_RUN / f"tier_{t}/eval/")]


def fig_scaling(d: dict, path: Path) -> list[str]:
    sc = d["dec"]["per_tier"]["M"]["scaling"]
    par3 = d["tiers"]["M"]["summary"]["table"]["O4|par3"]["heldout_all_device_mean"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ns = [3, 6, 12]
    ntr = {3: 36, 6: 72, 12: 144}
    for arm, col, mk in (("lrn", ORANGE, "o"), ("hyb", BLUE, "D")):
        m = [sc[f"{arm}_n{n}"]["heldout_all_device_mean"] for n in ns]
        g = [sc[f"{arm}_n{n}"]["gmr_vs_par3"] for n in ns]
        axs[0].plot(ns, m, color=col, marker=mk, lw=2, ms=7, label=arm)
        for n, v in zip(ns, m):
            axs[0].annotate(f"{v:.4f}", (n, v), textcoords="offset points", xytext=(6, 5), fontsize=8,
                            color=INK)
        off = -0.25 if arm == "lrn" else 0.25
        x = np.array(ns) * np.exp(off * 0.25)
        axs[1].plot(x, [q["gmr"] for q in g], color=col, marker=mk, lw=2, ms=7, label=arm)
        for xi, q in zip(x, g):
            axs[1].plot([xi, xi], q["ci95"], color=col, lw=1.4)
            axs[1].annotate(f"{q['gmr']:.3f}, {q['wins']}/36", (xi, q["gmr"]), textcoords="offset points",
                            xytext=(7, -3), fontsize=7.8, color=INK)
    axs[0].axhline(par3, color=INK, ls="--", lw=1)
    axs[0].text(3, par3 - 0.002, f"par3 (no training population) {par3:.4f}", fontsize=8, color=INK,
                va="top")
    axs[1].axhline(1.0, color=INK, ls="--", lw=1)
    for ax in axs:
        style(ax)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([f"{n}\n({ntr[n]} devices)" for n in ns])
        ax.minorticks_off()
        ax.set_xlabel("training draws per geometry (training devices)", fontsize=9.5)
    axs[0].legend(fontsize=9, frameon=False, loc="center right")
    axs[1].legend(fontsize=9, frameon=False, loc="lower center")
    axs[0].set_ylabel("held-out all-device mean RRMS (O4)", fontsize=9.5)
    axs[1].set_ylabel("GMR vs par3 with 95% CI (eps 0.01)", fontsize=9.5)
    axs[0].set_xlim(2.5, 17)
    fig.suptitle("I3-1 scaling arm, tier M (reported, not deciding): lrn and hyb trained on 3, 6 or 12 draws "
                 "per geometry; same 36 test devices; n = 6 is the primary arm", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I31_DEC), rel(I31_RUN / "tier_M/summary.json"), rel(I31_RUN / "tier_M/eval/")]


def fig_per_temp(d: dict, path: Path) -> list[str]:
    temps = [77, 110, 150, 190, 225, 260, 300]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    arms = ["par3", "pt_interp", "lrn", "lrn_mean", "hyb", "hyb_mean"]
    for ax, t in zip(axs, TIERS):
        style(ax)
        tab = d["tiers"][t]["summary"]["table"]
        for a in arms:
            pt = tab[f"O4|{a}"]["per_temp"]
            ax.plot(temps, [max(pt[str(T)], 1e-4) for T in temps], color=ARM_COLOR[a], lw=1.8,
                    marker="o", ms=4.5, label=a, ls="--" if a in ("lrn_mean", "hyb_mean") else "-")
        fl = tab["floor|floor"]["per_temp"]
        ax.scatter([int(T) for T in HELD], [fl[T] for T in HELD], color=RED, marker="x", s=40, zorder=5,
                   label="floor (fit on held-out data)")
        for T in HELD:
            ax.axvspan(int(T) - 6, int(T) + 6, color="#ecebe6", zorder=0)
        ax.set_yscale("log")
        ax.set_xticks(temps)
        ax.set_xlabel("temperature (K); shaded = held-out, others = O4 anchors", fontsize=9)
        ax.set_title(TIER_LABEL[t], fontsize=9.5, color=INK)
    axs[0].set_ylabel("all-device mean RRMS (log scale; values below 1e-4 drawn at 1e-4)", fontsize=9)
    axs[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("I3-1 per-temperature all-device mean RRMS, O4, 36 test devices per tier", fontsize=10.5,
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I31_RUN / f"tier_{t}/summary.json") for t in TIERS]


# ================================================================== I3-2 F1

def f1_load() -> dict:
    s = load(F1 / "summary.json")
    assert not s["smoke"]
    return s


def table_f1_checks(s: dict) -> rf.Table:
    c = s["checks"]
    tol = s["tolerances"]
    n_mod = len(s["export"])
    rows = [
        {"k": "F1.0", "c": "export text: g from emitted Verilog-A lines vs Python; OpenVAF compiles",
         "v": f"max |g diff| {c['F1.0_export_text']['max_abs_g_diff']:.1e} (<= {tol['F1.0_g']:.0e}); "
              f"{n_mod}/{n_mod} compiled",
         "ok": "pass" if not c["F1.0_export_text"]["fail"] else "fail"},
        {"k": "F1.1", "c": "ngspice-46 vs ngspice-41, uncorrected cards (published, start, C0_refit, M_bias)",
         "v": f"max rel. current {c['F1.1_simulator_reproduction']['max_rel_scored']:.1f} (<= 1e-3); max "
              f"|dRRMS| {c['F1.1_simulator_reproduction']['max_abs_delta_rrms']:.2f} (<= 1e-4)",
         "ok": "fail" if c["F1.1_simulator_reproduction"]["fail"] else "pass"},
        {"k": "", "c": "ngspice-41 reproduces stored C0_refit, M_bias, released-card RRMS",
         "v": f"max |diff| {c['F1.1_simulator_reproduction']['max_stored_check']:.1g} (<= 1e-9)",
         "ok": "pass" if c["F1.1_simulator_reproduction"]["stored_values_reproduced"] else "fail"},
        {"k": "F1.2", "c": "SPICE-native (ngspice-46 + OSDI) vs Python I_bsim exp(g) on ngspice-46",
         "v": f"max |dRRMS| {c['F1.2_va_implementation']['max_abs_delta_rrms']:.1e} (<= 1e-4); max rel. "
              f"current fold {c['F1.2_va_implementation']['max_rel_fold_scored']:.1e}, full "
              f"{c['F1.2_va_implementation']['max_rel_full_scored']:.1e} (<= 1e-3); 0 convergence messages, "
              f"0 non-finite",
         "ok": "fail" if c["F1.2_va_implementation"]["fail"] else "pass"},
        {"k": "F1.3", "c": "SPICE-native (ngspice-46) vs stored I3-2 M_bias (ngspice-41, Python)",
         "v": f"max per-device |d| {c['F1.3_end_to_end']['max_abs_delta_device']:.2f} (<= 2e-4); "
              f"|d all-device mean| {c['F1.3_end_to_end']['abs_delta_all_device_mean']:.3f} (<= 1e-4)",
         "ok": "fail" if c["F1.3_end_to_end"]["fail"] else "pass"},
        {"k": "diag.", "c": "tight tolerance (reltol 1e-9, abstol 1e-18): SPICE-native vs Python, ngspice-46",
         "v": f"max |dRRMS| {c['tight_tolerance_diagnostic']['max_abs_delta_rrms']:.1e}; max rel. current "
              f"{c['tight_tolerance_diagnostic']['max_rel_native_vs_python_scored']:.1e}",
         "ok": "no pass/fail"},
    ]
    cols = [{"key": "k", "head": "check", "md": "check", "kind": "text"},
            {"key": "c", "head": "comparison", "md": "comparison", "kind": "text"},
            {"key": "v", "head": "value (pre-registered threshold)", "md": "value (threshold)", "kind": "rtext"},
            {"key": "ok", "head": "outcome", "md": "outcome", "kind": "rtext"}]
    notes = [
        "Pre-registration out/research/explore/I3_2/followup_prereg.json (commit 4ae726a); amendment "
        "followup_prereg_amendment_2026-10-20.json (commit 4f4e7ef): relative current differences exclude "
        "points with |I| < 1e-20 A (numerical zeros at Vds = 0). Under the literal definition (|I| > 0) "
        f"F1.1 and F1.2 also fail (literal F1.2 fold value {c['F1.2_va_implementation']['max_rel_fold_scored_literal']:.1e}).",
        "Binaries: ngspice-41 arm64 build (/Users/anrunw/cryo-ng41/arm64, configured without OSDI) and "
        "Homebrew ngspice-46 (/opt/homebrew/bin/ngspice). The SPICE-native runs use ngspice-46 because the "
        "ngspice-41 build has no OSDI. Corrected netlist: BSIM4 drain on node ndi, compiled correction between "
        "nd and ndi with a zero-volt sense branch (V(sense) <+ RS * I(sense), RS = 0).",
        "F1.1 and F1.3 fail on pmos_L0p35_W0p55 and pmos_L8_W5 only (next table): ngspice-46 and ngspice-41 "
        "give different currents for the uncorrected cards on these two devices. On the other 10 devices the "
        "uncorrected ngspice-46 and ngspice-41 currents are equal (max relative difference 0).",
        f"F1.2 per-device |dRRMS| (SPICE-native minus Python, both ngspice-46) is at most "
        f"{max(abs(v['native']['delta_native_vs_python_ng46']) for v in s['per_device'].values()):.1e} on all "
        f"12 devices. The relative current difference of the fold models exceeds 1e-3 on "
        + ", ".join(f"{t} ({v['native']['max_rel_native_vs_python_ng46_scored']:.1e})"
                    for t, v in s["per_device"].items()
                    if v["native"]["max_rel_native_vs_python_ng46_scored"] > 1e-3)
        + "; on the other devices it is 7.9e-4 to 1.0e-3. With reltol 1e-9 the per-device |dRRMS| is at most "
        f"{c['tight_tolerance_diagnostic']['max_abs_delta_rrms']:.1e}; the relative current difference is "
        f"below 3e-6 on "
        f"{sum(v['native']['tight_tolerance_diagnostic']['max_rel_native_vs_python_scored'] < 3e-6 for v in s['per_device'].values())}"
        " devices and is "
        + ", ".join(f"{v['native']['tight_tolerance_diagnostic']['max_rel_native_vs_python_scored']:.1e} ({t})"
                    for t, v in s["per_device"].items()
                    if v["native"]["tight_tolerance_diagnostic"]["max_rel_native_vs_python_scored"] >= 3e-6)
        + " on the others.",
        "F1b (SPICE-native runs on an OSDI-enabled ngspice-41 build) is in progress (another agent); no F1b "
        "result is used here.",
    ]
    return rf.Table("11_table_i3_2f_f1_checks",
                    "I3-2 follow-up F1: SPICE-native Verilog-A validation, pre-registered checks", cols, rows,
                    notes)


def table_f1_devices(s: dict) -> tuple[rf.Table, list[str]]:
    rows, diff_devs = [], []
    agg = {k: [] for k in ("pub41", "pub46", "c041", "c046", "py41", "py46", "nat")}
    for tag, d in s["per_device"].items():
        r = d["rrms"]
        rp = d["reproduction"]
        n = d["native"]
        same46 = len({r["published_ng46"], r["start_ng46"], r["c0_ng46"], r["mbias_ng46"]}) == 1
        if rp["published"]["abs_delta_rrms"] > 0:
            diff_devs.append(tag)
        rows.append({"dev": tag, "pub41": r["published_ng41"], "pub46": r["published_ng46"],
                     "c041": r["c0_ng41"], "c046": r["c0_ng46"],
                     "py41": r["py_fold_ng41"], "py46": r["py_fold_ng46"], "nat": r["native_fold_ng46"],
                     "dn": f"{abs(n['delta_native_vs_python_ng46']):.1e}",
                     "rel": f"{n['max_rel_native_vs_python_ng46_scored']:.1e}",
                     "tight": f"{n['tight_tolerance_diagnostic']['abs_delta_rrms']:.1e}",
                     "same": "yes" if same46 and rp["published"]["abs_delta_rrms"] > 0 else ""})
        assert abs(r["py_fold_ng41"] - n["rrms_stored"]) < 1e-12, tag
        for k in agg:
            agg[k].append(rows[-1][k])
    for k in agg:
        agg[k] = dict(zip(s["per_device"], agg[k]))
    fa = {k: fam(v) for k, v in agg.items()}
    assert abs(fa["nat"]["all"] - s["aggregates"]["native_ng46"]["all_device_mean"]) < 1e-12
    assert abs(fa["py41"]["all"] - s["aggregates"]["stored_M_bias"]["all_device_mean"]) < 1e-12
    for lab, key in (("all-device", "all"), ("combined", "combined")):
        rows.append({"dev": lab, **{k: fa[k][key] for k in agg}, "dn": "", "rel": "", "tight": "", "same": ""})
    cols = [{"key": "dev", "head": "device", "md": "device", "kind": "text"},
            {"key": "pub41", "head": "ng41", "md": "published ng41", "kind": "rrms",
             "group": "published 77 K card"},
            {"key": "pub46", "head": "ng46", "md": "published ng46", "kind": "rrms",
             "group": "published 77 K card", "family": True},
            {"key": "c041", "head": "ng41", "md": "C0_refit card ng41", "kind": "rrms",
             "group": "C0_refit card"},
            {"key": "c046", "head": "ng46", "md": "C0_refit card ng46", "kind": "rrms", "group": "C0_refit card",
             "family": True},
            {"key": "py41", "head": "Python ng41", "md": "M_bias Python ng41 (stored)", "kind": "rrms",
             "group": "M_bias refit card x exp(g), fold model"},
            {"key": "py46", "head": "Python ng46", "md": "M_bias Python ng46", "kind": "rrms",
             "group": "M_bias refit card x exp(g), fold model", "family": True},
            {"key": "nat", "head": "native ng46", "md": "M_bias SPICE-native ng46", "kind": "rrms",
             "group": "M_bias refit card x exp(g), fold model", "family": True},
            {"key": "dn", "head": "|native - Py46|", "md": "abs(native - Python ng46)", "kind": "rtext",
             "group": "F1.2 (ngspice-46)"},
            {"key": "rel", "head": "max rel. I", "md": "max rel current diff", "kind": "rtext",
             "group": "F1.2 (ngspice-46)"},
            {"key": "tight", "head": "|dRRMS| tight", "md": "abs dRRMS at reltol 1e-9", "kind": "rtext",
             "group": "F1.2 (ngspice-46)"},
            {"key": "same", "head": "ng46 equal for 4 cards", "md": "ng46 RRMS equal for all 4 cards",
             "kind": "rtext"}]
    notes = [
        "Strict paper RRMS with frozen inclusion on the measured 77 K curves, 12 REAL_DEV devices; all-device = "
        "mean over 12, combined = mean of the nMOS (5) and pMOS (7) family means. Tinted columns: ngspice-46.",
        "Python ng41 = the stored I3-2 M_bias held-out value (reproduced with difference 0). published ng41 "
        "equals out/research/reference_scores.json 'released' (difference 0).",
        f"On {', '.join(diff_devs)} ngspice-46 returns the same RRMS for the published, starting, C0_refit and "
        "M_bias refit cards (last column), while ngspice-41 returns different values for the four cards. On "
        "the other 10 devices ngspice-46 and ngspice-41 give equal RRMS for every card.",
        "max rel. I = max |I_native - I_Python| / |I_Python| over scored points with |I| >= 1e-20 A "
        "(pre-registered threshold 1e-3). |dRRMS| tight = same comparison with '.options reltol=1e-9 "
        "abstol=1e-18' (diagnostic).",
    ]
    return rf.Table("12_table_i3_2f_f1_per_device",
                    "I3-2 follow-up F1 per device: ngspice-41 vs ngspice-46 and SPICE-native vs Python RRMS",
                    cols, rows, notes), diff_devs


def fig_f1_iv(s: dict, devs: list[str], path: Path) -> list[str]:
    z = np.load(F1 / "curves.npz")
    fig, axs = plt.subplots(len(devs), 2, figsize=(14, 5.2 * len(devs)), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for row, tag in zip(np.atleast_2d(axs), devs):
        r = s["per_device"][tag]["rrms"]
        idx = sorted({int(k.split("|")[1]) for k in z.files if k.startswith(tag + "|")})
        for i in idx:
            vd, vg = z[f"{tag}|{i}|vd"], z[f"{tag}|{i}|vg"]
            is_vd = np.ptp(vd) > 0.1
            ax = row[0] if is_vd else row[1]
            x = np.abs(vd if is_vd else vg)
            ax.scatter(x, np.abs(z[f"{tag}|{i}|measured"]), s=5, color=INK, zorder=4)
            for k2 in ("start_ng46", "c0_ng46", "mbias_ng46"):
                assert np.array_equal(z[f"{tag}|{i}|published_ng46"], z[f"{tag}|{i}|{k2}"]), (tag, i, k2)
            for key, col, ls in (("published_ng41", GREY, "--"), ("c0_ng41", BLUE, "-"),
                                 ("published_ng46", ORANGE, "-")):
                ax.plot(x, np.abs(z[f"{tag}|{i}|{key}"]), color=col, ls=ls, lw=1.3, zorder=3)
        for ax, kind in zip(row, ("Id-Vd at |Vgs| = 0.37-1.85 V", "Id-Vg at |Vds| = 0.01-1.85 V (log)")):
            style(ax)
            ax.set_title(f"{tag}: {kind}", fontsize=9.5, color=INK)
            ax.set_ylabel("|Id| (A)", fontsize=9)
        row[0].set_xlabel("|Vds| (V)", fontsize=9)
        row[1].set_xlabel("|Vgs| (V)", fontsize=9)
        row[1].set_yscale("log")
        row[1].set_ylim(1e-12, None)
        row[0].text(0.01, 0.98, f"RRMS: published ng41 {r['published_ng41']:.4f}, ng46 {r['published_ng46']:.4f}; "
                    f"C0_refit ng41 {r['c0_ng41']:.4f}, ng46 {r['c0_ng46']:.4f}", transform=row[0].transAxes,
                    fontsize=8.3, va="top", color=INK)
    handles = [Line2D([], [], color=INK, marker="o", lw=0, ms=3, label="measured 77 K"),
               Line2D([], [], color=GREY, ls="--", label="published 77 K card, ngspice-41"),
               Line2D([], [], color=BLUE, label="C0_refit card, ngspice-41"),
               Line2D([], [], color=ORANGE, label="ngspice-46: published, starting, C0_refit and M_bias refit "
                      "cards (identical currents, one line)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.8, frameon=False)
    fig.suptitle("I3-2 follow-up F1: uncorrected cards simulated with ngspice-41 and ngspice-46 on the two "
                 "devices where the simulators differ (same deck, same cards; no correction)", fontsize=10.5,
                 color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F1 / "curves.npz"), rel(F1 / "summary.json")]


# ================================================================== I3-2 F2

F2_DESC = {"a_seeds_5_9": "ensemble seeds 5-9", "a_seeds_10_14": "ensemble seeds 10-14",
           "a_seeds_15_19": "ensemble seeds 15-19", "b_width8": "MLP 3-8-8-1",
           "b_width32": "MLP 3-32-32-1", "b_depth1": "MLP 3-16-1",
           "c_train_round1_cards": "g trained on round-1 card records"}


def f2_load() -> dict:
    real = load(F2_REAL)
    assert not real["smoke"]
    c0 = load(I21_REAL)["per_device"]["C0_refit"]
    mb = load(I32_REAL)["per_device"]["M_bias"]
    summ = load(F2_SUM)
    for v, r in real["variants"].items():
        per = r["per_device"]
        check_ratio(r["vs_C0_refit"], paired_ratio(per, c0), v)
        check_ratio(r["vs_I3_2_M_bias"], paired_ratio(per, mb), v)
        assert abs(fam(per)["all"] - r["summary"]["all_device_mean"]) < 1e-12
    check_ratio(real["reference"]["I3_2_M_bias"]["vs_C0_refit"], paired_ratio(mb, c0), "M_bias")
    pdk = {}
    for k, p in F2_PDK.items():
        r = load(p)
        assert not r["smoke"] and r["n_targets"] == 48
        per_m, per_c = r["M_bias"]["per_device"], r["C0_refit"]["per_device"]
        cl = {t: t.split("#")[0] for t in per_m}
        check_ratio(r["M_bias"]["vs_C0_refit"], paired_ratio(per_m, per_c, cl), k)
        pdk[k] = r
    floors = {k: load(p)["summary"] for k, p in FLOORS.items()}
    return {"real": real, "c0": c0, "mb": mb, "summ": summ, "pdk": pdk, "floors": floors}


def table_f2_real(f: dict) -> rf.Table:
    real = f["real"]
    rows = []
    c0f = fam(f["c0"])
    rows.append({"v": "C0_refit", "d": "control (stored I2-1 rows)", "all": c0f["all"], "comb": c0f["combined"],
                 "n": c0f["nmos"], "p": c0f["pmos"], "med": c0f["median"], "fail": "0/12"})
    mbf = fam(f["mb"])
    ref = real["reference"]["I3_2_M_bias"]["vs_C0_refit"]
    rows.append({"v": "I3-2 M_bias", "d": "seeds 0-4, MLP 3-16-16-1, round-3 cards", "all": mbf["all"],
                 "comb": mbf["combined"], "n": mbf["nmos"], "p": mbf["pmos"], "med": mbf["median"], "fail": "0/12",
                 "g": r_text(ref), "w": f"{ref['wins']}/12", "le": "yes"})
    for v, desc in F2_DESC.items():
        r = real["variants"][v]
        fa = fam(r["per_device"])
        rows.append({"v": v, "d": desc, "all": fa["all"], "comb": fa["combined"], "n": fa["nmos"],
                     "p": fa["pmos"], "med": fa["median"], "fail": f"{r['summary']['n_failed']}/12",
                     "g": r_text(r["vs_C0_refit"]), "w": f"{r['vs_C0_refit']['wins']}/12",
                     "gm": r_text(r["vs_I3_2_M_bias"]), "wm": f"{r['vs_I3_2_M_bias']['wins']}/12",
                     "le": yn(r["all_device_mean_le_R2_threshold"]), "f": "fail" if r["fails"] else "pass"})
    cols = [{"key": "v", "head": "variant", "md": "variant", "kind": "text"},
            {"key": "d", "head": "change from I3-2 M_bias", "md": "change", "kind": "text"},
            {"key": "all", "head": "all-device", "md": "all-device", "kind": "rrms"},
            {"key": "comb", "head": "combined", "md": "combined", "kind": "rrms"},
            {"key": "n", "head": "nMOS", "md": "nMOS", "kind": "rrms", "family": True},
            {"key": "p", "head": "pMOS", "md": "pMOS", "kind": "rrms", "family": True},
            {"key": "med", "head": "median", "md": "median", "kind": "rrms"},
            {"key": "fail", "head": "failed", "md": "failed", "kind": "rtext"},
            {"key": "g", "head": "GMR vs C0_refit [CI]", "md": "GMR vs C0_refit [95% CI]", "kind": "rtext"},
            {"key": "w", "head": "< C0", "md": "devices below C0_refit", "kind": "rtext"},
            {"key": "gm", "head": "GMR vs I3-2 M_bias [CI]", "md": "GMR vs I3-2 M_bias [95% CI]",
             "kind": "rtext"},
            {"key": "wm", "head": "< M_bias", "md": "devices below I3-2 M_bias", "kind": "rtext"},
            {"key": "le", "head": "<= 0.08698", "md": "all-device <= 0.08698", "kind": "rtext"},
            {"key": "f", "head": "check", "md": "pre-registered check", "kind": "rtext"}]
    g = f["summ"]["real_groups"]
    rng = g["a"]["all_device_mean_range_4_seed_sets"]
    notes = [
        "REAL_DEV leave-one-device-out (12 devices: 5 nMOS, 7 pMOS), held-out card refit from the round-3 "
        "starting card with the frozen correction in the loop (max_nfev 400, irls 2, diff_step 0.01), strict "
        "paper RRMS. Each variant changes one item of the I3-2 M_bias protocol.",
        "GMR = paired geometric-mean ratio exp(mean ln(variant / reference)), device bootstrap 95% CI, 10000 "
        "resamples, default_rng(0); recomputed from the stored per-device values and checked (|difference| "
        "< 1e-12). A variant fails if the CI upper bound vs C0_refit is >= 1 or any device fails.",
        f"Group outcomes: seeds (a) {'fail' if g['a']['fails'] else 'pass'}, capacity (b) "
        f"{'fail' if g['b']['fails'] else 'pass'}, training cards (c) {'fail' if g['c']['fails'] else 'pass'}. "
        f"All-device range over the four seed sets (0-4, 5-9, 10-14, 15-19): {rng[0]:.4f}-{rng[1]:.4f}.",
        "0.08698 = the I3-2 R2 threshold (T1_inloop 0.08198 + 0.005); reported, not a check. "
        "c_train_round1_cards: the held-out refit still starts from the round-3 card; the round-1 card "
        "records reproduce the stored round-1 RRMS (difference 0).",
        "REAL_DEV was used by E1, I2-1 and I3-2; these values are development evidence. Queue job "
        "I3_2_F2_real 2026-10-20 02:59-03:15 (955 s).",
    ]
    return rf.Table("14_table_i3_2f_f2_real", "I3-2 follow-up F2 on REAL_DEV: seed, capacity and "
                    "training-card variants of M_bias", cols, rows, notes)


def table_f2_tier_b(f: dict) -> rf.Table:
    rows = []
    for k in ("gf180", "ihp"):
        r = f["pdk"][k]
        dg = f["summ"]["d_group"]["per_pdk"][k]
        for arm in ("C0_start", "C0_refit", "M_bias"):
            sm = r[arm]["summary"]
            row = {"pdk": PDK_LABEL[k], "arm": arm, "all": sm["all_device_mean"], "med": sm["median"],
                   "fail": f"{sm['n_failed']}/48"}
            if arm == "M_bias":
                c = r["M_bias"]["vs_C0_refit"]
                row.update({"g": r_text(c), "w": f"{c['wins']}/48",
                            "r4": "pass" if dg["passes_R4_rule"] else "fail", "ci": yn(dg["ci_high_lt_1"])})
            rows.append(row)
        fl = f["floors"][k]
        for key, lab in (("b/seven", "floor, seven-parameter refit"), ("b/wide", "floor, wide-box refit")):
            rows.append({"pdk": PDK_LABEL[k], "arm": lab + " (3 devices, set 0)", "all": fl[key]["mean_floor"]})
        rows.append({"pdk": PDK_LABEL[k], "arm": "injected noise level (3 devices, set 0)",
                     "all": fl["b/seven"]["mean_noise"]})
    cols = [{"key": "pdk", "head": "PDK", "md": "PDK", "kind": "text"},
            {"key": "arm", "head": "row", "md": "row", "kind": "text"},
            {"key": "all", "head": "all-target mean", "md": "all-target mean", "kind": "rrms"},
            {"key": "med", "head": "median", "md": "median", "kind": "rrms"},
            {"key": "fail", "head": "failed", "md": "failed", "kind": "rtext"},
            {"key": "g", "head": "GMR vs C0_refit [CI]", "md": "GMR vs C0_refit [95% CI]", "kind": "rtext"},
            {"key": "w", "head": "< C0_refit", "md": "targets below C0_refit", "kind": "rtext"},
            {"key": "r4", "head": "R4 rule", "md": "R4 rule (GMR <= 0.90, 0 failed)", "kind": "rtext"},
            {"key": "ci", "head": "CI upper < 1", "md": "CI upper < 1", "kind": "rtext"}]
    d = f["summ"]["d_group"]
    notes = [
        "Simulated tier b truth sets 0-2 (16 geometries x 3 = 48 targets per PDK), leave-one-geometry-out (16 "
        "folds), M_bias seeds 0-4; new starting-card records (polish from the truth's local-wide projection, "
        "wide box) and new C0_refit runs. GMR with geometry-cluster bootstrap 95% CI (10000 resamples, "
        "default_rng(0)), recomputed and checked.",
        f"F2(d) check: fails if on both PDKs GMR > 0.90 or M_bias has a failed target (the I3-2 R4 rule on tier "
        f"b). Outcome: {'fail' if d['fails'] else 'pass'} (GF180 {d['per_pdk']['gf180']['ratio']:.3f}, IHP "
        f"{d['per_pdk']['ihp']['ratio']:.3f}).",
        "Floor rows: out/research/pdk2/floors/2026-10-16_final (3 devices per PDK, truth set 0, per-device "
        "refit to its own tier b truth; seven-parameter and wide-box parameter sets); noise = RRMS of the "
        "injected measurement noise. These are separate runs on 3 devices, not the 48 F2 targets.",
        "C0_start = starting card without refit or correction. Queue jobs I3_2_F2_gf180_b 03:15-03:31 (982 s), "
        "I3_2_F2_ihp_b 03:31-03:49 (1086 s).",
    ]
    return rf.Table("15_table_i3_2f_f2_tier_b", "I3-2 follow-up F2(d): GF180 and IHP tier b, M_bias vs "
                    "C0_refit, with the tier b refit floors", cols, rows, notes)


def fig_f2_forest(f: dict, path: Path) -> list[str]:
    real = f["real"]
    items = [("I3-2 M_bias (seeds 0-4)", real["reference"]["I3_2_M_bias"]["vs_C0_refit"], INK, "REAL_DEV")]
    for v in F2_DESC:
        items.append((f"{v} ({F2_DESC[v]})", real["variants"][v]["vs_C0_refit"], BLUE, "REAL_DEV"))
    for k in ("gf180", "ihp"):
        items.append((f"{PDK_LABEL[k]} M_bias", f["pdk"][k]["M_bias"]["vs_C0_refit"], ORANGE, "tier b"))
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    ys = np.arange(len(items))[::-1]
    for y, (lab, c, col, grp) in zip(ys, items):
        ax.plot([c["ci_low"], c["ci_high"]], [y, y], color=col, lw=2.0)
        ax.scatter([c["ratio"]], [y], color=col, s=55, edgecolor=BG, zorder=4,
                   marker="D" if grp == "REAL_DEV" else "s")
        n = 12 if grp == "REAL_DEV" else 48
        ax.text(1.02, y, f"{r_text(c)}   {c['wins']}/{n}", transform=ax.get_yaxis_transform(), fontsize=8.6,
                va="center", color=INK)
    ax.axvline(1.0, color=INK, ls="--", lw=0.9)
    ax.axvline(0.90, color=RED, lw=1.2, ls=":")
    ax.text(0.90 * 0.997, len(items) - 0.45, "0.90 (R4 / F2(d) threshold)", color=RED, fontsize=8, ha="right")
    ax.axhline(1.5, color=GREY, lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], fontsize=9)
    ax.set_ylim(-0.7, len(items))
    ax.set_xlabel("paired geometric-mean ratio vs C0_refit (refit without correction), 95% bootstrap CI; "
                  "< 1: lower RRMS than C0_refit", fontsize=9)
    ax.set_title("I3-2 follow-up F2: M_bias / C0_refit on REAL_DEV (12 devices, device bootstrap)\nand on "
                 "simulated tier b (48 targets, geometry-cluster bootstrap)", fontsize=10, color=INK, loc="left")
    ax.text(1.02, len(items) - 0.45, "GMR [95% CI]   < C0_refit", transform=ax.get_yaxis_transform(),
            fontsize=8.6, color=MUTED, fontweight="bold")
    fig.subplots_adjust(left=0.33, right=0.77, top=0.87, bottom=0.11)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F2_REAL), rel(F2_PDK["gf180"]), rel(F2_PDK["ihp"]), rel(I21_REAL), rel(I32_REAL)]


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)

    d = load_i31()
    _, _, verdict = i31_conditions(d)
    s1 = f1_load()
    f2 = f2_load()

    items, tables = [], []

    def add_table(t: rf.Table, desc: str, srcs: list[str]) -> None:
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    i31_src = [rel(I31_PREREG), rel(I31_ADD), rel(I31_DEC)] + \
        [rel(I31_RUN / f"tier_{t}/summary.json") for t in TIERS]
    add_table(table_i31_decision(d), "Table of the I3-1 pre-registered conditions M1-M3, W1, H1, H2, F, the KILL "
              "conditions and the validity check, with values and the rule output (KILL)",
              i31_src + [rel(I31_RUN / f"tier_{t}/eval/") for t in TIERS])
    for k, t in enumerate(TIERS):
        png = f"{2 + k:02d}_i3_1_gmr_forest_tier_{t}.png"
        src = fig_forest(d, t, folder / png)
        items.append((png, f"Forest plot of the paired GMR (eps 0.01, 95% CI, wins) of hyb, lrn and hyb_mean "
                      f"against the other arms on {TIER_LABEL[t]}, with the pre-registered thresholds on hyb vs "
                      f"par3 and the eps 0.001 values", src))
    add_table(table_i31_gmr(d), "Table of every reported GMR comparison per tier (M, W, H): GMR [95% CI], wins, "
              "eps 0.001 sensitivity", [rel(I31_DEC)])
    add_table(table_i31_per_temp(d), "Table of held-out all-device mean, median and per-temperature RRMS "
              "(110/190/260 K), failed cells and anchor-temperature RRMS for every O4 arm and the floor, per "
              "tier", [rel(I31_RUN / f"tier_{t}/summary.json") for t in TIERS])
    src = fig_per_temp(d, folder / "07_i3_1_per_temperature.png")
    items.append(("07_i3_1_per_temperature.png", "All-device mean RRMS per temperature (77-300 K) for par3, "
                  "pt_interp, lrn, lrn_mean, hyb and hyb_mean, with the held-out floor, one panel per tier", src))
    src = fig_scaling(d, folder / "08_i3_1_scaling.png")
    items.append(("08_i3_1_scaling.png", "Scaling arm on tier M: held-out all-device mean (left) and GMR vs "
                  "par3 with 95% CI (right) for lrn and hyb trained on 3, 6 and 12 draws per geometry", src))
    add_table(table_i31_lrn(d), "Table of the E3b method lrn on tiers M, W and H: held-out means, GMR vs par3, "
              "lrn_mean and pt_interp, E3-style R_O4 and R_O2 and the E3 rule output per tier",
              [rel(I31_RUN / f"tier_{t}/summary.json") for t in TIERS] + [rel(I31_DEC)])
    add_table(table_i31_checks(d), "Table of the u0 unit-switch validity check per tier (truths, fitted, "
              "predicted, evaluated cards, clipped cards), hyb member best steps and input sets, reproduction "
              "of the E3b arms and wall time", [rel(I31_RUN / f"tier_{t}/summary.json") for t in TIERS]
              + [rel(I31_DEC)])

    f1_src = [rel(FU_PREREG), rel(FU_AMEND), rel(F1 / "summary.json")]
    add_table(table_f1_checks(s1), "Table of the F1 pre-registered checks F1.0-F1.3 with values, thresholds "
              "and outcomes, and the tight-tolerance diagnostic", f1_src)
    t13, diff_devs = table_f1_devices(s1)
    add_table(t13, "Table of F1 per device: published and C0_refit card RRMS on ngspice-41 and ngspice-46, "
              "M_bias refit card x exp(g) in Python (ngspice-41, ngspice-46) and SPICE-native (ngspice-46), "
              "SPICE-native minus Python differences, all-device and combined rows", f1_src)
    src = fig_f1_iv(s1, diff_devs, folder / "13_i3_2f_f1_iv_ng41_vs_ng46.png")
    items.append(("13_i3_2f_f1_iv_ng41_vs_ng46.png", f"I-V of {' and '.join(diff_devs)}: measured points and "
                  "NGSpice curves of the published 77 K card and the C0_refit card from ngspice-41, and the "
                  "ngspice-46 curve, which is identical for the published, starting, C0_refit and M_bias refit "
                  "cards (uncorrected)", src))
    f2_src = [rel(FU_PREREG), rel(F2_REAL), rel(F2_SUM), rel(I21_REAL), rel(I32_REAL)]
    add_table(table_f2_real(f2), "Table of F2 REAL_DEV variants (seed sets, widths, 3-16-1, round-1 training "
              "cards): all-device, combined, nMOS, pMOS, median, failed, GMR vs C0_refit and vs I3-2 M_bias "
              "with 95% CI and wins, check outcome", f2_src)
    add_table(table_f2_tier_b(f2), "Table of F2(d) on GF180 and IHP tier b: C0_start, C0_refit and M_bias "
              "all-target mean, median, failed, GMR vs C0_refit with cluster CI, R4 rule outcome, and the tier "
              "b refit floors and noise level", [rel(FU_PREREG), rel(F2_PDK["gf180"]), rel(F2_PDK["ihp"]),
                                                 rel(F2_SUM), rel(FLOORS["gf180"]), rel(FLOORS["ihp"])])
    src = fig_f2_forest(f2, folder / "16_i3_2f_f2_forest.png")
    items.append(("16_i3_2f_f2_forest.png", "Forest plot of the paired ratio vs C0_refit (95% CI, wins) for "
                  "I3-2 M_bias, the seven F2 REAL_DEV variants and M_bias on GF180 and IHP tier b, with the 0.90 "
                  "threshold", src))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = (f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i3_1_i3_2f.py "
           f"--date {args.date}`")
    md = [f"# Tables, {args.date} {TOPIC}", "",
          f"Generated {now} by {gen} from stored files only. RRMS is the paper metric with frozen curve "
          "inclusion, strict. I3-1 values are on simulated devices; I3-2 F1 and F2 REAL_DEV values are on the 12 "
          "measured REAL_DEV devices (development evidence).", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    dec = d["dec"]["per_tier"]
    g = {t: dec[t]["gmr"] for t in TIERS}
    tb = {t: d["tiers"][t]["summary"]["table"] for t in TIERS}
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen} from stored files only (no NGSpice, no training).", "",
              "## Contents", "",
              "- I3-1: hybrid temperature law hyb = per-device par3 law fitted on the O4 anchors plus a learned "
              "population-shared residual, decided on simulated tiers M, W and H (36 test devices each). "
              f"Pre-registration `{rel(I31_PREREG)}` (commit c877413) and `{rel(I31_ADD)}`; code commit "
              "c59ce14. Queue jobs I3_1_full_M (2026-10-20 01:55-02:22), I3_1_full_W (02:22-02:40), "
              f"I3_1_full_H (02:40-02:58), I3_1_decide (02:58). Rule output: {verdict}.",
              "- I3-2 follow-up F1 (SPICE-native validation of the exported Verilog-A correction) and F2 "
              f"(robustness of M_bias). Pre-registration `{rel(FU_PREREG)}` (commit 4ae726a), amendment "
              f"`{rel(FU_AMEND)}` (commit 4f4e7ef); code commit 5fb08d6. Queue jobs I3_2_F1 (02:58-02:59), "
              "I3_2_F2_real (02:59-03:15), I3_2_F2_gf180_b (03:15-03:31), I3_2_F2_ihp_b (03:31-03:49), "
              "I3_2_F2_summary (03:49). The follow-up does not change the I3-2 rule output (CONTINUE). F1b "
              "(SPICE-native runs on an OSDI-enabled ngspice-41 build) is in progress and not in this batch.",
              "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    rs = f2["real"]["variants"]
    rng = f2["summ"]["real_groups"]["a"]["all_device_mean_range_4_seed_sets"]
    c = s1["checks"]
    summary = [
        f"I3-1 (simulated, 36 test devices per tier, O4): GMR(hyb, par3) M {g_text(g['M']['hyb_vs_par3'])}, "
        f"W {g_text(g['W']['hyb_vs_par3'])}, H {g_text(g['H']['hyb_vs_par3'])}; GMR(hyb, lrn_mean) M "
        f"{g_text(g['M']['hyb_vs_lrn_mean'])}. Conditions met: M1, M3, F; not met: M2, W1, H1, H2; KILL "
        f"conditions W >= 1.50 and H >= 1.00 hit. Rule output {verdict}.",
        f"I3-1, lrn (E3b method): GMR vs par3 M {g_text(g['M']['lrn_vs_par3'])}, W "
        f"{g_text(g['W']['lrn_vs_par3'])}, H {g_text(g['H']['lrn_vs_par3'])}; vs pt_interp H "
        f"{g_text(g['H']['lrn_vs_pt_interp'])}; held-out mean H lrn {tb['H']['O4|lrn']['heldout_all_device_mean']:.4f}, "
        f"par3 {tb['H']['O4|par3']['heldout_all_device_mean']:.4f}, pt_interp "
        f"{tb['H']['O4|pt_interp']['heldout_all_device_mean']:.4f}; E3 rule output on tier H INCONCLUSIVE.",
        f"I3-2 F1: SPICE-native (ngspice-46 + OSDI) vs Python correction on ngspice-46, max per-device |dRRMS| "
        f"{c['F1.2_va_implementation']['max_abs_delta_rrms']:.1e} (tight tolerance "
        f"{c['tight_tolerance_diagnostic']['max_abs_delta_rrms']:.1e}); F1.1 and F1.3 fail because ngspice-46 "
        f"and ngspice-41 differ on {' and '.join(diff_devs)} for the uncorrected cards; F1.2 fails on the "
        f"current criterion (max relative difference {c['F1.2_va_implementation']['max_rel_fold_scored']:.1e} "
        "> 1e-3). F1b (OSDI-enabled ngspice-41) in progress.",
        f"I3-2 F2 REAL_DEV: all seven variants pass (all-device {min(r['summary']['all_device_mean'] for r in rs.values()):.4f}"
        f"-{max(r['summary']['all_device_mean'] for r in rs.values()):.4f}; seed-set range {rng[0]:.4f}-{rng[1]:.4f}; "
        f"I3-2 M_bias 0.0806, C0_refit 0.1072). Tier b: GMR(M_bias, C0_refit) GF180 "
        f"{r_text(f2['pdk']['gf180']['M_bias']['vs_C0_refit'])}, IHP {r_text(f2['pdk']['ihp']['M_bias']['vs_C0_refit'])}; "
        "R4 rule not met on either PDK (F2(d) fail); tier b means 0.0133 (GF180) and 0.0138 (IHP) vs "
        "seven-parameter floors 0.0136 and 0.0146.",
    ]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
