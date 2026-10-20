#!/usr/bin/env python3
"""Report batch for exploration direction I5-1: the I3-2 bias-only correction
M_bias replicated on IHP SG13G2 measured MOSFETs (PSP103) at 233, 300, 343
and 398 K.

Reads stored files only (no NGSpice, no training):

* ``out/research/explore/I5_1/{prereg.json,split.json,decision.json}``,
  ``data/manifest.json``, ``stock/stock_rrms.json``, ``start/summary.json``,
  ``start/b1_screen.json``, ``lodo_{233,300,343,398}K/results.json``,
  ``cross/results.json``, the I-V PNGs in ``figs/`` and ``stock/figs/``.
* ``out/research/explore/I4_1/{real_dev,ihp_c_logo}/results.json`` for the
  sky130 77 K and simulated IHP comparison rows.

Every stored aggregate, paired ratio, per-class ratio, cross-temperature
ratio and the I5-1 rule output is recomputed from the stored per-device
values and checked against the stored values.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i5_1.py --date 2026-10-20
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA,  # noqa: E402
                                   VIOLET, RED, DPI, EXP, load, rel, style)
from cryoml.explore.i3_2_bias import paired_ratio  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i5_1"
I5 = EXP / "I5_1"
PREREG = I5 / "prereg.json"
SPLIT = I5 / "split.json"
DEC = I5 / "decision.json"
MANI = I5 / "data/manifest.json"
STOCK = I5 / "stock/stock_rrms.json"
START = I5 / "start/summary.json"
SCREEN = I5 / "start/b1_screen.json"
CROSS = I5 / "cross/results.json"
FIGS = I5 / "figs"
STOCK_FIGS = I5 / "stock/figs"
TEMPS = (233, 300, 343, 398)
PRIMARY = 300
LODO = {t: I5 / f"lodo_{t}K/results.json" for t in TEMPS}
SKY = EXP / "I4_1/real_dev/results.json"
IHP_SIM = EXP / "I4_1/ihp_c_logo/results.json"

ARMS = ("start", "C0_refit", "M_bias", "T1_inloop", "B1", "B2", "B3_d3", "B5")
BASE = ("B1", "B2", "B3_d3")
ARM_DESC = {"start": "starting card (polish from the stock card), no held-out refit",
            "C0_refit": "control: held-out card refit, no correction",
            "M_bias": "I3-2 correction, refit with g in the loop (method under test)",
            "T1_inloop": "per-(polarity, curve key) gain table (not deployable)",
            "B1": "wider PSP parameter set (base + 70-73 screened extras), no correction",
            "B2": "extrinsic series resistance + self-heating factor, no learned term",
            "B3_d3": "shared polynomial g, total degree 3 (20 coefficients)",
            "B5": "per-DUT correction trained on the held-out DUT (in-sample reference)"}
ARM_COLOR = {"start": "#c9c7c0", "C0_refit": GREY, "M_bias": BLUE, "T1_inloop": INK, "B1": AQUA,
             "B2": VIOLET, "B3_d3": RED, "B5": "#7a9a3a"}
ARM_MARK = {"M_bias": "D", "B5": "v", "C0_refit": "s", "T1_inloop": "^", "start": "x"}
T_COLOR = {233: AQUA, 300: BLUE, 343: ORANGE, 398: RED}
IV_DUTS = ("lv_nmos_W0p3_L0p13", "lv_pmos_W0p15_L0p18", "hv_nmos_W10_L0p5", "hv_pmos_W10_L0p35")


def r_text(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def check_ratio(stored: dict, mine: dict, what: str) -> None:
    for k in ("ratio", "ci_low", "ci_high"):
        assert abs(stored[k] - mine[k]) < 1e-12, (what, k, stored[k], mine[k])
    assert stored["wins"] == mine["wins"], what


def pol(tag: str) -> str:
    return "nmos" if "_nmos_" in tag else "pmos"


def stats(per: dict) -> dict:
    """all-device, nMOS, pMOS, combined, median, failed over a {tag: rrms} dict."""
    v = {t: x for t, x in per.items() if np.isfinite(x)}
    n = [x for t, x in v.items() if pol(t) == "nmos"]
    p = [x for t, x in v.items() if pol(t) == "pmos"]
    return {"all": float(np.mean(list(v.values()))), "nmos": float(np.mean(n)), "pmos": float(np.mean(p)),
            "combined": (float(np.mean(n)) + float(np.mean(p))) / 2,
            "median": float(np.median(list(v.values()))), "failed": len(per) - len(v), "n": len(per)}


def sub(per: dict, cls: str) -> dict:
    return per if cls == "all" else {t: x for t, x in per.items() if t.startswith(cls + "_")}


def dev_key(tag: str):
    cls, p, w, l = tag.split("_")
    f = lambda s: float(s[1:].replace("p", "."))  # noqa: E731
    return (0 if cls == "lv" else 1, 0 if p == "nmos" else 1, f(w), f(l))


# ================================================================== loading and checks

def load_all() -> dict:
    dec, split, mani = load(DEC), load(SPLIT), load(MANI)
    assert not dec["smoke"] and dec["primary_T_K"] == PRIMARY
    res = {t: load(LODO[t]) for t in TEMPS}
    dev = split["DEV"]
    assert len(dev) == 18 and len(split["TEST"]) == 10
    for t, r in res.items():
        assert r["T_K"] == t and not r["smoke"] and r["preregistered_config"], t
        per = r["per_device"]
        assert sorted(per["M_bias"]) == sorted(dev), t          # DEV only
        for cls in ("all", "lv", "hv"):
            for arm in ARMS:
                s, mine = r["summary"][cls][arm], stats(sub(per[arm], cls))
                assert abs(s["all_device_mean"] - mine["all"]) < 1e-12, (t, cls, arm)
                assert abs(s["median"] - mine["median"]) < 1e-12, (t, cls, arm)
                assert s["n_failed"] == mine["failed"], (t, cls, arm)
            for name, st in r["comparisons"][cls].items():
                a, b = name.split("_vs_")
                check_ratio(st, paired_ratio(sub(per[a], cls), sub(per[b], cls)), f"{t} {cls} {name}")
        for arm in ARMS[1:]:
            for tag, row in r["rows"][arm].items():
                assert abs(row["rrms"] - per[arm][tag]) < 1e-15, (t, arm, tag)
        # stored decision recomputed
        key = "primary" if t == PRIMARY else None
        d = dec["primary"] if key else dec["secondary"][f"{t}K"]
        c = r["comparisons"]["all"]
        check_ratio(d["R1_detail"], c["M_bias_vs_C0_refit"], f"R1 {t}")
        s = r["summary"]["all"]
        r1 = c["M_bias_vs_C0_refit"]["ci_high"] < 1
        r2 = s["M_bias"]["all_device_mean"] <= s["T1_inloop"]["all_device_mean"] + 0.005
        r3 = s["M_bias"]["n_failed"] == 0
        assert d["checks"] == {"R1_ci_high_lt_1": r1, "R2_mean_le_T1_inloop_plus_0.005": r2,
                               "R3_zero_failed": r3}, t
        failed = [k for k, ok in (("R1_ci_high_lt_1", r1), ("R2_mean_le_T1_inloop_plus_0.005", r2),
                                  ("R3_zero_failed", r3)) if not ok]
        rep = "replicated" if not failed else f"not replicated ({', '.join(failed)})"
        assert d["replication"] == rep, (t, rep)
        for x in BASE:
            b = d["baselines"][x]
            check_ratio(b, c[f"M_bias_vs_{x}"], f"{t} baseline {x}")
            assert b["beaten"] == (c[f"M_bias_vs_{x}"]["ci_high"] < 1), (t, x)
        nb = [x for x in BASE if not d["baselines"][x]["beaten"]]
        verdict = ("supported against baselines" if rep == "replicated" and not nb
                   else "matched by baseline(s) " + ", ".join(nb))
        assert d["baseline_verdict"] == verdict, (t, verdict)
        for cls in ("lv", "hv"):
            for name, st in dec["per_class"][f"{t}K"][cls].items():
                check_ratio(st, r["comparisons"][cls][name], f"{t} class {cls} {name}")
        for arm, m in dec["means"][f"{t}K"].items():
            assert abs(m["all_device_mean"] - s[arm]["all_device_mean"]) < 1e-15, (t, arm)
    # cross-temperature arm
    cross = load(CROSS)
    assert not cross["smoke"]
    for t in (233, 343, 398):
        ct = cross["per_temperature"][str(t)]
        per = ct["per_device"]
        for arm in ("C0_refit", "M_bias"):   # the cross file carries the lodo values of these arms
            assert all(abs(per[arm][k] - res[t]["per_device"][arm][k]) < 1e-15 for k in dev), (t, arm)
        for cls in ("all", "lv", "hv"):
            for arm in ("X300", "C0_refit", "M_bias"):
                assert abs(ct["summary"][cls][arm]["all_device_mean"]
                           - stats(sub(per[arm], cls))["all"]) < 1e-12, (t, cls, arm)
        for name, st in ct["comparisons"].items():
            a, b = name.split("_vs_")
            check_ratio(st, paired_ratio(per[a], per[b]), f"cross {t} {name}")
            check_ratio(dec["cross_temperature"][str(t)]["comparisons"][name], st, f"dec cross {t} {name}")
        assert abs(dec["cross_temperature"][str(t)]["X300_mean"] - ct["summary"]["all"]["X300"]["all_device_mean"]) \
            < 1e-15
    # stock and start
    stock, start = load(STOCK), load(START)
    assert not start["smoke"]
    for t in TEMPS:
        pt = start["per_temperature"][str(t)]
        st_dev = {k: stock["per_device"][str(t)][k] for k in dev}
        assert abs(pt["stock"]["all_device_mean"] - np.mean(list(st_dev.values()))) < 1e-12, t
        sr = {k: v["start_rrms"] for k, v in pt["rows"].items()}
        assert all(abs(sr[k] - res[t]["per_device"]["start"][k]) < 1e-15 for k in dev), t
        assert abs(pt["start"]["all_device_mean"] - np.mean(list(sr.values()))) < 1e-12, t
        for part in ("DEV", "TEST", "ALL"):
            tags = list(stock["per_device"][str(t)]) if part == "ALL" else split[part]
            for cls in ("lv", "hv", "all"):
                sel = [stock["per_device"][str(t)][k] for k in tags if cls == "all" or k.startswith(cls)]
                assert abs(stock["summary"][f"{t}K|{part}|{cls}"]["all_device_mean"] - np.mean(sel)) < 1e-12
    sky, ihp_sim = load(SKY), load(IHP_SIM)
    for r in (sky, ihp_sim):
        assert not r["smoke"] and r["preregistered_config"]
    return {"dec": dec, "split": split, "mani": mani, "res": res, "cross": cross, "stock": stock,
            "start": start, "screen": load(SCREEN), "prereg": load(PREREG), "sky": sky, "ihp_sim": ihp_sim}


# ================================================================== tables

def table_dataset(d: dict) -> rf.Table:
    mani, split = d["mani"], d["split"]
    stock = d["stock"]["per_device"]
    rows = []
    tot = {p: {t: [0, 0] for t in TEMPS} for p in ("DEV", "TEST")}
    for dv in sorted(mani["devices"], key=lambda x: dev_key(x["tag"])):
        tag = dv["tag"]
        part = "DEV" if tag in split["DEV"] else "TEST"
        row = {"tag": tag, "cls": dv["cls"].upper(), "typ": dv["dev_type"], "w": f"{dv['W_um']:g}",
               "l": f"{dv['L_um']:g}", "vdd": f"{dv['vdd_nominal']:g}", "part": part}
        for t in TEMPS:
            fit = mani["fitted"][f"{tag}@{t}"]
            assert len(fit) == 11
            inc = sum(int(x.split("|")[5]) for x in fit)
            row[f"i{t}"] = f"{inc}/11"
            tot[part][t][0] += inc
            tot[part][t][1] += 11
        row["s300"] = stock[str(PRIMARY)][tag]
        rows.append(row)
    n_inc = 0
    for part in ("DEV", "TEST"):
        r = {"tag": f"{part} total ({len(split[part])} DUTs)", "part": part}
        for t in TEMPS:
            r[f"i{t}"] = f"{tot[part][t][0]}/{tot[part][t][1]}"
            n_inc += tot[part][t][0]
        rows.append(r)
    assert n_inc == mani["inclusion_summary"]["n_included"] == 946
    cols = [{"key": "tag", "head": "DUT", "md": "DUT", "kind": "text"},
            {"key": "cls", "head": "class", "md": "class", "kind": "text"},
            {"key": "typ", "head": "type", "md": "type", "kind": "text"},
            {"key": "w", "head": "W (um)", "md": "W (um)", "kind": "rtext"},
            {"key": "l", "head": "L (um)", "md": "L (um)", "kind": "rtext"},
            {"key": "vdd", "head": "VDD (V)", "md": "VDD (V)", "kind": "rtext"},
            {"key": "part", "head": "split", "md": "split", "kind": "text"},
            {"key": "i233", "head": "233 K", "md": "included 233 K", "kind": "rtext",
             "group": "included curves of the 11 Vb = 0 curves", "family": True},
            {"key": "i300", "head": "300 K", "md": "included 300 K", "kind": "rtext",
             "group": "included curves of the 11 Vb = 0 curves", "family": True},
            {"key": "i343", "head": "343 K", "md": "included 343 K", "kind": "rtext",
             "group": "included curves of the 11 Vb = 0 curves", "family": True},
            {"key": "i398", "head": "398 K", "md": "included 398 K", "kind": "rtext",
             "group": "included curves of the 11 Vb = 0 curves", "family": True},
            {"key": "s300", "head": "stock RRMS 300 K", "md": "stock card RRMS 300 K", "kind": "rrms"}]
    q, c, o = mani["quantization"], mani["compliance"], mani["off_floor"]
    notes = [
        f"Source: {mani['source']['repository']} {mani['source']['path']}, commit {mani['source']['commit'][:8]} "
        f"({mani['source']['license']}, {mani['source']['format']}); local copy not committed; parsed as text by "
        "cryoml.explore.i5_1_ihp_real.parse_mdm under python -I. 28 DUTs (LV 8 nMOS + 8 pMOS on the same 8 W/L, "
        "HV 6 nMOS + 6 pMOS), one DUT per geometry and type, 233/300/343/398 K (header TEMP -40/27/70/125 C).",
        "Fitted curves per DUT and temperature: the 11 Vb = 0 curves, Id-Vg at 3 Vd (idvg@d0..d2), Id-Vd at 5 Vg "
        "(dc_idvd, idvd@g0..g4) and at 3 near-threshold Vg (dc_idvd_vth, idvd@t0..t2). Body-bias curves are "
        "stored and not used. The dc_idvd / dc_idvd_vth gate biases are set per DUT and temperature by the "
        "measurement setup.",
        "Inclusion is frozen from the measured current only (pdk2 rule): trim the leading points up to the first "
        "|I| > 1e-10 A; include iff the trimmed mean |I| > 3e-7 A (Id-Vd) or > 2e-7 A (Id-Vg). "
        f"{mani['inclusion_summary']['n_included']} of {mani['inclusion_summary']['n_fitted_curves']} curves "
        f"included; 3 to 10 per DUT and temperature.",
        f"Up to {q['printed_significant_digits_max']} significant digits; smallest nonzero |Id| "
        f"{q['smallest_nonzero_abs_id_A']:.0e} A; largest |Id| is {c['max_abs_id_over_drain_compliance']:.3f} of "
        f"the {c['drain_SMU1_A']:g} A drain compliance; off-current floor {o['min_A']:.0e} to {o['max_A']:.1e} A "
        f"(median {o['median_A']:.1e} A).",
        f"Split {rel(SPLIT)} (seed {split['seed']}): LV by geometry (3 of 8 geometries, both polarities, to TEST), "
        "HV 2 of 6 per polarity to TEST; DEV 18 (LV 10, HV 8), TEST 10. Written before any fit; the fitting modes "
        "read only DEV. Stock card RRMS: NGSpice-46 + psp103.osdi, stock PDK card, no fit.",
    ]
    return rf.Table("01_table_i5_1_dataset", "I5-1 dataset: IHP SG13G2 measured MOSFETs, DUTs, geometries, "
                    "split and included curves per temperature", cols, rows, notes)


def table_stock_start(d: dict) -> rf.Table:
    st, start, res = d["stock"]["summary"], d["start"]["per_temperature"], d["res"]
    rows = []
    for t in TEMPS:
        pt = start[str(t)]
        r = res[t]
        c = r["comparisons"]["all"]["start_vs_C0_refit"]
        rows.append({"t": f"{t} K", "sd": st[f"{t}K|DEV|all"]["all_device_mean"],
                     "sdl": st[f"{t}K|DEV|lv"]["all_device_mean"], "sdh": st[f"{t}K|DEV|hv"]["all_device_mean"],
                     "sdm": st[f"{t}K|DEV|all"]["median"],
                     "stt": st[f"{t}K|TEST|all"]["all_device_mean"], "sta": st[f"{t}K|ALL|all"]["all_device_mean"],
                     "a": pt["start"]["all_device_mean"], "al": pt["start_lv"]["all_device_mean"],
                     "ah": pt["start_hv"]["all_device_mean"], "am": pt["start"]["median"],
                     "c0": r["summary"]["all"]["C0_refit"]["all_device_mean"],
                     "g": r_text(c), "w": f"{c['wins']}/18"})
    cols = [{"key": "t", "head": "T", "md": "temperature", "kind": "text"},
            {"key": "sd", "head": "DEV all", "md": "stock DEV all-device", "kind": "rrms",
             "group": "stock card (no fit)"},
            {"key": "sdl", "head": "DEV LV", "md": "stock DEV LV", "kind": "rrms", "group": "stock card (no fit)"},
            {"key": "sdh", "head": "DEV HV", "md": "stock DEV HV", "kind": "rrms", "group": "stock card (no fit)"},
            {"key": "sdm", "head": "DEV median", "md": "stock DEV median", "kind": "rrms",
             "group": "stock card (no fit)"},
            {"key": "stt", "head": "TEST all", "md": "stock TEST all-device", "kind": "rrms",
             "group": "stock card (no fit)"},
            {"key": "sta", "head": "all 28", "md": "stock all 28 DUTs", "kind": "rrms", "group": "stock card (no fit)"},
            {"key": "a", "head": "DEV all", "md": "starting card DEV all-device", "kind": "rrms",
             "group": "starting card (12-parameter fit)", "family": True},
            {"key": "al", "head": "DEV LV", "md": "starting card DEV LV", "kind": "rrms",
             "group": "starting card (12-parameter fit)", "family": True},
            {"key": "ah", "head": "DEV HV", "md": "starting card DEV HV", "kind": "rrms",
             "group": "starting card (12-parameter fit)", "family": True},
            {"key": "am", "head": "DEV median", "md": "starting card DEV median", "kind": "rrms",
             "group": "starting card (12-parameter fit)", "family": True},
            {"key": "c0", "head": "C0_refit", "md": "C0_refit DEV all-device", "kind": "rrms"},
            {"key": "g", "head": "GMR(start, C0_refit) [CI]", "md": "GMR(start, C0_refit) [95% CI]", "kind": "rtext"},
            {"key": "w", "head": "start < C0", "md": "DUTs with start < C0_refit", "kind": "rtext"}]
    o = d["start"]["options"]
    notes = [
        "Strict paper RRMS (per included curve RMSE over the trimmed points / trimmed mean |I_measured|, DUT value "
        "= mean over included curves) with the frozen measured inclusion; all-device = mean over DUTs (DEV 18: LV "
        "10, HV 8; TEST 10; all 28).",
        "Stock card: IHP PSP103 PDK .model block (cornerMOS{lv,hv}.lib mos_tt) as a top-level card, NGSpice-46 + "
        f"psp103.osdi, header temperature; matches the PDK subcircuit within {d['stock']['max_rel_diff_card_vs_pdk_subckt']:.1e} "
        "relative on all 28 x 4 x 11 curves.",
        f"Starting card: ngspice_fit.polish from the stock card on the 12-parameter pdk2 IHP wide set (VFBO, UO, CTL, "
        f"THESATO, AXO, RSW1, CFL, MUEW, THEMUO, DPHIBO, DPHIBL, THESATL), max_nfev {o['nfev']}, irls {o['irls']}, "
        f"diff_step {o['diff_step']}; one per DEV DUT and temperature (queue job I5_1_start, "
        f"{d['start']['runtime_s']:.0f} s). C0_refit starts from it (max_nfev 400, irls 2).",
        "The stock-card RRMS of HV nMOS W10/L0.5 is 1.60 at 300 K (near-threshold Id-Vd curves).",
    ]
    return rf.Table("02_table_i5_1_stock_start", "I5-1 stock-card and starting-card RRMS per temperature (no fit "
                    "vs 12-parameter PSP fit)", cols, rows, notes)


def table_decision(d: dict) -> rf.Table:
    dec, res = d["dec"], d["res"]
    rows = []
    for t in TEMPS:
        z = dec["primary"] if t == PRIMARY else dec["secondary"][f"{t}K"]
        s = res[t]["summary"]["all"]
        r1 = z["R1_detail"]
        row = {"t": f"{t} K" + (" (primary)" if t == PRIMARY else " (secondary)"),
               "r1": r_text(r1), "w": f"{r1['wins']}/{r1['n_used']}",
               "r1c": "met" if z["checks"]["R1_ci_high_lt_1"] else "not met",
               "r2": f"{s['M_bias']['all_device_mean']:.4f} vs {s['T1_inloop']['all_device_mean'] + 0.005:.4f}",
               "r2c": "met" if z["checks"]["R2_mean_le_T1_inloop_plus_0.005"] else "not met",
               "r3": f"{s['M_bias']['n_failed']}",
               "r3c": "met" if z["checks"]["R3_zero_failed"] else "not met",
               "rep": "replicated" if z["replication"] == "replicated" else "not replicated"}
        for x in BASE:
            b = z["baselines"][x]
            row[x] = f"{r_text(b)} {b['wins']}/18 ({'beaten' if b['beaten'] else 'not beaten'})"
        row["bv"] = z["baseline_verdict"].replace("matched by baseline(s)", "matched by")
        rows.append(row)
    cols = [{"key": "t", "head": "temperature", "md": "temperature", "kind": "text"},
            {"key": "r1", "head": "GMR(M_bias, C0_refit) [CI]", "md": "R1: GMR(M_bias, C0_refit) [95% CI]",
             "kind": "rtext", "group": "R1: CI upper < 1"},
            {"key": "w", "head": "M < C0", "md": "DUTs with M_bias < C0_refit", "kind": "rtext",
             "group": "R1: CI upper < 1"},
            {"key": "r1c", "head": "R1", "md": "R1", "kind": "rtext", "group": "R1: CI upper < 1"},
            {"key": "r2", "head": "M_bias vs T1 + 0.005", "md": "R2: M_bias mean vs T1_inloop mean + 0.005",
             "kind": "rtext", "group": "R2 and R3", "family": True},
            {"key": "r2c", "head": "R2", "md": "R2", "kind": "rtext", "group": "R2 and R3", "family": True},
            {"key": "r3", "head": "failed", "md": "R3: M_bias failed DUTs", "kind": "rtext", "group": "R2 and R3",
             "family": True},
            {"key": "r3c", "head": "R3", "md": "R3", "kind": "rtext", "group": "R2 and R3", "family": True},
            {"key": "rep", "head": "replication", "md": "replication rule output", "kind": "text"},
            {"key": "B1", "head": "B1", "md": "GMR(M_bias, B1) [95% CI] wins", "kind": "rtext",
             "group": "baselines: GMR(M_bias, X) [CI], M_bias < X; beaten if CI upper < 1"},
            {"key": "B2", "head": "B2", "md": "GMR(M_bias, B2) [95% CI] wins", "kind": "rtext",
             "group": "baselines: GMR(M_bias, X) [CI], M_bias < X; beaten if CI upper < 1"},
            {"key": "B3_d3", "head": "B3_d3", "md": "GMR(M_bias, B3_d3) [95% CI] wins", "kind": "rtext",
             "group": "baselines: GMR(M_bias, X) [CI], M_bias < X; beaten if CI upper < 1"},
            {"key": "bv", "head": "baseline rule output", "md": "baseline rule output", "kind": "text"}]
    notes = [
        f"Pre-registration {rel(PREREG)} (commit 9267a90, committed before any card fit). Decision at 300 K over the "
        "18 DEV DUTs: 'replicated' if R1 (CI upper bound of GMR(M_bias, C0_refit) < 1), R2 (M_bias all-device mean "
        "<= T1_inloop mean + 0.005) and R3 (0 failed M_bias DUTs) all hold. Baselines B1, B2, B3_d3: X is beaten if "
        "the CI upper bound of GMR(M_bias, X) < 1; 'supported against baselines' if replicated and every X is "
        "beaten, otherwise 'matched by' the X not beaten.",
        "Rule output at 300 K: not replicated (R1 fails); baseline rule output: matched by B1, B2. 233, 343 and "
        "398 K are secondary (reported, not deciding); R1 fails at each of them.",
        "GMR(A, B) = exp(mean ln(RRMS_A / RRMS_B)) over the 18 DEV DUTs; device bootstrap, 10000 resamples, "
        "default_rng(0); wins = DUTs with A < B. Recomputed here from the stored per-DUT values and checked "
        f"against {rel(DEC)}.",
        "DEV values are development evidence (the rule, budgets, box and arms were fixed after the stock-card RRMS "
        "was seen and before any fit). TEST (10 DUTs) is not loaded by any I5-1 fitting mode.",
    ]
    return rf.Table("03_table_i5_1_decision", "I5-1 decision: replication rule R1-R3 and baseline rule per "
                    "temperature (300 K decides)", cols, rows, notes)


def table_results(d: dict) -> rf.Table:
    rows = []
    for t in TEMPS:
        r = d["res"][t]
        per, cmp = r["per_device"], r["comparisons"]["all"]
        for arm in ARMS:
            s = stats(per[arm])
            row = {"t": f"{t} K", "arm": arm, "desc": ARM_DESC[arm], "all": s["all"], "n": s["nmos"],
                   "p": s["pmos"], "med": s["median"], "fail": f"{s['failed']}/18"}
            if arm != "M_bias":
                c = cmp[f"M_bias_vs_{arm}"]
                row.update({"gm": r_text(c), "wm": f"{c['wins']}/18"})
            if arm != "C0_refit":
                c = cmp[f"{arm}_vs_C0_refit"]
                row.update({"gc": r_text(c), "wc": f"{c['wins']}/18"})
            rows.append(row)
    cols = [{"key": "t", "head": "T", "md": "temperature", "kind": "text"},
            {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
            {"key": "desc", "head": "definition", "md": "definition", "kind": "text"},
            {"key": "all", "head": "all-device", "md": "all-device", "kind": "rrms"},
            {"key": "n", "head": "nMOS", "md": "nMOS", "kind": "rrms", "family": True},
            {"key": "p", "head": "pMOS", "md": "pMOS", "kind": "rrms", "family": True},
            {"key": "med", "head": "median", "md": "median", "kind": "rrms"},
            {"key": "fail", "head": "failed", "md": "failed", "kind": "rtext"},
            {"key": "gm", "head": "GMR(M_bias, X) [CI]", "md": "GMR(M_bias, X) [95% CI]", "kind": "rtext"},
            {"key": "wm", "head": "M_bias < X", "md": "DUTs with M_bias < X", "kind": "rtext"},
            {"key": "gc", "head": "GMR(X, C0_refit) [CI]", "md": "GMR(X, C0_refit) [95% CI]", "kind": "rtext"},
            {"key": "wc", "head": "X < C0", "md": "DUTs with X < C0_refit", "kind": "rtext"}]
    r3 = d["res"][PRIMARY]
    b2 = list(r3["rows"]["B2"].values())
    frac = np.array([x["rext_ohm"] / x["rext_max_ohm"] for x in b2])
    kap = np.array([x["heat_kappa"] for x in b2])
    b1 = list(r3["rows"]["B1"].values())
    sc = d["screen"]
    notes = [
        "Per temperature, leave-one-DUT-out over the 18 DEV DUTs (9 nMOS, 9 pMOS), corrections trained within class "
        "(LV: 9 training DUTs, HV: 7), VDD 1.2 V (LV) / 3.3 V (HV). Every held-out arm refits the held-out DUT's "
        "card from its starting card on the 12-parameter box (max_nfev 400, irls 2, diff_step 0.01) and scores a "
        "fresh NGSpice-46 simulation of the refit card, multiplied by exp(g) for M_bias, T1_inloop, B3_d3 and B5.",
        "all-device = mean over 18 DUTs; nMOS and pMOS family means have 9 DUTs each, so combined = all-device. "
        "start = the starting card without a held-out refit.",
        f"B1 extras from the edge screen ({rel(SCREEN)}; response >= 1e-4 at a box edge on a 300 K starting card of "
        f"the class): LV {sc['lv']['n_selected']} and HV {sc['hv']['n_selected']} of {len(sc['lv']['candidates'])} PSP "
        f"candidates; {min(x['n_extra_active'] for x in b1)}-{max(x['n_extra_active'] for x in b1)} active per DUT "
        "(nonzero stock value). B1 and B2 max_nfev scaled with the parameter count (same TRF evaluations per IRLS "
        f"round as the base box). B2 at 300 K: fitted R_ext / box upper median {np.median(frac):.4f} "
        f"[{frac.min():.1e}, {frac.max():.3f}], kappa median {np.median(kap):+.4f}.",
        "B5 is trained on the held-out DUT's own measured curves and is an in-sample reference, not a held-out "
        "method. Queue jobs I5_1_lodo_300 (07:59-08:07), _233 (08:07-08:15), _343 (08:15-08:23), _398 "
        f"(08:23-08:30), 2026-10-20; {r3['timing']['refit_n']} held-out refits per temperature, about "
        f"{r3['runtime_s']:.0f} s each.",
    ]
    return rf.Table("04_table_i5_1_results", "I5-1 results per temperature for every arm, with GMR against M_bias "
                    "and against C0_refit (18 DEV DUTs)", cols, rows, notes)


def table_class(d: dict) -> rf.Table:
    rows = []
    for t in TEMPS:
        r = d["res"][t]
        per = r["per_device"]
        for cls in ("lv", "hv"):
            cmp = r["comparisons"][cls]
            n = len(sub(per["M_bias"], cls))
            row = {"t": f"{t} K", "cls": f"{cls.upper()} ({n})"}
            for arm in ("C0_refit", "M_bias", "T1_inloop", "B1", "B2", "B3_d3", "B5"):
                row[arm] = stats(sub(per[arm], cls))["all"]
            for key, name in (("g0", "M_bias_vs_C0_refit"), ("g1", "B1_vs_C0_refit"), ("g2", "B2_vs_C0_refit"),
                              ("gb1", "M_bias_vs_B1"), ("gb2", "M_bias_vs_B2")):
                c = cmp[name]
                row[key] = f"{r_text(c)} {c['wins']}/{n}"
            rows.append(row)
    g = "class mean RRMS"
    cols = [{"key": "t", "head": "T", "md": "temperature", "kind": "text"},
            {"key": "cls", "head": "class (DUTs)", "md": "class (DUTs)", "kind": "text"}] + [
        {"key": a, "head": a, "md": a, "kind": "rrms", "group": g} for a in
        ("C0_refit", "M_bias", "T1_inloop", "B1", "B2", "B3_d3", "B5")] + [
        {"key": "g0", "head": "GMR(M_bias, C0_refit)", "md": "GMR(M_bias, C0_refit) [95% CI] wins", "kind": "rtext",
         "group": "GMR [95% CI], wins (A < B)", "family": True},
        {"key": "g1", "head": "GMR(B1, C0_refit)", "md": "GMR(B1, C0_refit) [95% CI] wins", "kind": "rtext",
         "group": "GMR [95% CI], wins (A < B)", "family": True},
        {"key": "g2", "head": "GMR(B2, C0_refit)", "md": "GMR(B2, C0_refit) [95% CI] wins", "kind": "rtext",
         "group": "GMR [95% CI], wins (A < B)", "family": True},
        {"key": "gb1", "head": "GMR(M_bias, B1)", "md": "GMR(M_bias, B1) [95% CI] wins", "kind": "rtext",
         "group": "GMR [95% CI], wins (A < B)", "family": True},
        {"key": "gb2", "head": "GMR(M_bias, B2)", "md": "GMR(M_bias, B2) [95% CI] wins", "kind": "rtext",
         "group": "GMR [95% CI], wins (A < B)", "family": True}]
    notes = [
        "Per-class statistics are pre-registered as secondary (reported, not deciding). LV: sg13_lv, 1.2 V, 5 nMOS "
        "+ 5 pMOS DEV DUTs; HV: sg13_hv, 3.3 V, 4 nMOS + 4 pMOS DEV DUTs. Corrections are trained within class, so "
        "the LV and HV rows come from separate fold models.",
        "GMR with a device bootstrap within the class (10000 resamples, default_rng(0)); recomputed here and checked "
        f"against the per_class entries of {rel(DEC)}.",
    ]
    return rf.Table("05_table_i5_1_class", "I5-1 LV / HV class split per temperature: class means and paired "
                    "ratios", cols, rows, notes)


def table_cross(d: dict) -> rf.Table:
    rows = []
    for t in (233, 343, 398):
        ct = d["cross"]["per_temperature"][str(t)]
        per = ct["per_device"]
        for cls in ("all", "lv", "hv"):
            pa = {a: sub(per[a], cls) for a in ("X300", "C0_refit", "M_bias")}
            n = len(pa["X300"])
            row = {"t": f"{t} K", "cls": f"{cls if cls == 'all' else cls.upper()} ({n})",
                   "x": stats(pa["X300"])["all"], "c0": stats(pa["C0_refit"])["all"],
                   "m": stats(pa["M_bias"])["all"]}
            for key, a, b in (("gc", "X300", "C0_refit"), ("gm", "X300", "M_bias")):
                c = ct["comparisons"][f"{a}_vs_{b}"] if cls == "all" else paired_ratio(pa[a], pa[b])
                row[key] = r_text(c)
                row[key + "w"] = f"{c['wins']}/{n}"
            rows.append(row)
    cols = [{"key": "t", "head": "T", "md": "temperature", "kind": "text"},
            {"key": "cls", "head": "DUTs", "md": "DUTs", "kind": "text"},
            {"key": "x", "head": "X300", "md": "X300 mean", "kind": "rrms", "group": "mean RRMS"},
            {"key": "c0", "head": "C0_refit", "md": "C0_refit mean", "kind": "rrms", "group": "mean RRMS"},
            {"key": "m", "head": "M_bias (at T)", "md": "M_bias (trained at T) mean", "kind": "rrms",
             "group": "mean RRMS"},
            {"key": "gc", "head": "GMR(X300, C0_refit) [CI]", "md": "GMR(X300, C0_refit) [95% CI]", "kind": "rtext",
             "family": True},
            {"key": "gcw", "head": "X300 < C0", "md": "DUTs with X300 < C0_refit", "kind": "rtext", "family": True},
            {"key": "gm", "head": "GMR(X300, M_bias) [CI]", "md": "GMR(X300, M_bias) [95% CI]", "kind": "rtext"},
            {"key": "gmw", "head": "X300 < M_bias", "md": "DUTs with X300 < M_bias", "kind": "rtext"}]
    notes = [
        "X300 (pre-registered, secondary): the fold M_bias trained on the 300 K training records of the class, "
        "applied unchanged at T; the held-out card is refit at T with it in the loop (same refit settings). M_bias "
        "(at T) is the fold correction trained on the training records at T (the lodo arm).",
        f"'all' rows: stored comparisons ({rel(CROSS)}, checked against {rel(DEC)}); LV and HV rows: computed here "
        "from the stored per-DUT values with the same device bootstrap (10000 resamples, default_rng(0)).",
        f"Queue job I5_1_cross 2026-10-20 08:30-08:32 ({d['cross']['runtime_s']:.0f} s).",
    ]
    return rf.Table("09_table_i5_1_cross_x300", "I5-1 cross-temperature arm X300: 300 K correction applied at 233, "
                    "343 and 398 K", cols, rows, notes)


def gap_row(label: str, r: dict, per_key: str | None, n: int, params: str) -> dict:
    """r: a results file with summary/comparisons; per_key 'all' for I5-1 layout."""
    if per_key:
        s = {a: r["summary"][per_key][a]["all_device_mean"] for a in ("C0_refit", "M_bias", "T1_inloop", "B1",
                                                                       "B2", "B3_d3", "B5")}
        cmp = r["comparisons"][per_key]
    else:
        s = {a: r["summary"][a]["all_device_mean"] for a in ("C0_refit", "M_bias", "T1_inloop", "B1", "B2",
                                                              "B3_d3", "B5")}
        cmp = r["comparisons"]
    c, b5 = cmp["M_bias_vs_C0_refit"], cmp["B5_vs_C0_refit"]
    return {"ds": label, "n": str(n), "par": params, **s,
            "g": f"{r_text(c)} {c['wins']}/{n}",
            "g1": f"{cmp['B1_vs_C0_refit']['ratio']:.3f}", "g2": f"{cmp['B2_vs_C0_refit']['ratio']:.3f}",
            "fm": f"{(s['C0_refit'] - s['M_bias']) / (s['C0_refit'] - s['B5']):+.3f}",
            "fl": f"{np.log(c['ratio']) / np.log(b5['ratio']):+.3f}"}


def table_compare(d: dict) -> tuple[rf.Table, dict, dict]:
    rows = [gap_row("sky130 77 K REAL_DEV (measured, I4-1)", d["sky"], None, 12, "43 BSIM4"),
            gap_row(f"IHP SG13G2 {PRIMARY} K DEV (measured, I5-1 primary)", d["res"][PRIMARY], "all", 18, "12 PSP")]
    for t in (233, 343, 398):
        rows.append(gap_row(f"IHP SG13G2 {t} K DEV (measured, I5-1 secondary)", d["res"][t], "all", 18, "12 PSP"))
    rows.append(gap_row("IHP SG13G2 tier c (simulated, I4-1)", d["ihp_sim"], None, 48, "12 PSP"))
    cols = [{"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "n", "head": "n", "md": "devices / targets", "kind": "rtext"},
            {"key": "par", "head": "card parameters", "md": "refit parameters", "kind": "rtext"}] + [
        {"key": a, "head": a, "md": a, "kind": "rrms", "group": "all-device / all-target mean RRMS"}
        for a in ("C0_refit", "M_bias", "T1_inloop", "B1", "B2", "B3_d3", "B5")] + [
        {"key": "g", "head": "GMR(M_bias, C0_refit) [CI] wins", "md": "GMR(M_bias, C0_refit) [95% CI] wins",
         "kind": "rtext", "family": True},
        {"key": "g1", "head": "B1/C0", "md": "GMR(B1, C0_refit)", "kind": "rtext", "family": True},
        {"key": "g2", "head": "B2/C0", "md": "GMR(B2, C0_refit)", "kind": "rtext", "family": True},
        {"key": "fm", "head": "of mean", "md": "fraction of C0->B5 gap closed by M_bias (means)", "kind": "rtext",
         "group": "C0->B5 gap closed by M_bias"},
        {"key": "fl", "head": "of log GMR", "md": "fraction of C0->B5 gap closed by M_bias (log GMR)",
         "kind": "rtext", "group": "C0->B5 gap closed by M_bias"}]
    sky = d["sky"]["summary"]["C0_refit"]["all_device_mean"]
    ihp = d["res"][PRIMARY]["summary"]["all"]["C0_refit"]["all_device_mean"]
    notes = [
        "sky130 77 K: 12 REAL_DEV devices (5 nMOS, 7 pMOS), BSIM4 card, ngspice-41, 43-parameter held-out refit from "
        f"the direct43_r3_fd card ({rel(SKY)}). IHP: 18 DEV DUTs, PSP103 card, NGSpice-46 + psp103.osdi, "
        "12-parameter held-out refit from the starting card. Both use the I3-2 M_bias recipe unchanged and the same "
        "refit settings (max_nfev 400, irls 2). The sky130 RRMS freezes inclusion from the published-card baseline; "
        "the IHP RRMS freezes inclusion from the measured current (pdk2 rule).",
        f"C0_refit level: sky130 {sky:.4f}, IHP 300 K {ihp:.4f} (ratio {sky / ihp:.1f}).",
        "fraction of mean gap = (C0_refit - M_bias) / (C0_refit - B5); fraction of log GMR = ln GMR(M_bias, "
        "C0_refit) / ln GMR(B5, C0_refit). A negative value means M_bias is above C0_refit.",
        "B1 and B2 differ between datasets: sky130 B1 has 19 screened BSIM4 extras, IHP measured B1 70-73 PSP "
        "extras, IHP tier c B1 71 PSP extras. The IHP tier c truths contain an extrinsic resistance of the B2 form "
        "(I4-1 pre-registration).",
    ]
    return rf.Table("10_table_i5_1_sky130_vs_ihp", "I5-1 comparison: sky130 77 K REAL_DEV (I3-2 / I4-1) vs IHP "
                    "SG13G2 measured DEV per temperature, same M_bias recipe", cols, rows, notes), rows[0], rows[1]


# ================================================================== figures

def fig_forest(d: dict, path: Path) -> list[str]:
    left = ("C0_refit", "T1_inloop", "B1", "B2", "B3_d3")
    right = ("M_bias", "T1_inloop", "B1", "B2", "B3_d3", "B5")
    fig, axs = plt.subplots(len(TEMPS), 2, figsize=(17, 13), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for i, t in enumerate(TEMPS):
        cmp = d["res"][t]["comparisons"]["all"]
        for j, (arms, fmt_name) in enumerate(((left, "M_bias_vs_{}"), (right, "{}_vs_C0_refit"))):
            ax = axs[i, j]
            style(ax)
            ys = np.arange(len(arms))[::-1]
            for y, x in zip(ys, arms):
                c = cmp[fmt_name.format(x)]
                col = ARM_COLOR[x]
                lw = 2.6 if (j == 0 and x == "C0_refit") or (j == 1 and x == "M_bias") else 1.8
                ax.plot([c["ci_low"], c["ci_high"]], [y, y], color=col, lw=lw)
                ax.scatter([c["ratio"]], [y], color=col, s=48, edgecolor=BG, zorder=4, marker=ARM_MARK.get(x, "o"))
                ax.text(1.01, y, f"{r_text(c)}  {c['wins']}/18", transform=ax.get_yaxis_transform(), fontsize=8,
                        va="center", color=INK)
            ax.axvline(1.0, color=INK, ls="--", lw=0.9)
            ax.set_xscale("log")
            if j == 0:
                ax.set_xlim(0.6, 2.4)
                ticks = [0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0]
                ax.set_yticklabels([])
                ax.set_yticks(ys)
                ax.set_yticklabels([f"X = {x}" + (" (R1)" if x == "C0_refit" else
                                                  " (baseline)" if x in BASE else " (not in rule)") for x in arms],
                                   fontsize=8.5)
            else:
                ax.set_xlim(0.15, 2.4)
                ticks = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
                ax.set_yticks(ys)
                ax.set_yticklabels([f"X = {x}" + (" (in-sample)" if x == "B5" else "") for x in arms], fontsize=8.5)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{v:g}" for v in ticks])
            ax.minorticks_off()
            ax.set_ylim(-0.7, len(arms) - 0.3)
            role = "primary, decides" if t == PRIMARY else "secondary"
            what = "GMR(M_bias, X)" if j == 0 else "GMR(X, C0_refit)"
            ax.set_title(f"{t} K ({role}): {what}", fontsize=9.5, color=INK, loc="left")
    axs[-1, 0].set_xlabel("GMR(M_bias, X), 95% device-bootstrap CI (log axis); < 1: M_bias lower; "
                          "rule: CI upper < 1", fontsize=8.5)
    axs[-1, 1].set_xlabel("GMR(X, C0_refit), 95% device-bootstrap CI (log axis); < 1: X lower than the "
                          "no-correction refit", fontsize=8.5)
    fig.suptitle("I5-1 IHP SG13G2 measured, 18 DEV DUTs, leave-one-DUT-out: held-out strict paper RRMS after the "
                 "card refit\nR1 at 300 K: GMR(M_bias, C0_refit) CI upper < 1 not met (not replicated); "
                 "baseline rule: matched by B1, B2", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.12, right=0.86, top=0.93, bottom=0.06, hspace=0.42, wspace=0.75)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO[t]) for t in TEMPS] + [rel(DEC)]


def fig_per_device(d: dict, path: Path) -> list[str]:
    r = d["res"][PRIMARY]
    per = r["per_device"]
    tags = sorted(per["M_bias"], key=dev_key)
    arms = ("start", "C0_refit", "B1", "B2", "B3_d3", "T1_inloop", "M_bias", "B5")
    fig, ax = plt.subplots(figsize=(14, 5.8), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    off = np.linspace(-0.33, 0.33, len(arms))
    for j, a in enumerate(arms):
        ax.scatter(np.arange(len(tags)) + off[j], [per[a][t] for t in tags], color=ARM_COLOR[a],
                   marker=ARM_MARK.get(a, "o"), s=26 if a != "start" else 22, zorder=3,
                   label=f"{a} (all-device {stats(per[a])['all']:.4f})")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(tags)))
    ax.set_xticklabels(tags, rotation=35, ha="right", fontsize=8)
    for i in range(len(tags) - 1):
        ax.axvline(i + 0.5, color="#e4e3df", lw=0.6)
    nlv = sum(t.startswith("lv_") for t in tags)
    ax.axvline(nlv - 0.5, color=MUTED, lw=1.2)
    ax.text(nlv / 2 - 0.5, 1.02, "LV (1.2 V)", transform=ax.get_xaxis_transform(), ha="center", fontsize=9,
            color=MUTED)
    ax.text(nlv + (len(tags) - nlv) / 2 - 0.5, 1.02, "HV (3.3 V)", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=9, color=MUTED)
    ax.set_ylabel("held-out strict paper RRMS (log scale)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.3))
    ax.set_title(f"I5-1 IHP SG13G2 measured, {PRIMARY} K, per DEV DUT: held-out RRMS of every arm after the card "
                 "refit (start: starting card, no held-out refit; B5: in-sample)", fontsize=10, color=INK, loc="left",
                 pad=18)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO[PRIMARY])]


def fig_ratio_by_temperature(d: dict, path: Path) -> list[str]:
    tags = sorted(d["split"]["DEV"], key=dev_key)
    arms = ("M_bias", "B1", "B2")
    fig, axs = plt.subplots(len(arms), 1, figsize=(14, 10.5), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(BG)
    off = {t: o for t, o in zip(TEMPS, np.linspace(-0.24, 0.24, len(TEMPS)))}
    for ax, a in zip(axs, arms):
        style(ax)
        for t in TEMPS:
            per = d["res"][t]["per_device"]
            v = [per[a][k] / per["C0_refit"][k] for k in tags]
            c = d["res"][t]["comparisons"]["all"][f"{a}_vs_C0_refit"]
            ax.scatter(np.arange(len(tags)) + off[t], v, color=T_COLOR[t], s=24, zorder=3,
                       marker="D" if t == PRIMARY else "o",
                       label=f"{t} K: GMR {c['ratio']:.3f}, {c['wins']}/18 below 1")
        ax.axhline(1.0, color=INK, ls="--", lw=0.9)
        ax.set_yscale("log")
        for i in range(len(tags) - 1):
            ax.axvline(i + 0.5, color="#e4e3df", lw=0.6)
        nlv = sum(t.startswith("lv_") for t in tags)
        ax.axvline(nlv - 0.5, color=MUTED, lw=1.2)
        ax.set_ylabel(f"RRMS({a}) / RRMS(C0_refit)", fontsize=9)
        vals = [d["res"][t]["per_device"][x][k] / d["res"][t]["per_device"]["C0_refit"][k]
                for t in TEMPS for x in arms for k in tags]
        ax.set_ylim(min(vals) / 1.15, max(vals) * 1.15)
        ax.set_yticks([0.25, 0.5, 1, 2, 4, 8])
        ax.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8"])
        ax.minorticks_off()
        ax.legend(fontsize=8, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
        ax.set_title(f"{a} vs C0_refit per DEV DUT and temperature (< 1: {a} lower)", fontsize=9.5, color=INK,
                     loc="left", pad=22)
    axs[-1].set_xticks(np.arange(len(tags)))
    axs[-1].set_xticklabels(tags, rotation=35, ha="right", fontsize=8)
    fig.suptitle("I5-1: per-DUT held-out RRMS ratio to the no-correction refit, M_bias, B1 and B2, at 233, 300, 343 "
                 "and 398 K (LV left of the dark line, HV right)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO[t]) for t in TEMPS]


def fig_iv(tag: str, d: dict, path: Path) -> list[str]:
    """Stock-card figure (top, when stored) and refit-card figure (bottom), both from stored PNGs."""
    import matplotlib.image as mpimg
    refit = FIGS / f"iv_{tag}_{PRIMARY}K.png"
    stock = STOCK_FIGS / f"stock_{tag}_{PRIMARY}K.png"
    per = d["res"][PRIMARY]["per_device"]
    st = d["stock"]["per_device"][str(PRIMARY)][tag]
    imgs = [(stock, "a. measured points and the NGSpice-46 PSP103 stock PDK card (no fit), "
                    f"stock RRMS {st:.3f}")] if stock.exists() else []
    imgs.append((refit, "b. measured points; dashed: NGSpice-46 C0_refit card (held-out refit, no correction), "
                        f"RRMS {per['C0_refit'][tag]:.4f}; solid: NGSpice-46 M_bias refit card x exp(g) of the frozen "
                        f"fold correction, RRMS {per['M_bias'][tag]:.4f}"))
    if len(imgs) == 1:
        imgs = [(refit, imgs[0][1][3:])]
    arrs = [mpimg.imread(p) for p, _ in imgs]
    w = max(a.shape[1] for a in arrs)
    hs = [a.shape[0] for a in arrs]
    head = 45
    fig = plt.figure(figsize=(w / 110, (sum(hs) + head * len(arrs) + 50) / 110), dpi=110)
    fig.patch.set_facecolor("white")
    total = sum(hs) + head * len(arrs) + 50
    y = total - 50
    fig.text(0.01, 1 - 15 / total, f"I5-1 {tag}, {PRIMARY} K (DEV): I-V from stored figures"
             + ("" if stock.exists() else " (no stored stock-card figure for this DUT)"),
             fontsize=12, fontweight="bold", va="top")
    for (p, lab), a in zip(imgs, arrs):
        y -= head
        fig.text(0.01, (y + head * 0.45) / total, lab, fontsize=9.5, va="center")
        ax = fig.add_axes([0, (y - a.shape[0]) / total, a.shape[1] / w, a.shape[0] / total])
        ax.imshow(a)
        ax.axis("off")
        y -= a.shape[0]
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return [rel(p) for p, _ in imgs] + [rel(LODO[PRIMARY])]


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)

    d = load_all()
    items, tables = [], []

    def add_table(t: rf.Table, desc: str, srcs: list[str]) -> None:
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    add_table(table_dataset(d), "Table of the IHP SG13G2 dataset: 28 DUTs with class, type, W, L, VDD, DEV/TEST "
              "split, included curves of the 11 Vb = 0 curves at 233/300/343/398 K and stock-card RRMS at 300 K; "
              "DEV and TEST totals", [rel(MANI), rel(SPLIT), rel(STOCK)])
    add_table(table_stock_start(d), "Table of stock-card (no fit) and starting-card (12-parameter PSP fit) RRMS per "
              "temperature: DEV all/LV/HV/median, TEST and all 28 for the stock card, C0_refit and GMR(start, "
              "C0_refit)", [rel(STOCK), rel(START)] + [rel(LODO[t]) for t in TEMPS])
    add_table(table_decision(d), "Table of the I5-1 rule per temperature: R1 GMR(M_bias, C0_refit) with CI and wins, "
              "R2, R3, replication output, GMR(M_bias, X) for B1, B2, B3_d3 and the baseline rule output; 300 K "
              "decides (not replicated; matched by B1, B2)", [rel(PREREG), rel(DEC)] + [rel(LODO[t]) for t in TEMPS])
    add_table(table_results(d), "Table of the results per temperature for every arm (start, C0_refit, M_bias, "
              "T1_inloop, B1, B2, B3_d3, B5): all-device, nMOS, pMOS, median, failed, GMR(M_bias, X) and GMR(X, "
              "C0_refit) with CI and wins", [rel(LODO[t]) for t in TEMPS] + [rel(SCREEN)])
    add_table(table_class(d), "Table of the LV / HV class split per temperature: class means of every held-out arm "
              "and GMR(M_bias, C0_refit), GMR(B1, C0_refit), GMR(B2, C0_refit), GMR(M_bias, B1), GMR(M_bias, B2) "
              "with CI and wins", [rel(LODO[t]) for t in TEMPS] + [rel(DEC)])
    src = fig_forest(d, folder / "06_i5_1_gmr_forest.png")
    items.append(("06_i5_1_gmr_forest.png", "Forest plot per temperature: left GMR(M_bias, X) for X in C0_refit "
                  "(R1), T1_inloop, B1, B2, B3_d3; right GMR(X, C0_refit) for M_bias, T1_inloop, B1, B2, B3_d3, B5; "
                  "95% device-bootstrap CI and wins", src))
    src = fig_per_device(d, folder / "07_i5_1_per_device_300K.png")
    items.append(("07_i5_1_per_device_300K.png", "Per-DUT held-out RRMS at 300 K for start, C0_refit, B1, B2, B3_d3, "
                  "T1_inloop, M_bias and B5 (log scale), LV and HV DUTs", src))
    src = fig_ratio_by_temperature(d, folder / "08_i5_1_ratio_by_temperature.png")
    items.append(("08_i5_1_ratio_by_temperature.png", "Per-DUT ratio RRMS(X) / RRMS(C0_refit) for X = M_bias, B1, B2 "
                  "at 233, 300, 343 and 398 K (log scale)", src))
    add_table(table_cross(d), "Table of the cross-temperature arm X300 (300 K fold correction applied at 233, 343, "
              "398 K): X300, C0_refit and M_bias means, GMR(X300, C0_refit) and GMR(X300, M_bias) with CI and wins, "
              "all DUTs and per class", [rel(CROSS), rel(DEC)])
    tc, sky_row, ihp_row = table_compare(d)
    add_table(tc, "Table comparing sky130 77 K REAL_DEV (I3-2/I4-1) with IHP SG13G2 measured DEV at each temperature "
              "and the simulated IHP tier c: C0_refit, M_bias, T1_inloop, B1, B2, B3_d3, B5 means, GMR(M_bias, "
              "C0_refit), GMR(B1, C0_refit), GMR(B2, C0_refit), fraction of the C0_refit -> B5 gap closed by M_bias",
              [rel(SKY), rel(IHP_SIM)] + [rel(LODO[t]) for t in TEMPS])
    for j, tag in enumerate(IV_DUTS):
        name = f"{11 + j:02d}_i5_1_iv_{tag}_{PRIMARY}K.png"
        src = fig_iv(tag, d, folder / name)
        has_stock = (STOCK_FIGS / f"stock_{tag}_{PRIMARY}K.png").exists()
        items.append((name, f"I-V of {tag} at {PRIMARY} K, assembled from stored figures: "
                      + ("(a) measured points and the NGSpice-46 stock PDK card; (b) " if has_stock else
                         "no stored stock-card figure for this DUT; ")
                      + "measured points, NGSpice-46 C0_refit card (dashed) and NGSpice-46 M_bias refit card "
                      "multiplied by exp(g) of the frozen fold correction (solid)", src))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i5_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "",
          f"Generated {now} by {gen} from stored files only. RRMS is the strict paper metric with the inclusion "
          "frozen from the measured current (pdk2 rule). I5-1 values are on the 18 IHP SG13G2 DEV DUTs "
          "(development evidence); TEST is not loaded.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    res = d["res"]
    s3 = res[PRIMARY]["summary"]["all"]
    c3 = res[PRIMARY]["comparisons"]["all"]
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen} from stored files only (no NGSpice, no training).", "",
              "## Contents", "",
              "- I5-1: the I3-2 bias-only correction M_bias (recipe unchanged) on IHP SG13G2 measured MOSFETs "
              "(IHP-Open-PDK `ihp-sg13g2/libs.doc/meas/MOS`, commit 5e6d592e, PSP103), 18 DEV DUTs, 233/300/343/398 "
              f"K, leave-one-DUT-out within class. Pre-registration `{rel(PREREG)}` (commit 9267a90); code, dataset "
              "manifest, split, stock-card RRMS, smoke runs commit c58c087; outcome log commit 8010a61. Queue jobs "
              "I5_1_start (2026-10-20 07:52-07:59), I5_1_screen (07:59), I5_1_lodo_300 (07:59-08:07), "
              "I5_1_lodo_233 (08:07-08:15), I5_1_lodo_343 (08:15-08:23), I5_1_lodo_398 (08:23-08:30), I5_1_cross "
              "(08:30-08:32), I5_1_decide and I5_1_figs (08:32).",
              f"- Rule output at {PRIMARY} K: {d['dec']['primary']['replication']}; baseline rule output: "
              f"{d['dec']['primary']['baseline_verdict']}. GMR(M_bias, C0_refit) {r_text(c3['M_bias_vs_C0_refit'])}, "
              f"{c3['M_bias_vs_C0_refit']['wins']}/18 DUTs with M_bias < C0_refit. All-device means: C0_refit "
              f"{s3['C0_refit']['all_device_mean']:.4f}, M_bias {s3['M_bias']['all_device_mean']:.4f}, T1_inloop "
              f"{s3['T1_inloop']['all_device_mean']:.4f}, B1 {s3['B1']['all_device_mean']:.4f}, B2 "
              f"{s3['B2']['all_device_mean']:.4f}, B3_d3 {s3['B3_d3']['all_device_mean']:.4f}, B5 (in-sample) "
              f"{s3['B5']['all_device_mean']:.4f}. 233, 343 and 398 K (secondary): not replicated.",
              "- Simulator: NGSpice-46 (Homebrew) + psp103.osdi (OpenVAF-Reloaded), the toolchain recorded for the "
              "IHP PSP103 model; the sky130 work uses ngspice-41.",
              "- I-V figures 11-14 are assembled from stored PNGs (no simulation in this batch). "
              "`out/research/explore/I5_1/figs/iv_*_300K.png` show measured points, the C0_refit card and the "
              "M_bias refit card x exp(g); they do not contain the stock-card curve. The stock-card curve comes from "
              "`out/research/explore/I5_1/stock/figs/stock_*_300K.png`, stored for 6 DEV DUTs; 3 of the 4 DUTs with "
              "a refit figure have one. hv_pmos_W10_L0p35 has no stored stock-card figure, and figure 14 shows "
              "the refit panel only. The M_bias curve is the NGSpice-46 current of the M_bias refit card multiplied "
              "by exp(g) of the frozen fold correction (applied in Python), not a SPICE-native simulation of the "
              "correction. B1 and B2 refit cards are not plotted.",
              "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    sky = d["sky"]
    summary = [
        f"I5-1 rule output ({PRIMARY} K, 18 IHP SG13G2 DEV DUTs): {d['dec']['primary']['replication']}; "
        f"baseline rule output: {d['dec']['primary']['baseline_verdict']}. GMR(M_bias, C0_refit) "
        f"{r_text(c3['M_bias_vs_C0_refit'])}, {c3['M_bias_vs_C0_refit']['wins']}/18. All-device: C0_refit "
        f"{s3['C0_refit']['all_device_mean']:.4f}, M_bias {s3['M_bias']['all_device_mean']:.4f}, T1_inloop "
        f"{s3['T1_inloop']['all_device_mean']:.4f}, B1 {s3['B1']['all_device_mean']:.4f}, B2 "
        f"{s3['B2']['all_device_mean']:.4f}, B3_d3 {s3['B3_d3']['all_device_mean']:.4f}, B5 "
        f"{s3['B5']['all_device_mean']:.4f}.",
        "Secondary temperatures, GMR(M_bias, C0_refit): " + "; ".join(
            f"{t} K {r_text(res[t]['comparisons']['all']['M_bias_vs_C0_refit'])}" for t in (233, 343, 398))
        + "; not replicated at each.",
        f"sky130 77 K REAL_DEV vs IHP 300 K: C0_refit {sky['summary']['C0_refit']['all_device_mean']:.4f} vs "
        f"{s3['C0_refit']['all_device_mean']:.4f}; GMR(M_bias, C0_refit) {sky['comparisons']['M_bias_vs_C0_refit']['ratio']:.3f} "
        f"vs {c3['M_bias_vs_C0_refit']['ratio']:.3f}; fraction of the C0_refit -> B5 gap closed (means / log GMR) "
        f"{sky_row['fm']} / {sky_row['fl']} vs {ihp_row['fm']} / {ihp_row['fl']}.",
        "X300 (300 K correction at T), GMR(X300, C0_refit): " + "; ".join(
            f"{t} K {r_text(d['cross']['per_temperature'][str(t)]['comparisons']['X300_vs_C0_refit'])}"
            for t in (233, 343, 398)) + ".",
    ]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
