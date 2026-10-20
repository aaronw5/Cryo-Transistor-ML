#!/usr/bin/env python3
"""Report batch for exploration direction I6-1: the I3-2 bias-only correction
M_bias on TU Delft 40-nm CMOS measured at 4.13 K and 295.0 K with a public
BSIM4 card (ngspice-41 CMOSedu 50 nm example card), leave-one-geometry-out over
45 DEV devices in 7 (polarity, W, L) groups.

Reads stored files only (no NGSpice, no training):

* ``out/research/explore/I6_1/{prereg.json, prereg_amendment_2026-10-20.json,
  split.json, box_trial.json, decision.json}``, ``data/manifest.json``,
  ``card/``, ``stock/stock_rrms.json``, ``start/summary.json``,
  ``start/b1_screen.json``, ``lodo_{4,295}K/results.json``, the I-V PNGs in
  ``figs/`` and ``stock/figs/``.
* ``out/research/explore/I4_1/real_dev/results.json`` (sky130 77 K REAL_DEV)
  and ``out/research/explore/I5_1/{decision.json, lodo_*K/results.json}``
  (IHP SG13G2) for the cross-dataset table.

Every stored aggregate (all devices and per group), paired ratio (device and
cluster bootstrap), the temperature contrast K, the premise P and the
pre-registered outcome label are recomputed from the stored per-device values
and checked against the stored values.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i6_1.py --date 2026-10-20
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA,  # noqa: E402
                                   VIOLET, RED, DPI, EXP, load, rel, style)
from cryoml.explore.i3_2_bias import paired_ratio  # noqa: E402
import cryoml.explore.i6_1_tudelft as m6  # noqa: E402  (card text, box rule; no simulation)

rf = pe.rf
plt = pe.plt

TOPIC = "i6_1"
I6 = EXP / "I6_1"
PREREG = I6 / "prereg.json"
AMEND = I6 / "prereg_amendment_2026-10-20.json"
SPLIT = I6 / "split.json"
DEC = I6 / "decision.json"
MANI = I6 / "data/manifest.json"
CARD = I6 / "card/ngspice41_ro_17_4_cmosedu50nm_bsim4.mod"
BOX = I6 / "box_trial.json"
STOCK = I6 / "stock/stock_rrms.json"
START = I6 / "start/summary.json"
SCREEN = I6 / "start/b1_screen.json"
FIGS = I6 / "figs"
STOCK_FIGS = I6 / "stock/figs"
TEMPS = (4, 295)
PRIMARY = 4
LODO = {t: I6 / f"lodo_{t}K/results.json" for t in TEMPS}
SKY = EXP / "I4_1/real_dev/results.json"
I5_DEC = EXP / "I5_1/decision.json"
I5_TEMPS = (233, 300, 343, 398)
I5_LODO = {t: EXP / f"I5_1/lodo_{t}K/results.json" for t in I5_TEMPS}

ARMS = ("start", "C0_refit", "M_bias", "T1_inloop", "B1", "B2", "B3_d3", "B5")
BASE = ("B1", "B2", "B3_d3")
ARM_DESC = {"start": "starting card (43-set fit from the stock card), no held-out refit",
            "C0_refit": "control: held-out card refit, no correction",
            "M_bias": "I3-2 correction, refit with g in the loop (method under test)",
            "T1_inloop": "per-(polarity, curve key) gain table (not deployable)",
            "B1": "wider BSIM4 parameter set (base + 16 screened extras), no correction",
            "B2": "extrinsic series resistance + self-heating factor, no learned term",
            "B3_d3": "shared polynomial g, total degree 3 (20 coefficients)",
            "B5": "per-device correction trained on the held-out device (in-sample reference)"}
ARM_COLOR = {"start": "#c9c7c0", "C0_refit": GREY, "M_bias": BLUE, "T1_inloop": INK, "B1": AQUA,
             "B2": VIOLET, "B3_d3": RED, "B5": "#7a9a3a"}
ARM_MARK = {"M_bias": "D", "B5": "v", "C0_refit": "s", "T1_inloop": "^", "start": "x"}
GROUP_COLOR = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3"]
IV_DEVS = ("t40_nmos_W0p12_L0p04_d152", "t40_nmos_W1_L0p04_d192", "t40_nmos_W1_L1_d128",
           "t40_pmos_W1_L0p04_d392", "t40_pmos_W1_L1_d328")
OUTCOMES = ("refuted", "inconclusive", "matched by baselines", "supported", "supported, premise not met",
            "temperature-graded", "not temperature-specific")


def r_text(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def cl_text(c: dict) -> str:
    b = c["cluster_bootstrap"]
    return f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}]"


def check_ratio(stored: dict, mine: dict, what: str) -> None:
    for k in ("ratio", "ci_low", "ci_high"):
        assert abs(stored[k] - mine[k]) < 1e-12, (what, k, stored[k], mine[k])
    assert stored["wins"] == mine["wins"], what


def pol(tag: str) -> str:
    return "nmos" if "_nmos_" in tag else "pmos"


def group_of(tag: str) -> str:
    return tag.rsplit("_d", 1)[0]


def stats(per: dict) -> dict:
    """all-device, nMOS, pMOS, combined, median, failed over a {tag: rrms} dict."""
    v = {t: x for t, x in per.items() if np.isfinite(x)}
    n = [x for t, x in v.items() if pol(t) == "nmos"]
    p = [x for t, x in v.items() if pol(t) == "pmos"]
    fam = [float(np.mean(f)) for f in (n, p) if f]
    return {"all": float(np.mean(list(v.values()))), "nmos": float(np.mean(n)) if n else float("nan"),
            "pmos": float(np.mean(p)) if p else float("nan"),
            "combined": float(np.mean(fam)) if len(fam) == 2 else float("nan"),
            "median": float(np.median(list(v.values()))), "failed": len(per) - len(v), "n": len(per)}


def sub(per: dict, group: str) -> dict:
    return per if group == "all" else {t: x for t, x in per.items() if group_of(t) == group}


def tlabel(t: int) -> str:
    return "4.13 K" if t == 4 else "295.0 K"


def glabel(g: str, mani: dict) -> str:
    x = mani["groups"][g]
    return f"{'nMOS' if x['dev_type'] == 'nmos' else 'pMOS'} {x['W_um']:g}/{x['L_um']:g}"


def gap(c0: float, mb: float, b5: float, g_m: float, g_b5: float) -> tuple[float, float]:
    """fraction of the C0_refit -> B5 gap closed by M_bias: means, log GMR."""
    return (c0 - mb) / (c0 - b5), float(np.log(g_m) / np.log(g_b5))


# ================================================================== outcome logic

def conditions(r: dict) -> dict:
    s, c = r["summary"]["all"], r["comparisons"]
    out = {"R1": c["M_bias_vs_C0_refit"]["ci_high"] < 1,
           "R2": s["M_bias"]["all_device_mean"] <= s["T1_inloop"]["all_device_mean"] + 0.005,
           "R3": s["M_bias"]["n_failed"] == 0}
    out["beaten"] = {x: c[f"M_bias_vs_{x}"]["ci_high"] < 1 for x in BASE}
    out["all_beaten"] = all(out["beaten"].values())
    return out


def outcome_table(c4: dict, c295: dict, k_holds: bool, premise: bool) -> list[tuple[str, str, bool]]:
    """Each pre-registered outcome with its condition evaluated on its own; the
    rule output is the first one in list order whose condition holds."""
    r123 = c4["R1"] and c4["R2"] and c4["R3"]
    full4 = r123 and c4["all_beaten"]
    return [
        ("refuted", "R1 fails at 4 K", not c4["R1"]),
        ("inconclusive", "R1 holds at 4 K, R2 or R3 fails at 4 K", c4["R1"] and not (c4["R2"] and c4["R3"])),
        ("matched by baselines", "R1-R3 hold at 4 K, at least one of B1, B2, B3_d3 not beaten at 4 K",
         r123 and not c4["all_beaten"]),
        ("supported", "R1-R3 hold and all baselines beaten at 4 K; R1 fails at 295 K; P holds",
         full4 and not c295["R1"] and premise),
        ("supported, premise not met", "as 'supported' but P fails", full4 and not c295["R1"] and not premise),
        ("temperature-graded", "R1-R3 hold and all baselines beaten at 4 K; R1 holds at 295 K; K holds",
         full4 and c295["R1"] and k_holds),
        ("not temperature-specific", "R1-R3 hold and all baselines beaten at 4 K; R1 holds at 295 K; K fails",
         full4 and c295["R1"] and not k_holds),
    ]


# ================================================================== loading and checks

def load_all() -> dict:
    dec, split, mani = load(DEC), load(SPLIT), load(MANI)
    assert not dec["smoke"]
    dev = split["DEV"]
    assert len(dev) == 45 and len(split["TEST"]) == 24 and split["n_dev"] == 45
    groups = split["DEV_groups"]
    clusters = {t: group_of(t) for t in dev}
    assert sorted(set(clusters.values())) == sorted(groups) and len(groups) == 7
    res = {t: load(LODO[t]) for t in TEMPS}
    for t, r in res.items():
        assert r["T_K"] == t and not r["smoke"] and r["preregistered_config"], t
        assert r["hyper_overrides"] == {}, t
        per = r["per_device"]
        assert sorted(per["M_bias"]) == sorted(dev), t          # DEV only
        assert sorted(per) == sorted(ARMS), t
        for g in ["all"] + groups:
            for arm in ARMS:
                s, mine = r["summary"][g][arm], stats(sub(per[arm], g))
                assert abs(s["all_device_mean"] - mine["all"]) < 1e-12, (t, g, arm)
                assert abs(s["median"] - mine["median"]) < 1e-12, (t, g, arm)
                assert s["n_failed"] == mine["failed"] == 0, (t, g, arm)
                if g == "all":
                    assert abs(s["combined"] - mine["combined"]) < 1e-12, (t, arm)
        for name, st in r["comparisons"].items():
            a, b = name.split("_vs_")
            check_ratio(st, paired_ratio(per[a], per[b]), f"{t} {name}")
            cl = paired_ratio(per[a], per[b], clusters=clusters)
            check_ratio(st["cluster_bootstrap"] | {"ratio": cl["ratio"], "wins": cl["wins"]}, cl,
                        f"{t} cluster {name}")
            assert st["cluster_bootstrap"]["resample_unit"] == "cluster (7)"
        for arm in ARMS[1:]:
            for tag, row in r["rows"][arm].items():
                assert abs(row["rrms"] - per[arm][tag]) < 1e-15, (t, arm, tag)
    # decision recomputed
    cond = {t: conditions(res[t]) for t in TEMPS}
    for t in TEMPS:
        d, r = dec[f"at_{t}K"], res[t]
        c = r["comparisons"]
        check_ratio(d["R1_detail"], c["M_bias_vs_C0_refit"], f"R1 {t}")
        assert d["R1_detail"]["cluster_bootstrap"] == c["M_bias_vs_C0_refit"]["cluster_bootstrap"]
        assert d["checks"] == {"R1_ci_high_lt_1": cond[t]["R1"], "R2_mean_le_T1_inloop_plus_0.005": cond[t]["R2"],
                               "R3_zero_failed": cond[t]["R3"]}, t
        for x in BASE:
            check_ratio(d["baselines"][x], c[f"M_bias_vs_{x}"], f"{t} baseline {x}")
            assert d["baselines"][x]["beaten"] == cond[t]["beaten"][x], (t, x)
        assert d["all_baselines_beaten"] == cond[t]["all_beaten"]
        for arm, mm in dec["means"][f"{t}K"].items():
            assert abs(mm["all_device_mean"] - r["summary"]["all"][arm]["all_device_mean"]) < 1e-15
        for g, row in dec["per_group"][f"{t}K"].items():
            for arm, v in row.items():
                assert abs(v - r["summary"][g][arm]["all_device_mean"]) < 1e-15, (t, g, arm)
    p4, p295 = res[4]["per_device"], res[295]["per_device"]
    q4 = {k: p4["M_bias"][k] / p4["C0_refit"][k] for k in dev}
    q295 = {k: p295["M_bias"][k] / p295["C0_refit"][k] for k in dev}
    kk = paired_ratio(q4, q295)
    kc = paired_ratio(q4, q295, clusters=clusters)
    check_ratio(dec["contrast_GMR_4K_over_295K"], kk, "contrast K")
    check_ratio(dec["contrast_GMR_4K_over_295K"]["cluster_bootstrap"] | {"ratio": kc["ratio"], "wins": kc["wins"]},
                kc, "contrast K cluster")
    k_holds = kk["ci_high"] < 1
    c0m = {t: res[t]["summary"]["all"]["C0_refit"]["all_device_mean"] for t in TEMPS}
    premise = c0m[4] >= 1.5 * c0m[295]
    assert dec["premise"]["holds"] == premise and dec["premise"]["factor"] == 1.5
    outs = outcome_table(cond[4], cond[295], k_holds, premise)
    assert [o[0] for o in outs] == list(OUTCOMES)
    label = next(o[0] for o in outs if o[2])
    assert dec["outcome"] == label == "refuted", label
    # stock and start
    stock, start = load(STOCK), load(START)
    assert not start["smoke"]
    for t in TEMPS:
        pt = start["per_temperature"][str(t)]
        sr = {k: v["start_rrms"] for k, v in pt["rows"].items()}
        assert sorted(sr) == sorted(dev)
        assert all(abs(sr[k] - res[t]["per_device"]["start"][k]) < 1e-15 for k in dev), t
        assert all(abs(pt["rows"][k]["stock_rrms"] - stock["per_device"][str(t)][k]) < 1e-12 for k in dev), t
        assert abs(pt["start"]["all_device_mean"] - np.mean(list(sr.values()))) < 1e-12, t
        assert abs(stock["summary"][f"{t}K"]["all_device_mean"] - np.mean(list(stock["per_device"][str(t)].values()))) \
            < 1e-12
        for g in groups:
            assert abs(pt["start_per_group"][g]["all_device_mean"] - stats(sub(sr, g))["all"]) < 1e-12
    # manifest inclusion
    inc = 0
    for key, rows in mani["fitted"].items():
        assert len(rows) == 23, key
        inc += sum(int(x.split("|")[5]) for x in rows)
    assert inc == mani["inclusion_summary"]["n_included"] == 3161
    screen = load(SCREEN)
    for t in TEMPS:
        assert sorted(screen[f"{t}K"]["selected"]) == sorted(res[t]["b1_names"]), t
    sky = load(SKY)
    assert not sky["smoke"] and sky["preregistered_config"]
    i5 = {t: load(I5_LODO[t]) for t in I5_TEMPS}
    for t, r in i5.items():
        assert r["T_K"] == t and not r["smoke"] and r["preregistered_config"]
    return {"dec": dec, "split": split, "mani": mani, "res": res, "stock": stock, "start": start,
            "screen": screen, "prereg": load(PREREG), "amend": load(AMEND), "box": load(BOX), "sky": sky,
            "i5": i5, "i5dec": load(I5_DEC), "cond": cond, "K": kk, "Kc": kc, "k_holds": k_holds,
            "premise": premise, "c0m": c0m, "outcomes": outs, "label": label, "groups": groups,
            "clusters": clusters, "q4": q4, "q295": q295}


# ================================================================== tables

def table_dataset(d: dict) -> rf.Table:
    mani, split, stock = d["mani"], d["split"], d["stock"]
    rows = []
    tot = {p: {t: [0, 0, 0] for t in TEMPS} for p in ("DEV", "TEST")}
    order = d["groups"] + split["TEST_groups"]
    for g in order:
        x = mani["groups"][g]
        part = "DEV" if g in split["DEV_groups"] else "TEST"
        devs = [v for v in mani["devices"] if v["group"] == g]
        assert [v["device"] for v in devs] == x["devices"]
        row = {"g": g, "typ": x["dev_type"], "w": f"{x['W_um']:g}", "l": f"{x['L_um']:g}",
               "n": str(len(devs)), "die": str(x["sample"]), "blk": x["sub_block"],
               "ids": f"{min(x['devices'])}-{max(x['devices'])}", "part": part}
        for t in TEMPS:
            inc = sum(int(r.split("|")[5]) for v in devs for r in mani["fitted"][f"{v['tag']}@{t}"])
            row[f"i{t}"] = f"{inc}/{23 * len(devs)}"
            tot[part][t][0] += inc
            tot[part][t][1] += 23 * len(devs)
            tot[part][t][2] += len(devs)
            row[f"s{t}"] = stock["per_group"][f"{t}K"][g]["all_device_mean"] if part == "DEV" else None
        rows.append(row)
    for part in ("DEV", "TEST"):
        r = {"g": f"{part} total ({tot[part][4][2]} devices)", "part": part}
        for t in TEMPS:
            r[f"i{t}"] = f"{tot[part][t][0]}/{tot[part][t][1]}"
        if part == "DEV":
            for t in TEMPS:
                r[f"s{t}"] = stock["summary"][f"{t}K"]["all_device_mean"]
        rows.append(r)
    cols = [{"key": "g", "head": "group", "md": "group (polarity, W, L)", "kind": "text"},
            {"key": "typ", "head": "type", "md": "type", "kind": "text"},
            {"key": "w", "head": "W (um)", "md": "W (um)", "kind": "rtext"},
            {"key": "l", "head": "L (um)", "md": "L (um)", "kind": "rtext"},
            {"key": "n", "head": "copies", "md": "copies", "kind": "rtext"},
            {"key": "ids", "head": "device ids", "md": "device ids", "kind": "rtext"},
            {"key": "die", "head": "die", "md": "die (sample)", "kind": "rtext"},
            {"key": "blk", "head": "array", "md": "array (sub-block)", "kind": "text"},
            {"key": "part", "head": "split", "md": "split", "kind": "text"},
            {"key": "i4", "head": "4.13 K", "md": "included curves 4.13 K", "kind": "rtext",
             "group": "included curves (23 per device)", "family": True},
            {"key": "i295", "head": "295.0 K", "md": "included curves 295.0 K", "kind": "rtext",
             "group": "included curves (23 per device)", "family": True},
            {"key": "s4", "head": "4.13 K", "md": "stock card RRMS 4.13 K (group mean)", "kind": "rrms",
             "group": "stock card RRMS, group mean"},
            {"key": "s295", "head": "295.0 K", "md": "stock card RRMS 295.0 K (group mean)", "kind": "rrms",
             "group": "stock card RRMS, group mean"}]
    src, q, c, st = mani["source"], mani["quantization"], mani["compliance"], mani["stage_temperature_K"]
    bg = mani["bias_grid"]
    notes = [
        f"Source: 4TU.ResearchData doi:{src['doi']} ({src['license']}), {src['title']}; {src['files']}; local copy "
        "not committed; read by cryoml.explore.i6_1_tudelft.read_npy_dict (unpickler restricted to numpy "
        "reconstruction globals) under python -I.",
        f"Selection: campaign 1, {mani['n_devices']} low-Vt core devices (NF = 1) with a campaign-1 file at both file "
        "temperatures; nMOS on die 3, pMOS on die 5; 'array' is the dataset's sub-block (LC low-current, HC "
        "high-current). Stage temperature (mean of the file's start and end readings): label 4 = "
        f"{st['4'][0]:.2f}-{st['4'][1]:.2f} K, label 295 = {st['295'][0]:.2f}-{st['295'][1]:.2f} K; deck "
        "temperature = stage temperature of the file.",
        "Curves per device and temperature: 12 Id-Vg (idvg@d0..d11; Vd 0.05 V and 11 values to 1.1 V) and 11 Id-Vd "
        "(idvd@g0..g10; Vd 0-1.1 V), 111 points each. Vg sweep start: nMOS "
        f"{bg['nmos@4K']['vg_sweep'][0]:g} V (4.13 K), {bg['nmos@295K']['vg_sweep'][0]:g} V (295 K); pMOS "
        f"{bg['pmos@4K']['vg_sweep'][0]:g} V, {bg['pmos@295K']['vg_sweep'][0]:g} V. No deep subthreshold data.",
        "Inclusion frozen from the measured current only (pdk2 rule): trim the leading points up to the first |I| > "
        "1e-10 A; include iff the trimmed mean |I| > 3e-7 A (Id-Vd) or > 2e-7 A (Id-Vg). "
        f"{mani['inclusion_summary']['n_included']} of {mani['inclusion_summary']['n_fitted_curves']} curves "
        "included, 22-23 per device and temperature.",
        f"Float64 currents; smallest nonzero |Id| {q['smallest_nonzero_abs_id_A']:.2e} A; no exact zeros; "
        f"{q['n_points_with_sign_opposite_to_the_bias']} points near Vd = 0 with sign opposite to the bias; no "
        f"compliance stored; largest |Id| {c['largest_abs_id_A'] * 1e3:.3f} mA; no curve repeats its maximum.",
        f"Split {rel(SPLIT)} (seed {split['seed']}, by W/L pair; written before any fit): TEST = W/L 4/4 (both "
        "polarities) and nMOS 1/0.36, 24 devices; DEV 45 devices in 7 groups. TEST curves are not loaded or "
        "simulated by any I6-1 mode. Stock card RRMS: DEV only (no fit), from "
        f"{rel(STOCK)}.",
    ]
    return rf.Table("01_table_i6_1_dataset", "I6-1 dataset: TU Delft 40-nm CMOS, campaign 1, device groups, split, "
                    "included curves per temperature and stock-card RRMS", cols, rows, notes)


def table_card_box(d: dict) -> rf.Table:
    boxes = {p: m6.base_box(SimpleNamespace(dev_type=p)) for p in ("nmos", "pmos")}
    ab = {t: {} for t in TEMPS}
    for t in TEMPS:
        for row in d["start"]["per_temperature"][str(t)]["rows"].values():
            for n in row["at_bound"]:
                ab[t][n] = ab[t].get(n, 0) + 1
    n_act = {p: int(b.n_active) for p, b in boxes.items()}
    assert n_act == {"nmos": 36, "pmos": 37}, n_act
    rows = []

    def box_text(b, n: str, i: int) -> str:
        lo, hi = b.bounds[n]
        if not b.active[i]:
            return "inactive"
        if n in m6.BASE_OFFSET:
            return f"{b.published[n]:+.3g} +/- {m6.BASE_OFFSET[n]:g} V"
        if b.log_scale[i]:
            return f"log [{lo:.3g}, {hi:.3g}]"
        return f"lin [{lo:.3g}, {hi:.3g}]"

    for i, n in enumerate(m6.BASE_SET):
        row = {"n": n.upper()}
        for p, b in boxes.items():
            row[f"v{p}"] = f"{b.published[n]:.4g}"
            row[f"b{p}"] = box_text(b, n, i)
        for t in TEMPS:
            row[f"a{t}"] = str(ab[t].get(n, 0))
        rows.append(row)
    cols = [{"key": "n", "head": "parameter", "md": "parameter", "kind": "text"},
            {"key": "vnmos", "head": "stock", "md": "nMOS stock value", "kind": "rtext", "group": "nMOS (N1)"},
            {"key": "bnmos", "head": "box", "md": "nMOS box", "kind": "rtext", "group": "nMOS (N1)"},
            {"key": "vpmos", "head": "stock", "md": "pMOS stock value", "kind": "rtext", "group": "pMOS (P1)",
             "family": True},
            {"key": "bpmos", "head": "box", "md": "pMOS box", "kind": "rtext", "group": "pMOS (P1)",
             "family": True},
            {"key": "a4", "head": "4.13 K", "md": "starting fits at a bound, 4.13 K (of 45)", "kind": "rtext",
             "group": "starting fits at a bound"},
            {"key": "a295", "head": "295.0 K", "md": "starting fits at a bound, 295.0 K (of 45)", "kind": "rtext",
             "group": "starting fits at a bound"}]
    pr, am = d["prereg"]["compact_model"], d["amend"]["changes"]
    sc = d["screen"]
    cp = m6.card_params("nmos")
    notes = [
        f"Card {rel(CARD)}: BSIM4 level 54 version {cp['version']}, N1 (nMOS) and P1 (pMOS) copied verbatim from "
        "ngspice-41 examples/mos/ro_17_4.cir ('Short channel models from CMOS Circuit Design, Layout, and "
        "Simulation, 50nm BSIM4 models VDD=1V', CMOSedu.com, R. J. Baker); ngspice Modified BSD license. Not a card "
        f"of the measured 40-nm process. lint {float(cp['lint']) * 1e9:g} nm, no xl (Leff 16 nm at L = 40 nm), toxe "
        f"{float(cp['toxe']) * 1e9:g} nm, TNOM {cp['tnom']} C.",
        "Temperature: deck .options temp = stage temperature; kt1, kt1l, kt2, ute, ua1, ub1, uc1, at, prt set to 0, "
        "so the card values apply at the deck temperature and temperature enters only through the built-in kT/q, "
        "band-gap and intrinsic-carrier terms. One card is fitted per device and temperature. BSIM4 is not "
        "designed for 4.2 K.",
        "Box (cryoml.explore.i6_1_tudelft.base_box, the same for the starting fit and every held-out refit): VTH0, "
        "VFB stock +/- 0.5 V, VOFF +/- 0.3 V; other nonzero values log |v|/10..10|v|; zero values: B0, B1 [0, "
        "1e-7] m, CIT, CDSCD [0, 2.5e-3] F/m^2, else inactive. UB has a box but is inactive under the ParameterBox "
        f"rule (as in the sky130 43-set fits). Active: nMOS {n_act['nmos']}, pMOS {n_act['pmos']} (PRWG is zero "
        "in N1).",
        f"B1 (wider set): universe of {len(m6.b1_universe())} BSIM4 names after amendment {rel(AMEND)} "
        f"({am[0]['item']}: {am[0]['now']}); edge screen per temperature kept {sc['4K']['n_selected']} names at "
        f"4.13 K and {sc['295K']['n_selected']} at 295.0 K, the same set: {', '.join(sc['4K']['selected'])}. "
        f"Amendment 2: {am[1]['now'].split(';')[0]}.",
        "Simulator: ngspice-41 native arm64 build, one process per device and card, 23 sweeps on the measured grid, "
        "0.03 s per run. 'at a bound' counts DEV starting cards (45 per temperature, both polarities) with the "
        "parameter at a box edge after the starting fit.",
    ]
    return rf.Table("02_table_i6_1_card_box", "I6-1 card and box: CMOSedu 50-nm BSIM4 card, the 43-parameter set, "
                    "stock values, box per polarity and starting-fit bound hits", cols, rows, notes)


def table_box_trial(d: dict) -> rf.Table:
    rows = []
    for r in d["box"]["rows"]:
        h = r["history"]
        rows.append({"tag": r["tag"], "t": tlabel(r["T_K"]), "f": f"{r['box_factor']}",
                     "h0": h[0], "h1": h[1], "h2": h[2], "h3": h[3], "ns": f"{r['n_sim']}",
                     "ab": ", ".join(r["at_bound"]) or "none"})
    cols = [{"key": "tag", "head": "device", "md": "device", "kind": "text"},
            {"key": "t", "head": "T", "md": "temperature", "kind": "text"},
            {"key": "f", "head": "box factor", "md": "box factor", "kind": "rtext"},
            {"key": "h0", "head": "stock", "md": "RRMS stock card", "kind": "rrms", "group": "RRMS after IRLS round"},
            {"key": "h1", "head": "1", "md": "RRMS round 1", "kind": "rrms", "group": "RRMS after IRLS round"},
            {"key": "h2", "head": "2", "md": "RRMS round 2", "kind": "rrms", "group": "RRMS after IRLS round"},
            {"key": "h3", "head": "3", "md": "RRMS round 3", "kind": "rrms", "group": "RRMS after IRLS round"},
            {"key": "ns", "head": "simulations", "md": "simulations", "kind": "rtext"},
            {"key": "ab", "head": "at bound", "md": "parameters at a bound", "kind": "text"}]
    notes = [
        f"{rel(BOX)}: written before the pre-registration; ngspice_fit.polish from the stock card, max_nfev 1500, "
        "irls 3, diff_step 0.01, 3 DEV devices at both temperatures. Factor 2: log |v|/2..2|v|; U0, VSAT, RDSW "
        "|v|/4..4|v|; VTH0, VFB +/- 0.3 V, VOFF +/- 0.2 V. Factor 10: log |v|/10..10|v|; VTH0, VFB +/- 0.5 V, "
        "VOFF +/- 0.3 V.",
        "The factor-10 box was adopted for the starting cards and all refits. No correction was fitted in this "
        "trial.",
    ]
    return rf.Table("03_table_i6_1_box_trial", "I6-1 box-width trial before the pre-registration: starting-fit "
                    "RRMS per IRLS round, box factor 2 vs 10", cols, rows, notes)


def table_stock_start(d: dict) -> rf.Table:
    rows = []
    st, start, res, mani = d["stock"], d["start"]["per_temperature"], d["res"], d["mani"]
    for t in TEMPS:
        r = res[t]
        per = r["per_device"]
        for g in ["all"] + d["groups"]:
            sp = sub(st["per_device"][str(t)], g)
            row = {"t": tlabel(t), "g": "all DEV" if g == "all" else glabel(g, mani),
                   "n": str(len(sp)), "s": stats(sp)["all"], "sm": stats(sp)["median"],
                   "a": stats(sub(per["start"], g))["all"], "am": stats(sub(per["start"], g))["median"],
                   "c0": stats(sub(per["C0_refit"], g))["all"]}
            if g == "all":
                c = r["comparisons"]["start_vs_C0_refit"]
                row["gr"] = r_text(c)
                row["gw"] = f"{c['wins']}/45"
                row["s"] = st["summary"][f"{t}K"]["all_device_mean"]
                row["a"] = start[str(t)]["start"]["all_device_mean"]
            rows.append(row)
    cols = [{"key": "t", "head": "T", "md": "temperature", "kind": "text"},
            {"key": "g", "head": "group", "md": "group", "kind": "text"},
            {"key": "n", "head": "devices", "md": "devices", "kind": "rtext"},
            {"key": "s", "head": "mean", "md": "stock mean", "kind": "rrms", "group": "stock card (no fit)"},
            {"key": "sm", "head": "median", "md": "stock median", "kind": "rrms", "group": "stock card (no fit)"},
            {"key": "a", "head": "mean", "md": "starting card mean", "kind": "rrms",
             "group": "starting card (43-set fit)", "family": True},
            {"key": "am", "head": "median", "md": "starting card median", "kind": "rrms",
             "group": "starting card (43-set fit)", "family": True},
            {"key": "c0", "head": "C0_refit", "md": "C0_refit mean", "kind": "rrms"},
            {"key": "gr", "head": "GMR(start, C0_refit) [CI]", "md": "GMR(start, C0_refit) [95% CI]", "kind": "rtext"},
            {"key": "gw", "head": "start < C0", "md": "devices with start < C0_refit", "kind": "rtext"}]
    o = d["start"]["options"]
    sv = [x for t in TEMPS for x in st["per_device"][str(t)].values()]
    smin, smax = min(sv), max(sv)
    a4, a295 = (start[str(t)]["start"]["all_device_mean"] for t in TEMPS)
    notes = [
        "Strict paper RRMS (per included curve RMSE over the trimmed points / trimmed mean |I_measured|, device "
        "value = mean over included curves) with the frozen measured inclusion; group and all-DEV means over "
        "devices.",
        f"Starting card: ngspice_fit.polish from the stock card on the 43-parameter box, max_nfev {o['nfev']}, irls "
        f"{o['irls']}, diff_step {o['diff_step']}; one per DEV device and temperature (queue job I6_1_start, "
        f"{d['start']['runtime_s']:.0f} s, 0 simulator timeouts). C0_refit starts from it (max_nfev 400, irls 2).",
        f"The stock card is a generic 50-nm card; its RRMS is {smin:.2f}-{smax:.1f} per device and temperature "
        f"(threshold and current level of another process). The starting-card all-DEV mean at 4.13 K is "
        f"{a4 / a295:.2f} times the 295.0 K value.",
    ]
    return rf.Table("04_table_i6_1_stock_start", "I6-1 stock-card, starting-card and C0_refit RRMS per temperature "
                    "and DEV group", cols, rows, notes)


def table_decision(d: dict) -> rf.Table:
    dec, res = d["dec"], d["res"]
    rows = []
    for t in TEMPS:
        z = dec[f"at_{t}K"]
        s = res[t]["summary"]["all"]
        r1 = z["R1_detail"]
        row = {"t": tlabel(t) + (" (decides)" if t == PRIMARY else " (control)"),
               "r1": r_text(r1), "r1c": cl_text(r1), "w": f"{r1['wins']}/{r1['n_used']}",
               "r1v": "met" if z["checks"]["R1_ci_high_lt_1"] else "not met",
               "r2": f"{s['M_bias']['all_device_mean']:.4f} vs {s['T1_inloop']['all_device_mean'] + 0.005:.4f}",
               "r2v": "met" if z["checks"]["R2_mean_le_T1_inloop_plus_0.005"] else "not met",
               "r3": f"{s['M_bias']['n_failed']}",
               "r3v": "met" if z["checks"]["R3_zero_failed"] else "not met"}
        for x in BASE:
            b = z["baselines"][x]
            row[x] = f"{r_text(b)} {cl_text(b)} {b['wins']}/45 ({'beaten' if b['beaten'] else 'not beaten'})"
        rows.append(row)
    cols = [{"key": "t", "head": "temperature", "md": "temperature", "kind": "text"},
            {"key": "r1", "head": "GMR [device CI]", "md": "R1: GMR(M_bias, C0_refit) [device 95% CI]",
             "kind": "rtext", "group": "R1: GMR(M_bias, C0_refit), device CI upper < 1"},
            {"key": "r1c", "head": "cluster CI", "md": "cluster 95% CI (7 groups)", "kind": "rtext",
             "group": "R1: GMR(M_bias, C0_refit), device CI upper < 1"},
            {"key": "w", "head": "M < C0", "md": "devices with M_bias < C0_refit", "kind": "rtext",
             "group": "R1: GMR(M_bias, C0_refit), device CI upper < 1"},
            {"key": "r1v", "head": "R1", "md": "R1", "kind": "rtext",
             "group": "R1: GMR(M_bias, C0_refit), device CI upper < 1"},
            {"key": "r2", "head": "M_bias vs T1 + 0.005", "md": "R2: M_bias mean vs T1_inloop mean + 0.005",
             "kind": "rtext", "group": "R2 and R3", "family": True},
            {"key": "r2v", "head": "R2", "md": "R2", "kind": "rtext", "group": "R2 and R3", "family": True},
            {"key": "r3", "head": "failed", "md": "R3: M_bias failed devices", "kind": "rtext", "group": "R2 and R3",
             "family": True},
            {"key": "r3v", "head": "R3", "md": "R3", "kind": "rtext", "group": "R2 and R3", "family": True}] + [
        {"key": x, "head": x, "md": f"GMR(M_bias, {x}) [device CI] [cluster CI] wins", "kind": "rtext",
         "group": "baselines: GMR(M_bias, X) [device CI] [cluster CI] wins; beaten if device CI upper < 1"}
        for x in BASE]
    notes = [
        f"Pre-registration {rel(PREREG)} (commit a967f04, before any correction fit and before the starting-card "
        f"fits); amendment {rel(AMEND)} (after the smoke runs, before any full run): DVT0W removed from the B1 "
        "universe, 10 s simulator timeout.",
        "Conditions per temperature over the 45 DEV devices: R1 device-bootstrap CI upper of GMR(M_bias, C0_refit) "
        "< 1; R2 M_bias all-device mean <= T1_inloop mean + 0.005; R3 0 failed M_bias devices; baseline X beaten if "
        "the device-bootstrap CI upper of GMR(M_bias, X) < 1. The prediction: R1-R3 and all baselines beaten at "
        "4.13 K, R1 not met at 295.0 K.",
        "At 4.13 K R1 is not met (CI upper 1.004); R2 and R3 are met; B1 is beaten, B2 and B3_d3 are not. At 295.0 K "
        "R1, R2, R3 are met and every baseline is beaten, which is the opposite of the 295 K prediction.",
        "GMR(A, B) = exp(mean ln(RRMS_A / RRMS_B)) over the 45 DEV devices; device bootstrap 10000 resamples, "
        "default_rng(0); cluster bootstrap resamples the 7 (polarity, W, L) groups (reported, not deciding; the "
        "2-8 copies of one geometry are not independent). wins = devices with A < B. Recomputed here and checked "
        f"against {rel(DEC)}.",
    ]
    return rf.Table("05_table_i6_1_decision", "I6-1 decision conditions per temperature (4.13 K decides, 295.0 K is "
                    "the control)", cols, rows, notes)


def table_outcome(d: dict) -> rf.Table:
    k, kc = d["K"], d["Kc"]
    c0m = d["c0m"]
    rows = [{"item": "premise P", "cond": "C0_refit mean at 4.13 K >= 1.5 x C0_refit mean at 295.0 K",
             "val": f"{c0m[4]:.4f} vs 1.5 x {c0m[295]:.4f} = {1.5 * c0m[295]:.4f} (ratio {c0m[4] / c0m[295]:.2f})",
             "res": "holds" if d["premise"] else "fails", "sel": ""},
            {"item": "contrast K", "cond": "device CI upper of K < 1, K = GMR of (M_bias / C0_refit at 4.13 K) / "
             "(M_bias / C0_refit at 295.0 K)",
             "val": f"{r_text(k)}, cluster {cl_text(k | {'cluster_bootstrap': kc})}; {k['wins']}/45 devices with the "
             "4.13 K ratio below the 295.0 K ratio",
             "res": "holds" if d["k_holds"] else "fails", "sel": ""}]
    for lab, cond, ok in d["outcomes"]:
        rows.append({"item": f"outcome: {lab}", "cond": cond, "val": "", "res": "holds" if ok else "fails",
                     "sel": "rule output" if lab == d["label"] else ""})
    cols = [{"key": "item", "head": "item", "md": "item", "kind": "text"},
            {"key": "cond", "head": "pre-registered condition", "md": "pre-registered condition", "kind": "text"},
            {"key": "val", "head": "value", "md": "value", "kind": "text"},
            {"key": "res", "head": "evaluated", "md": "evaluated", "kind": "text"},
            {"key": "sel", "head": "selected", "md": "selected", "kind": "text"}]
    notes = [
        "The outcome list is evaluated in the pre-registered order and the first label whose condition holds is the "
        "rule output (scripts/research/explore/i6_1_tudelft.py, function outcome). Each condition is also evaluated "
        f"on its own here. Rule output: {d['label']} ({d['dec']['interpretation']}).",
        "Pre-registered interpretation of 'refuted': the correction does not lower held-out error even where BSIM4 "
        "is used outside its validity range; together with I5-1 the I3-2 gain is not explained by model-form error "
        "of this kind.",
        "At 295.0 K the correction lowers the held-out RRMS (R1 met). K > 1 means the M_bias / C0_refit ratio is "
        "larger at 4.13 K than at 295.0 K, so the correction lowers the error more at 295.0 K.",
    ]
    return rf.Table("06_table_i6_1_outcome", "I6-1 outcome logic: premise P, temperature contrast K and the "
                    "pre-registered outcome list", cols, rows, notes)


def table_results(d: dict) -> rf.Table:
    rows = []
    for t in TEMPS:
        r = d["res"][t]
        per, cmp = r["per_device"], r["comparisons"]
        for arm in ARMS:
            s = stats(per[arm])
            row = {"t": tlabel(t), "arm": arm, "desc": ARM_DESC[arm], "all": s["all"], "n": s["nmos"],
                   "p": s["pmos"], "comb": s["combined"], "med": s["median"], "fail": f"{s['failed']}/45"}
            if arm != "M_bias":
                c = cmp[f"M_bias_vs_{arm}"]
                row.update({"gm": r_text(c), "gmc": cl_text(c), "wm": f"{c['wins']}/45"})
            if arm != "C0_refit":
                c = cmp[f"{arm}_vs_C0_refit"]
                row.update({"gc": r_text(c), "gcc": cl_text(c), "wc": f"{c['wins']}/45"})
            rows.append(row)
    cols = [{"key": "t", "head": "T", "md": "temperature", "kind": "text"},
            {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
            {"key": "desc", "head": "definition", "md": "definition", "kind": "text"},
            {"key": "all", "head": "all-device", "md": "all-device", "kind": "rrms"},
            {"key": "n", "head": "nMOS", "md": "nMOS (29)", "kind": "rrms", "family": True},
            {"key": "p", "head": "pMOS", "md": "pMOS (16)", "kind": "rrms", "family": True},
            {"key": "comb", "head": "combined", "md": "combined", "kind": "rrms"},
            {"key": "med", "head": "median", "md": "median", "kind": "rrms"},
            {"key": "fail", "head": "failed", "md": "failed", "kind": "rtext"},
            {"key": "gm", "head": "GMR [device CI]", "md": "GMR(M_bias, X) [device 95% CI]", "kind": "rtext",
             "group": "GMR(M_bias, X)", "family": True},
            {"key": "gmc", "head": "cluster CI", "md": "GMR(M_bias, X) cluster 95% CI", "kind": "rtext",
             "group": "GMR(M_bias, X)", "family": True},
            {"key": "wm", "head": "M < X", "md": "devices with M_bias < X", "kind": "rtext",
             "group": "GMR(M_bias, X)", "family": True},
            {"key": "gc", "head": "GMR [device CI]", "md": "GMR(X, C0_refit) [device 95% CI]", "kind": "rtext",
             "group": "GMR(X, C0_refit)"},
            {"key": "gcc", "head": "cluster CI", "md": "GMR(X, C0_refit) cluster 95% CI", "kind": "rtext",
             "group": "GMR(X, C0_refit)"},
            {"key": "wc", "head": "X < C0", "md": "devices with X < C0_refit", "kind": "rtext",
             "group": "GMR(X, C0_refit)"}]
    rr = {t: d["res"][t]["rows"] for t in TEMPS}
    flat = {t: sum(1 for x in rr[t]["B1"].values() if abs(x["rrms"] - x["rrms_start"]) < 1e-15) for t in TEMPS}
    b1_above = {t: sum(d["res"][t]["per_device"]["B1"][k] > d["res"][t]["per_device"]["C0_refit"][k]
                       for k in d["split"]["DEV"]) for t in TEMPS}
    b2 = {t: list(rr[t]["B2"].values()) for t in TEMPS}
    frac = {t: np.array([x["rext_ohm"] / x["rext_max_ohm"] for x in b2[t]]) for t in TEMPS}
    kap = {t: np.array([x["heat_kappa"] for x in b2[t]]) for t in TEMPS}
    tim = {t: d["res"][t]["timing"] for t in TEMPS}
    notes = [
        "Per temperature, leave-one-geometry-out over the 7 DEV groups: corrections are trained on the DEV devices "
        "of the other 6 groups (both polarities, one network, VDD 1.1 V); every device of the held-out group is "
        "refit from its starting card on the 43-parameter box (max_nfev 400, irls 2, diff_step 0.01) and scored on "
        "a fresh ngspice-41 simulation of the refit card, multiplied by exp(g) for M_bias, T1_inloop, B3_d3 and B5.",
        "all-device = mean over 45 devices (29 nMOS, 16 pMOS); combined = mean of the nMOS and pMOS family means. "
        "start = the starting card without a held-out refit. 0 failed devices and 0 simulator timeouts in every "
        "arm.",
        f"B1: base box plus the 16 screened extras (max_nfev 540-550). The B1 refit returned the starting card "
        f"unchanged (no accepted step) for {flat[4]}/45 devices at 4.13 K and {flat[295]}/45 at 295.0 K; B1 is above "
        f"C0_refit on {b1_above[4]}/45 and {b1_above[295]}/45 devices.",
        f"B2: fitted R_ext / box upper median {np.median(frac[4]):.1e} (4.13 K) and {np.median(frac[295]):.1e} "
        f"(295.0 K), maximum {frac[4].max():.3f} / {frac[295].max():.3f}; kappa range [{kap[4].min():+.3f}, "
        f"{kap[4].max():+.3f}] / [{kap[295].min():+.3f}, {kap[295].max():+.3f}].",
        "B5 is trained on the held-out device's own measured curves and is an in-sample reference, not a held-out "
        f"method. Queue jobs I6_1_lodo_4 ({d['res'][4]['runtime_s']:.0f} s; training {tim[4]['train_s']:.0f} s, "
        f"{tim[4]['refit_n']} refits {tim[4]['refit_s']:.0f} s) and I6_1_lodo_295 ({d['res'][295]['runtime_s']:.0f} "
        f"s), 2026-10-20.",
    ]
    return rf.Table("07_table_i6_1_results", "I6-1 results per temperature for every arm (45 DEV devices), with GMR "
                    "against M_bias and against C0_refit, device and cluster CIs", cols, rows, notes)


def table_groups(d: dict) -> rf.Table:
    rows, mani = [], d["mani"]
    arms = ("C0_refit", "M_bias", "T1_inloop", "B1", "B2", "B3_d3", "B5")
    for t in TEMPS:
        per = d["res"][t]["per_device"]
        for g in d["groups"]:
            pm, pc = sub(per["M_bias"], g), sub(per["C0_refit"], g)
            row = {"t": tlabel(t), "g": glabel(g, mani), "n": str(len(pm))}
            for a in arms:
                row[a] = stats(sub(per[a], g))["all"]
            gm = float(np.exp(np.mean([np.log(pm[k] / pc[k]) for k in pm])))
            row["gr"] = f"{gm:.3f}"
            row["w"] = f"{sum(pm[k] < pc[k] for k in pm)}/{len(pm)}"
            bx = min(BASE, key=lambda x: row[x])
            row["best"] = f"{bx} {row[bx]:.4f}"
            rows.append(row)
    cols = [{"key": "t", "head": "T", "md": "temperature", "kind": "text"},
            {"key": "g", "head": "held-out group", "md": "held-out group", "kind": "text"},
            {"key": "n", "head": "devices", "md": "devices", "kind": "rtext"}] + [
        {"key": a, "head": a, "md": a, "kind": "rrms", "group": "group mean RRMS"} for a in arms] + [
        {"key": "gr", "head": "GMR(M_bias, C0)", "md": "GMR(M_bias, C0_refit) within group", "kind": "rtext",
         "family": True},
        {"key": "w", "head": "M < C0", "md": "devices with M_bias < C0_refit", "kind": "rtext", "family": True},
        {"key": "best", "head": "lowest of B1, B2, B3_d3", "md": "lowest of B1, B2, B3_d3", "kind": "rtext"}]
    above = {t: [r["g"] for r in rows if r["t"] == tlabel(t) and r["M_bias"] > r["C0_refit"]] for t in TEMPS}
    assert above == {4: ["pMOS 1/0.04", "pMOS 1/1"], 295: ["nMOS 1/1"]}, above
    notes = [
        "Each row is one fold: the held-out group's devices, with corrections trained on the other 6 DEV groups. "
        "Per-group means are reported, not deciding (pre-registration). The within-group GMR has no CI (2-8 copies "
        f"of one geometry). Means checked against {rel(DEC)} per_group.",
        "At 4.13 K M_bias is above C0_refit in both pMOS groups (1/0.04 and 1/1) and below it in the 5 nMOS groups; "
        "at 295.0 K M_bias is above C0_refit in the nMOS 1/1 group only.",
    ]
    return rf.Table("08_table_i6_1_groups", "I6-1 per-group (fold) means of every held-out arm and GMR(M_bias, "
                    "C0_refit) within the group", cols, rows, notes)


def table_contrast_gap(d: dict) -> rf.Table:
    rows = []
    for t in TEMPS:
        r = d["res"][t]
        s = {a: r["summary"]["all"][a]["all_device_mean"] for a in ARMS}
        cm, cb = r["comparisons"]["M_bias_vs_C0_refit"], r["comparisons"]["B5_vs_C0_refit"]
        fm, fl = gap(s["C0_refit"], s["M_bias"], s["B5"], cm["ratio"], cb["ratio"])
        rows.append({"t": tlabel(t), "c0": s["C0_refit"], "m": s["M_bias"], "b5": s["B5"],
                     "gm": f"{r_text(cm)} {cl_text(cm)}", "gb": f"{r_text(cb)} {cl_text(cb)}",
                     "fm": f"{fm:+.3f}", "fl": f"{fl:+.3f}"})
    k, kc = d["K"], d["Kc"]
    rows.append({"t": "K (4.13 K over 295.0 K)", "gm": f"{r_text(k)} [{kc['ci_low']:.3f}, {kc['ci_high']:.3f}]",
                 "gb": f"{k['wins']}/45 devices: 4.13 K ratio < 295.0 K ratio"})
    cols = [{"key": "t", "head": "temperature", "md": "temperature", "kind": "text"},
            {"key": "c0", "head": "C0_refit", "md": "C0_refit mean", "kind": "rrms", "group": "all-device mean RRMS"},
            {"key": "m", "head": "M_bias", "md": "M_bias mean", "kind": "rrms", "group": "all-device mean RRMS"},
            {"key": "b5", "head": "B5", "md": "B5 mean (in-sample)", "kind": "rrms", "group": "all-device mean RRMS"},
            {"key": "gm", "head": "GMR(M_bias, C0_refit) [device CI] [cluster CI]",
             "md": "GMR(M_bias, C0_refit) [device CI] [cluster CI]", "kind": "rtext", "family": True},
            {"key": "gb", "head": "GMR(B5, C0_refit) [device CI] [cluster CI]",
             "md": "GMR(B5, C0_refit) [device CI] [cluster CI]", "kind": "rtext", "family": True},
            {"key": "fm", "head": "of mean", "md": "fraction of C0->B5 gap closed by M_bias (means)", "kind": "rtext",
             "group": "C0->B5 gap closed by M_bias"},
            {"key": "fl", "head": "of log GMR", "md": "fraction of C0->B5 gap closed by M_bias (log GMR)",
             "kind": "rtext", "group": "C0->B5 gap closed by M_bias"}]
    notes = [
        "fraction of mean gap = (C0_refit - M_bias) / (C0_refit - B5); fraction of log GMR = ln GMR(M_bias, "
        "C0_refit) / ln GMR(B5, C0_refit). B5 is the in-sample reference (correction trained on the held-out "
        "device), so the gap is the part of the C0_refit error that a bias-only correction of this form removes "
        "when it sees the device.",
        "K = exp(mean_i [ln(M4_i / C4_i) - ln(M295_i / C295_i)]) over the 45 DEV devices (pre-registered "
        "temperature contrast); the K row lists the device CI, then the cluster CI. K > 1: the M_bias / C0_refit "
        "ratio is larger at 4.13 K.",
    ]
    return rf.Table("09_table_i6_1_contrast_gap", "I6-1 temperature contrast K and the fraction of the C0_refit -> "
                    "B5 gap closed by M_bias per temperature", cols, rows, notes)


def cross_row(ds: str, n: int, card: str, calib: str, s: dict, cmp: dict, rule: str,
              cluster: str = "= device CI") -> dict:
    c, b5 = cmp["M_bias_vs_C0_refit"], cmp["B5_vs_C0_refit"]
    bx = min(BASE, key=lambda x: s[x])
    fm, fl = gap(s["C0_refit"], s["M_bias"], s["B5"], c["ratio"], b5["ratio"])
    cb = cmp[f"M_bias_vs_{bx}"]
    return {"ds": ds, "n": str(n), "card": card, "cal": calib, "c0": s["C0_refit"], "m": s["M_bias"],
            "g": f"{r_text(c)} {c['wins']}/{n}", "gcl": cluster,
            "best": f"{bx} {s[bx]:.4f}", "gb": r_text(cb), "b5": s["B5"], "fm": f"{fm:+.3f}", "fl": f"{fl:+.3f}",
            "rule": rule}


def table_cross(d: dict) -> tuple[rf.Table, list]:
    sky = d["sky"]
    ss = {a: sky["summary"][a]["all_device_mean"] for a in sky["summary"]}
    rows = [cross_row("sky130 77 K REAL_DEV (I3-2 / I4-1)", 12, "BSIM4, 43 parameters, ngspice-41",
                      "yes: the paper's 77 K sky130 card, refit", ss, sky["comparisons"],
                      "I3-2 CONTINUE; I4-1 supported against baselines")]
    i5d = d["i5dec"]
    for t in I5_TEMPS:
        r = d["i5"][t]
        s = {a: r["summary"]["all"][a]["all_device_mean"] for a in r["summary"]["all"]}
        z = i5d["primary"] if t == 300 else i5d["secondary"][f"{t}K"]
        rule = (z["replication"].split(" (")[0] + "; " + z["baseline_verdict"].replace("matched by baseline(s)",
                                                                                         "matched by"))
        rows.append(cross_row(f"IHP SG13G2 {t} K DEV (I5-1{', decides' if t == 300 else ''})", 18,
                              "PSP103, 12 parameters, NGSpice-46", "yes: foundry PSP103 card at a rated temperature",
                              s, r["comparisons"]["all"], rule))
    for t in TEMPS:
        r = d["res"][t]
        s = {a: r["summary"]["all"][a]["all_device_mean"] for a in ARMS}
        c = r["comparisons"]
        z = d["cond"][t]
        rule = ("R1 " + ("met" if z["R1"] else "not met") + "; baselines beaten: "
                + (", ".join(x for x in BASE if z["beaten"][x]) or "none"))
        if t == PRIMARY:
            rule = "refuted (" + rule + ")"
        rows.append(cross_row(f"TU Delft 40 nm {tlabel(t)} DEV (I6-1{', decides' if t == PRIMARY else ', control'})",
                              45, "BSIM4, 43-set (36/37 active), ngspice-41",
                              "no: generic ngspice example card (CMOSedu 50 nm), temperature coefficients zeroed",
                              s, c, rule, cluster=cl_text(c["M_bias_vs_C0_refit"])))
    cols = [{"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "n", "head": "n", "md": "devices", "kind": "rtext"},
            {"key": "card", "head": "card, refit parameters, simulator", "md": "card, refit parameters, simulator",
             "kind": "text"},
            {"key": "cal", "head": "card calibrated for the measured condition", "md": "card calibrated for the "
             "measured condition", "kind": "text"},
            {"key": "c0", "head": "C0_refit", "md": "C0_refit mean", "kind": "rrms", "group": "all-device mean RRMS"},
            {"key": "m", "head": "M_bias", "md": "M_bias mean", "kind": "rrms", "group": "all-device mean RRMS"},
            {"key": "g", "head": "GMR(M_bias, C0) [device CI] wins", "md": "GMR(M_bias, C0_refit) [device 95% CI] "
             "wins", "kind": "rtext", "family": True},
            {"key": "gcl", "head": "cluster CI", "md": "GMR(M_bias, C0_refit) cluster 95% CI", "kind": "rtext",
             "family": True},
            {"key": "best", "head": "best baseline", "md": "lowest-mean baseline of B1, B2, B3_d3", "kind": "rtext"},
            {"key": "gb", "head": "GMR(M_bias, best) [CI]", "md": "GMR(M_bias, best baseline) [device 95% CI]",
             "kind": "rtext"},
            {"key": "b5", "head": "B5", "md": "B5 mean (in-sample)", "kind": "rrms"},
            {"key": "fm", "head": "of mean", "md": "fraction of C0->B5 gap closed (means)", "kind": "rtext",
             "group": "C0->B5 gap closed", "family": True},
            {"key": "fl", "head": "of log GMR", "md": "fraction of C0->B5 gap closed (log GMR)", "kind": "rtext",
             "group": "C0->B5 gap closed", "family": True},
            {"key": "rule", "head": "pre-registered rule output", "md": "pre-registered rule output", "kind": "text"}]
    notes = [
        f"Sources: sky130 {rel(SKY)} (12 REAL_DEV devices, 5 nMOS + 7 pMOS; inclusion frozen from the "
        "published-card baseline); IHP " + ", ".join(rel(I5_LODO[t]) for t in I5_TEMPS) + f", {rel(I5_DEC)} (18 "
        "DEV DUTs; inclusion frozen from the measured current); TU Delft " + ", ".join(rel(LODO[t]) for t in TEMPS)
        + " (45 DEV devices in 7 groups; inclusion frozen from the measured current).",
        "Same M_bias recipe (I3-2, unchanged) and held-out refit settings (max_nfev 400, irls 2) in every row. "
        "Folds: sky130 leave-one-device-out; IHP leave-one-DUT-out within class (LV / HV); TU Delft "
        "leave-one-geometry-out (both polarities in one training pool).",
        "Card calibration: sky130 77 K uses the paper's 77 K card (parameters published for that temperature) as "
        "the refit start; IHP uses the foundry PSP103 card, which the PDK supports at 233-398 K; TU Delft uses a "
        "generic 50-nm example card that is not a card of the measured process, with the temperature coefficients "
        "set to 0, at 4.13 K and 295.0 K. Every row refits the card per device before scoring.",
        "best baseline = lowest all-device mean of B1, B2, B3_d3 (the baselines common to I4-1, I5-1 and I6-1). "
        "sky130 also has B3_d2 0.1034 and B4 0.1317, both above B3_d3. B1 and B2 differ between datasets (sky130 B1 "
        "19 screened BSIM4 extras, IHP 70-73 PSP extras, TU Delft 16 BSIM4 extras).",
        "cluster CI: TU Delft resamples the 7 geometry groups; sky130 and IHP have one device per geometry, so the "
        "device bootstrap is the cluster bootstrap there ('= device CI'). fraction of mean gap = (C0_refit - M_bias) / (C0_refit - "
        "B5); fraction of log GMR = ln GMR(M_bias, C0_refit) / ln GMR(B5, C0_refit); negative: M_bias above "
        "C0_refit.",
    ]
    return rf.Table("13_table_i6_1_cross_dataset", "Cross-dataset summary of the I3-2 bias-only correction: sky130 "
                    "77 K (I3-2 / I4-1), IHP SG13G2 233-398 K (I5-1), TU Delft 40 nm 4.13 K and 295.0 K (I6-1)",
                    cols, rows, notes), rows


# ================================================================== figures

def fig_forest(d: dict, path: Path) -> list[str]:
    left = ("C0_refit", "T1_inloop", "B1", "B2", "B3_d3")
    right = ("M_bias", "T1_inloop", "B1", "B2", "B3_d3", "B5")
    fig, axs = plt.subplots(2, 2, figsize=(20, 8.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for i, t in enumerate(TEMPS):
        cmp = d["res"][t]["comparisons"]
        for j, (arms, fmt_name) in enumerate(((left, "M_bias_vs_{}"), (right, "{}_vs_C0_refit"))):
            ax = axs[i, j]
            style(ax)
            ys = np.arange(len(arms))[::-1]
            for y, x in zip(ys, arms):
                c = cmp[fmt_name.format(x)]
                cb = c["cluster_bootstrap"]
                col = ARM_COLOR[x]
                lw = 2.8 if (j == 0 and x == "C0_refit") or (j == 1 and x == "M_bias") else 2.0
                ax.plot([c["ci_low"], c["ci_high"]], [y, y], color=col, lw=lw, solid_capstyle="butt")
                ax.plot([cb["ci_low"], cb["ci_high"]], [y + 0.22, y + 0.22], color=col, lw=0.9, alpha=0.75)
                for xx in (cb["ci_low"], cb["ci_high"]):
                    ax.plot([xx, xx], [y + 0.16, y + 0.28], color=col, lw=0.9, alpha=0.75)
                ax.scatter([c["ratio"]], [y], color=col, s=48, edgecolor=BG, zorder=4, marker=ARM_MARK.get(x, "o"))
                ax.text(1.01, y, f"{r_text(c)}  cl. [{cb['ci_low']:.3f}, {cb['ci_high']:.3f}]  {c['wins']}/45",
                        transform=ax.get_yaxis_transform(), fontsize=8, va="center", color=INK)
            ax.axvline(1.0, color=INK, ls="--", lw=0.9)
            ax.set_xscale("log")
            if j == 0:
                ax.set_xlim(0.5, 1.6)
                ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5]
                ax.set_yticks(ys)
                ax.set_yticklabels([f"X = {x}" + (" (R1)" if x == "C0_refit" else
                                                  " (baseline)" if x in BASE else " (not in rule)") for x in arms],
                                   fontsize=8.5)
            else:
                ax.set_xlim(0.12, 1.6)
                ticks = [0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
                ax.set_yticks(ys)
                ax.set_yticklabels([f"X = {x}" + (" (in-sample)" if x == "B5" else "") for x in arms], fontsize=8.5)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{v:g}" for v in ticks])
            ax.minorticks_off()
            ax.set_ylim(-0.7, len(arms) - 0.3)
            role = "decides" if t == PRIMARY else "control"
            what = "GMR(M_bias, X)" if j == 0 else "GMR(X, C0_refit)"
            ax.set_title(f"{tlabel(t)} ({role}): {what}", fontsize=9.5, color=INK, loc="left")
    axs[-1, 0].set_xlabel("GMR(M_bias, X) (log axis); thick: 95% device-bootstrap CI; thin with caps: 95% "
                          "cluster-bootstrap CI (7 groups); < 1: M_bias lower", fontsize=8.5)
    axs[-1, 1].set_xlabel("GMR(X, C0_refit) (log axis); thick: device CI; thin: cluster CI; < 1: X lower than the "
                          "no-correction refit", fontsize=8.5)
    fig.suptitle("I6-1 TU Delft 40 nm, 45 DEV devices, leave-one-geometry-out: held-out strict paper RRMS after the "
                 "card refit\n4.13 K: R1 not met (device CI upper 1.004), rule output 'refuted'; 295.0 K: "
                 "GMR(M_bias, C0_refit) 0.797, device CI upper 0.935, all baselines beaten", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.10, right=0.81, top=0.89, bottom=0.08, hspace=0.38, wspace=1.05)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO[t]) for t in TEMPS] + [rel(DEC)]


def dev_order(d: dict) -> list[str]:
    return sorted(d["split"]["DEV"], key=lambda t: (d["groups"].index(group_of(t)), t))


def fig_per_device(d: dict, path: Path) -> list[str]:
    tags = dev_order(d)
    arms = ("start", "C0_refit", "B1", "B2", "B3_d3", "T1_inloop", "M_bias", "B5")
    fig, axs = plt.subplots(2, 1, figsize=(16, 10), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(BG)
    off = np.linspace(-0.33, 0.33, len(arms))
    bounds = [i for i in range(1, len(tags)) if group_of(tags[i]) != group_of(tags[i - 1])]
    for ax, t in zip(axs, TEMPS):
        style(ax)
        per = d["res"][t]["per_device"]
        for j, a in enumerate(arms):
            ax.scatter(np.arange(len(tags)) + off[j], [per[a][k] for k in tags], color=ARM_COLOR[a],
                       marker=ARM_MARK.get(a, "o"), s=16 if a != "start" else 14, zorder=3,
                       label=f"{a} (all-device {stats(per[a])['all']:.4f})")
        ax.set_yscale("log")
        for b in bounds:
            ax.axvline(b - 0.5, color=MUTED, lw=1.0)
        starts = [0] + bounds + [len(tags)]
        for a, b in zip(starts[:-1], starts[1:]):
            ax.text((a + b - 1) / 2, 1.01, glabel(group_of(tags[a]), d["mani"]), transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=8.5, color=MUTED)
        ax.set_ylabel("held-out strict paper RRMS (log scale)", fontsize=9)
        ax.set_title(f"{tlabel(t)} ({'decides' if t == PRIMARY else 'control'})", fontsize=10, color=INK, loc="left",
                     pad=16)
        ax.legend(fontsize=7.5, frameon=False, ncol=4, loc="lower left")
    axs[-1].set_xticks(np.arange(len(tags)))
    axs[-1].set_xticklabels([k.split("_d")[-1] for k in tags], fontsize=7)
    axs[-1].set_xlabel("device id (grouped by held-out geometry)", fontsize=9)
    fig.suptitle("I6-1 TU Delft 40 nm, per DEV device: held-out RRMS of every arm after the card refit (start: "
                 "starting card, no held-out refit; B5: in-sample)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO[t]) for t in TEMPS]


def fig_ratio_scatter(d: dict, path: Path) -> list[str]:
    q4, q295 = d["q4"], d["q295"]
    fig, ax = plt.subplots(figsize=(8.6, 7.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    for gi, g in enumerate(d["groups"]):
        ks = [k for k in d["split"]["DEV"] if group_of(k) == g]
        gm4 = float(np.exp(np.mean([np.log(q4[k]) for k in ks])))
        gm295 = float(np.exp(np.mean([np.log(q295[k]) for k in ks])))
        ax.scatter([q295[k] for k in ks], [q4[k] for k in ks], color=GROUP_COLOR[gi], s=26, alpha=0.85,
                   label=f"{glabel(g, d['mani'])} ({len(ks)}): group GMR 295 K {gm295:.3f}, 4.13 K {gm4:.3f}")
    vals = list(q4.values()) + list(q295.values())
    lo, hi = min(vals) / 1.1, max(vals) * 1.1
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.9, ls=":")
    ax.axhline(1.0, color=INK, lw=0.8, ls="--")
    ax.axvline(1.0, color=INK, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    tk = [v for v in (0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0) if lo <= v <= hi]
    ax.set_xticks(tk)
    ax.set_xticklabels([f"{v:g}" for v in tk])
    ax.set_yticks(tk)
    ax.set_yticklabels([f"{v:g}" for v in tk])
    ax.minorticks_off()
    ax.set_xlabel("RRMS(M_bias) / RRMS(C0_refit) at 295.0 K", fontsize=9)
    ax.set_ylabel("RRMS(M_bias) / RRMS(C0_refit) at 4.13 K", fontsize=9)
    k, kc = d["K"], d["Kc"]
    c4, c295 = (d["res"][t]["comparisons"]["M_bias_vs_C0_refit"] for t in TEMPS)
    ax.set_title(f"I6-1 per DEV device: M_bias / C0_refit ratio at 4.13 K vs 295.0 K (dotted: equal ratio)\n"
                 f"GMR(M_bias, C0_refit) 4.13 K {r_text(c4)}, 295.0 K {r_text(c295)}\n"
                 f"K {r_text(k)}, cluster [{kc['ci_low']:.3f}, {kc['ci_high']:.3f}]; {45 - k['wins']}/45 devices "
                 "above the dotted line (ratio larger at 4.13 K)",
                 fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO[t]) for t in TEMPS] + [rel(DEC)]


def fig_iv(tag: str, d: dict, path: Path) -> list[str]:
    """Stored stock-card and refit-card figures at both temperatures, stacked."""
    import matplotlib.image as mpimg
    dev = next(v for v in d["mani"]["devices"] if v["tag"] == tag)
    st = d["stock"]["per_device"]
    imgs = []
    for t in TEMPS:
        per = d["res"][t]["per_device"]
        tk = dev["temps_K"][str(t)]
        stock = STOCK_FIGS / f"stock_{tag}_{t}K.png"
        refit = FIGS / f"iv_{tag}_{t}K.png"
        lab = "a" if t == 4 else "c"
        imgs.append((stock, f"{lab}. {tk:.2f} K: points = measured; solid lines = ngspice-41 stock CMOSedu BSIM4 card "
                            f"(no fit, temperature coefficients 0), stock RRMS {st[str(t)][tag]:.3f}"))
        lab = "b" if t == 4 else "d"
        imgs.append((refit, f"{lab}. {tk:.2f} K: points = measured; solid = ngspice-41 C0_refit card (held-out "
                            f"refit, no correction), RRMS {per['C0_refit'][tag]:.4f}; dashed = ngspice-41 M_bias "
                            f"refit card x exp(g) of the frozen fold correction (applied in Python), RRMS "
                            f"{per['M_bias'][tag]:.4f}"))
    arrs = [mpimg.imread(p) for p, _ in imgs]
    w = max(a.shape[1] for a in arrs)
    hs = [a.shape[0] for a in arrs]
    head = 40
    total = sum(hs) + head * len(arrs) + 50
    fig = plt.figure(figsize=(w / 110, total / 110), dpi=110)
    fig.patch.set_facecolor("white")
    y = total - 50
    fig.text(0.01, 1 - 15 / total, f"I6-1 {tag} (DEV, group {glabel(group_of(tag), d['mani'])}): I-V from stored "
             "figures; line colors follow the curve key; the legend lists the fixed bias of each curve",
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
    return [rel(p) for p, _ in imgs] + [rel(LODO[t]) for t in TEMPS]


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

    lodo = [rel(LODO[t]) for t in TEMPS]
    add_table(table_dataset(d), "Table of the TU Delft dataset: 10 (polarity, W, L) groups with copies, die, array, "
              "DEV/TEST split, included curves per temperature and DEV stock-card RRMS", [rel(MANI), rel(SPLIT),
                                                                                          rel(STOCK)])
    add_table(table_card_box(d), "Table of the CMOSedu 50-nm BSIM4 card and the 43-parameter box: stock value and "
              "box per polarity, starting fits at a bound per temperature", [rel(CARD), rel(START), rel(SCREEN),
                                                                            rel(PREREG), rel(AMEND)])
    add_table(table_box_trial(d), "Table of the box-width trial (factor 2 vs 10, 3 devices, both temperatures): "
              "RRMS per IRLS round, simulations, parameters at a bound", [rel(BOX)])
    add_table(table_stock_start(d), "Table of stock-card, starting-card and C0_refit RRMS per temperature and DEV "
              "group, GMR(start, C0_refit)", [rel(STOCK), rel(START)] + lodo)
    add_table(table_decision(d), "Table of the decision conditions per temperature: R1 with device and cluster CI, "
              "R2, R3, GMR(M_bias, X) for B1, B2, B3_d3 with device and cluster CI", [rel(PREREG), rel(DEC)] + lodo)
    add_table(table_outcome(d), "Table of the outcome logic: premise P, contrast K, each pre-registered outcome "
              "condition evaluated, rule output 'refuted'", [rel(PREREG), rel(DEC)] + lodo)
    add_table(table_results(d), "Table of the results per temperature for every arm (start, C0_refit, M_bias, "
              "T1_inloop, B1, B2, B3_d3, B5): all-device, nMOS, pMOS, combined, median, failed, GMR(M_bias, X) and "
              "GMR(X, C0_refit) with device and cluster CIs and wins", lodo + [rel(SCREEN)])
    add_table(table_groups(d), "Table of the per-group (fold) means of every held-out arm and GMR(M_bias, C0_refit) "
              "within the group", lodo + [rel(DEC)])
    add_table(table_contrast_gap(d), "Table of the temperature contrast K and the fraction of the C0_refit -> B5 gap "
              "closed by M_bias per temperature", lodo + [rel(DEC)])
    src = fig_forest(d, folder / "10_i6_1_gmr_forest.png")
    items.append(("10_i6_1_gmr_forest.png", "Forest plot at 4.13 K and 295.0 K: left GMR(M_bias, X) for X in "
                  "C0_refit (R1), T1_inloop, B1, B2, B3_d3; right GMR(X, C0_refit) for M_bias, T1_inloop, B1, B2, "
                  "B3_d3, B5; device-bootstrap CI (thick) and cluster-bootstrap CI (thin), wins", src))
    src = fig_per_device(d, folder / "11_i6_1_per_device.png")
    items.append(("11_i6_1_per_device.png", "Per-device held-out RRMS at 4.13 K and 295.0 K for start, C0_refit, B1, "
                  "B2, B3_d3, T1_inloop, M_bias and B5 (log scale), grouped by held-out geometry", src))
    src = fig_ratio_scatter(d, folder / "12_i6_1_ratio_4K_vs_295K.png")
    items.append(("12_i6_1_ratio_4K_vs_295K.png", "Per-device RRMS(M_bias) / RRMS(C0_refit) at 4.13 K against the "
                  "same ratio at 295.0 K, colored by group, with group GMRs and the contrast K", src))
    tc, cross_rows = table_cross(d)
    add_table(tc, "Cross-dataset table: sky130 77 K REAL_DEV, IHP SG13G2 233/300/343/398 K, TU Delft 4.13 K and 295.0 "
              "K: card calibration status, C0_refit, M_bias, GMR(M_bias, C0_refit) with device and cluster CI, best "
              "baseline, B5, fraction of the C0_refit -> B5 gap closed, rule output",
              [rel(SKY), rel(I5_DEC)] + [rel(I5_LODO[t]) for t in I5_TEMPS] + lodo)
    for j, tag in enumerate(IV_DEVS):
        name = f"{14 + j:02d}_i6_1_iv_{tag.replace('t40_', '')}.png"
        src = fig_iv(tag, d, folder / name)
        items.append((name, f"I-V of {tag} at 4.13 K and 295.0 K, assembled from stored figures: (a, c) measured "
                      "points and the ngspice-41 stock card; (b, d) measured points, ngspice-41 C0_refit card (solid) "
                      "and ngspice-41 M_bias refit card multiplied by exp(g) of the frozen fold correction (dashed)",
                      src))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i6_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "",
          f"Generated {now} by {gen} from stored files only. RRMS is the strict paper metric with the inclusion "
          "frozen from the measured current (pdk2 rule). I6-1 values are on the 45 TU Delft DEV devices "
          "(development evidence); TEST is not loaded.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    res, dec = d["res"], d["dec"]
    s4, s295 = (res[t]["summary"]["all"] for t in TEMPS)
    c4, c295 = (res[t]["comparisons"]["M_bias_vs_C0_refit"] for t in TEMPS)

    def means(s):
        return ", ".join(f"{a} {s[a]['all_device_mean']:.4f}" for a in ("C0_refit", "M_bias", "T1_inloop", "B1",
                                                                           "B2", "B3_d3")) + \
            f", B5 (in-sample) {s['B5']['all_device_mean']:.4f}"

    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen} from stored files only (no NGSpice, no training).", "",
              "## Contents", "",
              "- I6-1: the I3-2 bias-only correction M_bias (recipe unchanged) on TU Delft 40-nm CMOS (4TU "
              "doi:10.4121/c41c26f7-586f-48ca-8abe-9127b2d97c60.v1, CC0, campaign 1) measured at 4.13 K and 295.0 K, "
              "with the ngspice-41 CMOSedu 50-nm BSIM4 example card (temperature coefficients 0) refit per device; "
              "45 DEV devices in 7 (polarity, W, L) groups, leave-one-geometry-out. Pre-registration "
              f"`{rel(PREREG)}` (commit a967f04); amendment `{rel(AMEND)}`; data pipeline, card, stock RRMS, box "
              "trial, smoke runs commit 3e8957c; starting cards commit edef8f3; full-run outputs (B1 screen, LODO, "
              "decision, figures) commit 74f4fc4. Queue jobs I6_1_start (1083 s), I6_1_screen (50 s), I6_1_lodo_4 "
              f"({res[4]['runtime_s']:.0f} s), I6_1_lodo_295 ({res[295]['runtime_s']:.0f} s), I6_1_decide, "
              "I6_1_figs, 2026-10-20.",
              f"- Rule output: {dec['outcome']}. 4.13 K: GMR(M_bias, C0_refit) {r_text(c4)} (cluster {cl_text(c4)}), "
              f"{c4['wins']}/45 devices with M_bias < C0_refit; R1 not met, R2 and R3 met; B1 beaten, B2 and B3_d3 "
              f"not beaten. All-device means: {means(s4)}.",
              f"- 295.0 K (control): GMR(M_bias, C0_refit) {r_text(c295)} (cluster {cl_text(c295)}), "
              f"{c295['wins']}/45; R1, R2, R3 met; B1, B2, B3_d3 beaten (device CI upper < 1). All-device means: "
              f"{means(s295)}.",
              f"- Contrast K (4.13 K ratio over 295.0 K ratio) {r_text(d['K'])}, cluster [{d['Kc']['ci_low']:.3f}, "
              f"{d['Kc']['ci_high']:.3f}]: the M_bias / C0_refit ratio is larger at 4.13 K. Premise P holds "
              f"(C0_refit {d['c0m'][4]:.4f} vs {d['c0m'][295]:.4f}, ratio {d['c0m'][4] / d['c0m'][295]:.2f}).",
              "- The cluster-bootstrap CIs (7 geometry groups) are wider than the device-bootstrap CIs; the "
              "295.0 K R1 cluster CI includes 1. The rule uses the device bootstrap.",
              "- Simulator: ngspice-41 native arm64 build (the canonical simulator); NGSpice-46 is not used in "
              "I6-1.",
              "- I-V figures 14-18 are assembled from stored PNGs (no simulation in this batch): "
              "`out/research/explore/I6_1/stock/figs/stock_*_{4,295}K.png` (measured points and the stock card) "
              "and `out/research/explore/I6_1/figs/iv_*_{4,295}K.png` (measured points, C0_refit card as solid "
              "lines, M_bias refit card x exp(g) as dashed lines). There is no published or foundry card for this "
              "process, so the stock card is the no-fit reference. The M_bias curve is the ngspice-41 current of the "
              "M_bias refit card multiplied by exp(g) of the frozen fold correction in Python, not a SPICE-native "
              "simulation of the correction. T1_inloop, B1, B2, B3_d3 and B5 cards are not plotted.",
              "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    g4 = res[4]["summary"]["all"]
    fm = {t: gap(res[t]["summary"]["all"]["C0_refit"]["all_device_mean"],
                 res[t]["summary"]["all"]["M_bias"]["all_device_mean"],
                 res[t]["summary"]["all"]["B5"]["all_device_mean"],
                 res[t]["comparisons"]["M_bias_vs_C0_refit"]["ratio"],
                 res[t]["comparisons"]["B5_vs_C0_refit"]["ratio"]) for t in TEMPS}
    summary = [
        f"I6-1 rule output (TU Delft 40 nm, 45 DEV devices, BSIM4 example card): {dec['outcome']}. 4.13 K: "
        f"GMR(M_bias, C0_refit) {r_text(c4)}, cluster {cl_text(c4)}, {c4['wins']}/45 (R1 not met); all-device "
        f"C0_refit {g4['C0_refit']['all_device_mean']:.4f}, M_bias {g4['M_bias']['all_device_mean']:.4f}, T1_inloop "
        f"{g4['T1_inloop']['all_device_mean']:.4f}, B1 {g4['B1']['all_device_mean']:.4f}, B2 "
        f"{g4['B2']['all_device_mean']:.4f}, B3_d3 {g4['B3_d3']['all_device_mean']:.4f}, B5 "
        f"{g4['B5']['all_device_mean']:.4f}.",
        f"295.0 K: GMR(M_bias, C0_refit) {r_text(c295)}, cluster {cl_text(c295)}, {c295['wins']}/45; C0_refit "
        f"{s295['C0_refit']['all_device_mean']:.4f}, M_bias {s295['M_bias']['all_device_mean']:.4f}; B1, B2, B3_d3 "
        f"beaten. Contrast K {r_text(d['K'])}; premise P holds.",
        f"Fraction of the C0_refit -> B5 gap closed by M_bias (means / log GMR): 4.13 K {fm[4][0]:+.3f} / "
        f"{fm[4][1]:+.3f}, 295.0 K {fm[295][0]:+.3f} / {fm[295][1]:+.3f}; sky130 77 K {cross_rows[0]['fm']} / "
        f"{cross_rows[0]['fl']}; IHP 300 K {cross_rows[2]['fm']} / {cross_rows[2]['fl']}.",
    ]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    for r in cross_rows:
        print(" |", r["ds"], f"{r['c0']:.4f}", f"{r['m']:.4f}", r["g"], r["gcl"], r["best"], r["gb"],
              f"{r['b5']:.4f}", r["fm"], r["fl"], r["rule"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
