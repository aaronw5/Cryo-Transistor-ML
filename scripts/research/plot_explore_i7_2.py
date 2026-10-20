#!/usr/bin/env python3
"""Report batch for exploration direction I7-2: hierarchical bias-only correction
g_d = g_shared + phi . delta_d (phi = quadratic basis in normalized terminal bias,
6 coefficients) with an empirical-Bayes prior over delta, fitted together with the
card on K = 2 or 4 curves of the held-out device and scored on the device's other
included curves. Four datasets: sky130 77 K REAL_DEV, IHP SG13G2 300 K DEV, TU Delft
40 nm 4.13 K and 295 K DEV. Arms a_C0, b_shared, c_hier (primary), c_ml (flat prior),
d_scratch (per-device MLP from scratch on the K curves), e_crossinterp (non-learned
cross-curve residual).

Reads stored files only (no NGSpice, no training):

* ``out/research/explore/I7_2/{prereg.json, decision.json}``
* ``out/research/explore/I7_2/<dataset>/{results.json, cards.json}``
* the measured records of each dataset (E1 real_dev, I5-1 / I6-1 starting records)
  through the driver's dataset adapters (file reads only), the stored fold
  corrections g_shared (evaluated in Python on the measured bias points)

Consistency checks against the stored files: the curve split of every row is
recomputed from the records; per-device values (all draws, K = 2, K = 4), the
aggregates and every paired ratio with its device (and TU Delft cluster) bootstrap
are recomputed from the per-split rows; the decision (C1-C4, counts, outcome) is
recomputed from the recomputed ratios and compared with decision.json; each fold's
prior (m, S, sd) is recomputed from the stored training deltas, the selected mu from
the stored criterion, and every stored c_hier / c_ml delta from m + Lc z.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i7_2.py --date 2026-10-20
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style; adds scripts/research/explore to sys.path)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA, VIOLET, RED, DPI, EXP,  # noqa: E402
                                   load, rel, style)
from cryoml.explore import i7_2_hier as h  # noqa: E402
from cryoml.explore.e1_correction import raw_features  # noqa: E402
from cryoml.explore.i3_2_bias import paired_ratio  # noqa: E402
import i7_2_hier as drv  # noqa: E402  (driver: dataset adapters, aggregate; no simulation is called)

rf = pe.rf
plt = pe.plt

TOPIC = "i7_2"
I7 = EXP / "I7_2"
PREREG = I7 / "prereg.json"
DEC = I7 / "decision.json"
DATASETS = ("sky130", "ihp300", "tud4", "tud295")
RES = {d: I7 / d / "results.json" for d in DATASETS}
CARDS = {d: I7 / d / "cards.json" for d in DATASETS}
DS_LABEL = {"sky130": "sky130 77 K REAL_DEV", "ihp300": "IHP SG13G2 300 K DEV", "tud4": "TU Delft 40 nm 4.13 K DEV",
            "tud295": "TU Delft 40 nm 295 K DEV"}
DS_SHORT = {"sky130": "sky130", "ihp300": "IHP 300 K", "tud4": "TU Delft 4.13 K", "tud295": "TU Delft 295 K"}
ARMS = h.ARMS  # a_C0, b_shared, c_hier, c_ml, d_scratch, e_crossinterp
ARM_SHORT = {"a_C0": "a", "b_shared": "b", "c_hier": "c", "c_ml": "c_ml", "d_scratch": "d", "e_crossinterp": "e"}
ARM_DESC = {"a_C0": "card refit on the K curves, no correction",
            "b_shared": "card refit with the stored fold correction g_shared in the loop",
            "c_hier": "card + delta_d fitted jointly, MAP with the learned prior and selected mu (primary)",
            "c_ml": "as c_hier with mu = 0 (flat prior on the z box)",
            "d_scratch": "card refit + per-device M_bias-form MLP from scratch on the K curves",
            "e_crossinterp": "a's card + measured-minus-simulated residual at curve crossings (non-learned)"}
ARM_COLOR = {"a_C0": GREY, "b_shared": ORANGE, "c_hier": BLUE, "c_ml": AQUA, "d_scratch": VIOLET,
             "e_crossinterp": RED}
PAIRS = drv.PAIRS  # (name, method, control)
PAIR_LABEL = {"c_vs_b": "GMR(c_hier, b_shared)", "c_vs_a": "GMR(c_hier, a_C0)", "c_vs_d": "GMR(c_hier, d_scratch)",
              "c_vs_e": "GMR(c_hier, e_crossinterp)", "c_vs_cml": "GMR(c_hier, c_ml)", "b_vs_a": "GMR(b_shared, a_C0)",
              "d_vs_a": "GMR(d_scratch, a_C0)", "e_vs_a": "GMR(e_crossinterp, a_C0)", "cml_vs_a": "GMR(c_ml, a_C0)"}
PAIR_ROLE = {"c_vs_b": "C1", "c_vs_a": "C2", "c_vs_d": "C3"}
KPAIRS = ("c_vs_b", "c_vs_a", "c_vs_d", "c_vs_e", "d_vs_a", "e_vs_a")
DRAWS = h.DRAWS
KSETS = (("K2a", "K2a"), ("K2b", "K2b"), ("K = 2", 2), ("K4a", "K4a"), ("K4b", "K4b"), ("K = 4", 4), ("all", None))
OUTCOMES = ("invalid", "supported_continue", "delta_no_gain_stop", "sharing_no_gain", "inconclusive")
COEF = ("1", "a", "b", "a^2", "ab", "b^2")


def wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


def r_text(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def close(a, b, tol: float = 1e-12) -> bool:
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_ratio(stored: dict, mine: dict, what: str) -> None:
    for k in ("ratio", "ci_low", "ci_high"):
        assert close(stored[k], mine[k]), (what, k, stored[k], mine[k])
    for k in ("wins", "n_used"):
        if k in stored:
            assert stored[k] == mine[k], (what, k, stored[k], mine[k])


def pol(tag: str) -> str:
    return "nmos" if "nmos_" in tag else "pmos"


def fmu(mu) -> str:
    mu = float(mu)
    return "inf" if math.isinf(mu) else f"{mu:g}"


# ================================================================== per-draw values

def draw_values(rows: list[dict], arm: str, draw: str) -> dict:
    """{tag: held-out RRMS of one draw}; NaN if that split failed for ``arm``."""
    out = {}
    for r in rows:
        if r["draw"] == draw:
            v = r["arms"].get(arm, {}).get("heldout_rrms", math.nan)
            out[r["tag"]] = float(v) if v is not None else math.nan
    return out


def values(rows: list[dict], arm: str, which) -> dict:
    if isinstance(which, str):
        return draw_values(rows, arm, which)
    return h.device_values(rows, arm, which)


def pair_of(name: str) -> tuple[str, str]:
    for n, x, y in PAIRS:
        if n == name:
            return x, y
    raise KeyError(name)


# ================================================================== loading and checks

def load_all() -> dict:
    dec, prereg = load(DEC), load(PREREG)
    assert not dec["smoke"] and dec["missing_or_not_preregistered"] == []
    assert dec["datasets"] == list(DATASETS)
    res, cards, recs, clusters = {}, {}, {}, {}
    n_checks = 0
    for d in DATASETS:
        r = load(RES[d])
        assert r["dataset"] == d and not r["smoke"] and r["preregistered_config"], d
        assert r["options"] == drv.OPTIONS and r["d_hyper_overrides"] == {}, d
        assert r["draws"] == list(DRAWS), d
        res[d] = r
        cards[d] = load(CARDS[d])
        ds = drv.dataset(d)
        recs[d] = ds
        rows = r["rows"]
        # cards rows align with result rows
        assert [(c["tag"], c["draw"]) for c in cards[d]] == [(x["tag"], x["draw"]) for x in rows], d
        for c in cards[d]:
            assert sorted(c["cards"]) == sorted(["base", "a_C0", "b_shared", "c_hier", "c_ml", "d_scratch"]), d
        # curve split recomputed from the records
        held = sorted(r["held_out"])
        assert held == sorted(ds.by_tag), d
        skipped = []
        by = {(x["tag"], x["draw"]): x for x in rows}
        for tag in held:
            for dname, dr in h.draws(ds.by_tag[tag], DRAWS).items():
                if dr is None:
                    skipped.append({"tag": tag, "draw": dname})
                    assert (tag, dname) not in by
                    continue
                x = by[(tag, dname)]
                assert x["adapt"] == dr["adapt"] and x["heldout"] == dr["heldout"], (d, tag, dname)
                assert x["K"] == h.DRAW_K[dname] and x["fold"] == ds.fold_of(tag), (d, tag, dname)
                n_checks += 1
        assert sorted(map(str, skipped)) == sorted(map(str, r["skipped_draws"])), d
        assert len(rows) + len(skipped) == 4 * len(held), d
        clusters[d] = None if d in ("sky130", "ihp300") else {t: ds.fold_of(t) for t in held}
        # per-device values, aggregates and ratios recomputed
        for label, k in (("all", None), ("K2", 2), ("K4", 4)):
            s = r["summary"][label]
            vals = {a: h.device_values(rows, a, k) for a in ARMS}
            for a in ARMS:
                assert sorted(vals[a]) == sorted(s["per_device"][a]), (d, label, a)
                for t, v in vals[a].items():
                    assert close(v, s["per_device"][a][t]), (d, label, a, t)
                agg = drv.aggregate(vals[a])
                for kk, v in agg.items():
                    assert close(v, s["aggregate"][a][kk]), (d, label, a, kk)
            for name, x, y in PAIRS:
                mine = paired_ratio(vals[x], vals[y])
                check_ratio(s["comparisons"][name], mine, f"{d} {label} {name}")
                if clusters[d] is not None:
                    cl = paired_ratio(vals[x], vals[y], clusters=clusters[d])
                    sc = s["comparisons"][name]["cluster_bootstrap"]
                    assert close(sc["ci_low"], cl["ci_low"]) and close(sc["ci_high"], cl["ci_high"]), (d, name)
                    assert sc["resample_unit"] == "cluster (7)"
                n_checks += 1
        # decision table entries
        for name, st in dec["table"][d].items():
            c = r["summary"]["all"]["comparisons"][name]
            assert all(close(st[k], c[k]) for k in ("ratio", "ci_low", "ci_high")), (d, name)
            assert st["wins"] == c["wins"] and st["n_used"] == c["n_used"], (d, name)
        assert sorted(dec["table"][d]) == sorted(n for n, _, _ in PAIRS)
        # start choice, timeouts
        sc = {k: sum(x["start"]["chosen"] == k for x in rows) for k in ("center", "train_median")}
        assert sc == r["start_choice"], d
        assert r["n_sim_timeout"] == sum(x.get("n_sim_timeout", 0) for x in rows), d
        # priors, mu selection and deltas
        for f, fo in r["folds"].items():
            td = np.array(list(fo["train_deltas"].values()), float)
            pr = h.prior_from_deltas(td)
            assert np.allclose(pr.m, fo["prior"]["m"], rtol=0, atol=1e-12), (d, f)
            assert np.allclose(pr.S, fo["prior"]["S"], rtol=0, atol=1e-12), (d, f)
            assert np.allclose(np.sqrt(np.diag(pr.S)), fo["prior"]["sd"], rtol=0, atol=1e-12), (d, f)
            assert fo["prior"]["n_devices"] == len(td), (d, f)
            assert sorted(fo["train_deltas"]) == sorted(x.tag for x in ds.train_of(f)), (d, f)
            for kk, sel in fo["mu_selection"].items():
                crit = {float(m): c for m, c in sel["criterion"].items()}
                fin = [(c, m) for m, c in crit.items() if np.isfinite(c)]
                best = min(c for c, _ in fin)
                chosen = max(m for c, m in fin if c <= best)
                assert close(chosen, float(sel["mu"])), (d, f, kk)
        for x in rows:
            fo = r["folds"][x["fold"]]
            assert close(float(x["mu"]), float(fo["mu_selection"][str(x["K"])]["mu"])), (d, x["tag"], x["draw"])
            S = np.array(fo["prior"]["S"], float)
            Lc = np.linalg.cholesky(S)
            m = np.array(fo["prior"]["m"], float)
            for arm in ("c_hier", "c_ml"):
                a = x["arms"][arm]
                z = np.array(a["z"], float)
                assert np.allclose(m + Lc @ z, a["delta"], rtol=0, atol=1e-10), (d, x["tag"], x["draw"], arm)
                assert close(float(z @ z), a["z_norm2"], 1e-9)
            if math.isinf(float(x["mu"])):
                assert np.allclose(x["arms"]["c_hier"]["z"], 0.0)
    # decision recomputed from the recomputed ratios
    cmp, failed = {}, {}
    for d in DATASETS:
        rows = res[d]["rows"]
        vals = {a: h.device_values(rows, a, None) for a in ARMS}
        cmp[d] = {n: paired_ratio(vals[x], vals[y]) for n, x, y in PAIRS if n in ("c_vs_b", "c_vs_a", "c_vs_d")}
        agg = drv.aggregate(vals["c_hier"])
        failed[d] = {"n": agg["n"], "n_failed": agg["n_failed"],
                     "n_paired": min(cmp[d][n]["n_used"] for n in ("c_vs_b", "c_vs_a", "c_vs_d"))}
    mine = h.decide(cmp, failed)
    for k in ("conditions", "counts", "C4_per_dataset", "outcome", "datasets"):
        assert mine[k] == dec[k], (k, mine[k], dec[k])
    assert failed == dec["failed"], (failed, dec["failed"])
    assert dec["outcome"] == "inconclusive"
    crit = prereg["criterion"]
    assert crit["C1"].startswith("GMR(c_hier, b_shared): CI upper < 1 on at least 3")
    return {"dec": dec, "prereg": prereg, "res": res, "cards": cards, "ds": recs, "clusters": clusters,
            "failed": failed, "cmp_rule": cmp, "n_checks": n_checks}


# ================================================================== derived quantities

def outcome_rows(d: dict) -> list[tuple[str, str, bool]]:
    dec = d["dec"]
    C, n = dec["conditions"], dec["counts"]
    c4_fail = sum(1 for v in dec["C4_per_dataset"].values() if not v)
    return [("invalid", "C4 fails on 2 or more datasets", c4_fail >= 2),
            ("supported_continue", "C1, C2, C3 and C4 hold", all(C.values())),
            ("delta_no_gain_stop", "GMR(c_hier, b_shared) point estimate >= 0.98 on 2 or more datasets",
             n["c_vs_b_point_ge_0.98"] >= 2),
            ("sharing_no_gain", "C1 and C2 hold and C3 fails", C["C1"] and C["C2"] and not C["C3"]),
            ("inconclusive", "any other combination", True)]


def per_k_table(d: dict) -> dict:
    """{ds: {kset_label: {pair: paired_ratio}}} with per-draw and per-K values."""
    out = {}
    for ds in DATASETS:
        rows = d["res"][ds]["rows"]
        out[ds] = {}
        for lab, which in KSETS:
            vals = {a: values(rows, a, which) for a in ARMS}
            out[ds][lab] = {"pairs": {n: paired_ratio(vals[pair_of(n)[0]], vals[pair_of(n)[1]])
                                      for n in KPAIRS + ("b_vs_a", "cml_vs_a")},
                            "vals": vals}
            if which in (2, 4) or which is None:
                key = {2: "K2", 4: "K4", None: "all"}[which]
                for n in KPAIRS:
                    check_ratio(d["res"][ds]["summary"][key]["comparisons"][n], out[ds][lab]["pairs"][n],
                                f"{ds} {key} {n}")
    return out


def ranking(pairs: dict) -> str:
    """arms ordered by GMR(X, a_C0) (a_C0 = 1), lowest first."""
    g = {"a": 1.0, "b": pairs["b_vs_a"]["ratio"], "c": pairs["c_vs_a"]["ratio"], "c_ml": pairs["cml_vs_a"]["ratio"],
         "d": pairs["d_vs_a"]["ratio"], "e": pairs["e_vs_a"]["ratio"]}
    return " < ".join(k for k, _ in sorted(g.items(), key=lambda kv: kv[1]))


def held_points(rec, keys, corr) -> dict:
    """bias features, basis and g_shared on the trimmed included points of ``keys``."""
    out = {}
    for c in rec.curves:
        if c.key not in keys or not c.included or not (c.den > 0):
            continue
        s = slice(c.start, None)
        sim = np.asarray(c.sim, float)[s]
        phi = raw_features(rec, np.asarray(c.vg)[s], np.asarray(c.vd)[s], sim)
        n = len(sim)
        g0 = corr.g(phi, [f"{rec.family}|{c.key}"] * n, [f"{rec.family}|{c.kind}"] * n)
        out[c.key] = {"B": h.basis(phi), "g0": np.asarray(g0, float),
                      "x": np.abs(np.asarray(c.vg if c.kind == "idvg" else c.vd, float)[s]),
                      "y": np.abs(np.asarray(c.target, float)[s]), "kind": c.kind}
    return out


def prior_diag(d: dict) -> dict:
    """Per split: RMS over the held-out points of phi . delta (c_hier, c_ml) and of g_shared."""
    out = {}
    for ds in DATASETS:
        r, dset = d["res"][ds], d["ds"][ds]
        rows = []
        for x in r["rows"]:
            rec = dset.by_tag[x["tag"]]
            pts = held_points(rec, set(x["heldout"]), dset.g_shared(x["fold"]))
            B = np.concatenate([p["B"] for p in pts.values()])
            g0 = np.concatenate([p["g0"] for p in pts.values()])
            row = {"tag": x["tag"], "draw": x["draw"], "K": x["K"], "fold": x["fold"], "mu": float(x["mu"]),
                   "g0_rms": float(np.sqrt(np.mean(g0 ** 2)))}
            for arm in ("c_hier", "c_ml"):
                a = x["arms"][arm]
                pd_ = B @ np.asarray(a["delta"], float)
                row[f"{arm}_rms"] = float(np.sqrt(np.mean(pd_ ** 2)))
                row[f"{arm}_z_bound"] = int(np.sum(np.abs(np.asarray(a["z"], float)) >= h.Z_MAX - 1e-6))
                row[f"{arm}_z_norm2"] = float(a["z_norm2"])
            ratio = {arm: x["arms"][arm]["heldout_rrms"] for arm in ARMS}
            row["rrms"] = ratio
            rows.append(row)
        mus = {}
        for f, fo in r["folds"].items():
            for k, sel in fo["mu_selection"].items():
                mus.setdefault(int(k), []).append(float(sel["mu"]))
        td = [np.array(list(fo["train_deltas"].values()), float) for fo in r["folds"].values()]
        # one training delta per device: mean over the folds where the device is a training device
        per_dev = {}
        for fo in r["folds"].values():
            for t, v in fo["train_deltas"].items():
                per_dev.setdefault(t, []).append(v)
        dev_mean = np.array([np.mean(v, axis=0) for v in per_dev.values()], float)
        dev_spread = float(np.max([np.max(np.std(np.array(v, float), axis=0)) for v in per_dev.values()]))
        out[ds] = {"rows": rows, "mus": mus, "train_dev": dev_mean, "train_fold_spread": dev_spread,
                   "m": np.array([fo["prior"]["m"] for fo in r["folds"].values()], float),
                   "sd": np.array([fo["prior"]["sd"] for fo in r["folds"].values()], float),
                   "n_train": [len(t) for t in td]}
    return out


# ================================================================== tables

def t_decision(d: dict) -> rf.Table:
    dec = d["dec"]
    tab, n = dec["table"], dec["counts"]
    rows = []
    spec = (("C1", "c_vs_b", "CI upper < 1 on >= 3 of 4", f"{n['C1_ci_lt1_c_vs_b']}/4"),
            ("C2", "c_vs_a", "CI upper < 1 on 4 of 4", f"{n['C2_ci_lt1_c_vs_a']}/4"),
            ("C3", "c_vs_d", "point < 1 on 4 of 4 and CI upper < 1 on >= 3",
             f"point {n['C3_point_lt1_c_vs_d']}/4, CI {n['C3_ci_lt1_c_vs_d']}/4"))
    for cid, pair, req, cnt in spec:
        row = {"c": cid, "what": PAIR_LABEL[pair], "req": req, "cnt": cnt,
               "res": "holds" if dec["conditions"][cid] else "fails"}
        for ds in DATASETS:
            c = tab[ds][pair]
            mark = " *" if (c["ci_high"] < 1 if cid != "C3" else c["ratio"] < 1) else ""
            row[ds] = f"{r_text(c)} {c['wins']}/{c['n_used']}{mark}"
        rows.append(row)
    row = {"c": "C4", "what": "c_hier failed <= 10% and >= 8 paired", "req": "every dataset",
           "cnt": f"{sum(dec['C4_per_dataset'].values())}/4", "res": "holds" if dec["conditions"]["C4"] else "fails"}
    for ds in DATASETS:
        f = dec["failed"][ds]
        row[ds] = f"failed {f['n_failed']}/{f['n']}, paired {f['n_paired']}"
    rows.append(row)
    row = {"c": "stop check", "what": "GMR(c_hier, b_shared) point >= 0.98", "req": ">= 2 datasets for the stop "
           "outcome", "cnt": f"{n['c_vs_b_point_ge_0.98']}/4", "res": "not reached"}
    for ds in DATASETS:
        c = tab[ds]["c_vs_b"]
        row[ds] = f"{c['ratio']:.3f}" + (" (>= 0.98)" if c["ratio"] >= 0.98 else "")
    rows.append(row)
    for lab, cond, ok in outcome_rows(d):
        rows.append({"c": f"outcome {OUTCOMES.index(lab) + 1}", "what": lab, "req": cond,
                     "res": ("holds" if ok else "fails") + ("; rule output" if lab == dec["outcome"] else "")})
    cols = [{"key": "c", "head": "item", "md": "item", "kind": "text"},
            {"key": "what", "head": "statistic", "md": "statistic", "kind": "text"},
            {"key": "req", "head": "required", "md": "required", "kind": "text"}] + [
        {"key": ds, "head": DS_SHORT[ds], "md": f"{DS_SHORT[ds]}: GMR [device 95% CI] wins/paired",
         "kind": "rtext", "group": "per dataset: GMR [device 95% CI] wins/paired (* = meets the per-dataset test)"}
        for ds in DATASETS] + [
        {"key": "cnt", "head": "count", "md": "count", "kind": "rtext"},
        {"key": "res", "head": "evaluated", "md": "evaluated", "kind": "text"}]
    notes = [
        f"Pre-registration {rel(PREREG)} (commit be7d69e, before any I7-2 code was run on data); code and smoke "
        "commit 7c9aeec; queue jobs I7_2_sky130, I7_2_ihp300, I7_2_tud4, I7_2_tud295, I7_2_decide (2026-10-20).",
        "Per-device value = geometric mean over the device's defined draws (K2a, K2b, K4a, K4b) of the strict paper "
        "RRMS on the held-out curves; NaN if any of those draws failed for the arm. GMR(X, Y) = exp(mean ln(v_X / "
        "v_Y)) over devices with both values finite; 95% percentile bootstrap over devices (10000 resamples, "
        "default_rng(0)); wins = devices with v_X < v_Y.",
        "Outcomes are evaluated in the pre-registered order; the first one that holds is the rule output. "
        "C1, C2 and C3 fail and C4 holds, no stop condition is met: rule output 'inconclusive'.",
        "sky130: pmos_L0p35_W0p55 failed in every arm (draws K2b and K4b give non-finite held-out RRMS), so 11 "
        "devices are paired for c_vs_b and c_vs_a; d_scratch and e_crossinterp also failed on pmos_L0p5_W0p42 "
        "(draw K4b), so 10 are paired for c_vs_d.",
        f"Recomputed here from the per-split rows and checked against {rel(DEC)} and the four results.json files.",
    ]
    return rf.Table("01_table_i7_2_decision", "I7-2 decision: pre-registered conditions C1-C4 per dataset, counts and "
                    "the outcome list", cols, rows, notes)


def t_means(d: dict) -> rf.Table:
    rows = []
    for ds in DATASETS:
        r = d["res"][ds]
        s = r["summary"]["all"]
        per = s["per_device"]
        common = [t for t in per["a_C0"] if all(np.isfinite(per[a][t]) for a in ARMS)]
        cmp = s["comparisons"]
        gkey = {"b_shared": "b_vs_a", "c_hier": "c_vs_a", "c_ml": "cml_vs_a", "d_scratch": "d_vs_a",
                "e_crossinterp": "e_vs_a"}
        for a in ARMS:
            ag = s["aggregate"][a]
            ad = [x["arms"][a]["adapt_rrms"] for x in r["rows"]]
            ad = [v for v in ad if v is not None and np.isfinite(v)]
            ns = [x["arms"][a]["n_sim"] for x in r["rows"]]
            row = {"ds": DS_SHORT[ds], "arm": a, "desc": ARM_DESC[a],
                   "nv": f"{ag['n'] - ag['n_failed']}/{ag['n']}", "mean": ag["valid_only_mean"],
                   "cm": float(np.mean([per[a][t] for t in common])), "nc": str(len(common)),
                   "comb": ag["combined"], "med": ag["median"],
                   "ad": float(np.mean(ad)), "ns": f"{np.mean(ns):.0f}"}
            if a == "a_C0":
                row["g"] = "1 (reference)"
            else:
                c = cmp[gkey[a]]
                row["g"] = f"{r_text(c)} {c['wins']}/{c['n_used']}"
            rows.append(row)
    cols = [{"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
            {"key": "desc", "head": "definition", "md": "definition", "kind": "text"},
            {"key": "nv", "head": "valid", "md": "valid devices", "kind": "rtext"},
            {"key": "mean", "head": "all-device", "md": "all-device mean (valid devices)", "kind": "rrms",
             "group": "held-out strict paper RRMS"},
            {"key": "cm", "head": "common set", "md": "mean over devices valid in every arm", "kind": "rrms",
             "group": "held-out strict paper RRMS"},
            {"key": "nc", "head": "n", "md": "devices in the common set", "kind": "rtext",
             "group": "held-out strict paper RRMS"},
            {"key": "comb", "head": "combined", "md": "combined (nMOS / pMOS family means)", "kind": "rrms",
             "group": "held-out strict paper RRMS"},
            {"key": "med", "head": "median", "md": "median", "kind": "rrms", "group": "held-out strict paper RRMS"},
            {"key": "g", "head": "GMR(X, a_C0) [CI] wins", "md": "GMR(X, a_C0) [device 95% CI] wins/paired",
             "kind": "rtext", "family": True},
            {"key": "ad", "head": "adapt RRMS", "md": "mean adaptation-curve RRMS over splits", "kind": "rrms"},
            {"key": "ns", "head": "sims", "md": "mean NGSpice calls per split", "kind": "rtext"}]
    tim = {ds: d["res"][ds]["runtime_s"] for ds in DATASETS}
    notes = [
        "all-device = mean of the per-device values (geometric mean over the device's draws) over the devices with "
        "a finite value; for IHP and TU Delft no device failed, so it equals the all-device mean. common set = "
        "devices finite in all six arms. combined = mean of the nMOS and pMOS family means (IHP has 9 nMOS and 9 "
        "pMOS DUTs, so its combined value equals the all-device mean; sky130 has 5 nMOS and 7 pMOS, TU Delft 29 "
        "nMOS and 16 pMOS).",
        "Every scored current is a fresh NGSpice simulation of the arm's final card on all curves of the device, "
        "multiplied by exp(g) of the arm's correction in Python (b: g_shared; c, c_ml: g_shared + phi . delta; d: "
        "the from-scratch MLP; e: a's card plus the cross-curve residual). sky130 and TU Delft: ngspice-41; IHP: "
        "ngspice-46 (as I5-1).",
        "adapt RRMS = strict paper RRMS on the K adaptation curves (the fitted curves), mean over the splits with a "
        "finite value; e_crossinterp equals a_C0 on the adaptation curves. sims = NGSpice calls of the arm's fit "
        "(c_hier and c_ml reuse the cached card simulation for the z columns of the Jacobian; e needs none).",
        f"Queue wall time: sky130 {tim['sky130']:.0f} s (48 splits), IHP {tim['ihp300']:.0f} s (66 splits; 6 K4 "
        f"draws undefined), TU Delft 4.13 K {tim['tud4']:.0f} s (180 splits; 2 simulator timeouts), 295 K "
        f"{tim['tud295']:.0f} s (180 splits); 8 workers.",
    ]
    return rf.Table("03_table_i7_2_arm_means", "I7-2 held-out RRMS per arm and dataset: all-device mean, common-set "
                    "mean, combined, median, GMR against a_C0, adaptation-curve RRMS and simulations per split",
                    cols, rows, notes)


def t_per_k(d: dict, pk: dict) -> rf.Table:
    rows = []
    for ds in DATASETS:
        for lab, which in KSETS:
            p = pk[ds][lab]["pairs"]
            n = max(v["n_used"] for v in p.values())
            row = {"ds": DS_SHORT[ds], "k": lab, "n": str(n), "rank": ranking(p)}
            for name in KPAIRS:
                c = p[name]
                row[name] = f"{r_text(c)} {c['wins']}/{c['n_used']}"
            rows.append(row)
    cols = [{"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "k", "head": "draw / K", "md": "draw or K", "kind": "text"},
            {"key": "n", "head": "devices", "md": "devices (max paired)", "kind": "rtext"}] + [
        {"key": n, "head": PAIR_LABEL[n].replace("GMR", "").replace("_hier", "").replace("_shared", "")
         .replace("_scratch", "").replace("_crossinterp", "").replace("_C0", ""),
         "md": f"{PAIR_LABEL[n]} [device 95% CI] wins/paired", "kind": "rtext",
         "group": "GMR [device 95% CI] wins/paired", "family": n in ("d_vs_a", "e_vs_a")} for n in KPAIRS] + [
        {"key": "rank", "head": "order by GMR(X, a)", "md": "arms ordered by GMR(X, a_C0), lowest first",
         "kind": "text"}]
    notes = [
        "Rows K2a, K2b, K4a, K4b: per-device value = held-out RRMS of that draw alone. Rows 'K = 2', 'K = 4', 'all': "
        "geometric mean over the device's draws of that K (or of all draws), the pre-registered per-K tables; these "
        "rows are checked against the stored summaries K2, K4 and all.",
        "IHP: draws K4a and K4b are undefined for 3 of 18 DUTs (fewer than 2 held-out curves would remain), so K4 "
        "rows pair 15 DUTs. sky130: pmos_L0p35_W0p55 fails in K2b and K4b (all arms) and pmos_L0p5_W0p42 fails in "
        "K4b for d and e.",
        "Order column: arms a, b, c, c_ml, d, e sorted by GMR(X, a_C0) (a = 1); the pairs can use different device "
        "sets when an arm failed on a device (sky130 only).",
    ]
    return rf.Table("04_table_i7_2_per_draw", "I7-2 breakdown by adaptation draw and K: GMR(c, b), GMR(c, a), "
                    "GMR(c, d), GMR(c, e), GMR(d, a), GMR(e, a) per dataset", cols, rows, notes)


def t_prior(d: dict, pd_: dict) -> rf.Table:
    rows = []
    for ds in DATASETS:
        p = pd_[ds]
        r = d["res"][ds]
        mus = p["mus"]
        rr = p["rows"]
        for K in (2, 4):
            sub = [x for x in rr if x["K"] == K]
            mu_counts = {}
            for m in mus[K]:
                mu_counts[fmu(m)] = mu_counts.get(fmu(m), 0) + 1
            mu_txt = ", ".join(f"{k} x{v}" for k, v in sorted(mu_counts.items(),
                                                               key=lambda kv: float(kv[0])))
            c_rms = np.array([x["c_hier_rms"] for x in sub])
            m_rms = np.array([x["c_ml_rms"] for x in sub])
            g_rms = np.array([x["g0_rms"] for x in sub])
            rows.append({"ds": DS_SHORT[ds], "k": f"K = {K}", "nf": str(len(mus[K])), "mu": mu_txt,
                         "zero": f"{sum(1 for x in sub if math.isinf(x['mu']))}/{len(sub)}",
                         "g0": f"{np.median(g_rms):.4f}",
                         "c": f"{np.median(c_rms):.4f} [{np.min(c_rms):.4f}, {np.max(c_rms):.4f}]",
                         "cml": f"{np.median(m_rms):.4f} [{np.min(m_rms):.4f}, {np.max(m_rms):.4f}]",
                         "zb": f"{sum(1 for x in sub if x['c_ml_z_bound'] > 0)}/{len(sub)}",
                         "zn": f"{np.median([x['c_hier_z_norm2'] for x in sub]):.3f} / "
                               f"{np.median([x['c_ml_z_norm2'] for x in sub]):.3f}"})
        m, sd = p["m"], p["sd"]
        rows.append({"ds": DS_SHORT[ds], "k": "prior", "nf": str(len(m)),
                     "mu": "m (fold median): " + ", ".join(f"{v:+.3f}" for v in np.median(m, axis=0)),
                     "zero": "", "g0": "",
                     "c": "sd (fold median): " + ", ".join(f"{v:.3f}" for v in np.median(sd, axis=0)),
                     "cml": f"training devices per fold {min(p['n_train'])}-{max(p['n_train'])}"})
    cols = [{"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "k", "head": "K", "md": "K", "kind": "text"},
            {"key": "nf", "head": "folds", "md": "folds", "kind": "rtext"},
            {"key": "mu", "head": "selected mu (count over folds)", "md": "selected mu (count over folds)",
             "kind": "text"},
            {"key": "zero", "head": "z = 0", "md": "splits with mu = inf (delta = m)", "kind": "rtext"},
            {"key": "g0", "head": "RMS g_shared", "md": "median RMS of g_shared over the held-out points",
             "kind": "rtext", "group": "over the held-out points of each split"},
            {"key": "c", "head": "RMS phi.delta c_hier", "md": "RMS of phi . delta, c_hier: median [min, max]",
             "kind": "rtext", "group": "over the held-out points of each split"},
            {"key": "cml", "head": "RMS phi.delta c_ml", "md": "RMS of phi . delta, c_ml: median [min, max]",
             "kind": "rtext", "group": "over the held-out points of each split"},
            {"key": "zb", "head": "c_ml z at box", "md": "c_ml splits with a z coordinate at +/-5", "kind": "rtext"},
            {"key": "zn", "head": "|z|^2 c / c_ml", "md": "median |z|^2 c_hier / c_ml", "kind": "rtext"}]
    notes = [
        "phi = (1, a, b, a^2, ab, b^2), a = 2|Vgs|/VDD - 1, b = 2|Vds|/VDD - 1. c_hier's correction is g_shared + "
        "phi . delta and b_shared's is g_shared, so phi . delta is the difference between the two corrections in "
        "ln I; its RMS is taken over the trimmed included points of the split's held-out curves (bias values from "
        "the measured grid). The cards of b and c differ as well (each arm refits its card).",
        "mu is selected per fold and K on the training devices with their cards fixed (prereg "
        "mu_selection_empirical_bayes); mu = inf sets z = 0, delta = m (the prior mean), the same delta for every "
        "device of the fold. Ties go to the larger mu.",
        "Prior per fold: m = mean of the training deltas, S = 0.5 Sigma_hat + 0.5 diag(Sigma_hat) + 1e-6 I; "
        "recomputed from the stored training deltas and checked. Coefficient order 1, a, b, a^2, ab, b^2.",
        "c_ml (mu = 0) z lies in the box [-5, 5]^6; 'at box' counts splits with at least one coordinate at a bound.",
    ]
    return rf.Table("06_table_i7_2_prior", "I7-2 learned prior: selected mu per fold and K, prior mean and sd, and the "
                    "size of the per-device term phi . delta of c_hier and c_ml on the held-out points", cols, rows,
                    notes)


# ================================================================== figures

def fig_forest(d: dict, path: Path) -> list[str]:
    names = [n for n, _, _ in PAIRS]
    fig, axs = plt.subplots(1, 4, figsize=(22, 6.4), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    ys = np.arange(len(names))[::-1]
    for ax, ds in zip(axs, DATASETS):
        style(ax)
        cmp = d["res"][ds]["summary"]["all"]["comparisons"]
        for y, n in zip(ys, names):
            c = cmp[n]
            x = pair_of(n)[0]
            col = ARM_COLOR[x] if n.startswith("c_") is False else (BLUE if n in PAIR_ROLE else "#5b8fd6")
            lw = 2.8 if n in PAIR_ROLE else 1.8
            ax.plot([c["ci_low"], c["ci_high"]], [y, y], color=col, lw=lw, solid_capstyle="butt")
            if "cluster_bootstrap" in c:
                cb = c["cluster_bootstrap"]
                ax.plot([cb["ci_low"], cb["ci_high"]], [y + 0.22, y + 0.22], color=col, lw=0.9, alpha=0.75)
                for xx in (cb["ci_low"], cb["ci_high"]):
                    ax.plot([xx, xx], [y + 0.16, y + 0.28], color=col, lw=0.9, alpha=0.75)
            ax.scatter([c["ratio"]], [y], color=col, s=46, edgecolor=BG, zorder=4,
                       marker="D" if n in PAIR_ROLE else "o")
            ax.text(0.99, y - 0.3, f"{c['ratio']:.3f} [{c['ci_low']:.2f}, {c['ci_high']:.2f}] {c['wins']}/"
                    f"{c['n_used']}", transform=ax.get_yaxis_transform(), fontsize=7.3, ha="right", color=INK)
        ax.axvline(1.0, color=INK, ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlim(0.45, 1.75)
        ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:g}" for v in ticks])
        ax.minorticks_off()
        ax.set_ylim(-0.8, len(names) - 0.3)
        f = d["dec"]["failed"][ds]
        ax.set_title(f"{DS_LABEL[ds]}\n{f['n']} devices, c_hier failed {f['n_failed']}", fontsize=9.5, color=INK,
                     loc="left")
        ax.set_xlabel("GMR (log axis); < 1: first arm lower", fontsize=8.5)
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels([PAIR_LABEL[n] + (f" ({PAIR_ROLE[n]})" if n in PAIR_ROLE else " (not in rule)")
                            for n in names], fontsize=8.5)
    dec = d["dec"]
    fig.suptitle("I7-2 paired geometric-mean ratios of held-out strict paper RRMS (per device: geometric mean over "
                 "draws K2a, K2b, K4a, K4b); thick: 95% device-bootstrap CI; thin with caps (TU Delft): 95% "
                 "cluster-bootstrap CI over 7 geometry groups\n"
                 f"Rule output '{dec['outcome']}': C1 (CI upper of GMR(c, b) < 1) on {dec['counts']['C1_ci_lt1_c_vs_b']}"
                 f"/4 (needs 3), C2 (GMR(c, a)) on {dec['counts']['C2_ci_lt1_c_vs_a']}/4 (needs 4), C3 (GMR(c, d) "
                 f"point < 1) on {dec['counts']['C3_point_lt1_c_vs_d']}/4 (needs 4); diamonds: pairs in the rule",
                 fontsize=10, color=INK)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.83, bottom=0.09, wspace=0.08)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(RES[ds]) for ds in DATASETS] + [rel(DEC)]


def fig_per_k(d: dict, pk: dict, path: Path) -> list[str]:
    labs = [lab for lab, _ in KSETS]
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.6), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    xs = np.arange(len(labs))
    colors = {"c_vs_b": ORANGE, "c_vs_a": GREY, "c_vs_d": VIOLET, "c_vs_e": RED, "d_vs_a": "#7a9a3a",
              "e_vs_a": AQUA}
    marks = {"c_vs_b": "o", "c_vs_a": "s", "c_vs_d": "D", "c_vs_e": "^", "d_vs_a": "v", "e_vs_a": "P"}
    off = np.linspace(-0.25, 0.25, len(KPAIRS))
    for ax, ds in zip(axs, DATASETS):
        style(ax)
        for j, n in enumerate(KPAIRS):
            v = [pk[ds][lab]["pairs"][n] for lab in labs]
            ax.errorbar(xs + off[j], [c["ratio"] for c in v],
                        yerr=[[c["ratio"] - c["ci_low"] for c in v], [c["ci_high"] - c["ratio"] for c in v]],
                        fmt=marks[n], color=colors[n], ms=5, lw=1.0, capsize=0, label=PAIR_LABEL[n])
        ax.axhline(1.0, color=INK, ls="--", lw=0.9)
        for b in (2.5, 5.5):
            ax.axvline(b, color=MUTED, lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=8)
        ax.set_title(DS_LABEL[ds], fontsize=9.5, color=INK, loc="left")
    ticks = [0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
    axs[0].set_yticks(ticks)
    axs[0].set_yticklabels([f"{v:g}" for v in ticks])
    axs[0].minorticks_off()
    axs[0].set_ylim(0.33, 2.5)
    axs[0].set_ylabel("GMR with 95% device-bootstrap CI (log axis)", fontsize=9)
    hd, lb = axs[0].get_legend_handles_labels()
    fig.legend(hd, lb, fontsize=8, frameon=False, loc="lower center", ncol=6)
    fig.suptitle("I7-2 paired ratios per adaptation draw (single draw per device) and per K (geometric mean over the "
                 "K's draws); 'all' = the rule's per-device value", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.14, wspace=0.06)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(RES[ds]) for ds in DATASETS]


def fig_prior(d: dict, pd_: dict, path: Path) -> list[str]:
    fig, axs = plt.subplots(2, 4, figsize=(22, 9.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    xs = np.arange(len(COEF))
    for j, ds in enumerate(DATASETS):
        p = pd_[ds]
        ax = axs[0, j]
        style(ax)
        tr = p["train_dev"]
        ch = np.array([x for x in (r["arms"]["c_hier"]["delta"] for r in d["res"][ds]["rows"])], float)
        cm = np.array([x for x in (r["arms"]["c_ml"]["delta"] for r in d["res"][ds]["rows"])], float)
        rng = np.random.default_rng(0)
        for k in range(len(COEF)):
            ax.scatter(np.full(len(tr), k - 0.25) + rng.uniform(-0.06, 0.06, len(tr)), tr[:, k], s=9, color=GREY,
                       alpha=0.8, label="training devices (card fixed, mean over folds)" if k == 0 else None)
            ax.scatter(np.full(len(ch), k) + rng.uniform(-0.06, 0.06, len(ch)), ch[:, k], s=9, color=BLUE,
                       alpha=0.6, label="c_hier, held-out splits (MAP)" if k == 0 else None)
            ax.scatter(np.full(len(cm), k + 0.25) + rng.uniform(-0.06, 0.06, len(cm)), cm[:, k], s=9, color=AQUA,
                       alpha=0.6, label="c_ml, held-out splits (mu = 0)" if k == 0 else None)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"delta[{c}]" for c in COEF], fontsize=8)
        ax.set_title(f"{DS_LABEL[ds]}: fitted delta coefficients", fontsize=9.5, color=INK, loc="left")
        if j == 0:
            ax.set_ylabel("delta (ln I units per basis function)", fontsize=9)
            ax.legend(fontsize=7.5, frameon=False, loc="best")
        ax = axs[1, j]
        style(ax)
        rr = p["rows"]
        for K, mk in ((2, "o"), (4, "s")):
            sub = [x for x in rr if x["K"] == K]
            ax.scatter([x["g0_rms"] for x in sub], [x["c_hier_rms"] for x in sub], s=14, color=BLUE, marker=mk,
                       alpha=0.75, label=f"c_hier, K = {K}")
            ax.scatter([x["g0_rms"] for x in sub], [x["c_ml_rms"] for x in sub], s=14, color=AQUA, marker=mk,
                       alpha=0.75, label=f"c_ml, K = {K}")
        lim = max(max(x["g0_rms"] for x in rr), max(x["c_ml_rms"] for x in rr)) * 1.1
        ax.plot([0, lim], [0, lim], color=INK, lw=0.8, ls=":")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        mus = {K: ", ".join(sorted({fmu(m) for m in p["mus"][K]}, key=float)) for K in (2, 4)}
        ax.set_title(f"selected mu: K = 2 {{{mus[2]}}}; K = 4 {{{mus[4]}}}", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("RMS of g_shared over the held-out points (b's correction)", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("RMS of phi . delta over the held-out points\n(c's correction minus b's)", fontsize=9)
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle(wrap("I7-2 learned prior. Top: delta coefficients of the training devices (fitted with their starting "
                 "card fixed; one point per device, mean over the folds in which it is a training device) and of the "
                 "held-out splits (c_hier MAP, c_ml flat prior). Bottom: per split, RMS of phi . delta vs RMS of "
                 "g_shared on the held-out points (dotted: equal)", 190), fontsize=10, color=INK)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.07, wspace=0.2, hspace=0.3)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(RES[ds]) for ds in DATASETS]


def pick_examples(d: dict) -> dict:
    """Per dataset: the device with the median ln(c/d) among devices where c beats d,
    and the same among devices where d beats c (per-device values over all draws).
    Draw shown: the defined draw whose ln(c/d) is closest to the device value."""
    out = {}
    for ds in DATASETS:
        r = d["res"][ds]
        per = r["summary"]["all"]["per_device"]
        lr = {t: math.log(per["c_hier"][t] / per["d_scratch"][t]) for t in per["c_hier"]
              if np.isfinite(per["c_hier"][t]) and np.isfinite(per["d_scratch"][t])}
        picks = []
        for sign in (-1, 1):
            side = sorted([(v, t) for t, v in lr.items() if sign * v > 0])
            v, t = side[(len(side) - 1) // 2]
            rows = [x for x in r["rows"] if x["tag"] == t and np.isfinite(x["arms"]["c_hier"]["heldout_rrms"])
                    and np.isfinite(x["arms"]["d_scratch"]["heldout_rrms"])]
            best = min(rows, key=lambda x: abs(math.log(x["arms"]["c_hier"]["heldout_rrms"]
                                                        / x["arms"]["d_scratch"]["heldout_rrms"]) - v))
            picks.append({"tag": t, "lr": v, "row": best, "side": "c beats d" if sign < 0 else "d beats c"})
        out[ds] = picks
    return out


def fig_iv(ds: str, d: dict, picks: list, path: Path) -> list[str]:
    dset = d["ds"][ds]
    r = d["res"][ds]
    per = r["summary"]["all"]["per_device"]
    fig, axs = plt.subplots(2, 3, figsize=(20, 10.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for i, pk in enumerate(picks):
        x = pk["row"]
        rec = dset.by_tag[x["tag"]]
        keys = set(x["heldout"]) | set(x["adapt"])
        pts = held_points(rec, keys, dset.g_shared(x["fold"]))
        dc = np.asarray(x["arms"]["c_hier"]["delta"], float)
        dm = np.asarray(x["arms"]["c_ml"]["delta"], float)
        for j, kind in enumerate(("idvg", "idvd")):
            ax = axs[i, j]
            style(ax)
            ks = [k for k in pts if pts[k]["kind"] == kind]
            cmap = plt.get_cmap("viridis")
            for q, k in enumerate(ks):
                p = pts[k]
                col = cmap(q / max(1, len(ks) - 1))
                if k in x["adapt"]:
                    ax.scatter(p["x"], p["y"], s=11, facecolor="none", edgecolor=RED, lw=0.9, zorder=3,
                               label=f"{k} (adaptation)")
                else:
                    ax.scatter(p["x"], p["y"], s=6, color=col, zorder=2,
                               label=k if len(ks) <= 12 or q % 3 == 0 else None)
            if kind == "idvg":
                ax.set_yscale("log")
            ax.set_xlabel("|Vg| (V)" if kind == "idvg" else "|Vd| (V)", fontsize=8.5)
            ax.set_ylabel("measured |Id| (A)", fontsize=8.5)
            ax.legend(fontsize=6.3, frameon=False, ncol=2, loc="lower right" if kind == "idvg" else "upper left")
            ax.set_title(f"{x['tag']} ({pk['side']}), draw {x['draw']}: measured {'Id-Vg' if kind == 'idvg' else 'Id-Vd'}"
                         " points (red open: adaptation curves)", fontsize=8.5, color=INK, loc="left")
        ax = axs[i, 2]
        style(ax)
        for k in x["heldout"]:
            if k not in pts:
                continue
            p = pts[k]
            ls = "-" if p["kind"] == "idvg" else "--"
            ax.plot(p["x"], np.exp(p["B"] @ dc), color=BLUE, lw=1.0, ls=ls, alpha=0.8)
            ax.plot(p["x"], np.exp(p["B"] @ dm), color=AQUA, lw=1.0, ls=ls, alpha=0.8)
        ax.plot([], [], color=BLUE, lw=1.2, label="c_hier: exp(phi . delta)")
        ax.plot([], [], color=AQUA, lw=1.2, label="c_ml: exp(phi . delta)")
        ax.plot([], [], color=INK, lw=1.0, ls="-", label="Id-Vg held-out curves (x = |Vg|)")
        ax.plot([], [], color=INK, lw=1.0, ls="--", label="Id-Vd held-out curves (x = |Vd|)")
        ax.axhline(1.0, color=INK, lw=0.7)
        ax.set_xlabel("sweep voltage |V| (V)", fontsize=8.5)
        ax.set_ylabel("factor c / b correction = exp(phi . delta)", fontsize=8.5)
        ax.legend(fontsize=7, frameon=False, loc="best")
        a = x["arms"]
        txt = ("held-out RRMS, this draw / device value (all draws):\n" + "\n".join(
            f"{ARM_SHORT[k]:5s} {a[k]['heldout_rrms']:.4f} / {per[k][x['tag']]:.4f}" for k in ARMS)
            + f"\nmu {fmu(x['mu'])}; K = {x['K']}; {len(x['heldout'])} held-out curves")
        ax.text(0.99, 0.02, txt, transform=ax.transAxes, fontsize=7.3, ha="right", va="bottom", family="monospace",
                color=INK, bbox=dict(facecolor=BG, edgecolor=GREY, lw=0.6))
        ax.set_title("correction factor of c relative to b on the held-out points (Python, no simulation)",
                     fontsize=8.5, color=INK, loc="left")
    fig.suptitle(wrap(f"I7-2 {DS_LABEL[ds]}: two example devices (median device among those where c_hier beats d_scratch, "
                 "and among those where d_scratch beats c_hier). Measured points only: the arm I-V curves are not "
                 "stored (results.json keeps the RRMS, cards.json the cards; d_scratch's MLP is not stored); plotting "
                 "them needs NGSpice re-simulation of the cards, not done in this batch", 170), fontsize=9.5, color=INK)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.06, wspace=0.22, hspace=0.32)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(RES[ds]), rel(CARDS[ds])]


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)

    d = load_all()
    pk = per_k_table(d)
    pd_ = prior_diag(d)
    picks = pick_examples(d)
    items, tables = [], []
    srcs_all = [rel(RES[ds]) for ds in DATASETS]

    def add_table(t: rf.Table, desc: str, srcs: list[str]) -> None:
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    add_table(t_decision(d), "Table of the pre-registered decision: C1-C4 per dataset with GMR, device CI and wins, "
              "counts, the stop check and each outcome condition evaluated, rule output 'inconclusive'",
              [rel(PREREG), rel(DEC)] + srcs_all)
    src = fig_forest(d, folder / "02_i7_2_gmr_forest.png")
    items.append(("02_i7_2_gmr_forest.png", "Forest plot, one panel per dataset: GMR with 95% device-bootstrap CI for "
                  "the 9 pairs of decision.json (c vs b, a, d, e, c_ml; b, d, e, c_ml vs a), TU Delft cluster CI, "
                  "reference line at 1", src))
    add_table(t_means(d), "Table of held-out RRMS per arm and dataset (all-device, common set, combined, median), "
              "GMR(X, a_C0), adaptation-curve RRMS, NGSpice calls per split", srcs_all)
    add_table(t_per_k(d, pk), "Table of the per-draw (K2a, K2b, K4a, K4b) and per-K paired ratios GMR(c, b), "
              "GMR(c, a), GMR(c, d), GMR(c, e), GMR(d, a), GMR(e, a) with the arm order by GMR(X, a_C0)", srcs_all)
    src = fig_per_k(d, pk, folder / "05_i7_2_per_draw.png")
    items.append(("05_i7_2_per_draw.png", "Per-draw and per-K paired ratios with device CIs, one panel per "
                  "dataset", src))
    add_table(t_prior(d, pd_), "Table of the learned prior: selected mu per fold and K, splits with z = 0, prior mean "
              "and sd, RMS of g_shared and of phi . delta (c_hier, c_ml) over the held-out points, c_ml z at the box",
              srcs_all)
    src = fig_prior(d, pd_, folder / "07_i7_2_prior_delta.png")
    items.append(("07_i7_2_prior_delta.png", "Learned-prior diagnostic: delta coefficients of training devices, "
                  "c_hier and c_ml per dataset (top); per split RMS of phi . delta vs RMS of g_shared over the "
                  "held-out points (bottom)", src))
    for j, ds in enumerate(DATASETS):
        name = f"{8 + j:02d}_i7_2_examples_{ds}.png"
        src = fig_iv(ds, d, picks[ds], folder / name)
        items.append((name, f"{DS_LABEL[ds]}: two example devices ({picks[ds][0]['tag']}, c beats d; "
                      f"{picks[ds][1]['tag']}, d beats c): measured Id-Vg and Id-Vd points of the shown draw "
                      "(adaptation curves marked), the factor exp(phi . delta) of c_hier and c_ml on the held-out "
                      "points, per-arm held-out RRMS; no arm I-V curves (not stored)", src))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i7_2.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "",
          f"Generated {now} by {gen} from stored files only. RRMS is the strict paper metric on the held-out curves "
          "of each split (frozen inclusion). I7-2 values are on DEV devices (development evidence); REAL_TEST and "
          "TEST are not loaded.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    dec, tab = d["dec"], d["dec"]["table"]

    def row4(pair: str) -> str:
        return ", ".join(f"{DS_SHORT[ds]} {r_text(tab[ds][pair])} {tab[ds][pair]['wins']}/{tab[ds][pair]['n_used']}"
                         for ds in DATASETS)

    def k_line(ds: str) -> str:
        return "; ".join(f"{lab}: c/b {pk[ds][lab]['pairs']['c_vs_b']['ratio']:.3f}, c/d "
                         f"{pk[ds][lab]['pairs']['c_vs_d']['ratio']:.3f}, c/e {pk[ds][lab]['pairs']['c_vs_e']['ratio']:.3f}"
                         f", order {ranking(pk[ds][lab]['pairs'])}" for lab in ("K = 2", "K = 4"))

    zero = {ds: sum(1 for x in pd_[ds]["rows"] if math.isinf(x["mu"])) for ds in DATASETS}
    med = {ds: (np.median([x["c_hier_rms"] for x in pd_[ds]["rows"]]),
                np.median([x["c_ml_rms"] for x in pd_[ds]["rows"]]),
                np.median([x["g0_rms"] for x in pd_[ds]["rows"]])) for ds in DATASETS}
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen} from stored files only (no NGSpice, no training).", "",
              "## Contents", "",
              "- I7-2: hierarchical bias-only correction g_d = g_shared + phi . delta_d (phi = 1, a, b, a^2, ab, b^2 "
              "in a = 2|Vgs|/VDD - 1, b = 2|Vds|/VDD - 1; g_shared = the stored fold M_bias corrections of I3-2, "
              "I5-1, I6-1), with an empirical-Bayes Gaussian prior over delta from the fold's training devices and "
              "a MAP fit of card and delta on K = 2 or 4 curves of the held-out device (draws K2a, K2b, K4a, K4b), "
              "scored by strict paper RRMS on the device's other included curves. Datasets: sky130 77 K REAL_DEV "
              "(12, leave-one-device-out), IHP SG13G2 300 K DEV (18, leave-one-DUT-out within class), TU Delft 40 nm "
              "4.13 K and 295 K DEV (45, leave-one-geometry-out, 7 groups). Pre-registration "
              f"`{rel(PREREG)}` (commit be7d69e); code and smoke commit 7c9aeec; queue jobs I7_2_sky130 "
              f"({d['res']['sky130']['runtime_s']:.0f} s), I7_2_ihp300 ({d['res']['ihp300']['runtime_s']:.0f} s), "
              f"I7_2_tud4 ({d['res']['tud4']['runtime_s']:.0f} s), I7_2_tud295 "
              f"({d['res']['tud295']['runtime_s']:.0f} s), I7_2_decide, 2026-10-20.",
              f"- Rule output: {dec['outcome']}. C1 GMR(c_hier, b_shared), CI upper < 1 on "
              f"{dec['counts']['C1_ci_lt1_c_vs_b']}/4 (needs 3): {row4('c_vs_b')}. C2 GMR(c_hier, a_C0), CI upper "
              f"< 1 on {dec['counts']['C2_ci_lt1_c_vs_a']}/4 (needs 4): {row4('c_vs_a')}. C3 GMR(c_hier, d_scratch) "
              f"point < 1 on {dec['counts']['C3_point_lt1_c_vs_d']}/4 (needs 4): {row4('c_vs_d')}. C4 holds on all "
              "four (sky130: 1 of 12 devices failed in every arm; 10 paired for c vs d).",
              f"- Not in the rule: GMR(c_hier, e_crossinterp) {row4('c_vs_e')}. GMR(e_crossinterp, a_C0) "
              f"{row4('e_vs_a')}. GMR(c_hier, c_ml) {row4('c_vs_cml')}. GMR(b_shared, a_C0) {row4('b_vs_a')}. "
              f"GMR(d_scratch, a_C0) {row4('d_vs_a')}.",
              "- Per K (table 04, figure 05): " + " | ".join(f"{DS_SHORT[ds]}: {k_line(ds)}" for ds in DATASETS)
              + ".",
              f"- Learned prior (table 06, figure 07): the selected mu is inf (delta = prior mean m, z = 0) in "
              + ", ".join(f"{DS_SHORT[ds]} {zero[ds]}/{len(pd_[ds]['rows'])}" for ds in DATASETS)
              + " splits. Median RMS over the held-out points of phi . delta (c_hier / c_ml) and of g_shared: "
              + ", ".join(f"{DS_SHORT[ds]} {med[ds][0]:.4f} / {med[ds][1]:.4f} / {med[ds][2]:.4f}" for ds in DATASETS)
              + ".",
              "- Figures 08-11 (example devices) show measured points only. The arm I-V curves are not stored: "
              "results.json keeps per-split held-out and adaptation RRMS, cards.json keeps the final card of each "
              "arm, and d_scratch's per-device MLP is not stored. Plotting the arm curves needs an NGSpice "
              "re-simulation of the stored cards (and a refit of d_scratch's MLP), which this batch does not run. "
              "The third panel shows exp(phi . delta), the factor between c's and b's correction, computed in "
              "Python on the measured bias points; it is a correction factor, not a current.",
              "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    summary = [
        f"I7-2 rule output (hierarchical correction, K = 2 / 4 adaptation curves, 4 datasets): {dec['outcome']}. "
        f"C1 GMR(c_hier, b_shared) CI upper < 1 on {dec['counts']['C1_ci_lt1_c_vs_b']}/4: {row4('c_vs_b')}.",
        f"C2 GMR(c_hier, a_C0) on {dec['counts']['C2_ci_lt1_c_vs_a']}/4: {row4('c_vs_a')}. C3 GMR(c_hier, "
        f"d_scratch) point < 1 on {dec['counts']['C3_point_lt1_c_vs_d']}/4: {row4('c_vs_d')}.",
        f"Not in the rule: GMR(c_hier, e_crossinterp) {row4('c_vs_e')}; GMR(c_hier, c_ml) {row4('c_vs_cml')}.",
        "Selected mu = inf (delta = prior mean) in " + ", ".join(f"{DS_SHORT[ds]} {zero[ds]}/{len(pd_[ds]['rows'])}"
                                                               for ds in DATASETS) + " splits.",
    ]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    print(f"consistency checks passed ({d['n_checks']} split/ratio checks plus aggregates, priors, mu, deltas, "
          "decision)")
    for ds in DATASETS:
        for lab, _ in KSETS:
            p = pk[ds][lab]["pairs"]
            print(f" {ds:7s} {lab:6s} " + " ".join(f"{n} {p[n]['ratio']:.3f} [{p[n]['ci_low']:.3f}, "
                                                   f"{p[n]['ci_high']:.3f}] {p[n]['wins']}/{p[n]['n_used']}"
                                                   for n in KPAIRS) + f" | {ranking(p)}")
    for ds in DATASETS:
        p = pd_[ds]
        print(f" prior {ds}: mu K2 {sorted(fmu(m) for m in p['mus'][2])} K4 {sorted(fmu(m) for m in p['mus'][4])}; "
              f"z=0 {zero[ds]}/{len(p['rows'])}; median RMS phi.delta c {med[ds][0]:.4f} c_ml {med[ds][1]:.4f} "
              f"g_shared {med[ds][2]:.4f}; m median {np.round(np.median(p['m'], axis=0), 3).tolist()} sd median "
              f"{np.round(np.median(p['sd'], axis=0), 3).tolist()}; training-delta fold spread max "
              f"{p['train_fold_spread']:.4f}")
        for pkk in picks[ds]:
            print(f"   example {pkk['side']}: {pkk['tag']} draw {pkk['row']['draw']} ln(c/d) {pkk['lr']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
