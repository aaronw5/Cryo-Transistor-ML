#!/usr/bin/env python3
"""Report batch for exploration probes E2 (three-seed decision run), E3 (full
tier-M decision run and tier-W check) and E4 (43-parameter held-out-curve runs,
splits vd, sparse, vg).

Reads result JSON files only (no simulation, no training) and writes PNGs,
``tables.md`` and ``README.md`` to
``out/research/report/<date>_explore_e2_e3_e4/``, then adds the batch to
``out/research/report/index.json`` and ``INDEX.md``.

The E4 pre-registered rule is evaluated on the decision run (p43 vd) from the
stored ``decision`` block; for the secondary splits (sparse, vg) the same
function (``e4_analyze.decide``) is applied to the stored summaries for
reporting only.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_e2_e3_e4.py \
      --date 2026-10-19
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "explore"))
import report_figs as rf  # noqa: E402  (Table rendering, index update, style)
import matplotlib.pyplot as plt  # noqa: E402  (configured by report_figs)

BG, INK, MUTED, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9892"
BLUE, ORANGE, AQUA, VIOLET, RED, YELLOW = rf.BLUE, rf.ORANGE, rf.AQUA, rf.VIOLET, rf.RED, rf.YELLOW
DPI = 150

EXP = ROOT / "out/research/explore"
SPLIT = EXP / "real_split.json"
E2 = EXP / "E2"
E2_EVAL = E2 / "eval"
E2_PREREG = E2 / "prereg.json"
E2_TABLE = E2 / "table.json"
D7 = ROOT / "out/research/polished/direct7_fd/real/summary.json"
E3 = EXP / "E3"
E3_RUN = {"M": E3 / "runs/full/tier_M", "W": E3 / "runs/full/tier_W"}
E3_OUTLIER = "nmos_L1_W3__2"
E4 = EXP / "E4"
E4_SPLITS = ["vd", "sparse", "vg"]
E4_CAND = ["ens", "laplace", "learned"]
E4_METHODS = ["single", "ens", "laplace", "conformal", "crossinterp", "crossinterp_conformal",
              "learned"]
E4_COLOR = {"single": INK, "ens": BLUE, "laplace": ORANGE, "learned": VIOLET,
            "crossinterp": AQUA}
E4_LABEL = {"single": "single best-fit card", "ens": "near-optimal ensemble (median)",
            "laplace": "Laplace (mean of draws)", "learned": "learned discrepancy (kernel ridge)",
            "crossinterp": "crossinterp (non-learned control)",
            "conformal": "conformal (single + interval)",
            "crossinterp_conformal": "crossinterp + conformal interval"}
ARM_COLOR = {"N0": GREY, "M": BLUE, "S": ORANGE, "nn": AQUA}
ARM_LABEL = {"N0": "N0 no augmentation", "M": "M measurement-only augmentation",
             "S": "S structured + measurement augmentation", "nn": "nearest bank sample"}
GROUP_LABEL = {"real": "REAL_DEV", "a0": "tier a set 0", "a1": "tier a set 1",
               "c0": "tier c set 0", "c1": "tier c set 1"}


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def load(p: Path):
    return json.loads(Path(p).read_text())


def fnum(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def style(ax) -> None:
    ax.set_facecolor(BG)
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def yn(b) -> str:
    return "—" if b is None else ("yes" if b else "no")


def fam(per: dict) -> dict:
    """all-device, nMOS, pMOS, combined and median over finite values."""
    v = {t: x for t, x in per.items() if np.isfinite(x)}
    s = rf.family_stats(v)
    s["median"] = float(np.median(list(v.values()))) if v else None
    s["n"] = len(v)
    return s


def real_dev() -> list[str]:
    return sorted(load(SPLIT)["REAL_DEV"], key=rf.device_key)


# ================================================================== E2

def e2_seed_rows(stage: str, arm: str, group: str, seeds) -> dict | None:
    """{tag: [record per seed]} for the seeds whose file has the group."""
    per = {}
    for s in seeds:
        p = E2_EVAL / f"{stage}_{arm}_s{s}.json"
        if not p.exists():
            return None
        rows = load(p).get(group)
        if not rows:
            return None
        for t, r in rows.items():
            per.setdefault(t, []).append(r)
    return per


def e2_stats(stage: str, arm: str, group: str, seeds) -> dict | None:
    per = e2_seed_rows(stage, arm, group, seeds)
    if per is None:
        return None
    pen = {t: float(np.mean([r["penalized_rrms"] for r in v])) for t, v in per.items()}
    valid = [fnum(r["strict_rrms"]) for v in per.values() for r in v if r["status"] == "valid"]
    out = fam(pen)
    out.update({"per_device": pen, "per_seed": {t: [r["penalized_rrms"] for r in v]
                                                for t, v in per.items()},
                "failed_seed": {t: [r["status"] != "valid" for r in v] for t, v in per.items()},
                "n_failed": int(sum(r["status"] != "valid" for v in per.values() for r in v)),
                "valid_mean": float(np.mean(valid)) if valid else None,
                "n_pairs": int(sum(len(v) for v in per.values())),
                "n_seeds": min(len(v) for v in per.values()),
                "local_abs_err": float(np.nanmean([fnum(r.get("local_abs_err"))
                                                   for v in per.values() for r in v]))
                if any(r.get("local_abs_err") is not None for v in per.values() for r in v)
                else None})
    return out


def e2_decision(seeds) -> dict:
    D = 0.173493
    n0 = e2_stats("raw", "N0", "real", seeds)
    s_ = e2_stats("raw", "S", "real", seeds)
    m = e2_stats("raw", "M", "real", seeds)
    r0, rs = n0["all"], s_["all"]
    wins = sum(s_["per_device"][t] < n0["per_device"][t] for t in n0["per_device"])
    ta0 = np.mean([e2_stats("raw", "N0", g, seeds)["all"] for g in ("a0", "a1")])
    taS = np.mean([e2_stats("raw", "S", g, seeds)["all"] for g in ("a0", "a1")])
    checks = {"C1": rs <= 0.70 * r0, "C2": (rs - D) <= 0.70 * (r0 - D), "C3": wins >= 8,
              "C4_guard": bool(taS <= 1.5 * ta0 + 0.01), "K1": rs > 0.85 * r0, "K2": wins < 6}
    if checks["K1"] or checks["K2"]:
        verdict = "KILL"
    elif checks["C1"] and checks["C2"] and checks["C3"] and checks["C4_guard"]:
        verdict = "CONTINUE"
    else:
        verdict = "INCONCLUSIVE"
    return {"R_N0": r0, "R_S": rs, "R_M": m["all"], "D": D, "ratio": rs / r0,
            "gap_ratio": (rs - D) / (r0 - D), "wins": wins, "ta_N0": ta0, "ta_S": taS,
            "attr": rs / m["all"], "checks": checks, "verdict": verdict}


def table_e2_decision(full: dict, light: dict, stored: dict) -> rf.Table:
    assert abs(full["R_S"] - stored["R_raw_S"]) < 1e-9 and full["verdict"] == stored["verdict"]
    assert full["checks"] == stored["checks"]
    rows = []

    def add(check, rule, vf, vl, of, ol, role):
        rows.append({"check": check, "rule": rule, "full": vf, "light": vl, "of": of, "ol": ol,
                     "role": role})

    add("C1", "R_raw(S) <= 0.70 × R_raw(N0)", f"{full['ratio']:.3f}", f"{light['ratio']:.3f}",
        yn(full["checks"]["C1"]), yn(light["checks"]["C1"]), "continue (all of C1-C4)")
    add("C2", "(R(S) − D) <= 0.70 × (R(N0) − D), D = 0.1735", f"{full['gap_ratio']:.3f}",
        f"{light['gap_ratio']:.3f}", yn(full["checks"]["C2"]), yn(light["checks"]["C2"]),
        "continue (all of C1-C4)")
    add("C3", "S < N0 on >= 8 of 12 devices", f"{full['wins']}/12", f"{light['wins']}/12",
        yn(full["checks"]["C3"]), yn(light["checks"]["C3"]), "continue (all of C1-C4)")
    add("C4 guard", "tier a: R(S) <= 1.5 × R(N0) + 0.01",
        f"{full['ta_S']:.2f} vs {1.5 * full['ta_N0'] + 0.01:.2f}",
        f"{light['ta_S']:.2f} vs {1.5 * light['ta_N0'] + 0.01:.2f}",
        yn(full["checks"]["C4_guard"]), yn(light["checks"]["C4_guard"]), "continue (all of C1-C4)")
    add("K1", "R_raw(S) > 0.85 × R_raw(N0)", f"{full['ratio']:.3f}", f"{light['ratio']:.3f}",
        "met" if full["checks"]["K1"] else "not met",
        "met" if light["checks"]["K1"] else "not met", "kill (any of K1, K2)")
    add("K2", "S < N0 on fewer than 6 of 12 devices", f"{full['wins']}/12", f"{light['wins']}/12",
        "met" if full["checks"]["K2"] else "not met",
        "met" if light["checks"]["K2"] else "not met", "kill (any of K1, K2)")
    add("attribution", "R_raw(S) <= 0.90 × R_raw(M)", f"{full['attr']:.3f}", f"{light['attr']:.3f}",
        yn(full["attr"] <= 0.90), yn(light["attr"] <= 0.90), "secondary, not a decision input")
    add("verdict", "", "", "", full["verdict"], light["verdict"] + " (exploratory)", "")
    cols = [
        {"key": "check", "head": "check", "md": "check", "kind": "text"},
        {"key": "rule", "head": "pre-registered rule", "md": "pre-registered rule", "kind": "text"},
        {"key": "full", "head": "seeds 0-2", "group": "value", "md": "value, seeds 0-2", "kind": "rtext"},
        {"key": "light", "head": "seed 0", "group": "value", "md": "value, seed 0 (light run)", "kind": "rtext"},
        {"key": "of", "head": "seeds 0-2", "group": "outcome", "md": "outcome, seeds 0-2 (decision)", "kind": "rtext"},
        {"key": "ol", "head": "seed 0", "group": "outcome", "md": "outcome, seed 0", "kind": "rtext"},
        {"key": "role", "head": "role", "md": "role", "kind": "text"},
    ]
    notes = [
        f"R_raw(arm) = mean over the 12 REAL_DEV devices of the per-device penalized paper RRMS "
        f"(PENALTY 10 per failed included curve) of the predicted seven-parameter card, one "
        f"forward pass + one NGSpice validation, averaged over seeds per device first. Seeds 0-2: "
        f"R(N0) {full['R_N0']:.3f}, R(M) {full['R_M']:.3f}, R(S) {full['R_S']:.3f}. Seed 0: "
        f"R(N0) {light['R_N0']:.3f}, R(M) {light['R_M']:.3f}, R(S) {light['R_S']:.3f}.",
        "D = 0.173493: direct NGSpice seven-parameter fit + FD on REAL_DEV (fixed before the "
        "pre-registration). tier a value = mean of the set-0 and set-1 means (36 targets).",
        "The seed-0 column reproduces the light run of commit 773abe3 from the same seed-0 files; "
        "the decision uses seeds 0-2 (queue jobs E2_full and E2_fd_sim, 2026-10-19 11:26-11:52).",
    ]
    return rf.Table("01_table_e2_decision",
                    "E2 decision: one-pass seven-parameter network with output augmentation, "
                    "pre-registered checks on REAL_DEV raw RRMS", cols, rows, notes)


def fig_e2_per_device(path: Path) -> list[str]:
    tags = real_dev()
    x = np.arange(len(tags))
    st = {a: e2_stats("raw", a, "real", [0, 1, 2] if a != "nn" else [0])
          for a in ("N0", "M", "S", "nn")}
    released = {t: r["penalized_rrms"] for t, r in load(E2_EVAL / "refs.json")["real"].items()}
    d7 = load(D7)["devices"]
    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    off = {"N0": -0.24, "M": -0.08, "S": 0.08, "nn": 0.24}
    mk = {"N0": "o", "M": "D", "S": "s", "nn": "^"}
    for a in ("N0", "M", "S", "nn"):
        s = st[a]
        for i, t in enumerate(tags):
            seeds = s["per_seed"][t]
            fails = s["failed_seed"][t]
            if a != "nn":
                ax.scatter([i + off[a]] * len(seeds), seeds, s=10, color=ARM_COLOR[a],
                           alpha=0.35, zorder=2, lw=0)
            for v, f in zip(seeds, fails):
                if f:
                    ax.scatter(i + off[a], v, marker="x", s=30, color=RED, lw=1.4, zorder=5)
        lab = (f"{ARM_LABEL[a]}: all-device {s['all']:.3f}, combined {s['combined']:.3f}, "
               f"median {s['median']:.3f}, failed pairs {s['n_failed']}/{s['n_pairs']}")
        if a != "nn":
            lab += " (seed mean; small dots: seeds)"
        ax.scatter(x + off[a], [s["per_device"][t] for t in tags], marker=mk[a], s=34,
                   color=ARM_COLOR[a], zorder=4, label=lab, edgecolor=BG, lw=0.6)
    rel_v = [released[t] for t in tags]
    d7_v = [d7[t]["rrms"] for t in tags]
    ax.hlines(rel_v, x - 0.36, x + 0.36, color=INK, lw=1.0, ls=":", zorder=3,
              label=f"released 77 K card: all-device {np.mean(rel_v):.3f}")
    ax.hlines(d7_v, x - 0.36, x + 0.36, color=AQUA, lw=1.6, zorder=3,
              label=f"D: direct NGSpice 7-param fit + FD: all-device {np.mean(d7_v):.3f}")
    ax.scatter([], [], marker="x", color=RED, label="red ×: device-seed pair with a failed "
               "included curve (penalized value)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([rf.device_label(t) for t in tags], rotation=30, ha="right", fontsize=8)
    nm = sum(t.startswith("nmos") for t in tags)
    ax.axvline(nm - 0.5, color=GREY, lw=0.8, ls=":")
    ax.text(nm / 2 - 0.5, 1.02, "nMOS", transform=ax.get_xaxis_transform(), ha="center",
            fontsize=8, color=MUTED)
    ax.text(nm + (len(tags) - nm) / 2 - 0.5, 1.02, "pMOS", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=8, color=MUTED)
    ax.set_ylabel("raw paper RRMS, penalized (one forward pass, NGSpice-validated)", fontsize=9,
                  color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=1,
              bbox_to_anchor=(0.0, -0.17))
    ax.set_ylim(0.02, 400)
    wins = sum(st["S"]["per_device"][t] < st["N0"]["per_device"][t] for t in tags)
    ax.set_title(f"E2 on REAL_DEV, seeds 0-2: S / N0 = {st['S']['all'] / st['N0']['all']:.3f} "
                 f"(kill if > 0.85), S < N0 on {wins}/12 devices", fontsize=9.5, color=INK,
                 pad=14)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(E2_EVAL / f"raw_{a}_s{s}.json") for a in ("N0", "M", "S") for s in (0, 1, 2)] + \
        [rel(E2_EVAL / "raw_nn_s0.json"), rel(E2_EVAL / "refs.json"), rel(D7)]


def table_e2_results() -> rf.Table:
    rows = []
    for stage in ("raw", "fd"):
        for a in ("N0", "M", "S", "nn"):
            seeds = [0, 1, 2] if a != "nn" else [0]
            r = e2_stats(stage, a, "real", seeds)
            sims = {g: e2_stats(stage, a, g, seeds) for g in ("a0", "a1", "c0", "c1")}
            ta = [sims[g] for g in ("a0", "a1") if sims[g]]
            tc = [sims[g] for g in ("c0", "c1") if sims[g]]
            rows.append({
                "stage": {"raw": "raw (one pass)", "fd": "FD polish (100 nfev)"}[stage],
                "arm": ARM_LABEL[a].split(" ")[0] if a != "nn" else "nearest bank",
                "seeds": len(seeds), "all": r["all"], "combined": r["combined"],
                "nmos": r["nmos"], "pmos": r["pmos"], "median": r["median"],
                "valid": r["valid_mean"], "failed": f"{r['n_failed']}/{r['n_pairs']}",
                "ta": float(np.mean([s["all"] for s in ta])) if ta else None,
                "tc": float(np.mean([s["all"] for s in tc])) if tc else None,
                "tam": float(np.mean([s["median"] for s in ta])) if ta else None,
                "tcm": float(np.mean([s["median"] for s in tc])) if tc else None,
                "tfail": int(sum(s["n_failed"] for s in ta + tc)),
            })
    refs = load(E2_EVAL / "refs.json")
    relv = {t: r["penalized_rrms"] for t, r in refs["real"].items()}
    rs = fam(relv)
    tc_truth = [fam({t: r["penalized_rrms"] for t, r in refs[g].items()}) for g in ("c0", "c1")]
    rows.append({"stage": "reference", "arm": "released 77 K card", "seeds": None,
                 "all": rs["all"], "combined": rs["combined"], "nmos": rs["nmos"],
                 "pmos": rs["pmos"], "median": rs["median"], "valid": rs["all"], "failed": "0/12",
                 "ta": None, "tc": None, "tam": None, "tcm": None, "tfail": None})
    d7 = fam({t: v["rrms"] for t, v in load(D7)["devices"].items() if t in real_dev()})
    rows.append({"stage": "reference", "arm": "D: direct NGSpice 7 + FD", "seeds": None,
                 "all": d7["all"], "combined": d7["combined"], "nmos": d7["nmos"],
                 "pmos": d7["pmos"], "median": d7["median"], "valid": d7["all"], "failed": "0/12",
                 "ta": None, "tc": None, "tam": None, "tcm": None, "tfail": None})
    rows.append({"stage": "reference", "arm": "true seven values (tier c)", "seeds": None,
                 "all": None, "combined": None, "nmos": None, "pmos": None, "median": None,
                 "valid": None, "failed": "", "ta": 0.0,
                 "tc": float(np.mean([s["all"] for s in tc_truth])), "tam": 0.0,
                 "tcm": float(np.mean([s["median"] for s in tc_truth])), "tfail": None})
    cols = [
        {"key": "stage", "head": "stage", "md": "stage", "kind": "text"},
        {"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
        {"key": "seeds", "head": "seeds", "md": "seeds", "kind": "int"},
        {"key": "all", "head": "all-device", "group": "REAL_DEV (12), penalized", "md": "REAL_DEV all-device", "kind": "rrms"},
        {"key": "combined", "head": "combined", "group": "REAL_DEV (12), penalized", "md": "REAL_DEV combined", "kind": "rrms"},
        {"key": "nmos", "head": "nMOS", "group": "REAL_DEV (12), penalized", "md": "REAL_DEV nMOS", "kind": "rrms", "family": True},
        {"key": "pmos", "head": "pMOS", "group": "REAL_DEV (12), penalized", "md": "REAL_DEV pMOS", "kind": "rrms", "family": True},
        {"key": "median", "head": "median", "group": "REAL_DEV (12), penalized", "md": "REAL_DEV median", "kind": "rrms"},
        {"key": "valid", "head": "valid-only", "group": "REAL_DEV (12), penalized", "md": "REAL_DEV valid-only mean", "kind": "rrms"},
        {"key": "failed", "head": "failed", "group": "REAL_DEV (12), penalized", "md": "REAL_DEV failed device-seed pairs", "kind": "rtext"},
        {"key": "ta", "head": "tier a mean", "group": "simulated (36 + 36 targets)", "md": "tier a mean", "kind": "rrms"},
        {"key": "tam", "head": "tier a med.", "group": "simulated (36 + 36 targets)", "md": "tier a median", "kind": "rrms"},
        {"key": "tc", "head": "tier c mean", "group": "simulated (36 + 36 targets)", "md": "tier c mean", "kind": "rrms"},
        {"key": "tcm", "head": "tier c med.", "group": "simulated (36 + 36 targets)", "md": "tier c median", "kind": "rrms"},
        {"key": "tfail", "head": "failed", "group": "simulated (36 + 36 targets)", "md": "simulated failed target-seed pairs", "kind": "int"},
    ]
    notes = [
        "Per-device value = seed mean of the penalized paper RRMS (strict frozen inclusion; "
        "PENALTY 10 per failed included curve). all-device = mean over 12 devices; combined = mean "
        "of the nMOS (5) and pMOS (7) family means; valid-only = mean of the strict RRMS over "
        "device-seed pairs with no failed included curve.",
        "FD polish: cryoml.ngspice_fit.polish in the round-1 seven box from the predicted card, "
        "max_nfev 100, diff_step 0.01, irls_rounds 1, identical for every arm (secondary, not a "
        "decision input). FD on the simulated sets was run by queue job E2_fd_sim.",
        "tier a / tier c mean = mean of the set-0 and set-1 means (18 devices each); medians "
        "likewise. Tier c targets for S are partly in-distribution by construction "
        "(same component types as the tier c generator).",
        "The nearest bank sample uses no trained network: the device's own seven_r1 bank sample "
        "closest to the target in standardized signed-log features.",
    ]
    return rf.Table("03_table_e2_results",
                    "E2 raw and FD-polished RRMS per arm on REAL_DEV and the simulated sets",
                    cols, rows, notes)


# ================================================================== E3

E3_METHODS = {"O4": ["lrn", "par3", "par14", "par3_pop", "pt_interp", "lrn_mean", "lrn_noH",
                     "lrn_param"],
              "O2": ["lrn", "par3_pop", "par3_g1", "par14_g1", "pt_interp", "lrn_mean",
                     "lrn_noH", "lrn_param"]}
E3_CONTROL = {"O4": "par3", "O2": "par3_pop"}
HELD = ("110", "190", "260")


def e3_per_device(tier: str) -> dict:
    """{key: {device: held-out mean (NaN if any held-out cell is non-finite)}}"""
    out = {}
    for f in sorted((E3_RUN[tier] / "eval").glob("*.json")):
        d = load(f)
        for k, cells in d.items():
            vals = [fnum(cells.get(T)) for T in HELD]
            out.setdefault(k, {})[f.stem] = float(np.mean(vals)) if all(
                np.isfinite(vals)) else math.nan
    return out


def e3_outlier() -> dict:
    d = load(E3_RUN["M"] / "eval" / f"{E3_OUTLIER}.json")
    meth = {k: v for k, v in d.items() if not k.startswith("floor")}
    hm = [float(np.mean([v[T] for T in HELD])) for v in meth.values()]
    h110 = [v["110"] for v in meth.values()]
    fl = d["floor|floor"]
    return {"lo": min(hm), "hi": max(hm), "lo110": min(h110), "hi110": max(h110),
            "floor": float(np.mean([fl[T] for T in HELD]))}


def table_e3_decision(summ: dict) -> rf.Table:
    rows = []
    for tier in ("M", "W"):
        s = summ[tier]
        for S in ("O4", "O2"):
            d = s["decisions"][S]
            lc = d["lrn_vs_control"]
            lp = d["lrn_vs_pt_interp"]
            ff = d["failed_fraction"]
            c_r = lc["reduction"] >= 0.25
            c_ci = lc["ci95"][0] > 0
            c_pt = lp["reduction"] >= 0.10
            c_ff = all(v <= 0.10 for v in ff.values())
            rows.append({
                "tier": f"tier {tier}" + (" (decision)" if tier == "M" else " (secondary)"),
                "S": S, "control": d["control"],
                "R": lc["reduction"], "ci": f"[{lc['ci95'][0]:.3f}, {lc['ci95'][1]:.3f}]",
                "c_r": yn(c_r), "c_ci": yn(c_ci),
                "rp": lp["reduction"], "c_pt": yn(c_pt),
                "ff": f"{max(ff.values()):.3f}", "c_ff": yn(c_ff),
                "cont": yn(c_r and c_ci and c_pt and c_ff),
                "k_r": yn(lc["reduction"] < 0.10), "k_pt": yn(lp["reduction"] < 0),
                "verdict": s["verdict_rule_output"],
            })
    cols = [
        {"key": "tier", "head": "run", "md": "run", "kind": "text"},
        {"key": "S", "head": "S", "md": "observed temperatures S", "kind": "text"},
        {"key": "control", "head": "control", "md": "control", "kind": "text"},
        {"key": "R", "head": "R_S", "group": "lrn vs control", "md": "R_S = 1 − lrn/control", "kind": "rrms"},
        {"key": "ci", "head": "95% CI", "group": "lrn vs control", "md": "R_S bootstrap 95% CI", "kind": "rtext"},
        {"key": "c_r", "head": ">= 0.25", "group": "continue checks", "md": "R_S >= 0.25", "kind": "rtext"},
        {"key": "c_ci", "head": "CI lo > 0", "group": "continue checks", "md": "CI lower bound > 0", "kind": "rtext"},
        {"key": "rp", "head": "vs pt_interp", "group": "continue checks", "md": "1 − lrn/pt_interp", "kind": "rrms"},
        {"key": "c_pt", "head": ">= 0.10", "group": "continue checks", "md": "lrn >= 10% below pt_interp", "kind": "rtext"},
        {"key": "ff", "head": "max failed", "group": "continue checks", "md": "max failed-cell fraction (lrn, control, pt_interp)", "kind": "rtext"},
        {"key": "c_ff", "head": "<= 0.10", "group": "continue checks", "md": "failed <= 10%", "kind": "rtext"},
        {"key": "cont", "head": "all", "group": "continue checks", "md": "continue for S", "kind": "rtext"},
        {"key": "k_r", "head": "R_S < 0.10", "group": "kill conditions", "md": "R_S < 0.10", "kind": "rtext"},
        {"key": "k_pt", "head": "lrn > pt_int.", "group": "kill conditions", "md": "lrn worse than pt_interp", "kind": "rtext"},
        {"key": "verdict", "head": "rule output", "md": "rule output for the run", "kind": "rtext"},
    ]
    notes = [
        "Pre-registered rule (prereg.json): CONTINUE if for at least one S: R_S >= 0.25 AND "
        "bootstrap 95% CI lower bound of R_S > 0 AND lrn >= 10% below pt_interp in the same S AND "
        "failed-cell fraction <= 10% for lrn, control_S and pt_interp. KILL if R_S < 0.10 for "
        "both S, OR lrn worse than pt_interp for both S. Otherwise INCONCLUSIVE.",
        "Decision run: tier M (E3_full_M, 12 REAL_DEV geometries, 72 training and 36 test devices). "
        "Tier W (E3_full_W, truth inside the parametric family) is a secondary check; its rule "
        "output is reported, not used for the decision.",
        "Metric: frozen-inclusion paper RRMS of the NGSpice re-simulated card against the "
        "simulated truth curves, mean over held-out 110/190/260 K per device, all-device mean "
        "over 36 test devices (devices finite for all compared methods; tier W O4 has 35). "
        "Paired percentile bootstrap over test devices, 4000 resamples.",
    ]
    return rf.Table("04_table_e3_decision",
                    "E3 decision: learned shared temperature law (lrn) vs parametric laws, "
                    "pre-registered checks", cols, rows, notes)


def fig_e3(path: Path) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for r, tier in enumerate(("M", "W")):
        per = e3_per_device(tier)
        for c, S in enumerate(("O4", "O2")):
            ax = axes[r, c]
            style(ax)
            ctl = E3_CONTROL[S]
            a = per[f"{S}|lrn"]
            b = per[f"{S}|{ctl}"]
            pt = per[f"{S}|pt_interp"]
            tags = sorted(t for t in a if np.isfinite(a[t]) and np.isfinite(b[t]))
            xr = np.array([b[t] for t in tags])
            yr = np.array([a[t] for t in tags])
            xa, ya = np.maximum(xr, 1e-3), np.maximum(yr, 1e-3)
            colors = [BLUE if t.startswith("nmos") else ORANGE for t in tags]
            ax.scatter(xa, ya, c=colors, s=26, zorder=3, edgecolor=BG, lw=0.6)
            lo = min(xa.min(), ya.min()) * 0.7
            hi = max(xa.max(), ya.max()) * 1.4
            ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.9, ls="--", zorder=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            for t, xv, yv in zip(tags, xa, ya):
                if max(xv, yv) > 3:
                    ax.annotate(t.replace("__", " #"), (xv, yv), fontsize=6.5, color=MUTED,
                                xytext=(-4, -10), textcoords="offset points", ha="right")
            wins = int(sum(yr < xr))
            fa = fam({t: a[t] for t in tags})
            fb = fam({t: b[t] for t in tags})
            fp = fam({t: pt[t] for t in tags if np.isfinite(pt[t])})
            ax.text(0.02, 0.98,
                    f"lrn: mean {fa['all']:.3f}, median {fa['median']:.3f}\n"
                    f"{ctl}: mean {fb['all']:.3f}, median {fb['median']:.3f}\n"
                    f"pt_interp: mean {fp['all']:.3f}, median {fp['median']:.3f}\n"
                    f"lrn < {ctl} on {wins}/{len(tags)} test devices",
                    transform=ax.transAxes, va="top", fontsize=7.5, color=INK,
                    bbox=dict(facecolor=BG, edgecolor="none", alpha=0.85))
            ax.set_xlabel(f"{ctl} (control for {S}): held-out RRMS, mean of 110/190/260 K",
                          fontsize=8.5, color=INK)
            ax.set_ylabel("lrn (learned shared law): held-out RRMS", fontsize=8.5, color=INK)
            role = "decision run" if tier == "M" else "secondary check"
            ax.set_title(f"tier {tier} ({role}), {S}: R_S = "
                         f"{load(E3_RUN[tier] / 'summary.json')['decisions'][S]['lrn_vs_control']['reduction']:.3f}",
                         fontsize=9.5, color=INK)
    h1 = axes[0, 0].scatter([], [], color=BLUE, s=26, label="nMOS test device")
    h2 = axes[0, 0].scatter([], [], color=ORANGE, s=26, label="pMOS test device")
    fig.legend(handles=[h1, h2], fontsize=8, frameon=False, loc="lower center", ncol=2)
    fig.suptitle("E3, per test device: learned shared temperature law vs the pre-registered "
                 "control (points below the dashed line: lrn lower; values below 1e-3 drawn "
                 "at 1e-3)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(E3_RUN[t] / "eval") + "/*.json" for t in ("M", "W")] + \
        [rel(E3_RUN[t] / "summary.json") for t in ("M", "W")]


def table_e3_results(summ: dict) -> rf.Table:
    rows = []
    ol = e3_outlier()
    for tier in ("M", "W"):
        per = e3_per_device(tier)
        tab = summ[tier]["table"]
        for S in ("O4", "O2"):
            for m in E3_METHODS[S]:
                k = f"{S}|{m}"
                if k not in tab:
                    continue
                f = fam(per[k])
                excl = fam({t: v for t, v in per[k].items() if t != E3_OUTLIER}) \
                    if tier == "M" else None
                pt = tab[k]["per_temp"]
                rows.append({"tier": tier, "S": S, "m": m + (" (control)" if m == E3_CONTROL[S]
                                                            else ""),
                             "mean": tab[k]["heldout_all_device_mean"], "combined": f["combined"],
                             "median": f["median"], "excl": excl["all"] if excl else None,
                             "t110": pt["110"], "t190": pt["190"], "t260": pt["260"],
                             "anchor": tab[k]["anchor_all_device_mean"],
                             "failed": f"{tab[k]['n_failed_cells']}/{tab[k]['n_cells']}"})
        f = fam(per["floor|floor"])
        t = tab["floor|floor"]
        excl = fam({k: v for k, v in per["floor|floor"].items() if k != E3_OUTLIER})
        rows.append({"tier": tier, "S": "—", "m": "floor (reference)",
                     "mean": t["heldout_all_device_mean"], "combined": f["combined"],
                     "median": f["median"], "excl": excl["all"] if tier == "M" else None,
                     "t110": t["per_temp"]["110"], "t190": t["per_temp"]["190"],
                     "t260": t["per_temp"]["260"], "anchor": None,
                     "failed": f"{t['n_failed_cells']}/{t['n_cells']}"})
    cols = [
        {"key": "tier", "head": "tier", "md": "tier", "kind": "text"},
        {"key": "S", "head": "S", "md": "S", "kind": "text"},
        {"key": "m", "head": "method", "md": "method", "kind": "text"},
        {"key": "mean", "head": "all-device", "group": "held-out RRMS, 36 test devices", "md": "held-out all-device mean", "kind": "rrms"},
        {"key": "combined", "head": "combined", "group": "held-out RRMS, 36 test devices", "md": "held-out combined", "kind": "rrms"},
        {"key": "median", "head": "median", "group": "held-out RRMS, 36 test devices", "md": "held-out median", "kind": "rrms"},
        {"key": "excl", "head": "excl. #2", "group": "held-out RRMS, 36 test devices", "md": f"mean without {E3_OUTLIER} (diagnostic)", "kind": "rrms"},
        {"key": "t110", "head": "110 K", "group": "per temperature", "md": "110 K", "kind": "rrms", "family": True},
        {"key": "t190", "head": "190 K", "group": "per temperature", "md": "190 K", "kind": "rrms", "family": True},
        {"key": "t260", "head": "260 K", "group": "per temperature", "md": "260 K", "kind": "rrms", "family": True},
        {"key": "anchor", "head": "anchors", "md": "observed-temperature mean", "kind": "rrms"},
        {"key": "failed", "head": "failed", "md": "failed cells", "kind": "rtext"},
    ]
    notes = [
        "all-device = mean over the 36 test devices of the per-device mean over 110/190/260 K "
        "(from summary.json); combined = mean of the nMOS (15) and pMOS (21) family means; median "
        "over test devices. Means and medians in the combined/median/excl. columns use devices "
        "with all three held-out cells finite.",
        f"excl. #2 = tier M mean with test device {E3_OUTLIER} removed. Diagnostic only; the "
        f"pre-registered aggregate is the all-device mean. That device has held-out RRMS "
        f"{ol['lo']:.0f}-{ol['hi']:.0f} for every method (floor {ol['floor']:.4f}); its 110 K "
        f"cell is {ol['lo110']:.0f}-{ol['hi110']:.0f} for every method.",
        "Methods: lrn = learned shared law (primary); par3 = per-device law x = c + (mu − c) w^g; "
        "par14 = cryoml.temperature family; par3_pop = law with population-pooled g; *_g1 = g = 1; "
        "pt_interp = linear interpolation of the per-temperature anchor fits; lrn_mean = "
        "population-mean correction without a network; lrn_noH = trained without 110/190/260 K; "
        "lrn_param = parameter-MSE loss; floor = per-temperature fit on the held-out data.",
        "O4 observes 77/150/225/300 K; O2 observes 77/300 K. Anchor column: mean RRMS at the "
        "observed temperatures.",
    ]
    return rf.Table("06_table_e3_results",
                    "E3 held-out-temperature RRMS per method, tiers M and W", cols, rows, notes)


# ================================================================== E4

def e4_summaries() -> dict:
    out = {}
    for s in E4_SPLITS:
        out[s] = {"strict": load(E4 / "p43" / s / "summary_strict.json")}
        f = E4 / "p43" / s / "summary_feasible.json"
        if f.exists():
            out[s]["feasible"] = load(f)
    return out


def e4_decisions(summ: dict) -> dict:
    import e4_analyze as ea  # read-only: decide() reads prereg.json and the summary
    dec = {}
    for s in E4_SPLITS:
        d = ea.decide(summ[s]["strict"])
        stored = summ[s]["strict"].get("decision")
        if stored is not None:
            assert d["verdict"] == stored["verdict"]
            for k, v in d["per_method"].items():
                assert (v["C1"], v["C2"], v["C3"]) == (stored["per_method"][k]["C1"],
                                                       stored["per_method"][k]["C2"],
                                                       stored["per_method"][k]["C3"])
        dec[s] = d
    return dec


def table_e4_decision(summ: dict, dec: dict) -> rf.Table:
    rows = []
    for s in E4_SPLITS:
        d = dec[s]
        single = d["single_heldout_rrms"]
        for m in E4_CAND:
            v = d["per_method"][m]
            ps = v["paired_vs_single"]
            pc = v["paired_vs_crossinterp"]
            rows.append({
                "split": s + (" (decision)" if s == "vd" else " (secondary)"), "m": m,
                "cov": v["coverage_0.9"], "w": v["width_0.9"], "wl": 4 * single, "c1": yn(v["C1"]),
                "rr": v["heldout_rrms"], "rl": 0.95 * single,
                "wins": f"{v['wins']}/{v['n_valid']}", "c2": yn(v["C2"]),
                "c3": (yn(v["C3"]) + f" ({pc[0]:.3f} vs {pc[1]:.3f})") if m == "learned" else "—",
                "all": yn(v["C1"] and v["C2"] and v["C3"]),
                "verdict": d["verdict"] if m == E4_CAND[0] else "",
            })
    cols = [
        {"key": "split", "head": "split", "md": "split", "kind": "text"},
        {"key": "m", "head": "candidate", "md": "candidate", "kind": "text"},
        {"key": "cov", "head": "cov. 0.9", "group": "C1 calibration", "md": "coverage of 90% interval", "kind": "rrms"},
        {"key": "w", "head": "width 0.9", "group": "C1 calibration", "md": "width of 90% interval", "kind": "rrms"},
        {"key": "wl", "head": "4 × single", "group": "C1 calibration", "md": "width limit 4 × single RRMS", "kind": "rrms"},
        {"key": "c1", "head": "C1", "group": "C1 calibration", "md": "C1", "kind": "rtext"},
        {"key": "rr", "head": "held-out", "group": "C2 accuracy", "md": "held-out RRMS", "kind": "rrms"},
        {"key": "rl", "head": "0.95 × single", "group": "C2 accuracy", "md": "limit 0.95 × single", "kind": "rrms"},
        {"key": "wins", "head": "< single", "group": "C2 accuracy", "md": "devices below single", "kind": "rtext"},
        {"key": "c2", "head": "C2", "group": "C2 accuracy", "md": "C2", "kind": "rtext"},
        {"key": "c3", "head": "C3 (vs crossinterp)", "md": "C3 learned <= crossinterp", "kind": "rtext"},
        {"key": "all", "head": "all", "md": "C1 and C2 and C3", "kind": "rtext"},
        {"key": "verdict", "head": "rule output", "md": "rule output for the split", "kind": "rtext"},
    ]
    notes = [
        "Pre-registered rule (prereg.json, amendments 1 and 2): C1 = all-device mean coverage of "
        "the central 90% interval in [0.70, 0.95] AND all-device mean width <= 4 × single's "
        "held-out RRMS; C2 = held-out RRMS <= 0.95 × single (valid-only and paired) AND lower than "
        "single on >= 8 of 12 devices; C3 (learned only) = held-out RRMS <= crossinterp's "
        "(valid-only and paired). CONTINUE if a candidate meets C1, C2 and C3; "
        "PARTIAL_CALIBRATION_ONLY if no candidate meets all but one meets C1; KILL if none meets C1.",
        "Decision run: p43, split vd, 12 REAL_DEV devices, --select strict "
        "(queue job E4_p43_vd, stored decision block). Splits sparse and vg are secondary: the "
        "same decide() function was applied to their stored strict summaries for this report.",
        f"single held-out RRMS (all-device): vd {dec['vd']['single_heldout_rrms']:.4f}, sparse "
        f"{dec['sparse']['single_heldout_rrms']:.4f}, vg {dec['vg']['single_heldout_rrms']:.4f}. "
        "Width is mean (hi − lo) / den, in the same units as the per-curve RRMS.",
        "The seven-parameter runs (p7) are not included: at report time p7 vd and p7 vg had "
        "finished and p7 sparse was running; p7 is reported when all three splits finish.",
    ]
    return rf.Table("07_table_e4_decision",
                    "E4 decision (43 parameters): predictive intervals and accuracy on held-out "
                    "bias curves, pre-registered checks per candidate", cols, rows, notes)


def fig_e4_per_device(summ: dict, path: Path) -> list[str]:
    tags = real_dev()
    x = np.arange(len(tags))
    ms = ["single", "ens", "laplace", "learned", "crossinterp"]
    off = dict(zip(ms, np.linspace(-0.28, 0.28, len(ms))))
    mk = {"single": "o", "ens": "D", "laplace": "s", "learned": "v", "crossinterp": "^"}
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(BG)
    for ax, s in zip(axes, E4_SPLITS):
        style(ax)
        meth = summ[s]["strict"]["methods"]
        for m in ms:
            pd_ = {t: r["heldout_rrms"] for t, r in meth[m]["per_device"].items()}
            st = meth[m]["heldout_rrms"]
            vals = np.array([fnum(pd_.get(t)) for t in tags])
            ax.scatter(x + off[m], vals, marker=mk[m], s=30, color=E4_COLOR[m], zorder=3,
                       edgecolor=BG, lw=0.5,
                       label=f"{E4_LABEL[m]}: all-device {st['all_device_mean']:.4g}, combined "
                             f"{st['combined']:.4g}"
                             + (f", < single on {meth[m]['wins_vs_single']}/12" if m != "single" else ""))
        ax.set_yscale("log")
        top = max(fnum(r["heldout_rrms"]) for m in ms for r in meth[m]["per_device"].values())
        ax.set_ylim(0.01, top * 3)
        nm = sum(t.startswith("nmos") for t in tags)
        ax.axvline(nm - 0.5, color=GREY, lw=0.8, ls=":")
        ax.set_ylabel("held-out paper RRMS", fontsize=9, color=INK)
        role = "decision split" if s == "vd" else "secondary split"
        ax.set_title(f"p43, split {s} ({role})", fontsize=9.5, color=INK)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([rf.device_label(t) + (" n" if t.startswith("nmos") else " p")
                              for t in tags], rotation=30, ha="right", fontsize=8)
    fig.suptitle("E4, 43 parameters, REAL_DEV: held-out-curve RRMS per device and method "
                 "(n = nMOS, p = pMOS)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(E4 / "p43" / s / "summary_strict.json") for s in E4_SPLITS]


def fig_e4_calibration(summ: dict, path: Path) -> list[str]:
    ms = ["ens", "laplace", "learned", "conformal", "crossinterp_conformal"]
    col = {"ens": BLUE, "laplace": ORANGE, "learned": VIOLET, "conformal": GREY,
           "crossinterp_conformal": AQUA}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    w = 0.16
    for ax in axes:
        style(ax)
    for j, m in enumerate(ms):
        cov = [summ[s]["strict"]["methods"][m]["coverage_0.9"]["all_device_mean"] for s in E4_SPLITS]
        wid = [summ[s]["strict"]["methods"][m]["width_0.9"]["all_device_mean"] /
               summ[s]["strict"]["methods"]["single"]["heldout_rrms"]["all_device_mean"]
               for s in E4_SPLITS]
        xs = np.arange(3) + (j - 2) * w
        axes[0].bar(xs, cov, w * 0.92, color=col[m], label=E4_LABEL[m])
        axes[1].bar(xs, wid, w * 0.92, color=col[m], label=E4_LABEL[m])
        for xv, v in zip(xs, wid):
            axes[1].text(xv, v * 1.15, f"{v:.3g}", ha="center", va="bottom", fontsize=6,
                         color=INK, rotation=90)
    axes[0].axhspan(0.70, 0.95, color="#e4e3df", zorder=0, label="C1 coverage band [0.70, 0.95]")
    axes[0].axhline(0.90, color=MUTED, lw=0.8, ls="--")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("all-device mean coverage of the 90% interval", fontsize=9, color=INK)
    axes[1].axhline(4.0, color=MUTED, lw=0.9, ls="--", label="C1 width limit: 4 × single RRMS")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.1, 1e6)
    axes[1].set_ylabel("90% interval width / single's held-out RRMS", fontsize=9, color=INK)
    for ax in axes:
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{s} ({'decision' if s == 'vd' else 'secondary'})" for s in E4_SPLITS],
                           fontsize=8.5)
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1[:1], l0 + l1[:1], fontsize=7.5, frameon=False, loc="lower center",
               ncol=4)
    fig.suptitle("E4, 43 parameters, REAL_DEV: interval calibration (left) and width relative to "
                 "the C1 limit (right)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(E4 / "p43" / s / "summary_strict.json") for s in E4_SPLITS]


def table_e4_results(summ: dict) -> rf.Table:
    rows = []
    for s in E4_SPLITS:
        for sel in ("strict", "feasible"):
            if sel not in summ[s]:
                continue
            meth = summ[s][sel]["methods"]
            for m in E4_METHODS:
                e = meth[m]
                h = e["heldout_rrms"]
                med = fam({t: r["heldout_rrms"] for t, r in e["per_device"].items()})["median"]
                g = lambda k: e.get(k, {}).get("all_device_mean")  # noqa: E731
                rows.append({"split": s, "sel": sel, "m": m, "all": h["all_device_mean"],
                             "combined": h["combined"], "nmos": h["nmos"], "pmos": h["pmos"],
                             "median": med, "n": f"{h['n_valid']}/{h['n_devices']}",
                             "wins": e.get("wins_vs_single"), "c50": g("coverage_0.5"),
                             "c90": g("coverage_0.9"), "w90": g("width_0.9"),
                             "is90": g("interval_score_0.9")})
    cols = [
        {"key": "split", "head": "split", "md": "split", "kind": "text"},
        {"key": "sel", "head": "select", "md": "member selection", "kind": "text"},
        {"key": "m", "head": "method", "md": "method", "kind": "text"},
        {"key": "all", "head": "all-device", "group": "held-out RRMS, 12 devices", "md": "all-device", "kind": "rrms"},
        {"key": "combined", "head": "combined", "group": "held-out RRMS, 12 devices", "md": "combined", "kind": "rrms"},
        {"key": "nmos", "head": "nMOS", "group": "held-out RRMS, 12 devices", "md": "nMOS", "kind": "rrms", "family": True},
        {"key": "pmos", "head": "pMOS", "group": "held-out RRMS, 12 devices", "md": "pMOS", "kind": "rrms", "family": True},
        {"key": "median", "head": "median", "group": "held-out RRMS, 12 devices", "md": "median", "kind": "rrms"},
        {"key": "n", "head": "finite", "group": "held-out RRMS, 12 devices", "md": "finite devices", "kind": "rtext"},
        {"key": "wins", "head": "< single", "md": "devices below single", "kind": "int"},
        {"key": "c50", "head": "cov 0.5", "group": "intervals", "md": "coverage 0.5", "kind": "rrms"},
        {"key": "c90", "head": "cov 0.9", "group": "intervals", "md": "coverage 0.9", "kind": "rrms"},
        {"key": "w90", "head": "width 0.9", "group": "intervals", "md": "width 0.9", "kind": "rrms"},
        {"key": "is90", "head": "IS 0.9", "group": "intervals", "md": "interval score 0.9", "kind": "rrms"},
    ]
    notes = [
        "all-device = mean over the 12 REAL_DEV devices (strict: a non-finite held-out prediction "
        "makes the device NaN; finite = devices with a finite value; 0 single-card failures in all "
        "three splits); combined = mean of the nMOS (5) and pMOS (7) family means.",
        "Held-out curves: vd = Id-Vd curves except the highest gate bias; sparse = every included "
        "curve except Id-Vg at Vd 0.01/1.85 and Id-Vd at the highest gate bias; vg = Id-Vg at "
        "Vd 0.37/0.74/1.11/1.48.",
        "select feasible (amendment 1, secondary): members whose simulation of any included curve "
        "is non-finite are dropped before SINGLE and the ensemble are chosen. Feasible summaries "
        "exist where the analysis was run; on vd every value equals the strict one.",
        "conformal and crossinterp_conformal reuse the single and crossinterp point predictions; "
        "ens_env and laplace_param are reported in the JSON summaries and omitted here.",
    ]
    return rf.Table("09_table_e4_results",
                    "E4 held-out-curve RRMS and interval metrics per method, 43 parameters, "
                    "splits vd, sparse, vg", cols, rows, notes)


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    topic = "explore_e2_e3_e4"
    folder = rf.REPORT_ROOT / f"{args.date}_{topic}"
    folder.mkdir(parents=True, exist_ok=True)

    items = []  # (png, description, sources)
    # ---- E2
    full, light = e2_decision([0, 1, 2]), e2_decision([0])
    t1 = table_e2_decision(full, light, load(E2_TABLE)["decision"])
    rf.render_table(t1, folder / "01_table_e2_decision.png")
    items.append(("01_table_e2_decision.png",
                  "Table of the E2 pre-registered checks C1-C4, K1, K2 and the secondary "
                  "attribution check, with values and outcomes for the three-seed decision run "
                  "and the seed-0 light run", [rel(E2_PREREG), rel(E2_TABLE)]))
    src = fig_e2_per_device(folder / "02_e2_per_device.png")
    items.append(("02_e2_per_device.png",
                  "Per-device raw penalized RRMS on the 12 REAL_DEV devices for N0 (grey circle), "
                  "M (blue diamond), S (orange square) as seed means with seed values as small "
                  "dots, the nearest bank sample (green triangle), the released card (dotted) and "
                  "the direct seven-parameter fit + FD (green line); red × marks device-seed pairs "
                  "with a failed included curve; log scale", src))
    t3 = table_e2_results()
    rf.render_table(t3, folder / "03_table_e2_results.png")
    items.append(("03_table_e2_results.png",
                  "Table of E2 raw and FD-polished RRMS per arm: REAL_DEV all-device, combined, "
                  "nMOS, pMOS, median, valid-only mean and failed pairs; simulated tier a and "
                  "tier c means and medians; references",
                  [rel(E2_EVAL) + "/*.json", rel(D7)]))
    # ---- E3
    e3s = {t: load(E3_RUN[t] / "summary.json") for t in ("M", "W")}
    t4 = table_e3_decision(e3s)
    rf.render_table(t4, folder / "04_table_e3_decision.png")
    items.append(("04_table_e3_decision.png",
                  "Table of the E3 pre-registered checks per observation setting (O4, O2) for the "
                  "tier M decision run and the tier W secondary run: R_S with bootstrap CI, "
                  "reduction vs pt_interp, failed-cell fraction, continue and kill conditions, "
                  "rule output",
                  [rel(E3 / "prereg.json"), rel(E3 / "prereg_addendum.json")] +
                  [rel(E3_RUN[t] / "summary.json") for t in ("M", "W")]))
    src = fig_e3(folder / "05_e3_per_device.png")
    items.append(("05_e3_per_device.png",
                  "Scatter of per-test-device held-out RRMS (mean of 110/190/260 K), learned "
                  "shared law (y) vs the pre-registered control (x: par3 for O4, par3_pop for "
                  "O2), tier M (top) and tier W (bottom), log-log with the diagonal; nMOS blue, "
                  "pMOS orange", src))
    t6 = table_e3_results(e3s)
    rf.render_table(t6, folder / "06_table_e3_results.png")
    items.append(("06_table_e3_results.png",
                  "Table of E3 held-out RRMS per tier, setting and method: all-device, combined, "
                  "median, the tier M mean without the outlier device (diagnostic), per held-out "
                  "temperature, observed-temperature mean and failed cells",
                  [rel(E3_RUN[t] / "summary.json") for t in ("M", "W")]))
    # ---- E4
    e4 = e4_summaries()
    dec = e4_decisions(e4)
    t7 = table_e4_decision(e4, dec)
    rf.render_table(t7, folder / "07_table_e4_decision.png")
    items.append(("07_table_e4_decision.png",
                  "Table of the E4 pre-registered checks C1, C2, C3 for the candidates ens, "
                  "laplace and learned on the 43-parameter splits vd (decision), sparse and vg "
                  "(secondary), with rule output",
                  [rel(E4 / "prereg.json"), rel(E4 / "prereg_amendment_1.json"),
                   rel(E4 / "prereg_amendment_2.json")] +
                  [rel(E4 / "p43" / s / "summary_strict.json") for s in E4_SPLITS]))
    src = fig_e4_per_device(e4, folder / "08_e4_per_device.png")
    items.append(("08_e4_per_device.png",
                  "Per-device held-out RRMS on REAL_DEV for single card, ensemble, Laplace, "
                  "learned discrepancy and the crossinterp control, one panel per split (vd, "
                  "sparse, vg), log scale", src))
    t9 = table_e4_results(e4)
    rf.render_table(t9, folder / "09_table_e4_results.png")
    srcs9 = [rel(E4 / "p43" / s / f"summary_{k}.json") for s in E4_SPLITS for k in e4[s]]
    items.append(("09_table_e4_results.png",
                  "Table of E4 held-out RRMS (all-device, combined, nMOS, pMOS, median), devices "
                  "below single and interval metrics (coverage 0.5 and 0.9, width, interval "
                  "score) per split, member selection and method", srcs9))
    src = fig_e4_calibration(e4, folder / "10_e4_calibration.png")
    items.append(("10_e4_calibration.png",
                  "Bar charts per split of the all-device 90% interval coverage with the C1 band "
                  "[0.70, 0.95] (left) and the 90% interval width divided by the single card's "
                  "held-out RRMS with the C1 limit 4 (right, log scale), for ens, laplace, learned, "
                  "conformal and crossinterp + conformal", src))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = (f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_e2_e3_e4.py "
           f"--date {args.date}`")
    tables_md = [f"# Tables, {args.date} {topic}", "",
                 f"Generated {now} by {gen} from result files only. RRMS is the paper metric "
                 "with frozen published-baseline curve inclusion, strict.", ""]
    for t in (t1, t3, t4, t6, t7, t9):
        tables_md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(tables_md))

    readme = [f"# Report batch {args.date} · {topic}", "",
              f"Generated {now} by {gen} from result files only (no simulation, no training).", "",
              "## Contents", "",
              f"- E2 three-seed decision run (queue jobs E2_full, E2_fd_sim). Pre-registration: "
              f"`{rel(E2_PREREG)}`.",
              "- E3 full runs, tier M (decision) and tier W (secondary) (queue jobs E3_full_M, "
              f"E3_full_W). Pre-registration: `{rel(E3 / 'prereg.json')}`, "
              f"`{rel(E3 / 'prereg_addendum.json')}`.",
              "- E4 43-parameter runs, splits vd (decision), sparse and vg (secondary) (queue jobs "
              f"E4_p43_vd, E4_p43_sparse, E4_p43_vg). Pre-registration: `{rel(E4 / 'prereg.json')}` "
              "and amendments 1 and 2. The seven-parameter runs (p7) are in progress and not "
              "included.", "",
              "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{s}`" for s in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    m3 = e3s["M"]["decisions"]
    v = dec["vd"]["per_method"]
    summary = [
        f"E2 decision run (seeds 0-2), REAL_DEV raw penalized RRMS all-device: N0 {full['R_N0']:.3f}, "
        f"M {full['R_M']:.3f}, S {full['R_S']:.3f}; S/N0 = {full['ratio']:.3f}, S < N0 on "
        f"{full['wins']}/12. K1 met: KILL (confirms the seed-0 light run, S/N0 = "
        f"{light['ratio']:.3f}). Nearest bank sample 0.905; released card 0.285.",
        f"E3 decision run (tier M, 36 test devices): R_O4 = {m3['O4']['lrn_vs_control']['reduction']:.3f} "
        f"vs par3, R_O2 = {m3['O2']['lrn_vs_control']['reduction']:.3f} vs par3_pop; both < 0.10: "
        f"KILL. Means are dominated by test device {E3_OUTLIER} (held-out RRMS "
        f"{e3_outlier()['lo']:.0f}-{e3_outlier()['hi']:.0f} for all methods). Tier W secondary: "
        "rule output INCONCLUSIVE.",
        f"E4 p43 vd (decision): single held-out RRMS {dec['vd']['single_heldout_rrms']:.4f}; ens "
        f"coverage {v['ens']['coverage_0.9']:.3f}, width {v['ens']['width_0.9']:.3f} (C1 yes), "
        f"RRMS {v['ens']['heldout_rrms']:.4f}, < single on {v['ens']['wins']}/12 (C2 no); laplace "
        f"and learned fail C1. Rule output PARTIAL_CALIBRATION_ONLY. crossinterp control "
        f"{e4['vd']['strict']['methods']['crossinterp']['heldout_rrms']['all_device_mean']:.4f}. "
        f"Secondary: sparse {dec['sparse']['verdict']}, vg {dec['vg']['verdict']}. p7 in progress.",
    ]
    pngs = [i[0] for i in items]
    rf.update_index(args.date, topic, folder, pngs, summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
