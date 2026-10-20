#!/usr/bin/env python3
"""I3-2 follow-up analyses F1 (SPICE-native Verilog-A validation) and F2 (robustness).

Pre-registration: ``out/research/explore/I3_2/followup_prereg.json``.  These
analyses do not change the I3-2 decision.  REAL_TEST is not loaded.

Modes:
  f1          export the 12 per-fold M_bias corrections as Verilog-A, compile
              them (and the stored full model) with OpenVAF-Reloaded, simulate
              the REAL_DEV cards with ngspice-41 and ngspice-46 (uncorrected),
              run the M_bias refit cards inside the corrected netlist on
              ngspice-46 and compare with the Python-corrected currents and the
              stored I3-2 RRMS; writes followup/f1/{summary.json,curves.npz,va,osdi}
              and the three I-V figures.
  f1_plot     redraw the I-V figures from followup/f1/{summary.json,curves.npz}.
  f2_real     REAL_DEV variants a (seed sets), b (capacity), c (round-1 training
              cards), leave-one-device-out, held-out refit with g in the loop.
  f2_pdk2     tier b sets 0-2 of --pdk: collect starting-card records (E1
              recipe), leave-one-geometry-out M_bias and C0_refit.
  f2_summary  table of all F2 variants and the pre-registered checks.

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i3_2_followup.py --mode f1 --workers 4
  # F1b (amendment followup_prereg_amendment_F1b_2026-10-20.json): SPICE-native runs on
  # the ngspice-41 OSDI build instead of ngspice-46
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i3_2_followup.py --mode f1 --workers 4 \
      --native-bin /Users/anrunw/cryo-ng41/arm64-osdi/bin/ngspice --native-label ng41osdi \
      --out-name f1b --amendment out/research/explore/I3_2/followup_prereg_amendment_F1b_2026-10-20.json
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i3_2_followup.py --mode f2_real --workers 6
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i3_2_followup.py --mode f2_pdk2 --pdk gf180 --workers 6
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i3_2_followup.py --mode f2_summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryoml.explore.e1_correction import (Correction, load_devices,  # noqa: E402
                                          save_devices, score_device)
from cryoml.explore.i2_1_joint import with_sims  # noqa: E402
from cryoml.explore.i3_2_bias import (correction_from_json, eval_verilog_a_g,  # noqa: E402
                                      merge_seed_results, paired_ratio, split_seed_tasks,
                                      to_verilog_a_bias)
from cryoml.explore.i3_2_followup import (F2_GROUPS, F2_REAL_VARIANTS,  # noqa: E402
                                          OPENVAF_R, compile_va, exceeds, max_rel_diff, network_dims,
                                          osdi_va_text, real_variant_fails, scored_masks,
                                          simulate_sky130, tier_b_pdk_passes)

I32 = ROOT / "out/research/explore/I3_2"
FU = I32 / "followup"
PREREG = I32 / "followup_prereg.json"
AMENDMENT = I32 / "followup_prereg_amendment_2026-10-20.json"
E1 = ROOT / "out/research/explore/E1"
SPLIT = ROOT / "out/research/explore/real_split.json"
CARDS = ROOT / "out/research/polished/direct43_r3_fd/real/summary.json"
CARDS_R1 = ROOT / "out/research/polished/direct43_fd/real/summary.json"
REF_SCORES = ROOT / "out/research/reference_scores.json"
VDD = 1.85
OPTS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
PLOT_TAGS = ("pmos_L0p35_W0p55", "pmos_L2_W5", "nmos_L20_W0p64")
TOL = {"F1.0_g": 1e-9, "F1.1_rel": 1e-3, "F1.1_rrms": 1e-4, "F1.1_stored": 1e-9,
       "F1.2_rel": 1e-3, "F1.2_rrms": 1e-4, "F1.3_dev": 2e-4, "F1.3_mean": 1e-4}
R2_THRESHOLD = 0.08198283190529150 + 0.005
ZERO_FLOOR = 1e-20                 # amendment A1: numerical zeros at Vds = 0
TIGHT = "reltol=1e-9 abstol=1e-18"  # amendment A2: tolerance diagnostic


def rel(p: Path) -> str:
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def dev_tags() -> list[str]:
    return list(json.loads(SPLIT.read_text())["REAL_DEV"])


def aggregate(values: dict) -> dict:
    from e1_lodo import aggregate as agg
    return agg(values)


NATIVE_DESC = {"ng46": "ngspice-46", "ng41osdi": "ngspice-41 OSDI build"}


def native_label(args) -> str:
    return getattr(args, "native_label", "ng46") or "ng46"


def native_desc(label: str) -> str:
    return NATIVE_DESC.get(label, label)


def f1_out_name(args) -> str:
    name = getattr(args, "out_name", "") or "f1"
    return f"smoke_{name}" if args.smoke else name


def binaries(native_bin: str = "", label: str = "ng46") -> dict:
    """Canonical ngspice-41 ("ng41") and the second simulator used for the
    SPICE-native runs (key ``label``; default Homebrew ngspice-46, F1b: the
    ngspice-41 OSDI build passed as ``native_bin``)."""
    from cryoml.config import resolve_ngspice_bin
    from cryoml.pdk2.ihp import ngspice46_bin
    if label == "ng41":
        raise ValueError("native label must differ from 'ng41'")
    out = {"ng41": resolve_ngspice_bin(), label: native_bin or ngspice46_bin()}
    for k in list(out):
        v = subprocess.run([out[k], "-v"], capture_output=True, text=True, timeout=30).stdout
        out[f"{k}_version"] = next((ln.strip("* ").strip() for ln in v.splitlines()
                                    if "ngspice-" in ln), v[:80])
    return out


# ===========================================================================
# F1
# ===========================================================================
def _record_order(tag: str):
    """(device, curves in simulation order, index of each record curve)."""
    from cryoml.card_eval import device_by_tag
    from cryoml.data_io import load_device_curves
    from cryoml.metrics import tag_metric_curves
    device = device_by_tag(tag)
    curves = load_device_curves(device)
    pos = {id(c): i for i, c in enumerate(curves)}
    order = [pos[id(c)] for _, c in sorted(tag_metric_curves(curves).items())]
    return device, curves, order


def f1_device_task(task):
    """All F1 simulations of one REAL_DEV device."""
    from cryoml.cold_start import device_bins
    tag, cards, fold_osdi, full_osdi, bins, N = task
    rec = {r.tag: r for r in load_devices(E1 / "real_dev")}[tag]
    fold = correction_from_json(json.loads((I32 / "real_dev/corrections.json").read_text())
                                [f"M_bias|{tag}"])
    full = correction_from_json(json.loads((I32 / "real_dev/corrections.json").read_text())
                                ["full|M_bias"])
    device, curves, order = _record_order(tag)
    b_idx = device_bins()[tag]
    t0 = time.time()

    def run(params, ng, osdi=None, options=""):
        sims, msgs = simulate_sky130(device.dev_type, device.L_um, device.W_um, curves,
                                     params, b_idx, bins[ng], osdi=osdi, vdd=VDD,
                                     options=options)
        return [np.asarray(sims[i], float) for i in order], msgs

    sims, msgs = {}, {}
    for card, params in cards.items():
        for ng in ("ng41", N):
            sims[f"{card}_{ng}"], msgs[f"{card}_{ng}"] = run(params, ng)
    sims[f"native_fold_{N}"], msgs[f"native_fold_{N}"] = run(cards["mbias"], N, fold_osdi)
    sims[f"native_full_{N}"], msgs[f"native_full_{N}"] = run(cards["mbias"], N, full_osdi)
    sims[f"mbias_{N}_tight"], msgs[f"mbias_{N}_tight"] = run(cards["mbias"], N,
                                                              options=TIGHT)
    sims[f"native_fold_{N}_tight"], msgs[f"native_fold_{N}_tight"] = run(
        cards["mbias"], N, fold_osdi, options=TIGHT)
    sims[f"py_fold_{N}_tight"] = fold.apply(with_sims(rec, sims[f"mbias_{N}_tight"]))
    sims["py_fold_ng41"] = fold.apply(with_sims(rec, sims["mbias_ng41"]))
    sims[f"py_fold_{N}"] = fold.apply(with_sims(rec, sims[f"mbias_{N}"]))
    sims[f"py_full_{N}"] = full.apply(with_sims(rec, sims[f"mbias_{N}"]))
    scores = {k: float(score_device(rec, v)["rrms"]) for k, v in sims.items()}
    masks = scored_masks(rec)
    return {"tag": tag, "sims": sims, "msgs": msgs, "scores": scores,
            "masks": masks, "runtime_s": round(time.time() - t0, 1)}


def va_g_check(corr, text: str, n: int = 200) -> float:
    rng = np.random.default_rng(0)
    worst = 0.0
    for pmos in (0.0, 1.0):
        sgn = -1.0 if pmos else 1.0
        for _ in range(n):
            vgs, vds = sgn * rng.uniform(0, VDD), sgn * rng.uniform(0, VDD)
            phi = np.array([[pmos, abs(vgs) / VDD, abs(vds) / VDD, 0.0, 0.0, 0.0]])
            worst = max(worst, abs(eval_verilog_a_g(text, pmos, vgs, vds) - corr.g(phi)[0]))
    return float(worst)


def f1(args) -> int:
    t_start = time.time()
    N = native_label(args)
    out = FU / f1_out_name(args)
    va_dir, osdi_dir = out / "va", out / "osdi"
    va_dir.mkdir(parents=True, exist_ok=True)
    tags = dev_tags()
    if args.tags:
        tags = [t for t in args.tags.split(",") if t]
    bins = binaries(args.native_bin, N)
    compiler = {"openvaf": args.openvaf or OPENVAF_R,
                "extra_args": tuple(a for a in args.openvaf_args.split() if a)}
    cv = subprocess.run([str(compiler["openvaf"]), "--version"], capture_output=True, text=True,
                        timeout=30, check=False)
    compiler_info = {"openvaf": str(compiler["openvaf"]), "extra_args": list(compiler["extra_args"]),
                     "version": (cv.stdout + cv.stderr).strip()[:120]}
    corrs = json.loads((I32 / "real_dev/corrections.json").read_text())
    held = json.loads((I32 / "real_dev/heldout_cards.json").read_text())
    stored = json.loads((I32 / "real_dev/results.json").read_text())["per_device"]
    start_cards = json.loads(CARDS.read_text())["devices"]
    released = json.loads(REF_SCORES.read_text())["released"]["devices"]

    # ---- export + compile
    export = {}
    full_corr = correction_from_json(corrs["full|M_bias"])
    full_text = (I32 / "real_dev/m_bias_sky130_dev_full.va").read_text()
    models = [("full", full_corr, full_text)]
    models += [(f"fold_{t}", correction_from_json(corrs[f"M_bias|{t}"]),
                to_verilog_a_bias(correction_from_json(corrs[f"M_bias|{t}"]), vdd=VDD,
                                  legacy=True))
               for t in tags]
    for name, corr, text in models:
        (va_dir / f"m_bias_{name}.va").write_text(text)
        otext = osdi_va_text(text)
        c = compile_va(otext, va_dir / f"m_bias_{name}_osdi.va", osdi_dir / f"m_bias_{name}.osdi",
                       **compiler)
        export[name] = {"va": rel(va_dir / f"m_bias_{name}.va"),
                        "va_osdi": rel(c["va"]), "osdi": rel(c["osdi"]),
                        "sha256_va_osdi": hashlib.sha256(otext.encode()).hexdigest(),
                        "sha256_compile_key": c["sha256"],
                        "compile_warnings": c["log"].count("warning["),
                        "max_abs_g_diff_va_text_vs_python": va_g_check(corr, text),
                        "changed_lines_vs_export": sorted(
                            set(otext.splitlines()) ^ set(text.splitlines()))}
    export["full"]["export_text_equals_stored_file"] = (
        to_verilog_a_bias(full_corr, vdd=VDD, legacy=True) == full_text)
    print(json.dumps({k: {kk: v[kk] for kk in ("compile_warnings",
                                               "max_abs_g_diff_va_text_vs_python")}
                      for k, v in export.items()}), flush=True)

    # ---- simulations
    tasks = []
    for t in tags:
        cards = {"published": None, "start": dict(start_cards[t]["params"]),
                 "c0": dict(held["C0_refit_rerun"][t]), "mbias": dict(held["M_bias"][t])}
        tasks.append((t, cards, ROOT / export[f"fold_{t}"]["osdi"],
                      ROOT / export["full"]["osdi"], bins, N))
    with get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(f1_device_task, tasks, chunksize=1)

    per, arrays = {}, {}
    recs = {r.tag: r for r in load_devices(E1 / "real_dev")}
    for row in rows:
        t, s, sc, m = row["tag"], row["sims"], row["scores"], row["masks"]
        rec = recs[t]
        repro = {}
        for card in ("published", "start", "c0", "mbias"):
            a, b = s[f"{card}_{N}"], s[f"{card}_ng41"]
            repro[card] = {"rrms_ng41": sc[f"{card}_ng41"], f"rrms_{N}": sc[f"{card}_{N}"],
                           "abs_delta_rrms": abs(sc[f"{card}_{N}"] - sc[f"{card}_ng41"]),
                           "max_rel_scored": max_rel_diff(a, b, m, ZERO_FLOOR),
                           "max_rel_scored_literal": max_rel_diff(a, b, m),
                           "max_rel_all_ge_1e-12A": max_rel_diff(a, b, floor=1e-12)}
        nat = {f"rrms_native_{N}": sc[f"native_fold_{N}"], "rrms_stored": stored["M_bias"][t],
               "rrms_python_ng41": sc["py_fold_ng41"], f"rrms_python_{N}": sc[f"py_fold_{N}"],
               "delta_native_vs_stored": sc[f"native_fold_{N}"] - stored["M_bias"][t],
               f"delta_native_vs_python_{N}": sc[f"native_fold_{N}"] - sc[f"py_fold_{N}"],
               f"max_rel_native_vs_python_{N}_scored": max_rel_diff(
                   s[f"native_fold_{N}"], s[f"py_fold_{N}"], m, ZERO_FLOOR),
               f"max_rel_native_vs_python_{N}_scored_literal": max_rel_diff(
                   s[f"native_fold_{N}"], s[f"py_fold_{N}"], m),
               f"max_rel_native_vs_python_{N}_all_ge_1e-12A": max_rel_diff(
                   s[f"native_fold_{N}"], s[f"py_fold_{N}"], floor=1e-12),
               "max_rel_native_vs_python_ng41_scored": max_rel_diff(
                   s[f"native_fold_{N}"], s["py_fold_ng41"], m, ZERO_FLOOR),
               f"max_rel_full_native_vs_python_{N}_scored": max_rel_diff(
                   s[f"native_full_{N}"], s[f"py_full_{N}"], m, ZERO_FLOOR),
               f"max_rel_full_native_vs_python_{N}_scored_literal": max_rel_diff(
                   s[f"native_full_{N}"], s[f"py_full_{N}"], m),
               "tight_tolerance_diagnostic": {
                   "options": TIGHT,
                   "max_rel_native_vs_python_scored": max_rel_diff(
                       s[f"native_fold_{N}_tight"], s[f"py_fold_{N}_tight"], m, ZERO_FLOOR),
                   "abs_delta_rrms": abs(sc[f"native_fold_{N}_tight"]
                                         - sc[f"py_fold_{N}_tight"]),
                   "rrms_native_tight": sc[f"native_fold_{N}_tight"],
                   "max_rel_uncorrected_tight_vs_default_scored": max_rel_diff(
                       s[f"mbias_{N}_tight"], s[f"mbias_{N}"], m, ZERO_FLOOR)},
               f"max_rel_full_native_vs_python_{N}_all_ge_1e-12A": max_rel_diff(
                   s[f"native_full_{N}"], s[f"py_full_{N}"], floor=1e-12),
               "convergence_messages": {k: v for k, v in row["msgs"].items() if v},
               "n_nonfinite_native": int(sum(np.sum(~np.isfinite(x))
                                             for k in (f"native_fold_{N}", f"native_full_{N}")
                                             for x in s[k])),
               "n_nonfinite_native_tight": int(sum(np.sum(~np.isfinite(x))
                                                   for x in s[f"native_fold_{N}_tight"]))}
        stored_ok = {"c0_ng41_vs_stored_C0_refit": abs(sc["c0_ng41"] - stored["C0_refit"][t]),
                     "mbias_python_ng41_vs_stored_M_bias": abs(sc["py_fold_ng41"]
                                                               - stored["M_bias"][t]),
                     "published_ng41_vs_reference_released": abs(sc["published_ng41"]
                                                                 - released[t]["rrms"])}
        per[t] = {"reproduction": repro, "native": nat, "stored_check": stored_ok,
                  "rrms": sc, "runtime_s": row["runtime_s"]}
        for i, c in enumerate(rec.curves):
            arrays[f"{t}|{i}|vg"] = c.vg
            arrays[f"{t}|{i}|vd"] = c.vd
            arrays[f"{t}|{i}|measured"] = c.target
            for k, v in s.items():
                arrays[f"{t}|{i}|{k}"] = v[i]
        print(t, json.dumps({"rrms_native": nat[f"rrms_native_{N}"],
                             "rrms_stored": nat["rrms_stored"],
                             "max_rel_native_vs_python_same_sim": nat[f"max_rel_native_vs_python_{N}_scored"],
                             f"max_rel_{N}_vs_ng41": max(r["max_rel_scored"]
                                                         for r in repro.values())}), flush=True)

    def agg(key_fn):
        return aggregate({t: key_fn(per[t]) for t in tags})

    aggs = {f"native_{N}": agg(lambda p: p["native"][f"rrms_native_{N}"]),
            "stored_M_bias": agg(lambda p: p["native"]["rrms_stored"]),
            f"python_{N}": agg(lambda p: p["native"][f"rrms_python_{N}"]),
            "c0_card_ng41": agg(lambda p: p["rrms"]["c0_ng41"]),
            f"c0_card_{N}": agg(lambda p: p["rrms"][f"c0_{N}"]),
            "published_ng41": agg(lambda p: p["rrms"]["published_ng41"]),
            "mbias_card_uncorrected_ng41": agg(lambda p: p["rrms"]["mbias_ng41"])}
    def mx(f):
        vals = [float(f(per[t])) for t in tags]
        return float("nan") if any(np.isnan(v) for v in vals) else float(max(vals))
    checks = {
        "F1.0_export_text": {
            "max_abs_g_diff": max(v["max_abs_g_diff_va_text_vs_python"] for v in export.values()),
            "all_compiled": True},
        "F1.1_simulator_reproduction": {
            "max_rel_scored": mx(lambda p: max(r["max_rel_scored"]
                                               for r in p["reproduction"].values())),
            "max_rel_scored_literal": mx(lambda p: max(r["max_rel_scored_literal"]
                                                       for r in p["reproduction"].values())),
            "max_rel_all_ge_1e-12A": mx(lambda p: max(r["max_rel_all_ge_1e-12A"]
                                                      for r in p["reproduction"].values())),
            "max_abs_delta_rrms": mx(lambda p: max(r["abs_delta_rrms"]
                                                   for r in p["reproduction"].values())),
            "max_stored_check": mx(lambda p: max(p["stored_check"].values()))},
        "F1.2_va_implementation": {
            "max_rel_fold_scored": mx(lambda p: p["native"][f"max_rel_native_vs_python_{N}_scored"]),
            "max_rel_full_scored": mx(
                lambda p: p["native"][f"max_rel_full_native_vs_python_{N}_scored"]),
            "max_rel_fold_scored_literal": mx(
                lambda p: p["native"][f"max_rel_native_vs_python_{N}_scored_literal"]),
            "max_rel_full_scored_literal": mx(
                lambda p: p["native"][f"max_rel_full_native_vs_python_{N}_scored_literal"]),
            "max_abs_delta_rrms": mx(lambda p: abs(p["native"][f"delta_native_vs_python_{N}"])),
            "n_devices_with_convergence_messages": int(sum(
                any(not k.endswith("_tight") for k in per[t]["native"]["convergence_messages"])
                for t in tags)),
            "n_devices_with_convergence_messages_tight_runs": int(sum(
                any(k.endswith("_tight") for k in per[t]["native"]["convergence_messages"])
                for t in tags)),
            "n_nonfinite": int(sum(per[t]["native"]["n_nonfinite_native"] for t in tags))},
        "F1.3_end_to_end": {
            "max_abs_delta_device": mx(lambda p: abs(p["native"]["delta_native_vs_stored"])),
            "abs_delta_all_device_mean": abs(aggs[f"native_{N}"]["all_device_mean"]
                                             - aggs["stored_M_bias"]["all_device_mean"])}}
    c = checks
    c["F1.0_export_text"]["fail"] = bool(exceeds(c["F1.0_export_text"]["max_abs_g_diff"], TOL["F1.0_g"]))
    c["F1.1_simulator_reproduction"]["fail"] = bool(
        exceeds(c["F1.1_simulator_reproduction"]["max_rel_scored"], TOL["F1.1_rel"])
        or exceeds(c["F1.1_simulator_reproduction"]["max_abs_delta_rrms"], TOL["F1.1_rrms"]))
    c["F1.1_simulator_reproduction"]["fail_literal"] = bool(
        exceeds(c["F1.1_simulator_reproduction"]["max_rel_scored_literal"], TOL["F1.1_rel"])
        or exceeds(c["F1.1_simulator_reproduction"]["max_abs_delta_rrms"], TOL["F1.1_rrms"]))
    c["F1.1_simulator_reproduction"]["stored_values_reproduced"] = bool(
        c["F1.1_simulator_reproduction"]["max_stored_check"] <= TOL["F1.1_stored"])
    c["F1.2_va_implementation"]["fail"] = bool(
        exceeds(c["F1.2_va_implementation"]["max_rel_fold_scored"], TOL["F1.2_rel"])
        or exceeds(c["F1.2_va_implementation"]["max_rel_full_scored"], TOL["F1.2_rel"])
        or exceeds(c["F1.2_va_implementation"]["max_abs_delta_rrms"], TOL["F1.2_rrms"])
        or c["F1.2_va_implementation"]["n_devices_with_convergence_messages"] > 0
        or c["F1.2_va_implementation"]["n_nonfinite"] > 0)
    f12 = c["F1.2_va_implementation"]
    f12["fail_literal"] = bool(
        exceeds(f12["max_rel_fold_scored_literal"], TOL["F1.2_rel"])
        or exceeds(f12["max_rel_full_scored_literal"], TOL["F1.2_rel"])
        or exceeds(f12["max_abs_delta_rrms"], TOL["F1.2_rrms"])
        or f12["n_devices_with_convergence_messages"] > 0 or f12["n_nonfinite"] > 0)
    c["tight_tolerance_diagnostic"] = {
        "options": TIGHT,
        "max_rel_native_vs_python_scored": mx(
            lambda p: p["native"]["tight_tolerance_diagnostic"]["max_rel_native_vs_python_scored"]),
        "max_abs_delta_rrms": mx(
            lambda p: p["native"]["tight_tolerance_diagnostic"]["abs_delta_rrms"])}
    c["F1.3_end_to_end"]["fail"] = bool(
        exceeds(c["F1.3_end_to_end"]["max_abs_delta_device"], TOL["F1.3_dev"])
        or exceeds(c["F1.3_end_to_end"]["abs_delta_all_device_mean"], TOL["F1.3_mean"]))

    amendments = [rel(AMENDMENT)] + [rel(ROOT / a) for a in args.amendment]
    summary = {"prereg": rel(PREREG), "amendment": rel(AMENDMENT), "amendments": amendments,
               "zero_floor_A": ZERO_FLOOR, "smoke": bool(args.smoke), "tags": tags,
               "binaries": bins, "compiler": compiler_info, "native_label": N, "native_simulator": native_desc(N),
               "osdi_in_ngspice41": N != "ng46",
               "osdi_note": ("ngspice-41 arm64 configured without OSDI; SPICE-native runs use "
                             "ngspice-46 (prereg setup_facts)" if N == "ng46" else
                             f"SPICE-native runs use {bins[N]} ({native_desc(N)}); "
                             "uncorrected reference runs use the canonical ngspice-41"),
               "tolerances": TOL, "export": export, "aggregates": aggs, "checks": checks,
               "per_device": per, "runtime_s": round(time.time() - t_start, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    np.savez_compressed(out / "curves.npz", **arrays)
    print(json.dumps({"aggregates": {k: {kk: v[kk] for kk in ("all_device_mean", "combined")}
                                     for k, v in aggs.items()}, "checks": checks}, indent=1))
    if not c["F1.1_simulator_reproduction"]["stored_values_reproduced"]:
        print("STOP: ngspice-41 does not reproduce the stored values (code error)")
        return 2
    f1_plot(args, out)
    print(f"runtime {summary['runtime_s']} s")
    return 0


def f1_plot(args, out: Path | None = None) -> int:
    import plot_explore_e2_e3_e4 as pe
    from matplotlib.lines import Line2D
    plt = pe.plt
    out = out or FU / f1_out_name(args)
    summ = json.loads((out / "summary.json").read_text())
    N = summ.get("native_label", "ng46")
    desc = native_desc(N)
    stem = out.name.removeprefix("smoke_")
    with np.load(out / "curves.npz") as z:
        arr = {k: z[k] for k in z.files}
    recs = {r.tag: r for r in load_devices(E1 / "real_dev")}
    stored = json.loads((I32 / "real_dev/results.json").read_text())["per_device"]
    written = []
    for tag in PLOT_TAGS:
        if tag not in summ["per_device"]:
            continue
        rec, p = recs[tag], summ["per_device"][tag]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), dpi=pe.DPI)
        fig.patch.set_facecolor(pe.BG)
        for ax, kind in zip(axes, ("idvd", "idvg")):
            pe.style(ax)
            for i, c in enumerate(rec.curves):
                if c.kind != kind:
                    continue
                g = lambda k: np.abs(arr[f"{tag}|{i}|{k}"])
                xv = np.abs(c.vd if kind == "idvd" else c.vg)
                s = slice(c.start, None)
                alpha = 1.0 if c.included else 0.35
                meas = np.abs(c.target)
                ax.scatter(xv[s], meas[s], s=7, color=pe.INK, zorder=4, alpha=alpha, lw=0)
                if c.start:
                    ax.scatter(xv[:c.start], meas[:c.start], s=7, facecolor="none",
                               edgecolor=pe.GREY, lw=0.5, zorder=4, alpha=alpha)
                ax.plot(xv, g("published_ng41"), color=pe.GREY, ls="--", lw=1.2, alpha=alpha,
                        zorder=2)
                ax.plot(xv, g("c0_ng41"), color=pe.BLUE, lw=1.3, alpha=alpha, zorder=3)
                ax.plot(xv, g(f"native_fold_{N}"), color=pe.ORANGE, lw=1.3, alpha=alpha,
                        zorder=3)
                lab = c.key.split("@")[1]
                ax.annotate(f"{'|Vgs|' if kind == 'idvd' else '|Vds|'} {lab}"
                            + ("" if c.included else " (excluded)"),
                            (xv[-1], max(abs(meas[-1]), 1e-14)), fontsize=6.8, color=pe.MUTED,
                            xytext=(3, 0), textcoords="offset points", va="center")
            if kind == "idvg":
                ax.set_yscale("log")
                ax.set_ylim(bottom=1e-12)
                ax.set_xlabel("|Vgs| (V)")
                ax.set_title("Id-Vg at |Vds| = 0.01-1.85 V (log scale)", fontsize=9.5)
            else:
                ax.set_xlabel("|Vds| (V)")
                ax.set_title("Id-Vd at |Vgs| = 0.37-1.85 V", fontsize=9.5)
            ax.set_ylabel("|Id| (A)")
        r = p["rrms"]
        handles = [
            Line2D([], [], marker="o", ls="", color=pe.INK, ms=4,
                   label="measured 77 K (filled: scored points; hollow: trimmed before the metric)"),
            Line2D([], [], color=pe.GREY, ls="--", lw=1.2,
                   label=f"NGSpice (ngspice-41), published 77 K card: RRMS {r['published_ng41']:.4f}"),
            Line2D([], [], color=pe.BLUE, lw=1.3,
                   label=f"NGSpice (ngspice-41), C0_refit card (held-out refit, no correction): "
                         f"RRMS {r['c0_ng41']:.4f}"),
            Line2D([], [], color=pe.ORANGE, lw=1.3,
                   label=f"SPICE-native corrected ({desc}): M_bias refit card in the netlist with "
                         "the compiled fold Verilog-A correction (trained on the other 11 devices): "
                         f"RRMS {r[f'native_fold_{N}']:.4f}"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8.3, frameon=False,
                   bbox_to_anchor=(0.5, -0.005))
        fig.suptitle(
            f"{tag}, held out (I3-2 follow-up {stem.upper()}). Stored I3-2 values: C0_refit "
            f"{stored['C0_refit'][tag]:.4f}, M_bias {stored['M_bias'][tag]:.4f} (ngspice-41 x "
            f"exp(g) in Python).\nSPICE-native minus stored M_bias RRMS: "
            f"{p['native']['delta_native_vs_stored']:+.2e}; max relative current difference, "
            f"SPICE-native vs Python on {desc} (scored points, |I| >= 1e-20 A): "
            f"{p['native'][f'max_rel_native_vs_python_{N}_scored']:.2e}",
            fontsize=9.6, color=pe.INK)
        fig.tight_layout(rect=(0, 0.13, 1, 0.93))
        path = (out.parent if not summ["smoke"] else out) / f"{stem}_iv_{tag}.png"
        fig.savefig(path, facecolor=pe.BG)
        plt.close(fig)
        written.append(rel(path))
    print(json.dumps({"figures": written}))
    return 0


# ===========================================================================
# F2
# ===========================================================================
def train_task(task):
    from i3_2_bias import train_task as t
    return t(task)


def heldout(task):
    from i3_2_bias import heldout as h
    return h(task)


def collect_real_task(task):
    from e1_collect_real import collect
    tag, params = task
    return collect(tag, params)


def collect_pdk2_task(task):
    from e1_collect_pdk2 import collect
    pdk, tier, s, tag = task
    recs = collect(pdk, tier, s, "wide", 300, 3, [tag])
    if len(recs) != 1:
        raise RuntimeError(f"{pdk} {tier} set {s} {tag}: {len(recs)} records")
    return s, recs[0]


def smoke_hyper(hyper: dict, smoke: bool) -> dict:
    if not smoke:
        return dict(hyper)
    from cryoml.explore.i3_2_bias import HYPER_BIAS
    seeds = hyper.get("seeds", HYPER_BIAS["seeds"])
    return {**hyper, "seeds": (seeds[0],), "steps": 200}


def run_lodo(pool, variants: dict, records, key, groups, train_sets: dict, controls: list,
             clusters) -> tuple[dict, dict, dict, dict]:
    """variants: name -> hyper; train_sets: name -> records used for training
    (held-out group removed per fold).  controls: [(label, Correction)] refit
    for every held-out record.  Returns rows, corrections, timing, cards."""
    held = {g: [r for r in records if key(r) == g] for g in groups}
    tasks, names, hypers = [], [], {}
    for v, hyper in variants.items():
        for g in groups:
            tasks.append(("M_bias", [r for r in train_sets[v] if key(r) != g]))
            names.append((v, g))
            hypers[len(tasks) - 1] = hyper
    expanded, owner = [], []
    for i, t in enumerate(tasks):
        e, o = split_seed_tasks([t], hypers[i])
        expanded += e
        owner += [i] * len(o)
    timing = {}
    t0 = time.time()
    corrs = dict(zip(names, merge_seed_results(len(tasks), owner,
                                               pool.map(train_task, expanded, chunksize=1))))
    timing["train_s"] = round(time.time() - t0, 1)
    timing["train_n_seed_tasks"] = len(expanded)
    print(f"training done ({timing['train_s']} s, {len(expanded)} seed tasks)", flush=True)
    rtasks = []
    for (v, g), corr in corrs.items():
        rtasks += [(v, None, corr, rec, OPTS) for rec in held[g]]
    for label, corr in controls:
        rtasks += [(label, None, corr, rec, OPTS) for g in groups for rec in held[g]]
    t0 = time.time()
    out = pool.map(heldout, rtasks, chunksize=1)
    timing["heldout_refit_s"] = round(time.time() - t0, 1)
    timing["heldout_refit_n"] = len(rtasks)
    rows, cards = {}, {}
    for row in out:
        rows.setdefault(row["label"], {})[row["tag"]] = row["rrms"]
        cards.setdefault(row["label"], {})[row["tag"]] = row["params"]
    corr_json = {f"{v}|{g}": c.to_json() for (v, g), c in corrs.items()}
    return rows, corr_json, timing, cards


def describe(scores: dict, control: dict, clusters, extra: dict | None = None) -> dict:
    out = {"summary": aggregate(scores),
           "vs_C0_refit": paired_ratio(scores, control, clusters)}
    for k, v in (extra or {}).items():
        out[f"vs_{k}"] = paired_ratio(scores, v, clusters)
    return out


def f2_real(args) -> int:
    from i3_2_bias import load_records, stored_rows
    t_start = time.time()
    out = FU / ("smoke_f2_real" if args.smoke else "f2_real")
    out.mkdir(parents=True, exist_ok=True)
    records, key, _ = load_records("real", "gf180")
    if sorted(r.tag for r in records) != sorted(dev_tags()):
        raise RuntimeError("records differ from REAL_DEV")
    groups = sorted({key(r) for r in records})
    if args.folds_limit:
        groups = groups[:args.folds_limit]
    names = [v for v in (args.variants.split(",") if args.variants else F2_REAL_VARIANTS)
             if v]
    variants = {v: smoke_hyper(F2_REAL_VARIANTS[v]["hyper"], args.smoke) for v in names}
    print(f"F2 real: variants {variants}, folds {len(groups)}, smoke={args.smoke}", flush=True)
    train_sets = {v: records for v in names}
    r1_check = None
    with get_context("spawn").Pool(args.workers) as pool:
        if any(F2_REAL_VARIANTS[v]["train_cards"] == "round1" for v in names):
            r1 = json.loads(CARDS_R1.read_text())["devices"]
            tags = sorted(r.tag for r in records)
            r1_recs = pool.map(collect_real_task, [(t, dict(r1[t]["params"])) for t in tags])
            r1_check = {r.tag: {"rrms_record": float(score_device(r, [c.sim for c in r.curves])
                                                     ["rrms"]),
                                "rrms_summary": float(r1[r.tag]["rrms"])} for r in r1_recs}
            worst = max(abs(v["rrms_record"] - v["rrms_summary"]) for v in r1_check.values())
            print(f"round-1 card records: max |record - summary| RRMS {worst:.3e}", flush=True)
            if worst > 1e-9:
                raise RuntimeError("round-1 card records do not reproduce the stored RRMS")
            for v in names:
                if F2_REAL_VARIANTS[v]["train_cards"] == "round1":
                    train_sets[v] = r1_recs
        rows, corr_json, timing, cards = run_lodo(pool, variants, records, key, groups,
                                                  train_sets, [], None)
    held_tags = [r.tag for r in records if key(r) in groups]
    st = stored_rows(True, "gf180")
    c0 = {t: st["C0_refit"][t] for t in held_tags}
    i32 = json.loads((I32 / "real_dev/results.json").read_text())["per_device"]
    m_bias = {t: i32["M_bias"][t] for t in held_tags}
    res = {}
    for v in names:
        d = describe(rows[v], c0, None, {"I3_2_M_bias": m_bias})
        d["fails"] = real_variant_fails(d["vs_C0_refit"], d["summary"]["n_failed"])
        d["all_device_mean_le_R2_threshold"] = bool(d["summary"]["all_device_mean"]
                                                    <= R2_THRESHOLD)
        d["network"] = network_dims(3, F2_REAL_VARIANTS[v]["hyper"])
        d["hyper"] = {k: list(x) if isinstance(x, tuple) else x for k, x in variants[v].items()}
        d["train_cards"] = F2_REAL_VARIANTS[v]["train_cards"]
        d["per_device"] = rows[v]
        res[v] = d
    reference = {"I3_2_M_bias": describe(m_bias, c0, None),
                 "C0_refit": {"summary": aggregate(c0)}}
    result = {"prereg": rel(PREREG), "smoke": bool(args.smoke), "options": OPTS,
              "n_folds": len(groups), "variants": res, "reference": reference,
              "round1_card_check": r1_check,
              "control_source": "out/research/explore/I2_1/real_dev/results.json "
                                "per_device.C0_refit", "timing": timing,
              "runtime_s": round(time.time() - t_start, 1)}
    (out / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "corrections.json").write_text(json.dumps(corr_json) + "\n")
    (out / "heldout_cards.json").write_text(json.dumps(cards) + "\n")
    print_table({v: res[v] for v in names} | reference)
    print(f"runtime {result['runtime_s']} s")
    return 0


def tier_b_tags(pdk: str, s: int) -> list[str]:
    from cryoml.pdk2.problem import TRUTH_ROOTS, load_truth_meta
    root = TRUTH_ROOTS[pdk]
    with np.load(root / "b" / f"set_{s}.npz") as data:
        if "tags" in data.files:
            return [str(t) for t in data["tags"]]
    return sorted(load_truth_meta("b", root)["sets"][str(s)])


def f2_pdk2(args) -> int:
    t_start = time.time()
    pdk = args.pdk
    out = FU / (f"smoke_f2_{pdk}_b" if args.smoke else f"f2_{pdk}_b")
    rec_dir = out / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    sets = [0] if args.smoke else [0, 1, 2]
    timing = {}
    with get_context("spawn").Pool(args.workers) as pool:
        need = []
        for s in sets:
            if not (rec_dir / f"{pdk}_b_set{s}.npz").exists():
                tags = tier_b_tags(pdk, s)
                if args.smoke:
                    tags = tags[:2] + [t for t in tags if "pmos" in t][:1]
                need += [(pdk, "b", s, t) for t in tags]
        if need:
            t0 = time.time()
            got = pool.map(collect_pdk2_task, need, chunksize=1)
            timing["collect_s"] = round(time.time() - t0, 1)
            timing["collect_n"] = len(got)
            for s in sets:
                recs = [r for ss, r in got if ss == s]
                if recs:
                    save_devices(rec_dir / f"{pdk}_b_set{s}", recs)
        records = [r for s in sets for r in load_devices(rec_dir / f"{pdk}_b_set{s}")]
        print(f"{pdk} tier b: {len(records)} records, C0 start mean "
              f"{np.nanmean([r.meta['rrms_c0'] for r in records]):.5f}", flush=True)
        key = lambda r: r.geometry
        groups = sorted({key(r) for r in records})
        variants = {"M_bias": smoke_hyper({}, args.smoke)}
        rows, corr_json, tm, cards = run_lodo(pool, variants, records, key, groups,
                                              {"M_bias": records},
                                              [("C0_refit", Correction("C0"))], None)
        timing.update(tm)
    clusters = {r.tag: r.geometry for r in records}
    d = describe(rows["M_bias"], rows["C0_refit"], clusters)
    d["passes_R4_rule"] = tier_b_pdk_passes(d["vs_C0_refit"], d["summary"]["n_failed"])
    d["ci_high_lt_1"] = bool(d["vs_C0_refit"].get("ci_high", np.nan) < 1.0)
    d["per_device"] = rows["M_bias"]
    result = {"prereg": rel(PREREG), "smoke": bool(args.smoke), "pdk": pdk, "tier": "b",
              "sets": sets, "options": OPTS, "n_targets": len(records),
              "n_folds": len(groups), "M_bias": d,
              "C0_refit": {"summary": aggregate(rows["C0_refit"]),
                           "per_device": rows["C0_refit"]},
              "C0_start": {"summary": aggregate({r.tag: r.meta["rrms_c0"] for r in records})},
              "records": rel(rec_dir), "timing": timing,
              "runtime_s": round(time.time() - t_start, 1)}
    (out / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "corrections.json").write_text(json.dumps(corr_json) + "\n")
    (out / "heldout_cards.json").write_text(json.dumps(cards) + "\n")
    print_table({"M_bias": d, "C0_refit": result["C0_refit"]})
    print(f"runtime {result['runtime_s']} s")
    return 0


def print_table(rows: dict) -> None:
    print("| variant | all-device | combined | median | failed | R vs C0_refit [95% CI] | wins |")
    print("|---|---|---|---|---|---|---|")
    for k, d in rows.items():
        s = d["summary"]
        c = d.get("vs_C0_refit")
        ci = (f"{c['ratio']:.3f} [{c.get('ci_low', np.nan):.3f}, {c.get('ci_high', np.nan):.3f}]"
              if c else "-")
        print(f"| {k} | {s['all_device_mean']:.5f} | {s['combined']:.5f} | {s['median']:.5f} "
              f"| {s['n_failed']} | {ci} | {c['wins'] if c else '-'}/{c['n_used'] if c else '-'} |")


def f2_summary(args) -> int:
    real = FU / "f2_real/results.json"
    pd = {p: FU / f"f2_{p}_b/results.json" for p in ("gf180", "ihp")}
    out = {"prereg": rel(PREREG), "note": "follow-up evidence; does not change the I3-2 decision"}
    rows = {}
    if real.exists():
        r = json.loads(real.read_text())
        rows.update({f"REAL_DEV {v}": d for v, d in r["variants"].items()})
        rows["REAL_DEV I3-2 M_bias (seeds 0-4, 3-16-16-1)"] = r["reference"]["I3_2_M_bias"]
        groups = {g: {"fails": any(r["variants"][v]["fails"] for v in vs
                                   if v in r["variants"]),
                      "complete": all(v in r["variants"] for v in vs)}
                  for g, vs in F2_GROUPS.items()}
        means = [r["variants"][v]["summary"]["all_device_mean"] for v in F2_GROUPS["a"]
                 if v in r["variants"]]
        means.append(r["reference"]["I3_2_M_bias"]["summary"]["all_device_mean"])
        groups["a"]["all_device_mean_range_4_seed_sets"] = [min(means), max(means)]
        out["real_groups"] = groups
    tb = {}
    for p, path in pd.items():
        if path.exists():
            d = json.loads(path.read_text())
            rows[f"{p} tier b M_bias"] = d["M_bias"]
            rows[f"{p} tier b C0_refit"] = d["C0_refit"]
            tb[p] = {"passes_R4_rule": d["M_bias"]["passes_R4_rule"],
                     "ci_high_lt_1": d["M_bias"]["ci_high_lt_1"],
                     "ratio": d["M_bias"]["vs_C0_refit"]["ratio"]}
    if tb:
        out["d_group"] = {"per_pdk": tb, "complete": len(tb) == 2,
                          "fails": len(tb) == 2 and not any(v["passes_R4_rule"]
                                                           for v in tb.values())}
    out["rows"] = {k: {"summary": v["summary"], "vs_C0_refit": v.get("vs_C0_refit"),
                       "vs_I3_2_M_bias": v.get("vs_I3_2_M_bias"), "fails": v.get("fails")}
                   for k, v in rows.items()}
    (FU / "f2_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print_table(rows)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("f1", "f1_plot", "f2_real", "f2_pdk2", "f2_summary"))
    ap.add_argument("--pdk", choices=("gf180", "ihp"), default="gf180")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tags", default="", help="f1: subset of REAL_DEV tags")
    ap.add_argument("--variants", default="", help="f2_real: subset of variants")
    ap.add_argument("--folds-limit", type=int, default=0, help="f2_real smoke: first N folds")
    ap.add_argument("--native-bin", default="",
                    help="f1: ngspice binary for the SPICE-native runs (default: Homebrew "
                         "ngspice-46); F1b uses the ngspice-41 OSDI build")
    ap.add_argument("--native-label", default="ng46",
                    help="f1: key of the --native-bin simulator in the outputs (default ng46)")
    ap.add_argument("--out-name", default="f1",
                    help="f1/f1_plot: output directory under followup/ and figure prefix")
    ap.add_argument("--amendment", action="append", default=[],
                    help="f1: additional amendment file recorded in summary.json (repeatable)")
    ap.add_argument("--openvaf", default="",
                    help="f1: Verilog-A compiler (default: OpenVAF-Reloaded v24, OSDI 0.4); "
                         "F1c uses OpenVAF 23.5.0 (OSDI 0.3) for the ngspice-41 OSDI build")
    ap.add_argument("--openvaf-args", default="",
                    help="f1: extra compiler arguments, e.g. '--target aarch64-apple-darwin'")
    ap.add_argument("--smoke", action="store_true",
                    help="code check: smoke_* outputs; f2: first seed only, 200 steps")
    args = ap.parse_args()
    if args.mode != "f2_summary":
        FU.mkdir(parents=True, exist_ok=True)
    return {"f1": f1, "f1_plot": f1_plot, "f2_real": f2_real, "f2_pdk2": f2_pdk2,
            "f2_summary": f2_summary}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
