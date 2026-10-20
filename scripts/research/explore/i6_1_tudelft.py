#!/usr/bin/env python3
"""I6-1: the I3-2 bias-only correction on TU Delft 40-nm CMOS at 4.2 K and 295 K.

Pre-registration: ``out/research/explore/I6_1/prereg.json``.
Module: ``cryoml.explore.i6_1_tudelft``.

Modes:
  data    read the campaign-1 DC files (restricted unpickler), write
          ``data/tudelft40.npz`` (not committed) and ``data/manifest.json``
  split   DEV / TEST split by W/L pair (seed below) -> split.json
  stock   stock-card RRMS of every DEV device at both temperatures (no fit)
          and I-V check plots of --plot devices (DEV only)
  start   starting cards: polish from the stock card (DEV devices, --temps)
  screen  B1 edge screen on the starting cards (per temperature)
  lodo    leave-one-geometry-out at --temp: C0_refit, M_bias, T1_inloop, B1,
          B2, B3_d3, B5
  decide  pre-registered rule -> decision.json
  figs    per-device RRMS by arm, GMR by temperature, I-V plots of --plot devices

The fitting modes read only the DEV devices (``split.json``).  ``--smoke``
writes to ``smoke_*`` with the reduced budgets of ``SMOKE``.

Run with ``python -I`` (the script puts ``src`` on ``sys.path`` itself):

  .venv/bin/python -I scripts/research/explore/i6_1_tudelft.py --mode data
  .venv/bin/python -I scripts/research/explore/i6_1_tudelft.py --mode start --workers 6
  .venv/bin/python -I scripts/research/explore/i6_1_tudelft.py --mode lodo --temp 4 --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.explore import i6_1_tudelft as m  # noqa: E402

OUT = m.OUT
DATA = OUT / "data"
NPZ = DATA / "tudelft40.npz"
MANIFEST = DATA / "manifest.json"
SPLIT = OUT / "split.json"
SPLIT_SEED = 20261020

PREREG_OPTIONS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
START_OPTIONS = {"nfev": 3000, "irls": 3, "diff_step": 0.01}
SMOKE = {"start_nfev": 100, "folds": 2, "seeds": (0,), "steps": 200, "one_per_group": True}
ARMS = ("C0_refit", "M_bias", "T1_inloop", "B1", "B2", "B3_d3", "B5")
RULE_BASELINES = ("B1", "B2", "B3_d3")
T1_PLUS = 0.005
PRIMARY_T = 4
CONTROL_T = 295
PREMISE_FACTOR = 1.5


def load(tags=None):
    manifest = json.loads(MANIFEST.read_text())
    return m.load_dataset(NPZ, manifest, tags), manifest


def dev_devices(smoke: bool = False) -> list:
    """The DEV devices only (TEST arrays are not read).  Smoke: the
    lowest-numbered device of each DEV group."""
    split = json.loads(SPLIT.read_text())
    devices, _ = load(split["DEV"])
    if sorted(d.tag for d in devices) != sorted(split["DEV"]):
        raise RuntimeError("DEV devices missing from the dataset")
    if smoke and SMOKE["one_per_group"]:
        first = {}
        for d in sorted(devices, key=lambda d: d.device):
            first.setdefault(d.group, d)
        devices = sorted(first.values(), key=lambda d: d.tag)
    return devices


def odir(args, name: str) -> Path:
    return OUT / (f"smoke_{name}" if args.smoke else name)


def records_path(args, t: int) -> Path:
    return odir(args, "start") / f"records_{t}K"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def run_data(args) -> int:
    t0 = time.time()
    devices = m.discover()
    saved = m.save_dataset(devices, NPZ)
    fitted, quant, n_inc, stage = {}, [], {}, {t: [] for t in m.TEMPS_K}
    biases = {}
    for dev in devices:
        for t in m.TEMPS_K:
            stage[t].append(dev.temps_K[t])
            rows_t, k_inc = [], 0
            for c in m.fit_curves(dev, t):
                inc, st = m.inclusion(c), m.curve_stats(c)
                quant.append(st)
                k_inc += int(inc["included"])
                rows_t.append(f"{c.key}|{c.fixed:.6g}|{c.sweep[0]:.6g}..{c.sweep[-1]:.6g}/"
                              f"{len(c.sweep)}|{inc['start']}|{inc['den']:.4g}|"
                              f"{int(inc['included'])}|{st['max_abs_id']:.4g}")
            fitted[f"{dev.tag}@{t}"] = rows_t
            n_inc[f"{dev.tag}@{t}"] = k_inc
            fc = m.fit_curves(dev, t)
            key = f"{dev.dev_type}@{t}K"
            b = {"vg_sweep": [fc[0].sweep[0], fc[0].sweep[-1], len(fc[0].sweep)],
                 "vd_sweep": [fc[12].sweep[0], fc[12].sweep[-1], len(fc[12].sweep)],
                 "idvg_vd": [round(c.fixed, 4) for c in fc[:12]],
                 "idvd_vg": [round(c.fixed, 4) for c in fc[12:]]}
            if biases.setdefault(key, b) != b:
                raise RuntimeError(f"{dev.tag}: bias grid differs within {key}")
    inc = [r.split("|")[5] == "1" for v in fitted.values() for r in v]
    groups = {}
    for d in devices:
        g = groups.setdefault(d.group, {"dev_type": d.dev_type, "W_um": d.W_um, "L_um": d.L_um,
                                        "sub_block": d.sub_block, "sample": d.sample,
                                        "devices": []})
        g["devices"].append(d.device)
    steps = [q["min_step"] for q in quant if q["min_step"]]
    per_t_inc = {str(t): {"n_included": sum(v for k, v in n_inc.items() if k.endswith(f"@{t}")),
                          "min_per_device": min(v for k, v in n_inc.items() if k.endswith(f"@{t}")),
                          "max_per_device": max(v for k, v in n_inc.items() if k.endswith(f"@{t}"))}
                 for t in m.TEMPS_K}
    manifest = {
        "id": "I6_1_data",
        "source": {"doi": m.DOI, "license": "CC0",
                   "title": "Data underlying the publication: Cryogenic Characterization of "
                            "Low-Frequency Noise in 40-nm CMOS (G. Kiene, S. Ilik, L. Mastrodomenico, "
                            "M. Babaie, F. Sebastiano, TU Delft)",
                   "files": "main.zip: repository_for_upload_slim/measurement_automation/main/results/dc"
                            "/*.npy (1,717 files; I3-1 download, 16.3 MB on disk)",
                   "local_copy": "data/external/delft_kiene_40nm_lfn (not committed)",
                   "reader": "cryoml.explore.i6_1_tudelft.read_npy_dict: unpickler restricted to "
                             "numpy _reconstruct / ndarray / dtype / scalar; run with python -I"},
        "npz": str(NPZ.relative_to(ROOT)) + " (not committed; regenerate with --mode data)",
        "selection": "campaign 1 files (campaign = 'Standard triode (Vds = 50 mV) and diode (Vgs = Vds) "
                     "sweep, 11 points, main bias exploration' in the dataset's campaign slides); "
                     "devices with a campaign-1 file at file temperature 4 and 300. Other campaigns: "
                     "2 (1,400 files; 2 Id-Vg and 1 Id-Vd over Vg 1.05-1.1 V), 3-10 and 99 (debug, "
                     "repeat or temperature-sweep runs on few devices) are not used",
        "n_devices": len(devices), "groups": groups,
        "stage_temperature_K": {str(t): [float(min(v)), float(max(v))] for t, v in stage.items()},
        "deck_temperature": "per device and label, the mean of temperature_meas_start and "
                            "temperature_meas_end of the file",
        "process": "commercial 40-nm bulk CMOS (foundry not named in the dataset); all 69 devices "
                   "are low-Vt core devices, NF = 1; nMOS on sample (die) 3, pMOS on sample 5",
        "terminals": "voltages are Vgs and Vds (pMOS negative, source reference); the drain current is "
                     "> 0 for nMOS and < 0 for pMOS; the files hold no bulk bias and no gate, source or "
                     "bulk current; bulk is taken as tied to the source (Vbs = 0)",
        "bias_grid": biases,
        "fitted_curves": {
            "rule": "per device and temperature, the 12 Id-Vg curves (i_d_const_d; keys idvg@d0..d11 by "
                    "ascending |Vd|: 0.05 V, then 11 values from v_diode_start to 1.1 V) and the 11 Id-Vd "
                    "curves (i_d_const_g; keys idvd@g0..g10 by ascending |Vg|), 111 points each, 23 "
                    "curves. The 11-point i_d_diode and i_d_triode arrays are not used",
            "note": "the Vg start and the Vd / Vg matrix values depend on polarity and temperature "
                    "(bias_grid) and are identical across devices of one polarity and temperature"},
        "inclusion_rule": (
            "frozen from the measured current only (cryoml.pdk2.metric): clean_current (zero the "
            "points before the last exact zero), trim the leading points up to the first "
            "|I| > 1e-10 A, denominator = mean |I| of the trimmed curve; a curve is included iff "
            "the denominator > 3e-7 A (Id-Vd) or > 2e-7 A (Id-Vg)"),
        "inclusion_summary": {"n_fitted_curves": len(inc), "n_included": int(sum(inc)),
                              "per_temperature": per_t_inc},
        "quantization": {
            "storage": "float64 arrays (no printed-digit rounding)",
            "smallest_nonzero_abs_id_A": min(q["min_nonzero_abs_id"] for q in quant
                                             if q["min_nonzero_abs_id"] > 0),
            "smallest_step_between_distinct_abs_id_values_A": min(steps),
            "n_exact_zero_id_values": int(sum(q["n_exact_zero"] for q in quant)),
            "n_points_with_sign_opposite_to_the_bias": int(sum(q["n_sign_wrong"] for q in quant))},
        "compliance": {
            "stated_in_files": "no compliance or SMU setting is stored",
            "largest_abs_id_A": float(max(q["max_abs_id"] for q in quant)),
            "max_identical_values_at_curve_maximum": int(max(q["n_identical_at_max"] for q in quant)),
            "check": "a clipped curve would repeat its largest |id| over its last points; "
                     "max_identical_values_at_curve_maximum counts repeats per curve"},
        "devices": [m.device_row(d) for d in devices],
        "fitted_columns": "key|fixed_V|sweep_V first..last/n|trim_start|denominator_A|included|"
                          "max_abs_id_A",
        "fitted": fitted, **saved,
        "runtime_s": round(time.time() - t0, 1)}
    m.write_json(MANIFEST, manifest)
    print(json.dumps({k: manifest[k] for k in ("n_devices", "stage_temperature_K", "inclusion_summary",
                                                "quantization", "compliance", "bias_grid")}, indent=1))
    for g, v in groups.items():
        print(g, v["sub_block"], v["sample"], len(v["devices"]))
    return 0


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def run_split(args) -> int:
    devices, _ = load()
    split = m.make_split(devices, SPLIT_SEED)
    split["rule"] = m.make_split.__doc__.strip()
    split["DEV_groups"] = sorted({m.group_of(t) for t in split["DEV"]})
    split["TEST_groups"] = sorted({m.group_of(t) for t in split["TEST"]})
    split["written"] = "2026-10-20, before any fit; TEST devices are not loaded by the fitting modes"
    m.write_json(SPLIT, split)
    print(json.dumps({k: v for k, v in split.items() if k not in ("DEV", "TEST")}, indent=1))
    return 0


# ---------------------------------------------------------------------------
# stock
# ---------------------------------------------------------------------------
def stock_task(task):
    dev, t = task
    curves = m.fit_curves(dev, t)
    sims = m.simulate(dev, m.model_block(dev.dev_type), curves, dev.temps_K[t])
    sc = m.score(curves, sims)
    return {"tag": dev.tag, "T_K": t, "rrms": sc["rrms"], "n_curves": sc["n_curves"],
            "per_curve": sc["per_curve"], "failed": sc["failed"], "sims": [s.tolist() for s in sims]}


def plot_device(dev, t, sims_by_label: dict, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = m.fit_curves(dev, t)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    panels = (("idvg@d", "Id-Vg", "|Vg| (V)", True), ("idvg@d", "Id-Vg", "|Vg| (V)", False),
              ("idvd@g", "Id-Vd", "|Vd| (V)", False))
    styles = ["-", "--", ":"]
    for ax, (prefix, name, xl, logy) in zip(axes, panels):
        for k, c in enumerate(curves):
            if not c.key.startswith(prefix) or (prefix == "idvg@d" and not logy and k % 3):
                continue
            inc = m.inclusion(c)
            color = f"C{k % 10}"
            lab = f"{c.key} ({'Vd' if c.kind == 'idvg' else 'Vg'}={c.fixed:.3g})"
            ax.plot(np.abs(c.sweep), np.abs(c.id), "o", ms=2.5, color=color,
                    label=lab + ("" if inc["included"] else " excl."))
            for st, (lab_s, sims) in zip(styles, sims_by_label.items()):
                ax.plot(np.abs(c.sweep), np.abs(np.asarray(sims[k])), st, color=color, lw=1)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("|Id| (A)")
        ax.set_title(name + (" (log)" if logy else ""))
        ax.legend(fontsize=5.5, ncol=2)
    fig.suptitle(title + "  points: measured; lines: " + ", ".join(
        f"{s} {lab}" for s, lab in zip(styles, sims_by_label)), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_stock(args) -> int:
    devices = dev_devices()
    tasks = [(d, t) for d in devices for t in m.TEMPS_K]
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(stock_task, tasks, chunksize=1)
    by = {(r["tag"], r["T_K"]): r for r in rows}
    table = {t: {d.tag: by[(d.tag, t)]["rrms"] for d in devices} for t in m.TEMPS_K}
    summary = {f"{t}K": m.aggregate(table[t]) for t in m.TEMPS_K}
    out = {"what": "stock BSIM4 card (ngspice-41 examples/mos/ro_17_4.cir N1/P1, temperature "
                   "coefficients set to 0), stage temperature, measured bias grid, no fit; strict "
                   "paper RRMS on the 23 curves with the frozen measured inclusion; DEV devices only",
           "per_device": {f"{t}": table[t] for t in m.TEMPS_K}, "summary": summary,
           "per_group": {f"{t}K": {g: m.aggregate({k: v for k, v in table[t].items() if m.group_of(k) == g})
                                   for g in sorted({d.group for d in devices})} for t in m.TEMPS_K},
           "per_curve": {f"{r['tag']}@{r['T_K']}": r["per_curve"] for r in rows},
           "runtime_s": round(time.time() - t0, 1)}
    m.write_json(OUT / "stock" / "stock_rrms.json", out)
    fig_dir = OUT / "stock" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for tag in [x for x in args.plot.split(",") if x]:
        dev = [d for d in devices if d.tag == tag]
        if not dev:
            raise RuntimeError(f"{tag} is not a DEV device; plots are DEV only")
        for t in m.TEMPS_K:
            r = by[(tag, t)]
            plot_device(dev[0], t, {"stock card": r["sims"]}, fig_dir / f"stock_{tag}_{t}K.png",
                        f"{tag} {dev[0].temps_K[t]:.2f} K, stock RRMS {r['rrms']:.3f}")
    for t in m.TEMPS_K:
        print(f"{t} K", {k: round(v, 4) for k, v in summary[f"{t}K"].items()})
        for g, v in out["per_group"][f"{t}K"].items():
            print(f"   {g:24s} mean {v['all_device_mean']:.3f} median {v['median']:.3f} n {v['n']}")
    print("runtime", out["runtime_s"])
    return 0


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
def start_task(task):
    from cryoml.ngspice_fit import polish
    dev, t, opts = task
    t0 = time.time()
    n_timeout0 = m.N_TIMEOUT
    box = m.base_box(dev)
    prob = m.MeasProblem(dev, t, box)
    local0 = box.params_to_local(box.published)
    stock = m.score(prob.curves, prob.simulate(local0))
    res = polish(prob, local0, max_nfev=opts["nfev"], irls_rounds=opts["irls"],
                 diff_step=opts["diff_step"])
    params = box.local_to_params(res["local"])
    rec = m.to_record(dev, t, prob.simulate_params(params), params)
    sc = m.score(prob.curves, [c.sim for c in rec.curves])
    loc = np.asarray(res["local"])
    row = {"tag": dev.tag, "T_K": t, "stock_rrms": stock["rrms"], "start_rrms": sc["rrms"],
           "n_sim_timeout": m.N_TIMEOUT - n_timeout0,
           "n_curves": sc["n_curves"], "n_sim": prob.n_sim, "history": res["history"],
           "n_active": int(box.n_active),
           "at_bound": [n for n, z, a in zip(box.names, loc, box.active) if a and abs(z) > 0.999],
           "params": params, "runtime_s": round(time.time() - t0, 1)}
    return rec, row


def run_start(args) -> int:
    from cryoml.explore.e1_correction import save_devices
    t_start = time.time()
    devices = dev_devices(args.smoke)
    temps = [int(x) for x in args.temps.split(",")] if args.temps else list(m.TEMPS_K)
    opts = dict(START_OPTIONS)
    if args.smoke:
        opts["nfev"] = SMOKE["start_nfev"]
        temps = temps if args.temps else [PRIMARY_T]
    tasks = [(d, t, opts) for t in temps for d in devices]
    with get_context("spawn").Pool(args.workers) as pool:
        out = pool.map(start_task, tasks, chunksize=1)
    target = odir(args, "start")
    target.mkdir(parents=True, exist_ok=True)
    summary = {"options": opts, "box": m.base_box.__doc__.strip(), "smoke": bool(args.smoke),
               "per_temperature": {}}
    prev = target / "summary.json"
    if prev.exists():
        old = json.loads(prev.read_text())
        summary["per_temperature"] = {k: v for k, v in old.get("per_temperature", {}).items()
                                      if int(k) not in temps}
    for t in temps:
        pairs = sorted([(r, row) for r, row in out if row["T_K"] == t], key=lambda x: x[0].tag)
        save_devices(records_path(args, t), [r for r, _ in pairs])
        rows = {row["tag"]: row for _, row in pairs}
        summary["per_temperature"][str(t)] = {
            "stock": m.aggregate({k: v["stock_rrms"] for k, v in rows.items()}),
            "start": m.aggregate({k: v["start_rrms"] for k, v in rows.items()}),
            "start_per_group": {g: m.aggregate({k: v["start_rrms"] for k, v in rows.items()
                                                if m.group_of(k) == g})
                                for g in sorted({m.group_of(k) for k in rows})},
            "rows": rows}
        s = summary["per_temperature"][str(t)]
        print(f"{t} K: stock {s['stock']['all_device_mean']:.4f} -> start "
              f"{s['start']['all_device_mean']:.4f} (median {s['start']['median']:.4f})", flush=True)
        for k, v in rows.items():
            print(f"  {k:28s} {v['stock_rrms']:.4f} -> {v['start_rrms']:.4f}  n_sim {v['n_sim']} "
                  f"at_bound {v['at_bound']}  {v['runtime_s']} s")
    summary["runtime_s"] = round(time.time() - t_start, 1)
    m.write_json(target / "summary.json", summary)
    print(f"runtime {summary['runtime_s']} s")
    return 0


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def screen_task(task):
    from cryoml.explore.i4_1_baselines import edge_response
    dev, rec = task
    prob = m.MeasProblem(dev, rec.meta["T_K"], m.base_box(dev))
    start = rec.meta["params"]
    ref = prob.simulate_params(start)
    out = {}
    for name, e in m.b1_extras(dev, start).items():
        lo, hi = e["bounds"]
        if not hi > lo:
            out[name] = {"value": e["value"], "bounds": e["bounds"], "active": False}
            continue
        resp = [edge_response(ref, prob.simulate_params({**start, name: v}), prob.layout)
                for v in (lo, hi)]
        out[name] = {"value": e["value"], "bounds": e["bounds"], "log": e["log"], "active": True,
                     "response_lo": resp[0], "response_hi": resp[1]}
    return dev.tag, out


def run_screen(args) -> int:
    from cryoml.explore.e1_correction import load_devices
    from cryoml.explore.i4_1_baselines import B1_THRESHOLD
    t0 = time.time()
    devices = {d.tag: d for d in dev_devices(args.smoke)}
    temps = [int(x) for x in args.temps.split(",")] if args.temps else list(m.TEMPS_K)
    out = {"rule": f"per temperature: a candidate is kept if the edge response at either box edge is "
                   f">= {B1_THRESHOLD} on at least one starting card of the DEV devices at that "
                   f"temperature (non-finite responses count as 0)", "smoke": bool(args.smoke)}
    target = odir(args, "start") / "b1_screen.json"
    if target.exists():
        out.update({k: v for k, v in json.loads(target.read_text()).items()
                    if k.endswith("K") and int(k[:-1]) not in temps})
    with get_context("spawn").Pool(args.workers) as pool:
        for t in temps:
            path = records_path(args, t)
            if not path.with_suffix(".json").exists():
                continue
            recs = load_devices(path)
            rows = dict(pool.map(screen_task, [(devices[r.tag], r) for r in recs], chunksize=1))
            names = sorted({n for r in rows.values() for n in r})
            per = {}
            for n in names:
                resp = []
                for r in rows.values():
                    v = r.get(n, {})
                    if v.get("active"):
                        lo, hi = v["response_lo"], v["response_hi"]
                        resp.append(max(lo if np.isfinite(lo) else 0.0, hi if np.isfinite(hi) else 0.0))
                per[n] = {"max_response": float(max(resp)) if resp else 0.0,
                          "n_active_devices": len(resp),
                          "n_devices_ge_threshold": int(sum(x >= B1_THRESHOLD for x in resp))}
            sel = [n for n in names if per[n]["max_response"] >= B1_THRESHOLD]
            out[f"{t}K"] = {"candidates": names, "selected": sel, "n_selected": len(sel),
                            "per_parameter": per, "per_device": rows}
            print(f"{t} K: {len(names)} candidates, {len(sel)} selected: {sel}", flush=True)
    out["runtime_s"] = round(time.time() - t0, 1)
    m.write_json(target, out)
    print(f"runtime {out['runtime_s']} s")
    return 0


# ---------------------------------------------------------------------------
# lodo
# ---------------------------------------------------------------------------
def train_task(task):
    kind, arm, recs, hyper = task
    if kind == "bias":
        from cryoml.explore.i3_2_bias import fit_bias
        return fit_bias(arm, recs, hyper=hyper, threads=1)
    if kind == "t1":
        from cryoml.explore.e1_correction import fit_arm
        return fit_arm("T1", recs, hyper=hyper, threads=1)
    if kind == "poly":
        from cryoml.explore.i4_1_baselines import fit_poly
        return fit_poly(arm, 3, recs, hyper=hyper)
    raise KeyError(kind)


def refit_task(task):
    """One held-out refit (as the I5-1 refit_task)."""
    from cryoml.explore.e1_correction import Correction, score_device
    from cryoml.explore.i3_2_bias import fit_bias
    from cryoml.explore.i4_1_baselines import (AugProblem, b2_box, extend_box, matched_nfev,
                                               measured_pmax)
    from cryoml.ngspice_fit import polish
    label, kind, payload, rec, dev, opts = task
    t0 = time.time()
    n_timeout0 = m.N_TIMEOUT
    t = int(rec.meta["T_K"])
    start = dict(rec.meta["params"])
    box = m.base_box(dev)
    nfev, irls, step = opts["heldout_nfev"], opts["heldout_irls"], opts["diff_step"]
    row = {"label": label, "tag": rec.tag, "T_K": t}
    corr_final = Correction("C0")
    if kind in ("corr", "b5"):
        prob = m.MeasProblem(dev, t, box)
        if kind == "b5":
            corr = fit_bias("M_bias", [rec], hyper=payload.get("hyper"), threads=1)
            row["rrms_start_g0_insample"] = float(score_device(rec, corr.apply(rec))["rrms"])
        else:
            corr = payload["corr"]
        res = polish(m.corrected_problem(prob, corr), box.params_to_local(start), max_nfev=nfev,
                     irls_rounds=irls, diff_step=step)
        params = box.local_to_params(res["local"])
        new = m.to_record(dev, t, prob.simulate_params(params), params)
        corr_final = corr
        if kind == "b5":
            g1 = fit_bias("M_bias", [new], hyper=payload.get("hyper"), threads=1)
            row["rrms_g0_on_refit_card"] = float(score_device(new, corr.apply(new))["rrms"])
            corr_final = g1
        n_sim = prob.n_sim
        row["n_active"] = int(box.n_active)
    elif kind == "b1":
        extras = m.b1_extras(dev, start, payload["names"])
        box1 = extend_box(box, extras)
        prob = m.MeasProblem(dev, t, box1)
        nfev = matched_nfev(box.n_active, box1.n_active, nfev, irls)
        local0 = box1.params_to_local({**start, **{n: e["value"] for n, e in extras.items()}})
        res = polish(prob, local0, max_nfev=nfev, irls_rounds=irls, diff_step=step)
        params = box1.local_to_params(res["local"])
        new = m.to_record(dev, t, prob.simulate_params(params), params)
        n_sim = prob.n_sim
        row.update(n_active=int(box1.n_active), n_extra_active=int(box1.n_active - box.n_active),
                   max_nfev=nfev)
    elif kind == "b2":
        prob = m.MeasProblem(dev, t, box)
        rmax = m.rext_max(dev, start)
        box2 = b2_box(box, rmax)
        aug = AugProblem(prob, box2, lambda card, rext: prob.simulate_params(card, rext),
                         measured_pmax(rec))
        nfev = matched_nfev(box.n_active, box2.n_active, nfev, irls)
        local0 = box2.params_to_local({**start, "rext_ohm": 0.0, "heat_kappa": 0.0})
        res = polish(aug, local0, max_nfev=nfev, irls_rounds=irls, diff_step=step)
        params = box2.local_to_params(res["local"])
        new = m.to_record(dev, t, aug.simulate(res["local"]), params)
        n_sim = aug.n_sim
        row.update(n_active=int(box2.n_active), max_nfev=nfev, rext_max_ohm=rmax,
                   pmax_w=aug.pmax, rext_ohm=params["rext_ohm"], heat_kappa=params["heat_kappa"])
    else:
        raise KeyError(kind)
    after = score_device(new, corr_final.apply(new))
    if kind != "b5":
        row["rrms_start"] = float(score_device(rec, corr_final.apply(rec))["rrms"])
    row.update(rrms=after["rrms"], failed=after["failed"], n_sim=n_sim, history=res["history"],
               params=params, runtime_s=round(time.time() - t0, 2),
               n_sim_timeout=m.N_TIMEOUT - n_timeout0)
    print(f"  {label:10s} {rec.tag:28s} {row['rrms']:.4f} n_sim {n_sim} timeouts "
          f"{row['n_sim_timeout']} {row['runtime_s']} s", flush=True)
    return row


def comparisons(scores: dict, tags: list) -> dict:
    """GMR with device bootstrap (decision) and (polarity, W, L) cluster
    bootstrap (reported)."""
    from cryoml.explore.i3_2_bias import paired_ratio
    sub = {k: {t: v[t] for t in tags if t in v} for k, v in scores.items()}
    clusters = {t: m.group_of(t) for t in tags}
    out = {}
    for x in sub:
        pairs = []
        if x != "M_bias" and "M_bias" in sub:
            pairs.append((f"M_bias_vs_{x}", "M_bias", x))
        if x != "C0_refit" and "C0_refit" in sub:
            pairs.append((f"{x}_vs_C0_refit", x, "C0_refit"))
        for name, a, b in pairs:
            out[name] = paired_ratio(sub[a], sub[b])
            cl = paired_ratio(sub[a], sub[b], clusters=clusters)
            out[name]["cluster_bootstrap"] = {k: cl.get(k) for k in ("ci_low", "ci_high",
                                                                     "resample_unit")}
    return out


def summaries(scores: dict, tags: list) -> dict:
    out = {"all": {k: m.aggregate({t: v[t] for t in tags if t in v}) for k, v in scores.items()}}
    for g in sorted({m.group_of(t) for t in tags}):
        sel = [t for t in tags if m.group_of(t) == g]
        out[g] = {k: m.aggregate({t: v[t] for t in sel if t in v}) for k, v in scores.items()}
    return out


def run_lodo(args) -> int:
    from cryoml.explore.e1_correction import Correction, load_devices, score_device
    from cryoml.explore.i3_2_bias import merge_seed_results, split_seed_tasks
    t_start = time.time()
    t = int(args.temp)
    devices = {d.tag: d for d in dev_devices(args.smoke)}
    recs = load_devices(records_path(args, t))
    if sorted(r.tag for r in recs) != sorted(devices):
        raise RuntimeError("starting records differ from DEV")
    arms = [a for a in args.arms.split(",") if a]
    opts = dict(PREREG_OPTIONS)
    hyper = {} if not args.smoke else {"seeds": SMOKE["seeds"], "steps": SMOKE["steps"]}
    hyper_short = {} if not args.smoke else {"steps": SMOKE["steps"]}
    groups = sorted({r.meta["group"] for r in recs})
    if args.smoke:
        groups = [next(g for g in groups if "_nmos_" in g), next(g for g in groups if "_pmos_" in g)]
    held = sorted(r.tag for r in recs if r.meta["group"] in groups)
    by_tag = {r.tag: r for r in recs}
    train = {g: [r for r in recs if r.meta["group"] != g] for g in groups}
    screen = json.loads((odir(args, "start") / "b1_screen.json").read_text())
    b1_names = screen[f"{t}K"]["selected"] if "B1" in arms else []
    preregistered = (not args.smoke and set(ARMS) <= set(arms))
    print(f"I6_1 lodo {t} K: {len(groups)} folds, {len(held)} held-out devices, arms {arms}, "
          f"opts {opts}, hyper {hyper}, preregistered={preregistered}", flush=True)
    timing = {}
    corrs = {}
    with get_context("spawn").Pool(args.workers) as pool:
        t0 = time.time()
        bias_tasks = [("M_bias", train[g]) for g in groups] if "M_bias" in arms else []
        expanded, owner = split_seed_tasks(bias_tasks, hyper)
        tasks = [("bias", a, r, hp) for a, r, hp in expanded]
        other = []
        if "T1_inloop" in arms:
            other += [("t1", "T1", train[g], hyper_short) for g in groups]
        if "B3_d3" in arms:
            other += [("poly", "B3_d3", train[g], hyper_short) for g in groups]
        results = pool.map(train_task, tasks + other, chunksize=1)
        merged = merge_seed_results(len(bias_tasks), owner, results[:len(tasks)])
        if "M_bias" in arms:
            for g, c in zip(groups, merged):
                corrs[f"M_bias|{g}"] = c
        rest = iter(results[len(tasks):])
        for arm in ("T1_inloop", "B3_d3"):
            if arm in arms:
                for g in groups:
                    corrs[f"{arm}|{g}"] = next(rest)
        timing["train_s"] = round(time.time() - t0, 1)
        print(f"training done ({timing['train_s']} s, {len(tasks + other)} tasks)", flush=True)
        refits = []
        for h in held:
            rec, dev = by_tag[h], devices[h]
            g = rec.meta["group"]
            for arm in arms:
                if arm == "C0_refit":
                    refits.append((arm, "corr", {"corr": Correction("C0")}, rec, dev, opts))
                elif arm in ("M_bias", "T1_inloop", "B3_d3"):
                    refits.append((arm, "corr", {"corr": corrs[f"{arm}|{g}"]}, rec, dev, opts))
                elif arm == "B1":
                    refits.append((arm, "b1", {"names": b1_names}, rec, dev, opts))
                elif arm == "B2":
                    refits.append((arm, "b2", {}, rec, dev, opts))
                elif arm == "B5":
                    refits.append((arm, "b5", {"hyper": hyper}, rec, dev, opts))
        cost = {"B1": 0, "B5": 1, "B2": 2}
        order = sorted(range(len(refits)), key=lambda i: cost.get(refits[i][0], 3))
        t0 = time.time()
        rows_l = pool.map(refit_task, [refits[i] for i in order], chunksize=1)
        timing["refit_s"] = round(time.time() - t0, 1)
        timing["refit_n"] = len(refits)
    rows = {}
    for row in rows_l:
        rows.setdefault(row["label"], {})[row["tag"]] = row
    scores = {a: {tg: r["rrms"] for tg, r in v.items()} for a, v in rows.items()}
    scores["start"] = {h: float(score_device(by_tag[h], [c.sim for c in by_tag[h].curves])["rrms"])
                       for h in held}
    result = {"name": odir(args, f"lodo_{t}K").name, "T_K": t,
              "prereg": "out/research/explore/I6_1/prereg.json",
              "preregistered_config": preregistered, "smoke": bool(args.smoke), "options": opts,
              "hyper_overrides": hyper, "arms": arms, "folds": groups, "held_out": held,
              "b1_names": b1_names, "summary": summaries(scores, held),
              "comparisons": comparisons(scores, held), "per_device": scores,
              "rows": {a: {tg: {k: v for k, v in r.items() if k != "params"} for tg, r in v.items()}
                       for a, v in rows.items()},
              "timing": timing, "runtime_s": round(time.time() - t_start, 1)}
    target = odir(args, f"lodo_{t}K")
    m.write_json(target / "results.json", result)
    (target / "corrections.json").write_text(json.dumps({k: c.to_json() for k, c in corrs.items()}) + "\n")
    m.write_json(target / "cards.json", {a: {tg: r["params"] for tg, r in v.items()} for a, v in rows.items()})
    print_summary(result)
    return 0


def print_summary(result: dict) -> None:
    s = result["summary"]["all"]
    for a, v in s.items():
        print(f"  {a:12s} mean {v['all_device_mean']:.4f} combined {v['combined']:.4f} "
              f"median {v['median']:.4f} failed {v['n_failed']}")
    for k, v in result["comparisons"].items():
        cb = v.get("cluster_bootstrap", {})
        print(f"  {k:28s} GMR {v['ratio']:.3f} [{v.get('ci_low', float('nan')):.3f}, "
              f"{v.get('ci_high', float('nan')):.3f}] cluster [{cb.get('ci_low') or float('nan'):.3f}, "
              f"{cb.get('ci_high') or float('nan'):.3f}] wins {v['wins']}/{v['n_used']}")
    print(f"  timing {result['timing']} runtime {result['runtime_s']} s")


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def conditions(result: dict) -> dict:
    """R1-R3 and the baseline conditions at one temperature."""
    s = result["summary"]["all"]
    c = result["comparisons"]
    r1 = c["M_bias_vs_C0_refit"]
    checks = {"R1_ci_high_lt_1": bool(np.isfinite(r1["ci_high"]) and r1["ci_high"] < 1.0),
              "R2_mean_le_T1_inloop_plus_0.005": bool(
                  s["M_bias"]["all_device_mean"] <= s["T1_inloop"]["all_device_mean"] + T1_PLUS),
              "R3_zero_failed": bool(s["M_bias"]["n_failed"] == 0)}
    per = {}
    for x in RULE_BASELINES:
        cx = c.get(f"M_bias_vs_{x}", {})
        per[x] = {k: cx.get(k) for k in ("ratio", "ci_low", "ci_high", "wins", "n_used")}
        per[x]["cluster_bootstrap"] = cx.get("cluster_bootstrap")
        per[x]["beaten"] = bool(np.isfinite(cx.get("ci_high", np.nan)) and cx["ci_high"] < 1.0)
    return {"checks": checks, "R1_detail": r1,
            "R2_detail": {"M_bias_mean": s["M_bias"]["all_device_mean"],
                          "T1_inloop_mean": s["T1_inloop"]["all_device_mean"]},
            "baselines": per, "all_baselines_beaten": all(v["beaten"] for v in per.values()),
            "C0_refit_mean": s["C0_refit"]["all_device_mean"]}


def outcome(c4: dict, c295: dict, contrast: dict, premise: bool) -> tuple[str, str]:
    """Pre-registered outcome label and its interpretation (prereg
    criterion.outcomes)."""
    ch = c4["checks"]
    if not ch["R1_ci_high_lt_1"]:
        return "refuted", "4.2 K R1 fails: the correction does not lower held-out error where the " \
                          "compact model is used far outside its validity range"
    if not (ch["R2_mean_le_T1_inloop_plus_0.005"] and ch["R3_zero_failed"]):
        return "inconclusive", "4.2 K R1 holds but R2 or R3 fails"
    if not c4["all_baselines_beaten"]:
        return "matched by baselines", "4.2 K R1-R3 hold but at least one of B1, B2, B3_d3 is not " \
                                       "beaten: the 4.2 K error is absorbed equally by the baseline"
    r295 = c295["checks"]["R1_ci_high_lt_1"]
    k = bool(np.isfinite(contrast["ci_high"]) and contrast["ci_high"] < 1.0)
    if not r295:
        if premise:
            return "supported", "correction helps at 4.2 K (all baselines beaten) and not at 295 K; " \
                                "the 4.2 K refit floor exceeds the 295 K floor by the premise factor"
        return "supported, premise not met", \
            "correction helps at 4.2 K only, but the C0_refit level at 4.2 K is not >= 1.5x the 295 K " \
            "level: the contrast is not attributable to a larger model-form error at 4.2 K"
    if k:
        return "temperature-graded", "correction helps at both temperatures and more at 4.2 K " \
                                     "(contrast CI below 1); the prediction 'no gain at 295 K' fails"
    return "not temperature-specific", "correction helps at both temperatures by similar factors; " \
                                       "the temperature form of the hypothesis is refuted"


def run_decide(args) -> int:
    from cryoml.explore.i3_2_bias import paired_ratio
    res = {}
    for t in m.TEMPS_K:
        p = odir(args, f"lodo_{t}K") / "results.json"
        res[t] = json.loads(p.read_text()) if p.exists() else None
    if any(r is None for r in res.values()) or \
            (not args.smoke and not all(r.get("preregistered_config") for r in res.values())):
        print("not evaluated: missing or non-preregistered LODO result")
        return 1
    c4, c295 = conditions(res[PRIMARY_T]), conditions(res[CONTROL_T])
    p4, p295 = res[PRIMARY_T]["per_device"], res[CONTROL_T]["per_device"]
    tags = sorted(set(p4["M_bias"]) & set(p295["M_bias"]))
    ratio4 = {t: p4["M_bias"][t] / p4["C0_refit"][t] for t in tags}
    ratio295 = {t: p295["M_bias"][t] / p295["C0_refit"][t] for t in tags}
    contrast = paired_ratio(ratio4, ratio295)
    contrast["cluster_bootstrap"] = {k: v for k, v in paired_ratio(
        ratio4, ratio295, clusters={t: m.group_of(t) for t in tags}).items()
        if k in ("ci_low", "ci_high", "resample_unit")}
    premise = bool(c4["C0_refit_mean"] >= PREMISE_FACTOR * c295["C0_refit_mean"])
    label, text = outcome(c4, c295, contrast, premise)
    out = {"prereg": "out/research/explore/I6_1/prereg.json", "smoke": bool(args.smoke),
           "outcome": label, "interpretation": text,
           "at_4K": c4, "at_295K": c295,
           "contrast_GMR_4K_over_295K": contrast,
           "premise": {"C0_refit_mean_4K": c4["C0_refit_mean"], "C0_refit_mean_295K": c295["C0_refit_mean"],
                       "factor": PREMISE_FACTOR, "holds": premise},
           "means": {f"{t}K": {a: {kk: v[kk] for kk in ("all_device_mean", "combined", "median", "n_failed")}
                               for a, v in res[t]["summary"]["all"].items()} for t in m.TEMPS_K},
           "per_group": {f"{t}K": {g: {a: v["all_device_mean"] for a, v in s.items()}
                                   for g, s in res[t]["summary"].items() if g != "all"}
                         for t in m.TEMPS_K},
           "note": "DEV results are development evidence; TEST untouched"}
    target = OUT / ("smoke_decision.json" if args.smoke else "decision.json")
    m.write_json(target, out)
    print(json.dumps({k: out[k] for k in ("outcome", "interpretation", "premise")}, indent=1))
    for t, c in (("4K", c4), ("295K", c295)):
        print(t, c["checks"], {x: (round(v["ratio"], 3), round(v["ci_high"], 3)) for x, v in c["baselines"].items()})
    print("contrast", {k: contrast.get(k) for k in ("ratio", "ci_low", "ci_high", "wins", "n_used")})
    return 0


# ---------------------------------------------------------------------------
# figs
# ---------------------------------------------------------------------------
ARM_ORDER = ("start", "C0_refit", "M_bias", "T1_inloop", "B1", "B2", "B3_d3", "B5")


def run_figs(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cryoml.explore.i4_1_baselines import correction_from_json_any
    fig_dir = odir(args, "figs")
    fig_dir.mkdir(parents=True, exist_ok=True)
    res = {t: json.loads((odir(args, f"lodo_{t}K") / "results.json").read_text())
           for t in m.TEMPS_K if (odir(args, f"lodo_{t}K") / "results.json").exists()}
    for t, r in res.items():
        per = r["per_device"]
        tags = sorted(per["C0_refit"])
        arms = [a for a in ARM_ORDER if a in per]
        fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(tags) + 3), 4.5))
        for k, a in enumerate(arms):
            ax.plot(range(len(tags)), [per[a].get(x, np.nan) for x in tags],
                    "o-" if a == "M_bias" else "o", ms=4, lw=1.0, color=f"C{k}", label=a, alpha=0.85)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels([x.replace("t40_", "") for x in tags], rotation=75, ha="right", fontsize=6)
        ax.set_yscale("log")
        ax.set_ylabel("held-out strict paper RRMS")
        ax.set_title(f"I6-1 TU Delft 40 nm, {t} K, DEV leave-one-geometry-out" + (" [smoke]" if args.smoke else ""))
        ax.legend(fontsize=7, ncol=4)
        fig.tight_layout()
        fig.savefig(fig_dir / f"per_device_{t}K.png", dpi=110)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, y = [], 0
    for t, rt in res.items():
        for a in [x for x in ARM_ORDER if x not in ("start", "C0_refit")]:
            c = rt["comparisons"].get(f"{a}_vs_C0_refit")
            if not c:
                continue
            ax.errorbar(c["ratio"], y, xerr=[[c["ratio"] - c["ci_low"]], [c["ci_high"] - c["ratio"]]],
                        fmt="o", color=f"C{ARM_ORDER.index(a)}")
            labels.append(f"{t} K {a}")
            y += 1
    ax.axvline(1.0, color="k", lw=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("GMR(arm, C0_refit), device-bootstrap 95% CI")
    ax.set_title("I6-1 held-out RRMS ratio to the no-correction refit" + (" [smoke]" if args.smoke else ""))
    fig.tight_layout()
    fig.savefig(fig_dir / "gmr_vs_c0_by_temperature.png", dpi=110)
    plt.close(fig)
    if args.plot:
        devices = {d.tag: d for d in dev_devices(args.smoke)}
        for t, r in res.items():
            cards = json.loads((odir(args, f"lodo_{t}K") / "cards.json").read_text())
            corrs = json.loads((odir(args, f"lodo_{t}K") / "corrections.json").read_text())
            for tag in args.plot.split(","):
                if tag not in cards.get("C0_refit", {}):
                    continue
                dev = devices[tag]
                curves = m.fit_curves(dev, t)
                card = m.model_block(dev.dev_type)
                c0 = m.simulate(dev, card, curves, dev.temps_K[t], cards["C0_refit"][tag])
                mb_raw = m.simulate(dev, card, curves, dev.temps_K[t], cards["M_bias"][tag])
                rec = m.to_record(dev, t, mb_raw)
                mb = correction_from_json_any(corrs[f"M_bias|{dev.group}"]).apply(rec)
                per = r["per_device"]
                plot_device(dev, t, {f"C0_refit card ({per['C0_refit'][tag]:.4f})": c0,
                                     f"M_bias refit card x exp(g) ({per['M_bias'][tag]:.4f})": mb},
                            fig_dir / f"iv_{tag}_{t}K.png", f"{tag} {t} K" + (" [smoke]" if args.smoke else ""))
    print(f"figures in {fig_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("data", "split", "stock", "start", "screen", "lodo", "decide", "figs"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--plot", default="")
    ap.add_argument("--temps", default="", help="start / screen: comma list (default 4,295)")
    ap.add_argument("--temp", type=int, default=PRIMARY_T, help="lodo: temperature label (4 or 295)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    return {"data": run_data, "split": run_split, "stock": run_stock, "start": run_start,
            "screen": run_screen, "lodo": run_lodo, "decide": run_decide,
            "figs": run_figs}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
