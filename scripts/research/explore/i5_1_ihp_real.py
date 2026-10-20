#!/usr/bin/env python3
"""I5-1: the I3-2 bias-only correction on measured IHP SG13G2 transistors.

Pre-registration: ``out/research/explore/I5_1/prereg.json``.
Module: ``cryoml.explore.i5_1_ihp_real``.

Modes:
  data    parse the downloaded MDM files (read as text), write the dataset
          (``out/research/explore/I5_1/data/ihp_meas.npz``, not committed)
          and the manifest ``data/manifest.json`` (committed)
  split   DEV / TEST split of the 28 DUTs (seed in the prereg) -> split.json
  stock   stock-card RRMS of every DUT at every temperature (no fit), the
          card-path vs PDK-subcircuit check, and I-V check plots (DEV only)
  start   starting cards: polish from the stock card (DEV DUTs, --temps)
  screen  B1 edge screen on the 300 K starting cards (per class)
  lodo    leave-one-DUT-out at --temp: C0_refit, M_bias, T1_inloop, B1, B2,
          B3_d3, B5 (training within class)
  cross   X300: the fold's 300 K M_bias, held-out refit at 233/343/398 K
  decide  pre-registered rule at 300 K; secondary tables
  figs    per-DUT RRMS, GMR by temperature, I-V plots of --plot DUTs (DEV)

The fitting modes read only the DEV DUTs (``split.json``).  ``--smoke``
writes to ``smoke_*`` directories with reduced budgets (prereg compute.smoke).

Run with ``python -I`` (the script puts ``src`` on ``sys.path`` itself):

  .venv/bin/python -I scripts/research/explore/i5_1_ihp_real.py --mode data
  .venv/bin/python -I scripts/research/explore/i5_1_ihp_real.py --mode start --workers 6
  .venv/bin/python -I scripts/research/explore/i5_1_ihp_real.py --mode lodo --temp 300 --workers 6
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

from cryoml.explore import i5_1_ihp_real as m  # noqa: E402

OUT = ROOT / "out/research/explore/I5_1"
DATA = OUT / "data"
NPZ = DATA / "ihp_meas.npz"
MANIFEST = DATA / "manifest.json"
SPLIT = OUT / "split.json"
SPLIT_SEED = 20261020


PREREG_OPTIONS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
START_OPTIONS = {"nfev": 3000, "irls": 3, "diff_step": 0.01}
SMOKE = {"start_nfev": 100, "folds": 2, "seeds": (0,), "steps": 200}
ARMS = ("C0_refit", "M_bias", "T1_inloop", "B1", "B2", "B3_d3", "B5")
RULE_BASELINES = ("B1", "B2", "B3_d3")
T1_PLUS = 0.005
PRIMARY_T = 300


def load(tags=None):
    manifest = json.loads(MANIFEST.read_text())
    return m.load_dataset(NPZ, manifest, tags), manifest


def dev_devices() -> list:
    """The DEV DUTs only (TEST arrays are not read)."""
    split = json.loads(SPLIT.read_text())
    devices, _ = load(split["DEV"])
    if sorted(d.tag for d in devices) != sorted(split["DEV"]):
        raise RuntimeError("DEV DUTs missing from the dataset")
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
    DATA.mkdir(parents=True, exist_ok=True)
    saved = m.save_dataset(devices, NPZ)
    rows, fitted, comp, floors, quant = [], {}, [], {}, []
    for dev in devices:
        rows.append({"tag": dev.tag, "cls": dev.cls, "dev_type": dev.dev_type, "W_um": dev.W_um,
                     "L_um": dev.L_um, "dut": dev.dut, "geom": dev.geom,
                     "temps_C": {str(k): v for k, v in sorted(dev.temps_C.items())},
                     "vdd_nominal": dev.vdd})
        for t in sorted(dev.curves):
            fc = m.fit_curves(dev, t)
            rows_t = []
            for c in fc:
                inc, st = m.inclusion(c), m.curve_stats(c)
                rows_t.append(f"{c.key}|{c.fixed:.6g}|{c.sweep[0]:.6g}..{c.sweep[-1]:.6g}/"
                              f"{len(c.sweep)}|{inc['start']}|{inc['den']:.4g}|"
                              f"{int(inc['included'])}|{st['max_abs_id']:.4g}")
            fitted[f"{dev.tag}@{t}"] = rows_t
            floors[f"{dev.tag}@{t}"] = m.off_floor(dev, t)
            for c in dev.curves[t]:
                lim = getattr(c, "compliance", None)
                if lim:
                    comp.append(float(np.max(np.abs(c.id))) / lim["vd"])
                st = m.curve_stats(c)
                quant.append(st)
    setups = {}
    for dev in devices:
        for c in dev.curves[300]:
            s = setups.setdefault(f"{dev.cls}_{dev.dev_type}", {}).setdefault(c.setup, set())
            s.add((round(c.vb, 3), round(c.fixed, 3)) if c.setup == "dc_idvg" else round(c.vb, 3))
    setups = {k: {s: sorted(v) for s, v in d.items()} for k, d in setups.items()}
    inc = [r.split("|")[5] == "1" for v in fitted.values() for r in v]
    n_inc = {k: sum(r.split("|")[5] == "1" for r in v) for k, v in fitted.items()}
    zero_steps = [q["min_step_below_1e-10"] for q in quant if q["min_step_below_1e-10"]]
    manifest = {
        "id": "I5_1_data",
        "source": {"repository": "github.com/IHP-GmbH/IHP-Open-PDK",
                   "path": "ihp-sg13g2/libs.doc/meas/MOS", "commit": m.MEAS_COMMIT,
                   "license": "Apache-2.0", "format": "Keysight IC-CAP MDM text",
                   "local_copy": "data/external/ihp_sg13g2_open_pdk_meas (not committed)",
                   "reader": "cryoml.explore.i5_1_ihp_real.parse_mdm (text parsing only; run "
                             "with python -I)"},
        "npz": str(NPZ.relative_to(ROOT)) + " (not committed; regenerate with --mode data)",
        "n_devices": len(devices), "n_files": 560, "temperatures_K": list(m.TEMPS_K),
        "header_TEMP_C": [-40.0, 27.0, 70.0, 125.0],
        "terminals": "source grounded (vs = 0 in every block); id = current into the drain "
                     "terminal (nMOS > 0, pMOS < 0); inputs vg, vd, vb are node voltages = "
                     "Vgs, Vds, Vbs",
        "setups_at_300K": setups,
        "fitted_curves": {
            "rule": "the Vb = 0 curves only: dc_idvg at its three Vd values (keys idvg@d0..d2, "
                    "ascending |Vd|), dc_idvd at its five Vg values (idvd@g0..g4) and dc_idvd_vth "
                    "at its three Vg values (idvd@t0..t2): 11 per DUT and temperature. "
                    "The body-bias curves (dc_idvg at Vb != 0, dc_idvd_vbmin, dc_idvd_vth_vbmin) "
                    "are stored in the npz and not used",
            "note": "dc_idvd and dc_idvd_vth gate biases are set per DUT and temperature by the "
                    "measurement setup (LINVT_VALUE: from a threshold extraction), so their "
                    "values differ between DUTs; keys are by position"},
        "inclusion_rule": (
            "frozen from the measured current only (cryoml.pdk2.metric): clean_current (zero the "
            "points before the last exact zero), trim the leading points up to the first "
            "|I| > 1e-10 A, denominator = mean |I| of the trimmed curve; a curve is included iff "
            "the denominator > 3e-7 A (Id-Vd) or > 2e-7 A (Id-Vg). A DUT-temperature with no "
            "included curve is dropped (none is)"),
        "inclusion_summary": {"n_fitted_curves": len(inc), "n_included": int(sum(inc)),
                              "included_per_device_temperature_min_max":
                                  [min(n_inc.values()), max(n_inc.values())]},
        "quantization": {
            "printed_significant_digits_max": max(q["max_sig_digits"] for q in quant),
            "smallest_nonzero_abs_id_A": min(q["min_nonzero_abs_id"] for q in quant
                                             if q["min_nonzero_abs_id"] > 0),
            "smallest_step_between_distinct_values_below_1e-10_A": min(zero_steps),
            "n_exact_zero_id_values": int(sum(q["n_exact_zero"] for q in quant))},
        "compliance": {"drain_SMU1_A": 0.1, "gate_SMU2_A": 0.001, "bulk_SMU4_A": 0.1,
                       "source_SMU3_A": 0.1,
                       "max_abs_id_over_drain_compliance": float(max(comp))},
        "off_floor": {
            "definition": "median |id| at the three most-off gate biases of the Vb = 0, lowest-|Vd| "
                          "Id-Vg curve (Vg = -0.5..-0.4 V nMOS, +0.5..+0.4 V pMOS)",
            "min_A": float(min(floors.values())), "median_A": float(np.median(list(floors.values()))),
            "max_A": float(max(floors.values())), "per_device_temperature": floors},
        "devices": rows,
        "fitted_columns": "key|fixed_V|sweep_V first..last/n|trim_start|denominator_A|included|"
                          "max_abs_id_A",
        "fitted": fitted, **saved,
        "runtime_s": round(time.time() - t0, 1)}
    m.write_json(MANIFEST, manifest)
    print(json.dumps({k: manifest[k] for k in ("n_devices", "inclusion_summary", "quantization",
                                                "compliance")}, indent=1))
    print({k: manifest["off_floor"][k] for k in ("min_A", "median_A", "max_A")})
    return 0


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def run_split(args) -> int:
    devices, _ = load()
    split = m.make_split(devices, SPLIT_SEED)
    split["rule"] = m.make_split.__doc__.strip()
    split["DEV_by_class"] = {c: [t for t in split["DEV"] if t.startswith(c)] for c in ("lv", "hv")}
    split["TEST_by_class"] = {c: [t for t in split["TEST"] if t.startswith(c)] for c in ("lv", "hv")}
    split["written"] = "2026-10-20, before any fit; TEST DUTs are not loaded by the fitting modes"
    m.write_json(SPLIT, split)
    print(json.dumps(split, indent=1))
    return 0


# ---------------------------------------------------------------------------
# stock
# ---------------------------------------------------------------------------
def stock_task(task):
    dev, t = task
    curves = m.fit_curves(dev, t)
    card = m.stock_card(dev, m.BASE_SET)
    sims = m.simulate(dev, card, curves, dev.temps_C[t])
    ref = m.simulate_pdk_subckt(dev, curves, dev.temps_C[t])
    diff = max(float(np.nanmax(np.abs(a - b) / (np.abs(b) + 1e-12))) for a, b in zip(sims, ref))
    sc = m.score(curves, sims)
    return {"tag": dev.tag, "T_K": t, "rrms": sc["rrms"], "n_curves": sc["n_curves"],
            "per_curve": sc["per_curve"], "failed": sc["failed"],
            "max_rel_diff_card_vs_pdk_subckt": diff, "sims": [s.tolist() for s in sims]}


def plot_device(dev, t, sims, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = m.fit_curves(dev, t)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    groups = (("idvg@d", "Id-Vg (Vb = 0)", "Vg (V)", True), ("idvd@g", "Id-Vd, dc_idvd", "Vd (V)", False),
              ("idvd@t", "Id-Vd, dc_idvd_vth", "Vd (V)", False))
    for ax, (prefix, name, xl, logy) in zip(axes, groups):
        for k, (c, s) in enumerate(zip(curves, sims)):
            if not c.key.startswith(prefix):
                continue
            inc = m.inclusion(c)
            color = f"C{k % 10}"
            lab = f"{c.key} ({'Vd' if c.kind == 'idvg' else 'Vg'}={c.fixed:.3g})"
            y, ys = np.abs(c.id), np.abs(np.asarray(s))
            ax.plot(c.sweep, y, "o", ms=3, color=color, label=lab + ("" if inc["included"] else " excl."))
            ax.plot(c.sweep, ys, "-", color=color)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("|Id| (A)")
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.suptitle(title + "  (points: measured; lines: NGSpice-46 PSP103 stock card)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_stock(args) -> int:
    devices, _ = load()
    split = json.loads(SPLIT.read_text())
    tasks = [(d, t) for d in devices for t in m.TEMPS_K]
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(stock_task, tasks, chunksize=1)
    by = {(r["tag"], r["T_K"]): r for r in rows}
    table = {t: {d.tag: by[(d.tag, t)]["rrms"] for d in devices} for t in m.TEMPS_K}
    summary = {}
    for t in m.TEMPS_K:
        for part in ("DEV", "TEST", "ALL"):
            tags = [d.tag for d in devices] if part == "ALL" else split[part]
            for cls in ("lv", "hv", "all"):
                sel = {k: v for k, v in table[t].items() if k in tags and (cls == "all" or k.startswith(cls))}
                summary[f"{t}K|{part}|{cls}"] = m.aggregate(sel)
    out = {"what": "stock IHP PSP103 card (cornerMOS{lv,hv}.lib mos_tt), header temperature, measured "
                   "bias grid, no fit; strict paper RRMS on the 11 Vb = 0 curves with the frozen "
                   "measured inclusion",
           "simulator": "ngspice-46 + psp103.osdi (OpenVAF-Reloaded), top-level card path; checked "
                        "against the PDK subcircuit through the corner library",
           "max_rel_diff_card_vs_pdk_subckt": max(r["max_rel_diff_card_vs_pdk_subckt"] for r in rows),
           "per_device": {f"{t}": table[t] for t in m.TEMPS_K}, "summary": summary,
           "per_curve": {f"{r['tag']}@{r['T_K']}": r["per_curve"] for r in rows},
           "runtime_s": round(time.time() - t0, 1)}
    m.write_json(OUT / "stock" / "stock_rrms.json", out)
    fig_dir = OUT / "stock" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for tag in args.plot.split(","):
        dev = [d for d in devices if d.tag == tag][0]
        if tag not in split["DEV"]:
            raise RuntimeError(f"{tag} is not a DEV device; plots are DEV only")
        for t in (300, 233, 398):
            r = by[(tag, t)]
            plot_device(dev, t, r["sims"], fig_dir / f"stock_{tag}_{t}K.png",
                        f"{tag} {t} K, stock RRMS {r['rrms']:.3f}")
    print(f"{'device':22s} " + " ".join(f"{t:>8d}K" for t in m.TEMPS_K) + "  split")
    for d in devices:
        print(f"{d.tag:22s} " + " ".join(f"{table[t][d.tag]:9.4f}" for t in m.TEMPS_K)
              + ("  DEV" if d.tag in split["DEV"] else "  TEST"))
    for k, v in summary.items():
        if "|ALL|" in k or "|DEV|" in k:
            print(k, {kk: round(v[kk], 4) for kk in ("all_device_mean", "combined", "median")})
    print("max rel diff card vs subckt", out["max_rel_diff_card_vs_pdk_subckt"], "runtime", out["runtime_s"])
    return 0


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
def start_task(task):
    from cryoml.ngspice_fit import polish
    dev, t, opts = task
    t0 = time.time()
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
           "n_curves": sc["n_curves"], "n_sim": prob.n_sim, "history": res["history"],
           "n_at_bound": int(np.sum(np.abs(loc[box.active]) > 0.999)),
           "at_bound": [n for n, z, a in zip(box.names, loc, box.active) if a and abs(z) > 0.999],
           "params": params, "runtime_s": round(time.time() - t0, 1)}
    return rec, row


def run_start(args) -> int:
    from cryoml.explore.e1_correction import save_devices
    t_start = time.time()
    devices = dev_devices()
    temps = [int(x) for x in args.temps.split(",")] if args.temps else list(m.TEMPS_K)
    opts = dict(START_OPTIONS)
    if args.smoke:
        opts["nfev"] = SMOKE["start_nfev"]
        temps = temps if args.temps else [PRIMARY_T]
    tasks = [(d, t, opts) for t in temps for d in devices]
    tasks.sort(key=lambda x: -x[0].W_um)          # no effect on results; spreads load
    with get_context("spawn").Pool(args.workers) as pool:
        out = pool.map(start_task, tasks, chunksize=1)
    target = odir(args, "start")
    target.mkdir(parents=True, exist_ok=True)
    summary = {"options": opts, "box": m.base_box.__doc__.strip(), "smoke": bool(args.smoke),
               "per_temperature": {}}
    for t in temps:
        pairs = sorted([(r, row) for r, row in out if row["T_K"] == t], key=lambda x: x[0].tag)
        save_devices(records_path(args, t), [r for r, _ in pairs])
        rows = {row["tag"]: row for _, row in pairs}
        summary["per_temperature"][str(t)] = {
            "stock": m.aggregate({k: v["stock_rrms"] for k, v in rows.items()}),
            "start": m.aggregate({k: v["start_rrms"] for k, v in rows.items()}),
            "start_lv": m.aggregate({k: v["start_rrms"] for k, v in rows.items() if k.startswith("lv")}),
            "start_hv": m.aggregate({k: v["start_rrms"] for k, v in rows.items() if k.startswith("hv")}),
            "rows": rows}
        print(f"{t} K: stock {summary['per_temperature'][str(t)]['stock']['all_device_mean']:.4f} "
              f"-> start {summary['per_temperature'][str(t)]['start']['all_device_mean']:.4f}", flush=True)
        for k, v in rows.items():
            print(f"  {k:22s} {v['stock_rrms']:.4f} -> {v['start_rrms']:.4f}  n_sim {v['n_sim']} "
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
    box = m.base_box(dev)
    prob = m.MeasProblem(dev, rec.meta["T_K"], box)
    start = rec.meta["params"]
    ref = prob.simulate_params(start)
    out = {}
    for name, e in m.b1_extras(dev).items():
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
    devices = {d.tag: d for d in dev_devices()}
    recs = load_devices(records_path(args, PRIMARY_T))
    with get_context("spawn").Pool(args.workers) as pool:
        rows = dict(pool.map(screen_task, [(devices[r.tag], r) for r in recs], chunksize=1))
    out = {"rule": f"per class: a candidate is kept if the edge response at either box edge is >= "
                   f"{B1_THRESHOLD} on at least one {PRIMARY_T} K starting card of the class's DEV "
                   f"DUTs (non-finite responses count as 0)", "smoke": bool(args.smoke)}
    for cls in ("lv", "hv"):
        cls_rows = {k: v for k, v in rows.items() if k.startswith(cls)}
        names = sorted({n for r in cls_rows.values() for n in r})
        per = {}
        for n in names:
            resp = []
            for r in cls_rows.values():
                v = r.get(n, {})
                if v.get("active"):
                    lo, hi = v["response_lo"], v["response_hi"]
                    resp.append(max(lo if np.isfinite(lo) else 0.0, hi if np.isfinite(hi) else 0.0))
            per[n] = {"max_response": float(max(resp)) if resp else 0.0, "n_active_devices": len(resp),
                      "n_devices_ge_threshold": int(sum(x >= B1_THRESHOLD for x in resp))}
        sel = [n for n in names if per[n]["max_response"] >= B1_THRESHOLD]
        out[cls] = {"candidates": names, "selected": sel, "n_selected": len(sel),
                    "per_parameter": per, "per_device": cls_rows}
        print(f"{cls}: {len(names)} candidates, {len(sel)} selected: {sel}")
    out["runtime_s"] = round(time.time() - t0, 1)
    m.write_json(odir(args, "start") / "b1_screen.json", out)
    print(f"runtime {out['runtime_s']} s")
    return 0


# ---------------------------------------------------------------------------
# lodo / cross
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
    """One held-out refit (I4-1 refit_task on the measured problem)."""
    from cryoml.explore.e1_correction import Correction, score_device
    from cryoml.explore.i3_2_bias import fit_bias
    from cryoml.explore.i4_1_baselines import (AugProblem, b2_box, extend_box, matched_nfev,
                                               measured_pmax)
    from cryoml.ngspice_fit import polish
    label, kind, payload, rec, dev, opts = task
    t0 = time.time()
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
        extras = m.b1_extras(dev, payload["names"])
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
               params=params, runtime_s=round(time.time() - t0, 2))
    return row


def comparisons(scores: dict, tags: list) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    sub = {k: {t: v[t] for t in tags if t in v} for k, v in scores.items()}
    out = {}
    for x in sub:
        if x != "M_bias" and "M_bias" in sub:
            out[f"M_bias_vs_{x}"] = paired_ratio(sub["M_bias"], sub[x])
        if x != "C0_refit" and "C0_refit" in sub:
            out[f"{x}_vs_C0_refit"] = paired_ratio(sub[x], sub["C0_refit"])
    return out


def summaries(scores: dict, tags: list) -> dict:
    out = {}
    for part, sel in (("all", tags), ("lv", [t for t in tags if t.startswith("lv")]),
                      ("hv", [t for t in tags if t.startswith("hv")])):
        out[part] = {k: m.aggregate({t: v[t] for t in sel if t in v}) for k, v in scores.items()}
    return out


def run_lodo(args) -> int:
    from cryoml.explore.e1_correction import load_devices
    from cryoml.explore.i3_2_bias import merge_seed_results, split_seed_tasks
    t_start = time.time()
    t = int(args.temp)
    devices = {d.tag: d for d in dev_devices()}
    recs = load_devices(records_path(args, t))
    if sorted(r.tag for r in recs) != sorted(devices):
        raise RuntimeError("starting records differ from DEV")
    arms = [a for a in args.arms.split(",") if a]
    opts = dict(PREREG_OPTIONS)
    hyper = {} if not args.smoke else {"seeds": SMOKE["seeds"], "steps": SMOKE["steps"]}
    hyper_short = {} if not args.smoke else {"steps": SMOKE["steps"]}
    held = [r.tag for r in recs]
    if args.smoke:
        held = [next(r.tag for r in recs if r.meta["cls"] == c) for c in ("lv", "hv")][:SMOKE["folds"]]
    by_tag = {r.tag: r for r in recs}
    train = {h: [r for r in recs if r.meta["cls"] == by_tag[h].meta["cls"] and r.tag != h]
             for h in held}
    screen = json.loads((odir(args, "start") / "b1_screen.json").read_text())
    preregistered = (not args.smoke and set(ARMS) <= set(arms))
    print(f"I5_1 lodo {t} K: {len(held)} folds, arms {arms}, opts {opts}, hyper {hyper}, "
          f"preregistered={preregistered}", flush=True)
    timing = {}
    with get_context("spawn").Pool(args.workers) as pool:
        t0 = time.time()
        bias_tasks = [("M_bias", train[h]) for h in held] if "M_bias" in arms else []
        expanded, owner = split_seed_tasks(bias_tasks, hyper)
        tasks = [("bias", a, r, hp) for a, r, hp in expanded]
        other = []
        if "T1_inloop" in arms:
            other += [("t1", "T1", train[h], hyper_short) for h in held]
        if "B3_d3" in arms:
            other += [("poly", "B3_d3", train[h], hyper_short) for h in held]
        results = pool.map(train_task, tasks + other, chunksize=1)
        corrs = {}
        merged = merge_seed_results(len(bias_tasks), owner, results[:len(tasks)])
        for h, c in zip(held, merged):
            corrs[f"M_bias|{h}"] = c
        rest = iter(results[len(tasks):])
        if "T1_inloop" in arms:
            for h in held:
                corrs[f"T1_inloop|{h}"] = next(rest)
        if "B3_d3" in arms:
            for h in held:
                corrs[f"B3_d3|{h}"] = next(rest)
        timing["train_s"] = round(time.time() - t0, 1)
        print(f"training done ({timing['train_s']} s, {len(tasks + other)} tasks)", flush=True)
        from cryoml.explore.e1_correction import Correction
        refits = []
        for h in held:
            rec, dev = by_tag[h], devices[h]
            for arm in arms:
                if arm == "C0_refit":
                    refits.append((arm, "corr", {"corr": Correction("C0")}, rec, dev, opts))
                elif arm in ("M_bias", "T1_inloop", "B3_d3"):
                    refits.append((arm, "corr", {"corr": corrs[f"{arm}|{h}"]}, rec, dev, opts))
                elif arm == "B1":
                    refits.append((arm, "b1", {"names": screen[dev.cls]["selected"]}, rec, dev, opts))
                elif arm == "B2":
                    refits.append((arm, "b2", {}, rec, dev, opts))
                elif arm == "B5":
                    refits.append((arm, "b5", {"hyper": hyper}, rec, dev, opts))
        cost = {"B1": 0, "B2": 1, "B5": 2}
        order = sorted(range(len(refits)), key=lambda i: cost.get(refits[i][0], 3))
        t0 = time.time()
        rows_l = pool.map(refit_task, [refits[i] for i in order], chunksize=1)
        timing["refit_s"] = round(time.time() - t0, 1)
        timing["refit_n"] = len(refits)
    rows = {}
    for row in rows_l:
        rows.setdefault(row["label"], {})[row["tag"]] = row
    scores = {a: {tg: r["rrms"] for tg, r in v.items()} for a, v in rows.items()}
    from cryoml.explore.e1_correction import score_device
    scores["start"] = {h: float(score_device(by_tag[h], [c.sim for c in by_tag[h].curves])["rrms"])
                       for h in held}
    result = {"name": odir(args, f"lodo_{t}K").name, "T_K": t, "prereg": "out/research/explore/I5_1/prereg.json",
              "preregistered_config": preregistered, "smoke": bool(args.smoke), "options": opts,
              "hyper_overrides": hyper, "arms": arms, "held_out": held,
              "b1_names": {c: screen[c]["selected"] for c in ("lv", "hv")},
              "summary": summaries(scores, held),
              "comparisons": {part: comparisons(scores, sel) for part, sel in (
                  ("all", held), ("lv", [h for h in held if h.startswith("lv")]),
                  ("hv", [h for h in held if h.startswith("hv")]))},
              "per_device": scores,
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
    for k, v in result["comparisons"]["all"].items():
        print(f"  {k:28s} GMR {v['ratio']:.3f} [{v.get('ci_low', float('nan')):.3f}, "
              f"{v.get('ci_high', float('nan')):.3f}] wins {v['wins']}/{v['n_used']}")
    print(f"  timing {result['timing']} runtime {result['runtime_s']} s")


def run_cross(args) -> int:
    from cryoml.explore.e1_correction import load_devices
    from cryoml.explore.i4_1_baselines import correction_from_json_any
    t_start = time.time()
    devices = {d.tag: d for d in dev_devices()}
    stored = json.loads((odir(args, f"lodo_{PRIMARY_T}K") / "corrections.json").read_text())
    held = sorted(k.split("|", 1)[1] for k in stored if k.startswith("M_bias|"))
    opts = dict(PREREG_OPTIONS)
    temps = [x for x in m.TEMPS_K if x != PRIMARY_T
             and records_path(args, x).with_suffix(".json").exists()]
    tasks = []
    for t in temps:
        recs = {r.tag: r for r in load_devices(records_path(args, t))}
        for h in held:
            corr = correction_from_json_any(stored[f"M_bias|{h}"])
            tasks.append(("X300", "corr", {"corr": corr}, recs[h], devices[h], opts))
    with get_context("spawn").Pool(args.workers) as pool:
        rows_l = pool.map(refit_task, tasks, chunksize=1)
    out = {"prereg": "out/research/explore/I5_1/prereg.json", "smoke": bool(args.smoke),
           "what": "fold M_bias trained on the 300 K training records, held-out refit at T with it in "
                   "the loop", "per_temperature": {}}
    for t in temps:
        rows = {r["tag"]: r for r in rows_l if r["T_K"] == t}
        scores = {"X300": {k: v["rrms"] for k, v in rows.items()}}
        lodo = odir(args, f"lodo_{t}K") / "results.json"
        if lodo.exists():
            per = json.loads(lodo.read_text())["per_device"]
            for a in ("C0_refit", "M_bias"):
                if a in per:
                    scores[a] = per[a]
        tags = sorted(rows)
        cmp_ = {}
        from cryoml.explore.i3_2_bias import paired_ratio
        for a in ("C0_refit", "M_bias"):
            if a in scores:
                cmp_[f"X300_vs_{a}"] = paired_ratio({k: scores["X300"][k] for k in tags},
                                                    {k: scores[a][k] for k in tags if k in scores[a]})
        out["per_temperature"][str(t)] = {
            "summary": summaries(scores, tags), "comparisons": cmp_, "per_device": scores,
            "rows": {k: {kk: vv for kk, vv in v.items() if kk != "params"} for k, v in rows.items()},
            "cards": {k: v["params"] for k, v in rows.items()}}
        print(f"{t} K: X300 mean {out['per_temperature'][str(t)]['summary']['all']['X300']['all_device_mean']:.4f}",
              {k: round(v["ratio"], 3) for k, v in cmp_.items()})
    out["runtime_s"] = round(time.time() - t_start, 1)
    m.write_json(odir(args, "cross") / "results.json", out)
    print(f"runtime {out['runtime_s']} s")
    return 0


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def rule(result: dict) -> dict:
    s = result["summary"]["all"]
    c = result["comparisons"]["all"]
    r1 = c["M_bias_vs_C0_refit"]
    checks = {"R1_ci_high_lt_1": bool(np.isfinite(r1["ci_high"]) and r1["ci_high"] < 1.0),
              "R2_mean_le_T1_inloop_plus_0.005": bool(
                  s["M_bias"]["all_device_mean"] <= s["T1_inloop"]["all_device_mean"] + T1_PLUS),
              "R3_zero_failed": bool(s["M_bias"]["n_failed"] == 0)}
    replicated = all(checks.values())
    per, matched = {}, []
    for x in RULE_BASELINES:
        cx = c.get(f"M_bias_vs_{x}", {})
        beaten = bool(np.isfinite(cx.get("ci_high", np.nan)) and cx["ci_high"] < 1.0)
        per[x] = {"ratio": cx.get("ratio"), "ci_low": cx.get("ci_low"), "ci_high": cx.get("ci_high"),
                  "wins": cx.get("wins"), "n_used": cx.get("n_used"), "beaten": beaten}
        if not beaten:
            matched.append(x)
    return {"checks": checks,
            "replication": "replicated" if replicated else
            "not replicated (" + ", ".join(k for k, v in checks.items() if not v) + ")",
            "R1_detail": r1, "R2_detail": {"M_bias_mean": s["M_bias"]["all_device_mean"],
                                           "T1_inloop_mean": s["T1_inloop"]["all_device_mean"]},
            "baselines": per,
            "baseline_verdict": ("supported against baselines" if replicated and not matched else
                                 ("matched by baseline(s) " + ", ".join(matched) if matched
                                  else "baselines beaten, replication failed"))}


def run_decide(args) -> int:
    res = {}
    for t in m.TEMPS_K:
        p = odir(args, f"lodo_{t}K") / "results.json"
        res[t] = json.loads(p.read_text()) if p.exists() else None
    primary = res[PRIMARY_T]
    if primary is None or (not args.smoke and not primary.get("preregistered_config")):
        print("not evaluated: missing or non-preregistered 300 K result")
        return 1
    out = {"prereg": "out/research/explore/I5_1/prereg.json", "smoke": bool(args.smoke),
           "primary_T_K": PRIMARY_T, "primary": rule(primary), "secondary": {}}
    for t in m.TEMPS_K:
        if res[t] is None or t == PRIMARY_T:
            continue
        try:
            out["secondary"][f"{t}K"] = rule(res[t])
        except KeyError:
            pass
    out["per_class"] = {f"{t}K": {cls: {k: {kk: v.get(kk) for kk in ("ratio", "ci_low", "ci_high", "wins", "n_used")}
                                        for k, v in res[t]["comparisons"][cls].items()}
                                  for cls in ("lv", "hv")} for t in m.TEMPS_K if res[t]}
    out["means"] = {f"{t}K": {a: {kk: v[kk] for kk in ("all_device_mean", "combined", "median", "n_failed")}
                              for a, v in res[t]["summary"]["all"].items()} for t in m.TEMPS_K if res[t]}
    cross = odir(args, "cross") / "results.json"
    if cross.exists():
        cr = json.loads(cross.read_text())
        out["cross_temperature"] = {t: {"X300_mean": v["summary"]["all"]["X300"]["all_device_mean"],
                                        "comparisons": {k: {kk: c.get(kk) for kk in ("ratio", "ci_low", "ci_high", "wins", "n_used")}
                                                        for k, c in v["comparisons"].items()}}
                                    for t, v in cr["per_temperature"].items()}
    out["note"] = "DEV results are development evidence (prereg data_reuse_note); TEST untouched"
    target = OUT / ("smoke_decision.json" if args.smoke else "decision.json")
    m.write_json(target, out)
    print(json.dumps({"primary": {k: out["primary"][k] for k in ("checks", "replication", "baseline_verdict")},
                      "baselines": out["primary"]["baselines"]}, indent=1))
    return 0


# ---------------------------------------------------------------------------
# figs
# ---------------------------------------------------------------------------
ARM_ORDER = ("start", "C0_refit", "M_bias", "T1_inloop", "B1", "B2", "B3_d3", "B5")


def run_figs(args) -> int:
    """Per-DUT RRMS by arm (300 K), GMR vs C0_refit by temperature, and I-V
    plots (measured; C0_refit card; M_bias refit card x exp(g)) for --plot DUTs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cryoml.explore.i4_1_baselines import correction_from_json_any
    fig_dir = odir(args, "figs")
    fig_dir.mkdir(parents=True, exist_ok=True)
    res = {t: json.loads((odir(args, f"lodo_{t}K") / "results.json").read_text())
           for t in m.TEMPS_K if (odir(args, f"lodo_{t}K") / "results.json").exists()}
    r = res[PRIMARY_T]
    per = r["per_device"]
    tags = sorted(per["C0_refit"])
    arms = [a for a in ARM_ORDER if a in per]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for k, a in enumerate(arms):
        ax.plot(range(len(tags)), [per[a].get(t, np.nan) for t in tags], "o-" if a == "M_bias" else "o",
                ms=5, lw=1.2, color=f"C{k}", label=a, alpha=0.9)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("held-out strict paper RRMS")
    ax.set_title(f"I5-1 IHP SG13G2 measured, {PRIMARY_T} K, DEV leave-one-DUT-out (within class)"
                 + (" [smoke]" if args.smoke else ""))
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(fig_dir / f"per_device_{PRIMARY_T}K.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = []
    y = 0
    for t, rt in res.items():
        for a in [x for x in ARM_ORDER if x not in ("start", "C0_refit")]:
            c = rt["comparisons"]["all"].get(f"{a}_vs_C0_refit")
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
    ax.set_xlabel("GMR(arm, C0_refit), bootstrap 95% CI over DEV DUTs")
    ax.set_title("I5-1 held-out RRMS ratio to the no-correction refit" + (" [smoke]" if args.smoke else ""))
    fig.tight_layout()
    fig.savefig(fig_dir / "gmr_vs_c0_by_temperature.png", dpi=110)
    plt.close(fig)

    if args.plot:
        devices = {d.tag: d for d in dev_devices()}
        cards = json.loads((odir(args, f"lodo_{PRIMARY_T}K") / "cards.json").read_text())
        corrs = json.loads((odir(args, f"lodo_{PRIMARY_T}K") / "corrections.json").read_text())
        for tag in args.plot.split(","):
            dev = devices[tag]
            curves = m.fit_curves(dev, PRIMARY_T)
            card = m.stock_card(dev, m.BASE_SET)
            c0 = m.simulate(dev, card, curves, dev.temps_C[PRIMARY_T], cards["C0_refit"][tag])
            mb_raw = m.simulate(dev, card, curves, dev.temps_C[PRIMARY_T], cards["M_bias"][tag])
            rec = m.to_record(dev, PRIMARY_T, mb_raw)
            mb = correction_from_json_any(corrs[f"M_bias|{tag}"]).apply(rec)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
            for ax, prefix in zip(axes, ("idvg@d", "idvd@g", "idvd@t")):
                for k, c in enumerate(curves):
                    if not c.key.startswith(prefix):
                        continue
                    col = f"C{k % 10}"
                    inc = m.inclusion(c)["included"]
                    ax.plot(c.sweep, np.abs(c.id), "o", ms=3, color=col,
                            label=f"{c.key}{'' if inc else ' excl.'} measured")
                    ax.plot(c.sweep, np.abs(c0[k]), "--", color=col, lw=1)
                    ax.plot(c.sweep, np.abs(mb[k]), "-", color=col, lw=1.2)
                if prefix == "idvg@d":
                    ax.set_yscale("log")
                ax.set_xlabel("Vg (V)" if prefix == "idvg@d" else "Vd (V)")
                ax.set_ylabel("|Id| (A)")
                ax.legend(fontsize=6)
            fig.suptitle(f"{tag} {PRIMARY_T} K: points measured; dashed C0_refit card "
                         f"({per['C0_refit'][tag]:.4f}); solid M_bias refit card x exp(g) "
                         f"({per['M_bias'][tag]:.4f})" + (" [smoke]" if args.smoke else ""), fontsize=10)
            fig.tight_layout()
            fig.savefig(fig_dir / f"iv_{tag}_{PRIMARY_T}K.png", dpi=110)
            plt.close(fig)
    print(f"figures in {fig_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("data", "split", "stock", "start", "screen", "lodo", "cross", "decide",
                             "figs"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--plot", default="")
    ap.add_argument("--temps", default="", help="start: comma list (default all four)")
    ap.add_argument("--temp", type=int, default=PRIMARY_T, help="lodo: temperature (K)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    return {"data": run_data, "split": run_split, "stock": run_stock, "start": run_start,
            "screen": run_screen, "lodo": run_lodo, "cross": run_cross,
            "decide": run_decide, "figs": run_figs}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
