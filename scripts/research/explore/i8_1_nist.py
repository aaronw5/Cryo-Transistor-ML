#!/usr/bin/env python3
"""I8-1: NIST mds2-2997 sky130 4 K data (MPW-5 test tile), ingest and smoke.

Module: ``cryoml.explore.i8_1_nist``.  No model fitting on these data here.

Modes:
  sources    sha256 of the two downloaded ZIPs vs the NIST metadata values
             -> out/research/explore/I8_1/data/sources.json
  inventory  parse every MPW-5 transistor file and every ETest file, map W/L,
             checks -> inventory.json, inventory.md
  data       chosen families (nfet_01v8_lvt, pfet_01v8_lvt), Vb = 0 curves
             -> data/nist4k.npz (not committed), data/manifest.json
             (measured inclusion, 77 K card bin coverage per geometry)
  split      DEV / TEST split (geometry and bin coverage only) -> split.json
  smoke      stock 77 K paper card on one DEV device per family (deck at
             77 K and at 4.2 K) and the forced nearest bin on one uncovered
             DEV device per family; shape check and timing only, no RRMS
  start      (prereg) bins, stock readback, starting cards of the DEV devices
  stats      (prereg) S1_lodo, S1_pooled, S2, S3, per-family values, gates
             from the starting-card records only -> prediction.json
  commit_prediction  commit prediction.json alone and push (fails if missing)
  lodo       (prereg) C0, C0_refit, M_bias, T1_inloop, leave-one-device-out;
             refuses to start unless prediction.json is committed
  baselines  (prereg) B1 screen, B1, B2, B3_d3 (same guard)
  decide     (prereg) P_a, P_b, P_c, secondary outcomes, validity

--smoke: 4 DEV devices (per family one bin-covered and one forced-bin device),
reduced settings (SMOKE), outputs under smoke_* (not used for any decision).

Run with ``python -I`` from the repository root:
  .venv/bin/python -I scripts/research/explore/i8_1_nist.py --mode inventory
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml import spice_pdk as sp  # noqa: E402
from cryoml.explore import i8_1_nist as m  # noqa: E402

OUT = m.OUT
DATA = OUT / "data"
NPZ = DATA / "nist4k.npz"
MANIFEST = DATA / "manifest.json"
SPLIT = OUT / "split.json"
SPLIT_SEED = 20261021
# sha256 from the NIST metadata record (data.nist.gov/rmm/records, components)
NIST_SHA256 = {
    "MPW-5 Test Tile.zip": "3a2cd00d40a059ade77b36afbeb45bc819e77ca80858668226809e3e932a03e3",
    "ETest Tile.zip": "b1c8374056cf9288d5e2ea2b63f50d22a3956fbc46f9dee3c89088b75e7c1383",
}
NIST_SIZE = {"MPW-5 Test Tile.zip": 39471626, "ETest Tile.zip": 3807177}
EXCLUDE_BAD = True


# ---------------------------------------------------------------------------
def run_sources(args) -> int:
    rows = {}
    for name, d in m.DOWNLOADS.items():
        p = m.RAW / d["local"]
        rows[name] = {"url": d["url"], "local": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                      "sha256": m.sha256(p), "nist_sha256": NIST_SHA256[name],
                      "nist_bytes": NIST_SIZE[name]}
        rows[name]["match"] = (rows[name]["sha256"] == NIST_SHA256[name]
                               and rows[name]["bytes"] == NIST_SIZE[name])
    out = {"doi": m.DOI, "record_version": "1.1.2 (rmm metadata, 2026-10-20)",
           "downloaded": "2026-10-20", "license": "NIST public data (https://www.nist.gov/open/license)",
           "files": rows,
           "extraction": "zipfile, every member path checked to stay inside a new empty "
                         "directory, no symlinks, testzip() None; ext_mpw5/ and ext_etest/ under "
                         "out/research/data_raw/nist_mds2_2997/ (gitignored)"}
    m.write_json(DATA / "sources.json", out)
    print(json.dumps(rows, indent=1))
    return 0 if all(r["match"] for r in rows.values()) else 1


# ---------------------------------------------------------------------------
def _fmt(v):
    return ", ".join(f"{x:g}" for x in v)


def run_inventory(args) -> int:
    t0 = time.time()
    mpw = m.discover_mpw5()
    et = m.discover_etest()
    flags = m.read_flags()
    found = {(d.folder, d.module, d.position) for d in mpw}
    inv = {"id": "I8_1_inventory", "doi": m.DOI,
           "mpw5": m.inventory(mpw, "MPW-5"), "etest": m.inventory(et, "ETest"),
           "readme_flags": {f"{k[0]}/{k[1]}_{k[2]}": {"text": v, "files_present": k in found}
                            for k, v in flags.items()},
           "checks": {"gate_control_min": m.GATE_CONTROL_MIN,
                      "compliance_frac": m.COMPLIANCE_FRAC,
                      "note": "bad = nonfinite value, an all-zero curve, |Id| >= 0.98 x drain SMU "
                              "compliance, or |Id(Vg max)| / |Id(Vg 0)| at Vd max (Id-Vd, Vb 0) "
                              "< 100 (no gate control); identical_current_data lists devices "
                              "whose Id arrays are byte-identical"},
           "seconds": round(time.time() - t0, 1)}
    m.write_json(OUT / "inventory.json", inv)
    lines = ["# I8-1 inventory: NIST mds2-2997 (4 K chuck temperature)", "",
             "| tile | device type | folder | n devices | n W/L | W (um) | L (um) | Id-Vg: Vd (V) | "
             "Id-Vg: Vb (V) | Id-Vg points | Id-Vd: Vg (V) | Id-Vd: Vb (V) | Id-Vd points | "
             "readme flags | bad |", "|" + "---|" * 15]
    for tile in ("mpw5", "etest"):
        for t, r in inv[tile]["types"].items():
            s = r["sweeps"]
            lines.append(
                f"| {r['tile']} | {t} | {', '.join(r['folder'])} | {r['n_devices']} | {r['n_wl']} | "
                f"{r['W_um'][0]:g}-{r['W_um'][1]:g} | {r['L_um'][0]:g}-{r['L_um'][1]:g} | "
                f"{_fmt(s['idvg']['fixed_V'])} | {_fmt(s['idvg']['vb_V'])} | "
                f"{_fmt(s['idvg']['n_points'])} | {_fmt(s['idvd']['fixed_V'])} | "
                f"{_fmt(s['idvd']['vb_V'])} | {_fmt(s['idvd']['n_points'])} | "
                f"{'; '.join(f'{k}: {v}' for k, v in r['flagged_readme'].items()) or '-'} | "
                f"{'; '.join(f'{k}: {v}' for k, v in r['bad'].items()) or '-'} |")
    lines += ["", "Readme flags: " + "; ".join(f"{k} '{v['text']}' (files present: "
                                               f"{v['files_present']})"
                                               for k, v in inv["readme_flags"].items()),
              "", "Identical current data: MPW-5 "
              f"{inv['mpw5']['identical_current_data'] or 'none'}; ETest "
              f"{inv['etest']['identical_current_data'] or 'none'}", ""]
    (OUT / "inventory.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------------------
def bin_candidates(polarity: str, W_um: float, L_um: float) -> list[int]:
    sp.ensure_pdk77k()
    text = (sp.PDK77K_DIR / sp._CORNER_BASENAME[polarity]).read_text(errors="replace")
    out, e = [], 1e-12
    for idx, a, b in sp._bin_spans(text, polarity):
        g = sp._bin_geometry(text[a:b])
        if (g["lmin"] - e <= L_um * 1e-6 <= g["lmax"] + e
                and g["wmin"] - e <= W_um * 1e-6 <= g["wmax"] + e):
            out.append(idx)
    return out


def nearest_bin(polarity: str, W_um: float, L_um: float) -> tuple[int, float]:
    """Bin whose (log L, log W) box is nearest (0 inside)."""
    text = (sp.PDK77K_DIR / sp._CORNER_BASENAME[polarity]).read_text(errors="replace")
    best = None
    for idx, a, b in sp._bin_spans(text, polarity):
        g = {k: v * 1e6 for k, v in sp._bin_geometry(text[a:b]).items()}
        dl = max(np.log(g["lmin"] / L_um), 0.0, np.log(L_um / g["lmax"]))
        dw = max(np.log(g["wmin"] / W_um), 0.0, np.log(W_um / g["wmax"]))
        d = float(np.hypot(dl, dw))
        if best is None or d < best[1]:
            best = (idx, d)
    return best


def run_data(args) -> int:
    devices = m.discover_mpw5(folders=m.CHOSEN)
    saved = m.save_dataset(devices, NPZ)
    rows, inc_counts = [], {}
    for d in devices:
        ck = m.device_checks(d)
        cands = bin_candidates(d.polarity, d.W_um, d.L_um)
        nb = nearest_bin(d.polarity, d.W_um, d.L_um)
        curves = m.fit_curves(d)
        inc = []
        for c in curves:
            sign = -1.0 if d.polarity == "pmos" else 1.0
            r = m.inclusion_measured(c.kind, sign * c.id)
            inc.append(f"{c.key}|{c.fixed:g}|{len(c.sweep)}|{r['start']}|{r['den']:.4g}|"
                       f"{int(r['included'])}")
        n_inc = sum(int(s.rsplit("|", 1)[1]) for s in inc)
        inc_counts.setdefault(d.dev_type, []).append(n_inc)
        rows.append({**m.device_row(d), "bad": ck["bad"], "gate_control_ratio": ck["gate_control_ratio"],
                     "bin77k_candidates": cands, "bin77k_covered": bool(cands),
                     "bin77k_nearest": nb[0], "bin77k_log_distance": round(nb[1], 4),
                     "curves_key|fixed_V|n|trim_start|den_A|included": inc,
                     "n_included_measured_rule": n_inc})
    manifest = {
        "id": "I8_1_data", "doi": m.DOI, "tile": "MPW-5 Test Tile, transistormodule",
        "families": {f: m.FOLDERS[f][0] for f in m.CHOSEN},
        "curves": "Vb = 0 curves per device: Id-Vg at Vd 0.1 and 1.8 V (pMOS -0.1, -1.8; keys "
                  "idvg@d0, d1; Vg 0..+-1.8 V, 10 mV, 181 points) and Id-Vd at Vg 0, 0.3, ..., "
                  "1.8 V (keys idvd@g0..g6; Vd 0..+-1.8 V, 20 mV, 91 points); npz holds the raw "
                  "stored sign; to_pipeline_curves gives the 77 K convention (pMOS Id as a "
                  "positive magnitude, voltages negative)",
        "inclusion_measured_rule": "pdk2.metric: clean_current, trim to the first |I| > 1e-10 A, "
                                   "include iff trimmed mean |I| > 3e-7 A (Id-Vd) / 2e-7 A (Id-Vg)",
        "included_per_family": {k: {"n_devices": len(v), "total": int(sum(v)),
                                    "min": int(min(v)), "max": int(max(v))}
                                for k, v in inc_counts.items()},
        "bin77k": {"cards": {p: str((sp.PDK77K_DIR / sp._CORNER_BASENAME[p]).relative_to(ROOT))
                             for p in ("nmos", "pmos")},
                   "covered_per_family": {f: sum(1 for r in rows if r["dev_type"] == f
                                                 and r["bin77k_covered"])
                                          for f in inc_counts}},
        "devices": rows, **saved}
    m.write_json(MANIFEST, manifest)
    print(json.dumps({k: manifest[k] for k in ("included_per_family", "bin77k")}, indent=1))
    return 0


# ---------------------------------------------------------------------------
def run_split(args) -> int:
    manifest = json.loads(MANIFEST.read_text())
    devices = m.load_dataset(NPZ, manifest)
    info = {r["tag"]: r for r in manifest["devices"]}
    excluded = {t: r["bad"] + [f for f in r["flags"] if f.startswith("readme")]
                for t, r in info.items()
                if EXCLUDE_BAD and (r["bad"] or any(f.startswith("readme") for f in r["flags"]))}
    keep = [d for d in devices if d.tag not in excluded]
    strata = {}
    for d in keep:
        cov = "covered" if info[d.tag]["bin77k_covered"] else "uncovered"
        d_copy = m.NistDevice(d.tag, d.folder, f"{d.dev_type}|{cov}", d.polarity, d.module,
                              d.position, d.W_um, d.L_um)
        strata.setdefault(d.dev_type, []).append(d_copy)
    split = m.make_split([x for v in strata.values() for x in v], SPLIT_SEED)
    counts = {}
    for part in ("DEV", "TEST"):
        for t in split[part]:
            key = f"{info[t]['dev_type']}|{'covered' if info[t]['bin77k_covered'] else 'uncovered'}"
            counts.setdefault(key, {"DEV": 0, "TEST": 0})[part] += 1
    out = {"id": "I8_1_split", "seed": SPLIT_SEED,
           "rule": "exclude devices with a readme flag or a bad check; strata = family x 77 K "
                   "card bin coverage (covered: the geometry lies in a bin of the 77 K card); "
                   "per stratum devices sorted by (L, W), consecutive blocks of 5, seeded "
                   "permutation puts 3 of 5 in DEV (short last block: round(0.6 n)); "
                   "numpy.random.default_rng(seed), strata in sorted order "
                   "(cryoml.explore.i8_1_nist.make_split). Uses geometry, flags and bin "
                   "coverage only; no current value enters the assignment except through the "
                   "inventory bad checks",
           "excluded": excluded, "DEV": split["DEV"], "TEST": split["TEST"],
           "n_dev": split["n_dev"], "n_test": split["n_test"], "counts": counts,
           "blocks": split["blocks"],
           "TEST_policy": "TEST arrays are not loaded by any later I8-1 mode before the "
                          "December TEST run (dev_devices() loads DEV tags only)"}
    m.write_json(SPLIT, out)
    print(json.dumps({k: out[k] for k in ("excluded", "n_dev", "n_test", "counts")}, indent=1))
    return 0


def dev_devices() -> list:
    split = json.loads(SPLIT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    devices = m.load_dataset(NPZ, manifest, split["DEV"])
    if sorted(d.tag for d in devices) != sorted(split["DEV"]):
        raise RuntimeError("DEV devices missing from the dataset")
    return devices, {r["tag"]: r for r in manifest["devices"]}


# ---------------------------------------------------------------------------
def simulate_forced(polarity: str, L_um: float, W_um: float, curves, bin_index: int,
                    temp_K: float):
    """simulate_pdk through a library whose subckt instance names bin
    ``bin_index`` explicitly (spice_pdk._force_bin)."""
    sp.ensure_pdk77k()
    with tempfile.TemporaryDirectory(prefix="i8_1_force_") as td_s:
        td = Path(td_s)
        base = sp.PDK77K_DIR / sp._CORNER_BASENAME[polarity]
        corner = td / sp._CORNER_BASENAME[polarity]
        corner.write_text(sp._force_bin(base.read_text(errors="replace"), polarity, bin_index))
        lib = td / "lib77k.spice"
        lib.write_text(sp.LIB_FILE.read_text().replace(base.as_posix(), corner.as_posix()))
        return sp.simulate_pdk(polarity, L_um, W_um, curves, temp_K=temp_K, library_path=lib)


def _shape_row(curves, sims) -> dict:
    return {"n_curves": len(curves),
            "shapes_equal": all(np.shape(s) == np.shape(c.Id) for c, s in zip(curves, sims)),
            "n_nonfinite_curves": int(sum(not np.all(np.isfinite(s)) for s in sims)),
            "sign_agrees_at_max_bias": [bool(np.sign(s[-1]) == np.sign(c.Id[-1]))
                                        for c, s in zip(curves, sims) if abs(c.Id[-1]) > 1e-9],
            "sim_last_A": [float(s[-1]) for s in sims],
            "meas_last_A": [float(c.Id[-1]) for c in curves]}


def run_smoke(args) -> int:
    devices, info = dev_devices()
    out = {"id": "I8_1_smoke", "simulator": sp._NGSPICE_BIN, "note": "stock 77 K paper card, "
           "no fit, no RRMS; shape, finiteness, sign and timing only", "runs": []}
    picks = []
    for pol in ("nmos", "pmos"):
        cov = [d for d in devices if d.polarity == pol and info[d.tag]["bin77k_covered"]]
        unc = [d for d in devices if d.polarity == pol and not info[d.tag]["bin77k_covered"]]
        # covered: the DEV device with the median L among covered DEV devices
        cov.sort(key=lambda d: (d.L_um, d.W_um))
        unc.sort(key=lambda d: (d.L_um, d.W_um))
        picks.append(("native_bin", cov[len(cov) // 2]))
        if unc:
            picks.append(("forced_nearest_bin", unc[len(unc) // 2]))
    for how, d in picks:
        curves = m.to_pipeline_curves(d)
        for temp_K in (77.0, 4.2):
            t0 = time.time()
            if how == "native_bin":
                sims = sp.simulate_pdk(d.polarity, d.L_um, d.W_um, curves, temp_K=temp_K)
                bin_used = sp.find_bin_index(d.polarity, d.L_um, d.W_um)
            else:
                bin_used = info[d.tag]["bin77k_nearest"]
                sims = simulate_forced(d.polarity, d.L_um, d.W_um, curves, bin_used, temp_K)
            dt = time.time() - t0
            row = {"device": d.tag, "how": how, "bin": bin_used, "deck_temp_K": temp_K,
                   "seconds": round(dt, 3), **_shape_row(curves, sims)}
            out["runs"].append(row)
            print(json.dumps({k: row[k] for k in ("device", "how", "bin", "deck_temp_K", "seconds",
                                                  "n_curves", "shapes_equal",
                                                  "n_nonfinite_curves")}))
    # forced-bin path check: on the covered devices, forcing the native bin
    # must reproduce the native-selection currents, and forcing another bin
    # must change them
    out["force_check"] = []
    for how, d in picks:
        if how != "native_bin":
            continue
        curves = m.to_pipeline_curves(d)
        native = sp.simulate_pdk(d.polarity, d.L_um, d.W_um, curves, temp_K=77.0)
        b = sp.find_bin_index(d.polarity, d.L_um, d.W_um)
        same = simulate_forced(d.polarity, d.L_um, d.W_um, curves, b, 77.0)
        other_b = 0 if b != 0 else 1
        other = simulate_forced(d.polarity, d.L_um, d.W_um, curves, other_b, 77.0)
        rel = lambda a, c: max(float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-12)))
                               for x, y in zip(a, c))
        row = {"device": d.tag, "native_bin": b, "max_rel_diff_forced_same_bin": rel(native, same),
               "other_bin": other_b, "max_rel_diff_forced_other_bin": rel(native, other)}
        out["force_check"].append(row)
        print(json.dumps(row))
    m.write_json(OUT / "smoke" / "smoke.json", out)
    return 0


# ===========================================================================
# Pre-registered pipeline
# ===========================================================================
from multiprocessing import get_context  # noqa: E402

PREREG = OUT / "prereg.json"
PREDICTION = OUT / "prediction.json"
START_OPTIONS = {"nfev": 3000, "irls": 3, "diff_step": 0.01}
HELDOUT_OPTIONS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
S3_HYPER = {"seeds": (0,), "steps": 500}
SMOKE = {"start_nfev": 100, "start_irls": 1, "heldout_nfev": 60, "heldout_irls": 1,
         "bias_hyper": {"seeds": (0,), "steps": 200}, "short_hyper": {"steps": 200},
         "s3_hyper": {"seeds": (0,), "steps": 50}}
ARMS = ("C0_refit", "M_bias", "T1_inloop")
BASELINE_ARMS = ("B1", "B2", "B3_d3")
T1_PLUS = 0.005


def odir(args, name: str) -> Path:
    return OUT / (f"smoke_{name}" if args.smoke else name)


def prediction_path(args) -> Path:
    return odir(args, "stats") / "prediction.json" if args.smoke else PREDICTION


def smoke_tags(devices, info) -> list[str]:
    picks = []
    for pol in ("nmos", "pmos"):
        for cov in (True, False):
            ds = sorted([d for d in devices if d.polarity == pol and info[d.tag]["bin77k_covered"] == cov],
                        key=lambda d: (d.L_um, d.W_um))
            if ds:
                picks.append(ds[len(ds) // 2].tag)
    return sorted(picks)


def run_devices(args):
    devices, info = dev_devices()
    if args.smoke:
        keep = set(smoke_tags(devices, info))
        devices = [d for d in devices if d.tag in keep]
    return devices, info


def family_of(tags) -> dict:
    return {t: ("pmos" if t.startswith("pfet") else "nmos") for t in tags}


def records_path(args) -> Path:
    return odir(args, "start") / "records"


def load_bins(args) -> dict:
    return json.loads((odir(args, "start") / "bins.json").read_text())["devices"]


def load_records(args):
    from cryoml.explore.e1_correction import load_devices
    return load_devices(records_path(args))


def file_sha(p: Path) -> str:
    return m.sha256(p)


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
def start_task(task):
    from cryoml.ngspice_fit import polish
    dev, binfo, stock, opts = task
    t0 = time.time()
    n_to0 = m.N_TIMEOUT
    box = m.base_box(stock)
    prob = m.MeasProblem(dev, binfo, stock, box)
    stock_sims = prob.simulate_params(None)
    stock_rrms = float(np.mean(prob.curve_rrms(stock_sims)))
    local0 = box.params_to_local(box.published)
    res = polish(prob, local0, max_nfev=opts["nfev"], irls_rounds=opts["irls"],
                 diff_step=opts["diff_step"])
    params = box.local_to_params(res["local"])
    rec = m.to_record(dev, binfo, prob.simulate_params(params), params)
    from cryoml.explore.e1_correction import score_device
    sc = score_device(rec, [c.sim for c in rec.curves])
    loc = np.asarray(res["local"])
    row = {"tag": dev.tag, "stock_rrms": stock_rrms, "start_rrms": float(sc["rrms"]),
           "n_curves": sc["n_curves"], "n_sim": prob.n_sim, "history": res["history"],
           "n_active": int(box.n_active), "n_sim_timeout": m.N_TIMEOUT - n_to0,
           "at_bound": [n for n, z, a in zip(box.names, loc, box.active) if a and abs(z) > 0.999],
           "params": params, "runtime_s": round(time.time() - t0, 1)}
    print(f"  {dev.tag:34s} stock {stock_rrms:.4f} -> start {row['start_rrms']:.4f} "
          f"n_sim {prob.n_sim} {row['runtime_s']} s", flush=True)
    return rec, row


def bins_task(task):
    dev = task
    b = m.bin_choice(dev.polarity, dev.W_um, dev.L_um)
    names = m._base_set()
    stock = m.read_params(dev.polarity, dev.W_um, dev.L_um, b["bin"], names)
    in_text = m.params_in_bin_text(dev.polarity, b["bin"], names)
    curves = m.to_pipeline_curves(dev)
    plain = m.simulate(dev.polarity, dev.W_um, dev.L_um, b["bin"], curves, None)
    patched = m.simulate(dev.polarity, dev.W_um, dev.L_um, b["bin"], curves,
                         {n: v for n, v in stock.items() if v != 0.0 and n in in_text})
    rel = max(float(np.max(np.abs(a - c) / np.maximum(np.abs(a), 1e-12))) for a, c in zip(plain, patched))
    return dev.tag, {**b, "stock": stock, "absent_in_bin_text": [n for n in names if n not in in_text],
                     "stock_readback_vs_card_max_rel_diff": rel}


def run_start(args) -> int:
    from cryoml.explore.e1_correction import save_devices
    t_start = time.time()
    devices, info = run_devices(args)
    opts = dict(START_OPTIONS)
    if args.smoke:
        opts.update(nfev=SMOKE["start_nfev"], irls=SMOKE["start_irls"])
    target = odir(args, "start")
    target.mkdir(parents=True, exist_ok=True)
    with get_context("spawn").Pool(args.workers) as pool:
        bins = dict(pool.map(bins_task, devices, chunksize=1))
        m.write_json(target / "bins.json", {
            "rule": "decision D2: covered geometries the natively selected bin, others the nearest "
                    "bin in (ln L, ln W); stretch factors s_L, s_W >= 1",
            "devices": bins})
        print(f"bins: {sum(b['covered'] for b in bins.values())} covered of {len(bins)}; max stretch "
              f"{max(max(b['stretch_L'], b['stretch_W']) for b in bins.values()):.3f}; max stock "
              f"readback rel diff {max(b['stock_readback_vs_card_max_rel_diff'] for b in bins.values()):.2e}",
              flush=True)
        tasks = [(d, bins[d.tag], bins[d.tag]["stock"], opts) for d in devices]
        out = pool.map(start_task, tasks, chunksize=1)
    pairs = sorted(out, key=lambda x: x[0].tag)
    save_devices(records_path(args), [r for r, _ in pairs])
    rows = {row["tag"]: row for _, row in pairs}
    fam = family_of(rows)
    summary = {"options": opts, "smoke": bool(args.smoke), "deck_temp_K": m.DECK_TEMP_K,
               "stock": m.aggregate({k: v["stock_rrms"] for k, v in rows.items()}, fam),
               "start": m.aggregate({k: v["start_rrms"] for k, v in rows.items()}, fam),
               "n_sim_timeout": int(sum(v["n_sim_timeout"] for v in rows.values())),
               "rows": rows, "runtime_s": round(time.time() - t_start, 1),
               "records_sha256": {s: file_sha(records_path(args).with_suffix(s))
                                  for s in (".json", ".npz")}}
    m.write_json(target / "summary.json", summary)
    print(f"stock {summary['stock']['all_device_mean']:.4f} -> start "
          f"{summary['start']['all_device_mean']:.4f}; timeouts {summary['n_sim_timeout']}; "
          f"runtime {summary['runtime_s']} s")
    return 0


# ---------------------------------------------------------------------------
# stats (S1, S3, prediction)
# ---------------------------------------------------------------------------
_RECS: dict = {}


def _init_recs(path: str) -> None:
    from cryoml.explore.e1_correction import load_devices
    import torch
    torch.set_num_threads(1)
    _RECS.clear()
    _RECS.update({r.tag: r for r in load_devices(Path(path))})


def s3_pair_task(task):
    """Reduced M_bias trained on all records except the pair (a, b);
    v(a -> b) scores b, v(b -> a) scores a (post hoc, no refit)."""
    from cryoml.explore.i3_2_bias import fit_bias
    from cryoml.explore.i7_1_transfer import s3_fold_value
    a, b, hyper = task
    train = [r for t, r in sorted(_RECS.items()) if t not in (a, b)]
    corr = fit_bias("M_bias", train, hyper=hyper, threads=1)
    vb, va = s3_fold_value(corr, [_RECS[b], _RECS[a]])
    return a, b, vb, va


def frozen_rule() -> dict:
    return json.loads((ROOT / "out/research/explore/I7_1/prereg_amendment_2026-10-20.json")
                      .read_text())["frozen_rule"]


def s1_rows(recs, fam) -> dict:
    from cryoml.explore.i7_1_transfer import stats_for
    tags = [r.tag for r in recs]
    rows = {}
    for f in tags:
        train = [r for r in recs if r.tag != f]
        same = [r for r in train if fam[r.tag] == fam[f]]
        rows[f] = {**stats_for(train), "S1_lodo_family": stats_for(same)["S1_lodo"]}
    return rows


def run_stats(args) -> int:
    from cryoml.explore.i7_1_transfer import rule_predict
    t_start = time.time()
    recs = load_records(args)
    tags = sorted(r.tag for r in recs)
    fam = family_of(tags)
    rows = s1_rows(recs, fam)
    t_s1 = round(time.time() - t_start, 2)
    hyper = dict(SMOKE["s3_hyper"] if args.smoke else S3_HYPER)
    pairs = [(a, b, hyper) for i, a in enumerate(tags) for b in tags[i + 1:]]
    print(f"S1 done ({t_s1} s); S3: {len(pairs)} pair fits, hyper {hyper}", flush=True)
    t0 = time.time()
    v: dict = {}
    with get_context("spawn").Pool(args.workers, initializer=_init_recs,
                                   initargs=(str(records_path(args)),)) as pool:
        for a, b, vb, va in pool.imap_unordered(s3_pair_task, pairs, chunksize=1):
            v[(a, b)], v[(b, a)] = vb, va
    t_s3 = round(time.time() - t0, 1)
    for f in tags:
        vals = [v[(f, h)] for h in tags if h != f]
        same = [v[(f, h)] for h in tags if h != f and fam[h] == fam[f]]
        rows[f]["S3"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        rows[f]["S3_family"] = float(np.nanmean(same)) if np.isfinite(same).any() else float("nan")
        rows[f]["S3_inner"] = {h: v[(f, h)] for h in tags if h != f}
    rule = frozen_rule()

    def dmean(key, sel):
        x = np.array([rows[t][key] for t in sel], float)
        return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")

    ds = {k: dmean(k, tags) for k in ("S1_lodo", "S1_pooled", "S2", "S3")}
    y_hat = rule_predict(rule, ds["S1_lodo"])
    per_family = {}
    for f in ("nmos", "pmos"):
        sel = [t for t in tags if fam[t] == f]
        s1f, s3f = dmean("S1_lodo_family", sel), dmean("S3_family", sel)
        per_family[f] = {"n_folds": len(sel), "S1_lodo": s1f, "S3": s3f,
                         "predicted_y_rule": rule_predict(rule, s1f),
                         "P_a_predicted_sign": "gain" if rule_predict(rule, s1f) < 0 else "no gain",
                         "P_c_predicted_sign": "gain" if s3f < 0 else "no gain"}
    rc, head = m._git("rev-parse", "HEAD")
    pred = {
        "id": "I8_1_prediction", "smoke": bool(args.smoke),
        "prereg": "out/research/explore/I8_1/prereg.json",
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_head_at_run": head if rc == 0 else None,
        "records_sha256": {s: file_sha(records_path(args).with_suffix(s)) for s in (".json", ".npz")},
        "frozen_rule": rule, "n_folds": len(tags),
        "dataset": {**ds, "predicted_y": y_hat},
        "P_a": {"predicted_sign": "gain" if y_hat < 0 else "no gain",
                "rule": "gain iff a + b S1_lodo_D < 0", "predicted_y": y_hat},
        "P_c": {"predicted_sign": "gain" if ds["S3"] < 0 else "no gain", "rule": "gain iff S3_D < 0"},
        "per_family_secondary": per_family,
        "gates": {"G": "M_bias" if ds["S3"] < 0 else "C0_refit",
                  "G1": "M_bias" if y_hat < 0 else "C0_refit"},
        "S3_hyper": hyper, "per_fold": rows,
        "timing": {"s1_s": t_s1, "s3_s": t_s3, "n_s3_fits": len(pairs)},
        "note": "computed from the starting-card records only; no refit card or held-out score read"}
    target = prediction_path(args)
    m.write_json(target, pred)
    print(json.dumps({k: pred[k] for k in ("dataset", "P_a", "P_c", "per_family_secondary", "gates",
                                           "timing")}, indent=1))
    return 0


def run_commit_prediction(args) -> int:
    path = prediction_path(args)
    if not path.exists():
        print(f"missing {path}")
        return 1
    rel = str(path.relative_to(ROOT))
    msg = ("I8-1 prediction (pre-registered order step 4): S1_lodo, S3, predicted signs, per-family "
           "values and gate choices from the DEV starting-card records, committed before any "
           "held-out refit\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01FTjyogfhfbXwhjQosGAjKp\n")
    rc, out = m._git("add", "--", rel)
    if rc:
        print(out)
        return 1
    rc_d, _ = m._git("diff", "--cached", "--quiet", "--", rel)
    if rc_d:
        rc, out = m._git("commit", "-m", msg, "--", rel)
        print(out)
        if rc:
            return 1
    g = m.prediction_guard(path)
    print(json.dumps(g))
    if not g["ok"]:
        return 1
    # push; on failure the local commit stands (no automatic rebase, which
    # would touch other work in the tree); the guard below checks the local commit
    rc, out = m._git("push")
    print(out if rc == 0 else f"push failed (local commit kept): {out}")
    g = m.prediction_guard(path)
    print(json.dumps(g))
    return 0 if g["ok"] else 1


# ---------------------------------------------------------------------------
# lodo / baselines
# ---------------------------------------------------------------------------
def train_task(task):
    kind, arm, held, hyper = task
    train = [r for t, r in sorted(_RECS.items()) if t != held]
    if kind == "bias":
        from cryoml.explore.i3_2_bias import fit_bias
        return fit_bias(arm, train, hyper=hyper, threads=1)
    if kind == "t1":
        from cryoml.explore.e1_correction import fit_arm
        return fit_arm("T1", train, hyper=hyper, threads=1)
    if kind == "poly":
        from cryoml.explore.i4_1_baselines import fit_poly
        return fit_poly(arm, 3, train, hyper=hyper)
    raise KeyError(kind)


def refit_task(task):
    from cryoml.explore.e1_correction import Correction, score_device
    from cryoml.explore.i4_1_baselines import (AugProblem, b2_box, extend_box, matched_nfev,
                                               measured_pmax)
    from cryoml.ngspice_fit import polish
    label, kind, payload, rec, dev, binfo, opts = task
    t0 = time.time()
    n_to0 = m.N_TIMEOUT
    start = dict(rec.meta["params"])
    stock = binfo["stock"]
    box = m.base_box(stock)
    nfev, irls, step = opts["heldout_nfev"], opts["heldout_irls"], opts["diff_step"]
    row = {"label": label, "tag": rec.tag}
    corr_final = Correction("C0")
    if kind == "corr":
        prob = m.MeasProblem(dev, binfo, stock, box)
        corr = payload["corr"]
        res = polish(m.corrected_problem(prob, corr), box.params_to_local(start), max_nfev=nfev,
                     irls_rounds=irls, diff_step=step)
        params = box.local_to_params(res["local"])
        new = m.to_record(dev, binfo, prob.simulate_params(params), params)
        corr_final, n_sim = corr, prob.n_sim
        row["n_active"] = int(box.n_active)
    elif kind == "b1":
        extras = m.b1_extras(payload["values"], start, payload["names"])
        box1 = extend_box(box, extras)
        stock1 = {**stock, **{n: e["value"] for n, e in extras.items()}}
        prob = m.MeasProblem(dev, binfo, stock1, box1)
        nfev = matched_nfev(box.n_active, box1.n_active, nfev, irls)
        local0 = box1.params_to_local({**start, **{n: e["value"] for n, e in extras.items()}})
        res = polish(prob, local0, max_nfev=nfev, irls_rounds=irls, diff_step=step)
        params = box1.local_to_params(res["local"])
        new = m.to_record(dev, binfo, prob.simulate_params(params), params)
        n_sim = prob.n_sim
        row.update(n_active=int(box1.n_active), n_extra_active=int(box1.n_active - box.n_active),
                   max_nfev=nfev)
    elif kind == "b2":
        prob = m.MeasProblem(dev, binfo, stock, box)
        rmax = m.rext_max(dev, start)
        box2 = b2_box(box, rmax)
        aug = AugProblem(prob, box2, lambda card, rext: prob.simulate_params(card, rext),
                         measured_pmax(rec))
        nfev = matched_nfev(box.n_active, box2.n_active, nfev, irls)
        local0 = box2.params_to_local({**start, "rext_ohm": 0.0, "heat_kappa": 0.0})
        res = polish(aug, local0, max_nfev=nfev, irls_rounds=irls, diff_step=step)
        params = box2.local_to_params(res["local"])
        new = m.to_record(dev, binfo, aug.simulate(res["local"]), params)
        n_sim = aug.n_sim
        row.update(n_active=int(box2.n_active), max_nfev=nfev, rext_max_ohm=rmax,
                   rext_ohm=params["rext_ohm"], heat_kappa=params["heat_kappa"])
    else:
        raise KeyError(kind)
    after = score_device(new, corr_final.apply(new))
    row.update(rrms=float(after["rrms"]), failed=after["failed"], n_sim=n_sim,
               history=res["history"], params=params, runtime_s=round(time.time() - t0, 2),
               rrms_start_corrected=float(score_device(rec, corr_final.apply(rec))["rrms"]),
               n_sim_timeout=m.N_TIMEOUT - n_to0)
    print(f"  {label:10s} {rec.tag:34s} {row['rrms']:.4f} n_sim {n_sim} {row['runtime_s']} s",
          flush=True)
    return row


def screen_task(task):
    from cryoml.explore.i4_1_baselines import edge_response
    rec, dev, binfo = task
    stock = binfo["stock"]
    prob = m.MeasProblem(dev, binfo, stock, m.base_box(stock))
    start = rec.meta["params"]
    ref = prob.simulate_params(start)
    values = m.read_params(dev.polarity, dev.W_um, dev.L_um, binfo["bin"], m.b1_universe(),
                           required=False)
    out = {}
    for name, e in m.b1_extras(values, start).items():
        lo, hi = e["bounds"]
        if not hi > lo:
            out[name] = {"value": e["value"], "bounds": e["bounds"], "active": False}
            continue
        prob.stock = {**stock, name: e["value"]}
        resp = [edge_response(ref, prob.simulate_params({**start, name: x}), prob.layout)
                for x in (lo, hi)]
        out[name] = {"value": e["value"], "bounds": e["bounds"], "log": e["log"], "active": True,
                     "response_lo": resp[0], "response_hi": resp[1]}
    return rec.tag, {"values": values, "screen": out}


def require_prediction(args) -> dict:
    g = m.prediction_guard(prediction_path(args))
    print(f"prediction guard: {json.dumps(g)}", flush=True)
    if not g["ok"] and not args.smoke:
        raise SystemExit("prediction.json is not committed in an ancestor of HEAD; refusing to start")
    return g


def summaries(scores: dict, tags: list, bins: dict) -> dict:
    fam = family_of(tags)
    cov = [t for t in tags if bins[t]["covered"]]
    out = {"all": {k: m.aggregate({t: v[t] for t in tags if t in v}, fam) for k, v in scores.items()},
           "covered": {k: m.aggregate({t: v[t] for t in cov if t in v}, fam) for k, v in scores.items()}}
    for f in ("nmos", "pmos"):
        sel = [t for t in tags if fam[t] == f]
        out[f] = {k: m.aggregate({t: v[t] for t in sel if t in v}, fam) for k, v in scores.items()}
    return out


def comparisons(scores: dict, tags: list, pairs: list[tuple[str, str]]) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    out = {}
    for a, b in pairs:
        if a in scores and b in scores:
            out[f"{a}_vs_{b}"] = paired_ratio({t: scores[a][t] for t in tags if t in scores[a]},
                                              {t: scores[b][t] for t in tags if t in scores[b]})
    return out


def run_lodo_like(args, arms: tuple[str, ...], name: str) -> int:
    from cryoml.explore.e1_correction import Correction, score_device
    from cryoml.explore.i3_2_bias import merge_seed_results, split_seed_tasks
    t_start = time.time()
    guard = require_prediction(args)
    devices = {d.tag: d for d in run_devices(args)[0]}
    recs = load_records(args)
    by_tag = {r.tag: r for r in recs}
    if sorted(by_tag) != sorted(devices):
        raise RuntimeError("starting records differ from the run devices")
    bins = load_bins(args)
    tags = sorted(by_tag)
    opts = dict(HELDOUT_OPTIONS)
    hyper = {}
    hyper_short = {}
    if args.smoke:
        opts.update(heldout_nfev=SMOKE["heldout_nfev"], heldout_irls=SMOKE["heldout_irls"])
        hyper, hyper_short = dict(SMOKE["bias_hyper"]), dict(SMOKE["short_hyper"])
    timing, corrs, extra = {}, {}, {}
    with get_context("spawn").Pool(args.workers, initializer=_init_recs,
                                   initargs=(str(records_path(args)),)) as pool:
        t0 = time.time()
        bias_tasks = [("M_bias", h) for h in tags] if "M_bias" in arms else []
        expanded, owner = split_seed_tasks(bias_tasks, hyper)
        tasks = [("bias", a, h, hp) for a, h, hp in expanded]
        other = []
        if "T1_inloop" in arms:
            other += [("t1", "T1", h, hyper_short) for h in tags]
        if "B3_d3" in arms:
            other += [("poly", "B3_d3", h, hyper_short) for h in tags]
        results = pool.map(train_task, tasks + other, chunksize=1)
        for h, c in zip(tags, merge_seed_results(len(bias_tasks), owner, results[:len(tasks)])):
            corrs[f"M_bias|{h}"] = c
        rest = iter(results[len(tasks):])
        for arm in ("T1_inloop", "B3_d3"):
            if arm in arms:
                for h in tags:
                    corrs[f"{arm}|{h}"] = next(rest)
        timing["train_s"] = round(time.time() - t0, 1)
        timing["train_n"] = len(tasks + other)
        print(f"training done ({timing['train_s']} s, {len(tasks + other)} tasks)", flush=True)
        b1_names, screen_rows = [], {}
        if "B1" in arms:
            from cryoml.explore.i4_1_baselines import B1_THRESHOLD
            t0 = time.time()
            screen_rows = dict(pool.map(screen_task, [(by_tag[t], devices[t], bins[t]) for t in tags],
                                        chunksize=1))
            names = sorted({n for r in screen_rows.values() for n in r["screen"]})
            per = {}
            for n in names:
                resp = []
                for r in screen_rows.values():
                    e = r["screen"].get(n, {})
                    if e.get("active"):
                        lo, hi = e["response_lo"], e["response_hi"]
                        resp.append(max(lo if np.isfinite(lo) else 0.0, hi if np.isfinite(hi) else 0.0))
                per[n] = {"max_response": float(max(resp)) if resp else 0.0, "n_active": len(resp)}
            b1_names = [n for n in names if per[n]["max_response"] >= B1_THRESHOLD]
            extra["b1_screen"] = {"threshold": B1_THRESHOLD, "candidates": names, "selected": b1_names,
                                  "per_parameter": per}
            timing["screen_s"] = round(time.time() - t0, 1)
            print(f"B1 screen: {len(names)} candidates, selected {b1_names}", flush=True)
        refits = []
        for h in tags:
            rec, dev, binfo = by_tag[h], devices[h], bins[h]
            for arm in arms:
                if arm == "C0_refit":
                    refits.append((arm, "corr", {"corr": Correction("C0")}, rec, dev, binfo, opts))
                elif arm in ("M_bias", "T1_inloop", "B3_d3"):
                    refits.append((arm, "corr", {"corr": corrs[f"{arm}|{h}"]}, rec, dev, binfo, opts))
                elif arm == "B1":
                    refits.append((arm, "b1", {"names": b1_names, "values": screen_rows[h]["values"]},
                                   rec, dev, binfo, opts))
                elif arm == "B2":
                    refits.append((arm, "b2", {}, rec, dev, binfo, opts))
        t0 = time.time()
        rows_l = pool.map(refit_task, refits, chunksize=1)
        timing["refit_s"] = round(time.time() - t0, 1)
        timing["refit_n"] = len(refits)
    rows = {}
    for row in rows_l:
        rows.setdefault(row["label"], {})[row["tag"]] = row
    scores = {a: {t: r["rrms"] for t, r in v.items()} for a, v in rows.items()}
    scores["C0"] = {h: float(score_device(by_tag[h], [c.sim for c in by_tag[h].curves])["rrms"])
                    for h in tags}
    if name == "baselines":
        lodo = json.loads((odir(args, "lodo") / "results.json").read_text())
        for a in ("C0_refit", "M_bias"):
            scores[a] = {t: lodo["per_device"][a][t] for t in tags}
        pairs = [("M_bias", x) for x in arms] + [(x, "C0_refit") for x in arms]
    else:
        pairs = [("M_bias", "C0_refit"), ("M_bias", "T1_inloop"), ("T1_inloop", "C0_refit"),
                 ("C0_refit", "C0")]
    cov = [t for t in tags if bins[t]["covered"]]
    fam = family_of(tags)
    result = {"name": odir(args, name).name, "prereg": "out/research/explore/I8_1/prereg.json",
              "smoke": bool(args.smoke), "prediction_guard": guard, "options": opts,
              "hyper_overrides": {"bias": hyper, "short": hyper_short}, "arms": list(arms),
              "held_out": tags, "covered": cov,
              "summary": summaries(scores, tags, bins),
              "comparisons": {"all": comparisons(scores, tags, pairs),
                              "covered": comparisons(scores, cov, pairs),
                              "nmos": comparisons(scores, [t for t in tags if fam[t] == "nmos"], pairs),
                              "pmos": comparisons(scores, [t for t in tags if fam[t] == "pmos"], pairs)},
              "per_device": scores,
              "n_sim_timeout": int(sum(r.get("n_sim_timeout", 0) for v in rows.values() for r in v.values())),
              "rows": {a: {t: {k: x for k, x in r.items() if k != "params"} for t, r in v.items()}
                       for a, v in rows.items()},
              **extra, "timing": timing, "runtime_s": round(time.time() - t_start, 1)}
    target = odir(args, name)
    m.write_json(target / "results.json", result)
    (target / "corrections.json").write_text(json.dumps({k: c.to_json() for k, c in corrs.items()}) + "\n")
    m.write_json(target / "cards.json", {a: {t: r["params"] for t, r in v.items()} for a, v in rows.items()})
    for k, v in result["summary"]["all"].items():
        print(f"  {k:10s} mean {v['all_device_mean']:.4f} median {v['median']:.4f} failed {v['n_failed']}")
    for k, v in result["comparisons"]["all"].items():
        print(f"  {k:24s} GMR {v['ratio']:.3f} [{v.get('ci_low', float('nan')):.3f}, "
              f"{v.get('ci_high', float('nan')):.3f}] wins {v['wins']}/{v['n_used']}")
    print(f"timing {timing} runtime {result['runtime_s']} s")
    return 0


def run_lodo(args) -> int:
    return run_lodo_like(args, ARMS, "lodo")


def run_baselines(args) -> int:
    return run_lodo_like(args, BASELINE_ARMS, "baselines")


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def p_b_label(cmp: dict, r2: bool, r3: bool) -> str:
    lo, hi = cmp.get("ci_low", float("nan")), cmp.get("ci_high", float("nan"))
    if hi < 1:
        return "supported" if (r2 and r3) else "gain, inconclusive"
    if lo > 1:
        return "harm"
    return "not supported"


def run_decide(args) -> int:
    from cryoml.explore.i7_1_transfer import rule_predict
    recs = load_records(args)
    tags = sorted(r.tag for r in recs)
    fam = family_of(tags)
    bins = load_bins(args)
    pred = json.loads(prediction_path(args).read_text())
    lodo = json.loads((odir(args, "lodo") / "results.json").read_text())
    start = json.loads((odir(args, "start") / "summary.json").read_text())
    base_path = odir(args, "baselines") / "results.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else None
    rows = s1_rows(recs, fam)
    s1_d = float(np.mean([rows[t]["S1_lodo"] for t in tags]))
    s1_check = abs(s1_d - pred["dataset"]["S1_lodo"])
    if s1_check >= 1e-12:
        raise RuntimeError(f"S1_lodo_D {s1_d} differs from prediction.json ({s1_check})")
    sc = lodo["per_device"]

    def y_of(sel):
        lr = [math.log(sc["M_bias"][t] / sc["C0_refit"][t]) for t in sel
              if np.isfinite(sc["M_bias"][t]) and np.isfinite(sc["C0_refit"][t])
              and sc["M_bias"][t] > 0 and sc["C0_refit"][t] > 0]
        return float(np.mean(lr)) if lr else float("nan"), len(lr)

    y_d, n_pair = y_of(tags)
    sign = lambda x: "gain" if x < 0 else "no gain"
    s_all = lodo["summary"]["all"]
    cmp = lodo["comparisons"]["all"]["M_bias_vs_C0_refit"]
    r1 = cmp["ci_high"] < 1
    r2 = s_all["M_bias"]["all_device_mean"] <= s_all["T1_inloop"]["all_device_mean"] + T1_PLUS
    r3 = s_all["M_bias"]["n_failed"] == 0
    n = len(tags)
    v1 = sum(1 for t in tags if np.isfinite(start["rows"][t]["start_rrms"])
             and start["rows"][t]["n_curves"] >= 2) >= 0.9 * n
    v2 = all(s_all[a]["n_failed"] <= 0.1 * n for a in ("C0_refit", "M_bias")) and n_pair >= (45 if not args.smoke else 1)
    g = lodo["prediction_guard"]
    pred_commit = g.get("commit")
    # the lodo job checked the guard before producing any output (refuses otherwise)
    v3 = bool(g.get("ok")) and not args.smoke
    v4 = lodo["n_sim_timeout"] == 0 and start["n_sim_timeout"] == 0
    rule = frozen_rule()
    p_a = {"predicted": pred["P_a"]["predicted_sign"], "observed": sign(y_d),
           "correct": pred["P_a"]["predicted_sign"] == sign(y_d), "y_D": y_d,
           "predicted_y": pred["dataset"]["predicted_y"],
           "abs_error_y": abs(y_d - pred["dataset"]["predicted_y"])}
    p_c = {"predicted": pred["P_c"]["predicted_sign"], "observed": sign(y_d),
           "correct": pred["P_c"]["predicted_sign"] == sign(y_d), "S3_D": pred["dataset"]["S3"]}
    fam_out = {}
    for f in ("nmos", "pmos"):
        yf, nf = y_of([t for t in tags if fam[t] == f])
        pf = pred["per_family_secondary"][f]
        fam_out[f] = {"y": yf, "n": nf, "observed": sign(yf),
                      "rule_correct": pf["P_a_predicted_sign"] == sign(yf),
                      "S3_correct": pf["P_c_predicted_sign"] == sign(yf)}
    gates = {}
    for gname, choice in pred["gates"].items():
        ln_c0 = y_d if choice == "M_bias" else 0.0
        gates[gname] = {"choice": choice, "gmr_vs_C0_refit": math.exp(ln_c0),
                        "gmr_vs_M_bias": math.exp(ln_c0 - y_d),
                        "chose_lower": (choice == "M_bias") == (y_d < 0)}
    cov_cmp = lodo["comparisons"]["covered"].get("M_bias_vs_C0_refit")
    valid = v1 and v2 and v4
    out = {"id": "I8_1_decision", "smoke": bool(args.smoke), "prereg": "out/research/explore/I8_1/prereg.json",
           "prediction_commit": pred_commit, "S1_lodo_D_recomputed_abs_diff": s1_check,
           "validity": {"V1": v1, "V2": v2, "V3_prospective": v3, "V4": v4, "valid": valid},
           "P_a": {**p_a, "stated": valid, "prospective": v3},
           "P_b": {"label": p_b_label(cmp, r2, r3) if valid else "invalid", "R1": r1, "R2": r2, "R3": r3,
                   "gmr": cmp, "means": {a: s_all[a] for a in s_all}},
           "P_c": {**p_c, "stated": valid, "prospective": v3},
           "secondary": {"covered_subset": cov_cmp, "per_family": fam_out, "gates": gates,
                         "baselines": base["comparisons"]["all"] if base else None,
                         "bins_stretch": {t: [bins[t]["stretch_L"], bins[t]["stretch_W"]] for t in tags}},
           "rule": rule}
    m.write_json(odir(args, "decision") / "decision.json" if args.smoke else OUT / "decision.json", out)
    print(json.dumps({k: out[k] for k in ("validity", "P_a", "P_c")}, indent=1))
    print(f"P_b {out['P_b']['label']} GMR {cmp['ratio']:.3f} [{cmp['ci_low']:.3f}, {cmp['ci_high']:.3f}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("sources", "inventory", "data", "split", "smoke", "start", "stats",
                             "commit_prediction", "lodo", "baselines", "decide"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"sources": run_sources, "inventory": run_inventory, "data": run_data,
            "split": run_split, "smoke": run_smoke, "start": run_start, "stats": run_stats,
            "commit_prediction": run_commit_prediction, "lodo": run_lodo,
            "baselines": run_baselines, "decide": run_decide}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
