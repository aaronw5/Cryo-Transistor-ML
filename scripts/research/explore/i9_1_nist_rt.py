#!/usr/bin/env python3
"""I9-1: NIST mds2-2997 MPW-5 sky130 4 K data, families nfet_01v8 (nthin) and
pfet_01v8 (pthin), open sky130 room-temperature tt cards as stock cards.

Modules: ``cryoml.explore.i9_1_nist_rt`` (cards, deck), ``cryoml.explore.i8_1_nist``
(parsing, curves, inclusion, records, box, aggregates, guard).  Pipeline as
I8-1 with amendment A1 built in: a DEV device whose starting record has any
non-finite simulated point is excluded from every training set and S fold,
not refit, and scored as failed (NaN) in every arm.

Modes:
  cards      sha256 and bin counts of the open_pdks card files -> cards.json
  data       nfet_01v8 / pfet_01v8 Vb = 0 curves -> data/nist4k_rt.npz (not
             committed), data/manifest.json (measured inclusion, bin coverage)
  split      DEV / TEST split (geometry and bin coverage only) -> split.json
  smoke      card checks and stock-card checks on DEV devices (no fit, no RRMS)
             -> smoke/smoke.json
  start      starting cards of the DEV devices -> start/
  stats      S1_lodo, S1_pooled, S2, S3, per-family values, gates from the
             starting records only -> prediction.json
  commit_prediction  commit prediction.json alone and push
  lodo       C0, C0_refit, M_bias, T1_inloop, M_bias_lvt (refuses to start
             unless prereg.json is in force and prediction.json committed)
  baselines  B1 screen, B1, B2, B3_d3 (same guard)
  decide     P_a, P_b, P_c, secondary outcomes, validity -> decision.json

--smoke: smoke devices (smoke_tags), reduced settings (SMOKE), outputs under
smoke_* (not used for any decision).

Run with ``python -I`` from the repository root:
  .venv/bin/python -I scripts/research/explore/i9_1_nist_rt.py --mode data
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml import spice_pdk as sp  # noqa: E402
from cryoml.explore import i8_1_nist as b  # noqa: E402
from cryoml.explore import i9_1_nist_rt as m  # noqa: E402

OUT = m.OUT
DATA = OUT / "data"
NPZ = DATA / "nist4k_rt.npz"
MANIFEST = DATA / "manifest.json"
SPLIT = OUT / "split.json"
SPLIT_SEED = 20261022
EXCLUDE_BAD = True
MIN_INCLUDED = 2                 # measured-only rule; I8-1 V1 threshold (draft D12)
PREREG = OUT / "prereg.json"
PREDICTION = OUT / "prediction.json"
I8_1_RECORDS = ROOT / "out" / "research" / "explore" / "I8_1" / "start" / "records"

START_OPTIONS = {"nfev": 3000, "irls": 3, "diff_step": 0.01}
HELDOUT_OPTIONS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
S3_HYPER = {"seeds": (0,), "steps": 500}
SMOKE = {"start_nfev": 100, "start_irls": 1, "heldout_nfev": 60, "heldout_irls": 1,
         "bias_hyper": {"seeds": (0,), "steps": 200}, "short_hyper": {"steps": 200},
         "s3_hyper": {"seeds": (0,), "steps": 50}}
ARMS = ("C0_refit", "M_bias", "T1_inloop", "M_bias_lvt")
BASELINE_ARMS = ("B1", "B2", "B3_d3")
T1_PLUS = 0.005
V2_MAX_FAILED = 0.10
V2_MIN_PAIRED = 0.80
V1_MIN_FINITE = 0.90


# ---------------------------------------------------------------------------
# cards, data, split
# ---------------------------------------------------------------------------
def run_cards(args) -> int:
    src = m.card_sources()
    rows = {k: {"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                "resolved": str(p.resolve()), "bytes": p.stat().st_size, "sha256": m.sha256_file(p)}
            for k, p in src.items()}
    cards = {}
    for pol in ("nmos", "pmos"):
        parts = m.card_parts(pol)
        g = parts["geometry"]
        cards[pol] = {"subckt": m.SUBCKT[pol], "n_bins": len(parts["blocks"]),
                      "lines_not_parsed": parts["n_unparsed"],
                      "L_um": [min(x["lmin"] for x in g.values()) * 1e6, max(x["lmax"] for x in g.values()) * 1e6],
                      "W_um": [min(x["wmin"] for x in g.values()) * 1e6, max(x["wmax"] for x in g.values()) * 1e6]}
    out = {"id": "I9_1_cards", "open_pdks_commit": m.OPEN_PDKS_COMMIT,
           "provenance": "Volare build of open_pdks for sky130 (volare enable --pdk sky130 "
                         f"{m.OPEN_PDKS_COMMIT}, PDK_ROOT=data/raw/pdk; already on disk, used by the "
                         "canonical setup for the mismatch corners). Model files from "
                         "google/skywater-pdk-libs-sky130_fd_pr as packaged by open_pdks; "
                         "no download in I9-1",
           "files": rows, "cards": cards, "tnom_C": m.TNOM_C,
           "handling": "open_pdks tt: nfet_01v8 tt.pm3 included directly (corners/tt.spice), "
                       "pfet_01v8 tt.corner (.param block + tt.pm3); every '+' line of a bin "
                       "block rewritten with one assignment per line; per simulation the card "
                       "holds the subckt with the m-instance forced to one bin and that bin's "
                       "block only; library adds parameters/lod.spice and the mismatch corner, "
                       "mc_mm_switch = mc_pr_switch = 0"}
    b.write_json(OUT / "cards.json", out)
    print(json.dumps(cards, indent=1))
    return 0


def run_data(args) -> int:
    devices = b.discover_mpw5(folders=m.CHOSEN)
    saved = b.save_dataset(devices, NPZ)
    rows, inc_counts = [], {}
    for d in devices:
        ck = b.device_checks(d)
        bc = m.bin_choice(d.polarity, d.W_um, d.L_um)
        inc = []
        sign = -1.0 if d.polarity == "pmos" else 1.0
        for c in b.fit_curves(d):
            r = b.inclusion_measured(c.kind, sign * c.id)
            inc.append(f"{c.key}|{c.fixed:g}|{len(c.sweep)}|{r['start']}|{r['den']:.4g}|"
                       f"{int(r['included'])}")
        n_inc = sum(int(s.rsplit("|", 1)[1]) for s in inc)
        inc_counts.setdefault(d.dev_type, []).append(n_inc)
        rows.append({**b.device_row(d), "bad": ck["bad"], "gate_control_ratio": ck["gate_control_ratio"],
                     "bin_rt": bc["bin"], "bin_rt_covered": bc["covered"],
                     "bin_rt_candidates": bc["candidates"], "bin_rt_log_distance": round(bc["log_distance"], 4),
                     "curves_key|fixed_V|n|trim_start|den_A|included": inc,
                     "n_included_measured_rule": n_inc})
    manifest = {
        "id": "I9_1_data", "doi": b.DOI, "tile": "MPW-5 Test Tile, transistormodule",
        "families": {f: b.FOLDERS[f][0] for f in m.CHOSEN},
        "curves": "Vb = 0 curves per device (I8-1 curve set): Id-Vg at |Vd| 0.1 and 1.8 V (keys "
                  "idvg@d0, d1; |Vg| 0-1.8 V, 10 mV, 181 points) and Id-Vd at |Vg| 0, 0.3, ..., 1.8 V "
                  "(keys idvd@g0..g6; |Vd| 0-1.8 V, 20 mV, 91 points); npz holds the raw stored sign",
        "inclusion_measured_rule": "pdk2.metric: clean_current, trim to the first |I| > 1e-10 A, "
                                   "include iff trimmed mean |I| > 3e-7 A (Id-Vd) / 2e-7 A (Id-Vg)",
        "included_per_family": {k: {"n_devices": len(v), "total": int(sum(v)),
                                    "min": int(min(v)), "max": int(max(v))}
                                for k, v in inc_counts.items()},
        "bin_rt": {"cards": "open_pdks sky130 tt (cards.json)",
                   "covered_per_family": {f: sum(1 for r in rows if r["dev_type"] == f and r["bin_rt_covered"])
                                          for f in inc_counts},
                   "rule": "covered: the geometry lies in a closed bin box (1e-6 um tolerance); bin = "
                           "the bin ngspice selects with the full card (eta0 sentinel readback when "
                           "several boxes share an edge); uncovered: nearest bin in (ln L, ln W)"},
        "devices": rows, **saved}
    b.write_json(MANIFEST, manifest)
    print(json.dumps({k: manifest[k] for k in ("included_per_family", "bin_rt")}, indent=1))
    return 0


def run_split(args) -> int:
    manifest = json.loads(MANIFEST.read_text())
    devices = b.load_dataset(NPZ, manifest)
    info = {r["tag"]: r for r in manifest["devices"]}
    excluded = {t: r["bad"] + [f for f in r["flags"] if f.startswith("readme")]
                for t, r in info.items()
                if EXCLUDE_BAD and (r["bad"] or any(f.startswith("readme") for f in r["flags"]))}
    for t, r in info.items():                     # draft decision D12
        if r["n_included_measured_rule"] < MIN_INCLUDED:
            excluded.setdefault(t, []).append(f"n_included {r['n_included_measured_rule']} < {MIN_INCLUDED}")
    keep = [d for d in devices if d.tag not in excluded]
    strata = []
    for d in keep:
        cov = "covered" if info[d.tag]["bin_rt_covered"] else "uncovered"
        strata.append(b.NistDevice(d.tag, d.folder, f"{d.dev_type}|{cov}", d.polarity, d.module,
                                   d.position, d.W_um, d.L_um))
    split = b.make_split(strata, SPLIT_SEED)
    counts = {}
    for part in ("DEV", "TEST"):
        for t in split[part]:
            key = f"{info[t]['dev_type']}|{'covered' if info[t]['bin_rt_covered'] else 'uncovered'}"
            counts.setdefault(key, {"DEV": 0, "TEST": 0})[part] += 1
    out = {"id": "I9_1_split", "seed": SPLIT_SEED,
           "rule": "I8-1 procedure: exclude devices with a readme flag or a bad check, and (draft D12) "
                   "devices with fewer than 2 curves included by the measured-only rule; strata = family "
                   "x room-temperature card bin coverage; per stratum devices sorted by (L, W), "
                   "consecutive blocks of 5, seeded permutation puts 3 of 5 in DEV (short last block: "
                   "round(0.6 n)); numpy.random.default_rng(seed), strata in sorted order "
                   "(cryoml.explore.i8_1_nist.make_split). Uses geometry, flags and bin coverage only",
           "excluded": excluded, "DEV": split["DEV"], "TEST": split["TEST"],
           "n_dev": split["n_dev"], "n_test": split["n_test"], "counts": counts,
           "blocks": split["blocks"],
           "TEST_policy": "TEST arrays are not loaded by any later I9-1 mode (dev_devices loads DEV "
                          "tags only); TEST is used once in December with the method frozen then"}
    b.write_json(SPLIT, out)
    print(json.dumps({k: out[k] for k in ("excluded", "n_dev", "n_test", "counts")}, indent=1))
    return 0


def dev_devices():
    split = json.loads(SPLIT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    devices = b.load_dataset(NPZ, manifest, split["DEV"])
    if sorted(d.tag for d in devices) != sorted(split["DEV"]):
        raise RuntimeError("DEV devices missing from the dataset")
    return devices, {r["tag"]: r for r in manifest["devices"]}


# ---------------------------------------------------------------------------
# smoke (card and stock-card checks; no fit, no RRMS)
# ---------------------------------------------------------------------------
SMOKE_TEMPS = (("tnom_30C", 30.0, False), ("27C", 27.0, False), ("4.2K", 4.2 - 273.15, False),
               ("4.2K_tc0", 4.2 - 273.15, True))


def _rel(a, c) -> float:
    return max(float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-15))) for x, y in zip(a, c))


def _fatal_lines(log: str) -> list[str]:
    return sorted({ln.strip() for ln in log.splitlines()
                   if "fatal" in ln.lower() or "is not positive" in ln.lower()})


def stock_scan_task(task):
    dev, binfo = task
    curves = b.to_pipeline_curves(dev)
    t0 = time.time()
    sims, log = m.simulate(dev.polarity, dev.W_um, dev.L_um, binfo["bin"], curves, return_log=True)
    return dev.tag, {"bin": binfo["bin"], "covered": binfo["covered"],
                     "n_nonfinite_curves": int(sum(not np.all(np.isfinite(s)) for s in sims)),
                     "nonfinite_keys": [c.meta["key"] for c, s in zip(curves, sims)
                                        if not np.all(np.isfinite(s))],
                     "fatal_lines": _fatal_lines(log), "seconds": round(time.time() - t0, 3)}


def smoke_tags(devices, info, failed: list[str] | None = None) -> list[str]:
    """Per family the median-(L, W) DEV device of each bin-coverage stratum,
    plus every DEV device whose stock card gave a non-finite curve in the
    stock scan (smoke/smoke.json), at most 2."""
    picks = sorted(failed or [])[:2]
    for pol in ("nmos", "pmos"):
        for cov in (True, False):
            ds = sorted([d for d in devices if d.polarity == pol and info[d.tag]["bin_rt_covered"] == cov],
                        key=lambda d: (d.L_um, d.W_um))
            if ds:
                picks.append(ds[len(ds) // 2].tag)
    # if a family has no uncovered device: nMOS shortest-(L, W), pMOS longest-(L, W) DEV device
    for pol in ("nmos", "pmos"):
        if not any(d.polarity == pol and not info[d.tag]["bin_rt_covered"] for d in devices):
            ds = sorted([d for d in devices if d.polarity == pol], key=lambda d: (d.L_um, d.W_um))
            picks.append(ds[0].tag if pol == "nmos" else ds[-1].tag)
    # the DEV device with the fewest included curves (ties: smallest (W, L))
    low = min(devices, key=lambda d: (info[d.tag]["n_included_measured_rule"], d.W_um, d.L_um))
    picks.append(low.tag)
    return sorted(set(picks))


def run_smoke(args) -> int:
    from multiprocessing import get_context
    devices, info = dev_devices()
    bins = {d.tag: {"bin": info[d.tag]["bin_rt"], "covered": info[d.tag]["bin_rt_covered"]}
            for d in devices}
    out = {"id": "I9_1_smoke", "simulator": sp._NGSPICE_BIN,
           "note": "stock room-temperature card, no fit, no RRMS; finiteness, shape, sign, timing, "
                   "card checks; descriptive subthreshold swing", "runs": []}
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        scan = dict(pool.map(stock_scan_task, [(d, bins[d.tag]) for d in devices], chunksize=1))
    failed = sorted(t for t, r in scan.items() if r["n_nonfinite_curves"])
    out["stock_scan"] = {"deck_temp_C": m.DECK_TEMP_C, "n_devices": len(scan), "failed": failed,
                         "n_with_fatal_lines": sum(1 for r in scan.values() if r["fatal_lines"]),
                         "seconds_per_sim_median": float(np.median([r["seconds"] for r in scan.values()])),
                         "seconds_per_sim_max": float(max(r["seconds"] for r in scan.values())),
                         "wall_s": round(time.time() - t0, 1), "devices": scan}
    print(f"stock scan: {len(scan)} DEV devices, {len(failed)} with a non-finite curve {failed}; "
          f"median {out['stock_scan']['seconds_per_sim_median']:.3f} s per sim", flush=True)
    tags = smoke_tags(devices, info, failed)
    out["smoke_devices"] = tags
    by = {d.tag: d for d in devices}
    for t in tags:
        d = by[t]
        curves = b.to_pipeline_curves(d)
        bi = bins[t]["bin"]
        # card checks: forced single-bin card vs full normalized card (native
        # selection) vs the unmodified open_pdks library, at the deck temperature
        forced = m.simulate(d.polarity, d.W_um, d.L_um, bi, curves)
        row_c = {"device": t, "bin": bi, "covered": bins[t]["covered"]}
        if bins[t]["covered"]:
            full = m.simulate(d.polarity, d.W_um, d.L_um, bi, curves, all_bins=True)
            t1 = time.time()
            off = m.simulate_official(d.polarity, d.W_um, d.L_um, curves)
            row_c.update(max_rel_forced_vs_full=_rel(full, forced), max_rel_full_vs_official=_rel(off, full),
                         max_abs_full_vs_official_A=max(float(np.max(np.abs(x - y))) for x, y in zip(off, full)),
                         official_seconds=round(time.time() - t1, 2))
        out.setdefault("card_checks", []).append(row_c)
        print(json.dumps(row_c), flush=True)
        meas_ss = [m.min_swing_mV_dec(c.Vg, c.Id) for c in curves if c.kind == "idvg"]
        for label, temp_c, tc0 in SMOKE_TEMPS:
            t1 = time.time()
            params = None
            if tc0:
                names = [pre + n for n in m.TC_ZERO for pre in ("", "l", "w", "p")]
                stock_tc = m.read_params(d.polarity, d.W_um, d.L_um, bi, names, required=False)
                params = {k: 0.0 for k in stock_tc}
            sims, log = m.simulate(d.polarity, d.W_um, d.L_um, bi, curves, params=params, temp_c=temp_c,
                                   return_log=True)
            dt = time.time() - t1
            row = {"device": t, "deck": label, "deck_temp_C": round(temp_c, 4), "seconds": round(dt, 3),
                   "n_curves": len(curves),
                   "shapes_equal": all(np.shape(s) == np.shape(c.Id) for c, s in zip(curves, sims)),
                   "n_nonfinite_curves": int(sum(not np.all(np.isfinite(s)) for s in sims)),
                   "fatal_lines": _fatal_lines(log),
                   "sign_agrees_at_max_bias": [bool(np.sign(s[-1]) == np.sign(c.Id[-1]))
                                               for c, s in zip(curves, sims) if abs(c.Id[-1]) > 1e-9],
                   "sim_last_A": {c.meta["key"]: float(s[-1]) for c, s in zip(curves, sims)},
                   "meas_last_A": {c.meta["key"]: float(c.Id[-1]) for c in curves},
                   "sim_min_abs_A_idvg": [float(np.nanmin(np.abs(s))) for c, s in zip(curves, sims)
                                          if c.kind == "idvg"],
                   "sim_min_swing_mV_dec": [m.min_swing_mV_dec(c.Vg, s) for c, s in zip(curves, sims)
                                            if c.kind == "idvg"],
                   "meas_min_swing_mV_dec": meas_ss}
            out["runs"].append(row)
            print(json.dumps({k: row[k] for k in ("device", "deck", "seconds", "n_nonfinite_curves",
                                                  "fatal_lines", "sim_min_swing_mV_dec",
                                                  "meas_min_swing_mV_dec")}), flush=True)
    b.write_json(OUT / "smoke" / "smoke.json", out)
    return 0


# ===========================================================================
# Pre-registered pipeline (prereg.json)
# ===========================================================================
from multiprocessing import get_context  # noqa: E402


def odir(args, name: str) -> Path:
    return OUT / (f"smoke_{name}" if args.smoke else name)


def prediction_path(args) -> Path:
    return odir(args, "stats") / "prediction.json" if args.smoke else PREDICTION


def run_devices(args):
    devices, info = dev_devices()
    if args.smoke:
        smoke = json.loads((OUT / "smoke" / "smoke.json").read_text())
        keep = set(smoke["smoke_devices"])
        devices = [d for d in devices if d.tag in keep]
    return devices, info


def nan_scores(per_device: dict) -> dict:
    return {a: {t: (float("nan") if x is None else float(x)) for t, x in v.items()}
            for a, v in per_device.items()}


def family_of(tags) -> dict:
    return {t: ("pmos" if t.startswith("pfet") else "nmos") for t in tags}


def records_path(args) -> Path:
    return odir(args, "start") / "records"


def load_bins(args) -> dict:
    return json.loads((odir(args, "start") / "bins.json").read_text())["devices"]


def load_records(args):
    from cryoml.explore.e1_correction import load_devices
    return load_devices(records_path(args))


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
def start_task(task):
    from cryoml.ngspice_fit import polish
    from cryoml.explore.e1_correction import score_device
    dev, binfo, stock, opts = task
    t0 = time.time()
    n_to0 = m.N_TIMEOUT
    box = b.base_box(stock)
    prob = m.MeasProblemRT(dev, binfo, stock, box)
    stock_sims = prob.simulate_params(None)
    stock_rrms = float(np.mean(prob.curve_rrms(stock_sims)))
    res = polish(prob, box.params_to_local(box.published), max_nfev=opts["nfev"],
                 irls_rounds=opts["irls"], diff_step=opts["diff_step"])
    params = box.local_to_params(res["local"])
    rec = b.to_record(dev, binfo, prob.simulate_params(params), params)
    sc = score_device(rec, [c.sim for c in rec.curves])
    loc = np.asarray(res["local"])
    row = {"tag": dev.tag, "stock_rrms": stock_rrms, "start_rrms": float(sc["rrms"]),
           "n_curves": sc["n_curves"], "n_sim": prob.n_sim, "history": res["history"],
           "n_active": int(box.n_active), "n_sim_timeout": m.N_TIMEOUT - n_to0,
           "finite_record": b.finite_record(rec),
           "at_bound": [n for n, z, a in zip(box.names, loc, box.active) if a and abs(z) > 0.999],
           "params": params, "runtime_s": round(time.time() - t0, 1)}
    print(f"  {dev.tag:34s} n_sim {prob.n_sim} finite {row['finite_record']} {row['runtime_s']} s",
          flush=True)
    return rec, row


def bins_task(dev):
    bc = m.bin_choice(dev.polarity, dev.W_um, dev.L_um)
    names = b._base_set()
    stock = m.read_params(dev.polarity, dev.W_um, dev.L_um, bc["bin"], names)
    in_text = m.params_in_bin_text(dev.polarity, bc["bin"], names)
    curves = b.to_pipeline_curves(dev)
    plain = m.simulate(dev.polarity, dev.W_um, dev.L_um, bc["bin"], curves, None)
    patched = m.simulate(dev.polarity, dev.W_um, dev.L_um, bc["bin"], curves,
                         {n: v for n, v in stock.items() if v != 0.0 and n in in_text})
    pairs = [(a, c) for a, c in zip(plain, patched) if np.all(np.isfinite(a)) and np.all(np.isfinite(c))]
    rel = max((float(np.max(np.abs(a - c) / np.maximum(np.abs(a), 1e-12))) for a, c in pairs),
              default=float("nan"))
    return dev.tag, {**bc, "stock": stock, "absent_in_bin_text": [n for n in names if n not in in_text],
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
        for t, bc in bins.items():
            if bc["bin"] != info[t]["bin_rt"]:
                raise RuntimeError(f"{t}: bin {bc['bin']} != manifest {info[t]['bin_rt']}")
        b.write_json(target / "bins.json", {
            "rule": "covered geometries the natively selected bin, others the nearest bin in "
                    "(ln L, ln W); stretch factors s_L, s_W >= 1", "devices": bins})
        tasks = [(d, bins[d.tag], bins[d.tag]["stock"], opts) for d in devices]
        out = pool.map(start_task, tasks, chunksize=1)
    pairs = sorted(out, key=lambda x: x[0].tag)
    save_devices(records_path(args), [r for r, _ in pairs])
    rows = {row["tag"]: row for _, row in pairs}
    fam = family_of(rows)
    summary = {"options": opts, "smoke": bool(args.smoke), "deck_temp_C": m.DECK_TEMP_C,
               "stock": b.aggregate({k: v["stock_rrms"] for k, v in rows.items()}, fam),
               "start": b.aggregate({k: v["start_rrms"] for k, v in rows.items()}, fam),
               "n_sim_timeout": int(sum(v["n_sim_timeout"] for v in rows.values())),
               "non_finite_records": sorted(t for t, v in rows.items() if not v["finite_record"]),
               "rows": rows, "runtime_s": round(time.time() - t_start, 1),
               "records_sha256": {s: b.sha256(records_path(args).with_suffix(s)) for s in (".json", ".npz")}}
    b.write_json(target / "summary.json", summary)
    print(f"start: {len(rows)} devices; non-finite records {summary['non_finite_records']}; "
          f"timeouts {summary['n_sim_timeout']}; runtime {summary['runtime_s']} s")
    return 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
_RECS: dict = {}
_EXT: dict = {}


def _init_recs(path: str) -> None:
    from cryoml.explore.e1_correction import load_devices
    import torch
    torch.set_num_threads(1)
    _RECS.clear()
    good, _ = b.split_finite(load_devices(Path(path)))
    _RECS.update({r.tag: r for r in good})


def _ext_records() -> list:
    """I8-1 DEV starting-card records with finite simulated points (transfer
    arm M_bias_lvt), loaded once per worker."""
    if "recs" not in _EXT:
        from cryoml.explore.e1_correction import load_devices
        good, _ = b.split_finite(load_devices(I8_1_RECORDS))
        _EXT["recs"] = sorted(good, key=lambda r: r.tag)
    return _EXT["recs"]


def s3_pair_task(task):
    from cryoml.explore.i3_2_bias import fit_bias
    from cryoml.explore.i7_1_transfer import s3_fold_value
    a, c, hyper = task
    train = [r for t, r in sorted(_RECS.items()) if t not in (a, c)]
    corr = fit_bias("M_bias", train, hyper=hyper, threads=1)
    vc, va = s3_fold_value(corr, [_RECS[c], _RECS[a]])
    return a, c, vc, va


def frozen_rule() -> dict:
    return json.loads((ROOT / "out/research/explore/I7_1/prereg_amendment_2026-10-20.json")
                      .read_text())["frozen_rule"]


def s1_rows(recs, fam) -> dict:
    from cryoml.explore.i7_1_transfer import stats_for
    recs, _ = b.split_finite(recs)
    rows = {}
    for f in [r.tag for r in recs]:
        train = [r for r in recs if r.tag != f]
        same = [r for r in train if fam[r.tag] == fam[f]]
        rows[f] = {**stats_for(train), "S1_lodo_family": stats_for(same)["S1_lodo"]}
    return rows


def run_stats(args) -> int:
    from cryoml.explore.i7_1_transfer import rule_predict
    t_start = time.time()
    all_recs = load_records(args)
    recs, excluded = b.split_finite(all_recs)
    print(f"A1: {len(excluded)} records with non-finite simulated points excluded: {excluded}", flush=True)
    tags = sorted(r.tag for r in recs)
    fam = family_of(tags)
    rows = s1_rows(recs, fam)
    t_s1 = round(time.time() - t_start, 2)
    hyper = dict(SMOKE["s3_hyper"] if args.smoke else S3_HYPER)
    pairs = [(a, c, hyper) for i, a in enumerate(tags) for c in tags[i + 1:]]
    print(f"S1 done ({t_s1} s); S3: {len(pairs)} pair fits, hyper {hyper}", flush=True)
    t0 = time.time()
    v: dict = {}
    with get_context("spawn").Pool(args.workers, initializer=_init_recs,
                                   initargs=(str(records_path(args)),)) as pool:
        for a, c, vc, va in pool.imap_unordered(s3_pair_task, pairs, chunksize=1):
            v[(a, c)], v[(c, a)] = vc, va
    t_s3 = round(time.time() - t0, 1)
    for f in tags:
        vals = [v[(f, h)] for h in tags if h != f]
        if not all(np.isfinite(x) for x in vals):
            raise RuntimeError(f"fold {f}: non-finite S3 inner value")
        same = [v[(f, h)] for h in tags if h != f and fam[h] == fam[f]]
        rows[f]["S3"] = float(np.mean(vals)) if vals else float("nan")
        rows[f]["S3_family"] = float(np.mean(same)) if same else float("nan")
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
    rc, head = b._git("rev-parse", "HEAD")
    pred = {"id": "I9_1_prediction", "smoke": bool(args.smoke),
            "prereg": "out/research/explore/I9_1/prereg.json",
            "excluded_records": excluded, "n_records": len(all_recs),
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_head_at_run": head if rc == 0 else None,
            "records_sha256": {s: b.sha256(records_path(args).with_suffix(s)) for s in (".json", ".npz")},
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
            "note": "computed from the starting-card records only (A1: finite records); no refit card "
                    "or held-out score read"}
    target = prediction_path(args)
    b.write_json(target, pred)
    print(json.dumps({k: pred[k] for k in ("dataset", "P_a", "P_c", "per_family_secondary", "gates",
                                           "timing")}, indent=1))
    return 0


def run_commit_prediction(args) -> int:
    path = prediction_path(args)
    if args.smoke or not path.exists():
        print(f"missing {path}" if not path.exists() else "no commit in smoke mode")
        return 1
    rel = str(path.relative_to(ROOT))
    msg = ("I9-1 prediction (pre-registered order step): S1_lodo, S3, predicted signs, per-family "
           "values and gate choices from the DEV starting-card records, committed before any "
           "held-out refit\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01FTjyogfhfbXwhjQosGAjKp\n")
    rc, out = b._git("add", "--", rel)
    if rc:
        print(out)
        return 1
    rc_d, _ = b._git("diff", "--cached", "--quiet", "--", rel)
    if rc_d:
        rc, out = b._git("commit", "-m", msg, "--", rel)
        print(out)
        if rc:
            return 1
    rc, out = b._git("push")
    print(out if rc == 0 else f"push failed (local commit kept): {out}")
    g = b.prediction_guard(path)
    print(json.dumps(g))
    return 0 if g["ok"] else 1


# ---------------------------------------------------------------------------
# lodo / baselines
# ---------------------------------------------------------------------------
def train_task(task):
    kind, arm, held, hyper = task
    if kind == "bias_ext":
        from cryoml.explore.i3_2_bias import fit_bias
        return fit_bias("M_bias", _ext_records(), hyper=hyper, threads=1)
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
    box = b.base_box(stock)
    nfev, irls, step = opts["heldout_nfev"], opts["heldout_irls"], opts["diff_step"]
    row = {"label": label, "tag": rec.tag}
    corr_final = Correction("C0")
    if kind == "corr":
        prob = m.MeasProblemRT(dev, binfo, stock, box)
        corr = payload["corr"]
        res = polish(b.corrected_problem(prob, corr), box.params_to_local(start), max_nfev=nfev,
                     irls_rounds=irls, diff_step=step)
        params = box.local_to_params(res["local"])
        new = b.to_record(dev, binfo, prob.simulate_params(params), params)
        corr_final, n_sim = corr, prob.n_sim
        row["n_active"] = int(box.n_active)
    elif kind == "b1":
        extras = b.b1_extras(payload["values"], start, payload["names"])
        box1 = extend_box(box, extras)
        stock1 = {**stock, **{n: e["value"] for n, e in extras.items()}}
        prob = m.MeasProblemRT(dev, binfo, stock1, box1)
        nfev = matched_nfev(box.n_active, box1.n_active, nfev, irls)
        local0 = box1.params_to_local({**start, **{n: e["value"] for n, e in extras.items()}})
        res = polish(prob, local0, max_nfev=nfev, irls_rounds=irls, diff_step=step)
        params = box1.local_to_params(res["local"])
        new = b.to_record(dev, binfo, prob.simulate_params(params), params)
        n_sim = prob.n_sim
        row.update(n_active=int(box1.n_active), n_extra_active=int(box1.n_active - box.n_active),
                   max_nfev=nfev)
    elif kind == "b2":
        prob = m.MeasProblemRT(dev, binfo, stock, box)
        rmax = b.rext_max(dev, start)
        box2 = b2_box(box, rmax)
        aug = AugProblem(prob, box2, lambda card, rext: prob.simulate_params(card, rext),
                         measured_pmax(rec))
        nfev = matched_nfev(box.n_active, box2.n_active, nfev, irls)
        local0 = box2.params_to_local({**start, "rext_ohm": 0.0, "heat_kappa": 0.0})
        res = polish(aug, local0, max_nfev=nfev, irls_rounds=irls, diff_step=step)
        params = box2.local_to_params(res["local"])
        new = b.to_record(dev, binfo, aug.simulate(res["local"]), params)
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
    print(f"  {label:10s} {rec.tag:34s} n_sim {n_sim} {row['runtime_s']} s", flush=True)
    return row


def screen_task(task):
    from cryoml.explore.i4_1_baselines import edge_response
    rec, dev, binfo = task
    stock = binfo["stock"]
    prob = m.MeasProblemRT(dev, binfo, stock, b.base_box(stock))
    start = rec.meta["params"]
    ref = prob.simulate_params(start)
    values = m.read_params(dev.polarity, dev.W_um, dev.L_um, binfo["bin"], b.b1_universe(), required=False)
    out = {}
    for name, e in b.b1_extras(values, start).items():
        lo, hi = e["bounds"]
        if not hi > lo:
            out[name] = {"value": e["value"], "bounds": e["bounds"], "active": False}
            continue
        prob.stock = {**stock, name: e["value"]}
        resp = [edge_response(ref, prob.simulate_params({**start, name: x}), prob.layout) for x in (lo, hi)]
        out[name] = {"value": e["value"], "bounds": e["bounds"], "log": e["log"], "active": True,
                     "response_lo": resp[0], "response_hi": resp[1]}
    return rec.tag, {"values": values, "screen": out}


def require_prereg_and_prediction(args) -> dict:
    """Outside smoke: prereg.json in force (status not DRAFT) and prereg.json
    and prediction.json tracked, unmodified and in ancestors of HEAD."""
    out = {"prediction": b.prediction_guard(prediction_path(args))}
    if not args.smoke:
        out["prereg"] = b.prediction_guard(PREREG)
        status = json.loads(PREREG.read_text()).get("status", "") if PREREG.exists() else ""
        out["prereg_in_force"] = not status.upper().startswith("DRAFT")
    for k, v in out.items():
        print(f"guard {k}: {json.dumps(v)}", flush=True)
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict)) and out.get("prereg_in_force", True)
    out["commit"] = out["prediction"].get("commit")
    if not out["ok"] and not args.smoke:
        raise SystemExit("prereg.json not in force or prediction.json not committed in an ancestor of "
                         "HEAD; refusing to start")
    return out


def summaries(scores: dict, tags: list, bins: dict) -> dict:
    fam = family_of(tags)
    cov = [t for t in tags if bins[t]["covered"]]
    out = {"all": {k: b.aggregate({t: v[t] for t in tags if t in v}, fam) for k, v in scores.items()},
           "covered": {k: b.aggregate({t: v[t] for t in cov if t in v}, fam) for k, v in scores.items()}}
    for f in ("nmos", "pmos"):
        sel = [t for t in tags if fam[t] == f]
        out[f] = {k: b.aggregate({t: v[t] for t in sel if t in v}, fam) for k, v in scores.items()}
    return out


def comparisons(scores: dict, tags: list, pairs: list) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    return {f"{a}_vs_{c}": paired_ratio({t: scores[a][t] for t in tags if t in scores[a]},
                                        {t: scores[c][t] for t in tags if t in scores[c]})
            for a, c in pairs if a in scores and c in scores}


def run_lodo_like(args, arms: tuple, name: str) -> int:
    from cryoml.explore.e1_correction import Correction, score_device
    from cryoml.explore.i3_2_bias import merge_seed_results, split_seed_tasks
    t_start = time.time()
    guard = require_prereg_and_prediction(args)
    devices = {d.tag: d for d in run_devices(args)[0]}
    all_recs = load_records(args)
    if sorted(r.tag for r in all_recs) != sorted(devices):
        raise RuntimeError("starting records differ from the run devices")
    recs, excluded = b.split_finite(all_recs)
    print(f"A1: {len(excluded)} failed starting records excluded, scored as failed: {excluded}", flush=True)
    by_tag = {r.tag: r for r in recs}
    bins = load_bins(args)
    tags = sorted(by_tag)
    all_tags = sorted(r.tag for r in all_recs)
    opts = dict(HELDOUT_OPTIONS)
    hyper, hyper_short = {}, {}
    if args.smoke:
        opts.update(heldout_nfev=SMOKE["heldout_nfev"], heldout_irls=SMOKE["heldout_irls"])
        hyper, hyper_short = dict(SMOKE["bias_hyper"]), dict(SMOKE["short_hyper"])
    timing, corrs, extra = {}, {}, {}
    with get_context("spawn").Pool(args.workers, initializer=_init_recs,
                                   initargs=(str(records_path(args)),)) as pool:
        t0 = time.time()
        bias_tasks = [("M_bias", h) for h in tags] if "M_bias" in arms else []
        if "M_bias_lvt" in arms:
            bias_tasks.append(("M_bias_lvt", "__I8_1_DEV__"))
        expanded, owner = split_seed_tasks(bias_tasks, hyper)
        tasks = [("bias_ext" if a == "M_bias_lvt" else "bias", "M_bias", h, hp) for a, h, hp in expanded]
        other = []
        if "T1_inloop" in arms:
            other += [("t1", "T1", h, hyper_short) for h in tags]
        if "B3_d3" in arms:
            other += [("poly", "B3_d3", h, hyper_short) for h in tags]
        results = pool.map(train_task, tasks + other, chunksize=1)
        merged = merge_seed_results(len(bias_tasks), owner, results[:len(tasks)])
        for (arm, h), c in zip(bias_tasks, merged):
            corrs[f"{arm}|{h}"] = c
        rest = iter(results[len(tasks):])
        for arm in ("T1_inloop", "B3_d3"):
            if arm in arms:
                for h in tags:
                    corrs[f"{arm}|{h}"] = next(rest)
        timing["train_s"] = round(time.time() - t0, 1)
        timing["train_n"] = len(tasks + other)
        if "M_bias_lvt" in arms:
            extra["M_bias_lvt"] = {"training_records": str(I8_1_RECORDS.relative_to(ROOT)),
                                   "records_sha256": {s: b.sha256(I8_1_RECORDS.with_suffix(s))
                                                      for s in (".json", ".npz")},
                                   "n_training_records": len(_ext_records())}
        print(f"training done ({timing['train_s']} s, {timing['train_n']} tasks)", flush=True)
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
                elif arm == "M_bias_lvt":
                    refits.append((arm, "corr", {"corr": corrs["M_bias_lvt|__I8_1_DEV__"]}, rec, dev,
                                   binfo, opts))
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
    scores["C0"] = {h: float(score_device(by_tag[h], [c.sim for c in by_tag[h].curves])["rrms"]) for h in tags}
    for a in scores:
        for t in excluded:
            scores[a][t] = float("nan")
    if name == "baselines":
        lodo = json.loads((odir(args, "lodo") / "results.json").read_text())
        for a in ("C0_refit", "M_bias"):
            scores[a] = {t: nan_scores(lodo["per_device"])[a][t] for t in all_tags}
        pairs = [("M_bias", x) for x in arms] + [(x, "C0_refit") for x in arms]
    else:
        pairs = [("M_bias", "C0_refit"), ("M_bias", "T1_inloop"), ("T1_inloop", "C0_refit"),
                 ("C0_refit", "C0"), ("M_bias_lvt", "C0_refit"), ("M_bias_lvt", "M_bias")]
    tags = all_tags
    cov = [t for t in tags if bins[t]["covered"]]
    fam = family_of(tags)
    result = {"name": odir(args, name).name, "excluded_records": excluded,
              "prereg": "out/research/explore/I9_1/prereg.json", "smoke": bool(args.smoke),
              "prediction_guard": guard, "options": opts,
              "hyper_overrides": {"bias": hyper, "short": hyper_short}, "arms": list(arms),
              "held_out": tags, "covered": cov, "summary": summaries(scores, tags, bins),
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
    b.write_json(target / "results.json", result)
    (target / "corrections.json").write_text(json.dumps({k: c.to_json() for k, c in corrs.items()}) + "\n")
    b.write_json(target / "cards.json", {a: {t: r["params"] for t, r in v.items()} for a, v in rows.items()})
    print(f"{name}: timing {timing} runtime {result['runtime_s']} s; failed per arm "
          f"{ {k: v['n_failed'] for k, v in result['summary']['all'].items()} }")
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


def ci_label(cmp: dict) -> str:
    """CI reading of a GMR(A, B): 'A lower' (upper < 1), 'A higher' (lower > 1),
    'CI contains 1'."""
    lo, hi = cmp.get("ci_low", float("nan")), cmp.get("ci_high", float("nan"))
    return "A lower" if hi < 1 else ("A higher" if lo > 1 else "CI contains 1")


def run_decide(args) -> int:
    from cryoml.explore.i3_2_bias import paired_ratio
    all_recs = load_records(args)
    recs, excluded = b.split_finite(all_recs)
    all_tags = sorted(r.tag for r in all_recs)
    bins = load_bins(args)
    pred = json.loads(prediction_path(args).read_text())
    lodo = json.loads((odir(args, "lodo") / "results.json").read_text())
    start = json.loads((odir(args, "start") / "summary.json").read_text())
    base_path = odir(args, "baselines") / "results.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else None
    rows1 = s1_rows(all_recs, family_of(all_tags))
    good = sorted(r.tag for r in recs)
    s1 = float(np.mean([rows1[t]["S1_lodo"] for t in good]))
    checks = {"S1_abs_diff": abs(s1 - pred["dataset"]["S1_lodo"]),
              "excluded_equal": (excluded == pred["excluded_records"] == lodo["excluded_records"]
                                 and (base is None or base["excluded_records"] == excluded))}
    if checks["S1_abs_diff"] >= 1e-12 or not checks["excluded_equal"]:
        raise RuntimeError(f"decide checks failed: {checks}")
    sc = nan_scores(lodo["per_device"])
    arm_rows = [lodo["rows"]] + ([base["rows"]] if base else [])
    timeout_tags = sorted({t for rows in arm_rows for v in rows.values() for t, r in v.items()
                           if r.get("n_sim_timeout", 0) > 0})
    scored = [t for t in all_tags if t not in timeout_tags]

    def y_of(sel, a="M_bias", c="C0_refit"):
        lr = [math.log(sc[a][t] / sc[c][t]) for t in sel
              if np.isfinite(sc[a][t]) and np.isfinite(sc[c][t]) and sc[a][t] > 0 and sc[c][t] > 0]
        return float(np.mean(lr)) if lr else float("nan"), len(lr)

    def sub(arm, sel, s=None):
        s = sc if s is None else s
        return {t: s[arm][t] for t in sel}

    def vmean(arm, sel):
        x = np.array([sc[arm][t] for t in sel], float)
        return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")

    fam = family_of(all_tags)
    y_d, n_pair = y_of(scored)
    sign = lambda x: "gain" if x < 0 else "no gain"
    cmp = paired_ratio(sub("M_bias", scored), sub("C0_refit", scored))
    finite_scored = [t for t in scored if t not in excluded]
    r1 = cmp["ci_high"] < 1
    r2 = vmean("M_bias", finite_scored) <= vmean("T1_inloop", finite_scored) + T1_PLUS
    r3 = all(np.isfinite(sc["M_bias"][t]) for t in finite_scored)
    n = len(all_tags)
    v1 = sum(1 for t in all_tags if np.isfinite(start["rows"][t]["start_rrms"] or np.nan)
             and start["rows"][t]["finite_record"] and start["rows"][t]["n_curves"] >= 2) >= V1_MIN_FINITE * n
    n_fail = {a: sum(1 for t in scored if not np.isfinite(sc[a][t])) for a in ("C0_refit", "M_bias")}
    v2 = (all(x <= V2_MAX_FAILED * n for x in n_fail.values())
          and n_pair >= (math.ceil(V2_MIN_PAIRED * n) if not args.smoke else 1))
    g = lodo["prediction_guard"]
    v3 = bool(g.get("ok")) and not args.smoke
    valid = v1 and v2
    p_a = {"predicted": pred["P_a"]["predicted_sign"], "observed": sign(y_d),
           "correct": pred["P_a"]["predicted_sign"] == sign(y_d), "y_D": y_d,
           "predicted_y": pred["dataset"]["predicted_y"],
           "abs_error_y": abs(y_d - pred["dataset"]["predicted_y"])}
    p_c = {"predicted": pred["P_c"]["predicted_sign"], "observed": sign(y_d),
           "correct": pred["P_c"]["predicted_sign"] == sign(y_d), "S3_D": pred["dataset"]["S3"]}
    fam_out = {}
    for f in ("nmos", "pmos"):
        yf, nf = y_of([t for t in scored if fam[t] == f])
        pf = pred["per_family_secondary"][f]
        fam_out[f] = {"y": yf, "n": nf, "observed": sign(yf),
                      "rule_correct": pf["P_a_predicted_sign"] == sign(yf),
                      "S3_correct": pf["P_c_predicted_sign"] == sign(yf)}
    gates = {}
    for gname, choice in pred["gates"].items():
        ln_c0 = y_d if choice == "M_bias" else 0.0
        gates[gname] = {"choice": choice, "gmr_vs_C0_refit": math.exp(ln_c0),
                        "gmr_vs_M_bias": math.exp(ln_c0 - y_d), "chose_lower": (choice == "M_bias") == (y_d < 0)}
    cov = [t for t in scored if bins[t]["covered"]]
    transfer = None
    if "M_bias_lvt" in sc:
        t_c0 = paired_ratio(sub("M_bias_lvt", scored), sub("C0_refit", scored))
        t_mb = paired_ratio(sub("M_bias_lvt", scored), sub("M_bias", scored))
        transfer = {"M_bias_lvt_vs_C0_refit": {**t_c0, "reading": ci_label(t_c0)},
                    "M_bias_lvt_vs_M_bias": {**t_mb, "reading": ci_label(t_mb)}}
    base_cmp, b3 = None, None
    if base:
        bsc = nan_scores(base["per_device"])
        base_cmp = {f"M_bias_vs_{x}": paired_ratio(sub("M_bias", scored), sub(x, scored, bsc))
                    for x in BASELINE_ARMS}
        base_cmp.update({f"{x}_vs_C0_refit": paired_ratio(sub(x, scored, bsc), sub("C0_refit", scored))
                         for x in BASELINE_ARMS})
        mb3 = base_cmp["M_bias_vs_B3_d3"]
        b3c0 = base_cmp["B3_d3_vs_C0_refit"]
        b3 = {"GMR_M_bias_vs_B3_d3": mb3, "reading": ci_label(mb3),
              "log_gain_fraction_B3_d3": (math.log(b3c0["ratio"]) / y_d
                                          if np.isfinite(y_d) and y_d != 0 else float("nan"))}
    out = {"id": "I9_1_decision", "smoke": bool(args.smoke), "prereg": "out/research/explore/I9_1/prereg.json",
           "prediction_commit": g.get("commit"), "checks": checks, "excluded_start_failures": excluded,
           "dropped_timeouts_V4": timeout_tags,
           "validity": {"V1": v1, "V2": v2, "V3_prospective": v3, "V4_n_timeout_start": start["n_sim_timeout"],
                        "V4_dropped": timeout_tags, "valid": valid, "n_failed": n_fail, "n_paired": n_pair},
           "P_a": {**p_a, "stated": valid, "prospective": v3},
           "P_b": {"label": p_b_label(cmp, r2, r3) if valid else "invalid", "R1": r1, "R2": r2, "R3": r3,
                   "gmr": cmp, "means_valid_only": {a: vmean(a, finite_scored) for a in sc}},
           "P_c": {**p_c, "stated": valid, "prospective": v3},
           "secondary": {"M_bias_vs_B3_d3": b3, "transfer_M_bias_lvt": transfer,
                         "covered_subset": (paired_ratio(sub("M_bias", cov), sub("C0_refit", cov))
                                            if len(cov) < len(scored) else "all scored devices covered"),
                         "per_family": fam_out, "gates": gates, "baselines": base_cmp,
                         "bins_stretch": {t: [bins[t]["stretch_L"], bins[t]["stretch_W"]] for t in all_tags}},
           "rule": frozen_rule()}
    b.write_json(odir(args, "decision") / "decision.json" if args.smoke else OUT / "decision.json", out)
    print(json.dumps({k: out[k] for k in ("checks", "validity")}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("cards", "data", "split", "smoke", "start", "stats", "commit_prediction",
                             "lodo", "baselines", "decide"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"cards": run_cards, "data": run_data, "split": run_split, "smoke": run_smoke,
            "start": run_start, "stats": run_stats, "commit_prediction": run_commit_prediction,
            "lodo": run_lodo, "baselines": run_baselines, "decide": run_decide}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
