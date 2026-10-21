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

Run with ``python -I`` from the repository root:
  .venv/bin/python -I scripts/research/explore/i8_1_nist.py --mode inventory
"""
from __future__ import annotations

import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("sources", "inventory", "data", "split", "smoke"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"sources": run_sources, "inventory": run_inventory, "data": run_data,
            "split": run_split, "smoke": run_smoke}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
