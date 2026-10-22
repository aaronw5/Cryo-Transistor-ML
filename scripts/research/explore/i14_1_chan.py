#!/usr/bin/env python3
"""I14-1: deployment test (Part B) of the bias-only correction g v2 with the
channel-current replica, one BSIM4 model-check state (b4check.c clamp written
into every card) and the corrected Q2 reference; M_bias v1 full-data g as a
reference arm.  sky130 77 K, 18 devices.

Pre-registration: out/research/explore/I14_1/prereg.md (prereg.json).

Inputs (fixed, I13-1 state C): records I13_1/records, v1 rows and cards
I13_1/all18, v2 rows, fold corrections and held-out cards I13_1/lodo.

Stages (``--stage``):

  repcheck  replica pre-check on the published, start (state C), I13-1 v2
            held-out and I13-1 v1 held-out cards: forward grid (tight,
            17 digits) with the card's and the replica's BSIM4 outputs
  validity  V1: card-only library deck (clamp written) -> Python rows equal
            the I13-1 rows (v2 and v1, 18 devices)
  accuracy  element-deck RRMS of the held-out pairs (fold g + I13-1 held-out
            card) on the measured curves, v2 and v1; differences to the
            Python rows; channel-model Python rows from card + replica runs
  deploy    --arm v2|v1: compile (full-data g), M2 cards (refit in state C
            with the full-data g in the loop), q12, q3, report
  decide    validity + I13-1 accuracy + Part B (v2) -> decision.json

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i14_1_chan.py --stage repcheck --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "out/research/explore"
I13 = EXP / "I13_1"
# I13-1 state-C inputs (set before the I12-1 driver and i3_2_bias are imported)
os.environ["CRYOML_E1_RECORDS"] = str(I13 / "records")
os.environ["CRYOML_I3_2_ALL18"] = str(I13 / "all18")

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i12_1_symm as drv12  # noqa: E402  (I12-1 driver: devices, records, refit)
from cryoml.explore import i11_1_deploy as m1  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i14_1_chan as m  # noqa: E402

OUT = EXP / "I14_1"
PREREG = OUT / "prereg.json"
ALL18 = I13 / "all18"
LODO = I13 / "lodo"
NG_CANON = drv12.NG_CANON
REFIT_OPTS = dict(drv12.REFIT_OPTS)
SMOKE_TAGS = ["nmos_L1_W1p6", "pmos_L2_W5"]

jload, jdump, devices, start_cards = drv12.jload, drv12.jdump, drv12.devices, drv12.start_cards


def load_records() -> dict:
    return drv12.load_records()


# ---------------------------------------------------------------------------
# corrections and cards (I13-1 state C)
# ---------------------------------------------------------------------------
def corr_full(arm: str):
    """Full-data correction as a SymCorrection (v2: sym_v2, v1: abs_v1)."""
    from cryoml.explore.i3_2_bias import correction_from_json
    if arm == "v2":
        return v2.SymCorrection.from_json(jload(LODO / "corrections.json")["full|M_bias_v2"])
    return m.v1_as_sym(correction_from_json(jload(ALL18 / "corrections.json")["full|M_bias"]))


def corr_fold(arm: str, tag: str):
    from cryoml.explore.i3_2_bias import correction_from_json
    if arm == "v2":
        return v2.SymCorrection.from_json(jload(LODO / "corrections.json")[f"M_bias_v2|{tag}"])
    return m.v1_as_sym(correction_from_json(jload(ALL18 / "corrections.json")[f"M_bias|{tag}"]))


def heldout_card(arm: str, tag: str) -> dict:
    if arm == "v2":
        return dict(jload(LODO / "heldout_cards.json")[tag])
    return dict(jload(ALL18 / "heldout_cards.json")["M_bias"][tag])


def python_rows(arm: str) -> dict:
    """I13-1 per-device rows (strict paper RRMS, Python model)."""
    r = jload(LODO / "results.json")["per_device"]
    return dict(r["M_bias_v2"] if arm == "v2" else r["M_bias_v1"])


# ---------------------------------------------------------------------------
# repcheck (pre-registration)
# ---------------------------------------------------------------------------
CARD_OUT = ("id", "isub", "ibd", "ibs", "igidl", "igisl")
REP_OUT = ("id",) + m.NONCHANNEL_OUTPUTS + ("ibd", "ibs")


def repcheck_task(task):
    tag, label, params = task
    from cryoml import spice_pdk as sp
    dev = devices()[tag]
    pm = dev.dev_type == "pmos"
    s = -1.0 if pm else 1.0
    sub = sp._SUBCKT[dev.dev_type]
    c, r = f"@m.xm1.m{sub}", f"@m.xm1r.m{sub}"
    vec = [f"{c}[{k}]" for k in CARD_OUT] + [f"{r}[{k}]" for k in REP_OUT]
    with m1.tempdir("cryoml_i14r_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, {dev.dev_type: [(dev.bin, params)]})
        f = td / "g.dat"
        ctrl = ["save all " + " ".join(vec),
                f"dc VD {m1.GRID_VDS[0]:.6g} {m1.GRID_VDS[-1]:.6g} {m1.STEP:.6g} "
                f"VG 0 {s * m1.GRID_VGS[-1]:.6g} {s * m1.STEP:.6g}",
                f"wrdata {f.as_posix()} -i(VD) i(Vm1R) " + " ".join(vec)]
        rr = m1.run_deck(m.deck("repcheck", lib, m.device_body(dev, "card_rep"), ctrl, None,
                                options=m1.TIGHT), td, "r.sp", binary=NG_CANON, timeout_s=300)
        ncol = 2 * (2 + len(vec))
        rows = m1.read_wrdata(f, ncol)
        rm = m1.run_deck(m.deck("mcheck", lib, m.device_body(dev, "card_rep"),
                                ["op", f"print {c}[m] {r}[m]"], None), td, "mc.sp", binary=NG_CANON)
        mvals = [float(x.split("=")[1]) for x in rm["text"].splitlines() if "[m] =" in x]
        warn = sorted({ln.strip() for ln in (rr["text"] + rm["text"]).splitlines()
                       if "unrecognized" in ln.lower() or "unknown parameter" in ln.lower()})
    n = len(m1.GRID_VGS) * len(m1.GRID_VDS)
    out = {"tag": tag, "label": label, "msgs": rr["msgs"], "n_rows": int(rows.shape[0]),
           "bsim4_m": mvals, "param_warnings": warn}
    if rows.shape[0] != n:
        out["failed"] = True
        return out
    cols = {name: rows[:, 2 * k + 1] for k, name in enumerate(
        ["icard", "isense"] + [f"card_{k}" for k in CARD_OUT] + [f"rep_{k}" for k in REP_OUT])}
    vds = rows[:, 0]
    vgs = np.repeat(s * m1.GRID_VGS, len(m1.GRID_VDS))
    mult = sp.device_mult(dev.dev_type, dev.L_um, dev.W_um)
    ich_card = m.drain_channel_current(cols["card_id"], vds, pm)
    ich_rep = m.drain_channel_current(cols["rep_id"], vds, pm)
    fwd = (s * vds) > 0
    meas = fwd & (np.abs(vgs) <= 1.85 + 1e-9)
    big = np.abs(ich_card) > m1.I_FLOOR
    nonch_card = cols["icard"] - ich_card
    out.update({
        "mult": mult,
        # P1: replica non-channel outputs exactly zero; junction current = gmin x V
        "P1_rep_nonchannel_max": {k: float(np.max(np.abs(cols[f"rep_{k}"]))) for k in m.NONCHANNEL_OUTPUTS},
        # BSIM4 ibd = m gbd vbd, vbd = type (V(b) - V(d)); gbd = gmin with zero saturation current
        "P1_rep_ibd_minus_gmin_vbd_max": float(np.max(np.abs(
            cols["rep_ibd"] - m.M_DEV * m.GMIN * (-1.0 if pm else 1.0) * (0.0 - vds)))),
        # P2: sensed current vs the replica's own channel current
        "P2_sense_vs_rep_channel": m.rel_stats(cols["isense"], ich_rep, np.abs(ich_rep) > m1.I_FLOOR),
        "P2_sense_vs_rep_channel_all_max_abs": float(np.max(np.abs(cols["isense"] - ich_rep))),
        # the I12-1 per-point criterion |dI| <= 1e-6 |I| + 1e-13 A (all points)
        "P2_eq_max_ratio": v2.eq_violation(cols["isense"], ich_rep)["max_ratio"],
        # P3: replica channel vs card channel (BSIM4 outputs), and sensed vs card channel
        "P3_rep_id_vs_card_id_fwd": m.rel_stats(ich_rep, ich_card, meas & big),
        "P3_rep_id_vs_card_id_rev": m.rel_stats(ich_rep, ich_card, (~fwd) & big),
        "P3_sense_vs_card_channel_fwd": m.rel_stats(cols["isense"], ich_card, meas & big),
        "P3_sense_vs_card_channel_rev": m.rel_stats(cols["isense"], ich_card, (~fwd) & big),
        # card non-channel current at the drain terminal (forward quadrant)
        "card_nonchannel_fwd_max_abs": float(np.max(np.abs(nonch_card[meas]))),
        "card_nonchannel_fwd_max_rel": float(np.max(np.abs(nonch_card[meas & big])
                                                    / np.abs(ich_card[meas & big]))) if (meas & big).any() else 0.0,
        "card_isub_max": float(np.max(np.abs(cols["card_isub"]))),
    })
    # P3 restricted to forward points where the card's |isub| < 1e-15 A
    low = meas & big & (np.abs(cols["card_isub"]) < 1e-15)
    out["P3_rep_id_vs_card_id_fwd_isub_lt_1e-15"] = m.rel_stats(ich_rep, ich_card, low)
    return out


def stage_repcheck(workers: int, tags=None) -> dict:
    tags = tags or sorted(devices())
    sc = start_cards()
    sets = {"published": {t: None for t in tags}, "start_C": {t: sc[t] for t in tags},
            "v2_heldout_C": {t: heldout_card("v2", t) for t in tags},
            "v1_heldout_C": {t: heldout_card("v1", t) for t in tags}}
    tasks = [(t, lab, cards[t]) for lab, cards in sets.items() for t in tags]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(repcheck_task, tasks, chunksize=1)
    res = {"wall_s": round(time.time() - t0, 1), "binary": NG_CANON, "options": m1.TIGHT,
           "replica": m.REPLICA_CHAN, "gmin": m.GMIN, "rows": {f"{r['label']}|{r['tag']}": r for r in rows}}
    agg = {}
    for lab in sets:
        rs = [r for r in rows if r["label"] == lab and not r.get("failed")]
        agg[lab] = {
            "n": len(rs), "failed": sum(1 for r in rows if r["label"] == lab and r.get("failed")),
            "msgs": sum(len(r["msgs"]) for r in rows if r["label"] == lab),
            "bsim4_m_values": sorted({v for r in rs for v in r["bsim4_m"]}),
            "param_warnings": sorted({w for r in rs for w in r["param_warnings"]}),
            "P1_rep_nonchannel_max": max(max(r["P1_rep_nonchannel_max"].values()) for r in rs),
            "P1_rep_ibd_minus_gmin_vbd_max": max(r["P1_rep_ibd_minus_gmin_vbd_max"] for r in rs),
            "P2_max_rel": max(r["P2_sense_vs_rep_channel"]["max_rel"] for r in rs),
            "P2_max_abs": max(r["P2_sense_vs_rep_channel_all_max_abs"] for r in rs),
            "P2_eq_max_ratio": max(r["P2_eq_max_ratio"] for r in rs),
            "card_nonchannel_fwd_max_rel": max(r["card_nonchannel_fwd_max_rel"] for r in rs),
            "P3_rep_id_fwd_max_rel": max(r["P3_rep_id_vs_card_id_fwd"]["max_rel"] for r in rs),
            "P3_rep_id_fwd_isub_lt_1e-15_max_rel": max(
                r["P3_rep_id_vs_card_id_fwd_isub_lt_1e-15"]["max_rel"] for r in rs),
            "P3_rep_id_rev_max_rel": max(r["P3_rep_id_vs_card_id_rev"]["max_rel"] for r in rs),
            "P3_sense_fwd_max_rel": max(r["P3_sense_vs_card_channel_fwd"]["max_rel"] for r in rs),
            "devices_P3_rep_id_fwd_le_1e-12": sorted(
                r["tag"] for r in rs if r["P3_rep_id_vs_card_id_fwd"]["max_rel"] <= 1e-12),
        }
    res["summary"] = agg
    jdump(res, OUT / "repcheck" / "repcheck.json")
    print(json.dumps(agg, indent=1, default=float), flush=True)
    return res


# ---------------------------------------------------------------------------
# measured-curve decks (validity, accuracy, M2 native)
# ---------------------------------------------------------------------------
def _sense_curves(td: Path, curves, dev_type: str, fatal: bool) -> list[np.ndarray]:
    """Columns 3-4 (i(Vm1R)) of raw_<i>.dat resampled as ``_read_curves``
    (pMOS sign flipped)."""
    out = []
    for idx, c in enumerate(curves):
        target = np.asarray(c.Vg if c.kind == "idvg" else c.Vd, dtype=np.float64)
        p = td / f"raw_{idx}.dat"
        rows = m1.read_wrdata(p, 4) if (p.exists() and not fatal) else np.zeros((0, 4))
        if rows.shape[0] == 0:
            out.append(np.full_like(target, np.nan))
            continue
        o = np.argsort(rows[:, 0])
        v = np.interp(target, rows[o, 0], rows[o, 3])
        out.append(-v if dev_type == "pmos" else v)
    return out


def measured_run(tag: str, params: dict, mode: str, osdi: Path | None = None,
                 module: str = v2.VA_MODULE_V2) -> dict:
    """The measured-curve batch deck of ``tag`` with ``params`` (clamp
    written) in mode card / card_rep / elem on the OSDI ngspice-41 build.
    Returns the curve currents in record order (and the sensed replica
    currents for card_rep) and the convergence messages."""
    from cryoml import spice_pdk as sp
    from cryoml.card_eval import device_by_tag
    from cryoml.data_io import load_device_curves
    from cryoml.metrics import tag_metric_curves
    dev = devices()[tag]
    curves = load_device_curves(device_by_tag(tag))
    pos = {id(c): i for i, c in enumerate(curves)}
    order = [pos[id(c)] for _, c in sorted(tag_metric_curves(curves).items())]
    with m1.tempdir("cryoml_i14m_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, {dev.dev_type: [(dev.bin, params)]})
        deck = sp._batch_deck(dev.dev_type, dev.L_um, dev.W_um, curves, lib, td, 77.0)
        if mode == "elem":
            deck = m.insert_element(deck, osdi, dev, module)
        elif mode == "card_rep":
            lines = deck.splitlines(keepends=True)
            k = [i for i, ln in enumerate(lines) if ln.startswith("Xm1 nd ng 0 0 ")]
            if len(k) != 1:
                raise ValueError("instance line not found once")
            lines[k[0]] = m.instance("m1", dev, "nd", "ng", "0", "0", "card_rep")
            deck = "".join(lines).replace(" -i(VD)\n", " -i(VD) i(Vm1R)\n")
        (td / "d.sp").write_text(deck)
        proc = subprocess.run([m1.NG_OSDI, "-b", str(td / "d.sp")], cwd=str(td),
                              capture_output=True, text=True, timeout=300, check=False)
        text = proc.stdout + proc.stderr
        fatal = "could not find a valid modelname" in text.lower()
        sims = sp._read_curves(td, "raw_", curves, dev.dev_type, fatal)
        sense = _sense_curves(td, curves, dev.dev_type, fatal) if mode == "card_rep" else None
    out = {"sims": [np.asarray(sims[i], float) for i in order], "msgs": m1.convergence_messages(text)}
    if sense is not None:
        out["sense"] = [np.asarray(sense[i], float) for i in order]
    return out


def validity_task(task):
    arm, tag = task
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i2_1_joint import with_sims
    rec = load_records()[tag]
    corr = corr_fold(arm, tag)
    run = measured_run(tag, heldout_card(arm, tag), "card")
    py = corr.apply(with_sims(rec, run["sims"]))
    return {"arm": arm, "tag": tag, "rrms": float(score_device(rec, py)["rrms"]), "msgs": run["msgs"]}


UNIT_TESTS = ("tests.test_i14_1_chan", "tests.test_i13_1_clamp", "tests.test_spice_batch",
              "tests.test_sky130_misspec", "tests.test_i12_1_symm")


def run_unit_tests(rd: Path) -> dict:
    """V2: the clamp and element unit tests (unittest, one process)."""
    proc = subprocess.run([sys.executable, "-m", "unittest", *UNIT_TESTS], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=3600, check=False,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
    out = {"files": list(UNIT_TESTS), "returncode": proc.returncode, "passed": proc.returncode == 0,
           "tail": tail}
    jdump(out, rd / "unit_tests.json")
    print(f"unit tests: rc {proc.returncode}; {tail[-1] if tail else ''}", flush=True)
    return out


def stage_validity(workers: int, rd: Path, tags) -> dict:
    run_unit_tests(rd)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(validity_task, [(a, t) for a in ("v2", "v1") for t in tags], chunksize=1)
    out = {"rows": {}, "wall_s": None}
    worst = 0.0
    for r in rows:
        stored = python_rows(r["arm"])[r["tag"]]
        d = abs(r["rrms"] - stored)
        worst = max(worst, d)
        out["rows"][f"{r['arm']}|{r['tag']}"] = {**r, "stored": stored, "abs_diff": d}
    out["v1_max_rrms_diff"] = worst
    out["n_rows"] = {a: sum(1 for r in rows if r["arm"] == a) for a in ("v2", "v1")}
    out["wall_s"] = round(time.time() - t0, 1)
    jdump(out, rd / "validity.json")
    print(f"validity: max |RRMS card-only library deck - I13-1 row| {worst:.3e} "
          f"({len(rows)} rows, {out['wall_s']} s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# accuracy (element deck, held-out pairs)
# ---------------------------------------------------------------------------
def fold_osdi(rd: Path, arm: str, tag: str) -> Path:
    return rd / "osdi" / f"{arm}_{tag}.osdi"


def compile_task(task):
    rd_s, arm, tag = task
    rd = Path(rd_s)
    corr = corr_full(arm) if tag == "full" else corr_fold(arm, tag)
    text = m.to_verilog_a(corr)
    osdi = fold_osdi(rd, arm, tag)
    c = m1.compile_osdi(text, osdi.with_suffix(".va"), osdi)
    return {"arm": arm, "tag": tag, "sha256": c["sha256"], "compiled": bool(c["compiled"])}


def accuracy_task(task):
    rd_s, arm, tag = task
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i2_1_joint import with_sims
    rec = load_records()[tag]
    corr = corr_fold(arm, tag)
    card = heldout_card(arm, tag)
    dev = devices()[tag]
    cr = measured_run(tag, card, "card_rep")
    el = measured_run(tag, card, "elem", fold_osdi(Path(rd_s), arm, tag), m.module_name(corr))
    pm = dev.dev_type == "pmos"
    py = corr.apply(with_sims(rec, cr["sims"]))                      # I_card exp(g)
    chan = [m.current_fwd(ic, ir, corr.g_v(pm, c.vg, c.vd))          # I_card + (exp(g)-1) I_rep
            for ic, ir, c in zip(cr["sims"], cr["sense"], rec.curves)]
    eq = v2.eq_violation(np.concatenate(el["sims"]), np.concatenate(chan), rel=1e-3, absol=1e-12)
    # non-channel share of the card current at the measured points
    nonch = [np.abs(ic - ir) / np.maximum(np.abs(ic), 1e-30) for ic, ir in zip(cr["sims"], cr["sense"])]
    return {"arm": arm, "tag": tag,
            "rrms_native": float(score_device(rec, el["sims"])["rrms"]),
            "rrms_python": float(score_device(rec, py)["rrms"]),
            "rrms_python_channel": float(score_device(rec, chan)["rrms"]),
            "rrms_card_only": float(score_device(rec, cr["sims"])["rrms"]),
            "native_vs_channel_max_ratio_1e-3_1e-12": eq["max_ratio"],
            "card_nonchannel_share_max": float(max(np.nanmax(x) for x in nonch)),
            "msgs": cr["msgs"] + el["msgs"]}


def stage_accuracy(workers: int, rd: Path, tags) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    from e1_lodo import aggregate
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        comp = pool.map(compile_task, [(str(rd), a, t) for a in ("v2", "v1") for t in tags], chunksize=1)
        t_comp = time.time() - t0
        rows = pool.map(accuracy_task, [(str(rd), a, t) for a in ("v2", "v1") for t in tags], chunksize=1)
    out = {"compile": comp, "compile_s": round(t_comp, 1), "rows": {}, "per_arm": {}}
    for a in ("v2", "v1"):
        rr = {r["tag"]: r for r in rows if r["arm"] == a}
        stored = python_rows(a)
        for t, r in rr.items():
            r["stored_python"] = stored[t]
            r["native_minus_stored"] = r["rrms_native"] - stored[t]
            out["rows"][f"{a}|{t}"] = r
        nat = {t: r["rrms_native"] for t, r in rr.items()}
        pyr = {t: r["rrms_python"] for t, r in rr.items()}
        out["per_arm"][a] = {
            "native": nat, "python_rerun": pyr,
            "python_channel": {t: r["rrms_python_channel"] for t, r in rr.items()},
            "max_abs_native_minus_stored": max(abs(r["native_minus_stored"]) for r in rr.values()),
            "max_abs_python_rerun_minus_stored": max(abs(r["rrms_python"] - r["stored_python"])
                                                     for r in rr.values()),
            "max_native_vs_channel_ratio": max(r["native_vs_channel_max_ratio_1e-3_1e-12"]
                                               for r in rr.values()),
            "n_msgs": sum(len(r["msgs"]) for r in rr.values())}
        if len(rr) == 18:
            out["per_arm"][a]["summary_native"] = aggregate(nat)
            out["per_arm"][a]["summary_stored"] = aggregate({t: stored[t] for t in rr})
    if all(len([r for r in rows if r["arm"] == a]) == 18 for a in ("v2", "v1")):
        c0 = jload(LODO / "results.json")["per_device"]["C0_refit"]
        v1s = python_rows("v1")
        nv2 = out["per_arm"]["v2"]["native"]
        nv1 = out["per_arm"]["v1"]["native"]
        out["gmr_native"] = {
            "v2_native_vs_C0_refit": paired_ratio(nv2, c0),
            "v2_native_vs_v1_stored": paired_ratio(nv2, v1s),
            "v2_native_vs_v1_native": paired_ratio(nv2, nv1),
            "v1_native_vs_C0_refit": paired_ratio(nv1, c0)}
    out["wall_s"] = round(time.time() - t0, 1)
    jdump(out, rd / "accuracy.json")
    print(json.dumps({a: {k: v for k, v in out["per_arm"][a].items()
                          if k not in ("native", "python_rerun", "python_channel")}
                      for a in out["per_arm"]}, indent=1, default=float), flush=True)
    return out


# ---------------------------------------------------------------------------
# deploy: compile, M2, q12, q3, report (one arm)
# ---------------------------------------------------------------------------
def arm_osdi(rd: Path, arm: str) -> Path:
    return rd / "osdi" / f"{arm}_full.osdi"


def stage_compile(rd: Path, arm: str) -> dict:
    corr = corr_full(arm)
    text = m.to_verilog_a(corr)
    c = m1.compile_osdi(text, arm_osdi(rd, arm).with_suffix(".va"), arm_osdi(rd, arm))
    rng = np.random.default_rng(20261021)
    worst, worst_sym = 0.0, 0.0
    for _ in range(2000):
        p = int(rng.integers(0, 2))
        vgs, vds = rng.uniform(-1.85, 1.85, size=2)
        gt = m.eval_verilog_a(text, p, vgs, vds)
        gp = float(corr.g_v(bool(p), vgs, vds))
        gs = m.eval_verilog_a(text, p, vgs - vds, -vds)
        worst = max(worst, abs(gt - gp))
        worst_sym = max(worst_sym, abs(gs - gt))
    out = {"arm": arm, "module": m.module_name(corr), "sha256": c["sha256"],
           "compiled_now": bool(c["compiled"]), "lines": len(text.splitlines()),
           "g_text_max": worst, "g_sym_text_max": worst_sym, "n_points": 2000}
    jdump(out, rd / "compile.json")
    print(f"compile {arm}: |g text - g python| {worst:.2e}, |g exchanged - forward| {worst_sym:.2e}",
          flush=True)
    return out


def m2_task(task):
    tag, opts, arm = task
    from cryoml.explore.i3_2_bias import correction_from_json
    corr = (v2.SymCorrection.from_json(jload(LODO / "corrections.json")["full|M_bias_v2"]) if arm == "v2"
            else correction_from_json(jload(ALL18 / "corrections.json")["full|M_bias"]))
    return drv12.heldout(("M2", None, corr, load_records()[tag], opts))


def native_m2_task(task):
    tag, params, rd_s, arm = task
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i2_1_joint import with_sims
    rec = load_records()[tag]
    corr = corr_full(arm)
    pm = devices()[tag].dev_type == "pmos"
    cr = measured_run(tag, params, "card_rep")
    el = measured_run(tag, params, "elem", arm_osdi(Path(rd_s), arm), m.module_name(corr))
    py = corr.apply(with_sims(rec, cr["sims"]))
    chan = [m.current_fwd(ic, ir, corr.g_v(pm, c.vg, c.vd))
            for ic, ir, c in zip(cr["sims"], cr["sense"], rec.curves)]
    return {"tag": tag, "rrms_card_only": float(score_device(rec, cr["sims"])["rrms"]),
            "rrms_python": float(score_device(rec, py)["rrms"]),
            "rrms_python_channel": float(score_device(rec, chan)["rrms"]),
            "rrms_native": float(score_device(rec, el["sims"])["rrms"]),
            "msgs": cr["msgs"] + el["msgs"]}


def stage_refit(rd: Path, arm: str, tags, workers: int, opts: dict) -> dict:
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(m2_task, [(t, opts, arm) for t in tags], chunksize=1)
        t_refit = time.time() - t0
        cards = {r["tag"]: r["params"] for r in rows}
        jdump(cards, rd / "cards_M2.json")
        nat = pool.map(native_m2_task, [(t, cards[t], str(rd), arm) for t in tags], chunksize=1)
    out = {"arm": arm, "opts": opts, "refit_s": round(t_refit, 1),
           "rows": {r["tag"]: {k: r[k] for k in ("rrms_start", "rrms", "failed", "n_sim", "runtime_s")}
                    for r in rows},
           "native": {r["tag"]: r for r in nat}}
    jdump(out, rd / "refit.json")
    print(f"refit M2 ({arm}): {len(rows)} devices, {t_refit:.0f} s, in-sample mean "
          f"{np.mean([r['rrms'] for r in rows]):.4f}, native mean "
          f"{np.mean([v['rrms_native'] for v in nat]):.4f}", flush=True)
    return out


def q12_task(task):
    tag, params, rd_s, arm = task
    rd = Path(rd_s)
    dev = devices()[tag]
    corr = corr_full(arm)
    module = m.module_name(corr)
    osdi = arm_osdi(rd, arm)
    res = {"tag": tag, "msgs": {}, "wall": {}}
    a = {}
    with m1.tempdir("cryoml_i14q_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, {dev.dev_type: [(dev.bin, params)]})

        def go(label, body, control, el, options="", repeats=1):
            walls, r = [], None
            for _ in range(repeats):
                r = m1.run_deck(m.deck(label, lib, body, control, osdi if el else None, module, options),
                                td, f"{label}.sp", timeout_s=m1.TIMEOUT_DEVICE_S)
                walls.append(r["wall_s"])
            res["msgs"][label] = r["msgs"]
            res["wall"][label] = walls
            return r

        # runtime reference: card only (no replica), default tolerance, 3 repeats
        f = td / "grid_fwd_default_card.dat"
        go("grid_fwd_default_card", m.device_body(dev, "card"), m.grid_control(dev, f, False, False),
           False, "", repeats=3)
        a["grid_fwd_default_card"] = m1.grid_from_rows(m1.read_wrdata(f))
        for mode in ("card_rep", "elem"):
            el = mode == "elem"
            sense = mode == "card_rep"
            for tol, opt in (("default", ""), ("r4", m1.R4), ("tight", m1.TIGHT)):
                lab = f"grid_fwd_{tol}_{mode}"
                f = td / f"{lab}.dat"
                go(lab, m.device_body(dev, mode, module=module), m.grid_control(dev, f, False, sense),
                   el, opt, repeats=3 if (tol == "default" and el) else 1)
                rows = m1.read_wrdata(f, 4 if sense else 2)
                a[lab] = m1.grid_from_rows(rows[:, :2])
                if sense:
                    a[lab + "_rep"] = m1.grid_from_rows(rows[:, 2:4])
            lab = f"grid_swap_tight_{mode}"
            f = td / f"{lab}.dat"
            go(lab, m.device_body(dev, mode, swap=True, module=module), m.grid_control(dev, f, True, sense),
               el, m1.TIGHT)
            rows = m1.read_wrdata(f, 4 if sense else 2)
            a[lab] = m1.grid_from_rows(rows[:, :2])
            if sense:
                a[lab + "_rep"] = m1.grid_from_rows(rows[:, 2:4])
        pts = m1.op_points(dev)
        for mode in ("card_rep", "elem"):
            for ex in ("VG", "VD"):
                lab = f"ac_{ex}_{mode}"
                r = go(lab, m.device_body(dev, mode, module=module),
                       m.ac_control(pts, ex, mode == "card_rep"), mode == "elem", m1.TIGHT)
                for k, v in m.parse_ac(r["text"], len(pts)).items():
                    a[f"{lab}_{k}"] = v
        for mode in ("card", "elem"):
            lab = f"dense_{mode}"
            pre = td / lab
            go(lab, m.device_body(dev, mode, module=module), drv12.dense_control(dev, pre),
               mode == "elem", m1.TIGHT)
            for k in range(len(m1.GM_VDS)):
                a[f"{lab}_gm{k}"] = m1.read_wrdata(Path(f"{pre}_gm{k}.dat"))
            for k in range(len(m1.GDS_VGS)):
                a[f"{lab}_gds{k}"] = m1.read_wrdata(Path(f"{pre}_gds{k}.dat"))
    res.update(q12_metrics(dev, corr, pts, a))
    (rd / "arrays").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rd / "arrays" / f"q12_{tag}.npz", pts=pts,
                        **{k: v for k, v in a.items() if isinstance(v, np.ndarray)})
    return res


def q12_metrics(dev, corr, pts, a) -> dict:
    pm = dev.dev_type == "pmos"
    s = -1.0 if pm else 1.0
    VG, VD = np.meshgrid(s * m1.GRID_VGS, m1.GRID_VDS, indexing="ij")
    g_fwd = v2.g_terminal(corr, pm, VG, VD, 0.0)
    g_sw = v2.g_terminal(corr, pm, VG, 0.0, VD)          # module d at nz (0 V), s at nd
    out = {"g_sym_grid_max": float(np.max(np.abs(g_sw - g_fwd)))}
    # ---- Q1 equivalence (reference from the card + replica run of the same tolerance)
    for tol in ("default", "r4", "tight"):
        ref = m.current_fwd(a[f"grid_fwd_{tol}_card_rep"], a[f"grid_fwd_{tol}_card_rep_rep"], g_fwd)
        eq = v2.eq_violation(a[f"grid_fwd_{tol}_elem"], ref)
        out[f"q1_{tol}_fwd_viol"] = eq["n_viol"]
        out[f"q1_{tol}_fwd_max_ratio"] = eq["max_ratio"]
        out[f"q1_{tol}_max_rel_1nA"] = m1.nanmax0(m1.rel_diff(a[f"grid_fwd_{tol}_elem"], ref, m1.I_FLOOR))
    ref_sw = m.current_swap(a["grid_swap_tight_card_rep"], a["grid_swap_tight_card_rep_rep"], g_sw)
    eq = v2.eq_violation(a["grid_swap_tight_elem"], ref_sw)
    out["q1_tight_swap_viol"] = eq["n_viol"]
    out["q1_tight_swap_max_ratio"] = eq["max_ratio"]
    out["q1_tight_viol"] = int(out["q1_tight_fwd_viol"] + out["q1_tight_swap_viol"])
    elem_runs = [k for k in a if k.startswith("grid_") and k.endswith("_elem")]
    out["q1_nonfinite"] = int(sum(np.sum(~np.isfinite(a[k])) for k in elem_runs))
    out["q1_nonfinite_card"] = int(sum(np.sum(~np.isfinite(a[k])) for k in a
                                       if k.startswith("grid_") and ("_card" in k)))
    # card + replica vs card only (default tolerance): the replica draws no circuit current
    out["card_rep_vs_card_default_max_abs"] = float(np.nanmax(np.abs(
        a["grid_fwd_default_card_rep"] - a["grid_fwd_default_card"])))
    j0 = int(np.argmin(np.abs(m1.GRID_VDS)))
    out["q1_vds0_max_corr"] = float(max(np.nanmax(np.abs(a[f"grid_fwd_{t}_elem"][:, j0]))
                                        for t in ("default", "tight")))
    out["q1_vds0_max_card"] = float(max(np.nanmax(np.abs(a[f"grid_fwd_{t}_card_rep"][:, j0]))
                                        for t in ("default", "tight")))
    out["g_grid_min"], out["g_grid_max"] = float(np.min(g_fwd)), float(np.max(g_fwd))
    # replica share of the card current in the forward measured quadrant (tight)
    fw = (s * VD) > 0
    cc, rp = a["grid_fwd_tight_card_rep"], a["grid_fwd_tight_card_rep_rep"]
    big = fw & (np.abs(cc) > m1.I_FLOOR) & np.isfinite(cc) & np.isfinite(rp)
    out["nonchannel_share_fwd_max"] = float(np.max(np.abs(cc - rp)[big] / np.abs(cc[big]))) if big.any() else 0.0
    # exchange asymmetry (forward operation, |I| > 1 nA, tight)
    ic, isw = a["grid_fwd_tight_elem"], a["grid_swap_tight_elem"]
    csw = a["grid_swap_tight_card_rep"]
    keep = (fw & (np.abs(ic) > m1.I_FLOOR) & (np.abs(cc) > m1.I_FLOOR)
            & np.isfinite(isw) & np.isfinite(csw) & np.isfinite(ic) & np.isfinite(cc))
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_corr = np.log(np.abs(isw / ic))
        lr_card = np.log(np.abs(csw / cc))
    d = np.abs(lr_corr - lr_card)
    out["swap_asym_card_max"] = float(np.max(np.abs(lr_card[keep]))) if keep.any() else 0.0
    out["swap_asym_element_max"] = float(np.max(d[keep])) if keep.any() else 0.0
    if keep.any():
        k = np.unravel_index(np.argmax(np.where(keep, d, -1.0)), d.shape)
        out["swap_asym_element_at"] = {"vgs": float(VG[k]), "vds": float(VD[k]), "i_fwd": float(ic[k])}
    # ---- Q2 small signal (tight), element-identity reference
    vg, vd = pts[:, 0], pts[:, 1]
    gp = v2.g_terminal(corr, pm, vg, vd, 0.0)
    dgg, dgd = v2.g_partials_v2(corr, pm, vg, vd)
    i_card, i_rep = a["ac_VG_card_rep_i0"], a["ac_VG_card_rep_s0"]
    yc_dg = a["ac_VG_card_rep_yr"] + 1j * a["ac_VG_card_rep_ym"]
    yc_dd = a["ac_VD_card_rep_yr"] + 1j * a["ac_VD_card_rep_ym"]
    yr_dg = a["ac_VG_card_rep_sr"] + 1j * a["ac_VG_card_rep_sm"]
    yr_dd = a["ac_VD_card_rep_sr"] + 1j * a["ac_VD_card_rep_sm"]
    ye_dg = a["ac_VG_elem_yr"] + 1j * a["ac_VG_elem_ym"]
    ye_dd = a["ac_VD_elem_yr"] + 1j * a["ac_VD_elem_ym"]
    yp_dg = m.admittance(yc_dg, yr_dg, i_rep, gp, dgg)
    yp_dd = m.admittance(yc_dd, yr_dd, i_rep, gp, dgd)
    ipy = m.current_fwd(i_card, i_rep, gp)
    cnt = np.abs(ipy) > m1.I_FLOOR
    for k, e in (("gm", m1.norm_y_err(ye_dg.real, yp_dg.real)),
                 ("gds", m1.norm_y_err(ye_dd.real, yp_dd.real))):
        out[f"q2_err_{k}"] = float(np.nanmax(np.where(cnt, e, np.nan))) if cnt.any() else 0.0
        if np.any(cnt & ~np.isfinite(e)):
            out[f"q2_err_{k}"] = float("inf")
    # the I12-1 reference formula on the same runs (reported only)
    yo_dg = v2.python_admittance_v2(i_card, yc_dg, gp, dgg)
    yo_dd = v2.python_admittance_v2(i_card, yc_dd, gp, dgd)
    out["q2_err_i12_formula"] = float(np.nanmax(np.where(cnt, np.maximum(
        m1.norm_y_err(ye_dg.real, yo_dg.real), m1.norm_y_err(ye_dd.real, yo_dd.real)), np.nan))) if cnt.any() else 0.0
    out["q2_rep_imag_max"] = float(np.nanmax(np.abs(np.concatenate([yr_dg.imag, yr_dd.imag]))))
    out["q2_op_i_rel"] = m1.nanmax0(m1.rel_diff(a["ac_VG_elem_i0"], ipy, m1.I_FLOOR))
    caps = {"dd": (ye_dd.imag, yc_dd.imag), "dg": (ye_dg.imag, yc_dg.imag),
            "gg": (a["ac_VG_elem_yg"], a["ac_VG_card_rep_yg"])}
    for k, (e_, c_) in caps.items():
        err = v2.cap_err(e_, c_)
        out[f"cap_err_{k}"] = float(np.max(err)) if np.all(np.isfinite(err)) else float("inf")
    out["cap_max"] = max(out[f"cap_err_{k}"] for k in caps)
    # strong inversion and sign changes (I11-1 rule)
    jv = int(np.argmin(np.abs(np.abs(m1.GRID_VDS) - m1.GM_VDS[0]) + (s * m1.GRID_VDS < 0)))
    from cryoml.spice_pdk import device_mult
    w_eff = dev.W_um * device_mult(dev.dev_type, dev.L_um, dev.W_um)
    vt = m1.threshold_voltage(m1.GRID_VGS, a["grid_fwd_tight_card_rep"][:, jv], 1e-7 * w_eff / dev.L_um)
    out["vt_card"] = vt
    nv = len(m1.GRID_VGS)
    n_gm = len(m1.GM_VDS) * nv
    gm_mask = np.zeros(len(pts), bool)
    gm_mask[:n_gm] = np.abs(vg[:n_gm]) >= vt + 0.2
    gds_mask = np.zeros(len(pts), bool)
    gds_mask[n_gm:] = np.abs(vd[n_gm:]) >= 0.05
    gm_ac = [m1.curve_sign_change_introduced(ye_dg.real[k * nv:(k + 1) * nv], yc_dg.real[k * nv:(k + 1) * nv],
                                             gm_mask[k * nv:(k + 1) * nv]) for k in range(len(m1.GM_VDS))]
    gds_ac = [m1.curve_sign_change_introduced(ye_dd.real[n_gm + k * nv:n_gm + (k + 1) * nv],
                                              yc_dd.real[n_gm + k * nv:n_gm + (k + 1) * nv],
                                              gds_mask[n_gm + k * nv:n_gm + (k + 1) * nv])
              for k in range(len(m1.GDS_VGS))]
    gm_dn, gds_dn = [False] * len(m1.GM_VDS), [False] * len(m1.GDS_VGS)
    fd_err = 0.0
    for k, vds in enumerate(m1.GM_VDS):
        dc, dk = a[f"dense_card_gm{k}"], a[f"dense_elem_gm{k}"]
        if dc.shape[0] < 10 or dk.shape[0] != dc.shape[0]:
            fd_err = float("inf")
            gm_dn[k] = True
            continue
        o = np.argsort(np.abs(dc[:, 0]))
        x, ic_, ik_ = dc[o, 0], dc[o, 1], dk[o, 1]
        gm_c, gm_k = m1.central_diff(x, ic_), m1.central_diff(x, ik_)
        sel = slice(k * nv, (k + 1) * nv)
        at = np.searchsorted(np.round(np.abs(x), 6), np.round(np.abs(vg[sel]), 6))
        inner = (at >= 1) & (at <= len(x) - 2)
        at = np.clip(at, 1, len(x) - 2)
        cnt_k = (np.abs(ipy[sel]) > m1.I_FLOOR) & inner
        e1 = m1.norm_y_err(gm_k[at], ye_dg.real[sel])
        fd_err = max(fd_err, float(np.max(e1[cnt_k])) if cnt_k.any() else 0.0)
        gm_dn[k] = m1.curve_sign_change_introduced(gm_k, gm_c, np.abs(x) >= vt + 0.2)
    for k, vgs in enumerate(m1.GDS_VGS):
        dc, dk = a[f"dense_card_gds{k}"], a[f"dense_elem_gds{k}"]
        if dc.shape[0] < 10 or dk.shape[0] != dc.shape[0]:
            gds_dn[k] = True
            continue
        o = np.argsort(np.abs(dc[:, 0]))
        x, ic_, ik_ = dc[o, 0], dc[o, 1], dk[o, 1]
        gds_dn[k] = m1.curve_sign_change_introduced(m1.central_diff(x, ik_), m1.central_diff(x, ic_),
                                                    np.abs(x) >= 0.05)
    out.update({"q2_fd_err_osdi": fd_err,
                "q2_gm_curves_introduced": [bool(p or q) for p, q in zip(gm_ac, gm_dn)],
                "q2_gds_curves_introduced": [bool(p or q) for p, q in zip(gds_ac, gds_dn)],
                "q2_gm_neg_card_ac": int(np.sum(gm_mask & (yc_dg.real < 0))),
                "q2_gds_neg_card_ac": int(np.sum(gds_mask & (yc_dd.real < 0)))})
    return out


def stage_q12(rd: Path, arm: str, tags, workers: int) -> dict:
    cards = jload(rd / "cards_M2.json")
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(q12_task, [(t, cards[t], str(rd), arm) for t in tags], chunksize=1)
    out = {"wall_s": round(time.time() - t0, 1), "devices": {r["tag"]: r for r in rows}}
    jdump(out, rd / "q12.json")
    print(f"q12 ({arm}): {len(rows)} devices, {out['wall_s']} s", flush=True)
    return out


CIRCUITS = drv12.CIRCUITS
Q3_ARMS = drv12.Q3_ARMS


def arm_cards(arm_q3: str, tags, m2: dict) -> dict:
    devs = devices()
    held = jload(ALL18 / "heldout_cards.json")
    cards = {}
    for t in tags:
        d = devs[t]
        p = None if arm_q3 == "published" else (dict(held["C0_refit"][t]) if arm_q3 == "C0_refit"
                                                 else dict(m2[t]))
        cards.setdefault(d.dev_type, []).append((d.bin, p))
    return cards


def q3_task(task):
    name, kind, tags, settings, arm_q3, rd_s, arm = task
    rd = Path(rd_s)
    devs = devices()
    m2 = jload(rd / "cards_M2.json")
    mode = "elem" if arm_q3 == "M2_full" else "card"
    module = m.module_name(corr_full(arm))
    osdi = arm_osdi(rd, arm) if mode == "elem" else None
    reps = 3 if arm_q3 in ("M2_card", "M2_full") else 1
    res = {"circuit": name, "arm": arm_q3, "kind": kind}
    with m1.tempdir("cryoml_i14c_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, arm_cards(arm_q3, tags, m2))
        f = td / "out.dat"
        if kind == "vtc":
            body, ctrl = m.vtc_body(devs[tags[0]], devs[tags[1]], mode, module), m1.vtc_control(f)
        elif kind == "ring":
            body = m.ring_body(devs[tags[0]], devs[tags[1]], mode, module, stages=5)
            ctrl = m1.ring_control(f, settings["tstep"], settings["tstop"])
        else:
            body, ctrl = m.mirror_body(devs[tags[0]], mode, settings["iref"], module), m1.mirror_control(f)
        walls = []
        for _ in range(reps):
            r = m1.run_deck(m.deck(name, lib, body, ctrl, osdi, module, numdgt=12), td, "c.sp",
                            timeout_s=m1.TIMEOUT_CIRCUIT_S)
            walls.append(r["wall_s"])
            if r["msgs"]:          # failed run: no repeats (I12-1 amendment A1)
                break
        res["wall"], res["msgs"] = walls, r["msgs"]
        if kind == "mirror":
            rows = m1.read_wrdata(f, 4)
            x, y, vref = rows[:, 0], rows[:, 1], rows[:, 3]
        else:
            rows = m1.read_wrdata(f, 2)
            x, y = rows[:, 0], rows[:, 1]
    finite = bool(rows.size and np.all(np.isfinite(rows)))
    res["finite"] = finite
    if kind == "vtc" and finite:
        res.update(m1.vtc_metrics(x, y))
    elif kind == "ring" and finite:
        res.update(m1.oscillation(x, y, m1.VDD_CIRCUIT / 2, t_from=0.3 * settings["tstop"]))
    elif kind == "mirror" and finite:
        vr = float(np.median(vref))
        res["v_ref"] = vr
        res["ratio_at_0p9"] = m1.interp_at(x, y, 0.9) / settings["iref"]
        res["ratio_at_vref"] = m1.interp_at(x, y, vr) / settings["iref"]
    ok = finite and not r["msgs"]
    if kind == "ring" and finite:
        ok = ok and res["n_periods"] >= m1.THR["ro_min_periods"]
    res["ok"] = bool(ok)
    (rd / "arrays").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rd / "arrays" / f"q3_{name}_{arm_q3}.npz", rows=rows)
    return res


def stage_q3(rd: Path, arm: str, workers: int) -> dict:
    tasks = [(n, k, t, s, aq, str(rd), arm) for n, k, t, s in CIRCUITS for aq in Q3_ARMS]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(q3_task, tasks, chunksize=1)
    out = {"wall_s": round(time.time() - t0, 1),
           "rows": {f"{r['circuit']}|{r['arm']}": r for r in rows}}
    jdump(out, rd / "q3.json")
    print(f"q3 ({arm}): {len(rows)} runs, {out['wall_s']} s; not ok: "
          f"{[k for k, v in out['rows'].items() if not v['ok']]}", flush=True)
    return out


def summary_values(rd: Path) -> dict:
    q12 = jload(rd / "q12.json")["devices"]
    q3 = jload(rd / "q3.json")["rows"]
    comp = jload(rd / "compile.json")
    D = list(q12.values())
    elem_msgs = sum(len(v) for r in D for k, v in r["msgs"].items() if k.endswith("_elem"))
    ratios = {r["tag"]: float(np.median(r["wall"]["grid_fwd_default_elem"])
                              / np.median(r["wall"]["grid_fwd_default_card"])) for r in D}
    names = [n for n, *_ in CIRCUITS]
    circ = v2.circuit_runtime_ratios(q3, names)
    s = {"q1_tight_viol": int(sum(r["q1_tight_viol"] for r in D)),
         "q1_bad": int(elem_msgs + sum(r["q1_nonfinite"] for r in D)),
         "q1_vds0_max": max(r["q1_vds0_max_corr"] for r in D),
         "g_text_max": comp["g_text_max"], "g_sym_text_max": comp["g_sym_text_max"],
         "swap_asym_max": max(r["swap_asym_element_max"] for r in D),
         "q2_ac_max": max(max(r["q2_err_gm"], r["q2_err_gds"]) for r in D),
         "cap_max": max(r["cap_max"] for r in D),
         "q2_gm_sign": int(sum(sum(r["q2_gm_curves_introduced"]) for r in D)),
         "q3_fail": v2.q3_fail_count(q3, names),
         "q1_default_max": max(r["q1_default_max_rel_1nA"] for r in D),
         "q2_gds_sign": int(sum(sum(r["q2_gds_curves_introduced"]) for r in D)),
         "runtime_ratio_max": max(list(ratios.values()) + list(circ.values()))}
    return {"values": s, "runtime_grid": ratios, "runtime_circuits": circ,
            "runtime_circuits_excluded": [n for n, *_ in CIRCUITS if n not in circ],
            "n_devices": len(D), "q3_not_ok": [k for k, v in q3.items() if not v["ok"]],
            "q2_err_i12_formula_max": max(r["q2_err_i12_formula"] for r in D),
            "nonchannel_share_fwd_max": max(r["nonchannel_share_fwd_max"] for r in D),
            "card_rep_vs_card_default_max_abs": max(r["card_rep_vs_card_default_max_abs"] for r in D)}


def stage_report(rd: Path, arm: str) -> dict:
    sv = summary_values(rd)
    dec = v2.decide_deploy(sv["values"])
    refit = jload(rd / "refit.json")
    nat = refit["native"].values()
    out = {"arm": arm, "label": "I14-1 Part B" if arm == "v2" else
           "reference arm (M_bias v1 full-data g, non-symmetric inputs; not the I14-1 outcome)",
           **sv, "decision": dec, "refit_in_sample": {
               "mean_rrms_M2_python": float(np.mean([v["rrms_python"] for v in nat])),
               "mean_rrms_M2_python_channel": float(np.mean([v["rrms_python_channel"] for v in nat])),
               "mean_rrms_M2_native": float(np.mean([v["rrms_native"] for v in nat])),
               "mean_rrms_M2_card_only": float(np.mean([v["rrms_card_only"] for v in nat])),
               "n_failed": int(sum(bool(v["failed"]) for v in refit["rows"].values()))}}
    jdump(out, rd / "summary.json")
    print(json.dumps({"arm": arm, "values": sv["values"], **dec}, indent=1, default=float), flush=True)
    return out


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def stage_decide() -> dict:
    val = jload(OUT / "validity" / "validity.json")
    acc_native = jload(OUT / "accuracy" / "accuracy.json")
    dep = jload(OUT / "deploy_v2" / "summary.json")
    dep_v1 = jload(OUT / "deploy_v1" / "summary.json")
    unit = jload(OUT / "validity" / "unit_tests.json")
    i13 = jload(I13 / "summary.json")
    lodo = jload(LODO / "results.json")
    n_rows_min = min(len(acc_native["per_arm"][a]["native"]) for a in ("v2", "v1"))
    m2_failed = max(dep["refit_in_sample"]["n_failed"], dep_v1["refit_in_sample"]["n_failed"])
    validity = m.decide_validity({"v1_max_rrms_diff": val["v1_max_rrms_diff"],
                                  "v2_unit_tests_pass": bool(unit["passed"]),
                                  "v3_n_rows_min": n_rows_min, "v3_m2_failed": m2_failed})
    acc = v2.decide_accuracy(lodo["comparisons"]["A1_M_bias_v2_vs_C0_refit"],
                             lodo["comparisons"]["A2_M_bias_v2_vs_M_bias_v1"],
                             lodo["n_failed_M_bias_v2"])
    out = {"prereg": str(PREREG.relative_to(ROOT)), "validity": validity, "accuracy_I13_1": acc,
           "deploy_v2": dep["decision"], **m.decide_outcome(validity, acc, dep["decision"]),
           "A1": lodo["comparisons"]["A1_M_bias_v2_vs_C0_refit"],
           "A2": lodo["comparisons"]["A2_M_bias_v2_vs_M_bias_v1"],
           "reference_arm_v1_deploy": dep_v1["decision"],
           "element_deck_accuracy": {a: {k: v for k, v in acc_native["per_arm"][a].items()
                                         if k not in ("native", "python_rerun", "python_channel")}
                                     for a in ("v2", "v1")},
           "element_deck_gmr": acc_native.get("gmr_native"),
           "i13_1_table": i13["table"]}
    jdump(out, OUT / "decision.json")
    print(json.dumps({k: out[k] for k in ("validity", "accuracy_I13_1", "deploy_v2", "outcome",
                                          "action", "reference_arm_v1_deploy")}, indent=1, default=float),
          flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("repcheck", "validity", "accuracy", "deploy", "compile", "refit",
                             "q12", "q3", "report", "decide"))
    ap.add_argument("--arm", default="v2", choices=("v2", "v1"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="2 devices, refit max_nfev 20 (code check; not used for decisions)")
    args = ap.parse_args()
    base = OUT / "smoke" if args.smoke else OUT
    tags = SMOKE_TAGS if args.smoke else sorted(devices())
    t0 = time.time()
    if args.stage == "repcheck":
        stage_repcheck(args.workers, tags if args.smoke else None)
    elif args.stage == "validity":
        stage_validity(args.workers, base / "validity", tags)
    elif args.stage == "accuracy":
        stage_accuracy(args.workers, base / "accuracy", tags)
    elif args.stage == "decide":
        stage_decide()
    else:
        rd = base / f"deploy_{args.arm}"
        opts = dict(REFIT_OPTS, heldout_nfev=20) if args.smoke else dict(REFIT_OPTS)
        workers = min(args.workers, 4)
        stages = (("compile", "refit", "q12", "q3", "report") if args.stage == "deploy"
                  else (args.stage,))
        timing = {}
        for st in stages:
            t1 = time.time()
            if st == "compile":
                stage_compile(rd, args.arm)
            elif st == "refit":
                stage_refit(rd, args.arm, tags, workers, opts)
            elif st == "q12":
                stage_q12(rd, args.arm, tags, workers)
            elif st == "q3":
                if args.smoke:      # smoke: starting cards for circuit devices not refit
                    have = jload(rd / "cards_M2.json")
                    sc = start_cards()
                    for _, _, ts, _ in CIRCUITS:
                        for t in ts:
                            have.setdefault(t, sc[t])
                    jdump(have, rd / "cards_M2.json")
                stage_q3(rd, args.arm, workers)
            else:
                stage_report(rd, args.arm)
            timing[st] = round(time.time() - t1, 1)
        tf = rd / "timing.json"
        old = jload(tf) if tf.exists() else {}
        old.update(timing)
        jdump(old, tf)
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
