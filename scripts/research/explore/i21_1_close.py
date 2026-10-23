#!/usr/bin/env python3
"""I21-1: open items that need no user decision.

Pre-registration: out/research/explore/I21_1/prereg.md (prereg.json).

Stages (``--stage``):

  partA   --ds nist4|nistrt [--sub all|m2|compile|q12|q3|acc|report]:
          Part B deployment checks of the I20-1 full-data g v2 (lambda 0 V)
          -> partA_<ds>/
  s3tud   TU Delft v1-recipe S3_D on the I18-2 state-C records (+ state-L
          check of the first group) -> s3tud/s3tud.json
  ihp     --temp 233|343|398 --sub pens|base|inner|sel|decide: selected g v2
          recipe on IHP SG13G2 (I17-2 setup) -> ihp_<T>K/
  decide  every part -> decision.json

The IHP stages reuse the I17-2 driver with its temperature and output
directory rebound at import (env CRYOML_I21_1_IHP_T; spawned workers re-run
the rebinding).

  .venv/bin/python -I scripts/research/explore/i21_1_close.py --stage partA --ds nist4 --workers 8
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i17_2_ihp as d17  # noqa: E402  (I17-2 driver; rebound below)
import i20_1_nist_clamp as d20  # noqa: E402  (I20-1 driver: NIST devices, bins, state-C records, refit_task)
from cryoml.explore import i11_1_deploy as m1  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i14_1_chan as m14  # noqa: E402
from cryoml.explore import i17_1_vds0 as m17  # noqa: E402
from cryoml.explore import i17_2_ihp as m17b  # noqa: E402
from cryoml.explore import i21_1_close as m  # noqa: E402

EXP = ROOT / "out/research/explore"
OUT = EXP / "I21_1"
PREREG = OUT / "prereg.json"
_SMOKE = {"on": os.environ.get("CRYOML_I21_1_SMOKE") == "1"}
UNIT_A = ("tests.test_i21_1_close", "tests.test_i14_1_chan", "tests.test_i12_1_symm", "tests.test_i20_1_nist_clamp")
UNIT_C = ("tests.test_i21_1_close", "tests.test_i17_2_ihp", "tests.test_i12_1_symm", "tests.test_i15_2_ablate")
TOLS = (("default", ""), ("r4", m1.R4), ("tight", m1.TIGHT))
FULL_KEY = "full|sym_e1_l0"
SMOKE_N = 2


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def base_dir() -> Path:
    return OUT / "smoke" if _SMOKE["on"] else OUT


def run_unit_tests(files) -> dict:
    r = subprocess.run([sys.executable, "-m", "unittest", *files], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    return {"files": list(files), "rc": r.returncode, "passed": r.returncode == 0, "tail": tail}


# ===========================================================================
# Part A: NIST Part B
# ===========================================================================
class Dev:
    """Minimal device view for the netlists and metrics (polarity, W, L)."""

    def __init__(self, tag, polarity, W_um, L_um):
        self.tag, self.polarity, self.W_um, self.L_um = tag, polarity, float(W_um), float(L_um)
        self.dev_type = polarity            # q12 metrics use dev_type = "nmos" | "pmos"


def dev_of(ds: str, tag: str) -> Dev:
    d = d20.devices(ds)[tag]
    return Dev(tag, d.polarity, d.W_um, d.L_um)


def rdir(ds: str) -> Path:
    return base_dir() / f"partA_{ds}"


def full_corr(ds: str) -> v2.SymCorrection:
    js = jload(EXP / "I20_1" / f"select_{ds}" / "full_corrections.json")
    if list(js) != [FULL_KEY]:
        raise RuntimeError(f"unexpected full-data keys {list(js)}")
    return v2.SymCorrection.from_json(js[FULL_KEY])


def osdi_path(ds: str) -> Path:
    return rdir(ds) / "osdi" / "full.osdi"


def circuit_tags(ds: str) -> dict:
    """Pre-registered circuit devices, checked against the rule."""
    order = d20.finite_order(ds)
    devs = d20.devices(ds)
    got = m.circuit_devices({t: (devs[t].polarity, devs[t].L_um, devs[t].W_um) for t in order})
    if got != m.PREREG_CIRCUIT_DEVICES[ds]:
        raise RuntimeError(f"circuit devices {got} differ from the pre-registered ones")
    return got


def partA_tags(ds: str) -> list[str]:
    order = d20.finite_order(ds)
    if _SMOKE["on"]:
        n = [t for t in order if t.startswith("n")][:1]
        p = [t for t in order if t.startswith("p")][:1]
        return n + p
    return order


def written(ds: str, tag: str, params: dict | None) -> dict | None:
    d = d20.devices(ds)[tag]
    bi = d20.bins(ds)[tag]
    return m.card_params(ds, d.polarity, int(bi["bin"]), d.W_um, d.L_um, params, bi["stock"])


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------
def stage_compile(ds: str) -> dict:
    corr = full_corr(ds)
    text = v2.to_verilog_a_v2(corr)
    p = osdi_path(ds)
    p.parent.mkdir(parents=True, exist_ok=True)
    c = m1.compile_osdi(text, p.with_suffix(".va"), p)
    rng = np.random.default_rng(20261023)
    worst, worst_sym = 0.0, 0.0
    for _ in range(2000):
        q = int(rng.integers(0, 2))
        vgs, vds = rng.uniform(-m.VDD, m.VDD, size=2)
        gt = v2.eval_verilog_a_v2(text, q, vgs, vds)
        gp = float(corr.g_v(bool(q), vgs, vds))
        gs = v2.eval_verilog_a_v2(text, q, vgs - vds, -vds)
        worst = max(worst, abs(gt - gp))
        worst_sym = max(worst_sym, abs(gs - gt))
    out = {"ds": ds, "sha256": c["sha256"], "compiled": bool(c["compiled"]), "lines": len(text.splitlines()),
           "vdd": corr.vdd, "eps": corr.eps, "g_text_max": worst, "g_sym_text_max": worst_sym, "n_points": 2000}
    jdump(out, rdir(ds) / "compile.json")
    print(f"compile {ds}: |g text - python| {worst:.2e}, exchanged {worst_sym:.2e}", flush=True)
    return out


# ---------------------------------------------------------------------------
# M2 cards
# ---------------------------------------------------------------------------
def opts_a() -> dict:
    return dict(d20.HELDOUT, heldout_nfev=20, heldout_irls=1) if _SMOKE["on"] else dict(d20.HELDOUT)


def stage_m2(ds: str, workers: int) -> dict:
    t0 = time.time()
    corr = full_corr(ds)
    recs = d20.c_records(ds)
    tags = partA_tags(ds)
    if _SMOKE["on"]:
        tags = sorted(set(tags) | set(circuit_tags(ds).values()))
    tasks = [("M2", ds, t, corr, dict(recs[t].meta["params"]), "C", opts_a()) for t in tags]
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(d20.refit_task, tasks, chunksize=1)
    cards = {r["tag"]: r["params"] for r in rows}
    jdump(cards, rdir(ds) / "cards_M2.json")
    out = {"ds": ds, "opts": opts_a(), "wall_s": round(time.time() - t0, 1),
           "rows": {r["tag"]: {k: v for k, v in r.items() if k not in ("params", "history")} for r in rows},
           "n_failed": int(sum(bool(r["failed"]) or not np.isfinite(r["rrms"]) for r in rows)),
           "mean_rrms_python": float(np.mean([r["rrms"] for r in rows])),
           "n_sim": int(sum(r["n_sim"] for r in rows))}
    jdump(out, rdir(ds) / "m2.json")
    print(f"M2 {ds}: {len(rows)} refits, {out['wall_s']} s, failed {out['n_failed']}, in-sample mean "
          f"{out['mean_rrms_python']:.5f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Q1 / Q2
# ---------------------------------------------------------------------------
def q12_task(task):
    ds, tag, params, smoke = task
    _SMOKE["on"] = smoke
    dev = dev_of(ds, tag)
    corr = full_corr(ds)
    osdi = osdi_path(ds)
    bi = int(d20.bins(ds)[tag]["bin"])
    res = {"tag": tag, "bin": bi, "msgs": {}, "wall": {}}
    a = {}
    with m1.tempdir("cryoml_i21a_") as td_s:
        td = Path(td_s)
        lib = m.write_library(td, ds, {dev.polarity: (bi, written(ds, tag, params))}, replica=True)

        def go(label, body, control, mode, options="", repeats=1, out_file=None):
            walls, r = [], None
            for _ in range(repeats):
                if out_file is not None and Path(out_file).exists():
                    Path(out_file).unlink()          # serpentine decks append: one run per file
                r = m1.run_deck(m.deck(label, ds, lib, body, control, osdi if mode == "elem" else None, options),
                                td, f"{label}.sp", timeout_s=m1.TIMEOUT_DEVICE_S)
                walls.append(r["wall_s"])
            res["msgs"][label] = r["msgs"]
            res["wall"][label] = walls
            return r

        f = td / "grid_fwd_default_card.dat"
        go("grid_fwd_default_card", m.device_body(dev, "card", ds), m.grid_control_serp(dev.polarity, f, False),
           "card", "", repeats=3, out_file=f)
        a["grid_fwd_default_card"] = m.grid_rows(m1.read_wrdata(f, 2))[..., 0]
        for mode in ("card_rep", "elem"):
            sense = mode == "card_rep"
            nc = 3 if sense else 2
            for tol, opt in TOLS:
                lab = f"grid_fwd_{tol}_{mode}"
                f = td / f"{lab}.dat"
                go(lab, m.device_body(dev, mode, ds), m.grid_control_serp(dev.polarity, f, sense), mode, opt,
                   repeats=3 if (tol == "default" and mode == "elem") else 1, out_file=f)
                g = m.grid_rows(m1.read_wrdata(f, nc + (1 if sense else 0)))
                a[lab] = g[..., 0]
                if sense:
                    a[lab + "_rep"] = g[..., -1]
            lab = f"grid_swap_tight_{mode}"
            f = td / f"{lab}.dat"
            go(lab, m.device_body(dev, mode, ds, swap=True), m.grid_control_serp(dev.polarity, f, sense), mode,
               m1.TIGHT, out_file=f)
            g = m.grid_rows(m1.read_wrdata(f, nc + (1 if sense else 0)))
            a[lab] = g[..., 0]
            if sense:
                a[lab + "_rep"] = g[..., -1]
        pts = m.op_points(dev.polarity)
        for mode in ("card_rep", "elem"):
            for ex in ("VG", "VD"):
                lab = f"ac_{ex}_{mode}"
                r = go(lab, m.device_body(dev, mode, ds), m14.ac_control(pts, ex, mode == "card_rep"), mode, m1.TIGHT)
                for k, v in m14.parse_ac(r["text"], len(pts)).items():
                    a[f"{lab}_{k}"] = v
        for mode in ("card", "elem"):
            lab = f"dense_{mode}"
            pre = td / lab
            go(lab, m.device_body(dev, mode, ds), m.dense_control(dev.polarity, pre), mode, m1.TIGHT)
            for k in range(len(m.GM_VDS)):
                a[f"{lab}_gm{k}"] = m1.read_wrdata(Path(f"{pre}_gm{k}.dat"))
            for k in range(len(m.GDS_VGS)):
                a[f"{lab}_gds{k}"] = m1.read_wrdata(Path(f"{pre}_gds{k}.dat"))
        r = go("repcheck", m.device_body(dev, "card_rep", ds), m.nonchannel_control(pts[::4], ds, dev.polarity),
               "card_rep", m1.TIGHT)
        res["replica_nonchannel_max"] = m.parse_nonchannel(r["text"])
        res["replica_nonchannel_n_values"] = int(sum(1 for ln in r["text"].splitlines() if ln.strip().startswith("@m.xm1r")))
    res.update(q12_metrics(dev, corr, pts, a))
    (rdir(ds) / "arrays").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rdir(ds) / "arrays" / f"q12_{tag}.npz", pts=pts,
                        **{k: v for k, v in a.items() if isinstance(v, np.ndarray)})
    return res


def q12_metrics(dev, corr, pts, a) -> dict:
    """I18-2 ``q12_metrics`` on the NIST grids and op-point sets; the
    V(d,s) = 0 value is the raw element current (I17-1; no gate current)."""
    pm = dev.polarity == "pmos"
    s = -1.0 if pm else 1.0
    VG, VD = np.meshgrid(s * m.GRID_VGS, m.GRID_VDS, indexing="ij")
    g_fwd = v2.g_terminal(corr, pm, VG, VD, 0.0)
    g_sw = v2.g_terminal(corr, pm, VG, 0.0, VD)
    out = {"g_sym_grid_max": float(np.max(np.abs(g_sw - g_fwd)))}
    for tol, _ in TOLS:
        ref = m14.current_fwd(a[f"grid_fwd_{tol}_card_rep"], a[f"grid_fwd_{tol}_card_rep_rep"], g_fwd)
        eq = v2.eq_violation(a[f"grid_fwd_{tol}_elem"], ref)
        out[f"q1_{tol}_fwd_viol"] = eq["n_viol"]
        out[f"q1_{tol}_fwd_max_ratio"] = eq["max_ratio"]
        out[f"q1_{tol}_max_rel_1nA"] = m1.nanmax0(m1.rel_diff(a[f"grid_fwd_{tol}_elem"], ref, m1.I_FLOOR))
    ref_sw = m14.current_swap(a["grid_swap_tight_card_rep"], a["grid_swap_tight_card_rep_rep"], g_sw)
    eq = v2.eq_violation(a["grid_swap_tight_elem"], ref_sw)
    out["q1_tight_swap_viol"] = eq["n_viol"]
    out["q1_tight_swap_max_ratio"] = eq["max_ratio"]
    out["q1_tight_viol"] = int(out["q1_tight_fwd_viol"] + out["q1_tight_swap_viol"])
    elem_runs = [k for k in a if k.startswith("grid_") and k.endswith("_elem")]
    out["q1_nonfinite"] = int(sum(np.sum(~np.isfinite(a[k])) for k in elem_runs))
    out["q1_nonfinite_card"] = int(sum(np.sum(~np.isfinite(a[k])) for k in a
                                       if k.startswith("grid_") and "_card" in k))
    out["card_rep_vs_card_default_max_abs"] = float(np.nanmax(np.abs(
        a["grid_fwd_default_card_rep"] - a["grid_fwd_default_card"])))
    j0 = m17.vds0_column(m.GRID_VDS)
    out["vds0"] = m17.vds0_values(a, j0, m.GRID_VGS)          # raw element current per tolerance (I17-1)
    add = {tol: np.abs(a[f"grid_fwd_{tol}_elem"][:, j0] - a[f"grid_fwd_{tol}_card_rep"][:, j0]) for tol, _ in TOLS}
    out["q1_vds0_added_by_tol"] = {tol: float(np.nanmax(v)) for tol, v in add.items()}   # reported
    out["g_grid_min"], out["g_grid_max"] = float(np.min(g_fwd)), float(np.max(g_fwd))
    fw = (s * VD) > 0
    cc, rp = a["grid_fwd_tight_card_rep"], a["grid_fwd_tight_card_rep_rep"]
    big = fw & (np.abs(cc) > m1.I_FLOOR) & np.isfinite(cc) & np.isfinite(rp)
    out["nonchannel_share_fwd_max"] = float(np.max(np.abs(cc - rp)[big] / np.abs(cc[big]))) if big.any() else 0.0
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
    vg, vd = pts[:, 0], pts[:, 1]
    gp = v2.g_terminal(corr, pm, vg, vd, 0.0)
    dgg, dgd = v2.g_partials_v2(corr, pm, vg, vd)
    i_rep = a["ac_VG_card_rep_s0"]
    yc_dg = a["ac_VG_card_rep_yr"] + 1j * a["ac_VG_card_rep_ym"]
    yc_dd = a["ac_VD_card_rep_yr"] + 1j * a["ac_VD_card_rep_ym"]
    yr_dg = a["ac_VG_card_rep_sr"] + 1j * a["ac_VG_card_rep_sm"]
    yr_dd = a["ac_VD_card_rep_sr"] + 1j * a["ac_VD_card_rep_sm"]
    ye_dg = a["ac_VG_elem_yr"] + 1j * a["ac_VG_elem_ym"]
    ye_dd = a["ac_VD_elem_yr"] + 1j * a["ac_VD_elem_ym"]
    yp_dg = m14.admittance(yc_dg, yr_dg, i_rep, gp, dgg)
    yp_dd = m14.admittance(yc_dd, yr_dd, i_rep, gp, dgd)
    ipy = m14.current_fwd(a["ac_VG_card_rep_i0"], i_rep, gp)
    cnt = np.abs(ipy) > m1.I_FLOOR
    for k, e in (("gm", m1.norm_y_err(ye_dg.real, yp_dg.real)), ("gds", m1.norm_y_err(ye_dd.real, yp_dd.real))):
        out[f"q2_err_{k}"] = float(np.nanmax(np.where(cnt, e, np.nan))) if cnt.any() else 0.0
        if np.any(cnt & ~np.isfinite(e)):
            out[f"q2_err_{k}"] = float("inf")
    out["q2_rep_imag_max"] = float(np.nanmax(np.abs(np.concatenate([yr_dg.imag, yr_dd.imag]))))
    out["q2_op_i_rel"] = m1.nanmax0(m1.rel_diff(a["ac_VG_elem_i0"], ipy, m1.I_FLOOR))
    caps = {"dd": (ye_dd.imag, yc_dd.imag), "dg": (ye_dg.imag, yc_dg.imag),
            "gg": (a["ac_VG_elem_yg"], a["ac_VG_card_rep_yg"])}
    for k, (e_, c_) in caps.items():
        err = v2.cap_err(e_, c_)
        out[f"cap_err_{k}"] = float(np.max(err)) if np.all(np.isfinite(err)) else float("inf")
    out["cap_max"] = max(out[f"cap_err_{k}"] for k in caps)
    jv = int(np.argmin(np.abs(np.abs(m.GRID_VDS) - m.GM_VDS[0]) + (s * m.GRID_VDS < 0)))
    vt = m1.threshold_voltage(m.GRID_VGS, a["grid_fwd_tight_card_rep"][:, jv], m.VT_CRIT_PER_SQ * dev.W_um / dev.L_um)
    out["vt_card"] = vt
    nv = len(m.GRID_VGS)
    n_gm = len(m.GM_VDS) * nv
    gm_mask = np.zeros(len(pts), bool)
    gm_mask[:n_gm] = np.abs(vg[:n_gm]) >= vt + 0.2
    gds_mask = np.zeros(len(pts), bool)
    gds_mask[n_gm:] = np.abs(vd[n_gm:]) >= 0.05
    gm_ac = [m1.curve_sign_change_introduced(ye_dg.real[k * nv:(k + 1) * nv], yc_dg.real[k * nv:(k + 1) * nv],
                                             gm_mask[k * nv:(k + 1) * nv]) for k in range(len(m.GM_VDS))]
    gds_ac = [m1.curve_sign_change_introduced(ye_dd.real[n_gm + k * nv:n_gm + (k + 1) * nv],
                                              yc_dd.real[n_gm + k * nv:n_gm + (k + 1) * nv],
                                              gds_mask[n_gm + k * nv:n_gm + (k + 1) * nv])
              for k in range(len(m.GDS_VGS))]
    gm_dn, gds_dn = [False] * len(m.GM_VDS), [False] * len(m.GDS_VGS)
    gds_counts = [0] * len(m.GDS_VGS)
    fd_err = 0.0
    for k in range(len(m.GM_VDS)):
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
    for k in range(len(m.GDS_VGS)):
        dc, dk = a[f"dense_card_gds{k}"], a[f"dense_elem_gds{k}"]
        if dc.shape[0] < 10 or dk.shape[0] != dc.shape[0]:
            gds_dn[k] = True
            continue
        o = np.argsort(np.abs(dc[:, 0]))
        x, ic_, ik_ = dc[o, 0], dc[o, 1], dk[o, 1]
        gd_k, gd_c = m1.central_diff(x, ik_), m1.central_diff(x, ic_)
        gds_dn[k] = m1.curve_sign_change_introduced(gd_k, gd_c, np.abs(x) >= 0.05)
        gds_counts[k] = m1.sign_changes_introduced(gd_k, gd_c, np.abs(x) >= 0.05)
    out.update({"q2_fd_err_osdi": fd_err,
                "q2_gm_curves_introduced": [bool(p or q) for p, q in zip(gm_ac, gm_dn)],
                "q2_gds_curves_introduced": [bool(p or q) for p, q in zip(gds_ac, gds_dn)],
                "q2_gds_points_introduced_dense": gds_counts,
                "q2_gm_neg_card_ac": int(np.sum(gm_mask & (yc_dg.real < 0))),
                "q2_gds_neg_card_ac": int(np.sum(gds_mask & (yc_dd.real < 0)))})
    return out


def stage_q12(ds: str, workers: int) -> dict:
    cards = jload(rdir(ds) / "cards_M2.json")
    tags = partA_tags(ds)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(q12_task, [(ds, t, cards[t], _SMOKE["on"]) for t in tags], chunksize=1)
    out = {"ds": ds, "wall_s": round(time.time() - t0, 1), "devices": {r["tag"]: r for r in rows}}
    jdump(out, rdir(ds) / "q12.json")
    print(f"q12 {ds}: {len(rows)} devices, {out['wall_s']} s", flush=True)
    return out


# ---------------------------------------------------------------------------
# Q3 circuits
# ---------------------------------------------------------------------------
def arm_params(ds: str, arm: str, tag: str, m2: dict) -> dict | None:
    if arm == "published":
        return None
    if arm == "C0_refit":
        return dict(jload(EXP / "I20_1" / f"base_{ds}" / "heldout_cards.json")[f"C0_refit|{tag}"])
    return dict(m2[tag])


def q3_task(task):
    ds, name, kind, roles, settings, arm, smoke = task
    _SMOKE["on"] = smoke
    ctags = circuit_tags(ds)
    m2 = jload(rdir(ds) / "cards_M2.json")
    devs = [dev_of(ds, ctags[r]) for r in roles]
    mode = "elem" if arm == "M2_full" else "card"
    osdi = osdi_path(ds) if mode == "elem" else None
    reps = 3 if arm in ("M2_card", "M2_full") else 1
    res = {"circuit": name, "arm": arm, "kind": kind, "devices": {r: ctags[r] for r in roles},
           "bins": {r: int(d20.bins(ds)[ctags[r]]["bin"]) for r in roles}}
    cards = {}
    for r in roles:
        t = ctags[r]
        cards[d20.devices(ds)[t].polarity] = (int(d20.bins(ds)[t]["bin"]), written(ds, t, arm_params(ds, arm, t, m2)))
    with m1.tempdir("cryoml_i21c_") as td_s:
        td = Path(td_s)
        lib = m.write_library(td, ds, cards, replica=mode == "elem")
        f = td / "out.dat"
        if kind == "vtc":
            body, ctrl = m.vtc_body(devs[0], devs[1], mode, ds), m1.vtc_control(f)
        elif kind == "ring":
            body = m.ring_body(devs[0], devs[1], mode, ds, stages=5)
            ctrl = m1.ring_control(f, settings["tstep"], settings["tstop"])
        else:
            body, ctrl = m.mirror_body(devs[0], mode, settings["iref"], ds), m1.mirror_control(f)
        walls = []
        for _ in range(reps):
            r = m1.run_deck(m.deck(name, ds, lib, body, ctrl, osdi, numdgt=12), td, "c.sp",
                            timeout_s=m1.TIMEOUT_CIRCUIT_S)
            walls.append(r["wall_s"])
            if r["msgs"]:
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
    res["n_rows"] = int(rows.shape[0])
    if kind == "vtc" and finite:
        res.update(m1.vtc_metrics(x, y))
    elif kind == "ring" and finite:
        res.update(m1.oscillation(x, y, m.VDD / 2, t_from=0.3 * settings["tstop"]))
    elif kind == "mirror" and finite:
        vr = float(np.median(vref))
        res["v_ref"] = vr
        res["ratio_at_0p9"] = m1.interp_at(x, y, 0.9) / settings["iref"]
        res["ratio_at_vref"] = m1.interp_at(x, y, vr) / settings["iref"]
    ok = finite and not r["msgs"]
    if kind == "ring" and finite:
        ok = ok and res["n_periods"] >= v2.THR["ro_min_periods"]
    res["ok"] = bool(ok)
    (rdir(ds) / "arrays").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rdir(ds) / "arrays" / f"q3_{name}_{arm}.npz", rows=rows)
    return res


def stage_q3(ds: str, workers: int) -> dict:
    tasks = [(ds, n, k, r, s, arm, _SMOKE["on"]) for n, k, r, s in m.CIRCUITS for arm in m.Q3_ARMS]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(q3_task, tasks, chunksize=1)
    out = {"ds": ds, "wall_s": round(time.time() - t0, 1), "devices": circuit_tags(ds),
           "rows": {f"{r['circuit']}|{r['arm']}": r for r in rows}}
    jdump(out, rdir(ds) / "q3.json")
    print(f"q3 {ds}: {len(rows)} runs, {out['wall_s']} s; not ok: "
          f"{[k for k, v in out['rows'].items() if not v['ok']]}", flush=True)
    return out


# ---------------------------------------------------------------------------
# measured-grid decks: VA1 and element-deck accuracy (reported)
# ---------------------------------------------------------------------------
def measured_run(ds: str, tag: str, params: dict, mode: str) -> dict:
    """The I8-1 / I9-1 measured-grid deck (no numdgt, as ``simulate``) with
    the Part B library and the instance of ``mode``."""
    from cryoml import spice_pdk as sp
    from cryoml.explore import i8_1_nist as b8
    d = d20.devices(ds)[tag]
    dev = dev_of(ds, tag)
    bi = int(d20.bins(ds)[tag]["bin"])
    curves = b8.to_pipeline_curves(d)
    with m1.tempdir("cryoml_i21m_") as td_s:
        td = Path(td_s)
        lib = m.write_library(td, ds, {d.polarity: (bi, written(ds, tag, params))}, replica=True)
        lines = [f"* I21-1 measured-grid sweeps ({mode})\n", ".options scale=1.0\n",
                 f".options temp={m.DECK_TEMP_C[ds]:.4f}\n", f'.lib "{lib.as_posix()}" {m.LIB_SECTION[ds]}\n',
                 m.instance("m1", dev, "nd", "ng", "0", "0", mode, ds), "VG ng 0 DC 0\n", "VD nd 0 DC 0\n"]
        if mode == "elem":
            lines.append(m14.corr_models(v2.VA_MODULE_V2))
        lines += [".control\n"] + ([f"pre_osdi {osdi_path(ds).as_posix()}\n"] if mode == "elem" else [])
        lines += ["set filetype=ascii\n"]
        sw = sp._sweep_lines(curves, td, "raw_")
        if mode == "card_rep":
            sw = [ln.replace(" -i(VD)\n", " -i(VD) i(Vm1R)\n") for ln in sw]
        lines += sw + ["quit\n", ".endc\n", ".end\n"]
        r = m1.run_deck("".join(lines), td, "meas.sp", timeout_s=120)
        sims = sp._read_curves(td, "raw_", curves, d.polarity, False)
        sense = None
        if mode == "card_rep":
            sense = []
            for i, c in enumerate(curves):
                tgt = np.asarray(c.Vg if c.kind == "idvg" else c.Vd, float)
                rows = m1.read_wrdata(td / f"raw_{i}.dat", 4)
                if rows.shape[0] < 2:
                    sense.append(np.full(len(tgt), np.nan))
                    continue
                o = np.argsort(rows[:, 0])
                v = np.interp(tgt, rows[o, 0], rows[o, 3])
                sense.append(-v if d.polarity == "pmos" else v)
    return {"sims": sims, "sense": sense, "msgs": r["msgs"]}


def acc_task(task):
    from cryoml.explore import i8_1_nist as b8
    from cryoml.explore.e1_correction import score_device
    ds, tag, params, smoke = task
    _SMOKE["on"] = smoke
    d = d20.devices(ds)[tag]
    binfo = d20.bins(ds)[tag]
    corr = full_corr(ds)
    out = {"tag": tag, "msgs": []}
    runs = {mode: measured_run(ds, tag, params, mode) for mode in ("card", "card_rep", "elem")}
    for mode, rr in runs.items():
        out["msgs"] += rr["msgs"]
    rec_c = b8.to_record(d, binfo, runs["card"]["sims"], params)
    out["rrms_card_python"] = float(score_device(rec_c, corr.apply(rec_c))["rrms"])
    out["rrms_card_only"] = float(score_device(rec_c, [c.sim for c in rec_c.curves])["rrms"])
    rec_r = b8.to_record(d, binfo, runs["card_rep"]["sims"], params)
    pm = d.polarity == "pmos"
    chan = [m14.current_fwd(c.sim, ir, corr.g_v(pm, c.vg, c.vd)) for c, ir in zip(rec_r.curves, runs["card_rep"]["sense"])]
    out["rrms_python_channel"] = float(score_device(rec_r, chan)["rrms"])
    rec_e = b8.to_record(d, binfo, runs["elem"]["sims"], params)
    out["rrms_native"] = float(score_device(rec_e, [c.sim for c in rec_e.curves])["rrms"])
    return out


def stage_acc(ds: str, workers: int) -> dict:
    t0 = time.time()
    cards = jload(rdir(ds) / "cards_M2.json")
    m2 = jload(rdir(ds) / "m2.json")["rows"]
    tags = partA_tags(ds)
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(acc_task, [(ds, t, cards[t], _SMOKE["on"]) for t in tags], chunksize=1)
    rr = {r["tag"]: r for r in rows}
    diff = {t: abs(rr[t]["rrms_card_python"] - m2[t]["rrms"]) for t in rr}
    out = {"ds": ds, "rows": rr, "va1_max_diff": max(diff.values()), "va1_per_device": diff,
           "mean_native": float(np.mean([r["rrms_native"] for r in rows])),
           "mean_python_channel": float(np.mean([r["rrms_python_channel"] for r in rows])),
           "mean_card_python": float(np.mean([r["rrms_card_python"] for r in rows])),
           "mean_stored_python": float(np.mean([m2[t]["rrms"] for t in rr])),
           "max_abs_native_minus_stored": max(abs(rr[t]["rrms_native"] - m2[t]["rrms"]) for t in rr),
           "n_msgs": sum(len(r["msgs"]) for r in rows), "wall_s": round(time.time() - t0, 1)}
    jdump(out, rdir(ds) / "accuracy.json")
    print(json.dumps({k: out[k] for k in ("va1_max_diff", "mean_native", "mean_stored_python",
                                          "max_abs_native_minus_stored", "n_msgs")}, default=float), flush=True)
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def dev_check_a(ds: str, tags) -> dict:
    return d20.dev_check(ds, tags)


def stage_report(ds: str) -> dict:
    q12 = jload(rdir(ds) / "q12.json")["devices"]
    q3 = jload(rdir(ds) / "q3.json")
    comp = jload(rdir(ds) / "compile.json")
    m2 = jload(rdir(ds) / "m2.json")
    acc = jload(rdir(ds) / "accuracy.json")
    D = list(q12.values())
    elem_msgs = sum(len(v) for r in D for k, v in r["msgs"].items() if k.endswith("_elem"))
    ratios = {r["tag"]: float(np.median(r["wall"]["grid_fwd_default_elem"]) / np.median(r["wall"]["grid_fwd_default_card"]))
              for r in D}
    names = [n for n, *_ in m.CIRCUITS]
    circ = v2.circuit_runtime_ratios(q3["rows"], names)
    vds0 = m17.summarize_vds0({r["tag"]: r["vds0"] for r in D})
    tight = vds0["tight"]["elem_max"] if vds0["tight"]["n_nonfinite"] == 0 else float("inf")
    values = {"q1_tight_viol": int(sum(r["q1_tight_viol"] for r in D)),
              "q1_bad": int(elem_msgs + sum(r["q1_nonfinite"] for r in D)),
              "q1_vds0_max": tight,
              "g_text_max": comp["g_text_max"], "g_sym_text_max": comp["g_sym_text_max"],
              "swap_asym_max": max(r["swap_asym_element_max"] for r in D),
              "q2_ac_max": max(max(r["q2_err_gm"], r["q2_err_gds"]) for r in D),
              "cap_max": max(r["cap_max"] for r in D),
              "q2_gm_sign": int(sum(sum(r["q2_gm_curves_introduced"]) for r in D)),
              "q3_fail": v2.q3_fail_count(q3["rows"], names),
              "q1_default_max": max(r["q1_default_max_rel_1nA"] for r in D),
              "q2_gds_sign": int(sum(sum(r["q2_gds_curves_introduced"]) for r in D)),
              "runtime_ratio_max": max(list(ratios.values()) + list(circ.values()))}
    dec = m.decide_part_a(values)
    gds_curves = [[r["tag"], m.GDS_VGS[k]] for r in D for k, f in enumerate(r["q2_gds_curves_introduced"]) if f]
    gm_curves = [[r["tag"], m.GM_VDS[k]] for r in D for k, f in enumerate(r["q2_gm_curves_introduced"]) if f]
    rep = {}
    for r in D:
        for k, v in r.get("replica_nonchannel_max", {}).items():
            rep[k] = max(rep.get(k, 0.0), v)
    unit = run_unit_tests(UNIT_A)
    tags = sorted(q12)
    validity = m.validity_part_a({"va1_max_diff": acc["va1_max_diff"], "m2_failed": m2["n_failed"],
                                  "m2_n": len(m2["rows"]), "unit_tests_pass": unit["passed"], "n_rows": len(D),
                                  "n_q3": len(q3["rows"]), "dev_ok": dev_check_a(ds, tags)["ok"]}, ds)
    out = {"ds": ds, "validity": validity, "unit_tests": unit, "values": values, "decision": dec,
           "label": dec["deploy"] if validity["valid"] else "invalid", "vds0": vds0,
           "vds0_added_tight_max": max(r["q1_vds0_added_by_tol"]["tight"] for r in D),
           "runtime_grid": ratios, "runtime_circuits": circ, "gds_curves_introduced": gds_curves,
           "gm_curves_introduced": gm_curves, "n_gds_curves": len(D) * len(m.GDS_VGS),
           "n_gm_curves": len(D) * len(m.GM_VDS),
           "q3_not_ok": [k for k, v in q3["rows"].items() if not v["ok"]], "q3_devices": q3["devices"],
           "replica_nonchannel_max": rep,
           "replica_nonchannel_n_values": int(sum(r.get("replica_nonchannel_n_values", 0) for r in D)),
           "nonchannel_share_fwd_max": max(r["nonchannel_share_fwd_max"] for r in D),
           "card_rep_vs_card_default_max_abs": max(r["card_rep_vs_card_default_max_abs"] for r in D),
           "g_grid_range": [min(r["g_grid_min"] for r in D), max(r["g_grid_max"] for r in D)],
           "m2_in_sample_python_mean": m2["mean_rrms_python"], "n_devices": len(D),
           "accuracy": {k: acc[k] for k in ("va1_max_diff", "mean_native", "mean_python_channel",
                                            "mean_card_python", "mean_stored_python",
                                            "max_abs_native_minus_stored", "n_msgs")},
           "dev_check": dev_check_a(ds, tags)}
    jdump(out, rdir(ds) / "summary.json")
    print(json.dumps({"ds": ds, "validity": validity, "values": values, **dec}, indent=1, default=float), flush=True)
    return out


def stage_partA(ds: str, sub: str, workers: int) -> None:
    rdir(ds).mkdir(parents=True, exist_ok=True)
    subs = ("compile", "m2", "q12", "q3", "acc", "report") if sub == "all" else (sub,)
    timing = {}
    for st in subs:
        t1 = time.time()
        {"compile": lambda: stage_compile(ds), "m2": lambda: stage_m2(ds, workers),
         "q12": lambda: stage_q12(ds, workers), "q3": lambda: stage_q3(ds, workers),
         "acc": lambda: stage_acc(ds, workers), "report": lambda: stage_report(ds)}[st]()
        timing[st] = round(time.time() - t1, 1)
    tf = rdir(ds) / "timing.json"
    old = jload(tf) if tf.exists() else {}
    old.update(timing)
    jdump(old, tf)


# ===========================================================================
# Part B: TU Delft v1-recipe S3_D on state-C records
# ===========================================================================
S3_HYPER = {"seeds": (0,), "steps": 500}


def tud_records(t: int, state: str) -> list:
    from cryoml.explore.e1_correction import load_devices
    src = EXP / ("I18_2/start" if state == "C" else "I6_1/start") / f"records_{t}K"
    return load_devices(src)


def s3tud_task(task):
    import torch
    torch.set_num_threads(1)
    from cryoml.explore.i3_2_bias import fit_bias
    from cryoml.explore.i7_1_transfer import s3_fold_value
    t, state, k, steps = task
    recs = tud_records(t, state)
    order = [r.tag for r in recs]
    by = {r.tag: r for r in recs}
    grp = {r.tag: r.meta["group"] for r in recs}
    train = [x for x in order if grp[x] != k]
    scored = [x for x in order if grp[x] == k]
    t0 = time.time()
    corr = fit_bias("M_bias", [by[x] for x in train], hyper={**S3_HYPER, "steps": steps}, threads=1)
    return t, state, k, scored, s3_fold_value(corr, [by[x] for x in scored]), round(time.time() - t0, 2)


def stage_s3tud(workers: int) -> dict:
    from cryoml.explore import i19_1_dataset_gate as m19
    steps = 50 if _SMOKE["on"] else S3_HYPER["steps"]
    groups = {t: sorted({r.meta["group"] for r in tud_records(t, "C")}) for t in (4, 295)}
    tasks = []
    for t in (4, 295):
        gs = groups[t][:1] if _SMOKE["on"] else groups[t]
        tasks += [(t, "C", k, steps) for k in gs]
        tasks += [(t, "L", groups[t][0], steps)]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(s3tud_task, tasks, chunksize=1)
    split = jload(EXP / "I6_1/split.json")
    out = {"hyper": {**S3_HYPER, "steps": steps}, "smoke": _SMOKE["on"], "per_temperature": {},
           "wall_s": round(time.time() - t0, 1)}
    for t in (4, 295):
        per = {}
        for tt, st, k, sc, lr, _ in res:
            if tt == t and st == "C":
                per.update(dict(zip(sc, lr)))
        order = [r.tag for r in tud_records(t, "C")]
        vals = [per[x] for x in order if x in per]
        stored = jload(EXP / f"I19_1/s3/tud{t}_v1.json")
        chk = {}
        for tt, st, k, sc, lr, _ in res:
            if tt == t and st == "L":
                chk = {x: {"new": v, "stored": stored["per_record"][x], "diff": abs(v - stored["per_record"][x])}
                       for x, v in zip(sc, lr)}
        s3c = m19.s3_dataset(vals)
        tags = set(order)
        dev_ok = tags <= set(split["DEV"]) and not (tags & set(split["TEST"]))
        rep = m.part_b_report(s3c, float(stored["S3_D"]))
        vb = m.validity_part_b({"vb1_max_diff": max(v["diff"] for v in chk.values()), "dev_ok": dev_ok,
                                "n_finite": int(np.sum(np.isfinite(vals)))})
        out["per_temperature"][str(t)] = {**rep, "validity": vb, "per_record": per, "n_records": len(order),
                                         "n_scored": len(vals), "state_L_check": chk,
                                         "state_L_check_group": groups[t][0], "n_groups": len(groups[t])}
        print(f"TU Delft {t} K: S3_D v1 state C {s3c:+.4f} (state L {stored['S3_D']:+.4f}); "
              f"sign differs {rep['sign_differs']}; VB {vb}", flush=True)
    jdump(out, base_dir() / "s3tud" / "s3tud.json")
    return out


# ===========================================================================
# Part C: IHP SG13G2 233 / 343 / 398 K (I17-2 driver rebound)
# ===========================================================================
IHP_TEMPS = (233, 343, 398)
_T_ENV = os.environ.get("CRYOML_I21_1_IHP_T")


def bind_ihp(t: int) -> None:
    """Rebind the I17-2 driver to temperature ``t`` and output I21_1/ihp_<t>K."""
    d17.T = int(t)
    d17.OUT = OUT / f"ihp_{int(t)}K"        # smoke: d17.base_dir(True) = OUT / ihp_<t>K / smoke
    d17.PREREG = PREREG
    d17.UNIT_TESTS = UNIT_C
    d17._CACHE.pop("recs", None)


if _T_ENV:
    bind_ihp(int(_T_ENV))


def ihp_dir() -> Path:
    return d17.base_dir(_SMOKE["on"])


def stage_ihp_base(workers: int) -> dict:
    """C0_refit and M_bias v1 reruns (I5-1 recipe); VC1 vs I5_1/lodo_TK rows."""
    from cryoml.explore.e1_correction import Correction
    from cryoml.explore.i3_2_bias import merge_seed_results, split_seed_tasks
    rd = ihp_dir() / "base"
    smoke = _SMOKE["on"]
    order = d17.load_order()
    folds = list(m17b.SMOKE_FOLDS) if smoke else sorted(order)
    hyper = {"seeds": (0,), "steps": 200} if smoke else {}
    opts = dict(d17.OPTS, heldout_nfev=20) if smoke else dict(d17.OPTS)
    recs = d17.by_tag()
    unit = run_unit_tests(UNIT_C)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        bias_tasks = [("M_bias", [recs[t] for t in m17b.fold_train(order, h)]) for h in folds]
        expanded, owner = split_seed_tasks(bias_tasks, hyper)
        results = pool.map(d17.drv5.train_task, [("bias", a, r, hp) for a, r, hp in expanded], chunksize=1)
        corrs = {f"M_bias|{h}": c for h, c in zip(folds, merge_seed_results(len(bias_tasks), owner, results))}
        t_train = time.time() - t0
        tasks = [("C0_refit", Correction("C0"), h, opts) for h in folds]
        tasks += [("M_bias", corrs[f"M_bias|{h}"], h, opts) for h in folds]
        t1 = time.time()
        rows = pool.map(d17.refit_task, tasks, chunksize=1)
        t_refit = time.time() - t1
    per = {"C0_refit": {}, "M_bias": {}}
    for r in rows:
        per[r["label"]][r["tag"]] = r["rrms"]
    stored = jload(EXP / f"I5_1/lodo_{d17.T}K/results.json")["per_device"]
    vc1 = {a: {t: abs(per[a][t] - stored[a][t]) for t in per[a]} for a in per}
    out = {"T_K": d17.T, "smoke": smoke, "folds": folds, "opts": opts, "hyper_overrides": hyper, "per_device": per,
           "vc1": {"c0_max": max(vc1["C0_refit"].values()), "mb_max": max(vc1["M_bias"].values()), "n": len(folds),
                   "per_device": vc1},
           "unit_tests": unit,
           "rows": {f"{r['label']}|{r['tag']}": {k: v for k, v in r.items() if k != "params"} for r in rows},
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1)}}
    jdump(out, rd / "results.json")
    jdump({k: c.to_json() for k, c in corrs.items()}, rd / "corrections.json")
    jdump({f"{r['label']}|{r['tag']}": r["params"] for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"T": d17.T, "vc1": {k: out["vc1"][k] for k in ("c0_max", "mb_max", "n")},
                      "timing": out["timing"]}, indent=1), flush=True)
    return out


def stage_ihp_sel(workers: int) -> dict:
    """Selected g v2: per outer fold the 5-seed correction at lambda*_f
    (I17-2 inner selection) and its held-out refit."""
    smoke = _SMOKE["on"]
    rd = ihp_dir() / "sel"
    inner = jload(ihp_dir() / "inner" / "inner.json")
    order = d17.load_order()
    folds = list(m17b.SMOKE_FOLDS) if smoke else sorted(order)
    hyper = {"steps": 200} if smoke else {}
    seeds = (0,) if smoke else v2.HYPER_V2["seeds"]
    opts = dict(d17.OPTS, heldout_nfev=20) if smoke else dict(d17.OPTS)
    pens_dir = str(ihp_dir() / "pens")
    lam = {f: float(inner["lambda_star"][f]) for f in folds}
    names = {f: f"{m17b.variant(lam[f])}|{f}" for f in folds}
    tasks = [(names[f], m17b.fold_train(order, f), s, hyper, lam[f], pens_dir) for f in folds for s in seeds]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(d17.train_sym_task, tasks, chunksize=1)
        t_train = time.time() - t0
        corrs, secs = d17.merge_sym(trained, list(names.values()))
        t1 = time.time()
        rows = pool.map(d17.refit_task, [("selected", corrs[names[f]], f, opts) for f in folds], chunksize=1)
        t_refit = time.time() - t1
    per = {r["tag"]: r["rrms"] for r in rows}
    out = {"T_K": d17.T, "smoke": smoke, "lambda_star": lam, "names": names, "seeds": list(seeds),
           "hyper": v2.HYPER_V2, "hyper_overrides": hyper, "opts": opts, "per_device": per,
           "mean": float(np.mean(list(per.values()))),
           "heldout_rows": {r["tag"]: {k: v for k, v in r.items() if k != "params"} for r in rows},
           "train_seconds_per_member": secs,
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1), "n_seed_tasks": len(tasks)}}
    jdump(out, rd / "results.json")
    jdump({n: c.to_json() for n, c in corrs.items()}, rd / "corrections.json")
    jdump({r["tag"]: r["params"] for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"T": d17.T, "mean": out["mean"], "lambda": lam, "timing": out["timing"]}, indent=1), flush=True)
    return out


def stage_ihp_decide() -> dict:
    from cryoml.explore.i3_2_bias import correction_from_json, paired_ratio
    B = ihp_dir()
    base = jload(B / "base/results.json")
    inner = jload(B / "inner/inner.json")
    sel = jload(B / "sel/results.json")
    pj = jload(B / "pens/pens.json")
    order = d17.load_order()
    tags = sorted(base["folds"])
    rows = {"C0_refit": base["per_device"]["C0_refit"], "M_bias_v1": base["per_device"]["M_bias"],
            "selected": sel["per_device"]}
    split = jload(EXP / "I5_1/split.json")
    dev_ok = set(order) <= set(split["DEV"]) and not (set(order) & set(split["TEST"]))
    validity = m.validity_part_c({"vc1_c0_max": base["vc1"]["c0_max"], "vc1_mb_max": base["vc1"]["mb_max"],
                                  "vc1_n": base["vc1"]["n"], "unit_tests_pass": base["unit_tests"]["passed"],
                                  "rows": {k: len(v) for k, v in rows.items()},
                                  "inner_complete": bool(inner["complete"] and len(inner["lambda_star"]) == 18),
                                  "grids_ok": bool(pj["v4_ok"]), "dev_ok": dev_ok})
    cl = m17b.clusters(tags)
    n_failed = int(sum(not np.isfinite(rows[a][t]) for a in ("selected", "C0_refit") for t in tags))
    primary = paired_ratio(rows["selected"], rows["C0_refit"])
    secondary = paired_ratio(rows["selected"], rows["M_bias_v1"])
    g_v1 = paired_ratio(rows["M_bias_v1"], rows["C0_refit"])
    table = {}
    for a, r in rows.items():
        table[a] = dict(d17.agg(r))
        if a != "C0_refit":
            table[a]["gmr_vs_C0_refit"] = paired_ratio(r, rows["C0_refit"])
            table[a]["gmr_vs_C0_refit_cluster"] = paired_ratio(r, rows["C0_refit"], clusters=cl)
        if a != "M_bias_v1":
            table[a]["gmr_vs_v1"] = paired_ratio(r, rows["M_bias_v1"])
            table[a]["gmr_vs_v1_cluster"] = paired_ratio(r, rows["M_bias_v1"], clusters=cl)
    by_class = {}
    for c in ("lv", "hv"):
        s_ = [t for t in tags if t.startswith(c)]
        sub = lambda d: {t: d[t] for t in s_}  # noqa: E731
        by_class[c] = {"n": len(s_), "agg": {a: d17.agg(sub(r)) for a, r in rows.items()},
                       "selected_vs_C0_refit": paired_ratio(sub(rows["selected"]), sub(rows["C0_refit"])),
                       "selected_vs_v1": paired_ratio(sub(rows["selected"]), sub(rows["M_bias_v1"]))}
    pens = d17.load_pens(B / "pens")
    bc = jload(B / "base/corrections.json")
    sc = jload(B / "sel/corrections.json")
    negs = {"M_bias_v1": d17.neg_fractions(lambda t: correction_from_json(bc[f"M_bias|{t}"]), tags, pens, order,
                                           vdd_v1=True),
            "selected": d17.neg_fractions(lambda t: v2.SymCorrection.from_json(sc[sel["names"][t]]), tags, pens,
                                          order),
            "card_only": pj["card_negative_fraction"]}
    preds = jload(PREREG)["predictions"]
    T = str(d17.T)
    pred = {"ln GMR(selected, C0_refit)": m17b.sign_check(primary["ratio"], preds["C ln GMR(selected, C0_refit)"][T]),
            "ln GMR(selected, M_bias v1)": m17b.sign_check(secondary["ratio"], preds["C ln GMR(selected, M_bias v1)"][T])}
    out_lab = m17b.decide_outcome(validity, primary, n_failed, secondary)
    pred["primary label"] = out_lab["outcome"] == preds["C primary label"][T]
    pred["secondary label"] = out_lab.get("secondary") == preds["C secondary label"][T]
    cnt: dict = {}
    for t in tags:
        k = f"{float(inner['lambda_star'][t]):g}"
        cnt[k] = cnt.get(k, 0) + 1
    out = {"T_K": d17.T, "validity": validity, "outcome": out_lab["outcome"], "secondary": out_lab.get("secondary"),
           "primary_gmr_selected_vs_C0_refit": primary, "secondary_gmr_selected_vs_v1": secondary,
           "gmr_v1_vs_C0_refit": g_v1, "n_failed_selected_C0": n_failed, "prediction_check": pred,
           "lambda_star": inner["lambda_star"], "lambda_star_counts": cnt, "rows": rows, "table": table,
           "by_class": by_class, "clusters": cl, "negative_fractions": negs,
           "inner_timing": inner["timing"], "sel_timing": sel["timing"], "base_timing": base["timing"],
           "stored_I5_1": jload(EXP / f"I5_1/lodo_{d17.T}K/results.json")["comparisons"]["all"]["M_bias_vs_C0_refit"]}
    jdump(out, B / "decision.json")
    print(json.dumps({"T": d17.T, "validity": validity, "outcome": out["outcome"], "secondary": out["secondary"],
                      "primary": {k: primary[k] for k in ("ratio", "ci_low", "ci_high", "wins")},
                      "secondary_gmr": {k: secondary[k] for k in ("ratio", "ci_low", "ci_high", "wins")},
                      "lambda": cnt, "pred": pred}, indent=1, default=float), flush=True)
    return out


def stage_ihp(t: int, sub: str, workers: int) -> None:
    if t not in IHP_TEMPS:
        raise ValueError(t)
    if int(os.environ.get("CRYOML_I21_1_IHP_T", "0")) != t:
        os.environ["CRYOML_I21_1_IHP_T"] = str(t)        # spawned workers rebind at import
    bind_ihp(t)
    smoke = _SMOKE["on"]
    if sub == "pens":
        d17.stage_pens(workers, ihp_dir() / "pens")
    elif sub == "base":
        stage_ihp_base(workers)
    elif sub == "inner":
        d17.stage_inner(workers, ihp_dir() / "inner", smoke)
    elif sub == "sel":
        stage_ihp_sel(workers)
    else:
        stage_ihp_decide()


# ===========================================================================
# decide (all parts)
# ===========================================================================
def stage_decide() -> dict:
    B = base_dir()
    out = {"prereg": str(PREREG.relative_to(ROOT)), "part_A": {}, "part_B": None, "part_C": {}}
    preds = jload(PREREG)["predictions"]
    for ds in m.DATASETS:
        p = B / f"partA_{ds}" / "summary.json"
        if p.exists():
            s = jload(p)
            out["part_A"][ds] = {"validity": s["validity"], "label": s["label"], "core_fail": s["decision"]["core_fail"],
                                 "restrictions": s["decision"]["restrictions"], "values": s["values"],
                                 "gds_curves_introduced": s["gds_curves_introduced"],
                                 "n_gds_curves": s["n_gds_curves"], "q3_not_ok": s["q3_not_ok"],
                                 "runtime_circuits": s["runtime_circuits"], "accuracy": s["accuracy"],
                                 "prediction_label_correct": s["label"] == preds[f"A label {ds}"],
                                 "prediction_gds_correct": len(s["gds_curves_introduced"]) > 0}
    p = B / "s3tud" / "s3tud.json"
    if p.exists():
        s = jload(p)
        out["part_B"] = {t: {k: v for k, v in r.items() if k not in ("per_record", "state_L_check")}
                         for t, r in s["per_temperature"].items()}
        for t, r in out["part_B"].items():
            r["prediction_sign_correct"] = bool(r["S3_D_state_C"] < 0)
    for t in IHP_TEMPS:
        p = B / f"ihp_{t}K" / "decision.json"
        if p.exists():
            s = jload(p)
            out["part_C"][str(t)] = {k: s[k] for k in ("validity", "outcome", "secondary",
                                                       "primary_gmr_selected_vs_C0_refit",
                                                       "secondary_gmr_selected_vs_v1", "gmr_v1_vs_C0_refit",
                                                       "lambda_star_counts", "prediction_check")}
    jdump(out, B / "decision.json")
    print(json.dumps(out, indent=1, default=float)[:6000], flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("partA", "s3tud", "ihp", "decide"))
    ap.add_argument("--ds", choices=m.DATASETS)
    ap.add_argument("--sub", default="all")
    ap.add_argument("--temp", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="reduced settings (code check; not used for decisions)")
    args = ap.parse_args()
    if args.smoke:
        os.environ["CRYOML_I21_1_SMOKE"] = "1"
        _SMOKE["on"] = True
    t0 = time.time()
    if args.stage == "partA":
        stage_partA(args.ds, args.sub, args.workers)
    elif args.stage == "s3tud":
        stage_s3tud(args.workers)
    elif args.stage == "ihp":
        stage_ihp(args.temp, args.sub, args.workers)
    else:
        stage_decide()
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
