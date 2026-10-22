#!/usr/bin/env python3
"""I18-2 Part B: deployment checks (I17-1 rules) of the full-data g v2 at
lambda*_full on the TU Delft BSIM4 card, ngspice-41 OSDI build, per
temperature.  Q3 circuits not run (supply).

Pre-registration: out/research/explore/I18_2/prereg.md (prereg.json).

Stages (``--stage``, ``--temp T``):

  compile  OSDI module of the full-data g; Verilog-A g vs Python and the
           exchanged-terminal g check -> partB_<T>K/compile.json
  m2       M2 cards (starting card refit with the full-data g in the loop,
           state C) -> cards_M2.json, m2.json
  q12      Q1 grids (serpentine, 3 tolerances, forward / exchanged), Q2 AC
           and dense sweeps, replica pre-check -> q12.json, arrays/
  acc      element-deck RRMS of the 45 M2 pairs and the 45 selected
           held-out pairs (fold g + held-out card) -> accuracy.json
  report   summary values and the Part B label -> summary.json
  all      compile, m2, q12, acc, report
  decide   both temperatures -> partB_decision.json

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i18_2_partb.py --stage all --temp 4 --workers 4
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

import i18_2_tud as drv  # noqa: E402  (records, devices, ProblemC, refit_task, clamp state)
from cryoml.explore import i11_1_deploy as m1  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i14_1_chan as m14  # noqa: E402
from cryoml.explore import i17_1_vds0 as m17  # noqa: E402
from cryoml.explore import i18_2_partb as m  # noqa: E402
from cryoml.explore import i18_2_tud as mt  # noqa: E402

jload, jdump = drv.jload, drv.jdump
UNIT_TESTS = ("tests.test_i18_2_partb", "tests.test_i18_2_tud", "tests.test_i14_1_chan", "tests.test_i12_1_symm")
TOLS = (("default", ""), ("r4", m1.R4), ("tight", m1.TIGHT))
SMOKE_N = 2


def rdir(t: int) -> Path:
    return drv.base_dir(drv._SMOKE["on"]) / f"partB_{t}K"


def full_corr(t: int) -> v2.SymCorrection:
    js = jload(drv.base_dir(drv._SMOKE["on"]) / f"full_{t}K" / "corrections.json")
    (key, row), = js.items()
    return v2.SymCorrection.from_json(row)


def fold_corr(t: int, tag: str) -> tuple[v2.SymCorrection, dict]:
    """Selected held-out pair: fold g of sym_e1_l{lambda*_G} and its held-out card."""
    B = drv.base_dir(drv._SMOKE["on"])
    g = drv.group_of(tag)
    lam = float(drv.lambda_star(t)[g])
    name = mt.variant(lam)
    d = B / f"lodo_{t}K_{name}"
    return (v2.SymCorrection.from_json(jload(d / "corrections.json")[f"{name}|{g}"]),
            dict(jload(d / "heldout_cards.json")[tag]))


def osdi_path(t: int, key: str) -> Path:
    return rdir(t) / "osdi" / f"{key}.osdi"


def write_cards(td: Path, dev, params: dict) -> tuple[Path, Path]:
    card = td / "card.mod"
    rep = td / "rep.mod"
    card.write_text(m.card_text(dev.dev_type, params))
    rtext = m.card_text(dev.dev_type, params, replica=True)
    if not m.replica_ok(rtext):
        raise RuntimeError("replica text lacks a REPLICA value")
    rep.write_text(rtext)
    return card, rep


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------
def stage_compile(t: int) -> dict:
    corr = full_corr(t)
    text = v2.to_verilog_a_v2(corr)
    p = osdi_path(t, "full")
    c = m1.compile_osdi(text, p.with_suffix(".va"), p)
    rng = np.random.default_rng(20261022)
    worst, worst_sym = 0.0, 0.0
    for _ in range(2000):
        q = int(rng.integers(0, 2))
        vgs, vds = rng.uniform(-mt.VDD, mt.VDD, size=2)
        gt = v2.eval_verilog_a_v2(text, q, vgs, vds)
        gp = float(corr.g_v(bool(q), vgs, vds))
        gs = v2.eval_verilog_a_v2(text, q, vgs - vds, -vds)
        worst = max(worst, abs(gt - gp))
        worst_sym = max(worst_sym, abs(gs - gt))
    out = {"T_K": t, "sha256": c["sha256"], "compiled": bool(c["compiled"]), "lines": len(text.splitlines()),
           "vdd": corr.vdd, "eps": corr.eps, "g_text_max": worst, "g_sym_text_max": worst_sym, "n_points": 2000}
    jdump(out, rdir(t) / "compile.json")
    print(f"compile {t} K: |g text - python| {worst:.2e}, exchanged {worst_sym:.2e}", flush=True)
    return out


# ---------------------------------------------------------------------------
# M2 cards
# ---------------------------------------------------------------------------
def m2_task(task):
    tag, t, opts = task
    return drv.refit_task(("M2", full_corr(t), tag, t, opts, "C"))


def stage_m2(t: int, tags, workers: int, opts: dict) -> dict:
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(m2_task, [(x, t, opts) for x in tags], chunksize=1)
    cards = {r["tag"]: r["params"] for r in rows}
    jdump(cards, rdir(t) / "cards_M2.json")
    out = {"T_K": t, "opts": opts, "wall_s": round(time.time() - t0, 1),
           "rows": {r["tag"]: {k: v for k, v in r.items() if k != "params"} for r in rows},
           "n_failed": int(sum(bool(r["failed"]) or not np.isfinite(r["rrms"]) for r in rows)),
           "mean_rrms_python": float(np.mean([r["rrms"] for r in rows]))}
    jdump(out, rdir(t) / "m2.json")
    print(f"M2 {t} K: {len(rows)} refits, {out['wall_s']} s, failed {out['n_failed']}, in-sample mean "
          f"{out['mean_rrms_python']:.5f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Q1 / Q2
# ---------------------------------------------------------------------------
def q12_task(task):
    tag, t, params = task
    dev = drv.devices()[tag]
    corr = full_corr(t)
    osdi = osdi_path(t, "full")
    temp = dev.temps_K[t]
    res = {"tag": tag, "msgs": {}, "wall": {}}
    a = {}
    with m1.tempdir("cryoml_i18b_") as td_s:
        td = Path(td_s)
        card, rep = write_cards(td, dev, params)

        def go(label, body, control, mode, options="", repeats=1, out_file=None):
            walls, r = [], None
            el = mode == "elem"
            for _ in range(repeats):
                if out_file is not None and Path(out_file).exists():
                    Path(out_file).unlink()          # serpentine decks append: one run per file
                r = m1.run_deck(m.deck(label, temp, card, None if mode == "card" else rep, body, control,
                                       osdi if el else None, options), td, f"{label}.sp",
                                timeout_s=m1.TIMEOUT_DEVICE_S)
                walls.append(r["wall_s"])
            res["msgs"][label] = r["msgs"]
            res["wall"][label] = walls
            return r

        f = td / "grid_fwd_default_card.dat"
        go("grid_fwd_default_card", m.device_body(dev, "card"), m.grid_control_serp(dev.dev_type, f, False),
           "card", "", repeats=3, out_file=f)
        a["grid_fwd_default_card"] = m.grid_rows(m1.read_wrdata(f, 2))[..., 0]
        for mode in ("card_rep", "elem"):
            sense = mode == "card_rep"
            nc = 3 if sense else 2
            for tol, opt in TOLS:
                lab = f"grid_fwd_{tol}_{mode}"
                f = td / f"{lab}.dat"
                go(lab, m.device_body(dev, mode), m.grid_control_serp(dev.dev_type, f, sense), mode, opt,
                   repeats=3 if (tol == "default" and mode == "elem") else 1, out_file=f)
                g = m.grid_rows(m1.read_wrdata(f, nc + (1 if sense else 0)))
                a[lab] = g[..., 0]
                if sense:
                    a[lab + "_rep"] = g[..., -1]
            lab = f"grid_swap_tight_{mode}"
            f = td / f"{lab}.dat"
            go(lab, m.device_body(dev, mode, swap=True), m.grid_control_serp(dev.dev_type, f, sense), mode, m1.TIGHT,
               out_file=f)
            g = m.grid_rows(m1.read_wrdata(f, nc + (1 if sense else 0)))
            a[lab] = g[..., 0]
            if sense:
                a[lab + "_rep"] = g[..., -1]
        pts = m.op_points(dev.dev_type)
        for mode in ("card_rep", "elem"):
            for ex in ("VG", "VD"):
                lab = f"ac_{ex}_{mode}"
                r = go(lab, m.device_body(dev, mode), m14.ac_control(pts, ex, mode == "card_rep"), mode, m1.TIGHT)
                for k, v in m14.parse_ac(r["text"], len(pts)).items():
                    a[f"{lab}_{k}"] = v
        for mode in ("card", "elem"):
            lab = f"dense_{mode}"
            pre = td / lab
            go(lab, m.device_body(dev, mode), m.dense_control(dev.dev_type, pre), mode, m1.TIGHT)
            for k in range(len(m.GM_VDS)):
                a[f"{lab}_gm{k}"] = m1.read_wrdata(Path(f"{pre}_gm{k}.dat"))
            for k in range(len(m.GDS_VGS)):
                a[f"{lab}_gds{k}"] = m1.read_wrdata(Path(f"{pre}_gds{k}.dat"))
        r = go("repcheck", m.device_body(dev, "card_rep"), m.nonchannel_control(pts[::4]), "card_rep", m1.TIGHT)
        res["replica_nonchannel_max"] = m.parse_nonchannel(r["text"])
    res.update(q12_metrics(dev, corr, pts, a))
    (rdir(t) / "arrays").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rdir(t) / "arrays" / f"q12_{tag}.npz", pts=pts,
                        **{k: v for k, v in a.items() if isinstance(v, np.ndarray)})
    return res


def q12_metrics(dev, corr, pts, a) -> dict:
    """I14-1 ``q12_metrics`` on the TU Delft grids and op-point sets."""
    pm = dev.dev_type == "pmos"
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
    vt0 = {tol: float(np.nanmax(np.abs(a[f"grid_fwd_{tol}_elem"][:, j0]))) for tol, _ in TOLS}
    out["q1_vds0_by_tol"] = vt0          # raw element current at V(d,s) = 0 (reported)
    # amendment A1: the current the element adds at V(d,s) = 0 (element run minus card + replica run)
    add = {tol: np.abs(a[f"grid_fwd_{tol}_elem"][:, j0] - a[f"grid_fwd_{tol}_card_rep"][:, j0]) for tol, _ in TOLS}
    out["q1_vds0_added_by_tol"] = {tol: float(np.nanmax(v)) for tol, v in add.items()}
    out["q1_vds0_added_tight_n_above"] = int(np.sum(add["tight"] > m17.VDS0_THR))
    out["q1_vds0_tight_nonfinite"] = int(np.sum(~np.isfinite(add["tight"])))
    out["q1_vds0_max_card"] = float(max(np.nanmax(np.abs(a[f"grid_fwd_{x}_card_rep"][:, j0]))
                                        for x in ("default", "tight")))
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
    i_card, i_rep = a["ac_VG_card_rep_i0"], a["ac_VG_card_rep_s0"]
    yc_dg = a["ac_VG_card_rep_yr"] + 1j * a["ac_VG_card_rep_ym"]
    yc_dd = a["ac_VD_card_rep_yr"] + 1j * a["ac_VD_card_rep_ym"]
    yr_dg = a["ac_VG_card_rep_sr"] + 1j * a["ac_VG_card_rep_sm"]
    yr_dd = a["ac_VD_card_rep_sr"] + 1j * a["ac_VD_card_rep_sm"]
    ye_dg = a["ac_VG_elem_yr"] + 1j * a["ac_VG_elem_ym"]
    ye_dd = a["ac_VD_elem_yr"] + 1j * a["ac_VD_elem_ym"]
    yp_dg = m14.admittance(yc_dg, yr_dg, i_rep, gp, dgg)
    yp_dd = m14.admittance(yc_dd, yr_dd, i_rep, gp, dgd)
    ipy = m14.current_fwd(i_card, i_rep, gp)
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


def stage_q12(t: int, tags, workers: int) -> dict:
    cards = jload(rdir(t) / "cards_M2.json")
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(q12_task, [(x, t, cards[x]) for x in tags], chunksize=1)
    out = {"T_K": t, "wall_s": round(time.time() - t0, 1), "devices": {r["tag"]: r for r in rows}}
    jdump(out, rdir(t) / "q12.json")
    print(f"q12 {t} K: {len(rows)} devices, {out['wall_s']} s", flush=True)
    return out


# ---------------------------------------------------------------------------
# accuracy (measured-curve decks with the element; reported)
# ---------------------------------------------------------------------------
def measured_run(tag: str, t: int, params: dict, mode: str, osdi: Path | None) -> dict:
    from cryoml.explore import i6_1_tudelft as m6
    from cryoml.explore.i5_1_ihp_real import sweep_lines
    dev = drv.devices()[tag]
    curves = m6.fit_curves(dev, t)
    with m1.tempdir("cryoml_i18m_") as td_s:
        td = Path(td_s)
        card, rep = write_cards(td, dev, params)
        lines = [f"* I18-2 measured-grid sweeps ({mode})\n", f".options temp={dev.temps_K[t] - 273.15:.6g}\n",
                 f'.include "{card.as_posix()}"\n']
        if mode != "card":
            lines.append(f'.include "{rep.as_posix()}"\n')
        lines.append(m.instance("m1", dev, "nd", "ngt", "0", "nb", mode))
        lines += ["VG ngt 0 DC 0\n", "VD nd 0 DC 0\n", "VB nb 0 DC 0\n"]
        if mode == "elem":
            lines.append(m14.corr_models(v2.VA_MODULE_V2))
        lines += [".control\n"] + ([f"pre_osdi {osdi.as_posix()}\n"] if mode == "elem" else [])
        lines += ["set filetype=ascii\n"]
        sw = sweep_lines(curves, td, "raw_")
        if mode == "card_rep":
            sw = [ln.replace(" -i(VD)\n", " -i(VD) i(Vm1R)\n") for ln in sw]
        lines += sw + ["quit\n", ".endc\n", ".end\n"]
        r = m1.run_deck("".join(lines), td, "meas.sp", timeout_s=120)
        sims, sense = [], []
        for i, c in enumerate(curves):
            tgt = np.asarray(c.sweep, float)
            rows = m1.read_wrdata(td / f"raw_{i}.dat", 4 if mode == "card_rep" else 2)
            if rows.shape[0] < 2:
                sims.append(np.full(len(tgt), np.nan))
                sense.append(np.full(len(tgt), np.nan))
                continue
            o = np.argsort(rows[:, 0])
            sims.append(np.interp(tgt, rows[o, 0], rows[o, 1]))
            sense.append(np.interp(tgt, rows[o, 0], rows[o, -1]) if mode == "card_rep" else None)
    return {"sims": sims, "sense": sense, "msgs": r["msgs"], "curves": curves}


def acc_task(task):
    from cryoml.explore import i6_1_tudelft as m6
    kind, tag, t, rd_s = task
    dev = drv.devices()[tag]
    pm = dev.dev_type == "pmos"
    if kind == "M2":
        corr = full_corr(t)
        params = jload(Path(rd_s) / "cards_M2.json")[tag]
        osdi = osdi_path(t, "full")
    else:
        corr, params = fold_corr(t, tag)
        osdi = osdi_path(t, f"fold_{drv.group_of(tag)}")
    cr = measured_run(tag, t, params, "card_rep", None)
    el = measured_run(tag, t, params, "elem", osdi)
    curves = cr["curves"]
    py = [np.asarray(ic) * np.exp(corr.g_v(pm, c.vg, c.vd)) for ic, c in zip(cr["sims"], curves)]
    chan = [m14.current_fwd(ic, ir, corr.g_v(pm, c.vg, c.vd)) for ic, ir, c in zip(cr["sims"], cr["sense"], curves)]
    return {"kind": kind, "tag": tag, "rrms_card_only": m6.score(curves, cr["sims"])["rrms"],
            "rrms_python": m6.score(curves, py)["rrms"], "rrms_python_channel": m6.score(curves, chan)["rrms"],
            "rrms_native": m6.score(curves, el["sims"])["rrms"], "msgs": cr["msgs"] + el["msgs"]}


def compile_fold_task(task):
    t, g, smoke = task
    if smoke:
        os.environ["CRYOML_I18_2_SMOKE"] = "1"
        drv._SMOKE["on"] = True
    tag = next(x for x in drv.load_order(t) if drv.group_of(x) == g)
    corr, _ = fold_corr(t, tag)
    p = osdi_path(t, f"fold_{g}")
    c = m1.compile_osdi(v2.to_verilog_a_v2(corr), p.with_suffix(".va"), p)
    return {"group": g, "compiled": bool(c["compiled"]), "sha256": c["sha256"]}


def stage_acc(t: int, tags, workers: int) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    t0 = time.time()
    gs = sorted({drv.group_of(x) for x in tags})
    with get_context("spawn").Pool(workers) as pool:
        comp = pool.map(compile_fold_task, [(t, g, drv._SMOKE["on"]) for g in gs], chunksize=1)
        rows = pool.map(acc_task, [(k, x, t, str(rdir(t))) for k in ("M2", "sel") for x in tags], chunksize=1)
    dec = jload(drv.base_dir(drv._SMOKE["on"]) / "decision.json")["per_temperature"][str(t)] \
        if (drv.base_dir(drv._SMOKE["on"]) / "decision.json").exists() else None
    m2 = jload(rdir(t) / "m2.json")["rows"]
    out = {"T_K": t, "compile": comp, "rows": {f"{r['kind']}|{r['tag']}": r for r in rows}}
    for k in ("M2", "sel"):
        rr = {r["tag"]: r for r in rows if r["kind"] == k}
        stored = {x: m2[x]["rrms"] for x in rr} if k == "M2" else (
            {x: dec["selected_rows"][x] for x in rr} if dec else {})
        out[k] = {"mean_native": float(np.mean([r["rrms_native"] for r in rr.values()])),
                  "mean_python_rerun": float(np.mean([r["rrms_python"] for r in rr.values()])),
                  "mean_card_only": float(np.mean([r["rrms_card_only"] for r in rr.values()])),
                  "mean_stored_python": float(np.mean(list(stored.values()))) if stored else None,
                  "max_abs_native_minus_stored": (max(abs(rr[x]["rrms_native"] - stored[x]) for x in rr)
                                                  if stored else None),
                  "n_msgs": sum(len(r["msgs"]) for r in rr.values())}
        if k == "sel" and dec:
            nat = {x: r["rrms_native"] for x, r in rr.items()}
            out[k]["gmr_native_vs_C0_refit_cluster"] = paired_ratio(
                nat, dec["rows"]["C0_refit"], clusters={x: drv.group_of(x) for x in nat})
    out["wall_s"] = round(time.time() - t0, 1)
    jdump(out, rdir(t) / "accuracy.json")
    print(json.dumps({k: out[k] for k in ("M2", "sel")}, indent=1, default=float), flush=True)
    return out


# ---------------------------------------------------------------------------
# report / decide
# ---------------------------------------------------------------------------
def run_unit_tests() -> dict:
    r = subprocess.run([sys.executable, "-m", "unittest", *UNIT_TESTS], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    return {"files": list(UNIT_TESTS), "rc": r.returncode, "passed": r.returncode == 0, "tail": tail}


def stage_report(t: int) -> dict:
    q12 = jload(rdir(t) / "q12.json")["devices"]
    comp = jload(rdir(t) / "compile.json")
    m2 = jload(rdir(t) / "m2.json")
    D = list(q12.values())
    elem_msgs = sum(len(v) for r in D for k, v in r["msgs"].items() if k.endswith("_elem"))
    ratios = {r["tag"]: float(np.median(r["wall"]["grid_fwd_default_elem"]) / np.median(r["wall"]["grid_fwd_default_card"]))
              for r in D}
    tight = max(r["q1_vds0_added_by_tol"]["tight"] for r in D)      # amendment A1
    if any(r["q1_vds0_tight_nonfinite"] for r in D):
        tight = float("inf")
    values = {"q1_tight_viol": int(sum(r["q1_tight_viol"] for r in D)),
              "q1_bad": int(elem_msgs + sum(r["q1_nonfinite"] for r in D)),
              "q1_vds0_max": tight,
              "g_text_max": comp["g_text_max"], "g_sym_text_max": comp["g_sym_text_max"],
              "swap_asym_max": max(r["swap_asym_element_max"] for r in D),
              "q2_ac_max": max(max(r["q2_err_gm"], r["q2_err_gds"]) for r in D),
              "cap_max": max(r["cap_max"] for r in D),
              "q2_gm_sign": int(sum(sum(r["q2_gm_curves_introduced"]) for r in D)),
              "q1_default_max": max(r["q1_default_max_rel_1nA"] for r in D),
              "q2_gds_sign": int(sum(sum(r["q2_gds_curves_introduced"]) for r in D)),
              "runtime_ratio_max": max(ratios.values())}
    dec = m.decide_partb(values)
    gds_curves = [[r["tag"], m.GDS_VGS[k]] for r in D for k, f in enumerate(r["q2_gds_curves_introduced"]) if f]
    rep = {}
    for r in D:
        for k, v in r.get("replica_nonchannel_max", {}).items():
            rep[k] = max(rep.get(k, 0.0), v)
    unit = run_unit_tests()
    validity = m.decide_validity_b({"m2_failed": m2["n_failed"], "m2_n": len(m2["rows"]), "unit_tests_pass": unit["passed"],
                                    "n_rows": len(D)})
    out = {"T_K": t, "validity": validity, "unit_tests": unit, "values": values, "decision": dec,
           "vds0_raw_element_tight_max": max(r["q1_vds0_by_tol"]["tight"] for r in D),
           "vds0_card_max": max(r["q1_vds0_max_card"] for r in D),
           "vds0_added_default_max": max(r["q1_vds0_added_by_tol"]["default"] for r in D),
           "vds0_default_max": max(r["q1_vds0_by_tol"]["default"] for r in D),
           "vds0_r4_max": max(r["q1_vds0_by_tol"]["r4"] for r in D),
           "runtime_grid": ratios, "gds_curves_introduced": gds_curves,
           "replica_nonchannel_max": rep, "nonchannel_share_fwd_max": max(r["nonchannel_share_fwd_max"] for r in D),
           "card_rep_vs_card_default_max_abs": max(r["card_rep_vs_card_default_max_abs"] for r in D),
           "m2_in_sample_python_mean": m2["mean_rrms_python"], "n_devices": len(D),
           "lambda_star_full": float(jload(drv.base_dir(drv._SMOKE["on"]) / f"full_{t}K/full.json")["lambda_star_full"])}
    jdump(out, rdir(t) / "summary.json")
    print(json.dumps({"T": t, "validity": validity, "values": values, **dec}, indent=1, default=float), flush=True)
    return out


def stage_decide() -> dict:
    B = drv.base_dir(drv._SMOKE["on"])
    out = {"prereg": "out/research/explore/I18_2/prereg.json", "per_temperature": {}}
    for t in mt.TEMPS:
        p = B / f"partB_{t}K/summary.json"
        if not p.exists():
            continue
        s = jload(p)
        acc = jload(B / f"partB_{t}K/accuracy.json") if (B / f"partB_{t}K/accuracy.json").exists() else {}
        out["per_temperature"][str(t)] = {
            "validity": s["validity"],
            "label": (s["decision"]["label_full"] if s["validity"]["valid"] else "invalid"),
            "core_fail": s["decision"]["core_fail"], "restrictions": s["decision"]["restrictions"],
            "q3": s["decision"]["q3"], "values": s["values"], "lambda_star_full": s["lambda_star_full"],
            "accuracy": {k: acc.get(k) for k in ("M2", "sel")}}
    jdump(out, B / "partB_decision.json")
    print(json.dumps(out, indent=1, default=float), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("compile", "m2", "q12", "acc", "report", "all", "decide"))
    ap.add_argument("--temp", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="2 devices, refit max_nfev 20 (code check only)")
    args = ap.parse_args()
    if args.smoke:
        os.environ["CRYOML_I18_2_SMOKE"] = "1"
        drv._SMOKE["on"] = True
    t = args.temp
    tags = drv.load_order(t)
    if args.smoke:
        tags = tags[:SMOKE_N]
    opts = dict(drv.OPTS, heldout_nfev=20) if args.smoke else dict(drv.OPTS)
    rdir(t).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stages = ("compile", "m2", "q12", "acc", "report") if args.stage == "all" else (args.stage,)
    timing = {}
    for st in stages:
        t1 = time.time()
        if st == "compile":
            stage_compile(t)
        elif st == "m2":
            stage_m2(t, tags, args.workers, opts)
        elif st == "q12":
            stage_q12(t, tags, args.workers)
        elif st == "acc":
            stage_acc(t, tags, args.workers)
        elif st == "report":
            stage_report(t)
        else:
            stage_decide()
        timing[st] = round(time.time() - t1, 1)
    if args.stage != "decide":
        tf = rdir(t) / "timing.json"
        old = jload(tf) if tf.exists() else {}
        old.update(timing)
        jdump(old, tf)
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
