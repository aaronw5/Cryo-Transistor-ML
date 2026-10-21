#!/usr/bin/env python3
"""I11-1: 43-parameter card + shared bias-only correction (M_bias) as a
compact-model add-on in ngspice-41 + OSDI, sky130 77 K, all 18 devices.

Pre-registration: out/research/explore/I11_1/prereg.md (prereg.json).

Stages (``--stage``; ``all`` runs them in order):

  compile  OpenVAF 23.5.0 compile of the stored full-data model
           (I3_2/all18/m_bias_sky130_all18_full.va); g of the emitted text vs
           Python on 2000 random bias points.
  refit    M_full cards: per device, the 43-parameter card refit from the
           direct43_r3_fd start with the full-data g frozen in the loop (I3-2
           refit recipe: max_nfev 400, irls 2, diff_step 0.01; in-sample).
           In-sample RRMS: Python (card x exp(g)) and SPICE-native (OSDI).
  q12      per device (M_full card): dense Vgs x Vds grid (Q1), op-point AC
           small signal and 1 mV sweeps (Q2), grid wall time (Q4).
  q3       circuits (inverter VTC x2, 5-stage ring oscillator x2, current
           mirror x2) x arms published / C0_refit / M_card / M_full; wall time
           of M_card vs M_full (Q4).
  report   summary.json + decision.json.
  diagnose post hoc: location of core-check failures, output-conductance
           slopes vs measured curves (diagnosis.json; no effect on the decision).

  PYTHONPATH=src .venv/bin/python -I scripts/research/explore/i11_1_deploy.py --stage all --workers 4
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
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryoml.explore import i11_1_deploy as m  # noqa: E402

EXP = ROOT / "out/research/explore"
ALL18 = EXP / "I3_2/all18"
OUT = EXP / "I11_1"
FULL_VA = ALL18 / "m_bias_sky130_all18_full.va"
START = ROOT / "out/research/polished/direct43_r3_fd/real/summary.json"
REFIT_OPTS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
SMOKE_DEVICES = ("nmos_L1_W1p6", "pmos_L2_W5")
REPEATS = 3

# Q3 circuits: (name, kind, devices, settings)
CIRCUITS = (
    ("inv_A_vtc", "vtc", ("nmos_L0p15_W1p6", "pmos_L0p35_W1p6"), {}),
    ("inv_B_vtc", "vtc", ("nmos_L1_W1p6", "pmos_L2_W5"), {}),
    ("ro5_A", "ring", ("nmos_L0p15_W1p6", "pmos_L0p35_W1p6"), {"tstep": 1e-12, "tstop": 10e-9}),
    ("ro5_B", "ring", ("nmos_L1_W1p6", "pmos_L2_W5"), {"tstep": 5e-12, "tstop": 50e-9}),
    ("mirror_n", "mirror", ("nmos_L1_W1p6",), {"iref": 10e-6}),
    ("mirror_p", "mirror", ("pmos_L2_W5",), {"iref": 10e-6}),
)
Q3_ARMS = ("published", "C0_refit", "M_card", "M_full")


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def run_dir(name: str) -> Path:
    return OUT / "runs" / name


def osdi_path() -> Path:
    return OUT / "osdi" / "m_bias_sky130_all18_full.osdi"


def full_corr():
    from cryoml.explore.i3_2_bias import correction_from_json
    return correction_from_json(jload(ALL18 / "corrections.json")["full|M_bias"])


def devices() -> dict:
    from cryoml.card_eval import device_by_tag
    from cryoml.cold_start import device_bins
    bins = device_bins()
    out = {}
    for tag in sorted(bins):
        d = device_by_tag(tag)
        out[tag] = m.Dev(tag, d.dev_type, float(d.L_um), float(d.W_um), int(bins[tag]))
    return out


def load_records() -> dict:
    from cryoml.explore.e1_correction import load_devices
    recs = load_devices(EXP / "E1/real_dev") + load_devices(EXP / "E1/real_test")
    start = jload(START)["devices"]
    for r in recs:
        r.meta["params"] = dict(start[r.tag]["params"])
    return {r.tag: r for r in recs}


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------
def stage_compile(rd: Path) -> dict:
    from cryoml.explore.i3_2_bias import eval_verilog_a_g
    text = FULL_VA.read_text()
    c = m.compile_osdi(text, OUT / "osdi" / "m_bias_sky130_all18_full.va", osdi_path())
    corr = full_corr()
    rng = np.random.default_rng(20261021)
    worst = 0.0
    for _ in range(2000):
        p = int(rng.integers(0, 2))
        vgs, vds = rng.uniform(-1.85, 1.85, size=2)
        gv = eval_verilog_a_g(text, p, vgs, vds)
        gp = float(m.g_values(corr, bool(p), vgs, vds)[0])
        worst = max(worst, abs(gv - gp))
    out = {"va": str(FULL_VA.relative_to(ROOT)), "osdi": str(osdi_path().relative_to(ROOT)),
           "sha256": c["sha256"], "compiled_now": bool(c["compiled"]),
           "max_abs_g_diff_text_vs_python": worst, "n_points": 2000}
    jdump(out, rd / "compile.json")
    print(f"compile: max |g text - g python| {worst:.2e}", flush=True)
    return out


# ---------------------------------------------------------------------------
# refit (M_full cards) and in-sample RRMS
# ---------------------------------------------------------------------------
def refit_task(task):
    tag, opts = task
    from i2_1_joint import heldout_task
    rec = load_records()[tag]
    row, _ = heldout_task(("M_full", None, full_corr(), rec, opts))
    return row


def native_task(task):
    """In-sample RRMS of the M_full pair on the measured curves: SPICE-native
    (OSDI element) and Python (card current x exp(g)); card-only RRMS."""
    tag, params = task
    from cryoml.card_eval import device_by_tag
    from cryoml.data_io import load_device_curves
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i2_1_joint import with_sims
    from cryoml.explore.i3_2_followup import simulate_sky130
    from cryoml.metrics import tag_metric_curves
    rec = load_records()[tag]
    dev = devices()[tag]
    device = device_by_tag(tag)
    curves = load_device_curves(device)
    pos = {id(c): i for i, c in enumerate(curves)}
    order = [pos[id(c)] for _, c in sorted(tag_metric_curves(curves).items())]

    def run(osdi):
        sims, msgs = simulate_sky130(dev.dev_type, dev.L_um, dev.W_um, curves, params, dev.bin,
                                     m.NG_OSDI, osdi=osdi, vdd=m.VDD_NORM, pmos_param="is_pmos")
        return [np.asarray(sims[i], float) for i in order], msgs
    card, msg_c = run(None)
    native, msg_n = run(osdi_path())
    py = [np.asarray(x, float) for x in full_corr().apply(with_sims(rec, card))]
    rd = 0.0
    for a, b in zip(native, py):
        k = np.abs(b) > m.I_FLOOR
        if k.any():
            rd = max(rd, float(np.max(np.abs(a[k] - b[k]) / np.abs(b[k]))))
    return {"tag": tag, "rrms_card_only": float(score_device(rec, card)["rrms"]),
            "rrms_python": float(score_device(rec, py)["rrms"]),
            "rrms_native": float(score_device(rec, native)["rrms"]),
            "max_rel_native_vs_python_1nA": rd, "msgs": msg_c + msg_n}


def stage_refit(rd: Path, tags: list[str], workers: int, opts: dict) -> dict:
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(refit_task, [(t, opts) for t in tags], chunksize=1)
        t_refit = time.time() - t0
        cards = {r["tag"]: r["params"] for r in rows}
        nat = pool.map(native_task, [(t, cards[t]) for t in tags], chunksize=1)
    out = {"opts": opts, "refit_s": round(t_refit, 1),
           "rows": {r["tag"]: {k: r[k] for k in ("rrms_start", "rrms", "failed", "n_sim",
                                                 "runtime_s")} for r in rows},
           "native": {r["tag"]: r for r in nat}}
    jdump(cards, rd / "cards_M_full.json")
    jdump(out, rd / "refit.json")
    v = [r["rrms"] for r in rows]
    print(f"refit: {len(rows)} devices, {t_refit:.0f} s, in-sample mean RRMS {np.mean(v):.4f}",
          flush=True)
    return out


# ---------------------------------------------------------------------------
# q12: device-level grid, small signal, dense sweeps, wall time
# ---------------------------------------------------------------------------
def dense_control(dev, out_prefix: Path) -> list[str]:
    """1 mV sweeps: idvg at GM_VDS, idvd at GDS_VGS (forward polarity)."""
    s = -1.0 if dev.dev_type == "pmos" else 1.0
    lines = []
    for k, vd in enumerate(m.GM_VDS):
        lines += [f"alter VD dc = {s * vd:.6g}", f"dc VG 0 {s * 1.85:.6g} {s * 0.001:.6g}",
                  f"wrdata {out_prefix.as_posix()}_gm{k}.dat -i(VD)"]
    lines.append("alter VG dc = 0")
    for k, vg in enumerate(m.GDS_VGS):
        lines += [f"alter VG dc = {s * vg:.6g}", f"dc VD 0 {s * 1.85:.6g} {s * 0.001:.6g}",
                  f"wrdata {out_prefix.as_posix()}_gds{k}.dat -i(VD)"]
    return lines


def q12_task(task):
    tag, params, rd_s = task
    rd = Path(rd_s)
    dev = devices()[tag]
    corr = full_corr()
    pm = dev.dev_type == "pmos"
    osdi = osdi_path()
    res = {"tag": tag, "msgs": {}, "wall": {}}
    arrays = {}
    with m.tempdir() as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, {dev.dev_type: [(dev.bin, params)]})

        def go(label, body, control, el, options="", repeats=1):
            walls, r = [], None
            for _ in range(repeats):
                r = m.run_deck(m.deck(label, lib, body, control, osdi if el else None, options),
                               td, f"{label}.sp", timeout_s=m.TIMEOUT_DEVICE_S)
                walls.append(r["wall_s"])
            res["msgs"][label] = r["msgs"]
            res["wall"][label] = walls
            return r

        # Q1 grids
        for orient in ("fwd", "swap"):
            for el in (False, True):
                for tol, opt in (("default", ""), ("r4", m.R4), ("tight", m.TIGHT)):
                    if orient == "swap" and tol != "tight":
                        continue
                    lab = f"grid_{orient}_{tol}_{'corr' if el else 'card'}"
                    f = td / f"{lab}.dat"
                    rep = REPEATS if (orient == "fwd" and tol == "default") else 1
                    go(lab, m.device_body(dev, el, swap=orient == "swap"),
                       m.grid_control(dev, f), el, opt, repeats=rep)
                    arrays[lab] = m.grid_from_rows(m.read_wrdata(f))
        # Q2 AC at op points (tight)
        pts = m.op_points(dev)
        for el in (False, True):
            for ex in ("VG", "VD"):
                lab = f"ac_{ex}_{'corr' if el else 'card'}"
                r = go(lab, m.device_body(dev, el), m.ac_control(pts, ex), el, m.TIGHT)
                for k, v in m.parse_ac(r["text"], len(pts)).items():
                    arrays[f"{lab}_{k}"] = v
        # Q2 dense sweeps (tight)
        for el in (False, True):
            lab = f"dense_{'corr' if el else 'card'}"
            pre = td / lab
            go(lab, m.device_body(dev, el), dense_control(dev, pre), el, m.TIGHT)
            for k in range(len(m.GM_VDS)):
                arrays[f"{lab}_gm{k}"] = m.read_wrdata(Path(f"{pre}_gm{k}.dat"))
            for k in range(len(m.GDS_VGS)):
                arrays[f"{lab}_gds{k}"] = m.read_wrdata(Path(f"{pre}_gds{k}.dat"))
    res.update(q12_metrics(dev, corr, pts, arrays))
    np.savez_compressed(rd / "arrays" / f"q12_{tag}.npz", pts=pts, **{
        k: v for k, v in arrays.items() if isinstance(v, np.ndarray)})
    return res


def q12_metrics(dev, corr, pts, a) -> dict:
    pm = dev.dev_type == "pmos"
    s = -1.0 if pm else 1.0
    VG, VD = np.meshgrid(s * m.GRID_VGS, m.GRID_VDS, indexing="ij")
    g_fwd = m.g_values(corr, pm, VG, VD)
    out = {}
    # ---- Q1 equivalence
    for tol in ("default", "r4", "tight"):
        ref = m.python_current(a[f"grid_fwd_{tol}_card"], g_fwd)
        rdiff = m.rel_diff(a[f"grid_fwd_{tol}_corr"], ref, m.I_FLOOR)
        out[f"q1_{tol}_max"] = m.nanmax0(rdiff)
        out[f"q1_{tol}_n_counted"] = int(np.sum(~np.isnan(rdiff)))
        fwdmask = (s * VD) > 0
        out[f"q1_{tol}_max_forward"] = m.nanmax0(np.where(fwdmask, rdiff, np.nan))
        out[f"q1_{tol}_max_reverse"] = m.nanmax0(np.where(~fwdmask & (VD != 0), rdiff, np.nan))
        out[f"q1_{tol}_max_above_1uA"] = m.nanmax0(np.where(np.abs(ref) >= 1e-6, rdiff, np.nan))
    # swap orientation: element terminals d=0, s=nd -> V(g,s) = Vg - Vd, V(d,s) = -Vd
    g_swap = m.g_values(corr, pm, VG - VD, -VD)
    ref_sw = m.python_current(a["grid_swap_tight_card"], g_swap)
    # forward operation only: in reverse operation the swapped source-body
    # junction current does not pass the sense branch
    out["q1_swap_tight_max_forward"] = m.nanmax0(np.where(
        (s * VD) > 0, m.rel_diff(a["grid_swap_tight_corr"], ref_sw, m.I_FLOOR), np.nan))
    corr_runs = [k for k in a if k.startswith("grid_") and k.endswith("_corr")]
    out["q1_nonfinite"] = int(sum(np.sum(~np.isfinite(a[k])) for k in corr_runs))
    out["q1_nonfinite_card"] = int(sum(np.sum(~np.isfinite(a[k])) for k in a
                                       if k.startswith("grid_") and k.endswith("_card")))
    j0 = int(np.argmin(np.abs(m.GRID_VDS)))
    out["q1_vds0_max_corr"] = float(max(np.nanmax(np.abs(a[f"grid_fwd_{t}_corr"][:, j0]))
                                        for t in ("default", "tight")))
    out["q1_vds0_max_card"] = float(max(np.nanmax(np.abs(a[f"grid_fwd_{t}_card"][:, j0]))
                                        for t in ("default", "tight")))
    out["g_grid_min"], out["g_grid_max"] = float(np.min(g_fwd)), float(np.max(g_fwd))
    out["g_grid_vds0_min"] = float(np.min(g_fwd[:, j0]))
    out["g_grid_vds0_max"] = float(np.max(g_fwd[:, j0]))
    # terminal-swap asymmetry (forward polarity, |I| > 1 nA): element part
    fw = (s * VD) > 0
    ic, isw = a["grid_fwd_tight_corr"], a["grid_swap_tight_corr"]
    cc, csw = a["grid_fwd_tight_card"], a["grid_swap_tight_card"]
    keep = fw & (np.abs(ic) > m.I_FLOOR) & (np.abs(cc) > m.I_FLOOR) & np.isfinite(isw) & np.isfinite(csw)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_corr = np.log(np.abs(isw / ic))
        lr_card = np.log(np.abs(csw / cc))
    out["swap_asym_card_max"] = float(np.max(np.abs(lr_card[keep]))) if keep.any() else 0.0
    out["swap_asym_corr_max"] = float(np.max(np.abs(lr_corr[keep]))) if keep.any() else 0.0
    out["swap_asym_element_max"] = float(np.max(np.abs(lr_corr[keep] - lr_card[keep]))) if keep.any() else 0.0
    out["swap_asym_element_pred_max"] = float(np.max(np.abs((g_swap - g_fwd)[keep]))) if keep.any() else 0.0
    # ---- Q2 small signal (tight)
    vg, vd = pts[:, 0], pts[:, 1]
    gp = m.g_values(corr, pm, vg, vd)
    dgg, dgd = m.g_partials(corr, pm, vg, vd)
    eg = np.exp(gp)
    i_card = a["ac_VG_card_i0"]
    ycard_dg = a["ac_VG_card_yr"] + 1j * a["ac_VG_card_ym"]
    ycard_dd = a["ac_VD_card_yr"] + 1j * a["ac_VD_card_ym"]
    ycorr_dg = a["ac_VG_corr_yr"] + 1j * a["ac_VG_corr_ym"]
    ycorr_dd = a["ac_VD_corr_yr"] + 1j * a["ac_VD_corr_ym"]
    ypy_dg = m.python_admittance(i_card, ycard_dg, gp, dgg)
    ypy_dd = m.python_admittance(i_card, ycard_dd, gp, dgd)
    ipy = m.python_current(i_card, gp)
    cnt = np.abs(ipy) > m.I_FLOOR
    errs = {
        "gm": m.norm_y_err(ycorr_dg.real, ypy_dg.real),
        "gds": m.norm_y_err(ycorr_dd.real, ypy_dd.real),
        "im_dg": m.norm_y_err(ycorr_dg.imag, ypy_dg.imag, 1e-15),
        "im_dd": m.norm_y_err(ycorr_dd.imag, ypy_dd.imag, 1e-15),
    }
    for k, e in errs.items():
        out[f"q2_err_{k}"] = float(np.nanmax(np.where(cnt, e, np.nan))) if cnt.any() else 0.0
        if np.any(cnt & ~np.isfinite(e)):
            out[f"q2_err_{k}"] = float("inf")
    out["q2_op_i_rel"] = m.nanmax0(m.rel_diff(a["ac_VG_corr_i0"], ipy, m.I_FLOOR))
    # gate capacitance unchanged; drain displacement current scaled by exp(g)
    with np.errstate(divide="ignore", invalid="ignore"):
        rg = a["ac_VG_corr_yg"] / a["ac_VG_card_yg"]
        rdd = ycorr_dd.imag / ycard_dd.imag
        rdg = ycorr_dg.imag / ycard_dg.imag
    ok = np.isfinite(rg)
    out["cap_gate_ratio_maxdev"] = float(np.max(np.abs(rg[ok] - 1))) if ok.any() else float("nan")
    ok = np.isfinite(rdd) & (np.abs(ycard_dd.imag) > 1e-15)
    out["cap_dd_ratio_median"] = float(np.median(rdd[ok]))
    out["cap_dd_ratio_min"], out["cap_dd_ratio_max"] = float(np.min(rdd[ok])), float(np.max(rdd[ok]))
    out["cap_dd_ratio_over_expg_maxdev"] = float(np.max(np.abs(rdd[ok] / eg[ok] - 1)))
    ok = np.isfinite(rdg) & (np.abs(ycard_dg.imag) > 1e-15)
    out["cap_dg_ratio_over_expg_maxdev"] = float(np.max(np.abs(rdg[ok] / eg[ok] - 1)))
    out["cap_dd_ratio_maxdev_from_1"] = float(np.max(np.abs(rdd[ok] - 1))) if ok.any() else 0.0
    # strong inversion mask and sign changes
    jv = int(np.argmin(np.abs(np.abs(m.GRID_VDS) - m.GM_VDS[0]) + (s * m.GRID_VDS < 0)))
    w_eff = dev.W_um * __import__("cryoml.spice_pdk", fromlist=["x"]).device_mult(
        dev.dev_type, dev.L_um, dev.W_um)
    vt = m.threshold_voltage(m.GRID_VGS, a["grid_fwd_tight_card"][:, jv], 1e-7 * w_eff / dev.L_um)
    out["vt_card"] = vt
    n_gm = len(m.GM_VDS) * len(m.GRID_VGS)
    gm_mask = np.zeros(len(pts), bool)
    gm_mask[:n_gm] = np.abs(vg[:n_gm]) >= vt + 0.2
    gds_mask = np.zeros(len(pts), bool)
    gds_mask[n_gm:] = np.abs(vd[n_gm:]) >= 0.05
    # gm = dI_D/dVg and gds = dI_D/dVd (I_D into the drain) are positive in
    # forward operation for both polarities
    out["q2_gm_sign_ac_points"] = m.sign_changes_introduced(ycorr_dg.real, ycard_dg.real, gm_mask)
    out["q2_gds_sign_ac_points"] = m.sign_changes_introduced(ycorr_dd.real, ycard_dd.real, gds_mask)
    nv = len(m.GRID_VGS)
    gm_curves_ac = [m.curve_sign_change_introduced(ycorr_dg.real[k * nv:(k + 1) * nv],
                                                   ycard_dg.real[k * nv:(k + 1) * nv],
                                                   gm_mask[k * nv:(k + 1) * nv])
                    for k in range(len(m.GM_VDS))]
    gds_curves_ac = [m.curve_sign_change_introduced(ycorr_dd.real[n_gm + k * nv:n_gm + (k + 1) * nv],
                                                    ycard_dd.real[n_gm + k * nv:n_gm + (k + 1) * nv],
                                                    gds_mask[n_gm + k * nv:n_gm + (k + 1) * nv])
                     for k in range(len(m.GDS_VGS))]
    gm_curves_dense, gds_curves_dense = [False] * len(m.GM_VDS), [False] * len(m.GDS_VGS)
    out["q2_gm_neg_card_ac"] = int(np.sum(gm_mask & (ycard_dg.real < 0)))
    out["q2_gds_neg_card_ac"] = int(np.sum(gds_mask & (ycard_dd.real < 0)))
    # dense 1 mV sweeps: FD of OSDI current and of the Python model; sign changes
    fd_err, fd_err_py, n_gm_dense, n_gds_dense, kink = 0.0, 0.0, 0, 0, 0.0
    for k, vds in enumerate(m.GM_VDS):
        dc, dk = a[f"dense_card_gm{k}"], a[f"dense_corr_gm{k}"]
        if dc.shape[0] < 10 or dk.shape[0] != dc.shape[0]:
            fd_err = float("inf")
            continue
        o = np.argsort(np.abs(dc[:, 0]))
        x, ic_, ik_ = dc[o, 0], dc[o, 1], dk[o, 1]
        gq = m.g_values(corr, pm, x, s * vds)
        gm_c, gm_k = m.central_diff(x, ic_), m.central_diff(x, ik_)
        gm_p = m.central_diff(x, m.python_current(ic_, gq))
        sel = slice(k * len(m.GRID_VGS), (k + 1) * len(m.GRID_VGS))
        at = np.searchsorted(np.round(np.abs(x), 6), np.round(np.abs(vg[sel]), 6))
        inner = (at >= 1) & (at <= len(x) - 2)
        at = np.clip(at, 1, len(x) - 2)
        cnt_k = (np.abs(ipy[sel]) > m.I_FLOOR) & inner
        e1 = m.norm_y_err(gm_k[at], ycorr_dg.real[sel])
        e2 = m.norm_y_err(gm_p[at], ycorr_dg.real[sel])
        fd_err = max(fd_err, float(np.max(e1[cnt_k])) if cnt_k.any() else 0.0)
        fd_err_py = max(fd_err_py, float(np.max(e2[cnt_k])) if cnt_k.any() else 0.0)
        mk = np.abs(x) >= vt + 0.2
        n_gm_dense += m.sign_changes_introduced(gm_k, gm_c, mk)
        gm_curves_dense[k] = m.curve_sign_change_introduced(gm_k, gm_c, mk)
    for k, vgs in enumerate(m.GDS_VGS):
        dc, dk = a[f"dense_card_gds{k}"], a[f"dense_corr_gds{k}"]
        if dc.shape[0] < 10 or dk.shape[0] != dc.shape[0]:
            continue
        o = np.argsort(np.abs(dc[:, 0]))
        x, ic_, ik_ = dc[o, 0], dc[o, 1], dk[o, 1]
        gds_c, gds_k = m.central_diff(x, ic_), m.central_diff(x, ik_)
        n_gds_dense += m.sign_changes_introduced(gds_k, gds_c, np.abs(x) >= 0.05)
        gds_curves_dense[k] = m.curve_sign_change_introduced(gds_k, gds_c, np.abs(x) >= 0.05)
    # gm kink at Vgs = 0 (abs): jump I * exp(g) * 2 |dg/d|Vgs|| relative to card gm
    kink_abs, i_vgs0 = 0.0, 0.0
    for k, vds in enumerate(m.GM_VDS):
        i0 = i_card[k * len(m.GRID_VGS)]
        h = 1e-5
        gpl = m.g_values(corr, pm, s * h, s * vds)[0]
        gmi = m.g_values(corr, pm, -s * h, s * vds)[0]
        g0 = m.g_values(corr, pm, 0.0, s * vds)[0]
        jump = abs(i0 * np.exp(g0) * ((gpl - g0) / h - (g0 - gmi) / h))
        gmc = abs(ycard_dg.real[k * len(m.GRID_VGS)])
        kink = max(kink, jump / max(gmc, 1e-30))
        kink_abs = max(kink_abs, float(jump))
        i_vgs0 = max(i_vgs0, abs(float(i0)))
    out.update({"q2_fd_err_osdi": fd_err, "q2_fd_err_python": fd_err_py,
                "q2_gm_sign_dense_points": int(n_gm_dense),
                "q2_gds_sign_dense_points": int(n_gds_dense),
                "q2_gm_curves_introduced": [bool(a_ or b_) for a_, b_ in zip(gm_curves_ac, gm_curves_dense)],
                "q2_gds_curves_introduced": [bool(a_ or b_) for a_, b_ in zip(gds_curves_ac, gds_curves_dense)],
                "q2_gm_kink_vgs0_rel": float(kink), "q2_gm_kink_vgs0_abs_S": kink_abs,
                "q2_abs_i_vgs0_A": i_vgs0})
    return out


def stage_q12(rd: Path, tags: list[str], workers: int) -> dict:
    cards = jload(rd / "cards_M_full.json")
    (rd / "arrays").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(q12_task, [(t, cards[t], str(rd)) for t in tags], chunksize=1)
    out = {"wall_s": round(time.time() - t0, 1), "devices": {r["tag"]: r for r in rows}}
    jdump(out, rd / "q12.json")
    print(f"q12: {len(rows)} devices, {out['wall_s']} s", flush=True)
    return out


# ---------------------------------------------------------------------------
# q3: circuits
# ---------------------------------------------------------------------------
def arm_cards(arm: str, tags, m_full: dict) -> dict:
    """dev_type -> [(bin, params)] for the circuit's devices."""
    devs = devices()
    held = jload(ALL18 / "heldout_cards.json")
    cards = {}
    for t in tags:
        d = devs[t]
        if arm == "published":
            p = None
        elif arm == "C0_refit":
            p = dict(held["C0_refit"][t])
        else:
            p = dict(m_full[t])
        cards.setdefault(d.dev_type, []).append((d.bin, p))
    return cards


def q3_task(task):
    name, kind, tags, settings, arm, rd_s = task
    rd = Path(rd_s)
    devs = devices()
    m_full = jload(rd / "cards_M_full.json")
    el = arm == "M_full"
    osdi = osdi_path() if el else None
    reps = REPEATS if arm in ("M_card", "M_full") else 1
    res = {"circuit": name, "arm": arm, "kind": kind}
    with m.tempdir() as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, arm_cards(arm, tags, m_full))
        f = td / "out.dat"
        if kind == "vtc":
            body = m.vtc_deck_body(devs[tags[0]], devs[tags[1]], el)
            ctrl = m.vtc_control(f)
        elif kind == "ring":
            body = m.ring_body(devs[tags[0]], devs[tags[1]], el, stages=5)
            ctrl = m.ring_control(f, settings["tstep"], settings["tstop"])
        else:
            body = m.mirror_body(devs[tags[0]], el, settings["iref"])
            ctrl = m.mirror_control(f)
        walls = []
        for _ in range(reps):
            r = m.run_deck(m.deck(name, lib, body, ctrl, osdi), td, "c.sp",
                           timeout_s=m.TIMEOUT_CIRCUIT_S)
            walls.append(r["wall_s"])
        res["wall"] = walls
        res["msgs"] = r["msgs"]
        if kind == "mirror":
            rows = m.read_wrdata(f, 4)
            x, y, vref = rows[:, 0], rows[:, 1], rows[:, 3]
        else:
            rows = m.read_wrdata(f, 2)
            x, y = rows[:, 0], rows[:, 1]
    finite = bool(rows.size and np.all(np.isfinite(rows)))
    res["finite"] = finite
    if kind == "vtc" and finite:
        res.update(m.vtc_metrics(x, y))
    elif kind == "ring" and finite:
        res.update(m.oscillation(x, y, m.VDD_CIRCUIT / 2, t_from=0.3 * settings["tstop"]))
        res["n_points"] = int(len(x))
    elif kind == "mirror" and finite:
        iref = settings["iref"]
        vr = float(np.median(vref))
        res["v_ref"] = vr
        if devs[tags[0]].dev_type == "nmos":
            v_match = vr
        else:
            v_match = vr     # pMOS: equal |Vds| when V(out) = V(ref)
        res["ratio_at_0p9"] = m.interp_at(x, y, 0.9) / iref
        res["ratio_at_vref"] = m.interp_at(x, y, v_match) / iref
    ok = finite and not r["msgs"]
    if kind == "ring" and finite:
        ok = ok and res["n_periods"] >= m.THR["ro_min_periods"]
    res["ok"] = bool(ok)
    np.savez_compressed(rd / "arrays" / f"q3_{name}_{arm}.npz", rows=rows)
    return res


def stage_q3(rd: Path, workers: int) -> dict:
    (rd / "arrays").mkdir(parents=True, exist_ok=True)
    tasks = [(n, k, t, s, arm, str(rd)) for n, k, t, s in CIRCUITS for arm in Q3_ARMS]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(q3_task, tasks, chunksize=1)
    out = {"wall_s": round(time.time() - t0, 1),
           "circuits": {n: {"kind": k, "devices": list(t), "settings": s} for n, k, t, s in CIRCUITS},
           "rows": {f"{r['circuit']}|{r['arm']}": r for r in rows}}
    jdump(out, rd / "q3.json")
    print(f"q3: {len(rows)} runs, {out['wall_s']} s", flush=True)
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def stage_report(rd: Path) -> dict:
    q12 = jload(rd / "q12.json")["devices"]
    q3 = jload(rd / "q3.json")["rows"]
    refit = jload(rd / "refit.json")
    D = list(q12.values())
    elem_msgs = sum(len(v) for r in D for k, v in r["msgs"].items() if k.endswith("_corr"))
    card_msgs = sum(len(v) for r in D for k, v in r["msgs"].items() if k.endswith("_card"))
    ratios = {}
    for r in D:
        c = np.median(r["wall"]["grid_fwd_default_card"])
        e = np.median(r["wall"]["grid_fwd_default_corr"])
        ratios[r["tag"]] = {"card_s": float(c), "corr_s": float(e), "ratio": float(e / c)}
    circ_t = {}
    for n, *_ in CIRCUITS:
        c = np.median(q3[f"{n}|M_card"]["wall"])
        e = np.median(q3[f"{n}|M_full"]["wall"])
        circ_t[n] = {"card_s": float(c), "corr_s": float(e), "ratio": float(e / c)}
    q3_fail = sum(1 for n, *_ in CIRCUITS if not q3[f"{n}|M_full"]["ok"] and q3[f"{n}|M_card"]["ok"])
    q3_fail_any = [k for k, v in q3.items() if not v["ok"]]
    s = {
        "q1_tight_max": max(r["q1_tight_max"] for r in D),
        "q1_default_max": max(r["q1_default_max"] for r in D),
        "q1_bad": int(elem_msgs + sum(r["q1_nonfinite"] for r in D)),
        "q1_vds0_max": max(r["q1_vds0_max_corr"] for r in D),
        "q2_ac_max": max(max(r[f"q2_err_{k}"] for k in ("gm", "gds", "im_dg", "im_dd")) for r in D),
        "q2_gm_sign": int(sum(sum(r["q2_gm_curves_introduced"]) for r in D)),
        "q2_gds_sign": int(sum(sum(r["q2_gds_curves_introduced"]) for r in D)),
        "q3_fail": int(q3_fail),
        "q3_fail_cardonly": len([k for k in q3_fail_any if not k.endswith("|M_full")]),
        "swap_asym_max": max(r["swap_asym_element_max"] for r in D),
        "cap_scaled": bool(max(r["cap_dd_ratio_maxdev_from_1"] for r in D) > 1e-3),
        "runtime_ratio_max": max([v["ratio"] for v in ratios.values()]
                                 + [v["ratio"] for v in circ_t.values()]),
    }
    dec = m.decide(s)
    summary = {"summary_values": s, "decision": dec, "n_devices": len(D),
               "card_only_messages": card_msgs, "q3_not_ok": q3_fail_any,
               "runtime_grid": ratios, "runtime_circuits": circ_t,
               "refit_in_sample": {
                   "mean_rrms_M_full_python": float(np.mean([v["rrms_python"] for v in refit["native"].values()])),
                   "mean_rrms_M_full_native": float(np.mean([v["rrms_native"] for v in refit["native"].values()])),
                   "mean_rrms_M_card_only": float(np.mean([v["rrms_card_only"] for v in refit["native"].values()])),
                   "max_rel_native_vs_python_measured_1nA": float(max(v["max_rel_native_vs_python_1nA"] for v in refit["native"].values())),
                   "n_failed": int(sum(bool(v["failed"]) for v in refit["rows"].values()))}}
    jdump(summary, rd / "summary.json")
    jdump(dec, rd / "decision.json")
    print(json.dumps({"summary_values": s, **dec}, indent=1, default=float), flush=True)
    return summary


# ---------------------------------------------------------------------------
# diagnose (post hoc, after the pre-registered decision; does not change it)
# ---------------------------------------------------------------------------
def _slope(x, y, lo, hi) -> float:
    k = (x >= lo) & (x <= hi)
    return float(np.polyfit(x[k], y[k], 1)[0]) if k.sum() >= 3 else float("nan")


def stage_diagnose(rd: Path) -> dict:
    """Location of the Q1 tight and Q2 core failures; average output
    conductance (linear fit, |Vds| 1.0-1.85 V) of card, element and measured
    curves on the gds curves with an introduced sign change."""
    corr = full_corr()
    devs = devices()
    recs = load_records()
    q12 = jload(rd / "q12.json")["devices"]
    out = {"q1_tight": {}, "q2_off_kink_max": 0.0, "q2_off_kink_where": None, "gm_sign": [],
           "gds_slopes": []}
    for tag, dev in devs.items():
        a = np.load(rd / "arrays" / f"q12_{tag}.npz")
        pm = dev.dev_type == "pmos"
        s = -1.0 if pm else 1.0
        VG, VD = np.meshgrid(s * m.GRID_VGS, m.GRID_VDS, indexing="ij")
        ref = m.python_current(a["grid_fwd_tight_card"], m.g_values(corr, pm, VG, VD))
        r = m.rel_diff(a["grid_fwd_tight_corr"], ref, m.I_FLOOR)
        ad = np.abs(a["grid_fwd_tight_corr"] - ref)
        over = r > m.THR["q1_tight"]
        out["q1_tight"][tag] = {
            "n_over": int(np.sum(over)),
            "max_abs_diff_over_A": float(ad[over].max()) if over.any() else 0.0,
            "max_abs_I_over_A": float(np.abs(ref[over]).max()) if over.any() else 0.0,
            "max_rel_above_10nA": m.nanmax0(m.rel_diff(a["grid_fwd_tight_corr"], ref, 1e-8))}
        pts = a["pts"]
        vg, vd = pts[:, 0], pts[:, 1]
        gp = m.g_values(corr, pm, vg, vd)
        dgg, dgd = m.g_partials(corr, pm, vg, vd)
        ic = a["ac_VG_card_i0"]
        cnt = (np.abs(ic * np.exp(gp)) > m.I_FLOOR) & (np.abs(vg) > 1e-9) & (np.abs(vd) > 1e-9)
        for ex, dg in (("VG", dgg), ("VD", dgd)):
            yc = a[f"ac_{ex}_card_yr"] + 1j * a[f"ac_{ex}_card_ym"]
            e = m.norm_y_err(a[f"ac_{ex}_corr_yr"], m.python_admittance(ic, yc, gp, dg).real)
            w = float(np.nanmax(np.where(cnt, e, np.nan)))
            if w > out["q2_off_kink_max"]:
                out["q2_off_kink_max"], out["q2_off_kink_where"] = w, [tag, ex]
        nv = len(m.GRID_VGS)
        for k, flag in enumerate(q12[tag]["q2_gm_curves_introduced"]):
            if flag:
                d = a[f"dense_corr_gm{k}"]
                gmk = m.central_diff(d[:, 0], d[:, 1])
                neg = (gmk < -m.Y_NEG) & (np.abs(d[:, 0]) >= q12[tag]["vt_card"] + 0.2)
                out["gm_sign"].append({"tag": tag, "abs_vds": m.GM_VDS[k],
                                       "abs_vgs_neg_min": float(np.abs(d[neg, 0]).min()),
                                       "abs_vgs_neg_max": float(np.abs(d[neg, 0]).max()),
                                       "gm_min_S": float(gmk[neg].min())})
        for k, flag in enumerate(q12[tag]["q2_gds_curves_introduced"]):
            if not flag:
                continue
            vgk = m.GDS_VGS[k]
            dk, dc = a[f"dense_corr_gds{k}"], a[f"dense_card_gds{k}"]
            x = np.abs(dk[:, 0])
            meas = float("nan")
            for c in recs[tag].curves:
                if c.kind == "idvd" and abs(abs(float(np.median(c.vg))) - vgk) < 0.04:
                    meas = _slope(np.abs(np.asarray(c.vd, float)), np.abs(np.asarray(c.target, float)),
                                  1.0, 1.85)
            gk = m.central_diff(dk[:, 0], dk[:, 1])
            neg = (gk < -m.Y_NEG) & (x >= 0.05)
            out["gds_slopes"].append({
                "tag": tag, "abs_vgs": vgk, "card_S": _slope(x, np.abs(dc[:, 1]), 1.0, 1.85),
                "elem_S": _slope(x, np.abs(dk[:, 1]), 1.0, 1.85), "measured_S": meas,
                "abs_vds_neg_min": float(x[neg].min()) if neg.any() else None,
                "abs_vds_neg_max": float(x[neg].max()) if neg.any() else None,
                "gds_min_S": float(gk[neg].min()) if neg.any() else None})
    jdump(out, rd / "diagnosis.json")
    print(f"diagnose: Q2 max off the abs() kink lines {out['q2_off_kink_max']:.2e}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("compile", "refit", "q12", "q3", "report", "diagnose", "all"))
    ap.add_argument("--name", default="full")
    ap.add_argument("--smoke", action="store_true",
                    help="2 devices, refit max_nfev 20 (code check; not used for decisions)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rd = run_dir("smoke" if args.smoke else args.name)
    rd.mkdir(parents=True, exist_ok=True)
    tags = list(SMOKE_DEVICES) if args.smoke else sorted(devices())
    opts = dict(REFIT_OPTS, heldout_nfev=20) if args.smoke else dict(REFIT_OPTS)
    workers = min(args.workers, 4)
    timing = {}
    for st in ("compile", "refit", "q12", "q3", "report", "diagnose"):
        if args.stage not in (st, "all"):
            continue
        t0 = time.time()
        if st == "compile":
            stage_compile(rd)
        elif st == "refit":
            stage_refit(rd, tags, workers, opts)
        elif st == "q12":
            stage_q12(rd, tags, workers)
        elif st == "q3":
            if args.smoke:
                # smoke: circuits need M_full cards for the 4 circuit devices
                need = sorted({t for _, _, ts, _ in CIRCUITS for t in ts} - set(tags))
                if need:
                    have = jload(rd / "cards_M_full.json")
                    start = jload(START)["devices"]
                    for t in need:
                        have[t] = dict(start[t]["params"])   # smoke only: starting card
                    jdump(have, rd / "cards_M_full.json")
            stage_q3(rd, workers)
        elif st == "report":
            stage_report(rd)
        else:
            stage_diagnose(rd)
        timing[st] = round(time.time() - t0, 1)
    tf = rd / "timing.json"
    old = jload(tf) if tf.exists() else {}
    old.update(timing)
    jdump(old, tf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
