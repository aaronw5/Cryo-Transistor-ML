#!/usr/bin/env python3
"""I12-1 post hoc diagnosis of the failed part B core checks (after the
decision; does not change it).  Light compute: single-process ngspice-41 decks,
about 1 s per device.

Failed core checks (decision.json): terminal-exchange asymmetry > 1e-3,
Q2 derivative agreement (> 1e-3), Q2 capacitances differ from the card
(> 1e-3).  The smoke run (nmos_L1_W1p6, pmos_L2_W5) passed all three.

Parts written to ``out/research/explore/I12_1/diagnosis/diagnosis.json``:

A  per-device values of the three checks against the thresholds (from
   deploy/q12.json), devices over threshold.
B  BSIM4 parameter-check state.  ngspice-41 ``b4check.c`` checks each model
   name once per process (static ``modname``) and clamps the size-dependent
   ``A2 > 1`` to 1 (and ``A1`` to 0); ``b4temp.c`` rebuilds the
   size-dependent parameters at every analysis.  In a deck with one BSIM4
   model the clamp applies in the first analysis only; in the element deck
   (card model and replica model) the check alternates between the two
   names and the clamp applies in every analysis.  Test: two consecutive
   ``op`` at the same bias, card only, published / starting / M2 card, and
   single-parameter search on nmos_L0p19_W7 (published card + one M2 value).
C  Q2 recomputed with the card AC reference in the element deck's check
   state (card instance + replica instance driven by VCVS copies, replica
   current read through a 0 V source; no OSDI module), against the stored
   element AC arrays.  Two references: (1) registered formula
   Re Y = exp(g) Re Y_card + I exp(g) dg/dV; (2) element identity
   Re Y = Re Y_card + (exp(g) - 1) Re Y_rep + I_rep exp(g) dg/dV, with Y_rep
   the charge-free replica admittance (static conductance at 1 MHz).
   Capacitance error against Im Y_card in the matched state.
   C2: Re Y_dg of the card (card only) versus frequency (100 Hz .. 1 MHz) at
   the nmos_L100_W100 max point, card as is (rbodymod = 1) and with
   rbodymod = 0.
D  Exchange asymmetry: at the max point, drain-side non-channel current
   J = |I_fwd,card| - |i(VZ)| (card only, tight grid) and the prediction
   |1 - exp(-g)| J / |I|; J versus Vgs; impact-ionization test (card current
   with alpha0 = 0 at the max point of nmos_L100_W100 and nmos_L20_W0p64).
E  Restrictions (values only).

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i12_1_diagnosis.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("i12_1_drv", ROOT / "scripts/research/explore/i12_1_symm.py")
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)
m, m1 = drv.m, drv.m1

OUT = drv.OUT / "diagnosis"
DEP = drv.OUT / "deploy"
THR = {"swap_asym": 1e-3, "q2_ac": 1e-3, "cap": 1e-3}
SMOKE = ("nmos_L1_W1p6", "pmos_L2_W5")


def ops_twice(dev, params, vg: float, vd: float) -> list[float]:
    """Card-only deck, tight tolerance: op at (vg, vd) twice in one process."""
    with tempfile.TemporaryDirectory(prefix="cryoml_i12d_") as t:
        td = Path(t)
        lib = m.patched_library_v2(td, {dev.dev_type: [(dev.bin, params)]} if params else {},
                                   bsim4_state="legacy")   # I14-1: decks as diagnosed
        p = [f"alter VG dc = {vg:.6g}", f"alter VD dc = {vd:.6g}", "op", "print -i(VD)"]
        r = m1.run_deck(m.deck_v2("ops", lib, m.device_body_v2(dev, False), p + p, None, m1.TIGHT), td, "ops.sp")
    return [float(ln.split("=")[1]) for ln in r["text"].splitlines() if "i(vd) =" in ln.lower()]


def drift(v: list[float]) -> float:
    return abs(v[1] - v[0]) / abs(v[0]) if len(v) == 2 and v[0] != 0 else float("nan")


def matched_body(dev) -> str:
    """Card at (nd, ng, 0, 0) plus the replica driven by VCVS copies; the
    OSDI module is replaced by a 0 V source VRm1 (replica drain current)."""
    txt = m.instance_dev("m1", dev, "nd", "ng", "0", "0", True)
    lines = [ln for ln in txt.splitlines() if not ln.startswith("Nm1 ")]
    if len(lines) != len(txt.splitlines()) - 1:
        raise RuntimeError("module line not found")
    lines.append("VRm1 m1_rx m1_ri DC 0")
    return "\n".join(lines) + "\nVG ng 0 DC 0 AC 0\nVD nd 0 DC 0 AC 0\n"


def matched_ac_control(points: np.ndarray, excite: str) -> list[str]:
    lines = [f"alter {excite} ac = 1"]
    for k, (vg, vd) in enumerate(points):
        lines += [f"alter VG dc = {vg:.6g}", f"alter VD dc = {vd:.6g}", "op",
                  f"let q{k}i = -i(VD)", f"let q{k}j = i(VRm1)", f"print q{k}i q{k}j",
                  f"ac lin 1 {m1.AC_FREQ:g} {m1.AC_FREQ:g}",
                  f"let q{k}r = real(-i(VD))", f"let q{k}m = imag(-i(VD))",
                  f"let q{k}g = imag(-i(VG))", f"let q{k}s = real(i(VRm1))", f"let q{k}t = imag(i(VRm1))",
                  f"print q{k}r q{k}m q{k}g q{k}s q{k}t", "destroy all"]
    return lines


_P = re.compile(r"^\s*q(\d+)([ijrmgst])\s*=\s*([-+0-9.eEnaNinf]+)")


def parse(text: str, n: int) -> dict:
    out = {k: np.full(n, np.nan) for k in "ijrmgst"}
    for ln in text.splitlines():
        mm = _P.match(ln)
        if mm and int(mm.group(1)) < n:
            out[mm.group(2)][int(mm.group(1))] = float(mm.group(3))
    return out


def matched_ac(dev, params, pts) -> dict:
    res = {}
    with tempfile.TemporaryDirectory(prefix="cryoml_i12d_") as t:
        td = Path(t)
        lib = m.patched_library_v2(td, {dev.dev_type: [(dev.bin, params)]}, bsim4_state="legacy")
        for ex in ("VG", "VD"):
            r = m1.run_deck(m.deck_v2(f"m_{ex}", lib, matched_body(dev), matched_ac_control(pts, ex), None,
                                      m1.TIGHT), td, f"m_{ex}.sp")
            res[ex] = parse(r["text"], len(pts))
            res[ex]["msgs"] = r["msgs"]
    return res


def q2_device(tag, dev, params, corr) -> dict:
    a = np.load(DEP / "arrays" / f"q12_{tag}.npz")
    pts = a["pts"]
    pm = dev.dev_type == "pmos"
    vg, vd = pts[:, 0], pts[:, 1]
    g = m.g_terminal(corr, pm, vg, vd, 0.0)
    eg = np.exp(g)
    dgg, dgd = m.g_partials_v2(corr, pm, vg, vd)
    mt = matched_ac(dev, params, pts)
    i_m, i_rep = mt["VG"]["i"], mt["VG"]["j"]
    i_stored = a["ac_VG_card_i0"]
    ipy = i_m * eg
    cnt = np.abs(ipy) > m1.I_FLOOR
    out = {"msgs": mt["VG"]["msgs"] + mt["VD"]["msgs"],
           "op_card_stored_vs_matched_max_rel": m1.nanmax0(m1.rel_diff(i_stored, i_m, m1.I_FLOOR)),
           "op_rep_vs_card_matched_max_rel": m1.nanmax0(m1.rel_diff(i_rep, i_m, m1.I_FLOOR)),
           "op_elem_stored_vs_matched_ref_max_rel": m1.nanmax0(m1.rel_diff(a["ac_VG_corr_i0"], ipy, m1.I_FLOOR))}
    for name, ex, dg in (("gm", "VG", dgg), ("gds", "VD", dgd)):
        ye = a[f"ac_{ex}_corr_yr"]
        yc_st = a[f"ac_{ex}_card_yr"]
        yc, yrep = mt[ex]["r"], mt[ex]["s"]
        ref_reg_stored = eg * yc_st + a["ac_VG_card_i0"] * eg * dg
        ref_reg = eg * yc + i_m * eg * dg
        ref_id = yc + (eg - 1.0) * yrep + i_rep * eg * dg
        for lab, ref in (("registered_stored", ref_reg_stored), ("registered_matched", ref_reg),
                         ("identity_matched", ref_id)):
            e = np.where(cnt, m1.norm_y_err(ye, ref), np.nan)
            out[f"q2_err_{name}_{lab}"] = m1.nanmax0(e)
        e = np.where(cnt, m1.norm_y_err(yc, yrep), np.nan)
        out[f"re_y_card_vs_rep_{name}_max_rel"] = m1.nanmax0(e)
        k = int(np.nanargmax(np.where(cnt, m1.norm_y_err(ye, ref_reg), -1)))
        out[f"q2_{name}_registered_matched_argmax"] = {"vgs": float(vg[k]), "vds": float(vd[k]),
                                                       "I_A": float(i_m[k]), "g": float(g[k]),
                                                       "re_y_card": float(yc[k]), "re_y_rep": float(yrep[k]),
                                                       "im_y_card": float(mt[ex]["m"][k])}
    caps = {"dd": (a["ac_VD_corr_ym"], mt["VD"]["m"]), "dg": (a["ac_VG_corr_ym"], mt["VG"]["m"]),
            "gg": (a["ac_VG_corr_yg"], mt["VG"]["g"])}
    out["cap_err_matched"] = {k: float(np.max(m.cap_err(e_, c_))) for k, (e_, c_) in caps.items()}
    out["cap_max_matched"] = max(out["cap_err_matched"].values())
    out["rep_im_y_max_abs"] = float(np.nanmax(np.abs(np.r_[mt["VG"]["t"], mt["VD"]["t"]])))
    return out


def freq_test(dev, params, vg: float, vd: float) -> dict:
    out = {}
    for lab, extra in (("rbodymod_1_card", {}), ("rbodymod_0", {"rbodymod": 0})):
        with tempfile.TemporaryDirectory(prefix="cryoml_i12d_") as t:
            td = Path(t)
            lib = m.patched_library_v2(td, {dev.dev_type: [(dev.bin, {**params, **extra})]},
                                       bsim4_state="legacy")
            c = ["alter VG ac = 1", f"alter VG dc = {vg:.6g}", f"alter VD dc = {vd:.6g}", "op"]
            freqs = (1e2, 1e3, 1e4, 1e5, 1e6)
            for f in freqs:
                c += [f"ac lin 1 {f:g} {f:g}", "print real(-i(VD)) imag(-i(VD))", "destroy all"]
            r = m1.run_deck(m.deck_v2("f", lib, m.device_body_v2(dev, False), c, None, m1.TIGHT), td, "f.sp")
        re_ = [float(ln.split("=")[1]) for ln in r["text"].splitlines() if ln.lower().startswith("real(")]
        im_ = [float(ln.split("=")[1]) for ln in r["text"].splitlines() if ln.lower().startswith("imag(")]
        out[lab] = {f"{f:g}": {"re_y_dg": a, "im_y_dg": b} for f, a, b in zip(freqs, re_, im_)}
    return out


def asym_device(tag, dev, corr) -> dict:
    a = np.load(DEP / "arrays" / f"q12_{tag}.npz")
    pm = dev.dev_type == "pmos"
    s = -1.0 if pm else 1.0
    VG, VD = np.meshgrid(s * m1.GRID_VGS, m1.GRID_VDS, indexing="ij")
    g = m.g_terminal(corr, pm, VG, VD, 0.0)
    ic, isw = a["grid_fwd_tight_corr"], a["grid_swap_tight_corr"]
    cc, csw, vz = a["grid_fwd_tight_card"], a["grid_swap_tight_card"], a["grid_swap_tight_card_vz"]
    keep = ((s * VD) > 0) & (np.abs(ic) > m1.I_FLOOR) & (np.abs(cc) > m1.I_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.abs(np.log(np.abs(isw / ic)) - np.log(np.abs(csw / cc)))
    dd = np.where(keep, d, -1.0)
    i, j = np.unravel_index(int(np.argmax(dd)), dd.shape)
    J = np.abs(cc) - np.abs(vz)
    pred = np.abs(1.0 - np.exp(-g)) * J / np.abs(cc)
    jv = int(np.argmin(np.abs(np.abs(m1.GRID_VDS) - abs(VD[i, j]))))
    return {"max": float(d[i, j]), "vgs": float(VG[i, j]), "vds": float(VD[i, j]), "I_A": float(cc[i, j]),
            "g": float(g[i, j]), "J_A": float(J[i, j]), "J_over_I": float(J[i, j] / abs(cc[i, j])),
            "predicted": float(pred[i, j]),
            "pred_vs_obs_max_abs_diff_over_kept": float(np.max(np.abs(pred - d)[keep])) if keep.any() else 0.0,
            "n_points_over_1e-3": int(np.sum(keep & (d > THR["swap_asym"]))),
            "gmin_x_vds_A": float(1e-12 * abs(VD[i, j])),
            "J_vs_vgs_at_vds": {f"{m1.GRID_VGS[k]:.2f}": float(J[k, jv]) for k in (0, 8, 16, 24, 40, 56, 74)}}


def iii_test(tag, dev, params, vg: float, vd: float) -> dict:
    """Card-only op at the asymmetry max point with the M2 card and with
    alpha0 = 0 (impact ionization off); the difference is the impact-
    ionization substrate current at that point."""
    i_on = ops_twice(dev, params, vg, vd)[0]
    i_off = ops_twice(dev, {**params, "alpha0": 0.0}, vg, vd)[0]
    return {"vgs": vg, "vds": vd, "I_card_A": i_on, "I_card_alpha0_0_A": i_off, "difference_A": i_on - i_off}


def main() -> int:
    t0 = time.time()
    q12 = drv.jload(DEP / "q12.json")["devices"]
    summ = drv.jload(DEP / "summary.json")
    devs = drv.devices()
    m2 = drv.jload(DEP / "cards_M2.json")
    st = drv.start_cards()
    corr = drv.full_corr(drv.OUT / "lodo")
    out = {"label": "post hoc (after decision.json); the registered outcome 'not deployable' is unchanged",
           "thresholds": THR, "smoke_devices": list(SMOKE)}

    # A ------------------------------------------------------------------
    A = {}
    for t, r in q12.items():
        A[t] = {"swap_asym_element": r["swap_asym_element_max"], "q2_err_gm": r["q2_err_gm"],
                "q2_err_gds": r["q2_err_gds"], "q2_op_i_rel": r["q2_op_i_rel"], "cap_max": r["cap_max"],
                "cap_dd": r["cap_err_dd"], "cap_dg": r["cap_err_dg"], "cap_gg": r["cap_err_gg"],
                "a2_M2": m2[t].get("a2"), "a2_start": st[t].get("a2")}
    out["A_values"] = A
    out["A_over_threshold"] = {
        "swap_asym": sorted(t for t, v in A.items() if v["swap_asym_element"] > THR["swap_asym"]),
        "q2_ac": sorted(t for t, v in A.items() if max(v["q2_err_gm"], v["q2_err_gds"]) > THR["q2_ac"]),
        "cap": sorted(t for t, v in A.items() if v["cap_max"] > THR["cap"])}
    print("A over threshold:", out["A_over_threshold"], flush=True)

    # B ------------------------------------------------------------------
    B = {}
    for t, dev in devs.items():
        s = -1.0 if dev.dev_type == "pmos" else 1.0
        row = {}
        for name, p in (("published", None), ("start", st[t]), ("M2", m2[t])):
            row[name] = max(drift(ops_twice(dev, p, s * vg, s * vd))
                            for vg, vd in ((0.375, 0.05), (0.9, 0.925), (1.85, 1.85)))
        B[t] = row
    out["B_second_analysis_drift"] = {"what": "max over 3 bias points of |I_op2 - I_op1| / |I_op1|, "
                                              "two op in one card-only process, tight tolerance",
                                      "per_device": B,
                                      "devices_M2_drift_gt_1e-6": sorted(t for t, v in B.items() if v["M2"] > 1e-6),
                                      "devices_M2_a2_gt_1": sorted(t for t in devs if m2[t].get("a2", 0) > 1.0)}
    print("B drift > 1e-6 (M2):", out["B_second_analysis_drift"]["devices_M2_drift_gt_1e-6"], flush=True)
    tag = "nmos_L0p19_W7"
    single = {}
    for k, v in sorted(m2[tag].items()):
        dlt = drift(ops_twice(devs[tag], {k: v}, 0.375, 0.05))
        if dlt > 1e-9:
            single[k] = {"value": v, "drift": dlt}
    loo = {k: drift(ops_twice(devs[tag], {kk: vv for kk, vv in m2[tag].items() if kk != k}, 0.375, 0.05))
           for k in single}
    out["B_single_parameter_search"] = {"device": tag, "bias": [0.375, 0.05],
                                        "M2_full_drift": drift(ops_twice(devs[tag], m2[tag], 0.375, 0.05)),
                                        "single_M2_value_on_published_with_drift": single,
                                        "M2_without_that_parameter_drift": loo}
    print("B single-parameter search:", single, loo, flush=True)

    # C ------------------------------------------------------------------
    C = {}
    for t, dev in devs.items():
        C[t] = q2_device(t, dev, m2[t], corr)
    keys = ("registered_stored", "registered_matched", "identity_matched")
    out["C_q2_recomputed"] = {
        "per_device": C,
        "max_over_devices": {k: max(max(v[f"q2_err_gm_{k}"], v[f"q2_err_gds_{k}"]) for v in C.values())
                             for k in keys},
        "cap_max_matched": max(v["cap_max_matched"] for v in C.values()),
        "devices_over_1e-3": {k: sorted(t for t, v in C.items()
                                        if max(v[f"q2_err_gm_{k}"], v[f"q2_err_gds_{k}"]) > THR["q2_ac"])
                              for k in keys},
        "cap_devices_over_1e-3_matched": sorted(t for t, v in C.items() if v["cap_max_matched"] > THR["cap"])}
    k = out["C_q2_recomputed"]["per_device"]["nmos_L100_W100"]["q2_gm_registered_matched_argmax"]
    out["C2_frequency_test_nmos_L100_W100"] = {"vgs": k["vgs"], "vds": k["vds"],
                                               **freq_test(devs["nmos_L100_W100"], m2["nmos_L100_W100"],
                                                           k["vgs"], k["vds"])}
    print("C:", out["C_q2_recomputed"]["max_over_devices"], out["C_q2_recomputed"]["devices_over_1e-3"],
          "cap matched", out["C_q2_recomputed"]["cap_max_matched"], flush=True)

    # D ------------------------------------------------------------------
    D = {t: asym_device(t, dev, corr) for t, dev in devs.items()}
    iii = {t: iii_test(t, devs[t], m2[t], D[t]["vgs"], D[t]["vds"]) for t in ("nmos_L100_W100", "nmos_L20_W0p64")}
    for t, r in iii.items():
        r["J_A"] = D[t]["J_A"]
        r["gmin_x_vds_A"] = D[t]["gmin_x_vds_A"]
    out["D_exchange_asymmetry"] = {"per_device": D, "impact_ionization_test": iii,
                                   "prereg_bound_assumed_non_channel_current_A": 1.85e-12}
    print("D:", {t: (round(v["max"], 6), round(v["predicted"], 6)) for t, v in D.items()}, flush=True)
    print("D iii:", iii, flush=True)

    # E ------------------------------------------------------------------
    gds = {t: [m1.GDS_VGS[k] for k, f in enumerate(r["q2_gds_curves_introduced"]) if f]
           for t, r in q12.items() if any(r["q2_gds_curves_introduced"])}
    out["E_restrictions"] = {"q1_default_max_rel_1nA": summ["values"]["q1_default_max"],
                             "q1_default_per_device": {t: r["q1_default_max_rel_1nA"] for t, r in q12.items()},
                             "gds_sign_change_curves_abs_vgs": gds,
                             "runtime_grid": summ["runtime_grid"], "runtime_circuits": summ["runtime_circuits"]}
    out["wall_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    drv.jdump(out, OUT / "diagnosis.json")
    print(f"wrote {OUT / 'diagnosis.json'} ({out['wall_s']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
