#!/usr/bin/env python3
"""g_why: where and why the learned correction g (I = I_card x exp(g)) lowers the
paper RRMS on sky130 77 K, and why it does not on IHP SG13G2 300 K.

Light analysis on stored records and stored corrections only: no NGSpice run, no
training.  CPU, single thread.

Inputs
- sky130 77 K: I13-1 state-C starting-card records (direct43_r3_fd cards, the
  training data of g), full-data g = I15-2 'full|sym_e1_l1' (lambda 1 V, the model
  deployed in I16-1 / I17-1), LODO g = I15-2 lodo_sym_e1_l{lambda*_k}; refit
  cards of the full-data deployment (M2, refit with g in the loop) from the stored
  I17-1 Q1 grid currents (interpolated to the measured biases).
- NIST sky130 4 K lvt: I20-1 state-C starting records, full-data g
  'full|sym_e1_l0', LODO g 'selected|<tag>'.
- IHP SG13G2 300 K: I5-1 starting records, LODO g of I17-2 at the per-fold
  lambda*.

Output: out/research/explore/g_why/ (PNG, results.json, g_why.md).

Run: OMP_NUM_THREADS=1 .venv/bin/python -I scripts/research/explore/g_why.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from cryoml.explore.e1_correction import load_devices, score_device  # noqa: E402
from cryoml.explore.i12_1_symm import SymCorrection  # noqa: E402
from cryoml.metrics import clean_current  # noqa: E402

EXP = ROOT / "out/research/explore"
OUT = EXP / "g_why"
KB_Q = 8.617333262e-5  # V/K

C_BLUE, C_ORANGE, C_AQUA, C_VIOLET, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984"

# Region boundaries (overdrive Vov = |Vgs| - Vth_card of the device)
NEAR = 0.10      # V: sub = Vov < -NEAR, near = |Vov| <= NEAR, strong = Vov > NEAR
REGIONS = ("sub", "near", "SI_lin", "SI_sat")
VDS_MIN = 0.005  # V


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def jload(p):
    return json.loads(Path(p).read_text())


def corr_from(path, key):
    return SymCorrection.from_json(jload(path)[key])


def dataset_sky130():
    r = EXP / "I13_1/records"
    recs = load_devices(r / "real_dev") + load_devices(r / "real_test")
    recs.sort(key=lambda d: d.tag)
    full = corr_from(EXP / "I15_2/lodo_sym_e1_l1/corrections.json", "full|sym_e1_l1")
    lam = jload(EXP / "I15_2/decision.json")["lambda_star"]
    cache = {}

    def lodo(tag):
        lk = f"l{int(lam[tag])}"
        # lambda 10 V = the I13-1 g v2 run (M_bias_v2), as in i15_2_ablate.stored_rows
        path, key = ((EXP / "I13_1/lodo/corrections.json", f"M_bias_v2|{tag}") if lk == "l10" else
                     (EXP / f"I15_2/lodo_sym_e1_{lk}/corrections.json", f"sym_e1_{lk}|{tag}"))
        if lk not in cache:
            cache[lk] = jload(path)
        return SymCorrection.from_json(cache[lk][key])
    return dict(name="sky130 77 K", key="sky130", T=77.0, vdd=1.85, floor=1e-8, recs=recs,
                full=full, lodo=lodo, lo="idvg@0.01", hi="idvg@1.85",
                idvd_top="idvd@1.85", idvd_mid="idvd@1.11")


def dataset_nist4():
    recs = load_devices(EXP / "I20_1/start/records_nist4")
    lodo_all = jload(EXP / "I20_1/lodo_nist4/corrections.json")
    recs = [r for r in recs if f"selected|{r.tag}" in lodo_all]     # DEV devices
    recs.sort(key=lambda d: d.tag)
    full = corr_from(EXP / "I20_1/select_nist4/full_corrections.json", "full|sym_e1_l0")
    return dict(name="NIST sky130 4 K lvt", key="nist4", T=4.0, vdd=1.8, floor=1e-10, recs=recs,
                full=full, lodo=lambda t: SymCorrection.from_json(lodo_all[f"selected|{t}"]),
                lo="idvg@d0", hi="idvg@d1", idvd_top="idvd@g6", idvd_mid="idvd@g4")


def dataset_ihp():
    recs = load_devices(EXP / "I5_1/start/records_300K")
    recs.sort(key=lambda d: d.tag)
    lam = jload(EXP / "I17_2/decision.json")["lambda_star"]
    cache = {}

    def lodo(tag):
        lk = f"l{int(lam[tag])}"
        if lk not in cache:
            cache[lk] = jload(EXP / f"I17_2/lodo_sym_e1_{lk}/corrections.json")
        return SymCorrection.from_json(cache[lk][f"sym_e1_{lk}|{tag}"])
    return dict(name="IHP SG13G2 300 K", key="ihp300", T=300.0, vdd=None, floor=1e-10, recs=recs,
                full=None, lodo=lodo, lo="idvg@d0", hi="idvg@d2", idvd_top="idvd@g4",
                idvd_mid="idvd@g2")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def curve(dev, key):
    for c in dev.curves:
        if c.key == key:
            return c
    raise KeyError(key)


def apply_g(corr, dev):
    return corr.apply(dev) if corr is not None else [np.asarray(c.sim, float) for c in dev.curves]


def deriv(x, y, k=None):
    """dy/dx by a Savitzky-Golay filter (quadratic, window ~0.09 V, at least 5
    points); the measured sky130 curves carry ~5-10 % point-to-point noise in
    strong inversion, so a plain central difference is noise-dominated.  The same
    filter is applied to measured and model curves."""
    from scipy.signal import savgol_filter
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx = float(np.median(np.diff(x)))
    win = max(5, int(round(0.09 / abs(dx))) | 1)
    if len(y) < win + 2:
        return np.full_like(y, np.nan)
    d = savgol_filter(y, win, 2, deriv=1, delta=dx)
    d[: win // 2] = np.nan
    d[-(win // 2):] = np.nan
    return d


def vth_elr(vg, i, vds):
    """max-gm linear extrapolation on forward Vg (s V(g,s)), |I|; returns (Vth, Vg at gm max, gm max)."""
    x, y = np.asarray(vg, float), np.abs(i)
    gm = deriv(x, y)
    if not np.isfinite(gm).any():
        return np.nan, np.nan, np.nan
    j = int(np.nanargmax(gm))
    return float(x[j] - y[j] / gm[j] - abs(vds) / 2.0), float(x[j]), float(gm[j])


def ss_min(vg, i, lo, hi, win=4):
    """minimum subthreshold swing (mV/dec): linear fit of |Vg| on log10|I| over
    ``win`` consecutive points, all with lo <= |I| <= hi and I increasing (vg forward-signed)."""
    x, y = np.asarray(vg, float), np.abs(i)
    best = np.nan
    for s in range(len(x) - win + 1):
        yy = y[s:s + win]
        if np.all(yy >= lo) and np.all(yy <= hi) and np.all(np.diff(yy) > 0):
            sl = np.polyfit(np.log10(yy), x[s:s + win], 1)[0] * 1e3
            if sl > 0 and (not np.isfinite(best) or sl < best):
                best = sl
    return best


def vg_at_current(vg, i, icc):
    x, y = np.asarray(vg, float), np.abs(i)
    k = np.flatnonzero((y[:-1] < icc) & (y[1:] >= icc))
    if not k.size:
        return np.nan
    j = int(k[-1])
    a, b = math.log(y[j]) if y[j] > 0 else math.log(icc) - 10, math.log(y[j + 1])
    return float(x[j] + (math.log(icc) - a) / (b - a) * (x[j + 1] - x[j]))


def gds_norm(vd, i, lo, hi):
    """(1/I) dI/dVds (1/V) by a linear fit of |I| on |Vds| over lo..hi."""
    x, y = np.abs(vd), np.abs(i)
    k = (x >= lo) & (x <= hi)
    if k.sum() < 3:
        return np.nan
    sl = np.polyfit(x[k], y[k], 1)[0]
    return float(sl / np.mean(y[k]))


def gds_rise(vd, i, vdd):
    """gds(0.85..1.0 VDD) / gds(0.55..0.7 VDD): >1 means gds rises with Vds (kink-like)."""
    x, y = np.abs(vd), np.abs(i)

    def sl(a, b):
        k = (x >= a * vdd) & (x <= b * vdd)
        return np.polyfit(x[k], y[k], 1)[0] if k.sum() >= 3 else np.nan
    a, b = sl(0.55, 0.70), sl(0.85, 1.0)
    return float(b / a) if (np.isfinite(a) and a > 0) else np.nan


def dev_vdd(ds, dev):
    return ds["vdd"] if ds["vdd"] else float(dev.vdd)


# ---------------------------------------------------------------------------
# per-device point table
# ---------------------------------------------------------------------------
def device_points(ds, dev, gcorr):
    """Per measured point: |Vgs|, |Vds|, curve key/kind, measured, card, card x g,
    log residual, metric weight w = |I_meas| / den (included curve, after trim;
    0 otherwise), region label (needs Vth_card)."""
    sv = -1.0 if dev.family == "pmos" else 1.0
    clo = curve(dev, ds["lo"])
    vth_c = vth_elr(sv * clo.vg, clo.sim, clo.vd[0])[0]
    corr_sims = apply_g(gcorr, dev)
    rows = []
    for c, sg in zip(dev.curves, corr_sims):
        meas = clean_current(c.target)
        sv = -1.0 if dev.family == "pmos" else 1.0       # forward-signed voltages
        vg, vd = sv * np.asarray(c.vg, float), np.abs(np.asarray(c.vd, float))
        sim = np.abs(np.asarray(c.sim, float))
        sg = np.abs(np.asarray(sg, float))
        w = np.zeros(len(meas))
        in_metric = np.zeros(len(meas), bool)
        if c.included and c.den > 0:
            in_metric[c.start:] = True
            w[c.start:] = np.abs(meas[c.start:]) / c.den
        for j in range(len(meas)):
            rows.append((c.key, c.kind, vg[j], vd[j], abs(meas[j]), sim[j], sg[j], w[j], in_metric[j]))
    dt = np.dtype([("key", "U16"), ("kind", "U5"), ("vg", "f8"), ("vd", "f8"), ("im", "f8"),
                   ("ic", "f8"), ("icg", "f8"), ("w", "f8"), ("metric", "?")])
    a = np.array(rows, dtype=dt)
    vov = a["vg"] - vth_c
    reg = np.where(vov < -NEAR, "sub", np.where(vov <= NEAR, "near",
                   np.where(a["vd"] < vov, "SI_lin", "SI_sat")))
    # log residual where the measured current is above the floor and |Vds| >= VDS_MIN
    # (at V(d,s) = 0 the card current is ~1e-17 A and the measured value is SMU offset)
    ok = (a["im"] > ds["floor"]) & (a["ic"] > 0) & (a["vd"] >= VDS_MIN)
    r = np.full(len(a), np.nan)
    r[ok] = np.log(a["im"][ok] / a["ic"][ok])
    gval = np.full(len(a), np.nan)
    pos = a["ic"] > 0
    gval[pos] = np.log(a["icg"][pos] / a["ic"][pos])
    return dict(a=a, vov=vov, reg=reg, r=r, g=gval, vth=vth_c)


# ---------------------------------------------------------------------------
# RRMS and its exact decomposition
# ---------------------------------------------------------------------------
def curve_rrms_parts(dev, sims0, sims1, pts):
    """per included curve: rrms before / after and the exact split of the
    reduction over regions: dr_c = sum_R dS_R / (sqrt(n) den (sqrt(S0)+sqrt(S1)))."""
    out = []
    reg_all = pts["reg"]
    off = 0
    for c, s0, s1 in zip(dev.curves, sims0, sims1):
        n_c = len(c.target)
        reg = reg_all[off:off + n_c]
        off += n_c
        if not (c.included and c.den > 0):
            continue
        meas = clean_current(c.target)[c.start:]
        e0 = (np.asarray(s0, float)[c.start:] - meas) ** 2
        e1 = (np.asarray(s1, float)[c.start:] - meas) ** 2
        n = len(meas)
        S0, S1 = e0.sum() / n, e1.sum() / n
        r0, r1 = math.sqrt(S0) / c.den, math.sqrt(S1) / c.den
        fac = 1.0 / (n * c.den * (math.sqrt(S0) + math.sqrt(S1))) if (S0 + S1) > 0 else 0.0
        rr = reg[c.start:]
        by_reg = {R: float(fac * (e0[rr == R].sum() - e1[rr == R].sum())) for R in REGIONS}
        out.append(dict(key=c.key, kind=c.kind, r0=r0, r1=r1, by_reg=by_reg))
    return out


def curve_class(ds, key, kind):
    if kind == "idvd":
        return "IdVd"
    return "IdVg_lowVds" if key == ds["lo"] else "IdVg_highVds"


# ---------------------------------------------------------------------------
# residual-map consistency on a normalized common grid
# ---------------------------------------------------------------------------
NB = 25   # bins over 0..1 of |V| / VDD (and Vov / VDD over -0.5..1)


def binned_maps(ds, devs_pts, axis="raw"):
    """per device: dict bin -> (sum w2 r, sum w2, sum r, n, sum w2 g, sum g)."""
    maps = []
    for dev, P in devs_pts:
        vdd = dev_vdd(ds, dev)
        a = P["a"]
        x = (a["vg"] if axis == "raw" else P["vov"]) / vdd
        y = a["vd"] / vdd
        if axis == "raw":
            bx = np.floor(x * NB).astype(int)
        else:
            bx = np.floor((x + 0.5) * NB).astype(int)
        by = np.floor(np.minimum(y, 0.9999) * NB).astype(int)
        ok = np.isfinite(P["r"])
        m = {}
        for i in np.flatnonzero(ok):
            k = (int(bx[i]), int(by[i]))
            w2 = a["w"][i] ** 2
            s = m.setdefault(k, [0.0, 0.0, 0.0, 0, 0.0, 0.0])
            s[0] += w2 * P["r"][i]
            s[1] += w2
            s[2] += P["r"][i]
            s[3] += 1
            s[4] += w2 * P["g"][i]
            s[5] += P["g"][i]
        maps.append(m)
    return maps


def shared_stats(maps, weighted):
    """family-level: residual per device per bin; shared component = family mean
    (in-sample) and leave-one-device-out mean; fractions of the sum of squares
    explained, mean pairwise Pearson correlation, fraction explained by g."""
    keys = sorted(set().union(*[set(m) for m in maps]))
    nd = len(maps)
    R = np.full((nd, len(keys)), np.nan)
    W = np.zeros((nd, len(keys)))
    G = np.full((nd, len(keys)), np.nan)
    for d, m in enumerate(maps):
        for j, k in enumerate(keys):
            if k in m:
                s = m[k]
                if weighted:
                    if s[1] > 0:
                        R[d, j], W[d, j], G[d, j] = s[0] / s[1], s[1], s[4] / s[1]
                else:
                    R[d, j], W[d, j], G[d, j] = s[2] / s[3], s[3], s[5] / s[3]
    have = np.isfinite(R)
    cnt = have.sum(0)
    col = cnt >= 3
    R, W, G, have = R[:, col], W[:, col], G[:, col], have[:, col]
    Rz = np.where(have, R, 0.0)
    W = np.where(have, W, 0.0)
    tot = np.sum(W * Rz ** 2)
    # shared component per bin = W-weighted mean over devices (the minimiser of the
    # weighted sum of squares); LOO: the same mean without the scored device
    mean_in = np.sum(W * Rz, 0) / np.maximum(np.sum(W, 0), 1e-300)
    res_in = np.sum(W * np.where(have, (R - mean_in) ** 2, 0.0))
    res_loo, res_g = 0.0, np.sum(W * np.where(have, (R - G) ** 2, 0.0))
    for d in range(nd):
        Wo = W.copy()
        Wo[d] = 0.0
        so = np.sum(Wo, 0)
        m_o = np.where(so > 0, np.sum(Wo * Rz, 0) / np.maximum(so, 1e-300), 0.0)
        res_loo += np.sum(W[d] * np.where(have[d], (R[d] - m_o) ** 2, 0.0))
    cors = []
    for i in range(nd):
        for j in range(i + 1, nd):
            k = have[i] & have[j]
            if k.sum() >= 5 and np.std(R[i, k]) > 0 and np.std(R[j, k]) > 0:
                cors.append(float(np.corrcoef(R[i, k], R[j, k])[0, 1]))
    rms = math.sqrt(tot / np.sum(W * have)) if np.sum(W * have) > 0 else np.nan
    return dict(n_dev=nd, n_bins=int(col.sum()), rms_resid=rms,
                frac_shared_insample=float(1 - res_in / tot) if tot > 0 else np.nan,
                frac_shared_loo=float(1 - res_loo / tot) if tot > 0 else np.nan,
                frac_removed_by_g=float(1 - res_g / tot) if tot > 0 else np.nan,
                mean_pair_corr=float(np.mean(cors)) if cors else np.nan,
                median_pair_corr=float(np.median(cors)) if cors else np.nan, n_pairs=len(cors))


# ---------------------------------------------------------------------------
# physics metrics
# ---------------------------------------------------------------------------
def phys_metrics(ds, dev, sims_by_label):
    """per device: Vth (ELR, low Vds), SS (low and high Vds), gm peak position and
    value, gm(top Vgs)/gm_max, DIBL (constant current), gds/I in saturation at top
    and mid Vgs, gds rise at top Vgs.  For each current set in sims_by_label."""
    vdd = dev_vdd(ds, dev)
    lo, hi = curve(dev, ds["lo"]), curve(dev, ds["hi"])
    top, mid = curve(dev, ds["idvd_top"]), curve(dev, ds["idvd_mid"])
    idx = {c.key: i for i, c in enumerate(dev.curves)}
    wl = dev.W_um / dev.L_um
    sv = -1.0 if dev.family == "pmos" else 1.0
    xlo, xhi = sv * np.asarray(lo.vg, float), sv * np.asarray(hi.vg, float)
    out = {}
    for lab, sims in sims_by_label.items():
        il, ih = np.asarray(sims[idx[lo.key]], float), np.asarray(sims[idx[hi.key]], float)
        it, im = np.asarray(sims[idx[top.key]], float), np.asarray(sims[idx[mid.key]], float)
        if lab == "meas":
            il, ih, it, im = (clean_current(x) for x in (il, ih, it, im))
        vth, vpk, gmax = vth_elr(xlo, il, lo.vd[0])
        gm = deriv(xlo, np.abs(il))
        gm_top = gm[np.isfinite(gm)][-1] if np.isfinite(gm).any() else np.nan
        # current window for SS: above the measured floor x 3, below I at Vth
        i_th = np.interp(vth, xlo, np.abs(il)) if np.isfinite(vth) else np.nan
        i_th_h = np.interp(vth, xhi, np.abs(ih)) if np.isfinite(vth) else np.nan
        lo_cur = 3 * ds["floor"]
        icc = max(1e-7 * wl, 30 * ds["floor"])
        out[lab] = dict(
            vth=vth, vth_cc=vg_at_current(xlo, il, icc), vg_gm_peak=vpk, vov_gm_peak=vpk - vth if np.isfinite(vth) else np.nan,
            gm_max=gmax, gm_top_over_max=float(gm_top / gmax) if gmax else np.nan,
            ss_low=ss_min(xlo, il, lo_cur, 0.3 * i_th),
            ss_high=ss_min(xhi, ih, lo_cur, 0.3 * i_th_h),
            ss_low_nofloor=ss_min(xlo, il, 1e-14, 0.3 * i_th),
            dibl_mV_per_V=(vg_at_current(xlo, il, icc) - vg_at_current(xhi, ih, icc))
            / (abs(hi.vd[0]) - abs(lo.vd[0])) * 1e3,
            i_lin_top=float(np.abs(il)[np.argmax(xlo)]),
            lin_top_over_08=float(np.abs(il)[np.argmax(xlo)] / np.interp(0.8 * np.max(xlo), xlo, np.abs(il))),
            i_sat_top=float(np.abs(ih)[np.argmax(xhi)]),
            # channel conductance I/Vds at 0.15 VDD over that at 0.03 VDD, top-|Vgs| Id-Vd
            # (one sweep); a MOSFET in the linear region with ohmic access gives <= 1
            g_ratio=float((np.interp(0.15 * vdd, np.abs(top.vd), np.abs(it)) / (0.15 * vdd))
                          / (np.interp(0.03 * vdd, np.abs(top.vd), np.abs(it)) / (0.03 * vdd))),
            gds_top=gds_norm(top.vd, it, 0.65 * vdd, 0.97 * vdd),
            gds_mid=gds_norm(mid.vd, im, 0.65 * vdd, 0.97 * vdd),
            gds_rise_top=gds_rise(top.vd, it, vdd))
    return out


# ---------------------------------------------------------------------------
# sky130 M2 (refit-with-g) cards from the stored Q1 grids
# ---------------------------------------------------------------------------
GRID_VGS = np.round(np.arange(75) * 0.025, 6)
GRID_VDS = np.round(-1.85 + np.arange(149) * 0.025, 6)


def m2_card_sims(dev):
    """card-only current of the I17-1 deployed M2 card on the measured biases,
    by bilinear interpolation of the stored 25 mV forward grid (|Vgs| rows, signed
    V(d) columns; current -i(VD)); log-linear in |Vgs| where both neighbours > 0."""
    z = np.load(EXP / "I17_1/deploy_sel/arrays" / f"q12_{dev.tag}.npz")
    grid = z["grid_fwd_default_card"]
    sgn = -1.0 if dev.family == "pmos" else 1.0
    out = []
    for c in dev.curves:
        vg, vd = np.abs(np.asarray(c.vg, float)), np.asarray(c.vd, float)   # vd: signed V(d,s), grid column axis
        res = np.empty(len(vg))
        for j in range(len(vg)):
            # rows: |Vgs|; columns: signed V(d,s)
            fi = np.clip(vg[j] / 0.025, 0, 74 - 1e-9)
            i0 = int(fi)
            ti = fi - i0
            fj = np.clip((vd[j] + 1.85) / 0.025, 0, 148 - 1e-9)
            j0 = int(fj)
            tj = fj - j0
            rows = []
            for ii in (i0, i0 + 1):
                rows.append((1 - tj) * grid[ii, j0] + tj * grid[ii, j0 + 1])
            a, b = rows
            if a * b > 0:
                res[j] = math.copysign(math.exp((1 - ti) * math.log(abs(a)) + ti * math.log(abs(b))), a)
            else:
                res[j] = (1 - ti) * a + ti * b
        out.append(sgn * res)     # records hold the forward current with positive sign
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_g_maps(ds, vth_fam, path):
    corr = ds["full"]
    n = 149
    ax_v = np.linspace(0, 1.85, n)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    for row, fam in enumerate(("nmos", "pmos")):
        pm = fam == "pmos"
        s = -1.0 if pm else 1.0
        VG, VD = np.meshgrid(ax_v, ax_v, indexing="xy")
        G = corr.g_v(pm, s * VG, s * VD)
        lim = float(np.max(np.abs(G)))
        ax = axs[row, 0]
        im = ax.pcolormesh(ax_v, ax_v, G, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
        cs = ax.contour(ax_v, ax_v, G, levels=np.arange(-1, 1.01, 0.1), colors="k", linewidths=0.5)
        ax.clabel(cs, fmt="%.1f", fontsize=7)
        v = np.array(vth_fam[fam])
        vm = float(np.median(v))
        for x in (v.min(), v.max()):
            ax.axvline(x, color="k", ls=":", lw=0.9)
        ax.axvspan(vm - NEAR, vm + NEAR, color="k", alpha=0.08)
        ax.axvline(vm, color="k", ls="--", lw=1.2)
        xs = np.linspace(vm, 1.85, 50)
        ax.plot(xs, xs - vm, color="k", lw=1.2, ls="-.")
        ax.text(max(vm - 0.45, 0.02), 1.7, "sub-\nthreshold", fontsize=8)
        ax.text(vm + 0.6, 1.7, "saturation", fontsize=8)
        ax.text(1.35, 0.15, "linear", fontsize=8)
        ax.set_title(f"{fam}: g(|Vgs|, |Vds|), full-data model\nVth_card median {vm:.3f} V (dashed), "
                     f"range {v.min():.3f}-{v.max():.3f} V (dotted)", fontsize=9)
        ax.set_xlabel("|Vgs| (V)")
        ax.set_ylabel("|Vds| (V)")
        fig.colorbar(im, ax=ax, label="g = ln(I_corr / I_card)")
        # cuts vs Vgs
        ax = axs[row, 1]
        cols = plt.cm.viridis(np.linspace(0, 0.9, 6))
        for k, vds in enumerate((0.01, 0.37, 0.74, 1.11, 1.48, 1.85)):
            ax.plot(ax_v, corr.g_v(pm, s * ax_v, s * vds), color=cols[k], lw=2, label=f"|Vds| = {vds:.2f} V")
        ax.axvspan(0, vm - NEAR, color=C_GRAY, alpha=0.12)
        ax.axvspan(vm - NEAR, vm + NEAR, color=C_GRAY, alpha=0.25)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("|Vgs| (V)")
        ax.set_ylabel("g")
        ax.set_title(f"{fam}: g vs |Vgs| (gray: subthreshold / near-threshold band of median Vth_card)", fontsize=9)
        ax.legend(fontsize=7)
        # cuts vs Vds
        ax = axs[row, 2]
        for k, vgs in enumerate((0.37, 0.74, 1.11, 1.48, 1.85)):
            gg = corr.g_v(pm, s * vgs, s * ax_v)
            ax.plot(ax_v, gg, color=cols[k], lw=2, label=f"|Vgs| = {vgs:.2f} V (Vov {vgs - vm:+.2f})")
            if vgs - vm > 0:
                ax.plot([vgs - vm], [float(corr.g_v(pm, s * vgs, s * (vgs - vm)))], "o", color=cols[k], ms=6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("|Vds| (V)")
        ax.set_ylabel("g")
        ax.set_title(f"{fam}: g vs |Vds| (dot: |Vds| = Vov, linear/saturation boundary)", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("sky130 77 K full-data g v2 (I15-2 full|sym_e1_l1, lambda 1 V); exp(g) multiplies the card current",
                 fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_residual_vs_vov(ds, devs_pts, path, title_extra=""):
    """per family: ln(I_meas/I_card) vs Vov for each curve class, all devices,
    with g overlaid; one panel per (family, curve class)."""
    classes = ("IdVg_lowVds", "IdVg_highVds", "IdVd")
    fams = sorted({d.family for d, _ in devs_pts})
    fig, axs = plt.subplots(len(fams), 3, figsize=(15, 4.3 * len(fams)), constrained_layout=True, squeeze=False)
    for fi, fam in enumerate(fams):
        for ci, cl in enumerate(classes):
            ax = axs[fi, ci]
            for dev, P in devs_pts:
                if dev.family != fam:
                    continue
                a = P["a"]
                cls = np.array([curve_class(ds, k, kd) for k, kd in zip(a["key"], a["kind"])])
                k = (cls == cl) & np.isfinite(P["r"])
                if cl == "IdVd":
                    x = a["vd"][k]
                else:
                    x = P["vov"][k]
                ax.plot(x, P["r"][k], ".", ms=1.5, color=C_BLUE, alpha=0.35)
                ax.plot(x, P["g"][k], ".", ms=1.0, color=C_ORANGE, alpha=0.35)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_ylim(-2.5, 2.5)
            if cl == "IdVd":
                ax.set_xlabel("|Vds| (V)")
            else:
                ax.axvspan(-NEAR, NEAR, color=C_GRAY, alpha=0.2)
                ax.set_xlabel("Vov = |Vgs| - Vth_card (V)")
            ax.set_ylabel("ln(I_meas / I_card)")
            ax.set_title(f"{fam} {cl}: blue = residual (all devices), orange = g", fontsize=9)
    fig.suptitle(f"{ds['name']}: starting-card log residual and g, points with |I_meas| > {ds['floor']:.0e} A"
                 + title_extra, fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fig_rrms_split(split_tab, path):
    classes = ("IdVg_lowVds", "IdVg_highVds", "IdVd")
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    x = np.arange(len(REGIONS))
    wdt = 0.26
    cols = (C_BLUE, C_ORANGE, C_AQUA)
    for k, cl in enumerate(classes):
        vals = [split_tab[cl][R] for R in REGIONS]
        ax.bar(x + (k - 1) * wdt, vals, wdt - 0.02, color=cols[k], label=cl)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, REGIONS)
    ax.set_ylabel("share of all-device RRMS reduction")
    ax.set_title("sky130 77 K: split of the RRMS reduction (starting card -> starting card x g_full)\n"
                 "by bias region and curve class (exact decomposition, shares sum to 1)", fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_phys(ph, path):
    keys = [("ss_low", "SS at low Vds (mV/dec)"), ("vov_gm_peak", "Vov at gm peak, low Vds (V)"),
            ("gm_top_over_max", "gm(|Vgs|=1.85) / gm_max, low Vds"), ("gds_top", "gds/Id in sat., |Vgs| = 1.85 (1/V)"),
            ("dibl_mV_per_V", "DIBL (mV/V)"), ("g_ratio", "G(0.15 VDD) / G(0.03 VDD), G = I/Vds, top |Vgs|")]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (k, lab) in zip(axs.ravel(), keys):
        for fam, mk in (("nmos", "o"), ("pmos", "s")):
            tags = [t for t in ph if t.startswith(fam)]
            m = np.array([ph[t]["meas"][k] for t in tags])
            c = np.array([ph[t]["card"][k] for t in tags])
            g = np.array([ph[t]["card_g"][k] for t in tags])
            ax.plot(m, c, mk, color=C_BLUE, ms=7, mfc="none" if fam == "pmos" else C_BLUE,
                    label=f"card ({fam})")
            ax.plot(m, g, mk, color=C_ORANGE, ms=7, mfc="none" if fam == "pmos" else C_ORANGE,
                    label=f"card x exp(g) ({fam})")
            for a, b, cc in zip(m, c, g):
                ax.plot([a, a], [b, cc], color=C_GRAY, lw=0.6)
        v = np.array([ph[t][s][k] for t in ph for s in ("meas", "card", "card_g")], float)
        v = v[np.isfinite(v)]
        if v.size:
            lo_, hi_ = v.min(), v.max()
            pad = 0.05 * (hi_ - lo_ + 1e-12)
            ax.plot([lo_ - pad, hi_ + pad], [lo_ - pad, hi_ + pad], "k--", lw=0.7)
        if k == "ss_low":
            ax.axhline(KB_Q * 77 * math.log(10) * 1e3, color=C_VIOLET, lw=0.8)
            ax.text(ax.get_xlim()[0], KB_Q * 77 * math.log(10) * 1e3 + 0.5, " kT/q ln10 at 77 K", color=C_VIOLET,
                    fontsize=7)
        ax.set_xlabel(f"measured: {lab}", fontsize=8)
        ax.set_ylabel("model", fontsize=8)
        ax.set_title(lab, fontsize=9)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("sky130 77 K, 18 devices: device metrics, model vs measured (filled nMOS, open pMOS; "
                 "dashed = equality)", fontsize=11)
    fig.savefig(path, dpi=105)
    plt.close(fig)


def fig_profiles(ds, devs_pts, path, glabel):
    """log residual along the three curve classes, per device (thin) and the
    family mean over devices (black); g at the same points (orange, mean over
    devices).  x normalized by each device's VDD."""
    fams = ("nmos", "pmos")
    cols = ((ds["lo"], "idvg", "Id-Vg, low |Vds|"), (ds["hi"], "idvg", "Id-Vg, high |Vds|"),
            (ds["idvd_top"], "idvd", "Id-Vd, top |Vgs|"))
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    grid = np.linspace(-0.3, 1.0, 131)
    for fi, fam in enumerate(fams):
        for ci, (key, kind, lab) in enumerate(cols):
            ax = axs[fi, ci]
            rs, gs = [], []
            for dev, P in devs_pts:
                if dev.family != fam:
                    continue
                vdd = dev_vdd(ds, dev)
                k = P["a"]["key"] == key
                x = (P["a"]["vg"][k] if kind == "idvg" else P["a"]["vd"][k]) / vdd
                r, g = P["r"][k], P["g"][k]
                ax.plot(x, r, "-", color=C_BLUE, lw=0.6, alpha=0.45)
                o = np.argsort(x)
                ri = np.interp(grid, x[o], r[o], left=np.nan, right=np.nan)
                gi = np.interp(grid, x[o], g[o], left=np.nan, right=np.nan)
                ri[~np.isfinite(np.interp(grid, x[o], np.where(np.isfinite(r[o]), 1.0, np.nan),
                                          left=np.nan, right=np.nan))] = np.nan
                rs.append(ri)
                gs.append(gi)
            if rs:
                R, Gm = np.array(rs), np.array(gs)
                n = np.sum(np.isfinite(R), 0)
                with np.errstate(all="ignore"):
                    mr = np.where(n >= 3, np.nanmean(R, 0), np.nan)
                    mg = np.where(n >= 3, np.nanmean(Gm, 0), np.nan)
                ax.plot(grid, mr, "k-", lw=2.2, label="mean residual over devices")
                ax.plot(grid, mg, "-", color=C_ORANGE, lw=2.2, label=f"g ({glabel})")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlim(-0.05 if kind == "idvd" else -0.3, 1.0)
            ax.set_xlabel(("|Vgs|" if kind == "idvg" else "|Vds|") + " / VDD")
            ax.set_ylabel("ln(I_meas / I_card)")
            ax.set_title(f"{fam}: {lab} ({key})", fontsize=9)
            if fi == 0 and ci == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{ds['name']}: starting-card log residual per device (blue), mean (black), g (orange); "
                 f"|I_meas| > {ds['floor']:.0e} A", fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fig_idvg_devices(ds, path):
    """measured, card, card x g on the low-|Vds| Id-Vg of every device (linear current)."""
    recs = ds["recs"]
    nc = 6
    nr = int(math.ceil(len(recs) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(19, 3.0 * nr), constrained_layout=True)
    for ax, d in zip(axs.ravel(), recs):
        i = [k for k, c in enumerate(d.curves) if c.key == ds["lo"]][0]
        c = d.curves[i]
        cg = ds["full"].apply(d)[i]
        x = np.abs(np.asarray(c.vg, float))
        meas = clean_current(c.target)
        r0 = math.sqrt(np.mean((np.asarray(c.sim)[c.start:] - meas[c.start:]) ** 2)) / c.den
        r1 = math.sqrt(np.mean((np.asarray(cg)[c.start:] - meas[c.start:]) ** 2)) / c.den
        ax.plot(x, np.abs(meas) * 1e6, ".", color="k", ms=2.5, label="measured")
        ax.plot(x, np.abs(c.sim) * 1e6, "-", color=C_BLUE, lw=1.6, label="card")
        ax.plot(x, np.abs(cg) * 1e6, "-", color=C_ORANGE, lw=1.6, label="card x exp(g)")
        ax.set_title(f"{d.tag}: curve RRMS {r0:.3f} -> {r1:.3f}", fontsize=8)
        ax.set_xlabel("|Vgs| (V)", fontsize=7)
        ax.set_ylabel("|Id| (uA)", fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axs.ravel()[len(recs):]:
        ax.axis("off")
    axs.ravel()[0].legend(fontsize=7)
    fig.suptitle(f"{ds['name']}: Id-Vg at |Vds| = 0.01 V, starting card and card x exp(g_full), no refit",
                 fontsize=11)
    fig.savefig(path, dpi=90)
    plt.close(fig)


ONSET_X = (0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)


def onset_table(ds, devs_pts, key_name="idvd_top"):
    """log residual and g on the top-|Vgs| Id-Vd curve at |Vds|/VDD = ONSET_X:
    per family, mean and SD over devices (linear interpolation along the curve)."""
    out = {}
    for fam in ("nmos", "pmos"):
        R, G = [], []
        for dev, P in devs_pts:
            if dev.family != fam:
                continue
            k = P["a"]["key"] == ds[key_name]
            x = P["a"]["vd"][k] / dev_vdd(ds, dev)
            r, g = P["r"][k], P["g"][k]
            ok = np.isfinite(r)
            o = np.argsort(x[ok])
            R.append(np.interp(ONSET_X, x[ok][o], r[ok][o], left=np.nan))
            o2 = np.argsort(x)
            G.append(np.interp(ONSET_X, x[o2], g[o2]))
        R, G = np.array(R), np.array(G)
        with np.errstate(all="ignore"):
            out[fam] = {f"{xv:g}": dict(n=int(np.sum(np.isfinite(R[:, j]))), mean_r=float(np.nanmean(R[:, j])),
                                        sd_r=float(np.nanstd(R[:, j], ddof=1)), mean_g=float(np.nanmean(G[:, j])),
                                        n_neg=int(np.sum(R[:, j] < 0)), n_pos=int(np.sum(R[:, j] > 0)))
                        for j, xv in enumerate(ONSET_X)}
    return out


def lin_top_table(ds, devs_pts):
    """log residual and g on the low-|Vds| Id-Vg at |Vgs|/VDD in (0.8, 0.9, 1.0)."""
    out = {}
    xs = (0.8, 0.9, 1.0)
    for fam in ("nmos", "pmos"):
        R, G = [], []
        for dev, P in devs_pts:
            if dev.family != fam:
                continue
            k = P["a"]["key"] == ds["lo"]
            x = P["a"]["vg"][k] / dev_vdd(ds, dev)
            o = np.argsort(x)
            R.append(np.interp(xs, x[o], P["r"][k][o]))
            G.append(np.interp(xs, x[o], P["g"][k][o]))
        R, G = np.array(R), np.array(G)
        with np.errstate(all="ignore"):
            out[fam] = {f"{xv:g}": dict(n=int(np.sum(np.isfinite(R[:, j]))), mean_r=float(np.nanmean(R[:, j])),
                                        sd_r=float(np.nanstd(R[:, j], ddof=1)), mean_g=float(np.nanmean(G[:, j])),
                                        n_neg=int(np.sum(R[:, j] < 0)))
                        for j, xv in enumerate(xs)}
    return out


def fig_idvd_onset(ds, path, fam="pmos", xmax=0.7):
    """top-|Vgs| Id-Vd near the origin for every device of one family."""
    recs = [d for d in ds["recs"] if d.family == fam]
    nc = 5
    nr = int(math.ceil(len(recs) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(17, 3.2 * nr), constrained_layout=True, squeeze=False)
    for ax, d in zip(axs.ravel(), recs):
        i = [k for k, c in enumerate(d.curves) if c.key == ds["idvd_top"]][0]
        c = d.curves[i]
        cg = (ds["full"].apply(d) if ds["full"] is not None else ds["lodo"](d.tag).apply(d))[i]
        x = np.abs(np.asarray(c.vd, float))
        k = x <= xmax
        ax.plot(x[k], np.abs(clean_current(c.target))[k] * 1e6, ".", color="k", ms=3, label="measured")
        ax.plot(x[k], np.abs(c.sim)[k] * 1e6, "-", color=C_BLUE, lw=1.6, label="card")
        ax.plot(x[k], np.abs(cg)[k] * 1e6, "-", color=C_ORANGE, lw=1.6, label="card x exp(g)")
        ax.set_title(d.tag, fontsize=8)
        ax.set_xlabel("|Vds| (V)", fontsize=7)
        ax.set_ylabel("|Id| (uA)", fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axs.ravel()[len(recs):]:
        ax.axis("off")
    axs.ravel()[0].legend(fontsize=7)
    fig.suptitle(f"{ds['name']} {fam}: Id-Vd at top |Vgs| ({ds['idvd_top']}), |Vds| <= {xmax} V; starting card and "
                 f"card x exp(g), no refit", fontsize=11)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def fig_datasets(ds_res, path):
    labels = list(ds_res)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    fams = ("nmos", "pmos")
    x = np.arange(len(labels))
    for ax, (k, lab) in zip(axs, (("rms_resid", "RMS metric-weighted log residual"),
                                  ("frac_shared_loo", "fraction of residual SS explained\nby LOO family mean (raw bias grid)"),
                                  ("mean_pair_corr", "mean pairwise correlation of device\nresidual maps (raw bias grid)"))):
        for j, fam in enumerate(fams):
            vals = [ds_res[d]["shared_w_raw"].get(fam, {}).get(k, np.nan) for d in labels]
            ax.bar(x + (j - 0.5) * 0.38, vals, 0.36, color=(C_BLUE, C_ORANGE)[j], label=fam)
        ax.set_xticks(x, labels, fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(lab, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Starting-card residual ln(I_meas / I_card), weights w^2 = (|I_meas| / curve denominator)^2 "
                 "(first-order RRMS weight), 0.04 VDD bins", fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_maps_grid(ds, devs_pts, path):
    """per-device metric-weighted residual on the raw (|Vgs|/VDD, |Vds|/VDD) bins:
    devices x bins image per family, columns ordered by bin."""
    fams = sorted({d.family for d, _ in devs_pts})
    fig, axs = plt.subplots(1, len(fams), figsize=(7.5 * len(fams), 4.2), constrained_layout=True, squeeze=False)
    for ax, fam in zip(axs[0], fams):
        sub = [(d, P) for d, P in devs_pts if d.family == fam]
        maps = binned_maps(ds, sub, "raw")
        keys = sorted(set().union(*[set(m) for m in maps]), key=lambda k: (k[1], k[0]))
        M = np.full((len(maps), len(keys)), np.nan)
        for i, m in enumerate(maps):
            for j, k in enumerate(keys):
                if k in m and m[k][1] > 0:
                    M[i, j] = m[k][0] / m[k][1]
        keep = np.sum(np.isfinite(M), 0) >= 3
        M = M[:, keep]
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1), interpolation="nearest")
        ax.set_yticks(range(len(sub)), [d.tag for d, _ in sub], fontsize=6)
        ax.set_xlabel("bias bin (sorted by |Vds| then |Vgs|), metric-weighted bins only")
        ax.set_title(f"{ds['name']} {fam}: per-device residual per bin", fontsize=9)
        fig.colorbar(im, ax=ax, label="ln(I_meas / I_card)")
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def summ(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if not v.size:
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan)
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                median=float(np.median(v)))


def analyse_dataset(ds, use_full=True):
    """residual tables, shared-pattern stats, RRMS before/after g (full and LODO)."""
    recs = ds["recs"]
    res = dict(name=ds["name"], n=len(recs), floor=ds["floor"])
    gsel = (lambda d: ds["full"]) if (use_full and ds["full"] is not None) else (lambda d: ds["lodo"](d.tag))
    res["g_used_for_maps"] = "full-data" if (use_full and ds["full"] is not None) else "LODO (held-out device)"
    devs_pts = [(d, device_points(ds, d, gsel(d))) for d in recs]
    # RRMS start / start x g_full / start x g_lodo (no refit)
    rows = {}
    for d in recs:
        sims0 = [np.asarray(c.sim, float) for c in d.curves]
        r0 = score_device(d, sims0)["rrms"]
        rl = score_device(d, ds["lodo"](d.tag).apply(d))["rrms"]
        rf = score_device(d, ds["full"].apply(d))["rrms"] if ds["full"] is not None else np.nan
        rows[d.tag] = dict(start=r0, start_g_full=rf, start_g_lodo=rl, family=d.family)
    res["rrms_rows"] = rows
    for k in ("start", "start_g_full", "start_g_lodo"):
        v = np.array([rows[t][k] for t in rows])
        fam = {f: float(np.nanmean([rows[t][k] for t in rows if rows[t]["family"] == f])) for f in ("nmos", "pmos")}
        res[f"rrms_{k}"] = dict(all_device=float(np.nanmean(v)), combined=float(np.mean(list(fam.values()))), **fam)
    lr = np.log(np.array([rows[t]["start_g_lodo"] / rows[t]["start"] for t in rows]))
    res["gmr_lodo_noref"] = float(np.exp(np.nanmean(lr)))
    res["n_lower_lodo_noref"] = int(np.sum(lr < 0))
    # per-region residual (log), per device then across devices
    reg_tab = {}
    for fam in ("all", "nmos", "pmos"):
        for R in REGIONS:
            per_dev_r, per_dev_rg, per_dev_abs, per_dev_wr, npts, ndev = [], [], [], [], 0, 0
            for d, P in devs_pts:
                if fam != "all" and d.family != fam:
                    continue
                k = (P["reg"] == R) & np.isfinite(P["r"])
                if k.sum() < 3:
                    continue
                ndev += 1
                npts += int(k.sum())
                per_dev_r.append(float(np.mean(P["r"][k])))
                per_dev_rg.append(float(np.mean(P["r"][k] - P["g"][k])))
                per_dev_abs.append(float(np.sqrt(np.mean(P["r"][k] ** 2))))
                w2 = P["a"]["w"][k] ** 2
                per_dev_wr.append(float(np.sum(w2 * P["r"][k]) / np.sum(w2)) if w2.sum() > 0 else np.nan)
            reg_tab[f"{fam}|{R}"] = dict(n_dev=ndev, n_pts=npts, mean_r=summ(per_dev_r), mean_r_after_g=summ(per_dev_rg),
                                         rms_r=summ(per_dev_abs), wmean_r=summ(per_dev_wr))
    res["region_resid"] = reg_tab
    # shared-pattern statistics per family, raw and Vov grids, weighted and unweighted
    for wtd in (True, False):
        for axis in ("raw", "vov"):
            key = f"shared_{'w' if wtd else 'u'}_{axis}"
            res[key] = {}
            for fam in ("nmos", "pmos"):
                sub = [(d, P) for d, P in devs_pts if d.family == fam]
                if len(sub) >= 3:
                    res[key][fam] = shared_stats(binned_maps(ds, sub, axis), wtd)
    # IHP by class too
    if ds["key"] == "ihp300":
        res["shared_w_raw_by_class"] = {}
        for cls in ("hv", "lv"):
            for fam in ("nmos", "pmos"):
                sub = [(d, P) for d, P in devs_pts if d.family == fam and d.tag.startswith(cls)]
                res["shared_w_raw_by_class"][f"{cls}_{fam}"] = shared_stats(binned_maps(ds, sub, "raw"), True)
    return res, devs_pts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    # ---------------- sky130 ----------------
    sky = dataset_sky130()
    res_sky, pts_sky = analyse_dataset(sky, use_full=True)
    vth_fam = {f: [P["vth"] for d, P in pts_sky if d.family == f] for f in ("nmos", "pmos")}
    res_sky["vth_card"] = {d.tag: P["vth"] for d, P in pts_sky}
    fig_g_maps(sky, vth_fam, OUT / "01_g_full_sky130_maps.png")
    # RRMS split by region x curve class (start -> start x g_full)
    split = {cl: {R: 0.0 for R in REGIONS} for cl in ("IdVg_lowVds", "IdVg_highVds", "IdVd")}
    split_fam = {f: {cl: {R: 0.0 for R in REGIONS} for cl in split} for f in ("nmos", "pmos")}
    split_tag = {}
    per_dev_split = {}
    for d, P in pts_sky:
        sims0 = [np.asarray(c.sim, float) for c in d.curves]
        sims1 = sky["full"].apply(d)
        parts = curve_rrms_parts(d, sims0, sims1, P)
        nc = len(parts)
        per_dev_split[d.tag] = {}
        for p in parts:
            cl = curve_class(sky, p["key"], p["kind"])
            for R in REGIONS:
                split[cl][R] += p["by_reg"][R] / nc / len(pts_sky)
                split_fam[d.family][cl][R] += p["by_reg"][R] / nc / len(pts_sky)
                split_tag[f"{d.family}|{p['key']}"] = (split_tag.get(f"{d.family}|{p['key']}", 0.0)
                                                       + p["by_reg"][R] / nc / len(pts_sky))
                per_dev_split[d.tag][f"{cl}|{R}"] = per_dev_split[d.tag].get(f"{cl}|{R}", 0.0) + p["by_reg"][R] / nc
        # check: decomposition reproduces the device RRMS change
        dr = sum(p["r0"] - p["r1"] for p in parts) / nc
        rr = res_sky["rrms_rows"][d.tag]
        assert abs(dr - (rr["start"] - rr["start_g_full"])) < 1e-9, (d.tag, dr, rr)
        assert abs(sum(p["r0"] for p in parts) / nc - rr["start"]) < 1e-9
    total = sum(split[cl][R] for cl in split for R in REGIONS)
    res_sky["rrms_split_abs"] = split
    res_sky["rrms_split_share"] = {cl: {R: split[cl][R] / total for R in REGIONS} for cl in split}
    res_sky["rrms_split_total"] = total
    res_sky["rrms_split_per_device"] = per_dev_split
    res_sky["rrms_split_share_by_tag"] = {k: v / total for k, v in sorted(split_tag.items(), key=lambda kv: -kv[1])}
    res_sky["rrms_split_share_by_family"] = {f: {cl: {R: split_fam[f][cl][R] / total for R in REGIONS}
                                                 for cl in split} for f in split_fam}
    fig_idvg_devices(sky, OUT / "10_idvg_lowvds_all_devices_sky130.png")
    fig_profiles(sky, pts_sky, OUT / "11_residual_profiles_sky130.png", "full-data")
    fig_rrms_split(res_sky["rrms_split_share"], OUT / "03_rrms_reduction_split_sky130.png")
    fig_residual_vs_vov(sky, pts_sky, OUT / "02_residual_vs_vov_sky130.png")
    fig_maps_grid(sky, pts_sky, OUT / "06_device_residual_maps_sky130.png")
    # physics metrics
    ph = {}
    for d in sky["recs"]:
        sims = {"meas": [np.asarray(c.target, float) for c in d.curves],
                "card": [np.asarray(c.sim, float) for c in d.curves],
                "card_g": sky["full"].apply(d)}
        ph[d.tag] = phys_metrics(sky, d, sims)
    res_sky["phys"] = ph
    fig_phys(ph, OUT / "04_device_metrics_sky130.png")
    # M2 (refit with g in loop) cards: residual of the deployed card, by region
    m2 = {}
    for d, P in pts_sky:
        sims_c = m2_card_sims(d)
        sims_cg = [s * np.exp(sky["full"].g_v(d.family == "pmos", c.vg, c.vd)) for s, c in zip(sims_c, d.curves)]
        r_c = score_device(d, sims_c)["rrms"]
        r_cg = score_device(d, sims_cg)["rrms"]
        idx = 0
        rr = []
        for c, s in zip(d.curves, sims_c):
            n = len(c.target)
            im = np.abs(clean_current(c.target))
            ic = np.abs(s)
            ok = (im > sky["floor"]) & (ic > 0) & (np.abs(np.asarray(c.vd, float)) >= VDS_MIN)
            rv = np.full(n, np.nan)
            rv[ok] = np.log(im[ok] / ic[ok])
            rr.append(rv)
            idx += n
        rr = np.concatenate(rr)
        m2[d.tag] = dict(rrms_card=r_c, rrms_card_g=r_cg, r=rr)
    stored = jload(EXP / "I17_1/deploy_sel/refit.json")["rows"]
    res_sky["m2_check"] = dict(
        mean_rrms_card_interp=float(np.mean([m2[t]["rrms_card"] for t in m2])),
        mean_rrms_card_g_interp=float(np.mean([m2[t]["rrms_card_g"] for t in m2])),
        mean_rrms_card_g_stored=float(np.mean([stored[t]["rrms"] for t in m2])),
        max_abs_dev_card_g=float(np.max([abs(m2[t]["rrms_card_g"] - stored[t]["rrms"]) for t in m2])),
        stored_card_only_mean=jload(EXP / "I17_1/decision.json")["refit_in_sample"]["mean_rrms_M2_card_only"])
    m2reg = {}
    for fam in ("all", "nmos", "pmos"):
        for R in REGIONS:
            vals, vals_g = [], []
            for d, P in pts_sky:
                if fam != "all" and d.family != fam:
                    continue
                k = (P["reg"] == R) & np.isfinite(m2[d.tag]["r"])
                if k.sum() < 3:
                    continue
                vals.append(float(np.mean(m2[d.tag]["r"][k])))
                gk = np.log(np.abs(P["a"]["icg"][k]) / np.abs(P["a"]["ic"][k]))
                vals_g.append(float(np.mean(m2[d.tag]["r"][k] - gk)))
            m2reg[f"{fam}|{R}"] = dict(mean_r=summ(vals), mean_r_after_g=summ(vals_g))
    res_sky["m2_region_resid"] = m2reg
    pts_m2 = [(d, {**P, "r": m2[d.tag]["r"]}) for d, P in pts_sky]
    res_sky["onset_m2"] = onset_table(sky, pts_m2)
    res_sky["lin_top_m2"] = lin_top_table(sky, pts_m2)
    fig_profiles(sky, pts_m2, OUT / "13_residual_profiles_sky130_refit_M2.png", "full-data")
    res_sky["onset"] = onset_table(sky, pts_sky)
    res_sky["lin_top"] = lin_top_table(sky, pts_sky)
    fig_idvd_onset(sky, OUT / "12_idvd_onset_pmos_sky130.png")
    results["sky130"] = res_sky
    # ---------------- NIST 4 K ----------------
    nist = dataset_nist4()
    res_n, pts_n = analyse_dataset(nist, use_full=True)
    fig_profiles(nist, pts_n, OUT / "07_residual_profiles_nist4k.png", "full-data")
    ph_n = {}
    for d in nist["recs"]:
        sims = {"meas": [np.asarray(c.target, float) for c in d.curves],
                "card": [np.asarray(c.sim, float) for c in d.curves], "card_g": nist["full"].apply(d)}
        ph_n[d.tag] = phys_metrics(nist, d, sims)
    res_n["phys"] = ph_n
    res_n["onset"] = onset_table(nist, pts_n)
    res_n["lin_top"] = lin_top_table(nist, pts_n)
    results["nist4"] = res_n
    # ---------------- IHP 300 K ----------------
    ihp = dataset_ihp()
    res_i, pts_i = analyse_dataset(ihp, use_full=False)
    fig_profiles(ihp, pts_i, OUT / "08_residual_profiles_ihp300k.png", "LODO, held-out device")
    ph_i = {}
    for d in ihp["recs"]:
        sims = {"meas": [np.asarray(c.target, float) for c in d.curves],
                "card": [np.asarray(c.sim, float) for c in d.curves], "card_g": ihp["lodo"](d.tag).apply(d)}
        ph_i[d.tag] = phys_metrics(ihp, d, sims)
    res_i["phys"] = ph_i
    res_i["onset"] = onset_table(ihp, pts_i)
    res_i["lin_top"] = lin_top_table(ihp, pts_i)
    results["ihp300"] = res_i
    # sky130 with LODO g for the like-for-like comparison of 'removed by g'
    res_sky_l, _ = analyse_dataset(sky, use_full=False)
    results["sky130"]["shared_w_raw_lodo_g"] = res_sky_l["shared_w_raw"]
    res_n_l, _ = analyse_dataset(nist, use_full=False)
    results["nist4"]["shared_w_raw_lodo_g"] = res_n_l["shared_w_raw"]
    fig_datasets({"sky130 77 K": results["sky130"], "NIST 4 K lvt": results["nist4"],
                  "IHP 300 K": results["ihp300"]}, OUT / "05_shared_pattern_by_dataset.png")
    fig_maps_grid(ihp, pts_i, OUT / "09_device_residual_maps_ihp300k.png")
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
