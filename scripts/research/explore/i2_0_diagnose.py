#!/usr/bin/env python3
"""I2-0: diagnosis of the devices that determined the exploration-iteration-1 means.

A. Real devices pmos_L0p35_W5 and pmos_L0p35_W0p55 (REAL_DEV).
B. Simulated E3 tier-M test device nmos_L1_W3__2.
C. Iteration-1 decision quantities with and without those devices (diagnostic
   only; stored verdicts and result files are not changed).

Stages (run in order; ``sim`` runs about 40 single-device NGSpice simulations):

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i2_0_diagnose.py tables
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i2_0_diagnose.py sim
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i2_0_diagnose.py decisions
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i2_0_diagnose.py plots

Reads stored result files only (E1-E5 under out/research/explore, the
published-baseline inclusion set, out/research/polished/direct43_r3_fd and
direct7_fd).  Writes JSON tables and PNGs to out/research/explore/I2_0/.
REAL_TEST devices are not loaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.config import OUT_DIR  # noqa: E402

EXP = OUT_DIR / "research" / "explore"
OUT = EXP / "I2_0"
SPLIT = json.loads((EXP / "real_split.json").read_text())
REAL_DEV = SPLIT["REAL_DEV"]
PDEV = ("pmos_L0p35_W5", "pmos_L0p35_W0p55")
BEST43 = OUT_DIR / "research" / "polished" / "direct43_r3_fd" / "real" / "summary.json"
DIRECT7 = OUT_DIR / "research" / "polished" / "direct7_fd" / "real" / "summary.json"
BASELINE = OUT_DIR / "pdk_baseline" / "pdk_baseline.json"
E2_EVAL = EXP / "E2" / "eval"
E3_RUN = EXP / "E3" / "runs" / "full" / "tier_M"
E3_DEV = "nmos_L1_W3__2"
TAGS11 = ([f"idvd@{b:g}" for b in (0.37, 0.74, 1.11, 1.48, 1.85)]
          + [f"idvg@{b:g}" for b in (0.01, 0.37, 0.74, 1.11, 1.48, 1.85)])


def save(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1, default=_default) + "\n")
    print("wrote", p.relative_to(ROOT))
    return p


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(type(o))


def load(p: Path):
    return json.loads(Path(p).read_text())


def fnum(v):
    return float(v) if v is not None and np.isfinite(v) else None


# =========================================================================
# Stage "tables": stored records only
# =========================================================================
def measured_table(tag: str) -> dict:
    """Per metric curve: frozen inclusion, trim, denominator, quantization."""
    from cryoml.card_eval import device_by_tag, include_tags
    from cryoml.data_io import load_device_curves
    from cryoml.metrics import curve_trim, tag_metric_curves
    dev = device_by_tag(tag)
    curves = load_device_curves(dev)
    inc = include_tags(tag)
    base = load(BASELINE)["per_curve"][tag]
    rows = {}
    for (kind, b), c in tag_metric_curves(curves).items():
        key = f"{kind}@{b:g}"
        start, cleaned, den = curve_trim(dev.dev_type, dev.L_um, dev.W_um, kind, c.Id)
        seg = np.abs(cleaned[start:])
        uniq = np.unique(seg)
        steps = np.diff(uniq)
        step = float(np.min(steps[steps > 0])) if np.any(steps > 0) else None
        zero_pred = (math.sqrt(float(np.mean(seg ** 2))) / den) if den > 0 else None
        sweep = np.abs(np.asarray(c.Vd if kind == "idvd" else c.Vg, dtype=float))
        rows[key] = {
            "included_frozen": key in inc, "n_points": len(c.Id), "start": int(start),
            "n_scored": int(len(seg)),
            "scored_sweep_V": [float(sweep[start]), float(sweep[-1])] if len(seg) else None,
            "den_A": den, "max_abs_meas_A": float(np.max(seg)) if len(seg) else 0.0,
            "n_distinct_values": int(len(uniq)), "min_step_A": step,
            "step_over_den": (step / den) if (step and den > 0) else None,
            "rrms_if_sim_is_zero": zero_pred,
            "published_card_rrms": base.get(key, {}).get("rrms")}
    dens = [r["den_A"] for r in rows.values() if r["included_frozen"]]
    dmax = max(dens)
    for r in rows.values():
        r["den_over_device_max_den"] = r["den_A"] / dmax if r["included_frozen"] else None
    return {"tag": tag, "dev_type": dev.dev_type, "L_um": dev.L_um, "W_um": dev.W_um,
            "curves": rows, "max_included_den_A": dmax, "min_included_den_A": min(dens),
            "den_dynamic_range": dmax / min(dens)}


def e2_breakdown() -> dict:
    out = {}
    for stage in ("raw", "fd"):
        for arm in ("N0", "M", "S", "nn"):
            for seed in (0, 1, 2):
                p = E2_EVAL / f"{stage}_{arm}_s{seed}.json"
                if not p.exists():
                    continue
                real = load(p)["real"]
                for tag in PDEV:
                    r = real[tag]
                    pc = {k: fnum(v) for k, v in r["per_curve"].items()}
                    fin = {k: v for k, v in pc.items() if v is not None}
                    top = max(fin, key=fin.get) if fin else None
                    out[f"{stage}/{arm}/s{seed}/{tag}"] = {
                        "status": r["status"], "penalized_rrms": r["penalized_rrms"],
                        "per_curve": pc, "top_curve": top,
                        "top_curve_share_of_sum": (fin[top] / sum(fin.values())) if top else None,
                        "params": r.get("params"), "local": r.get("local")}
    return out


def e4_breakdown() -> dict:
    out = {}
    for run in ("p43", "p7"):
        for split in ("vd", "sparse", "vg"):
            s = load(EXP / "E4" / run / split / "summary_strict.json")
            for tag in PDEV:
                rec = load(EXP / "E4" / run / split / f"{tag}.json")
                row = {"test_tags": rec["test_tags"], "train_tags": rec["train_tags"]}
                for m in ("single", "ens", "laplace", "crossinterp", "learned"):
                    d = s["methods"][m]["per_device"].get(tag)
                    if d:
                        row[m] = {"heldout_rrms": d["heldout_rrms"],
                                  "per_curve": dict(zip(rec["test_tags"], d["heldout_curve_rrms"]))}
                row["learned_inner_mse"] = s["learned_hyper"].get(tag, {}).get("inner_mse")
                out[f"{run}/{split}/{tag}"] = row
            out[f"{run}/{split}/learned_inner_mse_all"] = {
                t: v.get("inner_mse") for t, v in s["learned_hyper"].items()}
    return out


def e4_learned_scale() -> dict:
    """E4 learned-discrepancy targets y = (meas - single) / sc with sc = mean
    |single prediction| over the curve: per device and test curve, sc/den and
    max |y| (the quantity whose squared mean is the inner LODO MSE)."""
    from cryoml.explore.e4_eval import load_device, point_table, scale_of, curve_of
    out = {}
    for run in ("p43", "p7"):
        for split in ("vd", "sparse"):
            d_run = EXP / "E4" / run / split
            opts = load(d_run / "options.json")
            per = {}
            for tag in REAL_DEV:
                dev = load_device(d_run, tag, opts["near_tol"], "strict")
                tab = point_table(dev)
                rows = {}
                for t, st, n in tab["meta"]:
                    row = dev["rows"][t]
                    pred = curve_of(dev["members"][dev["single"]], row)
                    sc = float(scale_of(pred))
                    y = (row["meas"] - pred)[st:] / sc
                    rows[t] = {"sc_A": sc, "den_A": row["den"], "sc_over_den": sc / row["den"],
                               "max_abs_y": float(np.max(np.abs(y))),
                               "mean_y2": float(np.mean(y ** 2))}
                per[tag] = {"curves": rows,
                            "device_mean_y2": float(np.mean(tab["y"] ** 2)),
                            "max_abs_y": float(np.max(np.abs(tab["y"])))}
            out[f"{run}/{split}"] = per
    return out


def e4_laplace_draws() -> dict:
    """Per-draw held-out RRMS of the Laplace draws (NGSpice curves stored in the
    E4 npz) on the held-out curves of the two devices, and the RRMS of the
    draw mean (the 'laplace' point predictor)."""
    from cryoml.explore.e4_eval import load_device, curve_of
    out = {}
    for run, split in (("p43", "vd"), ("p43", "sparse"), ("p7", "vd"), ("p7", "sparse")):
        d_run = EXP / "E4" / run / split
        opts = load(d_run / "options.json")
        for tag in PDEV:
            dev = load_device(d_run, tag, opts["near_tol"], "strict")
            rows = {}
            for t in dev["rec"]["test_tags"]:
                row = dev["rows"][t]
                st, y, den = row["start"], row["meas"][row["start"]:], row["den"]
                dr = curve_of(dev["draws"], row)
                ok = np.all(np.isfinite(dr), axis=1)
                dr = dr[ok]
                per = np.sqrt(np.mean((dr[:, st:] - y) ** 2, axis=1)) / den
                mean_rrms = math.sqrt(float(np.mean((dr.mean(0)[st:] - y) ** 2))) / den
                single = curve_of(dev["members"][dev["single"]], row)
                rows[t] = {"n_draws_finite": int(ok.sum()), "n_draws": int(len(ok)),
                           "draw_rrms_median": float(np.median(per)),
                           "draw_rrms_q90": float(np.quantile(per, 0.9)),
                           "draw_rrms_max": float(np.max(per)),
                           "draw_mean_rrms": mean_rrms,
                           "single_rrms": math.sqrt(float(np.mean((single[st:] - y) ** 2))) / den,
                           "frac_draws_rrms_gt_10": float(np.mean(per > 10))}
            ident = dev["rec"]["identifiability"]
            out[f"{run}/{split}/{tag}"] = {
                "curves": rows, "condition": ident.get("condition"),
                "eff_rank_1e-2": ident.get("eff_rank_1e-2"),
                "n_active": len(dev["rec"]["active_names"]),
                "laplace_n_sd_gt_0p1": ident.get("laplace_n_sd_gt_0p1")}
    return out


def e5_table() -> dict:
    s = load(EXP / "E5" / "stage1.json")
    out = {}
    for base in ("b43", "b7"):
        pdv = s["bases"][base]["per_device"]
        out[base] = {t: {"A0": pdv["A0_control"][t], "A2": pdv["A2_primary"][t],
                         "A2_minus_A0": pdv["A2_primary"][t] - pdv["A0_control"][t]}
                     for t in REAL_DEV}
    return out


def e2_box_position() -> dict:
    """Where the worst E2 single-card predictions sit in the 7-parameter box."""
    from cryoml.explore import e2_direct as E
    spec = E.load_spec()
    out = {}
    for tag in PDEV:
        box = E.box_for(spec, tag)
        pub = spec["published_by_device"][tag]
        rows = {}
        for arm in ("N0", "M", "S"):
            for seed in (0, 1, 2):
                r = load(E2_EVAL / f"raw_{arm}_s{seed}.json")["real"][tag]
                loc = np.asarray(r["local"], dtype=float)
                rows[f"{arm}/s{seed}"] = {
                    "penalized_rrms": r["penalized_rrms"],
                    "local": dict(zip(box.names, loc.tolist())),
                    "n_at_edge_abs_local_ge_0.98": int(np.sum(np.abs(loc) >= 0.98)),
                    "params": r["params"]}
        out[tag] = {"names": list(box.names), "published": pub,
                    "bounds": spec["bounds_by_device"][tag], "predictions": rows}
    return out


def add_scale(rows: dict, meas: dict) -> dict:
    """Absolute RMSE (A) of each curve, RRMS x den, and the same divided by
    the largest included denominator of the device (diagnostic scale only;
    the paper RRMS is unchanged)."""
    for key, rec in rows.items():
        tag = key.split("/")[-1]
        if tag not in meas or not isinstance(rec, dict):
            continue
        curves = meas[tag]["curves"]
        dmax = meas[tag]["max_included_den_A"]
        targets = [rec] + [v for v in rec.values() if isinstance(v, dict) and "per_curve" in v]
        for tgt in targets:
            pc = tgt.get("per_curve") or {}
            tgt["rmse_A"] = {k: (v * curves[k]["den_A"] if v is not None else None)
                             for k, v in pc.items()}
            tgt["rmse_over_device_max_den"] = {
                k: (v * curves[k]["den_A"] / dmax if v is not None else None)
                for k, v in pc.items()}
    return rows


def stage_tables() -> None:
    meas = {t: measured_table(t) for t in REAL_DEV}
    rank = sorted(REAL_DEV, key=lambda t: -meas[t]["den_dynamic_range"])
    save("A_measured_curves.json", {
        "note": ("Per metric curve of every REAL_DEV device: frozen published-baseline "
                 "inclusion, start-trim index (metrics.curve_trim), scored points, "
                 "denominator den = trimmed mean |I_meas|, SMU quantization step, "
                 "the RRMS a card would get by predicting 0 A, and the published-card "
                 "RRMS. den_dynamic_range = max/min den over included curves."),
        "devices_by_den_dynamic_range": [(t, meas[t]["den_dynamic_range"],
                                          meas[t]["min_included_den_A"]) for t in rank],
        "devices": meas})
    save("A_e2_per_curve.json", add_scale(e2_breakdown(), meas))
    save("A_e2_box_position.json", e2_box_position())
    save("A_e4_per_curve.json", add_scale(e4_breakdown(), meas))
    save("A_e4_learned_target_scale.json", e4_learned_scale())
    save("A_e4_laplace_draws.json", e4_laplace_draws())
    save("A_e5_per_device.json", e5_table())


# =========================================================================
# Stage "sim": single-device NGSpice checks
# =========================================================================
def per_curve_full(tag, curves, sims):
    from cryoml.card_eval import device_by_tag, include_tags
    from cryoml.metrics import score_device_new
    dev = device_by_tag(tag)
    r = score_device_new(dev.dev_type, dev.L_um, dev.W_um, curves, sims,
                         include_tags=include_tags(tag))
    return {"rrms": fnum(r["rrms"]),
            "per_curve": {k: fnum(v["rrms"]) for k, v in r["per_curve"].items()
                          if v.get("included")}}


def stage_sim() -> None:
    from cryoml.card_eval import device_by_tag
    from cryoml.cold_start import device_bins
    from cryoml.data_io import load_device_curves
    from cryoml.spice_pdk import simulate_pdk
    from cryoml.explore import shared_discrepancy as sd
    best = load(BEST43)["devices"]
    e5_models = load(EXP / "E5" / "models" / "b43_A2_folds.json")
    res = {"A": {}, "B": {}}
    curves_dump = {}
    for tag in PDEV:
        dev = device_by_tag(tag)
        curves = load_device_curves(dev)
        bins = device_bins()[tag]

        def sim(params):
            return [np.asarray(s, dtype=float) for s in
                    simulate_pdk(dev.dev_type, dev.L_um, dev.W_um, curves,
                                 params=params or None, bin_index=bins)]
        cards = {"published": None, "best43_direct43_r3_fd": best[tag]["params"]}
        # Worst stored single-card prediction among the iteration-1 runs with a
        # full card: E2 raw (one forward pass), the highest penalized RRMS.
        worst = None
        for arm in ("N0", "M", "S"):
            for seed in (0, 1, 2):
                r = load(E2_EVAL / f"raw_{arm}_s{seed}.json")["real"][tag]
                if r["status"] == "valid" and (worst is None or r["penalized_rrms"] > worst[1]):
                    worst = (f"E2 raw {arm} seed {seed}", r["penalized_rrms"], r["params"])
        cards["worst_e2_raw"] = worst[2]
        sims = {k: sim(v) for k, v in cards.items()}
        rows = {k: per_curve_full(tag, curves, s) for k, s in sims.items()}
        rows["worst_e2_raw"]["label"] = worst[0]
        rows["worst_e2_raw"]["stored_penalized_rrms"] = worst[1]
        # VTH0 sensitivity around the best 43-parameter card.
        vsens = {}
        v0 = best[tag]["params"]["vth0"]
        for dv in (-0.04, -0.02, -0.01, 0.01, 0.02, 0.04):
            p = dict(best[tag]["params"])
            # |vth0| larger by dv (pMOS vth0 < 0): positive dv = higher threshold magnitude.
            p["vth0"] = v0 + math.copysign(1.0, v0) * dv
            vsens[f"{dv * 1000:+.0f}mV"] = per_curve_full(tag, curves, sim(p))
        # E5 A2 (fold model) applied post hoc to the best-card simulation.
        model = sd.DeltaModel.from_json(e5_models[tag])
        corr = sd.corrected_sims(dev, curves, sims["best43_direct43_r3_fd"], model)
        rows["e5_A2_on_best43"] = per_curve_full(tag, curves, corr)
        res["A"][tag] = {"cards": rows, "vth0_shift_on_best43": vsens,
                         "vth0_best43": v0,
                         "e2_worst_params": worst[2]}
        curves_dump[tag] = {k: [s.tolist() for s in v] for k, v in sims.items()}
    # ---------------------------------------------------------------- B
    from cryoml.explore import e3_temp as E
    from cryoml.parameter_sets import PARAMS7
    rec = load(E3_RUN / "test" / f"{E3_DEV}.json")
    truth = rec["truth"]
    tag = truth["tag"]
    z = np.load(E3_RUN / "test" / f"{E3_DEV}.npz")
    temps = [float(t) for t in z["temps"]]
    ids = z["ids"]
    stock, lo, hi = E.stock_and_box(tag)
    xlo, xhi = E.x_bounds(tag)
    iu = PARAMS7.index("u0")
    preds = load(E3_RUN / "preds.json")
    T = 110.0
    target = ids[temps.index(T)]
    scorer = E.TempScorer(tag, T, target)

    def score_x(x):
        s = E.simulate_cards(tag, [np.asarray(x, dtype=float)], T)[0]
        cr = scorer.problem.curve_rrms(s)
        return s, {"rrms": float(np.mean(cr)) if scorer.strict_ok(s) else None,
                   "per_curve": {r["tag"]: float(v) for r, v in zip(scorer.layout, cr)}}
    b = {"T_K": T, "x_u0_switch": float(math.log(1.0 / stock[iu])),
         "stock": dict(zip(PARAMS7, stock.tolist())),
         "box_1p5_lo": dict(zip(PARAMS7, lo.tolist())), "box_1p5_hi": dict(zip(PARAMS7, hi.tolist())),
         "fit_x_lo": dict(zip(PARAMS7, xlo.tolist())), "fit_x_hi": dict(zip(PARAMS7, xhi.tolist())),
         "layout": [{"tag": r["tag"], "start": r["start"], "den_A": r["denominator"]}
                    for r in scorer.layout]}
    b["truth_x_by_T"] = {f"{t:g}": E.truth_x(truth, t).tolist() for t in E.TEMPS_DENSE}
    b["truth_u0_by_T"] = {f"{t:g}": float(stock[iu] * math.exp(E.truth_x(truth, t)[iu]))
                         for t in E.TEMPS_DENSE}
    b["truth_max_abs_current_by_T"] = {f"{t:g}": float(np.max(np.abs(r)))
                                       for t, r in zip(temps, ids)}
    # Truth re-simulated (with the extrinsic resistors) vs stored target.
    re_truth = E.simulate_truth_at(tag, truth, T)
    b["truth_resim_max_rel_diff"] = float(np.max(np.abs(re_truth - target))
                                          / np.max(np.abs(target)))
    sims_b = {}
    cards_b = {}
    for obs, meth in (("O4", "lrn"), ("O4", "par3"), ("O4", "pt_interp"),
                      ("O2", "lrn"), ("O2", "par3_pop")):
        cards_b[f"{obs}|{meth}"] = preds[obs][meth][E3_DEV][f"{T:g}"]
    cards_b["floor"] = rec["fits"]["floor"][f"{T:g}"]["x"]
    cards_b["truth_x_no_series_R"] = E.truth_x(truth, T).tolist()
    for k, x in cards_b.items():
        s, sc = score_x(x)
        sims_b[k] = s
        b.setdefault("cards", {})[k] = {"x": list(map(float, x)), "x_u0": float(x[iu]),
                                       "u0": float(stock[iu] * math.exp(x[iu])),
                                       "above_switch": bool(stock[iu] * math.exp(x[iu]) > 1.0),
                                       **sc,
                                       "params_at_fit_bound": [PARAMS7[i] for i in range(7)
                                                               if abs(x[i] - xlo[i]) < 1e-6
                                                               or abs(x[i] - xhi[i]) < 1e-6]}
    # u0 sweep across the switch: truth x at 110 K (no series R), only x_u0 varied.
    sweep = {}
    xt = E.truth_x(truth, T)
    for xu in (3.50, 3.60, 3.64, 3.65, 3.66, 3.67, 3.70, 3.75):
        x = xt.copy()
        x[iu] = xu
        s, sc = score_x(x)
        sweep[f"{xu:.2f}"] = {"u0": float(stock[iu] * math.exp(xu)), "rrms_vs_truth": sc["rrms"],
                              "max_abs_current_A": float(max(np.max(np.abs(c)) for c in s))}
    b["u0_sweep_truth_x_110K"] = sweep
    # Counterfactual: the O4 lrn card with only u0 moved to the truth's x_u0.
    x = np.asarray(cards_b["O4|lrn"], dtype=float).copy()
    x[iu] = xt[iu]
    s, sc = score_x(x)
    sims_b["O4|lrn with truth x_u0"] = s
    b["O4_lrn_with_truth_x_u0"] = {"x_u0": float(xt[iu]), **sc}
    # Training population: E3 tier-M truths with u0 above 1 at any dense temperature.
    pop = {}
    for split in ("train", "test"):
        for p in sorted((E3_RUN / split).glob("*.json")):
            tr = load(p)["truth"]
            st, _, _ = E.stock_and_box(tr["tag"])
            u = {f"{t:g}": float(st[iu] * math.exp(E.truth_x(tr, t)[iu])) for t in E.TEMPS_DENSE}
            if max(u.values()) > 1.0:
                fits = load(p).get("fits", {})
                pop[f"{split}/{p.stem}"] = {
                    "temps_above_switch": [t for t, v in u.items() if v > 1.0], "u0_by_T": u}
    b["population_truths_above_switch"] = pop
    b["n_population"] = {s: len(list((E3_RUN / s).glob("*.json"))) for s in ("train", "test")}
    res["B"] = b
    save("sim_checks.json", res)
    np.savez_compressed(OUT / "sim_curves_scratch.npz",
                        **{f"A__{t}__{k}": np.concatenate(v) for t, d in curves_dump.items()
                           for k, v in d.items()},
                        **{f"B__{k}": np.concatenate(v) for k, v in sims_b.items()},
                        B__truth=target)


# =========================================================================
# Stage "decisions": iteration-1 decision quantities with / without devices
# =========================================================================
def boot_ci(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    r = 1 - a[idx].mean(1) / b[idx].mean(1)
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def dec_e2(exclude: set) -> dict:
    t = load(EXP / "E2" / "table.json")["rows"]
    keep = [d for d in REAL_DEV if d not in exclude]
    d7 = load(DIRECT7)["devices"]
    D_all = 0.173493
    D = float(np.mean([d7[d]["rrms"] for d in keep]))
    n0 = t["raw/N0/real"]["per_device"]
    s_ = t["raw/S/real"]["per_device"]
    m_ = t["raw/M/real"]["per_device"]
    r0 = float(np.mean([n0[d] for d in keep]))
    rs = float(np.mean([s_[d] for d in keep]))
    rm = float(np.mean([m_[d] for d in keep]))
    wins = int(sum(s_[d] < n0[d] for d in keep))
    n = len(keep)
    # C3 / K2 thresholds scaled to the device count (8/12 and 6/12).
    c3_need, k2_need = math.ceil(8 * n / 12), math.ceil(6 * n / 12)

    def tier(arm, k):
        pdv = t[f"raw/{arm}/a{k}"]["per_device"]
        return float(np.mean([v for d, v in pdv.items()
                              if not any(d == e for e in exclude)]))
    ta0 = np.mean([tier("N0", k) for k in (0, 1)])
    taS = np.mean([tier("S", k) for k in (0, 1)])
    checks = {"C1": rs <= 0.70 * r0, "C2_with_D_recomputed": (rs - D) <= 0.70 * (r0 - D),
              "C2_with_stored_D": (rs - D_all) <= 0.70 * (r0 - D_all),
              f"C3_wins_ge_{c3_need}": wins >= c3_need,
              "C4_guard_tier_a_same_exclusion": bool(taS <= 1.5 * ta0 + 0.01),
              "K1": rs > 0.85 * r0, f"K2_wins_lt_{k2_need}": wins < k2_need}
    kill = checks["K1"] or checks[f"K2_wins_lt_{k2_need}"]
    cont = (checks["C1"] and checks["C2_with_D_recomputed"]
            and checks[f"C3_wins_ge_{c3_need}"] and checks["C4_guard_tier_a_same_exclusion"])
    return {"n_devices": n, "R_raw_N0": r0, "R_raw_S": rs, "R_raw_M": rm, "D": D,
            "ratio_S_over_N0": rs / r0, "gap_ratio": (rs - D) / (r0 - D), "S_wins": wins,
            "tier_a_N0": float(ta0), "tier_a_S": float(taS), "checks": checks,
            "attribution_S_over_M": rs / rm,
            "rule_output": "KILL" if kill else ("CONTINUE" if cont else "INCONCLUSIVE"),
            "medians": {"N0": float(np.median([n0[d] for d in keep])),
                        "S": float(np.median([s_[d] for d in keep])),
                        "M": float(np.median([m_[d] for d in keep]))}}


def dec_e3(exclude: set) -> dict:
    s = load(E3_RUN / "summary.json")
    out = {}
    per = {}
    # eval/<device>.json holds every method's per-temperature RRMS for one device.
    for p in sorted((E3_RUN / "eval").glob("*.json")):
        name = p.stem
        if name.startswith(("O2__", "O4__")):
            continue
        per[name] = load(p)
    devs = [d for d in sorted(per) if d not in exclude]
    held = ("110", "190", "260")

    def dev_mean(d, key):
        v = per[d].get(key)
        if v is None:
            return math.nan
        vals = [v.get(h) for h in held]
        if any(x is None or not np.isfinite(x) for x in vals):
            return math.nan
        return float(np.mean(vals))
    for S, ctrl in (("O4", "par3"), ("O2", "par3_pop")):
        lrn = np.array([dev_mean(d, f"{S}|lrn") for d in devs])
        c = np.array([dev_mean(d, f"{S}|{ctrl}") for d in devs])
        pti = np.array([dev_mean(d, f"{S}|pt_interp") for d in devs])
        ok = np.isfinite(lrn) & np.isfinite(c) & np.isfinite(pti)
        R = 1 - lrn[ok].mean() / c[ok].mean()
        ci = boot_ci(lrn[ok], c[ok])
        red_pt = 1 - lrn[ok].mean() / pti[ok].mean()
        out[S] = {"control": ctrl, "n_devices": int(ok.sum()), "lrn": float(lrn[ok].mean()),
                  "control_mean": float(c[ok].mean()), "pt_interp": float(pti[ok].mean()),
                  "R_S": float(R), "R_S_CI95": ci, "lrn_vs_pt_interp_reduction": float(red_pt),
                  "medians": {"lrn": float(np.median(lrn[ok])), "control": float(np.median(c[ok])),
                              "pt_interp": float(np.median(pti[ok]))},
                  "lrn_lower_than_control_on": int(np.sum(lrn[ok] < c[ok]))}
    cont = [S for S, v in out.items() if v["R_S"] >= 0.25 and v["R_S_CI95"][0] > 0
            and v["lrn_vs_pt_interp_reduction"] >= 0.10]
    kill = (all(v["R_S"] < 0.10 for v in out.values())
            or all(v["lrn_vs_pt_interp_reduction"] < 0 for v in out.values()))
    out["rule_output_without_failed_cell_check"] = (
        "CONTINUE" if cont else ("KILL" if kill else "INCONCLUSIVE"))
    out["stored_decision_reference"] = s.get("verdict_rule_output")
    return out


def _import_e4_analyze():
    spec = importlib.util.spec_from_file_location(
        "e4_analyze", ROOT / "scripts" / "research" / "explore" / "e4_analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dec_e4(run: str, split: str, exclude: set) -> dict:
    """Re-run the E4 analysis in memory on REAL_DEV minus ``exclude`` (the
    learned/conformal leave-one-device-out is recomputed on the reduced set)
    and apply the pre-registered decide(); nothing is written to E4/."""
    from cryoml.explore.e4_eval import (device_metrics, device_predictions, family_means,
                                        learned_and_conformal, load_device)
    ea = _import_e4_analyze()
    d_run = EXP / "E4" / run / split
    opts = load(d_run / "options.json")
    tags = [t for t in REAL_DEV if t not in exclude]
    devs = [load_device(d_run, t, opts["near_tol"], "strict") for t in tags]
    base = {d["tag"]: device_predictions(d) for d in devs}
    learned = learned_and_conformal(devs, base)
    per_device = {m: {} for m in ea.METHODS}
    for d in devs:
        for m in ea.METHODS:
            src = base[d["tag"]].get(m) if m in base[d["tag"]] else learned[m].get(d["tag"])
            if src is not None:
                per_device[m][d["tag"]] = device_metrics(d, src)
    summary = {"n_devices": len(devs), "methods": {}}
    single = {t: r["heldout_rrms"] for t, r in per_device["single"].items()}
    for m in ea.METHODS:
        rows = per_device[m]
        if not rows:
            continue
        rr = {t: r["heldout_rrms"] for t, r in rows.items()}
        entry = {"heldout_rrms": family_means(rr), "per_device": rows}
        if m != "single":
            common = [t for t in rr if np.isfinite(rr[t]) and np.isfinite(single.get(t, np.nan))]
            entry["wins_vs_single"] = int(sum(rr[t] < single[t] for t in common))
        if any("coverage_0.9" in r for r in rows.values()):
            for lev in (0.9,):
                entry[f"coverage_{lev:g}"] = family_means(
                    {t: r.get(f"coverage_{lev:g}", math.nan) for t, r in rows.items()})
                entry[f"width_{lev:g}"] = family_means(
                    {t: r.get(f"width_{lev:g}", math.nan) for t, r in rows.items()})
        summary["methods"][m] = entry
    dec = ea.decide(summary)
    means = {m: summary["methods"][m]["heldout_rrms"]["all_device_mean"]
             for m in ("single", "ens", "laplace", "crossinterp", "learned")}
    inner = {t: v.get("inner_mse") for t, v in learned["learned_hyper"].items()}
    return {"n_devices": len(devs), "heldout_means": means, "decision": dec,
            "learned_inner_mse_max": float(max(inner.values())) if inner else None,
            "learned_inner_mse_median": float(np.median(list(inner.values()))) if inner else None}


def dec_e5(exclude: set) -> dict:
    s = load(EXP / "E5" / "stage1.json")
    out = {}
    for base in ("b43", "b7"):
        pdv = s["bases"][base]["per_device"]
        keep = [d for d in REAL_DEV if d not in exclude]
        a0 = np.array([pdv["A0_control"][d] for d in keep])
        a2 = np.array([pdv["A2_primary"][d] for d in keep])
        out[base] = {"n": len(keep), "A0": float(a0.mean()), "A2": float(a2.mean()),
                     "reduction": float((a0.mean() - a2.mean()) / a0.mean()),
                     "A2_lt_A0_on": int(np.sum(a2 < a0))}
    return out


def dec_e1(exclude: set) -> dict:
    out = {}
    s1 = load(EXP / "E1" / "real_dev_lodo" / "results.json")["per_device"]
    s2 = load(EXP / "E1" / "real_dev_refit" / "results.json")["rows"]
    for name, rows in (("stage1", {a: s1[a] for a in ("C0", "T1", "M")}),
                       ("stage2", {a: {t: v["rrms"] for t, v in s2[a].items()}
                                   for a in ("C0", "T1", "M")})):
        keep = [d for d in REAL_DEV if d not in exclude]
        n = len(keep)
        m = np.array([rows["M"][d] for d in keep])
        t1 = np.array([rows["T1"][d] for d in keep])
        c0 = np.array([rows["C0"][d] for d in keep])
        need = math.ceil(8 * n / 12)
        out[name] = {"n": n, "M": float(m.mean()), "T1": float(t1.mean()), "C0": float(c0.mean()),
                     "M_le_0.090": bool(m.mean() <= 0.090), "M_lt_T1": bool(m.mean() < t1.mean()),
                     "M_beats_T1": int(np.sum(m < t1)), "M_beats_C0": int(np.sum(m < c0)),
                     "wins_needed": need,
                     "pass": bool(m.mean() <= 0.090 and m.mean() < t1.mean()
                                  and np.sum(m < t1) >= need and np.sum(m < c0) >= need)}
    return out


def stage_decisions() -> None:
    sets = {"all": set(), "without_pmos_L0p35_W5": {"pmos_L0p35_W5"},
            "without_pmos_L0p35_W0p55": {"pmos_L0p35_W0p55"},
            "without_both": set(PDEV)}
    res = {"label": ("DIAGNOSTIC ONLY. Decision quantities recomputed with devices removed "
                     "after the fact. The stored pre-registered verdicts are unchanged and "
                     "these values are not decision inputs."),
           "E1": {k: dec_e1(v) for k, v in sets.items()},
           "E2": {k: dec_e2(v) for k, v in sets.items()},
           "E4_p43_vd": {k: dec_e4("p43", "vd", v) for k, v in sets.items()},
           "E5": {k: dec_e5(v) for k, v in sets.items()},
           "E3": {"all": dec_e3(set()), "without_nmos_L1_W3__2": dec_e3({E3_DEV}),
                  "without_all_nmos_L1_W3_test": dec_e3({f"nmos_L1_W3__{i}" for i in range(3)})}}
    res["notes"] = {
        "E2": ("Real devices are evaluation-only in E2 (models trained on the simulated bank), "
               "so removing a device from the mean is exact. C3/K2 thresholds scaled to n "
               "(ceil(8n/12), ceil(6n/12)). D recomputed as the direct7_fd mean over the kept "
               "devices; C2 with the stored D = 0.173493 also shown. C4 tier-a means drop the "
               "simulated targets of the same geometries."),
        "E4": ("Learned and conformal leave-one-device-out recomputed on the reduced device set; "
               "win threshold round(0.6667 n) as in decide()."),
        "E5": ("Aggregate-only: the A2 fold models were trained with the removed devices in "
               "the other folds; retraining was not run."),
        "E1": ("Aggregate-only: fold models were trained with the removed devices in the other "
               "folds; retraining was not run. Win thresholds scaled to ceil(8n/12)."),
        "E3": ("Per-device mean over 110/190/260 K, paired bootstrap 4000 resamples, seed 0 "
               "(the stored run used its own seed, so CI bounds differ slightly). The "
               "failed-cell check is not recomputed (0 failed cells for lrn/par3/pt_interp in "
               "the stored run).")}
    save("C_decision_sensitivity.json", res)


# =========================================================================
# Stage "plots"
# =========================================================================
def stage_plots() -> None:
    sys.path.insert(0, str(ROOT / "scripts" / "research"))
    import report_figs as rf  # noqa: F401  (rcParams)
    import matplotlib.pyplot as plt
    from cryoml.card_eval import device_by_tag
    from cryoml.data_io import load_device_curves
    from cryoml.metrics import curve_trim
    BG, INK, ACCENT, GREY = "#fcfcfb", "#0b0b0b", "#2a78d6", "#9a9892"
    ORANGE, RED = rf.ORANGE, rf.RED
    DPI = 150
    simc = load(OUT / "sim_checks.json")
    arr = np.load(OUT / "sim_curves_scratch.npz")
    pngs = []
    for tag in PDEV:
        dev = device_by_tag(tag)
        curves = load_device_curves(dev)
        bounds, s0 = [], 0
        for c in curves:
            bounds.append((s0, s0 + len(c.Id)))
            s0 += len(c.Id)
        cards = simc["A"][tag]["cards"]
        series = [("published", "NGSpice, published card", GREY, "--"),
                  ("best43_direct43_r3_fd", "NGSpice, best stored 43-parameter card (direct43_r3_fd)",
                   ACCENT, "-"),
                  ("worst_e2_raw", f"NGSpice, worst stored prediction ({cards['worst_e2_raw']['label']})",
                   ORANGE, "-")]
        fig, axes = plt.subplots(3, 4, figsize=(15, 10.2), dpi=DPI)
        fig.patch.set_facecolor(BG)
        order = TAGS11
        keymap = {f"{c.kind}@{abs(float(c.fixed)):g}": i for i, c in enumerate(curves)}
        for ax, key in zip(axes.flat, order):
            ax.set_facecolor(BG)
            i = keymap[key]
            c = curves[i]
            kind = c.kind
            sweep = np.abs(np.asarray(c.Vd if kind == "idvd" else c.Vg, dtype=float))
            start, cleaned, den = curve_trim(dev.dev_type, dev.L_um, dev.W_um, kind, c.Id)
            meas = np.abs(np.asarray(c.Id, dtype=float))
            lo_, hi_ = bounds[i]
            ax.scatter(sweep[::3], np.maximum(meas[::3], 1e-12), s=7, color=INK, zorder=3,
                       label="measured (every 3rd point)")
            for k, lab, col, ls in series:
                y = np.abs(arr[f"A__{tag}__{k}"][lo_:hi_])
                ax.plot(sweep, np.maximum(y, 1e-12), color=col, ls=ls, lw=1.4, label=lab)
            if start > 0:
                ax.axvspan(sweep[0], sweep[start], color="#ecebe6", zorder=0,
                           label="trimmed (not scored)")
            ax.axhline(den, color=RED, lw=0.8, ls=":", label="denominator (trimmed mean |I_meas|)")
            ax.set_yscale("log")
            ax.set_ylim(1e-11, 3e-3)
            inc = key in {k2 for k2 in cards["published"]["per_curve"]}
            r = [cards[k]["per_curve"].get(key) for k, _, _, _ in series]
            txt = ("not in frozen inclusion set" if not inc else
                   "RRMS pub {:.3g} | best43 {:.3g} | worst {:.3g}".format(*[v if v is not None else float('nan') for v in r]))
            ax.set_title(f"{'Id-Vd' if kind == 'idvd' else 'Id-Vg'} at |{'Vg' if kind == 'idvd' else 'Vd'}| = {key.split('@')[1]} V\n{txt}",
                         fontsize=9)
            ax.set_xlabel(f"|{'Vd' if kind == 'idvd' else 'Vg'}| (V)", fontsize=9)
            ax.set_ylabel("|Id| (A)", fontsize=9)
            ax.tick_params(labelsize=8)
        ax = axes.flat[-1]
        ax.axis("off")
        h, l = axes.flat[1].get_legend_handles_labels()
        ax.legend(h, l, loc="center", fontsize=8.5, frameon=False)
        fig.suptitle(f"{tag}: measured I-V and NGSpice re-simulations, log scale "
                     f"(device RRMS: published {cards['published']['rrms']:.3f}, "
                     f"best43 {cards['best43_direct43_r3_fd']['rrms']:.3f}, "
                     f"worst {cards['worst_e2_raw']['rrms']:.1f})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = OUT / f"I2_0_A_{tag}_iv.png"
        fig.savefig(p, dpi=DPI)
        plt.close(fig)
        pngs.append(p)

    # Denominator vs device max for every REAL_DEV included curve.
    meas = load(OUT / "A_measured_curves.json")["devices"]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    tags = sorted(REAL_DEV, key=rf.device_key)
    for j, t in enumerate(tags):
        for key, r in meas[t]["curves"].items():
            if not r["included_frozen"]:
                continue
            v = r["den_over_device_max_den"]
            col = ORANGE if t in PDEV and v < 1e-2 else (ACCENT if t in PDEV else GREY)
            ax.scatter(j, v, s=22, color=col, zorder=3)
            if v < 3e-3:
                ax.annotate(key, (j, v), xytext=(4, 0), textcoords="offset points", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("curve denominator / largest\ndenominator of the device")
    ax.set_title("Frozen-included metric curves, REAL_DEV: per-curve RRMS denominator relative to "
                 "the device's largest", fontsize=11)
    fig.tight_layout()
    p = OUT / "I2_0_A_denominators.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    pngs.append(p)

    # VTH0 sensitivity.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    shifts = [-40, -20, -10, 0, 10, 20, 40]
    for ax, tag in zip(axes, PDEV):
        ax.set_facecolor(BG)
        vs = simc["A"][tag]["vth0_shift_on_best43"]
        base = simc["A"][tag]["cards"]["best43_direct43_r3_fd"]["per_curve"]
        keys = list(base)
        low = [k for k in keys if meas[tag]["curves"][k]["den_over_device_max_den"] < 1e-2]
        for k in keys:
            ys = [vs[f"{s:+d}mV"]["per_curve"][k] if s else base[k] for s in shifts]
            ax.plot(shifts, ys, marker="o", ms=3, lw=1.3 if k in low else 0.8,
                    color=ORANGE if k in low else GREY, label=k if k in low else None)
        dm = [vs[f"{s:+d}mV"]["rrms"] if s else simc["A"][tag]["cards"]["best43_direct43_r3_fd"]["rrms"]
              for s in shifts]
        ax.plot(shifts, dm, color=ACCENT, lw=2.2, marker="o", ms=4, label="device mean")
        ax.set_yscale("log")
        ax.set_xlabel("change of |VTH0| from the best 43-parameter card (mV)")
        ax.set_ylabel("curve RRMS (NGSpice)")
        ax.set_title(f"{tag}: orange = curves with den < 1% of device max; grey = other curves",
                     fontsize=9.5)
        ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    p = OUT / "I2_0_A_vth0_sensitivity.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    pngs.append(p)

    # B: E3 nmos_L1_W3__2.
    from cryoml.explore import e3_temp as E
    b = simc["B"]
    tag = "nmos_L1_W3"
    dev = device_by_tag(tag)
    curves = load_device_curves(dev)
    bounds, s0 = [], 0
    for c in curves:
        bounds.append((s0, s0 + len(c.Id)))
        s0 += len(c.Id)
    fig = plt.figure(figsize=(15, 8.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(BG)
    Ts = [float(t) for t in b["truth_u0_by_T"]]
    ax.plot(Ts, list(b["truth_u0_by_T"].values()), color=INK, marker="o", ms=3, label="truth u0 (card value)")
    ax.axhline(1.0, color=RED, ls=":", lw=1, label="u0 = 1 (BSIM4: u0 > 1 is divided by 1e4)")
    for k, col in (("O4|lrn", ACCENT), ("O4|par3", ORANGE), ("floor", GREY)):
        ax.scatter([110], [b["cards"][k]["u0"]], color=col, zorder=4, s=28, label=f"{k} card at 110 K")
    for t in (110, 190, 260):
        ax.axvline(t, color="#ecebe6", lw=6, zorder=0)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("u0 (m^2/Vs as written in the card)")
    ax.set_title("Truth u0 vs temperature (held-out temperatures shaded)", fontsize=10)
    ax.legend(fontsize=7.5)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(BG)
    tm = b["truth_max_abs_current_by_T"]
    ax.plot([float(t) for t in tm], list(tm.values()), color=INK, marker="o", ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("max |Id| of the truth curves (A)")
    ax.set_title("Truth current scale vs temperature (stored targets)", fontsize=10)
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor(BG)
    sw = b["u0_sweep_truth_x_110K"]
    xs = [float(k) for k in sw]
    ax.plot(xs, [v["rrms_vs_truth"] for v in sw.values()], color=ACCENT, marker="o", ms=4)
    ax.axvline(b["x_u0_switch"], color=RED, ls=":", lw=1, label="x_u0 where u0 = 1")
    ax.set_yscale("log")
    ax.set_xlabel("x_u0 = log(u0 / u0_stock)\n(other six parameters at the 110 K truth)")
    ax.set_ylabel("RRMS vs 110 K truth (NGSpice)")
    ax.set_title("RRMS against the 110 K truth when only u0 is varied", fontsize=10)
    ax.legend(fontsize=8)
    arr_t = arr["B__truth"]
    keymap = {f"{c.kind}@{abs(float(c.fixed)):g}": i for i, c in enumerate(curves)}
    for col_i, key in enumerate(("idvg@0.37", "idvd@1.85", "idvg@1.85")):
        ax = fig.add_subplot(gs[1, col_i])
        ax.set_facecolor(BG)
        i = keymap[key]
        c = curves[i]
        lo_, hi_ = bounds[i]
        sweep = np.abs(np.asarray(c.Vd if c.kind == "idvd" else c.Vg, dtype=float))
        ax.scatter(sweep[::4], np.maximum(np.abs(arr_t[lo_:hi_][::4]), 1e-13), s=8, color=INK,
                   zorder=3, label="110 K truth (simulated target, every 4th point)")
        for k, col, ls in (("O4|lrn", ACCENT, "-"), ("O4|par3", ORANGE, "-"),
                           ("floor", GREY, "--"), ("O4|lrn with truth x_u0", rf.AQUA, ":")):
            y = np.abs(arr[f"B__{k}"][lo_:hi_])
            r = (b["cards"][k]["per_curve"].get(key) if k in b["cards"]
                 else b["O4_lrn_with_truth_x_u0"]["per_curve"].get(key))
            ax.plot(sweep, np.maximum(y, 1e-13), color=col, ls=ls, lw=1.5,
                    label=f"NGSpice, {k} card (curve RRMS {r:.3g})")
        ax.set_yscale("log")
        ax.set_xlabel(f"|{'Vd' if c.kind == 'idvd' else 'Vg'}| (V)")
        ax.set_ylabel("|Id| (A)")
        ax.set_title(f"110 K, {key}", fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"E3 tier M test device {E3_DEV}: truth crosses the BSIM4 u0 unit switch "
                 "between 110 K and 130 K", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = OUT / "I2_0_B_nmos_L1_W3__2.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    pngs.append(p)
    for p in pngs:
        print("wrote", p.relative_to(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("tables", "sim", "decisions", "plots"))
    a = ap.parse_args()
    {"tables": stage_tables, "sim": stage_sim, "decisions": stage_decisions,
     "plots": stage_plots}[a.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
