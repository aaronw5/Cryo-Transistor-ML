#!/usr/bin/env python3
"""Held-out simulated truth devices for the pdk2 benchmarks.

``--pdk gf180`` (default; GF180MCU BSIM4, ngspice-41) or ``--pdk ihp`` (IHP
SG13G2 PSP103, ngspice-46).  Tiers (``--sets`` sets x 16 devices each,
fresh seeds, no bank contains them); the fitted card always runs at 27 C
with no extrinsic resistance:

* ``a`` (well-specified): the seven fitted parameters drawn uniformly in the
  device's seven-parameter box (``--draw-set wide``: the wide box); every
  other card value stock; 27 C; noise-free.
* ``b`` (mildly misspecified): a fresh in-box draw, plus
  - every non-fitted ``boxes.PDK_MISSPEC`` parameter (outside both fitted
    sets) scaled by a log-uniform factor in [1/(1+w), 1+w]
    (``--misspec-width`` w; default 0.2 GF180, 0.05 IHP, whose MUEO alone
    would otherwise dominate; both leave a refit floor near the noise);
  - extrinsic drain and source resistors outside the card, each ``r / W``:
    GF180 r ~ U(``--rext-range``) ohm-um (default 25-100); IHP
    r = k x RSW1, k ~ U(0.1, 0.4) (the same fraction of the card's own
    series resistance as GF180's range);
  - multiplicative Gaussian noise ``I (1 + sigma N(0, 1))`` (``--noise``
    sigma, default 0.01); optional global temperature offset
    (``--temp-offset``, default off).
* ``c`` (strongly misspecified; structure the fitted card cannot absorb):
  a fresh in-box draw, plus
  - non-fitted parameters with width ``--c-misspec-width`` (default 0.5);
  - extrinsic resistors r = k x (the card's own per-side series resistance:
    RDSW/2 for GF180, RSW1 for IHP), k ~ U(``--c-rext-k``) (default 1-2);
  - self-heating-like heating: curve i runs at 27 C + dT_max P_i / max P,
    P_i the curve's peak |I Vd| at 27 C, dT_max ~ U(``--c-dtmax``) K
    (default 20-40);
  - bias-dependent noise ``sigma (1 + |Vd| / VDD)`` relative per point
    (``--c-noise`` sigma, default 0.01) and a systematic per-curve gain error
    ``1 + N(0, --c-gain)`` (default 0.01).

Layout (as ``out/research/sim_targets`` + ``out/research/temp/truth``):
``<out>/<tier>/set_<k>.npz`` with ``tags``, ``<tag>__<i>`` (the 11 observed
curves, canonical order, positive magnitudes), ``<tag>__ids`` (observed,
concatenated), ``<tag>__clean_ids`` (noise-free), ``<tag>__local_seven`` /
``<tag>__local_wide`` (the truth's fitted-parameter projection in each box);
``<out>/<tier>/truth.json`` holds the cards, misspecification draws, seeds
and frozen include tags.  Defaults: ``out/research/pdk2/truth`` (GF180),
``out/research/pdk2/ihp_truth`` (IHP).

  PYTHONPATH=src .venv/bin/python scripts/research/pdk2/make_truth.py --tier all
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

from cryoml.pdk2 import boxes, metric  # noqa: E402
from cryoml.pdk2.problem import TRUTH_ROOTS  # noqa: E402

TIER_SEED = {"a": 910001, "b": 920001, "c": 930001}
PDK_SEED = {"gf180": 0, "ihp": 500000}
IHP_B_REXT_K = (0.1, 0.4)
B_WIDTH = {"gf180": 0.2, "ihp": 0.05}


def draw_truth(device, tier: str, seed: int, options: dict) -> dict:
    pdk = boxes.pdk_of_tag(device.tag)
    sim = boxes.backend(pdk)
    seven, wide = boxes.PDK_SETS[pdk]["seven"], boxes.PDK_SETS[pdk]["wide"]
    misspec_names = boxes.PDK_MISSPEC[pdk]
    rng = np.random.default_rng(seed)
    box7 = boxes.box_for("seven", device.tag)
    boxw = boxes.box_for("wide", device.tag)
    draw_box = boxw if options["draw_set"] == "wide" else box7
    stock_wide = boxes.device_box_rows(device, wide, log_scale=True)[0]
    extra: dict = {}
    for attempt in range(1, 51):
        local = np.where(draw_box.active,
                         rng.uniform(-1, 1, draw_box.n_parameters), 0.0)
        fitted = draw_box.local_to_params(local)
        card_params = dict(fitted)
        misspec, rs_ext, temp_C, curve_temps = {}, 0.0, sim.TEMP_C, None
        if tier in ("b", "c"):
            width = options["misspec_width"] if tier == "b" else options["c_misspec_width"]
            stock = sim.stock_params(device, misspec_names)
            half = math.log1p(width)
            for name in misspec_names:
                misspec[name] = float(stock[name] * math.exp(rng.uniform(-half, half)))
            card_params.update(misspec)
        if tier == "b":
            if pdk == "gf180":
                lo, hi = options["rext_range"]
                rs_ext = float(rng.uniform(lo, hi) / device.W_um)
            else:
                rs_ext = float(rng.uniform(*IHP_B_REXT_K)
                               * sim.series_r_per_side(device) / device.W_um)
            if options["temp_offset"] > 0:
                temp_C += float(rng.uniform(-options["temp_offset"],
                                            options["temp_offset"]))
        if tier == "c":
            k = float(rng.uniform(*options["c_rext_k"]))
            rs_ext = k * sim.series_r_per_side(device) / device.W_um
            dt_max = float(rng.uniform(*options["c_dtmax"]))
            cold = sim.simulate(device, card_params, temp_C=temp_C,
                                rs_ext_ohm=rs_ext)
            if not all(np.all(np.isfinite(c)) for c in cold):
                continue
            curves = sim.device_curves(device.dev_type)
            power = np.array([float(np.max(np.abs(i_d * cv.Vd)))
                              for i_d, cv in zip(cold, curves)])
            curve_temps = [temp_C + dt_max * p / power.max() for p in power]
            extra = {"rext_k": k, "dT_max": dt_max,
                     "curve_temps_C": [round(t, 6) for t in curve_temps]}
        clean = sim.simulate(device, card_params, temp_C=temp_C,
                             rs_ext_ohm=rs_ext, curve_temps_C=curve_temps)
        if all(np.all(np.isfinite(c)) for c in clean):
            break
    else:
        raise RuntimeError(f"{device.tag}: no converged truth draw")
    observed = [np.array(c) for c in clean]
    noise_seed, noise_rel = None, 0.0
    if tier == "b" and options["noise"] > 0:
        noise_seed, noise_rel = int(seed + 7), options["noise"]
        noise_rng = np.random.default_rng(noise_seed)
        observed = [c * (1.0 + options["noise"] * noise_rng.standard_normal(c.shape))
                    for c in clean]
    if tier == "c":
        noise_seed, noise_rel = int(seed + 7), options["c_noise"]
        noise_rng = np.random.default_rng(noise_seed)
        gains = 1.0 + options["c_gain"] * noise_rng.standard_normal(len(clean))
        curves = sim.device_curves(device.dev_type)
        observed = [g * c * (1.0 + noise_rel * (1.0 + np.abs(cv.Vd) / sim.VDD)
                             * noise_rng.standard_normal(c.shape))
                    for g, c, cv in zip(gains, clean, curves)]
        extra["curve_gains"] = [round(float(g), 8) for g in gains]
    full_wide = {**stock_wide, **{k: v for k, v in fitted.items()
                                  if k in stock_wide}}
    curves = sim.device_curves(device.dev_type, observed)
    return {
        "clean": clean, "observed": observed,
        "local_seven": box7.params_to_local(
            {**sim.stock_params(device, seven),
             **{k: v for k, v in fitted.items() if k in seven}}),
        "local_wide": boxw.params_to_local(full_wide),
        "meta": {"fitted_params": fitted, "misspec_params": misspec,
                 "rs_ext_ohm_per_side": rs_ext, "temp_C": temp_C,
                 "noise_rel": noise_rel, "noise_seed": noise_seed,
                 "seed": int(seed), "attempts": attempt,
                 "bin_index": sim.bin_index(device),
                 "include_tags": metric.include_tags(curves), **extra},
    }


def tier_options(args) -> dict:
    width = B_WIDTH[args.pdk] if args.misspec_width is None else args.misspec_width
    return {"draw_set": args.draw_set, "misspec_width": width,
            "rext_range": tuple(args.rext_range), "noise": args.noise,
            "temp_offset": args.temp_offset,
            "c_misspec_width": args.c_misspec_width,
            "c_rext_k": tuple(args.c_rext_k), "c_dtmax": tuple(args.c_dtmax),
            "c_noise": args.c_noise, "c_gain": args.c_gain}


def write_tier(tier: str, args, devices) -> dict:
    sim = boxes.backend(args.pdk)
    options = tier_options(args)
    name = tier if args.draw_set == "seven" else f"{tier}_{args.draw_set}"
    out = Path(args.out_root or TRUTH_ROOTS[args.pdk]) / name
    out.mkdir(parents=True, exist_ok=True)
    base = args.seed + TIER_SEED[tier] + PDK_SEED[args.pdk]
    meta = {"tier": tier, "draw_set": args.draw_set,
            "pdk": f"{sim.PDK_NAME}@{sim.PDK_VERSION}",
            "fit_temp_C": sim.TEMP_C, "options": options,
            "base_seed": base, "sets": {}}
    for k in range(args.sets):
        arrays = {"tags": np.asarray([d.tag for d in devices])}
        meta["sets"][str(k)] = {}
        for device in devices:
            seed = base + 1000 * k + sim.TAGS.index(device.tag)
            row = draw_truth(device, tier, seed, options)
            for i, curve in enumerate(row["observed"]):
                arrays[f"{device.tag}__{i}"] = curve
            arrays[f"{device.tag}__ids"] = np.concatenate(row["observed"])
            arrays[f"{device.tag}__clean_ids"] = np.concatenate(row["clean"])
            arrays[f"{device.tag}__local_seven"] = row["local_seven"]
            arrays[f"{device.tag}__local_wide"] = row["local_wide"]
            meta["sets"][str(k)][device.tag] = row["meta"]
        np.savez(out / f"set_{k}.npz", **arrays)
    (out / "truth.json").write_text(json.dumps(meta, indent=1) + "\n")
    return {"tier": name, "path": str(out), "sets": args.sets,
            "devices": len(devices)}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--pdk", choices=("gf180", "ihp"), default="gf180")
    p.add_argument("--tier", choices=("a", "b", "c", "all"), default="all")
    p.add_argument("--sets", type=int, default=3)
    p.add_argument("--devices", default="all")
    p.add_argument("--draw-set", choices=("seven", "wide"), default="seven")
    p.add_argument("--misspec-width", type=float, default=None,
                   help="tier b (default: 0.2 GF180, 0.05 IHP)")
    p.add_argument("--rext-range", type=float, nargs=2, default=(25.0, 100.0),
                   metavar=("LO", "HI"), help="GF180 tier b, ohm-um per side")
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--temp-offset", type=float, default=0.0,
                   help="max |truth temperature offset| in C (tier b)")
    p.add_argument("--c-misspec-width", type=float, default=0.5)
    p.add_argument("--c-rext-k", type=float, nargs=2, default=(1.0, 2.0))
    p.add_argument("--c-dtmax", type=float, nargs=2, default=(20.0, 40.0))
    p.add_argument("--c-noise", type=float, default=0.01)
    p.add_argument("--c-gain", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-root", default=None)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    devices = boxes.backend(args.pdk).parse_devices(args.devices)
    t0 = time.time()
    tiers = ("a", "b", "c") if args.tier == "all" else (args.tier,)
    for tier in tiers:
        print(json.dumps(write_tier(tier, args, devices)), flush=True)
    print(f"runtime {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
