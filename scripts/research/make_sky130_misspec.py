#!/usr/bin/env python3
"""Misspecified SkyWater DEV targets (tier c, ``sky130c``), paired with tier a.

Tier a is ``out/research/sim_targets/set_{0,1}.npz`` (seven fitted values
drawn per device, every other card value stock, noise-free).  For each DEV
set k and paper device the tier-c target keeps tier a's card as the base
card and adds structure that the fitted 43-parameter card cannot represent:

* base card = ``truth.json['sets'][k][tag]['params']`` merged over
  ``spec['published_by_device'][tag]`` (round-1 43-parameter spec); its
  projection ``z_true = box_for(spec, tag).params_to_local(card)`` must
  satisfy max |z_true| <= 1 (asserted).  The 43 fitted values are not
  redrawn.
* (i) non-fitted BSIM4 parameters: every ``MISSPEC`` name whose bin value is
  a nonzero number is multiplied by exp(U(-h, h)), h = s ln 1.5;
* (ii) extrinsic drain and source resistors (``sky130_misspec``), each
  R = kR rdsw_card / (2 W_um mult), kR ~ s U(1, 2);
* (iii) per-curve heating-like drift: an unheated simulation with (i)+(ii)
  gives P_i = max |Id Vd| per curve; dT_i = dT_max P_i / max_j P_j with
  dT_max ~ s U(5, 20) K; curve i runs with vth0_i = vth0 - sign(vth0) kv dT_i
  (kv ~ U(0.5, 1.0) mV/K) and u0_i = u0 (77 / (77 + dT_i))^beta
  (beta ~ U(1, 2)).  The released card has kt1 = ute = 0 and
  tnom = -196.15 C, so ``.temp`` would leave these terms at zero;
* (iv) measurement components (not scaled by s): observed_i =
  g_i clean_i (1 + 0.01 (1 + |Vd| / 1.8) N(0, 1)) pointwise,
  g_i = 1 + 0.01 N(0, 1).

Per-target seed: 940001 + 1000 k + the device index in ``PAPER_DEVICES``.
Each attempt draws, in this order, the (i) factors, kR, dT_max, kv, beta,
the 11 gains and the per-point normals, so attempt j consumes the same
stream positions for every strength (common random numbers across the
calibration grid).  An attempt is rejected (up to 50 attempts, base card
fixed) when (1) any curve of the unheated or heated simulation is
non-finite (all 11 curves, as tier a requires; this includes every
included curve), (2) a heated u0_i lies on the other side of 1 from u0
(BSIM4 divides u0 > 1 by 1e4), or (3) ``ngspice_fit.DeviceProblem`` built
under the target override has a different included-curve set from tier a's
for that device.  The base card without components is also simulated once
and compared with the tier-a target (``tier_a_pairing_max_rel``).

Outputs (``--out``, default ``out/research/sim_targets_sky130c``):
``set_{k}.npz`` (keys ``<tag>__<i>``: observed currents in curve load order;
198 keys for all 18 devices), ``clean_set_{k}.npz`` (noise-free heated
curves ``<tag>__<i>`` and ``<tag>__unheated_<i>``), ``truth.json`` (strength,
seeds, base cards, z_true, every component draw, MISSPEC subset per device,
rejections, git commit, sha1 of the source files, ngspice path and version).

``--strength auto`` reads the calibration decision
(``<out>/calib/calibration.json``, written by ``sky130c_floor.py
--calibrate``) and refuses to run until the pre-registered rule has chosen a
strength.  Set 2 (TEST) is refused.

  PYTHONPATH=src .venv/bin/python scripts/research/make_sky130_misspec.py \
      --sets 0,1 --strength 1 --workers 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
os.environ.pop("CRYOML_TARGET_OVERRIDE", None)

from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.sky130_misspec import MISSPEC, TEMP_K  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

SPEC = ROOT / "out" / "cold_loop" / "full" / "round_1" / "spec.json"
TIER_A = ROOT / "out" / "research" / "sim_targets"
DEFAULT_OUT = ROOT / "out" / "research" / "sim_targets_sky130c"
TAGS = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]

BASE_SEED = 940001
MAX_ATTEMPTS = 50
VDD = 1.8
MISSPEC_HALF_LOG = math.log(1.5)      # h = s ln 1.5        [s]
KR_RANGE = (1.0, 2.0)                 # kR ~ s U(1, 2)       [s]
DTMAX_RANGE_K = (5.0, 20.0)           # dT_max ~ s U(5, 20)  [s]
KV_RANGE_V_PER_K = (0.5e-3, 1.0e-3)   # kv ~ U(0.5, 1.0) mV/K
BETA_RANGE = (1.0, 2.0)               # beta ~ U(1, 2)
NOISE_REL = 0.01                      # (iv), not scaled by s
GAIN_SIGMA = 0.01                     # (iv), not scaled by s

SOURCES = ("scripts/research/make_sky130_misspec.py",
           "src/cryoml/sky130_misspec.py", "src/cryoml/spice_pdk.py",
           "src/cryoml/ngspice_fit.py", "src/cryoml/data_io.py",
           "src/cryoml/parameter_space.py", "scripts/research/cold_loop.py",
           "scripts/research/make_sim_targets.py")


# ------------------------------------------------------------------ records

def _git(*args: str) -> str | None:
    try:
        run = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, timeout=30, check=False)
    except OSError:
        return None
    return run.stdout.rstrip() if run.returncode == 0 else None


def provenance(sources=SOURCES) -> dict:
    from cryoml.spice_pdk import _NGSPICE_BIN
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    try:
        out = subprocess.run([_NGSPICE_BIN, "-v"], capture_output=True,
                             text=True, timeout=30, check=False).stdout
    except OSError:
        out = ""
    version = next((line.strip("* ").strip() for line in out.splitlines()
                    if "ngspice-" in line), out.strip()[:200])
    return {"git": {"commit": _git("rev-parse", "HEAD"),
                    "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                    "dirty_tracked_files": None if dirty is None else
                    [line[3:] for line in dirty.splitlines()]},
            "source_sha1": {rel: hashlib.sha1((ROOT / rel).read_bytes()).hexdigest()
                            for rel in sources if (ROOT / rel).exists()},
            "ngspice": {"binary": _NGSPICE_BIN, "version": version},
            "python": platform.python_version(), "numpy": np.__version__,
            "host": platform.node()}


def sha1_file(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------- worker

def problem_under(tag: str, box, npz_path: str):
    """``ngspice_fit.DeviceProblem`` built under the target override
    ``npz_path`` (no simulation); its layout holds the included curves."""
    from cryoml.data_io import TARGET_OVERRIDE_ENV
    from cryoml.ngspice_fit import DeviceProblem
    previous = os.environ.get(TARGET_OVERRIDE_ENV)
    os.environ[TARGET_OVERRIDE_ENV] = str(npz_path)
    try:
        return DeviceProblem(tag, box)
    finally:
        if previous is None:
            os.environ.pop(TARGET_OVERRIDE_ENV, None)
        else:
            os.environ[TARGET_OVERRIDE_ENV] = previous


def heated_overrides(vth0: float, u0: float, dT: list[float], kv: float,
                     beta: float) -> list[dict]:
    return [{"vth0": vth0 - math.copysign(1.0, vth0) * kv * t,
             "u0": u0 * (TEMP_K / (TEMP_K + t)) ** beta} for t in dT]


def draw_target(job: dict) -> dict:
    """One tier-c target (set k, device); one ngspice process at a time."""
    from cold_loop import box_for
    from cryoml.card_eval import device_by_tag
    from cryoml.cold_start import device_bins
    from cryoml.data_io import TARGET_OVERRIDE_ENV, load_device_curves
    from cryoml.sky130_misspec import bin_values, device_mult, simulate_truth
    t0 = time.time()
    os.environ.pop(TARGET_OVERRIDE_ENV, None)
    tag, s = job["tag"], float(job["strength"])
    device = device_by_tag(tag)
    curves = load_device_curves(device)          # measured bias grid only
    n = len(curves)
    bin_index = device_bins()[tag]
    mult = device_mult(device.dev_type, device.L_um, device.W_um)
    spec = json.loads(Path(job["spec"]).read_text())
    box = box_for(spec, tag)
    card = dict(job["base_card"])
    timeout = float(job["timeout_s"])

    def sim(params, **kwargs):
        return simulate_truth(device.dev_type, device.L_um, device.W_um, curves,
                              params, bin_index=bin_index, timeout_s=timeout,
                              **kwargs)

    problem_a = problem_under(tag, box, job["tier_a_npz"])
    tier_a_layout = [row["tag"] for row in problem_a.layout]
    with np.load(job["tier_a_npz"]) as data:
        tier_a = [np.asarray(data[f"{tag}__{i}"], dtype=np.float64)
                  for i in range(n)]
    # The base card without components against the tier-a target: equal up
    # to card-text rounding, except where the 43-value card sets a value the
    # released bin leaves to BSIM4 (pMOS vfb: absent in the card, computed
    # from vth0; the spec's published value -1.0 is written explicitly).
    base_sims = sim(card)
    pairing = max(float(np.max(np.abs(b - a)) / max(float(np.max(np.abs(a))), 1e-300))
                  for a, b in zip(tier_a, base_sims))
    tier_a_rrms = float(np.mean(problem_a.curve_rrms(base_sims)))

    available = {name: value for name, value in
                 bin_values(device.dev_type, bin_index, MISSPEC).items()
                 if value is not None and value != 0.0}
    vth0, u0, rdsw = float(card["vth0"]), float(card["u0"]), float(card["rdsw"])
    vd = [np.asarray(c.Vd, dtype=np.float64) for c in curves]
    rng = np.random.default_rng(int(job["seed"]))
    rejections = []
    with tempfile.TemporaryDirectory(prefix="sky130c_gen_") as td:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            h = s * MISSPEC_HALF_LOG
            factors = {name: float(math.exp(rng.uniform(-h, h)))
                       for name in available}
            rext_k = s * float(rng.uniform(*KR_RANGE))
            dt_max = s * float(rng.uniform(*DTMAX_RANGE_K))
            kv = float(rng.uniform(*KV_RANGE_V_PER_K))
            beta = float(rng.uniform(*BETA_RANGE))
            gains = 1.0 + GAIN_SIGMA * rng.standard_normal(n)
            normals = [rng.standard_normal(len(c.Id)) for c in curves]
            misspec = {name: available[name] * factors[name] for name in available}
            truth_card = {**card, **misspec}
            rs_ext = rext_k * rdsw / (2.0 * device.W_um * mult)
            draw = {"attempt": attempt, "misspec_factors": factors,
                    "misspec_values": misspec, "rext_k": rext_k,
                    "rs_ext_ohm_per_side": rs_ext, "dT_max_K": dt_max,
                    "kv_V_per_K": kv, "beta": beta,
                    "gains": [float(g) for g in gains]}
            unheated = sim(truth_card, rs_ext_ohm=rs_ext)
            bad = [i for i, c in enumerate(unheated) if not np.all(np.isfinite(c))]
            if bad:
                rejections.append({"attempt": attempt, "reason": "unheated_nonfinite",
                                   "curves": bad})
                continue
            power = [float(np.max(np.abs(i_d * v))) for i_d, v in zip(unheated, vd)]
            dT = [dt_max * p / max(power) for p in power]
            curve_params = heated_overrides(vth0, u0, dT, kv, beta)
            if any((cp["u0"] > 1.0) != (u0 > 1.0) for cp in curve_params):
                rejections.append({"attempt": attempt, "reason": "u0_unit_switch"})
                continue
            clean = sim(truth_card, rs_ext_ohm=rs_ext, curve_params=curve_params)
            bad = [i for i, c in enumerate(clean) if not np.all(np.isfinite(c))]
            if bad:
                rejections.append({"attempt": attempt, "reason": "heated_nonfinite",
                                   "curves": bad})
                continue
            observed = [g * c * (1.0 + NOISE_REL * (1.0 + np.abs(v) / VDD) * z)
                        for g, c, v, z in zip(gains, clean, vd, normals)]
            path = Path(td) / f"attempt_{attempt}.npz"
            np.savez(path, **{f"{tag}__{i}": o for i, o in enumerate(observed)})
            layout = [row["tag"] for row in problem_under(tag, box, str(path)).layout]
            if layout != tier_a_layout:
                rejections.append({"attempt": attempt, "reason": "inclusion_changed",
                                   "tier_a": tier_a_layout, "tier_c": layout})
                continue
            break
        else:
            raise RuntimeError(f"set {job['set']} {tag}: no accepted draw in "
                               f"{MAX_ATTEMPTS} attempts: {rejections[-3:]}")
    meta = {"seed": int(job["seed"]), "device_index": TAGS.index(tag),
            "bin_index": bin_index, "device_mult": mult, "attempts": attempt,
            "rejections": rejections, "misspec_available": available,
            "misspec_used": sorted(available, key=MISSPEC.index),
            **draw, "rdsw_card": rdsw, "vth0_card": vth0, "u0_card": u0,
            "power_W": power, "dT_K": dT, "curve_params": curve_params,
            "noise_rel": NOISE_REL, "gain_sigma": GAIN_SIGMA,
            "include_tags": layout, "tier_a_pairing_max_rel": pairing,
            "tier_a_rrms_at_z_true": tier_a_rrms,
            "runtime_s": round(time.time() - t0, 2)}
    return {"set": int(job["set"]), "tag": tag, "observed": observed,
            "clean": clean, "unheated": unheated, "meta": meta}


# --------------------------------------------------------------------- main

def resolve_strength(text: str, calibration: Path) -> tuple[float, dict]:
    if text != "auto":
        return float(text), {"source": "command line"}
    if not calibration.exists():
        raise SystemExit(f"--strength auto: {calibration} does not exist")
    decision = json.loads(calibration.read_text())
    if not decision.get("decided"):
        raise SystemExit(f"--strength auto: calibration not decided "
                         f"({decision.get('status')}: {decision.get('message')})")
    return float(decision["chosen_strength"]), {
        "source": str(calibration.relative_to(ROOT) if calibration.is_relative_to(ROOT)
                      else calibration),
        "sha1": sha1_file(calibration), "status": decision.get("status")}


def build_jobs(sets: list[int], tags: list[str], strength: float,
               timeout_s: float) -> tuple[list[dict], dict]:
    from cold_loop import box_for
    spec = json.loads(SPEC.read_text())
    overlap = set(MISSPEC) & set(spec["selected_parameters"])
    assert not overlap, f"MISSPEC overlaps the fitted set: {sorted(overlap)}"
    truth_a = json.loads((TIER_A / "truth.json").read_text())
    jobs, bases = [], {}
    for k in sets:
        for tag in tags:
            card = {**spec["published_by_device"][tag],
                    **truth_a["sets"][str(k)][tag]["params"]}
            z_true = box_for(spec, tag).params_to_local(card)
            assert float(np.max(np.abs(z_true))) <= 1.0 + 1e-9, (k, tag)
            bases[f"set_{k}/{tag}"] = {"card": card, "z_true": z_true.tolist()}
            jobs.append({"set": k, "tag": tag, "strength": strength,
                         "base_card": card, "spec": str(SPEC),
                         "tier_a_npz": str(TIER_A / f"set_{k}.npz"),
                         "seed": BASE_SEED + 1000 * k + TAGS.index(tag),
                         "timeout_s": timeout_s})
    return jobs, bases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", default="0,1")
    parser.add_argument("--strength", default="1",
                        help="s (number) or 'auto' (the calibration decision)")
    parser.add_argument("--calibration", default=None,
                        help="calibration.json for --strength auto "
                             "(default <out>/calib/calibration.json)")
    parser.add_argument("--devices", default="all")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    sets = sorted({int(x) for x in args.sets.split(",") if x.strip()})
    if any(k not in (0, 1) for k in sets):
        parser.error(f"--sets {args.sets}: only DEV sets 0 and 1 (set_2 is TEST)")
    tags = TAGS if args.devices == "all" else [t for t in TAGS
                                               if t in args.devices.split(",")]
    unknown = set(args.devices.split(",")) - set(TAGS) if args.devices != "all" else set()
    if unknown or not tags:
        parser.error(f"unknown or no devices: {sorted(unknown)}")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    calibration = Path(args.calibration) if args.calibration else out / "calib" / "calibration.json"
    if not calibration.is_absolute():
        calibration = ROOT / calibration
    strength, strength_source = resolve_strength(args.strength, calibration)
    if not strength > 0:
        parser.error("--strength must be positive")
    out.mkdir(parents=True, exist_ok=True)
    truth_path = out / "truth.json"
    previous = json.loads(truth_path.read_text()) if truth_path.exists() else None
    if previous and previous.get("strength") != strength and not (
            set(previous.get("sets", {})) <= {str(k) for k in sets}):
        raise SystemExit(f"{truth_path} holds strength {previous.get('strength')} "
                         f"for sets {sorted(previous.get('sets', {}))}; refusing to "
                         f"mix strength {strength}")
    t0 = time.time()
    jobs, bases = build_jobs(sets, tags, strength, args.timeout_s)
    if args.workers <= 1:
        results = [draw_target(job) for job in jobs]
    else:
        with get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            results = pool.map(draw_target, jobs)
    truth = previous if (previous and previous.get("strength") == strength) else {}
    truth.update({
        "tier": "sky130c", "strength": strength, "strength_source": strength_source,
        "spec": str(SPEC.relative_to(ROOT)),
        "tier_a": str(TIER_A.relative_to(ROOT)),
        "base_card_rule": "tier-a truth.json params merged over spec "
                          "published_by_device (43 values; not redrawn)",
        "seed_rule": f"{BASE_SEED} + 1000*k + PAPER_DEVICES index",
        "constants": {"misspec": list(MISSPEC), "misspec_half_log": "s*ln(1.5)",
                      "rext_k": "s*U(1,2)", "rs_ext_ohm_per_side":
                      "rext_k*rdsw_card/(2*W_um*device_mult)",
                      "dT_max_K": "s*U(5,20)", "kv_V_per_K": list(KV_RANGE_V_PER_K),
                      "beta": list(BETA_RANGE), "T0_K": TEMP_K, "vdd": VDD,
                      "noise_rel": NOISE_REL, "gain_sigma": GAIN_SIGMA,
                      "max_attempts": MAX_ATTEMPTS},
        "provenance": provenance(),
        "command": [sys.executable, *(argv if argv is not None else sys.argv)]})
    truth.setdefault("sets", {})
    for k in sets:
        chunk = [r for r in results if r["set"] == k]
        observed, clean = {}, {}
        for row in sorted(chunk, key=lambda r: TAGS.index(r["tag"])):
            for i, curve in enumerate(row["observed"]):
                observed[f"{row['tag']}__{i}"] = np.asarray(curve, dtype=np.float64)
            for i, curve in enumerate(row["clean"]):
                clean[f"{row['tag']}__{i}"] = np.asarray(curve, dtype=np.float64)
            for i, curve in enumerate(row["unheated"]):
                clean[f"{row['tag']}__unheated_{i}"] = np.asarray(curve, dtype=np.float64)
        if len(tags) == len(TAGS):
            assert len(observed) == 198, len(observed)
        np.savez(out / f"set_{k}.npz", **observed)
        np.savez(out / f"clean_set_{k}.npz", **clean)
        truth["sets"][str(k)] = {
            row["tag"]: {**bases[f"set_{k}/{row['tag']}"], **row["meta"]}
            for row in sorted(chunk, key=lambda r: TAGS.index(r["tag"]))}
        truth.setdefault("npz_sha1", {})[f"set_{k}"] = sha1_file(out / f"set_{k}.npz")
    truth["runtime_s"] = round(time.time() - t0, 1)
    truth_path.write_text(json.dumps(truth, indent=1) + "\n")
    summary = {k: {t: {"attempts": v["attempts"], "rs_ext": round(v["rs_ext_ohm_per_side"], 3),
                       "dT_max": round(v["dT_max_K"], 2),
                       "pair": f"{v['tier_a_pairing_max_rel']:.1e}",
                       "tier_a_rrms_z_true": f"{v['tier_a_rrms_at_z_true']:.1e}"}
                   for t, v in truth["sets"][str(k)].items()} for k in sets}
    print(json.dumps({"strength": strength, "out": str(out), "sets": summary},
                     indent=1))
    print(f"runtime {truth['runtime_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
