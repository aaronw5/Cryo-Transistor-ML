#!/usr/bin/env python3
"""Refit floors of the sky130c DEV targets, the strength calibration, and the
final floor table.

Floors (default mode).  Per (set, device) target in ``--targets`` (written by
``make_sky130_misspec.py``), a worker sets ``CRYOML_TARGET_OVERRIDE`` to the
set npz, builds ``ngspice_fit.DeviceProblem(tag, box)`` on the round-1
43-parameter box and runs ``polish(problem, z_true, max_nfev=M,
diff_step=1e-3, irls_rounds=3)`` from the base card's projection z_true
(1e-3 because 1e-2 probes cross the u0 = 1 cliff; hybrid_tr docstring).
Recorded per target:

* ``start``: paper RRMS (frozen inclusion) at z_true, failed curves at 10;
* ``polish``: the best frozen-inclusion paper RRMS seen by polish (failed
  curves at 10, ``ngspice_fit.PENALTY``);
* ``per_curve`` / ``final_rrms`` / ``official_dynamic_rrms``: a fresh
  ``ngspice_fit.final_score`` of the best card; ``strict``: its
  ``cryoml.strict_score`` status;
* ``noise_only``: the clean heated curves scored against the observed
  target (no simulation);
* ``n_sim``, ``runtime_s``.

Results go to ``<targets>/floors.json`` (rewritten after every target;
finished targets with the same settings are skipped on a rerun).  Before
``--finalize``, ``floor`` = ``polish`` and ``reference`` = null.

``--calibrate`` additionally applies the pre-registered rule (DEV set_0, the
six devices below, M = 1500) over every ``<targets>/../s*/floors.json`` and
writes ``<targets>/../calibration.json``: choose the strength whose 6-target
mean polish floor is closest to 0.09 among those with mean in [0.06, 0.12]
(the measured GF180/IHP tier-c floor range); if none, evaluate 0.25 (all
means above 0.12) or 4 (all means below 0.06) and apply the rule again; if
still none (or the means straddle the interval), use the strength whose mean
is closest to 0.09 and record that the range was not reached.
``--calibrate-next`` runs that one extra strength (generation + floors) when
the decision asks for it, else does nothing.

``--finalize --reference DIR`` adds the best fresh RRMS of the B20000
reference run (``DIR/set_<k>/ml_<tag>.json``, method ``sketch_tr``; a record
with a failed included curve counts as no reference), sets ``floor =
min(polish, reference)`` and the per-set all-device mean and combined (mean
of the nMOS / pMOS family means).  ``finalized`` is true when every target
has a reference record (used or left out for a failed curve).  The reference run's identity is checked
against the bank oracle recomputed from these targets (``dev_compare``).

  PYTHONPATH=src .venv/bin/python scripts/research/sky130c_floor.py \
      --targets out/research/sim_targets_sky130c --sets 0,1 --max-nfev 3000 --workers 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.pop("CRYOML_TARGET_OVERRIDE", None)

from make_sky130_misspec import SPEC, TAGS, provenance  # noqa: E402

CALIB_DEVICES = ("nmos_L0p15_W1p6", "nmos_L1_W3", "nmos_L100_W100",
                 "pmos_L0p35_W1p6", "pmos_L2_W5", "pmos_L8_W0p84")
CALIB_GRID = (0.5, 1.0, 2.0)
CALIB_EXTRA = {"low": 0.25, "high": 4.0}
CALIB_LO, CALIB_HI, CALIB_AIM = 0.06, 0.12, 0.09
CALIB_NFEV = 1500
DIFF_STEP = 1e-3
IRLS_ROUNDS = 3
REFERENCE_METHOD = "sketch_tr"
SOURCES = ("scripts/research/sky130c_floor.py",
           "scripts/research/make_sky130_misspec.py",
           "src/cryoml/sky130_misspec.py", "src/cryoml/ngspice_fit.py",
           "src/cryoml/strict_score.py", "src/cryoml/card_eval.py",
           "src/cryoml/metrics.py", "src/cryoml/spice_pdk.py")


def rel(path: Path) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def family_means(values: dict[str, float]) -> dict:
    """All-device mean and combined (mean of the nMOS / pMOS family means);
    a non-finite device makes every mean it enters non-finite."""
    v = {t: float(x) if x is not None else math.nan for t, x in values.items()}
    nmos = [x for t, x in v.items() if t.split("/")[-1].startswith("nmos")]
    pmos = [x for t, x in v.items() if t.split("/")[-1].startswith("pmos")]
    out = {"n": len(v), "all_device_mean": float(np.mean(list(v.values()))) if v else math.nan,
           "nmos": float(np.mean(nmos)) if nmos else math.nan,
           "pmos": float(np.mean(pmos)) if pmos else math.nan}
    out["combined"] = (out["nmos"] + out["pmos"]) / 2 if nmos and pmos else math.nan
    return out


# ------------------------------------------------------------------- worker

def floor_target(job: dict) -> dict:
    from cold_loop import box_for
    from cryoml import strict_score as ss
    from cryoml.data_io import TARGET_OVERRIDE_ENV
    from cryoml.ngspice_fit import DeviceProblem, final_score, polish
    t0 = time.time()
    os.environ[TARGET_OVERRIDE_ENV] = job["npz"]
    tag = job["tag"]
    spec = json.loads(Path(job["spec"]).read_text())
    problem = DeviceProblem(tag, box_for(spec, tag))
    z_true = np.asarray(job["z_true"], dtype=np.float64)
    start_curves = problem.curve_rrms(problem.simulate(z_true))
    result = polish(problem, z_true, max_nfev=int(job["max_nfev"]),
                    diff_step=DIFF_STEP, irls_rounds=IRLS_ROUNDS)
    final = final_score(problem, result["local"])
    strict = ss.record_status({"per_curve": final["per_curve"]}, tag)
    with np.load(job["clean_npz"]) as data:
        clean = [np.asarray(data[f"{tag}__{i}"], dtype=np.float64)
                 for i in range(len(problem.curves))]
    noise_curves = problem.curve_rrms(clean)
    return {"key": job["key"], "row": {
        "start": float(np.mean(start_curves)),
        "polish": float(result["objective"]),
        "reference": None, "floor": float(result["objective"]),
        "noise_only": float(np.mean(noise_curves)),
        "final_rrms": float(final["rrms"]),
        "official_dynamic_rrms": float(final["official_dynamic_rrms"]),
        "per_curve": final["per_curve"],
        "start_per_curve": {row["tag"]: float(v) for row, v in
                            zip(problem.layout, start_curves)},
        "noise_per_curve": {row["tag"]: float(v) for row, v in
                            zip(problem.layout, noise_curves)},
        "strict": {k: strict[k] for k in ("status", "n_included", "n_failed",
                                          "failed_tags")},
        "polish_history": result["history"],
        "local": [float(x) for x in result["local"]],
        # problem.n_sim (start + polish) plus the one fresh final_score run.
        "n_sim": int(problem.n_sim) + 1, "runtime_s": round(time.time() - t0, 1)}}


# --------------------------------------------------------------- floor runs

def settings_record(args, sets, devices, truth) -> dict:
    return {"max_nfev": args.max_nfev, "diff_step": DIFF_STEP,
            "irls_rounds": IRLS_ROUNDS, "start": "z_true (base card projection)",
            "polish": "cryoml.ngspice_fit.polish", "sets": sets,
            "devices": devices, "strength": truth.get("strength"),
            "truth_npz_sha1": truth.get("npz_sha1", {}),
            "spec": rel(SPEC)}


def write_floors(path: Path, report: dict) -> None:
    targets = report["targets"]
    report["per_set"] = {}
    for set_name in sorted({k.split("/")[0] for k in targets}):
        rows = {k: v for k, v in targets.items() if k.startswith(set_name + "/")}
        report["per_set"][set_name] = {
            field: family_means({k: r.get(field) for k, r in rows.items()})
            for field in ("start", "polish", "reference", "floor", "noise_only")
            if all(r.get(field) is not None for r in rows.values())}
    report["pooled"] = {
        field: family_means({k: r.get(field) for k, r in targets.items()})
        for field in ("start", "polish", "reference", "floor", "noise_only")
        if targets and all(r.get(field) is not None for r in targets.values())}
    partial = path.with_suffix(f".{os.getpid()}.tmp")
    partial.write_text(json.dumps(report, indent=1) + "\n")
    partial.replace(path)


def run_floors(args, targets_dir: Path, sets: list[int],
               devices: list[str] | None) -> dict:
    truth = json.loads((targets_dir / "truth.json").read_text())
    path = targets_dir / "floors.json"
    jobs, chosen = [], []
    for k in sets:
        rows = truth["sets"].get(str(k))
        if rows is None:
            raise SystemExit(f"{targets_dir}: set {k} not generated")
        for tag in TAGS:
            if tag not in rows or (devices and tag not in devices):
                continue
            chosen.append(tag)
            jobs.append({"key": f"set_{k}/{tag}", "tag": tag, "spec": str(SPEC),
                         "npz": str(targets_dir / f"set_{k}.npz"),
                         "clean_npz": str(targets_dir / f"clean_set_{k}.npz"),
                         "z_true": rows[tag]["z_true"], "max_nfev": args.max_nfev})
    if devices and set(devices) - set(chosen):
        raise SystemExit(f"devices not in {targets_dir}: {sorted(set(devices) - set(chosen))}")
    settings = settings_record(args, sets, sorted(set(chosen), key=TAGS.index), truth)
    report = json.loads(path.read_text()) if path.exists() else {}
    same = {k: v for k, v in report.get("settings", {}).items()
            if k not in ("sets", "devices")} == {
        k: v for k, v in settings.items() if k not in ("sets", "devices")}
    done = report.get("targets", {}) if same else {}
    report = {"kind": "sky130c_floors", "finalized": False, "settings": settings,
              "provenance": provenance(SOURCES),
              "command": [sys.executable, *sys.argv], "targets": dict(done)}
    todo = [job for job in jobs if job["key"] not in done]
    # Longest-running (pMOS) devices first keeps the pool busy.
    todo.sort(key=lambda job: not job["tag"].startswith("pmos"))
    t0 = time.time()
    if todo and args.workers <= 1:
        results = map(floor_target, todo)
    elif todo:
        pool = get_context("spawn").Pool(min(args.workers, len(todo)))
        results = pool.imap_unordered(floor_target, todo)
    else:
        results = iter(())
    for out in results:
        report["targets"][out["key"]] = out["row"]
        row = out["row"]
        print(json.dumps({"target": out["key"], **{k: round(row[k], 5) for k in
                          ("start", "polish", "noise_only", "final_rrms")},
                          "strict": row["strict"]["status"], "n_sim": row["n_sim"],
                          "runtime_s": row["runtime_s"]}), flush=True)
        report["targets"] = dict(sorted(report["targets"].items()))
        write_floors(path, report)
    if todo and args.workers > 1:
        pool.close()
        pool.join()
    report["runtime_s"] = round(time.time() - t0, 1)
    write_floors(path, report)
    print(json.dumps(report.get("per_set", {}), indent=1))
    return report


# -------------------------------------------------------------- calibration

def calibration_points(calib_root: Path) -> tuple[dict, list]:
    """Mean polish floor per strength over the pre-registered calibration
    runs under ``calib_root``; runs with other devices, sets or M are listed
    and left out."""
    points, ignored = {}, []
    for floors in sorted(calib_root.glob("s*/floors.json")):
        report = json.loads(floors.read_text())
        st = report.get("settings", {})
        keys = set(report.get("targets", {}))
        want = {f"set_0/{t}" for t in CALIB_DEVICES}
        s = st.get("strength")
        reason = None
        if s is None or float(s) not in CALIB_GRID + tuple(CALIB_EXTRA.values()):
            reason = f"strength {s} is not pre-registered"
        elif st.get("max_nfev") != CALIB_NFEV:
            reason = f"max_nfev {st.get('max_nfev')} != {CALIB_NFEV}"
        elif keys != want:
            reason = f"targets {sorted(keys)} != the six set_0 devices"
        if reason:
            ignored.append({"path": rel(floors), "reason": reason})
            continue
        values = [report["targets"][k]["polish"] for k in sorted(want)]
        points[float(s)] = {"mean_polish_floor": float(np.mean(values)),
                            "per_device": {k.split("/")[1]: report["targets"][k]["polish"]
                                           for k in sorted(want)},
                            "mean_start": float(np.mean([report["targets"][k]["start"]
                                                         for k in want])),
                            "mean_noise_only": float(np.mean(
                                [report["targets"][k]["noise_only"] for k in want])),
                            "path": rel(floors)}
    return points, ignored


def decide(means: dict[float, float]) -> dict:
    """The pre-registered calibration rule on {strength: mean polish floor}."""
    def distance(s):
        m = means[s]
        return (abs(m - CALIB_AIM) if math.isfinite(m) else math.inf, s)
    missing = [s for s in CALIB_GRID if s not in means]
    if missing:
        return {"decided": False, "status": "incomplete", "missing": missing,
                "message": f"strengths {missing} not evaluated yet"}
    inside = [s for s, m in means.items() if CALIB_LO <= m <= CALIB_HI]
    if inside:
        best = min(inside, key=distance)
        return {"decided": True, "status": "in_range", "chosen_strength": best,
                "message": f"s = {best:g}: mean polish floor {means[best]:.4f} "
                           f"in [{CALIB_LO}, {CALIB_HI}], closest to {CALIB_AIM}"}
    grid = [means[s] for s in CALIB_GRID]
    if all(m > CALIB_HI for m in grid):
        extra = CALIB_EXTRA["low"]
    elif all(m < CALIB_LO for m in grid):
        extra = CALIB_EXTRA["high"]
    else:
        extra = None
    if extra is not None and extra not in means:
        return {"decided": False, "status": "needs_extra", "next_strength": extra,
                "message": f"no grid strength in range; evaluate s = {extra:g}"}
    best = min(means, key=distance)
    why = ("the grid means straddle the interval" if extra is None
           else f"s = {extra:g} was evaluated")
    return {"decided": True, "status": "range_not_reached", "chosen_strength": best,
            "message": f"range [{CALIB_LO}, {CALIB_HI}] not reached ({why}); "
                       f"s = {best:g} has the mean closest to {CALIB_AIM} "
                       f"({means[best]:.4f})"}


def write_calibration(calib_root: Path) -> dict:
    points, ignored = calibration_points(calib_root)
    decision = decide({s: p["mean_polish_floor"] for s, p in points.items()})
    record = {**decision, "rule": {
        "devices": list(CALIB_DEVICES), "set": "set_0", "max_nfev": CALIB_NFEV,
        "grid": list(CALIB_GRID), "extra": CALIB_EXTRA,
        "interval": [CALIB_LO, CALIB_HI], "aim": CALIB_AIM,
        "interval_source": "out/research/pdk2/floors/2026-10-16_final "
                           "(GF180/IHP tier-c mean floors 0.063-0.123)",
        "statistic": "mean over the six set_0 targets of the polish floor"},
        "points": {f"{s:g}": p for s, p in sorted(points.items())},
        "ignored": ignored}
    path = calib_root / "calibration.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("decided", "status", "message")}
                     | {"means": {k: round(p["mean_polish_floor"], 5)
                                  for k, p in record["points"].items()}}, indent=1))
    return record


def calibrate_next(args, calib_root: Path) -> int:
    record = write_calibration(calib_root)
    if record["status"] == "incomplete":
        print(f"calibration grid incomplete: {record['message']}")
        return 1
    if record["status"] != "needs_extra":
        print(f"no extra strength needed ({record['status']})")
        return 0
    s = record["next_strength"]
    target = calib_root / f"s{s:g}"
    command = [sys.executable, str(ROOT / "scripts/research/make_sky130_misspec.py"),
               "--sets", "0", "--devices", ",".join(CALIB_DEVICES),
               "--strength", f"{s:g}", "--workers", str(args.workers),
               "--out", str(target)]
    print("running", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
    args.max_nfev = CALIB_NFEV
    run_floors(args, target, [0], list(CALIB_DEVICES))
    write_calibration(calib_root)
    return 0


# ----------------------------------------------------------------- finalize

def finalize(args, targets_dir: Path) -> int:
    from cryoml import strict_score as ss
    path = targets_dir / "floors.json"
    report = json.loads(path.read_text())
    reference_dir = Path(args.reference)
    if not reference_dir.is_absolute():
        reference_dir = ROOT / reference_dir
    oracle = None
    if not args.no_oracle_check:
        import dev_compare
        registry = {"bank": "data/processed/cold_loop/full_r1",
                    "round_dir": "out/cold_loop/full/round_1",
                    "targets_template": rel(targets_dir / "{set}.npz")}
        cache = Path(args.oracle_cache) if args.oracle_cache else (
            dev_compare.DEV / "oracle_reference.json")
        oracle = dev_compare.OracleReference(registry, cache, True)
    notes, missing, unused = [], [], []
    for key, row in report["targets"].items():
        set_name, tag = key.split("/")
        record_path = reference_dir / set_name / f"ml_{tag}.json"
        reference = None
        row["reference_calls"] = row["reference_budget"] = None
        if record_path.exists():
            record = json.loads(record_path.read_text())["methods"][REFERENCE_METHOD]
            status = ss.record_status(record, tag)
            if oracle is not None and abs(record["oracle_lhc"]["paper_rrms"]
                                          - oracle.get(set_name, tag)) > 1e-9:
                raise SystemExit(f"{record_path}: bank oracle differs from "
                                 f"{targets_dir} targets; not this target set")
            if status["status"] == ss.VALID:
                reference = float(record["rrms"])
            else:
                unused.append(key)
                notes.append(f"{key}: reference card has failed curves "
                             f"{status['failed_tags']}; not used")
            row["reference_calls"] = int(record["n_ngspice_calls"])
            row["reference_budget"] = int(record.get("budget", 0))
        else:
            missing.append(key)
            notes.append(f"{key}: no reference record ({record_path})")
        row["reference"] = reference
        candidates = [v for v in (row["polish"], reference)
                      if v is not None and math.isfinite(v)]
        row["floor"] = min(candidates) if candidates else math.inf
        row["floor_source"] = ("reference" if reference is not None
                               and reference < row["polish"] else "polish")
    # Finalized = every target had a reference record; a record whose card
    # has a failed included curve is read and left out (floor = polish).
    report["finalized"] = not missing
    report["finalize"] = {"reference": rel(reference_dir),
                          "method": REFERENCE_METHOD,
                          "oracle_checked": oracle is not None, "notes": notes,
                          "missing_reference": missing,
                          "unused_failed_reference": unused,
                          "rule": "floor = min(polish, reference); a reference "
                                  "card with a failed included curve is not used"}
    write_floors(path, report)
    for note in notes:
        print("NOTE", note)
    print(json.dumps(report["per_set"], indent=1))
    return 0


# --------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="out/research/sim_targets_sky130c")
    parser.add_argument("--sets", default="0,1")
    parser.add_argument("--devices", default=None)
    parser.add_argument("--max-nfev", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--calibrate", action="store_true",
                        help="floors on --targets, then the calibration rule "
                             "over its sibling s*/ directories")
    parser.add_argument("--calibrate-next", action="store_true",
                        help="run the one extra strength the rule asks for "
                             "(--targets = the calib root)")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--no-oracle-check", action="store_true")
    parser.add_argument("--oracle-cache", default=None,
                        help="bank-oracle cache (default: dev_compare's "
                             "out/research/dev/oracle_reference.json)")
    args = parser.parse_args()
    targets_dir = Path(args.targets)
    if not targets_dir.is_absolute():
        targets_dir = ROOT / targets_dir
    sets = sorted({int(x) for x in args.sets.split(",") if x.strip()})
    if any(k not in (0, 1) for k in sets):
        parser.error("only DEV sets 0 and 1 (set_2 is TEST)")
    devices = args.devices.split(",") if args.devices else None
    if args.calibrate_next:
        return calibrate_next(args, targets_dir)
    if args.finalize:
        if not args.reference:
            parser.error("--finalize needs --reference")
        return finalize(args, targets_dir)
    if args.calibrate:
        if sets != [0] or args.max_nfev != CALIB_NFEV:
            parser.error(f"--calibrate uses --sets 0 and --max-nfev {CALIB_NFEV}")
        run_floors(args, targets_dir, sets, devices or list(CALIB_DEVICES))
        write_calibration(targets_dir.parent)
        return 0
    run_floors(args, targets_dir, sets, devices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
