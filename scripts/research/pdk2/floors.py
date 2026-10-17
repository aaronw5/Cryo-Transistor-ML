#!/usr/bin/env python3
"""Per-tier refit floors: ``ngspice_fit.polish`` started at the truth card's
fitted-parameter projection (``<tag>__local_<set>``), on a few devices.

For every (tier, device, fitted set): the paper RRMS of the projection
(``start``), after polish (``floor``), and of the noise alone (the truth's
own clean curves scored against its observed curves, ``noise``).  Truth
sets are generated fresh (1 set) under ``<out>/truth_<pdk>``; one ngspice
process at a time.

``<out>/floors_<pdk>.json`` carries a ``settings`` record: the exact
``make_truth`` argv per tier, each tier's ``truth.json`` (resolved options,
base and per-device seeds, misspecification draws), the misspecified
parameter list and other generator constants, devices, polish settings,
the git commit (and dirty tracked files), source hashes, and the ngspice
binary and version.

  .venv/bin/python -I scripts/research/pdk2/floors.py --pdk gf180 \
      --devices gf180_nmos_L0p3_W0p3,gf180_pmos_L1_W5 --out <dir>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_truth  # noqa: E402
from cryoml.ngspice_fit import polish  # noqa: E402
from cryoml.pdk2 import boxes  # noqa: E402
from cryoml.pdk2.problem import Pdk2Problem, final_score  # noqa: E402

# Sources whose content decides the truth draws and the floors.
SOURCES = ("scripts/research/pdk2/floors.py", "scripts/research/pdk2/make_truth.py",
           "src/cryoml/pdk2/boxes.py", "src/cryoml/pdk2/problem.py",
           "src/cryoml/pdk2/metric.py", "src/cryoml/pdk2/gf180.py",
           "src/cryoml/pdk2/ihp.py", "src/cryoml/ngspice_fit.py")


def _git(*args: str) -> str | None:
    try:
        run = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, timeout=30, check=False)
    except OSError:
        return None
    return run.stdout.rstrip() if run.returncode == 0 else None


def git_record() -> dict:
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_tracked_files": None if dirty is None else
            [line[3:] for line in dirty.splitlines()]}


def source_hashes() -> dict:
    return {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            for rel in SOURCES if (ROOT / rel).exists()}


def simulator_record(pdk: str) -> dict:
    sim = boxes.backend(pdk)
    binary = sim.ngspice46_bin() if pdk == "ihp" else sim.NGSPICE_BIN
    out = subprocess.run([binary, "-v"], capture_output=True, text=True,
                         timeout=30, check=False).stdout
    version = next((line.strip("* ").strip() for line in out.splitlines()
                    if "ngspice-" in line), out.strip()[:200])
    record = {"binary": binary, "version": version,
              "pdk": f"{sim.PDK_NAME}@{sim.PDK_VERSION}", "fit_temp_C": sim.TEMP_C}
    if pdk == "ihp":
        record["osdi"] = str(sim.OSDI.relative_to(ROOT))
        record["osdi_sha256"] = hashlib.sha256(sim.OSDI.read_bytes()).hexdigest()
    return record


def polish_record(nfev: int, irls: int) -> dict:
    defaults = {name: p.default for name, p in inspect.signature(polish).parameters.items()
                if p.default is not inspect.Parameter.empty}
    return {"function": f"{polish.__module__}.polish", **defaults,
            "max_nfev": nfev, "irls_rounds": irls,
            "start": "<tag>__local_<set> (truth's fitted-parameter projection)",
            "score": "cryoml.pdk2.problem.final_score (fresh simulation)"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdk", choices=("gf180", "ihp"), required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--tiers", default="a,b,c")
    parser.add_argument("--fit-sets", default="seven,wide")
    parser.add_argument("--nfev", type=int, default=300)
    parser.add_argument("--irls", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--truth-args", default="",
                        help="extra make_truth.py arguments (quoted)")
    args = parser.parse_args()
    started = dt.datetime.now().astimezone()
    t_start = time.time()
    out = Path(args.out)
    truth_root = out / f"truth_{args.pdk}"
    tiers = args.tiers.split(",")
    devices = args.devices.split(",")
    fit_sets = args.fit_sets.split(",")
    truth_argv = {}
    for tier in tiers:
        truth_argv[tier] = ["--pdk", args.pdk, "--tier", tier, "--sets", "1",
                            "--devices", args.devices, "--out-root", str(truth_root),
                            *args.truth_args.split()]
        make_truth.main(truth_argv[tier])
    resolved = make_truth.tier_options(make_truth.parser().parse_args(truth_argv[tiers[0]]))
    rows = []
    for tier in tiers:
        with np.load(truth_root / tier / "set_0.npz") as data:
            arrays = {k: data[k] for k in data.files}
        for tag in devices:
            clean = np.split(arrays[f"{tag}__clean_ids"], 11)
            for set_name in fit_sets:
                if tier == "a" and set_name == "wide":
                    continue          # well-specified: the floor is 0 either way
                box = boxes.box_for(set_name, tag)
                problem = Pdk2Problem.from_truth(tag, box, tier, 0, truth_root)
                start_local = arrays[f"{tag}__local_{set_name}"]
                t0 = time.time()
                start = problem.objective(start_local)
                result = polish(problem, start_local, max_nfev=args.nfev,
                                irls_rounds=args.irls)
                floor = final_score(problem, result["local"])["rrms"]
                noise = float(np.mean(problem.curve_rrms(clean)))
                row = {"tier": tier, "device": tag, "set": set_name,
                       "start": round(start, 5), "floor": round(floor, 5),
                       "noise": round(noise, 5), "n_sim": problem.n_sim,
                       "runtime_s": round(time.time() - t0, 1)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    summary = {}
    for tier in tiers:
        for set_name in fit_sets:
            sel = [r for r in rows if r["tier"] == tier and r["set"] == set_name]
            if sel:
                summary[f"{tier}/{set_name}"] = {
                    "mean_floor": round(float(np.mean([r["floor"] for r in sel])), 5),
                    "mean_noise": round(float(np.mean([r["noise"] for r in sel])), 5),
                    "mean_start": round(float(np.mean([r["start"] for r in sel])), 5)}
    settings = {
        "command": [sys.executable, *sys.argv],
        "started": started.isoformat(timespec="seconds"),
        "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime_s": round(time.time() - t_start, 1),
        "pdk": args.pdk, "devices": devices, "tiers": tiers, "fit_sets": fit_sets,
        "skipped": ["a/wide (well-specified; the floor is 0 by construction)"]
        if "a" in tiers and "wide" in fit_sets else [],
        "truth_generator": {
            "script": "scripts/research/pdk2/make_truth.py",
            "root": str(truth_root), "sets": 1, "set_index": 0,
            "truth_args": args.truth_args, "argv": truth_argv,
            "options": resolved,
            "misspec_params": list(boxes.PDK_MISSPEC[args.pdk]),
            "b_width_defaults": dict(make_truth.B_WIDTH),
            "ihp_b_rext_k": list(make_truth.IHP_B_REXT_K),
            "seed_offsets": {"tier": dict(make_truth.TIER_SEED),
                             "pdk": dict(make_truth.PDK_SEED)},
            "per_tier": {tier: json.loads((truth_root / tier / "truth.json").read_text())
                         for tier in tiers}},
        "polish": polish_record(args.nfev, args.irls),
        "git": git_record(),
        "source_sha256": source_hashes(),
        "simulator": simulator_record(args.pdk),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "host": platform.node()}
    report = {"pdk": args.pdk, "devices": devices,
              "nfev": args.nfev, "irls": args.irls,
              "truth_args": args.truth_args, "settings": settings,
              "rows": rows, "summary": summary}
    (out / f"floors_{args.pdk}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"runtime {settings['runtime_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
