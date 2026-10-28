#!/usr/bin/env python3
"""Check that ``cryoml.extraction`` reproduces the stored I25-1 results.

  (r) round-1 ranges of all 18 devices, 7 and 45 parameters, from the
      template readback vs ``I25_1/specs/{seven,full}_r1.json``;
  (a) 7 parameters, one device, the whole procedure (3 starts, rounds 1-3,
      FD polish, start selection, final refit) vs ``I25_1/stepA/seven_*`` and
      ``I25_1/armA/c0/results.json``;
  (b) 45 parameters, one device, start 0, start draw and round 1 vs
      ``I25_1/stepA/full_s0_r1.json``.

(a) and (b) run in one pool of 4 workers.  Writes
``out/research/explore/extraction_equivalence/result.json``.

  PYTHONPATH=src .venv/bin/python scripts/research/extraction_equivalence.py
"""
from __future__ import annotations

import os

os.environ["CRYOML_PDK_TEMPLATE"] = "mult1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cryoml import extraction as ex  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.ngspice_fit import DeviceProblem  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

I25 = ROOT / "out/research/explore/I25_1"
OUT = ROOT / "out/research/explore/extraction_equivalence/result.json"
DEV_A = "nmos_L0p19_W7"      # 7 parameters; redraws on all 3 starts (5, 1, 1)
DEV_B = "nmos_L1_W3"         # 45 parameters, start 0; 1 redraw
REL_TOL = 1e-9


def jload(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def round1_task(job):
    tag, k, config, range1 = job
    seed = ex.start_seed(config, tag, k)
    draw = ex.draw_start(DeviceProblem(tag, range1.box()), np.random.default_rng(seed),
                         config.max_draws)
    return draw, ex.fit_round(tag, range1, config, draw.local, name="round 1")


class Table:
    def __init__(self):
        self.rows = []

    def add(self, check: str, quantity: str, stored, new) -> None:
        if isinstance(stored, dict):
            keys = sorted(stored)
            if sorted(new) != keys:
                raise KeyError(f"{check} {quantity}: keys differ")
            s = np.asarray([stored[k] for k in keys], dtype=float)
            n = np.asarray([new[k] for k in keys], dtype=float)
            diff = float(np.max(np.abs(s - n) / np.maximum(np.abs(s), 1e-300)))
            self.rows.append({"check": check, "quantity": quantity, "stored": "dict",
                              "new": "dict", "max_rel_diff": diff, "ok": diff <= REL_TOL})
            return
        s, n = float(stored), float(new)
        if s == n:
            diff = 0.0
        elif np.isfinite(s) and np.isfinite(n):
            diff = abs(s - n) / max(abs(s), 1e-300)
        else:
            diff = float("inf")
        self.rows.append({"check": check, "quantity": quantity, "stored": stored, "new": new,
                          "max_rel_diff": diff, "ok": diff <= REL_TOL})


def check_ranges(table: Table) -> None:
    tags = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]
    for pre, config in (("seven", ex.SEVEN), ("full", ex.FORTY_FIVE)):
        spec = jload(I25 / f"specs/{pre}_r1.json")
        worst, equal = 0.0, 0
        for tag in tags:
            r = ex.initial_range(config.parameters, ex.template_values(tag, config.parameters))
            got = r.to_json()
            if got["log_scale"] != spec["log_scale_by_device"][tag]:
                raise RuntimeError(f"{pre} {tag}: fixed parameters differ")
            s = np.asarray([spec["bounds_by_device"][tag][n] for n in config.parameters], float)
            g = np.asarray([got["bounds"][n] for n in config.parameters], float)
            d = float(np.max(np.abs(s - g) / np.maximum(np.abs(s), 1e-300)))
            worst, equal = max(worst, d), equal + int(d == 0.0)
        table.add("ranges", f"{pre} round-1 bounds, 18 devices (max rel diff; {equal}/18 identical)",
                  0.0, worst)


def check_a(table: Table, res: ex.DeviceResult) -> None:
    pre, tag = "seven", DEV_A
    for start in res.starts:
        for st, stage in zip(("r1", "r2", "r3", "fd"), start.stages):
            old = jload(I25 / f"stepA/{pre}_s{start.k}_{st}.json")["devices"][tag]
            c = f"(a) {tag} start {start.k} {st}"
            table.add(c, "device RRMS", old["rrms"], stage.rrms)
            table.add(c, "NGSpice runs", old["n_sim"], stage.n_sim)
            table.add(c, "card", old["params"], stage.params)
            if st == "r1":
                table.add(c, "start redraws", old["start_draws"], start.redraws)
                table.add(c, "draw runs", old["n_sim_start_draws"], start.draw_runs)
                table.add(c, "CMA-ES evaluations", old["cmaes_evals"], stage.cmaes_evals)
                table.add(c, "CMA-ES best", old["cmaes_best"], stage.cmaes_best)
    best = jload(I25 / f"stepA/{pre}_best.json")["devices"][tag]
    table.add(f"(a) {tag} selection", "kept start", best["start"], res.selected_start)
    table.add(f"(a) {tag} selection", "runs, all starts", best["n_sim"],
              res.ngspice_runs["all_starts"])
    table.add(f"(a) {tag} selection", "runs, kept start", best["n_sim_selected_chain"],
              res.ngspice_runs["selected_start"])
    c0 = jload(I25 / "armA/c0/results.json")["rows"][tag]
    table.add(f"(a) {tag} final refit", "device RRMS", c0["rrms"], res.final.rrms)
    table.add(f"(a) {tag} final refit", "NGSpice runs", c0["n_sim"], res.final.n_sim)
    table.add(f"(a) {tag} final refit", "card", c0["params"], res.final.params)


def check_b(table: Table, draw: ex.StartDraw, stage: ex.StageRecord) -> None:
    old = jload(I25 / "stepA/full_s0_r1.json")["devices"][DEV_B]
    c = f"(b) {DEV_B} start 0 r1 (45 parameters)"
    table.add(c, "start redraws", old["start_draws"], draw.redraws)
    table.add(c, "draw runs", old["n_sim_start_draws"], draw.runs)
    table.add(c, "CMA-ES evaluations", old["cmaes_evals"], stage.cmaes_evals)
    table.add(c, "CMA-ES best", old["cmaes_best"], stage.cmaes_best)
    table.add(c, "device RRMS", old["rrms"], stage.rrms)
    table.add(c, "NGSpice runs", old["n_sim"], stage.n_sim)
    table.add(c, "card", old["params"], stage.params)


def main() -> int:
    t0 = time.time()
    table = Table()
    check_ranges(table)
    print(f"ranges checked {time.time() - t0:.0f} s", flush=True)
    range_a = ex.initial_range(ex.SEVEN.parameters, ex.template_values(DEV_A, ex.SEVEN.parameters))
    range_b = ex.initial_range(ex.FORTY_FIVE.parameters,
                               ex.template_values(DEV_B, ex.FORTY_FIVE.parameters))
    with get_context("spawn").Pool(4) as pool:
        pending_b = pool.apply_async(round1_task, ((DEV_B, 0, ex.FORTY_FIVE, range_b),))
        res_a = ex.fit_devices([DEV_A], ex.SEVEN, {DEV_A: range_a},
                               map_fn=lambda fn, jobs: pool.map(fn, jobs, chunksize=1),
                               log=lambda m: print(m, flush=True))[DEV_A]
        draw_b, stage_b = pending_b.get()
    check_a(table, res_a)
    check_b(table, draw_b, stage_b)
    out = {"rel_tol": REL_TOL, "all_ok": all(r["ok"] for r in table.rows),
           "devices": {"a": DEV_A, "b": DEV_B}, "rows": table.rows,
           "wall_s": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float) + "\n")
    for r in table.rows:
        print(f"{'ok ' if r['ok'] else 'BAD'} {r['check']:42s} {r['quantity']:58s} "
              f"{r['stored']!s:>22.22} {r['new']!s:>22.22} {r['max_rel_diff']:.3g}")
    print(json.dumps({"all_ok": out["all_ok"], "wall_s": out["wall_s"]}), flush=True)
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
