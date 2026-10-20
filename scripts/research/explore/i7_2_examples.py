#!/usr/bin/env python3
"""I7-2 example curves: NGSpice re-simulation of the stored arm cards for the
example (device, draw) pairs of report figures 08-11.

Pairs: ``plot_explore_i7_2.pick_examples`` (two per dataset, 8 in total).
Per pair, every stored final card of ``cards.json`` (base, a_C0, b_shared,
c_hier, c_ml, d_scratch) is simulated on all curves of the device through the
I7-2 driver's dataset adapter (``i7_2_hier.dataset(name).full_record``), i.e.
the same simulator and settings as the I7-2 run:

  sky130   ngspice-41 (cryoml.config: native arm64 build), sky130 77 K card
  ihp300   ngspice-46 + psp103.osdi (cryoml.explore.i5_1_ihp_real)
  tud4/295 ngspice-41 arm64 (cryoml.explore.i6_1_tudelft.NGSPICE)

The arm output is then formed in Python exactly as in ``split_task``:

  a_C0           NGSpice card output (no correction)
  b_shared       NGSpice card output x exp(g_shared)        (stored fold M_bias)
  c_hier, c_ml   NGSpice card output x exp(g_shared + phi . delta)  (stored delta)
  d_scratch      NGSpice card output x exp(g_d); g_d = per-device M_bias-form MLP
                 refit on the K adaptation curves of d's record (run seeds and
                 hyperparameters; the run did not store this MLP)
  e_crossinterp  a_C0's NGSpice card output + interpolated measured-minus-
                 simulated residual of the adaptation curves at their crossings

Check: held-out and adaptation RRMS of every arm (and the base card) are
recomputed from the new curves and compared with ``<dataset>/results.json``.

Per curve and arm the JSON keeps ``ngspice_card`` (the card's NGSpice current)
and ``output`` (the arm's current after the Python step).  ``output`` is null for
base and a_C0 (equal to ``ngspice_card``); ``ngspice_card`` is null for
e_crossinterp (equal to a_C0's).

Output: ``out/research/explore/I7_2/examples/<dataset>__<tag>__<draw>.json`` and
``examples/checks.json`` (``--smoke``: ``smoke_examples/``).

  PYTHONPATH=src .venv/bin/python -I scripts/research/explore/i7_2_examples.py --workers 8
  PYTHONPATH=src .venv/bin/python -I scripts/research/explore/i7_2_examples.py --smoke \\
      --datasets sky130 --limit 1 --workers 1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryoml.explore import i7_2_hier as h  # noqa: E402
import i7_2_hier as drv  # noqa: E402

OUT = drv.OUT
CARD_ARMS = ("base", "a_C0", "b_shared", "c_hier", "c_ml", "d_scratch")
REL_TOL = 1e-6
OUTPUT_LABEL = {
    "base": "NGSpice card output (base fit on the K curves, no correction)",
    "a_C0": "NGSpice card output (no correction)",
    "b_shared": "NGSpice card output x exp(g_shared) applied in Python",
    "c_hier": "NGSpice card output x exp(g_shared + phi . delta) applied in Python",
    "c_ml": "NGSpice card output x exp(g_shared + phi . delta) applied in Python",
    "d_scratch": "NGSpice card output x exp(g_d) applied in Python (g_d refit on the K curves)",
    "e_crossinterp": "a_C0 NGSpice card output + crossing residual interpolation applied in Python",
}


def rel_diff(new: float, old: float) -> float:
    if not (np.isfinite(new) and np.isfinite(old)):
        return 0.0 if (np.isnan(new) and np.isnan(old)) else math.inf
    return abs(new - old) / max(abs(old), 1e-300)


def arr(x) -> list:
    return [float(v) for v in np.asarray(x, dtype=np.float64)]


def pair_task(task):
    """Re-simulate the stored cards of one (dataset, tag, draw) and form the arm outputs."""
    import torch
    torch.set_num_threads(1)
    from cryoml.explore.e1_correction import Correction
    from cryoml.explore.i3_2_bias import fit_bias
    name, tag, draw, row, cards = task
    t_start = time.time()
    ds = drv.dataset(name)
    tmo0 = ds.n_timeout() if hasattr(ds, "n_timeout") else 0
    adapt, held = row["adapt"], row["heldout"]
    g_sh = ds.g_shared(row["fold"])
    recs = {arm: ds.full_record(tag, cards[arm]) for arm in CARD_ARMS}
    t_sim = time.time() - t_start

    corr = {"base": Correction("C0"), "a_C0": Correction("C0"), "b_shared": g_sh}
    for arm in ("c_hier", "c_ml"):
        corr[arm] = h.ShiftedCorrection("hier", base=g_sh,
                                        delta=np.asarray(row["arms"][arm]["delta"], dtype=np.float64))
    t0 = time.time()
    corr["d_scratch"] = fit_bias("M_bias", [h.restrict(recs["d_scratch"], adapt)], hyper={}, threads=1)
    t_fit = time.time() - t0
    outputs = {arm: corr[arm].apply(recs[arm]) for arm in CARD_ARMS}
    outputs["e_crossinterp"], crossings = h.crossinterp_sims(recs["a_C0"], adapt, held)
    rec_of = {**recs, "e_crossinterp": recs["a_C0"]}

    check = {}
    for arm, sims in outputs.items():
        rec = rec_of[arm]
        stored = row["base"] if arm == "base" else row["arms"][arm]
        new_h = float(h.score_keys(rec, sims, held)["rrms"])
        new_a = float(h.score_keys(rec, sims, adapt)["rrms"])
        check[arm] = {"heldout_new": new_h, "heldout_stored": stored["heldout_rrms"],
                      "heldout_rel_diff": rel_diff(new_h, stored["heldout_rrms"]),
                      "adapt_new": new_a, "adapt_stored": stored["adapt_rrms"],
                      "adapt_rel_diff": rel_diff(new_a, stored["adapt_rrms"])}
    assert crossings == row["arms"]["e_crossinterp"]["crossings"], (name, tag, draw)

    curves = []
    for i, c in enumerate(recs["a_C0"].curves):
        cur = {"key": c.key, "kind": c.kind, "included": bool(c.included), "start": int(c.start),
               "den": float(c.den),
               "role": "adapt" if c.key in adapt else ("heldout" if c.key in held else "other"),
               "vg": arr(c.vg), "vd": arr(c.vd), "measured": arr(c.target), "arms": {}}
        for arm, sims in outputs.items():
            rc = rec_of[arm].curves[i]
            assert rc.key == c.key
            # null = same array as stated in the module docstring (no duplicate storage)
            cur["arms"][arm] = {
                "ngspice_card": None if arm == "e_crossinterp" else arr(rc.sim),
                "output": None if arm in ("base", "a_C0") else arr(sims[i])}
            if arm in ("base", "a_C0"):
                assert np.array_equal(np.asarray(sims[i]), np.asarray(rc.sim, float))
        curves.append(cur)
    out = {"dataset": name, "tag": tag, "draw": draw, "K": row["K"], "fold": row["fold"],
           "family": recs["a_C0"].family, "adapt": adapt, "heldout": held, "mu": row["mu"],
           "output_label": OUTPUT_LABEL, "cards": cards,
           "deltas": {a: row["arms"][a]["delta"] for a in ("c_hier", "c_ml")},
           "d_scratch_mlp": corr["d_scratch"].to_json(), "crossings": crossings,
           "check": check, "curves": curves,
           "n_sim_timeout": (ds.n_timeout() - tmo0) if hasattr(ds, "n_timeout") else 0,
           "timing": {"simulate_s": round(t_sim, 1), "d_refit_s": round(t_fit, 1),
                      "total_s": round(time.time() - t_start, 1)}}
    print(f"  {name} {tag} {draw}: " + " ".join(
        f"{a}={v['heldout_new']:.6f}({v['heldout_rel_diff']:.1e})" for a, v in check.items())
        + f" {out['timing']['total_s']} s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(drv.DATASETS))
    ap.add_argument("--limit", type=int, default=0, help="pairs per dataset (0 = both)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="write to smoke_examples/")
    args = ap.parse_args()
    import plot_explore_i7_2 as pe  # pick rule of figures 08-11 (single definition)
    names = [d for d in args.datasets.split(",") if d]
    res = {d: json.loads((OUT / d / "results.json").read_text()) for d in drv.DATASETS}
    cards = {d: json.loads((OUT / d / "cards.json").read_text()) for d in names}
    picks = pe.pick_examples({"res": res})
    tasks = []
    for d in names:
        by = {(c["tag"], c["draw"]): c["cards"] for c in cards[d]}
        sel = picks[d][:args.limit] if args.limit else picks[d]
        for p in sel:
            x = p["row"]
            tasks.append((d, x["tag"], x["draw"], x, by[(x["tag"], x["draw"])]))
    target = OUT / ("smoke_examples" if args.smoke else "examples")
    target.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"I7_2 examples: {len(tasks)} pairs -> {target.relative_to(ROOT)}", flush=True)
    with get_context("spawn").Pool(max(1, min(args.workers, len(tasks)))) as pool:
        outs = pool.map(pair_task, tasks, chunksize=1)
    summary = []
    for o in outs:
        f = target / f"{o['dataset']}__{o['tag']}__{o['draw']}.json"
        f.write_text(json.dumps(o) + "\n")
        for arm, v in o["check"].items():
            summary.append({"dataset": o["dataset"], "tag": o["tag"], "draw": o["draw"], "arm": arm, **v})
    worst = {}
    for s in summary:
        k = "d_scratch" if s["arm"] == "d_scratch" else "other"
        worst[k] = max(worst.get(k, 0.0), s["heldout_rel_diff"], s["adapt_rel_diff"])
    over = [s for s in summary if max(s["heldout_rel_diff"], s["adapt_rel_diff"]) > REL_TOL]
    chk = {"rel_tol": REL_TOL, "n_pairs": len(outs), "max_rel_diff": worst,
           "over_tol": over, "rows": summary, "runtime_s": round(time.time() - t0, 1),
           "n_sim_timeout": int(sum(o["n_sim_timeout"] for o in outs))}
    (target / "checks.json").write_text(json.dumps(chk, indent=1) + "\n")
    print(f"max relative difference vs results.json: {worst}; over {REL_TOL}: {len(over)} "
          f"(arm/pair rows); runtime {chk['runtime_s']} s")
    for s in over:
        print(f"  over tol: {s['dataset']} {s['tag']} {s['draw']} {s['arm']} held "
              f"{s['heldout_new']:.8f} vs {s['heldout_stored']:.8f}, adapt {s['adapt_new']:.8f} vs "
              f"{s['adapt_stored']:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
