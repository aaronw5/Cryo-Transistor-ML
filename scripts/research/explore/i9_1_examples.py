#!/usr/bin/env python3
"""I9-1 example curves: NGSpice re-simulation of stored cards for the example
devices of the report batch (scripts/research/plot_explore_i9_1.py).

Devices (``pick``): per family (nMOS, pMOS) the DEV device whose
ln(RRMS_M_bias / RRMS_C0_refit) (lodo/results.json) is nearest the family
median, and the DEV device with the largest ln(RRMS_M_bias / RRMS_C0_refit)
over both families.  3 devices.  Every I9-1 DEV device lies in a bin of the
room-temperature card (no forced bins), so there is no bin stratum.

Per device, with the I9-1 deck (``cryoml.explore.i9_1_nist_rt.simulate``:
ngspice-41 arm64, open_pdks sky130 tt room-temperature card with the device's
bin only, mult = m = 1, deck temperature 30 C), on all Vb = 0 curves of the
measured grid:

  stock     stock card (no parameter written; for reference, not scored in I9-1)
  C0        starting card (start/records.json meta params)
  C0_refit  lodo/cards.json C0_refit card (held-out refit, no correction)
  M_bias    lodo/cards.json M_bias card; NGSpice output, and the output
            multiplied in Python by exp(g) of the device's stored LODO
            correction (lodo/corrections.json 'M_bias|<tag>')

Check: the strict paper RRMS (e1_correction.score_device) of C0, C0_refit and
M_bias is recomputed from these curves and compared with lodo/results.json
per_device.  The stock card's score_device RRMS is stored beside
start/summary.json stock_rrms (computed there by MeasProblem.curve_rrms; not a
check).

Output: out/research/explore/I9_1/examples/examples.json (the measured
currents of the 3 devices on their Vb = 0 curves are included for plotting).

  PYTHONPATH=src .venv/bin/python -I scripts/research/explore/i9_1_examples.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml import spice_pdk as sp  # noqa: E402
from cryoml.explore import i8_1_nist as b  # noqa: E402
from cryoml.explore import i9_1_nist_rt as m  # noqa: E402
from cryoml.explore.e1_correction import load_devices, score_device  # noqa: E402
from cryoml.explore.i3_2_bias import correction_from_json  # noqa: E402

OUT = m.OUT
DATA = OUT / "data"
TARGET = OUT / "examples" / "examples.json"


def pick(lodo: dict) -> list[str]:
    sc = lodo["per_device"]
    lr = {t: float(np.log(sc["M_bias"][t] / sc["C0_refit"][t])) for t in sc["M_bias"]
          if sc["M_bias"][t] is not None and sc["C0_refit"][t] is not None}
    out = []
    for fam in ("nfet", "pfet"):
        sel = {t: v for t, v in lr.items() if t.startswith(fam)}
        med = float(np.median(list(sel.values())))
        out.append(min(sorted(sel), key=lambda t: abs(sel[t] - med)))
    worst = max(sorted(lr), key=lambda t: lr[t])
    if worst not in out:
        out.append(worst)
    return out


def main() -> int:
    t0 = time.time()
    lodo = json.loads((OUT / "lodo/results.json").read_text())
    start = json.loads((OUT / "start/summary.json").read_text())
    cards = json.loads((OUT / "lodo/cards.json").read_text())
    corrs = json.loads((OUT / "lodo/corrections.json").read_text())
    bins = json.loads((OUT / "start/bins.json").read_text())["devices"]
    recs = {r.tag: r for r in load_devices(OUT / "start/records")}
    tags = pick(lodo)
    manifest = json.loads((DATA / "manifest.json").read_text())
    devices = {d.tag: d for d in b.load_dataset(DATA / "nist4k_rt.npz", manifest, tags)}
    out = {"id": "I9_1_examples", "simulator": sp._NGSPICE_BIN, "deck_temp_C": m.DECK_TEMP_C,
           "rule": "per family the device with ln(M_bias / C0_refit) nearest the family median, and the "
                   "device with the largest ln(M_bias / C0_refit) (lodo/results.json)",
           "devices": {}, "checks": {}}
    for tag in tags:
        dev, binfo, rec0 = devices[tag], bins[tag], recs[tag]
        stock = binfo["stock"]
        curves = b.to_pipeline_curves(dev)
        cset = {"stock": None, "C0": rec0.meta["params"], "C0_refit": cards["C0_refit"][tag],
                "M_bias": cards["M_bias"][tag]}
        sims, scores = {}, {}
        for arm, params in cset.items():
            s = m.simulate(dev.polarity, dev.W_um, dev.L_um, int(binfo["bin"]), curves,
                           None if params is None else b.changed_params(params, stock))
            rec = b.to_record(dev, binfo, s, params)
            if arm == "M_bias":
                corr = correction_from_json(corrs[f"M_bias|{tag}"])
                out_s = corr.apply(rec)
                sims["M_bias_card"] = s
                sims["M_bias"] = out_s
                scores[arm] = float(score_device(rec, out_s)["rrms"])
            else:
                sims[arm] = s
                scores[arm] = float(score_device(rec, s)["rrms"])
        stored = {a: lodo["per_device"][a][tag] for a in ("C0", "C0_refit", "M_bias")}
        out["checks"][tag] = {a: {"recomputed": scores[a], "stored": stored[a],
                                  "rel_diff": abs(scores[a] - stored[a]) / stored[a]} for a in stored}
        st = start["rows"][tag]["stock_rrms"]
        out["stock_scores"] = out.get("stock_scores", {})
        out["stock_scores"][tag] = {"score_device": scores["stock"], "start_summary_stock_rrms": st}
        rows = []
        for i, (c, r) in enumerate(zip(curves, rec0.curves)):
            rows.append({"key": c.meta["key"], "kind": c.kind, "fixed_V": c.fixed,
                         "vg": np.asarray(c.Vg, float).tolist(), "vd": np.asarray(c.Vd, float).tolist(),
                         "meas": np.asarray(c.Id, float).tolist(), "included": bool(r.included),
                         "start": int(r.start),
                         "sim": {k: np.asarray(v[i], float).tolist() for k, v in sims.items()}})
        out["devices"][tag] = {"polarity": dev.polarity, "W_um": dev.W_um, "L_um": dev.L_um,
                               "bin": int(binfo["bin"]), "covered": bool(binfo["covered"]),
                               "scores": scores, "curves": rows}
        print(f"{tag}: " + ", ".join(f"{a} {scores[a]:.6f} (stored {stored[a]:.6f})" for a in stored)
              + f"; stock score_device {scores['stock']:.6f}", flush=True)
    out["max_rel_diff"] = max(v["rel_diff"] for c in out["checks"].values() for v in c.values())
    out["runtime_s"] = round(time.time() - t0, 1)
    b.write_json(TARGET, out)
    print(f"max rel diff {out['max_rel_diff']:.2e}; {out['runtime_s']} s -> {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
