#!/usr/bin/env python3
"""I8-1 example curves: NGSpice re-simulation of stored cards for the example
devices of the report batch (scripts/research/plot_explore_i8_1.py).

Devices (``pick``): per family (nMOS, pMOS) and bin stratum (native bin,
forced nearest bin), the DEV device whose ln(RRMS_M_bias / RRMS_C0_refit)
(lodo/results.json) is nearest the median of its stratum; devices with a NaN
score are not candidates.  4 devices.

Per device, with the I8-1 deck (``cryoml.explore.i8_1_nist.simulate``:
ngspice-41 arm64, 77 K card with the device's explicit bin, mult = m = 1,
deck temperature 77 K), on all Vb = 0 curves of the measured grid:

  C0        starting card (start/records.json meta params)
  C0_refit  lodo/cards.json C0_refit card (held-out refit, no correction)
  M_bias    lodo/cards.json M_bias card; NGSpice output, and the output
            multiplied in Python by exp(g) of the device's stored LODO
            correction (lodo/corrections.json 'M_bias|<tag>')

Check: the strict paper RRMS (e1_correction.score_device) of each arm is
recomputed from these curves and compared with lodo/results.json per_device.

Output: out/research/explore/I8_1/examples/examples.json (no measured raw
files; the measured currents of the 4 devices on their Vb = 0 curves are
included for plotting).

  PYTHONPATH=src .venv/bin/python -I scripts/research/explore/i8_1_examples.py
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
from cryoml.explore import i8_1_nist as m  # noqa: E402
from cryoml.explore.e1_correction import load_devices, score_device  # noqa: E402
from cryoml.explore.i3_2_bias import correction_from_json  # noqa: E402

OUT = m.OUT
DATA = OUT / "data"
TARGET = OUT / "examples" / "examples.json"


def pick(lodo: dict, bins: dict) -> list[str]:
    sc = lodo["per_device"]
    out = []
    for fam in ("nfet", "pfet"):
        for cov in (True, False):
            lr = {t: float(np.log(sc["M_bias"][t] / sc["C0_refit"][t])) for t in sc["M_bias"]
                  if t.startswith(fam) and bins[t]["covered"] == cov
                  and sc["M_bias"][t] is not None and sc["C0_refit"][t] is not None}
            med = float(np.median(list(lr.values())))
            out.append(min(sorted(lr), key=lambda t: abs(lr[t] - med)))
    return out


def main() -> int:
    t0 = time.time()
    lodo = json.loads((OUT / "lodo/results.json").read_text())
    cards = json.loads((OUT / "lodo/cards.json").read_text())
    corrs = json.loads((OUT / "lodo/corrections.json").read_text())
    bins = json.loads((OUT / "start/bins.json").read_text())["devices"]
    recs = {r.tag: r for r in load_devices(OUT / "start/records")}
    tags = pick(lodo, bins)
    manifest = json.loads((DATA / "manifest.json").read_text())
    devices = {d.tag: d for d in m.load_dataset(DATA / "nist4k.npz", manifest, tags)}
    out = {"id": "I8_1_examples", "simulator": sp._NGSPICE_BIN, "deck_temp_K": m.DECK_TEMP_K,
           "rule": "per family and bin stratum, the device with ln(M_bias / C0_refit) nearest the "
                   "stratum median (lodo/results.json)",
           "devices": {}, "checks": {}}
    for tag in tags:
        dev, binfo, rec0 = devices[tag], bins[tag], recs[tag]
        stock = binfo["stock"]
        curves = m.to_pipeline_curves(dev)
        cset = {"C0": rec0.meta["params"], "C0_refit": cards["C0_refit"][tag], "M_bias": cards["M_bias"][tag]}
        sims, scores = {}, {}
        for arm, params in cset.items():
            s = m.simulate(dev.polarity, dev.W_um, dev.L_um, int(binfo["bin"]), curves,
                           m.changed_params(params, stock))
            rec = m.to_record(dev, binfo, s, params)
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
                                  "rel_diff": abs(scores[a] - stored[a]) / stored[a]} for a in scores}
        rows = []
        for i, (c, r) in enumerate(zip(curves, rec0.curves)):
            rows.append({"key": c.meta["key"], "kind": c.kind, "fixed_V": c.fixed,
                         "vg": np.asarray(c.Vg, float).tolist(), "vd": np.asarray(c.Vd, float).tolist(),
                         "meas": np.asarray(c.Id, float).tolist(), "included": bool(r.included),
                         "start": int(r.start),
                         "sim": {k: np.asarray(v[i], float).tolist() for k, v in sims.items()}})
        out["devices"][tag] = {"polarity": dev.polarity, "W_um": dev.W_um, "L_um": dev.L_um,
                               "bin": int(binfo["bin"]), "covered": bool(binfo["covered"]),
                               "stretch": [binfo["stretch_L"], binfo["stretch_W"]], "curves": rows}
        print(f"{tag}: " + ", ".join(f"{a} {scores[a]:.6f} (stored {stored[a]:.6f})" for a in scores), flush=True)
    out["max_rel_diff"] = max(v["rel_diff"] for c in out["checks"].values() for v in c.values())
    out["runtime_s"] = round(time.time() - t0, 1)
    m.write_json(TARGET, out)
    print(f"max rel diff {out['max_rel_diff']:.2e}; {out['runtime_s']} s -> {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
