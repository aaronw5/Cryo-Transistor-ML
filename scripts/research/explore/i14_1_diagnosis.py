#!/usr/bin/env python3
"""I14-1 diagnosis (post hoc; the recorded outcome stays as decided): the
convergence warnings of the I14-1 Part B v2 run.

The only failed core check was 'Q1 runs': nmos_L8_W1p6 (M2 card of g v2),
exchanged-terminal bench at tight tolerance, element run printed 'dynamic gmin
stepping failed' and 'true gmin stepping failed' (the card + replica forward
tight reference run printed the same two lines).  This script reruns the
tight forward and exchanged grids of that card in single ngspice processes:

  card       card only (no replica)
  card_rep   card + channel replica + gmin VCCS + 0 V sense (I14-1 reference)
  card_rep_novccs   as card_rep without the gmin VCCS
  card_rep_i12      card + the I12-1 charge-free replica (junction, Isub kept)
  elem       I14-1 element (as in Part B)
  elem_novccs       element without the gmin VCCS

and records the convergence messages, the row count and, for the element
runs, the Q1 criterion against the matching card + replica run.  Also the
same six runs for the other 17 devices' M2 cards (messages only).

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i14_1_diagnosis.py
"""
from __future__ import annotations

import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i14_1_chan as drv  # noqa: E402
from cryoml.explore import i11_1_deploy as m1  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i14_1_chan as m  # noqa: E402

RD = drv.OUT / "deploy_v2"
OUT = drv.OUT / "diagnosis"
VARIANTS = ("card", "card_rep", "card_rep_novccs", "card_rep_i12", "elem", "elem_novccs")


def body_for(dev, variant: str, swap: bool) -> str:
    mode = {"card": "card", "elem": "elem", "elem_novccs": "elem"}.get(variant, "card_rep")
    b = m.device_body(dev, mode, swap=swap)
    if variant.endswith("novccs"):
        b = "".join(ln + "\n" for ln in b.splitlines() if not ln.startswith("Gm1c "))
    return b


def task(t):
    tag, params, variant = t
    keep = dict(m.REPLICA_CHAN)
    try:
        if variant == "card_rep_i12":
            m.REPLICA_CHAN.clear()
            m.REPLICA_CHAN.update(v2.REPLICA_ZERO)
        return _task(tag, params, variant)
    finally:
        m.REPLICA_CHAN.clear()
        m.REPLICA_CHAN.update(keep)


def _task(tag, params, variant):
    dev = drv.devices()[tag]
    osdi = drv.arm_osdi(RD, "v2") if variant.startswith("elem") else None
    out = {"tag": tag, "variant": variant}
    sense = variant.startswith("card_rep")
    with m1.tempdir("cryoml_i14d_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, {dev.dev_type: [(dev.bin, params)]})
        for swap in (False, True):
            lab = "swap" if swap else "fwd"
            f = td / f"{lab}.dat"
            r = m1.run_deck(m.deck(lab, lib, body_for(dev, variant, swap), m.grid_control(dev, f, swap, sense),
                                   osdi, options=m1.TIGHT), td, f"{lab}.sp", timeout_s=300)
            rows = m1.read_wrdata(f, 4 if sense else 2)
            out[lab] = {"msgs": r["msgs"], "n_rows": int(rows.shape[0])}
            out[f"{lab}_I"] = m1.grid_from_rows(rows[:, :2]).tolist()
            if sense:
                out[f"{lab}_rep"] = m1.grid_from_rows(rows[:, 2:4]).tolist()
    return out


def main() -> int:
    cards = drv.jload(RD / "cards_M2.json")
    tags = sorted(cards)
    tasks = [(t, cards[t], v) for t in tags for v in VARIANTS]
    with get_context("spawn").Pool(4) as pool:
        rows = pool.map(task, tasks, chunksize=1)
    res = {"note": "post hoc; recorded outcome unchanged", "options": m1.TIGHT, "messages": {}, "q1": {}}
    by = {(r["tag"], r["variant"]): r for r in rows}
    for (tag, var), r in by.items():
        msgs = {k: r[k]["msgs"] for k in ("fwd", "swap") if r[k]["msgs"]}
        if msgs:
            res["messages"][f"{tag}|{var}"] = msgs
    corr = drv.corr_full("v2")
    for tag in tags:
        dev = drv.devices()[tag]
        pm = dev.dev_type == "pmos"
        s = -1.0 if pm else 1.0
        VG, VD = np.meshgrid(s * m1.GRID_VGS, m1.GRID_VDS, indexing="ij")
        g_f = v2.g_terminal(corr, pm, VG, VD, 0.0)
        g_s = v2.g_terminal(corr, pm, VG, 0.0, VD)
        for el, ref in (("elem", "card_rep"), ("elem_novccs", "card_rep_novccs")):
            e, c = by[(tag, el)], by[(tag, ref)]
            f = v2.eq_violation(np.array(e["fwd_I"]), m.current_fwd(np.array(c["fwd_I"]), np.array(c["fwd_rep"]), g_f))
            w = v2.eq_violation(np.array(e["swap_I"]), m.current_swap(np.array(c["swap_I"]), np.array(c["swap_rep"]), g_s))
            res["q1"][f"{tag}|{el}"] = {"fwd_viol": f["n_viol"], "fwd_max_ratio": f["max_ratio"],
                                        "swap_viol": w["n_viol"], "swap_max_ratio": w["max_ratio"]}
        cc, c0 = np.array(by[(tag, "card_rep")]["fwd_I"]), np.array(by[(tag, "card")]["fwd_I"])
        res.setdefault("card_rep_minus_card_fwd_max_abs", {})[tag] = float(np.nanmax(np.abs(cc - c0)))
    res["devices_with_messages"] = sorted({k.split("|")[0] for k in res["messages"]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnosis.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"messages": res["messages"],
                      "q1_max_ratio": max(max(v["fwd_max_ratio"], v["swap_max_ratio"]) for v in res["q1"].values()),
                      "q1_viol": sum(v["fwd_viol"] + v["swap_viol"] for v in res["q1"].values())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
