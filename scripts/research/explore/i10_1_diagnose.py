#!/usr/bin/env python3
"""I10-1 failure diagnosis (NIST DEV): why the failed (device, arm) records are
non-finite.  Written after the I10-1 decision ("invalid", V2); not part of the
pre-registered pipeline and not used for any I10-1 label.

1. Analytic check, no simulation, for every stored NIST arm card
   (lodo_nist/cards.json) and the C0_RT starting cards: the size-dependent
   BSIM4 values of PCLM and LPEB of the device's bin,
       P = P0 + lP / Leff + wP / Weff + pP / (Leff Weff)   (binunit 2: metres),
       Leff = L - 2 dL, Weff = W - 2 dW   (dL from lint, ll, lw, lwl, lln, lwn;
                                           dW from wint, wl, ww, wwl, wln, wwn),
   against the BSIM4 parameter-check conditions that stop ngspice
   ("Fatal: Pclm = ... is not positive", "Fatal: Lpeb = ... is less than
   -Leff").  The predicted-invalid set is compared with the failed set of each
   arm (lodo_nist/results.json per_device).
2. ngspice messages: one .op run (|Vg| = |Vd| = 3.3 V, deck 30 C, the I10-1
   single-bin card with the stored parameters) per failed (device, arm) of the
   primary arms A1, A3, R_pop (A2 uses the A1 card) and for the C0_RT card of the
   device excluded by the A1 rule; the "Fatal:" lines are stored.

Output: out/research/explore/I10_1/diagnosis/diagnosis.json

  PYTHONPATH=src .venv/bin/python -I scripts/research/explore/i10_1_diagnose.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("i10_1_driver", ROOT / "scripts/research/explore/i10_1_rt4k.py")
drv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drv)
m, b = drv.m, drv.b

OUT = m.OUT
TARGET = OUT / "diagnosis" / "diagnosis.json"
PRIMARY_SIM = ("A1", "A3", "R_pop")
ARMS_CHECK = ("A0_rtdeck", "A1", "A1_L", "A3", "R_pop", "V_mean", "V_ind", "V_sub7", "V_deck")


def _val(blk: str, p: str, default: float = 0.0) -> float:
    mm = re.search(rf"^\+\s*{p}\s*=\s*([^\s]+)", blk, re.I | re.M)
    return float(mm.group(1)) if mm else default


def geometry(row: dict) -> tuple[str, float, float]:
    blk = m.card_parts(row["dev_type"])["blocks"][row["bin"]["bin"]]
    if _val(blk, "binunit", 1.0) != 2.0:
        raise RuntimeError(f"{row['tag']}: binunit {_val(blk, 'binunit', 1.0)} (expected 2)")
    L, W = row["L_um"] * 1e-6, row["W_um"] * 1e-6
    lln, lwn, wln, wwn = (_val(blk, k, 1.0) for k in ("lln", "lwn", "wln", "wwn"))
    dL = _val(blk, "lint") + _val(blk, "ll") / L ** lln + _val(blk, "lw") / W ** lwn \
        + _val(blk, "lwl") / (L ** lln * W ** lwn)
    dW = _val(blk, "wint") + _val(blk, "wl") / L ** wln + _val(blk, "ww") / W ** wwn \
        + _val(blk, "wwl") / (L ** wln * W ** wwn)
    return blk, L - 2 * dL, W - 2 * dW


def effective(row: dict, params: dict) -> dict:
    blk, Le, We = geometry(row)
    out = {"Leff_m": Le, "Weff_m": We}
    for name in ("pclm", "lpeb"):
        p0 = float(params[name])
        out[f"{name}_base"] = p0
        out[f"{name}_eff"] = p0 + _val(blk, "l" + name) / Le + _val(blk, "w" + name) / We \
            + _val(blk, "p" + name) / (Le * We)
    bad = []
    if out["pclm_eff"] <= 0:
        bad.append("Pclm <= 0")
    if out["lpeb_eff"] < -Le:
        bad.append("Lpeb < -Leff")
    out["predicted_invalid"] = bad
    # margins: Pclm_eff itself; Lpeb_eff / Leff (fatal below -1)
    out["lpeb_over_leff"] = out["lpeb_eff"] / Le
    return out


def op_fatal(ds, tag: str, params: dict) -> list[str]:
    d = ds.dev4[tag]
    p = b.changed_params(params, ds.stock(tag))
    sign = -1 if d.polarity == "pmos" else 1
    with tempfile.TemporaryDirectory(prefix="i10_1_diag_") as td_s:
        td = Path(td_s)
        lib = m.write_lib(td, d.dev_type, int(ds.binfo(tag)["bin"]), p)
        deck = td / "d.sp"
        deck.write_text(f"* I10-1 diagnosis .op\n.options scale=1.0\n.options temp={m.DECK_RT_C:.4f}\n"
                        f'.lib "{lib.as_posix()}" tt_i10\n' + m.instance_line(d.dev_type, d.W_um, d.L_um)
                        + f"VG ng 0 DC {sign * 3.3}\nVD nd 0 DC {sign * 3.3}\n.op\n.control\nrun\nquit\n.endc\n.end\n")
        _, log = m._ngspice(deck, td, 30)
    return sorted({ln.strip() for ln in log.splitlines() if ln.strip().lower().startswith("fatal:")})


def main() -> int:
    t0 = time.time()
    man = json.loads((OUT / "data/manifest.json").read_text())
    rows = {r["tag"]: r for r in man["devices"] if r["tag"] in set(man["DEV"])}
    cards = json.loads((OUT / "lodo_nist/cards.json").read_text())
    res = json.loads((OUT / "lodo_nist/results.json").read_text())
    pd = res["per_device"]
    excluded = res["excluded_A1_rule"]
    recrt = drv.load_fit_records(OUT / "start_nist/records_C0_RT")
    finite = lambda v: v is not None and math.isfinite(v)  # noqa: E731

    analytic, agreement = {}, {}
    analytic["C0_RT"] = {t: effective(rows[t], recrt[t].meta["params"]) for t in sorted(rows)}
    for t, v in analytic["C0_RT"].items():
        v["failed"] = not b.finite_record(recrt[t])
    for arm in ARMS_CHECK:
        analytic[arm] = {}
        for t, params in cards[arm].items():
            e = effective(rows[t], params)
            e["failed"] = not finite(pd[arm][t])
            analytic[arm][t] = e
    for arm, per in analytic.items():
        pred = {t for t, v in per.items() if v["predicted_invalid"]}
        fail = {t for t, v in per.items() if v["failed"]}
        agreement[arm] = {"n_cards": len(per), "n_predicted_invalid": len(pred), "n_failed": len(fail),
                          "failed_and_predicted": len(pred & fail),
                          "failed_not_predicted": sorted(fail - pred),
                          "predicted_not_failed": sorted(pred - fail)}

    ds = drv.get_ds("nist", False)
    sims = []
    for arm in PRIMARY_SIM:
        for t in sorted(pd[arm]):
            if t in excluded or finite(pd[arm][t]):
                continue
            sims.append({"arm": arm, "tag": t, "fatal": op_fatal(ds, t, cards[arm][t])})
    for t in excluded:
        sims.append({"arm": "C0_RT", "tag": t, "fatal": op_fatal(ds, t, recrt[t].meta["params"])})
    for s in sims:
        s["analytic"] = analytic[s["arm"]][s["tag"]]
        s["C0_RT_analytic"] = analytic["C0_RT"][s["tag"]]
        r = rows[s["tag"]]
        s.update(dev_type=r["dev_type"], W_um=r["W_um"], L_um=r["L_um"], bin=r["bin"]["bin"],
                 covered=bool(r["bin"]["covered"]), stretch_L=r["bin"]["stretch_L"], stretch_W=r["bin"]["stretch_W"])
        print(f"{s['arm']:6s} {s['tag']:44s} {'; '.join(s['fatal'])}", flush=True)

    counts = {}
    for s in sims:
        for f in s["fatal"]:
            key = "Pclm <= 0" if f.startswith("Fatal: Pclm") else "Lpeb < -Leff" if f.startswith("Fatal: Lpeb") else f
            counts.setdefault(s["arm"], {}).setdefault(key, 0)
            counts[s["arm"]][key] += 1
    out = {"id": "I10_1_diagnosis", "simulator": b._sp()._NGSPICE_BIN,
           "note": "post hoc; written after the I10-1 decision; not used for any I10-1 label",
           "rule": "analytic BSIM4 binning of PCLM and LPEB (binunit 2) vs the fatal parameter checks; ngspice .op "
                   "runs for the failed primary-arm records and the excluded C0_RT record",
           "excluded_A1_rule": excluded, "agreement": agreement, "fatal_counts": counts, "op_runs": sims,
           "n_op_runs": len(sims), "analytic": analytic, "runtime_s": round(time.time() - t0, 1)}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    b.write_json(TARGET, out)
    print(json.dumps({"agreement": {a: {k: v[k] for k in ("n_cards", "n_predicted_invalid", "n_failed",
                                                          "failed_and_predicted")} for a, v in agreement.items()},
                      "fatal_counts": counts}, indent=1))
    print(f"{len(sims)} .op runs, {out['runtime_s']} s -> {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
