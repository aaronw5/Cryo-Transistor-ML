#!/usr/bin/env python3
"""I17-1: re-registration of the I16-1 Part B test of the I15-2 selected
correction (``full|sym_e1_l1``, lambda 1 V) with the Q1 V(d,s) = 0 check at
tight tolerance only (threshold 1e-12 A kept).  Every other I16-1 rule,
procedure and threshold is unchanged; every stage is rerun into I17_1.

Pre-registration: out/research/explore/I17_1/prereg.md (prereg.json).

The stages are the I16-1 ones (scripts/research/explore/i16_1_sel.py, which
reuses the I14-1 / I15-1 drivers) with the output directory rebound to
I17_1 at import (spawned workers re-run the rebinding).  New: ``report``
writes the Part B summary under the I17-1 rule and the I16-1 rule label;
``decide`` applies the I17-1 outcome table.

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i17_1_vds0.py --stage validity --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i16_1_sel as drv16  # noqa: E402  (rebinds the I14-1 / I15-1 drivers to I16_1 and arm "sel")
from cryoml.explore import i11_1_deploy as m1  # noqa: E402
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i16_1_sel as m16  # noqa: E402
from cryoml.explore import i17_1_vds0 as m  # noqa: E402

drv14, drv15 = drv16.drv14, drv16.drv15
EXP = drv14.EXP
OUT = EXP / "I17_1"
PREREG = OUT / "prereg.json"
I16_1, I15_1, I15_2 = EXP / "I16_1", EXP / "I15_1", EXP / "I15_2"
ARM = drv16.ARM
jload, jdump, devices = drv14.jload, drv14.jdump, drv14.devices

# rebinding: every stage writes under I17_1
drv14.OUT = OUT
drv14.PREREG = PREREG
drv14.UNIT_TESTS = tuple(drv14.UNIT_TESTS) + ("tests.test_i17_1_vds0",)
drv15.OUT = OUT
drv16.OUT = OUT
drv16.PREREG = PREREG


def vds0_table(rd: Path) -> dict:
    q12 = jload(rd / "q12.json")["devices"]
    j0 = m.vds0_column(m1.GRID_VDS)
    per = {}
    for tag in sorted(q12):
        z = np.load(rd / "arrays" / f"q12_{tag}.npz")
        per[tag] = m.vds0_values(z, j0, m1.GRID_VGS)
    return {"per_device": per, "summary": m.summarize_vds0(per)}


def stage_report(rd: Path) -> dict:
    out = drv16.stage_report(rd)                  # I14-1 summary (I16-1 rule) -> summary.json
    vt = vds0_table(rd)
    dec = m.decide_deploy_i17(out["values"], vt["summary"])
    out["decision_I16_1_rule"] = out["decision"]
    out["decision"] = {k: dec[k] for k in ("deploy", "core_fail", "restrictions")}
    out["decision_I16_1_rule_recomputed"] = dec["under_I16_1_rule"]
    out["values_I17_1_rule"] = m.values_i17_rule(out["values"], vt["summary"])
    out["vds0"] = vt
    out["label"] = ("I17-1 Part B (I15-2 selected correction, full|sym_e1_l1, lambda 1 V; Q1 V(d,s) = 0 check "
                    "at tight tolerance only)")
    jdump(out, rd / "summary.json")
    print(json.dumps({"I17_1_rule": out["decision"], "I16_1_rule": out["decision_I16_1_rule"],
                      "vds0": vt["summary"]}, indent=1, default=float), flush=True)
    return out


def stage_decide() -> dict:
    val = jload(OUT / "validity" / "validity.json")
    acc = jload(OUT / "accuracy" / "accuracy.json")
    dep = jload(OUT / "deploy_sel" / "summary.json")
    rest = jload(OUT / "deploy_sel" / "restrictions.json")
    validity = m16.decide_validity({"v1_max_rrms_diff": val["v1_max_rrms_diff"], "v1_n_rows": val["v1_n_rows"],
                                    "v2_unit_tests_pass": bool(val["unit_tests_passed"]),
                                    "v3_n_rows": len(acc["native"]),
                                    "v3_m2_failed": dep["refit_in_sample"]["n_failed"],
                                    "v4_deployed_ok": bool(val["v4"]["ok"])})
    i16 = jload(I16_1 / "decision.json")
    i16_dep = jload(I16_1 / "deploy_sel" / "summary.json")
    diff = {k: [i16_dep["values"][k], dep["values"][k]] for k in dep["values"]
            if dep["values"][k] != i16_dep["values"].get(k)}
    out = {"prereg": str(PREREG.relative_to(ROOT)),
           "note": ("the V(d,s) = 0 check was re-registered after the observed I16-1 failure; the I16-1 outcome "
                    "stays 'not deployable'"),
           "validity": validity, "deploy_sel": dep["decision"],
           "deploy_sel_under_I16_1_rule": dep["decision_I16_1_rule"],
           **m.decide_outcome(validity, dep["decision"]),
           "values": dep["values"], "values_I17_1_rule": dep["values_I17_1_rule"], "vds0": dep["vds0"]["summary"],
           "thresholds": v2.THR, "runtime_grid": dep["runtime_grid"], "runtime_circuits": dep["runtime_circuits"],
           "refit_in_sample": dep["refit_in_sample"],
           "restriction_totals": {"sel": rest["sel"]["totals"],
                                  "ref_v2_I15_1": rest.get("ref_v2_I15_1", {}).get("totals")},
           "accuracy_I15_2_fixed": jload(I15_2 / "decision.json")["primary_gmr_selected_vs_v1"],
           "element_deck_accuracy": {k: acc.get(k) for k in (
               "summary_native", "summary_stored", "summary_reference", "gmr_native",
               "max_abs_native_minus_stored", "max_abs_python_rerun_minus_stored",
               "max_native_vs_channel_ratio", "n_msgs")},
           "I16_1": {"outcome": i16["outcome"], "deploy_sel": i16["deploy_sel"],
                     "values_differing_from_I17_1_run": diff,
                     "refit_in_sample": i16_dep["refit_in_sample"]}}
    jdump(out, OUT / "decision.json")
    print(json.dumps({k: out[k] for k in ("validity", "deploy_sel", "deploy_sel_under_I16_1_rule", "outcome",
                                          "action", "values", "vds0", "restriction_totals")},
                     indent=1, default=float), flush=True)
    print("values differing from I16-1:", json.dumps(diff, default=float), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("validity", "accuracy", "deploy", "compile", "refit", "q12", "q3", "report",
                             "restrict", "runtime", "decide"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="2 devices, refit max_nfev 20 (code check; not used for decisions)")
    args = ap.parse_args()
    base = OUT / "smoke" if args.smoke else OUT
    tags = drv16.SMOKE_TAGS if args.smoke else sorted(devices())
    rd = base / "deploy_sel"
    t0 = time.time()
    if args.stage == "validity":
        drv16.stage_validity(args.workers, base / "validity", tags)
    elif args.stage == "accuracy":
        drv16.stage_accuracy(args.workers, base / "accuracy", tags)
    elif args.stage == "decide":
        stage_decide()
    elif args.stage == "restrict":
        drv16.stage_restrict(rd, args.smoke)
    elif args.stage == "runtime":
        drv15.stage_runtime(rd, ARM)
    else:
        opts = dict(drv14.REFIT_OPTS, heldout_nfev=20) if args.smoke else dict(drv14.REFIT_OPTS)
        workers = min(args.workers, 4)
        stages = (("compile", "refit", "q12", "q3", "report") if args.stage == "deploy" else (args.stage,))
        timing = {}
        for st in stages:
            t1 = time.time()
            if st == "compile":
                drv14.stage_compile(rd, ARM)
            elif st == "refit":
                drv14.stage_refit(rd, ARM, tags, workers, opts)
            elif st == "q12":
                drv15.stage_q12(rd, ARM, tags, workers)
            elif st == "q3":
                if args.smoke:      # smoke: starting cards for circuit devices not refit
                    have = jload(rd / "cards_M2.json")
                    sc = drv14.start_cards()
                    for _, _, ts, _ in drv14.CIRCUITS:
                        for t in ts:
                            have.setdefault(t, sc[t])
                    jdump(have, rd / "cards_M2.json")
                drv14.stage_q3(rd, ARM, workers)
            else:
                stage_report(rd)
            timing[st] = round(time.time() - t1, 1)
        tf = rd / "timing.json"
        old = jload(tf) if tf.exists() else {}
        old.update(timing)
        jdump(old, tf)
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
