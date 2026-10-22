#!/usr/bin/env python3
"""I3-2: bias-only correction g(is_pmos, |Vgs|/VDD, |Vds|/VDD), held-out refit in the loop.

Pre-registration: ``out/research/explore/I3_2/prereg.json``.

Per fold, ``M_bias`` (3-16-16-1) and ``M_bias_geom`` (5-16-16-1) are trained
post-hoc on the E1 stored training records (``cryoml.explore.i3_2_bias``;
one pool task per fold, arm and seed); the held-out card is then refit from
its starting card with the frozen correction in the loop and scored with the
strict paper RRMS (the I2-1 held-out task, ``i2_1_joint.heldout_task``).

Modes:
  real    REAL_DEV leave-one-device-out.  C0_refit is read from the stored
          I2-1 rows and T1_inloop from the stored E1 stage-2 rows; both are
          re-run as reproduction checks (``--no-check`` skips).  Also trains a
          full M_bias on all 12 devices and writes it as Verilog-A text.
  pdk2    GF180 / IHP tier c sets 0-2, leave-one-geometry-out.  C0_refit read
          from I2-1 (re-run as a check); T1_inloop is a new held-out refit with
          the E1 leave-one-geometry-out fold T1 correction.
  real --devices all18
          all 18 sky130 77 K devices (REAL_DEV + REAL_TEST; the TEST devices
          were opened on 2026-10-21 by user decision), leave-one-device-out,
          same recipe.  C0_refit and T1_inloop are new runs (T1 per fold
          fitted on the other 17 records, e1_correction.fit_arm); C0_refit on
          the 12 REAL_DEV devices is compared with the stored I2-1 rows.
          Needs out/research/explore/E1/real_test.{npz,json}
          (e1_collect_real.py --split test).  Output out/research/explore/I3_2/all18/.
  decide  read real_dev / gf180_c_logo / ihp_c_logo results and evaluate R1-R4.
  export  write a stored correction as Verilog-A text.

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i3_2_bias.py \
      --mode real --name real_dev --workers 4
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i3_2_bias.py \
      --mode pdk2 --pdk gf180 --name gf180_c_logo --workers 6
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i3_2_bias.py --mode decide
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i3_2_bias.py \
      --mode real --devices all18 --name all18 --arms M_bias --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryoml.explore.e1_correction import Correction  # noqa: E402
from cryoml.explore.i2_1_joint import corrected_score  # noqa: E402
from cryoml.explore.i3_2_bias import (HYPER_BIAS, BiasCorrection,  # noqa: E402
                                      correction_from_json, eval_verilog_a_g, fit_bias,
                                      merge_seed_results, paired_ratio, split_seed_tasks,
                                      to_verilog_a_bias)

E1 = ROOT / "out/research/explore/E1"
I21 = ROOT / "out/research/explore/I2_1"
OUT = ROOT / "out/research/explore/I3_2"
PREREG = OUT / "prereg.json"
PREREG_OPTIONS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
ARMS = ("M_bias", "M_bias_geom")
T1_PLUS = 0.005
PDK_RATIO = 0.90


def train_task(task):
    arm, recs, hyper = task
    return fit_bias(arm, recs, hyper=hyper, threads=1)


def heldout(task):
    from i2_1_joint import heldout_task
    row, _ = heldout_task(task)
    return row


def aggregate(values: dict) -> dict:
    from e1_lodo import aggregate as agg
    return agg(values)


def load_records(mode: str, pdk: str):
    from i2_1_joint import load_records as lr
    return lr(argparse.Namespace(mode=mode, pdk=pdk))


def load_records_all18():
    """REAL_DEV + REAL_TEST starting-card records (direct43_r3_fd cards)."""
    from cryoml.explore.e1_correction import load_devices
    split = json.loads((ROOT / "out/research/explore/real_split.json").read_text())
    tags = sorted(split["REAL_DEV"] + split["REAL_TEST"])
    rec_dir = Path(os.environ.get("CRYOML_E1_RECORDS", str(E1)))   # I13-1: state-C records
    recs = load_devices(rec_dir / "real_dev") + load_devices(rec_dir / "real_test")
    if sorted(r.tag for r in recs) != tags or len(tags) != 18:
        raise RuntimeError("all18 records differ from REAL_DEV + REAL_TEST")
    cards = json.loads((ROOT / "out/research/polished/direct43_r3_fd/real/summary.json")
                       .read_text())["devices"]
    for r in recs:
        r.meta["params"] = dict(cards[r.tag]["params"])
    return recs, (lambda r: r.tag)


def t1_task(task):
    from cryoml.explore.e1_correction import fit_arm
    return fit_arm("T1", task, threads=1).to_json()


def stored_rows(real: bool, pdk: str) -> dict[str, dict]:
    """Stored per-device RRMS reused or reported (label -> {tag: rrms})."""
    if real:
        i21 = json.loads((I21 / "real_dev/results.json").read_text())["per_device"]
        e1 = json.loads((E1 / "real_dev_refit/results.json").read_text())["rows"]
        return {"C0_refit": i21["C0_refit"],
                "T1_inloop": {t: r["rrms"] for t, r in e1["T1"].items()},
                "ref_M_post_E1_stage2": {t: r["rrms"] for t, r in e1["M"].items()},
                "ref_M_joint_I2_1": i21["M_joint"], "ref_T1_joint_I2_1": i21["T1_joint"]}
    i21 = json.loads((I21 / f"{pdk}_c_logo/results.json").read_text())["per_device"]
    return {"C0_refit": i21["C0_refit"], "ref_M_post_I2_1": i21["M_post"],
            "ref_M_joint_I2_1": i21["M_joint"]}


def max_abs_diff(a: dict, b: dict) -> float:
    d = [abs(a[t] - b[t]) for t in a if t in b]
    return float(max(d)) if d else float("nan")


def export_va(corr: BiasCorrection, vdd: float, path: Path, n_check: int = 200,
              legacy: bool = False) -> dict:
    """Write the Verilog-A export and check its arithmetic.  ``legacy=True``
    writes the I3-2 stored form (``V(sense) <+ 0.0``, parameter ``pmos``)."""
    text = to_verilog_a_bias(corr, vdd=vdd, legacy=legacy)
    path.write_text(text)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n_check):
        pmos = float(rng.integers(0, 2))
        sgn = -1.0 if pmos else 1.0
        vgs, vds = sgn * rng.uniform(0, vdd), sgn * rng.uniform(0, vdd)
        L, W = 10 ** rng.uniform(-1, 2), 10 ** rng.uniform(-1, 2)
        phi = np.array([[pmos, abs(vgs) / vdd, abs(vds) / vdd, np.log10(L), np.log10(W), 0.0]])
        diffs.append(abs(eval_verilog_a_g(text, pmos, vgs, vds, L, W) - corr.g(phi)[0]))
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return {"path": str(shown), "lines": len(text.splitlines()),
            "max_abs_g_diff_va_vs_python": float(max(diffs))}


def decide() -> int:
    res = {}
    for name in ("real_dev", "gf180_c_logo", "ihp_c_logo"):
        p = OUT / name / "results.json"
        res[name] = json.loads(p.read_text()) if p.exists() else None
    missing = [k for k, v in res.items() if v is None or not v.get("preregistered_config")]
    if missing:
        print(f"not evaluated: missing or non-preregistered results {missing}")
        return 1
    real = res["real_dev"]
    s, cmp_ = real["summary"], real["comparisons"]
    r1 = cmp_["M_bias_vs_C0_refit"]
    checks = {
        "R1_real_ci_high_lt_1": bool(r1["ci_high"] < 1.0),
        "R2_real_mean_le_T1_inloop_plus_0.005": bool(
            s["M_bias"]["all_device_mean"] <= s["T1_inloop"]["all_device_mean"] + T1_PLUS),
        "R3_real_zero_failed": bool(s["M_bias"]["n_failed"] == 0)}
    r4 = {}
    for name, pdk in (("gf180_c_logo", "gf180"), ("ihp_c_logo", "ihp")):
        c = res[name]["comparisons"]["M_bias_vs_C0_refit"]
        r4[pdk] = {"ratio": c["ratio"], "n_failed": res[name]["summary"]["M_bias"]["n_failed"],
                   "pass": bool(c["ratio"] <= PDK_RATIO
                                and res[name]["summary"]["M_bias"]["n_failed"] == 0)}
    checks["R4_sim_ratio_le_0.90_on_gf180_or_ihp"] = any(v["pass"] for v in r4.values())
    out = {"prereg": str(PREREG.relative_to(ROOT)), "checks": checks, "R4_detail": r4,
           "R1_detail": r1, "R2_detail": {
               "M_bias_mean": s["M_bias"]["all_device_mean"],
               "T1_inloop_mean": s["T1_inloop"]["all_device_mean"]},
           "decision": "continue" if all(checks.values()) else "stop",
           "note": "REAL_DEV results are development evidence (prereg data_reuse_note)"}
    (OUT / "decision.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("real", "pdk2", "decide", "export"), required=True)
    ap.add_argument("--pdk", choices=("gf180", "ihp"), default="gf180")
    ap.add_argument("--name", default="")
    ap.add_argument("--devices", choices=("dev", "all18"), default="dev",
                    help="real: REAL_DEV (pre-registered I3-2) or all 18 devices")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--no-check", action="store_true",
                    help="skip the C0_refit / T1_inloop reproduction reruns (real)")
    ap.add_argument("--heldout-nfev", type=int, default=PREREG_OPTIONS["heldout_nfev"])
    ap.add_argument("--heldout-irls", type=int, default=PREREG_OPTIONS["heldout_irls"])
    ap.add_argument("--diff-step", type=float, default=PREREG_OPTIONS["diff_step"])
    ap.add_argument("--folds-limit", type=int, default=0, help="smoke: first N folds")
    ap.add_argument("--m-seeds", default="", help="smoke override of seeds")
    ap.add_argument("--m-steps", type=int, default=0, help="smoke override of steps")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default="", help="output directory (default I3_2/<name>)")
    ap.add_argument("--corrections", default="", help="export: corrections json")
    ap.add_argument("--key", default="full|M_bias", help="export: entry in corrections json")
    ap.add_argument("--vdd", type=float, default=1.85, help="export: VDD")
    ap.add_argument("--out", default="", help="export: output .va path")
    args = ap.parse_args()
    if args.mode == "decide":
        return decide()
    if args.mode == "export":
        corr = correction_from_json(json.loads(Path(args.corrections).read_text())[args.key])
        print(json.dumps(export_va(corr, args.vdd, Path(args.out).resolve())))
        return 0
    if not args.name:
        ap.error("--name is required")

    t_start = time.time()
    real = args.mode == "real"
    arms = [a for a in args.arms.split(",") if a]
    opts = {"heldout_nfev": args.heldout_nfev, "heldout_irls": args.heldout_irls,
            "diff_step": args.diff_step}
    hyper = {}
    if args.m_seeds:
        hyper["seeds"] = tuple(int(x) for x in args.m_seeds.split(","))
    if args.m_steps:
        hyper["steps"] = args.m_steps
    all18 = real and args.devices == "all18"
    preregistered = (opts == PREREG_OPTIONS and not hyper and not args.folds_limit
                     and set(ARMS) <= set(arms) and not all18)

    if all18:
        records, key = load_records_all18()
        g0 = None
    else:
        records, key, g0 = load_records(args.mode, args.pdk)
    groups = sorted({key(r) for r in records})
    if args.folds_limit:
        groups = groups[:args.folds_limit]
    held = {g: [r for r in records if key(r) == g] for g in groups}
    held_recs = [r for g in groups for r in held[g]]
    clusters = None if real else {r.tag: r.geometry for r in held_recs}
    print(f"I3_2 {args.mode} {'sky130' if real else args.pdk}: {len(groups)} folds, "
          f"{len(held_recs)} held-out records, arms {arms}, opts {opts}, hyper {hyper}, "
          f"preregistered={preregistered}", flush=True)

    train_tasks = [(arm, [r for r in records if key(r) != g]) for arm in arms for g in groups]
    train_names = [f"{arm}|{g}" for arm in arms for g in groups]
    if real:
        train_tasks.append(("M_bias", list(records)))
        train_names.append("full|M_bias")
    timing = {}
    with get_context("spawn").Pool(args.workers) as pool:
        t0 = time.time()
        if all18:
            t1s = pool.map(t1_task, [[r for r in records if key(r) != g] for g in groups])
            g0 = {g: {"T1": t} for g, t in zip(groups, t1s)}
            timing["t1_fit_s"] = round(time.time() - t0, 1)
            t0 = time.time()
        expanded, owner = split_seed_tasks(train_tasks, hyper)
        corrs = dict(zip(train_names, merge_seed_results(
            len(train_tasks), owner, pool.map(train_task, expanded, chunksize=1))))
        timing["train_s"] = round(time.time() - t0, 1)
        timing["train_n_seed_tasks"] = len(expanded)
        print(f"training done ({timing['train_s']} s, {len(expanded)} seed tasks)", flush=True)

        tasks = []
        for arm in arms:
            for g in groups:
                tasks += [(arm, None, corrs[f"{arm}|{g}"], rec, opts) for rec in held[g]]
        for g in groups:
            for rec in held[g]:
                t1 = Correction.from_json(g0[g]["T1"])
                if all18:
                    tasks += [("C0_refit", None, Correction("C0"), rec, opts),
                              ("T1_inloop", None, t1, rec, opts)]
                elif real:
                    if not args.no_check:
                        tasks += [("C0_refit_rerun", None, Correction("C0"), rec, opts),
                                  ("T1_inloop_rerun", None, t1, rec, opts)]
                else:
                    tasks.append(("T1_inloop", None, t1, rec, opts))
                    if not args.no_check:
                        tasks.append(("C0_refit_rerun", None, Correction("C0"), rec, opts))
        t0 = time.time()
        out = pool.map(heldout, tasks, chunksize=1)
        timing["heldout_refit_s"] = round(time.time() - t0, 1)
        timing["heldout_refit_n"] = len(tasks)

    rows = {}
    for row in out:
        rows.setdefault(row["label"], {})[row["tag"]] = row
    held_tags = [r.tag for r in held_recs]
    scores = {lab: {t: r["rrms"] for t, r in v.items()} for lab, v in rows.items()}
    if all18:
        stored = {}
        dev12 = stored_rows(True, args.pdk)["C0_refit"]
        repro_dev12 = {"C0_refit_vs_stored_I2_1_on_REAL_DEV": max_abs_diff(
            {t: v for t, v in scores["C0_refit"].items() if t in dev12}, dev12)}
    else:
        stored = {lab: {t: v[t] for t in held_tags}
                  for lab, v in stored_rows(real, args.pdk).items()}
        repro_dev12 = {}
    scores.update(stored)
    posthoc = {arm: {r.tag: corrected_score(corrs[f"{arm}|{key(r)}"], r) for r in held_recs}
               for arm in arms}
    posthoc["C0_start"] = {r.tag: corrected_score(Correction("C0"), r) for r in held_recs}

    summary = {lab: aggregate(v) for lab, v in scores.items()}
    pairs = [("M_bias", "C0_refit"), ("M_bias", "T1_inloop"), ("M_bias_geom", "C0_refit"),
             ("M_bias_geom", "M_bias"), ("T1_inloop", "C0_refit")]
    pairs += [("M_bias", lab) for lab in scores if lab.startswith("ref_")]
    comparisons = {f"{a}_vs_{b}": paired_ratio(scores[a], scores[b], clusters)
                   for a, b in pairs if a in scores and b in scores}
    checks_repro = {}
    for lab in ("C0_refit", "T1_inloop"):
        if f"{lab}_rerun" in scores:
            checks_repro[lab] = {"max_abs_diff_rerun_vs_stored":
                                 max_abs_diff(scores[f"{lab}_rerun"], scores[lab])}
    checks_repro.update(repro_dev12)

    result = {"name": args.name, "mode": args.mode, "pdk": "sky130" if real else args.pdk,
              "prereg": str(PREREG.relative_to(ROOT)), "preregistered_config": preregistered,
              "smoke": not preregistered, "options": opts, "hyper": HYPER_BIAS,
              "hyper_overrides": hyper, "arms": arms, "n_folds": len(groups),
              "n_heldout_records": len(held_recs),
              "devices": args.devices if real else None,
              "stored_sources": {} if all18 else {
                  "C0_refit": ("out/research/explore/I2_1/real_dev/results.json per_device.C0_refit"
                               if real else f"out/research/explore/I2_1/{args.pdk}_c_logo/"
                               "results.json per_device.C0_refit"),
                  "T1_inloop": ("out/research/explore/E1/real_dev_refit/results.json rows.T1"
                                if real else "new run (E1 leave-one-geometry-out fold T1)"),
                  "ref_rows": "reported only, not rerun"},
              "summary": summary, "comparisons": comparisons, "reproduction": checks_repro,
              "per_device": scores, "posthoc_no_refit": {
                  lab: {"summary": aggregate(v), "per_device": v} for lab, v in posthoc.items()},
              "timing": timing}
    if real and preregistered:
        r1 = comparisons["M_bias_vs_C0_refit"]
        result["criterion_real_part"] = {
            "R1_ci_high_lt_1": bool(r1["ci_high"] < 1.0),
            "R2_mean_le_T1_inloop_plus_0.005": bool(
                summary["M_bias"]["all_device_mean"]
                <= summary["T1_inloop"]["all_device_mean"] + T1_PLUS),
            "R3_zero_failed": bool(summary["M_bias"]["n_failed"] == 0),
            "note": "R4 needs the GF180/IHP runs; full decision: --mode decide"}
    elif not real and preregistered:
        c = comparisons["M_bias_vs_C0_refit"]
        result["criterion_R4_part"] = {
            "ratio": c["ratio"], "n_failed": summary["M_bias"]["n_failed"],
            "pass_this_pdk": bool(c["ratio"] <= PDK_RATIO
                                  and summary["M_bias"]["n_failed"] == 0)}
    target = Path(args.out_dir).resolve() if args.out_dir else OUT / args.name
    target.mkdir(parents=True, exist_ok=True)
    if all18:
        result["note"] = ("all 18 sky130 77 K devices; the 6 REAL_TEST devices were opened on "
                          "2026-10-21 by user decision; not pre-registered; I3-2 recipe unchanged")
        result["export"] = export_va(corrs["full|M_bias"], 1.85,
                                     target / "m_bias_sky130_all18_full.va")
    elif real:
        # legacy form: reproduces the stored I3-2 artifact; the corrected
        # export is m_bias_sky130_dev_full_sense_rs.va (--mode export)
        result["export"] = export_va(corrs["full|M_bias"], 1.85,
                                     target / "m_bias_sky130_dev_full.va", legacy=True)
    result["heldout_rows"] = {lab: {t: {k: v for k, v in r.items() if k != "params"}
                                    for t, r in v.items()} for lab, v in rows.items()}
    result["runtime_s"] = round(time.time() - t_start, 1)
    (target / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    (target / "corrections.json").write_text(json.dumps(
        {name: c.to_json() for name, c in corrs.items()}) + "\n")
    (target / "heldout_cards.json").write_text(json.dumps(
        {lab: {t: r["params"] for t, r in v.items()} for lab, v in rows.items()}) + "\n")
    print(json.dumps({"summary": {k: {kk: v[kk] for kk in ("all_device_mean", "combined",
                                                            "median", "n_failed")}
                                  for k, v in summary.items()},
                      "comparisons": {k: {kk: v.get(kk) for kk in ("ratio", "ci_low", "ci_high",
                                                                    "wins", "n_used")}
                                      for k, v in comparisons.items()},
                      "reproduction": checks_repro,
                      "criterion": result.get("criterion_real_part",
                                              result.get("criterion_R4_part")),
                      "export": result.get("export"), "timing": timing}, indent=1))
    print(f"runtime {result['runtime_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
