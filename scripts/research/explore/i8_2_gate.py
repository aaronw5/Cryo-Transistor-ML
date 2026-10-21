#!/usr/bin/env python3
"""I8-2: pre-registered TEST predictions of the I7-1 statistics, and the S3 /
S1_lodo gate (retrospective).

Modes:
  prereg  Part A.  For each of the 7 TEST datasets (sky130 REAL_TEST 77 K,
          IHP SG13G2 TEST at 233 / 300 / 343 / 398 K, TU Delft TEST at 4.13 /
          295 K): predicted sign of ln GMR(M_bias, C0_refit) from the stored
          DEV S1_lodo (frozen I7-1 rule) and from the stored DEV S3.  Secondary:
          S1_lodo computed on the training set that the TEST run uses (all DEV
          devices; IHP per class), from the stored DEV starting-card records.
          Writes out/research/explore/I8_2/prereg_test_predictions.{json,md}.
          Reads only DEV records, stored DEV statistics and the split tag lists;
          every loaded record tag is checked against the DEV and TEST lists.
  gate    Part B (retrospective; outcomes already seen).  Gated method G: on a
          dataset D apply M_bias if S3(D) < 0, else C0_refit (one decision per
          dataset from training-device statistics); G1: the same with the
          frozen I7-1 S1_lodo rule.  Evaluated on the 11 stored and the 20 I7-1
          simulated datasets from the stored per-device log ratios
          ln(RRMS_M_bias / RRMS_C0_refit); no new fits or simulations.
          Diagnostic: fold-level gates (decision per outer fold from that
          fold's training-set S3_f / S1_lodo_f, which exclude the scored
          device).  Writes out/research/explore/I8_2/gate_results.json.

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i8_2_gate.py --mode prereg
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i8_2_gate.py --mode gate
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryoml.explore.i7_1_transfer import rule_predict, stats_for  # noqa: E402
import i7_1_transfer as r71  # noqa: E402  (scripts/research/explore runner: specs, loaders)

EX = ROOT / "out/research/explore"
I7 = EX / "I7_1"
OUT = EX / "I8_2"
FILES_READ: list[str] = []


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def read_json(p: Path) -> dict:
    FILES_READ.append(rel(p))
    return json.loads(Path(p).read_text())


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def git_clean(paths: list[str]) -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return out.strip() == ""


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# TEST datasets and their DEV counterparts
# ---------------------------------------------------------------------------
TEST_DATASETS = [
    {"name": "sky130_77K", "dataset": "sky130 77 K REAL_TEST", "kind": "sky130", "temp_K": 77.0},
    *[{"name": f"ihp_meas_{t}K", "dataset": f"IHP SG13G2 TEST {t} K", "kind": "ihp_meas",
       "temp_K": float(t)} for t in (233, 300, 343, 398)],
    {"name": "tudelft_4K", "dataset": "TU Delft TEST 4.13 K", "kind": "tudelft", "temp_K": 4.13},
    {"name": "tudelft_295K", "dataset": "TU Delft TEST 295 K", "kind": "tudelft", "temp_K": 295.0},
]


def splits() -> dict:
    """DEV / TEST tag lists per dataset kind (tag lists only; no TEST data)."""
    real = read_json(EX / "real_split.json")
    ihp = read_json(EX / "I5_1/split.json")
    tud = read_json(EX / "I6_1/split.json")
    return {
        "sky130": {"DEV": real["REAL_DEV"], "TEST": real["REAL_TEST"], "file": "real_split.json"},
        "ihp_meas": {"DEV": ihp["DEV"], "TEST": ihp["TEST"], "TEST_by_class": ihp["TEST_by_class"],
                     "DEV_by_class": ihp["DEV_by_class"], "file": "I5_1/split.json"},
        "tudelft": {"DEV": tud["DEV"], "TEST": tud["TEST"], "TEST_groups": tud["TEST_groups"],
                    "file": "I6_1/split.json"},
    }


def scoring_protocol(sp: dict) -> dict:
    common = {
        "C0_refit": "held-out card refit of each TEST device without correction from its starting "
                    "card: ngspice_fit.polish, max_nfev 400, irls_rounds 2, diff_step 0.01 (I3-2 "
                    "settings), same parameter box as the dataset's DEV runs",
        "M_bias": "I3-2 recipe unchanged (cryoml.explore.i3_2_bias.fit_bias, 'M_bias'): "
                  "I = I_card * exp(g), g = tanh(h(x)), h = MLP 3-16-16-1 (tanh hidden, last "
                  "layer zero-initialized), x = (is_pmos 0/1, |Vgs|/VDD, |Vds|/VDD) with the two "
                  "bias inputs standardized by the training-point mean/std; loss = mean over "
                  "training devices of the paper per-device RRMS of the corrected current + "
                  "1e-4 * sum of squared weight-matrix entries; Adam lr 3e-3, 2000 full-batch "
                  "steps, float64, seeds 0-4, g = mean of the 5 members. Trained post hoc on the "
                  "starting-card records of the training set below, frozen; each TEST card is "
                  "refit with g in the loop (same polish settings as C0_refit), the best card "
                  "re-simulated, corrected with the same g and scored",
        "score": "strict paper RRMS per device (e1_correction.score_device: mean over the frozen "
                 "measured-inclusion curves of trimmed RMSE / trimmed mean |I_measured|)",
        "outcome": "y = mean over TEST devices of ln(RRMS_M_bias / RRMS_C0_refit); "
                   "GMR(M_bias, C0_refit) = exp(y) (i3_2_bias.paired_ratio; devices with a "
                   "non-finite or non-positive RRMS in either arm left out and counted). "
                   "Predicted sign correct iff sign(predicted y) = sign(y)",
        "code_status": "no TEST mode exists in the code at this commit; the protocol is the "
                       "TEST_policy / method-freeze text of the dataset's prereg, restated here",
    }
    return {
        "sky130": {**common,
                   "devices": sp["sky130"]["TEST"],
                   "source": "out/research/explore/real_split.json REAL_TEST (6 devices, 3 nMOS, "
                             "3 pMOS); I3-2 prereg data_reuse_note (REAL_TEST at the method "
                             "freeze, configuration fixed before loading)",
                   "starting_cards": "the 43-parameter direct-fit cards in "
                                     "out/research/polished/direct43_r3_fd/real/summary.json "
                                     "(as for REAL_DEV); records by scripts/research/explore/"
                                     "e1_collect_real.py --split test (one NGSpice run per device)",
                   "box": "round-3 43-parameter box (out/cold_loop/full_direct/round_3/spec.json, "
                          "cold_loop.box_for)",
                   "training_set": "all 12 REAL_DEV starting-card records "
                                   "(out/research/explore/E1/real_dev), both polarities, one "
                                   "M_bias ensemble"},
        "ihp_meas": {**common,
                     "devices": sp["ihp_meas"]["TEST"],
                     "source": "out/research/explore/I5_1/split.json TEST (10 DUTs: 6 LV, 4 HV); "
                               "I5-1 prereg data.split.TEST_policy",
                     "starting_cards": "per TEST DUT and temperature: ngspice_fit.polish from the "
                                       "stock card, max_nfev 3000, irls_rounds 3, diff_step 0.01, "
                                       "objective = mean included-curve RRMS (frozen measured "
                                       "inclusion), best card kept (I5-1 prereg cards)",
                     "box": "IHP_WIDE 12-parameter PSP box, i5_1_ihp_real.base_box",
                     "training_set": "per temperature and class: all DEV DUTs of the TEST DUT's "
                                     "class (LV 10, HV 8) at that temperature "
                                     "(out/research/explore/I5_1/start/records_<T>K); one M_bias "
                                     "ensemble per (temperature, class)"},
        "tudelft": {**common,
                    "devices": sp["tudelft"]["TEST"],
                    "source": "out/research/explore/I6_1/split.json TEST (24 devices in 3 W/L "
                              "groups: nMOS 1/0.36, nMOS 4/4, pMOS 4/4); I6-1 prereg "
                              "data.split.TEST_policy",
                    "starting_cards": "per TEST device and temperature: ngspice_fit.polish from "
                                      "the stock card, max_nfev 3000, irls_rounds 3, diff_step "
                                      "0.01, objective = mean included-curve RRMS (frozen "
                                      "measured inclusion), best card kept (I6-1 prereg cards)",
                    "box": "43-parameter sky130-set BSIM4 box around the 40 nm card, "
                           "i6_1_tudelft.base_box",
                    "training_set": "all 45 DEV devices at the temperature "
                                    "(out/research/explore/I6_1/start/records_<T>K), both "
                                    "polarities, one M_bias ensemble per temperature"},
    }


def check_dev_only(kind: str, recs: list, sp: dict) -> dict:
    tags = [r.tag for r in recs]
    dev, test = set(sp[kind]["DEV"]), set(sp[kind]["TEST"])
    in_test = sorted(set(tags) & test)
    not_dev = sorted(set(tags) - dev)
    if in_test or not_dev:
        raise RuntimeError(f"{kind}: TEST or non-DEV tags in loaded records: {in_test} {not_dev}")
    return {"n_records": len(tags), "all_in_DEV": True, "n_in_TEST": 0,
            "DEV_complete": sorted(set(tags)) == sorted(dev)}


def full_dev_s1(kind: str, recs: list, sp: dict) -> dict:
    """S1_lodo on the TEST run's training set(s) (prereg dataset_S weighting:
    held-out device count of the fold)."""
    if kind == "ihp_meas":
        parts = {}
        for cls in ("lv", "hv"):
            train = [r for r in recs if r.meta["cls"] == cls]
            parts[cls] = {**stats_for(train), "n_test": len(sp[kind]["TEST_by_class"][cls])}
        w = np.array([parts[c]["n_test"] for c in parts], float)
        s = np.array([parts[c]["S1_lodo"] for c in parts], float)
        return {"S1_lodo": float(np.sum(w * s) / w.sum()), "per_class": parts,
                "weighting": "per class S1_lodo of all DEV DUTs of the class, weighted by the "
                             "number of TEST DUTs of the class (LV 6, HV 4)"}
    st = stats_for(recs)
    return {"S1_lodo": st["S1_lodo"], "n_train": st["n_train"],
            "weighting": "one training set (all DEV devices)"}


def recompute_dataset_s(name: str, kind: str, recs: list, rows_stored: list,
                        s3_fold: dict) -> dict:
    """Recompute the stored dataset-level S1_lodo from the per-fold training sets
    and the dataset-level S3 from retro/s3.json (check of the stored values)."""
    folds = r71.folds_of(kind, recs)
    w = np.array([len(h) for _, h, _ in folds], float)
    s1 = np.array([stats_for(t)["S1_lodo"] for _, _, t in folds], float)
    s3 = np.array([s3_fold.get(f, np.nan) for f, _, _ in folds], float)
    ok = np.isfinite(s3)
    return {"S1_lodo": float(np.sum(w * s1) / w.sum()),
            "S3": float(np.sum(w[ok] * s3[ok]) / w[ok].sum()),
            "n_folds": len(folds), "fold_ids_match_stored":
                [f for f, _, _ in folds] == [r["fold"] for r in rows_stored]}


def prereg(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    head = git_head()
    me = Path(__file__).resolve()
    code_paths = [rel(me), "scripts/research/explore/i7_1_transfer.py",
                  "src/cryoml/explore/i7_1_transfer.py", "src/cryoml/explore/e1_correction.py"]
    clean = git_clean(code_paths)
    amend = read_json(I7 / "prereg_amendment_2026-10-20.json")["frozen_rule"]
    retro = read_json(I7 / "retro/results.json")
    s3_all = read_json(I7 / "retro/s3.json")["per_fold"]
    sp = splits()
    proto = scoring_protocol(sp)
    rows, checks, test_ok = [], {}, True
    for spec in TEST_DATASETS:
        name, kind = spec["name"], spec["kind"]
        stored = retro["datasets"][name]
        spec71 = next(s for s in r71.retro_specs() if s["name"] == name)
        FILES_READ.extend(rel(Path(str(p) + ext)) for p in spec71["records"]
                          for ext in (".npz", ".json"))
        k2, recs = r71.dataset_records(name)
        assert k2 == kind
        dev_check = check_dev_only(kind, recs, sp)
        rec = recompute_dataset_s(name, kind, recs, retro["per_fold"][name], s3_all[name])
        full = full_dev_s1(kind, recs, sp)
        s1, s3 = stored["S1_lodo"], stored["S3"]
        y_rule = rule_predict(amend, s1)
        y_full = rule_predict(amend, full["S1_lodo"])
        sign = lambda v: "gain" if v < 0 else "loss"  # noqa: E731
        row = {
            "name": name, "dataset": spec["dataset"], "temp_K": spec["temp_K"],
            "n_test_devices": len(sp[kind]["TEST"]), "n_dev_devices": len(sp[kind]["DEV"]),
            "S1_lodo_stored": s1, "predicted_y_rule": y_rule,
            "predicted_gmr_rule": math.exp(y_rule),
            "predicted_sign_S1": "negative (M_bias gain)" if y_rule < 0 else
                                 "positive (M_bias loss)",
            "S1_predicts": sign(y_rule),
            "S3_stored": s3, "predicted_sign_S3": "negative (M_bias gain)" if s3 < 0 else
                                                  "positive (M_bias loss)",
            "S3_predicts": sign(s3),
            "S1_and_S3_agree": sign(y_rule) == sign(s3),
            "secondary_S1_lodo_test_training_set": full["S1_lodo"],
            "secondary_predicted_y_rule": y_full,
            "secondary_S1_predicts": sign(y_full),
            "secondary_detail": full,
            "dev_dataset_outcome_already_seen": {"y_dev": stored["y"], "gmr_dev": stored["gmr"],
                                                 "note": "DEV leave-one-out outcome (I3-2 / "
                                                         "I5-1 / I6-1), known before this file"},
        }
        rows.append(row)
        checks[name] = {
            "records_DEV_only": dev_check,
            "S1_lodo_recomputed": rec["S1_lodo"], "S1_lodo_abs_diff": abs(rec["S1_lodo"] - s1),
            "S3_recomputed_from_s3_json": rec["S3"], "S3_abs_diff": abs(rec["S3"] - s3),
            "fold_ids_match_stored": rec["fold_ids_match_stored"]}
        test_ok &= dev_check["all_in_DEV"]
        if checks[name]["S1_lodo_abs_diff"] > 1e-9 or checks[name]["S3_abs_diff"] > 1e-12:
            raise RuntimeError(f"{name}: stored statistic not reproduced {checks[name]}")
    files_read = sorted(set(FILES_READ))
    test_markers = ("real_test", "REAL_TEST", "_test", "TEST")
    body = {
        "id": "I8_2_partA",
        "title": "Pre-registered predictions of the sign of ln GMR(M_bias, C0_refit) on the 7 "
                 "TEST datasets, from the DEV statistics S1_lodo (frozen I7-1 rule) and S3",
        "written": _dt.date.today().isoformat(),
        "code": {"git_head_at_run": head, "code_files": code_paths,
                 "code_files_clean_at_run": clean,
                 "script_sha256": sha256(me),
                 "command": "PYTHONPATH=src .venv/bin/python scripts/research/explore/"
                            "i8_2_gate.py --mode prereg"},
        "status": "written before any TEST curve is loaded, simulated or fitted. The final TEST "
                  "run is planned for about 2026-12-10",
        "test_data_not_opened": {
            "statement": "TEST data were not opened. The only TEST information used is the tag "
                         "lists in the split files (device names), which carry no measured "
                         "or simulated values",
            "verification": "every record loaded by this script is checked to be in the DEV "
                            "list and not in the TEST list of its split file (records_DEV_only "
                            "below); the record directories read are the I7-1 retro DEV "
                            "record paths; files_read lists every file this script opened",
            "files_read": files_read,
            "files_read_with_test_in_name": [f for f in files_read
                                             if any(m in f for m in test_markers)],
            "all_loaded_records_DEV": bool(test_ok)},
        "frozen_rule": {**amend, "source": "out/research/explore/I7_1/prereg_amendment_"
                                           "2026-10-20.json"},
        "S3_rule": "predicted gain (negative ln GMR) iff S3 < 0; S3 = stored dataset-level value "
                   "(retro/s3.json per-fold values, weighted by held-out devices per fold), "
                   "reduced M_bias recipe (1 seed, 500 steps)",
        "statistic_definition_used": "primary: the stored dataset-level DEV values of "
                                     "out/research/explore/I7_1/retro/results.json (mean over "
                                     "the DEV leave-one-out folds of the statistic on each fold's "
                                     "training devices; the definition the frozen rule was fit "
                                     "on). Secondary (reported, not the prediction): S1_lodo on "
                                     "the TEST run's training set (all DEV devices; IHP per "
                                     "class)",
        "how_TEST_is_scored": proto,
        "predictions": rows,
        "checks": checks,
        "evaluation_at_TEST": {
            "per_dataset": "sign of the TEST y against S1_predicts and S3_predicts; |TEST y - "
                           "predicted_y_rule|",
            "summary": "number of correct signs out of 7 for each statistic. The 4 IHP "
                       "temperatures share DUTs and the 2 TU Delft temperatures share devices; "
                       "the 7 outcomes are not independent (3 device sets)",
            "no_change": "these predictions, the rule and the protocol are not changed after "
                         "this file is committed; a change gets a dated amendment"},
        "dev_already_seen_note": "the DEV outcomes of all 7 datasets were known when the rule was "
                                 "fit (I7-1 step 2) and when this file was written; the "
                                 "predictions restate the DEV-calibrated rule for TEST",
    }
    (OUT / "prereg_test_predictions.json").write_text(json.dumps(body, indent=1) + "\n")
    write_md(body)
    for r in rows:
        print(f"{r['name']:14s} S1 {r['S1_lodo_stored']:+.4f} y_rule {r['predicted_y_rule']:+.4f} "
              f"({r['S1_predicts']})  S3 {r['S3_stored']:+.4f} ({r['S3_predicts']})  "
              f"S1_full {r['secondary_S1_lodo_test_training_set']:+.4f} "
              f"({r['secondary_S1_predicts']})")
    print("files read:", *files_read, sep="\n  ")
    return 0


def write_md(body: dict) -> None:
    lines = [
        "# I8-2 Part A: pre-registered TEST predictions",
        "",
        f"Written {body['written']}. Code: git HEAD `{body['code']['git_head_at_run']}` "
        f"(code files clean at run: {body['code']['code_files_clean_at_run']}), script "
        f"`scripts/research/explore/i8_2_gate.py --mode prereg` (sha256 "
        f"`{body['code']['script_sha256'][:16]}...`).",
        "",
        "TEST data were not opened. The script reads the DEV starting-card records, the stored "
        "I7-1 DEV statistics and the split tag lists; every loaded record tag is in the DEV list "
        "and not in the TEST list (`checks.*.records_DEV_only` in the json).",
        "",
        "Outcome: y = ln GMR(M_bias, C0_refit) over the TEST devices of a dataset (negative = "
        "M_bias lower RRMS). Frozen I7-1 rule: predicted y = "
        f"{body['frozen_rule']['a']:.6f} + ({body['frozen_rule']['b']:.6f}) * S1_lodo; gain iff "
        f"S1_lodo > {body['frozen_rule']['zero_crossing_S1_lodo']:.4f}. S3 rule: gain iff S3 < 0.",
        "",
        "| TEST dataset | n TEST | n DEV | S1_lodo (DEV) | rule y | rule GMR | S1 sign | S3 (DEV) "
        "| S3 sign | S1_lodo, TEST training set (secondary) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in body["predictions"]:
        lines.append(
            f"| {r['dataset']} | {r['n_test_devices']} | {r['n_dev_devices']} | "
            f"{r['S1_lodo_stored']:+.4f} | {r['predicted_y_rule']:+.4f} | "
            f"{r['predicted_gmr_rule']:.3f} | {r['S1_predicts']} | {r['S3_stored']:+.4f} | "
            f"{r['S3_predicts']} | {r['secondary_S1_lodo_test_training_set']:+.4f} "
            f"({r['secondary_S1_predicts']}) |")
    p = body["how_TEST_is_scored"]
    lines += [
        "",
        "## How TEST is scored",
        "",
        f"- C0_refit: {p['sky130']['C0_refit']}.",
        f"- M_bias: {p['sky130']['M_bias']}.",
        f"- Score: {p['sky130']['score']}.",
        f"- Outcome: {p['sky130']['outcome']}.",
        f"- sky130 77 K: {len(p['sky130']['devices'])} REAL_TEST devices; starting cards: "
        f"{p['sky130']['starting_cards']}; training set: {p['sky130']['training_set']}.",
        f"- IHP SG13G2: {len(p['ihp_meas']['devices'])} TEST DUTs per temperature; starting "
        f"cards: {p['ihp_meas']['starting_cards']}; training set: "
        f"{p['ihp_meas']['training_set']}.",
        f"- TU Delft: {len(p['tudelft']['devices'])} TEST devices per temperature; starting "
        f"cards: {p['tudelft']['starting_cards']}; training set: "
        f"{p['tudelft']['training_set']}.",
        f"- {p['sky130']['code_status']}.",
        "",
        "## Notes",
        "",
        "- The primary S values are the stored dataset-level DEV values (mean over the DEV "
        "leave-one-out folds), the definition the rule was fit on. The secondary column is "
        "S1_lodo on the TEST run's training set (all DEV devices; IHP: per class, weighted by "
        "TEST DUTs per class 6 LV / 4 HV); it gives the same sign as the primary value on all "
        "7 datasets." if all(r["S1_predicts"] == r["secondary_S1_predicts"]
                             for r in body["predictions"]) else
        "- The primary S values are the stored dataset-level DEV values; the secondary column "
        "is S1_lodo on the TEST run's training set. The two give different signs on at least "
        "one dataset; the primary value is the prediction.",
        "- The DEV outcomes of these datasets were known when the rule was fit (I7-1 step 2). "
        "The 4 IHP temperatures share DUTs and the 2 TU Delft temperatures share devices, so "
        "the 7 outcomes come from 3 device sets.",
        "- These predictions are not changed after this commit; a change gets a dated amendment.",
        "",
    ]
    (OUT / "prereg_test_predictions.md").write_text("\n".join(lines))


# ===========================================================================
# Part B: gate (retrospective)
# ===========================================================================
LOSS_TOL = 0.02          # "G loses to C0_refit by > 2 %": GMR(G, C0_refit) > 1.02
NEAR_S3 = 0.01           # post-hoc cut for datasets near the gate threshold (leakage check)


def _fold_rows_new(key: str, s3_new: dict) -> tuple[list[dict], dict]:
    pdk, rep, lam = key.split("|")
    path = r71.new_dir(pdk, int(rep), float(lam)) / "results.json"
    res = read_json(path)
    rows = []
    for r in res["per_fold"]:
        rows.append({"fold": r["fold"], "held": r["held"], "log_ratios": r["log_ratios"],
                     "S1_lodo": r["S1_lodo"], "S3": s3_new[f"new:{key}"].get(r["fold"], np.nan)})
    return rows, res


def _check_retro_ratios(name: str, rows: list[dict]) -> float:
    """Max |stored fold log ratio - ln(M_bias / C0_refit) from per-device RRMS|."""
    spec = next(s for s in r71.retro_specs() if s["name"] == name)
    FILES_READ.append(rel(spec["results"]))
    m, c0 = r71.load_outcomes(spec)
    diff = 0.0
    for r in rows:
        for t, v in zip(r["held"], r["log_ratios"]):
            a, b = m.get(t, np.nan), c0.get(t, np.nan)
            ref = math.log(a / b) if (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0) \
                else float("nan")
            if np.isfinite(ref) or np.isfinite(v):
                diff = max(diff, abs(ref - v))
    return diff


def _wmean(rows: list[dict], key: str) -> float:
    w = np.array([len(r["held"]) for r in rows], float)
    v = np.array([r[key] for r in rows], float)
    ok = np.isfinite(v)
    return float(np.sum(w[ok] * v[ok]) / np.sum(w[ok]))


def _gate_eval(rows: list[dict], choose_m) -> dict:
    """choose_m(row) -> True: M_bias for the held devices of that fold."""
    lg, n_m, n = [], 0, 0
    for r in rows:
        c = bool(choose_m(r))
        for v in r["log_ratios"]:
            if not np.isfinite(v):
                continue
            lg.append(v if c else 0.0)
            n_m += c
            n += 1
    lg = np.array(lg)
    return {"ln_gmr_vs_C0": float(lg.mean()), "n_devices": n, "n_devices_M_bias": int(n_m)}


def gate(args) -> int:
    amend = read_json(I7 / "prereg_amendment_2026-10-20.json")["frozen_rule"]
    retro = read_json(I7 / "retro/results.json")
    dec = read_json(I7 / "decision.json")
    s3_new = read_json(I7 / "new/s3.json")["per_fold"]
    datasets = []
    for name, st in retro["datasets"].items():
        rows = retro["per_fold"][name]
        diff = _check_retro_ratios(name, rows)
        datasets.append({"name": name, "group": "stored", "label": name, "rows": rows,
                         "S3": st["S3"], "S1_lodo": st["S1_lodo"], "y": st["y"],
                         "check_log_ratio_max_abs_diff": diff})
    for key, st in dec["datasets"].items():
        rows, res = _fold_rows_new(key, s3_new)
        datasets.append({"name": key, "group": "new", "label": key, "rows": rows, "S3": st["S3"],
                         "S1_lodo": st["S1_lodo"], "y": st["y"],
                         "check_log_ratio_max_abs_diff": abs(math.log(
                             res["M_bias"]["vs_C0_refit"]["ratio"]) - st["y"])})
    out_ds = {}
    for d in datasets:
        rows = d["rows"]
        lr = np.array([v for r in rows for v in r["log_ratios"]], float)
        y = float(np.nanmean(lr))
        s3c, s1c = _wmean(rows, "S3"), _wmean(rows, "S1_lodo")
        chk = {"y_abs_diff": abs(y - d["y"]), "S3_abs_diff": abs(s3c - d["S3"]),
               "S1_lodo_abs_diff": abs(s1c - d["S1_lodo"]),
               "log_ratio_vs_per_device_max_abs_diff": d["check_log_ratio_max_abs_diff"]}
        if max(chk.values()) > 1e-9:
            raise RuntimeError(f"{d['name']}: stored values not reproduced {chk}")
        rule_y = rule_predict(amend, d["S1_lodo"])
        g_m, g1_m = d["S3"] < 0, rule_y < 0
        ln_g = y if g_m else 0.0
        ln_g1 = y if g1_m else 0.0
        fold_g = _gate_eval(rows, lambda r: r["S3"] < 0)
        fold_g1 = _gate_eval(rows, lambda r: rule_predict(amend, r["S1_lodo"]) < 0)
        n_fold_dis_g = sum(1 for r in rows if (r["S3"] < 0) != g_m)
        n_fold_dis_g1 = sum(1 for r in rows if (rule_predict(amend, r["S1_lodo"]) < 0) != g1_m)
        s3f = np.array([r["S3"] for r in rows], float)
        out_ds[d["name"]] = {
            "group": d["group"], "n_folds": len(rows), "n_devices": int(np.isfinite(lr).sum()),
            "S3": d["S3"], "S1_lodo": d["S1_lodo"], "rule_y": rule_y,
            "y_M_bias_vs_C0": y, "gmr_M_bias_vs_C0": math.exp(y),
            "G": {"choice": "M_bias" if g_m else "C0_refit", "ln_gmr_vs_C0": ln_g,
                  "gmr_vs_C0": math.exp(ln_g), "ln_gmr_vs_M_bias": ln_g - y,
                  "gmr_vs_M_bias": math.exp(ln_g - y)},
            "G1": {"choice": "M_bias" if g1_m else "C0_refit", "ln_gmr_vs_C0": ln_g1,
                   "gmr_vs_C0": math.exp(ln_g1), "ln_gmr_vs_M_bias": ln_g1 - y,
                   "gmr_vs_M_bias": math.exp(ln_g1 - y)},
            "oracle_reference": {"ln_gmr_vs_C0": min(y, 0.0),
                                 "note": "dataset-level choice with the outcome known; reference "
                                         "only, not a method"},
            "diagnostic_fold_gate_S3": {**fold_g, "gmr_vs_C0": math.exp(fold_g["ln_gmr_vs_C0"]),
                                        "n_folds_decision_differs_from_dataset_gate": n_fold_dis_g},
            "diagnostic_fold_gate_S1": {**fold_g1, "gmr_vs_C0": math.exp(fold_g1["ln_gmr_vs_C0"]),
                                        "n_folds_decision_differs_from_dataset_gate":
                                            n_fold_dis_g1},
            "S3_fold_min": float(np.nanmin(s3f)), "S3_fold_max": float(np.nanmax(s3f)),
            "checks": chk}

    def pooled(names: list[str]) -> dict:
        out = {"n_datasets": len(names)}
        for arm in ("G", "G1", "M_bias", "C0_refit", "oracle_reference",
                    "diagnostic_fold_gate_S3", "diagnostic_fold_gate_S1"):
            if arm == "M_bias":
                lv = [out_ds[n]["y_M_bias_vs_C0"] for n in names]
            elif arm == "C0_refit":
                lv = [0.0 for n in names]
            else:
                lv = [out_ds[n][arm]["ln_gmr_vs_C0"] for n in names]
            ys = [out_ds[n]["y_M_bias_vs_C0"] for n in names]
            lm = [a - y for a, y in zip(lv, ys)]
            iw = int(np.argmax(lv))
            iwm = int(np.argmax(lm))
            out[arm] = {
                "mean_ln_gmr_vs_C0": float(np.mean(lv)), "gmr_of_mean_vs_C0": math.exp(np.mean(lv)),
                "worst_vs_C0": {"dataset": names[iw], "gmr": math.exp(lv[iw])},
                "n_lose_to_C0_gt_2pct": int(sum(math.exp(v) > 1 + LOSS_TOL for v in lv)),
                "mean_ln_gmr_vs_M_bias": float(np.mean(lm)),
                "worst_vs_M_bias": {"dataset": names[iwm], "gmr": math.exp(lm[iwm])},
                "n_lose_to_M_bias_gt_2pct": int(sum(math.exp(v) > 1 + LOSS_TOL for v in lm))}
            if arm in ("G", "G1"):
                out[arm]["n_choose_M_bias"] = int(sum(out_ds[n][arm]["choice"] == "M_bias"
                                                      for n in names))
                out[arm]["n_correct_choice"] = int(sum(
                    (out_ds[n][arm]["choice"] == "M_bias") == (out_ds[n]["y_M_bias_vs_C0"] < 0)
                    for n in names))
        return out

    stored = [n for n in out_ds if out_ds[n]["group"] == "stored"]
    new = [n for n in out_ds if out_ds[n]["group"] == "new"]
    measured = [n for n in stored if n.startswith(("sky130", "ihp_meas", "tudelft"))]
    body = {
        "id": "I8_2_partB",
        "label": "retrospective: every outcome used here was known before the gate was defined",
        "written": _dt.date.today().isoformat(),
        "code": {"git_head_at_run": git_head(), "script_sha256": sha256(Path(__file__).resolve()),
                 "command": "PYTHONPATH=src .venv/bin/python scripts/research/explore/"
                            "i8_2_gate.py --mode gate"},
        "definitions": {
            "G": "on dataset D: M_bias for every held-out device if S3(D) < 0, else C0_refit; "
                 "S3(D) = stored dataset-level S3 (mean over outer folds of the inner "
                 "leave-one-group-out post-hoc gain on each fold's training devices, weighted by "
                 "held-out devices per fold)",
            "G1": "the same with the frozen I7-1 rule: M_bias iff 0.007619 - 0.485184 * S1_lodo(D) "
                  "< 0 (S1_lodo(D) > 0.0157)",
            "ln_gmr": "ln GMR(G, C0_refit) = mean over held-out devices of ln(RRMS_G / "
                      "RRMS_C0_refit) = y_D if G chooses M_bias, else 0; ln GMR(G, M_bias) = "
                      "ln GMR(G, C0_refit) - y_D",
            "pooled": "mean over datasets (equal weight) of ln GMR; worst dataset = largest GMR; "
                      "n_lose_to_C0_gt_2pct = datasets with GMR(G, C0_refit) > 1.02",
            "diagnostic_fold_gate": "per outer fold: M_bias for the fold's held-out devices if the "
                                    "fold's training-set S3_f < 0 (resp. frozen rule on S1_lodo_f); "
                                    "these statistics exclude the scored devices",
            "oracle_reference": "min(y_D, 0): the dataset-level choice made with the outcome; a "
                                "reference bound, not a method",
            "not_best_of": "G and G1 make one choice per dataset from training-device statistics; "
                           "no per-device selection by outcome",
        },
        "leakage": {
            "where": "S3(D) and S1_lodo(D) are means over the outer folds of statistics computed "
                     "on each fold's training devices. A device d is in the training set of "
                     "every fold except its own, so d's starting-card record (its measured or "
                     "truth I-V and its starting-card simulation) enters S3(D) through the other "
                     "folds, and G's decision for d uses d's own record. The scored outcome of d "
                     "(held-out refit with and without M_bias) is not used by S3 or S1_lodo",
            "size": "d contributes to about (n_folds - 1) of n_folds fold values with weight "
                    "about 1 / n_train each, i.e. about 1 / n_devices of S3(D) (1/12 sky130, 1/9 "
                    "or 1/10 IHP measured within a class, 1/45 TU Delft, 1/48 or 1/16 simulated). "
                    "The leak can change d's decision only if S3(D) is within that share of 0",
            "check": "the fold-level gate (diagnostic_fold_gate_S3) uses S3_f of d's own fold, "
                     "which excludes d; where every fold's decision equals the dataset decision "
                     "the leak has no effect on the scored values",
            "at_TEST": "on TEST the gate uses S3 of the DEV devices only and the TEST devices are "
                       "not in it: no leakage of this kind",
        },
        "stored_S3_note": "S3 on the 11 stored datasets was computed after their outcomes were "
                          "known (I7-1 S3 job); the S3 sign agreement 9/11 and 19/20 was seen "
                          "before G was defined. The threshold 0 is the sign of the pre-registered "
                          "S3, not a fitted value. G1's rule was fit on the 11 stored datasets "
                          "(in-sample there) and frozen before the 20 new datasets were generated",
        "datasets": out_ds,
        "pooled": {"stored_11": pooled(stored), "measured_7": pooled(measured),
                   "new_20": pooled(new), "all_31": pooled(stored + new)},
        "files_read": sorted(set(FILES_READ)),
    }
    lk = {}
    for grp, names in (("stored_11", stored), ("new_20", new)):
        dif = [n for n in names if out_ds[n]["diagnostic_fold_gate_S3"][
            "n_folds_decision_differs_from_dataset_gate"] > 0]
        lk[grp] = {"datasets_with_any_fold_decision_different": dif,
                   "max_abs_ln_gmr_difference_dataset_vs_fold_gate": float(max(
                       abs(out_ds[n]["G"]["ln_gmr_vs_C0"]
                           - out_ds[n]["diagnostic_fold_gate_S3"]["ln_gmr_vs_C0"])
                       for n in names))}
    near = {n: {"S3": out_ds[n]["S3"], "abs_y": abs(out_ds[n]["y_M_bias_vs_C0"]),
                "n_devices": out_ds[n]["n_devices"]}
            for n in out_ds if abs(out_ds[n]["S3"]) < NEAR_S3}
    lk["near_threshold"] = {
        "definition": f"datasets with |S3(D)| < {NEAR_S3} (cut chosen post hoc): the datasets "
                      "whose decision is closest to the threshold; per-device inner values are "
                      "not stored, so the exact share of one device is not computed. If the leak "
                      "flipped the decision of a dataset, ln GMR(G, C0_refit) of the dataset changes by at most |y_D|",
        "datasets": near,
        "max_abs_y": float(max(v["abs_y"] for v in near.values())) if near else 0.0}
    lk["note"] = ("the fold-level gate differs from G by leakage and by the fold-level variability "
                  "of S3_f (S3_f uses one group fewer, and within a dataset S3_f and y_f have the "
                  "opposite rank relation, I7-1); its difference from G is not a pure leakage "
                  "estimate")
    body["leakage"]["observed"] = lk
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_results.json").write_text(json.dumps(body, indent=1) + "\n")
    for n, v in out_ds.items():
        print(f"{n:16s} S3 {v['S3']:+.4f} S1 {v['S1_lodo']:+.4f} GMR(M,C0) {v['gmr_M_bias_vs_C0']:.3f}"
              f"  G {v['G']['choice']:8s} {v['G']['gmr_vs_C0']:.3f}/{v['G']['gmr_vs_M_bias']:.3f}"
              f"  G1 {v['G1']['choice']:8s} {v['G1']['gmr_vs_C0']:.3f}/{v['G1']['gmr_vs_M_bias']:.3f}"
              f"  foldG {v['diagnostic_fold_gate_S3']['gmr_vs_C0']:.3f}"
              f" ({v['diagnostic_fold_gate_S3']['n_folds_decision_differs_from_dataset_gate']})")
    for g, p in body["pooled"].items():
        print(g, json.dumps({a: (round(x["gmr_of_mean_vs_C0"], 4), x["worst_vs_C0"]["gmr"].__round__(4),
                                 x["n_lose_to_C0_gt_2pct"]) for a, x in p.items() if isinstance(x, dict)}))
    print(json.dumps(lk, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("prereg", "gate"), required=True)
    args = ap.parse_args()
    return {"prereg": prereg, "gate": gate}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
