#!/usr/bin/env python3
"""Report batch for the method-freeze draft (2026-10-23): one summary table
of the DEV evidence per measured dataset.

Per dataset: C0_refit, M_bias v1, selected g v2 (where run) and the I19-1
dataset-level gated method: all-device mean RRMS, combined RRMS (mean of the
nMOS and pMOS family means), GMR vs C0_refit with the 95% bootstrap CI
(device; TU Delft 7 geometry groups), GMR(selected, v1), the I19-1 decision
and the Part B deployment label.  Every value is computed from the stored
per-device rows and checked against the stored decision files.  No NGSpice
run, no training, no TEST data.

  .venv/bin/python -I scripts/research/plot_freeze_draft.py --date 2026-10-23
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import plot_explore_e2_e3_e4 as pe  # noqa: E402
from plot_explore_e2_e3_e4 import EXP, load, rel  # noqa: E402

rf = pe.rf

TOPIC = "freeze_draft"
NAMES = {"sky130_77K": "sky130 77 K (all 18, LODO)", "nist_4K": "NIST sky130 4 K lvt (I8-1)",
         "nist_rt_card_4K": "NIST sky130 4 K, RT card (I9-1)", "ihp_300K": "IHP SG13G2 300 K",
         "ihp_233K": "IHP SG13G2 233 K", "ihp_343K": "IHP SG13G2 343 K", "ihp_398K": "IHP SG13G2 398 K",
         "tudelft_4K": "TU Delft 4.13 K", "tudelft_295K": "TU Delft 295.0 K"}


def pd_rows(path: Path, arm: str) -> dict:
    return {k: float(v) if v is not None else float("nan") for k, v in load(path)["per_device"][arm].items()}


def fl(d: dict) -> dict:
    return {k: (float(v) if v is not None else float("nan")) for k, v in d.items()}


def datasets() -> dict:
    """c0 / v1 / sel rows per dataset, cluster map, deployment label, sources."""
    from cryoml.explore.i6_1_tudelft import group_of
    ds = {}
    i13 = EXP / "I13_1"
    i15 = load(EXP / "I15_2/decision.json")
    i17 = load(EXP / "I17_1/decision.json")
    ds["sky130_77K"] = {"c0": pd_rows(i13 / "lodo/results.json", "C0_refit"),
                        "v1": pd_rows(i13 / "all18/results.json", "M_bias"),
                        "sel": fl(i15["selected_rows"]), "unit": None,
                        "deploy": f"{i17['label']} (I17-1, lambda 1 V full-data model)",
                        "src": [rel(i13 / "lodo/results.json"), rel(i13 / "all18/results.json"),
                                rel(EXP / "I15_2/decision.json"), rel(EXP / "I17_1/decision.json")]}
    i20 = EXP / "I20_1"
    i21 = EXP / "I21_1"
    for name, d in (("nist_4K", "nist4"), ("nist_rt_card_4K", "nistrt")):
        p = i20 / f"base_{d}/results.json"
        q = i20 / f"lodo_{d}/results.json"
        pb = i21 / f"partA_{d}/summary.json"
        dep = "not tested (Part B not run for the NIST cards; v1 element not deployable, I11-1)"
        if pb.exists():
            dep = f"{load(pb)['label']} (I21-1, lambda 0 V full-data model; Q3 sky130 circuits on NIST devices)"
        ds[name] = {"c0": pd_rows(p, "C0_refit"), "v1": pd_rows(p, "M_bias"), "sel": fl(load(q)["per_device"]),
                    "unit": None, "i20": d, "deploy": dep,
                    "src": [rel(p), rel(q), rel(i20 / "decision.json")] + ([rel(pb)] if pb.exists() else [])}
    i17_2 = load(EXP / "I17_2/decision.json")
    ds["ihp_300K"] = {"c0": pd_rows(EXP / "I17_2/base/results.json", "C0_refit"),
                      "v1": pd_rows(EXP / "I5_1/lodo_300K/results.json", "M_bias"),
                      "sel": fl(i17_2["selected_rows"]), "unit": None,
                      "deploy": "Part B not run (PSP103 replica, I17-2)",
                      "src": [rel(EXP / "I17_2/base/results.json"), rel(EXP / "I5_1/lodo_300K/results.json"),
                              rel(EXP / "I17_2/decision.json")]}
    for t in (233, 343, 398):
        p = EXP / f"I5_1/lodo_{t}K/results.json"
        q = i21 / f"ihp_{t}K/decision.json"
        sel = fl(load(q)["rows"]["selected"]) if q.exists() else None
        ds[f"ihp_{t}K"] = {"c0": pd_rows(p, "C0_refit"), "v1": pd_rows(p, "M_bias"), "sel": sel, "unit": None,
                           "i19_arm": "v1", "i21": q if q.exists() else None,
                           "deploy": "Part B not run (PSP103 replica)", "src": [rel(p)] + ([rel(q)] if q.exists() else [])}
    i18 = load(EXP / "I18_2/decision.json")
    pb = load(EXP / "I18_2/partB_decision.json")
    for t in (4, 295):
        dj = i18["per_temperature"][str(t)]
        lab = pb["per_temperature"][str(t)]["label"]
        lam = pb["per_temperature"][str(t)]["lambda_star_full"]
        ds[f"tudelft_{t}K"] = {"c0": fl(dj["rows"]["C0_refit"]), "v1": fl(dj["rows"]["M_bias_v1"]),
                               "sel": fl(dj["selected_rows"]), "unit": "cluster",
                               "deploy": f"{lab} (I18-2, lambda {lam:g} V full-data model)",
                               "src": [rel(EXP / "I18_2/decision.json"), rel(EXP / "I18_2/partB_decision.json")]}
        ds[f"tudelft_{t}K"]["clusters"] = {k: group_of(k) for k in dj["rows"]["C0_refit"]}
    return ds


def paired_tags(*rows) -> list[str]:
    tags = set(rows[0])
    for r in rows[1:]:
        tags &= set(r)
    return sorted(t for t in tags if all(np.isfinite(r[t]) and r[t] > 0 for r in rows))


def is_p(tag: str) -> bool:
    return "pmos" in tag or "pfet" in tag


def means(rows: dict, tags: list[str]) -> dict:
    a = np.array([rows[t] for t in tags])
    n = np.array([rows[t] for t in tags if not is_p(t)])
    p = np.array([rows[t] for t in tags if is_p(t)])
    comb = float(np.mean([x.mean() for x in (n, p) if len(x)]))
    return {"all": float(a.mean()), "combined": comb, "n": len(tags)}


def check(a: float, b: float, what: str, tol: float = 1e-12) -> None:
    if not abs(a - b) <= tol:
        raise AssertionError(f"{what}: {a} vs {b}")


def ci(g: dict) -> str:
    return f"{g['ratio']:.3f} [{g['ci_low']:.3f}, {g['ci_high']:.3f}] {g['wins']}/{g['n_used']}"


V1_S3 = {"sky130_77K": "sky130_v1", "ihp_300K": "ihp300_v1", "ihp_233K": "ihp233_v1", "ihp_343K": "ihp343_v1",
         "ihp_398K": "ihp398_v1", "tudelft_4K": "tud4_v1", "tudelft_295K": "tud295_v1"}


def frozen_s3(name: str) -> tuple[float, str]:
    """v1-recipe S3_D used by the frozen-method rule (I20-1): NIST from the I20-1 state-C records, TU Delft
    from the I21-1 state-C records (I18-2 starting records), the other datasets from the I19-1 v1 files."""
    if name.startswith("tudelft") and (EXP / "I21_1/s3tud/s3tud.json").exists():
        t = name.split("_")[1][:-1]
        return float(load(EXP / "I21_1/s3tud/s3tud.json")["per_temperature"][t]["S3_D_state_C"]), "I21_1/s3tud/s3tud.json"
    if name.startswith("nist"):
        d = {"nist_4K": "nist4", "nist_rt_card_4K": "nistrt"}[name]
        return float(load(EXP / f"I20_1/s3/{d}_v1.json")["S3_D"]), f"I20_1/s3/{d}_v1.json"
    return float(load(EXP / f"I19_1/s3/{V1_S3[name]}.json")["S3_D"]), f"I19_1/s3/{V1_S3[name]}.json"


def compute() -> tuple[list[dict], int, dict]:
    from cryoml.explore.i3_2_bias import paired_ratio
    ds = datasets()
    D19 = load(EXP / "I19_1/decision.json")["part_R"]["datasets"]
    D20 = load(EXP / "I20_1/decision.json")["per_dataset"]
    i15 = load(EXP / "I15_2/decision.json")
    i17_2 = load(EXP / "I17_2/decision.json")
    i18 = load(EXP / "I18_2/decision.json")["per_temperature"]
    n_checks = 0
    out = []
    for name, d in ds.items():
        cl = d.get("clusters")
        x19 = D19[name]
        apply = bool(x19["apply"])
        nist = "i20" in d
        arm_rows = d["v1"] if (nist or d.get("i19_arm") == "v1") else (d["sel"] if d["sel"] is not None
                                                                         else d["v1"])   # I19-1 arm
        gated = {t: (arm_rows[t] if apply else d["c0"][t]) for t in d["c0"] if t in arm_rows}
        s3f, s3src = frozen_s3(name)
        apply_f = bool(s3f < 0.0)
        frozen_arm = d["sel"] if d["sel"] is not None else d["v1"]
        gated_f = {t: (frozen_arm[t] if apply_f else d["c0"][t]) for t in d["c0"] if t in frozen_arm}
        tags = paired_tags(d["c0"], d["v1"], gated, gated_f, *([d["sel"]] if d["sel"] is not None else []))
        sub = lambda r: {t: r[t] for t in tags}  # noqa: E731
        g_v1 = paired_ratio(sub(d["v1"]), sub(d["c0"]), clusters=cl)
        g_gate = paired_ratio(sub(gated), sub(d["c0"]), clusters=cl)
        g_gate_f = paired_ratio(sub(gated_f), sub(d["c0"]), clusters=cl)
        if nist:      # I19-1 decision (apply, M_bias v1) on the I20-1 state-C rows
            ref = D20[d["i20"]]["gmr_v1_vs_C0_refit_stateC"]
            for k in ("ratio", "ci_low", "ci_high"):
                check(g_gate[k], ref[k], f"{name} gated (I20-1 state C) {k}")
                check(g_v1[k], ref[k], f"{name} v1 (I20-1 state C) {k}")
        else:
            for k in ("ratio", "ci_low", "ci_high"):
                check(g_gate[k], x19["gmr_gated_D_vs_C0_refit"][k], f"{name} gated {k}")
        n_checks += 6 if nist else 3
        row = {"name": name, "n": len(tags), "unit": "7 groups" if cl else "device",
               "c0": means(d["c0"], tags), "v1": means(d["v1"], tags), "gated": means(gated, tags),
               "gated_f": means(gated_f, tags), "g_gate_f": g_gate_f, "S3_f": s3f,
               "decision_f": "apply" if apply_f else "C0_refit", "s3_f_src": s3src,
               "g_v1": g_v1, "g_gate": g_gate, "decision": x19["decision"], "S3_D": x19["S3_D"],
               "arm": x19["arm"], "deploy": d["deploy"], "src": d["src"]}
        if d["sel"] is not None:
            g_sel = paired_ratio(sub(d["sel"]), sub(d["c0"]), clusters=cl)
            g_sv1 = paired_ratio(sub(d["sel"]), sub(d["v1"]), clusters=cl)
            row.update(sel=means(d["sel"], tags), g_sel=g_sel, g_sv1=g_sv1)
            if nist:
                P = D20[d["i20"]]
                for k in ("ratio", "ci_low", "ci_high"):
                    check(g_sel[k], P["primary_gmr_selected_vs_C0_refit"][k], f"{name} selected {k}")
                    check(g_sv1[k], P["secondary_gmr_selected_vs_v1"][k], f"{name} selected vs v1 {k}")
                n_checks += 6
                out.append(row)
                continue
            if d.get("i21") is not None:      # IHP 233 / 343 / 398 K: I21-1 selected rows, I19-1 arm v1
                P = load(d["i21"])
                for k in ("ratio", "ci_low", "ci_high"):
                    check(g_sel[k], P["primary_gmr_selected_vs_C0_refit"][k], f"{name} selected {k}")
                    check(g_sv1[k], P["secondary_gmr_selected_vs_v1"][k], f"{name} selected vs v1 {k}")
                    check(g_v1[k], x19["gmr_correction_vs_C0_refit"][k], f"{name} v1 {k}")
                n_checks += 9
                out.append(row)
                continue
            for k in ("ratio", "ci_low", "ci_high"):
                check(g_sel[k], x19["gmr_correction_vs_C0_refit"][k], f"{name} selected {k}")
            n_checks += 3
            ref = {"sky130_77K": (i15["gmr_selected_vs_C0_refit"], i15["primary_gmr_selected_vs_v1"]),
                   "ihp_300K": (i17_2["primary_gmr_selected_vs_C0_refit"], i17_2["secondary_gmr_selected_vs_v1"]),
                   "tudelft_4K": (i18["4"]["primary_gmr_selected_vs_C0_refit"], i18["4"]["secondary_gmr_selected_vs_v1"]),
                   "tudelft_295K": (i18["295"]["primary_gmr_selected_vs_C0_refit"],
                                    i18["295"]["secondary_gmr_selected_vs_v1"])}[name]
            for k in ("ratio", "ci_low", "ci_high"):
                check(g_sel[k], ref[0][k], f"{name} stored selected {k}")
                check(g_sv1[k], ref[1][k], f"{name} stored selected vs v1 {k}")
            n_checks += 6
        else:
            for k in ("ratio", "ci_low", "ci_high"):
                check(g_v1[k], x19["gmr_correction_vs_C0_refit"][k], f"{name} v1 {k}")
            n_checks += 3
        if name.startswith("tudelft"):
            ref = i18[name.split("_")[1][:-1]]["gmr_v1_vs_C0_refit"]
            for k in ("ratio", "ci_low", "ci_high"):
                check(g_v1[k], ref[k], f"{name} stored v1 {k}")
            n_checks += 3
        if name == "ihp_300K":
            for k in ("ratio", "ci_low", "ci_high"):
                check(g_v1[k], i17_2["gmr_v1_vs_C0_refit"][k], f"{name} stored v1 {k}")
            n_checks += 3
        out.append(row)
    pooled = {k: float(np.exp(np.mean([math.log(r[k]["ratio"]) for r in out]))) for k in ("g_v1", "g_gate", "g_gate_f")}
    pooled["g_corr"] = float(np.exp(np.mean([math.log((r["g_sel"] if "g_sel" in r else r["g_v1"])["ratio"])
                                             for r in out])))
    return out, n_checks, pooled


def table(rows: list[dict], pooled: dict) -> rf.Table:
    def col(key, head, kind="rtext", group=None):
        return {"key": key, "head": head, "md": head if not group else f"{group}: {head}", "kind": kind,
                "group": group}

    def m(x):
        return f"{x['all']:.4f} / {x['combined']:.4f}"

    trs = []
    for r in rows:
        trs.append({"d": NAMES[r["name"]], "n": f"{r['n']} ({r['unit']})", "c0": m(r["c0"]), "v1": m(r["v1"]),
                    "sel": m(r["sel"]) if "sel" in r else "not run", "gt": m(r["gated"]),
                    "gv1": ci(r["g_v1"]), "gsel": ci(r["g_sel"]) if "g_sel" in r else "-",
                    "gsv1": ci(r["g_sv1"]) if "g_sv1" in r else "-",
                    "dec": f"{r['decision']} ({r['S3_D']:+.3f})", "gg": ci(r["g_gate"]), "dep": r["deploy"],
                    "decf": f"{r['decision_f']} ({r['S3_f']:+.3f})", "ggf": ci(r["g_gate_f"])})
    cols = [col("d", "dataset", "text"), col("n", "n (CI unit)"),
            col("c0", "C0_refit", group="RRMS all-device / combined"), col("v1", "M_bias v1", group="RRMS all-device / combined"),
            col("sel", "selected g v2", group="RRMS all-device / combined"),
            col("gt", "gated (I19-1)", group="RRMS all-device / combined"),
            col("gv1", "v1", group="GMR vs C0_refit [95% CI] lower/n"),
            col("gsel", "selected g v2", group="GMR vs C0_refit [95% CI] lower/n"),
            col("gg", "gated (I19-1)", group="GMR vs C0_refit [95% CI] lower/n"),
            col("ggf", "gated (frozen rule)", group="GMR vs C0_refit [95% CI] lower/n"),
            col("gsv1", "GMR(selected, v1)"), col("dec", "I19-1 decision (S3_D)"),
            col("decf", "frozen-rule decision (v1 S3_D)"),
            col("dep", "Part B deployment label", "text")]
    notes = ["DEV only; sky130 77 K is the all-18 LODO set (no locked TEST since the 2026-10-21 user decision). "
             "RRMS: strict paper RRMS of the held-out refit card (x exp(g) for correction arms); all-device = mean "
             "over the paired devices, combined = mean of the nMOS and pMOS family means. GMR = exp(mean ln ratio), "
             "percentile bootstrap 10000 resamples, seed 0; unit device, TU Delft 7 geometry groups. 'lower/n' = "
             "devices with a lower RRMS than the reference.",
             "gated (I19-1): committed I19-1 decisions and arms (apply iff S3_D < 0, S3_D with the gated arm's "
             "recipe); arm selected g v2 on sky130, IHP 300 K and TU Delft, M_bias v1 on NIST and IHP 233 / 343 / "
             "398 K. gated (frozen rule, pre-registered in I20-1): the v1-recipe S3_D on state-C records decides "
             "(TU Delft: I21-1 state-C value; IHP: PSP103, no clamp state), the arm is selected g v2 (run on all 9 "
             "rows after I21-1). Pooled exp(mean ln GMR vs C0_refit) "
             f"over the 9 rows: gated I19-1 {pooled['g_gate']:.3f}, gated frozen rule {pooled['g_gate_f']:.3f}, "
             f"correction always (selected g v2 where run, else v1) {pooled['g_corr']:.3f}, v1 {pooled['g_v1']:.3f}.",
             "sky130 v1 and selected rows: I13-1 / I15-2 state C (BSIM4 A2 clamp written in the card); TU Delft rows: "
             "I18-2 state C; NIST rows: I20-1 state C (C0_refit and M_bias v1 rerun, selected g v2 with 6-block "
             "lambda selection); IHP: PSP103 (no BSIM4 clamp); IHP 233 / 343 / 398 K selected g v2 rows: I21-1 "
             "(I17-2 setup); NIST Part B labels: I21-1."]
    return rf.Table("01_table_freeze_draft_dev_evidence",
                    "Method-freeze draft (proposed, not frozen): DEV evidence per measured dataset", cols, trs, notes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    rows, n_checks, pooled = compute()
    t = table(rows, pooled)
    png = f"{t.name}.png"
    rf.render_table(t, folder / png)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`.venv/bin/python -I scripts/research/plot_freeze_draft.py --date {args.date}`"
    (folder / "tables.md").write_text("\n".join([f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.",
                                                 "", rf.table_markdown(t), ""]))
    srcs = sorted({s for r in rows for s in r["src"]} | {rel(EXP / "I19_1/decision.json")}
                  | {"out/research/explore/" + r["s3_f_src"] for r in rows})
    summ = ["Method-freeze draft DEV evidence table (9 measured DEV datasets; NIST rows state C from I20-1): C0_refit, "
            f"M_bias v1, selected g v2, gated I19-1 and gated frozen rule; pooled GMR vs C0_refit gated I19-1 "
            f"{pooled['g_gate']:.3f}, frozen rule {pooled['g_gate_f']:.3f}, v1 {pooled['g_v1']:.3f}. "
            "Draft: docs/research/method_freeze_draft_2026-10-23.md."]
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen}. {n_checks} recomputed values checked against the stored decision files. "
              "No NGSpice runs, no training, no TEST data.", "", "## Contents", "", *[f"- {x}" for x in summ], "",
              "## Figures and tables", "",
              f"- `{png}`: DEV evidence per dataset; sources: " + ", ".join(f"`{s}`" for s in srcs) + ".",
              "- `tables.md`: the table in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    rf.update_index(args.date, TOPIC, folder, [png], summ)
    print(rf.table_markdown(t))
    print(f"{n_checks} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
