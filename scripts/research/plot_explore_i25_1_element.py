#!/usr/bin/env python3
"""Report batch for the I25-1 element checks (prereg section 10, both arms):
I17-1 Part B rules on the I25-1 full-data card + g pairs (arm A 7 parameters,
arm B 45 parameters; lambda*_full 5 V); the I17-1 run on the old 43p card + g
(lambda 1 V) as a reference column.

Reads stored files only (no NGSpice):

* ``out/research/explore/I25_1/element/arm{A,B}/decision.json``
* ``.../arm{A,B}/pairs/pairs.json``, ``deploy/{q12,q3,summary,restrictions,runtime,compile}.json``
* ``out/research/explore/I25_1/element/unit_tests.json``
* ``out/research/explore/I17_1/{decision.json, deploy_sel/summary.json}``

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i25_1_element.py --date 2026-10-28
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_figs as rf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "out/research/explore"
EL = EXP / "I25_1" / "element"
I17 = EXP / "I17_1"
TOPIC = "i25_1_element"
ARMS = (("A", "7p card + g"), ("B", "45p card + g"))


def jl(p: Path):
    return json.loads(Path(p).read_text())


def rel(p: Path) -> str:
    return str(Path(p).relative_to(ROOT))


def col(key, head, kind="rtext"):
    return {"key": key, "head": head, "md": head, "kind": kind, "group": None}


def e(x) -> str:
    x = float(x)
    if x != x:
        return "nan"
    return f"{x:.2e}" if (x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4)) else f"{x:.4g}"


CHECKS = (  # key, label, threshold, pass(value), type
    ("q1_tight_viol", "Q1 element vs I_card + (e^g - 1) I_rep, tight, both benches: points with "
                      "|dI| > 1e-6 |I_py| + 1e-13 A", "0", lambda v: v == 0, "core"),
    ("q1_bad", "Q1 element runs: convergence messages + non-finite values + failed runs", "0",
     lambda v: v == 0, "core"),
    ("q1_vds0_max", "Q1 max |I_elem(V(d,s) = 0)| (A), tight tolerance", "<= 1e-12", lambda v: v <= 1e-12, "core"),
    ("g_text_max", "Verilog-A g vs Python g, max |dg| (2000 points)", "<= 1e-9", lambda v: v <= 1e-9, "core"),
    ("g_sym_text_max", "Verilog-A g, exchanged vs forward terminals, max |dg|", "<= 1e-12",
     lambda v: v <= 1e-12, "core"),
    ("swap_asym_max", "d-s exchange, element-introduced max |d ln I| (|I| > 1 nA)", "<= 1e-3",
     lambda v: v <= 1e-3, "core"),
    ("q2_ac_max", "Q2 AC gm, gds vs element identity, max normalized difference", "<= 1e-3",
     lambda v: v <= 1e-3, "core"),
    ("cap_max", "Q2 capacitances (Y_dd, Y_dg, Y_gg) vs card, max relative error", "<= 1e-3",
     lambda v: v <= 1e-3, "core"),
    ("q2_gm_sign", "gm curves with a sign change introduced in strong inversion (of 54)", "0",
     lambda v: v == 0, "core"),
    ("q3_fail", "Q3 circuits failing only with the element (of 6)", "0", lambda v: v == 0, "core"),
    ("q1_default_max", "default tolerance, max relative difference (|I| > 1 nA)", "<= 1e-3",
     lambda v: v <= 1e-3, "restriction"),
    ("q2_gds_sign", "gds curves with a sign change introduced (of 90)", "0", lambda v: v == 0, "restriction"),
    ("runtime_ratio_max", "wall-time ratio element / card, max (DC grid and circuits)", "<= 2",
     lambda v: v <= 2.0, "restriction"),
)


def verdict(ok: bool, kind: str) -> str:
    return "pass" if ok else ("FAIL" if kind == "core" else "restriction")


def t_checks(d) -> rf.Table:
    rows = []
    for key, lab, thr, ok, kind in CHECKS:
        r = {"chk": lab, "kind": kind, "thr": thr}
        for a, _ in ARMS:
            v = d[a]["dec"]["values"][key]
            r[a] = f"{e(v)} ({verdict(ok(v), kind)})"
        v = d["i17"]["values_I17_1_rule"][key]
        r["old"] = f"{e(v)} ({verdict(ok(v), kind)})"
        rows.append(r)
    r = {"chk": "label (I17-1 rule)", "kind": "", "thr": ""}
    for a, _ in ARMS:
        r[a] = d[a]["dec"]["outcome"]
    r["old"] = d["i17"]["deploy_sel"]["deploy"]
    rows.append(r)
    r = {"chk": "reported, no rule: label under the I16-1 V(d,s) = 0 rule (max of default and tight)", "kind": "",
         "thr": ""}
    for a, _ in ARMS:
        r[a] = d[a]["dec"]["deploy_under_I16_1_rule"]["deploy"]
    r["old"] = d["i17"]["deploy_sel_under_I16_1_rule"]["deploy"]
    rows.append(r)
    notes = ["arm A: I25-1 7-parameter full-data card + g; arm B: 45-parameter full-data card + g; g = full-data "
             "g v2 (sym_v2 inputs, EPS 0.01 V, lambda*_full 5 V); cards = I25-1 in-sample refits with g in the loop "
             "(multiplier-1 template, state C). Old 43p + g: I17-1 run (lambda 1 V), out/research/explore/I17_1.",
             "Element: I(d, s) = I_card + (exp(g) - 1) I_rep, I_rep = sensed channel current of a charge-free card "
             "replica; Verilog-A sense branch V(sense) <+ RS * I(sense), RS = 0; OpenVAF 23.5, ngspice-41 OSDI build.",
             "core fail -> not deployable; restriction -> deployable with restrictions (I17-1 thresholds)."]
    return rf.Table("01_table_element_checks", "Element checks (I17-1 rules): I25-1 arms A and B, old 43p card + g",
                    [col("chk", "check", "text"), col("kind", "type", "text"), col("thr", "limit"),
                     col("A", "arm A, 7p + g"), col("B", "arm B, 45p + g"), col("old", "old 43p + g (I17-1)")],
                    rows, notes)


def t_validity(d) -> rf.Table:
    rows = []
    for a, lab in ARMS:
        V = d[a]["dec"]["validity"]
        P = d[a]["pairs"]
        v4 = V["v4"]
        rows += [
            {"arm": a, "c": "V1 card-only deck (OSDI build) x exp(g) vs stored I25-1 rows (18 full + 18 LODO)",
             "t": "<= 1e-9", "v": e(V["v1_max_rrms_diff"]) + f" ({V['v1_n_rows']} rows)",
             "p": "pass" if V["v1_max_rrms_diff"] <= 1e-9 and V["v1_n_rows"] == 36 else "FAIL"},
            {"arm": a, "c": "V2 unit tests (I17-1 list + test_i25_1_scratch)", "t": "all pass",
             "v": d["unit"]["tail"][-1] if d["unit"]["tail"] else "", "p": "pass" if d["unit"]["passed"] else "FAIL"},
            {"arm": a, "c": "V3 finite element-deck rows full / LODO; M2 cards; failed",
             "t": "18 / 18; 18; 0",
             "v": f"{P['per_kind']['full']['n_finite_native']} / {P['per_kind']['lodo']['n_finite_native']}; "
                  f"18; {d[a]['dec']['refit_in_sample']['n_failed']}",
             "p": "pass" if not any("V3" in f for f in V["fails"]) else "FAIL"},
            {"arm": a, "c": "V4 deployed g: inputs / EPS / lambda*_full (results, inner)", "t": "sym_v2 / 0.01 / 5, 5",
             "v": f"{v4['corr_inputs']} / {v4['corr_eps']} / {v4['lambda_star_full_results']:g}, "
                  f"{v4['lambda_star_full_inner']:g}", "p": "pass" if v4["ok"] else "FAIL"}]
    notes = [f"arm A: {'valid' if d['A']['dec']['validity']['valid'] else 'invalid'}; arm B: "
             f"{'valid' if d['B']['dec']['validity']['valid'] else 'invalid'}. Unit tests run with the default "
             "template; every NGSpice stage with CRYOML_PDK_TEMPLATE=mult1."]
    return rf.Table("02_table_element_validity", "Validity V1-V4 per arm",
                    [col("arm", "arm", "text"), col("c", "check", "text"), col("t", "criterion"), col("v", "value"),
                     col("p", "result")], rows, notes)


def agg(s) -> str:
    return f"{s['all_device']:.4f} / {s['combined']:.4f}"


def t_accuracy(d) -> rf.Table:
    rows = []
    for a, lab in ARMS:
        for k, kl in (("full", "full data (card + g fitted on 18)"), ("lodo", "LODO (g without the device)")):
            P = d[a]["pairs"]["per_kind"][k]
            rows.append({"row": f"arm {a}, {lab}, {kl}", "nat": agg(P["summary_native"]),
                         "py": agg(P["summary_stored"]), "ch": agg(P["summary_python_channel"]),
                         "d": e(P["max_abs_native_minus_stored"]), "msg": str(P["n_msgs"])})
    notes = ["RRMS: paper RRMS with the released card's curve set (frozen); values all-device / combined "
             "(combined = mean of the nMOS and pMOS family means).",
             "element deck: NGSpice with the OSDI element; stored Python: I25-1 row, NGSpice card current x exp(g); "
             "channel model: I_card + (exp(g) - 1) I_rep from the card + replica deck; d: max per-device "
             "|element - stored|."]
    return rf.Table("03_table_element_accuracy", "Element-deck RRMS vs stored Python rows",
                    [col("row", "pair set", "text"), col("nat", "element deck"), col("py", "stored Python"),
                     col("ch", "channel model"), col("d", "max |d| per device"), col("msg", "msgs")], rows, notes)


def t_devices(d, a, lab) -> rf.Table:
    Q = d[a]["q12"]["devices"]
    P = d[a]["pairs"]["rows"]
    rt = d[a]["summary"]["runtime_grid"]
    vd = d[a]["summary"]["vds0"]["per_device"]
    rows = []
    for t in sorted(Q):
        r = Q[t]
        msgs = sum(len(v) for k, v in r["msgs"].items() if k.endswith("_elem"))
        rows.append({"dev": t, "q1": f"{max(r['q1_tight_fwd_max_ratio'], r['q1_tight_swap_max_ratio']):.2f}",
                     "v0": e(vd[t]["tight"]["elem_max"]), "as": e(r["swap_asym_element_max"]),
                     "q2": e(max(r["q2_err_gm"], r["q2_err_gds"])), "cap": e(r["cap_max"]),
                     "def": e(r["q1_default_max_rel_1nA"]), "gm": str(sum(r["q2_gm_curves_introduced"])),
                     "gds": str(sum(r["q2_gds_curves_introduced"])), "msg": str(msgs), "rt": f"{rt[t]:.2f}",
                     "rr": f"{P[f'full|{t}']['stored']:.4f} / {P[f'full|{t}']['rrms_native']:.4f}"})
    notes = ["q1: max |dI| / (1e-6 |I_py| + 1e-13 A), tight, forward and exchanged (pass <= 1); Vds0: max "
             "|I_elem(V(d,s) = 0)| (A), tight; asym: element-introduced |d ln(I_swap / I_fwd)|; q2: max normalized "
             "gm / gds difference; cap: capacitance error; def: default-tolerance relative difference (|I| > 1 nA); "
             "gm: gm curves with an introduced sign change in strong inversion (of 3); gds: gds curves with an "
             "introduced sign change (of 5); msgs: convergence lines in element runs; runtime: DC grid wall-time "
             "ratio element / card; RRMS: stored Python / element deck, full-data pair."]
    return rf.Table(f"0{4 if a == 'A' else 5}_table_element_devices_arm{a}",
                    f"Element checks per device, arm {a} ({lab})",
                    [col("dev", "device", "text"), col("q1", "Q1 ratio"), col("v0", "Vds0 (A)"), col("as", "asym"),
                     col("q2", "Q2"), col("cap", "cap"), col("def", "default tol"), col("gm", "gm"),
                     col("gds", "gds"), col("msg", "msgs"), col("rt", "runtime"), col("rr", "RRMS py / element")],
                    rows, notes)


def t_vds0(d) -> rf.Table:
    rows = []
    for tol, rule in (("tight", "core check (I17-1)"), ("default", "reported"), ("r4", "reported")):
        r = {"tol": tol, "rule": rule}
        for a, _ in ARMS:
            s = d[a]["dec"]["vds0"][tol]
            r[a] = f"{s['elem_max']:.3e} at {s['at'][0]}, |Vgs| {s['at'][1]:.3g} V; {s['n_above']} of {s['n_points']}"
        s = d["i17"]["vds0"][tol]
        r["old"] = f"{s['elem_max']:.3e}; {s['n_above']} of {s['n_points']}"
        rows.append(r)
    notes = ["max |I_elem| at V(d,s) = 0 over 18 devices x 75 V(g,s) values; 'k of n' = points above 1e-12 A. "
             "Tolerances: default (ngspice abstol 1e-12 A, reltol 1e-3); r4 reltol 1e-4; tight reltol 1e-9, abstol "
             "1e-18 A, vntol 1e-12 V."]
    return rf.Table("06_table_element_vds0", "Current at V(d,s) = 0 per tolerance",
                    [col("tol", "tolerance", "text"), col("rule", "use", "text"), col("A", "arm A, 7p + g"),
                     col("B", "arm B, 45p + g"), col("old", "old 43p + g (I17-1)")], rows, notes)


def q3_metric(r) -> str:
    if not r.get("finite", False):
        return "non-finite"
    if "f_hz" in r:
        return f"{r['f_hz'] / 1e9:.3f} GHz"
    if "freq_hz" in r:
        return f"{r['freq_hz'] / 1e9:.3f} GHz"
    if "ratio_at_0p9" in r:
        return f"Iout/Iref(0.9 V) {r['ratio_at_0p9']:.3f}"
    if "v_m" in r:
        return f"Vm {r['v_m']:.3f} V"
    for k in ("vm", "v_switch", "vsw"):
        if k in r:
            return f"Vm {r[k]:.3f} V"
    return ""


def t_q3(d) -> rf.Table:
    rows = []
    names = []
    for k in d["A"]["q3"]["rows"]:
        n = k.split("|")[0]
        if n not in names:
            names.append(n)
    for n in names:
        r = {"c": n}
        for a, _ in ARMS:
            R = d[a]["q3"]["rows"]
            parts = []
            for arm_q3, lab in (("published", "mult1"), ("C0_refit", "card"), ("M2_card", "M2 card"),
                                ("M2_full", "M2 + elem")):
                x = R[f"{n}|{arm_q3}"]
                parts.append(f"{lab}: {'ok' if x['ok'] else 'fail'}" + (f" {q3_metric(x)}" if x["ok"] else ""))
            r[a] = "; ".join(parts)
        rows.append(r)
    notes = ["Q3 runs per circuit: multiplier-1 template card (mult1), I25-1 C0_refit card (card), M2 card = card fitted "
             "with g, without the element (M2 card), and with the element (M2 + elem). Core check: M2 + elem fails "
             "while M2 card passes. inv: inverter VTC (Vm = switching voltage); ro5: 5-stage ring oscillator; mirror: "
             "current mirror, Iref 10 uA."]
    return rf.Table("07_table_element_q3", "Q3 circuits (1.8 V supply)",
                    [col("c", "circuit", "text"), col("A", "arm A", "text"), col("B", "arm B", "text")], rows, notes)


def t_restrict(d) -> rf.Table:
    rows = []
    for a, lab in ARMS:
        s = d[a]["summary"]
        tot = d[a]["restrict"]["arm"]["totals"]
        per = d[a]["restrict"]["arm"]["per_device"]
        gds = [f"{t} Vgs {c['vgs']:g} V ({c['n_neg_elem']} pts)" for t, v in sorted(per.items())
               for c in v["gds"] if c["introduced"]]
        rt = d[a]["runtime"]
        qd = d[a]["dec"]["q1_default_max_per_device"]
        n_def = sum(1 for v in qd.values() if v > 1e-3)
        rows += [
            {"arm": a, "item": "default tolerance > 1e-3: devices; max",
             "v": f"{n_def} of 18; {e(s['values']['q1_default_max'])}"},
            {"arm": a, "item": "gds curves with an introduced sign change",
             "v": f"{tot['n_gds_curves_introduced']} of 90" + (": " + ", ".join(gds) if gds else "")},
            {"arm": a, "item": "negative gds points: element / card x exp(g) / card",
             "v": f"{tot['n_gds_neg_points_elem']} / {tot['n_gds_neg_points_python']} / {tot['n_gds_neg_points_card']}"},
            {"arm": a, "item": "runtime element / card: DC grid; circuits",
             "v": f"{min(s['runtime_grid'].values()):.2f}-{max(s['runtime_grid'].values()):.2f}; "
                  + ", ".join(f"{k} {v:.2f}" for k, v in s["runtime_circuits"].items())},
            {"arm": a, "item": "ro5_A wall time card + replica / card; element / card",
             "v": f"{rt['ratio_card_rep']:.2f}; {rt['ratio_elem']:.2f}"}]
    notes = ["Restriction items (I17-1): reported in the label, not a core failure. card x exp(g): Python model "
             "on the dense card sweep."]
    return rf.Table("08_table_element_restrictions", "Restriction items per arm",
                    [col("arm", "arm", "text"), col("item", "item", "text"), col("v", "value", "text")], rows, notes)


def t_attrib(d) -> rf.Table:
    """Core failures by device (reported, no rule; from the q12 arrays and the card-only diagnostic)."""
    rows = []
    for a, _ in ARMS:
        q = jl(EL / f"arm{a}/diag_q1_nmos_L100_W100.json")
        for b, lab in (("fwd", "forward"), ("swap", "exchanged")):
            x = q[b]
            if x["n_viol"]:
                rows.append({"arm": a, "dev": "nmos_L100_W100", "chk": f"Q1 equivalence, tight, {lab} bench",
                             "v": f"{x['n_viol']} points; |dI| <= {x['max_abs_dI_viol']:.2e} A (ratio <= "
                                  f"{x['max_ratio']:.2f}); |I| {x['I_range_viol'][0]:.1e}-{x['I_range_viol'][1]:.1e} A, "
                                  f"|Vgs| {x['vgs_abs_range'][0]:.3g}-{x['vgs_abs_range'][1]:.3g} V, |Vds| "
                                  f"{x['vds_abs_range'][0]:.3g}-{x['vds_abs_range'][1]:.3g} V"})
    gm = d["B"]["restrict"]["arm"]["per_device"]["nmos_L100_W100"]["gm"][0]
    rows.append({"arm": "B", "dev": "nmos_L100_W100", "chk": "gm sign change in strong inversion, |Vds| 0.05 V",
                 "v": f"negative gm points: element {gm['n_neg_elem_si']}, card x exp(g) {gm['n_neg_python_si']}, "
                      f"card {gm['n_neg_card_si']} (of {gm['n_points_si']})"})
    q = d["B"]["q12"]["devices"]["nmos_L20_W0p64"]
    nm = {k: len(v) for k, v in q["msgs"].items() if v}
    diag = jl(EL / "armB/diag_card_only_nmos_L20_W0p64.json")
    rows.append({"arm": "B", "dev": "nmos_L20_W0p64", "chk": "Q1 runs, V(d,s) = 0, Q1 equivalence (tight)",
                 "v": f"convergence lines: element tight grids {nm.get('grid_fwd_tight_elem', 0)} / "
                      f"{nm.get('grid_swap_tight_elem', 0)}, card + replica tight {nm.get('grid_fwd_tight_card_rep', 0)}; "
                      f"exchanged card + replica tight run timed out (300 s); every tight element grid point "
                      f"non-finite ({q['q1_nonfinite']})"})
    rows.append({"arm": "B", "dev": "nmos_L20_W0p64", "chk": "card only, no replica, no element (diagnostic)",
                 "v": "tight forward grid convergence lines: M2 card " + str(diag["M2|tight"]["msgs"])
                      + ", C0_refit card " + str(diag["C0_refit|tight"]["msgs"]) + ", multiplier-1 card "
                      + str(diag["template_mult1|tight"]["msgs"]) + "; default tolerance 0 / 0 / 0"})
    rows.append({"arm": "B", "dev": "nmos_L20_W0p64", "chk": "gm curve |Vds| 1.85 V, gds curve |Vgs| 1.85 V",
                 "v": "dense element sweep incomplete (1724 / 1372 of 1851 points); counted as introduced by the rule"})
    notes = ["Reported without a rule (the label follows the pre-registered checks). ratio = |dI| / (1e-6 |I| + "
             "1e-13 A). Old 43p card + g (I17-1): nmos_L100_W100 0 points, ratio <= 0.11."]
    return rf.Table("09_table_element_core_failures", "Core failures by device (diagnosis, reported)",
                    [col("arm", "arm", "text"), col("dev", "device", "text"), col("chk", "check", "text"),
                     col("v", "value", "text")], rows, notes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    d = {"unit": jl(EL / "unit_tests.json"), "i17": jl(I17 / "decision.json")}
    d["i17"]["vds0"] = jl(I17 / "deploy_sel/summary.json")["vds0"]["summary"]
    src = {}
    for a, _ in ARMS:
        b = EL / f"arm{a}"
        d[a] = {"dec": jl(b / "decision.json"), "pairs": jl(b / "pairs/pairs.json"),
                "q12": jl(b / "deploy/q12.json"), "q3": jl(b / "deploy/q3.json"),
                "summary": jl(b / "deploy/summary.json"), "restrict": jl(b / "deploy/restrictions.json"),
                "runtime": jl(b / "deploy/runtime.json")}
        src[a] = [rel(b / "decision.json"), rel(b / "pairs/pairs.json"), rel(b / "deploy/summary.json")]
    tables = [(t_checks(d), "Element checks vs I17-1 thresholds, both arms and the old 43p card + g",
               src["A"] + src["B"] + [rel(I17 / "decision.json")]),
              (t_validity(d), "Validity V1-V4 per arm", src["A"] + src["B"] + [rel(EL / "unit_tests.json")]),
              (t_accuracy(d), "Element-deck RRMS (all-device / combined) vs stored Python rows",
               [rel(EL / "armA/pairs/pairs.json"), rel(EL / "armB/pairs/pairs.json")]),
              (t_devices(d, "A", "7p card + g"), "Per-device checks, arm A", [rel(EL / "armA/deploy/q12.json")]),
              (t_devices(d, "B", "45p card + g"), "Per-device checks, arm B", [rel(EL / "armB/deploy/q12.json")]),
              (t_vds0(d), "V(d,s) = 0 current per tolerance", src["A"] + src["B"]),
              (t_q3(d), "Q3 circuits", [rel(EL / "armA/deploy/q3.json"), rel(EL / "armB/deploy/q3.json")]),
              (t_restrict(d), "Restriction items",
               [rel(EL / "armA/deploy/restrictions.json"), rel(EL / "armB/deploy/restrictions.json")]),
              (t_attrib(d), "Core failures by device (diagnosis, reported)",
               [rel(EL / "armA/diag_q1_nmos_L100_W100.json"), rel(EL / "armB/diag_q1_nmos_L100_W100.json"),
                rel(EL / "armB/diag_card_only_nmos_L20_W0p64.json"), rel(EL / "armB/deploy/q12.json")])]
    items = []
    for t, desc, s in tables:
        rf.render_table(t, folder / f"{t.name}.png")
        items.append((f"{t.name}.png", desc, s))
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i25_1_element.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    md += [rf.table_markdown(t) + "\n" for t, _, _ in tables]
    (folder / "tables.md").write_text("\n".join(md))
    lines = []
    for a, lab in ARMS:
        D = d[a]["dec"]
        v = D["values"]
        acc = D["element_deck_accuracy"]
        lines.append(f"arm {a} ({lab}): {D['outcome']}; core failures {', '.join(D.get('core_fail', [])) or 'none'}; "
                     f"restrictions {'; '.join(D.get('restrictions', [])) or 'none'} (default tolerance "
                     f"{v['q1_default_max']:.2e}, gds {v['q2_gds_sign']} of 90, runtime {v['runtime_ratio_max']:.2f}); "
                     f"element-deck RRMS full data {agg(acc['full']['summary_native'])} (stored "
                     f"{agg(acc['full']['summary_stored'])}), LODO {agg(acc['lodo']['summary_native'])}.")
    readme = [f"# Report batch {args.date} · {TOPIC}", "", f"Generated {now} by {gen}; stored files only.", "",
              "I25-1 element checks (prereg section 10; pre-registration "
              "`out/research/explore/I25_1/element/prereg.md`): I17-1 Part B rules on the I25-1 full-data card + g "
              "pairs of both arms.", ""] + [f"- {x}" for x in lines] + ["", "## Tables", ""]
    readme += [f"- `{p}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in s) + "." for p, desc, s in items]
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], lines)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
