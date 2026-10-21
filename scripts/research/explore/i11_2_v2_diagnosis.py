#!/usr/bin/env python3
"""I11-2 post hoc diagnosis of validity check V2 (after the decision; does not change it).

V2 (prereg.json): "M_bias_near rows at 233 / 343 K equal the stored I5-1 X300
rows: max relative difference <= 1e-9 over the 36 records."  ``run_decide``
compared every M_bias_near row whose key exists in the stored X300 rows; the
stored X300 rows exist at 233, 343 and 398 K, so 54 rows entered the check.

This script writes ``out/research/explore/I11_2/v2_diagnosis.json``:

* per held-out temperature: the stored I5-1 fold correction used by
  M_bias_near (compared with ``lodo_ihp/corrections.json.gz``), the number of
  M_bias_near rows equal to the stored X300 rows and the max relative difference;
* V2 as stated in prereg.json (233 / 343 K, 36 records);
* two single-record refits for the record with the largest difference
  (held-out settings, ``refit_task``): with the stored I5-1 lodo_343K fold
  correction (M_bias_near definition at 398 K) and with the lodo_300K fold
  correction (X300 definition).

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i11_2_v2_diagnosis.py
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("i11_2_script", ROOT / "scripts/research/explore/i11_2_temp.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def main() -> int:
    t_start = time.time()
    near = json.loads((s.OUT / "lodo_ihp/results.json").read_text())["per_record"]["M_bias_near"]
    with gzip.open(s.OUT / "lodo_ihp/corrections.json.gz", "rt") as f:
        used = {kk: v for kk, v in json.load(f).items() if kk.startswith("M_bias_near|")}
    stored = {t: json.loads((s.I5 / f"lodo_{t}K/corrections.json").read_text()) for t in (300, 343)}
    x300 = s.stored_x300()
    per_t = {}
    for t in s.IHP_TEMPS:
        keys = sorted(kk for kk in near if s.unkey(kk)[1] == t)
        src = {}
        for kk in keys:
            tag = s.unkey(kk)[0]
            m = [ts for ts in (300, 343) if stored[ts].get(f"M_bias|{tag}") == used[f"M_bias_near|{kk}"]]
            src[str(m[0]) if len(m) == 1 else "none"] = src.get(str(m[0]) if len(m) == 1 else "none", 0) + 1
        d = [abs(near[kk] - x300[kk]) / x300[kk] for kk in keys if kk in x300]
        per_t[str(t)] = {"n_records": len(keys), "nearest_K_prereg": s.k.nearest_temp(t, s.IHP_TEMPS),
                         "correction_source_counts": src, "stored_X300_rows": len(d),
                         "n_equal_X300": int(sum(x <= s.REPRO_TOL for x in d)),
                         "max_rel_diff_vs_X300": max(d) if d else None,
                         "in_decide_V2": bool(d), "in_prereg_V2": t in (233, 343)}
    d_pre = [abs(near[kk] - x300[kk]) / x300[kk] for kk in near if kk in x300 and s.unkey(kk)[1] in (233, 343)]
    d_all = {kk: abs(near[kk] - x300[kk]) / x300[kk] for kk in near if kk in x300}
    worst = max(d_all, key=d_all.get)
    tag, t = s.unkey(worst)
    devices, rihp = s.load_ihp()
    rec = next(x for x in rihp[t] if x.tag == tag)
    rerun = {"record": worst, "stored_M_bias_near": near[worst], "stored_X300": x300[worst]}
    for src_t in (343, 300):
        corr = s.stored_corrections(s.I5, src_t)[tag]
        row = s.refit_task(("ihp", f"I5_1_lodo_{src_t}K", corr, rec, devices[tag], s.OPTS))
        rerun[f"refit_with_I5_1_lodo_{src_t}K"] = row["rrms"]
        rerun[f"refit_with_I5_1_lodo_{src_t}K_runtime_s"] = row["runtime_s"]
    rerun["rel_diff_343K_refit_vs_stored_near"] = abs(rerun["refit_with_I5_1_lodo_343K"] - near[worst]) / near[worst]
    rerun["rel_diff_300K_refit_vs_stored_X300"] = abs(rerun["refit_with_I5_1_lodo_300K"] - x300[worst]) / x300[worst]
    out = {"iteration": "I11-2", "post_hoc": True,
           "what": "diagnosis of V2 after the decision (decision.json unchanged): which M_bias_near rows the "
                   "decide check compared with the stored X300 rows, which stored fold correction each "
                   "M_bias_near row used, V2 restricted to the prereg temperatures, and two single-record refits",
           "per_temperature": per_t,
           "V2_as_decided": {"temperatures": [233, 343, 398], "n": len(d_all), "max_rel_diff": max(d_all.values())},
           "V2_as_prereg_text": {"temperatures": [233, 343], "n": len(d_pre), "max_rel_diff": max(d_pre),
                                 "ok": len(d_pre) == 36 and max(d_pre) <= s.REPRO_TOL},
           "cause": "run_decide selects M_bias_near rows by 'key in stored X300 rows'; the stored I5-1 X300 rows "
                    "exist at 233, 343 and 398 K, so the 18 rows at 398 K entered the check. At 398 K "
                    "M_bias_near uses the lodo_343K fold correction (prereg 398 -> 343) and X300 uses the "
                    "lodo_300K fold correction, so they are different refits by definition.",
           "rerun": rerun, "runtime_s": round(time.time() - t_start, 1)}
    path = s.OUT / "v2_diagnosis.json"
    s.write_json(path, out)
    print(json.dumps({kk: out[kk] for kk in ("per_temperature", "V2_as_decided", "V2_as_prereg_text", "rerun")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
