#!/usr/bin/env python3
"""Compare the direct IV->params extractor: no-surrogate vs with-surrogate,
on (a) held-out synthetic reconstruction and (b) the measured curve, against
the published baseline and the search-based pipeline (ml_final)."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cryoml.config import OUT_DIR, OUT_TABLES  # noqa: E402


def load(d: str) -> dict[str, dict]:
    out = {}
    for f in glob.glob(str(OUT_DIR / d / "ml_*.json")):
        r = json.load(open(f))
        out[r["device"]] = r
    return out


def m(r, key):
    return r.get("methods", {}).get(key, {}).get("rrms", float("nan"))


def main() -> int:
    ns = load("pdk_fwd_nosurr")
    su = load("pdk_fwd_surr")
    mlf = load("pdk_ml_final_perdev")
    base = {x["device"]: x["rrms"] for x in json.load(
        open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))["devices"]}
    devs = sorted(ns)
    print(f"{'device':22s} | recon_synth_ng     | measured one-shot  | measured+FD       | refs")
    print(f"{'':22s} | nosurr  surr       | nosurr  surr       | nosurr  surr      | pub    ml_final")
    print("-" * 104)
    cols = {k: [] for k in
            ["rec_ns", "rec_su", "raw_ns", "raw_su", "fd_ns", "fd_su", "pub", "mlf"]}
    for dv in devs:
        rn, rs = ns[dv], su.get(dv, {})
        rec_ns = rn.get("recon_rrms_synth_ng", float("nan"))
        rec_su = rs.get("recon_rrms_synth_ng", float("nan"))
        raw_ns, raw_su = m(rn, "direct"), m(rs, "direct")
        fd_ns, fd_su = m(rn, "direct+fd"), m(rs, "direct+fd")
        pub, mf = base.get(dv, float("nan")), mlf.get(dv, {}).get("rrms", float("nan"))
        for k, v in zip(cols, [rec_ns, rec_su, raw_ns, raw_su, fd_ns, fd_su, pub, mf]):
            cols[k].append(v)
        print(f"{dv:22s} | {rec_ns:6.3f}  {rec_su:6.3f}     "
              f"| {raw_ns:6.3f}  {raw_su:6.3f}     "
              f"| {fd_ns:6.3f}  {fd_su:6.3f}    | {pub:5.3f}  {mf:5.3f}")
    print("-" * 104)
    mean = {k: float(np.nanmean(v)) for k, v in cols.items()}
    print(f"{'MEAN':22s} | {mean['rec_ns']:6.3f}  {mean['rec_su']:6.3f}     "
          f"| {mean['raw_ns']:6.3f}  {mean['raw_su']:6.3f}     "
          f"| {mean['fd_ns']:6.3f}  {mean['fd_su']:6.3f}    "
          f"| {mean['pub']:5.3f}  {mean['mlf']:5.3f}")
    print()
    print("Takeaways:")
    print(f"  one-shot IN-DISTRIBUTION (synthetic) reconstruction: "
          f"no-surrogate {mean['rec_ns']:.3f} -> with-surrogate {mean['rec_su']:.3f}")
    print(f"  one-shot on MEASURED curves: "
          f"no-surrogate {mean['raw_ns']:.3f} / with-surrogate {mean['raw_su']:.3f} "
          f"(published baseline {mean['pub']:.3f})")
    print(f"  measured after FD polish: "
          f"no-surrogate {mean['fd_ns']:.3f} / with-surrogate {mean['fd_su']:.3f} "
          f"(search pipeline ml_final {mean['mlf']:.3f})")
    json.dump({"mean": mean, "n_devices": len(devs)},
              open(OUT_TABLES / "forward_compare.json", "w"), indent=2)
    print(f"\nwrote {OUT_TABLES / 'forward_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
