#!/usr/bin/env python3
"""Emit our results in the paper's EXACT table formats (arXiv:2604.21625).

Table 6 format (Appendix A, "Reported Errors for 77 K Models"):
  Transistor Type | Length (μm) | Width (μm) | RMSE (μA) | RRMS̄ | σ_RRMS
  - RMSE: 3 significant figures
  - RRMS̄: 3 decimals
  - σ_RRMS: 3 decimals below 1, else 3 significant figures
  - no mean row (the paper quotes means in text)

Table 4 format (per-bin extracted parameters): one sub-table per model
bin, headed "nMOS: Lmin = a, Lmax = b; Wmin = c, Wmax = d", rows U0, RDSW,
ETA0, VSAT, DELTA, VTH0, NFACTOR, comparing published 77 K vs ML 77 K.

Outputs:
  out/tables/table6_paper_format.md  (published-in-NGSpice, CMA-ES refit
                                      library, ML library — one table each)
  out/tables/table4_ml_params.md
  out/tables/paper_tables.json       (structured, consumed by make_slides)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, OUT_TABLES, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import per_curve_rrms  # noqa: E402
from cryoml.spice_pdk import PDK77K_DIR, _CORNER_BASENAME  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

PARAM_ORDER = ("U0", "RDSW", "ETA0", "VSAT", "DELTA", "VTH0", "NFACTOR")

CARD_SETS = [
    ("paper cards run in NGSpice", OUT_DIR / "pdk_baseline"),
    ("FD refit (one card per size bin)", OUT_DIR / "pdk_fd_deploy"),
    ("CMA-ES refit (one card per size bin)", OUT_DIR / "pdk_cma_deploy"),
    ("ML refit (one card per size bin)", OUT_DIR / "pdk_ml_final"),
    ("direct-predict ML (one card per transistor)", OUT_DIR / "pdk_fwd_surr_best"),
]

_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sig3(v: float) -> str:
    """3 significant figures, plain decimal (paper RMSE style)."""
    if v == 0 or not np.isfinite(v):
        return "0"
    return f"{v:.3g}"


def sigma_fmt(v: float) -> str:
    return f"{v:.3f}" if abs(v) < 1 else f"{v:.3g}"


def sci(v: float) -> str:
    """Paper Table-4 style: plain for moderate values, a×10^b otherwise."""
    if v == 0 or not np.isfinite(v):
        return "0"
    if 1e-2 <= abs(v) < 1e4:
        return f"{v:.3g}"
    m, e = f"{v:.2e}".split("e")
    return f"{m}×10{str(int(e)).translate(_SUP)}"


def device_metrics(dev, sims_dir: Path) -> tuple[float, float, float]:
    tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
    curves = load_device_curves(dev)
    saved = np.load(sims_dir / f"sims_{tag}.npz")
    per = [per_curve_rrms(np.asarray(saved[f"sim_{i}"]), c.Id)
           for i, c in enumerate(curves)]
    rr = [m.rrms for m in per if np.isfinite(m.rrms)]
    return (float(np.mean([m.rmse_uA for m in per])),
            float(np.mean(rr)), float(np.std(rr)))


def table6_block(name: str, sims_dir: Path) -> tuple[list[str], list[list[str]]]:
    rows = []
    for d in PAPER_DEVICES:
        rmse, rrms, sigma = device_metrics(d, sims_dir)
        rows.append([
            "nMOS" if d.dev_type == "nmos" else "pMOS",
            f"{d.L_um:g}", f"{d.W_um:g}",
            sig3(rmse), f"{rrms:.3f}", sigma_fmt(sigma),
        ])
    headers = ["Transistor Type", "Length (μm)", "Width (μm)",
               "RMSE (μA)", "RRMS̄", "σ_RRMS"]
    return headers, rows


def bin_boxes() -> dict[tuple[str, int], tuple[float, float, float, float]]:
    """Parse lmin/lmax/wmin/wmax (in μm) per bin from the corner files."""
    out = {}
    for dev_type, base in _CORNER_BASENAME.items():
        text = (PDK77K_DIR / base).read_text(errors="replace")
        # bins appear in order; model sections look like
        # .model <name>__<idx> ... lmin = <v> lmax = <v> wmin = <v> wmax = <v>
        pat = re.compile(
            r"\.model\s+\S+?\.(\d+)\s.*?"
            r"lmin\s*=\s*([0-9.eE+\-]+).*?lmax\s*=\s*([0-9.eE+\-]+).*?"
            r"wmin\s*=\s*([0-9.eE+\-]+).*?wmax\s*=\s*([0-9.eE+\-]+)",
            re.IGNORECASE | re.DOTALL)
        for m in pat.finditer(text):
            idx = int(m.group(1))
            vals = [float(m.group(i)) for i in range(2, 6)]
            # corner files use scale=1.0u → values are bare microns; stock
            # PDK uses meters — normalize to μm
            vals = [v * 1e6 if v < 1e-3 else v for v in vals]
            out[(dev_type, idx)] = tuple(vals)
    return out


def table4_blocks() -> list[dict]:
    boxes = bin_boxes()
    blocks, seen = [], set()
    for d in PAPER_DEVICES:
        tag = device_tag(d.dev_type, d.L_um, d.W_um)
        rec = json.loads((OUT_DIR / "pdk_ml_final" / f"ml_{tag}.json")
                         .read_text())
        key = (d.dev_type, int(rec["bin_index"]))
        if key in seen:
            continue
        seen.add(key)
        pub = rec["params_by_method"]["published"]
        ml = rec["params_by_method"][rec["best_method"]]
        box = boxes.get(key)
        pol = "nMOS" if d.dev_type == "nmos" else "pMOS"
        if box:
            head = (f"{pol}: Lmin = {box[0]:g}, Lmax = {box[1]:g}; "
                    f"Wmin = {box[2]:g}, Wmax = {box[3]:g}")
        else:
            head = f"{pol}: model bin {rec['bin_index']}"
        rows = [[f"`{p}`", sci(pub[p.lower()]), sci(ml[p.lower()])]
                for p in PARAM_ORDER]
        blocks.append({"head": head, "headers":
                       ["Parameter", "published 77 K", "ML 77 K"],
                       "rows": rows})
    return blocks


def main() -> int:
    ensure_dirs()
    t6 = []
    for name, sims_dir in CARD_SETS:
        headers, rows = table6_block(name, sims_dir)
        t6.append({"name": name, "headers": headers, "rows": rows})

    with open(OUT_TABLES / "table6_paper_format.md", "w") as f:
        f.write("# Reported Errors for 77 K Models — paper Table 6 format\n"
                "\nAll values computed in the identical corrected NGSpice "
                "chain.\n")
        for block in t6:
            f.write(f"\n## {block['name']}\n\n")
            f.write("| " + " | ".join(block["headers"]) + " |\n")
            f.write("|" + "|".join(["---"] * len(block["headers"])) + "|\n")
            for row in block["rows"]:
                f.write("| " + " | ".join(row) + " |\n")

    t4 = table4_blocks()
    with open(OUT_TABLES / "table4_ml_params.md", "w") as f:
        f.write("# Extracted BSIM4 parameters per model bin — paper "
                "Table 4 format\n\nPublished 77 K values vs the ML-refit "
                "values actually shipped in the library.\n")
        for block in t4:
            f.write(f"\n### {block['head']}\n\n")
            f.write("| " + " | ".join(block["headers"]) + " |\n")
            f.write("|" + "|".join(["---"] * 3) + "|\n")
            for row in block["rows"]:
                f.write("| " + " | ".join(row) + " |\n")

    json.dump({"table6": t6, "table4": t4},
              open(OUT_TABLES / "paper_tables.json", "w"), indent=1)
    print(f"wrote {OUT_TABLES}/table6_paper_format.md, table4_ml_params.md, "
          "paper_tables.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
