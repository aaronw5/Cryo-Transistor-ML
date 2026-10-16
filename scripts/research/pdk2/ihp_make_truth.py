#!/usr/bin/env python3
"""IHP SG13G2 (PSP103, ngspice-46) truth sets: ``make_truth.py --pdk ihp``.

Tiers a/b/c as documented in ``make_truth.py``; output
``out/research/pdk2/ihp_truth/<tier>``.

  PYTHONPATH=src .venv/bin/python -I scripts/research/pdk2/ihp_make_truth.py --tier all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_truth  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(make_truth.main(["--pdk", "ihp", *sys.argv[1:]]))
