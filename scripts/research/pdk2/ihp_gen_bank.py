#!/usr/bin/env python3
"""IHP SG13G2 (PSP103, ngspice-46) LHC banks: ``gen_bank.py --pdk ihp``.

Same schema as the GF180 banks; output ``data/processed/pdk2/ihp_<set>``.

  PYTHONPATH=src .venv/bin/python -I scripts/research/pdk2/ihp_gen_bank.py \
      --set seven --workers 4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_bank  # noqa: E402

if __name__ == "__main__":
    sys.argv[1:1] = ["--pdk", "ihp"]
    raise SystemExit(gen_bank.main())
