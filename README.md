# cryo-bsim4-ml

ML-guided extraction of seven cryogenic SKY130 BSIM4 parameters:

```text
VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0
```

Every result is validated in NGSpice using the setup from
`ogzamour/CryoSkywater130nm_CorrectedForNgspice`:

- published 77 K corner files,
- Volare SKY130 PDK revision `a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b`,
- native geometry-bin selection,
- corrected-repository deck convention,
- paper companion notebook RRMS over every measured curve.

NGSpice does not reproduce the paper-reported mean RRMS with the published
parameter cards. Results are therefore compared with the paper parameter cards
run through the identical NGSpice chain.

## Pipeline

1. `scripts/verify_simulator.py`: verify the local harness against the
   corrected repository's saved NGSpice sweeps.
2. `scripts/pdk_baseline.py`: run the published parameter cards.
3. `scripts/pdk_gen_data.py`: generate synthetic NGSpice samples in each
   device's native geometry bin.
4. `scripts/pdk_ml_extract.py`: train a per-device neural emulator and inverse
   MLP, search with gradients, validate candidates in NGSpice, and polish with
   NGSpice finite differences.
5. `scripts/pdk_compare.py`: recompute and report paper-exact RRMS.
6. `scripts/export_ml_cards.py`: export one combined nMOS/pMOS 77 K library.

Finite-difference and CMA-ES extractors are retained as non-ML controls.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_data.py
```

Install NGSpice separately. The current validated environment uses
`ngspice-46`.

## Run

```bash
python scripts/pdk_baseline.py
python scripts/verify_simulator.py
python scripts/pdk_gen_data.py --num-samples 1500 --workers 8
python scripts/pdk_ml_extract.py --device mps --resume
python scripts/pdk_compare.py --methods out/pdk_ml:ml
python scripts/export_ml_cards.py
```

## Current Result

Across all 18 Table-6 devices, the deployable ML-extracted cards reduce mean
paper-exact RRMS from `0.69187` to `0.54376` in the identical corrected
NGSpice flow, with 17 device-level wins. Devices mapped by NGSpice to the same
model bin share one jointly polished parameter vector.

Optional controls:

```bash
python scripts/pdk_fd_extract.py --workers 8
python scripts/pdk_cma_extract.py --workers 8
python scripts/pdk_compare.py \
  --methods out/pdk_fd:fd out/pdk_cma:cma out/pdk_ml:ml
```

Outputs are written under `out/`; the current comparison is
`out/tables/comparison.md`.
