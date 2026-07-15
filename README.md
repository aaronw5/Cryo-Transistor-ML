# Cryogenic SKY130 parameter extraction with ML

This repository reproduces the 77 K SKY130 BSIM4 workflow from
arXiv:2604.21625v1 in the authors' confirmed open simulation setup. It then
extracts the paper's seven parameters with a learned forward emulator
(`parameters -> I-V curves`) searched in parameter space, validates every
reported parameter vector in real NGSpice, and optionally applies numerical
finite-difference (FD) polish against the measured I-V data.

The primary comparison uses fixed methods over all 18 Table-6 transistors:

- direct MLP: measured I-V features -> seven parameters in one forward pass;
- per-device surrogate search, before FD;
- the same per-device surrogate search after FD.

There is no per-device method cherry-picking. The global foundation emulator
and high-voltage-preserving selection are fixed exploratory series. The
canonical exported card uses per-device surrogate search + FD for every bin.

## Confirmed setup

| component | pinned value |
|---|---|
| upstream flow | `ogzamour/CryoPDK_Skywater130nm_ML@39b1e518` |
| paper | arXiv:2604.21625v1 |
| simulator | conda-forge `ngspice-41` |
| device set | all 18 paper Table-6 geometries |
| curves per device | 5 output + 6 transfer |
| parameters | `VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0` |
| synthetic data | 10,000-point Latin hypercube in the published +/-10% box |
| metric | faithful upstream `rrmsCalc.py` port; unchanged from the paper |

The harness reproduces all 198 committed upstream sweeps. The worst absolute
current difference is `1.0000001e-11 A`; the worst peak-relative difference is
`0.0037594` on a tiny-current curve (`7.10543e-15 A` absolute). Correct pMOS
sweep direction, device multiplicity, deck tolerances, updated pFET card, and
native geometry-bin selection are required. The 18 geometries map to 18
distinct native bins.

## Results

Lower RRMS is better. `combined` is the paper/upstream convention,
`(nMOS mean + pMOS mean) / 2`; `all-device` is the arithmetic mean over all 18
devices. Primary comparisons freeze the published-card curve-inclusion set so
a candidate cannot improve by triggering the scorer's simulation-dependent
pMOS exclusion. The RRMS calculation itself is not modified.

| fixed method | nMOS | pMOS | combined | all-device | wins vs paper card |
|---|---:|---:|---:|---:|---:|
| paper parameters in confirmed NGSpice | 0.1198 | 0.3992 | 0.2595 | 0.2751 | - |
| direct MLP, one pass, no FD | 0.1284 | 0.3874 | 0.2579 | 0.2723 | 11/18 |
| per-device surrogate search, raw | 0.0828 | 0.3581 | 0.2204 | 0.2357 | 17/18 |
| **per-device surrogate search + FD** | **0.0788** | **0.3491** | **0.2140** | **0.2290** | **18/18** |
| global foundation surrogate + FD (exploratory) | 0.0794 | 0.3434 | 0.2114 | 0.2261 | 18/18 |
| high-voltage guarded selection (diagnostic) | 0.0806 | 0.3544 | 0.2175 | 0.2327 | 18/18 |

The paper reports `0.2629` combined and `0.2788` all-device using its HSPICE
flow. The confirmed NGSpice published-card baseline is close at `0.2595` and
`0.2751`.

### FD ablation

FD perturbs parameters, runs fresh NGSpice simulations, and computes residuals
against measured data. It does not optimize against surrogate-generated data.

- Published card -> FD alone: `0.2751 -> 0.2751`, 0/18 improvements. The local
  solver stalls at the published start under the production trust-region
  recipe.
- Direct MLP -> direct MLP + FD: `0.2723 -> 0.2328`, 18/18 improvements. This
  fixed one-start paired result took `373.5 s`; the raw one-pass MLP remains
  the primary direct-method comparison.
- Same raw surrogate winner -> FD: `0.2357 -> 0.2296`, 18/18 improvements.
- Production top-five surrogate candidate policy -> FD: `0.2357 -> 0.2290`.

FD is effective as local polish after the emulator finds a better basin, but
it is not an effective global extractor from the published card in this setup.

### Scaling

The accepted scaling study averages all 18 transistors. From 375 to 6,000
samples per transistor, held-out emulator MSE improves about 9.5x
(`0.004962 -> 0.000524`), while real-NGSpice RRMS after FD remains flat:
`0.2280, 0.2270, 0.2273, 0.2280, 0.2294`. The best mean occurs at 750 samples,
so the remaining all-device capacity/search grid was intentionally stopped.
The plot shows every device as a faded colored trace and the arithmetic mean in
bold. Capacity/search results are retained only as a labeled four-device pilot.

### Foundation emulator

One geometry-conditioned forward model was trained once over the 18 datasets
and reused for all inverse searches. It achieved `0.2261` all-device after FD,
slightly better than the 18 separate emulators (`0.2290`). Cache construction,
training, and all search/validation/FD work took `1200.7 s`, versus `2420.2 s`
for the per-device extraction campaign. Published-start FD alone took `16.2 s`
but did not improve RRMS.

This experiment demonstrates reuse across the 18 known geometries. Its random
within-geometry validation split does not establish generalization to an
unseen L/W geometry, so it remains exploratory and is not the exported card.

### High-voltage behavior

For pMOS `L=2 um, W=5 um`, the foundation result improves device RRMS from
`0.1907` to `0.1758` but worsens the strongest output curve
`idvd@1.85` from `0.0109` to `0.0470`. Selecting one separate card per voltage
reduces the included-curve mean to `0.1078`, proving that the single seven-
parameter card is making a cross-bias compromise; those separate cards are not
deployable.

The high-voltage guard keeps the paper RRMS unchanged and selects the lowest-
RRMS LHC candidate whose strongest included output and transfer curves remain
within `max(1.5 * paper curve RRMS, paper curve RRMS + 0.005)`. For this device,
it gives overall `0.1861` and restores `idvd@1.85` to `0.0153`. Across all 18,
it gives `0.2327` and still wins over every published card. This is a diagnostic
selection policy, not a new metric or the canonical export.

More emulator training is unlikely to remove this tradeoff by itself: a 10,000
real-NGSpice sample audit found no candidate in the seven-parameter +/-10% box
that improves every included curve for pMOS `L=2 um, W=5 um`. A hybrid
signed-log plus normalized-linear training loss could improve candidate
ranking, but materially better simultaneous strong- and weak-current fits
would require a separately labeled wider box or additional BSIM parameters.

## Method and artifacts

Detailed definitions are in [docs/METHODS.md](docs/METHODS.md). The current
handoff and exact commands are in [docs/HANDOFF.md](docs/HANDOFF.md).

- `out/tables/comparison.md`: all devices and all fixed series.
- `out/tables/table6_paper_format.md`: paper-format RMSE, RRMS, and sigma.
- `out/tables/table4_ml_params.md`: published and extracted parameters.
- `out/tables/fd_parameter_study.md`: FD-alone and FD-polish ablations.
- `out/tables/direct_mlp_fd_study.md`: direct MLP before/after FD for all 18.
- `out/tables/scaling_study.md`: accepted all-18 data scaling.
- `out/tables/foundation_emulator_study.md`: global model accuracy and timing.
- `out/tables/per_bias_pmos_L2_W5.md`: nondeployable per-voltage diagnostic.
- `out/tables/high_voltage_guarded_study.md`: unchanged-RRMS guard study.
- `figs/fig2_iv_77k.png`, `figs/fig4_bestfit.png`: measured points, published
  NGSpice curves, and NGSpice curves from every plotted parameter series.
- `figs/iv_<method>.png`: individual paper/direct/raw/FD/foundation plots;
  each ML plot also contains the published-card curve.
- `figs/fig5_rrms_heatmap.png`, `figs/table6_bars.png`: all-device summaries.
- `figs/devices/<tag>.png`: complete 18-device appendix.
- `out/pdk_ml_selected/cards`: uniform surrogate+FD exported library.
- `slides/cryo_ml_simple_results.pptx`: simplified presentation. Main slides
  show only direct MLP and surrogate+FD against measured/paper references; raw
  surrogate and direct MLP + FD appear only on the paired FD-improvement slide.

Final figures never use neural-emulator currents as physical predictions. Each
line is a real NGSpice re-simulation of the corresponding parameter vector.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export NGSPICE_BIN=/Users/anrunw/cryo-ng41/mm/envs/ng41/bin/ngspice
export PYTHONPATH=src
python scripts/setup_data.py
```

`setup_data.py` validates the pinned upstream checkout and installs its updated
pFET corner. NGSpice must be installed separately.

## Reproduce

Run heavy stages sequentially and inspect `ps` before starting one.

```bash
python scripts/verify_simulator.py
python scripts/pdk_baseline.py
python scripts/pdk_gen_data.py --num-samples 10000 --workers 10

python scripts/pdk_ml_extract.py --device mps \
  --emu-arch 512,512,512,512 \
  --n-adam-starts 2048 --adam-steps 600 --n-validate 14 \
  --n-polish 5 --max-nfev 120 --out-dir out/pdk_surrogate_final
python scripts/make_ml_variants.py --src out/pdk_surrogate_final
python scripts/pdk_direct_mlp.py --device mps
python scripts/fd_parameter_study.py
python scripts/direct_mlp_fd_study.py --resume

python scripts/scaling_study.py --device mps --data-only
python scripts/make_scaling_fig.py

# Exploratory stages, run after primary results.
python scripts/pdk_foundation_emulator.py --device mps --remove-cache
python scripts/per_bias_diagnostic.py
# Diagnostic table/data only; guarded cards are intentionally not plotted.
python scripts/high_voltage_guarded_study.py

python scripts/pdk_compare.py
python scripts/export_ml_cards.py --src out/pdk_ml_emu
python scripts/make_paper_tables.py
python scripts/make_figs.py
python scripts/make_simple_slides.py
```

The synthetic datasets are about 2.3 GB and are gitignored. Historical June
experiments used a different card, metric, and binning regime; they are
preserved in the research log but are not compatible with this workflow.

## Verify

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts
PYTHONPATH=src .venv/bin/python scripts/verify_simulator.py
.venv/bin/python scripts/setup_data.py --skip-clone
git diff --check
```
