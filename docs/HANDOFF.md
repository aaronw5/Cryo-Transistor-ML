# Confirmed-setup pipeline handoff

Last updated: 2026-07-15 EDT. This is the authoritative continuation record.
Read it with root `CLAUDE.md` before launching a heavy command.

## Immediate state

The requested pipeline and final verification are complete. No extraction,
training, scaling, or bulk NGSpice job should be resumed by default. At the
last process check no project heavy job was active.

Completed stages:

1. confirmed simulator/card/deck reproduction for all 198 sweeps;
2. published-card baseline for all 18 devices;
3. 10,000-sample +/-10% LHC data for every device;
4. clean per-device surrogate search and FD polish;
5. direct one-pass I-V-to-parameter MLP comparison;
6. FD-alone and paired FD parameter study;
7. complete all-18 data-scaling axis;
8. paper-format tables, real-NGSpice I-V figures, and canonical card export;
9. one global foundation emulator reused across all 18 devices;
10. pMOS L=2/W=5 per-voltage diagnostic;
11. fixed high-voltage-preserving candidate-selection study;
12. paired direct-MLP -> measured-data FD study and updated simple deck.

## Non-negotiable rules

1. Never report a per-device best-of as a method.
2. The canonical exported card uses `emu_search+fd` for all 18 native bins.
3. Keep RRMS exactly as defined by the paper's confirmed `rrmsCalc.py`.
4. Freeze the published-card inclusion set for optimization/primary comparison;
   retain dynamic official inclusion as an audit.
5. Final I-V curves are real NGSpice re-simulations of parameter vectors, not
   neural-emulator outputs.
6. The foundation and high-voltage guard are fixed exploratory series, not
   canonical exports.
7. Separate per-voltage cards are nondeployable diagnostics.
8. Keep high-voltage guarded results out of plots and slides; retain only the
   diagnostic tables/data.
9. Run only one compute-heavy stage at a time and inspect `ps` first.
10. Do not resume the intentionally stopped scaling grid without changing and
   documenting the protocol.

## Canonical setup

| item | value |
|---|---|
| upstream | `ogzamour/CryoPDK_Skywater130nm_ML` |
| pinned commit | `39b1e518e25120104225b8fa19f4cfc61a6766b3` |
| local upstream | `data/raw/CryoPDK_Skywater130nm_ML` |
| simulator | conda-forge ngspice-41 |
| simulator path | `/Users/anrunw/cryo-ng41/mm/envs/ng41/bin/ngspice` |
| pFET card | upstream updated 77 K pFET card |
| devices | 8 nMOS + 10 pMOS, all paper Table-6 rows |
| curves/device | 5 output + 6 transfer |
| tuned parameters | `VTH0,U0,NFACTOR,VSAT,DELTA,RDSW,ETA0` |
| synthetic data | 10,000 LHC samples/device in published +/-10% box |
| metric | faithful paper/upstream `rrmsCalc.py` port |

Use:

```bash
export NGSPICE_BIN=/Users/anrunw/cryo-ng41/mm/envs/ng41/bin/ngspice
export PYTHONPATH=src
```

The harness mirrors pMOS sweep direction, deck options, parasitics,
multiplicity, and native bin selection. All 18 current devices map to distinct
native bins. June results used an incompatible old card/metric/binning setup
and are history only.

## Verified inputs and score

The local harness matches all 198 upstream saved sweeps. Recorded maxima:

- worst absolute difference: `1.0000001e-11 A`;
- worst peak-relative difference with 1 pA floor: `0.0037594`;
- absolute difference on that tiny relative-error curve: `7.10543e-15 A`.

`src/cryoml/metrics.py` ports upstream current cleaning, pMOS trims, nMOS
measured-current exclusions, pMOS simulation-dependent exclusions, per-device
mean/sigma, and family aggregation. A literal upstream comparison had zero
score delta across all devices.

Do not mix aggregate conventions:

```text
combined_rrms = (mean(nMOS) + mean(pMOS)) / 2
all_device_mean = arithmetic mean of the 18 device RRMS values
```

| aggregate | confirmed NGSpice paper card | paper reported |
|---|---:|---:|
| nMOS mean | 0.1198250 | 0.1201250 |
| pMOS mean | 0.3992356 | 0.4057000 |
| combined | 0.2595303 | 0.2629125 |
| all-device | 0.2750531 | 0.2787778 |

All 18 `data/processed/pdk_synth/*.npz` files contain 10,000 valid `lhc10`
samples and all 11 curves, totaling about 2.3 GB.

## Completed fixed methods

| method | nMOS | pMOS | combined | all-device | wins |
|---|---:|---:|---:|---:|---:|
| direct MLP, no FD | 0.1283710 | 0.3873572 | 0.2578641 | 0.2722522 | 11/18 |
| surrogate raw | 0.0827710 | 0.3580934 | 0.2204322 | 0.2357279 | 17/18 |
| surrogate + FD | 0.0788212 | 0.3491185 | 0.2139699 | 0.2289864 | 18/18 |
| foundation + FD, exploratory | 0.0793719 | 0.3434197 | 0.2113958 | 0.2260651 | 18/18 |
| high-voltage guarded, diagnostic | 0.0806091 | 0.3544207 | 0.2175149 | 0.2327267 | 18/18 |

Primary directories:

```text
out/pdk_direct_mlp                 direct_mlp_forward_pass
out/pdk_direct_mlp_fd              direct_mlp_forward_pass+fd (ablation)
out/pdk_ml_emu_raw                emu_search
out/pdk_ml_emu                    emu_search+fd
out/pdk_foundation_emu            foundation raw and +FD
out/pdk_high_voltage_guarded      guarded real-NGSpice curves
out/pdk_ml_selected/cards         canonical uniform surrogate+FD export
```

The direct MLP follows the supplied notebook concept: 301 linear current
locations plus their signed-log values (602 inputs), min-max scaling,
512/512/512 LeakyReLU layers, seven normalized parameter outputs, Adam/L2,
exponential LR decay, early stopping, and exactly one inference pass. It has no
parameter search and no FD in the primary comparison. The separate
`out/pdk_direct_mlp_fd` ablation starts from that exact one-pass vector.

The per-device surrogate is a four-layer 512-wide GELU model from seven local
parameters to the full signed-log I-V vector. Production search uses 2,048 Adam
starts for 600 steps, validates 14 separated candidates in real NGSpice, and
FD-polishes the best five. The clean run took `2420.2 s`.

## FD study

FD is with respect to measured data. Every numerical parameter perturbation is
written into the compact-model card, simulated in fresh NGSpice, and compared
with measured curves using the frozen paper RRMS layout. It is not computed
against emulator or synthetic target curves.

Results:

- published -> FD alone: `0.2750531 -> 0.2750533`, effectively unchanged,
  0/18 wins, mean 13 actual objective evaluations and `16.2 s` total;
- direct MLP -> FD: `0.2722522 -> 0.2327984`, 18/18 wins, `373.5 s` total,
  mean 44.56 SciPy `nfev` and 253.78 actual residual evaluations per device;
- exact raw surrogate winner -> FD: `0.2357279 -> 0.2296494`, 18/18 wins;
- production top-five surrogate policy: `0.2289864`.

The published-start Jacobian was verified nonzero. The trust-region solve
nevertheless accepts no better local step, so FD alone is a real negative
result. FD works well after surrogate search supplies a better basin.

Artifacts: `out/tables/fd_parameter_study.{md,json}`,
`out/tables/fd_parameter_changes.csv`, `figs/fd_parameter_study.png`, and
`figs/fd_polish_ablation.png`. The direct pair is in
`out/tables/direct_mlp_fd_study.{md,json}`, `out/pdk_direct_mlp_fd`, and
`slides/plots/fd_improvement_comparison.png`.

## Scaling decision

The accepted final result is the five-point data axis complete for all 18
transistors, exactly 90 cells:

| samples/device | test MSE | raw NGSpice | NGSpice + FD |
|---:|---:|---:|---:|
| 375 | 0.004961851 | 0.2497051 | 0.2279589 |
| 750 | 0.003279399 | 0.2482016 | 0.2269846 |
| 1500 | 0.001669568 | 0.2393542 | 0.2273213 |
| 3000 | 0.001072415 | 0.2391011 | 0.2279737 |
| 6000 | 0.000524285 | 0.2398428 | 0.2294253 |

Emulator MSE improves about 9.5x, but physical NGSpice+FD RRMS is flat and best
at only 750 samples. The unfinished 10,000/capacity/search all-device grid was
stopped by design. Partial CSV rows are audit-only and excluded from reported
means. Capacity/search remain a 56-cell four-device pilot. The final plot uses
faded device traces plus a bold arithmetic all-device mean.

## Foundation emulator

`scripts/pdk_foundation_emulator.py` trains one model with inputs consisting of
seven local parameters, polarity, scaled `log(L)`, and scaled `log(W)`, and a
full 2,046-value signed-log I-V output. The same fixed search/validation/FD
recipe is used for all 18 devices.

- validation MSE: `0.000242765` global vs `0.000214598` mean per-device model;
- raw: `0.2269044`; after FD: `0.2260651`;
- cache `14.5 s`, training `189.5 s`, search/validation/FD `996.7 s`;
- total first campaign `1200.7 s`, versus per-device campaign `2420.2 s`.

This makes the foundation model the best fixed exploratory score and faster
than retraining 18 separate emulators for this known set. It is not proven on
an unseen geometry, still requires NGSpice validation, and remains excluded
from canonical export.

## High-voltage diagnosis

For pMOS L=2/W=5:

| method | included-curve mean | `idvd@1.85` |
|---|---:|---:|
| paper card | 0.1907 | 0.0109 |
| per-device surrogate + FD | 0.1900 | 0.0938 |
| foundation + FD | 0.1758 | 0.0470 |
| separate voltage cards | 0.1078 | 0.0046 |
| high-voltage guarded | 0.1861 | 0.0153 |

The per-voltage result selects a different best 10k LHC card for each curve and
re-simulates it, so it is not deployable. It demonstrates that a single
seven-parameter card must compromise across biases. No sampled candidate
improved every included curve.

The guarded study does not redefine RRMS. It protects the strongest included
output and transfer curves with the limit
`max(1.5 * paper curve RRMS, paper curve RRMS + 0.005)`, then selects the
lowest unchanged official RRMS among feasible real-NGSpice LHC candidates.
All 18 devices had feasible candidates. It is a diagnostic selection policy.

Training changes such as a hybrid signed-log plus normalized-linear loss may
rank strong-current candidates better. They cannot create physical degrees of
freedom missing from the seven-parameter +/-10% box. A wider box or added
strong-inversion/output-conductance parameters would be a separate experiment,
not a reproduction of the paper's original seven-parameter protocol.

## Figures and tables

All current method figures include measured points and the paper-card NGSpice
curve. The individual ML curve is also a fresh NGSpice simulation:

```text
figs/iv_direct.png
figs/iv_emu_raw.png
figs/iv_emu_fd.png
figs/iv_foundation_fd.png
```

Aggregate/paper analogues are `figs/fig2_iv_77k.png`,
`figs/fig4_bestfit.png`, `figs/fig5_rrms_heatmap.png`, and
`figs/table6_bars.png`. The 18-device appendix is `figs/devices/*.png`.
Diagnostic figures are `figs/scaling_laws.png`,
`figs/foundation_emulator_study.png`, and `figs/per_bias_pmos_L2_W5.png`.
The high-voltage guarded study is table/data only.

The simplified deck is `slides/cryo_ml_simple_results.pptx`, generated by
`scripts/make_simple_slides.py`. Its main comparisons contain only direct MLP
and surrogate+FD against measured/paper references. Raw surrogate appears only
on the FD-improvement slide, where it is paired with surrogate+FD and the fixed
direct MLP is paired with direct MLP+FD. Foundation, per-voltage, and guarded
material are excluded from the deck.

Tables are under `out/tables`; `comparison.md`, `table4_ml_params.md`, and
`table6_paper_format.md` are the primary reports.

## Reproduction commands

Heavy commands must run sequentially. Do not rerun completed stages unless an
artifact is missing or the protocol is intentionally changed.

```bash
# Inputs and baseline
PYTHONPATH=src .venv/bin/python scripts/verify_simulator.py
PYTHONPATH=src .venv/bin/python scripts/pdk_baseline.py
PYTHONPATH=src .venv/bin/python scripts/pdk_gen_data.py \
  --num-samples 10000 --workers 10

# Primary extraction
PYTHONPATH=src .venv/bin/python scripts/pdk_ml_extract.py --device mps \
  --emu-arch 512,512,512,512 \
  --n-adam-starts 2048 --adam-steps 600 --n-validate 14 \
  --n-polish 5 --max-nfev 120 --out-dir out/pdk_surrogate_final
PYTHONPATH=src .venv/bin/python scripts/make_ml_variants.py \
  --src out/pdk_surrogate_final
PYTHONPATH=src .venv/bin/python scripts/pdk_direct_mlp.py --device mps
PYTHONPATH=src .venv/bin/python scripts/fd_parameter_study.py
PYTHONPATH=src .venv/bin/python scripts/direct_mlp_fd_study.py --resume

# Scaling
PYTHONPATH=src .venv/bin/python scripts/scaling_study.py --device mps --data-only
PYTHONPATH=src .venv/bin/python scripts/make_scaling_fig.py

# Exploratory diagnostics, after primary work
PYTHONPATH=src .venv/bin/python scripts/pdk_foundation_emulator.py \
  --device mps --remove-cache
PYTHONPATH=src .venv/bin/python scripts/per_bias_diagnostic.py
PYTHONPATH=src .venv/bin/python scripts/high_voltage_guarded_study.py

# Reports/export
PYTHONPATH=src .venv/bin/python scripts/pdk_compare.py
PYTHONPATH=src .venv/bin/python scripts/export_ml_cards.py --src out/pdk_ml_emu
PYTHONPATH=src .venv/bin/python scripts/make_paper_tables.py
PYTHONPATH=src .venv/bin/python scripts/make_figs.py
.venv/bin/python scripts/make_simple_slides.py
```

## Cleanup status

Obsolete inverse-network, June control/CMA, direct/tandem, active-BO, floor,
and simulator-version scripts were removed. `cma` was removed from
requirements; `python-pptx` was restored for the new minimal deck generator.
Stale `out/pdk_ml_final/cards` was replaced by the uniform
`out/pdk_ml_selected/cards` export. Do not restore deleted historical methods
into current plots or tables.

## Final verification

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts
PYTHONPATH=src .venv/bin/python scripts/verify_simulator.py
.venv/bin/python scripts/setup_data.py --skip-clone
git diff --check
```

Audit gates:

- exactly 90 complete accepted all-device scaling cells;
- 18 direct, 18 raw surrogate, 18 polished surrogate records;
- 18 direct-MLP+FD paired-ablation records and fresh accepted simulations;
- 18 foundation and 18 guarded records;
- 18 FD-alone and 18 paired-surrogate records;
- canonical manifest says `uniform_method=emu_search+fd` and written-card
  re-simulation agrees at numeric precision;
- no inverse method appears in current figures or headline tables;
- all ML I-V figures include measured points, paper curves, and real-NGSpice
  predicted-parameter curves;
- README, GOAL, METHODS, RESEARCH_LOG, this handoff, and Claude memory agree.

Final gate result on 2026-07-15:

- 13/13 unit tests passed;
- `compileall` passed for `src` and `scripts`;
- simulator verification passed all 198 curves with the recorded maxima;
- `setup_data.py --skip-clone` validated the pinned local inputs;
- artifact audit passed every count/method/box/scaling/export assertion;
- `git diff --check` passed;
- final paper, method, scaling, foundation, FD (including MLP+FD), and
  per-voltage figures were
  visually inspected and are nonblank/readable;
- all final I-V PNGs are fully opaque with white backgrounds. A black area
  observed once in a batched tool preview was a preview-rendering issue; the
  original PNG and original-resolution inspection are correct.
