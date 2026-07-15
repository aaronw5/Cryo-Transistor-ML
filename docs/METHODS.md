# Methods

This document describes the confirmed July 2026 pipeline. Historical June
experiments used a different pFET card, simulator/metric combination, sampling
box, and reporting policy; see `docs/RESEARCH_LOG.md` for that history.

Current method note (2026-07-15): the primary ML method is surrogate search
reported before and after FD polish. Its fixed comparison is a direct MLP that
maps 602 linear/signed-log I-V features to seven parameters in one forward
pass, with no search and no FD. The former inverse-network experiment is
obsolete. `scripts/fd_parameter_study.py` separately evaluates FD from the
published card and raw surrogate; `scripts/direct_mlp_fd_study.py` adds the
fixed direct-MLP -> FD pair. The MLP+FD artifact is an ablation only, so the
one-pass MLP remains the primary comparison. One fixed global foundation
emulator and one high-voltage-preserving candidate-selection study are reported
as exploratory series; none changes the canonical export.

## Reproducibility inputs

- Measured data and paper Table-6 device list: arXiv:2604.21625v1 companion
  repository.
- Simulation setup: `ogzamour/CryoPDK_Skywater130nm_ML`, pinned locally at
  commit `39b1e518e25120104225b8fa19f4cfc61a6766b3`.
- Simulator: conda-forge ngspice-41.
- Model cards: the confirmed upstream's unchanged nFET card and updated pFET
  77 K card.
- Device set: eight nMOS and ten pMOS geometries, exactly the paper's 18
  Table-6 rows.

The harness mirrors the upstream deck options, negative-going pMOS sweep
direction, device geometry/parasitic instance parameters, and per-device
parallel multiplicity. Native geometry-bin selection is used for the baseline.
Under this setup the 18 devices resolve to 18 distinct native model bins.

`scripts/verify_simulator.py` re-simulates all 11 scored curves for every
device on the upstream output grids. Across all 198 curves, the recorded worst
absolute difference from the committed upstream sweeps is `1.00e-11 A`. The
worst peak-relative mismatch is `0.00376` on a tiny-current curve whose
absolute difference is `7.11e-15 A`.

## Score

The primary scorer is a direct port of the confirmed upstream's
`rrmsCalc.py`. For an included curve `k`,

```text
RMSE_k = sqrt(mean((I_sim - I_meas)^2))
RRMS_k = RMSE_k / mean(abs(I_meas))
RRMS_device = mean_k(RRMS_k)
sigma_device = population_std_k(RRMS_k)
```

The 11 candidate curves are five output curves at
`|VGS|={0.37,0.74,1.11,1.48,1.85} V` and six transfer curves at
`|VDS|={0.01,0.37,0.74,1.11,1.48,1.85} V`. The scorer also reproduces:

- measured-current cleaning before the last exact zero;
- pMOS device-specific leading-current trims;
- nMOS measured-current inclusion thresholds, including the L=W=100 um
  special case;
- pMOS simulated-final-current inclusion thresholds.

Because pMOS inclusion depends on the candidate simulation, optimizing the
literal dynamic metric could improve a score by turning a curve off. The
published-card baseline's included tag set is therefore frozen per device for
optimization and primary method comparison. The literal dynamic-inclusion
score is also calculated and stored as an audit value.

Two aggregate conventions are reported:

```text
all_device_mean = mean(RRMS_device over all 18 devices)
combined_rrms = (mean(nMOS devices) + mean(pMOS devices)) / 2
```

The latter matches the confirmed upstream scorer's final printout. Values from
different conventions are never compared directly.

The older all-curve metric remains in `src/cryoml/metrics.py` for historical
continuity but is not the current optimization objective.

No alternative RRMS is used in current work. The high-voltage study below
constrains which candidates are eligible while evaluating them with this same
paper-defined score.

## Published-card baseline

`scripts/pdk_baseline.py` performs no fitting. It simulates the confirmed
upstream cards in ngspice-41, records the native bin, saves all curves, and
evaluates both the current and legacy metrics. Its current confirmed score is
`0.119825` nMOS, `0.399236` pMOS, and `0.259530` combined (`0.275053`
all-device mean).

## Parameterization and sampling

Only seven BSIM4 parameters are varied:

```text
VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0
```

For each native bin, ngspice `showmod` reads the published effective parameter
values. `LhcBox` defines a sign-safe linear +/-10% box around that vector. A
sigmoid/logit transform maps the finite physical box to unconstrained network
coordinates `z`; `z=0` is the published vector.

`scripts/pdk_gen_data.py` uses a seeded seven-dimensional SciPy Latin
hypercube with 10,000 samples per device. Row zero is replaced with the exact
published vector. Each vector is simulated in ngspice-41 on all 11 measured
bias grids. The per-device NPZ stores physical parameters, transformed
coordinates, simulated currents, validity flags, curve layout, published
center, native bin, and `box_mode=lhc10`.

The +/-10% effective-parameter perturbation is behaviorally equivalent to the
upstream's +/-10% multiplier perturbation for these multiplicative card
expressions, while letting the downstream extractor patch and export ordinary
BSIM4 parameter values.

## Learned forward surrogate

Each device gets an independent fully connected forward model

```text
E: z(7) -> signed_log(I_D at every retained grid point)
hidden widths: 512, 512, 512, 512
activation: GELU
```

The signed transform is

```text
signed_log(I) = sign(I) * log(1 + abs(I) / 1 nA).
```

Training uses Adam, weight decay `1e-5`, cosine learning-rate decay, a 15%
held-out validation split, early stopping, and 4096-sample minibatches. The
forward model is trained for at most 2,000 epochs.

## Direct parameter MLP comparison

`scripts/pdk_direct_mlp.py` adapts the user-provided Keras notebook recipe to
the repository's existing PyTorch/MPS environment. It is an ordinary
supervised forward pass from I-V features to parameters, with no parameter
search and no finite-difference polish.

Each device uses 301 deterministic current locations distributed nearly
equally across all 11 curves. The raw currents and their signed-log values are
concatenated, producing 602 input features. Linear and signed-log features are
min-max scaled using only the training split. The targets are the seven
physical parameters expressed as fractions of their +/-10% `LhcBox` bounds.

The default fixed recipe is:

```text
input: 602
hidden widths: 512, 512, 512
activation: LeakyReLU(0.01)
dropout: 0
initialization: LeCun uniform
output: 7 linear parameter fractions
loss: parameter-space MSE
optimizer: Adam, learning rate 1e-4, L2 weight decay 1e-3
schedule: exponential 0.99 decay per 100 optimizer steps
batch: 256
maximum epochs: 100
early-stopping patience: 20
split: 64% train, 16% validation, 20% held-out test
```

The notebook-scale `1700,3900,4500` widths remain available via `--arch`, but
are not silently substituted into the fixed reported recipe. After training,
the cleaned measured vector is transformed with the training scalers, passed
through the model exactly once, clipped to the physical box, and validated in
real ngspice-41. This produces `direct_mlp_forward_pass`.

## Differentiable inverse search

The measured curves are also fit by optimizing parameters through the frozen
forward surrogate. The production configuration uses 2,048 simultaneous Adam
starts for 600 steps. Starts include only the published vector, local normal
perturbations, and uniform physical-box coverage transformed to logit space.
No inverse-network initializer, June control, or extracted card is used.

The differentiable loss is the mean per-curve RRMS over the baseline-frozen
included/trimmed curve spans. The best state for every start is retained. Up to
14 separated candidates are then evaluated in real ngspice-41, and the best
validated surrogate-family candidate is `emu_search`.

## Finite-difference polish ablation

FD polish uses SciPy trust-region least squares with two-point numerical
Jacobians, relative difference step `2e-2`, and at most 120 function
evaluations. Residuals are constructed so their sum of squares equals the
fixed-inclusion device RRMS objective.

Each residual evaluation converts the proposed parameters to a model card,
runs fresh NGSpice curves, and compares those curves with the measured data.
The neural surrogate is not used inside FD and synthetic curves are not its
target.

The best five validated surrogate candidates are eligible for polish. The
production polished result is the best non-regressing endpoint under that
fixed top-five policy. Reporting has two surrogate stages:

| fixed series | meaning |
|---|---|
| `emu_search` | raw NGSpice-validated surrogate search |
| `emu_search+fd` | surrogate search after FD polish |

`scripts/make_ml_variants.py` re-simulates each fixed series for all 18
devices. It rejects missing/current-setup-incompatible records and never
substitutes another method. `scripts/make_figs.py` produces the per-device and
aggregate raw-versus-polished comparison in
`out/tables/fd_polish_ablation.md` and `figs/fd_polish_ablation.png`.

`scripts/fd_parameter_study.py` adds two stricter controls:

1. FD alone, initialized exactly at the published card;
2. FD initialized at the exact raw surrogate winner.

The second control is paired without top-five candidate selection. The study
reports that paired endpoint separately from the production top-five result,
records endpoint acceptance, `nfev`, runtime, and signed/absolute movement of
all seven physical parameters, and writes
`out/tables/fd_parameter_study.{md,json}` plus
`out/tables/fd_parameter_changes.csv`.

The completed control shows published-start FD unchanged at `0.2750533`
all-device with 0/18 wins. The exact raw surrogate winners improve from
`0.2357279` to `0.2296494` with 18/18 wins; the production top-five policy
reaches `0.2289864`. The zero published-start movement is a measured optimizer
result, not a skipped solve: the finite-difference Jacobian is nonzero, but the
local trust-region recipe does not accept a better step from that start.

`scripts/direct_mlp_fd_study.py` applies the same measured-data FD concept to
the fixed direct MLP. Each device has exactly one start: its archived
`direct_mlp_forward_pass` parameter vector. The script verifies the archived
raw NGSpice curves, re-simulates the start, uses the baseline-frozen curve set,
and runs the same trust-region/two-point/relative-`2e-2`/120-`nfev` recipe. It
accepts an endpoint only if it does not regress and saves fresh accepted
NGSpice curves under method ID `direct_mlp_forward_pass+fd`.

This paired ablation improves the all-device mean from `0.2722522` to
`0.2327984`, the family-combined score from `0.2578641` to `0.2182856`, and all
18 devices. Total runtime was `373.5 s`, with mean `44.56` SciPy function
evaluations and `253.78` actual residual evaluations per device. It is
reported in `out/tables/direct_mlp_fd_study.{md,json}` and
`out/pdk_direct_mlp_fd`; it does not replace the raw MLP in primary comparisons
or the uniform surrogate+FD card export.

## Exploratory Foundation Emulator

`scripts/pdk_foundation_emulator.py` is intentionally the last experiment,
after the primary fixed-method outputs. It trains one conditional forward
network over all 18 synthetic datasets (up to 180,000 parameter/I-V pairs).
The ten inputs are seven local +/-10% box coordinates, device polarity, scaled
`log(L)`, and scaled `log(W)`; the 2,046 outputs are the full signed-log I-V
vector. A disk-backed float32 cache avoids repeatedly inflating the compressed
NPZ datasets during training.

The frozen shared model uses the same fixed 2,048-start/600-step inverse search,
14 real-NGSpice validations, and five FD attempts as the production per-device
emulators. Both its raw and polished parameter vectors are re-simulated in
NGSpice. The report compares measured one-time training, per-device search,
validation, FD, and total wall times with published-start FD-only and the 18
per-device emulators. The foundation result is exploratory: it is never used
for per-device selection or the canonical card export. The random validation
split contains samples from every geometry, so this experiment tests one-model
reuse across the 18 known devices; it does not by itself demonstrate
generalization to an entirely unseen L/W geometry.

The completed shared model has held-out signed-log MSE `0.000242765`, versus
`0.000214598` averaged over the 18 separate models. Its raw and FD-polished
all-device RRMS values are `0.2269044` and `0.2260651`. Cache build, training,
and all 18 search/validation/FD stages took `14.5 s`, `189.5 s`, and `996.7 s`
respectively (`1200.7 s` total), compared with `2420.2 s` for the per-device
campaign. Published-start FD-only took `16.2 s` but did not improve the cards.

## Per-voltage and high-voltage diagnostics

`scripts/per_bias_diagnostic.py` examines pMOS `L=2 um, W=5 um` by selecting a
different best real-NGSpice LHC sample for each fixed-bias curve and then
re-simulating it. The paper, per-device surrogate+FD, foundation+FD, and
separate-voltage included-curve means are `0.1907`, `0.1900`, `0.1758`, and
`0.1078`. Because it uses multiple parameter vectors for one transistor, this
is an upper-bound diagnostic rather than a deployable compact model.

The device exposes a cross-bias compromise. Foundation+FD improves its device
mean but changes the strongest output curve `idvd@1.85` from paper RRMS
`0.0109` to `0.0470`; the separate-voltage result is `0.0046`. A 10,000-sample
parameter-space audit found no candidate in the seven-parameter +/-10% box
that improves every included curve simultaneously.

`scripts/high_voltage_guarded_study.py` keeps the paper RRMS unchanged. For
each device it protects the strongest included output and transfer curves and
requires each candidate to satisfy

```text
curve RRMS <= max(1.5 * paper curve RRMS, paper curve RRMS + 0.005).
```

Among feasible real-NGSpice LHC samples it selects the lowest official RRMS
and re-simulates that parameter vector. All 18 devices had a feasible
candidate. The fixed guarded series scores `0.2327267` all-device and retains
18/18 wins over paper cards. On pMOS L=2/W=5 it scores `0.1861` and restores
`idvd@1.85` to `0.0153`. This series is diagnostic and is not exported.

## Selection and exported cards

No per-device method selection feeds headline scores, I-V plots, Table 4,
Table 6, or card export. The direct MLP, surrogate raw, and surrogate + FD are
the three primary fixed series across all 18 devices. Foundation+FD and the
high-voltage guard are additional fixed exploratory series. Per-voltage cards
appear only in their diagnostic figure and table. Guarded results are retained
in tables/data but are intentionally excluded from all figures and slides.

The deployable method is uniformly `emu_search+fd` for every device/bin. The
direct MLP is a comparison rather than an export candidate.
`scripts/export_ml_cards.py` rejects any other method, patches the chosen
parameters into the two corner files, writes a library, and re-simulates that
written library before recording its manifest.

## Scaling study

`scripts/scaling_study.py` repeats the standalone surrogate-search path while
varying:

- training examples: 375, 750, 1,500, 3,000, 6,000, 10,000;
- network capacity: 64x3 through 1024x4;
- Adam starts: 128, 512, 2,048, 8,192.

Each cell records held-out surrogate error, differentiable search loss, raw
NGSpice RRMS, and RRMS after a short FD polish. A versioned run configuration
prevents the June CSV from being mixed into the current study.

The final all-device result is the complete data axis at 375, 750, 1,500,
3,000, and 6,000 examples: 90 cells across all 18 transistors. Held-out
emulator MSE falls from `0.004962` to `0.000524`, but arithmetic mean
real-NGSpice+FD RRMS is flat (`0.2280, 0.2270, 0.2273, 0.2280, 0.2294`). The
unfinished 10,000-sample/capacity/search extension was therefore stopped and
its partial rows are excluded from all-device means. Capacity/search results
remain a clearly labeled 56-cell/four-device pilot. The final plot uses the
arithmetic mean across all 18 as a bold line and faded colored per-transistor
traces. `scaling_study.py --data-only` reproduces exactly the accepted 90-cell
scope. Never overlap scaling with another heavy job.

## Reproduction sequence

The exact live continuation commands and resource/verification gates are kept
in `docs/HANDOFF.md`. The high-level sequence is:

```bash
export NGSPICE_BIN=/Users/anrunw/cryo-ng41/mm/envs/ng41/bin/ngspice
.venv/bin/python scripts/setup_data.py
.venv/bin/python scripts/verify_simulator.py
.venv/bin/python scripts/pdk_baseline.py
.venv/bin/python scripts/pdk_gen_data.py --num-samples 10000 --workers 10
.venv/bin/python scripts/pdk_ml_extract.py --device mps \
  --emu-arch 512,512,512,512 \
  --n-adam-starts 2048 --adam-steps 600 --n-validate 14 --n-polish 5 \
  --max-nfev 120 --out-dir out/pdk_surrogate_final
.venv/bin/python scripts/make_ml_variants.py --src out/pdk_surrogate_final
.venv/bin/python scripts/pdk_direct_mlp.py --device mps
.venv/bin/python scripts/fd_parameter_study.py
.venv/bin/python scripts/direct_mlp_fd_study.py --resume
.venv/bin/python scripts/pdk_compare.py
.venv/bin/python scripts/make_paper_tables.py
.venv/bin/python scripts/make_figs.py
.venv/bin/python scripts/scaling_study.py --device mps --data-only
.venv/bin/python scripts/make_scaling_fig.py
.venv/bin/python scripts/pdk_foundation_emulator.py --device mps --remove-cache
.venv/bin/python scripts/per_bias_diagnostic.py
.venv/bin/python scripts/high_voltage_guarded_study.py
.venv/bin/python scripts/pdk_compare.py
.venv/bin/python scripts/make_paper_tables.py
.venv/bin/python scripts/make_figs.py
```
