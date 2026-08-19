# Foundation/full-parameter extraction experiment — 2026-08-18

Status: active. Results in this file are provisional until the complete fixed
series has been validated in NGSpice and the paper-format cards are regenerated.

## Requested comparison

Run the best geometry-conditioned foundation setup with the Monte Carlo work
from `ogzamour/CryoPDK_Skywater130nm_ML`, comparing:

1. the original seven-parameter forward surrogate;
2. a forward surrogate using all relevant continuous BSIM4 parameters, not
   only the seven variables in the upstream Monte Carlo script;
3. finite-difference parameter extraction through the surrogate as method 1;
4. direct I-V-to-parameter prediction with reconstruction through the frozen
   forward surrogate (autoencoder-style) as method 2.

Use the existing 18 Table-6 device set and paper-format reporting. Diagnose
high-voltage failures explicitly, including nMOS L=1 um, W=3 um at
VGS=1.85 V, so aggregate low-voltage accuracy cannot hide strong-inversion
errors.

## Starting state

- Local code revision: `95b3530` on clean `main` before this log was added.
- Documented upstream revision: `39b1e518e25120104225b8fa19f4cfc61a6766b3`.
- Current upstream `master` discovered on 2026-08-18:
  `c1c87615cd20de76a1759067398c0037d8565b3b`.
- Existing synthetic data: 10,000 seven-parameter LHC points per device,
  2.3 GB total.
- Existing accelerator/simulator: PyTorch 2.12 with Apple MPS available;
  conda-forge NGSpice 41.
- Existing primary metric: faithful upstream RRMS with the paper-card curve
  inclusion set frozen for comparisons and the dynamic score retained as an
  audit.
- No compute-heavy Python or NGSpice job was active at the start.

The current upstream README says its Monte Carlo script nominally sweeps all
parameters, but the implementation is seven-dimensional (`VTH0`, `VSAT`,
`RDSW`, `U0`, `NFACTOR`, `ETA0`, and `DELTA`) and the checked-in configuration
is limited to an nMOS model-0 VGS sweep at one fixed drain bias. This does not
satisfy the requested expanded experiment. The full candidate set will be
derived from the continuous DC-current parameters in the actual BSIM4 cards,
then screened on the complete 11-curve grid for all 18 devices. The upstream
18-variable list is a minimum audit reference, not a cap. Categorical switches,
geometry selectors/bounds, capacitance/noise-only terms, unsafe discontinuous
controls, exact aliases, and numerically inactive parameters will be excluded
with recorded reasons. There will be no arbitrary top-k cutoff.

## Guardrails for this run

- Keep seven-parameter and full-parameter results as complete fixed series;
  do not select a different method per device.
- Report overall, per-family, per-device, per-curve, and high-voltage-only
  metrics from real NGSpice simulations of extracted cards.
- Preserve the paper RRMS definition for direct comparison. Add a separate
  high-voltage diagnostic/objective rather than silently redefining RRMS.
- Run one heavy stage at a time and record its command, configuration, random
  seed, elapsed time, and output directory here.
- Treat any expanded parameter bounds as a distinct experiment and document
  them explicitly.

## Live log

### 2026-08-18 — audit started

Read the repository instructions, README, goal, handoff, existing model and
diagnostic inventory, and the linked upstream README. Confirmed that prior work
already contains a seven-parameter geometry-conditioned foundation surrogate,
a real-NGSpice finite-difference polish, a direct I-V-to-seven-parameter MLP,
and a high-voltage guarded candidate-selection diagnostic. The requested
full-parameter forward/inverse comparison is not yet present.

Next: diff upstream `c1c87615` against the documented `39b1e518`, audit its
Monte Carlo implementation and model-card parameter surface, then write a
small deterministic data-generation validation before launching a full run.

### 2026-08-18 — scope correction: expanded parameter surface

The requested “all parameters” run means all *relevant* BSIM4 parameters for
the ML model, not all seven parameters varied at once. Inspection confirmed
that upstream `nomSweep_latinHypercube.py` hard-codes exactly the original
seven variables and therefore supplies workflow ideas but not the requested
parameter dimensionality. The expanded parameter list will be chosen by a
documented NGSpice sensitivity screen, including separate high-voltage
sensitivity so low-voltage behavior cannot eliminate parameters needed in
strong inversion.

### 2026-08-18 — global failure coverage and maximum relevant set

The expanded experiment must address high-voltage failures across the entire
18-device set; nMOS L=1 um/W=3 um is a required regression probe but not the
only target. The parameter audit will cover every defensible continuous BSIM4
DC-current term present in either card, including threshold/short-channel,
mobility/high-field, resistance, subthreshold/DIBL, output-conductance, leakage,
junction, and active temperature terms. A parameter will be retained if it
produces a response above the measured NGSpice repeatability floor for any
device in either the full-grid or high-voltage diagnostics. Only inapplicable,
unsafe, aliased, or numerically inactive terms will be removed.

### 2026-08-18 — 115-parameter L1/W3 sensitivity pilot

Implemented `scripts/pdk_parameter_sensitivity.py` and a categorized
115-parameter continuous DC screening surface. The screen uses effective
`showmod` values, +/-10% perturbations (family scales for card-zero values),
safe retry bounds, two repeat baseline simulations, and separate full-grid and
|fixed bias| >= 1.48 V response. Selection threshold is `1e-4` normalized
current response with no top-k cap; the original seven are mandatory so the
expanded surface remains nested.

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/pdk_parameter_sensitivity.py \
  --devices nmos:1:3 \
  --out-dir out/pdk_parameter_sensitivity_pilot \
  --workers 1 --fraction 0.10 --threshold 1e-4
```

Result: 31/115 candidates retained for this device in 11.6 s. Highest
high-voltage responses included `TOXE`, `VTH0`, `TOXM`, `A2`, `UA`, `U0`,
`UB`, `AGS`, `WR`, `K1`, `LPE0`, `RDSW`, `PCLM`, `DELTA`, and `PDIBLC2`.
`PDIBLC1`, `DROUT`, and `DSUB` were numerically inactive for this particular
bin under the tested perturbations, reinforcing the need for a union over all
18 devices rather than selection from the named regression device alone.

### 2026-08-18 — all-device sensitivity screen

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/pdk_parameter_sensitivity.py \
  --out-dir out/pdk_parameter_sensitivity \
  --workers 6 --fraction 0.10 --threshold 1e-4
```

Wall time was 45.45 s (256.2 summed device-seconds). Repeat simulations were
bit-stable under the response calculation, so the effective threshold remained
`1e-4`. The union over all 18 devices retained 43/115 candidates:

```text
TOXE TOXM XJ NDEP NSD DWG RSH VTH0 K1 K2 K3 DVT0 DVT1 DVT1W W0 K3B
VFB LPE0 LPEB VSAT UA UB RDSW PRWG WR U0 A0 A2 AGS B0 B1 VOFF NFACTOR
CIT CDSCD ETA0 DSUB PCLM PDIBLC2 DROUT DELTA ALPHA0 BETA0
```

The strongest maximum high-voltage responses came from `UA`, `VTH0`, `UB`,
`TOXE`, `TOXM`, `WR`, `DVT1`, `NFACTOR`, `PCLM`, `K1`, `U0`, and `RDSW`.
The broader screen recovered 12 parameters not active in the L1/W3 pilot,
including `DSUB`, `DROUT`, `DVT1W`, `VFB`, `B0`, and `B1`. Of the upstream
18-variable shortlist, `UC`, `RDW`, `RSW`, and `PVAG` were below threshold;
the other 14 were retained. Once a term enters the all-device union, it remains
variable on every card for which NGSpice found safe nondegenerate bounds. A
small one-at-a-time response on one geometry is not treated as proof of
inactivity under simultaneous variation. Only absent, zero-across-family, or
non-finite dimensions are held at their published values, while preserving the
common 43-feature model input.

### 2026-08-18 — simultaneous expanded-LHC pilot

Implemented `scripts/pdk_gen_full_data.py` with a signed parameter manifest,
per-device active masks, sensitivity-safe bounds, fixed published values for
inactive terms, and all 2,046 current locations. Pilot command:

```bash
PYTHONPATH=src .venv/bin/python scripts/pdk_gen_full_data.py \
  --devices nmos:1:3 \
  --out-dir data/processed/pdk_synth_full_pilot \
  --num-samples 256 --workers 1 --seed 20260818
```

Result: 256/256 finite samples in 22.5 s. The common layout has 43 parameters;
38 are active for nMOS L=1 um/W=3 um. Currents are stored as float32 for model
training while parameters/bounds remain float64; all extracted cards will still
be validated from fresh float64 NGSpice output.

### 2026-08-18 — nominal-card and curve-layout checks

The simultaneous 43-parameter pilot's published row reproduces the untouched
foundry-card nMOS L=1 um/W=3 um curves to numerical precision (flat normalized
MSE `2.07e-14`); the existing seven-parameter published row gives `2.01e-14`.
This rules out parameter injection as the source of the observed high-voltage
misfit.

The existing per-device archives do not share a curve order between nMOS and
pMOS. The earlier foundation implementation checked only the total 2,046-point
length, so it implicitly learned a polarity/geometry-dependent permutation as
well as the current map. The new cache explicitly canonicalizes every device
to `idvd@0.37,0.74,1.11,1.48,1.85` followed by
`idvg@0.01,0.37,0.74,1.11,1.48,1.85`. This is a structural correction, not a
voltage-specific weighting trick.

Following the user's diagnostic clarification, the controlled seven-vs-43
comparison uses uniform voltage weights. High-voltage errors are reported and
used as a separate diagnostic, not silently upweighted as the presumed fix.
An explicit 1x/2x/4x weighting option remains available only as a later
ablation if holdout results demonstrate a learning-allocation problem after
the parameter surface and curve layout have been tested.

### 2026-08-18 — prior-run high-voltage failure map

The prior foundation+real-FD output confirms that L1/W3 is one instance of a
broader pattern. Its regressions relative to the published card include
`idvd@1.48` 0.08781 -> 0.11576, `idvg@1.48` 0.10905 -> 0.13165,
`idvg@1.85` 0.10503 -> 0.12699, and `idvd@1.85` 0.10581 -> 0.12354.
Larger high-voltage failures occur on several pMOS cards: L0.35/W0.55
`idvg@1.48` 0.07492 -> 0.19558, L8/W0.84 `idvd@1.85` 0.04061 ->
0.10309, L0.5/W0.42 `idvd@1.85` 0.04521 -> 0.09320, and L2/W5
`idvd@1.85` 0.01092 -> 0.04703. This is therefore being evaluated as a
device-wide high-field failure mode, with L1/W3 retained as a named regression
test rather than optimized in isolation.

### 2026-08-18 — matched foundation and extraction implementation

Implemented `scripts/foundation_parameter_study.py` and shared components in
`src/cryoml/foundation.py`. Both parameter-count experiments use the same
four-layer 1,024-wide geometry-conditioned network, deterministic per-device
80/10/10 splits, per-point signed-log standardization, curve-normalized linear
current loss, optimizer, and early stopping. The only intended forward-model
difference is seven local parameter inputs versus the common 43-parameter
layout. Holdout reporting separates every curve and the 1.48/1.85 V group from
ordinary curves.

Method 1 uses SciPy two-point finite differences entirely on the frozen
surrogate. Its residual is pointwise and curve-normalized, so each included
curve has equal weight regardless of bias or point count; a weak `1e-4`
published-parameter prior stabilizes non-identifiable directions. The LHC data
provide diverse starts, and only completed endpoints are re-simulated in fresh
NGSpice. The saved finite-difference Jacobian column norms will expose which of
the expanded parameters the measured curve set can actually identify.

Method 2 maps the complete canonical measured I-V vector plus geometry directly
to bounded local parameters. It is trained with active-parameter supervision
and reconstruction loss through the frozen forward surrogate. Its single
predicted parameter vector is then re-simulated in NGSpice; both surrogate
reconstruction error and physical-card error are retained so inverse ambiguity
cannot be mistaken for simulator agreement.

The exact paper RRMS and frozen paper-card inclusion set remain the primary
comparison. A separate high-voltage RRMS includes every nonzero 1.48/1.85 V
curve, including any curve excluded by the paper's dynamic policy.

### 2026-08-18 — full expanded Monte Carlo launched

Command (one heavy job, ten workers):

```bash
PYTHONPATH=src .venv/bin/python scripts/pdk_gen_full_data.py \
  --out-dir data/processed/pdk_synth_full \
  --num-samples 10000 --workers 10 --seed 20260818
```

The first completed batch has 10,000/10,000 finite samples per card. nMOS
cards have 38 safe active dimensions and the first pMOS cards have 40; the
remaining common features are fixed only where no safe nondegenerate
device-specific perturbation exists. Generation remains in progress at this
log point.

### 2026-08-18 — pre-training L1/W3 representation diagnostic

As a model-independent check, the stored currents were scored directly against
measurement before fitting any neural network. The best real-NGSpice sample in
the complete 10,000-point seven-parameter LHC has paper RRMS `0.11258` and
high-voltage RRMS `0.12153`. The 43-parameter pilot, despite containing only
256 samples, already has a real-NGSpice card at paper RRMS `0.08887` and
high-voltage RRMS `0.06856`. This cannot be caused by loss weighting or
surrogate capacity: it is direct evidence that added physical degrees of
freedom materially improve representability for the named L1/W3 failure. The
full-device conclusion remains pending the completed 18-card run and matched
surrogate holdouts.

### 2026-08-18 — complete expanded Monte Carlo and surface diagnosis

The expanded generation completed in `2145.05 s` wall time. All 180,000 of
180,000 simulations are finite. The common surface has 43 parameters; nMOS
cards expose 38 safe active dimensions and pMOS cards expose 39 or 40 depending
on the native bin.

`scripts/pdk_parameter_surface_diagnostic.py` then scored both stored LHCs
directly, without a neural network or optimizer. The per-device best sampled
paper-RRMS oracle is diagnostic/nondeployable and is not a headline method.
Its all-device mean improves from `0.22967` in the seven-parameter surface to
`0.20678` in the 43-parameter surface. At those paper-optimal samples, mean
high-voltage RRMS is nearly unchanged (`0.07465` -> `0.07440`), showing that
additional degrees of freedom do not by themselves prevent the aggregate
objective from selecting a cross-bias tradeoff.

A separate high-voltage-only query of the same real-NGSpice samples improves
on all 18/18 devices with the expanded surface; its mean high-voltage RRMS is
`0.05239` (seven) versus `0.04415` (43). Thus the current diagnosis has two
parts: the seven-variable surface lacks useful high-field degrees of freedom,
and aggregate selection can still mask those improvements. The matched uniform
foundation training remains the controlled test. Its finite-difference method
uses equal per-curve point residuals, which penalize a large single-curve miss
without assuming a special voltage multiplier. An explicit voltage weighting
ablation is reserved for the case where the controlled holdout/extraction still
selects the masking tradeoff.

### 2026-08-18 — seven-parameter controlled run

The first training attempt exposed an off-curve normalization instability: a
near-zero synthetic current scale allowed an early random linear-current
prediction to create an unbounded loss. That checkpoint was stopped and never
reported. Cache schema 3 adds the same device-relative `1e-6 * peak` scale floor
used by the sensitivity audit, uses Huber loss for the normalized-linear branch,
and clips only physically impossible signed-log predictions in that branch.
The restarted run was stable and completed all 200 forward epochs. Its selected
synthetic validation loss is below `1e-6`, so measured-card limitations are not
being attributed to an obviously inaccurate seven-parameter forward emulator.

The initial FD implementation validated only optimized endpoints, which could
discard a better real-NGSpice LHC start when the surrogate objective left its
physical basin. The fixed method validates both the two-point-FD endpoint and
its starting card, plus the published fallback. The complete seven-parameter
fixed series is: nMOS `0.07882`, pMOS `0.34814`, combined `0.21348`,
all-device `0.22844`, and separate high-voltage RRMS `0.07433`. This is close
to the direct real-LHC oracle (`0.22967`) and the prior foundation result, as
expected.

The synthetic supervised IV-to-parameter network reached test parameter MSE
`0.08732` and synthetic reconstruction loss `2.26e-5`, yet its direct measured
predictions were strongly out of distribution. A global reconstruction phase
on all 18 measured I-V inputs through the frozen surrogate reduced its
curve-normalized reconstruction objective only from `0.9638` to `0.9051`.
Fresh NGSpice validation remains poor: all-device RRMS `1.08139` and
high-voltage RRMS `0.49319`. This is a negative result for the requested direct
autoencoder method on the seven-parameter box, not a substituted search result.

### 2026-08-18 — voltage-split comparison checkpoint

For this comparison, **high voltage** is the mean RRMS of the four nonzero
`idvd`/`idvg` curves at 1.48 V and 1.85 V. **Low voltage** is the mean RRMS of
the remaining curves admitted by the same frozen paper-card inclusion rule.
Each number below is first averaged within a device and then across all 18
devices, matching the paper-card all-device aggregation rather than allowing a
device with more admitted curves to dominate.

| Result | Low-voltage RRMS | High-voltage RRMS | Overall RRMS |
|---|---:|---:|---:|
| Confirmed paper cards | 0.41366 | 0.07943 | 0.27505 |
| Seven-parameter surrogate FD, final | 0.33823 | 0.07433 | 0.22844 |
| 43-parameter real-LHC paper-objective oracle | 0.30028 | 0.07440 | 0.20678 |
| 43-parameter real-LHC high-voltage oracle | 0.41991 | 0.04415 | 0.26377 |

The final seven-parameter FD method is 18.2% better than the confirmed paper
cards at low voltage and 17.0% better overall, but only 6.4% better on the
high-voltage split. More importantly, it does **not** fix the named nMOS
L=1 um/W=3 um high-voltage failure: that device changes from paper-card
low/high/overall `0.19853 / 0.10192 / 0.15989` to seven-FD
`0.10034 / 0.12495 / 0.11019`. The aggregate improvement is being supplied by
the low-voltage curves while this device's high-voltage result becomes 22.6%
worse.

The expanded physical surface changes the diagnosis. For nMOS L=1 um/W=3 um,
the best 43-parameter real-NGSpice sample under the ordinary paper objective is
`0.07721 / 0.04980 / 0.06624` for low/high/overall, cutting high-voltage RRMS
by 51.1% versus the paper card. Across all devices, however, ordinary aggregate
selection still realizes only a 6.3% high-voltage improvement. Selecting the
same stored physical samples by their high-voltage metric reaches `0.04415`
across all devices, 44.4% below the paper-card high-voltage RRMS, at the cost of
low-voltage RRMS `0.41991`. Therefore the new parameters fix the lack of
high-field representational capacity, but the completed 43-parameter learned
extractor is still required before claiming that the ML method itself fixes the
failure without a low-voltage tradeoff. Its matched training remains in
progress at this checkpoint.

### 2026-08-18 — residual-shape diagnosis and corrective protocol

The L1/W3 high-voltage miss is not just a scalar RRMS or thresholding artifact.
The signed residual was inspected along each 1.48/1.85 V sweep. A gain below
one means that a least-squares projection of the simulation onto measurement
has insufficient current amplitude; the endpoint ratio compares simulated and
measured current at the 1.85 V end of the sweep.

| L1/W3 curve | Paper RRMS | 7-param FD RRMS | 7-param gain | 7-param endpoint ratio | 43-param real-LHC RRMS |
|---|---:|---:|---:|---:|---:|
| `idvg@1.48` | 0.10905 | 0.13222 | 0.897 | 0.886 | 0.03213 |
| `idvg@1.85` | 0.10503 | 0.12754 | 0.900 | 0.896 | 0.03106 |
| `idvd@1.48` | 0.08781 | 0.11626 | 0.904 | 0.893 | 0.05177 |
| `idvd@1.85` | 0.10581 | 0.12380 | 0.925 | 0.896 | 0.08423 |

For all four seven-parameter curves, the residual changes sign across the
sweep: current is close to or above measurement in the first quarter and is
well below measurement in the last quarter. The upper-gate `idvg` curves have
too little transconductance, while the `idvd` curves are too flat at high drain
bias. A threshold-only correction would translate the turn-on and retain this
slope error. The best expanded real-NGSpice sample brings the endpoint ratios
to 0.981--0.995 and the fitted gains to 0.987--1.018. This is direct evidence
that the missing correction is primarily high-field transport/output-shape
freedom rather than a different scalar voltage weight.

The strongest relevant measured-surface sensitivities are consistent with
that shape diagnosis. `UA`, `UB`, `U0`, and `VSAT` control mobility and its
vertical-field degradation; `RDSW`, `WR`, and `PRWG` affect series resistance;
and `PCLM`, `PDIBLC2`, `DROUT`, `ETA0`, and `DSUB` affect saturation-region
output conductance and drain-induced behavior. `TOXE`, `TOXM`, `VTH0`, `K1`,
`DVT1`, and `NFACTOR` remain coupled electrostatic/current-scale controls. All
43 screened parameters remain inputs to the foundation model; the staged fit
below does not revert to a hand-selected seven-parameter model.

#### Planned fix

1. **Finish the uniform 43-parameter foundation as the controlled baseline.**
   Preserve equal per-curve training and the frozen paper inclusion set. Report
   ordinary and high-voltage surrogate holdouts independently. This establishes
   whether local surrogate derivatives, rather than forward interpolation, are
   the next limitation.

2. **Fit identifiable parameter combinations in stages.** Start from several
   real-LHC cards, the published card, and the direct inverse prediction. First
   fit electrostatic/low-field controls on low and middle-bias curves. Next hold
   those combinations near their solution while fitting the high-field
   transport/output group on the four 1.48/1.85 V curves. Finally unlock all 43
   parameters for a joint refinement. At each stage, compute the two-point-FD
   Jacobian SVD and optimize only numerically identifiable directions; keep its
   null space near the prior. This retains maximum relevant physical capacity
   without allowing dozens of unobservable parameters to drift arbitrarily.

3. **Replace single-average candidate selection with guarded Pareto
   selection.** Generate endpoints for the ordinary paper objective, the
   high-voltage objective, and a smooth worst-curve objective. Re-simulate every
   finalist in fresh NGSpice. Among those physical results, choose the lowest
   overall RRMS that does not materially regress any device's high-voltage mean
   or any individual 1.48/1.85 V curve versus its paper card. Use an absolute
   numerical tolerance only for ties; if no feasible candidate exists, record
   the failure and trigger local data acquisition instead of silently relaxing
   the guard. This is a constraint/selection rule, not an arbitrary voltage
   multiplier.

4. **Active-learn where surrogate gradients disagree with NGSpice.** Around
   the overall, high-voltage, and Pareto-knee endpoints, simulate bounded local
   perturbations along the retained Jacobian singular vectors. Add those cards
   to training and refit only if surrogate-versus-NGSpice per-curve RRMS or
   candidate ranking is materially wrong. This directly improves the local
   derivatives used by finite differences without spending another uniform
   180,000 simulations.

5. **Use the IV-to-parameter network as an initializer, not an unquestioned
   final extractor.** Forty-three BSIM parameters are not uniquely identifiable
   from these I-V curves. Train reconstruction in the identifiable Jacobian
   subspace, permit multiple valid parameter solutions, then pass its proposal
   through the same guarded FD refinement and fresh-NGSpice validation. A
   direct inverse result that reconstructs only through the surrogate is not a
   successful card.

#### Acceptance gates

The expanded ML extraction is considered a high-voltage fix only if all of the
following hold on fresh NGSpice simulations:

- all-device overall and high-voltage RRMS both beat the completed seven-FD
  values (`0.22844` and `0.07433`);
- no device has a material high-voltage regression relative to its paper card,
  and regressions are also checked curve by curve rather than hidden in the
  four-curve mean;
- nMOS L1/W3 high-voltage RRMS is at most `0.06` (the stored 43-parameter
  physical result demonstrates `0.04980` is reachable);
- the stretch target is all-device overall RRMS near `0.21` and high-voltage
  RRMS at most `0.06`, between the ordinary 43-parameter oracle (`0.20678`,
  `0.07440`) and high-voltage oracle (`0.26377`, `0.04415`); and
- every headline value comes from the physical simulator, with surrogate-only
  metrics reported separately.

### 2026-08-18 — full curve audit: the failure is not high-voltage-only

Every admitted curve from the completed seven-parameter FD extraction was
compared with its corresponding confirmed paper-card curve. A *material
regression* in this diagnostic means more than 0.01 absolute RRMS and more than
10% relative RRMS. This threshold is for counting failure modes, not for
changing the reported paper metric.

| Region | Curves | Paper mean | 7-param FD mean | Regressed | Materially regressed |
|---|---:|---:|---:|---:|---:|
| Below 1.48 V | 103 | 0.41077 | 0.33500 | 28 | 15 |
| 1.48/1.85 V | 72 | 0.07943 | 0.07433 | 34 | 18 |

High voltage has the higher regression rate (25.0% material, versus 14.6% for
lower voltage), but it is not the only problem. The clearest systematic failure
is `idvd@1.85`: 15 of 18 devices regress, and its mean RRMS changes from
`0.05215` to `0.07154`. Conversely, `idvd@1.48` improves on average
(`0.09525` to `0.07142`). Combining those as a single "high-voltage" number
therefore hides a bias-specific output-conductance failure.

Mid-bias failures are also present. `idvg@1.11` changes from `0.08120` to
`0.08232` on average and contains the single largest regression: pMOS
L0.35/W0.55 changes from `0.10263` to `0.22138`. That same geometry also
regresses at `idvg@0.74` (`0.06353` to `0.14973`) and `idvg@1.48`
(`0.07492` to `0.16129`), which is a transfer-shape failure spanning regimes,
not an isolated voltage endpoint. Other large mid-bias failures include pMOS
L8/W0.84 `idvd@1.11` (`0.57825` to `0.69614`) and pMOS L0.5/W0.64
`idvd@1.11` (`0.27570` to `0.34503`).

The masking mechanism is now more specific: enormous improvements on a few
already-high-error curves dominate the arithmetic mean. For example,
L1/W3 `idvd@0.74` improves by `0.624` absolute RRMS, while each of its four
1.48/1.85 V curves regresses by much smaller absolute amounts. This makes the
overall device RRMS improve even though the strong-current shape is worse.
Across all devices, the same effect is visible in `idvd@0.74`, whose mean
improves from `0.55305` to `0.22824`.

#### General-method correction

The loss must not create one group for each of these 18 geometries. That would
make the method depend on the evaluation list and would not demonstrate
generalization. The revised seven-parameter ablation will use only general
physical labels available for any transistor characterization:

- polarity (`nMOS` and `pMOS`), balanced 50/50 rather than weighted by how many
  geometries happen to be present;
- sweep family (`idvd` and `idvg`);
- bias regime (low, middle, and strong-current/high-field); and
- curve-level RRMS-like residuals before aggregation, plus a CVaR/tail term
  over examples and curve regimes so a small number of bad shapes cannot be
  erased by a large win elsewhere.

Conceptually, each example first receives one normalized RMSE per curve. Those
are averaged within general sweep/bias regimes, the regime losses are summed
with equal mass, and nMOS/pMOS losses are finally averaged with equal mass. A
tail-risk term penalizes the worst fraction of curve-regime errors without
using a device ID. The existing uniform objective remains the control.

Generalization will be tested by holding out complete geometries, rather than
only random Monte Carlo rows from every known geometry. Per-device and
per-curve tables remain diagnostic outputs on those held-out geometries, but
they do not define training groups. Likewise, the earlier paper-card guard is
retained only as an evaluation/acceptance report; it is not used as a
device-specific training loss.

The sequence is:

1. implement and test this hierarchical polarity/sweep/regime loss with the
   seven-parameter data and both requested extraction methods;
2. compare uniform versus hierarchical training on held-out geometries using
   low, middle, `idvd@1.85`, other high-voltage, tail, and overall RRMS;
3. carry the loss to the 43-parameter model only if the seven-parameter test
   reduces regime/tail failures without sacrificing geometry generalization;
   and
4. keep fresh-NGSpice validation as the final authority for extracted cards.

### 2026-08-18 — synthetic-data coverage audit

`scripts/pdk_training_data_coverage.py` now audits the exact 175 curves in the
frozen paper-card inclusion set. For every voltage point it measures whether
the measured current is outside the min/max of all 10,000 synthetic cards. It
also reports an optimistic *pointwise envelope bound*: measured points are
independently clipped into that envelope before RRMS is computed. This bound
does not have to correspond to any single physical parameter vector, so a
small value proves marginal range coverage but not joint curve coverage.

| Data | Region | Measured points outside min/max | Outside central 98% | Pointwise envelope-bound RRMS | Best stored sample, selected separately per curve |
|---|---|---:|---:|---:|---:|
| 7 parameters | all | 12.88% | 15.11% | 0.08458 | 0.12879 |
| 7 parameters | below 1.48 V | 16.12% | 18.37% | 0.13953 | 0.19013 |
| 7 parameters | 1.48/1.85 V | 8.25% | 10.45% | 0.00598 | 0.04104 |
| 43 parameters | all | 5.12% | 7.56% | 0.02978 | 0.07750 |
| 43 parameters | below 1.48 V | 6.00% | 9.41% | 0.04791 | 0.10999 |
| 43 parameters | 1.48/1.85 V | 3.86% | 4.91% | 0.00385 | 0.03102 |

Training-data coverage is therefore a real cause, but not uniformly. The
largest marginal mismatch is the off/near-off `idvg@0.01` regime: even with 43
parameters, 24.85% of measured points are outside the complete synthetic
min/max and the optimistic bound is RRMS `0.26575`. This explains why the
direct measured inverse input is out of distribution and why improving a few
very poor low-current curves can dominate the ordinary average.

The principal high-field miss is not explained by marginal current range. In
the 43-parameter data, only 1.85% of `idvd@1.85` measured points are outside
the min/max and its pointwise bound is only `0.00163`. Across all high-voltage
curves the bound is `0.00385`, far below the ordinary 43-parameter sampled-card
high-voltage result `0.07440`. The currents needed at individual voltage points
are present; the remaining problem is finding one parameter vector that gives
the complete shape while also fitting the other biases.

There is also a joint-manifold/data-density issue that a min/max audit cannot
detect. For example, 43-parameter `idvd@0.74` has only 0.36% of points outside
the envelope, but the best complete-curve stored sample is still RRMS
`0.11856`. Ten thousand independent Latin-hypercube samples are dense enough
for seven dimensions but extremely sparse in 38--40 active dimensions, and
independent parameter combinations spend simulations away from the correlated
low-dimensional regions that produce plausible measured shapes. Thus the
expanded data improve physical range but are not an efficient local design for
finite-difference extraction.

The data correction will therefore be adaptive rather than simply making the
uniform LHC larger:

- treat instrument floors, quantization, and cleaned zero prefixes as a
  censored/noise region instead of asking BSIM to reproduce individual
  off-state glitches;
- add local NGSpice samples around the measured-curve projection, Pareto
  candidates, and retained Jacobian singular directions;
- sample correlated identifiable parameter combinations instead of filling a
  40-dimensional box uniformly;
- retrain and check local surrogate derivative agreement against NGSpice; and
- use complete-geometry holdouts. The current random split within every one of
  the 18 geometries measures interpolation in parameter space but does not
  establish geometry generalization.

The full reproducible audit is saved at
`out/tables/training_data_coverage.json`.

### 2026-08-18 — visual I-V inspection

`scripts/plot_iv_diagnostics.py` overlays measurement, the paper card, final
seven-parameter FD card, and the best stored 43-parameter real-NGSpice card.
Each plot contains linear output and transfer curves, log-scale transfer
curves, signed shape residuals, and per-curve RRMS. The same diagnostic was
generated for all 18 geometries in `out/figs/iv_diagnostics/`; these form a
fixed visual regression set and are not device-specific training targets.

The visual inspection confirms and extends the scalar diagnostics:

- **nMOS L1/W3:** the seven-parameter card is visibly low at the high-current
  endpoints of both output and transfer curves. It is close around turn-on but
  becomes progressively low with increasing sweep voltage, so the failure is
  transconductance/output-slope shape, not only threshold offset. The
  43-parameter physical card largely restores the endpoint amplitude. The plot
  also makes the masking obvious: the paper card's very large `idvd@0.74`
  error collapses in the seven-parameter fit while all four 1.48/1.85 V bars
  become worse.
- **pMOS L0.35/W0.55:** the mismatch spans `idvg@0.74`, `1.11`, and `1.48`,
  with a turn-on/slope crossover followed by severe upper-sweep
  underprediction. This is not a high-voltage-only error. The log plot also
  shows a measured low-current floor and scattered/quantized points that the
  smooth compact model cannot reproduce; modest improvement of the enormous
  `idvg@0.01` RRMS masks degradation of several strong-current curves.
- **pMOS L0.5/W0.42 and L8/W0.84:** the output plots show a cross-bias scaling
  failure. The fit can be high for the 1.48 V curve while low for the 1.85 V
  curve, so a global current multiplier or threshold shift cannot correct both.
  Mobility degradation, saturation/output-conductance, and bias-coupled
  parameters must change together. Low-current output curves also contain
  visible quantization/noise, producing the jagged normalized residuals.
- **Longer pMOS L2/W5 and L4/W7:** the seven-parameter curves look close in an
  ordinary linear plot, but the 1.85 V output curve is consistently low in the
  saturation region. Their overall scores still improve because the very
  large off-state `idvg@0.01` bar shrinks. This confirms that the masking is
  present even where the strong-current visual miss is subtler than L1/W3.

The revised diagnosis is therefore three-part: (1) off-state measurement/data
coverage and noise modeling, (2) mid/high-field joint shape capacity and sparse
high-dimensional sampling, and (3) arithmetic-mean selection that lets a very
large improvement on one poor curve hide several smaller but physically
important regressions. The seven-parameter general-loss ablation must report
all three separately before its loss is promoted to the 43-parameter run.

### 2026-08-18 — seven-parameter general robust-loss gate

Before retraining a neural network, `scripts/seven_general_loss_ablation.py`
tested candidate objectives directly on the stored 10,000-card real-NGSpice
seven-parameter LHC. This isolates objective behavior from surrogate error. No
device identifier is used: the robust objective balances the four general
groups `idvd ordinary`, `idvd high`, `idvg ordinary`, and `idvg high`, then
mixes their mean with the worst 25% of individual curve errors at weight 0.25.

| Seven-parameter LHC selection | Overall RRMS | Low-voltage RRMS | High-voltage RRMS | Mean worst-curve RRMS | Material curve regressions vs paper |
|---|---:|---:|---:|---:|---:|
| Ordinary paper mean | 0.22967 | 0.33999 | 0.07465 | 1.29021 | 32/175 |
| Mean squared curve RRMS | 0.29032 | 0.37489 | 0.17308 | 0.99581 | 78/175 |
| Regime-balanced + CVaR | 0.23346 | 0.33287 | 0.09447 | 1.25923 | 40/175 |
| Minimax curve RRMS | 0.38754 | 0.46113 | 0.28398 | 0.84034 | 112/175 |
| High-voltage-only reference | 0.27191 | 0.42862 | 0.05239 | 1.50229 | 33/175 |

Raw robust aggregation fails this gate. It lowers the largest numerical curve
error, but high-voltage RRMS becomes worse (`0.07465` to `0.09447`) and the
number of material regressions increases. For pMOS L0.35/W0.55, the robust
selection reduces the low-voltage mean from `0.92453` to `0.87627` by attacking
the enormous off-state tail, while high-voltage RRMS nearly doubles from
`0.12417` to `0.23732`. For nMOS L1/W3 it changes high-voltage RRMS from
`0.12153` to `0.12631`. Minimax is worse still.

This result rejects a plain sum, squared sum, or worst-case loss over raw RRMS.
The largest raw RRMS curves are precisely those affected by off-state floors,
quantization, glitches, or inadequate synthetic coverage. Giving them more
tail weight pulls the physical card away from reliable strong-current shapes.
The next seven-parameter objective must first use a general measurement model:

1. estimate a noise/detection band from repeated/quantized low-current points
   and local non-monotonic variation, without using geometry identity;
2. treat measurements inside that band as censored (a prediction below the
   detection limit is not penalized for failing to reproduce a noise plateau);
3. use robust, smoothed current and slope residuals above the signal-to-noise
   threshold; and
4. only then apply polarity/sweep/regime balancing and a modest tail term.

The ordinary paper RRMS remains the final reported metric, but an unreliable
off-state curve will no longer dictate optimization merely because its
denominator makes its raw RRMS numerically large. This noise-aware seven-
parameter gate must succeed before the loss is used for 43-parameter training.

### 2026-08-18 — bounded/noise-aware objective result on seven parameters

The seven-parameter LHC objective gate was extended with two general
alternatives. `capped_regime` clips each curve's optimization contribution at
RRMS 0.25 before equally balancing ordinary/high output and transfer regimes.
This approximates censoring a curve once it is clearly noise/coverage limited,
while all reported scores remain the unchanged raw paper RRMS.
`attainable_regret` instead subtracts and normalizes by the best complete-curve
error found on the physical LHC, with a denominator floor of 0.02; this asks
whether a joint card is unnecessarily far from what the surface can attain on
each curve.

| Seven-parameter selection | Overall | Low voltage | High voltage | Material regressions vs paper |
|---|---:|---:|---:|---:|
| Ordinary paper mean | 0.22967 | 0.33999 | 0.07465 | 32/175 |
| Capped regime | 0.25357 | 0.39138 | 0.06064 | 26/175 |
| Attainable regret | 0.23595 | 0.35321 | 0.07106 | 32/175 |

The cap succeeds at preventing unreliable curves from masking high-voltage
errors: high-voltage RRMS falls 18.8% and six material curve regressions are
removed. It does not provide a no-tradeoff seven-parameter solution. Overall
and low-voltage RRMS worsen. For nMOS L1/W3, capped selection changes
low/high/overall from `0.10661 / 0.12153 / 0.11258` to
`0.21217 / 0.09334 / 0.16464`; the high-only seven-parameter endpoint can reach
`0.08381` but has overall RRMS `0.24061`. By contrast, the stored 43-parameter
physical card reaches `0.07721 / 0.04980 / 0.06624` on the same device. The
loss correction exposes and reduces masking, while the expanded physics are
required to remove the underlying cross-bias tradeoff.

The production fix will not use a hard RRMS cap as an unexplained tuning knob.
It will replace it with an estimated detection/noise band and one-sided
censored residual below that band, then use bounded pseudo-Huber current and
smoothed-slope residuals above it. Low/ordinary and high-field objectives will
form a Pareto set. Selection will minimize normalized excess from attainable
curve floors, computed for the new device from the physical/surrogate surface,
not from a stored device identity. Finalists are always rerun in NGSpice.

### 2026-08-18 — post-baseline noise-aware implementation prepared

The next experiment is implemented without modifying the still-running
controlled 43-parameter baseline. `src/cryoml/noise_aware.py` estimates a
detection band from each measured sweep using monotone smoothing, robust local
variation, and the appropriate initial region for output versus transfer
curves. It applies a one-sided censored loss below that band and bounded
pseudo-Huber current plus smoothed-slope losses above it. Aggregation uses only
general curve labels (`idvd`/`idvg`, ordinary/high), a bounded contribution,
and a bounded tail term. It contains no evaluation-device identity.

`scripts/noise_aware_foundation_extract.py` loads a frozen controlled
foundation checkpoint, performs numerical two-point finite-difference
optimization from ordinary LHC, high/robust LHC, direct-inverse, and published
starts, and re-simulates both endpoints and physical starts in fresh NGSpice.
The noise-aware objective selects the physical finalist; unchanged paper RRMS
and separate high-voltage RRMS remain the headline outputs. The direct inverse
result is retained as a same-checkpoint control. The script writes to a new
output directory and cannot overwrite the controlled run.

The detection audit is physically plausible on the current measurements:
1.48/1.85 V output curves retain about 99.3% of their points as signal, while
transfer curves censor the subthreshold/noise-floor portion and retain the
turn-on/strong-current portion. All 26 repository tests, including three new
noise-aware unit tests, pass. Per the requested ordering, execution waits for
the current 43-parameter baseline to finish, then runs seven parameters before
the revised 43-parameter extraction.

### 2026-08-18 — controlled 43-parameter foundation result

The controlled expanded run completed with the 43 screened BSIM4 DC
parameters and the same architecture, split, curve layout, and extraction
procedure used for the seven-parameter control. The frozen forward surrogate
reached validation loss `0.00011178` and test loss `0.00106326`; its high-bias
normalized-linear test component is only `5.43e-6`. The direct inverse remains
poorly identified: test parameter loss is `0.27538`, and measured-card
fine-tuning reconstruction remains `0.29559`.

| Card/method | Overall RRMS | nMOS | pMOS | Low <=0.74 V | Mid 1.11 V | High >=1.48 V |
|---|---:|---:|---:|---:|---:|---:|
| Published paper cards | 0.27505 | 0.11983 | 0.39924 | 0.55453 | 0.15727 | 0.07943 |
| Seven-parameter foundation FD | 0.22844 | 0.07882 | 0.34814 | 0.45098 | 0.13524 | 0.07433 |
| 43-parameter foundation FD | 0.20380 | 0.07044 | 0.31048 | 0.40165 | 0.10142 | 0.07602 |
| 43-parameter direct inverse | 0.52300 | 0.22639 | 0.76029 | — | — | 0.28774 |

The added parameters improve overall, nMOS, pMOS, low, and mid-bias means, but
do **not** improve the aggregate high-bias result versus seven parameters:
`0.07433` becomes `0.07602`. They do fix the named nMOS L1/W3 failure:
overall changes from `0.11019` to `0.06575`, and high bias from `0.12495` to
`0.05046`. The direct inverse/autoencoder method is still rejected as a card
extractor because its aggregate reconstruction is much worse than both paper
and finite-difference cards.

The per-device result is heterogeneous. The largest 43-versus-seven high-bias
improvements include nMOS L1/W3 (`-0.07450`), pMOS L8/W5 (`-0.0162`), and pMOS
L0.35/W1.6 (`-0.0142`). The largest high-bias regressions are pMOS
L0.35/W0.55 (`+0.0559`), pMOS L8/W0.84 (`+0.0286`), pMOS L8/W1.6 (`+0.0240`),
and pMOS L4/W7 (`+0.0234`). Thus parameter count fixes a capacity problem but
does not fix scalar-card selection.

Updated `scripts/plot_iv_diagnostics.py` now overlays the actual trained
43-parameter finite-difference result, rather than only the stored 43-D LHC
oracle. Visual inspection confirms two distinct cases. For nMOS L1/W3, the
43-parameter card restores high-current output and transfer slope, so the
numeric improvement is physical. For pMOS L0.35/W0.55, the total falls mainly
because `idvg@0.01` and `idvd@0.74` improve; the strong-current transfer
endpoints from 0.74 through 1.85 V visibly worsen. This is the same masking
mechanism at higher model capacity.

Reproducible aggregate/device/curve tables are generated by
`scripts/analyze_foundation_comparison.py` in
`out/tables/foundation_comparison.{json,md}`. Overview figures are in
`out/figs/foundation_comparison/`, and detailed measured-I-V overlays are in
`out/figs/iv_diagnostics_43_run/`.

### 2026-08-18 — bounded-tradeoff selector and polarity ablation

The initial noise-aware guard was non-inferior to the *published* card in
ordinary and high noise-aware loss. That was too weak relative to the already
better controlled foundation result and could still hide a large raw curve
regression. In the all-device seven-parameter run it improved high-bias mean
to `0.07104`, but overall worsened to `0.24044`; pMOS L0.35/W0.55 allowed
`idvd@0.74` to move from RRMS `0.3228` to `1.9737` while other regimes
improved. This candidate is rejected.

The replacement selector always includes the controlled foundation FD card,
all physical starts, and optimized endpoints, reruns finalists in NGSpice, and
uses the best real overall-RRMS candidate as its reference. A candidate may
spend `max(0.005 absolute, 5% relative)` RRMS in overall, ordinary, and high
regimes, but no included curve may spend more than
`max(0.05 absolute, 25% relative)`. Among eligible candidates it minimizes the
noise-aware balanced objective. These settings are global and the reference
is recomputed for an arbitrary input transistor; no device identity appears
in training or selection.

The revised two-device seven-parameter smoke test behaves as intended. nMOS
L1/W3 spends only `+0.00028` overall while improving high bias by `-0.00240`.
pMOS L0.35/W0.55 spends `+0.00160` overall while improving high bias by
`-0.01610`, and the catastrophic low-current endpoint is excluded. A
representative 43-parameter smoke test is running before the all-device run.

Because the controlled 43-parameter result is `0.07044` on nMOS but `0.31048`
on pMOS, a polarity-specialist ablation is also justified. Splitting nMOS and
pMOS is a general physical partition, unlike fitting one model per listed
geometry. The next diagnostic first evaluates the frozen shared surrogate by
polarity and curve regime, then compares shared and separate-polarity models
on identical samples, losses, and splits. Separate models will only be
promoted if the pMOS improvement survives the same card and curve tests.

The first frozen-model polarity diagnostic does **not** support shared-network
interference as the cause of poor pMOS cards. On the identical synthetic test
split, the shared 43-parameter surrogate has loss `2.36e-3` for nMOS but only
`2.47e-5` for pMOS. High-bias normalized-linear MSE is similarly small for
both (`6.81e-6` nMOS, `4.32e-6` pMOS). At extracted endpoints, mean absolute
surrogate-versus-real-NGSpice RRMS discrepancy is `0.00270` for nMOS and
`0.00323` for pMOS; high-bias discrepancies are `0.00340` and `0.00232`.
Thus the pMOS card gap is not a gross pMOS forward-emulation failure. A
separate-polarity training run remains an ablation, but it should not replace
the shared model unless it improves local derivatives and real extracted
cards. Current evidence instead prioritizes selection, identifiability, and
adaptive/correlated pMOS sampling. The diagnostic is reproducible with
`scripts/foundation_polarity_ablation.py`; output is stored in
`out/pdk_foundation_polarity_full/summary.json`.

### 2026-08-18 — correction: L1/W3 is not fixed across the full sweep

The earlier L1/W3 statement used whole-curve and high-fixed-bias aggregates.
The plotted `idvd@1.85` curve shows a more specific failure: the 43-parameter
card fixes saturation and the endpoint but is worse than the published card
below about 0.75 V drain sweep.  Splitting that curve into drain-sweep regions
gives the following unchanged reported RRMS:

| Card | Vd 0–0.37 | Vd 0.37–0.75 | Vd 0.75–1.48 | Vd 1.48–2.0 |
|---|---:|---:|---:|---:|
| Published | 0.1289 | 0.0607 | 0.1106 | 0.1070 |
| Seven-parameter FD | 0.1469 | 0.0674 | 0.1303 | 0.1298 |
| 43-parameter FD | 0.1533 | 0.0842 | 0.0252 | 0.0176 |

Thus “high voltage” alone is not the diagnosis.  This is a within-curve
cross-region tradeoff: at fixed `Vgs=1.85 V`, low/intermediate `Vd` is poor
while high `Vd` is excellent.  Whole-curve RRMS hides the location of the
error, just as all-device RRMS can hide a bad device.

This is also not primarily a forward-MLP error for this card.  The frozen
surrogate-versus-NGSpice discrepancy in the same four regions is only
`0.00065 / 0.00093 / 0.00283 / 0.00376`, whereas NGSpice-versus-measurement is
`0.1533 / 0.0842 / 0.0252 / 0.0176`.  The surrogate accurately predicts the
bad physical card selected inside the current search box.

The selected card has `rdsw` at its upper local bound (about `317 ohm-um`) and
`vsat` at a bound; `prwg` is inactive because its published family value is
zero and the original multiplicative range collapses to zero.  Direct
NGSpice probes establish causality rather than correlation: raising `rdsw` to
400 reduces the four regional errors to
`0.1157 / 0.0598 / 0.0521 / 0.0440`, better than the published card in every
region.  Raising it further to 500 continues improving the first region but
starts degrading the tail, demonstrating the need for correlated parameters
and a segmented objective.  More uniform random samples inside the old box
cannot discover the `rdsw=400` solution.

The implemented general correction splits every curve by normalized sweep
position `(0–0.2, 0.2–0.4, 0.4–0.7, 0.7–1.0)`, averages the segment losses,
and adds a bounded worst-segment term.  Final real-NGSpice selection now guards
each segment against the better of the best-overall candidate and published
card.  This contains no device name, geometry list, or special 1.85 V rule.
It exposes the existing bound limitation; a follow-up adaptive sampler must
expand bound-hit, physically relevant directions and retrain locally before
the low-sweep shape can actually be fixed.

### 2026-08-18 — prior inverse notebook used only as an audit reference

The supplied Keras notebook is not being copied.  It provides one useful
ablation hypothesis: retaining linear current alongside log current may help
the direct inverse recognize strong-current knees.  Its limitations are kept
out of this pipeline: random point-level splitting can leak near-duplicate
Monte Carlo curves, parameter MSE assumes an identifiable one-to-one inverse,
and the shown diagnostics accidentally subtract `y_pred[i][0]` from every
parameter.  A low parameter MSE is not evidence of a good reconstructed card.

The current inverse therefore has an optional input consisting of
curve-scale-normalized linear current concatenated with standardized signed-log
current.  It retains geometry and active-parameter masks.  Synthetic training
uses a weak parameter regularizer and makes reconstruction through the frozen
surrogate primary.  Reconstruction is curve- and sweep-segment-balanced, and
measured global fine-tuning uses the same equal segment allocation.  Existing
single-channel checkpoints remain compatible, making this a controlled
representation/loss ablation rather than a pipeline replacement.  Fourteen
focused foundation/noise-aware tests pass after the change.

### 2026-08-18 — completed pre-segmentation 43-parameter selector run

The all-device bounded-tradeoff run that was already in progress completed in
988.5 seconds.  It used the whole-curve guard loaded at process start, not the
later sweep-segment guard, so it is retained as a controlled intermediate
result:

| Card/method | Overall | nMOS | pMOS | Low <=0.74 V | Mid 1.11 V | High >=1.48 V |
|---|---:|---:|---:|---:|---:|---:|
| Published | 0.27505 | 0.11983 | 0.39924 | 0.55453 | 0.15727 | 0.07943 |
| Seven FD | 0.22844 | 0.07882 | 0.34814 | 0.45098 | 0.13524 | 0.07433 |
| 43 FD | 0.20380 | 0.07044 | 0.31048 | 0.40165 | 0.10142 | 0.07602 |
| 43 bounded tradeoff | 0.20219 | 0.06944 | 0.30839 | 0.41232 | 0.08656 | 0.06861 |

The selector improves overall and high-bias means over both controlled
foundation cards, and it reduces materially worse-than-paper individual
curves from 38 to 27 versus the 43 FD card.  It spends some low-bias accuracy
(`0.40165` to `0.41232`) while remaining substantially better than the paper
cards (`0.55453`).  However, its nMOS L1/W3 `idvd@1.85` low-sweep error is
essentially unchanged because the old `rdsw` ceiling remains.  This run
therefore improves aggregate selection but is not the claimed fix for the
within-curve failure.

The comparison table and all 18 IV overlays were regenerated after this run.
The aggregate source of truth is `out/tables/foundation_comparison.md`; the
figures are in `out/figs/foundation_comparison/` and
`out/figs/iv_diagnostics/`.  All 32 repository tests pass.

### 2026-08-18 — regional-loss smoke and retraining launch

The first L1/W3 extraction with the new segment guard found **zero feasible
candidates** inside the existing 43-parameter box.  It therefore retained the
controlled baseline (`overall 0.06575`, high-fixed-bias `0.05046`) rather than
claiming an aggregate improvement.  For `idvd@1.85`, its four normalized
sweep-region RRMS values are `0.1527 / 0.0806 / 0.0261 / 0.0191`.  This is the
desired failure mode: the first two regions are now visible and block
selection, while the result also demonstrates that a loss change alone cannot
cross the `rdsw` bound.

The regional loss is now used in forward-surrogate training as well as inverse
reconstruction and physical finalist selection.  Every curve is divided into
four normalized x-axis regions; each region contributes its own differentiable
RRMS plus standardized-log and robust curve-scale-normalized linear-current
terms.  Regions and curves are averaged equally, with an additional 25%
worst-region term.  This is uniform across ordinary/high biases and all
geometries.  In physical extraction, the section RRMS uses the unchanged
measured-curve denominator; a smaller noise-aware term stabilizes censored
regions.

At 2026-08-18 14:56 local time, a fresh seven-parameter training run was
started in `out/pdk_foundation_segmented_seven_rrms_bounded/` with the same four-layer
1024-neuron architecture, 200-epoch limit, batch size 512, seed, cache, and
split as the controlled model.  Command:

```text
PYTHONPATH=src .venv/bin/python scripts/foundation_parameter_study.py \
  --parameter-set seven --stage forward --device mps --segmented-loss \
  --epochs 200 --batch-size 512 --arch 1024,1024,1024,1024 \
  --linear-weight 1.0 --out-dir out/pdk_foundation_segmented_seven_rrms_bounded
```

Two short implementation checks were stopped and discarded before the final
run.  The first verified section Huber/MSE plumbing.  The second exposed an
RRMS training spike from synthetic curves with essentially zero mean current
and from physically impossible early signed-log predictions.  The final loss
keeps a literal independent RRMS per section but (1) floors its denominator at
`1e-3` of the per-device/per-curve 99th-percentile training scale, (2) applies
a pseudo-Huber loss to the section RRMS, and (3) caps only its *training-loss
contribution* at RRMS 5.  Uncapped section RRMS is still recorded and used in
final NGSpice tables/guards.  A 20k-sample audit shows the denominator floor
affects 12.0% of synthetic `idvd@0.37`, 1.12% of `idvd@0.74`, 0.88% of
`idvd@1.11`, and none of the other eight curve types.  The final run begins at
training loss `8.79` rather than the unstable `426`, and reaches validation
`0.479` by epoch 2.

The 43-parameter training will be launched with the identical configuration
after the seven-parameter run completes, so this remains a controlled nested
comparison.

Independent section reporting is implemented by
`scripts/analyze_sweep_segments.py`.  The current 700 admitted
device/curve/section comparisons show 115 material section regressions versus
paper for seven FD, 126 for 43 FD, and 103 for the pre-segmentation 43
tradeoff selector.  Aggregate 43 performance is better in every normalized
x-axis section, but the L1/W3 plot isolates the opposite result in its first
40%, confirming why both aggregate and per-curve tables are needed.  Outputs:
`out/tables/sweep_segment_comparison.{json,md}` and
`out/figs/foundation_comparison/sweep_segments.png`.

`scripts/foundation_bound_diagnostic.py` also records 168 selected-card
coordinates with `|local| >= 0.95`.  A bound hit alone is not evidence for
blind expansion because many parameters are non-identifiable; outward physical
probes must improve multiple sweep sections.  The repeated hits do show that
the original one-at-a-time +/-10% safe box is often acting as a search limit:
`rdsw`, `u0`, `ua`, `pdiblc2`, and `alpha0` each hit seven cards.  Separately,
four globally selected terms are fixed for every nMOS (`vfb`, `prwg`, `b0`,
`b1`) and two for every pMOS (`lpe0`, `lpeb`).  The sensitivity code now uses
the other polarity's nonzero median as an *absolute probe scale* when one
polarity is all zero, rather than silently declaring the term inactive.  This
will test relevance before the adaptive-data run and is general across
devices.

The follow-up physical audit is prepared in
`scripts/pdk_adaptive_boundary_probe.py` but is deliberately not running in
parallel with foundation training.  It takes only parameters that repeatedly
hit a bound, probes 2x and 4x the original half-span in fresh NGSpice, and
scores the same four independent x-axis RRMS sections.  A direction is
recommended for adaptive data only if it improves at least two distinct
devices in the same polarity without worsening any section by more than
0.02 RRMS.  This prevents blindly expanding 168 saturated coordinates and
turning non-identifiability into an even larger random box.

### 2026-08-18 — seven-parameter regional surrogate completed; 43 started

The final bounded-section-RRMS seven-parameter surrogate completed all 200
epochs in 3184.7 seconds.  The selected checkpoint has validation loss
`1.8331e-5`; its held-out synthetic test metrics are:

| Metric | Original point-average surrogate | Regional-RRMS surrogate |
|---|---:|---:|
| high normalized-linear MSE | 2.1224e-7 | **1.3673e-7** |
| ordinary normalized-linear MSE | 7.4956e-7 | **3.9167e-7** |
| standardized signed-log component | 6.3727e-7 | 4.5074e-6 |
| uncapped mean section RRMS | not recorded | 0.0011368 |
| complete regional training objective | not comparable | 1.8135e-5 |

The total losses are not directly comparable because the new objective adds
independent RRMS and a worst-section term.  The separately comparable linear
metrics improve in both ordinary and high regimes.  Signed-log MSE is larger,
which is an intentional trade toward curve shape in physical current; final
measured-card extraction will determine whether that trade is useful.

Immediately after completion, the 43-parameter surrogate was launched in
`out/pdk_foundation_segmented_full_rrms_bounded/` with the identical
architecture, loss, seed, batch size, and schedule.  Only the nested parameter
surface and its already-controlled cache differ.

The per-curve held-out audit shows that the seven-parameter improvement is
broad but not uniform.  Normalized-linear MSE improves for 10 of 11 canonical
curve families.  The largest improvements are `idvd@0.37` (regional/original
`0.379`), `idvd@0.74` (`0.392`), and `idvd@1.85` (`0.515`).  The exception is
`idvg@0.01`, which regresses by a factor of `1.367`.  Therefore the surrogate
result alone is not accepted as a model-card improvement; the next decision is
based on fresh measured-card NGSpice simulations and the unchanged reported
RRMS.  The generated audit is
`out/tables/surrogate_training_comparison.{json,md}` with plot
`out/figs/foundation_comparison/surrogate_training.png`.

A post-hoc checkpoint audit now recomputes held-out normalized-linear error
from the saved model by transistor polarity and by the same four x-axis
sections.  For seven parameters, nMOS MSE falls from `3.3558e-7` to
`1.8276e-7` (ratio `0.545`) and pMOS from `7.2687e-7` to `3.9185e-7`
(`0.539`).  Aggregate section ratios are `0.557 / 0.316 / 0.713 / 0.537`;
all improve.  The weakest subpartition is nMOS section 1 at ratio `0.847`, so
the measured L1/W3 low-sweep case remains the decisive check.  The surrogate
comparison plot now includes these ratios with a no-change line at 1.0.

The measured finite-difference extractor now accepts explicit
`--forward-checkpoint` and `--inverse-checkpoint` paths, validates their
parameter names and dimensions against the selected cache, and writes the
resolved sources into every result.  This closes a provenance gap that could
otherwise cause a regional extraction to silently load the old controlled
surrogate.  A dual-channel inverse checkpoint is also accepted by reading its
recorded input width; the inverse remains only a start/control for this FD
experiment.

The direct inverse protocol was also corrected for generalization.  Previously,
`--stage inverse` automatically fine-tuned one shared network on all 18
measured paper-card devices and then reported those same devices.  This is
transductive leakage even without a per-device network.  Measured fine-tuning
is now disabled by default and requires explicit `--measured-finetune` (or
explicit `--stage finetune`) as a labeled ablation.  The primary
IV-to-parameter autoencoder will be trained only on held-out-split Monte Carlo
curves, with geometry as an input and no paper-card current used for training.
Its new checkpoints record `paper_cards_used_for_training: false`.

The extraction CLI now supports `--extract-method inverse`, so the direct
method can be evaluated without rerunning an unrelated FD search.  The common
comparison table now includes the original direct controls: seven-parameter
inverse overall/high RRMS is `1.0814 / 0.4932`, while 43-parameter inverse is
`0.5230 / 0.2877`.  Both are substantially worse than their FD counterparts,
and the old runs used measured-card fine-tuning; they are controls, not a
generalization result.

`scripts/plot_iv_diagnostics.py` now accepts a comma-separated `--methods`
filter and knows both old and general inverse artifacts.  FD and inverse
overlays will therefore be generated separately: each retains measured points,
physical simulations, residual panels, log transfer curves, and per-curve
RRMS without turning one figure into an unreadable stack of line styles.

The adaptive-bound physical audit now writes a Markdown decision table and a
two-panel plot in addition to raw JSON.  The plot separates improvement in the
section-balanced objective from the worst individual section regression; green
groups satisfy the repeated-device recommendation rule.  This makes it
possible to reject a parameter expansion that lowers one aggregate while
damaging a local sweep region.

After the checkpoint-provenance, detailed-audit, and inverse-protocol changes,
the repository's `unittest` suite passed all 33 tests.

The adaptive-data path is now implemented but will not be activated until the
physical probe finishes.  `scripts/build_adaptive_parameter_bounds.py` accepts
only repeated-device recommendations, caps the outward multiplier, and applies
each supported parameter/polarity/direction to **every** device of that
polarity.  This avoids creating a special box for the individual failing
geometry.  `scripts/pdk_gen_full_data.py --sampling adaptive_mixture` then
retains 50% unmodified global Latin-hypercube rows, puts 35% in single newly
opened boundary slabs, and puts 15% in paired slabs to expose compensating
parameter interactions.  Non-focal coordinates retain stratified global
coverage.  Two tests verify polarity-wide application and boundary-slab
coverage; the full suite now passes 35 tests.

### 2026-08-18 — 43-parameter regional surrogate completed

The bounded-section-RRMS 43-parameter surrogate completed all 200 epochs in
3131.5 seconds.  Its selected validation objective is `2.1822e-4`; held-out
test high and ordinary normalized-linear MSE are `1.6392e-6` and `1.8805e-4`.
Relative to the original 43 surrogate, those comparable metrics improve by
69.8% and 77.6%.  Every one of the 11 canonical curve families improves;
`idvd@1.85` improves by a factor of `0.265`, and `idvd@0.37` by `0.219`.
All four normalized sweep sections improve with ratios
`0.342 / 0.392 / 0.496 / 0.196`.

This does **not** mean the 43-dimensional training problem is solved.  The new
seven surrogate's high/ordinary errors are `1.3673e-7 / 3.9167e-7`, so the
43 surrogate remains about 12x worse at high bias and 480x worse in the
ordinary partition.  The 43 polarity audit is also strongly asymmetric:
nMOS normalized-linear MSE is `2.6462e-4`, versus `4.7764e-6` for pMOS
(about 55x).  The loss fixed masking within the 43 model, but sparse coverage
and effective dimensionality still limit absolute forward accuracy,
especially for nMOS.  This supports a future nMOS-specific/capacity or active-
subspace ablation, while the measured NGSpice extraction determines whether
the current surrogate is already useful enough for physical search.

### 2026-08-18 — measured regional FD, seven parameters

The all-18-device seven-parameter regional extraction completed in 517.1
seconds.  It gives all-device overall/high RRMS `0.228403 / 0.074108`, versus
the controlled seven FD result `0.228444 / 0.074325`.  This is only a tiny
aggregate improvement.  nMOS L1/W3 retains the controlled physical start at
`0.110187 / 0.124954`; its `idvd@1.85` section RRMS is
`0.1453 / 0.0654 / 0.1293 / 0.1299`.  The section-aware loss detects the bad
shape and rejects worse endpoints, but the seven-parameter +/-10% surface does
not contain a safe better candidate.  Better surrogate training alone did not
fix this physical limitation.

The 43-parameter extraction was then launched with the same number of starts,
validated endpoints, finite-difference step, iteration limit, regularization,
section guards, measured curves, and scorer.  Only the parameter surface and
its corresponding regional forward checkpoint differ.

### 2026-08-18 — completed measured regional FD, 43 parameters

The 43-parameter regional extraction completed in `2058.0 s` over all 18
devices, using forward checkpoint
`out/pdk_foundation_segmented_full_rrms_bounded/forward_model.pt` and the
retained `pdk_foundation_vb_full` inverse as a control.

| Card/method | Overall | nMOS | pMOS | Low <=0.74 V | Mid 1.11 V | High >=1.48 V | Material curve regressions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Published | 0.27505 | 0.11983 | 0.39924 | 0.55453 | 0.15727 | 0.07943 | 0/175 |
| Seven FD | 0.22844 | 0.07882 | 0.34814 | 0.45098 | 0.13524 | 0.07433 | 33/175 |
| Seven regional | 0.22840 | 0.07873 | 0.34814 | 0.45113 | 0.13518 | 0.07411 | 33/175 |
| 43 FD | 0.20380 | 0.07044 | 0.31048 | 0.40165 | 0.10142 | 0.07602 | 38/175 |
| 43 bounded tradeoff | 0.20219 | 0.06944 | 0.30839 | 0.41232 | 0.08656 | 0.06861 | 27/175 |
| 43 regional | 0.20290 | 0.06968 | 0.30947 | 0.40320 | 0.09728 | 0.07429 | 34/175 |

The regional run beats the controlled 43 FD card on overall, mid, and high
means and on material curve regressions, but it does not beat the earlier
whole-curve bounded-tradeoff selector on overall or high aggregates.  Its
mean section RRMS is the best of all six cards in three of the four normalized
sweep sections (`0.06228 / 0.09192 / 0.16220 / 0.28536`), and its material
section regressions fall from 126/700 for 43 FD to 108/700.

The decisive observation is in the selector, not the aggregate.  With the
sweep-section guard active, **16 of 18 devices had zero feasible candidates**
and the run had to retain an already-validated physical start; only 5 of 18
accepted a new finite-difference endpoint.  The seven-parameter regional run
was blocked on 17 of 18 devices and accepted exactly one endpoint, which is
why its aggregate reproduces the controlled seven FD result to five decimals.
Under the earlier whole-curve guard no device was blocked and 10 of 18
endpoints were accepted.  The section guard is therefore working as designed
and is now almost always binding: inside the current +/-10% one-at-a-time box
the optimizer can lower a curve's aggregate error only by materially damaging
at least one normalized sweep section.

nMOS L1/W3 again retained the controlled baseline card (`0.06575` overall,
`0.05046` high) with `idvd@1.85` section RRMS `0.1527 / 0.0806 / 0.0261 /
0.0191`.  Its first two sections remain worse than the published card
(`0.1289 / 0.0598`), so the within-curve failure is still unfixed.

`scripts/foundation_bound_diagnostic.py` gained `--method` and `--manifest`
so any extraction summary can be audited, not only the original controlled
run.  Recomputed on the regional cards it finds 159 saturated coordinates
across 18 devices (168 for the controlled 43 FD cards), with the same
directions dominating: `a0` and `pdiblc2` at 7 devices each, then `delta`,
`eta0`, `nfactor`, `rdsw`, `u0`, `ua`, `vsat`, and `wr` at 6 each.

### 2026-08-18 — outward physical bound probe: the single-parameter rdsw fix is refuted

`scripts/pdk_adaptive_boundary_probe.py` ran 318 fresh NGSpice probes in
`7.2 s` (317 finite), centered on the selected regional cards at 2x and 4x the
original one-at-a-time half-span.  Only 42 of 318 probes improved the
section-balanced objective without worsening any admitted section by more than
`0.02` RRMS.

This directly corrects the earlier `rdsw` entry.  Raising L1/W3 `rdsw` to
`403.5 ohm-um` does reproduce the reported `idvd@1.85` regional improvement
(`0.1141 / 0.0594 / 0.0542 / 0.0464`), but the earlier check looked only at
that one curve.  Scored over the whole admitted curve set, the same change
worsens every transfer curve's upper sections — `idvg@0.01` by `+0.0923`, and
`idvg@0.74/1.11/1.48/1.85` by `+0.043` to `+0.046` — and raises the device's
unchanged reported paper RRMS from `0.06575` to `0.07544`.  The 2x probe is
also net worse (`0.06785`).  A single outward `rdsw` expansion is therefore
**not** a fix for the L1/W3 low-drain-sweep failure; the compensating
parameters have to move together, which is a correlated-direction problem
rather than a box-size problem along one coordinate.  `rdsw` is not recommended
for expansion in any polarity or direction.

A correctness defect in the recommendation rule was found and fixed before any
bound was widened.  The rule counted improving devices pooled over all probed
multipliers, and `build_adaptive_parameter_bounds.py` then widened the box by
the multiplier of the *best objective* probe, which may itself have failed the
section guard.  Under that rule, `ags`/pMOS would have been widened 4x on the
strength of a probe whose worst section regressed `+0.0538`, and `xj`/nMOS
would have been widened even though its two improving devices improve at
different multipliers while a third device regresses at both.  The rule is now
evaluated per probed multiplier: an expansion is supported at a multiplier only
if at least `min_hits` distinct devices improve *at that multiplier* and no
probed device of that polarity regresses there; the largest supported
multiplier is used.  The builder now refuses probe files that predate the rule.
Two regression tests cover both failure modes; the suite is 37 tests.

Under the corrected rule, four polarity-general expansions are supported:

| Parameter | Polarity | Direction | Supported multiplier | Evidence devices |
|---|---|---|---:|---|
| `ags` | pMOS | upper | 2 | L0.35/W0.55, L4/W7, L8/W5 |
| `k2` | pMOS | lower | 4 | L0.5/W0.42, L4/W7 |
| `lpeb` | nMOS | lower | 2 | L0.19/W7, L0.25/W1.6 |
| `prwg` | pMOS | upper | 2 | L0.5/W0.42, L4/W7 |

`xj`/nMOS/lower is now rejected.  The physical gains are real but small: the
largest is `ags` at 2x on pMOS L4/W7 (`0.18555 -> 0.18059` reported RRMS) and
pMOS L0.35/W0.55 (`0.62034 -> 0.61177`).  `prwg` is notable because it was
previously fixed inactive on nMOS for having a published value of zero; the
cross-polarity absolute probe scale makes its pMOS expansion testable.

`scripts/build_adaptive_parameter_bounds.py` applied the four groups to every
device of the relevant polarity, widening 38 device coordinates with no
degenerate (zero-width) expansion.  Tables:
`out/tables/adaptive_boundary_probe.{json,md}`,
`out/tables/adaptive_parameter_bounds.{json,md}`, and plot
`out/figs/foundation_comparison/adaptive_boundary_probe.png`.

The honest reading is that axis-aligned box expansion is a weak lever here:
only four of dozens of saturated directions survive a whole-device section
audit, and their gains are an order of magnitude smaller than the remaining
L1/W3 and pMOS errors.  The adaptive mixture's paired boundary slabs, which
sample two newly opened directions together, are the part of this stage most
likely to matter, and the Jacobian-singular-direction local sampling from step
4 of the planned fix remains the higher-value follow-up.

### 2026-08-18 — why the low-sweep region of nMOS L1/W3 `idvd@1.85` is unlearnable

Direct NGSpice probes around the selected regional card isolate the cause.  It
is not network capacity, not the loss, and not the surrogate: it is that the
sampling box excludes the parameter values that produce the correct shape.

`DELTA` (the Vdsat smoothing parameter) is the controlling knob.  Its published
value is `0.05`, the one-at-a-time +/-10% box is `[0.045, 0.055]`, and the
selected card sits at `0.0539`, near the ceiling.  Setting `DELTA = 0.08`,
roughly 48% beyond that ceiling, gives `idvd@1.85` section RRMS
`0.1059 / 0.0483 / 0.0530 / 0.0356`, which beats the published card
(`0.1289 / 0.0598 / 0.1106 / 0.1070`) in **all four** normalized sweep regions.
`AGS = 1.5`, about 53% below its box, reproduces the published card's regional
profile almost exactly.  The shape is therefore reachable in this compact
model; it is simply outside the box that generated all 10,000 training cards
per device.  A surrogate cannot learn a response it has never been shown, and
finite differences inside the same box cannot walk to it, which is why both the
`rdsw` and `vsat` coordinates report "at bound" on this card.

Inside the box only `rdsw` has authority in that region, and it is coupled to
every other curve: `rdsw = 400` improves the low-sweep sections
(`0.1533 -> 0.1157`) but adds `+0.0923` to `idvg@0.01` and `+0.043` to `+0.046`
to the four strong-current transfer curves, raising the device's reported RRMS
from `0.06575` to `0.07544`.

Two further structural holes were found on this device:

- `VSAT` is a dead coordinate for this bin.  Its effective published value is
  `-3.04`, which is unphysical for a saturation velocity.  Every negative value
  produces bit-identical curves, so NGSpice is clamping it and the parameter
  carries no gradient; setting a physical `1e5` raises the device's RRMS to
  `0.53369`, because the card was calibrated against the clamped behavior.  The
  primary physical control of the triode-to-saturation knee is therefore
  unavailable here, and the optimizer's "at bound" `vsat` is a null direction.
- `PRWG`, `B0`, `B1`, and `VFB` are frozen at exactly zero for every nMOS card
  because the box is multiplicative around a published zero.  Unfreezing has
  real effect: `B0 = 1e-6` moves the mean of the first two `idvd@1.85` sections
  from `0.1193` to `0.1111`, and `PRWG` monotonically controls the upper
  sections of both `idvd@1.85` and `idvg@1.85`.

Finally, even a reachable fix is currently vetoed by an unfittable curve.  The
`DELTA = 0.08` card regresses `idvg@0.01` by `+0.1118`, and that is the guard's
binding constraint.  `idvg@0.01` is the off-state sweep whose measured points
are at the instrument floor: the coverage audit already showed 24.85% of its
points lie outside the entire 10,000-card synthetic envelope with an optimistic
pointwise bound of RRMS `0.26575`.  It is also 35.9% of the summed curve error
across all 18 devices; excluding it, the 43-parameter regional all-device mean
falls from `0.20290` to `0.09952` (published: `0.27505` to `0.14617`).  A curve
nothing can fit is therefore blocking cards that are physically better
everywhere else.

#### Corrected fix order

1. Widen the knee family (`DELTA`, `AGS`, `A0`, `A2`) using physical absolute
   ranges rather than +/-10% multiplicative spans, and probe them in pairs,
   since single-axis moves are vetoed by other curves.
2. Give parameters published as zero an absolute half-span so `PRWG`, `B0`,
   `B1`, `VFB` (nMOS) and `LPE0`, `LPEB` (pMOS) stop being structurally frozen.
3. Detect and report clamped/dead coordinates per device, such as `VSAT` on
   this bin, instead of letting the optimizer consume search dimensions on
   them; a missing physical control is a finding, not a tuning failure.
4. Regenerate local, correlated training data inside the widened region along
   retained Jacobian singular directions and their pairs, then retrain.
5. Censor the detection-floor region in the objective and the guard, keeping
   the reported paper RRMS unchanged and adding a signal-limited companion
   value, so `idvg@0.01` can no longer veto a physically better card.

The adaptive generation running at this log point expands `ags`/pMOS,
`k2`/pMOS, `lpeb`/nMOS, and `prwg`/pMOS.  None of these is the binding
direction for this nMOS failure, so that run remains a valid control and
paired-slab dataset but is not expected to fix L1/W3.  The corrected bounds
above are the next generation run.

All 18 measured-I-V overlays for the completed regional comparison are in
`out/figs/iv_diagnostics_regional/`, showing measurement, the published card,
the seven-parameter regional card, the controlled 43-parameter FD card, and the
43-parameter regional card, with signed shape residuals and per-curve RRMS.

### 2026-08-18 — simplest fix: put the required values inside the training box

Confirmed the diagnosis before changing anything.  The frozen surrogate is
accurate where it was trained: on the exact `idvd@1.85` sweep regions in
question its predicted-versus-NGSpice discrepancy is `0.00065 / 0.00093 /
0.00283 / 0.00376`, while NGSpice-versus-measurement is `0.1533 / 0.0842 /
0.0252 / 0.0176`.  It faithfully reproduces a bad card.  The missing ingredient
is training coverage: every one of the 10,000 cards per device was drawn with
`DELTA` in `[0.045, 0.055]`, so the `DELTA ~ 0.08` behavior that fixes the
region was never in the training set and neither the surrogate nor its finite
differences could reach it.

The applied fix is deliberately minimal.  `scripts/build_adaptive_parameter_bounds.py`
gained `--coverage-expansion name:multiplier`, which widens named parameters
symmetrically on every device of every polarity.  It is recorded separately
from the probe-evidence groups, because widening the box is a statement about
what the surrogate is allowed to see, not a claim that the wider values improve
a card; the unchanged paper scorer and the section guard still select the card.
Two parameters were widened 8x, both with direct physical evidence on this
card: `DELTA` (Vdsat smoothing) and `AGS` (bulk-charge gate-bias dependence).
For nMOS L1/W3 the box becomes `DELTA [0.01, 0.09]` and `AGS [0.687, 6.185]`,
so the demonstrated `DELTA = 0.08` and `AGS = 1.5` are now inside it.  The four
probe-supported evidence expansions are retained, giving 38 evidence
coordinates and 72 coverage coordinates.

A uniformly larger box would dilute sampling density near the published card,
which the coverage audit already flagged as a problem in 38--40 active
dimensions.  The newly opened coverage regions are therefore also registered as
adaptive-mixture slabs: 50% of rows keep the original global Latin hypercube,
35% resolve one newly opened region, and 15% resolve two jointly, so `DELTA`
and `AGS` are sampled together.  That paired sampling is what the earlier
single-axis probes showed to be necessary, since a single-coordinate move is
vetoed by the other curves.

Bounds construction reused `min`/`max` on both sides so parameters with
negative published values, such as `AGS` on pMOS L0.35/W5, widen in the correct
direction; a regression test covers that case.  The suite is 40 tests.

The superseded `ags`/`k2`/`lpeb`/`prwg`-only generation was stopped and its
partial output removed, since the widened box is a strict superset of it and
the resource policy allows only one heavy job at a time.  Generation of the
widened 10,000-sample, 18-device cache started at 18:14 local time:

```bash
PYTHONPATH=src .venv/bin/python scripts/pdk_gen_full_data.py \
  --sensitivity out/tables/adaptive_parameter_bounds.json \
  --sampling adaptive_mixture \
  --out-dir data/processed/pdk_synth_full_widened \
  --num-samples 10000 --workers 10 --seed 20260818
```

The forward surrogate retrain and measured re-extraction follow it in sequence,
with the same architecture, loss, seed, split, and section guard, so the only
intended difference from the completed 43-parameter regional run is the
training box.

### 2026-08-19 — final result: widened box plus low-drain constraint

Two changes carried the 43-parameter method past the seven-parameter one.

**Sampling box.** Direct NGSpice probes showed the values needed for the
triode-to-saturation knee lie outside the one-at-a-time +/-10% box: nMOS L1/W3
needs `DELTA ~ 0.08` against a `[0.045, 0.055]` box and `AGS` far from its
published value. `DELTA` and `AGS` were widened 8x on every device of both
polarities and the cache regenerated. This is training coverage, not a claim
that the wider values improve a card. Scored directly on the stored physical
samples, the attainable per-device oracle improved on 17 of 18 devices, mean
`0.20678 -> 0.18414`.

**Selection.** The measured low-drain triode region (`|Vd| < 0.6 V` on `idvd`
curves at `|Vg| >= 1.48 V`) was worse than the published card on 12 of 18
devices, and on 7 of 8 nMOS, while pMOS improved. The one structural
difference is `PRWG`, the gate-bias coefficient of series resistance: every
pMOS card carries `0.052`, every nMOS card carries exactly zero, so the
multiplicative box pins it to a point.

Two objective changes were tried and both failed, on evidence:

- weighting sweep sections by absolute RRMS chases the numerically largest
  sections, which are transfer-curve upper ends, not the triode region; the
  all-device mean went `0.18232 -> 0.20161` and low drain `0.0585 -> 0.0813`;
- weighting sections by their ratio to the published card is degenerate. The
  published card scores exactly 1.0 in every section, so any card better
  somewhere and worse elsewhere loses on the worst-ratio term. Tested on the
  stored samples it simply reselects the published card.

The rule that works is a hard constraint rather than a weight: discard any
candidate whose low-drain RRMS exceeds that device's published card, then
select among survivors by the existing rule. A matching start ranking
(`lhc_low_drain_*`: best overall RRMS subject to the same constraint) is
required, or the candidate pool never contains a qualifying card.

| Card | All-device RRMS | High bias | Low drain | Above knee | Worse than paper, low drain |
|---|---:|---:|---:|---:|---:|
| Published | 0.27505 | 0.07943 | 0.0792 | 0.0677 | - |
| Seven-parameter | 0.22840 | 0.07411 | - | - | - |
| 43-parameter, widened box | 0.18232 | 0.06005 | 0.0585 | 0.0519 | 12/18 |
| 43-parameter, constrained | 0.18565 | 0.05811 | 0.0495 | 0.0506 | **0/18** |

The constraint costs `+1.8%` all-device RRMS and no device regresses against
its published card in the low-drain region. High-bias and above-knee RRMS
improved rather than paying for it. Split by family, all-device RRMS is
`0.07157` nMOS and `0.27691` pMOS; low drain is `0.0442` nMOS and `0.0538`
pMOS.

Reproduce with `scripts/pdk_parameter_sensitivity.py`,
`scripts/pdk_adaptive_boundary_probe.py`,
`scripts/build_adaptive_parameter_bounds.py --coverage-expansion delta:8,ags:8`,
`scripts/pdk_gen_full_data.py --sampling adaptive_mixture`,
`scripts/foundation_parameter_study.py --parameter-set full --stage forward
--segmented-loss`, then `scripts/noise_aware_foundation_extract.py`. Report
with `scripts/low_drain_diagnostic.py` and
`scripts/analyze_foundation_comparison.py`; figures and deck come from
`scripts/make_paper_style_plots.py` and `scripts/make_paper_style_slides.py`.

Sweep-region primitives are shared in `src/cryoml/regions.py` so the section
edges and the low-drain definition cannot drift between the loss, the guard,
the diagnostics and the figures.
