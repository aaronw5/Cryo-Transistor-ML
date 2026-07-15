# Goal

Reproduce the paper authors' confirmed 77 K SKY130 simulation and scoring
workflow, then extract the paper's seven BSIM4 parameters with a
parameter-to-I-V surrogate used inversely. Report the same device/table forms
as arXiv:2604.21625v1, reproduce its I-V plots with measured and predicted
curves, characterize scaling, and export a usable cryogenic model library.

## Scientific Requirements

- Pin `CryoPDK_Skywater130nm_ML@39b1e518` and conda-forge ngspice-41.
- Use all 18 Table-6 devices and their native geometry bins.
- Tune only `VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0`.
- Generate 10,000 Latin-hypercube samples per device in the published
  parameter vector's +/-10% box.
- Validate every reported parameter vector in real NGSpice.
- Score with the faithful upstream `rrmsCalc.py` behavior.
- Freeze curve inclusion to the published-card baseline for optimization and
  primary comparison; retain the dynamic official score as an audit.
- Report family-combined and all-device aggregates separately.

## Fixed Comparisons

The following are three distinct fixed series. Each must be run and reported
across all 18 devices:

1. Surrogate inverse search, raw.
2. Surrogate inverse search + FD polish.
3. Direct MLP forward pass from I-V features to parameters, with no search or
   FD polish.

Do not label a per-device best-of as "the ML method." Figures, tables,
headline RRMS, and the exported library must use fixed series. The shipped
library uses surrogate search + FD across every device/bin.

A separate FD study must compare published-card parameters -> FD alone and
surrogate raw -> surrogate + FD. It reports per-device RRMS improvement,
optimizer effort, and movement of all seven parameters. A second paired
ablation must apply the same measured-data FD stage to the fixed direct-MLP
prediction, without replacing the raw MLP in the primary comparison.

## Final Result Status (2026-07-15)

| fixed method | combined RRMS | all-device mean | wins vs cards |
|---|---:|---:|---:|
| published cards, confirmed NGSpice flow | 0.2595 | 0.2751 | - |
| direct MLP forward pass, no FD | 0.2579 | 0.2723 | 11/18 |
| surrogate search, raw | 0.2204 | 0.2357 | 17/18 |
| **surrogate search + FD** | **0.2140** | **0.2290** | **18/18** |
| foundation surrogate + FD (exploratory) | 0.2114 | 0.2261 | 18/18 |
| high-voltage guarded (diagnostic) | 0.2175 | 0.2327 | 18/18 |

All rows are complete fixed-series results. The obsolete inverse-network rows
were removed from current reporting. The canonical export remains the uniform
per-device surrogate+FD series; foundation and guarded cards are reported but
do not replace it.

FD uses fresh NGSpice perturbations and residuals against measured data. From
the published card it gives `0.2751 -> 0.2751` with 0/18 wins. From the same
raw surrogate winner it gives `0.2357 -> 0.2296` with 18/18 improvements; the
production top-five policy reaches `0.2290`. This establishes FD as local
polish, not a standalone global extractor under this recipe.

From the fixed direct-MLP prediction, the paired FD ablation gives
`0.2723 -> 0.2328` all-device and `0.2579 -> 0.2183` family-combined, with
18/18 improvements in `373.5 s`. The raw MLP remains the one-pass primary row;
the polished MLP is shown only in the dedicated FD study/slide.

The final scaling result covers five complete training-data configurations for
all 18 transistors (90 cells). It shows the arithmetic all-device mean in bold
and every transistor as a faded colored trace. Emulator test MSE improves by
about 9.5x from 375 to 6,000 samples, but final real-NGSpice+FD RRMS is flat;
the unfinished all-device capacity/search grid was therefore stopped, while
its completed four-device pilot remains labeled as such. After all primary
outputs are complete. The exploratory geometry-conditioned foundation emulator
was trained once across all 18 datasets and reached `0.2261` after FD in
`1200.7 s`, versus `0.2290` and `2420.2 s` for the 18 per-device emulators.
This tests reuse on known training geometries, not unseen-geometry
generalization.

The pMOS `L=2 um, W=5 um` per-voltage diagnostic shows that separate
nondeployable cards can reduce its included-curve mean from `0.1907` to
`0.1078`. A high-voltage-preserving selection study keeps the official RRMS
unchanged and improves the strong `idvd@1.85` curve while retaining an 18/18
overall win. It is a diagnostic constraint, not a replacement metric.
