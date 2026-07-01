# Methods

## Simulator

The sole supported simulator is the NGSpice flow from
`ogzamour/CryoSkywater130nm_CorrectedForNgspice`. It uses the paper's
published 77 K model cards with the specified Volare SKY130 PDK revision.
Instances use native geometry-bin selection and the corrected-repository deck
convention. When model-bin ranges overlap, the pipeline tags the candidate
bins and asks NGSpice which one it selects; it does not choose a bin by score.

## Metric

The sole supported score matches the paper companion `RMS_functions.ipynb`:

```text
RRMS_k = sqrt(mean((I_sim - I_meas)^2)) / mean(abs(I_meas))
RRMS_device = mean_k(RRMS_k)
```

An all-zero measured curve receives `RRMS_k = 0.0`, matching the companion
notebook. Every measured curve is included in the device mean.

## Baseline

`scripts/pdk_baseline.py` runs the published parameter cards without fitting.
The resulting NGSpice score is the baseline used for extraction comparisons.

## ML Extraction

`scripts/pdk_gen_data.py` samples seven-parameter vectors and simulates them in
the device's native geometry bin.

`scripts/pdk_ml_extract.py` trains two networks per device:

- an emulator from parameter-space `z` to signed-log current curves;
- an inverse MLP from signed-log current curves to `z`.

The emulator supports batched gradient search. Candidate parameter vectors are
scored in real NGSpice, then the strongest candidates are polished with
finite-difference least squares. Candidate selection and final reporting use
the paper-exact metric. Devices that NGSpice maps to the same model bin are
jointly polished and receive one common final parameter vector.

### Tandem multi-head parameter predictor

`scripts/pdk_mlp_tandem.py` is a standalone inverse-MLP experiment. One MLP
emits 2,048 region-specialized candidate parameter vectors. Its output layer
is calibrated on the measured curves through a frozen forward emulator, then
the strongest candidates are validated in real NGSpice and finite-difference
polished.

Unlike the production ML pipeline, this experiment uses no classical warm
starts. A development run and two predeclared confirmatory seeds score
**0.4943**, **0.4957**, and **0.4957** mean RRMS over all 18 devices,
respectively, each beating the budget-matched 8,500-evaluation CMA control's
**0.4981**. The confirmatory-run mean is **0.4957**. Results are in
`out/pdk_mlp_tandem_full` and `out/pdk_mlp_tandem_seed{1,2}`.

```bash
python scripts/pdk_mlp_tandem.py --device mps \
  --out-dir out/pdk_mlp_tandem_full --resume
python scripts/verify_dl_no_cherry_pick.py --runs \
  out/pdk_mlp_tandem_full out/pdk_mlp_tandem_seed1 out/pdk_mlp_tandem_seed2
```

The verifier rejects missing devices, mixed non-seed configurations, duplicate
seeds, any final stage other than `tandem_best+fd`, saved-simulation score
mismatches, a classical comparator other than the full 8,500-evaluation
control, missing/non-deep forward emulators, or any confirmatory seed that
does not beat the classical full-set mean.

## Controls

`scripts/pdk_fd_extract.py` performs multistart finite-difference extraction.
`scripts/pdk_cma_extract.py` performs CMA-ES search followed by finite-
difference polish. Both use the same simulator, geometry bin, seven parameters,
and metric as the ML extractor.
