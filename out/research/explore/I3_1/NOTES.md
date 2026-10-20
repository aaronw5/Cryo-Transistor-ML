# I3-1 notes

Follow-up of E3b (rule output CONTINUE on tier M). Pre-registration:
`prereg.json` (commit c877413) and `prereg_addendum.json` (hyb inputs and output
penalty, written before any NGSpice run of the new arms).

## Design summary

- New primary arm `hyb` (O4 only): per-device par3 law fitted on the O4 anchors
  plus a population-shared network residual, output layer initialised to 0,
  output penalty 0.01 mean(NN^2), early stopping that can return the
  zero-correction state.
- Each ensemble member trains two networks and keeps the one with the lower
  validation criterion: inputs `geo_t` (geometry, T) and `geo_t_res`
  (geometry, par3 residuals at the four anchors, T).
- `hyb_mean`: par3 plus the population-mean residual per device type and
  temperature (no network).
- Training par3 fits: stage `train_law` fits par3 on each training device's
  O4 anchors (truth re-simulated at 77/150/225/300 K from the stored truth).
- Tier H: piecewise-linear law with a kink at Tkink_k in [90, 260] K plus a
  5 K wide freeze-out step at Tf in [90, 140] K on vth0 (+0.02 to +0.06) and
  u0 (-0.20 to -0.05); no extrinsic resistance. Definition in `prereg.json`
  `world_and_protocol.tier_H`.
- Decision statistic: GMR(a, b) = exp(mean log((a + 0.01) / (b + 0.01))) over
  test devices, percentile bootstrap 95% CI (4000 resamples, seed 0), wins.
- Scaling arm (tier M, reported): lrn and hyb with 3, 6, 12 training draws per
  geometry.

## Findings before any run of the new arms

1. Unit test with synthetic data (no NGSpice): with the originally
   pre-registered hyb inputs (geometry, anchor fits, par3 parameters, T; 53
   inputs) and 16 training devices, the training loss fell to 1e-6 and the
   device-level validation loss rose from 0.028 (zero correction) to 0.11;
   early stopping returned step 25 and the correction did not depend on T.
   With T as the only input the validation loss was 6e-8. With geometry plus
   par3 anchor residuals plus T (60 devices) the validation loss was 2e-7
   (shared residual) and 2e-6 (geometry-dependent residual) at steps
   2900-2975.
2. The E3b `lrn` ensemble members stopped at steps 25-100 of 3000 (tier M,
   O4) and 25-75 (tier W, O4); see
   `out/research/explore/E3b/runs/full/tier_*/learn_info.json`. The E3b `lrn`
   network was therefore trained for at most 100 steps in every member, which
   is consistent with the small lrn vs lrn_mean difference (reduction 0.033,
   CI -0.154 to 0.161).
3. Output penalty 0.1 mean(NN^2) gave about half of the residual as
   correction in the synthetic test; changed to 0.01 in the addendum.
4. Tier H streams (seed 20261019, 12 REAL_DEV geometries, 6 training and 3
   test draws): 108 truths, 10 rejected draws for the u0 unit switch, 5
   truths affected; no accepted truth reaches the switch (unit test).

## Smoke run (pipeline check, not a decision input)

`runs/smoke/tier_M`: tags nmos_L1_W3, pmos_L2_W5; n_train 2, n_test 1; dense
77/110/150/190/225/260/300 K; law_iter 3; learn_seeds 1; workers 2; scaling
1,2. Wall time 123 s (an I3-2 queue job with 4 workers ran at the same time).
The existing arms reproduce the E3b smoke values (O4 par3 0.6972, lrn 0.3886,
lrn_mean 0.3737, pt_interp 0.7283). New arms: O4 hyb 0.6958, hyb_mean
0.6409, hyb_n1 0.6441, lrn_n1 0.6440 (held-out all-device mean, 2 test
devices). u0 check valid; 0 clipped cards. The hyb member chose `geo_t` with
best step 0. The `--decide` path was run on copies of the smoke summary and
removed.

## Queue jobs and wall-time estimate

E3b tier runs: 910 s (M) and 989 s (W) with 8 workers; train stage 529-568 s
for 72 devices. Estimates with 8 workers and no other heavy job:

| job | estimate |
|---|---|
| I3_1_full_M (144 training devices, scaling arms) | about 27 min |
| I3_1_full_W | about 18 min |
| I3_1_full_H | about 18 min |
| I3_1_decide | under 10 s |
| total | about 63 min (range 55-75 min) |
