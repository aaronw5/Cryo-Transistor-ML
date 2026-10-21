# I8-1 pre-registration amendment A1 (2026-10-20)

Amends `prereg.json` (in force, d022c32). Written after the prediction commit 0c184f8 and before any held-out outcome existed (I8_1_lodo stopped before writing output). Machine-readable: `prereg_amendment_2026-10-20.json`.

## Finding

4 of 57 DEV starting-card records have a NaN Id-Vg curve at Vd = 0.1 V (idvg@d0); the other 8 curves are finite.

| device | bin (forced) | stretch L | stretch W | stock effective PCLM |
|---|---|---|---|---|
| nfet_01v8_lvt_W0p55_L0p25_m20p7 | 0 | 1.389 | 1.636 | -0.338647 |
| nfet_01v8_lvt_W3p01_L0p15_m25p1 | 1 | 1.2 | 1.678 | -0.130636 |
| nfet_01v8_lvt_W7_L0p15_m26p2 | 1 | 1.2 | 1.0 | -0.029894 |
| nfet_01v8_lvt_W7_L2_m26p7 | 4 | 2.0 | 2.333 | -0.121279 |

Cause: all 4 are bin-uncovered nMOS geometries with a forced nearest bin (decision D2). For these geometries the binned size-dependent PCLM (pclm + lpclm/L + wpclm/W + ppclm/(L W)) of the forced bin is negative, and BSIM4 4.8.2 parameter checking stops with 'Fatal: Pclm = ... is not positive' at the first analysis of the ngspice process. ngspice writes no output for that first analysis (the first sweep in the deck, idvg@d0) and runs the later sweeps. The stock card has the same failure, so the polish objective carried the constant penalty 10 for that curve from the start (objective histories end at 1.67-1.70 = 10/6 + the other curves) and the fit did not leave the region.

Consequence: e1/i3_2 fit_bias trains on every record of a training set; a record with NaN simulated points makes the training loss NaN and every trained correction NaN. In prediction.json (commit 0c184f8) S3_D, every S3 inner value and S3_fam are null; P_c ('no gain') and gate G ('C0_refit') follow only from NaN < 0 being false. The S1 statistics are finite but include the 4 records: i7_1_transfer.device_residual drops non-finite points, keeps the records' finite curves, and divides the point weights by a curve count that includes the NaN curve.

## Rules

- A1_exclusion: a DEV device whose starting-card record has any non-finite simulated point (a failed start, counted in V1) is excluded from every training set (S1_lodo, S1_pooled, S2, S3, M_bias, T1_inloop, B3_d3, the B1 screen) and has no fold in the S statistics (S_D = mean over the folds of the 53 devices with finite records); it is not refit and is reported as a failed device (NaN) in every arm.
- R2: the means in R2 are taken over the scored devices with a finite starting record (the all-device mean is NaN when any device failed).
- R3: R3 counts M_bias failures among the devices with a finite starting record; start failures are counted by V1.
- V2: failed devices per arm include the start failures (4 of 57 = 7%).
- V4: 1 ngspice process was stopped by the timeout in the start job (pfet_01v8_lvt_W1_L0p35_m28p6, during its starting fit; its record is finite). V4 is read as written: the start-job count is reported; devices with a timeout in an arm (lodo or baselines refits) are dropped from every arm in decide.

## Predictions

- P_a_primary: unchanged: the committed prediction.json (0c184f8) value, S1_lodo_D 0.5024710627554667, predicted y_D -0.23617, sign 'gain'.
- P_a_secondary: S1_lodo_D recomputed with A1 (written to prediction_v2.json). Computed while writing this amendment from start/records: S1_lodo_D 0.5071181784404928 (53 folds), predicted y_D -0.23845, sign 'gain'; S1_pooled 0.5528165517456682, S2 0.6421768120396391 (committed: 0.5479497, 0.6157307). Same sign as the primary.
- void: P_c, gate G and the per-family S3 values and P_c signs of prediction.json are void (computed from NaN). P_c, G, G1 and the per-family secondary predictions are taken from prediction_v2.json. The prediction.json G1 choice (M_bias, from the committed S1) is recorded.
- prediction_v2: out/research/explore/I8_1/prediction_v2.json, written by scripts/research/explore/i8_1_nist.py --mode stats_v2 from start/records only (no refit card or held-out score), committed alone by --mode commit_prediction_v2 before the lodo job; prediction.json stays as committed.

## Guards

lodo and baselines refuse to start unless prediction.json, prediction_v2.json and this amendment are tracked, unmodified and in commits that are ancestors of HEAD; the worker processes hold only finite records (cryoml.explore.i8_1_nist.split_finite); stats_v2 refuses to write a non-finite S3 inner value; decide recomputes S1 (v1 and v2 definitions) and checks the excluded sets in prediction_v2, lodo and baselines.
