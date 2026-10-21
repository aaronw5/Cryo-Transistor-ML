# I8-2 Part A: pre-registered TEST predictions

Written 2026-10-20. Code: git HEAD `3543948b8089211b5ee3971cc11d442b3c918cde` (code files clean at run: True), script `scripts/research/explore/i8_2_gate.py --mode prereg` (sha256 `2eb027a2b20f8020...`).

TEST data were not opened. The script reads the DEV starting-card records, the stored I7-1 DEV statistics and the split tag lists; every loaded record tag is in the DEV list and not in the TEST list (`checks.*.records_DEV_only` in the json).

Outcome: y = ln GMR(M_bias, C0_refit) over the TEST devices of a dataset (negative = M_bias lower RRMS). Frozen I7-1 rule: predicted y = 0.007619 + (-0.485184) * S1_lodo; gain iff S1_lodo > 0.0157. S3 rule: gain iff S3 < 0.

| TEST dataset | n TEST | n DEV | S1_lodo (DEV) | rule y | rule GMR | S1 sign | S3 (DEV) | S3 sign | S1_lodo, TEST training set (secondary) |
|---|---|---|---|---|---|---|---|---|---|
| sky130 77 K REAL_TEST | 6 | 12 | +0.3216 | -0.1484 | 0.862 | gain | -0.1126 | gain | +0.3365 (gain) |
| IHP SG13G2 TEST 233 K | 10 | 18 | -0.3144 | +0.1602 | 1.174 | loss | +0.2545 | loss | -0.2456 (loss) |
| IHP SG13G2 TEST 300 K | 10 | 18 | -0.2233 | +0.1160 | 1.123 | loss | +0.3662 | loss | -0.1887 (loss) |
| IHP SG13G2 TEST 343 K | 10 | 18 | -0.3307 | +0.1681 | 1.183 | loss | +0.3749 | loss | -0.2659 (loss) |
| IHP SG13G2 TEST 398 K | 10 | 18 | -0.3789 | +0.1915 | 1.211 | loss | +0.3945 | loss | -0.2904 (loss) |
| TU Delft TEST 4.13 K | 24 | 45 | +0.5041 | -0.2370 | 0.789 | gain | -0.0329 | gain | +0.5051 (gain) |
| TU Delft TEST 295 K | 24 | 45 | +0.3231 | -0.1492 | 0.861 | gain | -0.0409 | gain | +0.3045 (gain) |

## How TEST is scored

- C0_refit: held-out card refit of each TEST device without correction from its starting card: ngspice_fit.polish, max_nfev 400, irls_rounds 2, diff_step 0.01 (I3-2 settings), same parameter box as the dataset's DEV runs.
- M_bias: I3-2 recipe unchanged (cryoml.explore.i3_2_bias.fit_bias, 'M_bias'): I = I_card * exp(g), g = tanh(h(x)), h = MLP 3-16-16-1 (tanh hidden, last layer zero-initialized), x = (is_pmos 0/1, |Vgs|/VDD, |Vds|/VDD) with the two bias inputs standardized by the training-point mean/std; loss = mean over training devices of the paper per-device RRMS of the corrected current + 1e-4 * sum of squared weight-matrix entries; Adam lr 3e-3, 2000 full-batch steps, float64, seeds 0-4, g = mean of the 5 members. Trained post hoc on the starting-card records of the training set below, frozen; each TEST card is refit with g in the loop (same polish settings as C0_refit), the best card re-simulated, corrected with the same g and scored.
- Score: strict paper RRMS per device (e1_correction.score_device: mean over the frozen measured-inclusion curves of trimmed RMSE / trimmed mean |I_measured|).
- Outcome: y = mean over TEST devices of ln(RRMS_M_bias / RRMS_C0_refit); GMR(M_bias, C0_refit) = exp(y) (i3_2_bias.paired_ratio; devices with a non-finite or non-positive RRMS in either arm left out and counted). Predicted sign correct iff sign(predicted y) = sign(y).
- sky130 77 K: 6 REAL_TEST devices; starting cards: the 43-parameter direct-fit cards in out/research/polished/direct43_r3_fd/real/summary.json (as for REAL_DEV); records by scripts/research/explore/e1_collect_real.py --split test (one NGSpice run per device); training set: all 12 REAL_DEV starting-card records (out/research/explore/E1/real_dev), both polarities, one M_bias ensemble.
- IHP SG13G2: 10 TEST DUTs per temperature; starting cards: per TEST DUT and temperature: ngspice_fit.polish from the stock card, max_nfev 3000, irls_rounds 3, diff_step 0.01, objective = mean included-curve RRMS (frozen measured inclusion), best card kept (I5-1 prereg cards); training set: per temperature and class: all DEV DUTs of the TEST DUT's class (LV 10, HV 8) at that temperature (out/research/explore/I5_1/start/records_<T>K); one M_bias ensemble per (temperature, class).
- TU Delft: 24 TEST devices per temperature; starting cards: per TEST device and temperature: ngspice_fit.polish from the stock card, max_nfev 3000, irls_rounds 3, diff_step 0.01, objective = mean included-curve RRMS (frozen measured inclusion), best card kept (I6-1 prereg cards); training set: all 45 DEV devices at the temperature (out/research/explore/I6_1/start/records_<T>K), both polarities, one M_bias ensemble per temperature.
- no TEST mode exists in the code at this commit; the protocol is the TEST_policy / method-freeze text of the dataset's prereg, restated here.

## Notes

- The primary S values are the stored dataset-level DEV values (mean over the DEV leave-one-out folds), the definition the rule was fit on. The secondary column is S1_lodo on the TEST run's training set (all DEV devices; IHP: per class, weighted by TEST DUTs per class 6 LV / 4 HV); it gives the same sign as the primary value on all 7 datasets.
- The DEV outcomes of these datasets were known when the rule was fit (I7-1 step 2). The 4 IHP temperatures share DUTs and the 2 TU Delft temperatures share devices, so the 7 outcomes come from 3 device sets.
- These predictions are not changed after this commit; a change gets a dated amendment.
