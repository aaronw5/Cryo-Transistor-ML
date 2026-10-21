# Tables, 2026-10-21 i11_2

Generated 2026-10-21 19:40 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i11_2.py --date 2026-10-21` from stored files only.

### I11-2 INVALID RUN: validity, labels (descriptive) and the outcome

| item | condition | value | result |
|---|---|---|---|
| OUTCOME | invalid iff V1, V2, V3 or V5 fails | V2 fails | INVALID (stop: fix and re-register) |
| V1 | 6 stored C0_refit rows reproduced (<= 1e-9) | max 0e+00 on 6 | pass |
| V2 | M_bias_near = stored X300 rows (<= 1e-9) | max 0.263 on 54 rows (233, 343, 398 K) | FAIL |
| V2 (prereg text, post hoc) | same, 233 / 343 K, 36 records | max 0e+00 on 36 rows | would pass |
| V3 | <= 2 failed records per primary arm | ihp_M_bias_T 0, ihp_M_bias_pool 0, ihp_M_bias_near 0, tud_M_bias_X 0 | pass |
| V4 | TU Delft simulator timeouts (reported) | 0 | reported |
| V5 | 3 stored fold corrections reproduced (<= 1e-9) | max 0e+00 on 3 | pass |
| label IHP interp. (300, 343 K) | pre-registered label rule (descriptive) |  | no gain |
| label IHP extrap. (233, 398 K) | pre-registered label rule (descriptive) |  | no gain |
| label TU Delft 4.13 K held out | pre-registered label rule (descriptive) |  | inconclusive |
| label TU Delft 295 K held out | pre-registered label rule (descriptive) |  | inconclusive |

- Pre-registration in force from 964a828; code 02df93c; prediction committed 812d14c before any held-out refit; validity run 886271a.
- V2 cause (post hoc, v2_diagnosis.json): run_decide compared every M_bias_near row with a stored X300 row; X300 rows exist at 233, 343 and 398 K. At 398 K M_bias_near uses the lodo_343K fold correction (398 -> 343 by definition), X300 the lodo_300K one. 233 / 343 K: 36 of 36 rows equal (difference 0).
- With V2 restricted to 233 / 343 K as written in prereg.json the rule would give 'not supported' (stop: no cross-temperature work on the shared bias-only correction). Post hoc; the recorded outcome stays invalid.


### I11-2 primary contrasts (descriptive, run invalid)

| stratum | contrast | GMR [Bonferroni CI] | 95% CI | units lower | resample |
|---|---|---:|---:|---:|---|
| IHP interp. (300, 343 K) | M_bias_T vs C0_refit | 1.121 [0.928, 1.477] | [0.963, 1.359] | 11/18 | unit |
| IHP interp. (300, 343 K) | M_bias_T vs M_bias_pool | 1.008 [0.990, 1.029] | [0.995, 1.023] | 10/18 | unit |
| IHP interp. (300, 343 K) | M_bias_T vs M_bias_near | 0.923 [0.846, 0.989] | [0.867, 0.974] | 11/18 | unit |
| IHP extrap. (233, 398 K) | M_bias_T vs C0_refit | 1.209 [0.968, 1.614] | [1.018, 1.477] | 10/18 | unit |
| IHP extrap. (233, 398 K) | M_bias_T vs M_bias_pool | 1.008 [0.964, 1.053] | [0.977, 1.041] | 8/18 | unit |
| IHP extrap. (233, 398 K) | M_bias_T vs M_bias_near | 1.010 [0.954, 1.062] | [0.972, 1.047] | 6/18 | unit |
| TU Delft 4.13 K held out | M_bias_X vs C0_refit | 0.978 [0.885, 1.128] | [0.895, 1.088] | 23/45 | cluster (7) |
| TU Delft 295 K held out | M_bias_X vs C0_refit | 1.460 [0.669, 2.631] | [0.849, 2.363] | 15/45 | cluster (7) |

- DESCRIPTIVE: run INVALID (V2 failed); no label is a decision.
- GMR = exp(mean over units of ln(arm / control)); IHP unit = DUT (per DUT the mean over the stratum's two temperatures, device bootstrap), TU Delft unit = device with a 7-group cluster bootstrap; Bonferroni level 0.99375 over 8 contrasts; 10,000 resamples, seed 0.
- The M_bias_near rows are the pre-registered arm (fold correction of the nearest training temperature); the V2 failure does not come from them (table 08).


### IHP SG13G2 per held-out temperature: mean RRMS and GMR vs C0_refit (descriptive)

| T | arm | mean RRMS | median | n | GMR vs C0_refit [95% CI] | lower |
|---|---|---:|---:|---:|---:|---:|
| 233 K | C0_refit | 0.0273 | 0.0208 | 18 | — | — |
| 233 K | M_bias_T | 0.0275 | 0.0209 | 18 | 1.154 [0.971, 1.432] | 11/18 |
| 233 K | M_bias_pool | 0.0286 | 0.0197 | 18 | 1.159 [1.000, 1.382] | 9/18 |
| 233 K | M_bias_near (g 300 K) | 0.0271 | 0.0191 | 18 | 1.116 [0.934, 1.382] | 12/18 |
| 233 K | M_bias_in | 0.0280 | 0.0194 | 18 | 1.093 [0.941, 1.319] | 11/18 |
| 233 K | M_bias_sameT | 0.0281 | 0.0179 | 18 | 1.106 [0.908, 1.425] | 11/18 |
| 233 K | lin_T | 0.0278 | 0.0196 | 18 | 1.002 [0.931, 1.079] | 8/18 |
| 233 K | lin_pool | 0.0277 | 0.0200 | 18 | 0.999 [0.928, 1.077] | 10/18 |
| 300 K | C0_refit | 0.0191 | 0.0134 | 18 | — | — |
| 300 K | M_bias_T | 0.0196 | 0.0133 | 18 | 1.133 [0.970, 1.375] | 11/18 |
| 300 K | M_bias_pool | 0.0197 | 0.0136 | 18 | 1.130 [0.975, 1.348] | 11/18 |
| 300 K | M_bias_near (g 343 K) | 0.0200 | 0.0155 | 18 | 1.234 [1.044, 1.526] | 9/18 |
| 300 K | M_bias_in | 0.0196 | 0.0130 | 18 | 1.137 [0.971, 1.379] | 11/18 |
| 300 K | M_bias_sameT | 0.0203 | 0.0142 | 18 | 1.214 [1.008, 1.537] | 9/18 |
| 300 K | lin_T | 0.0192 | 0.0137 | 18 | 0.974 [0.921, 1.031] | 9/18 |
| 300 K | lin_pool | 0.0189 | 0.0136 | 18 | 0.960 [0.915, 0.999] | 11/18 |
| 343 K | C0_refit | 0.0190 | 0.0120 | 18 | — | — |
| 343 K | M_bias_T | 0.0190 | 0.0118 | 18 | 1.110 [0.953, 1.347] | 13/18 |
| 343 K | M_bias_pool | 0.0191 | 0.0120 | 18 | 1.094 [0.936, 1.340] | 12/18 |
| 343 K | M_bias_near (g 300 K) | 0.0198 | 0.0118 | 18 | 1.197 [0.980, 1.549] | 9/18 |
| 343 K | M_bias_in | 0.0192 | 0.0123 | 18 | 1.146 [0.983, 1.395] | 10/18 |
| 343 K | M_bias_sameT | 0.0194 | 0.0127 | 18 | 1.181 [0.983, 1.502] | 8/18 |
| 343 K | lin_T | 0.0186 | 0.0123 | 18 | 0.941 [0.880, 0.995] | 9/18 |
| 343 K | lin_pool | 0.0189 | 0.0122 | 18 | 0.960 [0.913, 1.003] | 10/18 |
| 398 K | C0_refit | 0.0197 | 0.0111 | 18 | — | — |
| 398 K | M_bias_T | 0.0202 | 0.0110 | 18 | 1.267 [1.038, 1.601] | 9/18 |
| 398 K | M_bias_pool | 0.0203 | 0.0108 | 18 | 1.241 [1.015, 1.581] | 8/18 |
| 398 K | M_bias_near (g 343 K) | 0.0204 | 0.0107 | 18 | 1.285 [1.018, 1.688] | 10/18 |
| 398 K | M_bias_in | 0.0200 | 0.0116 | 18 | 1.245 [1.012, 1.584] | 10/18 |
| 398 K | M_bias_sameT | 0.0202 | 0.0126 | 18 | 1.272 [1.030, 1.627] | 11/18 |
| 398 K | lin_T | 0.0196 | 0.0113 | 18 | 1.004 [0.968, 1.041] | 8/18 |
| 398 K | lin_pool | 0.0196 | 0.0113 | 18 | 0.997 [0.964, 1.031] | 9/18 |

- DESCRIPTIVE: run INVALID (V2 failed); no label is a decision.
- Held-out RRMS (strict paper RRMS of I_card x exp(g) after the refit with g frozen in the loop), 18 IHP DEV DUTs per temperature, 0 failed records. GMR vs C0_refit with a 95% device-bootstrap CI (decide secondary).
- M_bias_sameT = stored I5-1 same-temperature fold M_bias (reference, uses the held-out temperature); M_bias_in = M_bias_T trained on all temperatures, held-out DUT excluded (reference).


### TU Delft 40 nm per held-out temperature: mean RRMS and GMR vs C0_refit (descriptive)

| T | arm | mean RRMS | median | n | GMR [95% device CI] | 95% cluster CI | lower |
|---|---|---:|---:|---:|---:|---:|---:|
| 4 K held out | C0_refit | 0.0575 | 0.0408 | 45 | — | — | — |
| 4 K held out | M_bias_X | 0.0558 | 0.0439 | 45 | 0.978 [0.931, 1.028] | [0.895, 1.088] | 23/45 |
| 4 K held out | M_bias_in | 0.0521 | 0.0377 | 45 | 0.909 [0.861, 0.958] | [0.817, 1.005] | 30/45 |
| 4 K held out | M_bias_sameT | 0.0533 | 0.0373 | 45 | 0.939 [0.877, 1.004] | [0.811, 1.081] | 27/45 |
| 295 K held out | C0_refit | 0.0255 | 0.0226 | 45 | — | — | — |
| 295 K held out | M_bias_X | 0.0353 | 0.0362 | 45 | 1.460 [1.168, 1.812] | [0.849, 2.363] | 15/45 |
| 295 K held out | M_bias_in | 0.0196 | 0.0171 | 45 | 0.806 [0.711, 0.925] | [0.610, 1.059] | 35/45 |
| 295 K held out | M_bias_sameT | 0.0196 | 0.0175 | 45 | 0.797 [0.688, 0.935] | [0.576, 1.087] | 35/45 |

- DESCRIPTIVE: run INVALID (V2 failed); no label is a decision; TU Delft labels never enter the action (one training temperature).
- 45 DEV devices in 7 geometry groups. M_bias_X = stored I6-1 fold M_bias of the other temperature (= pool = near); M_bias_in and M_bias_sameT use the held-out temperature (references).
- Primary C7 / C8 use the cluster bootstrap at the Bonferroni level (table 03).


### I11-2 secondary prediction vs observed (IHP, 12 signs)

| T | prediction | statistic | predicted | observed ln GMR | observed | correct |
|---|---|---:|---|---:|---|---|
| 233 K | M_bias_T_vs_C0_refit (S3) | S3 +0.504 | no gain | +0.1433 | no gain | yes |
| 233 K | M_bias_pool_vs_C0_refit (S3) | S3 +0.501 | no gain | +0.1472 | no gain | yes |
| 233 K | M_bias_pool_vs_C0_refit (I7-1 rule on S1_loto) | S1_loto -0.329, y +0.167 | no gain | +0.1472 | no gain | yes |
| 300 K | M_bias_T_vs_C0_refit (S3) | S3 +0.565 | no gain | +0.1248 | no gain | yes |
| 300 K | M_bias_pool_vs_C0_refit (S3) | S3 +0.479 | no gain | +0.1223 | no gain | yes |
| 300 K | M_bias_pool_vs_C0_refit (I7-1 rule on S1_loto) | S1_loto -0.410, y +0.206 | no gain | +0.1223 | no gain | yes |
| 343 K | M_bias_T_vs_C0_refit (S3) | S3 +0.571 | no gain | +0.1041 | no gain | yes |
| 343 K | M_bias_pool_vs_C0_refit (S3) | S3 +0.475 | no gain | +0.0902 | no gain | yes |
| 343 K | M_bias_pool_vs_C0_refit (I7-1 rule on S1_loto) | S1_loto -0.406, y +0.205 | no gain | +0.0902 | no gain | yes |
| 398 K | M_bias_T_vs_C0_refit (S3) | S3 +0.550 | no gain | +0.2366 | no gain | yes |
| 398 K | M_bias_pool_vs_C0_refit (S3) | S3 +0.467 | no gain | +0.2161 | no gain | yes |
| 398 K | M_bias_pool_vs_C0_refit (I7-1 rule on S1_loto) | S1_loto -0.395, y +0.199 | no gain | +0.2161 | no gain | yes |

- 12 of 12 predicted signs correct. Prediction committed in 812d14c before any held-out refit; training-only statistics of the IHP DEV starting-card records.
- Observed = sign of the mean over 18 DUTs of ln(arm / C0_refit) at the held-out temperature. Rows use the invalid run's records (the arm rows themselves are not affected by the V2 defect).


### V2 diagnosis: M_bias_near rows vs stored X300 rows per held-out temperature

| held-out T | near g from | stored fold correction used | X300 rows | equal | max rel. diff. | in decide V2 | in prereg V2 |
|---|---:|---|---:|---:|---:|---:|---:|
| 233 K | 300 K | lodo_300K x 18 | 18 | 18/18 | 0 | yes | yes |
| 300 K | 343 K | lodo_343K x 18 | — | — | — | no | no |
| 343 K | 300 K | lodo_300K x 18 | 18 | 18/18 | 0 | yes | yes |
| 398 K | 343 K | lodo_343K x 18 | 18 | 0/18 | 0.263 | yes | no |

- Post hoc (after the decision; does not change it). Source: v2_diagnosis.json (scripts/research/explore/i11_2_v2_diagnosis.py).
- Correction source: the stored I5-1 fold correction equal to the one used by each M_bias_near row (lodo_ihp/corrections.json.gz); 72/72 rows use the prereg mapping 233->300, 300->343, 343->300, 398->343.
- Single-record refits of hv_pmos_W0p3_L10|398 (largest difference): with the lodo_343K correction 0.006829 = stored M_bias_near 0.006829; with the lodo_300K correction 0.009270 = stored X300 0.009270 (both differences 0).
- Cause: the decide code selected rows by 'key in stored X300 rows' instead of the prereg temperatures 233 / 343 K. Affected: the validity flag only; no arm row and no contrast uses X300.

