# Tables, 2026-10-20 i7_1

Generated 2026-10-20 18:39 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i7_1.py --date 2026-10-20` from stored files only. y = ln GMR(M_bias, C0_refit) over the held-out targets (strict paper RRMS after the held-out card refit); negative = gain.

### I7-1 prospective test (20 new simulated datasets): pre-registered P1, P2, verdict and secondary statistics including S3

| item | pre-registered definition | value | evaluated |
|---|---|---|---|
| P1 (decides) | Spearman(S1_lodo_D, y_D) over 20 datasets <= -0.50 | -0.970 | holds |
| P2 (decides) | frozen-rule sign correct in >= 15 of 20 | 16/20 | holds |
| verdict | supported if P1 and P2; partially supported if one; not supported if neither | supported |  |
| frozen rule (step 2) | OLS over the 11 stored datasets, amendment before step 3 | y = 0.007619 -0.485184 S1_lodo, R^2 0.785, zero crossing 0.0157 | reported |
| prospective OLS | OLS over the 20 new datasets | y = -0.0102 -1.419 S1_lodo, R^2 0.864, zero crossing -0.0072 | reported |
| rule MAE | mean |a + b S1_lodo_D - y_D| over 20 datasets | 0.357 | reported |
| permutation p | one-sided, 10000 permutations of y_D, default_rng(0) | 1.0e-04 | reported |
| per-PDK Spearman | S1_lodo vs y over 10 datasets per PDK | GF180 -0.988, IHP -0.976 | reported |
| S1_pooled, S2 | dataset Spearman with y_D (P1 form) | S1_pooled -0.974, S2 -0.979 | reported |
| fold level, pooled | Spearman(S_f, y_f) over 320 folds | S1_lodo -0.751, S1_pooled -0.755, S2 -0.759 | reported |
| manipulation check | Spearman(lambda, S1_lodo_D), Spearman(lambda, y_D) | 0.981, -0.956 | reported |
| S3, new datasets | inner leave-one-group-out post-hoc gain (secondary, rank test only, expected positive): Spearman(S3_D, y_D) over 20 datasets | Spearman +0.985; OLS y = -0.0298 +1.214 S3, R^2 0.992; sign(S3_D) = sign(y_D) in 19/20 | reported |
| S3, stored datasets | same over the 11 stored datasets (step 2, outcomes already seen) | Spearman +0.927; OLS y = -0.0957 +0.797 S3, R^2 0.862; sign(S3_D) = sign(y_D) in 9/11 | reported |
| S3, fold level | Spearman(S3_f, y_f) pooled over folds and within each dataset | new: pooled +0.736 (OLS slope +1.173, R^2 0.727), within negative in 20/20 (-0.97 to -0.30, mean -0.81); stored: pooled +0.277, within negative in 9/11 (mean -0.35) | reported |
| S3, compute | 1 seed, 500 steps per inner fit; wall time of the S3 queue jobs | S3 queue jobs: stored datasets 2409 s (1760 inner fits), new datasets 3442 s (4800 inner fits); S1_lodo, S1_pooled, S2 on the 162 stored folds 2.1 s, no training | reported |

- Pre-registration out/research/explore/I7_1/prereg.json (commit d406939); amendment out/research/explore/I7_1/prereg_amendment_2026-10-20.json (frozen rule, commit 968d973, written after step 2 and before any step-3 truth set). Decision file out/research/explore/I7_1/decision.json; every value is recomputed here from the per-fold values in new/*/results.json and checked.
- y_D = ln GMR(M_bias, C0_refit) = mean over the 16 held-out targets of ln(RRMS_M_bias / RRMS_C0_refit); negative = gain. S_D = mean over the 16 leave-one-geometry-out folds of the statistic computed on the 15 training records of the fold. 0 targets left out, 0 datasets excluded.
- S3 per fold = mean over the fold's training records of ln(RRMS with correction / RRMS of the starting card), the correction = M_bias reduced recipe trained on the fold's training records minus the record's inner group, applied post hoc without a card refit (out/research/explore/I7_1/new/s3.json, out/research/explore/I7_1/retro/s3.json). S3 does not enter P1, P2 or the frozen rule; its sign agreement is reported here, not pre-registered.


### I7-1 new simulated datasets: training statistics, held-out outcome and frozen-rule prediction

| PDK | replicate | lambda | S1_lodo | S1_pooled | S2 | S3 | C0_refit mean RRMS | M_bias mean RRMS | y = ln GMR(M_bias, C0_refit) | GMR(M_bias, C0_refit) | frozen-rule predicted y | predicted | observed | sign correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| GF180 | 0 | 0 | -0.103 | +0.177 | +0.058 | +0.006 | 0.0728 | 0.0715 | -0.0449 | 0.956 | +0.0578 | no gain | gain | NO |
| GF180 | 0 | 0.25 | +0.050 | +0.294 | +0.237 | -0.046 | 0.0693 | 0.0628 | -0.1429 | 0.867 | -0.0165 | gain | gain | yes |
| GF180 | 0 | 0.5 | +0.358 | +0.523 | +0.522 | -0.268 | 0.0717 | 0.0515 | -0.3885 | 0.678 | -0.1659 | gain | gain | yes |
| GF180 | 0 | 0.75 | +0.630 | +0.732 | +0.759 | -0.551 | 0.0749 | 0.0369 | -0.7487 | 0.473 | -0.2979 | gain | gain | yes |
| GF180 | 0 | 1 | +0.892 | +0.929 | +0.972 | -1.394 | 0.0806 | 0.0146 | -1.7125 | 0.180 | -0.4251 | gain | gain | yes |
| GF180 | 1 | 0 | -0.194 | +0.116 | -0.006 | +0.078 | 0.0695 | 0.0751 | +0.0708 | 1.073 | +0.1016 | no gain | no gain | yes |
| GF180 | 1 | 0.25 | +0.006 | +0.262 | +0.171 | -0.011 | 0.0666 | 0.0649 | -0.0392 | 0.962 | +0.0045 | no gain | gain | NO |
| GF180 | 1 | 0.5 | +0.261 | +0.449 | +0.430 | -0.185 | 0.0675 | 0.0550 | -0.2355 | 0.790 | -0.1190 | gain | gain | yes |
| GF180 | 1 | 0.75 | +0.529 | +0.648 | +0.669 | -0.458 | 0.0694 | 0.0419 | -0.5564 | 0.573 | -0.2490 | gain | gain | yes |
| GF180 | 1 | 1 | +0.873 | +0.903 | +0.935 | -1.097 | 0.0726 | 0.0171 | -1.5150 | 0.220 | -0.4162 | gain | gain | yes |
| IHP | 0 | 0 | -0.100 | +0.176 | +0.010 | +0.061 | 0.0723 | 0.0750 | +0.0297 | 1.030 | +0.0561 | no gain | no gain | yes |
| IHP | 0 | 0.25 | +0.223 | +0.416 | +0.327 | -0.111 | 0.0735 | 0.0637 | -0.1433 | 0.866 | -0.1004 | gain | gain | yes |
| IHP | 0 | 0.5 | +0.461 | +0.593 | +0.572 | -0.307 | 0.0754 | 0.0528 | -0.3697 | 0.691 | -0.2159 | gain | gain | yes |
| IHP | 0 | 0.75 | +0.700 | +0.768 | +0.794 | -0.608 | 0.0766 | 0.0390 | -0.6959 | 0.499 | -0.3319 | gain | gain | yes |
| IHP | 0 | 1 | +0.912 | +0.925 | +0.955 | -1.228 | 0.0749 | 0.0165 | -1.5380 | 0.215 | -0.4347 | gain | gain | yes |
| IHP | 1 | 0 | -0.050 | +0.236 | +0.159 | -0.102 | 0.0798 | 0.0741 | -0.1308 | 0.877 | +0.0317 | no gain | gain | NO |
| IHP | 1 | 0.25 | -0.001 | +0.268 | +0.178 | -0.003 | 0.0665 | 0.0651 | -0.0367 | 0.964 | +0.0082 | no gain | gain | NO |
| IHP | 1 | 0.5 | +0.266 | +0.460 | +0.460 | -0.180 | 0.0659 | 0.0535 | -0.2426 | 0.785 | -0.1213 | gain | gain | yes |
| IHP | 1 | 0.75 | +0.602 | +0.703 | +0.697 | -0.471 | 0.0679 | 0.0397 | -0.5731 | 0.564 | -0.2845 | gain | gain | yes |
| IHP | 1 | 1 | +0.950 | +0.955 | +0.971 | -1.286 | 0.0730 | 0.0148 | -1.4972 | 0.224 | -0.4534 | gain | gain | yes |

- 16 targets per dataset (8 nMOS, 8 pMOS geometries), 16 leave-one-geometry-out folds; truth = make_truth tier-a card x exp(e_d) x (1 + 0.01 N(0,1)), e_d = 0.10 (sqrt(lambda) s_polarity + sqrt(1 - lambda) u_d). The five lambda levels of one (PDK, replicate) share base cards, s, u_d and noise draws.
- Frozen rule: predicted y = 0.007619 -0.485184 S1_lodo; predicted gain iff S1_lodo > 0.0157. Sign correct in 16/20. The 4 errors are datasets with predicted 'no gain' and an observed gain. S3 (secondary, no rule): sign(S3_D) = sign(y_D) in 19/20; the exception is GF180 r0 lambda 0 (S3 +0.0062, y -0.0449).


### I7-1 post hoc (not pre-registered): S1_lodo vs y within lambda levels and after removing lambda

| subset or model | datasets | Spearman(S1_lodo, y) | Pearson(S1_lodo, y) | OLS slope of residual y on residual S1_lodo | one-sided permutation p (y permuted within cells) | detail |
|---|---:|---:|---:|---:|---:|---|
| within lambda = 0 | 4 | -0.800 | -0.874 |  |  | S1_lodo -0.194 to -0.050; y -0.131 to +0.071; order by S1_lodo: GF180 r1, GF180 r0, IHP r0, IHP r1 |
| within lambda = 0.25 | 4 | -1.000 | -0.739 |  |  | S1_lodo -0.001 to +0.223; y -0.143 to -0.037; order by S1_lodo: IHP r1, GF180 r1, GF180 r0, IHP r0 |
| within lambda = 0.5 | 4 | -0.800 | -0.848 |  |  | S1_lodo +0.261 to +0.461; y -0.388 to -0.235; order by S1_lodo: GF180 r1, IHP r1, GF180 r0, IHP r0 |
| within lambda = 0.75 | 4 | -0.800 | -0.725 |  |  | S1_lodo +0.529 to +0.700; y -0.749 to -0.556; order by S1_lodo: GF180 r1, IHP r1, GF180 r0, IHP r0 |
| within lambda = 1 | 4 | +0.400 | +0.369 |  |  | S1_lodo +0.873 to +0.950; y -1.713 to -1.497; order by S1_lodo: GF180 r1, GF180 r0, IHP r0, IHP r1 |
| mean of the 5 within-lambda Spearman | 20 | -0.600 |  |  |  |  |
| residual after lambda (5 levels) | 20 | -0.696 | -0.590 | -0.659 | 0.0037 | df 14; S1_lodo and y each minus their lambda cell mean |
| residual after lambda x PDK (10 levels) | 20 | -0.749 | -0.751 | -0.911 | 0.0020 | df 9; S1_lodo and y each minus their lambda x PDK cell mean |

- Post hoc, not pre-registered. Question: does S1_lodo order the datasets beyond the manipulated lambda? Within one lambda level the 4 datasets differ in PDK and replicate (base cards, error functions, noise).
- Residual rows: S1_lodo and y are each regressed on the categorical variable (OLS with one indicator per level = subtract the cell mean); the Pearson and Spearman correlations and the OLS slope are computed on the residuals. Permutation p: 10000 permutations of y within the cells, default_rng(0), statistic = residual Pearson r, one-sided (negative). The 'lambda x PDK' cells hold 2 datasets (the 2 replicates).


### I7-1 step 2: the 11 stored datasets that calibrate the frozen rule

| dataset | folds | held-out devices | S1_lodo | S1_pooled | S2 | S3 | y = ln GMR | GMR | frozen-rule y (in-sample) | sign correct | sign(S3) = sign(y) | within-dataset fold Spearman(S1_lodo, y_f) | within-dataset fold Spearman(S3, y_f) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| sky130 77 K | 12 | 12 | +0.322 | +0.647 | +0.679 | -0.113 | -0.2631 | 0.769 | -0.1484 | yes | yes | +0.69 | -0.74 |
| IHP meas 233 K | 18 | 18 | -0.314 | +0.297 | +0.228 | +0.254 | +0.1012 | 1.106 | +0.1602 | yes | yes | +0.11 | -0.09 |
| IHP meas 300 K | 18 | 18 | -0.223 | +0.371 | +0.321 | +0.366 | +0.1943 | 1.214 | +0.1160 | yes | yes | +0.30 | -0.22 |
| IHP meas 343 K | 18 | 18 | -0.331 | +0.336 | +0.340 | +0.375 | +0.1664 | 1.181 | +0.1681 | yes | yes | -0.10 | +0.25 |
| IHP meas 398 K | 18 | 18 | -0.379 | +0.534 | +0.314 | +0.395 | +0.2405 | 1.272 | +0.1915 | yes | yes | +0.11 | +0.45 |
| TU Delft 4.13 K | 7 | 45 | +0.504 | +0.531 | +0.781 | -0.033 | -0.0634 | 0.939 | -0.2370 | yes | yes | -0.57 | -0.93 |
| TU Delft 295 K | 7 | 45 | +0.323 | +0.374 | +0.476 | -0.041 | -0.2265 | 0.797 | -0.1492 | yes | yes | +0.79 | -0.43 |
| GF180 tier c | 16 | 48 | +0.248 | +0.307 | +0.435 | -0.066 | -0.1275 | 0.880 | -0.1125 | yes | yes | +0.72 | -0.80 |
| IHP tier c | 16 | 48 | +0.224 | +0.302 | +0.373 | +0.037 | -0.1219 | 0.885 | -0.1013 | yes | NO | +0.53 | -0.45 |
| GF180 tier b | 16 | 48 | +0.012 | +0.042 | +0.076 | +0.001 | -0.0039 | 0.996 | +0.0017 | NO | NO | +0.79 | -0.44 |
| IHP tier b | 16 | 48 | +0.032 | +0.064 | +0.140 | -0.004 | -0.0148 | 0.985 | -0.0078 | yes | yes | +0.94 | -0.46 |

- Step 2, retrospective calibration on outcomes already seen (exploratory), out/research/explore/I7_1/retro/results.json (commit 968d973). The IHP measured temperatures share devices, as do the TU Delft temperatures; the 11 points are not independent.
- Dataset Spearman with y: S1_lodo -0.873, S1_pooled -0.145, S2 -0.636. The rule is the OLS line over these 11 points (R^2 0.785); in-sample sign correct 10/11, leave-one-dataset-out 10/11.
- S3 (computed after the rule was frozen; out/research/explore/I7_1/retro/s3.json, 2409 s): dataset Spearman +0.927; sign(S3) = sign(y) in 9/11.

