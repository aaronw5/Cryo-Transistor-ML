# Tables, 2026-10-20 i8_2

Generated 2026-10-20 22:42 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i8_2.py --date 2026-10-20` from stored files only. y = ln GMR(M_bias, C0_refit) over the held-out devices (strict paper RRMS after the held-out card refit); negative = gain.

### I8-2 Part A: pre-registered TEST predictions of the sign of ln GMR(M_bias, C0_refit)

| TEST dataset | n TEST devices | n DEV devices | S1_lodo (DEV, stored) | predicted y | predicted GMR | S1_lodo prediction | S3 (DEV, stored) | S3 prediction | S1_lodo on TEST training set (secondary) | secondary prediction | DEV GMR(M_bias, C0_refit), already seen |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|---:|
| sky130 77 K REAL_TEST | 6 | 12 | +0.3216 | -0.1484 | 0.862 | gain | -0.1126 | gain | +0.3365 | gain | 0.769 |
| IHP SG13G2 TEST 233 K | 10 | 18 | -0.3144 | +0.1602 | 1.174 | loss | +0.2545 | loss | -0.2456 | loss | 1.106 |
| IHP SG13G2 TEST 300 K | 10 | 18 | -0.2233 | +0.1160 | 1.123 | loss | +0.3662 | loss | -0.1887 | loss | 1.214 |
| IHP SG13G2 TEST 343 K | 10 | 18 | -0.3307 | +0.1681 | 1.183 | loss | +0.3749 | loss | -0.2659 | loss | 1.181 |
| IHP SG13G2 TEST 398 K | 10 | 18 | -0.3789 | +0.1915 | 1.211 | loss | +0.3945 | loss | -0.2904 | loss | 1.272 |
| TU Delft TEST 4.13 K | 24 | 45 | +0.5041 | -0.2370 | 0.789 | gain | -0.0329 | gain | +0.5051 | gain | 0.939 |
| TU Delft TEST 295 K | 24 | 45 | +0.3231 | -0.1492 | 0.861 | gain | -0.0409 | gain | +0.3045 | gain | 0.797 |

- Pre-registered 2026-10-20 before any TEST curve was loaded, simulated or fitted (out/research/explore/I8_2/prereg_test_predictions.json; code at git 3543948). The final TEST run is planned for about 2026-12-10. 'gain' = predicted ln GMR(M_bias, C0_refit) < 0 on the TEST devices.
- Frozen I7-1 rule: predicted y = 0.007619 -0.485184 S1_lodo; gain iff S1_lodo > 0.0157. S3 rule: gain iff S3 < 0. Primary S values = stored dataset-level DEV values (mean over the DEV leave-one-out folds). Secondary: S1_lodo on the TEST run's training set (all DEV devices; IHP per class, weighted 6 LV / 4 HV TEST DUTs).
- TEST scoring: M_bias (I3-2 recipe, 5 seeds, 2000 steps) trained on all DEV starting-card records (IHP: DEV DUTs of the same class and temperature), TEST card refit with and without M_bias in the loop (polish max_nfev 400, irls 2, diff_step 0.01) from the TEST starting card, strict paper RRMS, y = mean ln(RRMS_M_bias / RRMS_C0_refit) over the TEST devices.
- TEST data were not opened: every record read is in the DEV list of its split file. The DEV outcomes (last column) were known when the rule was fitted. The 4 IHP temperatures share DUTs and the 2 TU Delft temperatures share devices (3 device sets).


### I8-2 Part B (retrospective): S3 gate G and S1_lodo gate G1 per dataset

| set | dataset | held-out devices | S3(D) | S1_lodo(D) | GMR(M_bias, C0_refit) | G choice | GMR(G, C0_refit) | GMR(G, M_bias) | G1 choice | GMR(G1, C0_refit) | GMR(G1, M_bias) | fold-level S3 gate GMR vs C0_refit (diagnostic) | folds whose decision differs from G |
|---|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| stored | sky130 77 K | 12 | -0.1126 | +0.3216 | 0.769 | M_bias | 0.769 | 1.000 | M_bias | 0.769 | 1.000 | 0.769 | 0/12 |
| stored | IHP meas 233 K | 18 | +0.2545 | -0.3144 | 1.106 | C0_refit | 1.000 | 0.904 | C0_refit | 1.000 | 0.904 | 1.000 | 0/18 |
| stored | IHP meas 300 K | 18 | +0.3662 | -0.2233 | 1.214 | C0_refit | 1.000 | 0.823 | C0_refit | 1.000 | 0.823 | 1.000 | 0/18 |
| stored | IHP meas 343 K | 18 | +0.3749 | -0.3307 | 1.181 | C0_refit | 1.000 | 0.847 | C0_refit | 1.000 | 0.847 | 1.000 | 0/18 |
| stored | IHP meas 398 K | 18 | +0.3945 | -0.3789 | 1.272 | C0_refit | 1.000 | 0.786 | C0_refit | 1.000 | 0.786 | 1.000 | 0/18 |
| stored | TU Delft 4.13 K | 45 | -0.0329 | +0.5041 | 0.939 | M_bias | 0.939 | 1.000 | M_bias | 0.939 | 1.000 | 1.027 | 2/7 |
| stored | TU Delft 295 K | 45 | -0.0409 | +0.3231 | 0.797 | M_bias | 0.797 | 1.000 | M_bias | 0.797 | 1.000 | 1.040 | 4/7 |
| stored | GF180 tier c | 48 | -0.0656 | +0.2476 | 0.880 | M_bias | 0.880 | 1.000 | M_bias | 0.880 | 1.000 | 0.880 | 0/16 |
| stored | IHP tier c | 48 | +0.0367 | +0.2244 | 0.885 | C0_refit | 1.000 | 1.130 | M_bias | 0.885 | 1.000 | 0.993 | 1/16 |
| stored | GF180 tier b | 48 | +0.0015 | +0.0122 | 0.996 | C0_refit | 1.000 | 1.004 | C0_refit | 1.000 | 1.004 | 1.003 | 2/16 |
| stored | IHP tier b | 48 | -0.0043 | +0.0318 | 0.985 | M_bias | 0.985 | 1.000 | M_bias | 0.985 | 1.000 | 0.985 | 0/16 |
| new (I7-1) | GF180 r0 lambda 0 | 16 | +0.0062 | -0.1034 | 0.956 | C0_refit | 1.000 | 1.046 | C0_refit | 1.000 | 1.046 | 1.103 | 8/16 |
| new (I7-1) | GF180 r0 lambda 0.25 | 16 | -0.0464 | +0.0497 | 0.867 | M_bias | 0.867 | 1.000 | M_bias | 0.867 | 1.000 | 0.973 | 4/16 |
| new (I7-1) | GF180 r0 lambda 0.5 | 16 | -0.2685 | +0.3577 | 0.678 | M_bias | 0.678 | 1.000 | M_bias | 0.678 | 1.000 | 0.678 | 0/16 |
| new (I7-1) | GF180 r0 lambda 0.75 | 16 | -0.5513 | +0.6297 | 0.473 | M_bias | 0.473 | 1.000 | M_bias | 0.473 | 1.000 | 0.473 | 0/16 |
| new (I7-1) | GF180 r0 lambda 1 | 16 | -1.3940 | +0.8918 | 0.180 | M_bias | 0.180 | 1.000 | M_bias | 0.180 | 1.000 | 0.180 | 0/16 |
| new (I7-1) | GF180 r1 lambda 0 | 16 | +0.0777 | -0.1937 | 1.073 | C0_refit | 1.000 | 0.932 | C0_refit | 1.000 | 0.932 | 1.000 | 0/16 |
| new (I7-1) | GF180 r1 lambda 0.25 | 16 | -0.0114 | +0.0065 | 0.962 | M_bias | 0.962 | 1.000 | C0_refit | 1.000 | 1.040 | 1.031 | 3/16 |
| new (I7-1) | GF180 r1 lambda 0.5 | 16 | -0.1854 | +0.2609 | 0.790 | M_bias | 0.790 | 1.000 | M_bias | 0.790 | 1.000 | 0.790 | 0/16 |
| new (I7-1) | GF180 r1 lambda 0.75 | 16 | -0.4583 | +0.5289 | 0.573 | M_bias | 0.573 | 1.000 | M_bias | 0.573 | 1.000 | 0.573 | 0/16 |
| new (I7-1) | GF180 r1 lambda 1 | 16 | -1.0973 | +0.8734 | 0.220 | M_bias | 0.220 | 1.000 | M_bias | 0.220 | 1.000 | 0.220 | 0/16 |
| new (I7-1) | IHP r0 lambda 0 | 16 | +0.0609 | -0.0998 | 1.030 | C0_refit | 1.000 | 0.971 | C0_refit | 1.000 | 0.971 | 1.000 | 0/16 |
| new (I7-1) | IHP r0 lambda 0.25 | 16 | -0.1108 | +0.2227 | 0.866 | M_bias | 0.866 | 1.000 | M_bias | 0.866 | 1.000 | 0.866 | 0/16 |
| new (I7-1) | IHP r0 lambda 0.5 | 16 | -0.3073 | +0.4607 | 0.691 | M_bias | 0.691 | 1.000 | M_bias | 0.691 | 1.000 | 0.691 | 0/16 |
| new (I7-1) | IHP r0 lambda 0.75 | 16 | -0.6076 | +0.6997 | 0.499 | M_bias | 0.499 | 1.000 | M_bias | 0.499 | 1.000 | 0.499 | 0/16 |
| new (I7-1) | IHP r0 lambda 1 | 16 | -1.2284 | +0.9116 | 0.215 | M_bias | 0.215 | 1.000 | M_bias | 0.215 | 1.000 | 0.215 | 0/16 |
| new (I7-1) | IHP r1 lambda 0 | 16 | -0.1024 | -0.0495 | 0.877 | M_bias | 0.877 | 1.000 | C0_refit | 1.000 | 1.140 | 0.877 | 0/16 |
| new (I7-1) | IHP r1 lambda 0.25 | 16 | -0.0027 | -0.0011 | 0.964 | M_bias | 0.964 | 1.000 | C0_refit | 1.000 | 1.037 | 1.046 | 9/16 |
| new (I7-1) | IHP r1 lambda 0.5 | 16 | -0.1800 | +0.2657 | 0.785 | M_bias | 0.785 | 1.000 | M_bias | 0.785 | 1.000 | 0.785 | 0/16 |
| new (I7-1) | IHP r1 lambda 0.75 | 16 | -0.4708 | +0.6021 | 0.564 | M_bias | 0.564 | 1.000 | M_bias | 0.564 | 1.000 | 0.564 | 0/16 |
| new (I7-1) | IHP r1 lambda 1 | 16 | -1.2860 | +0.9502 | 0.224 | M_bias | 0.224 | 1.000 | M_bias | 0.224 | 1.000 | 0.224 | 0/16 |

- Retrospective: all outcomes were known before the gates were defined. G: M_bias on dataset D iff S3(D) < 0, else C0_refit; G1: M_bias iff the frozen I7-1 rule predicts a gain (S1_lodo(D) > 0.0157). One choice per dataset from training-device statistics; GMR(G, C0_refit) = GMR(M_bias, C0_refit) if G chooses M_bias, else 1.
- S3 on the 11 stored datasets was computed after their outcomes were known; G1's rule was fitted on the 11 stored datasets (in-sample there) and frozen before the 20 simulated datasets were generated.
- Fold-level S3 gate (diagnostic): the decision for the held-out devices of an outer fold uses that fold's training-set S3_f, which excludes the scored devices. G's dataset-level S3(D) averages S3_f over all folds, so a scored device's own starting-card record enters G's decision through the other folds.


### I8-2 Part B (retrospective): pooled gate results over datasets

| datasets | method | choices (M_bias, sign correct) | mean ln GMR(., C0_refit) | exp(mean ln GMR(., C0_refit)) | worst GMR(., C0_refit) | datasets GMR(., C0_refit) > 1.02 | mean ln GMR(., M_bias) | worst GMR(., M_bias) | datasets GMR(., M_bias) > 1.02 |
|---|---|---|---:|---:|---|---:|---:|---|---:|
| 11 stored datasets | G (S3 gate) | 5 M_bias, 9 correct | -0.0632 | 0.939 | 1.000 (no dataset above 1) | 0/11 | -0.0524 | 1.130 (IHP tier c) | 1/11 |
| 11 stored datasets | G1 (S1_lodo rule gate) | 6 M_bias, 10 correct | -0.0743 | 0.928 | 1.000 (no dataset above 1) | 0/11 | -0.0635 | 1.004 (GF180 tier b) | 0/11 |
| 11 stored datasets | M_bias always |  | -0.0108 | 0.989 | 1.272 (IHP meas 398 K) | 4/11 | +0.0000 | 1.000 (no dataset above 1) | 0/11 |
| 11 stored datasets | C0_refit always |  | +0.0000 | 1.000 | 1.000 (no dataset above 1) | 0/11 | +0.0108 | 1.301 (sky130 77 K) | 5/11 |
| 11 stored datasets | dataset oracle (reference) |  | -0.0746 | 0.928 | 1.000 (no dataset above 1) | 0/11 | -0.0638 | 1.000 (no dataset above 1) | 0/11 |
| 11 stored datasets | fold-level S3 gate (diagnostic) |  | -0.0313 | 0.969 | 1.040 (TU Delft 295 K) | 2/11 | -0.0205 | 1.304 (TU Delft 295 K) | 3/11 |
| 11 stored datasets | fold-level S1_lodo gate (diagnostic) |  | -0.0736 | 0.929 | 1.005 (IHP meas 300 K) | 0/11 | -0.0628 | 1.008 (GF180 tier b) | 0/11 |
| 7 measured (subset of stored) | G (S3 gate) | 3 M_bias, 7 correct | -0.0790 | 0.924 | 1.000 (no dataset above 1) | 0/7 | -0.1003 | 1.000 (no dataset above 1) | 0/7 |
| 7 measured (subset of stored) | G1 (S1_lodo rule gate) | 3 M_bias, 7 correct | -0.0790 | 0.924 | 1.000 (no dataset above 1) | 0/7 | -0.1003 | 1.000 (no dataset above 1) | 0/7 |
| 7 measured (subset of stored) | M_bias always |  | +0.0213 | 1.022 | 1.272 (IHP meas 398 K) | 4/7 | +0.0000 | 1.000 (no dataset above 1) | 0/7 |
| 7 measured (subset of stored) | C0_refit always |  | +0.0000 | 1.000 | 1.000 (no dataset above 1) | 0/7 | -0.0213 | 1.301 (sky130 77 K) | 3/7 |
| 7 measured (subset of stored) | dataset oracle (reference) |  | -0.0790 | 0.924 | 1.000 (no dataset above 1) | 0/7 | -0.1003 | 1.000 (no dataset above 1) | 0/7 |
| 7 measured (subset of stored) | fold-level S3 gate (diagnostic) |  | -0.0282 | 0.972 | 1.040 (TU Delft 295 K) | 2/7 | -0.0495 | 1.304 (TU Delft 295 K) | 2/7 |
| 7 measured (subset of stored) | fold-level S1_lodo gate (diagnostic) |  | -0.0786 | 0.924 | 1.005 (IHP meas 300 K) | 0/7 | -0.0999 | 1.000 (no dataset above 1) | 0/7 |
| 20 I7-1 simulated datasets | G (S3 gate) | 17 M_bias, 19 correct | -0.5283 | 0.590 | 1.000 (no dataset above 1) | 0/20 | -0.0028 | 1.046 (GF180 r0 lambda 0) | 1/20 |
| 20 I7-1 simulated datasets | G1 (S1_lodo rule gate) | 14 M_bias, 16 correct | -0.5180 | 0.596 | 1.000 (no dataset above 1) | 0/20 | +0.0076 | 1.140 (IHP r1 lambda 0) | 4/20 |
| 20 I7-1 simulated datasets | M_bias always |  | -0.5255 | 0.591 | 1.073 (GF180 r1 lambda 0) | 2/20 | +0.0000 | 1.000 (no dataset above 1) | 0/20 |
| 20 I7-1 simulated datasets | C0_refit always |  | +0.0000 | 1.000 | 1.000 (no dataset above 1) | 0/20 | +0.5255 | 5.543 (GF180 r0 lambda 1) | 18/20 |
| 20 I7-1 simulated datasets | dataset oracle (reference) |  | -0.5305 | 0.588 | 1.000 (no dataset above 1) | 0/20 | -0.0050 | 1.000 (no dataset above 1) | 0/20 |
| 20 I7-1 simulated datasets | fold-level S3 gate (diagnostic) |  | -0.5101 | 0.600 | 1.103 (GF180 r0 lambda 0) | 3/20 | +0.0154 | 1.153 (GF180 r0 lambda 0) | 4/20 |
| 20 I7-1 simulated datasets | fold-level S1_lodo gate (diagnostic) |  | -0.5093 | 0.601 | 1.063 (GF180 r1 lambda 0.25) | 3/20 | +0.0162 | 1.163 (IHP r1 lambda 0) | 5/20 |
| all 31 datasets | G (S3 gate) | 22 M_bias, 28 correct | -0.3633 | 0.695 | 1.000 (no dataset above 1) | 0/31 | -0.0204 | 1.130 (IHP tier c) | 2/31 |
| all 31 datasets | G1 (S1_lodo rule gate) | 20 M_bias, 26 correct | -0.3605 | 0.697 | 1.000 (no dataset above 1) | 0/31 | -0.0177 | 1.140 (IHP r1 lambda 0) | 4/31 |
| all 31 datasets | M_bias always |  | -0.3429 | 0.710 | 1.272 (IHP meas 398 K) | 6/31 | +0.0000 | 1.000 (no dataset above 1) | 0/31 |
| all 31 datasets | C0_refit always |  | +0.0000 | 1.000 | 1.000 (no dataset above 1) | 0/31 | +0.3429 | 5.543 (GF180 r0 lambda 1) | 23/31 |
| all 31 datasets | dataset oracle (reference) |  | -0.3688 | 0.692 | 1.000 (no dataset above 1) | 0/31 | -0.0259 | 1.000 (no dataset above 1) | 0/31 |
| all 31 datasets | fold-level S3 gate (diagnostic) |  | -0.3402 | 0.712 | 1.103 (GF180 r0 lambda 0) | 5/31 | +0.0027 | 1.304 (TU Delft 295 K) | 7/31 |
| all 31 datasets | fold-level S1_lodo gate (diagnostic) |  | -0.3547 | 0.701 | 1.063 (GF180 r1 lambda 0.25) | 3/31 | -0.0119 | 1.163 (IHP r1 lambda 0) | 5/31 |

- Retrospective. Pooled = equal weight per dataset. 'worst dataset' = largest GMR. The dataset oracle chooses with the outcome known; it is a reference bound, not a method. Fold-level gates are diagnostics (decision per outer fold from that fold's training-set statistic).
- Leakage of G: S3(D) contains each scored device's own starting-card record through the folds in which it is a training device (about 1/n_devices of S3(D)); the held-out refit outcomes are not used. Datasets with |S3(D)| < 0.01: GF180 tier b, IHP tier b, GF180 r0 lambda 0, IHP r1 lambda 0.25; there a flipped decision changes ln GMR(G, C0_refit) by at most |y_D| <= 0.045. On TEST the gate uses DEV devices only and has no leakage of this kind.
- The 4 IHP measured temperatures share DUTs and the 2 TU Delft temperatures share devices; the 7 measured datasets come from 3 device sets.

