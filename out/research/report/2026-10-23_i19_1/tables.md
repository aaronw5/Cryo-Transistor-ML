# Tables, 2026-10-23 i19_1

Generated 2026-10-23 02:57 by `.venv/bin/python -I scripts/research/plot_explore_i19_1.py --date 2026-10-23`.

### I19-1 pre-registered rules, validity and outcome

| item | value |
|---|---|
| validity | valid |
| V2 S3 code reproduces stored fold values (3 folds) | max |diff| 0.0e+00 |
| R1: GMR(gated_D, C0_refit) <= 1 on each of 9 DEV datasets | holds |
| R2: pooled gated_D below pooled correction always | holds (0.921 vs 0.958) |
| Part R outcome (retrospective) | supported (retrospective) |
| C: simulated sign accuracy >= 19/20 | holds (19/20; constant apply 18/20; wrong: gf180|0|0.0) |
| overall outcome (pre-registered) | supported |
| threshold (fixed before any S3_D) | 0.0 (strict S3_D < 0) |

- Pre-registration commit 45b4f37; code 4da313c. Part R is retrospective: DEV outcomes and the I8-2 gate G values were known.


### I19-1 Part R (retrospective): dataset-level decision and GMR per DEV dataset

| dataset | gated arm | S3_D | decision | at +0.02455 | I8-2 G: fold-mean S3 | GMR(gated_D, C0_refit) [95% CI] | per-fold gate GMR (folds) | GMR(correction always, C0_refit) | decision = realised sign |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sky130 77 K (all 18) | selected g v2 (I15-2) | -0.1328 | apply | apply | -0.175 (apply) | 0.743 [0.654, 0.855] | 0.743 (18/18) | 0.743 [0.654, 0.855] | yes |
| NIST sky130 4 K lvt (I8-1) | M_bias v1 | -0.1027 | apply | apply | -0.108 (apply) | 0.806 [0.758, 0.864] | 0.806 (53/53) | 0.806 [0.758, 0.864] | yes |
| NIST sky130 4 K, RT card (I9-1) | M_bias v1 | -0.0689 | apply | apply | -0.071 (apply) | 0.797 [0.748, 0.859] | 0.797 (87/87) | 0.797 [0.748, 0.859] | yes |
| IHP SG13G2 300 K | selected g v2 (I17-2) | +0.3078 | C0_refit | C0_refit | +0.353 (C0_refit) | 1.000 [1.000, 1.000] | 1.000 (0/18) | 1.255 [1.026, 1.600] | yes |
| IHP SG13G2 233 K | M_bias v1 | +0.1866 | C0_refit | C0_refit | +0.254 (C0_refit) | 1.000 [1.000, 1.000] | 1.000 (0/18) | 1.106 [0.908, 1.425] | yes |
| IHP SG13G2 343 K | M_bias v1 | +0.3669 | C0_refit | C0_refit | +0.375 (C0_refit) | 1.000 [1.000, 1.000] | 1.000 (0/18) | 1.181 [0.983, 1.502] | yes |
| IHP SG13G2 398 K | M_bias v1 | +0.3667 | C0_refit | C0_refit | +0.395 (C0_refit) | 1.000 [1.000, 1.000] | 1.000 (0/18) | 1.272 [1.030, 1.627] | yes |
| TU Delft 4.13 K | selected g v2 (I18-2, state C) | +0.0080 | C0_refit | apply | +0.018 (C0_refit) | 1.000 [1.000, 1.000] | 0.981 (2/7) | 0.826 [0.669, 1.014] | no |
| TU Delft 295.0 K | selected g v2 (I18-2, state C) | +0.2640 | C0_refit | C0_refit | +0.201 (C0_refit) | 1.000 [1.000, 1.000] | 1.099 (1/7) | 0.829 [0.523, 1.247] | no |

- S3_D: mean over all DEV records of ln(RRMS(start x exp(g)) / RRMS(start)), g trained with the gated arm's recipe (seed 0, 500 steps) on the LOGO fold that holds out the record's group (IHP: same class). Decision: apply iff S3_D < 0. Per-fold gate: I18-1.


### I19-1 Part R pooled exp(mean ln GMR vs C0_refit), equal weight per dataset

| method | 9 datasets | 7 (without TU Delft) |
|---|---:|---:|
| gated_D (I19-1) | 0.921 | 0.900 |
| per-fold gate (I18-1) | 0.929 | 0.900 |
| correction always | 0.958 | 0.999 |
| I8-2 gate G (fold-mean S3 < 0) | 0.921 | - |
| post hoc dataset oracle (outcome known) | 0.883 | - |
| decision = realised sign of ln GMR(correction, C0_refit) | 7/9 | - |
| v1 secondary: gated_D / v1 always | 0.892 / 0.965 | - |

- The 4 IHP temperatures share DUTs and the 2 TU Delft temperatures share devices.


### I19-1 secondary: dataset-level gate with M_bias v1 and the v1 recipe (reported, no rule)

| dataset | S3_D (v1) | decision | GMR(gated_D, C0_refit) | GMR(v1 always, C0_refit) | decision = realised sign |
|---|---:|---:|---:|---:|---:|
| sky130 77 K (all 18) | -0.1777 | apply | 0.745 [0.660, 0.850] | 0.745 [0.660, 0.850] | yes |
| NIST sky130 4 K lvt (I8-1) | -0.1027 | apply | 0.806 [0.758, 0.864] | 0.806 [0.758, 0.864] | yes |
| NIST sky130 4 K, RT card (I9-1) | -0.0689 | apply | 0.797 [0.748, 0.859] | 0.797 [0.748, 0.859] | yes |
| IHP SG13G2 233 K | +0.1866 | C0_refit | 1.000 [1.000, 1.000] | 1.106 [0.908, 1.425] | yes |
| IHP SG13G2 300 K | +0.3114 | C0_refit | 1.000 [1.000, 1.000] | 1.214 [1.008, 1.537] | yes |
| IHP SG13G2 343 K | +0.3669 | C0_refit | 1.000 [1.000, 1.000] | 1.181 [0.983, 1.502] | yes |
| IHP SG13G2 398 K | +0.3667 | C0_refit | 1.000 [1.000, 1.000] | 1.272 [1.030, 1.627] | yes |
| TU Delft 4.13 K | -0.0292 | apply | 0.939 [0.811, 1.081] | 0.939 [0.811, 1.081] | yes |
| TU Delft 295.0 K | -0.0510 | apply | 0.797 [0.576, 1.087] | 0.797 [0.576, 1.087] | yes |

- sky130: state-C records and rows (I13-1); TU Delft: I6-1 state-L records and rows; other datasets: the primary v1 values.


### I19-1 Part P: committed TEST gate decisions (TEST not evaluated)

| TEST split | n TEST | DEV records | arm | S3_D (DEV) | decision | at +0.02455 | v1 S3_D (decision) | I8-2 S3 / S1 prediction |
|---|---:|---:|---|---:|---:|---:|---:|---|
| NIST 4 K lvt TEST (I8-1 split) | 38 | 53 | M_bias v1 | -0.1027 | apply | apply | - | DEV P_c gain |
| NIST 4 K room-temperature card TEST (I9-1 split) | 58 | 87 | M_bias v1 | -0.0689 | apply | apply | - | DEV P_c gain |
| IHP SG13G2 TEST 233 K | 10 | 18 | M_bias v1 | +0.1866 | C0_refit | C0_refit | - | loss / loss |
| IHP SG13G2 TEST 300 K | 10 | 18 | selected g v2 (I17-2) | +0.3078 | C0_refit | C0_refit | +0.3114 (C0_refit) | loss / loss |
| IHP SG13G2 TEST 343 K | 10 | 18 | M_bias v1 | +0.3669 | C0_refit | C0_refit | - | loss / loss |
| IHP SG13G2 TEST 398 K | 10 | 18 | M_bias v1 | +0.3667 | C0_refit | C0_refit | - | loss / loss |
| TU Delft TEST 4.13 K | 24 | 45 | selected g v2 (I18-2, state C) | +0.0080 | C0_refit | apply | -0.0292 (apply) | gain / gain |
| TU Delft TEST 295.0 K | 24 | 45 | selected g v2 (I18-2, state C) | +0.2640 | C0_refit | C0_refit | -0.0510 (apply) | gain / gain |

- Committed in its own commit before the decide step; written at git HEAD 4da313c. Disagreements with I8-2 (I8-2 predicts the sign for M_bias v1): tudelft_4K I19-1 primary C0_refit vs I8-2 S3_predicts gain; tudelft_4K I19-1 primary C0_refit vs I8-2 S1_predicts gain; tudelft_295K I19-1 primary C0_refit vs I8-2 S3_predicts gain; tudelft_295K I19-1 primary C0_refit vs I8-2 S1_predicts gain.


### I19-1 Part C: 20 simulated I7-1 datasets, S3_D (v1) vs realised y_D

| dataset (PDK, replicate, lambda) | S3_D | I7-1 fold-mean S3 | y_D = ln GMR(M_bias, C0_refit) | decision | correct |
|---|---:|---:|---:|---:|---:|
| gf180 r0 lambda 0.0 | +0.0003 | +0.0062 | -0.0449 | C0_refit | NO |
| gf180 r0 lambda 0.25 | -0.0596 | -0.0464 | -0.1429 | apply | yes |
| gf180 r0 lambda 0.5 | -0.2904 | -0.2685 | -0.3885 | apply | yes |
| gf180 r0 lambda 0.75 | -0.5658 | -0.5513 | -0.7487 | apply | yes |
| gf180 r0 lambda 1.0 | -1.4042 | -1.3940 | -1.7125 | apply | yes |
| gf180 r1 lambda 0.0 | +0.0832 | +0.0777 | +0.0708 | C0_refit | yes |
| gf180 r1 lambda 0.25 | -0.0120 | -0.0114 | -0.0392 | apply | yes |
| gf180 r1 lambda 0.5 | -0.1894 | -0.1854 | -0.2355 | apply | yes |
| gf180 r1 lambda 0.75 | -0.4550 | -0.4583 | -0.5564 | apply | yes |
| gf180 r1 lambda 1.0 | -1.1025 | -1.0973 | -1.5150 | apply | yes |
| ihp r0 lambda 0.0 | +0.0449 | +0.0609 | +0.0297 | C0_refit | yes |
| ihp r0 lambda 0.25 | -0.1181 | -0.1108 | -0.1433 | apply | yes |
| ihp r0 lambda 0.5 | -0.3155 | -0.3073 | -0.3697 | apply | yes |
| ihp r0 lambda 0.75 | -0.6186 | -0.6076 | -0.6959 | apply | yes |
| ihp r0 lambda 1.0 | -1.2310 | -1.2284 | -1.5380 | apply | yes |
| ihp r1 lambda 0.0 | -0.1021 | -0.1024 | -0.1308 | apply | yes |
| ihp r1 lambda 0.25 | -0.0061 | -0.0027 | -0.0367 | apply | yes |
| ihp r1 lambda 0.5 | -0.1879 | -0.1800 | -0.2426 | apply | yes |
| ihp r1 lambda 0.75 | -0.4740 | -0.4708 | -0.5731 | apply | yes |
| ihp r1 lambda 1.0 | -1.2780 | -1.2860 | -1.4972 | apply | yes |

- Spearman(S3_D, y_D) +0.985; OLS y = -0.0215 + 1.217 S3_D (R^2 0.993); pooled GMR gated_D 0.590, M_bias always 0.591, oracle 0.588; sensitivity threshold +0.02455: 20/20 correct.

