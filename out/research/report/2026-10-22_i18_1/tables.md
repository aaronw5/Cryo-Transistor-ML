# Tables, 2026-10-22 i18_1

Generated 2026-10-22 23:15 by `.venv/bin/python -I scripts/research/plot_explore_i18_1.py --date 2026-10-22`.

### I18-1 pre-registered rule, validity and outcome

| item | value |
|---|---|
| validity | valid |
| C1 never worse: CI lower of GMR(gated, C0_refit) <= 1 on all 9 datasets | holds |
| C2 sky130 gain: CI upper < 1 on sky130 77 K | holds |
| outcome (pre-registered rule) | supported |
| predicted outcome | supported |
| TU Delft rows | I18-2 |
| gate | apply iff S3_f < 0 (strict); not fitted (sign used by I8-2 G and I10-2 S_gate) |
| sign convention | S3_f < 0: the reduced correction lowered the starting-card RRMS on inner held-out training records (predicted gain) |
| pooled exp(mean ln GMR) over 9 datasets (reported): gated / correction always | 0.929 / 0.958 |

- Pre-registration commit b384a42; code 8fc4f32.


### I18-1 per dataset: gate decisions and GMR(gated, C0_refit)

| dataset | gated arm | unit | folds applied | predicted | GMR(gated, C0_refit) [95% CI] | CI lower <= 1 | GMR(correction always, C0_refit) | gated, other unit CI |
|---|---|---:|---:|---|---:|---:|---:|---:|
| sky130 77 K (all 18) | selected g v2 (I15-2) | device | 18/18 (18/18 dev) | 18/18 | 0.743 [0.654, 0.855] | yes | 0.743 [0.654, 0.855] | 0.743 [0.654, 0.855] |
| NIST sky130 4 K lvt (I8-1) | M_bias v1 | device | 53/53 (53/53 dev) | 53/53 (determined) | 0.806 [0.758, 0.864] | yes | 0.806 [0.758, 0.864] | 0.806 [0.758, 0.864] |
| NIST sky130 4 K, RT card (I9-1) | M_bias v1 | device | 87/87 (87/87 dev) | 87/87 (determined) | 0.797 [0.748, 0.859] | yes | 0.797 [0.748, 0.859] | 0.797 [0.748, 0.859] |
| IHP SG13G2 300 K | selected g v2 (I17-2) | device | 0/18 (0/18 dev) | 0/18 | 1.000 [1.000, 1.000] | yes | 1.255 [1.026, 1.600] | 1.000 [1.000, 1.000] |
| IHP SG13G2 233 K | M_bias v1 | device | 0/18 (0/18 dev) | 0/18 each (determined) | 1.000 [1.000, 1.000] | yes | 1.106 [0.908, 1.425] | 1.000 [1.000, 1.000] |
| IHP SG13G2 343 K | M_bias v1 | device | 0/18 (0/18 dev) | 0/18 each (determined) | 1.000 [1.000, 1.000] | yes | 1.181 [0.983, 1.502] | 1.000 [1.000, 1.000] |
| IHP SG13G2 398 K | M_bias v1 | device | 0/18 (0/18 dev) | 0/18 each (determined) | 1.000 [1.000, 1.000] | yes | 1.272 [1.030, 1.627] | 1.000 [1.000, 1.000] |
| TU Delft 4.13 K | selected g v2 (I18-2, state C) | cluster | 2/7 (16/45 dev) | 5/7 groups | 0.981 [0.863, 1.097] | yes | 0.826 [0.669, 1.014] | 0.981 [0.936, 1.028] |
| TU Delft 295.0 K | selected g v2 (I18-2, state C) | cluster | 1/7 (8/45 dev) | 3/7 groups | 1.099 [1.000, 1.327] | yes | 0.829 [0.523, 1.247] | 1.099 [1.029, 1.190] |

- Gate: correction row for a fold's held-out devices iff S3_f < 0 (training devices only); otherwise the C0_refit row. Other unit: device bootstrap for TU Delft, per-fold cluster (= device) elsewhere.


### I18-1 secondary: the same gate with M_bias v1 and the v1-recipe S3 (reported, no rule)

| dataset | folds applied | GMR(v1-gated, C0_refit) | GMR(v1 always, C0_refit) | dataset-level gate S3_D -> GMR | Spearman(S3_f, y_f) |
|---|---:|---:|---:|---:|---:|
| sky130 77 K (all 18) | 18/18 | 0.745 [0.660, 0.850] | 0.745 [0.660, 0.850] | -0.163 -> 0.745 | -0.31 |
| NIST sky130 4 K lvt (I8-1) | 53/53 | 0.806 [0.758, 0.864] | 0.806 [0.758, 0.864] | -0.108 -> 0.806 | -0.38 |
| NIST sky130 4 K, RT card (I9-1) | 87/87 | 0.797 [0.748, 0.859] | 0.797 [0.748, 0.859] | -0.071 -> 0.797 | -0.60 |
| IHP SG13G2 233 K | 0/18 | 1.000 [1.000, 1.000] | 1.106 [0.908, 1.425] | +0.254 -> 1.000 | -0.09 |
| IHP SG13G2 300 K | 0/18 | 1.000 [1.000, 1.000] | 1.214 [1.008, 1.537] | +0.366 -> 1.000 | -0.22 |
| IHP SG13G2 343 K | 0/18 | 1.000 [1.000, 1.000] | 1.181 [0.983, 1.502] | +0.375 -> 1.000 | +0.25 |
| IHP SG13G2 398 K | 0/18 | 1.000 [1.000, 1.000] | 1.272 [1.030, 1.627] | +0.395 -> 1.000 | +0.45 |
| TU Delft 4.13 K | 5/7 | 1.027 [0.943, 1.112] | 0.939 [0.811, 1.081] | -0.033 -> 0.939 | -0.93 |
| TU Delft 295.0 K | 3/7 | 1.040 [0.868, 1.228] | 0.797 [0.576, 1.087] | -0.041 -> 0.797 | -0.43 |

- sky130: state-C rows (I13-1), S3_v1 recomputed on 18 folds; TU Delft: I6-1 stored rows (state L) and stored I7-1 S3; other datasets: stored rows and stored S3.


### I18-1 S3_v2 per fold (g v2 recipe at lambda*_f, seed 0, 500 steps)

| dataset | folds | S3_f < 0 | range | S3_D (weighted mean) | Spearman(S3_f, y_f) | post hoc oracle GMR |
|---|---:|---:|---:|---:|---:|---:|
| sky130 77 K (all 18) | 18 | 18 | [-0.214, -0.100] | -0.175 | -0.37 | 0.743 |
| IHP SG13G2 300 K | 18 | 0 | [+0.184, +0.642] | +0.353 | -0.12 | 1.000 |
| TU Delft 4.13 K | 7 | 2 | [-0.039, +0.091] | +0.018 | -0.21 | 0.826 |
| TU Delft 295.0 K | 7 | 1 | [-0.106, +0.471] | +0.201 | -0.68 | 0.829 |

- y_f: mean held-out ln(RRMS correction / RRMS C0_refit) of fold f. Oracle: min(1, GMR(correction, C0_refit)) chosen with the outcome known (reference only).

