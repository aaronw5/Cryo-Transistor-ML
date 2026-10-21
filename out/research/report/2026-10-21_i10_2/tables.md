# Tables, 2026-10-21 i10_2

Generated 2026-10-21 14:40 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i10_2.py --date 2026-10-21` from stored files only. GMR = geometric mean ratio of held-out strict paper RRMS (stored I9-2 rows of the selected class); < 1 = the first method is lower.

### I10-2 PRIMARY RESULT (in-force rule): stop. Class selection for the shared bias-only correction from training data, per selector

| selector | pooled GMR(S, C0_refit) [Bonferroni CI] | pooled GMR(S, mlp16) [Bonferroni CI] (C2) | pooled GMR(S, S_gate) [Bonferroni CI] | C1: per-dataset upper CI of GMR(S, C0_refit) <= 1.03 | C2: pooled upper CI of GMR(S, mlp16) <= 1.03 | beats mlp16 (pooled upper < 1) | label |
|---|---:|---:|---:|---|---|---|---|
| S_loss | 0.838 [0.778, 0.900] | 0.914 [0.827, 0.992] | 0.938 [0.917, 0.958] | fails (TU Delft 4.13 K) | holds | yes | partly supported |
| S_refit | 0.834 [0.776, 0.897] | 0.910 [0.828, 0.979] | 0.934 [0.912, 0.959] | fails (IHP 300 K, TU Delft 4.13 K) | holds | yes | partly supported |
| S_gate | 0.893 [0.836, 0.951] | 0.974 [0.885, 1.049] | - | fails (TU Delft 4.13 K) | fails | no | not supported |
| S_cv | 0.814 [0.742, 0.888] | 0.888 [0.797, 0.964] | 0.912 [0.865, 0.948] | fails (TU Delft 4.13 K) | holds | yes | partly supported |
| PRIMARY RULE (in force) |  |  |  |  |  |  | stop (supported: none) |

- PRIMARY RESULT. Pre-registration out/research/explore/I10_2/prereg.json in force from commit 55c4cfc (code d16c929); decision out/research/explore/I10_2/decision.json. Every value is recomputed here from the per-device rows and checked against decision.json.
- Held-out score of a device = the stored I9-2 row (held-out card refit with g frozen, strict paper RRMS) of the class selected in its outer fold from that fold's training devices; none = stored C0_refit row. Pooled GMR = exp(mean over the 4 datasets of the per-dataset mean log ratio); equal dataset weight; 20000 stratified resamples, TU Delft resampled by geometry group (7 clusters); level 1 - 0.05/11 (alpha 0.00455).
- Rule: supported = C1 and C2; partly supported = one of them; overall 'stop' if no selector is supported: class selection from training data is not supported on DEV, the deployed rule stays the I8-2 gate between C0_refit and M_bias, and no class selector goes to TEST.
- S_cv is not strictly nested: the stored row of a training device was produced with a g trained on folds that include fold f's starting-card records. The amendment A1 secondary C1 report is in table 07 and does not change any label or the rule.


### I10-2 compute per selector and dataset (new CPU time; S_gate and S_cv use stored values)

| dataset | outer folds / inner folds per outer | inner trainings / CPU-h | inner refits / CPU-h | refit scope | failed inner refits | S_loss CPU-h | S_refit CPU-h (training + refits) | wall s (inner / refit) | V_fold check CPU s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| sky130 77 K | 12 / 11 | 528 / 0.75 | 528 / 2.30 | all | 0 | 0.75 | 3.04 | 453 / 1383 | 16 |
| NIST 4 K | 53 / 4 | 848 / 1.63 | 2756 / 5.61 | one | 0 | 1.63 | 7.23 | 995 / 3372 | 104 |
| TU Delft 4.13 K | 7 / 6 | 168 / 1.33 | 1080 / 3.34 | all | 0 | 1.33 | 4.67 | 810 / 2017 | 45 |
| IHP 300 K | 18 / 7,9 | 584 / 0.19 | 584 / 1.90 | all | 0 | 0.19 | 2.09 | 114 / 1147 | 8 |
| total | 90 outer | 2128 / 3.89 | 4948 / 13.15 |  | 0 | 3.89 | 17.04 | 2372 / 7920 | 172 |

- CPU time = sum of the per-task seconds stored in inner/<dataset>/results.json (timing; 1 torch thread per task) and refit/<dataset>/results.json (rows runtime_s); 6 workers. Wall: the job's own runtime_s; the queue times are I10_2_inner 2372 s (4 datasets), refits sky130_77K 1384 s, ihp_meas_300K 1147 s, tudelft_4K 2017 s, nist_4K 3372 s, decide 2 s. Plan: 3.73 CPU-h training, 13.15 CPU-h refits.
- S_gate and S_cv: 0 new CPU-h in I10-2. S_gate reads the stored fold-level S3_f (M_bias recipe, seeds [0], 500 steps; computed in I7-1 retro / I8-1). S_cv reads the stored I9-2 held-out rows of every menu class; at TEST it needs those rows (leave-one-group-out runs with refits of 4 classes) on each DEV set.
- V_fold: lin retrained on each outer training set reproduces the stored I9-2 lin fold model (max |coefficient difference| 0 on all 4 datasets); not part of a selector's cost.


### I10-2 SECONDARY (amendment A1): criterion C1 under the TU Delft device bootstrap and at 95%, for the 4 selectors and fixed mlp16

| C1 variant | S_loss: C1 | TU Delft upper CI | S_refit: C1 | TU Delft upper CI | S_gate: C1 | TU Delft upper CI | S_cv: C1 | TU Delft upper CI | fixed_mlp16: C1 | TU Delft upper CI |
|---|---|---|---|---|---|
| primary: TU Delft cluster bootstrap, Bonferroni | fail: TU | 1.089 | fail: IHP, TU | 1.089 | fail: TU | 1.144 | fail: TU | 1.164 | fail: IHP, TU | 1.137 |
| (a) TU Delft device bootstrap, Bonferroni | fail: TU | 1.057 | fail: IHP, TU | 1.043 | fail: TU | 1.091 | fail: TU | 1.050 | fail: IHP, TU | 1.031 |
| (b) TU Delft cluster bootstrap, 95% | fail: TU | 1.065 | fail: IHP, TU | 1.056 | fail: TU | 1.111 | fail: TU | 1.099 | fail: IHP, TU | 1.081 |
| (a+b) TU Delft device bootstrap, 95% | fail: TU | 1.037 | fail: IHP | 1.021 | fail: TU | 1.071 | pass | 1.016 | fail: IHP | 1.003 |

- SECONDARY (amendment A1, in force from commit 7a93dfc, committed before I10_2_decide and before any S_loss / S_refit outer score). This table does not change the primary labels or the rule (table 01). The first row is the primary C1.
- TU Delft per-dataset GMR(S, C0_refit): S_loss 0.993; S_refit 0.976; S_gate 1.027; S_cv 0.939; fixed_mlp16 0.939. IHP 300 K under (a+b): S_loss 1.000 [1.000, 1.000]; S_refit 1.020 [0.997, 1.060]; S_gate 1.000 [1.000, 1.000]; S_cv 0.962 [0.914, 1.001]; fixed_mlp16 1.214 [1.008, 1.539].
- Under (a+b) S_cv passes C1 on every dataset; fixed mlp16 (M_bias in every fold) fails C1 under every variant (IHP 300 K in all four, TU Delft in three). Fail entries name the failing datasets.


### I10-2 secondary analyses: per-dataset GMR, fixed classes, oracle references, S_cv full menu, I8-2 gate, tie sensitivity, TU Delft device bootstrap, NIST room temperature

| item | sky130 77 K | NIST 4 K | TU Delft 4.13 K | IHP 300 K | pooled |
|---|---:|---:|---:|---:|---:|
| S_loss per dataset vs C0_refit (95%) | 0.741 [0.630, 0.869] | 0.670 [0.614, 0.734] | 0.993 [0.910, 1.065] | 1.000 [1.000, 1.000] | 0.838 [0.796, 0.880] |
| S_refit per dataset vs C0_refit (95%) | 0.727 [0.622, 0.850] | 0.670 [0.614, 0.734] | 0.976 [0.898, 1.056] | 1.020 [0.997, 1.060] | 0.834 [0.793, 0.877] |
| S_gate per dataset vs C0_refit (95%) | 0.769 [0.663, 0.887] | 0.806 [0.758, 0.865] | 1.027 [0.943, 1.111] | 1.000 [1.000, 1.000] | 0.893 [0.853, 0.934] |
| S_cv per dataset vs C0_refit (95%) | 0.727 [0.622, 0.850] | 0.670 [0.614, 0.734] | 0.939 [0.793, 1.099] | 0.962 [0.914, 1.001] | 0.814 [0.764, 0.866] |
| S_loss per dataset vs mlp16 (95%) | 0.964 [0.926, 1.001] | 0.831 [0.800, 0.863] | 1.058 [0.944, 1.200] | 0.823 [0.650, 0.992] | 0.914 [0.855, 0.969] |
| S_refit per dataset vs mlp16 (95%) | 0.946 [0.910, 0.983] | 0.831 [0.800, 0.863] | 1.040 [0.950, 1.155] | 0.840 [0.671, 0.996] | 0.910 [0.855, 0.960] |
| S_gate per dataset vs mlp16 (95%) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.095 [1.000, 1.212] | 0.823 [0.650, 0.992] | 0.974 [0.914, 1.028] |
| S_cv per dataset vs mlp16 (95%) | 0.946 [0.910, 0.983] | 0.831 [0.800, 0.863] | 1.001 [0.911, 1.115] | 0.792 [0.608, 0.965] | 0.888 [0.827, 0.942] |
| fixed lin vs C0_refit (95%) | 1.002 [0.998, 1.007] | 0.998 [0.993, 1.003] | 0.922 [0.893, 0.946] | 0.983 [0.917, 1.058] | 0.976 [0.957, 0.995] |
| fixed gam vs C0_refit (95%) | 0.727 [0.622, 0.850] | 0.670 [0.614, 0.734] | 0.874 [0.713, 1.072] | 1.471 [1.141, 1.987] | 0.889 [0.808, 0.982] |
| fixed mlp8 vs C0_refit (95%) | 0.759 [0.664, 0.866] | 0.797 [0.746, 0.858] | 0.919 [0.812, 1.036] | 1.202 [1.005, 1.491] | 0.904 [0.845, 0.971] |
| fixed mlp16 vs C0_refit (95%) | 0.769 [0.663, 0.887] | 0.806 [0.758, 0.865] | 0.939 [0.811, 1.081] | 1.214 [1.008, 1.539] | 0.917 [0.852, 0.992] |
| oracle class (post hoc) | gam | gam | gam | lin | gam |
| S_loss vs per-dataset oracle (post hoc, 95%) | 1.020 [1.000, 1.048] | 1.000 [1.000, 1.000] | 1.137 [0.961, 1.362] | 1.017 [0.945, 1.091] | 1.042 [0.995, 1.094] |
| S_refit vs per-dataset oracle (post hoc, 95%) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.117 [0.965, 1.314] | 1.038 [0.977, 1.103] | 1.038 [0.998, 1.083] |
| S_gate vs per-dataset oracle (post hoc, 95%) | 1.058 [1.017, 1.099] | 1.203 [1.159, 1.250] | 1.176 [1.013, 1.376] | 1.017 [0.945, 1.091] | 1.111 [1.064, 1.162] |
| S_cv vs per-dataset oracle (post hoc, 95%) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.075 [1.000, 1.241] | 0.978 [0.927, 1.026] | 1.013 [0.984, 1.052] |
| S_cv full 13-option menu (95%): picks / pooled vs C0_refit | gam 12 | gam 53 | lin 1, poly5 1, gam 5 | none 8, lin 10 | 0.822 [0.776, 0.869] |
| I8-2 dataset-level gate (95%): class / pooled vs C0_refit | mlp16 | mlp16 | mlp16 | none | 0.873 [0.826, 0.921] |
| S_loss tie sensitivity (Bonferroni), pooled vs C0_refit / vs mlp16 |  |  |  |  | TIE 0.0: 0.841 / 0.917; TIE 0.01: 0.834 / 0.910; TIE 0.02: 0.842 / 0.919 |
| S_refit tie sensitivity (Bonferroni), pooled vs C0_refit / vs mlp16 |  |  |  |  | TIE 0.0: 0.832 / 0.907; TIE 0.01: 0.834 / 0.910; TIE 0.02: 0.832 / 0.907 |
| S_cv tie sensitivity (Bonferroni), pooled vs C0_refit / vs mlp16 |  |  |  |  | TIE 0.0: 0.814 / 0.888; TIE 0.01: 0.824 / 0.899; TIE 0.02: 0.824 / 0.899 |
| S_loss TU Delft device bootstrap (Bonferroni) |  |  | 0.993 [0.934, 1.057] |  | 0.838 [0.784, 0.898] |
| S_refit TU Delft device bootstrap (Bonferroni) |  |  | 0.976 [0.917, 1.043] |  | 0.834 [0.781, 0.895] |
| S_gate TU Delft device bootstrap (Bonferroni) |  |  | 1.027 [0.970, 1.091] |  | 0.893 [0.842, 0.950] |
| S_cv TU Delft device bootstrap (Bonferroni) |  |  | 0.939 [0.825, 1.050] |  | 0.814 [0.759, 0.876] |
| NIST room temperature (I9-1) S_gate, menu none/poly3/mlp16 (95%) |  | mlp16 87/87 folds |  |  | vs C0_refit 0.797 [0.749, 0.859] |
| NIST room temperature (I9-1) S_cv, menu none/poly3/mlp16 (95%) |  | mlp16 87/87 folds |  |  | vs C0_refit 0.797 [0.749, 0.859] |

- Secondary analyses of the pre-registration (descriptive, 95% unless marked; TU Delft cluster bootstrap unless marked). Oracle rows use held-out outcomes and cannot be deployed (D15). NIST room temperature row: the 'NIST 4 K' column holds the I9-1 NIST room-temperature selection count (87 folds).
- All values recomputed from the per-device rows and checked against decision.json (secondary, post_hoc_references).

