# I10-2 pre-registration (in force): selection of the class of the shared bias-only correction from training data

Status: IN FORCE (this file and prereg.json committed alone, before any I10-2 selector result). Date 2026-10-21. Code commit d16c929 (library `src/cryoml/explore/i10_2_select.py`, driver `scripts/research/explore/i10_2_select.py`, 19 unit tests `tests/test_i10_2_select.py`, smoke, plan).

## Question

Can the function class of the shared bias-only correction g (I = I_card x exp(g)), including "no correction" (none = C0_refit), be selected per dataset from training data only, so that the selected method is not worse than C0_refit on any dataset and close to the best fixed class?

Retrospective DEV analysis. The class menu, the four datasets, the I9-2 per-class held-out rows and the I8-2 S3 values were all seen before this pre-registration (I9-2 outcome, I8-2 gate). The prospective test is the December TEST stage (protocol below).

## Background (stored results)

- I9-2 per dataset GMR(X, C0_refit), 95%, descriptive: sky130 77 K lin 1.002, gam 0.727, mlp8 0.759, mlp16 0.769; NIST 4 K lin 0.998, gam 0.670, mlp8 0.797, mlp16 0.806; TU Delft 4.13 K lin 0.922, gam 0.874, mlp8 0.919, mlp16 0.939; IHP measured 300 K lin 0.983, gam 1.471, mlp8 1.202, mlp16 1.214. Pooled: lin 0.976, gam 0.889, mlp8 0.904, mlp16 0.917.
- I9-2 labels vs mlp16 (Bonferroni): gam inconclusive 0.970 [0.922, 1.030], mlp8 non-inferior, mlp16_s1 and mlp32 equivalent, poly2 / poly3 / poly4 / gp worse.
- I8-2: dataset-level gate G (M_bias iff S3(D) < 0) never above 1.02 vs C0_refit on 31 datasets; the fold-level S3 gate (diagnostic) gave 5/31 above 1.02.
- Stored S3_f signs (S3_f < 0 = gate picks mlp16): sky130 77 K 12/12 folds, NIST 4 K 53/53, TU Delft 4.13 K 5/7, IHP measured 300 K 0/18, NIST room temperature (I9-1) 87/87. The S_gate selections below are therefore known before this pre-registration.

## Decisions

| id | decision | reason |
|---|---|---|
| D1 | Primary class menu M5 = {none, lin, gam, mlp8, mlp16} (0, 6, 66, 565, 1765 trained values). Secondary: S_cv on the full I9-2 menu (none + 12 classes, mlp16_rep excluded). | One class per capacity step from no correction to M_bias. poly2-poly5 lie between lin and gam in size and three were labelled worse in I9-2; mlp16_s1 and mlp32 were equivalent to mlp16; gp was worse, had 2 failed devices and costs about 30 s more per refit. Four refit classes keep S_refit at 13 CPU-h. The menu was chosen with the I9-2 results known. |
| D2 | Selection unit = outer fold of the stored I9-2 runs (sky130 77 K, NIST 4 K, IHP 300 K: leave-one-device-out, IHP within class LV / HV; TU Delft: leave-one-group-out, 7 groups). In each outer fold the selector uses only that fold's training devices (`d92.load_records(...)["train"][fold]`). | Simulates the per-dataset selection with the training set that produced each stored held-out row. At TEST there is one selection per dataset on all DEV devices. |
| D3 | Held-out score of fold f = the stored I9-2 per-device row (`out/research/explore/I9_2/<dataset>/results.json`, `per_device`) of the class selected in f, for every device of f; none -> the stored C0_refit row. No outer refit is run. NIST: the 4 non-finite starting records are NaN (failed) in every series. | Each stored row of class X at device d was produced by the class-X model of fold(d), trained on `train[fold(d)]`, with the held-out card refit with g frozen (I9-2 protocol). Checks: V_rows and V_fold below. |
| D4 | Inner folds (`inner_folds`): groups (device, or TU Delft geometry group) of the outer training set sorted by (family of the first device, group name). Inner leave-one-group-out when the outer training set has at most 12 groups (sky130 11, TU Delft 6, IHP 7-9); otherwise 4 inner folds, group i -> inner fold i mod 4 (NIST 52 devices -> 4 folds of 13). | Small training sets lose a large fraction of their devices in a 4-fold split (sky130 11 -> 8; the I9-2 learning curves show the gain depends on the number of training devices below 8). On NIST 4 folds keep 39 of 52 devices; leave-one-out there would cost 13x more training. |
| D5 | Inner training recipe: the I9-2 class settings (E1 loss + L2 1e-4, Adam lr 3e-3, 2000 full-batch steps, float64, 1 torch thread); MLP classes (mlp8, mlp16) with member seed 0 only (`fit_class(..., {"seeds": (0,)})`). | 5x less MLP training. I9-2: mlp16_s1 (seed 0 only) vs mlp16 0.999 [0.982, 1.015], equivalent. The outer score still uses the stored 5-member class row. |
| D6 | S_loss score of class X in fold f: mean over all inner held-out devices (pooled over the inner folds) of ln(RRMS(I_sim x exp(g_X)) / RRMS(I_sim)) on the starting-card records, g_X trained on the inner-training devices (`i7_1_transfer.s3_fold_value`, strict paper RRMS, frozen inclusion). No refit. none = 0. | The log ratio has the scale of the primary GMR statistic and of S3; the E1 loss value itself (mean RRMS) weights devices by their RRMS. |
| D7 | S_refit score: the same inner corrections (from the inner job); each inner held-out card is refit with g_X frozen through the I9-2 refit path with the outer settings (ngspice_fit.polish, max_nfev 400, irls_rounds 2, diff_step 0.01; ngspice-41 arm64, IHP ngspice-46 + PSP 103 OSDI); score ln(RRMS_refit,X / RRMS_C0_refit) with the stored C0_refit row of the inner held-out device; none = 0. Refit scope: all inner held-out devices on sky130 77 K, TU Delft 4.13 K and IHP 300 K; on NIST 4 K only inner fold (outer fold position mod 4), 13 devices per outer fold. | The uncorrected refit of a device starts from its own stored starting card and uses only its own data, so the stored C0_refit row is the inner reference for any fold. All NIST inner devices would be 11024 refits (26 CPU-h); one inner fold gives 2756 refits (6.5 CPU-h). |
| D8 | S_gate: mlp16 iff the stored fold-level S3_f < 0, else none (S3_f: inner leave-one-group-out of the M_bias recipe with seed 0, 500 steps, post hoc gain; `I7_1/retro/s3.json` for sky130 77 K, TU Delft 4.13 K, IHP 300 K; `I8_1/prediction_v2.json` for NIST 4 K; `I9_1/prediction.json` for NIST room temperature). The I8-2 dataset-level gate (mean of S3_f) is reported as a secondary. | S3_f uses only fold f's training devices; the dataset-level S3(D) contains the scored device's starting record (I8-2 leakage note). |
| D9 | S_cv: score of class X in fold f = mean over the fold's training devices of ln(RRMS_X / RRMS_C0_refit) from the stored I9-2 rows; none = 0. | No new computation. Not strictly nested: the stored row of a training device d was produced with a g trained on all folds except fold(d), which includes fold f's starting-card records (weight about 1/n_train; TU Delft about 1/6 of the training points); held-out refit outcomes of fold f are not used. At TEST the same selector uses DEV rows only and is nested. The S_cv vs S_refit agreement is reported. |
| D10 | Selection rule (`select_class`): the class with the lowest score; none scores 0 and is always selectable; classes within TIE = 0.005 of the lowest score are tied and the class with fewer trained values is chosen (none first); a class with a non-finite inner value on a device whose reference value (starting RRMS, or C0_refit) is finite is not selectable in that fold. Tie sensitivity (TIE 0, 0.01, 0.02) is secondary. | Prefers the smaller class when the inner evidence does not separate classes by more than 0.5% in GMR; a failed refit of a class is evidence against deploying it. |
| D11 | Statistics: equal-dataset-weight pooled GMR over the 4 primary datasets (`i9_2_classes.pooled_ratio`), 20000 stratified resamples, default_rng(0); TU Delft resampled by geometry group (cluster bootstrap) in the primary analysis, device bootstrap secondary. | On TU Delft the selection unit is the group, so devices of one group share the selection. |
| D12 | Multiplicity: 11 primary contrasts at the Bonferroni level 1 - 0.05/11 (alpha 0.004545): GMR(S, C0_refit) for the 4 selectors, GMR(S, mlp16) for the 4 selectors, GMR(S, S_gate) for S_loss, S_refit, S_cv. Per-dataset CIs are taken at the same level; criterion C1 (all datasets) is an intersection-union test and needs no further adjustment. Every other interval is 95%, descriptive. | |
| D13 | Margin 1.03 (as I9-2 D2). | ln 1.03 is 11-14% of the stored M_bias log gain on sky130 77 K and NIST 4 K. |
| D14 | NIST room temperature DEV (I9-1, 87 devices, leave-one-device-out) is secondary: its stored rows hold C0_refit, M_bias (= mlp16) and B3_d3 (= poly3) only, not lin / gam / mlp8. Reported: S_gate and S_cv over {none, poly3, mlp16}, GMR vs C0_refit and vs mlp16 (95%). | The M5 menu has no stored rows there; new rows would need 87-fold training and refits of 3 classes. |
| D15 | Post hoc references (not tests): the oracle pooled class (lowest pooled GMR(X, C0_refit) over M5; from the I9-2 rows this is gam) and the per-dataset oracle class (lowest per-dataset GMR; from the I9-2 rows: gam on sky130 / NIST / TU Delft, lin on IHP 300 K). GMR(S, oracle) with 95% CIs. | The oracle uses held-out outcomes and cannot be deployed. |

## Selectors

| selector | inputs (outer fold f) | new compute |
|---|---|---|
| S_loss | inner corrections of lin, gam, mlp8 (seed 0), mlp16 (seed 0) on the inner-training devices; post hoc log ratio on the inner held-out starting records | inner training only |
| S_refit | the same inner corrections; inner held-out cards refit with g frozen; log ratio vs the stored C0_refit | inner refits (queue) |
| S_gate | stored S3_f | none |
| S_cv | stored I9-2 rows of the fold's training devices | none |

## Decision rule

Per selector S (`selector_label`):

- C1 (not worse than C0_refit): on every primary dataset the upper bound of the Bonferroni CI of GMR(S, C0_refit) <= 1.03.
- C2 (close to M_bias): upper bound of the pooled Bonferroni CI of GMR(S, mlp16) <= 1.03.
- Label: 'supported' if C1 and C2; 'partly supported' if one holds; 'not supported' otherwise. Also reported: 'beats mlp16' (pooled upper < 1), pooled GMR(S, C0_refit) and its CI.

Overall (`overall_decision`):

- 'stop' if no selector is 'supported': selection of the class from training data is not supported on DEV; the deployed rule stays the I8-2 gate between C0_refit and M_bias, and no class selector goes to TEST.
- 'continue: class selection adds to the gate' if a supported multi-class selector (S_loss, S_refit, S_cv) beats S_gate (upper bound of the pooled Bonferroni CI of GMR(S, S_gate) < 1); frozen TEST selector = the one of those with the lowest pooled GMR(S, S_gate).
- else 'continue: gate only' if S_gate is supported; frozen TEST selector = S_gate.
- else 'continue: multi-class selector': frozen TEST selector = the supported selector with the lowest pooled GMR(S, C0_refit) (ties within 0.005: cheaper selector, order S_gate, S_cv, S_loss, S_refit).

## Validity checks

- V_rows (decide): every held-out device has a value (finite, or NaN = failed) in the C0_refit row and in the row of every class of the full menu; the fold model of every class exists for every fold (I9-2 `corrections.json` key `<class>|<fold>` for the classes trained in I9-2 and mlp16_s1; stored model files for the stored classes); NIST non-finite starting records are NaN in every class. Checked before this pre-registration (presence only): pass on all 4 datasets.
- V_fold (inner job): lin retrained on each outer fold's full training set with the I9-2 settings reproduces the stored I9-2 lin fold model, max |coefficient difference| <= 1e-12 (the fold -> training-set mapping equals I9-2's). A dataset failing V_fold is excluded and reported; the refit and decide modes refuse it. Smoke: 0 on 2 sky130 folds. The stored classes not trained in I9-2 (mlp16 everywhere; poly2, poly3, gp, mlp8, mlp32 on sky130) come from the stored runs on the same folds (I9-2 V1: poly3 retrain and refit reproduced the stored B3_d3 rows and coefficients with difference 0 on 2 devices per dataset).
- V_s3: the stored S3_f keys equal the I9-2 folds on every dataset (checked: pass; NIST room temperature 87/87).
- Failed inner refits per class and dataset are reported (S_refit); a failed inner value blocks the class in that fold (D10).

## Secondary analyses

- Selection counts per selector, dataset and class (folds and devices).
- Agreement of the selections between every pair of selectors (fraction of folds); S_cv vs S_refit measures the effect of the S_cv leak plus the seed-0 / inner-training-set differences.
- Per-dataset GMR(S, C0_refit), GMR(S, mlp16), GMR(S, S_gate) at 95%.
- TU Delft device bootstrap instead of the cluster bootstrap.
- Tie sensitivity (TIE 0, 0.01, 0.02) for S_loss, S_refit, S_cv.
- S_cv on the full I9-2 menu (13 options).
- I8-2 dataset-level gate on the 4 datasets.
- NIST room temperature (D14).
- Post hoc oracle references (D15).

## TEST protocol (December, prospective; frozen selector)

- Before any TEST array is loaded: the selector frozen by the rule above (menu M5, inner design D4, recipe D5, score D6 / D7, TIE 0.005, gate D8 as applicable) is run on all DEV devices of each TEST dataset's DEV counterpart (training set at TEST = all DEV devices; IHP within class and temperature). For S_loss / S_refit / S_cv this needs the DEV leave-one-group-out rows or inner runs of lin, gam, mlp8, mlp16 on the DEV sets that I9-2 did not cover (IHP 233 / 343 / 398 K, TU Delft 295 K, NIST TEST counterparts); these DEV-only jobs are pre-registered separately before they run.
- The selected class per TEST dataset is written to a file and committed before TEST; then the selected class is trained on all DEV starting-card records (I9-2 class recipe, 5 seeds for MLP classes) and each TEST device is refit with g frozen (max_nfev 400, irls 2, diff_step 0.01) from its TEST starting card (I8-2 Part A TEST scoring protocol).
- TEST datasets: the 7 I8-2 TEST datasets (sky130 77 K REAL_TEST, IHP SG13G2 TEST 233 / 300 / 343 / 398 K, TU Delft TEST 4.13 / 295 K) and the NIST TEST splits of I8-1 (4 K, 38 devices) and I9-1 (room temperature, 58 devices).
- TEST contrasts: GMR(selected, C0_refit) per TEST dataset and pooled, GMR(selected, M_bias) pooled; criteria C1 and C2 with margin 1.03 at 95%; the selected class per dataset reported against the TEST per-dataset best class (post hoc).

## Compute and queue

Plan (`plan.json`, 6 workers, 85% efficiency; cost basis I9-2 `cost_basis.json`, refit seconds from the stored M_bias held-out refits):

| dataset | outer folds | inner folds | inner trainings | training CPU-h | inner refits | refit CPU-h |
|---|---|---|---|---|---|---|
| sky130_77K | 12 | 11 (LOGO) | 528 | 0.71 | 528 | 1.99 |
| nist_4K | 53 | 4 | 848 | 1.58 | 2756 | 6.49 |
| tudelft_4K | 7 | 6 (LOGO) | 168 | 1.33 | 1080 | 3.03 |
| ihp_meas_300K | 18 | 7-9 (LOGO) | 584 | 0.11 | 584 | 1.64 |

Total 3.73 CPU-h training, 13.15 CPU-h refits, about 199 min wall. Queue (one heavy job at a time): I10_2_inner (about 45 min), I10_2_refit_{sky130_77K, ihp_meas_300K, tudelft_4K, nist_4K} (about 23 / 19 / 36 / 76 min), I10_2_decide (seconds). The stored mode (S_gate, S_cv only, seconds) may run directly after this file is committed.

## Smoke

Commit d16c929: sky130 77 K, 2 outer folds, inner training 50 steps (seed 0), refits of inner fold 0 with max_nfev 10, irls 1 (18 s, 2 workers); V_fold difference 0; decide path run on synthetic outer rows (random values, no stored held-out row read). Code checks only.
