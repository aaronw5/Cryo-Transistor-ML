# I19-1 pre-registration (IN FORCE): dataset-level gate, one decision per dataset from S3_D computed on all DEV devices of the dataset; Part R (retrospective, 9 measured DEV datasets), Part P (prospective, TEST gate decisions committed before any TEST evaluation), Part C (calibration on the 20 I7-1 simulated datasets)

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-23), committed alone, before any I19-1 code and before any S3_D value is computed. TEST stays locked
for NIST 4 K (I8-1, 38 devices), NIST 4 K room-temperature card (I9-1, 58), IHP SG13G2 (10 DUTs x 4 temperatures)
and TU Delft 40 nm (24 devices x 2 temperatures). sky130 77 K has no locked TEST since 2026-10-21 (user decision).

## Reason

I7-1: S3 tracks the held-out correction gain across datasets (new datasets Spearman +0.985) but the fold-level
relation reverses within a dataset (within-dataset Spearman negative in 20/20 simulated and 9/11 stored datasets).
I18-1 gated per outer fold; on TU Delft the per-fold gate kept C0_refit on the groups with the largest held-out
gains and applied the correction to the group where it raised the error (within-dataset Spearman(S3_f, y_f) -0.21 /
-0.68). I19-1 makes one gate decision per dataset, the level at which S3 tracks the gain.

## Method under test

For a dataset D with DEV records R_D:

- S3_D = mean over the records r in R_D of ln( RRMS(r, I_start x exp(g_{-k(r)})) / RRMS(r, I_start) );
- k(r): the inner group of r (device for sky130, NIST, IHP and the simulated datasets; geometry group for TU Delft;
  the I7-1 / I18-1 groups);
- g_{-k(r)}: the correction trained with the gated arm's recipe at the I7-1 reduced budget (1 seed = seed 0, 500
  steps) on the DEV records of the training set of the leave-one-group-out fold that holds out k(r), in record load
  order, applied post hoc (no card refit). The training set of that fold is: all DEV records outside k(r)
  (sky130, NIST, TU Delft, simulated); the DEV records of the same class outside k(r) (IHP; the corrections are
  trained per class, I5-1 / I17-2 / I8-2 TEST protocol);
- I_start: the stored starting-card simulation of r; RRMS: strict paper RRMS on r's included curves
  (`e1_correction.score_device`, as `i7_1_transfer.s3_fold_value`);
- non-finite log ratios are dropped from the mean (`np.nanmean`).

S3_D is the I7-1 S3 statistic computed once with the training set = all DEV devices of the dataset (inner
leave-one-group-out over all DEV records). It uses starting-card records only: no held-out refit, no RRMS of any
arm row enters it.

**Decision: apply the correction to every device of the dataset iff S3_D < 0 (strict); otherwise every device gets
C0_refit.** One decision per dataset; no per-fold or per-device choice.

### Threshold (fixed now, before any S3_D value)

Threshold 0. Reasons: (1) S3_D < 0 has a direct meaning (the reduced correction lowered the starting-card RRMS of the
inner held-out DEV records on average); (2) it is the threshold of the I8-2 gate G and the I18-1 per-fold gate, so the
comparison with the per-fold gate changes only the aggregation level; (3) the alternative, the zero crossing of the
I7-1 OLS fit of y on S3 over the 20 simulated datasets (y = -0.0298 + 1.214 S3, crossing S3 = +0.02455), was fitted
to a different statistic (held-out-weighted mean of fold S3_f, I7-1 definition) with the v1 recipe only; its
intercept (-0.030) is within the scatter of the 20 points. Decisions at S3_D < +0.02455 are reported as a
sensitivity column, without a rule.

### Recipe per dataset (the gated arm of I18-1)

| dataset | gated arm (rows, as I18-1) | S3_D recipe | records (load order) | lambda of the inner fit |
|---|---|---|---|---|
| sky130 77 K all-18 | selected g v2 (I15-2 `selected_rows`) | g v2: `i12_1_symm.fit_v2`, sym_v2, EPS 0.01 V, vdd 1.85 V, penalty grids of the inner-training devices (I12-1 card grids) | `I13_1/records` (state C) | lambda*_f of the outer fold that holds out k(r) (`I15_2/decision.json` lambda_star) |
| NIST 4 K lvt (I8-1) | M_bias v1 | v1: `i3_2_bias.fit_bias` M_bias, S3 hyper (seed 0, 500 steps) | `I8_1/start/records`, finite records only (I8-1 amendment A1; 53) | - |
| NIST 4 K, room-temperature card (I9-1) | M_bias v1 | v1 | `I9_1/start/records`, finite only (87) | - |
| IHP SG13G2 300 K | selected g v2 (I17-2 `selected_rows`) | g v2, vdd = class supply, grids `I17_2/pens` | `I5_1/start/records_300K` | lambda*_f of the outer fold that holds out k(r) (`I17_2/inner/inner.json`) |
| IHP SG13G2 233 / 343 / 398 K | M_bias v1 | v1 | `I5_1/start/records_{T}K` | - |
| TU Delft 4.13 K / 295.0 K | selected g v2 (I18-2, state C) | g v2, vdd 1.1 V, I18-2 grids | I18-2 state-C starting records | lambda*_f of the outer fold that holds out k(r) (I18-2 `lambda_star`) |

For a g v2 dataset the inner fit that leaves out k uses the same training set as outer fold k and that fold's
training-only lambda selection lambda*_k; no full-data lambda is needed (IHP 300 K has none stored).

Secondary recipe (reported, no rule): v1 S3_D on the datasets whose primary is g v2: sky130 (state-C records
`I13_1/records`; v1 rows `I13_1/all18`), IHP 300 K (`I5_1/start/records_300K`; I5-1 rows), TU Delft 4.13 / 295.0 K
(I6-1 state-L records `I6_1/start/records_{T}K`; I6-1 rows). These are the arm and records of the I8-2 TEST protocol.

## Part R (retrospective; DEV outcomes already seen)

The 9 primary datasets of I18-1 with the I18-1 rows (sky130 18, NIST 4 K 53, NIST room-temperature card 87, IHP 18
at each of 4 temperatures, TU Delft 45 at each of 2 temperatures). Gated_D rows: the stored correction-arm row of
every device if S3_D < 0, else the stored C0_refit row. GMR(gated_D, C0_refit) with `i3_2_bias.paired_ratio`
(10,000 resamples, seed 0, percentile 95% CI; device bootstrap; 7-group cluster bootstrap for TU Delft).

Label: retrospective. The DEV outcomes (correction-always and per-fold gated GMR per dataset) are known, and the I8-2
gate G values (held-out-weighted mean of the fold S3_f) are known for every dataset (I18-1 `dataset_gate`: sky130
-0.175, NIST -0.108 / -0.071, IHP +0.353 / +0.254 / +0.375 / +0.395, TU Delft +0.018 / +0.201). S3_D shares inner
training sets with those values except for one group per fold, so the Part R decisions are close to predictable.
Leakage note: on DEV every scored device's own starting-card record enters S3_D (as an inner-scored record and as a
training record of the other inner fits); its held-out refit rows do not. On TEST (Part P) the scored devices are not
in R_D.

| condition | definition |
|---|---|
| R1 never worse | GMR(gated_D, C0_refit) point value <= 1 on each of the 9 datasets |
| R2 pooled below correction-always | exp(mean over the 9 datasets of ln GMR(gated_D, C0_refit)) < exp(mean of ln GMR(correction always, C0_refit)) (I18-1: 0.958) |

Part R outcome: "supported (retrospective)" iff R1 and R2; otherwise "not supported (retrospective)" with the failed
condition(s) named.

Reported without a rule: per dataset S3_D, decision, GMR(gated_D, C0_refit) with CI, GMR of the I18-1 per-fold gate
(pooled 0.929), GMR(correction always, C0_refit), the I8-2 gate G decision, decision vs the realised sign of
ln GMR(correction, C0_refit) (count of 9), the decision at the sensitivity threshold +0.02455, per-class S3_D for IHP,
the v1 secondary S3_D and its gated GMR on the datasets with v1 rows, the pooled values without the TU Delft pair.

## Part P (prospective; TEST locked, not evaluated)

For each locked TEST split with a DEV counterpart, the TEST gate decision is the decision from the dataset's DEV
S3_D (the TEST run trains the correction on all DEV devices of the dataset, IHP per class, so S3_D is computed on the
TEST run's training set):

| TEST split | n TEST | DEV counterpart (n) | primary arm | secondary |
|---|---|---|---|---|
| NIST 4 K lvt TEST (I8-1 split) | 38 | NIST 4 K DEV (57; 53 finite records) | M_bias v1 | - |
| NIST 4 K room-temperature card TEST (I9-1 split) | 58 | DEV (87) | M_bias v1 | - |
| IHP SG13G2 TEST 233 / 343 / 398 K | 10 each | DEV 18 | M_bias v1 | - |
| IHP SG13G2 TEST 300 K | 10 | DEV 18 | selected g v2 | M_bias v1 |
| TU Delft TEST 4.13 / 295.0 K | 24 each | DEV 45 | selected g v2 | M_bias v1 (I8-2 TEST protocol) |

File `test_gate_decisions.json` / `.md`, written by a dedicated mode and committed in its own commit before any TEST
evaluation and before the Part R / Part C decide step. It lists S3_D, the decision at threshold 0, the sensitivity
decision, the I8-2 committed TEST sign predictions (S3 sign and frozen S1_lodo rule; IHP and TU Delft), and the
disagreements between the I19-1 decision and each I8-2 prediction (primary and v1 secondary). The NIST datasets have
no I8-2 TEST prediction; their committed DEV P_c sign (I8-1 `prediction_v2.json`, I9-1 `prediction.json`) is listed
for reference. TEST data are not opened: the mode loads DEV records only, and every loaded record tag is checked
to be in the DEV list and not in the TEST list of the dataset's split file.

Evaluation at TEST (December, separate run, not part of I19-1): per split, gated_D TEST rows = correction-arm TEST
rows if the committed decision is "apply", else C0_refit TEST rows; the decision is correct iff it equals the sign
of the TEST ln GMR(correction, C0_refit). No change to these decisions after the commit; a change gets a dated
amendment.

## Part C (calibration on the 20 simulated I7-1 datasets; threshold fixed)

The 20 I7-1 step-3 datasets (2 PDKs x 2 replicates x 5 mixing levels x 16 devices; truth and records under
`out/research/pdk2/...` and `I7_1/new/`), v1 recipe, inner group = geometry (I7-1 `group_fn("pdk2")`). Realised
y_D = ln GMR(M_bias, C0_refit) stored in `I7_1/decision.json` (18 of 20 negative). Decision correct iff
(S3_D < 0) = (y_D < 0).

| condition | definition |
|---|---|
| C sign accuracy | correct on at least 19 of 20 datasets |

x = 19: the constant decision "apply on every dataset" is correct on 18/20; 19 requires the gate to exceed it.
The I7-1 fold-mean S3 had 19/20 (this is known). Reported: GMR(gated_D, C0_refit) per dataset (1 or exp(y_D)),
pooled exp(mean ln GMR) for gated_D, M_bias always and the post-hoc dataset oracle; Spearman(S3_D, y_D); OLS of y_D
on S3_D; S3_D vs the I7-1 fold-mean S3. No new simulated datasets are generated in I19-1.

## Overall outcome

| outcome | condition |
|---|---|
| supported | valid, Part R supported (retrospective) and Part C holds |
| partially supported | valid, exactly one of Part R and Part C holds |
| not supported | valid, neither holds |
| invalid | any validity check fails |

Part P has no outcome until the TEST run.

## Validity (checked first)

| check | pass |
|---|---|
| V1 rows | Part R row counts 18 / 53 / 87 / 18 x 4 / 45 x 2 and the correction / C0_refit rows equal the I18-1 rows (difference 0) |
| V2 S3 code | the I19-1 S3 code, given an I18-1 / I7-1 fold training set, reproduces stored fold values: IHP 300 K v1 folds hv_nmos_W0p3_L0p5 and lv_nmos_W0p15_L0p18 (I7-1 `retro/s3.json`), sky130 g v2 fold nmos_L0p15_W1p6 (I18-1 `s3/sky130_v2.json`); \|difference\| <= 1e-9 |
| V3 complete | S3_D finite for every primary dataset, every v1 secondary dataset and every simulated dataset; every inner fit ran (one per inner group) |
| V4 unit tests | `tests.test_i19_1_dataset_gate` passes |
| V5 TEST closed | every record loaded by the S3_D and Part P code is in the DEV list and not in the TEST list of its split (NIST, IHP, TU Delft) |

## Order and compute

This prereg (alone) -> code + unit tests (commit) -> smoke (2 inner fits per part, 50 steps; 1 simulated dataset;
not used for decisions) -> queue, one heavy job at a time: I19_1_s3_real (S3_D on 9 primary + 4 v1 secondary
datasets + V2 checks; about 420 inner fits), I19_1_testgate (Part P file; committed alone), I19_1_s3_sim (320 inner
fits), I19_1_decide (Part R, Part C, validity), I19_1_figs (report batch). Estimated 25-40 min of queue time.

## Outputs

`out/research/explore/I19_1/{s3/, test_gate_decisions.json, test_gate_decisions.md, decision.json}`; report batch
`out/research/report/<date>_i19_1/` (tables, forest plot of GMR(., C0_refit) per dataset for gated_D, per-fold gate
and correction always, scatter of S3_D vs realised y_D with the threshold for the 9 measured and 20 simulated
datasets); site claim `i19-1`; RESEARCH_LOOP entry.
