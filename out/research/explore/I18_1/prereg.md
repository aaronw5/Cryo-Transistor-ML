# I18-1 pre-registration (IN FORCE): gated correction, "selected g v2 applied only if the training-only S3 gate predicts a gain, else C0_refit", on every measured DEV dataset with stored rows; retrospective (stored outcomes) and prospective (TU Delft g v2 rows from I18-2)

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-22), committed alone, before any I18-1 code, before any S3 value with the g v2 recipe and before any
I18-2 run. TEST stays locked for IHP (10 DUTs) and TU Delft (24 devices); sky130 77 K has no locked TEST since
2026-10-21 (user decision).

## Method under test

For a dataset D with leave-one-out outer folds f (held-out device or geometry group), the gated method assigns to
every held-out device of fold f:

- the stored row of the correction arm (held-out refit with g in the loop) if S3_f < 0;
- the stored C0_refit row otherwise (S3_f >= 0 or S3_f non-finite).

One decision per outer fold, from the fold's training devices only. No per-device choice, no held-out information.

## Gate statistic S3_f (I7-1 definition; sign convention)

S3_f = mean over the training records r of fold f of ln( RRMS(r, I_start x exp(g_{-k(r)})) / RRMS(r, I_start) ),

- I_start: the stored starting-card simulation of r (strict paper RRMS on r's included curves, `score_device`);
- k(r): the inner group of r (device for sky130, NIST, IHP; geometry group for TU Delft, as I7-1);
- g_{-k(r)}: the correction trained with the gated arm's recipe at the I7-1 reduced budget (1 seed = seed 0, 500
  steps) on the fold's training records outside k(r), in record load order, applied post hoc (no card refit);
- non-finite log ratios are dropped from the mean (`np.nanmean`, I7-1 `run_s3`).

Sign convention: S3_f < 0 means the reduced correction lowered the RRMS of the starting card on the inner held-out
training records (predicted gain). **Threshold: apply the correction iff S3_f < 0 (strict).** The threshold 0 is the
sign used by the I8-2 gate G and the I10-2 selector S_gate; it is not fitted.

S3 uses the same recipe as the arm it gates:

| gated arm | S3 recipe | status |
|---|---|---|
| selected g v2 (sky130, IHP 300 K, TU Delft) | `i12_1_symm.fit_v2`, inputs sym_v2, EPS 0.01 V, vdd = dataset supply (sky130 1.85 V; IHP class supply 1.2 / 3.3 V; TU Delft 1.1 V), lambda = lambda*_f of the outer fold (the nested-selection output of that fold, training devices only), penalty grids of the inner-training devices (sky130: the I12-1 card grids used by I15-2; IHP: `I17_2/pens`; TU Delft: I18-2 grids), HYPER_V2 with seeds (0,), steps 500 | **new (S3_v2), recomputed** |
| M_bias v1 (NIST 4 K, NIST 4 K room-temperature card, IHP 233 / 343 / 398 K) | I7-1 recipe: `i3_2_bias.fit_bias`, S3_HYPER (seed 0, 500 steps) | stored S3_f (I8-1 `prediction_v2.json`, I9-1 `prediction.json`, I7-1 `retro/s3.json`), not recomputed |

## Datasets (primary: one arm per dataset)

| dataset | devices / folds | bootstrap unit | gated arm (rows) | C0_refit rows | S3 |
|---|---|---|---|---|---|
| sky130 77 K, all 18 | 18 / 18 (LODO) | device | selected g v2: `I15_2/decision.json` selected_rows (state C) | `I13_1/lodo/results.json` per_device C0_refit (as I15-2) | S3_v2 new; records `I13_1/records`; lambda*_f `I15_2/decision.json` lambda_star |
| NIST sky130 4 K, lvt (I8-1) | 53 paired / 53 | device | M_bias v1: `I8_1/lodo/results.json` | same file | stored `I8_1/prediction_v2.json` per_fold S3 |
| NIST sky130 4 K, nfet_01v8 / pfet_01v8, room-temperature card (I9-1) | 87 / 87 | device | M_bias v1: `I9_1/lodo/results.json` | same file | stored `I9_1/prediction.json` per_fold S3 |
| IHP SG13G2 300 K (I17-2) | 18 / 18 (LODO within class) | device | selected g v2: `I17_2/decision.json` selected_rows | `I17_2/base/results.json` C0_refit (= I5-1 rows) | S3_v2 new; records `I5_1/start/records_300K`; lambda*_f `I17_2/inner/inner.json` |
| IHP SG13G2 233 K, 343 K, 398 K (I5-1) | 18 / 18 each | device | M_bias v1: `I5_1/lodo_{T}K/results.json` | same file | stored `I7_1/retro/s3.json` |
| TU Delft 40 nm 4.13 K (I18-2) | 45 / 7 groups | cluster (7 geometry groups) | selected g v2 from I18-2 (state C) | I18-2 C0_refit (state C) | S3_v2 new on the I18-2 state-C starting records; lambda*_f from the I18-2 inner selection |
| TU Delft 40 nm 295.0 K (I18-2) | 45 / 7 groups | cluster (7) | same | same | same |

Nine primary datasets. The g v2 rows exist for sky130 and IHP 300 K (I15-2, I17-2) and are produced for TU Delft by
I18-2; the other datasets have stored rows for M_bias v1 only, which is the stored correction arm used there.
Fallback (stated now): if I18-2 is invalid or its selected rows are missing, the two TU Delft datasets enter with
the I6-1 stored rows (M_bias v1 and C0_refit, BSIM4 clamp state L) and the stored I7-1 S3_f, labelled "fallback".

## Rules

Per dataset: GMR(gated, C0_refit) = exp(mean over devices of ln(RRMS_gated / RRMS_C0_refit)) with
`i3_2_bias.paired_ratio` (10,000 resamples, seed 0, percentile 95% CI; unit as in the table; devices with a
non-finite row in either arm are left out and counted). A fold assigned C0_refit contributes ln 1 = 0.

| condition | definition |
|---|---|
| C1 never worse | CI lower bound of GMR(gated, C0_refit) <= 1 on every primary dataset |
| C2 sky130 gain | CI upper bound of GMR(gated, C0_refit) < 1 on sky130 77 K |

| outcome | condition |
|---|---|
| supported | valid, C1 and C2 |
| not supported (worse than C0_refit on <datasets>) | valid, C1 fails |
| not supported (no gain on sky130) | valid, C1 holds, C2 fails |
| invalid | any validity check fails |

## Validity (checked first)

| check | pass |
|---|---|
| V1 rows | finite paired rows per dataset: sky130 18, NIST 4 K 53, NIST room-temperature card 87, IHP 18 at each temperature, TU Delft 45 at each temperature; the g v2 rows equal the stored `selected_rows` of I15-2 / I17-2 / I18-2 (difference 0) |
| V2 S3 code | the I18-1 S3 code with the v1 recipe reproduces the stored I7-1 S3_f of IHP 300 K folds hv_nmos_W0p3_L0p5 and lv_nmos_W0p15_L0p18 (\|difference\| <= 1e-9) |
| V3 S3_v2 complete | every outer fold of sky130, IHP 300 K and TU Delft (both temperatures) has a finite S3_v2_f; every inner fit ran |
| V4 unit tests | `tests.test_i18_1_gate` passes |

## Reported without a rule

- Per dataset: decisions (folds and devices with the correction), GMR(gated, C0_refit) under the other bootstrap
  unit (device for TU Delft; geometry cluster for IHP), GMR(gated, correction always), GMR(correction, C0_refit).
- v1 gate (secondary): arm M_bias v1 with the v1-recipe S3 on every dataset with v1 rows: sky130 all-18 (state C
  rows `I13_1/all18` M_bias, S3_v1 new on `I13_1/records`, 18 folds), NIST 4 K, NIST room-temperature card, IHP 233 /
  300 / 343 / 398 K (stored), TU Delft 4.13 / 295.0 K (I6-1 stored rows, state L, stored S3).
- Dataset-level gate (I8-2 G: sign of the held-out-weighted mean of S3_f; reuses each device's own record through
  the other folds) per dataset, as a reference.
- S3_v2_f vs S3_v1_f per fold (sky130, IHP 300 K); within-dataset Spearman(S3_f, y_f), y_f = mean held-out ln
  ratio of the fold; equal-weight pooled exp(mean ln GMR) over the primary datasets; per-dataset oracle (post hoc).

## Predictions (stated before any S3_v2 value; not rules)

| dataset | predicted decisions | predicted GMR(gated, C0_refit) | basis |
|---|---|---|---|
| sky130 77 K | correction on 18 / 18 folds | equal to GMR(selected, C0_refit) 0.743 [0.654, 0.855]; C2 holds | stored S3_v1 on the 12 DEV folds -0.144 .. -0.075 |
| NIST 4 K (I8-1) | 53 / 53 (determined from stored S3_f, range -0.134 .. -0.100) | 0.806 [0.758, 0.864] (known) | stored |
| NIST room-temperature card (I9-1) | 87 / 87 (determined, range -0.088 .. -0.063) | 0.797 [0.748, 0.859] (known) | stored |
| IHP 300 K | 0 / 18 | 1.000 [1.000, 1.000] | stored S3_v1 0.187 .. 0.641 |
| IHP 233 / 343 / 398 K | 0 / 18 each (determined; S3_f 0.033 .. 0.753) | 1.000 | stored |
| TU Delft 4.13 K | 5 / 7 groups (the groups with stored S3_v1 < 0) | ln GMR < 0; CI lower <= 1 | I7-1 S3_v1 per group |
| TU Delft 295.0 K | 3 / 7 groups | ln GMR < 0; CI lower <= 1 | I7-1 S3_v1 per group |
| pooled | | outcome "supported" | |

## Known before registration

The v1-arm datasets are retrospective: their S3_f values, decisions and rows were all stored (I8-2 fold-level gate
diagnostic, I10-2 S_gate: sky130 DEV-12 0.769, NIST 4 K 0.806, TU Delft 4.13 K 1.027, IHP 300 K 1.000). The sky130
and IHP 300 K g v2 rows are stored (outcomes known); their S3_v2 values are not computed. The TU Delft g v2 rows do not
exist (prospective part).

## Order and compute

This prereg (alone) -> I18-1 code + unit tests -> smoke (2 folds each, 50 steps; not used for decisions) -> I18-2
pre-registration (alone) -> I18-2 code -> queue, one heavy job at a time: I18_1_s3 (sky130 S3_v2 306 inner fits,
sky130 S3_v1 306, IHP 300 K S3_v2 146; about 10 min), I18-2 jobs, I18_1_s3_tud (TU Delft S3_v2, 2 x 7 x 6 inner
fits), I18_1_decide.

## Outputs

`out/research/explore/I18_1/{s3/, decision.json}`; report batch `out/research/report/<date>_i18_1/` (tables vs
rules, per-device and per-fold plots, I-V figures with measured points, published / stock-card NGSpice and
re-simulated C0_refit and gated-arm cards); site claim `i18-1`; RESEARCH_LOOP entry.
