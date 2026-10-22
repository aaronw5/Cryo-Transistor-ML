# I15-2 pre-registration (IN FORCE): which g v2 change costs accuracy; ablation of the three training changes on the 18-fold LODO and a training-data-only selection of the penalty weight; sky130 77 K, 18 devices, state C

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-22), committed alone, before any I15-2 code, training or held-out refit.

## Background

g v2 (I12-1) changed three things in the training of the bias-only correction relative to M_bias v1 (I3-2 recipe):
(1) d/s-symmetric inputs x1 = (s V(g,s) + s V(g,d) + m) / (2 VDD), x2 = m / VDD, (2) the smooth absolute value
m = sqrt(V(d,s)^2 + EPS^2) with EPS = 0.01 V (v1: |V(g,s)| / VDD, |V(d,s)| / VDD), (3) the penalty lambda x P on
negative corrected gm / gds on the 25 mV forward-quadrant grid of each training card, lambda = 10 V. In the I13-1
state-C rows (I12-1 LODO recipe, 18 folds, 5 seeds, held-out card refit with the fold g in the loop), g v2 is
0.0772 / 0.0732 (all-device / combined) against M_bias v1 0.0735 / 0.0695; GMR(v2, v1) 1.053 [0.966, 1.137], 4/18
lower. On the training data (forward operation only: s V(d,s) >= 0 at every measured point), change (1) with EPS = 0
gives the v1 inputs up to floating-point rounding: x1 = s V(g,s) / VDD = |V(g,s)| / VDD and x2 = |V(d,s)| / VDD.

Done before this registration: none of the I15-2 runs. The stored rows and corrections are the I13-1 ones. The I15-1
pre-registration and runs are a separate experiment (deployment checks; no training).

## Inputs (fixed)

Records `I13_1/records` (state C); stored rows M_bias v1 `I13_1/all18/results.json` (`per_device.M_bias`), g v2
`I13_1/lodo/results.json` (`per_device.M_bias_v2`), C0_refit `I13_1/lodo/results.json` (`per_device.C0_refit`);
stored corrections `I13_1/all18/corrections.json`, `I13_1/lodo/corrections.json`; penalty grids
`I12_1/grids/card_grids.npz` (starting cards; one dc analysis per process, identical in clamp states L and C).

## Variants (all with the I12-1 / I13-1 recipe: 3-16-16-1 tanh MLP, 5 seeds 0-4, zero-init last layer, Adam lr 3e-3, 2000 full-batch steps, L2 1e-4 on weight matrices, float64, E1 loss; training sets in record load order; held-out refit TRF max_nfev 400, irls 2, diff_step 0.01, state C, strict paper RRMS with frozen curve inclusion)

| name | inputs | EPS (V) | lambda (V) | source |
|---|---|---|---|---|
| v1 | abs (is_pmos, \|V(g,s)\| / VDD, \|V(d,s)\| / VDD) | - | 0 | stored, I13_1/all18 `M_bias` |
| sym_e0_l0 | symmetric | 0 | 0 | new LODO |
| sym_e1_l0 | symmetric | 0.01 | 0 | new LODO |
| sym_e0_l10 | symmetric | 0 | 10 | new LODO |
| sym_e1_l10 (= g v2) | symmetric | 0.01 | 10 | stored, I13_1/lodo `M_bias_v2` |
| sym_e1_l1, sym_e1_l2, sym_e1_l5 | symmetric | 0.01 | 1, 2, 5 | new LODO |

Each new LODO: 18 fold corrections (17 training devices) and the full-data correction (18 devices), the 18 held-out
refits. EPS = 0 variants are ablation arms only (x2 = |V(d,s)| / VDD has no derivative at V(d,s) = 0 and is not
exported to Verilog-A).

## Ablation contrasts (descriptive labels, pre-registered)

Paired GMR with the device bootstrap (`i3_2_bias.paired_ratio`, 10000 resamples, seed 0), 18 devices:

| change | contrasts |
|---|---|
| (1) input form, abs -> symmetric (EPS 0, lambda 0) | GMR(sym_e0_l0, v1) |
| (2) EPS 0 -> 0.01 | GMR(sym_e1_l0, sym_e0_l0) at lambda 0; GMR(sym_e1_l10, sym_e0_l10) at lambda 10 |
| (3) penalty lambda 0 -> 10 | GMR(sym_e0_l10, sym_e0_l0) at EPS 0; GMR(sym_e1_l10, sym_e1_l0) at EPS 0.01 |
| lambda curve | GMR(sym_e1_lk, sym_e1_l0) and GMR(sym_e1_lk, v1), k in {1, 2, 5, 10} |

Label per contrast: **cost** if the CI lower bound > 1; **no cost beyond the margin** if the CI upper bound <= 1.05;
**inconclusive** otherwise. A change is named as costing accuracy when at least one of its contrasts is labelled cost.
Reported with each contrast: point GMR, CI, devices lower, all-device / combined / median of both arms.

## Selection rule (training data only; pre-registered)

Grid lambda in {0, 1, 2, 5, 10} V, symmetric inputs, EPS = 0.01 V. For each outer fold (held-out device h; 17
training devices) and for the full-data model (18 devices):

1. Inner folds: the training devices sorted by (polarity, tag), nMOS first; the i-th device in that order goes to
   inner fold i mod 3 (3 inner folds of 5-6 devices; `inner_folds`, unit-tested).
2. For each inner fold j and each lambda: train one member (seed 0, otherwise the recipe above) on the training
   devices outside fold j (record load order), with the penalty grids of those devices only; refit the card of each
   device in fold j with that member in the loop (held-out refit recipe above); score its strict paper RRMS.
3. Inner score S(lambda) = mean over the 17 (18) inner held-out devices of ln RRMS. A non-finite inner RRMS counts as
   ln(1.0).
4. Selected lambda* = argmin S; ties within 1e-12 go to the larger lambda.

The held-out device h enters no step of its own fold's selection (not in any inner training set, inner held-out set,
penalty grid or score). The selected variant's row for device h is the held-out row of the 5-seed LODO variant
sym_e1_l{lambda*_h} for fold h. The full-data lambda* is reported (it would be the deployed choice); it is not scored.

## Primary and secondary results

- **Primary:** GMR(selected, v1) over the 18 devices, device bootstrap 95% CI; non-inferiority margin 1.05.
- Secondary: GMR(selected, C0_refit); GMR(selected, sym_e1_l10); per-variant table (all-device, combined, median,
  GMR vs v1 and vs C0_refit with CIs, devices lower, failed devices); lambda*_h per fold and for the full-data model;
  inner scores S(lambda) per fold.
- **Negative gm / gds fractions** for every variant (including v1, sym_e1_l10 and the selected variant): fraction of
  penalty-grid points (card forward current > 1 nA) with negative corrected gm and gds (dg/du by central differences,
  h = 1e-5 V; `i12_1_symm.penalty_value_numpy`), (a) on each fold's training grids (mean over folds), (b) on the
  held-out device's starting-card grid (mean over folds and pooled over points), (c) the full-data model on all 18
  grids; card-only fractions for reference.
- Aggregates: all-device mean and combined (equal-weight nMOS / pMOS family means) for every row set; RRMS as
  registered (strict paper RRMS, published-card curve inclusion), not redefined.

## Outcome (primary)

| condition | outcome |
|---|---|
| any validity check fails | invalid (stop: fix and re-register) |
| CI upper of GMR(selected, v1) <= 1.05 | selected variant non-inferior to M_bias v1 |
| CI upper > 1.05 | selected variant not shown non-inferior to M_bias v1 |

Reported with the outcome: whether the CI upper is < 1 (selected lower than v1) and whether GMR(selected, C0_refit) has
CI upper < 1.

## Validity (checked first)

| check | pass |
|---|---|
| V1 trainer: seed 0 of fold pmos_L2_W5 reproduces the stored member 0, (a) sym_e1_l10 vs `I13_1/lodo` `M_bias_v2\|pmos_L2_W5`, (b) abs inputs lambda 0 vs `I13_1/all18` `M_bias\|pmos_L2_W5` | max \|weight difference\| = 0 |
| V2 held-out refit of the stored g v2 fold corrections of nmos_L1_W1p6 and pmos_L2_W5 reproduces the stored rows | max \|RRMS difference\| <= 1e-9 |
| V3 rows | 18 held-out rows per new variant; inner CV complete (19 selections x 5 lambdas x 3 inner folds trained; every inner held-out device scored) |

## Order and compute

This prereg commit (alone) -> code + unit tests commit -> smoke (2 outer folds, 200 steps, refit max_nfev 20; not used
for decisions) -> queue, one heavy job at a time, 6 workers: I15_2_validity (about 3 min), I15_2_lodo_sym_e0_l0 and
_sym_e1_l0 (about 6 min each; lambda 0 members train in about 17 s), I15_2_lodo_sym_e0_l10, _sym_e1_l1, _sym_e1_l2,
_sym_e1_l5 (about 28 min each; penalty members about 100 s), I15_2_inner (285 single-member trainings of 11-12
devices and 1620 inner refits; about 2 h), I15_2_decide. Total about 4.3 h. I15-1 jobs run first in the same queue.

## Outputs

`out/research/explore/I15_2/{validity, lodo_<variant>/{results,corrections,heldout_cards}.json, inner/inner.json,
decision.json}`; report batch, site claim `i15-2`, RESEARCH_LOOP entry after the decision.
