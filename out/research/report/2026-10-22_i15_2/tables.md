# Tables, 2026-10-22 i15_2

Generated 2026-10-22 07:23 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i15_2.py --date 2026-10-22`.

### I15-2 primary result and lambda selection

| quantity | value | devices lower | outcome |
|---|---:|---:|---|
| primary: GMR(selected, M_bias v1), 18 devices | 0.997 [0.962, 1.027] | 7/18 | selected variant non-inferior to M_bias v1 |
| GMR(selected, C0_refit) | 0.743 [0.654, 0.855] | 17/18 |  |
| GMR(selected, g v2 = sym_e1_l10) | 0.947 [0.885, 1.023] | 13/18 |  |
| lambda* over the 18 outer folds (V) | 0: 1, 1: 10, 2: 6, 5: 0, 10: 1 |  |  |
| lambda* of the full-data model (reported, not scored) | 1 V |  |  |
| changes named as costing accuracy (pre-registered label rule) | none |  |  |

- Selection (pre-registered): per outer fold, 3 inner folds of the 17 training devices, one member (seed 0) per inner fold and lambda in {0, 1, 2, 5, 10} V (symmetric inputs, EPS 0.01 V), inner held-out card refits; lambda* = argmin mean ln RRMS, ties to the larger lambda. Selected row of device h = 5-seed LODO row of sym_e1_l{lambda*_h}.
- Outcome rule: CI upper of GMR(selected, v1) <= 1.05 -> non-inferior; GMR device bootstrap 10,000, 95% CI. CI upper < 1: False. Validity: valid.


### I15-2 variants: held-out RRMS and paired GMRs

| variant | all-device | combined | median | GMR vs v1 | GMR vs C0_refit | failed |
|---|---:|---:|---:|---:|---:|---:|
| M_bias v1 (abs inputs, lambda 0) | 0.07348 | 0.06953 | 0.06370 | - | 0.745 [0.660, 0.850] (17/18) | 0 |
| sym, EPS 0, lambda 0 | 0.07397 | 0.06999 | 0.06362 | 1.006 [0.997, 1.015] (3/18) | 0.750 [0.663, 0.859] (17/18) | 0 |
| sym, EPS 0.01, lambda 0 | 0.07359 | 0.06964 | 0.06243 | 0.999 [0.985, 1.013] (7/18) | 0.745 [0.657, 0.855] (17/18) | 0 |
| sym, EPS 0, lambda 10 | 0.07753 | 0.07354 | 0.05568 | 1.055 [0.970, 1.135] (3/18) | 0.786 [0.696, 0.900] (16/18) | 0 |
| sym, EPS 0.01, lambda 1 | 0.07313 | 0.06924 | 0.05535 | 0.985 [0.948, 1.019] (8/18) | 0.734 [0.649, 0.841] (17/18) | 0 |
| sym, EPS 0.01, lambda 2 | 0.07259 | 0.06875 | 0.05486 | 0.973 [0.931, 1.013] (9/18) | 0.725 [0.640, 0.831] (17/18) | 0 |
| sym, EPS 0.01, lambda 5 | 0.07337 | 0.06956 | 0.05493 | 0.984 [0.926, 1.041] (8/18) | 0.733 [0.646, 0.845] (17/18) | 0 |
| g v2: sym, EPS 0.01, lambda 10 | 0.07716 | 0.07323 | 0.05437 | 1.053 [0.966, 1.137] (4/18) | 0.785 [0.693, 0.896] (16/18) | 0 |
| selected (nested lambda*) | 0.07391 | 0.06997 | 0.05504 | 0.997 [0.962, 1.027] (7/18) | 0.743 [0.654, 0.855] (17/18) | 0 |
| C0_refit (no correction) | 0.10509 | 0.09839 | 0.08198 | 1.341 [1.176, 1.516] (1/18) | - | 0 |

- 18-fold LODO, 5 seeds, held-out card refit with the fold g in the loop (state C); strict paper RRMS, curve inclusion frozen to the published card. all-device: mean over 18 devices; combined: mean of the nMOS and pMOS family means. v1 and g v2: stored I13-1 rows; other variants: I15-2 runs. GMR: device bootstrap, 95% CI (devices lower).


### I15-2 ablation contrasts (paired GMR, 18 devices)

| change | contrast | GMR [95% CI] | lower | label |
|---|---|---:|---:|---|
| inputs_abs_to_sym | sym_e0_l0 vs v1 | 1.006 [0.997, 1.015] | 3/18 | no cost beyond the margin |
| eps_0_to_0.01 | sym_e1_l0 vs sym_e0_l0 | 0.994 [0.982, 1.003] | 10/18 | no cost beyond the margin |
| eps_0_to_0.01 | sym_e1_l10 vs sym_e0_l10 | 0.998 [0.978, 1.018] | 7/18 | no cost beyond the margin |
| penalty_0_to_10 | sym_e0_l10 vs sym_e0_l0 | 1.049 [0.963, 1.130] | 3/18 | inconclusive |
| penalty_0_to_10 | sym_e1_l10 vs sym_e1_l0 | 1.054 [0.968, 1.136] | 5/18 | inconclusive |
| lambda curve | sym_e1_l1 vs sym_e1_l0 | 0.986 [0.952, 1.019] | 7/18 | no cost beyond the margin |
| lambda curve | sym_e1_l2 vs sym_e1_l0 | 0.973 [0.934, 1.010] | 9/18 | no cost beyond the margin |
| lambda curve | sym_e1_l5 vs sym_e1_l0 | 0.985 [0.930, 1.037] | 9/18 | no cost beyond the margin |
| lambda curve | sym_e1_l10 vs sym_e1_l0 | 1.054 [0.968, 1.136] | 5/18 | inconclusive |
| lambda curve | sym_e1_l0 vs v1 | 0.999 [0.985, 1.013] | 7/18 | no cost beyond the margin |
| lambda curve | sym_e1_l1 vs v1 | 0.985 [0.948, 1.019] | 8/18 | no cost beyond the margin |
| lambda curve | sym_e1_l2 vs v1 | 0.973 [0.931, 1.013] | 9/18 | no cost beyond the margin |
| lambda curve | sym_e1_l5 vs v1 | 0.984 [0.926, 1.041] | 8/18 | no cost beyond the margin |
| lambda curve | sym_e1_l10 vs v1 | 1.053 [0.966, 1.137] | 4/18 | inconclusive |

- Labels (pre-registered): cost if CI lower > 1; no cost beyond the margin if CI upper <= 1.05; otherwise inconclusive. A change is named as costing accuracy when one of its contrasts is labelled cost. wins: devices where the first arm is lower.


### I15-2 negative corrected gm / gds on the penalty grids (%)

| variant | gm train | gds train | gm held-out | gds held-out | gm full | gds full |
|---|---:|---:|---:|---:|---:|---:|
| M_bias v1 (abs inputs, lambda 0) | 0.192 | 8.093 | 0.190 | 11.487 | 0.200 | 8.130 |
| sym, EPS 0, lambda 0 | 0.191 | 8.015 | 0.190 | 11.434 | 0.200 | 8.094 |
| sym, EPS 0.01, lambda 0 | 0.190 | 8.015 | 0.190 | 11.377 | 0.197 | 8.142 |
| sym, EPS 0, lambda 10 | 0.059 | 0.265 | 0.107 | 1.499 | 0.062 | 0.237 |
| sym, EPS 0.01, lambda 1 | 0.137 | 1.875 | 0.138 | 4.357 | 0.138 | 1.588 |
| sym, EPS 0.01, lambda 2 | 0.114 | 1.199 | 0.128 | 3.477 | 0.110 | 0.938 |
| sym, EPS 0.01, lambda 5 | 0.086 | 0.529 | 0.116 | 1.402 | 0.087 | 0.362 |
| g v2: sym, EPS 0.01, lambda 10 | 0.061 | 0.270 | 0.109 | 1.290 | 0.063 | 0.229 |
| selected (nested lambda*) | 0.129 | 1.887 | 0.138 | 4.730 | 0.138 | 1.588 |
| card only (starting cards) | 0.128 | 0.000 |  |  |  |  |

- Percent of penalty-grid points (25 mV forward quadrant of each starting card, |I| > 1 nA; 72,475 points over 18 devices) with negative corrected gm / gds = (gm_card / I_card + dg/du_g), (gds_card / I_card + dg/du_d); dg/du by central differences. train: each fold model on its 17 training grids, mean over folds; held-out: each fold model on the held-out device's grid, mean over folds; full: the full-data model on all 18 grids (selected: full-data model of lambda* (full)).


### I15-2 validity

| check | criterion | value | result |
|---|---:|---:|---:|
| V1 trainer, seed 0 of fold pmos_L2_W5, member 0 vs stored: g v2 recipe / v1 recipe | = 0 | 0 / 0 | pass |
| V2 held-out refit of the stored g v2 fold corrections (nmos_L1_W1p6, pmos_L2_W5) | <= 1e-9 | 0 | pass |
| V3 18 rows per new variant / inner CV complete | yes / yes | True / True | pass |

- Inner selection: 285 single-member trainings (3183 s) and 1620 inner held-out refits (5807 s) on 6 workers.

