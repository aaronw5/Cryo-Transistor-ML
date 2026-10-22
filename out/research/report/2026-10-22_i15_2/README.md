# Report batch 2026-10-22 · i15_2

Generated 2026-10-22 07:23 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i15_2.py --date 2026-10-22`. NGSpice runs: re-simulation of the held-out cards of the 4 I-V devices (checked against the stored rows); everything else from stored files.

## Contents

- I15-2 (pre-registration `out/research/explore/I15_2/prereg.md`, commit 7924961; code d9e79ac): ablation of the three g v2 training changes on the 18-fold LODO and a training-data-only nested 3-fold selection of lambda in {0, 1, 2, 5, 10} V.
- Outcome (pre-registered rule): **selected variant non-inferior to M_bias v1**; primary GMR(selected, v1) 0.997 [0.962, 1.027], 7/18 lower. Selected all-device / combined 0.07391 / 0.06997; v1 0.07348 / 0.06953; g v2 0.07716 / 0.07323.
- Changes named as costing accuracy: none.

## Figures and tables

- `01_table_i15_2_primary.png`: Primary GMR(selected, v1), secondary GMRs, lambda* counts, outcome; sources: `out/research/explore/I15_2/decision.json`.
- `02_table_i15_2_variants.png`: Per-variant held-out RRMS (all-device, combined, median) and paired GMRs; sources: `out/research/explore/I15_2/decision.json`.
- `03_table_i15_2_contrasts.png`: Ablation contrasts with the pre-registered labels; sources: `out/research/explore/I15_2/decision.json`.
- `04_table_i15_2_negative.png`: Negative corrected gm / gds fractions on the penalty grids for every variant; sources: `out/research/explore/I15_2/decision.json`.
- `05_table_i15_2_validity.png`: Validity V1-V3; sources: `out/research/explore/I15_2/validity/validity.json`, `out/research/explore/I15_2/inner/inner.json`.
- `06_i15_2_lambda.png`: GMR vs v1 and negative gm / gds fractions vs lambda; sources: `out/research/explore/I15_2/decision.json`.
- `07_i15_2_per_device.png`: Per-device held-out RRMS and ratios to v1, lambda* per fold; sources: `out/research/explore/I15_2/decision.json`, `out/research/explore/I13_1/lodo/results.json`, `out/research/explore/I13_1/all18/results.json`, `out/research/explore/I15_2/lodo_sym_e1_l0/results.json`.
- `08_i15_2_inner_scores.png`: Inner selection scores per outer fold and lambda; sources: `out/research/explore/I15_2/inner/inner.json`, `out/research/explore/I15_2/decision.json`.
- `09_i15_2_iv_nmos_L1_W1p6.png`: I-V of nmos_L1_W1p6: measured 77 K points, NGSpice of the published card, and NGSpice re-simulated held-out cards of C0_refit, v1, sym EPS 0.01 lambda 0, g v2 and the selected variant (x exp(fold g)); sources: `out/research/explore/I13_1/records/real_dev.npz`, `out/research/explore/I13_1/records/real_test.npz`, `out/research/explore/I13_1/all18/heldout_cards.json`, `out/research/explore/I13_1/lodo/heldout_cards.json`, `out/research/explore/I15_2/lodo_sym_e1_l0/heldout_cards.json`, `I15_2/lodo_<selected>/heldout_cards.json`, `upstream committed {vd,vg}_sweep/*.txt`.
- `10_i15_2_iv_pmos_L0p35_W1p6.png`: I-V of pmos_L0p35_W1p6: measured 77 K points, NGSpice of the published card, and NGSpice re-simulated held-out cards of C0_refit, v1, sym EPS 0.01 lambda 0, g v2 and the selected variant (x exp(fold g)); sources: `out/research/explore/I13_1/records/real_dev.npz`, `out/research/explore/I13_1/records/real_test.npz`, `out/research/explore/I13_1/all18/heldout_cards.json`, `out/research/explore/I13_1/lodo/heldout_cards.json`, `out/research/explore/I15_2/lodo_sym_e1_l0/heldout_cards.json`, `I15_2/lodo_<selected>/heldout_cards.json`, `upstream committed {vd,vg}_sweep/*.txt`.
- `11_i15_2_iv_pmos_L8_W1p6.png`: I-V of pmos_L8_W1p6: measured 77 K points, NGSpice of the published card, and NGSpice re-simulated held-out cards of C0_refit, v1, sym EPS 0.01 lambda 0, g v2 and the selected variant (x exp(fold g)); sources: `out/research/explore/I13_1/records/real_dev.npz`, `out/research/explore/I13_1/records/real_test.npz`, `out/research/explore/I13_1/all18/heldout_cards.json`, `out/research/explore/I13_1/lodo/heldout_cards.json`, `out/research/explore/I15_2/lodo_sym_e1_l0/heldout_cards.json`, `I15_2/lodo_<selected>/heldout_cards.json`, `upstream committed {vd,vg}_sweep/*.txt`.
- `12_i15_2_iv_pmos_L2_W5.png`: I-V of pmos_L2_W5: measured 77 K points, NGSpice of the published card, and NGSpice re-simulated held-out cards of C0_refit, v1, sym EPS 0.01 lambda 0, g v2 and the selected variant (x exp(fold g)); sources: `out/research/explore/I13_1/records/real_dev.npz`, `out/research/explore/I13_1/records/real_test.npz`, `out/research/explore/I13_1/all18/heldout_cards.json`, `out/research/explore/I13_1/lodo/heldout_cards.json`, `out/research/explore/I15_2/lodo_sym_e1_l0/heldout_cards.json`, `I15_2/lodo_<selected>/heldout_cards.json`, `upstream committed {vd,vg}_sweep/*.txt`.

- `tables.md`: the tables above in markdown.
