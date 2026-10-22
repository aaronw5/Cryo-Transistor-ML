# Report batch 2026-10-22 · i17_2

Generated 2026-10-22 12:51 by `.venv/bin/python -I scripts/research/plot_explore_i17_2.py --date 2026-10-22`. NGSpice runs: re-simulation of the stock, C0_refit, M_bias v1 and selected cards for the 4 I-V DUTs (checked against the stored rows); everything else from stored files.

## Contents

- I17-2: second-PDK replication of the selected g v2 recipe on measured IHP SG13G2 (PSP103, I5-1 data), 300 K, 18 DEV DUTs, LODO within class. Pre-registration `out/research/explore/I17_2/prereg.md`, commit c071472; code bd660cb.
- Outcome (pre-registered rule): **worse than C0_refit**; secondary: not shown non-inferior to M_bias v1; Part B not run. GMR(selected, C0_refit) 1.255 [1.026, 1.600] (9/18 lower); GMR(selected, M_bias v1) 1.034 [0.990, 1.107] (7/18 lower).
- All-device RRMS: C0_refit 0.01906, M_bias v1 0.02029, selected 0.02045; lambda* counts 0 V: 2, 1 V: 3, 2 V: 6, 5 V: 2, 10 V: 5.

## Figures and tables

- `01_table_i17_2_setup.png`: Dataset choice, setup and the reasons Part B was not run; sources: `out/research/explore/I17_2/prereg.json`.
- `02_table_i17_2_validity.png`: Validity V1-V4; sources: `out/research/explore/I17_2/base/results.json`, `out/research/explore/I17_2/inner/inner.json`, `out/research/explore/I17_2/pens/pens.json`.
- `03_table_i17_2_rules.png`: Primary and secondary GMR vs the pre-registered rules, predicted signs; sources: `out/research/explore/I17_2/decision.json`.
- `04_table_i17_2_arms.png`: Held-out RRMS of every arm, GMR vs C0_refit and vs M_bias v1 (device and cluster CI); sources: `out/research/explore/I17_2/decision.json`.
- `05_table_i17_2_class.png`: LV / HV split; sources: `out/research/explore/I17_2/decision.json`.
- `06_table_i17_2_lambda.png`: Nested lambda selection per outer fold; sources: `out/research/explore/I17_2/inner/inner.json`, `out/research/explore/I17_2/decision.json`.
- `07_table_i17_2_negative.png`: Negative corrected gds / gm fractions on the penalty grids; sources: `out/research/explore/I17_2/decision.json`.
- `08_i17_2_per_device.png`: Per-DUT held-out RRMS: C0_refit, M_bias v1, lambda 10, selected; sources: `out/research/explore/I17_2/decision.json`.
- `09_i17_2_ratio_per_device.png`: Per-DUT RRMS ratio to C0_refit (v1, selected) with lambda*; sources: `out/research/explore/I17_2/decision.json`.
- `10_i17_2_lambda_curve.png`: GMR vs lambda (vs C0_refit and vs v1) and the selected variant; sources: `out/research/explore/I17_2/decision.json`.
- `11_i17_2_iv_lv_nmos_W0p3_L10.png`: I-V of lv_nmos_W0p3_L10: measured 300 K points, NGSpice of the stock PDK card, and NGSpice re-simulated C0_refit, M_bias v1 and selected held-out cards (x exp(g)); sources: `out/research/explore/I17_2/base/heldout_cards.json`, `I17_2/lodo_sym_e1_l{lambda*}/{corrections,heldout_cards}.json`, `I5_1/data/ihp_meas.npz (not in git)`.
- `12_i17_2_iv_lv_pmos_W10_L10.png`: I-V of lv_pmos_W10_L10: measured 300 K points, NGSpice of the stock PDK card, and NGSpice re-simulated C0_refit, M_bias v1 and selected held-out cards (x exp(g)); sources: `out/research/explore/I17_2/base/heldout_cards.json`, `I17_2/lodo_sym_e1_l{lambda*}/{corrections,heldout_cards}.json`, `I5_1/data/ihp_meas.npz (not in git)`.
- `13_i17_2_iv_hv_pmos_W0p6_L0p5.png`: I-V of hv_pmos_W0p6_L0p5: measured 300 K points, NGSpice of the stock PDK card, and NGSpice re-simulated C0_refit, M_bias v1 and selected held-out cards (x exp(g)); sources: `out/research/explore/I17_2/base/heldout_cards.json`, `I17_2/lodo_sym_e1_l{lambda*}/{corrections,heldout_cards}.json`, `I5_1/data/ihp_meas.npz (not in git)`.
- `14_i17_2_iv_hv_pmos_W0p3_L10.png`: I-V of hv_pmos_W0p3_L10: measured 300 K points, NGSpice of the stock PDK card, and NGSpice re-simulated C0_refit, M_bias v1 and selected held-out cards (x exp(g)); sources: `out/research/explore/I17_2/base/heldout_cards.json`, `I17_2/lodo_sym_e1_l{lambda*}/{corrections,heldout_cards}.json`, `I5_1/data/ihp_meas.npz (not in git)`.

- `tables.md`: the tables above in markdown.
