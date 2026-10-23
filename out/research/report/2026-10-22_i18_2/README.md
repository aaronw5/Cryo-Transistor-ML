# Report batch 2026-10-22 · i18_2

Generated 2026-10-22 23:15 by `.venv/bin/python -I scripts/research/plot_explore_i18_2.py --date 2026-10-22`. NGSpice runs: re-simulation of the stock, C0_refit, M_bias v1 and selected cards for the I-V devices (checked against the stored rows); everything else from stored files.

## Contents

- I18-2: selected g v2 recipe on measured TU Delft 40 nm (I6-1 data), 4.13 K and 295.0 K, 45 DEV devices, leave-one-group-out, BSIM4 clamp state C; Part B with I17-1 rules (amendment A1). Pre-registration commit 2454b15, amendment 1795e10, code 254e77d.
- I18-2 TU Delft 4.13 K (45 DEV, 7 groups): inconclusive; non-inferior to M_bias v1; GMR(selected, C0_refit) 0.826 [0.669, 1.014]; GMR(selected, v1) 0.908 [0.848, 0.965]; Part B: not deployable (Q3 not run).
- I18-2 TU Delft 295.0 K (45 DEV, 7 groups): inconclusive; not shown non-inferior to M_bias v1; GMR(selected, C0_refit) 0.829 [0.523, 1.247]; GMR(selected, v1) 1.022 [0.975, 1.056]; Part B: deployable with restrictions (Q3 not run).

## Figures and tables

- `01_table_i18_2_setup.png`: Setup, clamp state C audit, amendment A1, Q3 status; sources: `out/research/explore/I18_2/start/start.json`.
- `02_table_i18_2_validity.png`: Validity V1-V4 and VB; sources: `out/research/explore/I18_2/start/start.json`, `out/research/explore/I18_2/decision.json`.
- `03_table_i18_2_rules.png`: Primary / secondary GMR, Part B labels vs the pre-registered rules and predictions; sources: `out/research/explore/I18_2/decision.json`.
- `04_table_i18_2_arms.png`: Held-out RRMS of every arm, GMR with cluster and device CIs; sources: `out/research/explore/I18_2/decision.json`.
- `05_table_i18_2_groups.png`: Per geometry group; sources: `out/research/explore/I18_2/decision.json`.
- `06_table_i18_2_lambda.png`: Nested lambda selection and the full-data selection; sources: `out/research/explore/I18_2/decision.json`.
- `07_table_i18_2_negative.png`: Negative corrected gds / gm fractions; sources: `out/research/explore/I18_2/decision.json`.
- `08_table_i18_2_partb.png`: Part B core checks, restrictions and labels; sources: `I18_2/partB_<T>K/summary.json`.
- `09_table_i18_2_partb_accuracy.png`: Part B element-deck accuracy; sources: `I18_2/partB_<T>K/accuracy.json`.
- `10_i18_2_per_device.png`: Per-device held-out RRMS (C0_refit, M_bias v1, selected); sources: `out/research/explore/I18_2/decision.json`.
- `11_i18_2_ratio_by_group.png`: Per-device ratio to C0_refit by group with lambda*; sources: `out/research/explore/I18_2/decision.json`.
- `12_i18_2_lambda_curve.png`: GMR vs lambda with cluster CIs; sources: `out/research/explore/I18_2/decision.json`.
- `13_i18_2_partb_per_device.png`: Part B per device: default tolerance, runtime, V(d,s) = 0 current; sources: `out/research/explore/I18_2/partB_4K/q12.json`, `out/research/explore/I18_2/partB_295K/q12.json`.
- `14_i18_2_iv_4K_nmos_W0p12_L0p04_d158.png`: I-V of t40_nmos_W0p12_L0p04_d158 at 4.13 K: measured points, NGSpice of the stock card and re-simulated C0_refit, M_bias v1 and selected held-out cards (x exp(g)); sources: `I18_2/base_4K/heldout_cards.json`, `I18_2/lodo_4K_*/heldout_cards.json`, `I6_1/data/tudelft40.npz (not in git)`.
- `15_i18_2_iv_4K_pmos_W1_L1_d331.png`: I-V of t40_pmos_W1_L1_d331 at 4.13 K: measured points, NGSpice of the stock card and re-simulated C0_refit, M_bias v1 and selected held-out cards (x exp(g)); sources: `I18_2/base_4K/heldout_cards.json`, `I18_2/lodo_4K_*/heldout_cards.json`, `I6_1/data/tudelft40.npz (not in git)`.
- `16_i18_2_iv_295K_nmos_W0p12_L0p05_d162.png`: I-V of t40_nmos_W0p12_L0p05_d162 at 295.0 K: measured points, NGSpice of the stock card and re-simulated C0_refit, M_bias v1 and selected held-out cards (x exp(g)); sources: `I18_2/base_295K/heldout_cards.json`, `I18_2/lodo_295K_*/heldout_cards.json`, `I6_1/data/tudelft40.npz (not in git)`.
- `17_i18_2_iv_295K_pmos_W1_L1_d335.png`: I-V of t40_pmos_W1_L1_d335 at 295.0 K: measured points, NGSpice of the stock card and re-simulated C0_refit, M_bias v1 and selected held-out cards (x exp(g)); sources: `I18_2/base_295K/heldout_cards.json`, `I18_2/lodo_295K_*/heldout_cards.json`, `I6_1/data/tudelft40.npz (not in git)`.

- `tables.md`: the tables above in markdown.
