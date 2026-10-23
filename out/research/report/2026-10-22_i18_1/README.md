# Report batch 2026-10-22 · i18_1

Generated 2026-10-22 23:15 by `.venv/bin/python -I scripts/research/plot_explore_i18_1.py --date 2026-10-22`. 347 recomputed values checked against decision.json. NGSpice runs: re-simulation of TU Delft I-V devices only.

## Contents

- I18-1 gated correction over 9 measured DEV datasets: supported (C1 holds, C2 holds).
- sky130 GMR(gated, C0_refit) 0.743 [0.654, 0.855]; folds applied: sky130 77 K (all 18) 18/18, NIST sky130 4 K lvt (I8-1) 53/53, NIST sky130 4 K, RT card (I9-1) 87/87, IHP SG13G2 300 K 0/18, IHP SG13G2 233 K 0/18, IHP SG13G2 343 K 0/18, IHP SG13G2 398 K 0/18, TU Delft 4.13 K 2/7, TU Delft 295.0 K 1/7.

## Figures and tables

- `01_table_i18_1_rule.png`: Rule C1 / C2, validity, outcome; sources: `out/research/explore/I18_1/decision.json`, `out/research/explore/I18_1/prereg.json`.
- `02_table_i18_1_datasets.png`: Per dataset: arm, unit, gate decisions vs prediction, GMR(gated, C0_refit); sources: `out/research/explore/I18_1/decision.json`.
- `03_table_i18_1_secondary_v1.png`: Secondary v1 gate on every dataset with v1 rows; sources: `out/research/explore/I18_1/decision.json`.
- `04_table_i18_1_s3_v2.png`: S3_v2 per fold summary, Spearman with the fold outcome, oracle reference; sources: `out/research/explore/I18_1/decision.json`.
- `05_i18_1_forest.png`: GMR vs C0_refit per dataset: gated vs correction always; sources: `out/research/explore/I18_1/decision.json`.
- `06_i18_1_s3_per_fold.png`: S3_f per outer fold and dataset with the threshold; sources: `out/research/explore/I18_1/decision.json`, `I18_1/s3/*.json`.
- `07_i18_1_s3_scatter.png`: S3_v2 vs S3_v1, and S3_f vs fold outcome; sources: `out/research/explore/I18_1/decision.json`.
- `08_i18_1_iv_tudelft_4K_nmos_W1_L1_d128.png`: I-V of t40_nmos_W1_L1_d128 (4.13 K), gated: selected g v2 applied (S3_f -0.003 < 0): measured points, NGSpice stock card, re-simulated C0_refit and selected held-out cards; sources: `I18_2/base_*/heldout_cards.json`, `I18_2/lodo_*/heldout_cards.json`.
- `09_i18_1_iv_tudelft_4K_nmos_W0p12_L0p04_d152.png`: I-V of t40_nmos_W0p12_L0p04_d152 (4.13 K), gated: C0_refit kept (S3_f +0.001 >= 0): measured points, NGSpice stock card, re-simulated C0_refit and selected held-out cards; sources: `I18_2/base_*/heldout_cards.json`, `I18_2/lodo_*/heldout_cards.json`.
- `10_i18_1_iv_tudelft_295K_pmos_W1_L1_d328.png`: I-V of t40_pmos_W1_L1_d328 (295.0 K), gated: selected g v2 applied (S3_f -0.106 < 0): measured points, NGSpice stock card, re-simulated C0_refit and selected held-out cards; sources: `I18_2/base_*/heldout_cards.json`, `I18_2/lodo_*/heldout_cards.json`.
- `11_i18_1_iv_tudelft_295K_nmos_W0p12_L0p04_d152.png`: I-V of t40_nmos_W0p12_L0p04_d152 (295.0 K), gated: C0_refit kept (S3_f +0.451 >= 0): measured points, NGSpice stock card, re-simulated C0_refit and selected held-out cards; sources: `I18_2/base_*/heldout_cards.json`, `I18_2/lodo_*/heldout_cards.json`.

- `tables.md`: the tables above in markdown.
