# Report batch 2026-10-23 · i19_1

Generated 2026-10-23 02:57 by `.venv/bin/python -I scripts/research/plot_explore_i19_1.py --date 2026-10-23`. 94 recomputed values checked against decision.json and the S3_D files. No NGSpice runs, no training.

## Contents

- I19-1 dataset-level gate: supported. Part R (retrospective): supported (retrospective); pooled GMR gated_D 0.921 vs correction always 0.958 and per-fold gate 0.929. Part C: 19/20 sign correct.
- Part P TEST decisions: nist_4K apply, nist_rt_card_4K apply, ihp_233K C0_refit, ihp_300K C0_refit, ihp_343K C0_refit, ihp_398K C0_refit, tudelft_4K C0_refit, tudelft_295K C0_refit.

## Figures and tables

- `01_table_i19_1_rule.png`: Rules R1 / R2 / C, validity, outcome; sources: `out/research/explore/I19_1/decision.json`, `out/research/explore/I19_1/prereg.json`.
- `02_table_i19_1_part_r.png`: Part R per dataset: S3_D, decision, GMR vs per-fold gate and correction always; sources: `out/research/explore/I19_1/decision.json`.
- `03_table_i19_1_pooled.png`: Part R pooled GMR per method; sources: `out/research/explore/I19_1/decision.json`.
- `04_table_i19_1_secondary_v1.png`: Secondary v1 dataset-level gate; sources: `out/research/explore/I19_1/decision.json`.
- `05_table_i19_1_test_decisions.png`: Part P committed TEST gate decisions and I8-2 disagreements; sources: `out/research/explore/I19_1/test_gate_decisions.json`.
- `06_table_i19_1_part_c.png`: Part C per simulated dataset; sources: `out/research/explore/I19_1/decision.json`.
- `07_i19_1_forest.png`: GMR vs C0_refit per DEV dataset: dataset-level gate, per-fold gate, correction always; sources: `out/research/explore/I19_1/decision.json`.
- `08_i19_1_s3d_scatter.png`: S3_D vs realised gain with the threshold, measured and simulated datasets; sources: `out/research/explore/I19_1/decision.json`, `I19_1/s3/*.json`, `I7_1/decision.json`.

- `tables.md`: the tables above in markdown.
