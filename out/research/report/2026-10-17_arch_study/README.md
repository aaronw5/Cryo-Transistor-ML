# Report batch 2026-10-17 · arch_study

Generated 2026-10-17 07:37 by `scripts/research/report_figs.py --topic arch_study --date 2026-10-17` from result files only (no simulation). RRMS is the paper metric with frozen published-baseline curve inclusion, re-simulated in NGSpice, unless an item says otherwise.

## Protocol caveat

The end-to-end search in this study ran on the 18 measured devices; it was queued before the DEV/TEST protocol was fixed. It is reported as a diagnostic ablation and is not used for method selection.

## Figures and tables

- `01_table_arch.png`: Table, one row per surrogate architecture plus three reference rows: trainable parameters, epochs, training time, test loss, median and mean curve NRMSE, mean absolute score error at the reference cards, devices where the surrogate and NGSpice rank the direct-fit card better, and end-to-end all-device, combined, nMOS and pMOS RRMS with the failed-device count; sources: `out/research/surrogate_arch/full_round_1/paper_mlp_f1.json`, `out/research/surrogate_arch/full_round_1/paper_mlp_wide_f1.json`, `out/research/surrogate_arch/full_round_1/resmlp_f1.json`, `out/research/surrogate_arch/full_round_1/resmlp_large_f1.json`, `out/research/surrogate_arch/full_round_1/deeponet_f1.json`, `out/research/surrogate_arch/full_round_1/fno_f1.json`, `out/research/swap_search/full_round_1/summary.json`, `out/cold_loop/full/round_1/extract/summary.json`, `out/research/pipe_fixed/full_r1/summary.json`, `out/research/surrogate_gap_full_round_1.json`.
- `02_arch_loss_vs_e2e.png`: Scatter of surrogate test loss (x) against end-to-end all-device RRMS (y), one labelled point per architecture, with a horizontal line at the all-device RRMS of the same search with NGSpice in place of the surrogate; sources: `out/research/surrogate_arch/full_round_1/paper_mlp_f1.json`, `out/research/surrogate_arch/full_round_1/paper_mlp_wide_f1.json`, `out/research/surrogate_arch/full_round_1/resmlp_f1.json`, `out/research/surrogate_arch/full_round_1/resmlp_large_f1.json`, `out/research/surrogate_arch/full_round_1/deeponet_f1.json`, `out/research/surrogate_arch/full_round_1/fno_f1.json`, `out/research/swap_search/full_round_1/summary.json`.
- `03_arch_per_device.png`: Matrix of end-to-end RRMS per device (columns) for each architecture and for the same search with NGSpice in place of the surrogate (rows), with the all-device mean in the last column; sources: `out/research/surrogate_arch/full_round_1/paper_mlp_f1.json`, `out/research/surrogate_arch/full_round_1/paper_mlp_wide_f1.json`, `out/research/surrogate_arch/full_round_1/resmlp_f1.json`, `out/research/surrogate_arch/full_round_1/resmlp_large_f1.json`, `out/research/surrogate_arch/full_round_1/deeponet_f1.json`, `out/research/surrogate_arch/full_round_1/fno_f1.json`, `out/research/swap_search/full_round_1/summary.json`.
- `04_table_pipe_fixed_fd.png`: Table of the corrected pipeline 43 cards (rounds 1 and 4) raw and after NGSpice FD polish: all-device, combined, nMOS and pMOS RRMS, the FD change in all-device RRMS, failed devices and NGSpice calls per device; sources: `out/research/pipe_fixed/full_r1/summary.json`, `data/processed/cold_loop/full_r1/manifest.json`, `out/research/polished/pipe43_r1_fixed_fd/real/summary.json`, `out/research/pipe_fixed/full_r4/summary.json`, `data/processed/cold_loop/full_r4/manifest.json`, `out/research/polished/pipe43_r4_fixed_fd/real/summary.json`.

- `tables.md`: the tables above in markdown.

## Skipped inputs

- none

## Result files present but not used in this batch

- none
