# Report batch 2026-10-16 · baseline_results

Generated 2026-10-16 15:04 by `scripts/research/report_figs.py --topic baseline_results --date 2026-10-16` from result files only (no simulation). RRMS is the paper metric with frozen published-baseline curve inclusion, re-simulated in NGSpice, unless an item says otherwise.

## Figures and tables

- `01_real_methods.png`: Paper RRMS on the 18 measured devices for each method row, raw (open) and after NGSpice FD polish (filled) where a polished version exists, in the all-device convention with the three reference-card lines and in the combined convention, annotated with mean NGSpice calls per device; sources: `data/processed/cold_loop/full_r1/manifest.json`, `data/processed/cold_loop/full_r4/manifest.json`, `data/processed/cold_loop/seven_r1/manifest.json`, `data/processed/cold_loop/seven_r4/manifest.json`, `out/cold_loop/full/round_1/extract/summary.json`, `out/cold_loop/full/round_4/extract/summary.json`, `out/cold_loop/seven/round_1/extract/summary.json`, `out/cold_loop/seven/round_4/extract/summary.json`, `out/research/ngspice_direct/full_r1_direct_cold/summary.json`, `out/research/ngspice_direct/full_r1_direct_seeded/summary.json`, `out/research/ngspice_direct/full_r3_direct/summary.json`, `out/research/ngspice_direct/seven_r1_direct/summary.json`, `out/research/polished/direct43_fd/real/summary.json`, `out/research/polished/direct43_r3_fd/real/summary.json`, `out/research/polished/direct7_fd/real/summary.json`, `out/research/polished/pipe43_r1_fd/real/summary.json`, `out/research/polished/pipe43_r4_fd/real/summary.json`, `out/research/polished/pipe_r4_fd/real/summary.json`, `out/research/polished/swap_full_r1_fd/real/summary.json`, `out/research/swap_search/full_round_1/summary.json`, `out/research/swap_search/seven_round_1/summary.json`.
- `02_table_real_methods.png`: Table of the rows in the previous figure plus the reference cards: all-device, combined, nMOS and pMOS RRMS and NGSpice calls per device (this run and cumulative); sources: `data/processed/cold_loop/full_r1/manifest.json`, `data/processed/cold_loop/full_r4/manifest.json`, `data/processed/cold_loop/seven_r1/manifest.json`, `data/processed/cold_loop/seven_r4/manifest.json`, `out/cold_loop/full/round_1/extract/summary.json`, `out/cold_loop/full/round_4/extract/summary.json`, `out/cold_loop/seven/round_1/extract/summary.json`, `out/cold_loop/seven/round_4/extract/summary.json`, `out/research/ngspice_direct/full_r1_direct_cold/summary.json`, `out/research/ngspice_direct/full_r1_direct_seeded/summary.json`, `out/research/ngspice_direct/full_r3_direct/summary.json`, `out/research/ngspice_direct/seven_r1_direct/summary.json`, `out/research/polished/direct43_fd/real/summary.json`, `out/research/polished/direct43_r3_fd/real/summary.json`, `out/research/polished/direct7_fd/real/summary.json`, `out/research/polished/pipe43_r1_fd/real/summary.json`, `out/research/polished/pipe43_r4_fd/real/summary.json`, `out/research/polished/pipe_r4_fd/real/summary.json`, `out/research/polished/swap_full_r1_fd/real/summary.json`, `out/research/swap_search/full_round_1/summary.json`, `out/research/swap_search/seven_round_1/summary.json`, `out/research/reference_scores.json`.
- `03_rounds.png`: All-device RRMS (raw method output) against round for the cold pipeline at 7 and 43 parameters and for direct NGSpice at 43 parameters (chain of runs linked by spec source_run), with the round-1 NGSpice-swap points as diamonds and reference-card lines; sources: `data/processed/cold_loop/full_r1/manifest.json`, `data/processed/cold_loop/full_r2/manifest.json`, `data/processed/cold_loop/full_r3/manifest.json`, `data/processed/cold_loop/full_r4/manifest.json`, `data/processed/cold_loop/seven_r1/manifest.json`, `data/processed/cold_loop/seven_r2/manifest.json`, `data/processed/cold_loop/seven_r3/manifest.json`, `data/processed/cold_loop/seven_r4/manifest.json`, `out/cold_loop/full/round_1/extract/summary.json`, `out/cold_loop/full/round_2/extract/summary.json`, `out/cold_loop/full/round_3/extract/summary.json`, `out/cold_loop/full/round_4/extract/summary.json`, `out/cold_loop/seven/round_1/extract/summary.json`, `out/cold_loop/seven/round_2/extract/summary.json`, `out/cold_loop/seven/round_3/extract/summary.json`, `out/cold_loop/seven/round_4/extract/summary.json`, `out/research/ngspice_direct/full_r1_direct_seeded/summary.json`, `out/research/ngspice_direct/full_r2_direct/summary.json`, `out/research/ngspice_direct/full_r3_direct/summary.json`, `out/research/swap_search/full_round_1/summary.json`, `out/research/swap_search/seven_round_1/summary.json`.
- `04_table_rounds.png`: Table of the points in the rounds figure: all-device, combined, nMOS and pMOS RRMS and NGSpice calls per device (this run and cumulative); sources: `data/processed/cold_loop/full_r1/manifest.json`, `data/processed/cold_loop/full_r2/manifest.json`, `data/processed/cold_loop/full_r3/manifest.json`, `data/processed/cold_loop/full_r4/manifest.json`, `data/processed/cold_loop/seven_r1/manifest.json`, `data/processed/cold_loop/seven_r2/manifest.json`, `data/processed/cold_loop/seven_r3/manifest.json`, `data/processed/cold_loop/seven_r4/manifest.json`, `out/cold_loop/full/round_1/extract/summary.json`, `out/cold_loop/full/round_2/extract/summary.json`, `out/cold_loop/full/round_3/extract/summary.json`, `out/cold_loop/full/round_4/extract/summary.json`, `out/cold_loop/seven/round_1/extract/summary.json`, `out/cold_loop/seven/round_2/extract/summary.json`, `out/cold_loop/seven/round_3/extract/summary.json`, `out/cold_loop/seven/round_4/extract/summary.json`, `out/research/ngspice_direct/full_r1_direct_seeded/summary.json`, `out/research/ngspice_direct/full_r2_direct/summary.json`, `out/research/ngspice_direct/full_r3_direct/summary.json`, `out/research/swap_search/full_round_1/summary.json`, `out/research/swap_search/seven_round_1/summary.json`.
- `05_surrogate_gap.png`: Per-device surrogate-predicted RRMS against NGSpice RRMS (log axes, y = x line) for the round-1 43-parameter pipeline picks and for the direct-fit reference cards (full_r1_direct_cold); sources: `out/research/surrogate_gap_full_round_1.json`, `out/research/ngspice_direct/full_r1_direct_cold/summary.json`, `out/cold_loop/full/round_1/extract/summary.json`.
- `06_per_device.png`: Per-device RRMS after NGSpice FD polish (log y, nMOS then pMOS) for pipeline 43 round 4, NGSpice swap 43 round 1 and direct NGSpice 43 round 3, with the paper 43-parameter card per device; sources: `out/research/polished/pipe43_r4_fd/real/summary.json`, `out/research/polished/swap_full_r1_fd/real/summary.json`, `out/research/polished/direct43_r3_fd/real/summary.json`, `out/research/reference_scores.json`.
- `07_temperature.png`: All-device RRMS against temperature (77-300 K, log y) for every temperature method found in out/research/temp, raw dashed and +FD solid, held-out temperatures (110, 190, 260 K) shaded; mean over simulated sets; sources: `out/research/temp/direct/set_0.json`, `out/research/temp/direct/set_1.json`, `out/research/temp/direct/set_2.json`, `out/research/temp/ngspice_search/set_0.json`, `out/research/temp/ngspice_search/set_1.json`, `out/research/temp/ngspice_search/set_2.json`, `out/research/temp/surrogate_paper_mlp/set_0.json`, `out/research/temp/surrogate_paper_mlp/set_1.json`, `out/research/temp/surrogate_paper_mlp/set_2.json`.
- `08_table_temperature.png`: Table of observed and held-out all-device, combined, nMOS and pMOS RRMS and NGSpice calls per device for each temperature method, raw and +FD; sources: `out/research/temp/direct/set_0.json`, `out/research/temp/direct/set_1.json`, `out/research/temp/direct/set_2.json`, `out/research/temp/ngspice_search/set_0.json`, `out/research/temp/ngspice_search/set_1.json`, `out/research/temp/ngspice_search/set_2.json`, `out/research/temp/surrogate_paper_mlp/set_0.json`, `out/research/temp/surrogate_paper_mlp/set_1.json`, `out/research/temp/surrogate_paper_mlp/set_2.json`.
- `09_table_temperature_per_temp.png`: Table of the values plotted in the temperature figure (all-device RRMS per temperature, raw and +FD); sources: `out/research/temp/direct/set_0.json`, `out/research/temp/direct/set_1.json`, `out/research/temp/direct/set_2.json`, `out/research/temp/ngspice_search/set_0.json`, `out/research/temp/ngspice_search/set_1.json`, `out/research/temp/ngspice_search/set_2.json`, `out/research/temp/surrogate_paper_mlp/set_0.json`, `out/research/temp/surrogate_paper_mlp/set_1.json`, `out/research/temp/surrogate_paper_mlp/set_2.json`.

- `tables.md`: the tables above in markdown.

## Notes

- out/research/polished/pipe43_r1_fd/real/summary.json: FD start re-score differs from the method's own score on 1 device(s): nmos_L0p15_W1p6 0.2035 vs 1.1832 (method's own score shown as raw)
- out/research/polished/pipe43_r4_fd/real/summary.json: FD start re-score differs from the method's own score on 1 device(s): nmos_L8_W1p6 0.0263 vs 9.0026 (method's own score shown as raw)

## Skipped inputs

- none

## Result files present but not used in this batch

- `out/research/ngspice_direct/seven_r1_direct_1k/summary.json`
- `out/research/ngspice_direct/seven_r1_direct_flowinit_1k/summary.json`
- `out/research/ngspice_direct/seven_r1_direct_long/summary.json`
- `out/research/ngspice_direct/sim0/full_r1_direct_seeded/summary.json`
- `out/research/ngspice_direct/sim0/seven_r1_direct/summary.json`
- `out/research/ngspice_direct/sim1/full_r1_direct_seeded/summary.json`
- `out/research/ngspice_direct/sim1/seven_r1_direct/summary.json`
- `out/research/ngspice_direct/sim2/full_r1_direct_seeded/summary.json`
- `out/research/ngspice_direct/sim2/seven_r1_direct/summary.json`
- `out/research/ngspice_direct/smoke/summary.json`
- `out/research/polished/A_r4_fd/real/summary.json`
- `out/research/polished/A_r4_fd/sim0/summary.json`
- `out/research/polished/A_r4_fd/sim1/summary.json`
- `out/research/polished/A_r4_fd/sim2/summary.json`
- `out/research/polished/direct43_fd/sim0/summary.json`
- `out/research/polished/direct43_fd/sim1/summary.json`
- `out/research/polished/direct43_fd/sim2/summary.json`
- `out/research/polished/direct7_fd/sim0/summary.json`
- `out/research/polished/direct7_fd/sim1/summary.json`
- `out/research/polished/direct7_fd/sim2/summary.json`
- `out/research/polished/flow_pooled_fd/real/summary.json`
- `out/research/polished/flow_pooled_fd/sim0/summary.json`
- `out/research/polished/flow_pooled_fd/sim1/summary.json`
- `out/research/polished/flow_pooled_fd/sim2/summary.json`
- `out/research/polished/flow_r1_fd/real/summary.json`
- `out/research/polished/flow_r1_fd/sim0/summary.json`
- `out/research/polished/flow_r1_fd/sim1/summary.json`
- `out/research/polished/flow_r1_fd/sim2/summary.json`
- `out/research/polished/pipe43_r4_fd/sim0/summary.json`
- `out/research/polished/pipe43_r4_fd/sim1/summary.json`
- `out/research/polished/pipe43_r4_fd/sim2/summary.json`
- `out/research/polished/pipe_r4_fd/sim0/summary.json`
- `out/research/polished/pipe_r4_fd/sim1/summary.json`
- `out/research/polished/pipe_r4_fd/sim2/summary.json`
- `out/research/polished/wta_pooled_fd/real/summary.json`
- `out/research/polished/wta_pooled_fd/sim0/summary.json`
- `out/research/polished/wta_pooled_fd/sim1/summary.json`
- `out/research/polished/wta_pooled_fd/sim2/summary.json`
- `out/research/polished/wta_r1_fd/real/summary.json`
- `out/research/polished/wta_r1_fd/sim0/summary.json`
- `out/research/polished/wta_r1_fd/sim1/summary.json`
- `out/research/polished/wta_r1_fd/sim2/summary.json`
