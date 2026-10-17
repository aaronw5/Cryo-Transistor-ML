# Research report index

Newest batch first. Each batch folder holds the PNGs, `tables.md` and a `README.md` that names the result files behind every figure and table. Batches are written by `scripts/research/report_figs.py --topic <topic> --date <YYYY-MM-DD>`.

## 2026-10-16 · baseline_results

Folder: [2026-10-16_baseline_results/](2026-10-16_baseline_results/) · [tables.md](2026-10-16_baseline_results/tables.md) · [README.md](2026-10-16_baseline_results/README.md)

- Measured 77 K data, all-device RRMS (raw / +FD): Pipeline 43, round 4 0.1513 / 0.1465 (1 failed device(s) raw; 1 failed device(s) +FD); NGSpice swap 43, round 1 0.1125 / 0.1106; Direct NGSpice 43, round 3 0.1049 / 0.1044. References: released 0.2751, paper 7 0.2284, paper 43 0.2038.
- All-device RRMS by round (raw): Pipeline 7 r1-r4 0.2367 -> 0.1921 -> 0.1758 -> 0.1697; Pipeline 43 r1-r4 0.2375 -> 0.1965 -> 0.1716 -> 0.1513; Direct NGSpice 43 r1-r3 0.1201 -> 0.1094 -> 0.1049.
- Round-1 43-parameter surrogate, mean predicted / NGSpice RRMS: pipeline picks 0.2034 / 0.2375, direct-fit cards 0.5074 / 0.1269; opposite ordering on 17 of 18 devices.
- Temperature benchmark, held-out all-device RRMS (raw / +FD): direct 0.0362 / 0.0117; ngspice_search 0.0021 / 0.0004; surrogate_paper_mlp 0.0081 / 0.0007; surrogate_deeponet 0.0120 / 0.0022.

Figures and tables: [01_real_methods.png](2026-10-16_baseline_results/01_real_methods.png) · [02_table_real_methods.png](2026-10-16_baseline_results/02_table_real_methods.png) · [03_rounds.png](2026-10-16_baseline_results/03_rounds.png) · [04_table_rounds.png](2026-10-16_baseline_results/04_table_rounds.png) · [05_surrogate_gap.png](2026-10-16_baseline_results/05_surrogate_gap.png) · [06_per_device.png](2026-10-16_baseline_results/06_per_device.png) · [07_temperature.png](2026-10-16_baseline_results/07_temperature.png) · [08_table_temperature.png](2026-10-16_baseline_results/08_table_temperature.png) · [09_table_temperature_per_temp.png](2026-10-16_baseline_results/09_table_temperature_per_temp.png)
