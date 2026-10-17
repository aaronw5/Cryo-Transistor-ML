# Research report index

Newest batch first. Each batch folder holds the PNGs, `tables.md` and a `README.md` that names the result files behind every figure and table. Batches are written by `scripts/research/report_figs.py --topic <topic> --date <YYYY-MM-DD>`.

## 2026-10-16 · dev_iteration1

Folder: [2026-10-16_dev_iteration1/](2026-10-16_dev_iteration1/) · [tables.md](2026-10-16_dev_iteration1/tables.md) · [README.md](2026-10-16_dev_iteration1/README.md)

- DEV, 36 simulated targets, pooled all-device mean RRMS at B500 / B5000: sketch_surrogate_k8 0.0313 / 0.0091; sketch_none_k8 0.0370 / 0.0021; hybrid_surrogate 0.0463 / 0.0114; hybrid_none 0.0579 / 0.0058; anytime_sur_presearch_race 0.0287 / 0.0070; anytime_none_race 0.0428 / 0.0082; cmaes_trf 0.0565 / 0.0109.
- Paired mean difference [95% CI] and win rate: sketch_surrogate_k8 − sketch_none_k8: B500 -0.0058 [-0.0126, 0.0003], win rate 0.67; B1000 0.0028 [-0.0027, 0.0085], win rate 0.53; B2000 0.0057 [1.4e-05, 0.0127], win rate 0.44; B5000 0.0069 [0.0024, 0.0133], win rate 0.25 | hybrid_surrogate − hybrid_none: B500 -0.0116 [-0.0230, -0.0024], win rate 0.64; B1000 -0.0003 [-0.0129, 0.0138], win rate 0.64; B2000 0.0025 [-0.0022, 0.0070], win rate 0.56; B5000 0.0056 [0.0018, 0.0112], win rate 0.28.
- Failed-curve audit: 16 of 227 audited runs have a device with a failed included curve; 14 of the 20 audited headline numbers come from runs with a failed device. Example: pipeline 43 round 4 raw all-device stored / valid-only / penalized 0.1513 / 0.1586 / 0.6500.

Figures and tables: [01_compare_curves.png](2026-10-16_dev_iteration1/01_compare_curves.png) · [02_table_dev_primary.png](2026-10-16_dev_iteration1/02_table_dev_primary.png) · [03_table_dev_paired.png](2026-10-16_dev_iteration1/03_table_dev_paired.png) · [04_table_audit.png](2026-10-16_dev_iteration1/04_table_audit.png)

## 2026-10-16 · baseline_results

Folder: [2026-10-16_baseline_results/](2026-10-16_baseline_results/) · [tables.md](2026-10-16_baseline_results/tables.md) · [README.md](2026-10-16_baseline_results/README.md)

- Measured 77 K data, all-device RRMS (raw / +FD): Pipeline 43, round 4 0.1513 / 0.1465 (1 failed device(s) raw; 1 failed device(s) +FD); NGSpice swap 43, round 1 0.1125 / 0.1106; Direct NGSpice 43, round 3 0.1049 / 0.1044. References: released 0.2751, paper 7 0.2284, paper 43 0.2038.
- All-device RRMS by round (raw): Pipeline 7 r1-r4 0.2367 -> 0.1921 -> 0.1758 -> 0.1697; Pipeline 43 r1-r4 0.2375 -> 0.1965 -> 0.1716 -> 0.1513; Direct NGSpice 43 r1-r3 0.1201 -> 0.1094 -> 0.1049.
- Round-1 43-parameter surrogate, mean predicted / NGSpice RRMS: pipeline picks 0.2034 / 0.2375, direct-fit cards 0.5074 / 0.1269; opposite ordering on 17 of 18 devices.
- Temperature benchmark, held-out all-device RRMS (raw / +FD): direct 0.0362 / 0.0117; ngspice_search 0.0021 / 0.0004; surrogate_paper_mlp 0.0081 / 0.0007; surrogate_deeponet 0.0120 / 0.0022.

Figures and tables: [01_real_methods.png](2026-10-16_baseline_results/01_real_methods.png) · [02_table_real_methods.png](2026-10-16_baseline_results/02_table_real_methods.png) · [03_rounds.png](2026-10-16_baseline_results/03_rounds.png) · [04_table_rounds.png](2026-10-16_baseline_results/04_table_rounds.png) · [05_surrogate_gap.png](2026-10-16_baseline_results/05_surrogate_gap.png) · [06_per_device.png](2026-10-16_baseline_results/06_per_device.png) · [07_temperature.png](2026-10-16_baseline_results/07_temperature.png) · [08_table_temperature.png](2026-10-16_baseline_results/08_table_temperature.png) · [09_table_temperature_per_temp.png](2026-10-16_baseline_results/09_table_temperature_per_temp.png)
