# Research report index

Newest batch first. Each batch folder holds the PNGs, `tables.md` and a `README.md` that names the result files behind every figure and table. Batches are written by `scripts/research/report_figs.py --topic <topic> --date <YYYY-MM-DD>`.

## 2026-10-19 · dev_iteration2_ablation

Folder: [2026-10-19_dev_iteration2_ablation/](2026-10-19_dev_iteration2_ablation/) · [tables.md](2026-10-19_dev_iteration2_ablation/tables.md) · [README.md](2026-10-19_dev_iteration2_ablation/README.md)

- DEV iteration 2 (ablation), tier a, pooled mean paper RRMS (strict) at B500 / B5000: sketch_surrogate_k8 0.0334 / 0.0082; sketch_none_k8 0.0367 / 0.0024; lam_curve_k8 0.0317 / 0.0062; hybrid_surrogate 0.0463 / 0.0114; hybrid_none 0.0579 / 0.0058; dfols_style 0.0579 / 0.0076; cmaes_trf 0.0565 / 0.0109.
- DEV iteration 2 (ablation), tier c, pooled mean excess over floor (RRMS − floor) at B500 / B5000: sketch_surrogate_k8 0.0221 / 0.0047; sketch_none_k8 0.0225 / 0.0022; lam_curve_k8 0.0200 / 0.0039; hybrid_surrogate 0.0304 / 0.0064; hybrid_none 0.0401 / 0.0041; dfols_style 0.0406 / 0.0062; cmaes_trf 0.0388 / 0.0151.
- Pre-registered λ decision (n2b): negative; frozen candidate per family: sketch family: none; hybrid family: none.
- Pre-registered C1-C6: C1_a_B5000, C2_a_B500, C3_a_B5000, C5_a_B500, C5_a_B2000, C5_a_B5000, C5_c_B500, C5_c_B2000, C5_c_B5000, C6_a_B500, C6_c_B500, C6_c_B2000 have Holm p ≤ 0.05.

Figures and tables: [01_anytime_families_tier_a.png](2026-10-19_dev_iteration2_ablation/01_anytime_families_tier_a.png) · [02_anytime_families_tier_c.png](2026-10-19_dev_iteration2_ablation/02_anytime_families_tier_c.png) · [03_table_primary_tier_a.png](2026-10-19_dev_iteration2_ablation/03_table_primary_tier_a.png) · [04_table_primary_tier_c.png](2026-10-19_dev_iteration2_ablation/04_table_primary_tier_c.png) · [05_table_paired_lambda.png](2026-10-19_dev_iteration2_ablation/05_table_paired_lambda.png) · [06_table_paired_surrogate.png](2026-10-19_dev_iteration2_ablation/06_table_paired_surrogate.png) · [07_table_paired_dfols.png](2026-10-19_dev_iteration2_ablation/07_table_paired_dfols.png) · [08_table_lambda_decision.png](2026-10-19_dev_iteration2_ablation/08_table_lambda_decision.png)

## 2026-10-17 · arch_study

Folder: [2026-10-17_arch_study/](2026-10-17_arch_study/) · [tables.md](2026-10-17_arch_study/tables.md) · [README.md](2026-10-17_arch_study/README.md)

- Surrogate architecture study, 43 parameters, round-1 box, 6 architectures: end-to-end all-device RRMS 0.2274 (DeepONet) to 0.2549 (FNO); same search with NGSpice in place of the surrogate 0.1125; corrected pipeline r1 0.2382.
- Lowest test loss: FNO 0.0259, end-to-end 0.2549 (rank 6 of 6, 1 = lowest RRMS). Devices where the surrogate ranks the direct-fit card better than the pipeline pick: paper MLP 6, wide MLP 6, ResMLP 8, large ResMLP 5, DeepONet 6, FNO 6 (NGSpice: 18 of 18).
- Corrected pipeline 43 + NGSpice FD, all-device (combined), raw → FD: r1 0.2382 → 0.1800 (0.2289 → 0.1703), 350 FD calls/device, failed devices 0 / 0; r4 0.1522 → 0.1477 (0.1437 → 0.1394), 381 FD calls/device, failed devices 0 / 0.
- The end-to-end search in this study ran on the 18 measured devices; it was queued before the DEV/TEST protocol was fixed. It is reported as a diagnostic ablation and is not used for method selection.

Figures and tables: [01_table_arch.png](2026-10-17_arch_study/01_table_arch.png) · [02_arch_loss_vs_e2e.png](2026-10-17_arch_study/02_arch_loss_vs_e2e.png) · [03_arch_per_device.png](2026-10-17_arch_study/03_arch_per_device.png) · [04_table_pipe_fixed_fd.png](2026-10-17_arch_study/04_table_pipe_fixed_fd.png)

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
