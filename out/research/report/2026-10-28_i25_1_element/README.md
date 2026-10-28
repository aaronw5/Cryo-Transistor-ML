# Report batch 2026-10-28 · i25_1_element

Generated 2026-10-28 03:00 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i25_1_element.py --date 2026-10-28`; stored files only.

I25-1 element checks (prereg section 10; pre-registration `out/research/explore/I25_1/element/prereg.md`): I17-1 Part B rules on the I25-1 full-data card + g pairs of both arms.

- arm A (7p card + g): not deployable; core failures Q1 equivalence |dI| <= 1e-6 |I| + 1e-13 A (tight); restrictions default tolerance above 1e-3; gds sign changes introduced; runtime overhead > 2x (default tolerance 1.98e-03, gds 6 of 90, runtime 4.49); element-deck RRMS full data 0.0783 / 0.0757 (stored 0.0783 / 0.0757), LODO 0.0823 / 0.0795.
- arm B (45p card + g): not deployable; core failures Q1 equivalence |dI| <= 1e-6 |I| + 1e-13 A (tight), Q1 convergence / non-finite, Q1 I_D at Vds = 0, Q2 gm sign change in strong inversion; restrictions default tolerance above 1e-3; gds sign changes introduced; runtime overhead > 2x (default tolerance 1.59e-03, gds 10 of 90, runtime 4.30); element-deck RRMS full data 0.0667 / 0.0635 (stored 0.0667 / 0.0635), LODO 0.0693 / 0.0661.

## Tables

- `01_table_element_checks.png`: Element checks vs I17-1 thresholds, both arms and the old 43p card + g; sources: `out/research/explore/I25_1/element/armA/decision.json`, `out/research/explore/I25_1/element/armA/pairs/pairs.json`, `out/research/explore/I25_1/element/armA/deploy/summary.json`, `out/research/explore/I25_1/element/armB/decision.json`, `out/research/explore/I25_1/element/armB/pairs/pairs.json`, `out/research/explore/I25_1/element/armB/deploy/summary.json`, `out/research/explore/I17_1/decision.json`.
- `02_table_element_validity.png`: Validity V1-V4 per arm; sources: `out/research/explore/I25_1/element/armA/decision.json`, `out/research/explore/I25_1/element/armA/pairs/pairs.json`, `out/research/explore/I25_1/element/armA/deploy/summary.json`, `out/research/explore/I25_1/element/armB/decision.json`, `out/research/explore/I25_1/element/armB/pairs/pairs.json`, `out/research/explore/I25_1/element/armB/deploy/summary.json`, `out/research/explore/I25_1/element/unit_tests.json`.
- `03_table_element_accuracy.png`: Element-deck RRMS (all-device / combined) vs stored Python rows; sources: `out/research/explore/I25_1/element/armA/pairs/pairs.json`, `out/research/explore/I25_1/element/armB/pairs/pairs.json`.
- `04_table_element_devices_armA.png`: Per-device checks, arm A; sources: `out/research/explore/I25_1/element/armA/deploy/q12.json`.
- `05_table_element_devices_armB.png`: Per-device checks, arm B; sources: `out/research/explore/I25_1/element/armB/deploy/q12.json`.
- `06_table_element_vds0.png`: V(d,s) = 0 current per tolerance; sources: `out/research/explore/I25_1/element/armA/decision.json`, `out/research/explore/I25_1/element/armA/pairs/pairs.json`, `out/research/explore/I25_1/element/armA/deploy/summary.json`, `out/research/explore/I25_1/element/armB/decision.json`, `out/research/explore/I25_1/element/armB/pairs/pairs.json`, `out/research/explore/I25_1/element/armB/deploy/summary.json`.
- `07_table_element_q3.png`: Q3 circuits; sources: `out/research/explore/I25_1/element/armA/deploy/q3.json`, `out/research/explore/I25_1/element/armB/deploy/q3.json`.
- `08_table_element_restrictions.png`: Restriction items; sources: `out/research/explore/I25_1/element/armA/deploy/restrictions.json`, `out/research/explore/I25_1/element/armB/deploy/restrictions.json`.
- `09_table_element_core_failures.png`: Core failures by device (diagnosis, reported); sources: `out/research/explore/I25_1/element/armA/diag_q1_nmos_L100_W100.json`, `out/research/explore/I25_1/element/armB/diag_q1_nmos_L100_W100.json`, `out/research/explore/I25_1/element/armB/diag_card_only_nmos_L20_W0p64.json`, `out/research/explore/I25_1/element/armB/deploy/q12.json`.

- `tables.md`: the tables above in markdown.
