# Report batch 2026-10-16 · dev_iteration1

Generated 2026-10-16 22:07 by `scripts/research/report_figs.py --topic dev_iteration1 --date 2026-10-16` from result files only (no simulation). RRMS is the paper metric with frozen published-baseline curve inclusion, re-simulated in NGSpice, unless an item says otherwise.

## Figures and tables

- `01_compare_curves.png`: Copy of the DEV comparison figure: best-so-far paper RRMS against cumulative NGSpice calls per target on the 36 DEV targets, with the all-device mean (solid line) and median (dashed line) of the arms hybrid_surrogate, hybrid_none, cmaes_trf, lq_cmaes, sketch_surrogate_k8, sketch_none_k8, sketch_surrogate_k4, sketch_surrogate_k16, the fixed-budget run means as dots, and the bank-oracle, pipeline r1 and pipeline r1 + FD references as open markers (the arms sketch_fullfd_surrogate, sketch_fullfd_none, anytime_sur_presearch_equal, anytime_sur_presearch_race, anytime_none_race are not drawn: dev_compare.py's plot has 8 colours and stops after the 8th arm); sources: `out/research/dev/compare_curves.png`, `out/research/dev/compare.json`.
- `02_table_dev_primary.png`: Table of the pooled all-device mean paper RRMS of every DEV arm at 500, 1000, 2000, 5000 NGSpice calls per target, with AUC, median calls to RRMS 0.05 and 0.01 and failed targets per budget, plus the reference values in the notes; sources: `out/research/dev/compare.json`, `out/research/dev/compare.md`.
- `03_table_dev_paired.png`: Table of paired per-target differences A − B at each budget for sketch_surrogate_k8 vs sketch_none_k8; hybrid_surrogate vs hybrid_none; anytime_sur_presearch_race vs anytime_none_race; sketch_surrogate_k8 vs cmaes_trf; sketch_none_k8 vs hybrid_none, with the mean difference, the win rate of A and their bootstrap 95% CIs over the 36 DEV targets; sources: `out/research/dev/compare.json`.
- `04_table_audit.png`: Table of the failed-curve audit's headline numbers, each as stored, as the mean over valid devices only and in the penalized convention (failed included curves at RRMS 10), all-device and combined, with the failed-device count and names; sources: `out/research/audit/failed_curves.json`, `out/research/audit/failed_curves.md`.

- `tables.md`: the tables above in markdown.

## Notes

- compare_curves.png draws only the first 8 arms of compare.json; not drawn: sketch_fullfd_surrogate, sketch_fullfd_none, anytime_sur_presearch_equal, anytime_sur_presearch_race, anytime_none_race

## Skipped inputs

- none

## Result files present but not used in this batch

- none

- `05_dev_families.png`: anytime all-device mean RRMS vs NGSpice calls for each method family (full Jacobian, sketched k = 8, racing), surrogate arm solid, NGSpice-only control dashed, CMA-ES + TRF grey; from `out/research/dev/compare.json` via `scripts/research/plot_dev_families.py`.
