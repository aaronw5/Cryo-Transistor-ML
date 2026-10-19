# Report batch 2026-10-19 · dev_iteration2_ablation

Generated 2026-10-19 09:18 by `scripts/research/report_figs.py --topic dev_iteration2_ablation --date 2026-10-19` from result files only (no simulation). RRMS is the paper metric with frozen published-baseline curve inclusion, re-simulated in NGSpice, unless an item says otherwise.

## Scope

Ablation report of DEV iteration 2. DEV = simulated targets set_0 and set_1 (18 devices each, 36 targets); TEST (set_2 and the 18 measured devices) is not used. Tier a: well-specified targets (out/research/sim_targets). Tier c: misspecified sky130c targets (out/research/sim_targets_sky130c, strength 2), scored as excess over the per-target refit floor. No per-target or cross-arm minimum is taken.

## Pre-registered λ decision (n2b)

Status: decided. Result: negative. Frozen candidate per family: sketch family: none; hybrid family: none. TEST run: none (negative result).

Rule: an arm is frozen when the upper end of its 95% CI meets all of A1-A4; order lam_curve_k8, lam_scalar_k8, lam_cusum_k8 (sketch family) and hyb_lam_curve (hybrid family); lam_radius_k8 is reported, not selectable. CI: geometry_clustered (dev_stats.compare ci95_clustered).

sketch family controls: surrogate = sketch_surrogate_k8, none = sketch_none_k8 (parity exact: iteration-1 controls); candidate before the validity rule: none.

hybrid family controls: surrogate = hybrid_surrogate, none = hybrid_none (parity exact: iteration-1 controls); candidate before the validity rule: none.

Hard checks: 0 FAIL rows of 1739; 11 WARN rows for batched-probe score gaps (see README).

Pre-registration: lambda_tr.py PRE_REGISTERED (n2b, rules A1-A4) and out/research/dev/prereg_iter2.json (n2c, comparisons C1-C6, Holm over 36 tests), committed 2026-10-17T02:41:55-04:00.

## Pre-registered C1-C6 (n2c)

36 of 36 tests computed; 12 Holm-adjusted p ≤ 0.05: C1_a_B5000, C2_a_B500, C3_a_B5000, C5_a_B500, C5_a_B2000, C5_a_B5000, C5_c_B500, C5_c_B2000, C5_c_B5000, C6_a_B500, C6_c_B500, C6_c_B2000. Per-test values are in the paired tables (rows with a pre-registered id) and in out/research/dev/stats_iter2/stats.md.

## Check change in this batch

The first runs of the seven comparison jobs (2026-10-17) exited 1 for two reasons. (1) The registries listed the seed-1/2 replicate arms (sketch_surrogate_k8__seed1/2, sketch_none_k8__seed1/2) before their runs existed; the runs finished later the same day and are present now. (2) dev_compare's check 'fresh rrms == best_at(trace)' failed on 14 tier-c records, all pmos_L8_W0p84, with relative gaps 2.5e-5 to 4.6e-5: the in-loop best was a Jacobian probe simulated in a multi-card altermod NGSpice process after a non-converged card in the same process, and its fresh single-process score differs. A reproduction of lam_scalar_k8 set_0 B1000 shows the gap at call 225 (batch of 8, cards 0, 4, 5 failed, best = card 3: in-loop 0.0870815702, fresh 0.0870854113); every new best from a batch without failed cards matches exactly.

dev_compare.py now reports such a gap as a WARN row when the record's best_kind is a batched probe and the relative gap is at most 1e-4; any other gap remains a FAIL. The fresh rrms is the value used in every table. lambda_tr.py decide lists these WARN rows. Under the pre-registered validity rule these rows had withdrawn the sketch-family candidate; no arm passed A1-A4 before or after the change, so the decision outcome does not depend on it.

## Figures and tables

- `01_anytime_families_tier_a.png`: Anytime curves, tier a: well-specified simulated targets (sim_targets set_0, set_1): the all-device mean of the best-so-far paper RRMS against cumulative NGSpice calls per target from the largest-budget traces, one panel per method family (sketched k = 8 with the λ arms, full-Jacobian hybrid TR, racing, baselines); in each panel the λ = 1 surrogate arm is blue solid, the λ = 0 NGSpice-only control orange dashed, CMA-ES + TRF and DFO-LS-style are drawn in light grey as shared baselines; seed-replicated arms show the mean of their seed curves; dots = the fixed-budget fresh NGSpice values (seed mean) at B500/B1000/B2000/B5000; arms drawn: sketch_surrogate_k8, sketch_none_k8, lam_curve_k8, lam_scalar_k8, lam_cusum_k8, lam_radius_k8, hybrid_surrogate, hybrid_none, hyb_lam_curve, anytime_sur_presearch_race, anytime_none_race, dfols_style, dfols_style_1start, cmaes_trf, lq_cmaes; sources: `out/research/dev/compare_iter2/compare.json`.
- `02_anytime_families_tier_c.png`: Anytime curves, tier c: misspecified sky130c targets (set_0, set_1), excess over the per-target floor: the mean over targets of best-so-far RRMS minus the target's floor (compare.json mean curve minus the mean floor 0.0478 of the 36 targets, exact because the floor is constant per target) against cumulative NGSpice calls per target from the largest-budget traces, one panel per method family (sketched k = 8 with the λ arms, full-Jacobian hybrid TR, racing, baselines); in each panel the λ = 1 surrogate arm is blue solid, the λ = 0 NGSpice-only control orange dashed, CMA-ES + TRF and DFO-LS-style are drawn in light grey as shared baselines; seed-replicated arms show the mean of their seed curves; dots = the fixed-budget fresh NGSpice values (seed mean) at B500/B1000/B2000/B5000; arms drawn: sketch_surrogate_k8, sketch_none_k8, lam_curve_k8, lam_scalar_k8, lam_cusum_k8, lam_radius_k8, hybrid_surrogate, hybrid_none, hyb_lam_curve, anytime_sur_presearch_race, anytime_none_race, dfols_style, dfols_style_1start, cmaes_trf, lq_cmaes; sources: `out/research/dev_sky130c/compare/compare.json`.
- `03_table_primary_tier_a.png`: Table of the pooled mean paper RRMS (strict) of every iteration-2 arm on tier a at B500/B1000/B2000/B5000, seed counts and failed targets; per-target values through dev_stats (dev_compare hard checks); sources: `out/research/dev/arms_iter2.json`.
- `04_table_primary_tier_c.png`: Table of the pooled mean excess over floor (RRMS − floor) of every iteration-2 arm on tier c at B500/B1000/B2000/B5000 with raw RRMS at B500 and B5000, seed counts and failed targets; per-target values through dev_stats (dev_compare hard checks); sources: `out/research/dev_sky130c/arms.json`, `out/research/sim_targets_sky130c/floors.json`.
- `05_table_paired_lambda.png`: Table of paired differences A − B at B500/B2000/B5000 on both tiers for λ arms vs λ = 0 and λ = 1 (same code, surrogate weight varied): lam_curve_k8 − sketch_none_k8; lam_curve_k8 − sketch_surrogate_k8; lam_scalar_k8 − sketch_none_k8; lam_scalar_k8 − sketch_surrogate_k8; lam_cusum_k8 − sketch_none_k8; lam_cusum_k8 − sketch_surrogate_k8; lam_radius_k8 − sketch_none_k8; lam_radius_k8 − sketch_surrogate_k8; hyb_lam_curve − hybrid_none; hyb_lam_curve − hybrid_surrogate, with geometry-clustered 95% CIs, Wilcoxon p (Holm p for the pre-registered C1-C6 rows) and W/T/L; sources: `out/research/dev/arms_iter2.json`, `out/research/dev_sky130c/arms.json`, `out/research/dev/stats_iter2/stats.json`.
- `06_table_paired_surrogate.png`: Table of paired differences A − B at B500/B2000/B5000 on both tiers for surrogate term vs NGSpice-only control, same code: sketch_surrogate_k8 − sketch_none_k8; hybrid_surrogate − hybrid_none; anytime_sur_presearch_race − anytime_none_race; sketch_fullfd_surrogate − sketch_fullfd_none, with geometry-clustered 95% CIs, Wilcoxon p (Holm p for the pre-registered C1-C6 rows) and W/T/L; sources: `out/research/dev/arms_iter2.json`, `out/research/dev_sky130c/arms.json`, `out/research/dev/stats_iter2/stats.json`.
- `07_table_paired_dfols.png`: Table of paired differences A − B at B500/B2000/B5000 on both tiers for DFO-LS-style baseline vs the other arms: dfols_style − sketch_none_k8; dfols_style − sketch_surrogate_k8; lam_curve_k8 − dfols_style; dfols_style − hybrid_none; dfols_style − cmaes_trf; dfols_style − dfols_style_1start, with geometry-clustered 95% CIs, Wilcoxon p (Holm p for the pre-registered C1-C6 rows) and W/T/L; sources: `out/research/dev/arms_iter2.json`, `out/research/dev_sky130c/arms.json`, `out/research/dev/stats_iter2/stats.json`.
- `08_table_lambda_decision.png`: Table of the pre-registered n2b λ decision: rules A1-A4 for each evaluated λ arm and the reported ablation arms, with mean difference, clustered 95% CI, required bound and pass/fail, and the outcome in the notes; sources: `out/research/dev/lambda_decision.json`.

- `tables.md`: the tables above in markdown.

## Skipped inputs

- none

## Result files present but not used in this batch

- `out/research/dev_sky130c/compare_floor_lambda/compare_floor.json`
