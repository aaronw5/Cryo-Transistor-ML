# Report batch 2026-10-28 · i25_1

Script `scripts/research/plot_explore_i25_1.py`. Curves are the NGSpice re-simulations stored by `i25_1_scratch.py --stage final/old` (each reproduces its stored RRMS); every plotted RRMS recomputed from them and checked against decision.json. I-V figures of all 18 devices for both arms in `iv/iv_arm{A,B}_<device>.png` (measured points, paper card NGSpice, from-scratch card fit NGSpice, from-scratch card fit + g NGSpice x exp(g); per-curve RRMS in the panel titles). Deck `slides/i25_1_from_scratch.pptx` (not in git).

## Contents

- `01_table_main_primary.png`: Main table, primary curve set: stock card at 77 K, multiplier-1 card, paper card, 7p and screened-set card fits without and with g, full data and LODO, NGSpice calls per device; sources: out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json.
- `02_table_main_primary_full.png`: Main table, primary curve set, full data only (slide version); sources: out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json.
- `03_table_main_m_rule.png`: Main table, secondary M-rule curve set; sources: out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json.
- `04_table_stages.png`: RRMS after every round of the selected start, 7p and screened-set chains vs the old 43p chain, then C0_refit and g; sources: out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json.
- `05_table_multistart.png`: Spread over the 3 random starts per device (min / median / max); sources: out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json.
- `06_fig_stages.png`: RRMS after every round (log scale); sources: out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json.
- `07_table_gmr.png`: GMR with 95% device-bootstrap CI and pre-registered labels, both curve sets; sources: out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json.
- `08_fig_forest.png`: Forest plot of the primary GMRs; sources: out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json.
- `09_fig_per_device.png`: Per-device RRMS: multiplier-1 card, paper, card fit, card fit + g (full data), both arms; sources: out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json.
- `10_table_per_device_armA.png`: Per-device RRMS, arm A (7 parameters); sources: out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json.
- `11_table_per_device_armB.png`: Per-device RRMS, arm B (screened set); sources: out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json.
- `12_table_parameter_set.png`: Screened set around the multiplier-1 card vs the earlier 43; sources: out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json.
