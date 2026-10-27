# Report batch 2026-10-27 · i23_1

Generated 2026-10-27 09:19 by `.venv/bin/python -I scripts/research/plot_explore_i23_1.py --date 2026-10-27`. NGSpice runs: I-V re-simulation of the published / stock cards and the held-out cards; every re-simulated RRMS checked against the stored rows; 503 stored values recomputed.

## Contents

- I23-1 call-matched IHP control for B1 and fixed-lambda g v2. Pre-registration `out/research/explore/I23_1/prereg.md` (a9aa227). DEV only.
- Part A rule: B1 gain survives the call match.
- IHP SG13G2 233 K: GMR(B1, C0_call) 0.796 [0.661, 0.927]; GMR(C0_call, C0_refit) 0.998 [0.996, 0.999]; GMR(B1_cap, C0_refit) 0.977 [0.891, 1.035]
- IHP SG13G2 300 K: GMR(B1, C0_call) 0.776 [0.654, 0.905]; GMR(C0_call, C0_refit) 0.998 [0.997, 1.000]; GMR(B1_cap, C0_refit) 0.933 [0.868, 0.991]
- IHP SG13G2 343 K: GMR(B1, C0_call) 0.696 [0.580, 0.818]; GMR(C0_call, C0_refit) 0.999 [0.998, 1.000]; GMR(B1_cap, C0_refit) 0.944 [0.880, 1.011]
- IHP SG13G2 398 K: GMR(B1, C0_call) 0.811 [0.706, 0.916]; GMR(C0_call, C0_refit) 0.998 [0.996, 0.999]; GMR(B1_cap, C0_refit) 0.960 [0.899, 1.019]
- Part B rule: fixed lambda 1 V not shown non-inferior.
- sky130 77 K (18, LODO): GMR(fixed, selected) 0.988 [0.962, 1.014]; GMR(fixed, C0_refit) 0.734 [0.649, 0.841]
- NIST sky130 4 K lvt (53): GMR(fixed, selected) 1.018 [0.991, 1.044]; GMR(fixed, C0_refit) 0.817 [0.782, 0.860]
- NIST sky130 4 K RT card (87): GMR(fixed, selected) 1.030 [1.002, 1.058]; GMR(fixed, C0_refit) 0.835 [0.797, 0.883]
- IHP SG13G2 233 K: GMR(fixed, selected) 1.012 [1.000, 1.027]; GMR(fixed, C0_refit) 1.097 [0.905, 1.400]
- IHP SG13G2 300 K: GMR(fixed, selected) 0.963 [0.911, 1.000]; GMR(fixed, C0_refit) 1.209 [0.998, 1.536]
- IHP SG13G2 343 K: GMR(fixed, selected) 1.015 [1.004, 1.030]; GMR(fixed, C0_refit) 1.206 [0.995, 1.541]
- IHP SG13G2 398 K: GMR(fixed, selected) 1.004 [0.992, 1.015]; GMR(fixed, C0_refit) 1.265 [1.026, 1.617]
- TU Delft 4.13 K: GMR(fixed, selected) 1.070 [1.025, 1.137]; GMR(fixed, C0_refit) 0.884 [0.726, 1.077]
- TU Delft 295.0 K: GMR(fixed, selected) 1.030 [0.980, 1.094]; GMR(fixed, C0_refit) 0.854 [0.561, 1.237]
- Pooled vs C0_refit: fixed lambda 1 V 0.971, selected g v2 0.957.

## Figures and tables

- `01_table_i23_1_setup.png`: I23-1 setup (pre-registered); sources: `out/research/explore/I23_1/prereg.json`.
- `02_table_i23_1_a_validity.png`: Part A validity per temperature; sources: `out/research/explore/I23_1/decision.json`.
- `03_table_i23_1_a_rrms.png`: Part A RRMS per arm and temperature; sources: `out/research/explore/I23_1/decision.json`.
- `04_table_i23_1_a_gmr.png`: Part A GMR with CI and labels, rule outcome, pooled; sources: `out/research/explore/I23_1/decision.json`.
- `05_table_i23_1_a_calls.png`: Part A NGSpice calls per DUT; sources: `out/research/explore/I23_1/decision.json`.
- `06_table_i23_1_b_validity.png`: Part B row sources and validity; sources: `out/research/explore/I23_1/decision.json`, `out/research/explore/I23_1/verify_sky130.json`, `out/research/explore/I23_1/verify_ihp300.json`, `out/research/explore/I23_1/verify_tud4.json`, `out/research/explore/I23_1/verify_tud295.json`, `out/research/explore/I23_1/verify_nist4.json`.
- `07_table_i23_1_b_rrms.png`: Part B RRMS per arm and dataset; sources: `out/research/explore/I23_1/decision.json`.
- `08_table_i23_1_b_gmr.png`: Part B GMR with CI and labels, rule outcome, pooled; sources: `out/research/explore/I23_1/decision.json`.
- `09_table_i23_1_b_cost.png`: Part B NGSpice calls per held-out device next to the I22-1 cost table; sources: `out/research/explore/I23_1/decision.json`, `out/research/explore/I22_1/cost.json`.
- `10_i23_1_forest_a.png`: Part A forest plot: B1 vs call-matched C0_refit and reverse control per temperature; sources: `out/research/explore/I23_1/decision.json`.
- `11_i23_1_forest_b.png`: Part B forest plot: fixed lambda 1 V vs selected g v2, and both vs C0_refit, per dataset; sources: `out/research/explore/I23_1/decision.json`.
- `12_i23_1_cost_b.png`: Part B NGSpice calls per held-out device per arm and dataset; sources: `out/research/explore/I23_1/decision.json`, `out/research/explore/I22_1/cost.json`.
- `13_i23_1_per_device_a.png`: Part A per-DUT log ratios; sources: `out/research/explore/I23_1/decision.json`.
- `14_i23_1_per_device_b_sky130.png`: Part B sky130 77 K (18, LODO): per-device log ratios; sources: `out/research/explore/I23_1/decision.json`.
- `15_i23_1_per_device_b_nist4.png`: Part B NIST sky130 4 K lvt (53): per-device log ratios; sources: `out/research/explore/I23_1/decision.json`.
- `16_i23_1_per_device_b_nistrt.png`: Part B NIST sky130 4 K RT card (87): per-device log ratios; sources: `out/research/explore/I23_1/decision.json`.
- `17_i23_1_iv_a_ihp233_hv_pmos_W0p6_L0p5.png`: Part A I-V of hv_pmos_W0p6_L0p5 (IHP SG13G2 233 K; lowest ln(B1 / C0_call) -1.346): measured points, NGSpice stock card, re-simulated C0_refit, C0_call, B1 and B1_cap held-out cards; sources: `out/research/explore/I22_1/partA_ihp233/heldout_cards.json`, `out/research/explore/I23_1/partA_ihp233/heldout_cards.json`.
- `18_i23_1_iv_a_ihp233_hv_pmos_W10_L0p35.png`: Part A I-V of hv_pmos_W10_L0p35 (IHP SG13G2 233 K; highest ln(B1 / C0_call) +0.108): measured points, NGSpice stock card, re-simulated C0_refit, C0_call, B1 and B1_cap held-out cards; sources: `out/research/explore/I22_1/partA_ihp233/heldout_cards.json`, `out/research/explore/I23_1/partA_ihp233/heldout_cards.json`.
- `19_i23_1_iv_b_sky130_pmos_L8_W1p6.png`: Part B I-V of pmos_L8_W1p6 (sky130 77 K (18, LODO); highest ln(fixed lambda 1 V / selected) +0.146): measured points, NGSpice published / stock card, re-simulated C0_refit, selected g v2 and fixed lambda 1 V held-out cards; sources: `stored C0_refit (I22-1), selected g v2 and fixed lambda 1 V held-out cards and corrections`.
- `20_i23_1_iv_b_sky130_pmos_L4_W7.png`: Part B I-V of pmos_L4_W7 (sky130 77 K (18, LODO); lowest ln(fixed lambda 1 V / selected) -0.161): measured points, NGSpice published / stock card, re-simulated C0_refit, selected g v2 and fixed lambda 1 V held-out cards; sources: `stored C0_refit (I22-1), selected g v2 and fixed lambda 1 V held-out cards and corrections`.
- `21_i23_1_iv_b_nist4_pfet_01v8_lvt_W3_L1p5_m29p7.png`: Part B I-V of pfet_01v8_lvt_W3_L1p5_m29p7 (NIST sky130 4 K lvt (53); highest ln(fixed lambda 1 V / selected) +0.272): measured points, NGSpice published / stock card, re-simulated C0_refit, selected g v2 and fixed lambda 1 V held-out cards; sources: `stored C0_refit (I22-1), selected g v2 and fixed lambda 1 V held-out cards and corrections`.
- `22_i23_1_iv_b_nist4_pfet_01v8_lvt_W1_L20_m29p4.png`: Part B I-V of pfet_01v8_lvt_W1_L20_m29p4 (NIST sky130 4 K lvt (53); lowest ln(fixed lambda 1 V / selected) -0.309): measured points, NGSpice published / stock card, re-simulated C0_refit, selected g v2 and fixed lambda 1 V held-out cards; sources: `stored C0_refit (I22-1), selected g v2 and fixed lambda 1 V held-out cards and corrections`.
- `23_i23_1_iv_b_nistrt_nfet_01v8_W0p65_L4_m4p8.png`: Part B I-V of nfet_01v8_W0p65_L4_m4p8 (NIST sky130 4 K RT card (87); highest ln(fixed lambda 1 V / selected) +0.242): measured points, NGSpice published / stock card, re-simulated C0_refit, selected g v2 and fixed lambda 1 V held-out cards; sources: `stored C0_refit (I22-1), selected g v2 and fixed lambda 1 V held-out cards and corrections`.
- `24_i23_1_iv_b_nistrt_pfet_01v8_W5_L8_m18p7.png`: Part B I-V of pfet_01v8_W5_L8_m18p7 (NIST sky130 4 K RT card (87); lowest ln(fixed lambda 1 V / selected) -0.354): measured points, NGSpice published / stock card, re-simulated C0_refit, selected g v2 and fixed lambda 1 V held-out cards; sources: `stored C0_refit (I22-1), selected g v2 and fixed lambda 1 V held-out cards and corrections`.
- `25_i23_1_iv_b_ihp233_hv_pmos_W0p3_L10.png`: Part B I-V of hv_pmos_W0p3_L10 (IHP SG13G2 233 K; highest ln(fixed lambda 1 V / selected) +0.098): measured points, NGSpice published / stock card, re-simulated C0_refit, selected g v2 and fixed lambda 1 V held-out cards; sources: `stored C0_refit (I22-1), selected g v2 and fixed lambda 1 V held-out cards and corrections`.
- `26_i23_1_iv_b_tud4_t40_nmos_W0p12_L0p04_d158.png`: Part B I-V of t40_nmos_W0p12_L0p04_d158 (TU Delft 4.13 K; highest ln(fixed lambda 1 V / selected) +0.483): measured points, NGSpice published / stock card, re-simulated C0_refit, selected g v2 and fixed lambda 1 V held-out cards; sources: `stored C0_refit (I22-1), selected g v2 and fixed lambda 1 V held-out cards and corrections`.

- `tables.md`: the tables above in markdown.
