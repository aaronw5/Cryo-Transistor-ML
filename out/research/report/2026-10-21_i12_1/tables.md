# Tables, 2026-10-21 i12_1

Generated 2026-10-21 23:46 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i12_1.py --date 2026-10-21`.

### I12-1 LODO RRMS per arm and the accuracy contrasts

| arm | all-device | combined | median | nMOS mean | pMOS mean | GMR vs C0_refit [95% CI], wins |
|---|---:|---:|---:|---:|---:|---:|
| M_bias v2 (I12-1) | 0.0774 | 0.0732 | 0.0534 | 0.0353 | 0.1110 | 0.787 [0.703, 0.884], 16/18 |
| M_bias v1 (I3-2 all18) | 0.0732 | 0.0691 | 0.0643 | 0.0326 | 0.1056 | 0.755 [0.673, 0.852], 16/18 |
| T1_inloop (per-tag gains) | 0.0744 | 0.0703 | 0.0672 | 0.0330 | 0.1075 | — |
| C0_refit (no correction) | 0.1043 | 0.0975 | 0.0820 | 0.0362 | 0.1587 | — |
| published 77 K card | 0.2751 | 0.2595 | 0.2304 | 0.1198 | 0.3992 | — |

- LODO over 18 sky130 77 K devices; strict paper RRMS (curve inclusion frozen to the published card), held-out card refit with the fold g in the loop (max_nfev 400, irls 2). M_bias v1, T1_inloop, C0_refit: stored I3-2 all18 rows. published: upstream released card (reference_scores.json).
- combined = equal-weight mean of the nMOS (8) and pMOS (10) family means; all-device = mean over 18.
- A1 GMR(v2, C0_refit) 0.787 [0.703, 0.884], 16/18 lower: CI upper < 1, passes. A2 GMR(v2, v1) 1.042 [0.967, 1.111], 5/18 lower: CI upper > 1.05, non-inferiority fails. Reported: GMR(v2, T1_inloop) 1.015 [0.930, 1.096], 5/18.
- Outcome (pre-registered rule, prereg commit 3619a2b): not deployable; action: stop; diagnose the failed core checks.


### I12-1 part B deployment checks: values vs thresholds

| check | value | threshold | result | devices over threshold |
|---|---:|---:|---|---|
| Q1 equivalence, tight, fwd + exchanged (core) | 0 points | 0 points with |dI| > 1e-6|I| + 1e-13 A | pass | — |
| Q1 convergence msgs + non-finite (core) | 0 | 0 | pass | — |
| Q1 max |I_elem(Vds = 0)| (core) | 9.63e-13 A | <= 1e-12 A | pass | — |
| Verilog-A g text vs Python (core) | 2.22e-16 | <= 1e-9 | pass | — |
| Verilog-A g exchanged vs forward (core) | 1.67e-16 | <= 1e-12 | pass | — |
| exchange asymmetry, element-introduced |d ln I| (core) | 1.35e-02 | <= 1e-3 | fail | nmos_L100_W100, nmos_L20_W0p64 |
| Q2 normalized gm / gds difference (core) | 2.31e+00 | <= 1e-3 | fail | nmos_L0p19_W7, nmos_L0p25_W1p6, nmos_L100_W100, nmos_L20_W0p64, nmos_L8_W1p6 |
| Q2 capacitance error Y_dd, Y_dg, Y_gg (core) | 1.77e-03 | <= 1e-3 | fail | nmos_L0p19_W7, nmos_L20_W0p64 |
| Q2 gm curves with introduced sign change (core) | 0 of 54 | 0 | pass | — |
| Q3 circuits failing only with the element (core) | 0 | 0 of 6 | pass | — |
| Q1 default-tolerance max rel. diff. (restriction) | 1.80e-03 | <= 1e-3 | restriction | 13 of 18 devices |
| Q2 gds curves with introduced sign change (restriction) | 2 of 90 | 0 | restriction | pmos_L8_W0p84, pmos_L8_W1p6 |
| Q4 max wall-time ratio element / card (restriction) | 4.72 | <= 2 | restriction | ro5_A 4.72; DC grid 1.91-2.00 |

- Deployment label (pre-registered rule): not deployable. Core fails: terminal-exchange asymmetry > 1e-3; Q2 derivative agreement; Q2 capacitances differ from the card. Restrictions: default tolerance above 1e-3; gds sign changes introduced; runtime overhead > 2x.
- M2 card per device (43-parameter card refit with the full-data g v2 in the loop, in-sample) + full-data g v2 as OSDI element (OpenVAF 23.5.0) sensing a charge-free replica of the card; ngspice-41 OSDI build. tight = reltol 1e-9, abstol 1e-18, vntol 1e-12. ro5_B excluded from Q4 (fails at card level, amendment A1).
- Smoke (nmos_L1_W1p6, pmos_L2_W5) passed every core check; the full run adds 16 devices. Post hoc diagnosis: table 05.


### Post hoc diagnosis of the three failed core checks

| check | registered value | devices over 1e-3 | source of failure | cause | post hoc recomputed |
|---|---:|---|---|---|---|
| Q2 capacitances | 1.77e-03 | nmos_L0p19_W7, nmos_L20_W0p64 | reference run | BSIM4 A2 > 1 clamp state differs (card-only vs element deck) | matched state: 9.9e-13 |
| Q2 derivatives | 2.31e+00 | L0p19_W7, L0p25_W1p6, L20_W0p64, L8_W1p6 (nmos) | reference run | same A2 clamp state; card op current differs by up to 0.183 | matched state: 2.8e-06 |
| Q2 derivatives | 1.67e-02 | nmos_L100_W100 | reference formula | Re Y_card at 1 MHz != static conductance (rbodymod = 1) | element identity: 6.6e-06 (18 devices) |
| exchange asymmetry | 1.35e-02 | nmos_L100_W100, nmos_L20_W0p64 | element | impact-ionization current in the sensed replica drain current | predicted 0.0136, 0.0065 (observed 0.0135, 0.0065) |

- Post hoc (after decision.json); the registered outcome 'not deployable' is unchanged. Source: scripts/research/explore/i12_1_diagnosis.py (about 18 s of single-process ngspice runs).
- A2 clamp state: ngspice-41 b4check.c checks each model name once per process (static modname) and clamps the size-dependent A2 > 1 to 1 (A1 to 0); b4temp.c rebuilds the size-dependent parameters at every analysis. The card-only AC deck (600 x op + AC in one process) therefore uses A2 = 1 only in its first analysis; in the element deck the card and replica model names alternate, the check runs every time and A2 = 1 throughout. M2 card a2: nmos_L0p19_W7 1.888, nmos_L0p25_W1p6 1.042, nmos_L20_W0p64 2.003, nmos_L8_W1p6 1.030; single-parameter search on nmos_L0p19_W7: a2 alone produces the drift, removing a2 removes it. Matched check state: the card-only AC reference rerun with the replica instance in the deck (VCVS-driven, current through a 0 V source, no OSDI module).
- nmos_L100_W100: Re Y_dg at Vgs 0, Vds 0.05 V is 5.7972e-07 S at 1 MHz, 4.6987e-07 S at 100 Hz, and 4.6986e-07 S at 1 MHz with rbodymod = 0 (body resistor network x 100 um x 100 um capacitances). The registered reference exp(g) Re Y_card + I exp(g) dg/dV assumes Re Y_card is the static conductance; the element adds (exp(g) - 1) x the replica's static conductance. With that identity (matched state) the max difference over 18 devices is 6.6e-06.
- Exchange asymmetry: nmos_L100_W100 max at Vgs 0.00 V, Vds 1.50 V, I 7.10e-07 A; drain non-channel current J = |I_fwd,card| - |i(VZ)| = 1.820e-07 A (J/I 0.26); setting alpha0 = 0 lowers the card current by 1.820e-07 A. J rises with Vgs and Vds (impact ionization, drain to body). The replica drain-terminal current carries it in the forward bench only; the pre-registered bound assumed J = gmin x V <= 1.85e-12 A.
- The A2 clamp state also applies to every card-only multi-analysis deck, including the measured-curve batch deck (one dc analysis per curve): for cards with a2 > 1 the first curve uses A2 = 1 and the other curves the card value. Second-analysis drift > 1e-6 (M2 cards): nmos_L0p19_W7, nmos_L0p25_W1p6, nmos_L20_W0p64, nmos_L8_W1p6; the starting cards (direct43_r3_fd) show the same drift on the same devices; published cards 0.
- Smoke devices (nmos_L1_W1p6, pmos_L2_W5): a2 <= 1, J/I <= 1e-3, Re Y_card at 1 MHz equal to the static value.

