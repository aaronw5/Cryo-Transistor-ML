# I18-2 pre-registration (IN FORCE): second BSIM4 dataset for the deployable element; selected g v2 recipe on measured TU Delft 40-nm MOSFETs (I6-1 data), 4.13 K and 295.0 K, 45 DEV devices, leave-one-geometry-group-out; Part B deployment checks (I17-1 rules) with ngspice-41 OSDI

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-22), committed alone, before any I18-2 code, simulation, training or refit. The I18-1
pre-registration (commit b384a42) uses the TU Delft selected rows of this study as its prospective datasets.

## Data, card, simulator

- I6-1 data (doi:10.4121/c41c26f7-586f-48ca-8abe-9127b2d97c60.v1, CC0), campaign 1, `I6_1/split.json`: DEV 45 devices
  in 7 geometry groups (nMOS W/L 0.12/0.04, 0.12/0.05, 0.36/0.04, 1/0.04, 1/1; pMOS 1/0.04, 1/1); TEST 24 devices
  stay locked (not loaded). Temperatures 4.13 K and 295.0 K (stage readings; deck temperature = stage reading). I6-1
  frozen measured inclusion; strict paper RRMS (`i6_1_tudelft.score`).
- Card: BSIM4 4.5.0 N1 / P1 (CMOSedu 50 nm, from ngspice-41 `examples/mos/ro_17_4.cir`, temperature coefficients 0),
  43-parameter fitted set and box of I6-1. Simulator: ngspice-41 native arm64 build (I6-1), timeout 10 s per process.
- **BSIM4 clamp state C (change from I6-1).** ngspice-41 `b4check.c` clamps a2 > 1 to 1 (a1 to 0) and a2 < 0.01 to
  0.01 once per model name and process; the I6-1 decks (one `dc` per curve in one process) ran the first curve with
  the clamped value and later curves with the card value (state L). Known before registration: the I6-1 starting
  cards have a2 > 1 on 25 / 45 devices at 4.13 K and 20 / 45 at 295.0 K (a2 box 0.1 .. 10; the card has no
  la1 / wa1 / pa1 / la2 / wa2 / pa2 terms, a1 = 0). Every I18-2 deck writes the clamp into the card (a2 > 1 -> a2 = 1,
  a1 = 0 if nonzero; a2 < 0.01 -> 0.01), so every curve is simulated in the fresh-process state (I13-1 state C).
  Cards with a2 in [0.01, 1] give identical decks. Other `b4check.c` messages of the starting cards are listed by an
  audit stage (fresh-process `op`, no clamp written); reported.
- Starting cards: the I6-1 starting parameters (not refit); starting records re-simulated in state C at both
  temperatures.

## Arms (leave-one-geometry-group-out, 7 folds per temperature)

Held-out group G; training = the records of the other 6 groups, both polarities, in record load order (as I6-1).
Held-out refit for every arm: polish from the starting card with g frozen in the loop (max_nfev 400, irls 2,
diff_step 0.01; I6-1 options), state C.

| arm | definition |
|---|---|
| C0_refit | g = 0 |
| M_bias v1 | I3-2 recipe (`i3_2_bias.fit_bias`, inputs is_pmos, \|V(g,s)\| / VDD, \|V(d,s)\| / VDD, VDD 1.1 V as I6-1; hidden 16, 2 layers, l2 1e-4, lr 3e-3, 2000 steps, seeds 0-4) |
| sym_e1_l{lambda}, lambda in {0, 1, 2, 5, 10} V | g v2 recipe (`i12_1_symm.fit_v2`): x1 = (s V(g,s) + s V(g,d) + m) / (2 VDD), x2 = m / VDD, m = sqrt(V(d,s)^2 + EPS^2), EPS = 0.01 V, VDD = 1.1 V; same hyperparameters and seeds; loss + lambda P |
| selected | per outer fold the row of sym_e1_l{lambda*_G} |

Penalty P (I12-1 definition, `penalty_arrays`): mean over training devices of the mean over grid points of
relu(-(gm/I + dg/dV(g,s))) + relu(-(gds/I + dg/dV(d,s))). Grid per device: the state-C starting card at the dataset
temperature on the forward quadrant u_g = s V(g,s), u_d = s V(d,s) in {0, 0.025, ..., 1.1} V (45 x 45; step 25 mV =
the sky130 step); points with u_d > 0 and forward current > 1e-9 A; central differences.

Nested selection of lambda (I15-2 rule at the fold unit): per outer fold the 6 training groups sorted by (polarity,
nMOS first; group name) go to inner fold i mod 3 (2 groups each); for each inner fold and lambda one member (seed 0)
is trained on the other training groups and every device of the inner fold is refit with it; lambda* = argmin over
lambda of the mean ln RRMS of the inner held-out refits (non-finite -> ln 1 = 0); ties within 1e-12 to the larger
lambda. Full-data model (Part B): lambda*_full by the same selection over all 7 groups (sorted, i mod 3), then 5
seeds at lambda*_full on all 45 devices.

## Rules (per temperature)

Primary: GMR(selected, C0_refit), cluster bootstrap over the 7 geometry groups (devices of a group are copies of one
layout and share the outer fold), 10,000 resamples, seed 0, 95% CI (`i3_2_bias.paired_ratio` with clusters).

| condition | primary label |
|---|---|
| CI upper < 1 and 0 failed devices in selected and C0_refit | gain over C0_refit replicated |
| CI lower > 1 | worse than C0_refit |
| otherwise | inconclusive |

Secondary: GMR(selected, M_bias v1), cluster bootstrap: CI upper <= 1.05 -> non-inferior to M_bias v1; otherwise not
shown non-inferior.

Reported without a rule: every arm's all-device / combined / median; GMR of every arm vs C0_refit and vs v1 with
cluster and device CIs; per group; lambda* counts; negative corrected gm / gds fractions on the held-out and
training grids; state-C rows vs the I6-1 stored state-L rows; the I6-1 contrast K (4.13 K ratio over 295.0 K ratio)
for selected vs C0_refit.

## Part B (per temperature; I17-1 rules)

Model under test: the full-data g v2 at lambda*_full(T). M2 cards: every DEV device's starting card refit with the
full-data g in the loop (state C, max_nfev 400, irls 2). Element (BSIM4 channel replica, I14-1 construction): card
instance M (model `i6card`, w, l, nf = 1, the M2 parameters, clamp written); replica instance with model `i6card_ch`
= the same card with REPLICA_ZERO (xpart -1; cgso, cgdo, cgbo, cgsl, cgdl, cf, cjs, cjd, cjsws, cjswd, cjswgs,
cjswgd = 0) and REPLICA_CHAN (alpha0 = alpha1 = beta0 = 0, agidl = agisl = 0, jss = jsd = jsws = jswd = jswgs =
jswgd = 0, igcmod = igbmod = 0; the card has no l / w / p terms of these), driven by unit-gain VCVS copies of
V(d), V(g), V(s), V(b), a VCCS of -gmin (1e-12 S) between replica drain and body copies, and the OSDI module
(`i12_1_symm.to_verilog_a_v2`, OpenVAF 23.5, vdd 1.1 V in the module text) adding (exp(g) - 1) I_rep from d to s.
Card-specific notes: the TU Delft card has igcmod = igbmod = 1 (gate tunneling), rgatemod = 1, rbodymod = 1, nonzero
agidl, alpha0, beta0 and junction saturation currents; the replica switches the DC non-channel currents off as
listed. Replica pre-check (reported): the replica's BSIM4 outputs isub, igidl, igisl, igs, igd, igb, igcs, igcd are
0 on every M2 card at the Q2 op points.

Benches and checks (I14-1 / I15-1 / I17-1 definitions; values scaled to VDD 1.1 V):

- Q1 grids: forward (d, g, s, b) = (nd, ng, 0, 0) and exchanged bench; V(g,s) 0 .. s 1.1 V (45 values, 25 mV),
  V(d,s) -1.1 .. 1.1 V (89 values), serpentine sweep (I15-1); tolerances default, r4 (reltol 1e-4), tight (reltol
  1e-9, abstol 1e-18 A, vntol 1e-12 V); modes card, card + replica, element. Core: equivalence |dI| <= 1e-6 |I| +
  1e-13 A at tight tolerance (forward and exchanged; reference I_card + (exp(g) - 1) I_rep from the card + replica
  run); 0 convergence messages / non-finite values / failed runs in element runs; |I(V(d,s) = 0)| <= 1e-12 A on the
  tight forward grid only (I17-1); Verilog-A g vs Python <= 1e-9, exchanged vs forward <= 1e-12; exchange asymmetry
  <= 1e-3 (|I| > 1 nA, tight). Restriction: default-tolerance relative difference > 1e-3 (|I| > 1 nA).
- Q2 (tight, 1 MHz): op points gm sets |V(d,s)| in {0.05, 0.55, 1.1} V x |V(g,s)| on the grid, gds sets |V(g,s)| in
  {0.35, 0.525, 0.725, 0.9, 1.1} V x |V(d,s)| on the grid. Core: derivative error vs the element identity Re Y_card +
  (exp(g) - 1) Re Y_rep + I_rep exp(g) dg/dV <= 1e-3 (normalized, |I| > 1 nA); capacitance error (Im Y_dd, Im Y_dg,
  Im Y_gg vs card) <= 1e-3; 0 gm curves with an introduced sign change in strong inversion (|V(g,s)| >= Vt + 0.2 V,
  Vt at 1e-7 A x W / L). Restriction: gds curves with an introduced sign change (|V(d,s)| >= 0.05 V).
- Q3 circuits: **not run**. The sky130 circuit decks (inverters, 5-stage ring, current mirrors) run at a 1.8 V supply;
  the TU Delft card is a 1 V card (CMOSedu "VDD=1V") and the measured range ends at 1.1 V, so the supply is not
  compatible. The Q3 core check is recorded as not evaluated (count 0, labelled "Q3 not run").
- Q4 runtime: element / card wall-time ratio of the default-tolerance forward DC grid (3 repeats); restriction if >
  2. No circuit runtime (Q3 not run).
- Reported: element-deck RRMS of the 45 M2 in-sample pairs and of the 45 selected held-out pairs (fold g + held-out
  card) vs the Python rows, per temperature.

Part B label per temperature (`i12_1_symm.decide_deploy` with the I17-1 V(d,s) = 0 value): not deployable (any core
check fails) / deployable with restrictions / deployable (simulator-level); every label carries "Q3 not run".

## Validity (checked first; any fail -> invalid)

| check | pass |
|---|---|
| V1 code path | (a) state-L re-simulation of the 90 I6-1 starting cards reproduces the stored I6-1 starting records (max \|RRMS difference\| <= 1e-9); (b) state-C records equal state-L records on every card with a2 in [0.01, 1] (difference 0); (c) state-L held-out refits of C0_refit and M_bias v1 (stored I6-1 fold correction) for t40_nmos_W1_L1_d130 and t40_pmos_W1_L1_d330 at 4.13 K reproduce the I6-1 rows (\|difference\| <= 1e-9) |
| V2 unit tests | `tests.test_i18_2_tud`, `tests.test_i12_1_symm`, `tests.test_i15_2_ablate` pass |
| V3 rows | 45 rows in every arm at both temperatures; inner selection complete (7 x 3 x 5 trained per temperature, every inner held-out device scored; full-data selection complete) |
| V4 grids | 90 penalty grids 45 x 45 finite; forward current > 0 at u_g = u_d = 1.1 V |
| VB Part B | 0 failed M2 refits (45 per temperature); Part B unit tests pass; 45 Part B rows per temperature |

## Decision

| condition | outcome |
|---|---|
| any validity check fails | invalid (stop: fix and re-register) |
| valid | per temperature: primary label + secondary label + Part B label (Q3 not run) |

## Predictions (stated before any run; not rules)

| quantity | 4.13 K | 295.0 K | basis |
|---|---|---|---|
| ln GMR(selected, C0_refit) | < 0 | < 0 | I6-1 GMR(M_bias v1, C0_refit) 0.939 / 0.797 (state L) |
| ln GMR(selected, M_bias v1) | > 0 | > 0 | sky130 0.997 and IHP 1.034 (point estimates near or above 1; penalty constrains g) |
| ln GMR(M_bias v1, C0_refit), state C | < 0 | < 0 | I6-1 |
| primary label | inconclusive | gain over C0_refit replicated | 7-cluster CIs; I6-1 cluster CIs [0.811, 1.081] / [0.576, 1.087] |
| Part B label | deployable with restrictions (Q3 not run) | deployable with restrictions (Q3 not run) | I17-1 sky130 |

## Order and compute

This prereg (alone) -> code + unit tests -> smoke (2 groups, 1 device each, 200 steps, 1 seed, refit max_nfev 20;
Part B on 2 devices; not used for decisions) -> queue, one heavy job at a time: I18_2_start (state-C records, audit,
V1 a/b), I18_2_pens, I18_2_base_{4,295} (C0_refit + M_bias v1 + V1 c), I18_2_lodo_{T}_l{0,1,2,5,10},
I18_2_inner_{4,295}, I18_2_full_{4,295}, I18_2_decide, I18_2_partB_{4,295}, I18_2_decideB. Estimated 4-6 h.

## Outputs

`out/research/explore/I18_2/{start, pens, base_*, lodo_*, inner_*, full_*, partB_*, decision.json}`; arrays and the
measured dataset not in git; report batch `out/research/report/<date>_i18_2/` (tables vs rules, per-device plots,
I-V figures with measured points, stock-card NGSpice and re-simulated C0_refit / v1 / selected cards), site claim
`i18-2`, RESEARCH_LOOP entry.
