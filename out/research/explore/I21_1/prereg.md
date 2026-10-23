# I21-1 pre-registration (IN FORCE): open items that need no user decision; (A) NIST Part B of the selected g v2 full-data element, (B) TU Delft v1-recipe S3_D on state-C records, (C) selected g v2 recipe on IHP SG13G2 233 / 343 / 398 K

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-23), committed alone, before any I21-1 code, simulation, training or refit. DEV data only; every TEST
split stays locked (no TEST tag is loaded; every loaded tag is checked against the DEV / TEST lists of its split
file). This study writes no amendment of the committed I19-1 TEST gate decisions (a755e18); Part B reports a value
only (the amendment question is a pending user decision).

The three parts are independent. Each part has its own validity checks and labels; there is no combined outcome.

## Part A: NIST Part B (deployment checks, I17-1 rules) of the selected g v2 full-data element

### Model, cards, simulator

- Models under test: the I20-1 full-data g v2 models (5 seeds, lambda*_full = 0 V on both datasets, VDD 1.8 V,
  EPS 0.01 V): `I20_1/select_nist4/full_corrections.json` and `I20_1/select_nistrt/full_corrections.json`, key
  `full|sym_e1_l0`. Module text `i12_1_symm.to_verilog_a_v2`; OpenVAF 23.5.0 (OSDI 0.3); simulator ngspice-41 built
  with `--enable-osdi` (`~/cryo-ng41/arm64-osdi/bin/ngspice`).
- Datasets: NIST lvt (I8-1; 53 DEV devices with finite state-C starting records; deck: sky130 77 K lvt library
  `tt_77k`, explicit bin per device (I8-1 decision D2), 77 K) and NIST RT card (I9-1; 87 DEV devices; deck: open_pdks
  tt `sky130_fd_pr__{n,p}fet_01v8` card reduced to one explicit bin per device, library `tt_rt` with `lod.spice` and
  the mismatch file, 30 C). Instance line as in I8-1 / I9-1 (nf 1, ad / as / pd / ps / nrd / nrs from W, mult = m =
  1).
- M2 cards: each device's state-C starting card (`I20_1/start/records_<ds>`) refit with the full-data g frozen in the
  loop (I20-1 `refit_task`, state C, max_nfev 400, irls 2, diff_step 0.01). Card text written into the bin: the
  parameters that differ from the stock readback (`i8_1_nist.changed_params`) plus the b4check.c A1 / A2 clamp
  (state C; `i8_1_nist.bsim4_clamp_params` / `i9_1_nist_rt.bsim4_clamp_params`), as in the I20-1 decks.
- Gate current: every bin of both cards has igcmod = igbmod = 0 (read from the card text before registration). The
  I18-2 amendment A1 (V(d,s) = 0 check on the element-added current) therefore does not apply; the V(d,s) = 0 check
  is the I17-1 check on the raw element current.

### Element (I14-1 construction)

Card instance X (subcircuit, explicit bin, M2 parameters, clamp written); channel-current replica X..r of subcircuit
`<subckt>_ch` with model `<model>_ch`: REPLICA_ZERO (xpart -1; cgso, cgdo, cgbo, cgsl, cgdl, cf, cjs, cjd, cjsws,
cjswd, cjswgs, cjswgd = 0) and REPLICA_CHAN (alpha0 = alpha1 = beta0 = 0, agidl = agisl = 0, with their l / w / p
terms; jss = jsd = jsws = jswd = jswgs = jswgd = 0; igcmod = igbmod = 0) written into every bin of the replica
(NIST lvt: `i14_1_chan.replica_corner_chan` of the patched 77 K corner text; NIST RT card: the
`.subckt sky130_fd_pr__{n,p}fet_01v8 ... .ends` block of the patched card text, subcircuit and model renamed with
suffix `_ch`, REPLICA written into its bin block); unit-gain VCVS copies of V(d), V(g), V(s), V(b); a VCCS of
-gmin (1e-12 S, ngspice default; no deck sets gmin) between replica drain and body copies; OSDI module
`N.. d g s rx ri` adding (exp(g) - 1) I(sense) from d to s. Both cards carry rbodymod = 1 (body resistor network);
no change.

### Benches and checks (I14-1 / I15-1 / I17-1 definitions; VDD 1.8 V)

- Q1 grids: forward bench (d, g, s, b) = (nd, ng, 0, 0) and exchanged bench; V(g,s) 0 .. s 1.8 V (73 values,
  25 mV), V(d,s) -1.8 .. 1.8 V (145 values), serpentine order (I15-1); tolerances default, r4 (reltol 1e-4), tight
  (reltol 1e-9, abstol 1e-18 A, vntol 1e-12 V); modes card, card + replica, element. Core: equivalence
  |dI| <= 1e-6 |I| + 1e-13 A at tight tolerance (forward and exchanged; reference I_card + (exp(g) - 1) I_rep from
  the card + replica run); 0 convergence messages, non-finite values and failed runs in element decks; |I(V(d,s) =
  0)| <= 1e-12 A on the tight forward grid (I17-1; raw element current); Verilog-A g vs Python <= 1e-9 (2000 random
  points in [-1.8, 1.8] V^2), exchanged vs forward <= 1e-12; element-introduced exchange asymmetry <= 1e-3 (|I| > 1 nA,
  tight). Restriction: default-tolerance relative difference > 1e-3 (|I| > 1 nA).
- Q2 (tight, AC 1 MHz): gm sets |V(d,s)| in {0.05, 0.9, 1.8} V x |V(g,s)| on the grid; gds sets |V(g,s)| in {0.6,
  0.9, 1.2, 1.5, 1.8} V x |V(d,s)| on the grid; dense 1 mV sweeps of the same curves. Core: normalized gm / gds error
  vs the element identity Re Y_card + (exp(g) - 1) Re Y_rep + I_rep exp(g) dg/dV <= 1e-3 (|I| > 1 nA); capacitance
  error (Im Y_dd, Im Y_dg, Im Y_gg vs card) <= 1e-3; 0 gm curves with an introduced sign change in strong inversion
  (|V(g,s)| >= Vt + 0.2 V, Vt at 1e-7 A x W / L). Restriction: gds curves with an introduced sign change
  (|V(d,s)| >= 0.05 V). At lambda 0 the penalty is off; the I20-1 held-out grids have negative corrected gds on
  8.7% (lvt) and 13.2% (RT card) of the points, so gds sign changes are expected and are classified as restrictions
  (registered classification, I12-1 / I17-1).
- Q3 circuits: the sky130 circuit decks of I14-1 / I17-1 (supply 1.8 V, the NIST supply): inv_A_vtc, inv_B_vtc
  (DC VTC, 1 mV), ro5_A (tran 1 ps / 10 ns), ro5_B (5 ps / 50 ns), mirror_n, mirror_p (I_ref 10 uA, V_out 0 .. 1.8 V).
  Circuit devices: per sky130 circuit device the NIST DEV device of the same polarity with the smallest Euclidean
  distance in (ln L, ln W) (ties: first tag), at its own geometry, explicit bin and card:

  | sky130 role (I14-1) | NIST lvt device (bin) | NIST RT card device (bin) |
  |---|---|---|
  | nmos L0.15 W1.6 (inv_A, ro5_A) | nfet_01v8_lvt_W1p65_L0p18_m23p3 (bin 0) | nfet_01v8_W1p68_L0p15_m6p7 (bin 44) |
  | pmos L0.35 W1.6 (inv_A, ro5_A) | pfet_01v8_lvt_W1_L0p35_m28p6 (bin 0) | pfet_01v8_W1p68_L0p5_m16p1 (bin 41) |
  | nmos L1 W1.6 (inv_B, ro5_B, mirror_n) | nfet_01v8_lvt_W1p65_L1_m23p6 (bin 3) | nfet_01v8_W1p68_L1_m7p2 (bin 40) |
  | pmos L2 W5 (inv_B, ro5_B, mirror_p) | pfet_01v8_lvt_W5_L1p5_m30p6 (bin 11, bin-uncovered) | pfet_01v8_W5_L2_m18p5 (bin 12) |

  Arms: published (stock bin, state C), C0_refit (I20-1 state-C held-out card of that device), M2_card, M2_full (M2
  card with the element). Core: number of circuits that fail with M2_full while M2_card passes = 0 (failure: a
  convergence message, a non-finite output, or fewer than 5 ring periods). Runtime: M2_full / M2_card median wall
  time over 3 repeats per circuit where both pass.
- Q4 runtime restriction: max over the default-tolerance forward DC grid ratio (element / card, 3 repeats, per
  device) and the circuit ratios > 2.
- Label per dataset (`i12_1_symm.decide_deploy` with the I17-1 V(d,s) = 0 value): not deployable (any core check
  fails) / deployable with restrictions (no core failure, any restriction) / deployable (simulator-level).
- Reported without a rule: replica pre-check (BSIM4 outputs isub, igidl, igisl, igs, igd, igb, igcs, igcd of the
  replica at every 4th Q2 op point; expected 0); gds curves with an introduced sign change listed per device and
  |V(g,s)|; V(d,s) = 0 values at default and r4 tolerance; element-deck RRMS of the M2 pairs (native element) vs the
  Python rows; circuit metrics (V_M, maximum gain, ring frequency, mirror ratio) per arm.

### Validity (Part A, per dataset; any fail -> invalid)

| check | pass |
|---|---|
| VA1 code path | card-only measured-grid deck from the Part B library builder (ngspice-41 OSDI build) on every M2 card x exp(g) reproduces the M2 in-sample Python RRMS row (`i8_1_nist.simulate` / `i9_1_nist_rt.simulate`, state C): max \|difference\| <= 1e-9 |
| VA2 M2 | 0 failed M2 refits; 53 (lvt) / 87 (RT card) M2 rows |
| VA3 unit tests | `tests.test_i21_1_close`, `tests.test_i14_1_chan`, `tests.test_i12_1_symm`, `tests.test_i20_1_nist_clamp` pass |
| VA4 rows | 53 / 87 Part B rows (Q1 / Q2), 24 Q3 runs |
| VA5 DEV only | every loaded tag in the DEV list of `I8_1/split.json` / `I9_1/split.json`, none in TEST |

## Part B: TU Delft v1-recipe S3_D on state-C records (report only)

- Definition: I19-1 S3_D (`i19_1_dataset_gate`; `i7_1_transfer.s3_fold_value`), v1 recipe (`i3_2_bias.fit_bias`
  M_bias, seed 0, 500 steps), leave-one-geometry-group-out over the 7 DEV groups, applied to the starting-card records
  without a refit; S3_D = mean over the 45 DEV records of ln(RRMS(I_start x exp(g_{-k})) / RRMS(I_start)).
- Records: the I18-2 state-C starting records `I18_2/start/records_{4,295}K` (I19-1 used the I6-1 state-L records:
  S3_D -0.0292 at 4.13 K, -0.0510 at 295.0 K).
- Reported: S3_D (state C) per temperature; whether its sign differs from the state-L value; the decision word under
  threshold 0 (apply iff S3_D < 0). No amendment of a755e18 is written; the frozen-rule TU Delft decision in the
  freeze draft is updated as a report value only.
- Validity: VB1 a state-L recompute of the first group in sorted order at each temperature (I6-1 records) reproduces
  the stored I19-1 per-record values (`I19_1/s3/tud{4,295}_v1.json`, max |difference| <= 1e-9); VB2 every record tag
  in the DEV list of `I6_1/split.json`, none in TEST; VB3 45 finite per-record values per temperature.

## Part C: selected g v2 recipe on IHP SG13G2 233 / 343 / 398 K

- Setup of I17-2 at each temperature T (prereg c071472): I5-1 data, PSP103, ngspice-46 + psp103.osdi; 18 DEV DUTs
  (LV 10, HV 8), TEST 10 locked; starting records `I5_1/start/records_TK`; leave-one-DUT-out within class; held-out
  refit from the I5-1 starting card with g frozen (max_nfev 400, irls 2, diff_step 0.01).
- Arms: C0_refit and M_bias v1 (I5-1 recipe, reruns); selected g v2: g v2 recipe (`fit_v2`, symmetric inputs, EPS
  0.01 V, inputs normalized to the class supply 1.2 V (LV) / 3.3 V (HV), 5 seeds, 2000 steps), penalty on 75 x 75
  forward-quadrant grids of the starting card at T (0 .. class supply), lambda*_f per outer fold from the I17-2
  nested selection (inner folds: training DUTs sorted by (pMOS, tag), i mod 3; one member, seed 0, per inner fold
  and lambda in {0, 1, 2, 5, 10} V; inner held-out refits; lambda* = argmin mean ln RRMS, non-finite -> 0, ties to
  the larger lambda).
- Change from the I17-2 procedure (compute): the fixed-lambda arms sym_e1_l{lambda} are not run on every fold; the
  selected row of fold f is computed directly as the 5-seed fold correction at lambda*_f and its held-out refit. The
  training of one (fold, lambda, seed) is deterministic and does not depend on the other folds, so the selected rows
  equal the I17-2 construction.
- Primary per T: GMR(selected, C0_refit), device bootstrap (10,000 resamples, seed 0, 95% CI): CI upper < 1 and 0
  failed DUTs in selected and C0_refit -> gain over C0_refit replicated; CI lower > 1 -> worse than C0_refit; else
  inconclusive. Secondary: GMR(selected, M_bias v1): CI upper <= 1.05 -> non-inferior to M_bias v1; else not shown
  non-inferior. Reported: aggregates, geometry-cluster CIs (12 clusters), LV / HV, lambda* counts, negative
  corrected gm / gds fractions, GMR(M_bias v1, C0_refit) of the reruns.
- Validity per T: VC1 C0_refit and M_bias v1 reruns reproduce the stored I5-1 rows (`I5_1/lodo_TK/results.json`,
  max |difference| <= 1e-9, 18 each); VC2 unit tests (`tests.test_i21_1_close`, `tests.test_i17_2_ihp`,
  `tests.test_i12_1_symm`, `tests.test_i15_2_ablate`); VC3 18 rows in selected, C0_refit and M_bias v1; inner
  selection complete (18 x 3 x 5 members, every inner held-out DUT scored); VC4 18 finite 75 x 75 grids with positive
  forward current at the corner; VC5 every loaded tag in the DEV list of `I5_1/split.json`.

## Predictions (stated before any run; not rules)

| quantity | prediction | basis |
|---|---|---|
| Part A label, NIST lvt | deployable with restrictions (gds sign changes, default tolerance, runtime) | I17-1 sky130 (same 77 K library), lambda 0 |
| Part A label, NIST RT card | deployable with restrictions | as above |
| Part A gds curves with an introduced sign change | > 0 on both datasets | I20-1 negative corrected gds 8.7% / 13.2% at lambda 0 |
| Part B sign of S3_D (state C) | < 0 at 4.13 K and 295.0 K (no sign change vs state L) | state-L -0.029 / -0.051; I20-1 NIST v1 S3_D kept its sign under state C |
| Part C ln GMR(selected, C0_refit) | > 0 at 233, 343, 398 K | I5-1 GMR(v1, C0_refit) 1.106 / 1.181 / 1.272; I17-2 300 K selected 1.255 |
| Part C primary label | 233 K inconclusive, 343 K inconclusive, 398 K worse than C0_refit | I5-1 v1 CIs [0.908, 1.425], [0.983, 1.502], [1.030, 1.627] |
| Part C ln GMR(selected, M_bias v1) | > 0 at all three | I17-2 300 K 1.034 |
| Part C secondary label | not shown non-inferior at all three | I17-2 300 K CI upper 1.107 |

## Order and compute

This prereg (alone) -> code + unit tests -> smoke (Part A: 2 devices per dataset, refit max_nfev 20, Q3 on the
smoke cards; Part B: 1 group per temperature, 50 steps; Part C: 2 outer folds at 233 K, 200 steps, 1 seed, refit
max_nfev 20; not used for decisions) -> queue, one heavy job at a time: I21_1_s3tud, I21_1_partA_nist4,
I21_1_partA_nistrt, I21_1_ihp_{233,343,398}_{pens,base,inner,sel}, I21_1_decide. Estimated 5-6 h.

## Outputs

`out/research/explore/I21_1/{partA_nist4, partA_nistrt, s3tud, ihp_<T>K/{pens,base,inner,sel}, decision.json}`;
arrays and the measured data not in git; report batch `out/research/report/2026-10-23_i21_1/` (tables vs rules,
per-device values, I-V figures with measured points, the stock-card NGSpice curve and re-simulated cards), site
claim `i21-1`, RESEARCH_LOOP entry, method-freeze draft evidence table and open items.
