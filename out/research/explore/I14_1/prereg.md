# I14-1 pre-registration (IN FORCE): Part B deployment test of g v2 with a channel-current replica, one BSIM4 model-check state and the corrected Q2 reference; M_bias v1 reference arm; sky130 77 K, 18 devices

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-22), committed alone.

Done before it (not committed, not used for any decision):

- code draft: `src/cryoml/explore/i14_1_chan.py`, `scripts/research/explore/i14_1_chan.py`, the clamp changes listed
  under fix 1, `tests/test_i14_1_chan.py` (16 tests; with the I13-1, spice_batch, sky130_misspec and I12-1 suites
  68 tests pass);
- the replica pre-check (stage `repcheck`, results below, `repcheck/repcheck.json`);
- a binary check: the card-only measured-curve deck run on the OSDI ngspice-41 build through the clamped library
  equals `simulate_pdk` (canonical ngspice-41, state C) on 3 starting cards (nmos_L1_W1p6, pmos_L2_W5, nmos_L0p19_W7
  with a2 1.877): difference 0.0;
- one OpenVAF compile of a v1 fold module (0.9 s).

No I14-1 element run, M2 refit, accuracy run or Part B check has been made. No correction is trained in I14-1.

## Question

With (1) one BSIM4 model-check state in every deck, (2) an element that scales only the channel current of the card
and (3) a Q2 reference that matches the element identity, does the full-data g v2 (I13-1 state C) pass the I12-1
Part B deployment checks in ngspice-41 + OSDI on all 18 devices? How much does the element change the DC current of
the corrected device relative to the trained Python model (exp(g) times the card's drain-terminal current)?

## Inputs (fixed; I13-1 state C)

- Records: `I13_1/records` (starting cards direct43_r3_fd, state C).
- g v2: fold corrections `I13_1/lodo/corrections.json` (`M_bias_v2|<tag>`), full-data `full|M_bias_v2`; held-out
  cards `I13_1/lodo/heldout_cards.json`; rows `I13_1/lodo/results.json`.
- M_bias v1 (reference arm): fold corrections `I13_1/all18/corrections.json` (`M_bias|<tag>`), full-data
  `full|M_bias`; held-out cards `I13_1/all18/heldout_cards.json` (`M_bias`); rows `I13_1/all18/results.json`.
- C0_refit cards for Q3: `I13_1/all18/heldout_cards.json` (`C0_refit`).

## Fix 1: clamp state in every deck builder

The ngspice-41 `b4check.c` A1/A2 clamp (a2 > 1 -> a2 = 1 and a1 = 0; a2 < 0.01 -> 0.01) is written into the card
(`spice_pdk.bsim4_clamp_params`) by every deck builder that runs more than one analysis per process:
`spice_batch.simulate_pdk_multi` (altermod, parallel, separate; new argument `bsim4_state`, default "clamped") and
`ngspice_jacobian`; `sky130_misspec.simulate_truth`; `i3_2_followup.simulate_sky130` (used by the M_bias deck
re-simulation); `i11_1_deploy.patched_library` and `i12_1_symm.patched_library_v2` (all OSDI element, card-only and
circuit decks of I11-1, I12-1 and I14-1); `e3_temp.simulate_truth_at` (resistor path). "legacy" reproduces the old
decks; `i12_1_diagnosis.py` uses "legacy" (it studied the legacy state). Unit tests (difference 0 for a card with
a2 = 1.9 on nmos_L0p19_W7): `simulate_pdk_multi` altermod and separate = `simulate_pdk` state C for four cards
(a2 1.9, a2 1.9, a2 0.5, published); parallel within 1e-6 relative (its documented near-exact behavior);
`simulate_pdk_multi` legacy = `simulate_pdk` legacy and differs from state C; `simulate_truth` = `simulate_pdk`
state C; `simulate_sky130` (canonical binary, no element) = `simulate_pdk` state C; `patched_library` writes a2 = 1
for a2 = 1.9 and identical text for a2 = 0.5 in both states.

## Fix 2: channel-current replica

The sensed replica is the card (same subcircuit, same bin patch, clamp written) with, in every bin:
the I12-1 charge-free set (xpart = -1, cgso = cgdo = cgbo = cgsl = cgdl = cf = 0, cjs = cjd = cjsws = cjswd = cjswgs
= cjswgd = 0); impact ionization off: alpha0 = alpha1 = beta0 = 0 and their size terms lalpha0, walpha0, palpha0,
lalpha1, walpha1, palpha1, lbeta0, wbeta0, pbeta0 = 0 (the nMOS bins carry nonzero lalpha0 / walpha0 / palpha0 /
lbeta0 / wbeta0 / pbeta0; without them the replica of nmos_L0p15_W1p6 and nmos_L1_W3 had a substrate current up to
0.17 A in a first pre-check run); GIDL / GISL off: agidl = agisl = 0 and lagidl, wagidl, pagidl, lagisl, wagisl,
pagisl = 0; junction saturation current densities jss = jsd = jsws = jswd = jswgs = jswgd = 0; gate tunneling
igcmod = igbmod = 0. No other DC parameter changes. With zero saturation current BSIM4 keeps the junction conductance
gmin between the internal drain and body nodes (`b4ld.c`: gbd = CKTgmin, cbd = gbd vbd); a VCCS of value -gmin
(1e-12 S, the ngspice default; no deck sets gmin) between the replica drain and body copies removes it from the sensed
current. The BSIM4 instance multiplier inside the subcircuit is m = 1 for every device in these decks (printed in
every pre-check run; the X-line `mult=.. m=..` of the six pMOS devices with 2 or 3 does not reach the BSIM4
instance), so the compensation is 1 x gmin. The replica terminals are driven by unit-gain VCVS copies of V(d), V(g),
V(s), V(b); the module senses the current through the zero-volt branch rx -> ri and contributes
I(d, s) <+ (exp(g) - 1) I(sense) (I12-1 module text, unchanged).

### Replica pre-check (before registration)

Forward and exchanged-polarity grid of I11-1 (Vgs 0 to +-1.85 V x Vds -1.85 to 1.85 V, 25 mV, 11175 points), tight
tolerance (reltol 1e-9, abstol 1e-18, vntol 1e-12), card + replica deck, 17 digits, canonical ngspice-41; 18 devices
x 4 card sets (published, starting cards in state C, I13-1 v2 held-out cards, I13-1 v1 held-out cards), 72 runs,
8 s, 0 failed runs, 0 convergence messages, 0 unrecognized-parameter messages.

| quantity | published | start (C) | v2 held-out (C) | v1 held-out (C) |
|---|---|---|---|---|
| P1 replica isub, igidl, igisl, igs, igd, igb, igcs, igcd, max abs (A) | 0 | 0 | 0 | 0 |
| P1 replica ibd minus gmin x type x V(b,d), max abs (A) | 6.0e-16 | 5.8e-16 | 5.7e-16 | 5.7e-16 |
| P2 sensed current vs the replica's own channel current (BSIM4 `id`), max abs (A) | 2.9e-13 | 3.6e-13 | 1.7e-12 | 4.8e-13 |
| P2 max relative at \|I\| > 1 nA | 1.1e-4 | 2.6e-5 | 2.2e-5 | 2.5e-5 |
| P2 max of \|dI\| / (1e-6 \|I\| + 1e-13 A), all points | 1.68 | 0.43 | 0.36 | 0.46 |
| P3 replica channel vs card channel (`id` outputs), forward quadrant, \|I\| > 1 nA, max relative | 1.1e-5 | 1.2e-3 | 1.1e-3 | 1.2e-3 |
| P3 same, points with card \|Isub\| < 1e-15 A, max relative | 6.4e-11 | 2.1e-8 | 2.0e-8 | 2.1e-8 |
| P3 devices with max relative <= 1e-12 on every forward point | 2 of 18 | 6 of 18 | 5 of 18 | 6 of 18 |
| P3 reverse quadrant (drain junction forward-biased, body at the source), max relative | 218 | 18.8 | 13.5 | 15.1 |
| card non-channel share \|I_card - I_channel\| / \|I_channel\|, forward, \|I\| > 1 nA, max | 2.9e-3 | 1.04 | 1.04 | 1.02 |

Result against the requested criterion (replica drain current = card channel current within 1e-12 relative): **met
only in part.** P1 holds exactly: the replica has no substrate, GIDL / GISL or gate current, and its junction current
is the gmin term that the VCCS removes. P2: the sensed current equals the replica's own channel current to the tight
solver tolerance (max 1.7e-12 A absolute), not to 1e-12 relative at small currents; this is a property of the Newton
solution (the BSIM4 `id` output is evaluated at the last iterate, the branch current at the solution). P3: the
replica's channel current equals the card's channel current within 1e-12 relative on every forward point for 2 to 6
of the 18 devices; the largest forward differences (up to 1.2e-3, nmos_L100_W100, card Isub up to 1.2e-4 A) and the
reverse-quadrant differences come from the card's own non-channel currents: they flow through the card's body
resistor network (rbodymod = 1, rbpb = rbpd = rbps = rbdb = rbsb = 50 ohm) and change its internal body voltage and
hence its channel current; the replica, without these currents, has a different internal body voltage. A replica that
keeps these currents carries them in its drain-terminal current, and the BSIM4 terminal currents do not separate the
channel current from the drain-side non-channel currents, so no replica of this form meets 1e-12 at those points.
Registered consequence: the element scales the replica's channel current I_rep, and every Part B reference uses
I_rep measured in a card + replica run of the same deck, tolerance and clamp state (not the card's `id` output).

## Fix 3: Q2 reference (element identity)

Re Y_elem = Re Y_card + (exp(g) - 1) Re Y_rep + I_rep exp(g) dg/dV, Im Y_elem = Im Y_card, with Y_card (card drain
current), Y_rep (sensed replica current), I_rep and I_card from one card + replica run (op + AC at 1 MHz per point,
tight tolerance, clamp written) and g, dg/dV (central differences, h = 1e-5 V) at the module terminals. Capacitance
check unchanged (Y_dd, Y_dg from the drain current, Y_gg from the gate current; element run vs card + replica run).
Reported, no rule: the I12-1 formula exp(g) Re Y_card + I_card exp(g) dg/dV on the same runs; max |Im Y_rep|.

## Part B (I12-1 design; changes listed)

Card **M2** per device and arm: the 43-parameter card refit from the starting card (direct43_r3_fd) with the arm's
full-data g frozen in the loop (I3-2 / I12-1 refit recipe: TRF, max_nfev 400, irls 2, diff_step 0.01, strict paper
RRMS, frozen curve inclusion; `simulate_pdk` state C). In-sample, reported as such. Compiler OpenVAF 23.5.0
(`--target aarch64-apple-darwin`), simulator ngspice-41 OSDI build `~/cryo-ng41/arm64-osdi/bin/ngspice`.
Tolerances: default; r4 (reltol 1e-4, descriptive); tight (reltol 1e-9, abstol 1e-18, vntol 1e-12).

- **Q1 device equivalence.** I11-1 grid; forward bench (card at (nd, ng, 0, 0), module d = nd, s = 0) and exchanged
  bench (card drain at nz with VZ 0 V to ground, card source at nd; module d = nz, s = nd). Reference (change):
  forward I_py = I_card + (exp(g) - 1) I_rep; exchanged I_py = I_card(nd) - (exp(g) - 1) I_rep (I_rep = current into
  the replica drain, the copy of nz); I_card and I_rep from the card + replica run at the same tolerance; g at the
  module terminals. Criterion per point |I_elem - I_py| <= 1e-6 |I_py| + 1e-13 A (tight, both benches).
- **Verilog-A text:** g of the text vs Python g on 2000 random points (`default_rng(20261021)`), and the text at
  exchanged vs forward terminals.
- **Exchange asymmetry:** element-introduced |ln(I_swap / I_fwd)_elem - ln(I_swap / I_fwd)_card| in forward
  operation, |I| > 1 nA, tight; card values from the card + replica run. Expected (stated before the run): with the
  channel-only replica the element current is antisymmetric under the exchange where the replica channel current is;
  the remaining part is bounded by the card's own asymmetry delta (reported as `swap_asym_card_max`) times
  (1 - exp(-|g|)).
- **Q2 small signal:** op points, AC at 1 MHz, tight; reference as fix 3; derivative error
  |Y_elem - Y_py| / max(|Y_py|, 1e-9 S) on real parts at |I_py| > 1 nA; capacitance error
  |Im Y_elem - Im Y_card| / max(|Im Y_card|, 1e-15 S); I11-1 sign rule for gm (54 curves) and gds (90 curves) with the
  1 mV sweeps (element vs card only).
- **Q3 circuits:** inv_A_vtc, inv_B_vtc, ro5_A, ro5_B, mirror_n, mirror_p; arms published, C0_refit (I13-1 state C),
  M2_card, M2_full; I12-1 amendment A1 applies (circuit runtime ratio over circuits where M2_card and M2_full pass;
  no repeats after a failed run).
- **Q4 runtime:** median of 3 wall times, element vs card only (no replica), default-tolerance grid (18 devices) and
  the circuits.

### Part B rule (unchanged from I12-1)

| check | core (any fail -> not deployable) |
|---|---|
| Q1 equivalence | 0 grid points violating \|dI\| <= 1e-6 \|I_py\| + 1e-13 A (tight, forward and exchanged, 18 devices) |
| Q1 runs | 0 convergence messages, 0 non-finite values, 0 failed runs (element runs) |
| Q1 Vds = 0 | max \|I_elem(Vds = 0)\| <= 1e-12 A (default and tight) |
| Verilog-A g | max \|g_text - g_python\| <= 1e-9; max \|g_text(exchanged) - g_text(forward)\| <= 1e-12 |
| exchange asymmetry | max element-introduced \|d ln\| <= 1e-3 |
| Q2 derivatives | max normalized gm / gds difference <= 1e-3 (element-identity reference) |
| Q2 capacitances | max capacitance error (Y_dd, Y_dg, Y_gg) <= 1e-3 |
| Q2 gm sign | 0 gm curves with an introduced sign change in strong inversion (54 curves) |
| Q3 circuits | 0 circuits failing with the element while M2_card passes |

Restrictions: default-tolerance max relative difference (|I_py| > 1 nA) > 1e-3; >= 1 gds curve (of 90) with an
introduced sign change; wall-time ratio > 2. Label: any core fail -> **not deployable**; core pass and >= 1
restriction -> **deployable with restrictions**; otherwise **deployable (simulator-level)**.

### Reference arm: M_bias v1 full-data g (not the I14-1 outcome)

The same Part B stages with the I13-1 full-data M_bias v1 g (`full|M_bias`, inputs is_pmos, |V(g,s)| / VDD,
|V(d,s)| / VDD; module `i14_1_bias_corr_v1`, same element structure) and its own M2 cards (refit with the v1 g in the
loop). Its label is computed with the same rule and reported as "reference arm"; it does not enter the outcome. The
exchanged-terminal Verilog-A check is expected to fail for v1 (abs() inputs are not d/s symmetric).

## Accuracy

Part A is not rerun. The registered accuracy flags are the I13-1 state-C values (Python model, I_card exp(g)):
A1 GMR(M_bias_v2, C0_refit) 0.785 [0.693, 0.896], 16/18 lower, 0 failed devices: **passes**; A2 GMR(M_bias_v2,
M_bias_v1) 1.053 [0.966, 1.137]: CI upper above 1.05, **fails**.

Element-deck accuracy (reported, no rule): for each device and arm (v2 and v1) the held-out pair (fold g + I13-1
held-out card) on the measured curves: SPICE-native RRMS (element deck), Python RRMS from the card-only sims of the
same deck (must reproduce the I13-1 row, validity V1), channel-model Python RRMS (I_card + (exp(g) - 1) I_rep from a
card + replica run), per-device differences native - I13-1 row; all-device / combined / median; GMR(native v2,
C0_refit), GMR(native v2, stored v1), GMR(native v2, native v1), GMR(native v1, C0_refit) (device bootstrap 10000,
`i3_2_bias.paired_ratio`). Expected (stated before the run): native - Python = the RRMS effect of
(exp(g) - 1)(I_card - I_rep); I_card - I_rep is the substrate current plus the gmin junction term plus the
body-voltage effect of P3; the forward non-channel share reaches 1.04 on the state-C cards (nmos_L100_W100), so a
nonzero difference is expected on the nMOS devices with substrate current.

## Validity (checked first; any fail -> invalid)

| check | pass |
|---|---|
| V1 card-only library deck (OSDI build, clamp written) x exp(fold g) reproduces the I13-1 per-device rows, v2 (I13_1/lodo) and v1 (I13_1/all18), 36 rows | max \|RRMS difference\| <= 1e-9 |
| V2 unit tests (`tests.test_i14_1_chan`, `test_i13_1_clamp`, `test_spice_batch`, `test_sky130_misspec`, `test_i12_1_symm`) | all pass |
| V3 rows | 18 element-deck accuracy rows per arm; 0 failed M2 refits per arm |

## Decision

| condition | outcome | action |
|---|---|---|
| any validity check fails | invalid | stop: fix and re-register |
| Part B (v2) "not deployable" | not deployable | stop; diagnose the failed core checks |
| Part B (v2) "deployable with restrictions" or "deployable (simulator-level)" | deployable, accuracy below v1 (not non-inferior) | stop v2; report the accuracy cost of the deployment constraints |

(I12-1 outcome table with A1 = pass and A2 = fail fixed by I13-1; "supported" and "no gain over C0_refit" are not
reachable.) The element-deck accuracy and the reference arm are reported with the outcome and do not change it.

## Other decisions

- D1 The M2 refits and all decks run in state C; cards are stored unclamped and clamped at deck build.
- D2 Before every deck the replica library is checked (`replica_bins_ok`: every bin carries every REPLICA_CHAN value
  once).
- D3 Smoke (after the code commit; not used for decisions): 2 devices (nmos_L1_W1p6, pmos_L2_W5), refit max_nfev 20,
  all stages.
- D4 Order: this prereg commit (alone) -> code + unit tests commit -> smoke -> queue (one heavy job at a time):
  I14_1_validity, I14_1_accuracy, I14_1_deploy_v2, I14_1_deploy_v1, I14_1_decide.

## Outputs

`out/research/explore/I14_1/{repcheck, validity, accuracy, deploy_v2, deploy_v1, decision.json}`; arrays (`*.npz`)
and OSDI binaries not in git; report batch, site claim `i14-1` and RESEARCH_LOOP entry after the decision.

## Compute (estimate)

Validity about 1 min; accuracy 36 compiles (about 1 s each) + 72 measured-curve decks, about 3 min; deploy per arm:
M2 refit about 100 s, native M2 rows about 30 s, q12 about 4 min on 4 workers, q3 about 1 min. No other heavy job runs
at registration (queue empty since 00:37).
