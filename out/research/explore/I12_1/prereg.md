# I12-1 pre-registration (IN FORCE): bias-only correction g v2 as a deployable ngspice-41 OSDI add-on, sky130 77 K, 18 devices

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-21), committed alone. Done before it (not committed, not used for any decision): code draft
(`src/cryoml/explore/i12_1_symm.py`, `scripts/research/explore/i12_1_symm.py`); a charge-free replica check on the
published card (nmos_L1_W1p6 at Vgs 1.2 V / Vds 0.9 V and pmos_L2_W5 at -1.2 V / -0.9 V, tight tolerance: replica DC
drain current equal to the card's to 12 digits, imaginary AC replica current 0 at 1 MHz for drain and gate
excitation); the penalty grids of the 18 starting cards (`grids/`) and the penalty value of the stored M_bias v1
corrections on them (`calib/calib_v1.json`). No v2 correction has been trained and no I12-1 held-out refit has run.
All 18 sky130 77 K devices are open (user decision 2026-10-21). There is no measured circuit data: the circuit
checks are simulator-level.

## Question

Does a bias-only correction g v2 that removes the I11-1 deployment failures (source/drain asymmetry, scaled
displacement currents, abs() kinks, negative gm / gds) keep the leave-one-device-out (LODO) accuracy of M_bias v1
(I3-2 recipe, all-18 LODO: all-device 0.0732, combined 0.0691; C0_refit 0.1043 / 0.0975; T1_inloop 0.0744 / 0.0703),
and does it pass the I11-1 deployment checks in ngspice-41 + OSDI?

## g v2: the changes from M_bias v1 (each exactly)

1. **Source/drain-symmetric inputs.** s = +1 (nMOS), -1 (pMOS). With the transistor terminal voltages V(g,s),
   V(g,d) = V(g,s) - V(d,s), V(d,s): a = s V(g,s), b = s V(g,d), m = sabs(V(d,s)), x1 = (a + b + m) / (2 VDD),
   x2 = m / VDD, VDD = 1.85 V. Network input (is_pmos, x1, x2), standardized with the training-point mean and std
   (is_pmos kept 0/1). x1 is a smooth max(a, b) / VDD. Exchanging d and s maps (a, b, V(d,s)) to (b, a, -V(d,s)),
   so g is unchanged and I_card x exp(g) changes sign with I_card. On the measured (forward) points x1 differs from
   |Vgs| / VDD by at most EPS / (2 VDD).
2. **exp(g) multiplies the static drain-terminal current only.** A Verilog-A BSIM4 is out of scope. The add-on
   element senses a charge-free replica of the card: the same subcircuit with every model bin given xpart = -1
   (BSIM4 then skips the intrinsic charge: ngspice-41 `b4ld.c`, `if ((model->BSIM4xpart < 0) || ...)` -> charges 0,
   `goto finished`) and cgso = cgdo = cgbo = cgsl = cgdl = cf = 0, cjs = cjd = cjsws = cjswd = cjswgs = cjswgd = 0
   (overlap, fringing, junction capacitance); no DC parameter is changed. The replica's four terminals are driven by
   unit-gain VCVS copies of V(d), V(g), V(s), V(b), so the replica draws no current from the circuit. The module
   `i12_1_bias_corr_v2(d, g, s, rx, ri)` senses the replica drain current I_rep through the zero-volt branch rx -> ri
   (`V(sense) <+ RS*I(sense)`, RS = 0) and contributes `I(d, s) <+ (exp(g) - 1) * I_rep`. In DC I_rep equals the
   card's drain-terminal current (channel, GIDL and junction currents, as in the training simulations); in AC and
   transient the element adds no charge. The card instance itself is unchanged.
3. **Training penalty on negative gm and gds** of the corrected current. Per training device: grid u_g in
   {0, 0.025, ..., 1.85} V x u_d in {0.025, ..., 1.85} V (u = s V, forward quadrant; the measured range of every
   device is |Vgs|, |Vds| in [0, 1.85] V); card-only current I_card of the device's training card (direct43_r3_fd,
   canonical ngspice-41, default tolerances); points kept where the forward current s I_card > 1 nA.
   r_gm = gm_card / I_card + dg/du_g, r_gds = gds_card / I_card + dg/du_d (1/V; gm_card, gds_card by central
   differences of the card grid, one-sided at the edges; dg/du by autograd through the input map). The sign of r
   is the sign of the corrected derivative (exp(g) I_card > 0). P = mean over training devices of the per-device mean
   of relu(-r_gm) + relu(-r_gds). Objective per ensemble member: E1 loss + lambda P + 1e-4 sum ||W||^2, with
   **lambda = 10 V**. Rule for lambda (fixed before any v2 training): the stored full-data v1 model has P = 0.00873
   /V and in-sample E1 loss 0.0732 on the 18 starting-card records; lambda is the smallest value of the 1-2-5-10
   sequence with lambda x P(v1) > 0.0732 (8.4 -> 10).
4. **No abs():** sabs(x) = sqrt(x^2 + EPS^2), **EPS = 0.01 V**.
5. **Equivalence criterion** Python vs OSDI ngspice-41: |I_elem - I_py| <= 1e-6 |I_py| + 1e-13 A at every grid
   point (tight tolerance).

Unchanged from v1: 5 seeds (0-4, `torch.manual_seed(seed)` per member), MLP 3-16-16-1 tanh, zero-initialized last
layer, output tanh (OUT_SCALE 1), Adam lr 3e-3, 2000 full-batch steps, L2 1e-4 on weight matrices, float64, E1 loss
(mean per-device paper RRMS of the corrected current), training cards fixed (the stored starting-card records);
held-out card refit from direct43_r3_fd with g frozen in the loop (TRF, max_nfev 400, irls 2, diff_step 0.01);
strict paper RRMS with curve inclusion frozen to the published card's curve set; all-device and combined reported.

## Part A: accuracy (LODO, 18 devices)

Per held-out device: g v2 trained on the other 17 starting-card records and their 17 penalty grids; held-out refit
with the fold g in the loop. Also a full-data g v2 trained on all 18 (for Part B). Comparators are the stored
all-18 rows (`I3_2/all18/results.json` per_device): C0_refit, M_bias v1 (`M_bias`), T1_inloop.

| contrast | statistic | pass |
|---|---|---|
| A1 primary | GMR(M_bias_v2, C0_refit) = exp(mean ln ratio) over 18 devices, device bootstrap (10000 resamples, `default_rng(0)`, 95% percentile CI) | CI upper bound < 1 and 0 failed devices |
| A2 non-inferiority | GMR(M_bias_v2, M_bias_v1), same bootstrap | CI upper bound <= 1.05 |

Margin 1.05: v1's gain over C0_refit is GMR 0.755 (ln -0.281); a 5% increase (ln 0.049) keeps at least 83% of
that log gain. Reported (no rule): all-device mean, combined (equal-weight nMOS / pMOS family means), median, wins,
GMR(M_bias_v2, T1_inloop), post-hoc (no-refit) RRMS, the training penalty P and the fraction of grid points with
negative corrected gm / gds on the training grids.

## Part B: deployment (full-data g v2, 18 devices)

Card **M2** per device: the 43-parameter card refit from direct43_r3_fd with the full-data g v2 frozen in the loop
(same refit recipe; in-sample, reported as such). Compiler OpenVAF 23.5.0 (`--target aarch64-apple-darwin`),
simulator ngspice-41 OSDI build `~/cryo-ng41/arm64-osdi/bin/ngspice`; card-only runs use the same binary without
`pre_osdi`. Tolerances: default; r4 (reltol 1e-4, descriptive); tight (reltol 1e-9, abstol 1e-18, vntol 1e-12).

- **Q1 device equivalence.** Grid as I11-1: Vgs 0 to +-1.85 V (75 values) x Vds -1.85 to 1.85 V (149 values),
  nested DC sweep, body at 0, all 18 devices. Forward bench: card at (nd, ng, 0, 0), module d = nd, s = 0.
  Exchanged bench: card drain at node nz with a 0 V source VZ to ground, card source at nd; module d = nz, s = nd.
  Python reference: forward I_py = I_card exp(g); exchanged I_py = I_card(nd) + (exp(g) - 1) i(VZ)_card (i(VZ) =
  minus the card drain-terminal current); g evaluated at the module terminals. Card-only and element runs at
  default, r4 (forward only) and tight (both benches).
- **Verilog-A text:** g of the emitted text (assignment lines evaluated in Python) vs the Python g on 2000 random
  points (both polarities, V(g,s), V(d,s) uniform in [-1.85, 1.85] V, `default_rng(20261021)`); and g of the text at
  exchanged terminals vs forward terminals on the same points.
- **Exchange asymmetry:** element-introduced |ln(I_swap / I_fwd)_elem - ln(I_swap / I_fwd)_card| in forward
  operation, |I| > 1 nA, tight (I11-1 statistic). Expected nonzero part (stated before the run): BSIM4 junction
  currents include gmin x V_junction (gmin 1e-12 S); the element scales the replica's drain-terminal current, whose
  junction part sits at the drain in the forward bench and at the source in the exchanged bench; bound about
  (1 - exp(-|g|)) x 1.85e-12 A / 1 nA <= 1e-3.
- **Q2 small signal.** Op points, AC at 1 MHz, tight tolerance, gm / gds sets and 1 mV sweeps as I11-1. Python:
  Re Y = exp(g) Re Y_card + I_card exp(g) dg/dV (central differences of g, h = 1e-5 V), Im Y = Im Y_card. Derivative
  error |Y_elem - Y_py| / max(|Y_py|, 1e-9 S) on real parts at |I_py| > 1 nA. Capacitance error
  |Im Y_elem - Im Y_card| / max(|Im Y_card|, 1e-15 S) for Y_dd, Y_dg (drain current) and Y_gg (gate current) at all
  op points. Sign rule (I11-1, curve level): strong inversion |Vgs| >= V_T + 0.2 V (V_T at |I| = 1e-7 W_eff / L A,
  |Vds| = 0.05 V, card only); a gm curve has an introduced sign change if the element curve has gm < -1e-12 S in
  strong inversion (AC points or 1 mV sweep) and the card-only curve has none; same for gds curves at |Vds| >= 0.05 V.
- **Q3 circuits** (I11-1 set, default tolerances): inv_A_vtc, inv_B_vtc, ro5_A, ro5_B, mirror_n, mirror_p; arms
  published, C0_refit (stored all18), M2_card (M2 card, no element), M2_full (M2 card + element).
- **Q4 runtime:** median of 3 ngspice wall times, element / card only, default-tolerance grid deck (18 devices) and
  the six circuits (M2_full vs M2_card). Expected above 2 (each corrected transistor evaluates the BSIM4 card twice).

### Part B rule

| check | core (any fail -> not deployable) |
|---|---|
| Q1 equivalence | 0 grid points violating \|dI\| <= 1e-6 \|I_py\| + 1e-13 A (tight, forward and exchanged benches, 18 devices) |
| Q1 runs | 0 convergence messages, 0 non-finite values, 0 failed runs (element runs) |
| Q1 Vds = 0 | max \|I_elem(Vds = 0)\| <= 1e-12 A (default and tight) |
| Verilog-A g | max \|g_text - g_python\| <= 1e-9; max \|g_text(exchanged) - g_text(forward)\| <= 1e-12 |
| exchange asymmetry | max element-introduced \|d ln\| <= 1e-3 |
| Q2 derivatives | max normalized gm / gds difference <= 1e-3 |
| Q2 capacitances | max capacitance error (Y_dd, Y_dg, Y_gg) <= 1e-3 |
| Q2 gm sign | 0 gm curves with an introduced sign change in strong inversion (54 curves) |
| Q3 circuits | 0 circuits failing with the element (convergence message, non-finite output, timeout 120 s, < 5 ring periods) while M2_card passes |

Restrictions (listed when they apply): default-tolerance max relative difference (|I_py| > 1 nA) > 1e-3; >= 1 gds
curve (of 90) with an introduced sign change; wall-time ratio > 2. Label: any core fail -> **not deployable**; core
pass and >= 1 restriction -> **deployable with restrictions**; otherwise **deployable (simulator-level)**.

Changes from the I11-1 rule: the Q1 tight threshold (relative 1e-5 at |I| > 1 nA) is replaced by the per-point
criterion above (task item 5); exchange asymmetry and capacitance moved from restrictions to core checks (they are
design claims of v2); the Verilog-A g checks are added as core checks.

## Validity (checked first; any fail -> invalid)

| check | rows (exact selection, unit-tested) | pass |
|---|---|---|
| V1 trainer reproduction | the trainer `fit_v2` with v1 inputs (is_pmos, \|Vgs\|/VDD, \|Vds\|/VDD), lambda 0, v1 hyperparameters, on the fold training sets of `nmos_L0p15_W1p6` and `pmos_L2_W5` (17 records each); compared with the stored `I3_2/all18/corrections.json` keys `M_bias\|nmos_L0p15_W1p6`, `M_bias\|pmos_L2_W5` | max \|g_new - g_stored\| over the 17 records' training points <= 1e-9 on both |
| V2 held-out refit reproduction | held-out refit with the stored v1 fold correction for `nmos_L1_W1p6` and `pmos_L8_W5`; compared with the stored all18 `per_device.M_bias` rows of those two devices | \|RRMS difference\| <= 1e-9 on both |
| V3 rows | M_bias_v2 LODO rows | exactly one row per device for the 18 devices |

## Decision

| condition | outcome | action |
|---|---|---|
| any validity check fails | invalid | stop: fix and re-register |
| A1 fails | no gain over C0_refit | stop |
| A1 and A2 pass, Part B not "not deployable" | supported | continue: v2 replaces v1 as the deployable add-on |
| A1 passes, A2 fails, Part B not "not deployable" | deployable, accuracy below v1 | stop v2; report the accuracy cost of the deployment constraints |
| A1 passes, Part B "not deployable" | not deployable | stop; diagnose the failed core checks |

No prediction commit: I11-1 (the pattern followed here) had none, and no training-set statistic is pre-registered
for this question.

## Other decisions

- D1 No card is shifted: held-out and M2 cards are refit inside the stored base box (I3-2 recipe; 0 failed devices in
  the all-18 run), so the I10-1 failure mechanism (shifted effective size-dependent BSIM4 values) does not arise and
  no clamp is applied.
- D2 The replica library is built from the same patched corner text as the card (same bin patch), then
  REPLICA_ZERO is written into every bin; a check that every replica bin carries every REPLICA_ZERO value runs
  before each deck.
- D3 Smoke (after the code commit; not used for decisions): 2 folds, 1 seed, 200 steps, refit max_nfev 20; Part B
  stages on 2 devices with starting cards.
- D4 Order: this prereg commit (alone) -> code + unit tests commit -> smoke -> queue (validity, lodo, deploy, decide;
  one heavy job at a time).

## Outputs

`out/research/explore/I12_1/{grids, calib, validity, lodo, deploy, decision}`; arrays (`*.npz`) not in git; report
batch and site claim after the decision.

## Compute (estimate)

Validity about 4 min; LODO 95 seed tasks (about 100 s each with the penalty) + 18 refits on 6 workers about 30 min;
Part B about 10 min on 4 workers. No other heavy job runs at registration (queue empty since 18:44).
