# I20-1 pre-registration (IN FORCE): BSIM4 A2 clamp state of the NIST decks, audit and fix; selected g v2 recipe on both NIST DEV datasets

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-23), committed alone, before any I20-1 RRMS value, rerun, training or S3_D value.

DEV data only. Every TEST split stays locked: the NIST 4 K lvt TEST (I8-1 split, 38), the NIST 4 K room-temperature
card TEST (I9-1 split, 58), IHP SG13G2 TEST and TU Delft TEST. Every loaded tag is checked against the DEV list of
its split file.

## Done before this file (not committed, not used for any decision)

- Code draft: `bsim4_state` argument ("clamped" / "legacy" / "unclamped") of `i8_1_nist.simulate` and
  `i9_1_nist_rt.simulate`, `MeasProblem.bsim4_state`, `i8_1_nist.bsim4_clamp_params` (calls
  `spice_pdk.bsim4_clamp_params`), `i9_1_nist_rt.effective_a1a2` / `bsim4_clamp_params`, and
  `src/cryoml/explore/i20_1_nist_clamp.py` (pure helpers).
- Mechanism check without RRMS: fresh-process `op` model check (b4check.c messages, no clamp written) of the stock
  card and the stored starting card of every DEV device. I8-1: 16 of 57 starting cards with A2_eff > 1, 0 stock
  cards; I9-1: 35 of 87 starting cards with A2_eff > 1 (34 with the base a2 > 1; the la2 / wa2 / pa2 terms of the
  room-temperature pMOS bins move one card over 1), 0 with A2_eff < 0.01, 0 stock cards. The A2 value printed by
  b4check.c equals the computed A2_eff on all 51 clamped cards (relative difference <= 1e-5, print precision); no A2
  message on the other cards. Other b4check.c lines on these cards are warnings without a clamp (Eta0 < 0,
  Nfactor < 0, Pdibl2 < 0, Cdscd < 0, TNOIMOD) and the known fatal "Pclm is not positive" (I8-1 amendment A1).
- Timing of g v2 training (one member, 40 steps, I9-1 state-L records with a synthetic 73 x 73 penalty grid, not
  used for any decision): 0.038 s per step at lambda 0 and 0.24 s per step at lambda > 0 with 86 training devices.
- Known results (DEV): I8-1, I9-1, I9-2, I18-1, I19-1 outcomes and the committed I19-1 TEST gate decisions
  (a755e18): NIST lvt apply (S3_D -0.1027), NIST RT card apply (S3_D -0.0689), both with the M_bias v1 arm.

## Part A: audit (states L, C, U)

Datasets: `nist_4K` (I8-1, 57 DEV devices, 77 K sky130 lvt cards, deck 77 K), `nist_rt_card_4K` (I9-1, 87 DEV
devices, sky130 room-temperature tt cards, deck 30 C).

States (I13-1 definitions): **L** the stored deck (one ngspice process for all sweeps of a device; b4check.c clamps
the size-dependent A2 only in the first analysis); **C** the clamp written into the card (A2_eff > 1 -> a2 = 1,
la2 = wa2 = pa2 = 0, and a1 = la1 = wa1 = pa1 = 0 if A1_eff != 0; A2_eff < 0.01 -> a2 = 0.01, la2 = wa2 = pa2 = 0);
**U** one `op` before the curves (every curve with the card values; audit only). An unclamped card is written
unchanged in state C (identical deck text).

Effective A2 per card: 77 K bins have no la1 / wa1 / pa1 / la2 / wa2 / pa2 terms (asserted by
`spice_pdk.bsim4_clamp_params`), so A2_eff = bin a2. Room-temperature bins: b4temp.c binning
A2_eff = a2 + la2 Inv_L + wa2 Inv_W + pa2 Inv_LW with Leff = L + xl - 2 dl, Weff = W / nf + xw - 2 dw (lint, wint,
ll, lw, lwl, lln, lwn, wl, ww, wwl, wln, wwn of the bin) and binunit 2 (Inv_L = 1 / Leff).

Arms rescored (stored cards and stored corrections; no retraining, no refit):

| dataset | arms |
|---|---|
| nist_4K | stock, start (= C0, no refit), C0_refit, T1_inloop, M_bias v1 (I8-1 lodo); B1, B2, B3_d3 (I8-1 baselines); I9-2 classes gp, const, lin, poly2, poly4, poly5, gam, mlp8, mlp32, mlp16_s1 (I9-2 nist_4K) and mlp16_rep (I9-2 replicate) |
| nist_rt_card_4K | stock, start (= C0), C0_refit, T1_inloop, M_bias v1, M_bias_lvt (I9-1 lodo); B1, B2, B3_d3 (I9-1 baselines) |

Not rescored: I9-2 lc_nist_4K (no stored cards). The I18-1 / I19-1 NIST rows are the I8-1 / I9-1 M_bias v1 and
C0_refit rows (no separate cards).

Per arm and state: per-device strict paper RRMS (frozen measured inclusion of the dataset), differences C - L and
U - L, all-device mean, combined (mean of the nMOS and pMOS family means), median, number of devices changed; per
card: A2_eff, clamp class (high / low / none), b4check.c lines. GMRs per state (device bootstrap 10000, seed 0):
M_bias v1 vs C0_refit, B3_d3 vs C0_refit, M_bias v1 vs B3_d3. Reported: the other b4check.c clamps on
size-dependent parameters (ckappas / ckappad < 0.02, prwg < 0, rdsw < 0, rds0 < 0, rdswmin < 0, xn < 3, lc < 0)
for every stored card; if a stored card triggers one, the fix is extended to it by an amendment before the reruns.

Audit validity:

- **V1** state L reproduces the stored row of start, C0_refit and M_bias v1 for every device (|diff| <= 1e-12, NaN
  in both counts as equal). For the other arms V1 is reported per arm; an arm that does not reproduce is listed as
  "not reproduced" and its C / U values are not reported as audit values.
- **V2** C = L = U (difference 0) for every card with 0.01 <= A2_eff <= 1.
- **V3** on every clamped card: A2_eff equals the b4check.c printed value (relative 1e-5); no A2 message on the other
  cards; the state-C first curve equals the state-L first curve (difference 0) where the first curve is finite.

## Part B: fix

Both NIST simulators run in state C by default (`bsim4_state="clamped"`); "legacy" reproduces the stored decks.
Unit tests (`tests/test_i20_1_nist_clamp.py`): (1) stock cards and cards with 0.01 <= A2_eff <= 1 give the
unchanged parameters and identical deck text, and identical simulations in C and L (one stock card per dataset);
(2) a card with A2_eff > 1 is written with A2_eff = 1 and A1_eff = 0 (77 K and room-temperature, incl. la2 / wa2 /
pa2 terms); (3) the low clamp; (4) the computed A2_eff equals the b4check.c value for one room-temperature pMOS card
with size terms; (5) an unknown state raises; (6) the amendment trigger, selection blocks, frozen-rule decision.
The TEST run of the frozen method uses state C.

## Part C: reruns in state C (queue, one job at a time)

1. Starting records of both datasets re-simulated in state C from the stored starting cards (no refit of the
   starting cards). Required: records of unclamped starting cards identical to the stored records.
2. C0_refit and M_bias v1, LODO, I8-1 / I9-1 recipe unchanged (M_bias 3-16-16-1, 5 seeds, 2000 steps; held-out refit
   `ngspice_fit.polish` max_nfev 400, irls 2, diff_step 0.01; strict paper RRMS; I8-1 amendment A1: the 4 failed
   starting records are excluded from training and scored as failed in every arm), on the state-C records with
   state-C simulation. T1_inloop and the baselines are not rerun (audit values only).
3. S3_D, v1 recipe (I19-1 definition: M_bias v1, seed 0, 500 steps, leave-one-device-out over all finite DEV
   records, applied to the starting record without refit), on the state-C records.

## Part D: selected g v2 recipe on both NIST DEV datasets (state C)

Recipe (method-freeze draft 1.C): I = I_card x exp(g); symmetric inputs (is_pmos, x1 = (a + b + m) / (2 VDD),
x2 = m / VDD), smooth abs EPS 0.01 V, **VDD 1.8 V** (NIST supply); 3-16-16-1 tanh, zero-initialized last layer,
5 seeds (0-4), Adam 3e-3, 2000 steps, L2 1e-4, E1 loss; penalty P on a **73 x 73 grid, 0 to 1.8 V in 25 mV steps**
(forward quadrant of each training device's state-C starting card, u_d > 0, forward current > 1e-9 A); lambda grid
{0, 1, 2, 5, 10} V; training records in load order; held-out refit with g frozen (Part C settings).

**Selection of lambda (deviation from the I15-2 per-outer-fold selection, for compute).** The I15-2 selection runs
3 inner folds x 5 lambdas per outer fold: N x 15 single-member trainings and 5 N (N - 1) inner refits (nist_4K 795 /
13,780; nist_rt_card_4K 1,305 / 37,410), estimated above 30 h on 8 workers. Here the finite DEV devices are split
into **6 selection blocks** (sorted by polarity, nMOS first, then tag; device i -> block i mod 6). For block j: the
training devices are all finite DEV devices outside block j; inner folds by the I15-2 rule (sorted by polarity, tag;
i mod 3); per inner fold and lambda one member (seed 0) trained on the other inner folds (load order); every device
of the inner fold refit with it (state C); score = mean ln RRMS over the inner held-out devices (non-finite counts
as 0); lambda*_j = argmin, ties within 1e-12 to the larger lambda. A held-out device h uses lambda*_h = lambda* of
its block (no data of h enters its selection). Full-data lambda*: the same selection over all finite DEV devices;
the 5-seed full-data model at lambda*_full is stored (for TEST; not scored on DEV).

**LODO.** For each finite DEV device h: the 5-seed g v2 model at lambda*_h trained on all other finite DEV starting
records (load order); held-out refit with g frozen; strict paper RRMS = the "selected g v2" row. Only lambda*_h is
trained per fold.

**S3_D of the g v2 recipe** (I19-1 definition): per device k, one member (seed 0, 500 steps, lambda = lambda*_k)
trained on all other finite DEV records, applied to record k without refit; S3_D = mean ln ratio.

## Part E: contrasts and labels

- **Primary (per NIST dataset):** GMR(selected g v2, C0_refit), both state C, device bootstrap (10000, seed 0),
  paired over the finite-start devices (nist_4K 53, nist_rt_card_4K 87). Label (I17-2 rule): "gain over C0_refit
  replicated" if CI upper < 1 and 0 failed selected / C0_refit rows among the finite-start devices; "worse than
  C0_refit" if CI lower > 1; else "inconclusive".
- **Secondary:** GMR(selected g v2, M_bias v1), both state C, margin 1.05: "non-inferior to M_bias v1" if CI upper
  <= 1.05, else "not shown non-inferior to M_bias v1".
- Reported: GMR(M_bias v1, C0_refit) state C vs the stored state-L value; all-device / combined / median per arm;
  per-device rows; per family; lambda* per block and full-data lambda*; negative gm / gds fractions on the grids;
  NGSpice calls and wall time.

## Part F: gate, amendment rule, S3_D recipe of the frozen method

**Amendment rule (I19-1 Part P, commit a755e18).** For each NIST dataset the decision is recomputed from (a) S3_D,
v1 recipe, state C (Part C.3) and (b) S3_D, g v2 recipe, state C (Part D). If any of these four decisions differs
from the committed decision (both committed: apply), a dated amendment
(`out/research/explore/I19_1/test_gate_amendment_<date>.{json,md}`) is committed before any TEST run. It lists the
changed decisions with their S3_D values and statistic, and the decision of the frozen-method rule below; the
committed files of a755e18 are not edited and stay in the record. If no decision changes, no amendment is written
and the a755e18 decisions stand (the TEST arm per dataset is fixed at the method-freeze commit).

**S3_D recipe of the frozen method (open item 9), fixed here before any NIST I20-1 value.** S3_D is computed with
the v1 recipe (I19-1 v1 definition: M_bias v1, seed 0, 500 steps, leave-one-group-out, starting records in state C)
for every dataset, whichever correction the gate applies; the gate applies the selected g v2 correction iff this
S3_D < 0. Reasons: (1) the threshold 0 and the calibration of S3 (I7-1 OLS on 20 simulated datasets; I19-1 Part C,
19/20) exist for the v1 recipe only; (2) the v1 statistic needs no lambda selection. Known when this rule was
chosen (retrospective): on the 4 DEV rows with selected g v2 results (sky130 77 K, IHP 300 K, TU Delft 4.13 K /
295.0 K) the sign of the v1-recipe S3_D (-0.178, +0.311, -0.029, -0.051) equals the realised sign of ln
GMR(selected g v2, C0_refit) (GMR 0.743, 1.255, 0.826, 0.829) on 4/4; the g v2-recipe S3_D (-0.133, +0.308, +0.008,
+0.264) on 2/4. The two NIST datasets add 2 rows checked after this rule: for each, whether sign(S3_D v1, state C)
and sign(S3_D g v2) equal the sign of the realised ln GMR(selected g v2, C0_refit). Reported as k/2 per statistic,
no threshold.

## Validity of the reruns

V4 unit tests pass; V5 rows complete (C0_refit, M_bias v1 and selected g v2 rows for every finite-start device;
6 x 15 + 15 selection trainings per dataset; S3_D for every finite record); V6 TEST closed (every loaded tag in the
DEV list of its split and not in the TEST list); V7 penalty grids 73 x 73, finite, forward corner current > 0;
V8 state-C starting records of unclamped starting cards identical to the stored records. Any failure: outcome
"invalid", stop, fix and re-register.

## Order

prereg (alone) -> code, unit tests, smoke (reduced settings, not used for decisions) -> queue: audit, start records
(C), S3_D v1 (C), C0_refit / M_bias v1 LODO (C), penalty grids, block selection, full-data selection, S3_D g v2,
g v2 LODO nist_4K, g v2 LODO nist_rt_card_4K, decide, figures -> amendment (if triggered) -> report batch, site,
RESEARCH_LOOP, freeze draft.

Compute estimate (8 workers, one job at a time): audit about 10 min; Part C about 25 min; penalty grids about 3 min;
selection about 2.5 h; S3_D g v2 about 30 min; g v2 LODO about 10 h (5 x 140 members at 300-500 s); total about
13-14 h wall.
