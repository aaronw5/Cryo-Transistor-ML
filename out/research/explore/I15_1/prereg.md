# I15-1 pre-registration (IN FORCE): re-registration of the I14-1 Part B deployment test of g v2 with the Q1 grid swept in serpentine order; I14-1 replica, module, references and thresholds unchanged; sky130 77 K, 18 devices

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-22), committed alone.

## Why a re-registration

I14-1 (`out/research/explore/I14_1/prereg.md`, outcome in `docs/RESEARCH_LOOP.md`) failed one core check, 'Q1 runs':
two convergence lines ('dynamic gmin stepping failed', 'true gmin stepping failed') in 1 of 126 element decks
(nmos_L8_W1p6, M2 card of g v2, exchanged bench, tight tolerance). Every other core check passed. The I14-1 post hoc
diagnosis attributed the warnings to the channel-current replica (zero junction saturation current). The requested
fix for I15-1 was a replica with small junction saturation currents whose current the element subtracts exactly, or
another justified fix, with a check on all 18 devices before registration that the replica channel current and the
convergence behaviour are unchanged elsewhere. The work below (done before this registration) tested that fix and
alternatives; the registered fix is a change of the Q1 grid sweep order, not of the replica.

## Done before this registration (not committed, not used for any decision)

All runs on the OSDI ngspice-41 build, I14-1 full-data g v2 module, tight / r4 / default tolerances as in I14-1.

1. **Location of the warnings.** The nested Q1 sweep of the two failing decks (nmos_L8_W1p6 I14-1 M2 card: forward
   card + replica, exchanged element) split into one `dc VD` per V(g) line: the warnings occur only at the start of
   the V(g) = 1.85 V line, i.e. at the restart of the inner sweep from V(d) = +1.85 V (end of the V(g) = 1.825 V line)
   to V(d) = -1.85 V. An isolated `op` at (V(g), V(d)) = (1.85, -1.85) V and sweeps that start at that point, at
   1.80 V or from +1.85 V print no warning. Raising the Newton iteration limits (itl2 100, 200, 500; itl1 500 with
   itl2 500) leaves the warnings unchanged.
2. **Replica variants** (forward and exchanged grids x default / r4 / tight x card + replica / element, 18 devices x
   4 card sets: published, start state C, I13-1 v2 held-out C, I13-1 v1 held-out C; 864 decks per variant;
   `explore_replicas/`):

   | replica | decks with warnings (of 864) | nmos_L8_W1p6 M2 (I14-1 failure) |
   |---|---|---|
   | I14-1 (zero junction saturation current) | 0 | warnings (2 decks) |
   | small junction saturation current, card values x 1e-20 | 1 (v2 held-out pmos_L0p35_W5, forward tight element) | no warning |
   | card values x 1e-10 | 1 (start pmos_L8_W5, forward tight element) | not run |
   | card values x 1 | 1 (v1 held-out pmos_L0p5_W0p64, forward tight element) | not run |
   | card values x 1e-60 | not run | warnings |
   | rbodymod = 0 (zero saturation current) | 3 of 1008 incl. AC decks (start pmos_L8_W0p84 forward tight card + replica; v2 held-out nmos_L0p25_W1p6 exchanged tight card + replica and element) | no warning |

   Compensation of the junction current: (a) without compensation the sensed current of the small-saturation-current
   replicas contains the forward-biased drain-junction current in the reverse quadrant (up to 0.039 A at
   V(d,s) = -1.85 V, max |sense - replica channel current| / (1e-6 |I| + 1e-13 A) = 3.5e11); in the forward quadrant
   the sensed current equals the I14-1 one (same maximum, 1.68). (b) Exact subtraction with a second, junction-only
   replica (same card, u0 = 1e-30, its BSIM4 `id` output 0.0) through a current-controlled current source into the
   sense node: the sensed current equals the replica channel current to 5.7e-5 of the Q1 tolerance in the exchanged
   bench, but the forward tight card + replica run on the nmos_L8_W1p6 M2 card aborts ('timestep too small' at
   V(d) = -1.825 V): the sense branch carries the difference of two ~0.04 A junction currents, and the tight
   tolerance (reltol 1e-9) on that branch is below the cancellation error. (c) Routing the replica body current
   through the sense node (body VCVS between the sense node and the body copy; exact drain-junction cancellation by
   Kirchhoff's current law) gives warnings on the nmos_L8_W1p6 M2 forward decks.

   Result against the requested pre-condition: no tested replica change keeps the convergence behaviour unchanged on
   the other devices (each adds warnings in decks where the I14-1 replica has none), and the exact subtraction does
   not converge at tight tolerance. The replica is therefore kept as in I14-1.
3. **Serpentine pre-check** (`precheck/precheck.json`, 89 s): the Q1 grid decks in the I14-1 nested order and in the
   serpentine order of this registration, on the 4 card sets x 18 devices x forward / exchanged x default / r4 /
   tight x card / card + replica / element (1296 decks per order), plus the nmos_L8_W1p6 I14-1 M2 card (disclosed; no
   other M2 card was run):

   | quantity | value |
   |---|---|
   | decks with warnings, nested / serpentine (4 card sets) | 0 / 0 |
   | nmos_L8_W1p6 I14-1 M2 card, nested / serpentine | 2 decks (forward tight card + replica, exchanged tight element) / 0 |
   | serpentine - nested, tight: max \|d\| / (1e-6 \|I\| + 1e-13 A), card / card + replica / sensed / element | 1.69 / 1.69 / 1.74 / 2.10 |
   | same, max relative at \|I\| > 1 nA, tight | 3.4e-5 |
   | same, max relative at \|I\| > 1 nA, r4 / default (element) | 1.0e-3 / 2.0e-3 |
   | wall-time ratio serpentine / nested (card only, default, median) | 1.12 |

   The serpentine - nested differences are of the order of the solver tolerance of each setting; the Q1 checks
   compare element and card + replica runs of the same order and tolerance.
4. **Runtime decomposition** (ro5_A, I14-1 M2 cards, median of 3; ngspice `rusage`): card only 0.24 s, card +
   replica 0.48 s, element 1.13 s; transient load time 0.12 / 0.27 / 0.88 s; element with a module of the same ports
   and a trivial g (1e-3 V(g,s)) 0.52 s. The neural-network evaluation in the OSDI module takes about 0.6 s of the
   0.89 s added by the element. Replica-side reductions measured: unit-gain gate and body copies removed (replica
   gate and body tied to the circuit nodes) 1.10 s (-2.5 %); rbodymod = 0 replica 0.449 s for card + replica (-7 %)
   but adds warnings (item 2). Module-side: tanh written as 1 - 2 / (exp(2x) + 1): 1.09 s; OpenVAF `--target_cpu`
   (native, apple-m1, apple-m2, generic) crashes the compiler. No runtime-reduction variant is registered: the
   largest replica-side reduction without new warnings (gate and body copies removed) changes the ro5_A ratio from
   4.7 to 4.6, and no change of the module evaluation that keeps g identical was found. The decomposition is reported
   with the outcome.
5. Code draft (`src/cryoml/explore/i15_1_serp.py`, `scripts/research/explore/i15_1_chan.py`) and unit tests
   (`tests/test_i15_1_serp.py`).

Not done before: any Part B run on the M2 cards other than nmos_L8_W1p6 (item 3), any M2 refit, any accuracy run,
any training.

## Question

With the Q1 grid swept in serpentine order (every grid point reached from a neighbouring point) and everything else
as I14-1, does the full-data g v2 (I13-1 state C) pass the I12-1 / I14-1 Part B deployment checks in ngspice-41 +
OSDI on all 18 devices?

## Inputs (fixed; I13-1 state C; identical to I14-1)

Records `I13_1/records`; g v2 fold / full corrections, held-out cards and rows `I13_1/lodo`; M_bias v1 fold / full
corrections, held-out cards and rows `I13_1/all18`; C0_refit cards `I13_1/all18/heldout_cards.json`.

## The change (test procedure)

Every Q1 grid deck (card only, card + replica, element; forward and exchanged bench; default, r4 and tight
tolerance; also the default-tolerance grid used for the Q4 runtime) runs one `dc VD` command per V(g) line,
V(g) = 0 to +-1.85 V in 25 mV steps (outer order unchanged), with V(d) ascending (-1.85 to +1.85 V) on even lines and
descending on odd lines; `wrdata` appends each line to one file (`set appendwrite`); the rows are put back in the
nested order (V(d) ascending within each line) before any check (`i15_1_serp.grid_rows`, which rejects a file with a
missing or duplicated grid value). Grid points (75 x 149 = 11175), benches, tolerances, the replica (I14-1
`REPLICA_CHAN` with the gmin VCCS), the module text, the references and every threshold are the I14-1 ones. The AC
decks (op + AC per point), the 1 mV dense gm / gds sweeps, the measured-curve decks and the circuit decks are
unchanged. A grid run with fewer or more rows than 11175 counts as a failed run.

Justification: the warnings are located at the restart jump of the nested sweep (V(d) from +1.85 to -1.85 V at
V(g) = 1.85 V), a step of 3.7 V in V(d) taken in one Newton solve at the tight tolerance; the same bias point converges
without warnings when reached from a neighbouring point or from a fresh operating point. The jump is a property of the
nested sweep command, not of an operating condition of the device or of the circuits. Limitation (stated before the
run, reported with the outcome): a nested sweep with this restart jump at the tight tolerance can print gmin-stepping
warnings for the element deck where the card-only deck prints none (I14-1: 1 of 126 element decks; 1 of 108 card + replica
decks).

## Part B (I14-1 design; checks, thresholds and label unchanged)

Card M2 per device and arm: refit from the starting card (direct43_r3_fd) with the arm's full-data g in the loop (TRF,
max_nfev 400, irls 2, diff_step 0.01, state C); in-sample. Reported: max relative parameter difference of the I15-1
M2 cards vs the I14-1 M2 cards (expected 0, deterministic refit). Q1 equivalence (forward and exchanged bench,
reference I_card +- (exp(g) - 1) I_rep from the card + replica run of the same order and tolerance; per point
|dI| <= 1e-6 |I_py| + 1e-13 A, tight), Q1 runs (0 convergence messages, 0 non-finite values, 0 failed runs in every
element deck: grids, AC, dense), Q1 V(d,s) = 0 (<= 1e-12 A, default and tight), Verilog-A g (2000 points,
`default_rng(20261021)`; <= 1e-9, exchanged <= 1e-12), exchange asymmetry (<= 1e-3), Q2 derivatives vs the element
identity (<= 1e-3), Q2 capacitances (<= 1e-3), Q2 gm sign (0 of 54), Q3 circuits (0 failing only with the element;
I12-1 amendment A1 applies). Restrictions: default-tolerance max relative difference (|I_py| > 1 nA) > 1e-3; >= 1 of
90 gds curves with an introduced sign change; wall-time ratio > 2. Label: any core fail -> not deployable; core pass
and >= 1 restriction -> deployable with restrictions; otherwise deployable (simulator-level).

Reference arm M_bias v1 (full-data g, abs() inputs, same element, its own M2 cards): same stages, label reported, not
the outcome.

### Restriction items (reported, no rule; stage `restrict`)

- Q1 max relative difference at |I_py| > 1 nA per device at default, r4 and tight tolerance (I14-1: default 1.8e-3,
  13 of 18 devices above 1e-3).
- Each gds curve with an introduced sign change (I14-1: pmos_L8_W0p84 and pmos_L8_W1p6 at |V(g,s)| = 1.85 V): the
  number of 1 mV sweep points with dI/dV(d,s) < 0 for the element, the card and the Python product
  I_card exp(g) of the card sweep, and the range of g on the curve.
- Runtime: per-device grid ratios and circuit ratios (Q4), and the ro5_A decomposition (card, card + replica,
  element; wall time and rusage load time) on the I15-1 M2 cards.

## Accuracy (unchanged from I14-1)

Part A is not rerun. Registered flags (I13-1 state C): A1 GMR(M_bias_v2, C0_refit) 0.785 [0.693, 0.896], 16/18, 0
failed: passes; A2 GMR(M_bias_v2, M_bias_v1) 1.053 [0.966, 1.137]: CI upper above 1.05, fails. Element-deck accuracy
of the held-out pairs (v2 and v1) reported without a rule as in I14-1 (measured-curve decks unchanged; expected to
reproduce the I14-1 values).

## Validity (checked first; any fail -> invalid)

| check | pass |
|---|---|
| V1 card-only library deck x exp(fold g) reproduces the 36 I13-1 rows (v2, v1) | max \|RRMS difference\| <= 1e-9 |
| V2 unit tests (I14-1 list + `tests.test_i15_1_serp`) | all pass |
| V3 rows | 18 element-deck accuracy rows per arm; 0 failed M2 refits per arm |

## Decision (I14-1 table)

| condition | outcome | action |
|---|---|---|
| any validity check fails | invalid | stop: fix and re-register |
| Part B (v2) "not deployable" | not deployable | stop; diagnose the failed core checks |
| Part B (v2) "deployable with restrictions" or "deployable (simulator-level)" | deployable, accuracy below v1 (not non-inferior) | stop v2; report the accuracy cost of the deployment constraints |

## Order and compute

This prereg commit (alone) -> code + unit tests commit -> smoke (2 devices, refit max_nfev 20; not used for decisions)
-> queue, one heavy job at a time: I15_1_validity (about 10 s), I15_1_accuracy (about 15 s), I15_1_deploy_v2 (about 4
min), I15_1_deploy_v1 (about 5 min), I15_1_restrict (about 10 s), I15_1_decide. No other heavy job runs at
registration (queue empty since 01:22).

## Outputs

`out/research/explore/I15_1/{precheck, explore_replicas, validity, accuracy, deploy_v2, deploy_v1, decision.json}`;
arrays (`*.npz`) and OSDI binaries not in git; report batch, site claim `i15-1` and RESEARCH_LOOP entry after the
decision.
