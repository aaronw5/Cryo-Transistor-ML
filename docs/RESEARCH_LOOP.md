# Cold-start research loop (started 2026-10-15)

Goal (user): fit the 77 K data as well as possible — better than the paper
(all-device RRMS 0.2038 with 43 parameters, 0.2284 with seven; released card
0.2751) — and later at other temperatures. ML preferred, not required.

All numbers are all-device mean RRMS with the frozen published-baseline curve
inclusion unless stated; NGSpice re-simulation is always the final score.

## Environment change

The pinned x86_64 conda-forge ngspice-41 no longer runs (macOS 27, no
Rosetta). A native arm64 build of the same ngspice-41 release is at
`~/cryo-ng41/arm64/bin/ngspice` and is preferred by `cryoml.config`.
It reproduces the reference scores exactly
(`out/research/reference_scores.json`):

| card | all-device | nMOS | pMOS |
|---|---:|---:|---:|
| released (Beall et al.) | 0.27505 | 0.1198 | 0.3992 |
| paper, seven parameters | 0.22844 | 0.0788 | 0.3481 |
| paper, 43 parameters | 0.20380 | 0.0704 | 0.3105 |
| cold start: every released multiplier = 1 | 11.6006 | 0.6847 | 20.3333 |

`verify_simulator.py` agrees with the upstream sweeps to <=1e-11 A
(`out/tables/simulator_verification_arm64_ng41.json`).

## What "cold start" means here

The released 77 K card writes each tuned value as `{base * multiplier}`;
the base is the stock SkyWater value.  The paper's banks, starts and fallback
were all centered on the released (multiplied) card.  The cold loop instead
centers round 1 on the multiplier-free card and searches log-uniformly:
vth0 within x3, the other six paper parameters within x100, the remaining
sensitivity-screened parameters within x3 (`src/cryoml/cold_start.py`).
Each later round re-centers every device's box on its selected,
NGSpice-validated card; interior coordinates halve their log width, edge
coordinates keep it so the box moves outward (same rule for all devices).
The surrogate is always trained from random initialization.

## Methods under test

- Paper pipeline (cold): bank -> surrogate -> multistart surrogate FD ->
  NGSpice validation (`scripts/research/cold_loop.py`).
- A, tandem autoencoder: the study script's `inverse_autoencoder`, run in
  every cold round.
- B, conditional flow-matching posterior and K-head winner-takes-all model in
  a round-independent log-multiplier coordinate, pooled over rounds
  (`src/cryoml/posterior.py`, `scripts/research/posterior_study.py`).
  Evaluation: K samples -> NGSpice -> best -> optional NGSpice polish.
  Measured curves are out of distribution for any bank, so training uses
  discrepancy-transfer augmentation (another device's measured-vs-closest-
  simulation signed-log gap, never the device's own) and samples are clipped
  to the device's prior support.
- Non-ML reference: CMA-ES + TRF directly on NGSpice
  (`scripts/research/ngspice_direct.py`).

Heavy jobs run one at a time via `scripts/research/queue_runner.sh`
(`out/research/queue.txt`, `done.txt`, `logs/`).

## Results log

(appended per round)

## Temperature plan (user direction, 2026-10-15)

User: for other temperatures, simulate several temperatures to build the
dataset rather than waiting for measurements. Facts that shape this:

- The 77 K card has `tnom=-196.15` and zeroed temperature coefficients
  (kt1 = ute = 0), so it cannot be extrapolated; changing `.temp` on it does
  nothing meaningful.
- The stock SkyWater card (tnom = 27 C) carries real BSIM4 temperature
  coefficients and is itself a fit to 300 K silicon, so it serves as the 300 K
  anchor; the 77 K measurements are the cold anchor.

Plan: banks sweep `.temp` (e.g. 77, 100, 150, 200, 250, 300 K) as an input
alongside the card parameters, including the BSIM4 temperature coefficients
(kt1, kt2, ute, ua1, ub1, uc1, at, prt) as learnable parameters.  Train
temperature-conditioned models (I-V, T -> parameters; and a T-aware
surrogate).  Target: one card at tnom = 27 C whose temperature coefficients
reproduce the stock card at 300 K and the measurements at 77 K.  Caveat:
BSIM4's temperature laws are not built for cryogenic range, so intermediate
temperatures are a model prediction, not validated, until measured data at
another temperature exist.

## Results: seven parameters, cold start (2026-10-15)

All scores re-checked independently with `scripts/research/verify_cold_results.py`
(`out/research/verify_seven_r4.json`); they match the pipeline exactly.

| method (fixed procedure, all 18 devices) | all-device | combined | nMOS | pMOS | NGSpice calls/device |
|---|---:|---:|---:|---:|---:|
| released card | 0.2751 | — | 0.1198 | 0.3992 | — |
| paper, seven (hot) | 0.2284 | — | 0.0788 | 0.3481 | — |
| paper, 43 (hot) | 0.2038 | — | 0.0704 | 0.3105 | — |
| cold paper pipeline, round 1 | 0.2367 | 0.2251 | 0.1211 | 0.3291 | 10k bank |
| cold paper pipeline, round 2 | 0.1921 | 0.1820 | 0.0917 | 0.2723 | 20k cumulative |
| cold paper pipeline, round 3 | 0.1758 | 0.1660 | 0.0774 | 0.2545 | 30k |
| cold paper pipeline, round 4 | **0.1697** | **0.1602** | 0.0746 | 0.2457 | 40k |
| direct NGSpice (CMA-ES + TRF), round-1 box | **0.1676** | **0.1575** | 0.0665 | 0.2484 | 3.3k (291 s wall) |
| A: tandem autoencoder, round 4 | 0.6857 | 0.6859 | 0.6874 | 0.6843 | 0 |
| B: flow posterior, best of 256 + polish (r1) | 0.3719 | — | — | — | ~700 |
| B: 16-head WTA, best of 16 + polish (r1) | 0.5122 | — | — | — | ~400 |

Observations:

- Seven parameters, cold, beat the paper's 43-parameter hot result
  (0.2038). The paper's own boxes (±10% of released) were the binding
  limit, not the method: round-1 cold boxes allow ×100.
- Which candidate the paper pipeline keeps (NGSpice-validated among FD
  endpoints, top bank samples, fallback): rounds 1–4 = FD endpoint
  11, 8, 5, 3 of 18. By round 4 the winner is usually a raw NGSpice bank
  sample, so the iterated loop behaves mostly like an NGSpice
  cross-entropy search; the surrogate's FD contributes less each round.
  (The paper's own vb_seven/vb_full runs also kept bank samples on 11/18.)
- Direct NGSpice matches the 4-round pipeline with ~12× fewer
  simulations. Many direct parameters sit at the ×0.01/×100 box edges
  (eta0, vsat, rdsw, delta), and rdsw reaches ×2·10^4 of released on some
  devices: fits may rely on non-physical compensation. Needs a physicality
  check before any card is presented.
- A and B are far behind on measured data. B recovers simulated
  parameters well, so the gap is the simulation-to-measurement mismatch,
  not the model.

Queued next: 43 parameters by direct NGSpice seeded from the seven-parameter
fit and fully cold; a 3× longer seven-parameter direct run (convergence
check).

## Held-out simulated devices: simulation-vs-measurement gap (user request)

Every method also runs, unchanged, on held-out simulated devices: for each
of three sets, one card per paper device drawn log-uniformly from the round-4
seven-parameter box (clipped into the round-1 box), simulated in NGSpice on
that device's measured bias grid (`scripts/research/make_sim_targets.py`,
`out/research/sim_targets/`). No bank contains these cards. The targets
are realistic: a true card scores a median RRMS of about 0.45 against the
real device it imitates. `CRYOML_TARGET_OVERRIDE` substitutes them for the
measurements in `load_device_curves` and in every bank `meas` reader; bias
grids, frozen curve inclusion and the metric are unchanged (true card scores
exactly 0). `scripts/research/simgap_table.py` builds the table: the real
all-device RRMS, the mean sim RRMS over sets, and median parameter error.
Low sim RRMS with high real RRMS = model-form/measurement gap; high sim
RRMS = the method itself fails to invert the simulator.

Caveat: the pipeline's sim column reuses its round-4 stage (surrogate and
bank centered on the real-device fits; targets are drawn in that box), so it
is optimistic relative to a full four-round run per target. Direct NGSpice
and B use only round-1 information.

## Results: gap table, 43 parameters, matched budgets (2026-10-15 11:30)

Sim = mean over three held-out simulated-device sets (18 devices each).
Re-scored independently (`verify_direct.py`, `verify_cold_results.py`).

| method | real all-device | real combined | real median | sim all-device | sim median | param err (sim) |
|---|---:|---:|---:|---:|---:|---:|
| cold paper pipeline, round 4 (surrogate+FD) | 0.1697 | 0.1602 | 0.1585 | 0.0157 | 0.0141 | ×1.04 |
| A: tandem autoencoder, round 4 | 0.6857 | 0.6859 | 0.6344 | 32.41* | 0.0258 | ×1.05 |
| direct NGSpice, seven, round-1 box | 0.1676 | 0.1575 | 0.1460 | 0.0191 | 0.0000 | ×1.00 |
| B: flow r1, best of 256 + polish | 0.3719 | 0.3514 | 0.2670 | 0.3812 | 0.1839 | ×1.91 |
| B: WTA r1, best of 16 + polish | 0.5122 | 0.4912 | 0.3509 | 0.6932 | 0.2468 | ×2.09 |
| direct NGSpice, 43, seeded from seven fit | **0.1201** | **0.1126** | 0.1071 | pending | | |
| direct NGSpice, 43, fully cold | 0.1269 | 0.1186 | | | | |

\* one device (nmos L1/W3) explodes to RRMS 500–1200 in two sets; the median
device is fine.

Reading the gap:

- NGSpice-in-the-loop methods (pipeline, direct) invert the simulator almost
  perfectly (sim ≈ 0.015–0.02, parameters within 4%). Their real-data error
  of ~0.17 is therefore the seven-parameter model-form/measurement gap, not
  an optimization failure — which is why 43 parameters cut it to 0.12.
- A inverts simulations well on the median device but collapses on measured
  curves: its problem is purely simulation-to-measurement transfer.
- B is as bad on simulations as on measurements: the model itself fails to
  invert the round-1 (×100) prior. Next: pool rounds 1–4 (≈650k cards),
  150 epochs.
- B's NGSpice polish rarely moves (TRF terminates in ~25 sims): its best
  samples sit in local basins; only CMA-ES escapes them.

Other runs:

- Seven-parameter direct with 3× budget (7.9k calls): 0.16708 vs 0.16757 —
  converged; seven parameters cannot go much below ~0.167.
- Matched budget (1k CMA + 300 TRF): bank-sample starts 0.1777 (1505 calls)
  vs flow-best-sample start 0.1782 (1416 calls, plus the flow's 257): the
  flow start gives no saving.
- 43-parameter direct fits leave ~6 of 43 parameters per device at the box
  edge (mostly secondary ones held to ×3: dwg, k3b, pdiblc2, xj, prwg, k2).
  Round 2 re-centers with the cold-loop rule (running).

Temperature note: the stock SkyWater PDK is local under
`data/raw/pdk/volare/.../sky130A`. The 77 K fits move nfactor ×26 from stock;
BSIM4 has no temperature law for nfactor (and u0 ×6.6 would need ute ≈ −1.4),
so a single 77–300 K card needs temperature-dependent parameter expressions
(ngspice evaluates `temper` in model expressions), not just stock temperature
coefficients.

## Uniform NGSpice FD polish for every method (user request, 2026-10-15 17:30)

Bug found first: `ngspice_fit.polish` used SciPy's relative `diff_step`/`xtol`;
a start at the box center (local = 0, as in B's ×1.5 polish box) stopped
after one rejected step without moving. Fixed with the absolute-step local
Jacobian of `scripts/real_fd_polish.py` (`xtol=ftol=gtol=1e-10`). Direct
NGSpice runs were unaffected (they polish from CMA-ES endpoints).

Every method's cards then get the identical final stage
(`scripts/research/polish_cards.py`): bounded TRF on fixed-inclusion per-curve
NGSpice residuals against the target curves, ~400 simulations/device budget,
in the method's own search box (pooled B: union of the round 1–4 boxes,
`union_spec.py`; a round-1-only box clipped its cards). Real and all three
held-out simulated sets.

| method + NGSpice FD | real all-device | real combined | real median | sim all-device | sim median | FD calls/device |
|---|---:|---:|---:|---:|---:|---:|
| cold paper pipeline, seven, round 4 | **0.1643** | **0.1548** | 0.1513 | 0.0003 | 0.0000 | 270 |
| direct NGSpice, seven | 0.1670 | 0.1570 | 0.1446 | 0.0190 | 0.0000 | 187 |
| B flow pooled r1–4, best of 256 | 0.1728 | 0.1632 | 0.1662 | 0.0942 | 0.0001 | 248 |
| B flow r1, best of 256 | 0.2209 | 0.2067 | 0.1664 | 0.1546 | 0.0000 | 239 |
| B WTA pooled, best of 16 | 0.2268 | 0.2242 | 0.1658 | 0.0697 | 0.0000 | 251 |
| A tandem, round 4 | 0.2270 | 0.2245 | 0.1746 | 0.6112* | 0.0000 | 298 |
| B WTA r1, best of 16 | 0.2587 | 0.2396 | 0.1512 | 0.1108 | 0.0001 | 248 |
| direct NGSpice, 43, seeded (round 1) | 0.1191 | 0.1117 | 0.1071 | 0.0073 | 0.0000 | 319 |
| direct NGSpice, 43, round 3 | **0.1044** | **0.0976** | — | not run | — | 314 |

\* the nmos L1/W3 outlier from A's raw prediction survives FD.

Before → after FD (real all-device): pipeline 0.1697→0.1643, direct-7
0.1676→0.1670, B flow pooled 0.2922→0.1728, B flow r1 0.3801→0.2209,
B WTA pooled 0.4608→0.2268, A 0.6857→0.2270, B WTA r1 0.5148→0.2587,
direct-43 0.1201→0.1191, direct-43 r3 0.1049→0.1044.

Reading: after the same FD, the ML paper pipeline (surrogate) is the best
seven-parameter method, slightly ahead of direct NGSpice. FD rescues A and B
substantially (all recover the median simulated device exactly), but their
real-data means stay behind because FD is local: devices where the proposal
lands in the wrong basin stay there. Pooling rounds for B moves flow+FD from
0.221 to 0.173. With 43 parameters, direct NGSpice rounds 1→2→3 give
0.1201→0.1094→0.1049 (re-scored independently). Seven-parameter model-form
floor ≈ 0.164; 43 parameters is where the remaining gain is.

Next: cold 43-parameter paper pipeline (rounds 1–4) is running, so the ML
pipeline can be compared at 43 parameters.

## Cold 43-parameter paper pipeline, round 1, and why it trails direct NGSpice

Round 1 (`cold_loop.py --set full`, 10k bank/device, ~40% non-converged
samples in the ×100 box): pipeline 0.2375 all-device (combined 0.2281); tandem
A 0.5922. Direct NGSpice in the same box: 0.1269 cold, 0.1201 seeded.

Difference between the two procedures: the paper pipeline spends its NGSpice
budget on a 10k-card random bank, trains a neural surrogate (params → I–V),
searches on the surrogate (8 starts, surrogate FD), and uses NGSpice only to
check the final candidates. Direct NGSpice spends ~3.3k simulations/device
inside the optimizer (CMA-ES then TRF), all near the solution.

Diagnostic (`scripts/research/surrogate_gap.py`,
`out/research/surrogate_gap_full_round_1.json`, inference only): the round-1
surrogate predicts paper RRMS 0.203 for its own picks (NGSpice 0.2375) and
**0.507** for the direct-fit cards (NGSpice 0.127). On 17/18 devices it ranks
its own pick better while NGSpice ranks the direct card better. The good cards
sit on the box faces (max |local| ≈ 1), where the LHC is sparse and failures
cluster. So at 43 parameters the bottleneck is surrogate accuracy where it
matters, not the search. (At seven parameters the surrogate was accurate:
predicted 0.235 vs NGSpice 0.2367 in round 1.)

Queued: `surrogate_arch_study.py` on the round-1 cache: paper 4×1024 MLP,
6×2048 MLP, residual MLP (27M, 139M), DeepONet (branch ResMLP, trunk over
bias point), and a conditioned FNO along each curve's sweep axis
(`src/cryoml/operators.py`; truncated-DFT layers as GEMMs, verified equal to
FFT to 4e-7). Same split, loss and recipe; metrics are held-out loss,
emulator curve NRMSE, and whether the surrogate predicts/ranks the known
NGSpice scores of the pipeline and direct-fit cards. Learning curves at
1/8, 1/4, 1/2 of the data separate "too small" from "too little data".

## Simulated temperature benchmark (77–300 K)

Only 77 K measurements exist, so temperature methods are scored on simulated
devices (`src/cryoml/temperature.py`, `temp_fit.py`,
`scripts/research/temp_bank.py`, `temp_study.py`).

* The released 77 K card cannot be heated: with `tnom = -196.15 C` the
  junction model gives ~0.8 A at 150 K and NGSpice hangs at 300 K. With
  `tnom = 27 C` all 18 devices simulate cleanly from 77 to 300 K.
* ngspice-41 evaluates `temper` in model-parameter expressions, and
  `option temp=` inside one control block re-evaluates them, so all seven
  temperatures run in one process (identical to separate decks; 0.12 s for
  7 temperatures × 11 curves).
* Card: each paper parameter p(T) = p_stock · m^(w(T)^g),
  w = (27 C − T)/(223.15 K), clipped at 0. p(77 K) = p_stock·m,
  p(300.15 K) = p_stock (checked: equals the stock card to 3e-11 A). 14 numbers
  per device: log m and log g for the seven parameters. BSIM4's own
  temperature coefficients stay zero as released.
* 77 K multipliers anchored on the cold seven-parameter round-4 box (the real
  device neighbourhood); bank = 10k LHC/device in 1.5× that box with
  g ∈ [1/4, 4]; truths = 3 sets × 18 devices in 0.75× with g ∈ [1/3, 3].
* Observed temperatures 77, 150, 225, 300 K; held out 110, 190, 260 K.
  Score = frozen-inclusion paper RRMS vs the simulated truth at each
  temperature (all-device mean and combined), plus parameter error.
* Methods (same box, bank starts, NGSpice FD polish): direct NGSpice
  (CMA-ES 1500 + TRF 400), temperature-conditioned surrogate + gradient
  multistart + NGSpice validation (+FD) for paper MLP, ResMLP, DeepONet
  (temperature in the trunk) and FNO, and a one-pass direct MLP (+FD as a
  diagnostic).

Reference re-check after the `spice_pdk.py` multi-temperature change:
released card 0.27505 / nMOS 0.1198 / pMOS 0.3992 (unchanged); simulator
verification passes.

## NGSpice-swap controls for every surrogate ablation (2026-10-16, queued)

Each surrogate study gets a control row in which the surrogate is replaced by
NGSpice and nothing else changes: same 8 diverse bank starts, same TRF
2-point search (max_nfev 100, diff_step 0.02, reg 1e-4), same NGSpice
validation of top-4 endpoints + 4 starts + fallback, same FD polish. The gap
to that row is pure surrogate error.

- Pipelines: `scripts/research/swap_search.py` calls
  `finite_difference_surrogate_extract` unchanged with an NGSpice evaluator
  (round trip vs bank ≤5e-6 relative; failed points penalized at curve
  RRMS 10). Queued: seven r1, full r1, and FD polish of full r1 for both the
  swap and the pipeline.
- Architecture study: every trained model also runs that search end to end
  (`"end_to_end"` in its report; CPU spawn pool, identical results to the
  sequential path); the full-r1 swap row is the shared ceiling, also for
  the learning curves.
- Temperature study: the surrogate search is now the same pipeline search
  (Adam multistart removed, since it cannot be matched on NGSpice);
  `temp_study.py ngspice_search` is the swap row.
- Correction: direct NGSpice 43-parameter fits used ~21.9k NGSpice calls per
  device (TRF Jacobian evaluations included), not ~3.3k.

## Cold 43-parameter paper pipeline, rounds 1-4 (finished 2026-10-16 05:39, 12.4 h)

| round | surrogate+search all-device | combined | nMOS | pMOS | forward val loss | A (tandem) all-device |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.2375 | 0.2281 | 0.1435 | 0.3128 | 0.0322 | 0.592 |
| 2 | 0.1965 | 0.1875 | 0.1058 | 0.2691 | 0.0207 | 5.83 |
| 3 | 0.1716 | 0.1631 | 0.0865 | 0.2396 | 0.0113 | nan |
| 4 | **0.1513** | **0.1427** | 0.0652 | 0.2201 | 0.0085 | 1.70 |

Beats paper-43 (0.2038) and the cold seven-parameter pipeline r4 (0.1697),
still behind direct NGSpice 43 r3 (0.1049). Surrogate accuracy improves
every round as the box narrows and convergence rises. A is unstable at 43.

## Gradients through NGSpice (2026-10-16 scoping)

- Per-call cost ~31 ms: 5 ms process start, 8.5 ms library parse, ~11 ms
  Newton solving, ~6 ms output and Python I/O.
- `src/cryoml/spice_batch.py`: `simulate_pdk_multi` (altermod, bit-identical
  to separate calls on all 18 devices, failure-isolated) and
  `ngspice_jacobian`. The 43-parameter forward Jacobian takes 1.37 s → 0.65 s exact
  (0.30 s with the near-exact one-circuit "parallel" mode). `.sens` is wrong
  for delta/rdsw/eta0/pclm/dwg and in saturation/subthreshold; unusable.
- Differentiable BSIM4 port scoping: v4.8.2, rdsmod=0 but rbodymod=1 and
  rsh give 5 internal nodes, so the port needs a batched Newton solve plus
  implicit gradients. About 600 lines of b4ld DC + 1,150 lines of b4temp (half
  binning boilerplate). Estimate 2-3 weeks; parity is limited to ~1e-4 by
  ngspice tolerances. Not started.
- Side finding: BSIM4 divides u0 by 1e4 when u0 > 1 (cm² units). The
  cold round-1 box (u0 ×100) crosses that for ~14% of nMOS samples, a
  10^4 mobility cliff inside the box.

## NGSpice-swap and FD results (2026-10-16)

Real measurements, all-device (combined):

| method | raw | +FD | NGSpice calls/device |
|---|---:|---:|---:|
| pipeline 7, round 1 (surrogate search) | 0.2367 | | bank 10k |
| same search, NGSpice instead of surrogate, 7, r1 | 0.1887 (0.1767) | | 1.9k |
| pipeline 43, round 1 | 0.2375 (0.2281) | 0.1824 (0.1730) | bank 10k |
| same search, NGSpice instead of surrogate, 43, r1 | **0.1125 (0.1053)** | **0.1106 (0.1036)** | 19.9k |
| pipeline 43, round 4 | 0.1513 (0.1427) | 0.1465 (0.1380) | 4 × 10k banks |
| direct NGSpice (CMA+TRF) 43 r1 cold / r3 | 0.1269 / 0.1049 | — / 0.1044 | 21.9k / round |

The surrogate costs 0.125 RRMS at 43 parameters and 0.048 at seven in round 1.
The pipeline's 8-start TRF search on NGSpice beats CMA-ES+TRF in the same box.
The earlier claim that "the seven-parameter surrogate is accurate" held only for its own picks.

Simulated column, pipeline 43 r4 + FD: set 1 0.113. Sets 0 and 2 have mean
4.9 and 5.0 from one device (nmos_L1_W3 RRMS ~87, the same device that
exploded for A). Median ≈ 0.09-0.10. The 7-parameter sim truths may lie
outside the 43-parameter round-4 box. This needs a check.

Temperature benchmark (3 sets × 18 simulated devices, well-specified), observed
/ held-out all-device: direct CMA 0.0129/0.0362 (+FD 0.0015/0.0117);
NGSpice pipeline-search 0.0005/0.0021 (+FD 0.0000/0.0004); paper-MLP
surrogate search 0.0051/0.0081 (+FD 0.0000/0.0007). 110 K is the
hardest held-out temperature. Parameter error stays 0.16-0.27 (local RMS) even
at RRMS 1e-4, so the g exponents are weakly identifiable. The
well-specified task is too easy to separate methods, so it needs a misspecified tier.

## Research loop iteration 1 (2026-10-16, AI-Scientist-style)

Ideation lenses: method designer, literature scout, ICML reviewer. Nodes
implemented and adversarially verified:
- n1a: sketched-Jacobian TR (k NGSpice directional probes chosen from the
  surrogate Jacobian, Broyden secant, full batched Jacobian on demand); the
  r_s = 0 control is a random-subspace/Broyden Gauss-Newton TR.
- n1b: budget-aware allocation (zero-call surrogate presearch, racing with
  optimistic-bound drops, surrogate restarts).
- n1c: DEV harness (36 targets = sim sets 0 and 1, budgets 500/1000/2000/5000)
  with CMA-ES+TRF and lq-CMA-ES call-trace baselines and the pipeline r1 row.
81 DEV jobs queued, about 6-9 h. The real-device hybrid runs were removed from the
queue: real devices are TEST data and run once per frozen method. Tree:
out/research/tree/tree.json.
