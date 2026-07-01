# Research log

Goal: beat the paper's extraction method (arXiv:2604.21625) with ML/DL,
everything scored with the paper-exact all-curve RRMS in the identical
corrected NGSpice chain. Headline metric: **mean RRMS over the 18 Table-6
devices** (lower is better).

## Scoreboard

| date | method | mean RRMS | wins vs paper cards | notes |
|---|---|---:|---:|---|
| — | paper reported (HSPICE/Mystic) | 0.279 | — | not comparable: different simulator |
| 2026-06-09 | paper cards in NGSpice (baseline) | 0.6919 | — | published params, native bins; ngspice-46 |
| 2026-06-16 | paper cards in NGSpice-41 (exact ogzamour recipe) | 0.7434 | — | conda-forge ngspice=41; WORSE than -46, not closer to paper |
| 2026-06-09 | ML deployable shared-bin cards | **0.5438** | 17/18 | emulator search + FD polish + joint shared-bin polish |
| 2026-06-10 | fd control (paper-method retry) | 0.5010 | 18/18 | per-device cards; multistart FD least squares |
| 2026-06-10 | **cma control** | **0.4991** | 18/18 | per-device cards; CMA-ES + FD polish — STRONGEST CLASSICAL, the bar ML must beat |
| 2026-06-10 | fd control, deployable | 0.5497 | | one card per model bin (joint polish) |
| 2026-06-10 | cma control, deployable | 0.5445 | | ML v1's 0.5438 already edges this by 0.0007 |
| 2026-06-10 | ml v2, per-device | 0.4917 | 18/18 | beats cma 0.4991 & fd 0.5010; 5 strict wins, 13 ties, 0 losses |
| 2026-06-10 | ml v2, standalone (emu_search+fd, no warm starts) | 0.4938 | | pure-ML pipeline alone also beats both classical controls |
| 2026-06-10 | ml v2, deployable | 0.5368 | 17/18 | beats cma-deploy & fd-deploy |
| 2026-06-10 | cma control, 8500 evals (budget-matched) | 0.4981 | 18/18 | 3.5x budget buys 0.001 → classical plateau |
| 2026-06-10 | cma control deployable (8.5k) | 0.5411 | 17/18 | rebuilt; bin-2 joint 0.684 |
| 2026-06-10 | ml v3 active-BO, per-device | 0.4914 | 18/18 | 3 more strict wins (nmos_L20 0.148, pmos_L0p5_W0p64 0.469, pmos_L8_W1p6 0.566); standalone active_bo+fd 0.4928 |
| 2026-06-10 | **ml FINAL (best of v2/v3), per-device** | **0.4911** | 18/18 | vs strongest classical 0.4981 (-1.4%); 3 strict wins, 15 ties, 0 losses after folding in CMA-8.5k warm starts |
| 2026-06-10 | **ml FINAL, deployable** | **0.5363** | 17/18 | vs strongest classical deployable 0.5411 (-0.9%); out/pdk_ml_final, cards exported |

### 2026-06-10 — ML v2 results (goal met)
Strict per-device wins by the scaled emulator search over the best
classical result: nmos_L20_W0p64 0.150 (vs 0.194), pmos_L0p5_W0p64 0.472
(vs 0.477), pmos_L4_W7 0.609 (vs 0.641), pmos_L8_W1p6 0.569 (vs 0.593),
pmos_L8_W0p84 0.564 (vs 0.565). Joint emulator search improved the bin-2
shared card to 0.690 pair-mean (controls: 0.709/0.726); bin-10 pair stays
~0.876 for every method (real deployability cost). Run config: 6k
samples/device (quarter FD-centered), emu 512x4, inverse [1024,512],
2048 starts x 600 steps, 14 validations, 5 polishes nfev 120, warm starts
from both controls. Wall-clock ~62 min for 18 devices on MPS + 12-core
NGSpice.

Per-device detail: `out/tables/comparison.md` (incl. per-family means),
`out/tables/table6.md`. Family split of the final result (per-device /
deployable): nMOS — classical 0.378/0.378, ml 0.373/0.373; pMOS —
classical 0.594/0.671, ml 0.586/0.667. The deployment cost is entirely
pMOS (both shared bins are pMOS).

## History

### 2026-06-16 — Tried the EXACT ogzamour recipe (ngspice-41): does NOT match the paper
The corrected repo (`ogzamour/CryoSkywater130nm_CorrectedForNgspice`) pins
`conda install -c conda-forge ngspice=41`; this project had been using
ngspice-46 and recorded "ngspice 41 not needed". Tested the pinned version
directly: installed conda-forge `ngspice=41` (osx-64 build via Rosetta, the
same binary the recipe produces) and re-ran the 18-device published-card
baseline.

- **ngspice-41 mean RRMS = 0.7434**, vs ngspice-46 **0.6919**, vs paper
  **0.2788**. ngspice-41 is *worse*, not closer to the paper. The
  HSPICE/Mystic → ngspice gap is **not** a simulator-version artifact.
- Cross-version isolation (`scripts/compare_ng_versions.py`,
  `out/tables/ng41_vs_ng46.json`): on **13/18 devices both versions select
  the same native bin and the per-device RRMS is byte-identical (max diff
  0.0e+00)** — the cryo BSIM4 evaluation is version-independent. All 8 nMOS
  match exactly.
- The only differences are the **5 overlapping-box pMOS devices** where the
  two versions tie-break native bin selection differently (e.g.
  pmos_L0p35_W0p55: -46 → bin 10 = 0.931, -41 → bin 5 = 3.070; pmos_L8_W5:
  -46 → bin 2 = 1.458, -41 → bin 1 = 0.852). This reconfirms that "native
  bin selection" is itself ambiguous for the overlapping pMOS boxes (the
  reason we force per-device best bin elsewhere) — it is now shown to be
  *version-dependent*, not just bin-overlap-dependent.
- `verify_simulator.py` under ngspice-41 reproduces the corrected repo's
  saved reference sweeps to the same 3.6e-7 A as ngspice-46 — i.e. both our
  ngspice builds reproduce ogzamour's saved ngspice-41 output identically.
- Conclusion: stay on ngspice-46. The paper's 0.279 remains unreachable
  through any open NGSpice (41 or 46); method-vs-method inside the identical
  chain remains the only fair comparison. ngspice-41 saved to
  `/Users/anrunw/cryo-ng41/mm/envs/ng41/bin/ngspice`; ng46 baseline preserved
  in `out/pdk_baseline`, ng41 baseline in `out/pdk_baseline_ng41`.

### 2026-06-16 — Floor reconciliation: the paper gap is the METRIC, not the sim
Recomputed every method both all-curve and with a device-off floor (curves
whose mean|I| < frac*device-peak counted as 0, applied identically to all),
from each method's saved sims (`scripts/floored_comparison.py`,
`out/tables/floored_comparison.json`):

| method | all-curve | floor 1% | floor 2% | floor 5% |
|---|---:|---:|---:|---:|
| published baseline | 0.692 | 0.181 | 0.166 | 0.141 |
| fd control | 0.501 | 0.098 | 0.090 | 0.078 |
| cma-8500 control | 0.498 | 0.102 | 0.094 | 0.082 |
| ml_final (search) | 0.491 | 0.098 | 0.090 | 0.078 |
| tandem seed0 | 0.494 | 0.104 | 0.096 | 0.083 |
| paper reported | 0.279 | — | — | — |

- Floored, even the published baseline (0.14-0.18) drops BELOW the paper's
  0.279. The all-curve 0.692 is inflated by near-off curves: across all 18
  devices' near-off curves (9,672 pts) **41% are hard 0, plus SMU glitch
  spikes up to 51 uA** that no BSIM4 card can fit. So the "0.69 vs 0.28" gap
  is the all-curve weighting of unfittable noise-floor curves, not HSPICE-vs-
  NGSpice and not ngspice-41-vs-46. (Consistent with the long-known best-bin
  median-floor ~0.282.) Near-off curves are padded to 0 by the instrument,
  NOT to a high sentinel; what blows up is RRMS = rmse/mean|I| (tiny denom).
- The ML win is ROBUST to the metric: under both all-curve and floored,
  ml_final stays at/tied-for best (0.090 floored), still ~2x better than the
  baseline. The advantage is not an artifact of all-curve weighting.
- Caveat: "count near-off as 0" can be gamed (a device with many off curves
  scores low for free), so it is reported ALONGSIDE all-curve, never instead,
  and applied identically to every method.

### 2026-06-16 — Direct IV->params amortized extractor (no-surrogate + surrogate)
New `scripts/pdk_forward_direct.py`: a one-shot curve->7-param network (the
classic amortized inverse / iPREFER-style extractor), trained two ways and
applied once to the measured curve, every prediction NGSpice-validated.
Loss = pure parameter distance (z-MSE); `--recon-weight>0` adds a frozen-
emulator curve-reconstruction term. Early stopping + z-clamp to the inner box
(off-manifold measured inputs otherwise extrapolate to pathological corners,
e.g. 905x current). Compared in `scripts/forward_compare.py`
(`out/tables/forward_compare.json`):

| metric (median/18) | no-surrogate | with-surrogate |
|---|---:|---:|
| synthetic in-dist reconstruction | 0.190 | 0.134 |
| measured one-shot | 1.39 | 0.768 |
| measured one-shot (mean) | 52.8 | 3.95 |
| measured + FD polish | 0.578 | 0.545 |
| measured + FD polish (mean) | 0.579 | 0.516 |

- **In-distribution (synthetic) reconstruction is excellent** (median ~0.13-
  0.19; 14/18 devices < 0.3) — reproduces the literature result. The 4 high
  values (nmos_L0p15, nmos_L8, pmos_L0p35_W5, pmos_L4_W7) are the low-current
  near-threshold devices where RRMS is ill-conditioned, not model failures.
- **Adding the surrogate reconstruction loss helps on every aggregate metric**
  and rescues the catastrophic measured blow-ups (nmos_L1_W1p6 905->0.87,
  pmos_L0p35_W1p6 15.3->4.3) by anchoring predictions to curve-space.
- **But pure one-shot is not usable for extraction on measured cryo data**
  (median 0.77 even with surrogate vs the search pipeline's 0.491): the
  measured curves sit off the clean synthetic manifold (glitch spikes,
  quantization, real leakage outside the 7 params). FD polish from the
  with-surrogate start reaches 0.516 mean, near the controls but not beating
  NGSpice-in-the-loop search. This is exactly why the winning pipeline keeps
  NGSpice in the loop rather than amortized inference.

### 2026-06-16 — Tandem training continued (seeds 3-5): converged to the floor
Ran 3 more confirmatory tandem seeds (identical config, only --seed differs).
6-seed per-device ensemble (min over NGSpice-validated cards) = 0.4921 (was
0.4928 at 3 seeds); best-of-all (tandem ensemble + ml_final) = 0.4910 ≈ the
0.4911 ml_final. The extra seeds tightened but did not break the data/model
floor — confirms the four method families converge ~0.49.
(`scripts/tandem_ensemble.py`, `out/tables/tandem_ensemble.json`.)

### 2026-06-09 — NGSpice binning blocker solved
The long-standing "ngspice can't bin the cryo corner files" failure was a
scale-units issue: corner files set `.option scale=1.0u`, so X-instances
must pass bare micron numbers (`l=0.15 w=1.6`, no `u` suffix). Working
backend: `src/cryoml/spice_pdk.py` (ngspice-46, Volare sky130
`a918dc7c...`). pMOS bin boxes overlap (12 bins / 10 devices), so auto-bin
can pick a bin fit to a different device.

### 2026-06-09 — Baseline validated, old claims audited
Published cards through the PDK chain with per-device *best* bin: 0.2823
mean median-floor RRMS vs paper-reported 0.279 (within 1.2% → benchmark copy
is good). Exact reproduction impossible for pmos 0.35/1.6, nmos 20/0.64,
nmos 100/100 (~2x off even best-bin) — genuine HSPICE-vs-NGSpice gap. An
older "0.221 beats 0.279" claim was retracted: cross-simulator comparison +
over-aggressive curve filtering. Current fairness rules (native bins, all-curve
metric, same simulator both sides) date from this audit.

### 2026-06-09 — ML pipeline v1 (current best: 0.5438)
Per device: ~1,500 NGSpice samples in the native bin → MLP emulator
(z → signed-log curves) + inverse MLP (curves → z) → multistart Adam search
through the frozen emulator (|z| ≤ 3.5) → top candidates re-scored in real
NGSpice → FD least-squares polish → devices sharing a model bin jointly
polished into one card. Stage means over 18 devices:

| stage | mean RRMS |
|---|---:|
| paper cards | 0.6919 |
| inverse MLP raw | 390.19 (unusable alone) |
| inverse MLP + FD polish | 0.6159 |
| emulator search | 0.6011 |
| emulator search + FD polish | 0.5020 (per-device best, not deployable) |
| deployable shared-bin cards | **0.5438** |

Worst devices after v1: pmos_L0p5_W0p42 1.181 (regressed vs 1.098),
nmos_L0p25_W1p6 0.913, pmos_L2_W5 0.963, pmos_L0p35_W1p6 0.754,
pmos_L8_W5 0.739.

### 2026-06-10 — Figures, README, slides
`scripts/make_figs.py` reproduces paper Figs. 2/4/5 + Table 6 with
measured vs paper-cards vs ML overlays; `scripts/make_slides.py` builds
`slides/cryo-ml-77k.pptx` (14 slides, non-ML audience).

### 2026-06-10 — Classical controls launched (paper-method retry)
`pdk_fd_extract.py --workers 8` running: 6-start FD least squares per
device, max_nfev=150, warm start at published params — the closest analogue
of the paper's extraction process inside our simulator. CMA-ES control
queued behind it.

### 2026-06-10 — Diagnostics: where the error actually lives
Per-curve RRMS breakdown of ML v1 sims (all 18 devices):

- The device score is dominated by the **lowest-|Vg| output curves**
  (`idvd @ |Vg|=0.37`, near/below threshold): e.g. nmos_L0p25_W1p6 — that
  single curve scores 9.61 and contributes 0.87 of the device's 0.913;
  pmos_L2_W5 — two such curves are 77% of the error.
- **Emulator val-MSE does not correlate with final RRMS** (best emulator
  0.026 on a 0.576 device; worst 0.92 on a 0.058 device) → surrogate
  capacity is not the current bottleneck; the objective landscape is.
- Killer curves split three ways:
  1. *Spike-corrupted off-curves* (median 0, isolated 6 µA codes):
     irreducible, RRMS ≈ 1/sqrt(spike fraction) for any card.
  2. *Quantization floors*: measured currents quantized at ~6 µA (some
     ranges) — one sweep reads 0 where another reads 6 µA at the same bias.
  3. *Real high-Vd leakage floors* (nmos_L100_W100: flat ~70–83 µA for
     Vg=0.3–0.6 at Vd=1.85, self-consistent across sweep families;
     also nmos_L20_W0p64, pmos_L0p35_W0p55, pmos_L0p5 pair): GIDL-like,
     controlled by BSIM4 params outside the allowed 7, but partially
     mimicable via large ETA0 (DIBL) within bounds. **This is real,
     fittable headroom that no method has exploited yet** — every card
     (published, FD, ML v1) sits 3–4 decades below these floors.
- Parameter pinning at box edges is rare (vsat/rdsw on 4 devices) — bounds
  are mostly not the constraint.

### 2026-06-10 — FD control interim (13/18 devices)
FD (paper-method retry) is nearly matching ML v1 per-device: ML ahead by
0.002–0.05 on most devices, FD ahead on nmos_L0p19_W7 (0.429 vs 0.430) and
pmos_L2_W5 (0.945 vs 0.963). Implication: ML v1's edge over the classical
method is thin; the planned scaling experiments must target the leakage-floor
basins (item 3 above) where global search can win.

### 2026-06-10 — Pipeline audit + scaling prep (ML v2 design)
Audit of v1: emulator MLP [7,256,256,256,P~2046], inverse [P,512,256,7],
512 Adam starts x 400 steps, 8 NGSpice validations, 3 FD polishes (nfev 80),
~1.5k synth samples/device.

Probes:
- ETA0 (DIBL) is numerically inert for long channels (BSIM4 cosh(L/lt)
  term) → nmos_L100_W100's high-Vd leakage floor is unfittable within the
  7 params; bounded headroom ~0.03 there. Short-channel pMOS DIBL is live.
- pmos_L2_W5 over-conducts at Vg=-0.74 (sim nA vs meas ~0) → needs VTH0
  shift, opposite direction from the leakage devices.
- pmos_L0p5_W0p42 per-device ML was already 0.525 (beats FD 0.526) but the
  shared-bin joint polish wrecked it to 1.181; joint optimum for the
  (0p35_W0p55, 0p42) bin-10 pair sits ~0.875 pair-mean even with a new
  joint emulator search → genuine deployability cost, not a search bug.
- Emulator arch A/B (held-out val MSE): 512x4 ≈ 3-4x better than v1 256x3
  on 2/3 testbed devices; capacity chosen for v2: emu 512x4, inverse
  [1024,512].

v2 pipeline upgrades implemented in `pdk_ml_extract.py`:
1. `--emu-arch/--inv-arch` flags; minibatch path for >4k samples.
2. `--starts-from out/pdk_fd[,out/pdk_cma]`: control winners become warm
   starts AND directly-validated candidates → ML ≥ controls per device by
   construction.
3. Trained emulators saved per device; new **joint emulator gradient
   search** on the summed objective now seeds the shared-bin joint polish.
4. Training data v2 (`pdk_gen_data.py --centers-from out/pdk_fd`): 6000
   samples/device, quarter tight around published, quarter around the FD
   winner, quarter wide, quarter uniform box. (~5 min/device, 10 workers.)
5. `make_fd_deploy.py`: FD control under the same shared-bin deployment
   constraint, so deployable-vs-deployable is method-fair.

### 2026-06-10 — Data race found & fixed: 4 devices re-run
Timestamp cross-check showed the v2 extraction overtook the slower second
wave of data generation: pmos_L0p35_W0p55, pmos_L0p5_W0p42, pmos_L0p5_W0p64
and pmos_L8_W1p6 trained on the OLD 1.5k data (results still valid — all
NGSpice-scored — but without the v2 data benefit). Also: the FD-centered
sampling has a much lower ok-rate on these devices (67-95% vs 100%),
because FD winners near box edges produce more failing sims. Re-ran those
4 devices on the full 6k data (out/pdk_ml2 backed up to out/pdk_ml2_bak;
joint shared-bin cards redone — bin-2 group restored from backup if the
redo regresses).

### 2026-06-10 — Budget-matched CMA: classical method has plateaued
CMA-ES with 8,500 evals/device (≈ ML's total simulation budget, 3.5x the
regular control): mean **0.4981** vs 0.4991 at 2,400 evals. Tripling the
classical budget buys 0.001 — the ML margin cannot be attributed to
simulation budget. NOTE: this run replaced out/pdk_cma (the 2,400-eval
per-device records); the 0.4991 summary survives here and in git-less
history only. cma_deploy rebuilt from the 8.5k-eval results.

### 2026-06-10 — New goal: best possible ML vs control + scaling laws
New method implemented (`pdk_ml_active.py`, ML v3 = **ensemble active-BO**):
K=3 emulator ensemble, pessimistic acquisition (mean+std; half the starts
optimistic mean−std for disagreement-driven exploration), 4 active rounds
of search → NGSpice-validate top 8 → append TRUE evals to training set →
fine-tune ensemble. Targets the diagnosed v2 bottleneck (surrogate
exploitation error). Warm starts: fd, cma, ml2 winners (validated round 0
→ incumbent never worse than best known). Round-3 data appended first:
+2k samples/device densified around ML v2 winners (pool now 8k).

Also queued for rigor/science:
- Budget-matched CMA control (8,500 evals/device ≈ ML's total sim budget)
  so the ML win can't be attributed to budget alone.
- Scaling-law study (`scaling_study.py` → `make_scaling_fig.py`): data
  {375..6000} / capacity {64x3..1024x4} / search {128..8192 starts} sweeps
  on 4 testbed devices, standalone pipeline, fixed round-2 pool.
- Figures: log-scale I-V panels removed per user request (fig2 twin axis,
  fig4 row 2, appendix); slides to be refreshed at the end.

### 2026-06-10 — Scaling laws (out/scaling/results.csv, figs/scaling_laws.png)
52-cell sweep, standalone pipeline, 4 testbed devices, single seed:
- **Training data is the dominant axis**: final polished RRMS geo-mean
  0.66 (n=375) → 0.30 (n=6000), with catastrophic failures below ~1k
  samples on some devices (nmos_L20: 1.28 at n=375 vs 0.15 at full data);
  saturates by ~3-6k.
- **Capacity**: emulator val MSE follows a clean power law, ~params^-0.25
  (0.114 at 142k params → 0.025 at 5.3M); final RRMS improves modestly all
  the way to 1024x4 (0.49 → 0.41 on the testbed) — some headroom may
  remain beyond 512x4.
- **Search budget is saturated**: surrogate search loss flat from 128 →
  8192 starts (α≈0.02); consistent with v2→v3 gains coming from data and
  acquisition design, not more search.
- Final extraction error decouples from surrogate quality once the
  surrogate is good enough — the measurement/model floor takes over.

## Literature notes (2026-06-10)

ML-driven compact-model extraction is an active area; our contribution is
the cryogenic + open-simulator-verified setting, not the NN machinery:

- iPREFER (arXiv:2404.07827): feature-based NN parameter extractor for
  BSIM-CMG — amortized inverse like our inverse MLP.
- "A single neural network global I-V and C-V parameter extractor for
  BSIM-CMG" (Solid-State Electronics, 2024): one NN predicting params from
  curves across geometries; motivates a future multi-device inverse net.
- "Scientific machine learning for generic compact model parameter
  extraction" (Eng. Appl. AI, 2025): random-forest + ANN inverse modeling.
- "ML-Based Standard Compact Model Binning Parameter Extraction" (Adv.
  Intelligent Systems, 2026): ML for bin-level extraction — adjacent to our
  shared-bin joint card problem.
- None target cryogenic cards or score across two simulators; the
  simulator-verified (NGSpice-in-the-loop) candidate validation + classical
  controls protocol appears to be our differentiator.

## Experiment queue (ML scaling)

- [x] Audit architectures + budgets; A/B emulator capacity (512x4 wins)
- [x] More synthetic training data (1.5k → 6k, FD-centered quarter)
- [ ] ML v2 full run (out/pdk_ml2): emu 512x4, 2048 starts x 600 steps,
      14 validations, 5 polishes nfev 120, warm starts from FD **and CMA**
      (relaunched after CMA finished at 0.4991)
- [x] CMA-ES control (out/pdk_cma): **0.4991**, 18/18 vs published; beats
      FD control on 11/18 devices; notable basins ML v1 missed:
      pmos_L8_W5 0.648, nmos_L20_W0p64 0.194, nmos_L0p15 0.038
- [ ] FD-deployable control (out/pdk_fd_deploy)
- [ ] If v2 insufficient: emulator ensembles, round-3 active sampling
      around v2 winners, stronger inverse (curve-embedding arch)
