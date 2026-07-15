# cryo-ml continuation instructions

Read `docs/HANDOFF.md` before changing code or launching a job. It is the
authoritative live handoff for the July 2026 rebase.

## Canonical setup

- Upstream: `ogzamour/CryoPDK_Skywater130nm_ML`, commit
  `39b1e518e25120104225b8fa19f4cfc61a6766b3`.
- Simulator: conda-forge ngspice-41 at
  `/Users/anrunw/cryo-ng41/mm/envs/ng41/bin/ngspice`.
- pFET card: the upstream `update_sky130_fd_pr__pfet_01v8_lvt__tt_77k.corner.spice`.
- Metric: the faithful `rrmsCalc.py` port in `src/cryoml/metrics.py`.
- Tuned parameters: `VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0` only.
- Synthetic data: 10,000-point Latin hypercube in the published parameter
  vector's +/-10% box, with all 11 metric curves simulated.

Older June results, README text, research notes, and Claude memories describe a
different corrected-repository/ngspice-46/all-curve setup. They are historical,
not instructions for the current run.

## Non-negotiable reporting policy

Never report a per-device best-of result as the headline ML result.

Materialize and report these three primary fixed comparison series across all
18 devices:

1. `direct_mlp_forward_pass`: measured I-V -> parameters in exactly one MLP
   forward pass, with real-NGSpice validation and no search or FD.
2. `emu_search`: multistart parameter optimization through the frozen
   parameter-to-I-V emulator, before FD.
3. `emu_search+fd`: the same surrogate-search result after numerical FD
   least-squares polish.

The direct MLP implements the Keras-notebook recipe in PyTorch because this
environment already uses PyTorch/MPS: 301 current locations concatenated in
linear and signed-log form (602 inputs), min-max scaling, dense LeakyReLU
blocks, seven normalized parameter outputs, Adam/L2, exponential learning-rate
decay, early stopping, and one inference pass. Do not call it an inverse MLP in
current reporting; its experiment label is `direct MLP forward pass`.

Use `scripts/make_ml_variants.py` for the two surrogate stages. It fails if a
stage is missing rather than substituting another result. Export only the one
uniform `emu_search+fd` method. No per-device method selection may enter
figures, headline RRMS, Table 4, Table 6, or the exported card.

Every I-V figure must show measured points, the NGSpice curve from the paper's
published parameters, and NGSpice curves re-simulated from the fixed ML
parameter predictions (direct MLP, surrogate raw, surrogate + FD). Never plot
the neural emulator's current output as if it were the final physical curve.

The completed `foundation_plus_fd` fixed exploratory series may also appear,
clearly labeled. Retain `high_voltage_guarded` results in diagnostic tables and
data, but do not include the guarded series in plots or slides. Neither enters
the canonical card export. The per-voltage pMOS L=2/W=5 result uses a different
card per curve and is a nondeployable diagnostic only.

Slide policy: keep main presentation comparisons to direct MLP and
surrogate+FD against measured/paper references. Show raw surrogate only on the
dedicated FD-improvement slide. That slide must show both fixed paired tests:
direct MLP -> direct MLP + FD and surrogate raw -> surrogate + FD. Do not add
foundation or per-voltage slides to the simple deck.

Run `scripts/fd_parameter_study.py` as a separate experiment. It must report
published-card -> FD alone and surrogate raw -> surrogate + FD, including
per-device RRMS gains, optimizer effort, and all seven parameter changes. Run
`scripts/direct_mlp_fd_study.py` for the paired direct-MLP -> FD ablation. The
direct MLP stays unpolished in the primary comparison; its polished artifact
is diagnostic and must not replace the raw one-pass series in main results.

Scores used for primary comparisons must freeze curve inclusion to the
published-card baseline's included curve set. Also retain the upstream's
dynamic-inclusion score as a secondary audit value.

Do not redefine RRMS. The high-voltage study changes candidate feasibility,
not the paper metric. FD residuals are evaluated against measured I-V curves
using fresh NGSpice parameter perturbations, never against surrogate output.

Report both aggregate conventions and compare like with like: the upstream
headline `combined_rrms` is the equal-weight mean of the nMOS and pMOS family
means; `all_device_mean_rrms` weights all 18 devices equally. Do not compare a
combined value with the paper's all-device mean.

## Resource policy

Run only one compute-heavy task at a time. In particular, do not overlap:

- `pdk_ml_extract.py`
- `pdk_direct_mlp.py`
- `fd_parameter_study.py`
- `direct_mlp_fd_study.py`
- `scaling_study.py`
- `pdk_gen_data.py`
- bulk NGSpice re-simulation

Check `ps`, the active log, and system load before starting one. Lightweight
source inspection, documentation edits, syntax checks, and unit tests are fine.

## Required final verification

1. `scripts/verify_simulator.py` passes all 18 devices against upstream sweeps.
2. The metric port matches the upstream scorer and unit tests pass.
3. The direct MLP and both fixed surrogate stages contain exactly 18 current
   `lhc10` records.
4. Every table/figure is generated from those fixed series, never a best-of.
5. Exported cards are re-simulated from the written library.
6. The scaling study is from the confirmed setup, not the stale June CSV.
7. The FD studies cover all 18 devices: published start, raw surrogate start,
   and fixed direct-MLP start. Direct MLP + FD currently improves all-device
   RRMS `0.2722522 -> 0.2327984` on 18/18 devices in `373.5 s`.
8. Final scaling is the complete all-18 training-data axis: 375, 750, 1,500,
   3,000, and 6,000 examples per transistor (90 cells). Its plot uses the
   arithmetic all-device mean as a bold line and faded colored traces for
   every transistor. Emulator MSE improves about 9.5x while real-NGSpice+FD
   RRMS stays flat, so the remaining all-18 capacity/search grid was stopped
   by user-approved design. Capacity/search remain a labeled four-device
   pilot; incomplete CSV rows must never enter all-18 means.
9. The completed global conditional emulator must remain exploratory and fixed
   across devices; its current result is `0.2260651` all-device after FD and
   `1200.7 s` first-campaign time. It must not become a best-of selector.
10. Preserve the pMOS L=2/W=5 per-voltage and high-voltage-guard studies as
    diagnostics. Keep official RRMS unchanged and do not export those cards.
11. Update `README.md`, `GOAL.md`, `docs/METHODS.md`, `docs/RESEARCH_LOG.md`,
   `docs/HANDOFF.md`, and Claude project memory with the final numbers.
