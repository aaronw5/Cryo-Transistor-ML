# Goal

**Beat the paper's extraction method with machine learning / deep learning.**

Produce SPICE-compatible cryogenic BSIM4 parameter cards that reduce
paper-exact all-curve RRMS relative to (a) the paper's published parameter
cards and (b) the paper's extraction method re-run in our simulator
(classical least-squares / evolutionary search controls), when everything is
evaluated through the identical corrected-repository NGSpice chain.

## Fairness Requirements

- Use the published 77 K corner files and Volare SKY130 PDK.
- Use native geometry-bin selection; do not select bins from measured scores.
- Tune only `VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0`.
- Optimize, select, and report the paper companion notebook's all-curve RRMS.
- Validate every final parameter vector in real NGSpice.
- Compare extracted cards with paper cards only in the identical simulator.
- Give the classical controls (the paper-method retry) the same simulator
  budget and the same metric as the ML pipeline.

## Required comparisons

| column | what it is |
|---|---|
| paper cards | published cards, no fitting (lower bound on effort) |
| fd | multistart finite-difference least squares — the paper-method analogue |
| cma | CMA-ES + FD polish — stronger classical control |
| ml | emulator + inverse-MLP pipeline (this project's contribution) |

The ML pipeline must beat `fd` and `cma`, not just the published cards.

## Current Result (2026-06-10) — GOAL MET

| method | per-device | deployable |
|---|---:|---:|
| paper cards (no fit) | 0.6919 | 0.6919 |
| fd control (paper-method retry) | 0.5010 | 0.5497 |
| cma control (strongest classical) | 0.4991 | 0.5445 |
| **ml v2** | **0.4917** | **0.5368** |

ML beats the published cards (17/18 devices) and both classical
paper-method retries, per-device (5 strict wins, 13 ties, 0 losses) and
under the deployable one-card-per-bin constraint. Standalone ML (no warm
starts from the controls) scores 0.4938, also ahead of both controls.

Progress log: `docs/RESEARCH_LOG.md`.
