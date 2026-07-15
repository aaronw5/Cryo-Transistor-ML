# Collaboration conventions

This repository is shared by people and coding agents. Read `CLAUDE.md` and
`docs/HANDOFF.md` before starting a run or changing the experiment protocol.

## Shared assets

Source, tests, generated experiment outputs, figures, and method documentation
are repository-wide assets. Name them for the method or purpose, not for the
person or agent that created them. Anyone may use, review, or extend them.

## Experiment integrity

- Do not mix the historical June setup with the confirmed July setup.
- Do not use any per-device best-of as a reported method.
- Report the direct MLP, surrogate raw, and surrogate + FD as fixed series
  across the complete device set.
- Keep the direct MLP as a one-pass/no-FD comparison and report the separate
  published-start and paired-surrogate FD controls.
- Keep RRMS exactly as defined by the confirmed upstream scorer. Diagnostic
  constraints may change candidate selection but must not redefine the metric.
- Keep primary curve inclusion frozen to the published-card baseline.
- Validate learned parameters and exported cards in real NGSpice.
- Plot real NGSpice re-simulations of paper and ML-predicted parameters, never
  the neural emulator output as the final I-V curve.
- Run only one compute-heavy job at a time and inspect running processes first.
- Treat capacity/search scaling as a four-device pilot. The final all-device
  result is the complete five-point training-data axis (90 cells across all 18
  transistors). Do not resume the intentionally stopped grid or include
  incomplete rows in all-18 means unless the experiment protocol is explicitly
  changed again.
- Keep the completed shared foundation emulator fixed across all 18 devices,
  label it exploratory, and do not use it for per-device selection or export.
- Treat separate per-voltage cards as nondeployable diagnostics. The
  high-voltage guard is also diagnostic and must remain separate from the
  canonical surrogate+FD card. Keep guarded results out of plots and slides.
- Record setup commits, command lines, aggregate convention, and output paths
  when adding an experiment.

## Working logs

Verified conclusions belong in neutral project documentation, especially
`docs/RESEARCH_LOG.md` and `docs/HANDOFF.md`. Agent-specific scratch notes may
remain separate, but they are not authoritative and must not replace the
shared handoff.

Do not use agent-specific names for executable code or experiment output
directories.
