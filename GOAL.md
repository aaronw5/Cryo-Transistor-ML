# Goal

Produce SPICE-compatible cryogenic BSIM4 parameter cards that reduce
paper-exact all-curve RRMS relative to the paper's published parameter cards
when both are run through the identical corrected-repository NGSpice chain.

## Fairness Requirements

- Use the published 77 K corner files and Volare SKY130 PDK.
- Use native geometry-bin selection; do not select bins from measured scores.
- Tune only `VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0`.
- Optimize, select, and report the paper companion notebook's all-curve RRMS.
- Validate every final parameter vector in real NGSpice.
- Compare extracted cards with paper cards only in the identical simulator.

## Current Result

The ML pipeline produces deployable shared-bin parameter cards and evaluates
them against all 18 Table-6 devices in the identical corrected NGSpice flow.
The current cards reduce mean paper-exact RRMS from `0.69187` to `0.54376`.
