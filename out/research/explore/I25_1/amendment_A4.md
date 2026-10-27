# I25-1 amendment A4 (2026-10-27): non-finite random starts; committed alone before every affected run

Requested by the coordinating session after its check of the arm-A Step A outputs (not a user instruction).

## Observation (arm A, 7 parameters, A3 random starts)

Finite final RRMS per start: start 0 rounds 1 / 2 / 3 / FD 16 / 16 / 16 / 16 of 18 (pmos_L8_W0p84, pmos_L0p35_W5
non-finite); start 1 round 1 15 of 18 (pmos_L0p35_W1p6, nmos_L0p15_W1p6, pmos_L0p5_W0p42); start 2 round 1 13 of 18
(pmos_L8_W0p84, nmos_L0p15_W1p6, nmos_L0p19_W7, nmos_L0p25_W1p6, pmos_L0p35_W5). Start 1 / 2 rounds 2 were running
when the queue was stopped.

Cause (re-simulation of the start cards, ngspice-41 messages): two failure types.
1. Every curve NaN at the start card (start objective = PENALTY 10.0, CMA-ES best 10.0, polish 10.0): BSIM4 fatal
   "u0 at current temperature ... is not positive". The bins carry negative size terms (lu0 / wu0 / pu0, e.g.
   pmos wu0 = -1.3e-9); a u0 drawn in the lower part of the x/÷100 box makes the effective u0 negative and NGSpice aborts
   the card. Around such a start every CMA-ES candidate also aborts, so the objective is flat at 10 and CMA-ES and TRF
   do not move (nmos_L0p15_W1p6 start 1: u0 0.0048, effective u0 -0.025; pmos_L8_W0p84 start 0: u0 3.2e-5).
2. Some included curves NaN (sweep non-convergence) while others are finite (pmos_L0p35_W5 starts 0 and 2,
   pmos_L0p5_W0p42 start 1): the start objective is large (43-105), CMA-ES accepts a card with two failed curves
   (each counted as 10) because the remaining curves improve more; the paper RRMS of that card is non-finite.
The box widths are the registered rule; they are not changed.

## Rule

R1 (start redraw). For start k and device index i, draws j = 0, 1, 2, ... come from one stream
`numpy.random.default_rng(20261027 + 1000 k + i)`, each draw local[active] = rng.uniform(-1, 1, n_active) (j = 0 is the
A3 draw). The start is the first draw whose card simulates with every included curve finite (one NGSpice call per
draw, counted in the device's calls); at most 200 draws; the number of draws is logged.
R2 (eligibility). A start is eligible if its final card (round 3 + FD) has every included curve finite. Per device the
output is the eligible start with the lowest training RRMS among starts 0-2. If no start 0-2 is eligible, starts k = 3,
4, ... (same R1 and chain) run for that device one at a time until one is eligible (at most k = 9); the output is
that start. Logged: eligible starts per device, extra starts.
R3 (execution). Each (device, start) runs the whole registered chain (round 1 from the R1 start, rounds 2 and 3 with
the registered re-centering on that device's previous card, FD polish) as one task; the tasks of an arm share one pool
(no barrier between rounds). Per-device round-2 / round-3 boxes are computed by the same `next_round_spec` rule (it is
per device). Output files keep the A3 layout (`stepA/{seven,full}_s{k}_{r1,r2,r3,fd}.json`).

## Completed runs

Device-start pairs whose j = 0 draw has every included curve finite get the same start under R1, and every stage is
deterministic (fixed CMA-ES seed), so their completed records are reused (checked: the reused pair's start local
equals the R1 start). Pairs whose j = 0 draw is non-finite are rerun from their R1 start. Arm A start 0 is complete;
starts 1 and 2 are complete through round 2 where finished, else rerun in the chain stage. Arm B has not started and
runs entirely under R1-R3. g v2 stages unchanged.
