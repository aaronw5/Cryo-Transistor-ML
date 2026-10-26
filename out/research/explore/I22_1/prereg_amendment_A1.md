# I22-1 amendment A1 (2026-10-26, before any Part C run or result)

Recorded before any Part A or Part C run of the pre-registered configuration (only smoke runs with reduced budgets
under `I22_1/smoke` exist; their values are not used).

Change: Part C TuRBO-1 runs (primary and secondary configuration) stop at the matched budget B_d instead of 4 B_d.
The matched-budget comparison (best-so-far at B_d, the primary Part C statistic and labels) is unchanged. The
secondary "4 B_d" comparison is reported for CMA-ES (runs to 4 B_d) vs the LS max_nfev 1600 trace only.

Reason (measured before any run): BoTorch SingleTaskGP fitting in 43 dimensions takes 1.0 / 3.0 / 5.9 / 16.2 s at
100 / 400 / 800 / 1600 points (single thread, `scratchpad` timing), and a 500-call toy run took 312 s; with B_d up to
404 calls on sky130 (C0_refit n_sim), a run to 4 B_d (up to 1616 calls, about 400 GP fits) takes about 45 min, and
the 108 TuRBO runs about 40 h of CPU time, above the about 6 h limit for Part C. Runs to B_d take about 3-4 min each.

Also fixed in the code before any run: TuRBO and CMA-ES workers run with OMP_NUM_THREADS = VECLIB_MAXIMUM_THREADS = 1
(BLAS threads), set on the queue command line.
