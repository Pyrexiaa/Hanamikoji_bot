# Task 4: Nash solvers and CFR improvements

Both experiments use `student_attempts/solver.py` and `student_attempts/lp_solver.py`. Use the Python environment and repository-root working directory described in [../README.md](../README.md).

## Equilibrium quality

For realization plans x and y and chance-weighted Player 1 payoff matrix A, the full saddle-point residual is `max_{x'} x'^T A y - min_{y'} x^T A y'`, over feasible plans. It sums the two unilateral improvement opportunities; there is no division by two. A near-zero self-play payoff alone does not establish equilibrium.

LP uses the same treeplex/payoff data with a separate maximin/minimax optimization method. Primal/dual feasibility, agreement of game-value bounds and best responses check the LP solution. Exhaustive engine traversal separately checks the payoff representation.

## Direction D: alternative solvers

CFR uses local counterfactual regrets and averages realization plans. Fictitious Play simultaneously best-responds to the previous empirical mixtures, retaining one uniform prior observation. Double Oracle solves a restricted zero-sum game and adds full-game pure best responses. It begins with uniform candidate plans and reuses an unchanged restricted solution while continuing global SPR checks.

| Scenario | Initial condition | Sequences P1 / P2 | Infosets P1 / P2 |
|---|---|---:|---:|
| neutral | Neutral favors, uniformly random starter | 3672 / 3672 | 2757 / 2757 |
| first_player | Neutral favors, P1 starts | 1902 / 1771 | 1297 / 1460 |
| second_player | Neutral favors, P2 starts | 1771 / 1902 | 1460 / 1297 |
| favored_g3 | P1 starts and holds G3 favor | 1902 / 1771 | 1297 / 1460 |

The protocol is 20,000 iterations, three repetitions, two warm-up iterations and rotated method order: 36 full iterative runs, plus separately timed LP references. Exact weighted chance enumeration is used. CFR records every 25 iterations; FP/DO record every iteration. The first recorded SPR <= 0.002 is retrospective and never ends a full-budget run. No crossing is interpolated.

`wall_seconds` includes solver construction, updates, averaging, SPR checks and in-memory logging. Tree construction, warm-up, file export and subsequent validation are excluded. Iterations contain different work across methods. Separate DO runs stop at SPR <= 1e-9 and retain restricted-LP size/time traces; these are distinct from the main experiment. Starting-player decomposition, LP-versus-CFR comparisons, empirical convergence fits and constrained-LP non-uniqueness analyses remain in `experiments.json`.

## Direction C: CFR ablation

| Method | Regrets | Strategy average | Update schedule |
|---|---|---|---|
| CFR | Signed cumulative | Uniform | Simultaneous |
| CFR-linear | Signed cumulative | Linear | Simultaneous |
| RM+ | Clipped cumulative | Uniform | Simultaneous |
| RM+-linear | Clipped cumulative | Linear | Simultaneous |
| CFR+ | Clipped cumulative | Linear | Alternating |

Clipping is `R <- max(R + r, 0)`. Linear averaging weights realization plans by iteration t; regret increments are unweighted, so CFR-linear is not full LCFR. Both players' plans at the start of the iteration enter the average. With alternation, P1 regrets update first and P2 updates against P1's refreshed plan.

**CFR+ here denotes the zero-delay variant with pre-update averaging.** Holding this averaging point fixed makes the fifth comparison isolate update order conditional on clipping and linear averaging. It is not a claim that the finite-iteration average equals the pass-wise opponent average in [Tammelin's Algorithm 1](https://arxiv.org/abs/1407.5042). No averaging-delay experiment is included. Training retains the stored payoff entries, including zeros; the sparse LP removes zero entries from its matrix without changing the payoff function.

All three cases choose the starter uniformly: neutral favors; advantage (G1 favors P1); contested (G1 favors P1 and G3 favors P2). Each of five configurations runs for 20,000 iterations and three repetitions, after two warm-up iterations and with rotated method order. Fourteen checkpoints are `1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000`: 45 runs and 630 observations.

`elapsed_solve_seconds` sums segmented solve calls including entry/exit SPR checks. It excludes tree construction, reference LPs, separate current-strategy diagnostics, exports and plotting. Its timing boundary differs from Direction D. Repeated deterministic residuals measure the same algorithmic trajectory; repetitions characterize timing variation.

## Commands and outputs

From the repository root, using `$projectPython` from the setup instructions:

```powershell
& $projectPython -B final_attempt/task4/run_experiments.py --iterations 20000 --repeats 3 --check-every 25 --spr-threshold 0.002 --full-budget
& $projectPython -B final_attempt/task4/plot_results.py
& $projectPython -B final_attempt/task4/run_cfr_ablation.py --iterations 20000 --repeats 3 --warmup 2
& $projectPython -B final_attempt/task4/plot_report_figures.py
& $projectPython -B final_attempt/task4/validate_submission.py
& $projectPython -B final_attempt/task4/verify_evidence.py
& $projectPython -B final_attempt/task4/extract_report_data.py
```

All outputs reside in `results/`:

- `experiments.json` and `histories/*.jsonl.gz`: main and subsidiary Direction D measurements, source hashes and full intermediate histories.
- `summary.json` and `convergence.csv`: Direction C definitions, LP checks, every checkpoint, averaged/current SPR, best-response bounds and timings.
- `convergence.png` and `decomposition.png`: Direction D convergence and structural/restricted-LP analyses.
- `convergence_iterations.png` and `convergence_time.png`: Direction C checkpoint medians across repetitions. The driver generates these; `plot_report_figures.py` redraws the same data with 12 pt labels at report width and shared vertical limits.
- `cfr_plus_p1.npz` and `cfr_plus_p2.npz`: experimental averages, separate from submitted LP policies in `tasks/`.
- `validation.json`: numerical checks of the submitted policies. Tests and validators print their results to the console; notebook outputs are saved in the notebooks themselves.

`extract_report_data.py` prints the three numerical tables and computed comparisons directly from the primary datasets. It needs no execution logs or document-production metadata. `experiment_config.json` supplies Direction D's defaults.

Direction D figures select the repetition with median full runtime; values below 1e-8 share a disclosed visual floor. Table crossing times are separate medians. Exact values remain in the primary files.

The five test modules cover posterior inference, exhaustive responses/payoffs, flow, all CFR configurations, explicit alternating-update arithmetic, averaging, continuation/reset, FP, DO and experimental protocols. The evidence checker validates 36 histories, 630 observations, source hashes, seven scenario LPs, short solver replays and policy exports. It prints a summary without writing files.

These results concern single-round Tiny Hanamikoji. They establish neither multi-round equilibrium nor scalability to larger games. Three repetitions limit fine runtime comparisons, and LP/best-response checks remain numerical.
