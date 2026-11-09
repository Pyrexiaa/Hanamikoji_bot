# Experiment guide and findings

I compare three iterative algorithms: CFR (my Task 3 solver), fictitious play (FP), and double oracle (DO). A sparse sequence-form LP is the exact ground truth and supplies the submitted policies. Earlier CFR variants remain in the implementation and in `../archive/cfr_study/`; they are not part of the current report.

## What one iteration does

| Method | Work in one outer iteration | Returned strategy | How the target is checked |
|---|---|---|---|
| CFR | Compute both players' counterfactual action values; update both sets of regrets simultaneously; derive the next policies through regret matching; accumulate the pre-update realization plans. | Uniform average of the `t` plans accumulated after `t` iterations. | Two extra best responses (BRs) to the average every `check_every` iterations. |
| FP | Compute one full-game BR per player to the opponent's current average; update both empirical mixtures simultaneously. | Average of `t` BRs and one uniform prior observation: denominator `t + 1`. | Free, every iteration: the two BRs answer the current averages, so their values are the exact SPR of the returned profile. |
| DO | Solve the restricted game (two LPs) for `(x_t, y_t)`; call both global BR oracles against it; add any new plans. | Current restricted-equilibrium mixture of realization plans. | Free, every iteration: the oracle values give `SPR(x_t, y_t) = u(BR1(y_t), y_t) - u(x_t, BR2(x_t))`. |

Every BR respects information sets; it cannot observe the opponent's hidden hand. Mixtures and time averages are formed in realization-plan coordinates, preserving own-action reach and flow conservation. Averaging raw behavioral probabilities would generally be wrong.

**2,000 is a maximum number of these outer updates.** It is not 2,000 played games, sampled deals, tree nodes, or LP pivots. It is a chosen computational budget, not a special mathematical constant. Equal outer-iteration limits do not equalize computation, so elapsed seconds and work counts are also reported.

## SPR and the stopping rules

For P1 payoff `x.T @ A @ y`, the **saddle-point residual** is

```text
SPR(x, y) = max_x' x'.T A y - min_y' x.T A y'
          = P1's best unilateral gain + P2's best unilateral gain.
```

Zero means a Nash equilibrium in this two-player zero-sum game, up to floating-point precision. At `SPR <= 0.002`, neither player can improve expected payoff by more than `0.002` through a unilateral deviation. This implementation uses the **full gap**, not half the gap. The payoff is `P(win) - P(loss)` in one round, so `0.002` is not directly a percentage-point win-rate guarantee. Self-play value close to zero does not imply a small SPR.

Each method stops at the first certified profile that meets the target. FP and DO get the certificate for free at every iteration from best responses they need anyway, so their stopping iterations are exact first hits. CFR's average needs two extra BRs per check (about the cost of one CFR iteration), so it is checked at entry, every 25 iterations and at exit; its stop can overshoot the first crossing by at most 24 iterations.

**DO termination criterion.** The restricted game's own gap is ~0 by construction and is never used. DO stops when the oracle certificate is at most the target. If neither oracle returns a new plan, both best-response values are attained inside the restricted game and the restricted equilibrium is an exact full-game equilibrium (`stop_reason = "no_new_strategy"`). In this game exact ties (interchangeable G1/G2, identical split cards) make the oracle return different but equally good plans, so "no new plan" can take far longer than reaching `SPR <= 1e-9`; the analysis therefore calls a profile exact at a certified `SPR <= 1e-9` (LP accuracy).

## Experiment controls

The authoritative common settings are in [experiment_config.json](experiment_config.json). Command-line overrides and the complete resolved configuration are saved in `results/experiments.json`.

| Control | Submitted value | Meaning / effect |
|---|---|---|
| `iterations` / `--iterations` | `2000` | Maximum outer updates per fresh run. |
| `spr_threshold` / `--spr-threshold` | `0.002` | Common full-game stopping target; smaller requires greater accuracy. |
| `check_every` / `--check-every` | `25` | CFR's certificate spacing; for FP it only thins the stored history (DO stores every iteration). |
| `repeats` / `--repeats` | `3` | Fresh timed runs per method per scenario, with rotated method order. These measure runtime variation. |
| `warmup_iterations` | `2` | Discarded warm-up for each iterative method before measurement; the full LP is also warmed once per scenario. |
| `full_budget` / `--full-budget` | `false` | With the flag, disable early stopping and run all requested updates; retain `spr_threshold` for reporting target crossings. |
| `lp_tolerance` | `1e-9` | Full sequence-form reference LP feasibility/optimality request. See DO's separate fixed tolerances below. |
| `scenarios` | neutral, first_player, second_player, favored_g3 | Same single-round starting conditions for every method. |
| `methods` | CFR, Fictitious play, Double oracle | Labels for the implemented benchmark set; actual classes are wired in `METHODS` in the runner. |
| Initial profile | Uniform behavioral policies | CFR begins with zero regrets; FP keeps one uniform prior observation; DO begins with one uniform candidate per player. |
| Chance handling | Exact weighted enumeration | All deals/hidden states are integrated, with no training sampling seed. |
| Payoff scale | win `+1`, loss `-1`, unresolved `0` | SPR is measured in these expected-payoff units. Rescaling payoffs also rescales SPR. |
| Engine | Python, Tiny Hanamikoji | No engine selection or game-size sweep is part of this study. |

`neutral` uses neutral favors and a random starter. `first_player` and `second_player` fix P1 or P2 as starter with neutral favors; they are the two disjoint halves of `neutral` (the starting player is the first chance event and both players observe it) and mirror images of each other. `favored_g3` fixes P1 as starter and already controlling G3; G1/G2 remain neutral. The default tree has 2,757 information sets and 3,672 sequences per player; the fixed-start trees have 1,297/1,460 information sets and 1,902/1,771 sequences for P1/P2 (swapped for `second_player`).

Other implementation settings and numerical checks:

- `CFRSolver` has optional `alternating`, `regret_matching_plus`, and `linear_averaging` flags; **all are false** in this benchmark. Local actions use positive-regret normalization with a uniform fallback.
- Best-response ties choose the first maximizing action in supplied action order. Parent values are accumulated in the same order as the notebook's sequential reversed-infoset pass, so the vectorized BR is bit-identical to it, including at exact ties (tested). FP is sensitive to how ties are broken: with the earlier accumulation order its first certified iteration on the default game was 1,493 instead of 1,576; CFR's trajectory was bit-identical.
- FP's uniform prior has fixed weight one. DO begins with one uniform plan per player, deduplicates exact best-response plan bytes, and skips the LP for the trivial 1-by-1 game. These are explicit algorithm conventions, not tuned per scenario.
- DO uses SciPy `linprog(method='highs')` with primal and dual feasibility tolerance `1e-9`; its IPM optimality tolerance is left at the installed SciPy default. The full reference LP explicitly sets all three to `1e-9`. Changing the config's `lp_tolerance` changes only the full reference, not DO's fixed settings. Neither tolerance is the strategic SPR target.
- When the restricted game has several equilibria, DO follows whichever one HiGHS returns, so DO's trajectory can differ across platforms even with identical package versions (observed between macOS and Linux with the earlier DO code). CFR and FP trajectories are platform-independent in my runs.
- Full LP feasibility tolerances are clamped to at least `1e-10`, and its IPM optimality tolerance to at least `1e-12`, matching the supported lower limits. This does not affect the submitted `1e-9` setting.
- Runner validation requires realization nonnegativity within `1e-12`, maximum flow error below `1e-8`, agreement of logged and independently recomputed SPR within `1e-10`, and reference LP SPR below `1e-7`. These are numerical checks, not training controls. Reloaded exported policies must meet the common SPR target.
- DO's reported active support counts use mixture weights greater than `1e-12`. Stored candidate counts include zero-weight strategies and the initial uniform plan. Only the count report uses this cutoff; no strategy is pruned by it.
- The convergence plot floors displayed SPR at `1e-12` for log axes; stored results and stopping tests retain the actual values.
- Direct `CFRSolver.solve()` still has the starter API defaults `max_iter=1000`, `spr_threshold=0.01`. The benchmark **always passes 2000 and 0.002 explicitly**. FP and DO default to 2000 and 0.002.
- `solve(max_iter=...)` counts additional iterations. Continuing a solver preserves its state; `reset=True` restarts. `None` or a negative threshold disables stopping. `max_iter=0` certifies the existing profile. Fresh instances are used for every benchmark repetition.
- Task 3's guided CFR demonstration deliberately runs all 2,000 iterations with stopping disabled; its final submission export comes from the LP. Game-demo seeds (`6208`, or `6208 + k`) affect notebook play examples only.

## What solve time measures

`wall_seconds` is monotonic wall-clock elapsed time from immediately before the solver constructor to immediately after `solve()` returns. It includes topology preprocessing in the constructor, algorithm initialization and updates, averaging, and the SPR certificates. DO includes restricted payoff expansion, LP solves, and mixtures. It is neither pure CPU time nor time spent playing games.

Tree construction is measured once as `build_seconds` and excluded from all solver timings. Warm-up, post-run independent validation, policy conversion/export, the untimed analysis section, plots, notebook execution, and report generation are excluded. LP reference timing includes sparse assembly, both optimizations, and its internal certificates; its separate final best-response validation is outside the timer.

Solver-internal `elapsed_seconds` covers `solve()` calls only and accumulates on continuation. The benchmark adds construction time to plotted history entries and retains the original internal value as `solve_elapsed_seconds`. Each plotted curve uses that method's median-total-time repetition.

The JSON retains all times, convergence histories, source hashes, hardware/software versions, training and certificate BR calls, regret sweeps, restricted LP solves, candidate/active strategy counts, and an `analysis` section (below). Peak RSS is for the whole benchmark process; it is not a per-algorithm memory comparison.

## Analysis section (untimed, one run each)

- `double_oracle_exact`: DO continued to a certified `SPR <= 1e-9` in every scenario, with the iteration where it first met `0.002`, support sizes, the total time spent in restricted LPs, and `restricted_lp_trace` (restricted-game size and solve time for every restricted solve; Figure 2b).
- `rates`: CFR and FP over the full 2,000 iterations on the default game; log-log slope of SPR from iteration 100, and CFR's last-iterate SPR for contrast with its average.
- `lp_versus_cfr`: for the default and `first_player` games, LP and CFR values, each player's exploitability, cross-pairings (LP P1 with CFR P2 and vice versa), mixing counts, and an LP over P1's optimal face that ranges one first-decision probability to prove the equilibrium is not unique.

## Measured findings

These numbers come from the run behind the current report (`results/benchmark.txt`; Linux, Python 3.12, the locked package versions). Re-running the pipeline regenerates the JSON, figure and report but not this table, so compare it with the new `benchmark.txt`; CFR and FP iteration counts are platform-independent, DO's may shift slightly.

Entries are iterations to the first certified `SPR <= 0.002` / median seconds, with the same 2,000 cap throughout. "DO exact" continues to a certified `SPR <= 1e-9`.

| Algorithm | Neutral/random starter | P1 starts | P2 starts | P1 starts, holds G3 favor |
|---|---:|---:|---:|---:|
| CFR | 1550 / 0.279 | 1550 / 0.167 | 1550 / 0.167 | 675 / 0.074 |
| FP | 1576 / 0.265 | 1576 / 0.157 | 1576 / 0.158 | 1386 / 0.140 |
| DO | 101 / 0.381 | 57 / 0.127 | 59 / 0.133 | 55 / 0.119 |
| DO exact | 117 | 57 | 63 | 68 |
| LP value / s | 0 / 0.040 | +2/105 / 0.021 | -2/105 / 0.021 | +2/35 / 0.023 |

- **Decomposition.** CFR and FP give identical stopping iterations and SPR on the default game and on each half: they act separately on disjoint subtrees, ignore the 1/2 payoff scaling, and the halves are mirror images. DO's restricted game is built from whole-game pure plans whose mixture weights are shared by both halves, so on the default game it needs nearly the sum of the halves' iterations (101 vs 57 + 59), not their maximum. On each half DO is faster than CFR; on the union it is slower.
- **Work.** DO needs ~15x fewer iterations than CFR, but ~90% of its time is restricted LP solving (two LPs per iteration; reading P2's mixture from the duals of P1's LP would roughly halve this). CFR and FP cost about the same per iteration and both decay like `T^-0.67` here.
- **LP vs CFR.** Values agree within CFR's SPR; strategies do not, because the equilibrium set is not a singleton (the LP vertex mixes at 7/0 information sets, CFR's average at 466/193 with P1 starting; one first-decision probability ranges over [0, 0.417] among optimal P1 plans). Cross-pairing isolates each CFR strategy's exploitability.
- **Exactness.** DO reaches an exact equilibrium with only ~12 active plans per player; CFR's last iterate stays at SPR ~0.11 after 2,000 iterations while its average reaches ~0.0017.
- The warmed full LP is about 7x faster than the fastest iterative method on these small games and supplies the submitted policies. None of these results establish Medium Hanamikoji scalability.

## Reproduction and artifacts

In the collaborative repository, I run the command sequence in [../README.md](../README.md). In a grader copy where the ZIP is extracted directly over the supplied starter, the same Task 4 scripts use `task4/` paths, the policies use `tasks/`, and the solution modules use `student_attempts/` (without the `jiayu_attempt/` prefix). That copy still needs the supplied engine and a Python environment with `task4/requirements-lock.txt`; the solution ZIP does not duplicate the engine. The notebooks and wrapper remain in the collaborative repository because the original packager does not collect them. I run `execute_notebooks.py` only in the source checkout; it rejects an extracted archive without the attempt marker, so it cannot accidentally execute the grader's original starter notebooks and overwrite the final policies. Core files:

- `../student_attempts/utility.py`: chance/opponent-weighted posterior.
- `../student_attempts/solver.py`: best responses, full SPR, CFR and optional variants.
- `../student_attempts/fictitious_play.py` and `double_oracle.py`: the additional algorithm families.
- `../student_attempts/lp_solver.py`: independent sparse LP reference with lazy SciPy imports.
- `../tasks/nash_p1.npz`, `../tasks/nash_p2.npz`: final LP policies, reloaded and independently checked.
- `../tasks/my_nash_tiny.json`: both-seat neutral first-round policy table.
- `../report.pdf`: the Task 4 report (Direction D only), at most five A4 pages, 12 pt text and 2.5 cm margins, generated by `build_report.py`. Its sections: introduction (choice of direction and how it extends Tasks 2-3), how each solver works and why it converges, how correctness is verified, experimental setup, results (two figures, three tables), and conclusions.
- `results/experiments.json`, `benchmark.txt`: measurements and analysis.
- `results/convergence.png` (report Figure 1: SPR against iterations and time on the default game) and `results/decomposition.png` (Figure 2: CFR and DO on the default game and its halves; restricted-LP cost against stored plans), both drawn by `plot_results.py`.
- `results/tests.txt`: the unit-test log; four executed notebooks live in `../tasks/`.
- `requirements-lock.txt`: exact package versions used for this run.

For a separate fixed-budget study:

```sh
jiayu_attempt/.venv/bin/python jiayu_attempt/task4/run_experiments.py --full-budget --output jiayu_attempt/task4/results/full_budget.json
jiayu_attempt/.venv/bin/python jiayu_attempt/task4/plot_results.py --input jiayu_attempt/task4/results/full_budget.json --output jiayu_attempt/task4/results/full_budget.png
```

The main report reads `experiments.json` and asserts its stated shared settings. Changing the study configuration requires updating the report narrative too. The runner also regenerates the same LP policy artifacts even when a different result output path is chosen.

To play the exported policy table from the repository root:

```sh
jiayu_attempt/.venv/bin/python tournament.py --variant tiny --policy-table jiayu_attempt/tasks/my_nash_tiny.json
```

That table covers the solved neutral first round. Later-round fallback play has no equilibrium guarantee. The report's baseline scores use exact one-round expectations, not the notebook's seeded multi-round demonstrations.

## Primary references

- Zinkevich et al. (2007), [Regret Minimization in Games with Incomplete Information](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html).
- Brown (1951), [Iterative Solution of Games by Fictitious Play](https://cowles.yale.edu/sites/default/files/2022-09/m13-all.pdf), chapter XXIV.
- Robinson (1951), An Iterative Method of Solving a Game, Annals of Mathematics 54(2).
- Daskalakis and Pan (2014), A Counter-Example to Karlin's Strong Conjecture for Fictitious Play, FOCS.
- Heinrich, Lanctot, and Silver (2015), Fictitious Self-Play in Extensive-Form Games, ICML.
- McMahan, Gordon, and Blum (2003), [Planning in the Presence of Cost Functions Controlled by an Adversary](https://www.cs.cmu.edu/~ggordon/mcmahan-ggordon-blum.icml2003.pdf).
- Bosansky, Kiekintveld, Lisy, and Pechoucek (2014), An Exact Double-Oracle Algorithm for Zero-Sum Extensive-Form Games with Imperfect Information, JAIR 51.
- von Stengel (1996), [Efficient Computation of Behavior Strategies](https://doi.org/10.1006/game.1996.0050).
- Koller, Megiddo, and von Stengel (1996), Efficient Computation of Equilibria for Extensive Two-Person Games, GEB 14(2).
