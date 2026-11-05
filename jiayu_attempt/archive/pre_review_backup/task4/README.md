# Experiment guide and findings

I compare three distinct iterative algorithms: CFR, fictitious play (FP), and double oracle (DO). A separate sequence-form LP serves as a numerical reference and supplies the submitted policies. Earlier CFR variants remain in the implementation and in `../archive/cfr_study/`; they are not the three main algorithms in the current report.

## What one iteration does

| Method | Work in one outer iteration | Returned strategy |
|---|---|---|
| CFR | Compute both players' counterfactual action values; update both sets of regrets simultaneously; derive the next policies through regret matching; accumulate the pre-update realization plans. | Uniform average of the `t` plans accumulated after `t` iterations. |
| FP | Compute one full-game best response per player to the opponent's previous average; update both empirical mixtures simultaneously. | Average of `t` best responses and one uniform prior observation: denominator `t + 1`. |
| DO | Compute both global best responses; add new candidate plans; update the restricted payoff matrix and solve its two LPs when either pool grows. | Current restricted-equilibrium mixture of realization plans. |

Every best response respects information sets; it cannot observe the opponent's hidden hand. Mixtures and time averages are formed in realization-plan coordinates, preserving own-action reach and flow conservation. Averaging raw behavioral probabilities would generally be wrong.

**2,000 is a maximum number of these outer updates.** It is not 2,000 played games, sampled deals, tree nodes, or LP pivots. It is a chosen computational budget, not a special mathematical constant. Here the slowest successful stop is iteration 1,550, so 2,000 is sufficient for the chosen target in the tested scenarios. That does not guarantee sufficiency in another game. Raising the cap only matters when the target has not yet been reached; lowering it can leave a solver above the target. Equal outer-iteration limits do not equalize computation, so elapsed seconds and work counts are also reported.

## SPR and the stopping rule

For P1 payoff `x.T @ A @ y`, the **saddle-point residual** is

```text
SPR(x, y) = max_x' x'.T A y - min_y' x.T A y'
          = P1's best unilateral gain + P2's best unilateral gain.
```

Zero means a Nash equilibrium in this two-player zero-sum game, up to floating-point precision. At `SPR <= 0.002`, neither player can improve expected payoff by more than `0.002` through a unilateral deviation. This implementation uses the **full gap**, not half the gap. The payoff is `P(win) - P(loss)` in one round, so `0.002` is not directly a percentage-point win-rate guarantee. Self-play value close to zero does not imply a small SPR.

All algorithms stop at the first scheduled check satisfying the same target. Checks occur before updates, every 25 outer iterations, and at return. A successful periodic check is recomputed on exit; history keeps only one entry for that iteration, while work counts include both computations. Every SPR evaluation requires two global best responses. Checkpoint spacing can delay detection of convergence and can produce different terminal SPR values; the records do not claim exact first hitting times or permanently monotone improvement.

## Experiment controls

The authoritative common settings are in [experiment_config.json](experiment_config.json). Command-line overrides and the complete resolved configuration are saved in `results/experiments.json`.

| Control | Submitted value | Meaning / effect |
|---|---|---|
| `iterations` / `--iterations` | `2000` | Maximum outer updates per fresh run. |
| `spr_threshold` / `--spr-threshold` | `0.002` | Common full-game stopping target; smaller requires greater accuracy. |
| `check_every` / `--check-every` | `25` | SPR checkpoint spacing; smaller gives finer stopping resolution with more diagnostic overhead. |
| `repeats` / `--repeats` | `3` | Fresh timed runs per method per scenario, with rotated method order. These measure runtime variation. |
| `warmup_iterations` | `2` | Discarded warm-up for each iterative method before measurement; the full LP is also warmed once per scenario. |
| `full_budget` / `--full-budget` | `false` | With the flag, disable early stopping and run all requested updates; retain `spr_threshold` for reporting target crossings. |
| `lp_tolerance` | `1e-9` | Full sequence-form reference LP feasibility/optimality request. See DO's separate fixed tolerances below. |
| `scenarios` | neutral, first_player, favored_g3 | Same three single-round starting conditions for every method. |
| `methods` | CFR, Fictitious play, Double oracle | Labels for the implemented benchmark set; actual classes are wired in `METHODS` in the runner. |
| Initial profile | Uniform behavioral policies | CFR begins with zero regrets; FP keeps one uniform prior observation; DO begins with one uniform candidate per player. |
| Chance handling | Exact weighted enumeration | All deals/hidden states are integrated, with no training sampling seed. |
| Payoff scale | win `+1`, loss `-1`, unresolved `0` | SPR is measured in these expected-payoff units. Rescaling payoffs also rescales SPR. |
| Engine | Python, Tiny Hanamikoji | No engine selection or game-size sweep is part of this study. |

`neutral` uses neutral favors and a random starter. `first_player` fixes P1 as starter with neutral favors. `favored_g3` fixes P1 as starter and already controlling G3; G1/G2 remain neutral. The default tree has 2,757 information sets and 3,672 sequences per player. The fixed-start trees have 1,297/1,460 information sets and 1,902/1,771 sequences. These sizes come from the game, not tuning parameters. The JSON payoff and initialization descriptions document the implemented setup; changing their text alone does not change the engine or solver.

Other implementation settings and numerical checks:

- `CFRSolver` has optional `alternating`, `regret_matching_plus`, and `linear_averaging` flags; **all are false** in this benchmark. Local actions use positive-regret normalization with a uniform fallback. Best-response ties choose the first maximizing action in supplied action order.
- FP's uniform prior has fixed weight one. DO begins with one uniform plan per player and deduplicates exact best-response plan bytes. These are explicit algorithm conventions, not tuned per scenario.
- DO uses SciPy `linprog(method='highs')` with primal and dual feasibility tolerance `1e-9`; its IPM optimality tolerance is left at the installed SciPy default. The full reference LP explicitly sets all three to `1e-9`. Changing the config's `lp_tolerance` changes only the full reference, not DO's fixed settings. Neither tolerance is the strategic SPR target.
- Full LP feasibility tolerances are clamped to at least `1e-10`, and its IPM optimality tolerance to at least `1e-12`, matching the supported lower limits. This does not affect the submitted `1e-9` setting.
- Runner validation requires realization nonnegativity within `1e-12`, maximum flow error below `1e-8`, agreement of logged and independently recomputed SPR within `1e-10`, and reference LP SPR below `1e-7`. These are numerical checks, not training controls. Reloaded exported policies must meet the common SPR target.
- DO's reported active support counts use mixture weights greater than `1e-12`. Stored candidate counts include zero-weight strategies and the initial uniform plan. Only the count report uses this cutoff; no strategy is pruned by it.
- The convergence plot floors displayed SPR at `1e-12` for log axes; stored results and stopping tests retain the actual values.
- Direct `CFRSolver.solve()` still has the starter API defaults `max_iter=1000`, `spr_threshold=0.01`. The benchmark **always passes 2000 and 0.002 explicitly**. FP and DO default to 2000 and 0.002. I use the runner for comparisons rather than relying on mixed API defaults.
- `solve(max_iter=...)` counts additional iterations. Continuing a solver preserves its state; `reset=True` restarts. `None` or a negative threshold disables stopping. `max_iter=0` evaluates the existing profile. Fresh instances are used for every benchmark repetition.
- Task 3's guided CFR demonstration deliberately runs all 2,000 iterations with stopping disabled; its final submission export comes from LP. Task 4 displays the common early-stopping benchmark. Game-demo seeds (`6208`, or `6208 + k`) affect notebook play examples only, not training or the report's exact baseline evaluations.

## What solve time measures

`wall_seconds` is monotonic wall-clock elapsed time from immediately before the solver constructor to immediately after `solve()` returns. It includes topology preprocessing in the constructor, algorithm initialization and updates, averaging, and exact SPR diagnostics. DO includes restricted payoff expansion, LP solves, and mixtures. It is neither pure CPU time nor time spent playing games.

Tree construction is measured once as `build_seconds` and excluded from all solver timings. Warm-up, post-run independent validation, policy conversion/export, plots, notebook execution, and report generation are excluded. LP reference timing includes sparse assembly, both optimizations, and its internal certificates; its separate final best-response validation is outside the timer. All methods are warmed before timing.

Solver-internal `elapsed_seconds` covers `solve()` calls only and accumulates on continuation. The benchmark adds construction time to plotted history entries and retains the original internal value as `solve_elapsed_seconds`. Thus the report's plots and table use matching constructor-plus-solve scope. Each plotted curve uses that method's median-total-time repetition, rather than averaging timestamps from different runs.

The JSON retains all three times, full convergence histories, source hashes, hardware/software versions, training BR calls, diagnostic BR calls, regret sweeps, restricted LP solves, and candidate/active strategy counts. Peak RSS is for the whole benchmark process; it is not a per-algorithm memory comparison.

## Measured findings

Entries below are stopping iteration / median seconds, with the same 2,000 cap and 0.002 target throughout.

| Algorithm | Neutral/random starter | Neutral/P1 starter | P1 starter and holds G3 |
|---|---:|---:|---:|
| CFR | 1550 / 0.303 | 1550 / 0.198 | 675 / 0.074 |
| FP | 1500 / 0.292 | 1500 / 0.156 | 1400 / 0.174 |
| DO | 125 / 0.591 | 75 / 0.203 | 50 / 0.101 |

DO is most iteration-efficient here but pays for restricted LPs: the neutral run performs 250 training BR calls plus 252 LP solves and ends with 126 stored candidates per player, of which 23 per player have positive weight above the reporting cutoff. CFR instead performs 3,100 regret sweeps; FP performs 3,000 training BRs. Additional diagnostic BRs are counted separately.

Neutral FP and CFR timing ranges overlap, so the small median difference is not decisive. CFR is fastest when P1 starts with G3. DO reaches near-zero SPR in the neutral fixed-start scenario, which exceeds the required accuracy; equal stopping thresholds do not force identical terminal gaps. The warmed full LP is faster on these small fully enumerated games (medians 0.041, 0.021, 0.022 seconds) and supplies the submitted policies. None of these results establish larger-game scalability.

## Reproduction and artifacts

In the collaborative repository, I run the command sequence in [../README.md](../README.md). In a grader copy where the ZIP is extracted directly over the supplied starter, the same Task 4 scripts use `task4/` paths, the policies use `tasks/`, and the solution modules use `student_attempts/` (without the `jiayu_attempt/` prefix). That copy still needs the supplied engine and a Python environment with `task4/requirements-lock.txt`; the solution ZIP does not duplicate the engine. The notebooks and wrapper remain in the collaborative repository because the original packager does not collect them. I run `execute_notebooks.py` only in the source checkout; it rejects an extracted archive without the attempt marker, so it cannot accidentally execute the grader's original starter notebooks and overwrite the final policies. Core files:

- `../student_attempts/utility.py`: chance/opponent-weighted posterior.
- `../student_attempts/solver.py`: best responses, full SPR, CFR and optional variants.
- `../student_attempts/fictitious_play.py` and `double_oracle.py`: the additional algorithm families.
- `../student_attempts/lp_solver.py`: independent sparse LP reference with lazy SciPy imports.
- `../tasks/nash_p1.npz`, `../tasks/nash_p2.npz`: final LP policies, reloaded and independently checked.
- `../tasks/my_nash_tiny.json`: both-seat neutral first-round policy table.
- `../report.pdf`: five A4 pages, 12 pt text and 2.5 cm margins.
- `results/experiments.json`, `benchmark.txt`, `convergence.png`: measurements and figure.
- `results/tests.txt`: all 35 tests passing; four executed notebooks live in `../tasks/`.
- `requirements-lock.txt`: exact package versions used for this run.

For a separate fixed-budget study:

```sh
jiayu_attempt/.venv/bin/python jiayu_attempt/task4/run_experiments.py --full-budget --output jiayu_attempt/task4/results/full_budget.json
jiayu_attempt/.venv/bin/python jiayu_attempt/task4/plot_results.py --input jiayu_attempt/task4/results/full_budget.json --output jiayu_attempt/task4/results/full_budget.png
```

The main report intentionally reads `experiments.json` and asserts its stated shared settings. Changing the study configuration requires updating the report narrative too. Full-budget curves answer accuracy-after-equal-outer-updates; the current report answers time/work-to-a-common-accuracy target. DO iterations can be much more expensive as candidate pools grow. The runner also regenerates the same LP policy artifacts even when a different result output path is chosen.

To play the exported policy table from the repository root:

```sh
jiayu_attempt/.venv/bin/python tournament.py --variant tiny --policy-table jiayu_attempt/tasks/my_nash_tiny.json
```

That table covers the solved neutral first round. Later-round fallback play has no equilibrium guarantee. The report's baseline scores use exact one-round expectations, not the notebook's seeded multi-round demonstrations.

## Primary references

- Zinkevich et al. (2007), [Regret Minimization in Games with Incomplete Information](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html).
- Brown (1951), [Iterative Solution of Games by Fictitious Play](https://cowles.yale.edu/sites/default/files/2022-09/m13-all.pdf), chapter XXIV.
- McMahan, Gordon, and Blum (2003), [Planning in the Presence of Cost Functions Controlled by an Adversary](https://www.cs.cmu.edu/~ggordon/mcmahan-ggordon-blum.icml2003.pdf).
- von Stengel (1996), [Efficient Computation of Behavior Strategies](https://doi.org/10.1006/game.1996.0050).
