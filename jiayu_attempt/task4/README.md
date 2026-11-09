# Experiment guide and findings

I compare three iterative algorithms: CFR (my Task 3 solver), fictitious play (FP), and double oracle (DO). A sparse sequence-form LP is the exact ground truth and supplies the submitted policies. Earlier CFR variants remain in the implementation and in `../archive/cfr_study/`; they are not part of the current report.

## What one iteration does

| Method | Work in one outer iteration | Returned strategy | Stored certificate |
|---|---|---|---|
| CFR | Compute both players' counterfactual action values; update both sets of regrets simultaneously; derive the next policies through regret matching; accumulate the pre-update realization plans. | Uniform average of the `t` plans accumulated after `t` iterations. | Two extra best responses (BRs) to the average every `check_every` iterations. |
| FP | Compute one full-game BR per player to the opponent's current average; update both empirical mixtures simultaneously. | Average of `t` BRs and one uniform prior observation: denominator `t + 1`. | Free, every iteration: the two BRs answer the current averages, so their values are the exact SPR of the returned profile. |
| DO | Solve the restricted game for `(x_t, y_t)` when new candidates change it; otherwise reuse the mixture. Call both global BR oracles and add any new plans. | Current restricted-equilibrium mixture of realization plans. | Free, every iteration: the oracle values give `SPR(x_t, y_t) = u(BR1(y_t), y_t) - u(x_t, BR2(x_t))`. |

Every BR respects information sets; it cannot observe the opponent's hidden hand. Mixtures and time averages are formed in realization-plan coordinates, preserving own-action reach and flow conservation. Averaging raw behavioral probabilities would generally be wrong.

**20,000 is the number of outer iterations executed in every main benchmark run.** It is not a count of played games, sampled deals, tree nodes, or LP pivots. It is a chosen computational budget. Equal iteration counts do not equalize work, so elapsed seconds and actual operation counts are also reported. All runs continue beyond the 0.002 target; both endpoint quality and first observed target crossings are retained.

## SPR and retrospective target crossings

For P1 payoff `x.T @ A @ y`, the **saddle-point residual** is

```text
SPR(x, y) = max_x' x'.T A y - min_y' x.T A y'
          = P1's best unilateral gain + P2's best unilateral gain.
```

Zero means a Nash equilibrium in this two-player zero-sum game, up to floating-point precision. At `SPR <= 0.002`, neither player can improve expected payoff by more than `0.002` through a unilateral deviation. This implementation uses the **full gap**, not half the gap. The payoff is `P(win) - P(loss)` in one round, so `0.002` is not directly a percentage-point win-rate guarantee. Self-play value close to zero does not imply a small SPR.

The measured runs disable threshold stopping. Afterward, the first stored certificate with `SPR <= 0.002` identifies the first observed crossing. FP and DO store a certificate at every iteration; CFR's average is checked at entry, every 25 iterations, and at exit. A transient crossing between CFR checkpoints may be missed, because SPR is not guaranteed to decrease monotonically. A first crossing therefore does not imply that all later iterates satisfy the target; the final SPR is a separate result.

**DO exactness and full-budget continuation.** The restricted game's own gap is approximately zero by construction and cannot certify the full game. The two global BRs provide the full-game SPR. If neither oracle returns a new plan, the restricted equilibrium is an exact full-game equilibrium up to numerical precision. In a target-stopped solve that condition can end computation. Here threshold stopping is disabled, so DO continues actual BR/certificate cycles to iteration 20,000. Once the candidate pool is unchanged, it reuses the restricted-equilibrium mixture and avoids redundant LP solves; the later certificates are recomputed rather than copied or padded. This continuation is useful for the common-budget comparison, but is unnecessary for playing an already certified strategy. Exact ties can add equally good plans after the first small SPR, so the analysis also records the first observed `SPR <= 1e-9` (the LP accuracy).

## Experiment controls

The authoritative common settings are in [experiment_config.json](experiment_config.json). Command-line overrides and the complete resolved configuration are saved in `results/experiments.json`.

| Control | Submitted value | Meaning / effect |
|---|---|---|
| `iterations` / `--iterations` | `20000` | Full outer-iteration budget per fresh run. |
| `spr_threshold` / `--spr-threshold` | `0.002` | Retrospective full-game accuracy target; does not stop the measured runs. |
| `check_every` / `--check-every` | `25` | CFR certificate spacing. Main FP and DO runs store every iteration. |
| `history_output_every` | `25` | Common plotting grid in the main JSON; complete measured histories remain in gzip JSONL. |
| `summary_checkpoints` | 25, 100, 500, 1000, 2000, 5000, 10000, 20000 | Exact requested records retained in the main JSON without interpolation. |
| `repeats` / `--repeats` | `3` | Fresh timed runs per method per scenario, with rotated method order. These measure runtime variation. |
| `warmup_iterations` | `2` | Discarded warm-up for each iterative method before measurement; the full LP is also warmed once per scenario. |
| `full_budget` / `--full-budget` | `true` | Disable threshold stopping and complete the requested iterations; retain `spr_threshold` for retrospective crossing statistics. |
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
- Best-response ties choose the first maximizing action in supplied action order. Parent values are accumulated in the same order as the notebook's sequential reversed-infoset pass, so the vectorized BR is bit-identical to it, including at exact ties (tested). FP is sensitive to how exact ties are broken: an earlier change in accumulation order changed its first certified target crossing while leaving CFR's trajectory unchanged.
- FP's uniform prior has fixed weight one. DO begins with one uniform plan per player, deduplicates exact best-response plan bytes, and skips the LP for the trivial 1-by-1 game. These are explicit algorithm conventions, not tuned per scenario.
- DO uses SciPy `linprog(method='highs')` with primal and dual feasibility tolerance `1e-9`; its IPM optimality tolerance is left at the installed SciPy default. The full reference LP explicitly sets all three to `1e-9`. Changing the config's `lp_tolerance` changes only the full reference, not DO's fixed settings. Neither tolerance is the strategic SPR target.
- When the restricted game has several equilibria, DO follows whichever one HiGHS returns, so DO's trajectory can differ across platforms even with identical package versions (observed between macOS and Linux with the earlier DO code). CFR and FP trajectories are platform-independent in my runs.
- Full LP feasibility tolerances are clamped to at least `1e-10`, and its IPM optimality tolerance to at least `1e-12`, matching the supported lower limits. This does not affect the submitted `1e-9` setting.
- Runner validation requires realization nonnegativity within `1e-12`, maximum flow error below `1e-8`, agreement of logged and independently recomputed SPR within `1e-10`, and reference LP SPR below `1e-7`. These are numerical checks, not training controls. Reloaded exported policies must meet the common SPR target.
- DO's reported active support counts use mixture weights greater than `1e-12`. Stored candidate counts include zero-weight strategies and the initial uniform plan. Only the count report uses this cutoff; no strategy is pruned by it.
- The convergence plot floors displayed SPR at `1e-8` for log axes; stored results and stopping tests retain the actual values.
- Direct solver API defaults are separate from the experiment. The benchmark explicitly passes a 20,000-iteration budget and disables threshold stopping. It retains 0.002 in the result configuration solely for retrospective accuracy comparisons.
- `solve(max_iter=...)` counts additional iterations. Continuing a solver preserves its state; `reset=True` restarts. `None` or a negative threshold disables stopping. `max_iter=0` certifies the existing profile. Fresh instances are used for every benchmark repetition.
- Task 3's guided CFR demonstration deliberately runs all 20,000 iterations with stopping disabled; its final submission export comes from the LP. Game-demo seeds (`6208`, or `6208 + k`) affect notebook play examples only.

## What solve time measures

`wall_seconds` is monotonic wall-clock elapsed time from immediately before the solver constructor to immediately after `solve()` returns. It includes topology preprocessing in the constructor, algorithm initialization and updates, averaging, and the SPR certificates. DO includes restricted payoff expansion, LP solves, and mixtures. It is neither pure CPU time nor time spent playing games.

Tree construction is measured once as `build_seconds` and excluded from all solver timings. Warm-up, post-run independent validation, policy conversion/export, history serialization, the untimed analysis section, plots, notebook execution, and report generation are excluded. LP reference timing includes sparse assembly, both optimizations, and its internal certificates; its separate final best-response validation is outside the timer.

Solver-internal `elapsed_seconds` covers `solve()` calls only and accumulates on continuation. The benchmark adds construction time to plotted history entries and retains the original internal value as `solve_elapsed_seconds`. Each plotted curve uses that method's median-total-time repetition. `wall_seconds` describes all 20,000 iterations. The elapsed time on `first_target` describes the first recorded qualifying certificate; it must not be presented as the full-run time.

The main JSON retains compact convergence checkpoints, first-crossing and endpoint statistics, source hashes, hardware/software versions, training and certificate BR calls, regret sweeps, restricted LP solves, candidate/active strategy counts, and an `analysis` section (below). The complete measured histories live in per-run gzip JSON Lines files. Peak RSS is for the whole benchmark process; it is not a per-algorithm memory comparison.

## Complete per-run histories

`results/experiments.json` (schema 3) keeps the common 25-iteration plotting grid, requested summary checkpoints, and each run's first target crossing and final record. This compact array is not the complete FP or DO trace. Every measured record is preserved in `results/histories/*.jsonl.gz`: CFR at its 25-iteration certificate cadence, FP and DO at every iteration, including the initial and final certificates. The run's `history_file` is relative to the result JSON's parent directory; `history_records` reports its complete record count, and `history_cadence` records the interval. `history_sha256` hashes the compressed file bytes. `summary_checkpoints` holds the recorded observations at 25, 100, 500, 1,000, 2,000, 5,000, 10,000, and 20,000; any requested point absent from the measured cadence is listed in `unrecorded_summary_checkpoints` rather than interpolated.

From the repository root, I can inspect a complete run with the standard library:

```python
import gzip
import hashlib
import json
from pathlib import Path

results_dir = Path('jiayu_attempt/task4/results')
study = json.loads((results_dir / 'experiments.json').read_text())
run = study['scenarios']['neutral']['methods']['Fictitious play'][0]
history_path = results_dir / run['history_file']
assert hashlib.sha256(history_path.read_bytes()).hexdigest() == run['history_sha256']
with gzip.open(history_path, 'rt') as stream:
    history = [json.loads(line) for line in stream]
assert len(history) == run['history_records']
assert history[-1]['iteration'] == 20_000
print(history[0], history[-1])
```

Each record's `elapsed_seconds` includes construction time; `solve_elapsed_seconds` excludes construction. The plot reads compact checkpoints, while the compressed files preserve the complete observations for later analysis. No interpolated or padded measurements are substituted for solver output.

## Analysis section (untimed, one run each)

These subsidiary checks have their own stopping rules, saved in `analysis.protocols`; they are separate from the 36 main runs that each complete 20,000 iterations.

- `double_oracle_exact`: DO stops at a certified `SPR <= 1e-9` or the no-new-strategy certificate, with a separate 1,000-iteration maximum. It records the first `0.002` crossing, support sizes, the total time spent in restricted LPs, and `restricted_lp_trace` (restricted-game size and solve time for every restricted solve; Figure 2b).
- `rates`: CFR and FP over the full 20,000 iterations on the default game, with records every 25 iterations; log-log slope of SPR from iteration 100, and CFR's last-iterate SPR for contrast with its average.
- `lp_versus_cfr`: for the default and `first_player` games, CFR stops at its first recorded 0.002 crossing (within a 20,000-iteration maximum); this compares LP and CFR values, each player's exploitability, cross-pairings (LP P1 with CFR P2 and vice versa), mixing counts, and an LP over P1's optimal face that ranges one first-decision probability to prove the equilibrium is not unique.

## Measured findings

This full-budget run uses `macOS-15.6-arm64-arm-64bit`, Python 3.12.14, and the saved package versions. All 36 main runs execute 20,000 iterations; source hashes and complete histories are recorded.

**Endpoint SPR / median full-run seconds:**

| Method | Neutral/random starter | P1 starts | P2 starts | P1 starts, holds G3 favor |
|---|---:|---:|---:|---:|
| CFR | 0.0004155273 / 7.596 | 0.0004155273 / 4.623 | 0.0004155273 / 4.587 | 6.937044e-05 / 4.776 |
| Fictitious play | 0.0004388273 / 8.261 | 0.0004388273 / 4.881 | 0.0004388273 / 4.741 | 0.0002128465 / 4.783 |
| Double oracle | 4.483718e-15 / 16.413 | 9.59996e-15 / 5.707 | 4.614364e-15 / 5.370 | 7.313594e-15 / 6.472 |

**Retrospective first observed iteration at SPR <= 0.002 / median elapsed seconds at that certificate:**

| Method | Neutral/random starter | P1 starts | P2 starts | P1 starts, holds G3 favor |
|---|---:|---:|---:|---:|
| CFR | 1,550 / 0.600 | 1,550 / 0.361 | 1,550 / 0.358 | 675 / 0.158 |
| Fictitious play | 1,576 / 0.659 | 1,576 / 0.387 | 1,576 / 0.378 | 1,386 / 0.313 |
| Double oracle | 113 / 1.043 | 59 / 0.308 | 59 / 0.295 | 66 / 0.345 |

**Independent LP value / median reference solve seconds:**

| LP reference | Neutral/random starter | P1 starts | P2 starts | P1 starts, holds G3 favor |
|---|---:|---:|---:|---:|
| Value / median seconds | 0 / 0.088 | +0.01904762 / 0.044 | -0.01904762 / 0.047 | +0.05714286 / 0.061 |

The separate DO exactness check first certifies SPR <= 1e-9 at these iterations: neutral: 122, first_player: 59, second_player: 59, favored_g3: 66. This diagnostic uses its own stopping rule and is distinct from the complete 20,000-iteration benchmark.

- **Later accuracy.** At 20,000 iterations on the default game, CFR has SPR `0.00041552733`, FP `0.00043882726`, and DO `4.4837181e-15`. First crossings alone do not reveal these later outcomes.
- **Work after convergence.** DO keeps recomputing real BR certificates through the common budget. Once no new candidates arrive, it reuses the restricted equilibrium instead of repeating its LP solves. This explains why full-run cost differs from cost to first reach the target.
- **Decomposition.** The default game consists of two observed starting-player subgames. CFR and FP act independently on these subtrees; a DO restricted game formed from whole-game candidate plans can couple their mixture weights. Both first-crossing behavior and final accuracy are visible in the saved histories.
- **LP versus CFR.** The separate optimal-face and cross-pairing analysis checks values and exploitability, rather than requiring identical policy vectors; this game has multiple equilibria. Its CFR comparison deliberately stops at the accuracy target and does not substitute for the main 20,000-iteration endpoint.
- **Final policies.** The exported NPZ files remain the LP solution, with reloaded SPR `8.67361738e-18`. The longer iterative runs do not change that submission choice. No experiment here establishes Medium Hanamikoji scalability.

The first-crossing table and the endpoint table answer different questions: how soon a recorded profile meets the target, and how much accuracy remains after the full budget. Timings depend on hardware and the chosen run; DO's restricted-equilibrium choices can also change across platforms. The same initial game and 20,000 iterations do not imply equal computational cost.

## Reproduction and artifacts

In the collaborative repository, I run the command sequence in [../README.md](../README.md). In a grader copy where the ZIP is extracted directly over the supplied starter, the same Task 4 scripts use `task4/` paths, the policies use `tasks/`, and the solution modules use `student_attempts/` (without the `jiayu_attempt/` prefix). That copy still needs the supplied engine and a Python environment with `task4/requirements-lock.txt`; the solution ZIP does not duplicate the engine. The notebooks and wrapper remain in the collaborative repository because the original packager does not collect them. I run `execute_notebooks.py` only in the source checkout; it rejects an extracted archive without the attempt marker, so it cannot accidentally execute the grader's original starter notebooks and overwrite the final policies. Core files:

- `../student_attempts/utility.py`: chance/opponent-weighted posterior.
- `../student_attempts/solver.py`: best responses, full SPR, CFR and optional variants.
- `../student_attempts/fictitious_play.py` and `double_oracle.py`: the additional algorithm families.
- `../student_attempts/lp_solver.py`: independent sparse LP reference with lazy SciPy imports.
- `../tasks/nash_p1.npz`, `../tasks/nash_p2.npz`: final LP policies, reloaded and independently checked.
- `../tasks/my_nash_tiny.json`: both-seat neutral first-round policy table.
- `../report.pdf`: the Task 4 report (Direction D only), at most five A4 pages, 12 pt text and 2.5 cm margins, generated by `build_report.py`. Its sections: introduction (choice of direction and how it extends Tasks 2-3), how each solver works and why it converges, how correctness is verified, experimental setup, results with figures and tables, and conclusions.
- `results/experiments.json`, `benchmark.txt`: compact measurements, history index, and analysis.
- `results/histories/*.jsonl.gz`: complete per-run measured histories, with record counts and SHA-256 hashes in the main JSON.
- `results/convergence.png` (report Figure 1: SPR against iterations and time on the default game) and `results/decomposition.png` (Figure 2: CFR and DO on the default game and its halves; restricted-LP cost against stored plans), both drawn by `plot_results.py`.
- `results/tests.txt`: the unit-test log; four executed notebooks live in `../tasks/`.
- `requirements-lock.txt`: exact package versions used for this run.

The main study is already a full-budget experiment. A separate output path preserves its measured JSON when exploring another budget; the report's assertions must also match any intentionally changed study settings. The runner regenerates the same LP policy artifacts even if a different result-output path is chosen. Archived 2,000-iteration results remain in `../archive/2000_iteration_study/` and are not mixed into the new tables or figures.

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
