# Reproducing the project

The graded implementations in `student_attempts/utility.py` and `solver.py` use only NumPy and the supplied engine. `treeplex_representation.py` is unchanged.

This is the `vladimir_attempt/` contribution within the shared repository. The game engine and official submission checker remain in the repository root. Scripts automatically locate that shared engine while importing the implementations in this attempt. Python 3.9+ is sufficient for the implementations.

From the shared repository root, enter this attempt before running the commands below:

```sh
cd vladimir_attempt
```

Use your environment's Python interpreter in place of `python3` if needed.

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s task4 -p 'test_*.py' -v
```

Optional dependencies for the experiments and report:

```sh
python3 -m pip install -r task4/requirements-experiments.txt
python3 task4/run_experiments.py --iterations 20000 --repeats 3
python3 task4/validate_submission.py
python3 task4/build_report.py
python3 make_submission.py
```

The experiment runner creates `task4/results/convergence.csv`, a machine-readable summary, and convergence charts. It also saves the neutral-game CFR+ time-average policies to the required `tasks/nash_p1.npz` and `tasks/nash_p2.npz` using the provided save API. Those policies cover a single round with neutral starting favors and a random starter. An optional SciPy sequence-form LP independently checks all three experimental games. Its strategies are reference checks; the submitted policies come from iterative CFR+.

The first four methods form a 2x2 ablation with simultaneous updates: signed cumulative regret or nonnegative cumulative regret, crossed with uniform or linear strategy averaging. The fifth adds alternating updates to RM+ with linear averaging. Linear strategy averaging alone is not the algorithm called LCFR, because regrets are not weighted by iteration here.

All numerical values are obtained from the supplied Python engine, with no sampled trajectories in training or exploitability evaluation. Timing repeats use rotated method order. Training time excludes tree extraction, payoff preparation, exact best-response diagnostics and file export. Small timing differences should be interpreted cautiously.

`validate_submission.py` independently checks serialized strategies, exact engine expected payoffs, and seeded single-round play. `build_report.py` regenerates the five-page report from saved results. Figures in `results/` are supporting evidence. The PDF is the Task 4 deliverable.

