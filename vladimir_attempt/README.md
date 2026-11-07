# Vladimir's complete Hanamikoji solution

This folder contains the implementation and evidence for all four assignment tasks. It is separate from JiaYu's implementation, so both approaches can be reviewed and compared without replacing each other's files.

The game engine stays in the shared repository. This folder supplies its own `student_attempts/` implementations, notebooks, strategies, experiments and report. Import setup selects these implementations before the shared engine; there is no second copy of the engine.

## Contents

- **Task 1:** posterior inference in `student_attempts/utility.py`.
- **Task 2:** exact best responses and saddle-point residual in `student_attempts/solver.py`.
- **Task 3:** full-tree CFR with saved neutral-game strategies in `tasks/nash_p1.npz` and `tasks/nash_p2.npz`.
- **Task 4:** CFR/CFR+ experiments, an independent sequence-form LP check, tests and recorded evidence in `task4/`.
- **Report:** `report.pdf`, with methods, correctness checks, convergence plots and findings.

The submitted strategies use alternating CFR+ with linear strategy averaging. The experiments also compare vanilla CFR, linear strategy averaging, RM+, and RM+ with linear averaging across three initial favor states. Linear strategy averaging changes the average only; it is not iteration-weighted regret accumulation. All saved policies cover one round from neutral favors, with either player starting.

`student_attempts/treeplex_representation.py` is an unchanged copy of the supplied file. The report and recorded experiment results are included as generated; moving the implementation into this folder changes import paths only.

## Run from the shared repository root

Use the Python interpreter for your environment in place of `python3` if needed. The implementations need NumPy; the other packages in the supplied requirements support the optional C++ engine.

```sh
python3 -m pip install -r vladimir_attempt/requirements.txt
python3 -m unittest discover -s vladimir_attempt/task4 -p 'test_*.py' -v
python3 vladimir_attempt/task4/check_notebooks.py
python3 vladimir_attempt/task4/validate_submission.py
python3 vladimir_attempt/make_submission.py
```

The final command runs the shared repository's official checker against this attempt and writes `vladimir_attempt/submission.zip`. The upload folder intentionally omits generated ZIP files, environments and caches.

Open the notebooks with the working directory set to `vladimir_attempt/tasks/`. The setup cells import the shared engine and this attempt's own implementation. Running the Task 3 solve cell regenerates and replaces the two strategy files; rerun the experiment command below to regenerate the reported CFR+ submission strategies afterward.

## Reproduce experiments and report

These commands replace the saved experiment outputs and saved strategies. Timing results vary by machine.

```sh
python3 -m pip install -r vladimir_attempt/task4/requirements-experiments.txt
python3 vladimir_attempt/task4/run_experiments.py --iterations 20000 --repeats 3
python3 vladimir_attempt/task4/validate_submission.py
python3 vladimir_attempt/task4/build_report.py
```

The tests and saved-policy validator run against the shared Python engine. The optional experiment dependencies are SciPy, Matplotlib, ReportLab and pypdf. Details of the ablation and timing measurements are in `task4/README.md`.
