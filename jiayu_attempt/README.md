# Jiayu's Hanamikoji attempt

My solutions and generated artifacts live here so the shared starter remains available for other students. The 80 original starter files at repository root are restored byte-for-byte. My copy of the protected `treeplex_representation.py` is unchanged too.

```text
Hanamikoji_bot/
├── engine/, tiny_hana/, ...       original shared engines and examples
├── student_attempts/             original starter stubs
├── tasks/                        original starter notebooks
├── make_submission.py            original checker and packager
└── jiayu_attempt/
    ├── student_attempts/         my posterior, CFR, FP, DO, and LP implementations
    ├── tasks/                    my executed notebooks and exported policies
    ├── task4/                    experiment scripts, configuration, tests, results
    ├── report.pdf                Task 4 report (Direction D, at most five pages)
    ├── submission.zip            current grader-compatible archive
    ├── make_submission.py        wrapper around the original packager
    ├── verify_layout.py          starter, import, and archive checks
    ├── starter_manifest.json     original starter SHA-256 hashes
    ├── archive/cfr_study/         preserved earlier CFR-variant study
    ├── archive/pre_review_backup/ my files as they were before the 5 Nov review edits
    └── .venv/                    local environment, ignored by Git
```

The main study compares **CFR, fictitious play, and double oracle** with a **2,000-iteration cap** and a **0.002 full-SPR target** on four single-round scenarios (random starter, P1 starts, P2 starts, P1 starts holding the G3 favor), with **three timing repetitions**. FP and DO certify their output at every iteration for free from their own best responses; CFR's average is certified every 25 iterations. A separate sequence-form LP provides the exact reference and the submitted policies, and an untimed analysis section runs DO to exactness, measures convergence rates, and compares the LP and CFR equilibria. The experiment guide in [task4/README.md](task4/README.md) explains every control, the meaning of an iteration, SPR, the DO termination criterion, and solve time.

## Reproduce from the repository root

The relocated environment is ready. To recreate it with Python 3.12:

```sh
python3.12 -m venv jiayu_attempt/.venv
jiayu_attempt/.venv/bin/python -m pip install -r jiayu_attempt/task4/requirements-lock.txt
```

Run the complete workflow:

```sh
jiayu_attempt/.venv/bin/python -m unittest discover -s jiayu_attempt/task4 -p 'test_*.py' -v > jiayu_attempt/task4/results/tests.txt 2>&1
jiayu_attempt/.venv/bin/python jiayu_attempt/task4/run_experiments.py > jiayu_attempt/task4/results/benchmark.txt
jiayu_attempt/.venv/bin/python jiayu_attempt/task4/plot_results.py
jiayu_attempt/.venv/bin/python jiayu_attempt/task4/execute_notebooks.py
jiayu_attempt/.venv/bin/python jiayu_attempt/task4/build_report.py
jiayu_attempt/.venv/bin/python jiayu_attempt/make_submission.py
jiayu_attempt/.venv/bin/python jiayu_attempt/verify_layout.py
```

The benchmark overwrites measured results and exported policies inside this folder. The report reads the recorded results; it does not rerun experiments. Its common-configuration assertions prevent accidentally labelling a changed experiment as the submitted one. The four notebooks execute independently of the original starter notebooks.

For my submission, I run **`jiayu_attempt/make_submission.py`**. It uses the unchanged starter checker but points it at this attempt. The archive contains `student_attempts/`, `tasks/nash_p1.npz`, `tasks/nash_p2.npz`, `report.pdf`, and `task4/` at its top level, without a `jiayu_attempt/` prefix. The environment, old study, and other students' work stay outside the archive.

`verify_layout.py --skip-zip` checks starter identity and imports before packaging. Without that flag, it also verifies that every archived file matches the current solution. Other students can use separate attempt folders; my scripts import my own solver modules and the shared root engine explicitly. The manifest check describes this migration's original starter baseline; it will report a difference if shared starter files are deliberately changed later.

## Results at the shared target

| Default neutral game | Stop iteration | Full SPR | Median seconds |
|---|---:|---:|---:|
| CFR | 1,550 | 0.00196952 | 0.279 |
| Fictitious play | 1,576 | 0.00199948 | 0.265 |
| Double oracle | 101 (exact at 117) | 0.00196605 | 0.381 |

From the Linux run behind the current report; `task4/results/benchmark.txt` has the full log. Double oracle saves iterations but about 90% of its time is restricted LP solving, and because its restricted strategies are whole-game plans it needs nearly the sum of the two fixed-start halves' iterations on the random-start game, while CFR and FP decompose exactly. The full LP reference takes about 0.04 seconds on the default game and the reloaded submitted policies have SPR below `1e-17`.

All unit tests pass (`task4/results/tests.txt`), all four notebooks execute, and all 80 original starter hashes match. No commits or remote changes were made.
