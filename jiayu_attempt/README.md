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
    ├── archive/2000_iteration_study/ preserved previous benchmark and report
    └── .venv/                    local environment, ignored by Git
```

The main study compares **CFR, fictitious play, and double oracle** over the **full 20,000-iteration budget** on four single-round scenarios (random starter, P1 starts, P2 starts, P1 starts holding the G3 favor), with **three timing repetitions**. The **0.002 full-SPR target is retrospective**: it identifies the first observed crossing without stopping the run. All configured histories are saved through iteration 20,000 in per-run gzip JSON Lines files under `task4/results/histories/`; `experiments.json` indexes those files and retains compact checkpoints for plotting. CFR's average is certified every 25 iterations; FP and DO certify and store their profiles every iteration. DO continues actual best-response and certificate cycles after its candidate pool stops changing, reusing the restricted equilibrium until a new candidate requires another LP solve. This later work measures a common budget and is unnecessary for playing an already certified equilibrium.

A separate sequence-form LP remains the reference and the source of the submitted policies. The untimed analysis examines exactness, convergence rates, and differences between the LP and CFR strategies. The experiment guide in [task4/README.md](task4/README.md) explains the controls, iteration work, retrospective crossing statistics, and endpoint timings.

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

## Results from the full budget

The default neutral game gives the following results on `macOS-15.6-arm64-arm-64bit`, Python 3.12.14. Times are medians of three fresh runs; `task4/results/benchmark.txt` contains the full log.

| Method | First observed target iteration | Seconds to first crossing | SPR at 20,000 | Full-run seconds |
|---|---:|---:|---:|---:|
| CFR | 1,550 | 0.600 | 0.00041552733 | 7.596 |
| Fictitious play | 1,576 | 0.659 | 0.00043882726 | 8.261 |
| Double oracle | 113 | 1.043 | 4.4837181e-15 | 16.413 |

The reloaded LP policies have SPR `8.67361738e-18`. Main histories contain 801 measured CFR certificates and 20,001 FP/DO certificates per run (iteration zero included), indexed by `experiments.json`.

First-crossing times are read from recorded certificates and differ from the time needed to finish all 20,000 iterations. CFR's first observed crossing is limited to its 25-iteration checkpoint grid; FP and DO retain every iteration. Since SPR can rise again after crossing, the final SPR is reported separately.

The full LP supplies `tasks/nash_p1.npz` and `tasks/nash_p2.npz`. The test log, executed notebooks, and `verify_layout.py` provide the current correctness and starter-integrity checks.
