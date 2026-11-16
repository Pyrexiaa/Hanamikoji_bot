# Tiny Hanamikoji

Tasks 1–3 implement posterior inference, sequence-form best responses, CFR and LP-derived Nash policies. Task 4 studies alternative Nash solvers (Direction D) and five CFR configurations (Direction C).

Keep `final_attempt/` inside the complete lecturer repository, beside `engine/`, `tiny_hana/`, the original `student_attempts/` and `tasks/`. Commands below run from that **repository root**. The final scripts select `final_attempt/student_attempts/` and use the shared game engine. Contributor folders are not needed.

## Install

On Windows, use a short checkout path such as `C:\course\hana` so Jupyter dependencies fit systems without long-path support. With Python 3.12 available as `python`, create one environment and select its interpreter:

```powershell
python -m venv final_attempt/.venv
$projectPython = (Resolve-Path 'final_attempt/.venv/Scripts/python.exe').Path
& $projectPython -m pip install -r final_attempt/task4/requirements.txt
```

For an existing environment, just set `$projectPython`. Calling the interpreter directly also works when PowerShell blocks activation scripts. The pinned dependencies cover numerical work, plotting and Jupyter execution. Graded interfaces need NumPy; SciPy is loaded only for LP functionality.

For the commands below, use the Python engine and keep runtime files local:

```powershell
$env:HANA_ENGINE = 'python'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:MPLCONFIGDIR = (Resolve-Path 'final_attempt/.venv').Path + '/.matplotlib'
$env:JUPYTER_RUNTIME_DIR = (Resolve-Path 'final_attempt/.venv').Path + '/.jupyter-runtime'
```

On macOS and Linux the same setup in a POSIX shell is:

```bash
python3 -m venv final_attempt/.venv        # a Python 3.12 interpreter
projectPython="$PWD/final_attempt/.venv/bin/python"
"$projectPython" -m pip install -r final_attempt/task4/requirements.txt
export HANA_ENGINE=python PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
export MPLCONFIGDIR="$PWD/final_attempt/.venv/.matplotlib"
export JUPYTER_RUNTIME_DIR="$PWD/final_attempt/.venv/.jupyter-runtime"
```

Every command in the sections below is then the same with `"$projectPython"` in place of `& $projectPython`.

## Run and verify

```powershell
& $projectPython -B -m unittest discover -s final_attempt/task4 -p "test_*.py" -v
& $projectPython -B final_attempt/task4/execute_notebooks.py task1 task2 task3 task4
& $projectPython -B final_attempt/task4/validate_submission.py
& $projectPython -B final_attempt/task4/verify_evidence.py
```

The five test modules contain 62 tests. The notebook helper saves genuine outputs in the original four solution notebooks. `check_notebooks.py` additionally checks Tasks 1–2 without Jupyter. Validation prints results; `validate_submission.py` also updates the numerical policy checks in `task4/results/validation.json`.

These checks give the same result on Windows, macOS and Linux. The experiment records store raw-byte SHA-256 hashes of the source files, and the bytes of a text file depend on its line endings (LF or CRLF), which differ between checkouts. `task4/source_digest.py` therefore compares each recorded hash with the file under both conventions: only line endings are ignored, and any other change to a source still fails. Recorded paths are read with either separator.

## Reproduce experiments and figures

```powershell
& $projectPython -B final_attempt/task4/run_experiments.py --iterations 20000 --repeats 3 --full-budget
& $projectPython -B final_attempt/task4/plot_results.py
& $projectPython -B final_attempt/task4/run_cfr_ablation.py --iterations 20000 --repeats 3
& $projectPython -B final_attempt/task4/plot_report_figures.py
& $projectPython -B final_attempt/task4/extract_report_data.py
```

Run the complete benchmarks sequentially. [task4/README.md](task4/README.md) defines scenarios, update conventions, checkpoints and timing. Primary outputs in `task4/results/` are `experiments.json`, 36 compressed histories, `summary.json`, `convergence.csv`, four figures and two experimental policies. The plotting commands reproduce the saved figures without rerunning training; the table helper prints values directly from these datasets.

Task 3 and Experiment A write LP policies to `tasks/nash_p1.npz` and `tasks/nash_p2.npz`. Preserve validated copies before intentional reruns and validate the exports afterward. `tasks/my_nash_tiny.json` supports the policy-table demonstration; its fallback does not solve the multi-round game. Experimental CFR+ policies remain separate under `task4/results/`.

## Report and submission

Edit [report.docx](report.docx) directly and export [report.pdf](report.pdf) after review. No document-generation scripts are required. Word or LibreOffice can export it; the committed PDF comes from LibreOffice, which sets Times New Roman text in the metric-compatible Liberation Serif:

```bash
soffice --headless --convert-to 'pdf:writer_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"},"UseTaggedPDF":{"type":"boolean","value":"true"}}' --outdir final_attempt final_attempt/report.docx
```

The two figures are embedded at their printed size, so leave their size unchanged. A page break precedes the references. After any export, check that the PDF is A4, that the body ends on page 5 and that page 6 holds only the references.

When packaging is authorized, run `& $projectPython -B final_attempt/make_submission.py` from the repository root. This delegates to the lecturer's checker with the final folder selected, producing `final_attempt/submission.zip` with top-level `student_attempts/`, `tasks/nash_p1.npz`, `tasks/nash_p2.npz`, `report.pdf` and `task4/`. The Git layout and submission ZIP layout differ; the original starter implementation is not packaged.

The repository-root `.gitignore` excludes environments, Python/Jupyter caches and submission ZIPs. The small nested file adds Word lock files and OS/editor artifacts. The nested `.gitattributes` checks text files out with LF line endings on every system, so `treeplex_representation.py` stays byte-identical to the provided file, as the lecturer's checker requires. Scientific data, figures, policies and executed notebooks remain shareable.
