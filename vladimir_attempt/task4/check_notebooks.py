"""Run the unmodified Task 1 and Task 2 notebook code cells without Jupyter.

Run from this attempt folder: python3 task4/check_notebooks.py
The notebook working directory is tasks/, exactly as their setup cells require.
Task 3 is deliberately excluded because it writes the submitted strategy files.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import traceback


ROOT = Path(__file__).resolve().parents[1]


def run_notebook(path):
    namespace = {"__name__": "__main__", "__file__": str(path)}
    output = io.StringIO()
    cells_run = 0
    with redirect_stdout(output), redirect_stderr(output):
        for cell_index, cell in enumerate(json.loads(path.read_text())["cells"]):
            if cell["cell_type"] != "code":
                continue
            print(f"--- Code cell {cell_index} ---")
            source = "".join(cell["source"])
            exec(compile(source, f"{path.name}:cell{cell_index}", "exec"), namespace)
            cells_run += 1
    captured = output.getvalue()
    if "TODO:" in captured:
        raise AssertionError(f"{path.name} reported an unfinished task:\n{captured}")
    return f"=== {path.name} ===\n{captured}PASS: {cells_run} code cells executed.\n"


def main():
    destination = ROOT / "task4" / "results" / "notebook_checks.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    original_directory = Path.cwd()
    reports = []
    try:
        os.chdir(ROOT / "tasks")
        for name in ("task1.ipynb", "task2.ipynb"):
            reports.append(run_notebook(ROOT / "tasks" / name))
    except Exception:
        reports.append("FAIL:\n" + traceback.format_exc())
        destination.write_text("\n".join(reports))
        raise
    finally:
        os.chdir(original_directory)
    destination.write_text("\n".join(reports))
    print("\n".join(reports), end="")
    print(f"Saved checks to {destination}")


if __name__ == "__main__":
    main()
