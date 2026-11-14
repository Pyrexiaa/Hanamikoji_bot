"""Execute and save the four task notebooks with the active Python kernel."""

import argparse
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


ATTEMPT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", choices=("task1", "task2", "task3", "task4"))
    args = parser.parse_args()
    if not (ATTEMPT / "student_attempts" / "solver.py").is_file():
        parser.error(
            "Run this helper from the source checkout containing student_attempts/ and tasks/."
        )
    for name in args.notebooks or ("task1", "task2", "task3", "task4"):
        path = ATTEMPT / "tasks" / f"{name}.ipynb"
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
        ]
        NotebookClient(
            notebook, timeout=180, kernel_manager=manager,
            resources={"metadata": {"path": str(ATTEMPT / "tasks")}},
        ).execute()
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    raise RuntimeError(output)
                if "TODO:" in output.get("text", ""):
                    raise RuntimeError(output["text"])
        nbformat.write(notebook, path)
        print(f"{name}: all cells executed successfully", flush=True)


if __name__ == "__main__":
    main()
