"""Package this attempt with the shared repository's official submission checker.

From the repository root:
    python3 vladimir_attempt/make_submission.py

The default output is vladimir_attempt/submission.zip. An explicit --out path
continues to use the caller's working directory, as in the supplied checker.
"""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = next((path for path in ROOT.parents if (path / "engine").is_dir()), ROOT.parent)
STUDENT = ROOT / "student_attempts"
sys.path[:0] = [str(STUDENT), str(ROOT), str(REPO)]


def main():
    shared_checker = REPO / "make_submission.py"
    if not shared_checker.is_file():
        raise FileNotFoundError(
            "The shared repository make_submission.py is required beside the engine/ folder."
        )
    spec = importlib.util.spec_from_file_location("_shared_submission_checker", shared_checker)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker.ROOT = str(ROOT)
    checker.STUDENT = str(STUDENT)
    if not any(argument == "--out" or argument.startswith("--out=") for argument in sys.argv[1:]):
        sys.argv.extend(["--out", str(ROOT / "submission.zip")])
    checker.main()


if __name__ == "__main__":
    main()
