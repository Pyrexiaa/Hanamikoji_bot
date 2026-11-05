"""Run the unchanged starter checks and package only my attempt directory."""

import argparse
import importlib.util
from pathlib import Path
import sys


ATTEMPT = Path(__file__).resolve().parent
REPO = ATTEMPT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="submission.zip",
                        help="Archive filename inside jiayu_attempt (default: submission.zip)")
    args = parser.parse_args()
    output = (ATTEMPT / args.out).resolve()
    if output.parent != ATTEMPT:
        parser.error("The submission archive must be saved directly inside jiayu_attempt.")

    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ATTEMPT / "student_attempts"), str(ATTEMPT), str(REPO)]
    spec = importlib.util.spec_from_file_location("_starter_make_submission", REPO / "make_submission.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the original submission checker.")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker.ROOT = str(ATTEMPT)
    checker.STUDENT = str(ATTEMPT / "student_attempts")
    # The original main() retains its checks and its exact grader archive layout.
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), "--out", str(output)]
        checker.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
