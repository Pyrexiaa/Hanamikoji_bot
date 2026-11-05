"""Check starter byte identity, isolated solution imports, and submission layout."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


ATTEMPT = Path(__file__).resolve().parent
REPO = ATTEMPT.parent
IGNORED = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_starter():
    manifest = json.loads((ATTEMPT / "starter_manifest.json").read_text())
    errors = []
    for relative, expected in manifest.items():
        path = REPO / relative
        if not path.is_file():
            errors.append(f"Missing starter file: {relative}")
        elif digest(path) != expected:
            errors.append(f"Modified starter file: {relative}")
    # The original solution slots must contain only their original starter files.
    for folder in ("student_attempts", "tasks"):
        for path in (REPO / folder).rglob("*"):
            if path.is_file() and not any(part in IGNORED for part in path.parts):
                relative = path.relative_to(REPO).as_posix()
                if relative not in manifest:
                    errors.append(f"Solution artifact outside jiayu_attempt: {relative}")
    for relative in ("task4", "report.pdf", "submission.zip", ".venv"):
        if (REPO / relative).exists():
            errors.append(f"Relocated artifact still present at repository root: {relative}")
    protected = "student_attempts/treeplex_representation.py"
    if digest(ATTEMPT / protected) != manifest[protected]:
        errors.append("My copy of the protected treeplex representation was modified.")
    if errors:
        raise RuntimeError("\n".join(errors))
    print(f"Starter: {len(manifest)} files match their original SHA-256 hashes.", flush=True)


def verify_imports():
    code = """
import importlib
from pathlib import Path
import sys
attempt, repo = map(Path, sys.argv[1:])
sys.path[:0] = [str(attempt / 'student_attempts'), str(attempt), str(repo)]
for name in ('utility', 'solver', 'env_config', 'treeplex_representation', 'lp_solver',
             'fictitious_play', 'double_oracle'):
    module = importlib.import_module(name)
    expected = attempt / 'student_attempts' / (name + '.py')
    assert Path(module.__file__).resolve() == expected, (name, module.__file__)
for name in ('engine', 'tiny_hana', 'shared_example_agents'):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().parent == repo / name, (name, module.__file__)
print('Imports: my modules resolve inside jiayu_attempt; shared engine modules resolve at repository root.')
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(ATTEMPT), str(REPO)],
                   cwd=ATTEMPT, check=True)


def verify_archive():
    archive = ATTEMPT / "submission.zip"
    if not archive.exists():
        print("Archive: not present yet; archive checks skipped.")
        return
    required = {
        "student_attempts/utility.py", "student_attempts/solver.py",
        "student_attempts/env_config.py", "student_attempts/treeplex_representation.py",
        "tasks/nash_p1.npz", "tasks/nash_p2.npz", "report.pdf",
    }
    current_files = {"tasks/nash_p1.npz", "tasks/nash_p2.npz", "report.pdf"}
    for folder in ("student_attempts", "task4"):
        for path in (ATTEMPT / folder).rglob("*"):
            if (path.is_file() and path.suffix != ".pyc"
                    and not any(part in IGNORED or part == ".git" for part in path.parts)):
                current_files.add(path.relative_to(ATTEMPT).as_posix())
    with zipfile.ZipFile(archive) as submission:
        names = submission.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("The submission archive contains duplicate entries.")
        missing = required - set(names)
        if missing:
            raise RuntimeError(f"Missing submission entries: {sorted(missing)}")
        if set(names) != current_files:
            raise RuntimeError("The archive file list differs from my current solution; rebuild it.")
        if not any(name.startswith("task4/") for name in names):
            raise RuntimeError("The submission archive has no Task 4 implementation.")
        for name in names:
            relative = Path(name)
            allowed = (name.startswith("student_attempts/") or name.startswith("task4/")
                       or name in {"tasks/nash_p1.npz", "tasks/nash_p2.npz", "report.pdf"})
            if not allowed or relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unexpected archive path: {name}")
            if any(part in IGNORED or part in {".venv", ".git"} for part in relative.parts):
                raise RuntimeError(f"Cache or environment file in submission: {name}")
            current = ATTEMPT / relative
            if not current.is_file() or submission.read(name) != current.read_bytes():
                raise RuntimeError(f"Stale archive entry: {name}; rebuild my submission archive.")
        corrupt = submission.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Corrupt archive entry: {corrupt}")
    print(f"Archive: {len(names)} current files use the original grader layout.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-zip", action="store_true",
                        help="Check source isolation before rebuilding the archive")
    args = parser.parse_args()
    verify_starter()
    verify_imports()
    if not args.skip_zip:
        verify_archive()


if __name__ == "__main__":
    main()
