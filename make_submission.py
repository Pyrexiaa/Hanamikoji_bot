"""Build and check your submission zip.

Run from the project root (the folder with this file):

    python make_submission.py                  # writes submission.zip
    python make_submission.py --out my.zip

It collects exactly these paths, which the autograder expects (see "What to
submit" in README.md):

    student_attempts/        every file in it (Tasks 1-3)
    tasks/nash_p1.npz        Task 3 strategy, player 1
    tasks/nash_p2.npz        Task 3 strategy, player 2
    report.pdf               Task 4 report
    task4/                   Task 4 code and notebooks (optional)

Before writing the zip it checks them. Errors (your code cannot be imported,
or a strategy file does not match the Task 3 treeplex) stop it: fix them and
run it again. Warnings (for example a function that is not implemented yet,
or a missing strategy file or report) are printed, and the zip is still
written without the missing files.
"""

import argparse
import hashlib
import os
import sys
import traceback
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
STUDENT = os.path.join(ROOT, "student_attempts")
REQUIRED_CODE = ["utility.py", "solver.py"]
STRATEGIES = [("tasks/nash_p1.npz", 0), ("tasks/nash_p2.npz", 1)]
REPORT = "report.pdf"
TASK4 = "task4"
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}

# The provided treeplex_representation.py, which the autograder uses to read
# the strategy files.
TREEPLEX_SHA256 = "6b59cacf00e51e5b46f63995b139f910837f371b52f0740fb6836f49f652c626"

errors, warnings = [], []


def files_under(folder):
    """Every file under `folder` (relative to ROOT), skipping caches."""
    found = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, folder)):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(".pyc"):
                continue
            found.append(os.path.relpath(os.path.join(dirpath, name), ROOT))
    return found


def check_files():
    for name in REQUIRED_CODE:
        if not os.path.isfile(os.path.join(STUDENT, name)):
            errors.append(f"student_attempts/{name} is missing.")
    for path, _ in STRATEGIES:
        if not os.path.isfile(os.path.join(ROOT, path)):
            warnings.append(f"{path} is missing: Task 3 is assessed on it. Run the last code cell of tasks/task3.ipynb.")
    report = os.path.join(ROOT, REPORT)
    if not os.path.isfile(report):
        warnings.append(f"{REPORT} is missing: Task 4 is assessed on the report.")
    else:
        with open(report, "rb") as f:
            if f.read(5) != b"%PDF-":
                warnings.append(f"{REPORT} does not look like a PDF.")
    if not os.path.isdir(os.path.join(ROOT, TASK4)):
        warnings.append(f"No {TASK4}/ folder: put your Task 4 code and notebooks there.")
    treeplex = os.path.join(STUDENT, "treeplex_representation.py")
    if os.path.isfile(treeplex):
        with open(treeplex, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != TREEPLEX_SHA256:
                warnings.append("student_attempts/treeplex_representation.py differs from the provided one. "
                                "The autograder reads your strategy files with the provided one.")


def check_code_and_strategies():
    """Import the student's code and check the strategy files against the default treeplex."""
    sys.path.insert(0, ROOT)
    sys.path.insert(0, STUDENT)
    try:
        import numpy as np
        from env_config import env, PLAYER_1, PLAYER_2
        from treeplex_representation import TreeplexGame, convert_behavioral_to_sequence_form_strategy
        game = TreeplexGame()
    except Exception:
        errors.append("Could not build the Task 3 treeplex:\n" + traceback.format_exc(limit=2))
        return

    # Strategy files: right player, right size, a distribution at every information set.
    for path, player in STRATEGIES:
        full = os.path.join(ROOT, path)
        if not os.path.isfile(full):
            continue
        try:
            data = np.load(full)
            strategy, stored_player = data["strategy"], int(data["player_id"][0])
        except Exception as e:
            errors.append(f"{path} cannot be read ({e}). Save it with TreeplexGame.save_agent.")
            continue
        treeplex = game.treeplex_p1 if player == PLAYER_1 else game.treeplex_p2
        if stored_player != player:
            errors.append(f"{path} holds player {stored_player + 1}'s strategy, not player {player + 1}'s.")
        elif strategy.size != treeplex.num_sequences:
            errors.append(f"{path} has {strategy.size} entries; the default TreeplexGame() has "
                          f"{treeplex.num_sequences}. Solve the default game (one round of tiny_hana, neutral favors).")
        elif not np.all(np.isfinite(strategy)) or np.any(strategy < -1e-9):
            errors.append(f"{path} contains negative or non-finite probabilities.")
        else:
            bad = [i for i in range(treeplex.get_num_infosets())
                   if abs(strategy[treeplex.info_state_start_seq[i]:treeplex.info_state_end_seq[i]].sum() - 1) > 1e-6]
            if bad:
                errors.append(f"{path}: at {len(bad)} information sets the probabilities do not sum to 1. "
                              "Save a behavioral strategy (see Task 3).")

    # The student's functions: present, and implemented. Any failure is a warning.
    def attempt(label, call, not_implemented=""):
        try:
            call()
        except NotImplementedError:
            warnings.append(f"{label} is not implemented yet{not_implemented}.")
        except Exception:
            warnings.append(f"{label} raised an error:\n" + traceback.format_exc(limit=3))

    try:
        import utility
        import solver
    except Exception:
        errors.append("Your code cannot be imported:\n" + traceback.format_exc(limit=3))
        return
    missing = [f"{module.__name__}.py has no {name}."
               for module, names in ((utility, ["compute_posterior_given_opponent_agent"]),
                                     (solver, ["Solver", "CFRSolver"]))
               for name in names if not hasattr(module, name)]
    if missing:
        errors.extend(missing)
        return

    info = utility.generate_random_infoset(seed=1)
    attempt("compute_posterior_given_opponent_agent",
            lambda: utility.compute_posterior_given_opponent_agent(env.GameState(), info, env.RandomAgent()))
    def uniform(treeplex):
        """The sequence form of playing uniformly at random everywhere."""
        behavioral = np.zeros(treeplex.num_sequences)
        for i in range(treeplex.get_num_infosets()):
            start, end = treeplex.info_state_start_seq[i], treeplex.info_state_end_seq[i]
            behavioral[start:end] = 1.0 / (end - start)
        return convert_behavioral_to_sequence_form_strategy(behavioral, treeplex, in_place=False)

    s = solver.Solver(game)
    uniform_p1, uniform_p2 = uniform(game.treeplex_p1), uniform(game.treeplex_p2)
    attempt("Solver.compute_best_response", lambda: s.compute_best_response(PLAYER_1, uniform_p2))
    attempt("Solver.saddle_point_residual", lambda: s.saddle_point_residual(uniform_p1, uniform_p2))
    attempt("CFRSolver.solve", lambda: solver.CFRSolver(game).solve(max_iter=1),
            not_implemented=" (fine if you used another method for Task 3)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="submission.zip", help="zip file to write (default: submission.zip)")
    args = parser.parse_args()

    check_files()
    check_code_and_strategies()

    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")
    if errors:
        print(f"\nNo zip written: fix the {len(errors)} error(s) above and run this again.")
        sys.exit(1)

    paths = files_under("student_attempts")
    paths += [p for p, _ in STRATEGIES if os.path.isfile(os.path.join(ROOT, p))]
    if os.path.isfile(os.path.join(ROOT, REPORT)):
        paths.append(REPORT)
    if os.path.isdir(os.path.join(ROOT, TASK4)):
        paths += files_under(TASK4)
    out = os.path.abspath(args.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(os.path.join(ROOT, p), arcname=p.replace(os.sep, "/"))
    size = os.path.getsize(out) / 2**20
    print(f"\nWrote {out} ({len(paths)} files, {size:.1f} MB). Upload it to Canvas.")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
