"""Independent certificates and continuation checks for double oracle."""

import importlib.util
import itertools
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ATTEMPT_ROOT.parent
STUDENT_ROOT = ATTEMPT_ROOT / "student_attempts"
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(ATTEMPT_ROOT))
sys.path.insert(0, str(STUDENT_ROOT))

from double_oracle import DoubleOracleSolver
from env_config import make_round_start_state
from solver import Solver
from treeplex_representation import Treeplex, TreeplexGame


def make_matrix_game(matrix):
    matrix = np.asarray(matrix, dtype=float)
    p1, p2 = Treeplex(), Treeplex()
    p1.add_infoset(0, matrix.shape[0])
    p2.add_infoset(0, matrix.shape[1])
    return SimpleNamespace(
        treeplex_p1=p1, treeplex_p2=p2,
        seq_pair_to_leaves={(i + 1, j + 1): value
                            for (i, j), value in np.ndenumerate(matrix)},
    )


def assert_flow(test, plan, treeplex):
    test.assertAlmostEqual(plan[0], 1.0, places=10)
    test.assertGreaterEqual(float(np.min(plan)), -1e-10)
    for start, end, parent in zip(treeplex.info_state_start_seq,
                                  treeplex.info_state_end_seq,
                                  treeplex.info_state_parent_seq):
        test.assertAlmostEqual(float(plan[start:end].sum()), float(plan[parent]), places=10)


def pure_plans(treeplex):
    plans = []
    ranges = [range(start, end) for start, end in zip(
        treeplex.info_state_start_seq, treeplex.info_state_end_seq)]
    for choices in itertools.product(*ranges):
        plan = np.zeros(treeplex.num_sequences)
        plan[0] = 1.0
        for action, parent in zip(choices, treeplex.info_state_parent_seq):
            plan[action] = plan[parent]
        plans.append(plan)
    return np.asarray(plans)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "optional SciPy is unavailable")
class DoubleOracleTests(unittest.TestCase):
    def test_matching_pennies_stops_at_uniform_entry_certificate(self):
        solver = DoubleOracleSolver(make_matrix_game([[1, -1], [-1, 1]]))
        x, y = solver.solve()
        np.testing.assert_allclose(x, [1, .5, .5])
        np.testing.assert_allclose(y, [1, .5, .5])
        self.assertEqual(solver.iterations, 0)
        self.assertEqual(solver.support_counts, (1, 1))
        self.assertEqual(solver.training_br_calls, 0)
        self.assertEqual(solver.diagnostic_br_calls, 2)
        self.assertEqual(solver.restricted_lp_solves, 0)
        self.assertEqual(solver.spr, 0)
        self.assertEqual(solver.stop_reason, "spr_threshold")

    def test_nonuniform_equilibrium_and_negative_value(self):
        for offset in (0.0, -4.0):
            with self.subTest(offset=offset):
                game = make_matrix_game(np.array([[2, -1], [-1, 1]]) + offset)
                solver = DoubleOracleSolver(game)
                x, y = solver.solve(max_iter=100, spr_threshold=1e-9)
                np.testing.assert_allclose(x, [1, .4, .6], atol=1e-9)
                np.testing.assert_allclose(y, [1, .4, .6], atol=1e-9)
                self.assertAlmostEqual(solver.val, .2 + offset, places=9)
                self.assertLess(solver.spr, 1e-9)
                self.assertEqual(solver.stop_reason, "spr_threshold")
                self.assertLess(solver.iterations, 10)
                # Every iteration is certified, and the run stops at the first hit.
                self.assertEqual([row["iteration"] for row in solver.history],
                                 list(range(solver.iterations + 1)))
                self.assertTrue(all(row["spr"] > 1e-9 for row in solver.history[:-1]))
                self.assertEqual(solver.training_br_calls, 2 * solver.iterations)
                self.assertEqual(solver.diagnostic_br_calls, 2)
                self.assertEqual(solver.restricted_lp_solves, 2 * solver.restricted_game_solves)

    def test_restricted_gap_cannot_be_used_as_global_certificate(self):
        solver = DoubleOracleSolver(make_matrix_game([[2, -1], [-1, 1]]))
        solver.solve(max_iter=1, spr_threshold=1e-9)
        self.assertEqual(solver.iterations, 1)
        self.assertLess(solver.restricted_gap, 1e-9)
        self.assertGreater(solver.spr, .9)
        self.assertFalse(solver.converged)
        self.assertEqual(solver.stop_reason, "iteration_limit")
        solver.solve(max_iter=24, spr_threshold=1e-9)
        self.assertLess(solver.iterations, 25)
        self.assertTrue(solver.converged)
        self.assertEqual(solver.stop_reason, "spr_threshold")

    def test_realization_mixture_respects_shared_parent_flow(self):
        p1, p2 = Treeplex(), Treeplex()
        for tree in (p1, p2):
            tree.add_infoset(0, 2)
            tree.add_infoset(1, 2)
            tree.add_infoset(1, 2)
            tree.add_infoset(2, 2)
        rng = np.random.default_rng(32)
        matrix = rng.normal(size=(p1.num_sequences, p2.num_sequences))
        game = SimpleNamespace(treeplex_p1=p1, treeplex_p2=p2,
                               seq_pair_to_leaves=dict(np.ndenumerate(matrix)))
        solver = DoubleOracleSolver(game)
        np.testing.assert_allclose(solver.support_p1[0], [1, .5, .5, .25, .25,
                                                         .25, .25, .25, .25])
        x, y = solver.solve(max_iter=100, spr_threshold=1e-9)
        assert_flow(self, x, p1)
        assert_flow(self, y, p2)
        np.testing.assert_allclose(x, solver.weights_p1 @ np.asarray(solver.support_p1))
        np.testing.assert_allclose(y, solver.weights_p2 @ np.asarray(solver.support_p2))
        upper = float(np.max(pure_plans(p1) @ matrix @ y))
        lower = float(np.min(x @ matrix @ pure_plans(p2).T))
        self.assertLess(upper - lower, 1e-8)
        self.assertAlmostEqual(solver.spr, upper - lower, places=10)
        for support in (solver.support_p1[1:], solver.support_p2[1:]):
            self.assertTrue(all(np.isin(plan, [0.0, 1.0]).all() for plan in support))

    def test_continuation_and_reset_match_fresh_runs(self):
        # One fixed-start round needs more than 25 expansions, so no run stops early.
        game = TreeplexGame(make_round_start_state(0))
        split, whole = DoubleOracleSolver(game), DoubleOracleSolver(game)
        split.solve(max_iter=7, spr_threshold=None)
        actual = split.solve(max_iter=18, spr_threshold=None)
        expected = whole.solve(max_iter=25, spr_threshold=None)
        for first, second in zip(actual, expected):
            np.testing.assert_allclose(first, second, atol=1e-12)
        self.assertEqual(whole.stop_reason, "iteration_limit")
        self.assertEqual(split.iterations, 25)
        self.assertEqual(split.last_iterations, 18)
        self.assertEqual(split.training_br_calls, whole.training_br_calls)
        self.assertEqual(split.restricted_lp_solves, whole.restricted_lp_solves)
        self.assertEqual(split.support_counts, whole.support_counts)
        actual = split.solve(max_iter=7, spr_threshold=None, reset=True)
        fresh = DoubleOracleSolver(game)
        expected = fresh.solve(max_iter=7, spr_threshold=None)
        for first, second in zip(actual, expected):
            np.testing.assert_allclose(first, second, atol=1e-12)
        self.assertEqual(split.training_br_calls, fresh.training_br_calls)
        self.assertEqual(split.restricted_lp_solves, fresh.restricted_lp_solves)

    def test_zero_iterations_and_changed_payoffs(self):
        game = make_matrix_game([[2, -1], [-1, 1]])
        solver = DoubleOracleSolver(game)
        x, y = solver.solve(max_iter=0)
        np.testing.assert_allclose(x, [1, .5, .5])
        self.assertEqual(solver.restricted_lp_solves, 0)
        solver.solve(max_iter=25)
        game.seq_pair_to_leaves = make_matrix_game([[-1, 1], [2, -1]]).seq_pair_to_leaves
        x, y = solver.solve(max_iter=25, spr_threshold=None)
        np.testing.assert_allclose(x, [1, .6, .4], atol=1e-9)
        np.testing.assert_allclose(y, [1, .4, .6], atol=1e-9)
        self.assertLess(solver.spr, 1e-9)
        # Without a target, DO stops once neither oracle offers a new plan.
        self.assertEqual(solver.stop_reason, "no_new_strategy")
        old_iterations = solver.iterations
        old_plans = x.copy(), y.copy()
        result = solver.solve(max_iter=0)
        self.assertEqual(solver.iterations, old_iterations)
        for actual, expected in zip(result, old_plans):
            np.testing.assert_array_equal(actual, expected)

    def test_full_tiny_certificate_against_independent_linear_programs(self):
        from scipy.optimize import linprog
        from lp_solver import payoff_matrix, sequence_form_constraints, solve_sequence_form_lp

        game = TreeplexGame()
        solver = DoubleOracleSolver(game)
        x, y = solver.solve(max_iter=250, spr_threshold=.002)
        assert_flow(self, x, game.treeplex_p1)
        assert_flow(self, y, game.treeplex_p2)
        A = payoff_matrix(game)
        E, e = sequence_form_constraints(game.treeplex_p1)
        F, f = sequence_form_constraints(game.treeplex_p2)
        upper_lp = linprog(-(A @ y), A_eq=E, b_eq=e, bounds=(0, None), method="highs")
        lower_lp = linprog(A.T @ x, A_eq=F, b_eq=f, bounds=(0, None), method="highs")
        self.assertTrue(upper_lp.success)
        self.assertTrue(lower_lp.success)
        upper, lower = -upper_lp.fun, lower_lp.fun
        self.assertAlmostEqual(solver.spr, upper - lower, places=8)
        self.assertLessEqual(upper - lower, .002)
        exact = solve_sequence_form_lp(game)["value"]
        self.assertLessEqual(lower - 1e-8, exact)
        self.assertLessEqual(exact, upper + 1e-8)
        self.assertEqual(solver.stop_reason, "spr_threshold")
        self.assertEqual(solver.history[-1]["iteration"], solver.iterations)
        self.assertTrue(all(row["spr"] > .002 for row in solver.history[:-1]))
        self.assertLess(solver.support_count_p1, game.treeplex_p1.num_sequences)
        self.assertEqual(solver.restricted_payoffs.shape, solver.support_counts)
        self.assertEqual(solver.training_br_calls, 2 * solver.iterations)

    def test_oracle_certificate_matches_independent_spr_each_iteration(self):
        game = TreeplexGame(make_round_start_state(0))
        solver, evaluator = DoubleOracleSolver(game), Solver(game)
        for _ in range(12):
            x, y = solver.solve(max_iter=1, spr_threshold=None)
            self.assertAlmostEqual(solver.spr, evaluator.saddle_point_residual(x, y), places=12)
            self.assertAlmostEqual(solver.val,
                                   evaluator.compute_payoffs_from_seq_form_strategies(x, y),
                                   places=12)
        self.assertEqual(solver.iterations, 12)


class DoubleOracleImportTests(unittest.TestCase):
    def test_import_and_zero_iteration_evaluation_do_not_require_scipy(self):
        code = """
import builtins
original = builtins.__import__
def without_scipy(name, *args, **kwargs):
    if name == 'scipy' or name.startswith('scipy.'):
        raise ImportError('blocked optional dependency')
    return original(name, *args, **kwargs)
builtins.__import__ = without_scipy
from double_oracle import DoubleOracleSolver
from treeplex_representation import Treeplex
from types import SimpleNamespace
p1, p2 = Treeplex(), Treeplex()
p1.add_infoset(0, 2)
p2.add_infoset(0, 2)
game = SimpleNamespace(treeplex_p1=p1, treeplex_p2=p2,
    seq_pair_to_leaves={(1,1):2, (1,2):-1, (2,1):-1, (2,2):1})
solver = DoubleOracleSolver(game)
solver.solve(max_iter=0)
assert 'scipy' not in __import__('sys').modules
try:
    solver.solve(max_iter=1)
except ImportError as error:
    assert 'requires scipy' in str(error)
else:
    raise AssertionError('restricted LP should require scipy')
"""
        prefix = (f"import sys; sys.path[:0] = [{str(STUDENT_ROOT)!r}, "
                  f"{str(ATTEMPT_ROOT)!r}, {str(REPOSITORY_ROOT)!r}]\n")
        result = subprocess.run([sys.executable, "-c", prefix + code],
                                cwd=REPOSITORY_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
