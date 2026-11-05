"""Check fictitious play against matrix-game equilibria and sequence-form invariants."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ATTEMPT = Path(__file__).resolve().parents[1]
REPOSITORY = ATTEMPT.parent
sys.path[:0] = [str(ATTEMPT / "student_attempts"), str(REPOSITORY)]

from env_config import PLAYER_1, PLAYER_2
from fictitious_play import FictitiousPlaySolver
from solver import Solver
from treeplex_representation import Treeplex, TreeplexGame


def matrix_game(matrix):
    matrix = np.asarray(matrix, dtype=float)
    first, second = Treeplex(), Treeplex()
    first.add_infoset(0, matrix.shape[0])
    second.add_infoset(0, matrix.shape[1])
    return SimpleNamespace(
        treeplex_p1=first, treeplex_p2=second,
        seq_pair_to_leaves={(row + 1, column + 1): float(matrix[row, column])
                           for row in range(matrix.shape[0])
                           for column in range(matrix.shape[1])})


def assert_feasible(test, plan, tree):
    test.assertEqual(plan.shape, (tree.num_sequences,))
    test.assertTrue(np.isfinite(plan).all())
    test.assertGreaterEqual(float(plan.min()), 0.0)
    test.assertEqual(float(plan[0]), 1.0)
    for start, end, parent in zip(tree.info_state_start_seq, tree.info_state_end_seq,
                                  tree.info_state_parent_seq):
        test.assertAlmostEqual(float(plan[start:end].sum()), float(plan[parent]), places=12)


class FictitiousPlayTests(unittest.TestCase):
    def test_simultaneous_response_and_one_observation_prior(self):
        # P2's response to the old uniform mean is column 1, but its response
        # to P1's updated mean would be column 2; distinguish the update orders.
        game = matrix_game([[2, 0], [-2, 1]])
        solver = FictitiousPlaySolver(game)
        x, y = solver.solve(max_iter=1, spr_threshold=None)
        np.testing.assert_array_equal(x, [1.0, .75, .25])
        np.testing.assert_array_equal(y, [1.0, .75, .25])
        self.assertEqual(solver.br_calls, 2)
        self.assertEqual(solver.training_br_calls, 2)
        self.assertEqual(solver.diagnostic_br_calls, 4)

    def test_average_converges_toward_known_nonuniform_equilibrium(self):
        game = matrix_game([[2, -1], [-1, 1]])
        solver = FictitiousPlaySolver(game)
        solver.solve(max_iter=10, spr_threshold=None)
        early_gap = solver.spr
        x, y = solver.solve(max_iter=1990, spr_threshold=None)
        assert_feasible(self, x, game.treeplex_p1)
        assert_feasible(self, y, game.treeplex_p2)
        self.assertLess(solver.spr, early_gap)
        self.assertLess(solver.spr, .1)
        np.testing.assert_allclose(x[1:], [.4, .6], atol=.04)
        np.testing.assert_allclose(y[1:], [.4, .6], atol=.04)
        self.assertAlmostEqual(solver.val, .2, delta=.01)
        self.assertAlmostEqual(solver.spr, Solver(game).saddle_point_residual(x, y), places=12)
        self.assertEqual(solver.br_calls, 4000)

    def test_continuation_and_reset_match_fresh_run(self):
        game = matrix_game([[2, -1], [-1, 1]])
        split, whole = FictitiousPlaySolver(game), FictitiousPlaySolver(game)
        split.solve(max_iter=37, spr_threshold=None)
        pieces = split.solve(max_iter=63, spr_threshold=None)
        together = whole.solve(max_iter=100, spr_threshold=None)
        self.assertEqual(split.iterations, 100)
        self.assertEqual(split.last_iterations, 63)
        self.assertEqual(split.br_calls, whole.br_calls)
        for first, second in zip(pieces, together):
            np.testing.assert_array_equal(first, second)
        reset = split.solve(max_iter=100, spr_threshold=None, reset=True)
        for first, second in zip(reset, together):
            np.testing.assert_array_equal(first, second)
        self.assertEqual(split.iterations, 100)
        self.assertEqual(split.diagnostic_br_calls, whole.diagnostic_br_calls)
        self.assertEqual(split.history[0]['iteration'], 0)

    def test_initial_equilibrium_and_zero_budget(self):
        game = matrix_game([[1, -1], [-1, 1]])
        solver = FictitiousPlaySolver(game)
        x, y = solver.solve()
        self.assertEqual(solver.iterations, 0)
        self.assertEqual(solver.br_calls, 0)
        self.assertTrue(solver.converged)
        self.assertEqual(solver.spr, 0.0)
        for plan in (x, y):
            np.testing.assert_array_equal(plan, [1.0, .5, .5])
        solver.solve(max_iter=0, spr_threshold=None)
        self.assertFalse(solver.converged)
        self.assertEqual(solver.iterations, 0)

    def test_payoff_changes_refresh_between_calls(self):
        game = matrix_game([[2, -1], [-1, 1]])
        solver = FictitiousPlaySolver(game)
        plans = solver.solve(max_iter=10, spr_threshold=None)
        old_gap, old_value = solver.spr, solver.val
        game.seq_pair_to_leaves = {key: value * 3 for key, value in game.seq_pair_to_leaves.items()}
        refreshed = solver.solve(max_iter=0, spr_threshold=None)
        self.assertAlmostEqual(solver.spr, 3 * old_gap)
        self.assertAlmostEqual(solver.val, 3 * old_value)
        for first, second in zip(plans, refreshed):
            np.testing.assert_array_equal(first, second)

    def test_tiny_game_full_budget_preserves_realization_flows(self):
        game = TreeplexGame()
        solver = FictitiousPlaySolver(game)
        initial_gap = Solver(game).saddle_point_residual(solver.sol_seq_form_p1,
                                                        solver.sol_seq_form_p2)
        x, y = solver.solve(max_iter=2000, spr_threshold=None)
        assert_feasible(self, x, game.treeplex_p1)
        assert_feasible(self, y, game.treeplex_p2)
        self.assertEqual(solver.iterations, 2000)
        self.assertEqual(solver.br_calls, 4000)
        self.assertLess(solver.spr, initial_gap)
        self.assertLess(solver.spr, .02)
        self.assertAlmostEqual(solver.spr, Solver(game).saddle_point_residual(x, y), places=12)
        self.assertEqual([entry['iteration'] for entry in solver.history], list(range(0, 2001, 25)))


if __name__ == '__main__':
    unittest.main()
