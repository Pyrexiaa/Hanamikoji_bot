"""Independent tests: brute-force responses, known matrix game, and full tiny_hana.

Run from the project root: python -m unittest discover -s task4 -p 'test_solver.py' -v
"""
import itertools
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Keep this attempt's implementations ahead of the shared game engine.
# The ancestor fallback also supports a locally staged copy before upload.
REPO = next((path for path in ROOT.parents if (path / "engine").is_dir()), ROOT.parent)
sys.path[:0] = [str(ROOT / "student_attempts"), str(ROOT), str(REPO)]
from solver import Solver, CFRSolver
from treeplex_representation import Treeplex, TreeplexGame
from treeplex_representation import convert_behavioral_to_sequence_form_strategy


def sequence(treeplex, behavioral):
    # Independent scalar forward conversion, to avoid validating by implementation.
    result = behavioral.copy()
    result[0] = 1.0
    for start, end, parent in zip(treeplex.info_state_start_seq,
                                  treeplex.info_state_end_seq,
                                  treeplex.info_state_parent_seq):
        for action in range(start, end):
            result[action] = behavioral[action] * result[parent]
    return result


def pure_strategies(treeplex):
    counts = np.array(treeplex.info_state_end_seq) - np.array(treeplex.info_state_start_seq)
    for choices in itertools.product(*(range(k) for k in counts)):
        behavioral = np.zeros(treeplex.num_sequences)
        behavioral[0] = 1.0
        for start, choice in zip(treeplex.info_state_start_seq, choices):
            behavioral[start + choice] = 1.0
        yield sequence(treeplex, behavioral)


def matrix_game(matrix):
    matrix = np.asarray(matrix, dtype=float)
    p1, p2 = Treeplex(), Treeplex()
    p1.add_infoset(0, matrix.shape[0])
    p2.add_infoset(0, matrix.shape[1])
    return SimpleNamespace(treeplex_p1=p1, treeplex_p2=p2,
                           seq_pair_to_leaves={(i + 1, j + 1): value
                                              for (i, j), value in np.ndenumerate(matrix)})


def branched_game():
    p1, p2 = Treeplex(), Treeplex()
    p1.add_infoset(0, 2)
    p1.add_infoset(1, 2)
    p1.add_infoset(1, 2)  # two future infosets share one parent: their values add
    p1.add_infoset(2, 2)
    p2.add_infoset(0, 3)
    random = np.random.default_rng(43)
    return SimpleNamespace(treeplex_p1=p1, treeplex_p2=p2,
                           seq_pair_to_leaves={(s1, s2): random.uniform(-1, 1)
                                              for s1 in range(3, 9) for s2 in range(1, 4)})


class SolverTests(unittest.TestCase):
    def assert_flow(self, strategy, treeplex):
        self.assertAlmostEqual(strategy[0], 1.0)
        self.assertGreaterEqual(strategy.min(), -1e-12)
        for start, end, parent in zip(treeplex.info_state_start_seq,
                                      treeplex.info_state_end_seq,
                                      treeplex.info_state_parent_seq):
            self.assertAlmostEqual(strategy[start:end].sum(), strategy[parent], places=11)

    def test_branched_best_responses_match_exhaustive_pure_strategies(self):
        game = branched_game()
        solver = Solver(game)
        random = np.random.default_rng(7)
        for _ in range(5):
            own = []
            for treeplex in (game.treeplex_p1, game.treeplex_p2):
                behavioral = np.zeros(treeplex.num_sequences)
                for start, end in zip(treeplex.info_state_start_seq, treeplex.info_state_end_seq):
                    behavioral[start:end] = random.dirichlet(np.ones(end - start))
                own.append(sequence(treeplex, behavioral))
            x, y = own
            untouched_x, untouched_y = x.copy(), y.copy()
            br1 = solver.compute_best_response(0, y, True)
            br2 = solver.compute_best_response(1, x, True)
            upper = max(solver.compute_payoffs_from_seq_form_strategies(pure, y)
                        for pure in pure_strategies(game.treeplex_p1))
            lower = min(solver.compute_payoffs_from_seq_form_strategies(x, pure)
                        for pure in pure_strategies(game.treeplex_p2))
            self.assertAlmostEqual(solver.compute_payoffs_from_seq_form_strategies(br1, y), upper)
            self.assertAlmostEqual(solver.compute_payoffs_from_seq_form_strategies(x, br2), lower)
            self.assertAlmostEqual(solver.saddle_point_residual(x, y), upper - lower)
            self.assert_flow(br1, game.treeplex_p1)
            self.assert_flow(br2, game.treeplex_p2)
            np.testing.assert_array_equal(x, untouched_x)
            np.testing.assert_array_equal(y, untouched_y)

    def test_known_nash_and_exploitability(self):
        solver = Solver(matrix_game([[3, 0], [0, 1]]))
        equilibrium = np.array([1.0, 0.25, 0.75])
        self.assertAlmostEqual(solver.saddle_point_residual(equilibrium, equilibrium), 0)
        self.assertAlmostEqual(solver.compute_payoffs_from_seq_form_strategies(equilibrium, equilibrium), 0.75)
        pure = np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(solver.saddle_point_residual(pure, pure), 3.0)

    def test_cfr_convergence_against_known_matrix_equilibrium(self):
        game = matrix_game([[3, 0], [0, 1]])
        for plus, averaging, alternating in ((False, 'uniform', False),
                                              (True, 'uniform', False),
                                              (False, 'linear', False),
                                              (True, 'linear', True)):
            with self.subTest(plus=plus, averaging=averaging, alternating=alternating):
                solver = CFRSolver(game, plus, averaging, alternating)
                x, y = solver.solve(4000, spr_threshold=None, check_every=4000)
                self.assertLess(solver.spr, 0.07)
                self.assertAlmostEqual(solver.val, 0.75, delta=0.01)
                self.assertAlmostEqual(solver.spr, Solver(game).saddle_point_residual(x, y))
                if plus:
                    self.assertGreaterEqual(min(solver.regrets[0].min(), solver.regrets[1].min()), 0.0)

    def test_continuation_is_identical_to_uninterrupted_run(self):
        game = branched_game()
        for plus in (False, True):
            for alternating in (False, True):
                a = CFRSolver(game, plus, 'linear', alternating)
                b = CFRSolver(game, plus, 'linear', alternating)
                expected = a.solve(47, None, check_every=1000)
                b.solve(19, None, check_every=1000)
                actual = b.solve(28, None, check_every=1000)
                for left, right in zip(expected, actual):
                    np.testing.assert_allclose(left, right, rtol=0, atol=0)
                self.assertEqual(b.iterations, 47)
                self.assertEqual(b.average_weight, 47 * 48 / 2)
                reset = b.solve(47, None, reset=True, check_every=1000)
                for left, right in zip(expected, reset):
                    np.testing.assert_allclose(left, right, rtol=0, atol=0)

    def test_sequence_average_and_flow_on_deep_tree(self):
        game = branched_game()
        solver = CFRSolver(game)
        own_sum = [np.zeros(game.treeplex_p1.num_sequences), np.zeros(game.treeplex_p2.num_sequences)]
        for _ in range(13):
            current = {p: solver._ops[p].regret_matching(solver.regrets[p]) for p in (0, 1)}
            for p, treeplex in enumerate((game.treeplex_p1, game.treeplex_p2)):
                own_sum[p] += sequence(treeplex, current[p])
            returned = solver.solve(1, None)
        for p, treeplex in enumerate((game.treeplex_p1, game.treeplex_p2)):
            np.testing.assert_allclose(returned[p], own_sum[p] / 13)
            self.assert_flow(returned[p], treeplex)

    def test_payoff_changes_are_seen_when_continuing(self):
        game = matrix_game([[3, 0], [0, 1]])
        solver = CFRSolver(game)
        solver.solve(10, None)
        for key in game.seq_pair_to_leaves:
            game.seq_pair_to_leaves[key] *= -1
        x, y = solver.solve(10, None)
        self.assertAlmostEqual(solver.spr, Solver(game).saddle_point_residual(x, y))
        self.assertAlmostEqual(solver.val, Solver(game).compute_payoffs_from_seq_form_strategies(x, y))

    def test_empty_treeplex_and_zero_iterations_are_defined(self):
        empty = SimpleNamespace(treeplex_p1=Treeplex(), treeplex_p2=Treeplex(),
                                seq_pair_to_leaves={(0, 0): 0.37})
        solver = CFRSolver(empty)
        x, y = solver.solve(0)
        np.testing.assert_array_equal(x, [1.0])
        self.assertEqual(solver.spr, 0.0)
        self.assertAlmostEqual(solver.val, 0.37)
        self.assertEqual(solver.iterations, 0)

    def test_tiny_game_full_engine_uniform_payoff_and_flow(self):
        game = TreeplexGame()
        solver = CFRSolver(game)
        x, y = solver.solve(30, None, check_every=30)
        self.assert_flow(x, game.treeplex_p1)
        self.assert_flow(y, game.treeplex_p2)
        self.assertAlmostEqual(solver.val, 0.0, delta=1e-12)
        self.assertAlmostEqual(solver.spr, Solver(game).saddle_point_residual(x, y), places=12)
        self.assertGreaterEqual(solver.training_seconds, 0.0)


if __name__ == '__main__':
    unittest.main()
