"""Independent best-response, sequence-form, CFR, and optional LP checks.

Run from final_attempt with ``python -m unittest discover -s task4``.
"""

import importlib.util
import itertools
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "student_attempts"))

from env_config import PLAYER_1, PLAYER_2, env
from solver import CFRSolver, Solver
from treeplex_representation import Treeplex, TreeplexGame


def matrix_game(matrix):
    matrix = np.asarray(matrix, dtype=float)
    first, second = Treeplex(), Treeplex()
    first.add_infoset(0, matrix.shape[0])
    second.add_infoset(0, matrix.shape[1])
    game = SimpleNamespace(treeplex_p1=first, treeplex_p2=second)
    game.seq_pair_to_leaves = {
        (row + 1, column + 1): float(matrix[row, column])
        for row in range(matrix.shape[0])
        for column in range(matrix.shape[1])
    }
    return game


def branching_treeplex():
    treeplex = Treeplex()
    treeplex.add_infoset(0, 2)
    treeplex.add_infoset(0, 2)
    treeplex.add_infoset(1, 2)
    treeplex.add_infoset(1, 2)
    treeplex.add_infoset(4, 2)
    return treeplex


def independent_realization(behavioral, treeplex):
    realization = np.ones(treeplex.num_sequences)
    for start, end, parent in zip(treeplex.info_state_start_seq,
                                  treeplex.info_state_end_seq,
                                  treeplex.info_state_parent_seq):
        realization[start:end] = behavioral[start:end] * realization[parent]
    return realization


def random_strategy(treeplex, rng):
    behavioral = np.ones(treeplex.num_sequences)
    for start, end in zip(treeplex.info_state_start_seq, treeplex.info_state_end_seq):
        behavioral[start:end] = rng.dirichlet(np.ones(end - start))
    return behavioral, independent_realization(behavioral, treeplex)


def enumerate_pure_plans(treeplex):
    choices = [range(start, end) for start, end in zip(
        treeplex.info_state_start_seq, treeplex.info_state_end_seq)]
    plans = []
    for selected in itertools.product(*choices):
        behavioral = np.zeros(treeplex.num_sequences)
        behavioral[list(selected)] = 1
        plans.append(independent_realization(behavioral, treeplex))
    return np.asarray(plans)


def assert_feasible(test, plan, treeplex, tolerance=1e-10):
    test.assertEqual(plan.shape, (treeplex.num_sequences,))
    test.assertTrue(np.isfinite(plan).all())
    test.assertGreaterEqual(float(plan.min()), -tolerance)
    test.assertAlmostEqual(float(plan[0]), 1.0, delta=tolerance)
    for start, end, parent in zip(treeplex.info_state_start_seq,
                                  treeplex.info_state_end_seq,
                                  treeplex.info_state_parent_seq):
        test.assertAlmostEqual(float(plan[start:end].sum()), float(plan[parent]),
                               delta=tolerance)


def direct_round_value(game, behavioral_p1, behavioral_p2):
    """Enumerate the live game without using sequence-form payoffs."""
    state = env.GameState()
    trees = (game.treeplex_p1, game.treeplex_p2)
    keys = (game.info_states_key_to_id_p1, game.info_states_key_to_id_p2)
    strategies = (behavioral_p1, behavioral_p2)

    def visit():
        if state.is_round_complete():
            return float(state.terminal_payoff(player_id=PLAYER_1))
        if state.current_player == env.CHANCE_PLAYER:
            branches = list(state.get_chance_actions_with_probs())
        else:
            player = state.current_player
            info_id = keys[player][state.get_info_state(player)]
            start = trees[player].info_state_start_seq[info_id]
            branches = [(action, strategies[player][start + index])
                        for index, action in enumerate(state.get_legal_actions())]
        result = 0.0
        for action, probability in branches:
            if probability == 0:
                continue
            state.execute_action(action)
            try:
                result += probability * visit()
            finally:
                state.undo_move()
        return result

    return visit()


class SolverUnitTests(unittest.TestCase):
    def test_best_response_matches_exhaustive_normal_form(self):
        rng = np.random.default_rng(90210)
        p1, p2 = branching_treeplex(), branching_treeplex()
        matrix = rng.normal(size=(p1.num_sequences, p2.num_sequences))
        game = SimpleNamespace(treeplex_p1=p1, treeplex_p2=p2,
                               seq_pair_to_leaves=dict(np.ndenumerate(matrix)))
        solver = Solver(game)
        pure1, pure2 = enumerate_pure_plans(p1), enumerate_pure_plans(p2)
        for _ in range(5):
            _, x = random_strategy(p1, rng)
            _, y = random_strategy(p2, rng)
            br1 = solver.compute_best_response(PLAYER_1, y, convert_to_seq_form=True)
            br2 = solver.compute_best_response(PLAYER_2, x, convert_to_seq_form=True)
            assert_feasible(self, br1, p1)
            assert_feasible(self, br2, p2)
            upper = float(np.max(pure1 @ matrix @ y))
            lower = float(np.min(x @ matrix @ pure2.T))
            self.assertAlmostEqual(float(br1 @ matrix @ y), upper, places=11)
            self.assertAlmostEqual(float(x @ matrix @ br2), lower, places=11)
            self.assertAlmostEqual(solver.saddle_point_residual(x, y), upper - lower,
                                   places=11)
            behavioral = solver.compute_best_response(PLAYER_1, y)
            for start, end in zip(p1.info_state_start_seq, p1.info_state_end_seq):
                self.assertEqual(float(behavioral[start:end].sum()), 1.0)
                self.assertEqual(int(np.count_nonzero(behavioral[start:end])), 1)

    def test_all_child_infosets_contribute_to_their_parent(self):
        p1, p2 = branching_treeplex(), Treeplex()
        game = SimpleNamespace(treeplex_p1=p1, treeplex_p2=p2,
                               seq_pair_to_leaves={(2, 0): 10.0,
                                                   (5, 0): 8.0, (7, 0): 7.0})
        solver = Solver(game)
        br = solver.compute_best_response(PLAYER_1, np.ones(1), True)
        self.assertEqual(br[1], 1.0)
        self.assertEqual(br[2], 0.0)
        self.assertEqual(solver.compute_payoffs_from_seq_form_strategies(br, np.ones(1)),
                         15.0)

    def test_payoff_reward_signs_and_known_matrix_equilibrium(self):
        matrix = np.array([[2.0, -1.0], [-1.0, 1.0]])
        solver = Solver(matrix_game(matrix))
        x, y = np.array([1.0, .3, .7]), np.array([1.0, .8, .2])
        reward1 = solver.compute_reward_vector(PLAYER_1, y)
        reward2 = solver.compute_reward_vector(PLAYER_2, x)
        np.testing.assert_allclose(reward1[1:], matrix @ y[1:])
        np.testing.assert_allclose(reward2[1:], -matrix.T @ x[1:])
        value = float(x[1:] @ matrix @ y[1:])
        self.assertAlmostEqual(solver.compute_payoffs_from_seq_form_strategies(x, y), value)
        self.assertAlmostEqual(float(x @ reward1), value)
        self.assertAlmostEqual(float(y @ reward2), -value)
        equilibrium = np.array([1.0, .4, .6])
        self.assertAlmostEqual(solver.saddle_point_residual(equilibrium, equilibrium), 0.0)
        pure = np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(solver.saddle_point_residual(pure, pure), 3.0)

    def test_game_without_decisions_keeps_root_payoff(self):
        game = SimpleNamespace(treeplex_p1=Treeplex(), treeplex_p2=Treeplex(),
                               seq_pair_to_leaves={(0, 0): .75})
        solver = Solver(game)
        for player in (PLAYER_1, PLAYER_2):
            np.testing.assert_allclose(solver.compute_best_response(player, np.ones(1), True),
                                       np.ones(1))
        self.assertEqual(solver.saddle_point_residual(np.ones(1), np.ones(1)), 0.0)

    def test_cfr_average_converges_toward_nonuniform_equilibrium(self):
        game = matrix_game([[2, -1], [-1, 1]])
        cfr = CFRSolver(game)
        initial = cfr.solve(max_iter=10, spr_threshold=0)
        initial_gap = cfr.saddle_point_residual(*initial)
        x, y = cfr.solve(max_iter=5990, spr_threshold=0)
        assert_feasible(self, x, game.treeplex_p1)
        assert_feasible(self, y, game.treeplex_p2)
        self.assertLess(cfr.spr, initial_gap / 3)
        self.assertLess(cfr.spr, .05)
        self.assertAlmostEqual(cfr.val, .2, delta=.02)
        self.assertAlmostEqual(cfr.spr, cfr.saddle_point_residual(x, y), places=12)

    def test_cfr_continuation_and_reset_match_uninterrupted_runs(self):
        game = matrix_game([[2, -1], [-1, 1]])
        configurations = (
            {},
            {"alternating": True},
            {"regret_matching_plus": True, "alternating": True},
            {"regret_matching_plus": True, "linear_averaging": True,
             "alternating": True},
        )
        for options in configurations:
            with self.subTest(options=options):
                split, whole = CFRSolver(game, **options), CFRSolver(game, **options)
                split.solve(max_iter=37, spr_threshold=None)
                split_result = split.solve(max_iter=63, spr_threshold=None)
                whole_result = whole.solve(max_iter=100, spr_threshold=None)
                self.assertEqual(split.iterations, 100)
                self.assertEqual(split.last_iterations, 63)
                for split_plan, whole_plan in zip(split_result, whole_result):
                    np.testing.assert_allclose(split_plan, whole_plan, atol=1e-13)
                reset_result = split.solve(max_iter=37, spr_threshold=None, reset=True)
                fresh_result = CFRSolver(game, **options).solve(max_iter=37,
                                                               spr_threshold=None)
                self.assertEqual(split.iterations, 37)
                for reset_plan, fresh_plan in zip(reset_result, fresh_result):
                    np.testing.assert_allclose(reset_plan, fresh_plan, atol=1e-13)

    def test_cfr_plus_clips_cumulative_regret_after_each_update(self):
        game = matrix_game([[2, -1], [-1, 1]])
        for alternating, linear in ((False, False), (True, False), (True, True)):
            with self.subTest(alternating=alternating, linear=linear):
                cfr = CFRSolver(game, regret_matching_plus=True,
                                alternating=alternating, linear_averaging=linear)
                for _ in range(20):
                    cfr.solve(max_iter=1, spr_threshold=None)
                    self.assertGreaterEqual(float(np.min(cfr.regrets_p1)), 0.0)
                    self.assertGreaterEqual(float(np.min(cfr.regrets_p2)), 0.0)

    def test_cfr_converges_on_shared_parent_treeplexes(self):
        rng = np.random.default_rng(1999)
        p1, p2 = branching_treeplex(), branching_treeplex()
        matrix = rng.normal(size=(p1.num_sequences, p2.num_sequences))
        game = SimpleNamespace(treeplex_p1=p1, treeplex_p2=p2,
                               seq_pair_to_leaves=dict(np.ndenumerate(matrix)))
        pure1, pure2 = enumerate_pure_plans(p1), enumerate_pure_plans(p2)
        solver = CFRSolver(game)
        x, y = solver.solve(max_iter=10, spr_threshold=None)
        first_gap = float(np.max(pure1 @ matrix @ y) - np.min(x @ matrix @ pure2.T))
        x, y = solver.solve(max_iter=4990, spr_threshold=None)
        final_gap = float(np.max(pure1 @ matrix @ y) - np.min(x @ matrix @ pure2.T))
        self.assertLess(final_gap, first_gap / 20)
        self.assertLess(final_gap, .05)
        self.assertAlmostEqual(solver.spr, final_gap, places=11)

    def test_changed_payoffs_are_used_after_continuation_and_reset(self):
        game = matrix_game([[1, 1], [0, 0]])
        cfr = CFRSolver(game)
        x, _ = cfr.solve(max_iter=20, spr_threshold=None)
        self.assertGreater(x[1], .9)
        game.seq_pair_to_leaves = matrix_game([[-1, -1], [1, 1]]).seq_pair_to_leaves
        np.testing.assert_allclose(cfr.compute_reward_vector(PLAYER_1, np.array([1, .5, .5])),
                                   [0, -1, 1])
        x, _ = cfr.solve(max_iter=1000, spr_threshold=None)
        self.assertGreater(x[2], .9)
        actual = cfr.solve(max_iter=75, spr_threshold=None, reset=True)
        expected = CFRSolver(game).solve(max_iter=75, spr_threshold=None)
        for actual_plan, expected_plan in zip(actual, expected):
            np.testing.assert_allclose(actual_plan, expected_plan, atol=1e-13)

    def test_zero_iteration_calls_preserve_valid_average(self):
        game = matrix_game([[2, -1], [-1, 1]])
        cfr = CFRSolver(game)
        x, y = cfr.solve(max_iter=0)
        np.testing.assert_allclose(x, [1, .5, .5])
        np.testing.assert_allclose(y, [1, .5, .5])
        self.assertEqual(cfr.iterations, 0)
        expected = cfr.solve(max_iter=75, spr_threshold=None)
        actual = cfr.solve(max_iter=0)
        self.assertEqual(cfr.iterations, 75)
        for actual_plan, expected_plan in zip(actual, expected):
            np.testing.assert_allclose(actual_plan, expected_plan, atol=1e-13)

    def test_optional_lp_does_not_make_scipy_an_import_requirement(self):
        code = """
import builtins
original = builtins.__import__
def without_scipy(name, *args, **kwargs):
    if name == 'scipy' or name.startswith('scipy.'):
        raise ImportError('blocked optional dependency')
    return original(name, *args, **kwargs)
builtins.__import__ = without_scipy
import solver
import lp_solver
assert 'scipy' not in __import__('sys').modules
try:
    lp_solver.solve_sequence_form_lp(None)
except ImportError as error:
    assert 'optional LP reference requires scipy' in str(error)
else:
    raise AssertionError('LP should require its optional dependency')
"""
        result = subprocess.run([sys.executable, "-c",
                                 f"import sys; sys.path[:0] = {[str(ROOT / 'student_attempts'), str(ROOT), str(REPO)]!r}\n" + code],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class TinyGameIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = TreeplexGame()

    def test_sequence_payoff_matches_direct_game_tree_enumeration(self):
        rng = np.random.default_rng(489)
        b1, x = random_strategy(self.game.treeplex_p1, rng)
        b2, y = random_strategy(self.game.treeplex_p2, rng)
        direct = direct_round_value(self.game, b1, b2)
        sequence = Solver(self.game).compute_payoffs_from_seq_form_strategies(x, y)
        self.assertAlmostEqual(sequence, direct, places=11)

    def test_best_response_is_bit_identical_to_sequential_reverse_pass(self):
        """Exact ties (G1/G2 symmetry, identical split cards) resolve as in the
        notebook's sequential pass, not only with the same value."""
        from tiny_hana.example_agents import CardCountingAgent
        from treeplex_representation import convert_behavioral_to_sequence_form_strategy

        def sequential(player, opponent):
            treeplex = self.game.treeplex_p1 if player == PLAYER_1 else self.game.treeplex_p2
            vec = np.zeros(treeplex.num_sequences)
            for (first, second), payoff in self.game.seq_pair_to_leaves.items():
                if player == PLAYER_1:
                    vec[first] += opponent[second] * payoff
                else:
                    vec[second] -= opponent[first] * payoff
            for info in reversed(range(treeplex.get_num_infosets())):
                start = treeplex.info_state_start_seq[info]
                end = treeplex.info_state_end_seq[info]
                best = start + int(np.argmax(vec[start:end]))
                vec[treeplex.info_state_parent_seq[info]] += vec[best]
                vec[start:end] = 0.0
                vec[best] = 1.0
            return vec

        solver = Solver(self.game)
        trees = (self.game.treeplex_p1, self.game.treeplex_p2)
        for agent in (env.RandomAgent(), CardCountingAgent()):
            for player in (PLAYER_1, PLAYER_2):
                opponent = 1 - player
                plan = convert_behavioral_to_sequence_form_strategy(
                    self.game.agent_to_behavioral_strategy(agent, opponent), trees[opponent],
                    in_place=False)
                np.testing.assert_array_equal(solver.compute_best_response(player, plan)[1:],
                                              sequential(player, plan)[1:])

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "optional SciPy is unavailable")
    def test_full_tiny_best_responses_match_independent_linear_programs(self):
        from scipy.optimize import linprog
        from lp_solver import payoff_matrix, sequence_form_constraints

        rng = np.random.default_rng(1776)
        _, x = random_strategy(self.game.treeplex_p1, rng)
        _, y = random_strategy(self.game.treeplex_p2, rng)
        A = payoff_matrix(self.game)
        solver = Solver(self.game)
        for player, treeplex, reward, opponent in (
            (PLAYER_1, self.game.treeplex_p1, A @ y, y),
            (PLAYER_2, self.game.treeplex_p2, -A.T @ x, x),
        ):
            E, e = sequence_form_constraints(treeplex)
            reference = linprog(-reward, A_eq=E, b_eq=e, bounds=(0, None), method="highs")
            self.assertTrue(reference.success, reference.message)
            response = solver.compute_best_response(player, opponent, True)
            assert_feasible(self, response, treeplex)
            self.assertAlmostEqual(float(reward @ response), -reference.fun, places=9)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "optional SciPy is unavailable")
    def test_lp_equilibrium_has_independent_best_response_certificate(self):
        from lp_solver import solve_sequence_form_lp

        result = solve_sequence_form_lp(self.game)
        x, y = result["sol_seq_form_p1"], result["sol_seq_form_p2"]
        assert_feasible(self, x, self.game.treeplex_p1)
        assert_feasible(self, y, self.game.treeplex_p2)
        self.assertEqual(result["status"], "optimal")
        for name in ("feasibility_p1", "feasibility_p2", "dual_feasibility_p1",
                     "dual_feasibility_p2", "optimality_p1", "optimality_p2", "gap"):
            self.assertLess(result[name], 1e-8, name)
        self.assertLess(Solver(self.game).saddle_point_residual(x, y), 1e-8)
        self.assertAlmostEqual(result["value"], 0.0, places=9)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "optional SciPy is unavailable")
class LPReferenceTests(unittest.TestCase):
    def test_nonzero_value_and_unrestricted_dual_variables(self):
        from lp_solver import solve_sequence_form_lp

        for offset in (0.0, -4.0, 7.0):
            game = matrix_game(np.array([[2.0, -1.0], [-1.0, 1.0]]) + offset)
            result = solve_sequence_form_lp(game)
            np.testing.assert_allclose(result["sol_seq_form_p1"], [1, .4, .6], atol=1e-9)
            np.testing.assert_allclose(result["sol_seq_form_p2"], [1, .4, .6], atol=1e-9)
            self.assertAlmostEqual(result["value"], .2 + offset, places=9)
            self.assertLess(result["gap"], 1e-9)

    def test_lp_reference_matches_zero_sum_bounds_across_ablation_scenarios(self):
        from lp_solver import solve_sequence_form_lp

        for favors, expected_sign in (
            ({}, 0),
            ({env.Suit.GEISHA_1: -1}, 1),
            ({env.Suit.GEISHA_1: -1, env.Suit.GEISHA_3: 1}, 1),
        ):
            with self.subTest(favors=favors):
                state = env.GameState()
                state.favors = {**state.favors, **favors}
                game = TreeplexGame(state)
                result = solve_sequence_form_lp(game)
                x, y = result["sol_seq_form_p1"], result["sol_seq_form_p2"]
                assert_feasible(self, x, game.treeplex_p1)
                assert_feasible(self, y, game.treeplex_p2)
                self.assertEqual(result["status"], "optimal")
                self.assertLess(result["gap"], 1e-8)
                self.assertLess(Solver(game).saddle_point_residual(x, y), 1e-8)
                if expected_sign == 0:
                    self.assertAlmostEqual(result["value"], 0.0, places=9)
                else:
                    self.assertGreater(result["value"], 0.0)


class IntegratedAblationAndBranchedSolverTests(unittest.TestCase):
    """Unique branched-tree, realization-average, and CFR-variant tests on canonical Solver."""

    @staticmethod
    def _deep_branched_game():
        from treeplex_representation import Treeplex

        p1, p2 = Treeplex(), Treeplex()
        p1.add_infoset(0, 2)
        p1.add_infoset(1, 2)
        p1.add_infoset(1, 2)  # two infosets share parent sequence 1
        p1.add_infoset(2, 2)  # depth-2 descendant under sequence 2
        p2.add_infoset(0, 3)
        rng = np.random.default_rng(43)
        return SimpleNamespace(
            treeplex_p1=p1,
            treeplex_p2=p2,
            seq_pair_to_leaves={
                (s1, s2): float(rng.uniform(-1.0, 1.0))
                for s1 in range(3, 9)
                for s2 in range(1, 4)
            },
        )

    def test_branched_best_responses_preserve_inputs_and_match_exhaustive_pure(self):
        game = self._deep_branched_game()
        solver = Solver(game)
        rng = np.random.default_rng(7)
        pure1 = enumerate_pure_plans(game.treeplex_p1)
        pure2 = enumerate_pure_plans(game.treeplex_p2)
        for _ in range(5):
            _, x = random_strategy(game.treeplex_p1, rng)
            _, y = random_strategy(game.treeplex_p2, rng)
            untouched_x, untouched_y = x.copy(), y.copy()
            br1 = solver.compute_best_response(PLAYER_1, y, True)
            br2 = solver.compute_best_response(PLAYER_2, x, True)
            upper = max(solver.compute_payoffs_from_seq_form_strategies(p, y) for p in pure1)
            lower = min(solver.compute_payoffs_from_seq_form_strategies(x, p) for p in pure2)
            self.assertAlmostEqual(solver.compute_payoffs_from_seq_form_strategies(br1, y), upper, places=11)
            self.assertAlmostEqual(solver.compute_payoffs_from_seq_form_strategies(x, br2), lower, places=11)
            self.assertAlmostEqual(solver.saddle_point_residual(x, y), upper - lower, places=11)
            assert_feasible(self, br1, game.treeplex_p1)
            assert_feasible(self, br2, game.treeplex_p2)
            np.testing.assert_array_equal(x, untouched_x)
            np.testing.assert_array_equal(y, untouched_y)

    def test_asymmetric_matrix_known_nash_and_exploitability(self):
        solver = Solver(matrix_game([[3, 0], [0, 1]]))
        equilibrium = np.array([1.0, 0.25, 0.75])
        self.assertAlmostEqual(solver.saddle_point_residual(equilibrium, equilibrium), 0.0)
        self.assertAlmostEqual(solver.compute_payoffs_from_seq_form_strategies(equilibrium, equilibrium), 0.75)
        pure = np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(solver.saddle_point_residual(pure, pure), 3.0)

    def test_all_five_cfr_ablation_variants_converge_on_asymmetric_game(self):
        game = matrix_game([[3, 0], [0, 1]])
        configs = (
            ("CFR", {}),
            ("CFR-linear", {"linear_averaging": True}),
            ("RM+", {"regret_matching_plus": True}),
            ("RM+-linear", {"regret_matching_plus": True, "linear_averaging": True}),
            ("CFR+", {"regret_matching_plus": True, "linear_averaging": True, "alternating": True}),
        )
        for name, kwargs in configs:
            with self.subTest(variant=name):
                solver = CFRSolver(game, **kwargs, check_every=4000)
                x, y = solver.solve(max_iter=4000, spr_threshold=None)
                assert_feasible(self, x, game.treeplex_p1)
                assert_feasible(self, y, game.treeplex_p2)
                self.assertLess(solver.spr, 0.07)
                self.assertAlmostEqual(solver.val, 0.75, delta=0.01)
                self.assertAlmostEqual(solver.spr, Solver(game).saddle_point_residual(x, y), places=11)
                if kwargs.get("regret_matching_plus"):
                    self.assertGreaterEqual(float(solver.regrets_p1.min()), 0.0)
                    self.assertGreaterEqual(float(solver.regrets_p2.min()), 0.0)

    def test_branched_continuation_and_linear_weight_sum(self):
        game = self._deep_branched_game()
        for plus in (False, True):
            for alternating in (False, True):
                with self.subTest(plus=plus, alternating=alternating):
                    opts = dict(regret_matching_plus=plus, linear_averaging=True,
                                alternating=alternating, check_every=1000)
                    whole = CFRSolver(game, **opts)
                    part = CFRSolver(game, **opts)
                    expected = whole.solve(max_iter=47, spr_threshold=None)
                    part.solve(max_iter=19, spr_threshold=None)
                    actual = part.solve(max_iter=28, spr_threshold=None)
                    for left, right in zip(expected, actual):
                        np.testing.assert_allclose(left, right, rtol=0, atol=1e-13)
                    self.assertEqual(part.iterations, 47)
                    self.assertEqual(part._total_weight, 47 * 48 / 2)
                    reset = part.solve(max_iter=47, spr_threshold=None, reset=True)
                    for left, right in zip(expected, reset):
                        np.testing.assert_allclose(left, right, rtol=0, atol=1e-13)

    def test_sequence_average_and_flow_on_deep_tree(self):
        game = self._deep_branched_game()
        solver = CFRSolver(game)
        own_sum = [
            np.zeros(game.treeplex_p1.num_sequences),
            np.zeros(game.treeplex_p2.num_sequences),
        ]
        for _ in range(13):
            b1 = solver._operations[PLAYER_1].regret_matching(solver.regrets_p1)
            b2 = solver._operations[PLAYER_2].regret_matching(solver.regrets_p2)
            own_sum[0] += solver._operations[PLAYER_1].realization_plan(b1)
            own_sum[1] += solver._operations[PLAYER_2].realization_plan(b2)
            returned = solver.solve(max_iter=1, spr_threshold=None)
        for p, treeplex in enumerate((game.treeplex_p1, game.treeplex_p2)):
            np.testing.assert_allclose(returned[p], own_sum[p] / 13.0, atol=1e-13)
            assert_feasible(self, returned[p], treeplex)

    def test_update_schedule_and_pre_update_averages_on_known_matrix(self):
        """Two hand-calculated steps distinguish refreshed opponents and average timing."""
        game = matrix_game([[3, 0], [0, 1]])
        configurations = [(False, False, False), (False, True, False),
                          (True, False, False), (True, True, False), (True, True, True)]
        for plus, linear, alternating in configurations:
            with self.subTest(plus=plus, linear=linear, alternating=alternating):
                learner = CFRSolver(game, regret_matching_plus=plus,
                                    linear_averaging=linear, alternating=alternating)
                x, y = learner.solve(max_iter=1, spr_threshold=None)
                np.testing.assert_allclose(x, [1, .5, .5], atol=1e-14)
                np.testing.assert_allclose(y, [1, .5, .5], atol=1e-14)
                # P1 switches to its first row before an alternating P2 update.
                magnitude = 1.5 if alternating else .5
                np.testing.assert_allclose(learner.regrets_p2,
                    [0, 0 if plus else -magnitude, magnitude], atol=1e-14)
                x, y = learner.solve(max_iter=1, spr_threshold=None)
                weight = 2 if linear else 1
                np.testing.assert_allclose(x, [1, (.5 + weight)/(1 + weight), .5/(1 + weight)], atol=1e-14)
                np.testing.assert_allclose(y, [1, .5/(1 + weight), (.5 + weight)/(1 + weight)], atol=1e-14)
                np.testing.assert_allclose(learner.current_seq_form_p1,
                    [1, 1/3, 2/3] if plus else [1, .5, .5], atol=1e-14)

    def test_empty_treeplex_cfr_zero_iterations_is_defined(self):
        empty = SimpleNamespace(
            treeplex_p1=Treeplex(),
            treeplex_p2=Treeplex(),
            seq_pair_to_leaves={(0, 0): 0.37},
        )
        solver = CFRSolver(empty)
        x, y = solver.solve(max_iter=0)
        np.testing.assert_array_equal(x, [1.0])
        np.testing.assert_array_equal(y, [1.0])
        self.assertEqual(solver.spr, 0.0)
        self.assertAlmostEqual(solver.val, 0.37)
        self.assertEqual(solver.iterations, 0)

    def test_tiny_game_full_engine_uniform_payoff_and_flow(self):
        game = TreeplexGame()
        solver = CFRSolver(game, check_every=30)
        x, y = solver.solve(max_iter=30, spr_threshold=None)
        assert_feasible(self, x, game.treeplex_p1)
        assert_feasible(self, y, game.treeplex_p2)
        self.assertAlmostEqual(solver.val, 0.0, delta=1e-12)
        self.assertAlmostEqual(solver.spr, Solver(game).saddle_point_residual(x, y), places=12)
        self.assertGreaterEqual(solver.elapsed_seconds, 0.0)

    def test_payoff_flip_updates_spr_and_value_on_continuation(self):
        game = matrix_game([[3, 0], [0, 1]])
        solver = CFRSolver(game, check_every=10)
        solver.solve(max_iter=10, spr_threshold=None)
        for key in game.seq_pair_to_leaves:
            game.seq_pair_to_leaves[key] *= -1
        x, y = solver.solve(max_iter=10, spr_threshold=None)
        ref = Solver(game)
        self.assertAlmostEqual(solver.spr, ref.saddle_point_residual(x, y), places=12)
        self.assertAlmostEqual(solver.val, ref.compute_payoffs_from_seq_form_strategies(x, y), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
