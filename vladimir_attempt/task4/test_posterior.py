"""Posterior checks, including a separately weighted full-trajectory reference.

Run from the project root with ``python -m unittest discover -s task4 -p 'test_posterior.py'``.
"""

import copy
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
# Keep this attempt's implementations ahead of the shared game engine.
# The ancestor fallback also supports a locally staged copy before upload.
REPO = next((path for path in ROOT.parents if (path / "engine").is_dir()), ROOT.parent)
sys.path[:0] = [str(ROOT / "student_attempts"), str(ROOT), str(REPO)]

from env_config import env
from utility import compute_posterior_given_opponent_agent


class WeightedAgent(env.Agent):
    """Nonuniform, information-dependent strategy with positive action support."""

    def get_action_distribution(self, info_state):
        actions = list(info_state.get_legal_actions())
        weights = [float(i + 1) for i in range(len(actions))]
        total = sum(weights)
        return {a: w / total for a, w in zip(actions, weights)}


class ReserveFirstAgent(env.Agent):
    def get_action_distribution(self, info_state):
        actions = list(info_state.get_legal_actions())
        chosen = next((a for a in actions if a.action_type == env.ActionType.RESERVE), actions[0])
        # Include zeros explicitly to check zero-probability branch handling.
        return {a: float(a == chosen) for a in actions}


def reference_posterior(initial_state, target, opponent):
    """Condition ordinary joint trajectory probabilities, including *both* players.

    This uses independent state copies and breadth-first enumeration rather than
    undo traversal. The target player's nonuniform policy cancels under perfect
    recall, giving the same answer as counterfactual opponent/chance weighting.
    """
    own_policy = WeightedAgent()
    queue = deque([(copy.deepcopy(initial_state), 1.0)])
    mass = defaultdict(float)
    while queue:
        state, probability = queue.popleft()
        if state.is_round_complete():
            continue
        player = state.current_player
        observed = state.get_info_state(target.player_id)
        if player == target.player_id and observed == target:
            mass[state.to_key()] += probability
            continue
        # Event count is a safe depth limit without comparing event contents.
        if len(observed.events) > len(target.events):
            continue
        if player == env.CHANCE_PLAYER:
            branches = state.get_chance_actions_with_probs()
        else:
            agent = own_policy if player == target.player_id else opponent
            branches = agent.get_action_distribution(state.get_info_state(player)).items()
        for action, weight in branches:
            if weight == 0.0:
                continue
            child = copy.deepcopy(state)
            if not child.execute_action(action):
                raise AssertionError("Reference generated an illegal action")
            queue.append((child, probability * weight))
    total = sum(mass.values())
    return {key: value / total for key, value in mass.items()} if total else {}


def first_decision(starting_player=0):
    """A known deal: each player initially has suit A; starter then draws B."""
    state = env.GameState()
    a, b, _ = list(env.Suit)
    moves = [
        env.Action(env.ActionType.CHANCE_START, [], choice=starting_player),
        env.Action(env.ActionType.CHANCE_DEAL, [env.Card(a), env.Card(a)]),
        env.Action(env.ActionType.CHANCE_DRAW, [env.Card(b)], choice=starting_player),
    ]
    for action in moves:
        if not state.execute_action(action):
            raise AssertionError(f"Invalid test setup: {action}")
    return state


def sampled_late_infosets():
    """Deterministic sample with a past own move and past opponent move."""
    infos = {}
    for seed in range(30):
        rng = random.Random(seed)
        state = env.GameState()
        while not state.is_round_complete():
            player = state.current_player
            if player == env.CHANCE_PLAYER:
                actions, weights = zip(*state.get_chance_actions_with_probs())
                action = rng.choices(actions, weights=weights)[0]
            else:
                info = state.get_info_state(player)
                if state.turn_count >= 2:
                    infos.setdefault(player, info)
                    if len(infos) == 2:
                        return [infos[0], infos[1]]
                action = rng.choice(list(state.get_legal_actions()))
            state.execute_action(action)
    raise AssertionError("Could not sample both players")


class PosteriorTests(unittest.TestCase):
    def assert_distribution_close(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key, value in expected.items():
            self.assertAlmostEqual(actual[key], value, places=12)
        if actual:
            self.assertAlmostEqual(sum(actual.values()), 1.0, places=12)
            self.assertTrue(all(value > 0 for value in actual.values()))

    def test_first_decision_matches_card_counting_for_both_players(self):
        # After observing own A and draw B, the five unseen cards are A,B,C,C,C.
        a, b, c = list(env.Suit)
        for player in (0, 1):
            state = first_decision(player)
            posterior = compute_posterior_given_opponent_agent(
                env.GameState(), state.get_info_state(player), WeightedAgent())
            by_opponent_hand = defaultdict(float)
            for key, probability in posterior.items():
                hand = key.p1_hand if player == 0 else key.p0_hand
                by_opponent_hand[hand[0]] += probability
            self.assert_distribution_close(dict(by_opponent_hand), {a: 0.2, b: 0.2, c: 0.6})

    def test_weighted_full_trajectory_reference_for_both_players(self):
        for target in sampled_late_infosets():
            for opponent in (WeightedAgent(), ReserveFirstAgent()):
                with self.subTest(player=target.player_id, opponent=type(opponent).__name__):
                    state = env.GameState()
                    actual = compute_posterior_given_opponent_agent(state, target, opponent)
                    expected = reference_posterior(state, target, opponent)
                    self.assert_distribution_close(actual, expected)

    def test_unreachable_information_set_returns_empty(self):
        state = first_decision(0)
        split = next(a for a in state.get_legal_actions() if a.action_type == env.ActionType.SPLIT)
        state.execute_action(split)
        target = state.get_info_state(1)  # Player 1 is responding to an immediate split.
        self.assertEqual(compute_posterior_given_opponent_agent(
            env.GameState(), target, ReserveFirstAgent()), {})
        impossible_round = replace(target, round_number=2)
        self.assertEqual(compute_posterior_given_opponent_agent(
            env.GameState(), impossible_round, WeightedAgent()), {})

    def test_input_state_and_undo_stack_are_restored(self):
        state = first_decision(0)
        original_key = state.to_key()
        original_infos = [state.get_info_state(p) for p in (0, 1)]
        original_undo = list(state._undo_stack)
        # This target is deeper than the supplied non-root initial state.
        child = copy.deepcopy(state)
        split = next(a for a in child.get_legal_actions() if a.action_type == env.ActionType.SPLIT)
        child.execute_action(split)
        posterior = compute_posterior_given_opponent_agent(
            state, child.get_info_state(1), WeightedAgent())
        self.assertTrue(posterior)
        self.assertEqual(state.to_key(), original_key)
        self.assertEqual([state.get_info_state(p) for p in (0, 1)], original_infos)
        self.assertEqual(state._undo_stack, original_undo)
        # An initial state already in the requested decision set is a point mass.
        self.assertEqual(compute_posterior_given_opponent_agent(
            state, original_infos[0], WeightedAgent()), {original_key: 1.0})

    def test_input_state_restored_when_opponent_raises(self):
        class RaisingAgent(env.Agent):
            def get_action_distribution(self, info_state):
                raise RuntimeError("test opponent failure")

        child = first_decision(0)
        split = next(a for a in child.get_legal_actions() if a.action_type == env.ActionType.SPLIT)
        child.execute_action(split)
        target = child.get_info_state(1)
        state = env.GameState()
        key = state.to_key()
        infos = [state.get_info_state(p) for p in (0, 1)]
        with self.assertRaisesRegex(RuntimeError, "test opponent failure"):
            compute_posterior_given_opponent_agent(state, target, RaisingAgent())
        self.assertEqual(state.to_key(), key)
        self.assertEqual([state.get_info_state(p) for p in (0, 1)], infos)
        self.assertEqual(state._undo_stack, [])


if __name__ == "__main__":
    unittest.main()
