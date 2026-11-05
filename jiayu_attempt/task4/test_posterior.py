"""Posterior checks against physical-card Bayes calculations and a joint-law oracle."""

import copy
from dataclasses import replace
from pathlib import Path
import pickle
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "student_attempts"))

from env_config import env
from utility import compute_posterior_given_opponent_agent, generate_random_infosets


A, B, C = tuple(env.Suit)


def play(state, action_type, suits=(), choice=None):
    action = env.Action(action_type, [env.Card(suit) for suit in suits], choice=choice)
    if not state.execute_action(action):
        raise AssertionError(f"Invalid test fixture action: {action}")


def dealt_state(starting_player, hands):
    state = env.GameState()
    play(state, env.ActionType.CHANCE_START, choice=starting_player)
    play(state, env.ActionType.CHANCE_DEAL, hands)
    return state


class ReserveSignalAgent(env.Agent):
    """A hidden B draw causes reserve with 3/4; a hidden C draw with 1/4."""

    def __init__(self, expected_player, reserve=True):
        self.expected_player = expected_player
        self.reserve = reserve

    def get_action_distribution(self, info_state):
        if info_state.player_id != self.expected_player:
            raise AssertionError("The target player's strategy must not be queried.")
        hand, _ = info_state.get_current_hand_and_actions()
        drawn = info_state.events[-1].card
        reserve = env.Action(env.ActionType.RESERVE, [env.Card(drawn)])
        split = env.Action(env.ActionType.SPLIT, hand)
        probability = {A: 0.5, B: 0.75, C: 0.25}[drawn] if self.reserve else 0.0
        return {reserve: probability, split: 1.0 - probability}


class NeverCalledAgent(env.Agent):
    def get_action_distribution(self, info_state):
        raise AssertionError("No opponent decision should occur on this search.")


class WeightedAgent(env.Agent):
    def get_action_distribution(self, info_state):
        actions = list(info_state.get_legal_actions())
        total = len(actions) * (len(actions) + 1) / 2
        return {action: (index + 1) / total for index, action in enumerate(actions)}


def joint_law_posteriors(initial_state, targets, opponent):
    """Independent oracle: full joint reach with uniform own behavior and copied children.

    Own reach is constant within a perfect-recall infoset, so conditioning
    the full joint distribution cancels it. No observation-prefix pruning.
    """
    targets = set(targets)
    owner = next(iter(targets)).player_id
    masses = {info: {} for info in targets}
    own = env.RandomAgent()

    def visit(state, probability):
        if state.is_round_complete():
            return
        actor = state.current_player
        if actor == env.CHANCE_PLAYER:
            distribution = state.get_chance_actions_with_probs()
        else:
            info = state.get_info_state(actor)
            if actor == owner and info in targets:
                key = state.to_key()
                masses[info][key] = masses[info].get(key, 0.0) + probability
            strategy = own if actor == owner else opponent
            distribution = strategy.get_action_distribution(info).items()
        for action, weight in distribution:
            if weight:
                child = copy.deepcopy(state)
                if not child.execute_action(action):
                    raise AssertionError("Invalid oracle action")
                visit(child, probability * weight)

    visit(copy.deepcopy(initial_state), 1.0)
    return {info: {key: p / sum(values.values()) for key, p in values.items()}
            for info, values in masses.items()}


class PosteriorTests(unittest.TestCase):
    def assert_distribution(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key, probability in expected.items():
            self.assertAlmostEqual(actual[key], probability, places=12)
        if actual:
            self.assertAlmostEqual(sum(actual.values()), 1.0, places=12)
            self.assertTrue(all(probability > 0.0 for probability in actual.values()))

    def test_opening_hidden_card_counts_for_both_seats(self):
        for target in (0, 1):
            with self.subTest(target=target):
                expected = {}
                for hidden, probability in ((A, 1 / 5), (B, 1 / 5), (C, 3 / 5)):
                    hands = (A, hidden) if target == 0 else (hidden, A)
                    state = dealt_state(target, hands)
                    play(state, env.ActionType.CHANCE_DRAW, (B,), choice=target)
                    info = state.get_info_state(target)
                    expected[state.to_key()] = probability
                # Observing A then B leaves multiplicities A:1, B:1, C:3.
                actual = compute_posterior_given_opponent_agent(
                    env.GameState(), info, NeverCalledAgent())
                self.assert_distribution(actual, expected)

    def test_hidden_reserve_bayes_likelihood_for_both_seats(self):
        for target in (0, 1):
            with self.subTest(target=target):
                opponent = 1 - target
                hands = (A, B) if target == 0 else (B, A)
                initial = dealt_state(opponent, hands)
                expected = {}
                for hidden in (B, C):
                    state = copy.deepcopy(initial)
                    play(state, env.ActionType.CHANCE_DRAW, (hidden,), choice=opponent)
                    play(state, env.ActionType.RESERVE, (hidden,))
                    play(state, env.ActionType.CHANCE_DRAW, (A,), choice=target)
                    info = state.get_info_state(target)
                    expected[state.to_key()] = 0.5
                # B: (1/5)(3/4)(1/4) = C: (3/5)(1/4)(1/4) = 3/80.
                actual = compute_posterior_given_opponent_agent(
                    initial, info, ReserveSignalAgent(opponent))
                self.assert_distribution(actual, expected)
                unreachable = compute_posterior_given_opponent_agent(
                    initial, info, ReserveSignalAgent(opponent, reserve=False))
                self.assertEqual(unreachable, {})

    def test_matches_independent_joint_law_after_own_actions(self):
        sampled = generate_random_infosets(16, seed=6028, prob=0.18)
        for target in (0, 1):
            with self.subTest(target=target):
                infos = [info for info in sampled if info.player_id == target]
                self.assertTrue(any(info.get_current_turn() >= 2 for info in infos))
                expected = joint_law_posteriors(env.GameState(), infos, WeightedAgent())
                for info in infos:
                    actual = compute_posterior_given_opponent_agent(
                        env.GameState(), info, WeightedAgent())
                    self.assert_distribution(actual, expected[info])

    def test_matching_initial_decision_stops_immediately(self):
        state = dealt_state(0, (A, B))
        play(state, env.ActionType.CHANCE_DRAW, (C,), choice=0)
        posterior = compute_posterior_given_opponent_agent(
            state, state.get_info_state(0), NeverCalledAgent())
        self.assertEqual(posterior, {state.to_key(): 1.0})

    def test_wrong_round_and_terminal_initial_state_are_unreachable(self):
        state = dealt_state(0, (A, B))
        play(state, env.ActionType.CHANCE_DRAW, (C,), choice=0)
        info = state.get_info_state(0)
        wrong_round = replace(info, round_number=info.round_number + 1)
        self.assertEqual(compute_posterior_given_opponent_agent(
            state, wrong_round, NeverCalledAgent()), {})
        while not state.is_round_complete():
            if state.current_player == env.CHANCE_PLAYER:
                action, _ = next(state.get_chance_actions_with_probs())
            else:
                action = next(state.get_legal_actions())
            self.assertTrue(state.execute_action(action))
        self.assertEqual(compute_posterior_given_opponent_agent(
            state, info, NeverCalledAgent()), {})

    def test_initial_state_including_undo_history_is_preserved(self):
        initial = dealt_state(1, (A, B))
        target_state = copy.deepcopy(initial)
        play(target_state, env.ActionType.CHANCE_DRAW, (C,), choice=1)
        play(target_state, env.ActionType.RESERVE, (C,))
        play(target_state, env.ActionType.CHANCE_DRAW, (A,), choice=0)
        info = target_state.get_info_state(0)
        before = pickle.dumps(initial)
        compute_posterior_given_opponent_agent(initial, info, ReserveSignalAgent(1))
        self.assertEqual(pickle.dumps(initial), before)
        with self.assertRaisesRegex(AssertionError, "No opponent decision"):
            compute_posterior_given_opponent_agent(initial, info, NeverCalledAgent())
        self.assertEqual(pickle.dumps(initial), before)


if __name__ == "__main__":
    unittest.main()
