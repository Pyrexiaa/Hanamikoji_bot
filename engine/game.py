"""Game controller for running full matches."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .agents import Agent
from .constants import CHANCE_PLAYER
from .enums import ActionType
from .models import Action, GameConfig
from .state import GameState


class Game:
    """Main game controller, parameterized by GameConfig."""

    def __init__(self, agent0: Agent, agent1: Agent, config: GameConfig, verbose: bool = True, seed: Optional[int] = None):
        self.config = config
        self.state = GameState(config)
        self.agents = [agent0, agent1]
        self.verbose = verbose
        self.rng = random.Random(seed)

    @staticmethod
    def sample_action_from_distribution(
        distribution: Dict[Action, float],
        rng: Optional[random.Random] = None,
    ) -> Action:
        """Validate and sample one action from an action-probability dict."""
        if not distribution:
            raise ValueError("Action distribution cannot be empty.")

        total_prob = 0.0
        actions: List[Action] = []
        weights: List[float] = []
        for action, prob in distribution.items():
            if not isinstance(action, Action):
                raise TypeError("Distribution action must be an Action instance.")
            if prob < 0:
                raise ValueError("Action probabilities must be non-negative.")
            weight = float(prob)
            total_prob += weight
            actions.append(action)
            weights.append(weight)

        if total_prob <= 0:
            raise ValueError("Action distribution must have positive total probability.")

        # Validate probabilities sum to 1.0 (with floating point tolerance)
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(f"Action probabilities must sum to 1.0, got {total_prob:.6f}")

        sampler = rng if rng is not None else random
        return sampler.choices(actions, weights=weights, k=1)[0]

    def log(self, message: str):
        if self.verbose:
            print(message)

    def play_decision(self):
        """Execute a single decision point in the game tree."""
        if self.state.current_player is None:
            raise RuntimeError("Round is complete; begin the next round before playing more decisions.")

        # Handle chance nodes automatically
        if self.state.current_player == CHANCE_PLAYER:
            action = self.state.sample_chance_action(self.rng)
            if action.action_type == ActionType.CHANCE_START:
                self.log(f"\nChance chooses starting player: Player {action.choice}")
            elif action.action_type == ActionType.CHANCE_DEAL:
                self.log("\nChance deals initial cards")
            elif action.action_type == ActionType.CHANCE_DRAW:
                self.log(f"\nChance draws for Player {action.choice}")
            if not self.state.execute_action(action):
                raise ValueError(f"Invalid chance action: {action}")
            return

        current_player = self.state.current_player
        agent = self.agents[current_player]

        pending_offer = self.state.pending_offer

        if pending_offer is None:
            self.log(f"\n--- Turn {self.state.turn_count + 1}, Player {current_player} ---")
            self.log(f"Player {current_player} hand size: {len(self.state.players[current_player].hand)}")
        else:
            self.log(f"\n--- Player {current_player} responds to {pending_offer[1].name} ---")

        info_state = self.state.get_info_state(current_player)
        distribution = agent.get_action_distribution(info_state)
        action = self.sample_action_from_distribution(distribution, self.rng)

        # Log the action
        if action.action_type == ActionType.RESERVE:
            self.log(f"Player {current_player} plays: RESERVE (card hidden)")
        elif action.action_type == ActionType.DISCARD:
            revealed = action.cards[action.choice]
            self.log(f"Player {current_player} plays: DISCARD - revealed: {revealed}, hidden: (card hidden)")
        elif action.action_type == ActionType.SPLIT:
            self.log(f"Player {current_player} plays: SPLIT with {action.cards}")
        elif action.action_type == ActionType.SPLIT_RESPONSE:
            offerer_id, cards = self.state.pending_split
            self.log(f"Player {current_player} chooses card {action.choice}: {cards[action.choice]}")
        elif action.action_type == ActionType.TRADE_OFF:
            self.log(f"Player {current_player} plays: TRADE_OFF (2 cards hidden)")
        elif action.action_type == ActionType.GIFT:
            self.log(f"Player {current_player} plays: GIFT with {action.cards}")
        elif action.action_type == ActionType.COMPETITION:
            self.log(f"Player {current_player} plays: COMPETITION with pairs "
                     f"{action.cards[0:2]} / {action.cards[2:4]}")
        elif action.action_type == ActionType.GIFT_RESPONSE:
            _, _, cards = pending_offer
            self.log(f"Player {current_player} takes card {action.choice}: {cards[action.choice]}")
        elif action.action_type == ActionType.COMPETITION_RESPONSE:
            _, _, cards = pending_offer
            pair = cards[2 * action.choice:2 * action.choice + 2]
            self.log(f"Player {current_player} takes pair {action.choice}: {pair}")

        if not self.state.execute_action(action):
            raise ValueError(f"Invalid action selected by agent {current_player}: {action}")

    def play_step(self):
        """Execute one gameplay step (one action, plus the response to an offer)."""
        self.play_decision()
        if self.state.pending_offer is not None:
            self.play_decision()

    def play_round(self):
        # If a previous round finished, automatically begin the next one.
        if self.state.phase == "round_complete":
            self.state._begin_round()
        if self.state.winner is not None:
            raise ValueError("Game already has a winner; cannot play another round.")

        self.log(f"\n{'='*50}")
        self.log(f"ROUND {self.state.round_count}")
        self.log(f"{'='*50}")

        # Resolve chance nodes (choose starting player, deal cards)
        while self.state.current_player == CHANCE_PLAYER:
            self.play_decision()

        while not self.state.is_round_complete():
            self.play_step()

        self.state.update_favors()

        self.log(f"\n--- End of Round {self.state.round_count} ---")
        self.log("Card counts:")
        for i, player in enumerate(self.state.players):
            counts = player.get_suit_count()
            self.log(f"  Player {i}: {counts}")

        self.log("Favors:")
        for suit, favor in self.state.favors.items():
            owner = "Player 0" if favor == -1 else "Player 1" if favor == 1 else "Neutral"
            self.log(f"  {suit.name} (rank {suit.rank}): {owner}")

        self.state.winner = self.state.check_winner()

        if self.state.winner is not None:
            self.log(f"\n*** Player {self.state.winner} WINS! ***")
        else:
            self.log("\nNo winner yet, starting new round...")

    def play_game(self, max_rounds: int = 10) -> int:
        """Play a complete game. Returns winner (0 or 1) or -1 for draw."""
        rounds_played = 0
        while self.state.winner is None and rounds_played < max_rounds:
            self.play_round()
            rounds_played += 1

        if self.state.winner is None:
            self.log(f"\nGame reached maximum rounds ({max_rounds}). Draw!")
            return -1

        return self.state.winner
