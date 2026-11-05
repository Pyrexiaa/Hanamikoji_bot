"""
Hanamikoji - The standard variant of the Hanamikoji card game.

Game Overview:
- 11 cards total across 5 geishas (suits) with ranks [1, 1, 2, 3, 4]
- 2 players compete for geisha favors
- Each player starts with 2 cards and takes 3 actions per round
- Win by controlling 6+ favor points OR 3+ geishas

Actions (each can only be used once per round):
1. RESERVE: Take 1 card and hide it (you get it at end of round)
2. DISCARD: Choose 2 cards, reveal one, hide the other (both removed)
3. SPLIT: Choose 2 cards, opponent picks one, you get the other
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from engine import (
    CHANCE_PLAYER, GameConfig, ActionType, ObservationType, ObservationEvent,
    RoundStartEvent, DrawCardEvent, OwnReserveEvent, OpponentReserveEvent,
    OwnDiscardEvent, OpponentDiscardEvent, OwnSplitEvent, OpponentSplitEvent,
    SplitChoiceEvent, RoundEndEvent,
    RoundInfoState, Card, Action, PlayerState, Agent, RandomAgent,
    POLICY_TABLE_FORMAT, POLICY_TABLE_KEY_ENCODING, PolicyTableAgent,
    MARKOV_POLICY_TABLE_KEY_ENCODING, MarkovPolicyTableAgent, markov_policy_key,
    COMPACT_POLICY_TABLE_KEY_ENCODING, CompactPolicyTableAgent, load_policy_table_agent,
    dump_policy_table_artifact, load_policy_table_artifact,
    GameState as _BaseGameState, Game as _BaseGame,
    register_suit_enum,
)


# ---------------------------------------------------------------------------
# Suit enum — 5 geishas, ranks [1, 1, 2, 3, 4]  (11 cards total)
# ---------------------------------------------------------------------------

@register_suit_enum
class Suit(Enum):
    """Represents the 5 geisha suits with their respective ranks/values."""
    GEISHA_1 = 0  # Rank 1
    GEISHA_2 = 1  # Rank 1
    GEISHA_3 = 2  # Rank 2
    GEISHA_4 = 3  # Rank 3
    GEISHA_5 = 4  # Rank 4

    @property
    def rank(self):
        rank_map = {
            Suit.GEISHA_1: 1,
            Suit.GEISHA_2: 1,
            Suit.GEISHA_3: 2,
            Suit.GEISHA_4: 3,
            Suit.GEISHA_5: 4,
        }
        return rank_map[self]


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------

GAME_CONFIG = GameConfig(
    name="Hanamikoji",
    suits=Suit,
    available_actions=frozenset({ActionType.RESERVE, ActionType.DISCARD, ActionType.SPLIT}),
    initial_hand_size=2,
    win_favor_points=6,
    win_geisha_count=3,
)


# ---------------------------------------------------------------------------
# Thin subclasses with default config
# ---------------------------------------------------------------------------

class GameState(_BaseGameState):
    def __init__(self, config: GameConfig = GAME_CONFIG):
        super().__init__(config)


class Game(_BaseGame):
    def __init__(self, agent0: Agent, agent1: Agent, verbose: bool = True,
                 seed: Optional[int] = None):
        super().__init__(agent0, agent1, GAME_CONFIG, verbose, seed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Variant-specific
    'Suit',
    'GAME_CONFIG',
    'GameState',
    'Game',
    # Re-exported from engine
    'CHANCE_PLAYER',
    'GameConfig',
    'ActionType',
    'ObservationType',
    'ObservationEvent',
    'RoundStartEvent',
    'DrawCardEvent',
    'OwnReserveEvent',
    'OpponentReserveEvent',
    'OwnDiscardEvent',
    'OpponentDiscardEvent',
    'OwnSplitEvent',
    'OpponentSplitEvent',
    'SplitChoiceEvent',
    'RoundEndEvent',
    'RoundInfoState',
    'Card',
    'Action',
    'PlayerState',
    'Agent',
    'RandomAgent',
    'POLICY_TABLE_FORMAT',
    'POLICY_TABLE_KEY_ENCODING',
    'PolicyTableAgent',
    'MARKOV_POLICY_TABLE_KEY_ENCODING',
    'MarkovPolicyTableAgent',
    'COMPACT_POLICY_TABLE_KEY_ENCODING',
    'CompactPolicyTableAgent',
    'load_policy_table_agent',
    'markov_policy_key',
    'dump_policy_table_artifact',
    'load_policy_table_artifact',
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Example usage of the simulator."""
    print("Hanamikoji Simulator - Example Game\n")

    agent0 = RandomAgent()
    agent1 = RandomAgent()

    game = Game(agent0, agent1, verbose=True)
    winner = game.play_game(max_rounds=5)

    print(f"\n{'='*50}")
    print(f"Final Result: Player {winner} wins!" if winner >= 0 else "Draw!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
