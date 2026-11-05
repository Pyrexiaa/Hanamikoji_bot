"""
Tiny Hanamikoji - A simplified version of the Hanamikoji card game.

Game Overview:
- 7 cards total across 3 geishas (suits) with ranks [2, 2, 3]
- 2 players compete for geisha favors
- Each player starts with 1 card and takes 2 actions per round
- Win by controlling 2 or more geishas

Actions (each can only be used once per round):
1. RESERVE: Take 1 card and hide it (you get it at end of round)
2. SPLIT: Choose 2 cards, opponent picks one, you get the other
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
from engine.enums import ActionType as _PyActionType  # always Python ActionType for GAME_CONFIG


# ---------------------------------------------------------------------------
# Suit enum — 3 geishas, ranks [2, 2, 3]  (7 cards total)
# ---------------------------------------------------------------------------

@register_suit_enum
class Suit(Enum):
    """The three geishas with their ranks."""
    GEISHA_1 = 0  # Rank 2
    GEISHA_2 = 1  # Rank 2
    GEISHA_3 = 2  # Rank 3

    @property
    def rank(self):
        rank_map = {
            Suit.GEISHA_1: 2,
            Suit.GEISHA_2: 2,
            Suit.GEISHA_3: 3,
        }
        return rank_map[self]


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------

GAME_CONFIG = GameConfig(
    name="Tiny Hanamikoji",
    suits=Suit,
    available_actions=frozenset({_PyActionType.RESERVE, _PyActionType.SPLIT}),
    initial_hand_size=1,
    win_favor_points=None,
    win_geisha_count=2,
)


# ---------------------------------------------------------------------------
# Thin subclasses with default config
# ---------------------------------------------------------------------------

class GameState(_BaseGameState):
    def __init__(self, config: GameConfig = GAME_CONFIG):
        super().__init__(config)


class Game(_BaseGame):
    def __init__(self, agent0: Agent, agent1: Agent, verbose: bool = False,
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

if __name__ == "__main__":
    agent0 = RandomAgent()
    agent1 = RandomAgent()
    game = Game(agent0, agent1, verbose=True)
    winner = game.play_game()
    print(f"\nFinal winner: Player {winner}")
