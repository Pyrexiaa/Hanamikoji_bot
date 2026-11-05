"""
Full Hanamikoji - the published two-player card game.

Game Overview:
- 21 cards across 7 geishas (suits) with ranks [2, 2, 2, 3, 3, 4, 5]; a
  geisha's rank is both its number of cards and its charm points
- Each round: 6 cards each, then a draw at the start of every turn; one card
  stays undealt (the card set aside face down in the physical game)
- Each player takes 4 actions per round (8 turns); a geisha's favor goes to
  whoever has more of her cards at round end, and stays put on a tie
- Win at the end of a round with 11+ charm points, otherwise with 4+
  geishas; the starting player alternates between rounds

Actions (each used exactly once per round):
1. RESERVE (secret): keep 1 card face down; it counts at round end
2. TRADE_OFF: discard 2 cards face down; they take no part in the round
3. GIFT: offer 3 cards face up; the opponent takes 1, you keep the other 2
4. COMPETITION: offer 2 pairs face up; the opponent takes a pair, you keep the other

The game tree is far too large to enumerate (GAME_CONFIG.enumerable is
False): play, sampling and single chance nodes work, but exhaustive
tree-walking tools (treeplex extraction, CFR, the Markov solver) refuse it.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from engine import (
    CHANCE_PLAYER, GameConfig, ActionType, ObservationType, ObservationEvent,
    RoundStartEvent, DrawCardEvent, OwnReserveEvent, OpponentReserveEvent,
    OwnTradeOffEvent, OpponentTradeOffEvent, OwnGiftEvent, OpponentGiftEvent,
    GiftChoiceEvent, OwnCompetitionEvent, OpponentCompetitionEvent,
    CompetitionChoiceEvent, RoundEndEvent,
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
# Suit enum — 7 geishas, ranks [2, 2, 2, 3, 3, 4, 5]  (21 cards total)
# ---------------------------------------------------------------------------

@register_suit_enum
class Suit(Enum):
    """The seven geishas with their ranks (charm points = number of cards)."""
    GEISHA_1 = 0  # Rank 2
    GEISHA_2 = 1  # Rank 2
    GEISHA_3 = 2  # Rank 2
    GEISHA_4 = 3  # Rank 3
    GEISHA_5 = 4  # Rank 3
    GEISHA_6 = 5  # Rank 4
    GEISHA_7 = 6  # Rank 5

    @property
    def rank(self):
        return _RANKS[self.value]


_RANKS = (2, 2, 2, 3, 3, 4, 5)


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------

GAME_CONFIG = GameConfig(
    name="Full Hanamikoji",
    suits=Suit,
    available_actions=frozenset({
        _PyActionType.RESERVE, _PyActionType.TRADE_OFF,
        _PyActionType.GIFT, _PyActionType.COMPETITION,
    }),
    initial_hand_size=6,
    win_favor_points=11,
    win_geisha_count=4,
    enumerable=False,
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
    'OwnTradeOffEvent',
    'OpponentTradeOffEvent',
    'OwnGiftEvent',
    'OpponentGiftEvent',
    'GiftChoiceEvent',
    'OwnCompetitionEvent',
    'OpponentCompetitionEvent',
    'CompetitionChoiceEvent',
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
