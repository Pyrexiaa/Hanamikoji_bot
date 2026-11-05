"""Core immutable/mutable value objects for the engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Set, Tuple

from .enums import OFFER_RESPONSES, ActionType


@dataclass(frozen=True)
class GameConfig:
    """Configuration that defines a Hanamikoji variant."""

    name: str
    suits: type
    available_actions: FrozenSet[ActionType]
    initial_hand_size: int
    win_favor_points: Optional[int]
    win_geisha_count: int
    # False when the game tree is far too large to enumerate (full
    # Hanamikoji).  Tree-walking tools (treeplex extraction, CFR, the Markov
    # solver) check this and refuse instead of running out of memory.  Play,
    # sampling and single chance nodes still work.
    enumerable: bool = True

    @property
    def turns_per_round(self) -> int:
        """Total turns per round = 2 * number of available actions."""

        return 2 * len(self.available_actions)


def require_enumerable(config: Optional[GameConfig], what: str) -> None:
    """Raise if `config`'s game tree is too large to enumerate.

    Called by tools that walk the whole game tree (treeplex extraction, CFR,
    the Markov solver) so they fail at once instead of running out of memory.
    `config` may be None (a default-constructed tiny_hana state).
    """
    if config is not None and not config.enumerable:
        raise ValueError(
            f"{config.name}: the game tree is far too large to enumerate, so {what} "
            "is not supported. Use sampling-based methods (simulation, Monte Carlo "
            "search or sampled CFR) instead."
        )


@dataclass(frozen=True)
class Card:
    """Represents a single card in the game."""

    suit: object

    def __repr__(self):
        return f"Card({self.suit.name})"


@dataclass(frozen=True, init=False, repr=False)
class Action:
    """Represents a player's action."""

    action_type: ActionType
    cards: Tuple[Card, ...]
    choice: Optional[int]

    def __init__(self, action_type: ActionType, cards: Sequence[Card], choice: Optional[int] = None):
        object.__setattr__(self, "action_type", action_type)
        # Defensive copy prevents caller-side list mutation from affecting action payload.
        cards_tuple = tuple(cards)
        # Canonicalize card order so that equal actions (and the info states
        # they produce) are identical regardless of the order the caller used.
        #
        # CHANCE_DEAL gives cards[i] to player i % 2.  A hand is a multiset, so
        # the order cards are dealt in carries no information: each player's
        # cards (even / odd positions) are sorted by suit value.
        if action_type == ActionType.CHANCE_DEAL:
            hands = (sorted(cards_tuple[0::2], key=lambda c: c.suit.value),
                     sorted(cards_tuple[1::2], key=lambda c: c.suit.value))
            cards_tuple = tuple(hands[i % 2][i // 2] for i in range(len(cards_tuple)))
        # TRADE_OFF and GIFT: card order carries no information.
        elif action_type in (ActionType.TRADE_OFF, ActionType.GIFT):
            cards_tuple = tuple(sorted(cards_tuple, key=lambda c: c.suit.value))
        # COMPETITION: cards = pair 0 + pair 1.  Each pair is sorted, then the
        # two pairs are ordered.  (The response's choice indexes the pairs.)
        elif action_type == ActionType.COMPETITION and len(cards_tuple) == 4:
            pairs = sorted(
                (tuple(sorted(cards_tuple[i:i + 2], key=lambda c: c.suit.value)) for i in (0, 2)),
                key=lambda pair: (pair[0].suit.value, pair[1].suit.value),
            )
            cards_tuple = pairs[0] + pairs[1]
        # SPLIT and DISCARD: two cards in suit order.
        if action_type in (ActionType.SPLIT, ActionType.DISCARD) and len(cards_tuple) == 2:
            c0, c1 = cards_tuple
            if c0.suit.value > c1.suit.value:
                cards_tuple = (c1, c0)
                # choice is a direct index into cards (revealed card for DISCARD),
                # so it must be flipped when the cards are swapped.
                if choice is not None:
                    choice = 1 - choice
        object.__setattr__(self, "cards", cards_tuple)
        object.__setattr__(self, "choice", choice)

    # Number of cards each card-playing action takes from the hand.
    CARD_COUNTS = {
        ActionType.RESERVE: 1,
        ActionType.DISCARD: 2,
        ActionType.SPLIT: 2,
        ActionType.TRADE_OFF: 2,
        ActionType.GIFT: 3,
        ActionType.COMPETITION: 4,
    }

    def validate(
        self,
        hand: List[Card],
        used_actions: Set[ActionType],
        pending_split: bool = False,
        available_actions: Optional[FrozenSet[ActionType]] = None,
        pending_offer: Optional[ActionType] = None,
    ) -> bool:
        """Validate if this action is legal given the current hand and used actions.

        `pending_offer` is the type of offer (SPLIT, GIFT, COMPETITION) awaiting
        this player's response, if any.  `pending_split=True` is the older
        spelling of `pending_offer=ActionType.SPLIT`.
        """

        # Chance actions are validated separately inside execute_action
        if self.action_type in (ActionType.CHANCE_START, ActionType.CHANCE_DEAL, ActionType.CHANCE_DRAW):
            return True

        if pending_offer is None and pending_split:
            pending_offer = ActionType.SPLIT

        # A response is only valid for the matching pending offer, and while an
        # offer is pending nothing else is.
        if pending_offer is not None:
            response_type, n_choices = OFFER_RESPONSES[pending_offer]
            # (As before, a response's cards are ignored.)
            return self.action_type == response_type and self.choice in range(n_choices)

        if self.action_type not in self.CARD_COUNTS:
            return False

        # Check action is available in this variant
        if available_actions is not None and self.action_type not in available_actions:
            return False

        if self.action_type in used_actions:
            return False

        hand_copy = hand.copy()
        for card in self.cards:
            if card not in hand_copy:
                return False
            hand_copy.remove(card)

        if len(self.cards) != self.CARD_COUNTS[self.action_type]:
            return False
        if self.action_type == ActionType.DISCARD:
            if self.choice not in [0, 1]:
                return False
        elif self.action_type in (ActionType.TRADE_OFF, ActionType.GIFT, ActionType.COMPETITION):
            if self.choice is not None:
                return False

        return True

    def __repr__(self):
        return f"Action({self.action_type.name}, {self.cards}, choice={self.choice})"
