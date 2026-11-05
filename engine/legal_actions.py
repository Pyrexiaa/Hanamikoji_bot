"""Shared legal-action enumeration helpers for player decision nodes.

Ordering guarantee:
- Actions are yielded in a canonical, deterministic order.
- For player actions, ordering is based on suit-value ordering, not the
  incidental in-hand card order.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Generator, List, Sequence, Set, Tuple

from .enums import OFFER_RESPONSES, ActionType
from .models import Action, Card


def iter_split_response_actions() -> Generator[Action, None, None]:
    """Yield the two legal SPLIT_RESPONSE actions."""
    yield from iter_offer_response_actions(ActionType.SPLIT)


def iter_offer_response_actions(offer_type: ActionType) -> Generator[Action, None, None]:
    """Yield the legal responses to a pending SPLIT / GIFT / COMPETITION offer.

    Every choice index is listed, even when two offered cards (or pairs) are
    the same and the choices lead to the same outcome (as for SPLIT_RESPONSE).
    """
    response_type, n_choices = OFFER_RESPONSES[offer_type]
    for choice in range(n_choices):
        yield Action(response_type, [], choice=choice)


def _iter_suit_multisets(
    suits_sorted: Sequence[object], counts: Dict[object, int], size: int, start: int = 0
) -> Generator[Tuple[object, ...], None, None]:
    """Yield each multiset of `size` suits available in `counts` as an
    ascending tuple, in lexicographic order."""
    if size == 0:
        yield ()
        return
    for i in range(start, len(suits_sorted)):
        suit = suits_sorted[i]
        if counts[suit] > 0:
            counts[suit] -= 1
            for rest in _iter_suit_multisets(suits_sorted, counts, size - 1, i):
                yield (suit,) + rest
            counts[suit] += 1


def iter_player_legal_actions(
    hand: Sequence[Card],
    used_actions: Set[ActionType],
    available_actions: FrozenSet[ActionType],
) -> Generator[Action, None, None]:
    """Yield deduplicated legal non-chance actions for a player node."""
    # Group cards by suit; preserve per-suit index order for deterministic
    # representative selection when multiple identical-suit cards exist.
    suit_to_indices: Dict[object, List[int]] = {}
    for idx, card in enumerate(hand):
        suit_to_indices.setdefault(card.suit, []).append(idx)

    suits_sorted = sorted(suit_to_indices.keys(), key=lambda suit: suit.value)

    # RESERVE: one action per suit, ordered by suit value.
    if ActionType.RESERVE in available_actions and ActionType.RESERVE not in used_actions:
        for suit in suits_sorted:
            idx = suit_to_indices[suit][0]
            yield Action(ActionType.RESERVE, [hand[idx]])

    if len(hand) < 2:
        return

    # DISCARD: ordered by (low_suit, high_suit), then choice (0 before 1).
    # For distinct suits, yield two choices against one canonical card order.
    # For same-suit pairs, only one unique reveal exists (choice=0).
    if ActionType.DISCARD in available_actions and ActionType.DISCARD not in used_actions:
        for i, suit_a in enumerate(suits_sorted):
            for suit_b in suits_sorted[i:]:
                if suit_a == suit_b:
                    idxs = suit_to_indices[suit_a]
                    if len(idxs) < 2:
                        continue
                    cards = [hand[idxs[0]], hand[idxs[1]]]
                    yield Action(ActionType.DISCARD, cards, choice=0)
                    continue

                idx_a = suit_to_indices[suit_a][0]
                idx_b = suit_to_indices[suit_b][0]
                cards = [hand[idx_a], hand[idx_b]]
                yield Action(ActionType.DISCARD, cards, choice=0)
                yield Action(ActionType.DISCARD, cards, choice=1)

    # SPLIT: ordered by (low_suit, high_suit) in canonical card order.
    if ActionType.SPLIT in available_actions and ActionType.SPLIT not in used_actions:
        for i, suit_a in enumerate(suits_sorted):
            for suit_b in suits_sorted[i:]:
                if suit_a == suit_b:
                    idxs = suit_to_indices[suit_a]
                    if len(idxs) < 2:
                        continue
                    yield Action(ActionType.SPLIT, [hand[idxs[0]], hand[idxs[1]]])
                    continue

                idx_a = suit_to_indices[suit_a][0]
                idx_b = suit_to_indices[suit_b][0]
                yield Action(ActionType.SPLIT, [hand[idx_a], hand[idx_b]])

    # Full Hanamikoji actions, each over distinct suit multisets in
    # lexicographic order of the canonical card tuple.
    counts = {suit: len(idxs) for suit, idxs in suit_to_indices.items()}

    # TRADE_OFF: (low_suit, high_suit).
    if ActionType.TRADE_OFF in available_actions and ActionType.TRADE_OFF not in used_actions:
        for pair in _iter_suit_multisets(suits_sorted, counts, 2):
            yield Action(ActionType.TRADE_OFF, [Card(suit) for suit in pair])

    # GIFT: (a, b, c) with a <= b <= c.
    if ActionType.GIFT in available_actions and ActionType.GIFT not in used_actions:
        for triple in _iter_suit_multisets(suits_sorted, counts, 3):
            yield Action(ActionType.GIFT, [Card(suit) for suit in triple])

    # COMPETITION: pair0 + pair1, each pair ascending and pair0 <= pair1, so
    # each way of splitting 4 cards into two pairs is listed once.
    if ActionType.COMPETITION in available_actions and ActionType.COMPETITION not in used_actions:
        pairs = list(_iter_suit_multisets(suits_sorted, counts, 2))
        for i, pair0 in enumerate(pairs):
            for pair1 in pairs[i:]:
                need: Dict[object, int] = {}
                for suit in pair0 + pair1:
                    need[suit] = need.get(suit, 0) + 1
                if all(counts[suit] >= n for suit, n in need.items()):
                    yield Action(ActionType.COMPETITION, [Card(suit) for suit in pair0 + pair1])
