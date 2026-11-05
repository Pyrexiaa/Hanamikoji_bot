"""Deck utilities.

This engine treats cards as indistinguishable except by suit. Instead of an
ordered list "deck", we model the remaining cards as a multiset:

    remaining_card_counts[suit] = how many cards of that suit remain

The `Deck` class owns that multiset and provides the chance-node helpers:
- Enumerating distinct initial deals (pairs of hands) and draws
- Computing exact probabilities for those outcomes
- Sampling random deals/draws by remaining suit counts
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Dict, Generator, List, Optional, Sequence, Tuple

from .enums import ActionType
from .models import Action, Card


class Deck:
    """A multiset deck represented as remaining counts by suit."""

    def __init__(self, remaining_card_counts: Dict[object, int]):
        # Copy to prevent caller from holding an alias.
        self.remaining_card_counts: Dict[object, int] = dict(remaining_card_counts)

    @classmethod
    def full(cls, suits_enum: type) -> "Deck":
        """Create a fresh full deck for a round from a suit enum (using `suit.rank`)."""
        return cls({suit: suit.rank for suit in suits_enum})

    def total(self) -> int:
        """Total number of remaining cards in the deck."""
        return sum(self.remaining_card_counts.values())

    def nonzero_suit_counts(self) -> Dict[object, int]:
        """Return a compact {suit: count} view excluding 0-count suits."""
        return {suit: count for suit, count in self.remaining_card_counts.items() if count > 0}

    def draw_specific(self, suit: object) -> Optional[Card]:
        """Remove one card of `suit` if available and return a Card; otherwise None."""
        count = self.remaining_card_counts.get(suit, 0)
        if count <= 0:
            return None
        self.remaining_card_counts[suit] = count - 1
        return Card(suit)

    def remove_cards(self, cards: Sequence[Card]) -> bool:
        """Remove the given multiset of cards (by suit). Returns False if impossible."""
        needed: Dict[object, int] = {}
        for card in cards:
            suit = card.suit
            needed[suit] = needed.get(suit, 0) + 1
            if needed[suit] > self.remaining_card_counts.get(suit, 0):
                return False

        for suit, count in needed.items():
            self.remaining_card_counts[suit] -= count
        return True

    def _sorted_suits(self, suits: Sequence[object]) -> List[object]:
        # Suits are Enums in provided variants; `.value` gives a stable ordering.
        return sorted(suits, key=lambda suit: suit.value)

    def iter_distinct_suit_sequences(self, draw_count: int) -> Generator[Tuple[object, ...], None, None]:
        """Yield each distinct ordered suit sequence for `draw_count` draws.

        Sequences are drawn without replacement from the multiset described by
        `remaining_card_counts`.

        This uses a simple product+filter approach (readability first). The
        game variants in this repo have small `draw_count`, so this is fast.
        """
        remaining_total = self.total()
        if draw_count < 0 or draw_count > remaining_total:
            raise ValueError(f"draw_count must be in [0, {remaining_total}] (got {draw_count}).")

        suit_counts = self.nonzero_suit_counts()
        suits = self._sorted_suits(list(suit_counts.keys()))

        for suit_order in itertools.product(suits, repeat=draw_count):
            used: Dict[object, int] = {}
            ok = True
            for suit in suit_order:
                used[suit] = used.get(suit, 0) + 1
                if used[suit] > suit_counts[suit]:
                    ok = False
                    break
            if ok:
                yield suit_order

    def iter_distinct_suit_sequences_with_probs(
        self, draw_count: int
    ) -> Generator[Tuple[Tuple[object, ...], float], None, None]:
        """Yield each distinct suit sequence with its exact probability."""
        remaining_total = self.total()
        if draw_count < 0 or draw_count > remaining_total:
            raise ValueError(f"draw_count must be in [0, {remaining_total}] (got {draw_count}).")

        suits = self._sorted_suits(list(self.nonzero_suit_counts().keys()))

        for suit_order in itertools.product(suits, repeat=draw_count):
            counts = dict(self.remaining_card_counts)
            total = remaining_total
            prob = 1.0
            ok = True
            for suit in suit_order:
                count = counts.get(suit, 0)
                if count <= 0:
                    ok = False
                    break
                prob *= count / total
                counts[suit] = count - 1
                total -= 1
            if ok:
                yield suit_order, prob

    def count_distinct_suit_sequences(self, draw_count: int) -> int:
        """Count distinct ordered suit sequences for `draw_count` draws."""
        return sum(1 for _ in self.iter_distinct_suit_sequences(draw_count))

    def _iter_hands(self, counts: Dict[object, int], suits: List[object], size: int,
                    min_index: int = 0) -> Generator[Tuple[object, ...], None, None]:
        """Yield every hand of `size` cards drawable from `counts` as an
        ascending suit tuple, in lexicographic order.  `counts` is reduced
        while each hand is being yielded."""
        if size == 0:
            yield ()
            return
        for i in range(min_index, len(suits)):
            suit = suits[i]
            if counts[suit] > 0:
                counts[suit] -= 1
                for rest in self._iter_hands(counts, suits, size - 1, i):
                    yield (suit,) + rest
                counts[suit] += 1

    @staticmethod
    def _arrangements(hand: Tuple[object, ...]) -> int:
        """Distinct orderings of a hand: len! / prod(count of each suit)!."""
        result = math.factorial(len(hand))
        for suit in set(hand):
            result //= math.factorial(hand.count(suit))
        return result

    def iter_distinct_deals_with_probs(
        self, deal_count: int
    ) -> Generator[Tuple[Tuple[object, ...], float], None, None]:
        """Yield each distinct initial deal of `deal_count` cards with its exact probability.

        A deal is identified by the two hands as multisets, so each is listed
        once, as the canonical interleaved sequence (position i goes to player
        i % 2; each player's cards in ascending suit order; see Action).
        Order: hand0 then hand1, each as an ascending suit tuple, lexicographic.
        With one card per player this is exactly the ordered-sequence listing.

        Probability: P(that interleaved sequence), multiplied card by card as
        `iter_distinct_suit_sequences_with_probs` does, times the number of
        sequences giving the same pair of hands.
        """
        remaining_total = self.total()
        if deal_count < 0 or deal_count % 2 or deal_count > remaining_total:
            raise ValueError(
                f"deal_count must be even and in [0, {remaining_total}] (got {deal_count})."
            )
        hand_size = deal_count // 2
        counts = self.nonzero_suit_counts()
        suits = self._sorted_suits(list(counts.keys()))

        for hand0 in self._iter_hands(counts, suits, hand_size):
            for hand1 in self._iter_hands(counts, suits, hand_size):
                sequence = tuple(card for pair in zip(hand0, hand1) for card in pair)
                remaining = dict(self.remaining_card_counts)
                total = remaining_total
                prob = 1.0
                for suit in sequence:
                    prob *= remaining[suit] / total
                    remaining[suit] -= 1
                    total -= 1
                prob *= self._arrangements(hand0) * self._arrangements(hand1)
                yield sequence, prob

    def count_distinct_deals(self, deal_count: int) -> int:
        """Count distinct initial deals (pairs of hands) of `deal_count` cards."""
        return sum(1 for _ in self.iter_distinct_deals_with_probs(deal_count))

    def iter_distinct_draw_suits(self) -> Generator[object, None, None]:
        """Yield each distinct drawable suit (count > 0), in deterministic order."""
        suit_counts = self.nonzero_suit_counts()
        for suit in self._sorted_suits(list(suit_counts.keys())):
            yield suit

    def sample_deal_cards(self, deal_count: int, rng: Optional[random.Random] = None) -> List[Card]:
        """Sample a deal sequence as a list of Cards, without replacement."""
        remaining_total = self.total()
        if deal_count < 0 or deal_count > remaining_total:
            raise ValueError("Not enough remaining cards to sample a deal.")

        rand = rng if rng is not None else random
        remaining = dict(self.remaining_card_counts)

        dealt: List[Card] = []
        for _ in range(deal_count):
            suits = [suit for suit, count in remaining.items() if count > 0]
            if not suits:
                raise ValueError("Unexpected empty remaining-card pool during deal sampling.")
            weights = [remaining[suit] for suit in suits]
            suit = rand.choices(suits, weights=weights, k=1)[0]
            remaining[suit] -= 1
            dealt.append(Card(suit))

        return dealt

    def sample_draw_suit(self, rng: Optional[random.Random] = None) -> object:
        """Sample one suit proportional to remaining counts."""
        rand = rng if rng is not None else random
        suits = [suit for suit, count in self.remaining_card_counts.items() if count > 0]
        if not suits:
            raise ValueError("No remaining cards available to sample a draw.")
        weights = [self.remaining_card_counts[suit] for suit in suits]
        return rand.choices(suits, weights=weights, k=1)[0]

    # ---------------------------------------------------------------------
    # Chance action helpers (Action objects)
    # ---------------------------------------------------------------------

    def iter_chance_deal_actions(self, deal_count: int) -> Generator[Action, None, None]:
        """Yield one `CHANCE_DEAL` action per distinct deal (pair of hands)."""
        for sequence, _ in self.iter_distinct_deals_with_probs(deal_count):
            yield Action(ActionType.CHANCE_DEAL, [Card(suit) for suit in sequence])

    def iter_chance_draw_actions(self, pending_draw_player: int) -> Generator[Action, None, None]:
        """Yield one distinct `CHANCE_DRAW` action per distinct drawable suit."""
        if pending_draw_player not in (0, 1):
            raise ValueError("pending_draw_player must be 0 or 1.")
        for suit in self.iter_distinct_draw_suits():
            yield Action(ActionType.CHANCE_DRAW, [Card(suit)], choice=pending_draw_player)

    def sample_chance_deal_action(self, deal_count: int, rng: Optional[random.Random] = None) -> Action:
        """Sample a `CHANCE_DEAL` action without full enumeration.

        Cards are drawn in sequence; the Action sorts each player's hand.
        """
        return Action(ActionType.CHANCE_DEAL, self.sample_deal_cards(deal_count, rng))

    def sample_chance_draw_action(self, pending_draw_player: int, rng: Optional[random.Random] = None) -> Action:
        """Sample a `CHANCE_DRAW` action by remaining suit counts."""
        if pending_draw_player not in (0, 1):
            raise ValueError("pending_draw_player must be 0 or 1.")
        suit = self.sample_draw_suit(rng)
        return Action(ActionType.CHANCE_DRAW, [Card(suit)], choice=pending_draw_player)

    def iter_chance_deal_actions_with_probs(
        self, deal_count: int
    ) -> Generator[Tuple[Action, float], None, None]:
        """Yield `(CHANCE_DEAL action, probability)` pairs, one per distinct deal (exact)."""
        for sequence, prob in self.iter_distinct_deals_with_probs(deal_count):
            yield (Action(ActionType.CHANCE_DEAL, [Card(suit) for suit in sequence]), prob)

    def iter_chance_draw_actions_with_probs(
        self, pending_draw_player: int
    ) -> Generator[Tuple[Action, float], None, None]:
        """Yield `(CHANCE_DRAW action, probability)` pairs."""
        if pending_draw_player not in (0, 1):
            raise ValueError("pending_draw_player must be 0 or 1.")
        total_cards = self.total()
        if total_cards <= 0:
            raise ValueError("Cannot enumerate CHANCE_DRAW with empty remaining-card pool.")

        suit_counts = self.nonzero_suit_counts()
        for suit in self._sorted_suits(list(suit_counts.keys())):
            prob = suit_counts[suit] / total_cards
            yield (Action(ActionType.CHANCE_DRAW, [Card(suit)], choice=pending_draw_player), prob)
