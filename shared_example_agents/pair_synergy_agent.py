"""PairSynergyAgent: favor duplicate-suit synergies when available."""

from __future__ import annotations

from collections import Counter
from typing import Dict

from engine import Action, ActionType

from ._base import HeuristicAgentBase, HeuristicContext


class PairSynergyAgent(HeuristicAgentBase):
    """Prefers actions built from same-suit pairs."""

    def _suit_priority(self, ctx: HeuristicContext) -> Dict[object, float]:
        priority: Dict[object, float] = {}
        for suit in ctx.suits:
            diff = ctx.my_counts[suit] - ctx.opp_counts[suit]
            score = float(suit.rank)
            if diff <= 0:
                score += 1.5
            if ctx.favors.get(suit, 0) != -1:
                score += 0.5
            priority[suit] = score
        return priority

    def _action_score(
        self,
        action: Action,
        ctx: HeuristicContext,
        suit_priority: Dict[object, float],
    ) -> float:
        base = self._default_action_score(action, ctx, suit_priority)
        hand_counts = Counter(card.suit for card in ctx.hand)

        if action.action_type == ActionType.RESERVE:
            suit = action.cards[0].suit
            if hand_counts.get(suit, 0) >= 2:
                base += 2.5

        if action.action_type in (ActionType.DISCARD, ActionType.SPLIT):
            if action.cards[0].suit == action.cards[1].suit:
                base += 4.0
            elif action.action_type == ActionType.SPLIT:
                base -= 0.25

        if action.action_type == ActionType.SPLIT_RESPONSE and ctx.pending_split is not None:
            chosen = ctx.pending_split[action.choice].suit
            if hand_counts.get(chosen, 0) >= 1:
                base += 0.75

        # Full Hanamikoji: same synergies.
        if action.action_type == ActionType.TRADE_OFF and action.cards[0].suit == action.cards[1].suit:
            base += 4.0
        if action.action_type == ActionType.COMPETITION:
            for pair in (action.cards[0:2], action.cards[2:4]):
                if pair[0].suit == pair[1].suit:
                    base += 2.0
        if action.action_type in (ActionType.GIFT_RESPONSE, ActionType.COMPETITION_RESPONSE):
            taken = self._offer_taken(ctx, action) or ()
            base += 0.75 * sum(1 for card in taken if hand_counts.get(card.suit, 0) >= 1)

        return base
