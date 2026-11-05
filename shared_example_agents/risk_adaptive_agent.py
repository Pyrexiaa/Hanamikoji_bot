"""RiskAdaptiveAgent: switch between safe and aggressive posture."""

from __future__ import annotations

from typing import Dict

from engine import Action, ActionType

from ._base import HeuristicAgentBase, HeuristicContext


class RiskAdaptiveAgent(HeuristicAgentBase):
    """Risk posture follows current favor-board situation."""

    @staticmethod
    def _mode(ctx: HeuristicContext) -> str:
        my_favors = sum(1 for owner in ctx.favors.values() if owner == -1)
        opp_favors = sum(1 for owner in ctx.favors.values() if owner == 1)
        if my_favors > opp_favors:
            return "safe"
        if my_favors < opp_favors:
            return "aggressive"
        return "neutral"

    def _suit_priority(self, ctx: HeuristicContext) -> Dict[object, float]:
        mode = self._mode(ctx)
        priority: Dict[object, float] = {}

        for suit in ctx.suits:
            diff = ctx.my_counts[suit] - ctx.opp_counts[suit]
            favor = ctx.favors.get(suit, 0)

            if mode == "safe":
                score = float(suit.rank)
                if favor == -1:
                    score += 2.0
                if diff < 0:
                    score -= 0.5 * min(3, -diff)
            elif mode == "aggressive":
                score = float(suit.rank) * 1.4
                if diff <= 0:
                    score += 3.0
                if favor == -1:
                    score -= 1.5
            else:
                score = float(suit.rank) * 1.1
                if diff <= 0:
                    score += 1.5

            priority[suit] = score

        return priority

    def _action_score(
        self,
        action: Action,
        ctx: HeuristicContext,
        suit_priority: Dict[object, float],
    ) -> float:
        mode = self._mode(ctx)
        base = self._default_action_score(action, ctx, suit_priority)

        if action.action_type == ActionType.SPLIT:
            same_suit = action.cards[0].suit == action.cards[1].suit
            if mode == "safe" and same_suit:
                base += 1.5
            elif mode == "aggressive" and not same_suit:
                rank_gap = abs(action.cards[0].suit.rank - action.cards[1].suit.rank)
                base += 0.5 * rank_gap

        if action.action_type in (ActionType.DISCARD, ActionType.TRADE_OFF):
            total_rank = action.cards[0].suit.rank + action.cards[1].suit.rank
            if mode == "safe":
                base -= 0.2 * total_rank
            elif mode == "aggressive":
                base += 0.2 * total_rank

        # Full Hanamikoji offers, by analogy with SPLIT: matched cards are
        # safe; spread-out ranks are aggressive.
        if action.action_type in (ActionType.GIFT, ActionType.COMPETITION):
            ranks = [card.suit.rank for card in action.cards]
            if action.action_type == ActionType.COMPETITION:
                matched = sum(action.cards[i].suit == action.cards[i + 1].suit for i in (0, 2))
            else:
                matched = len(ranks) - len({card.suit for card in action.cards})
            if mode == "safe":
                base += 0.75 * matched
            elif mode == "aggressive":
                base += 0.25 * (max(ranks) - min(ranks))

        return base
