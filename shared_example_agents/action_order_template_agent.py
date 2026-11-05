"""ActionOrderTemplateAgent: use hand-profile action sequencing templates."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence

from engine import Action, ActionType

from ._base import HeuristicAgentBase, HeuristicContext


class ActionOrderTemplateAgent(HeuristicAgentBase):
    """Choose action type order from simple deterministic templates."""

    def _suit_priority(self, ctx: HeuristicContext) -> Dict[object, float]:
        priority: Dict[object, float] = {}
        for suit in ctx.suits:
            diff = ctx.my_counts[suit] - ctx.opp_counts[suit]
            score = float(suit.rank)
            if diff <= 0:
                score += 2.0
            if ctx.favors.get(suit, 0) != -1:
                score += 0.75
            priority[suit] = score
        return priority

    def _select_distribution(
        self,
        ctx: HeuristicContext,
        legal_actions: Sequence[Action],
    ) -> Dict[Action, float]:
        responses = (ActionType.SPLIT_RESPONSE, ActionType.GIFT_RESPONSE, ActionType.COMPETITION_RESPONSE)
        if legal_actions and all(a.action_type in responses for a in legal_actions):
            return super()._select_distribution(ctx, legal_actions)

        suit_priority = self._suit_priority(ctx)

        hand_counts = Counter(card.suit for card in ctx.hand)
        has_pair = any(count >= 2 for count in hand_counts.values())
        max_rank = max(suit.rank for suit in ctx.suits) if ctx.suits else 1
        high_cards = sum(1 for card in ctx.hand if card.suit.rank >= max_rank - 1)

        # Each template interleaves full-Hanamikoji actions (TRADE_OFF like
        # DISCARD; GIFT and COMPETITION like SPLIT); a variant only ever has
        # some of them, so the others are skipped.
        if has_pair:
            template = [ActionType.SPLIT, ActionType.COMPETITION, ActionType.RESERVE,
                        ActionType.GIFT, ActionType.DISCARD, ActionType.TRADE_OFF]
        elif high_cards >= max(1, len(ctx.hand) // 2):
            template = [ActionType.RESERVE, ActionType.SPLIT, ActionType.GIFT,
                        ActionType.COMPETITION, ActionType.DISCARD, ActionType.TRADE_OFF]
        else:
            template = [ActionType.DISCARD, ActionType.TRADE_OFF, ActionType.RESERVE,
                        ActionType.SPLIT, ActionType.GIFT, ActionType.COMPETITION]

        by_type = {}
        for action in legal_actions:
            by_type.setdefault(action.action_type, []).append(action)

        for action_type in template:
            options = by_type.get(action_type, [])
            if options:
                best = self._best_action(ctx, options, suit_priority)
                return {best: 1.0}

        best = self._best_action(ctx, legal_actions, suit_priority)
        return {best: 1.0}
