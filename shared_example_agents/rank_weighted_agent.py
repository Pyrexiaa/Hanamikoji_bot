"""RankWeightedAgent: emphasize high-rank geishas."""

from __future__ import annotations

from typing import Dict

from ._base import HeuristicAgentBase, HeuristicContext


class RankWeightedAgent(HeuristicAgentBase):
    """Simple rank-dominant suit valuation."""

    def _suit_priority(self, ctx: HeuristicContext) -> Dict[object, float]:
        priority: Dict[object, float] = {}
        for suit in ctx.suits:
            score = float(suit.rank) * 10.0
            if ctx.favors.get(suit, 0) != 1:
                score += 1.0
            if ctx.my_counts[suit] <= ctx.opp_counts[suit]:
                score += 0.75
            priority[suit] = score
        return priority
