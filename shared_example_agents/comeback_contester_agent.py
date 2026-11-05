"""ComebackContesterAgent: focus on neutral/losing contested suits."""

from __future__ import annotations

from typing import Dict

from ._base import HeuristicAgentBase, HeuristicContext


class ComebackContesterAgent(HeuristicAgentBase):
    """Invest mostly where a swing is still realistic."""

    def _suit_priority(self, ctx: HeuristicContext) -> Dict[object, float]:
        priority: Dict[object, float] = {}
        for suit in ctx.suits:
            diff = ctx.my_counts[suit] - ctx.opp_counts[suit]
            favor = ctx.favors.get(suit, 0)

            score = float(suit.rank) * 0.8
            if diff <= -1:
                score += 6.0 - 0.5 * min(4, -diff - 1)
            elif diff == 0:
                score += 4.5
            elif diff == 1:
                score += 2.0
            else:
                score -= 1.2 * min(3, diff - 1)

            if favor == 1:
                score += 1.5
            elif favor == -1 and diff >= 2:
                score -= 1.5

            priority[suit] = score

        return priority
