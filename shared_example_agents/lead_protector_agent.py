"""LeadProtectorAgent: lock in suits where we are already ahead."""

from __future__ import annotations

from typing import Dict

from ._base import HeuristicAgentBase, HeuristicContext


class LeadProtectorAgent(HeuristicAgentBase):
    """Prioritize actions that reinforce current leads and owned favors."""

    def _suit_priority(self, ctx: HeuristicContext) -> Dict[object, float]:
        priority: Dict[object, float] = {}
        for suit in ctx.suits:
            lead = ctx.my_counts[suit] - ctx.opp_counts[suit]
            favor = ctx.favors.get(suit, 0)

            score = float(suit.rank)
            if lead >= 1:
                score += 4.0
            elif lead == 0:
                score += 2.0
            else:
                score -= 1.5 * min(3, -lead)

            if favor == -1:
                score += 3.0
            elif favor == 1:
                score -= 2.0

            priority[suit] = score

        return priority
