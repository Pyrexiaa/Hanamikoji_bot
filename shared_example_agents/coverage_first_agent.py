"""CoverageFirstAgent: improve coverage of underrepresented suits."""

from __future__ import annotations

from typing import Dict

from ._base import HeuristicAgentBase, HeuristicContext


class CoverageFirstAgent(HeuristicAgentBase):
    """Prefer suits where our footprint is currently smallest."""

    def _suit_priority(self, ctx: HeuristicContext) -> Dict[object, float]:
        priority: Dict[object, float] = {}
        max_my = max(ctx.my_counts.values()) if ctx.my_counts else 0

        for suit in ctx.suits:
            coverage_need = (max_my - ctx.my_counts[suit]) + 1
            contested_bonus = 1.5 if ctx.my_counts[suit] <= ctx.opp_counts[suit] else 0.0
            favor_bonus = 1.0 if ctx.favors.get(suit, 0) != -1 else 0.0

            score = coverage_need * 3.0 + float(suit.rank) * 0.5 + contested_bonus + favor_bonus
            priority[suit] = score

        return priority
