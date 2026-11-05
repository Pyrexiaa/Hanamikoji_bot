"""MajorityThresholdAgent: prioritize near-threshold suit swings."""

from __future__ import annotations

from typing import Dict

from ._base import HeuristicAgentBase, HeuristicContext


class MajorityThresholdAgent(HeuristicAgentBase):
    """Rewards suits close to changing majority."""

    def _suit_priority(self, ctx: HeuristicContext) -> Dict[object, float]:
        priority: Dict[object, float] = {}
        for suit in ctx.suits:
            diff = ctx.my_counts[suit] - ctx.opp_counts[suit]
            distance_to_swing = abs(diff) + 1

            score = float(suit.rank) * 1.2
            score += 7.0 / distance_to_swing

            if diff in (-1, 0, 1):
                score += 2.0

            favor = ctx.favors.get(suit, 0)
            if favor == -1 and diff >= 0:
                score += 1.0
            elif favor == 1 and diff <= 0:
                score += 1.0

            priority[suit] = score

        return priority
