"""DenialFirstAgent: prioritize blocking opponent threat suits."""

from __future__ import annotations

from typing import Dict

from ._base import HeuristicAgentBase, HeuristicContext


class DenialFirstAgent(HeuristicAgentBase):
    """Heuristic that values denial before pure self-maximization."""

    def _suit_priority(self, ctx: HeuristicContext) -> Dict[object, float]:
        priority: Dict[object, float] = {}
        for suit in ctx.suits:
            opp_threat = float(ctx.opp_counts[suit]) * 3.0
            favor_threat = 4.0 if ctx.favors.get(suit, 0) == 1 else 0.0
            uncertainty = min(2.0, float(ctx.unseen_counts[suit])) * 0.5
            oversecured_penalty = 2.0 if ctx.my_counts[suit] >= ctx.opp_counts[suit] + 2 else 0.0

            score = float(suit.rank) * 0.7 + opp_threat + favor_threat + uncertainty - oversecured_penalty
            priority[suit] = score

        return priority
