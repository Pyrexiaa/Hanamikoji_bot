"""BeliefBucketAgent: lightweight hidden-card uncertainty buckets."""

from __future__ import annotations

from typing import Dict, Sequence

from engine import Action

from ._base import HeuristicAgentBase, HeuristicContext


class BeliefBucketAgent(HeuristicAgentBase):
    """Use coarse unseen-count buckets and return deterministic mixed policies."""

    def _suit_priority(self, ctx: HeuristicContext) -> Dict[object, float]:
        priority: Dict[object, float] = {}
        for suit in ctx.suits:
            unseen = ctx.unseen_counts[suit]
            deficit = max(0, ctx.opp_counts[suit] - ctx.my_counts[suit])

            if unseen <= 1:
                bucket_bonus = 3.0  # scarce -> immediate swings matter more
            elif unseen <= 2:
                bucket_bonus = 1.5
            else:
                bucket_bonus = 0.5

            score = float(suit.rank) + bucket_bonus + 2.0 * deficit
            if ctx.favors.get(suit, 0) == 1:
                score += 1.0
            priority[suit] = score

        return priority

    def _select_distribution(
        self,
        ctx: HeuristicContext,
        legal_actions: Sequence[Action],
    ) -> Dict[Action, float]:
        suit_priority = self._suit_priority(ctx)
        return self._top_two_mixture(
            ctx,
            legal_actions,
            suit_priority,
            p_top=0.7,
            min_gap_for_pure=1.25,
        )
