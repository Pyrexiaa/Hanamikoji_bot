"""Shared deterministic utilities for handcrafted heuristic agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from engine import Action, ActionType, Agent, Card, RoundInfoState


@dataclass
class HeuristicContext:
    player_id: int
    suits: Tuple[object, ...]
    hand: Tuple[Card, ...]
    used_actions: frozenset
    my_counts: Dict[object, int]
    opp_counts: Dict[object, int]
    favors: Dict[object, int]
    unseen_counts: Dict[object, int]
    pending_split: Optional[Tuple[Card, Card]]
    opp_reserved_count: int
    # The opponent's offer awaiting a response (SPLIT, GIFT or COMPETITION):
    # (offer type, offered cards); a COMPETITION's pairs are cards[0:2], cards[2:4].
    pending_offer: Optional[Tuple[object, Tuple[Card, ...]]] = None


class HeuristicAgentBase(Agent, ABC):
    """Base class for deterministic handcrafted heuristic agents.

    - Uses only RoundInfoState-visible information.
    - Returns deterministic distributions for each infoset.
    - Optionally memoizes distributions by infoset string.
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache: Dict[str, Dict[Action, float]] = {}

    def get_action_distribution(self, info_state: RoundInfoState) -> Dict[Action, float]:
        key = str(info_state)
        if self.use_cache and key in self._cache:
            return dict(self._cache[key])

        legal_actions = self._stable_legal_actions(info_state)
        if not legal_actions:
            raise ValueError(f"{self.__class__.__name__}: no legal actions.")

        ctx = self._build_context(info_state)
        distribution = self._select_distribution(ctx, legal_actions)
        self._validate_distribution(distribution, legal_actions)

        if self.use_cache:
            self._cache[key] = distribution
        return distribution

    def _select_distribution(
        self,
        ctx: HeuristicContext,
        legal_actions: Sequence[Action],
    ) -> Dict[Action, float]:
        suit_priority = self._suit_priority(ctx)
        best = self._best_action(ctx, legal_actions, suit_priority)
        return {best: 1.0}

    @abstractmethod
    def _suit_priority(self, ctx: HeuristicContext) -> Dict[object, float]:
        """Return a desirability score per suit (higher is better to acquire)."""

    def _action_score(
        self,
        action: Action,
        ctx: HeuristicContext,
        suit_priority: Dict[object, float],
    ) -> float:
        return self._default_action_score(action, ctx, suit_priority)

    def _best_action(
        self,
        ctx: HeuristicContext,
        legal_actions: Sequence[Action],
        suit_priority: Dict[object, float],
    ) -> Action:
        best = legal_actions[0]
        best_score = self._action_score(best, ctx, suit_priority)

        for action in legal_actions[1:]:
            score = self._action_score(action, ctx, suit_priority)
            if score > best_score + 1e-12:
                best = action
                best_score = score
        return best

    def _top_two_mixture(
        self,
        ctx: HeuristicContext,
        legal_actions: Sequence[Action],
        suit_priority: Dict[object, float],
        p_top: float = 0.75,
        min_gap_for_pure: float = 1.0,
    ) -> Dict[Action, float]:
        scored = [
            (action, self._action_score(action, ctx, suit_priority))
            for action in legal_actions
        ]
        scored.sort(key=lambda item: (-item[1], self._action_sort_key(item[0])))

        if len(scored) == 1:
            return {scored[0][0]: 1.0}

        top_action, top_score = scored[0]
        second_action, second_score = scored[1]
        if top_score - second_score >= min_gap_for_pure:
            return {top_action: 1.0}

        return {top_action: p_top, second_action: 1.0 - p_top}

    @staticmethod
    def _stable_legal_actions(info_state: RoundInfoState) -> List[Action]:
        actions = list(info_state.get_legal_actions())
        actions.sort(key=HeuristicAgentBase._action_sort_key)
        return actions

    @staticmethod
    def _action_sort_key(action: Action) -> Tuple[int, Tuple[int, ...], int]:
        order = {
            ActionType.RESERVE: 0,
            ActionType.DISCARD: 1,
            ActionType.SPLIT: 2,
            ActionType.SPLIT_RESPONSE: 3,
            ActionType.CHANCE_START: 4,
            ActionType.CHANCE_DEAL: 5,
            ActionType.CHANCE_DRAW: 6,
            ActionType.TRADE_OFF: 7,
            ActionType.GIFT: 8,
            ActionType.GIFT_RESPONSE: 9,
            ActionType.COMPETITION: 10,
            ActionType.COMPETITION_RESPONSE: 11,
        }
        cards_sig = tuple(card.suit.value for card in action.cards)
        choice = -1 if action.choice is None else int(action.choice)
        return (order.get(action.action_type, 99), cards_sig, choice)


    @staticmethod
    def _default_action_score(
        action: Action,
        ctx: HeuristicContext,
        suit_priority: Dict[object, float],
    ) -> float:
        if action.action_type == ActionType.SPLIT_RESPONSE:
            if ctx.pending_split is None or action.choice not in (0, 1):
                return -1e9
            chosen_suit = ctx.pending_split[action.choice].suit
            return suit_priority.get(chosen_suit, 0.0)

        if action.action_type == ActionType.RESERVE:
            return suit_priority.get(action.cards[0].suit, 0.0) + 0.1

        if action.action_type == ActionType.DISCARD:
            s0 = action.cards[0].suit
            s1 = action.cards[1].suit
            score = -(suit_priority.get(s0, 0.0) + suit_priority.get(s1, 0.0))
            if action.choice in (0, 1):
                revealed = action.cards[action.choice].suit
                hidden = action.cards[1 - action.choice].suit
                # Slight preference: reveal the lower-priority card, hide higher-priority.
                score += 0.1 * (suit_priority.get(hidden, 0.0) - suit_priority.get(revealed, 0.0))
            return score

        if action.action_type == ActionType.SPLIT:
            v0 = suit_priority.get(action.cards[0].suit, 0.0)
            v1 = suit_priority.get(action.cards[1].suit, 0.0)
            # Pessimistic split value with a small upside term.
            return min(v0, v1) + 0.15 * max(v0, v1)

        # ── Full Hanamikoji ──
        values = [suit_priority.get(card.suit, 0.0) for card in action.cards]

        if action.action_type == ActionType.TRADE_OFF:
            # Like DISCARD: give up the two least useful cards.
            return -sum(values)

        if action.action_type == ActionType.GIFT:
            # Pessimistic, as for SPLIT: the opponent takes the best card and
            # we keep the other two; small upside term.
            return sum(values) - max(values) + 0.15 * max(values)

        if action.action_type == ActionType.COMPETITION:
            # The opponent takes the better pair; we keep the other.
            pair0, pair1 = values[0] + values[1], values[2] + values[3]
            return min(pair0, pair1) + 0.15 * max(pair0, pair1)

        if action.action_type in (ActionType.GIFT_RESPONSE, ActionType.COMPETITION_RESPONSE):
            taken = HeuristicAgentBase._offer_taken(ctx, action)
            if taken is None:
                return -1e9
            return sum(suit_priority.get(card.suit, 0.0) for card in taken)

        return -1e9

    @staticmethod
    def _offer_taken(ctx: HeuristicContext, action: Action) -> Optional[Tuple[Card, ...]]:
        """Cards a GIFT_ / COMPETITION_RESPONSE takes from the pending offer."""
        if ctx.pending_offer is None or action.choice is None:
            return None
        _, cards = ctx.pending_offer
        if action.action_type == ActionType.COMPETITION_RESPONSE:
            return tuple(cards[2 * action.choice:2 * action.choice + 2])
        return (cards[action.choice],)

    @staticmethod
    def _build_context(info_state: RoundInfoState) -> HeuristicContext:
        hand, used_actions = info_state.get_current_hand_and_actions()
        my_collected, opp_collected = info_state.get_collected_cards()
        my_reserved = info_state.get_my_reserved()
        my_revealed, my_hidden, opp_revealed = info_state.get_discarded_cards()
        raw_favors = dict(info_state.get_favors())
        # Normalize favor ownership so `-1` always means "me", `+1` means "opponent".
        if info_state.player_id == 0:
            favors = raw_favors
        else:
            favors = {suit: (0 if owner == 0 else -owner) for suit, owner in raw_favors.items()}
        pending_split_list = info_state.get_pending_split()
        pending_split = tuple(pending_split_list) if pending_split_list is not None else None
        raw_offer = info_state.get_pending_offer()
        pending_offer = None if raw_offer is None else (raw_offer[0], tuple(raw_offer[1]))
        opp_reserved_count = info_state.get_opponent_reserved_count()

        suit_set = set(favors.keys())
        for seq in (
            hand,
            my_collected,
            opp_collected,
            my_reserved,
            my_revealed,
            my_hidden,
            opp_revealed,
        ):
            for card in seq:
                suit_set.add(card.suit)
        if pending_offer is not None:
            for card in pending_offer[1]:
                suit_set.add(card.suit)

        suits = tuple(sorted(suit_set, key=lambda suit: suit.value))

        my_counts = {suit: 0 for suit in suits}
        for card in my_collected:
            my_counts[card.suit] += 1
        for card in my_reserved:
            my_counts[card.suit] += 1

        opp_counts = {suit: 0 for suit in suits}
        for card in opp_collected:
            opp_counts[card.suit] += 1
        for card in opp_revealed:
            opp_counts[card.suit] += 1

        visible_counts = {suit: 0 for suit in suits}
        visible_cards = (
            list(hand)
            + my_collected
            + opp_collected
            + my_reserved
            + my_revealed
            + my_hidden
            + opp_revealed
            + ([] if pending_offer is None else list(pending_offer[1]))
        )
        for card in visible_cards:
            visible_counts[card.suit] += 1

        unseen_counts = {
            suit: max(0, int(getattr(suit, "rank", 0)) - visible_counts[suit])
            for suit in suits
        }

        return HeuristicContext(
            player_id=info_state.player_id,
            suits=suits,
            hand=tuple(hand),
            used_actions=frozenset(used_actions),
            my_counts=my_counts,
            opp_counts=opp_counts,
            favors=favors,
            unseen_counts=unseen_counts,
            pending_split=pending_split,
            opp_reserved_count=opp_reserved_count,
            pending_offer=pending_offer,
        )

    @classmethod
    def _validate_distribution(
        cls,
        distribution: Mapping[Action, float],
        legal_actions: Sequence[Action],
    ) -> None:
        if not distribution:
            raise ValueError("Action distribution cannot be empty.")

        legal_set = set(legal_actions)
        total_prob = 0.0
        for action, prob in distribution.items():
            if action not in legal_set:
                raise ValueError(f"Distribution includes non-legal action: {action}")
            if prob < 0:
                raise ValueError(f"Negative probability for action {action}: {prob}")
            total_prob += float(prob)

        if abs(total_prob - 1.0) > 1e-9:
            raise ValueError(f"Probabilities must sum to 1.0, got {total_prob:.12f}")
