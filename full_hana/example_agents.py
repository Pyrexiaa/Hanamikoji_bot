"""
Example AI agents for full Hanamikoji.

Both agents score every legal action with a value per geisha and play the
best one.  They show how to read the full game's information state:
get_pending_offer() for GIFT / COMPETITION responses, and the four actions
RESERVE (secret), TRADE_OFF, GIFT and COMPETITION.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from full_hana import Action, ActionType, Agent, Card, RoundInfoState


def _score_action(action: Action, value: Dict[object, float], offer_cards: Sequence[Card]) -> float:
    """How good `action` is for us, given a value per suit.

    Offers assume the opponent takes what is best for them, which we
    approximate by what is best for us.
    """
    v = [value[card.suit] for card in action.cards]
    at = action.action_type
    if at == ActionType.RESERVE:
        return v[0]                                   # the secret card is ours
    if at == ActionType.TRADE_OFF:
        return -(v[0] + v[1])                         # give up the least useful pair
    if at == ActionType.GIFT:
        return sum(v) - max(v)                        # opponent takes the best card
    if at == ActionType.COMPETITION:
        return min(v[0] + v[1], v[2] + v[3])          # opponent takes the better pair
    if at == ActionType.GIFT_RESPONSE:
        return value[offer_cards[action.choice].suit]
    if at == ActionType.COMPETITION_RESPONSE:
        pair = offer_cards[2 * action.choice:2 * action.choice + 2]
        return sum(value[card.suit] for card in pair)
    return float("-inf")


def _best_action(info_state: RoundInfoState, value: Dict[object, float]) -> Action:
    offer = info_state.get_pending_offer()
    offer_cards = offer[1] if offer is not None else []
    actions = list(info_state.get_legal_actions())
    # max() keeps the first of equal scores, so ties go to the engine's
    # canonical (suit-ordered) action order — deterministic on both backends.
    return max(actions, key=lambda a: _score_action(a, value, offer_cards))


class GreedyAgent(Agent):
    """
    Greedy strategy: a geisha is worth her charm points (rank).

    - Keeps the highest-rank card secret
    - Trades off the two lowest-rank cards
    - Offers gifts / competitions that leave it the most value
    - Takes the most valuable card or pair when responding
    """

    def get_action_distribution(self, info_state: RoundInfoState):
        favors = info_state.get_favors()
        value = {suit: float(suit.rank) for suit in favors}
        return {_best_action(info_state, value): 1.0}


class CardCountingAgent(Agent):
    """
    Card-counting strategy: a geisha is worth her rank only while her favor
    is still in play this round.

    For each geisha, count the cards already committed to each side (cards
    collected this round plus our secret card).  A geisha is won by holding
    more of her cards than the opponent; once either side holds a majority
    of her cards (rank // 2 + 1) she is settled and worth little.  An open
    geisha is worth more the closer the contest, and a little more if a tie
    would leave her favor with us.
    """

    def get_action_distribution(self, info_state: RoundInfoState):
        return {_best_action(info_state, self._values(info_state)): 1.0}

    @staticmethod
    def _values(info_state: RoundInfoState) -> Dict[object, float]:
        me = info_state.player_id
        mine_favor = -1 if me == 0 else 1             # favor encoding: -1 = player 0
        favors = info_state.get_favors()
        my_cards, opp_cards = info_state.get_collected_cards()
        committed: Dict[object, List[int]] = {suit: [0, 0] for suit in favors}
        for card in my_cards + info_state.get_my_reserved():
            committed[card.suit][0] += 1
        for card in opp_cards:
            committed[card.suit][1] += 1

        value: Dict[object, float] = {}
        for suit, owner in favors.items():
            mine, theirs = committed[suit]
            majority = suit.rank // 2 + 1
            if mine >= majority or theirs >= majority:
                value[suit] = 0.25 * suit.rank        # settled either way
                continue
            closeness = 1.0 - abs(mine - theirs) / suit.rank
            bonus = 0.5 if owner == mine_favor else 0.0
            value[suit] = suit.rank * (0.5 + closeness) + bonus
        return value
