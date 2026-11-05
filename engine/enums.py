"""Core enums used by the Hanamikoji engine."""

from enum import Enum


class ActionType(Enum):
    """All action types across all variants."""

    RESERVE = "reserve"
    DISCARD = "discard"
    SPLIT = "split"
    SPLIT_RESPONSE = "split_response"
    CHANCE_START = "chance_start"
    CHANCE_DEAL = "chance_deal"
    CHANCE_DRAW = "chance_draw"
    # Full Hanamikoji.  RESERVE is the "secret" action.
    TRADE_OFF = "trade_off"                        # discard 2 cards face down
    GIFT = "gift"                                  # offer 3 cards; opponent takes 1
    GIFT_RESPONSE = "gift_response"                # choice = index of the card taken
    COMPETITION = "competition"                    # offer 2 pairs; opponent takes a pair
    COMPETITION_RESPONSE = "competition_response"  # choice = index of the pair taken


class ObservationType(Enum):
    """All observable event types across all variants."""

    ROUND_START = "round_start"
    DRAW_CARD = "draw_card"
    OWN_RESERVE = "own_reserve"
    OWN_DISCARD = "own_discard"
    OWN_SPLIT = "own_split"
    OPPONENT_RESERVE = "opponent_reserve"
    OPPONENT_DISCARD = "opponent_discard"
    OPPONENT_SPLIT = "opponent_split"
    SPLIT_CHOICE = "split_choice"
    ROUND_END = "round_end"
    OWN_TRADE_OFF = "own_trade_off"
    OPPONENT_TRADE_OFF = "opponent_trade_off"
    OWN_GIFT = "own_gift"
    OPPONENT_GIFT = "opponent_gift"
    GIFT_CHOICE = "gift_choice"
    OWN_COMPETITION = "own_competition"
    OPPONENT_COMPETITION = "opponent_competition"
    COMPETITION_CHOICE = "competition_choice"


# Offer actions that the opponent answers within the same turn, and the
# response type and number of response choices for each.
OFFER_RESPONSES = {
    ActionType.SPLIT: (ActionType.SPLIT_RESPONSE, 2),
    ActionType.GIFT: (ActionType.GIFT_RESPONSE, 3),
    ActionType.COMPETITION: (ActionType.COMPETITION_RESPONSE, 2),
}
