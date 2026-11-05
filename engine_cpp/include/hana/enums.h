#pragma once

#include <cstdint>

namespace hana {

// -----------------------------------------------------------------------
// ActionType
// The .value strings below must match Python's ActionType enum values
// exactly — they appear verbatim in policy-table artifact JSON.
// -----------------------------------------------------------------------
enum class ActionType : uint8_t {
    RESERVE        = 0,   // "reserve"
    DISCARD        = 1,   // "discard"
    SPLIT          = 2,   // "split"
    SPLIT_RESPONSE = 3,   // "split_response"
    CHANCE_START   = 4,   // "chance_start"
    CHANCE_DEAL    = 5,   // "chance_deal"
    CHANCE_DRAW    = 6,   // "chance_draw"
    // Full Hanamikoji.  RESERVE is the "secret" action.
    TRADE_OFF            = 7,   // "trade_off"            discard 2 cards face down
    GIFT                 = 8,   // "gift"                 offer 3 cards; opponent takes 1
    GIFT_RESPONSE        = 9,   // "gift_response"        choice = index of the card taken
    COMPETITION          = 10,  // "competition"          offer 2 pairs; opponent takes a pair
    COMPETITION_RESPONSE = 11,  // "competition_response" choice = index of the pair taken
};

// Bitmask over ActionType (12 types, so 16 bits).
using ActionMask = uint16_t;

inline constexpr ActionMask action_bit(ActionType t) noexcept {
    return static_cast<ActionMask>(1u << static_cast<int>(t));
}

// Returns the Python-compatible .value string for an ActionType.
// Used in policy artifact serialization — must be byte-identical to Python.
inline const char* action_type_value_str(ActionType t) noexcept {
    switch (t) {
        case ActionType::RESERVE:        return "reserve";
        case ActionType::DISCARD:        return "discard";
        case ActionType::SPLIT:          return "split";
        case ActionType::SPLIT_RESPONSE: return "split_response";
        case ActionType::CHANCE_START:   return "chance_start";
        case ActionType::CHANCE_DEAL:    return "chance_deal";
        case ActionType::CHANCE_DRAW:    return "chance_draw";
        case ActionType::TRADE_OFF:            return "trade_off";
        case ActionType::GIFT:                 return "gift";
        case ActionType::GIFT_RESPONSE:        return "gift_response";
        case ActionType::COMPETITION:          return "competition";
        case ActionType::COMPETITION_RESPONSE: return "competition_response";
    }
    return "unknown";
}

// Returns the Python-compatible .name string for an ActionType.
inline const char* action_type_name(ActionType t) noexcept {
    switch (t) {
        case ActionType::RESERVE:        return "RESERVE";
        case ActionType::DISCARD:        return "DISCARD";
        case ActionType::SPLIT:          return "SPLIT";
        case ActionType::SPLIT_RESPONSE: return "SPLIT_RESPONSE";
        case ActionType::CHANCE_START:   return "CHANCE_START";
        case ActionType::CHANCE_DEAL:    return "CHANCE_DEAL";
        case ActionType::CHANCE_DRAW:    return "CHANCE_DRAW";
        case ActionType::TRADE_OFF:            return "TRADE_OFF";
        case ActionType::GIFT:                 return "GIFT";
        case ActionType::GIFT_RESPONSE:        return "GIFT_RESPONSE";
        case ActionType::COMPETITION:          return "COMPETITION";
        case ActionType::COMPETITION_RESPONSE: return "COMPETITION_RESPONSE";
    }
    return "UNKNOWN";
}

// Offers the opponent answers within the same turn: the response type and
// its number of choices.  Mirrors Python's engine.enums.OFFER_RESPONSES.
inline bool is_offer(ActionType t) noexcept {
    return t == ActionType::SPLIT || t == ActionType::GIFT || t == ActionType::COMPETITION;
}
inline ActionType offer_response_type(ActionType offer) noexcept {
    switch (offer) {
        case ActionType::GIFT:        return ActionType::GIFT_RESPONSE;
        case ActionType::COMPETITION: return ActionType::COMPETITION_RESPONSE;
        default:                      return ActionType::SPLIT_RESPONSE;
    }
}
inline int offer_response_choices(ActionType offer) noexcept {
    return offer == ActionType::GIFT ? 3 : 2;
}
// Number of cards each card-playing action takes from the hand (0 = none).
inline int action_card_count(ActionType t) noexcept {
    switch (t) {
        case ActionType::RESERVE:     return 1;
        case ActionType::DISCARD:     return 2;
        case ActionType::SPLIT:       return 2;
        case ActionType::TRADE_OFF:   return 2;
        case ActionType::GIFT:        return 3;
        case ActionType::COMPETITION: return 4;
        default:                      return 0;
    }
}

// -----------------------------------------------------------------------
// Phase — internal game phase, replaces Python's phase string field.
// -----------------------------------------------------------------------
enum class Phase : uint8_t {
    CHOOSE_START,
    DEAL_INITIAL,
    DRAW_FOR_TURN,
    PLAY,
    ROUND_COMPLETE,
};

inline const char* phase_name(Phase p) noexcept {
    switch (p) {
        case Phase::CHOOSE_START:   return "choose_start";
        case Phase::DEAL_INITIAL:   return "deal_initial";
        case Phase::DRAW_FOR_TURN:  return "draw_for_turn";
        case Phase::PLAY:           return "play";
        case Phase::ROUND_COMPLETE: return "round_complete";
    }
    return "unknown";
}

// -----------------------------------------------------------------------
// ObservationType — mirrors Python's ObservationType enum.
// -----------------------------------------------------------------------
enum class ObservationType : uint8_t {
    ROUND_START      = 0,
    DRAW_CARD        = 1,
    OWN_RESERVE      = 2,
    OWN_DISCARD      = 3,
    OWN_SPLIT        = 4,
    OPPONENT_RESERVE = 5,
    OPPONENT_DISCARD = 6,
    OPPONENT_SPLIT   = 7,
    SPLIT_CHOICE     = 8,
    ROUND_END        = 9,
    OWN_TRADE_OFF        = 10,
    OPPONENT_TRADE_OFF   = 11,
    OWN_GIFT             = 12,
    OPPONENT_GIFT        = 13,
    GIFT_CHOICE          = 14,
    OWN_COMPETITION      = 15,
    OPPONENT_COMPETITION = 16,
    COMPETITION_CHOICE   = 17,
};

} // namespace hana
