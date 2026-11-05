#pragma once

#include "constants.h"
#include "enums.h"
#include "suit.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace hana {

// -----------------------------------------------------------------------
// Event structs
//
// Each struct mirrors one Python ObservationEvent subclass.
// to_string() on each must produce byte-identical output to Python's
// __str__. Reference strings were extracted directly from the Python
// engine and are reproduced in test_events.cpp.
//
// Suit values are stored as raw Suit enums — never as strings — so the
// engine inner loop (execute/undo) never allocates. to_string() is
// called lazily only for artifact I/O and PolicyTableAgent lookups.
// -----------------------------------------------------------------------

// Python format helpers (mirrors events.py):
//   _fmt_value(suit)  -> str(suit.value)  e.g. "0", "1", "2"
//   _fmt_favors(favors) -> "0:-1,1:0,2:1"  sorted by suit value
//   _fmt_values(suits)  -> "0,2"           comma-separated suit values

inline std::string fmt_suit_value(Suit s) {
    return std::to_string(suit_value(s));
}

// Format a fixed-size suit array as comma-separated values.
inline std::string fmt_suits(const Suit* suits, int count) {
    std::string out;
    for (int i = 0; i < count; i++) {
        if (i > 0) out += ',';
        out += fmt_suit_value(suits[i]);
    }
    return out;
}

// Format favors array as "suit_value:owner,..." sorted by suit index.
// n_suits limits how many entries are printed (only active suits).
inline std::string fmt_favors(const int8_t favors[], int n_suits) {
    std::string out;
    for (int i = 0; i < n_suits; i++) {
        if (i > 0) out += ',';
        out += std::to_string(i);
        out += ':';
        out += std::to_string(static_cast<int>(favors[i]));
    }
    return out;
}

// Python: ",".join(sorted(action.value for action in available_actions))
inline std::string fmt_actions(ActionMask mask) {
    std::vector<std::string> values;
    for (int t = 0; t < 16; t++)
        if (mask & (1u << t))
            values.emplace_back(action_type_value_str(static_cast<ActionType>(t)));
    std::sort(values.begin(), values.end());
    std::string out;
    for (std::size_t i = 0; i < values.size(); i++) {
        if (i > 0) out += ',';
        out += values[i];
    }
    return out;
}

// "[[a,b],[c,d]]" for a competition's two pairs (pair i = suits[2i..2i+1]).
inline std::string fmt_pairs(const Suit* suits) {
    return "[[" + fmt_suits(suits, 2) + "],[" + fmt_suits(suits + 2, 2) + "]]";
}

// -----------------------------------------------------------------------
// Compact key encoding helpers.
//
// append_compact() writes the same fields as to_string(), in the same order,
// one byte each: turn, ints, suit values; variable-length suit lists and
// favors are preceded by their length; favors are stored as owner + 1; the
// action set is the 16-bit ActionMask, low byte first. The specification,
// the decoder and the Python encoder are in engine/compact_key.py.
// -----------------------------------------------------------------------

inline void put_byte(std::string& out, int v) {
    assert(v >= 0 && v < 256);
    out.push_back(static_cast<char>(static_cast<uint8_t>(v)));
}

inline void put_suits(std::string& out, const Suit* suits, int count) {
    for (int i = 0; i < count; i++) put_byte(out, suit_value(suits[i]));
}

inline void put_suit_list(std::string& out, const Suit* suits, int count) {
    put_byte(out, count);
    put_suits(out, suits, count);
}

inline void put_favors(std::string& out, const int8_t favors[], int n_suits) {
    put_byte(out, n_suits);
    for (int i = 0; i < n_suits; i++) put_byte(out, favors[i] + 1);
}

inline void put_actions(std::string& out, ActionMask mask) {
    put_byte(out, mask & 0xFF);
    put_byte(out, (mask >> 8) & 0xFF);
}

// -----------------------------------------------------------------------
// Individual event structs
// -----------------------------------------------------------------------

struct RoundStartEvent {
    int     turn;
    Suit    initial_hand[MAX_HAND_SIZE];
    int8_t  initial_hand_count;
    int8_t  favors[SUIT_CAPACITY];       // -1, 0, 1 per suit
    int8_t  n_suits;                     // number of active suits
    int8_t  starting_player;
    ActionMask actions_bitmask;          // the variant's available actions

    std::string to_string() const {
        std::string s = "ROUND_START@t";
        s += std::to_string(turn);
        s += "(hand=[";
        s += fmt_suits(initial_hand, initial_hand_count);
        s += "],favors=[";
        s += fmt_favors(favors, n_suits);
        s += "],start=";
        s += std::to_string(static_cast<int>(starting_player));
        s += ",actions=[";
        s += fmt_actions(actions_bitmask);
        s += "])";
        return s;
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suit_list(out, initial_hand, initial_hand_count);
        put_favors(out, favors, n_suits);
        put_byte(out, starting_player);
        put_actions(out, actions_bitmask);
    }
};

struct DrawCardEvent {
    int  turn;
    Suit card;

    std::string to_string() const {
        return "DRAW_CARD@t" + std::to_string(turn)
             + "(card=" + fmt_suit_value(card) + ")";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suits(out, &card, 1);
    }
};

struct OwnReserveEvent {
    int  turn;
    Suit card;

    std::string to_string() const {
        return "OWN_RESERVE@t" + std::to_string(turn)
             + "(card=" + fmt_suit_value(card) + ")";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suits(out, &card, 1);
    }
};

struct OpponentReserveEvent {
    int turn;

    std::string to_string() const {
        return "OPPONENT_RESERVE@t" + std::to_string(turn);
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
    }
};

struct OwnSplitEvent {
    int  turn;
    Suit cards[2];

    std::string to_string() const {
        return "OWN_SPLIT@t" + std::to_string(turn)
             + "(cards=[" + fmt_suits(cards, 2) + "])";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suits(out, cards, 2);
    }
};

struct OpponentSplitEvent {
    int  turn;
    Suit cards[2];

    std::string to_string() const {
        return "OPPONENT_SPLIT@t" + std::to_string(turn)
             + "(cards=[" + fmt_suits(cards, 2) + "])";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suits(out, cards, 2);
    }
};

struct SplitChoiceEvent {
    int    turn;
    int8_t chooser;
    int8_t choice_index;
    Suit   chooser_gets;
    Suit   offerer_gets;

    std::string to_string() const {
        return "SPLIT_CHOICE@t" + std::to_string(turn)
             + "(chooser="      + std::to_string(static_cast<int>(chooser))
             + ",choice="       + std::to_string(static_cast<int>(choice_index))
             + ",chooser_gets=" + fmt_suit_value(chooser_gets)
             + ",offerer_gets=" + fmt_suit_value(offerer_gets)
             + ")";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_byte(out, chooser);
        put_byte(out, choice_index);
        put_suits(out, &chooser_gets, 1);
        put_suits(out, &offerer_gets, 1);
    }
};

struct OwnDiscardEvent {
    int  turn;
    Suit revealed_card;
    Suit hidden_card;

    std::string to_string() const {
        return "OWN_DISCARD@t"  + std::to_string(turn)
             + "(revealed="     + fmt_suit_value(revealed_card)
             + ",hidden="       + fmt_suit_value(hidden_card)
             + ")";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suits(out, &revealed_card, 1);
        put_suits(out, &hidden_card, 1);
    }
};

struct OpponentDiscardEvent {
    int  turn;
    Suit revealed_card;

    std::string to_string() const {
        return "OPPONENT_DISCARD@t" + std::to_string(turn)
             + "(revealed="         + fmt_suit_value(revealed_card)
             + ")";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suits(out, &revealed_card, 1);
    }
};

struct RoundEndEvent {
    int    turn;
    Suit   p0_reserved[2]; int8_t p0_reserved_count;
    Suit   p1_reserved[2]; int8_t p1_reserved_count;
    int8_t final_favors[SUIT_CAPACITY];
    int8_t n_suits;

    std::string to_string() const {
        return "ROUND_END@t"   + std::to_string(turn)
             + "(p0_reserved=[" + fmt_suits(p0_reserved, p0_reserved_count)
             + "],p1_reserved=[" + fmt_suits(p1_reserved, p1_reserved_count)
             + "],favors=["     + fmt_favors(final_favors, n_suits)
             + "])";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suit_list(out, p0_reserved, p0_reserved_count);
        put_suit_list(out, p1_reserved, p1_reserved_count);
        put_favors(out, final_favors, n_suits);
    }
};

// ---- Full Hanamikoji ----------------------------------------------------

struct OwnTradeOffEvent {
    int  turn;
    Suit cards[2];

    std::string to_string() const {
        return "OWN_TRADE_OFF@t" + std::to_string(turn)
             + "(cards=[" + fmt_suits(cards, 2) + "])";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suits(out, cards, 2);
    }
};

struct OpponentTradeOffEvent {
    int turn;

    std::string to_string() const {
        return "OPPONENT_TRADE_OFF@t" + std::to_string(turn);
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
    }
};

struct OwnGiftEvent {
    int  turn;
    Suit cards[3];

    std::string to_string() const {
        return "OWN_GIFT@t" + std::to_string(turn)
             + "(cards=[" + fmt_suits(cards, 3) + "])";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suits(out, cards, 3);
    }
};

struct OpponentGiftEvent {
    int  turn;
    Suit cards[3];

    std::string to_string() const {
        return "OPPONENT_GIFT@t" + std::to_string(turn)
             + "(cards=[" + fmt_suits(cards, 3) + "])";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suits(out, cards, 3);
    }
};

struct GiftChoiceEvent {
    int    turn;
    int8_t chooser;
    int8_t choice_index;
    Suit   chooser_gets;
    Suit   offerer_gets[2];

    std::string to_string() const {
        return "GIFT_CHOICE@t" + std::to_string(turn)
             + "(chooser="       + std::to_string(static_cast<int>(chooser))
             + ",choice="        + std::to_string(static_cast<int>(choice_index))
             + ",chooser_gets="  + fmt_suit_value(chooser_gets)
             + ",offerer_gets=[" + fmt_suits(offerer_gets, 2)
             + "])";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_byte(out, chooser);
        put_byte(out, choice_index);
        put_suits(out, &chooser_gets, 1);
        put_suits(out, offerer_gets, 2);
    }
};

struct OwnCompetitionEvent {
    int  turn;
    Suit cards[4];   // pair 0 = cards[0..1], pair 1 = cards[2..3]

    std::string to_string() const {
        return "OWN_COMPETITION@t" + std::to_string(turn) + "(pairs=" + fmt_pairs(cards) + ")";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suits(out, cards, 4);
    }
};

struct OpponentCompetitionEvent {
    int  turn;
    Suit cards[4];   // pair 0 = cards[0..1], pair 1 = cards[2..3]

    std::string to_string() const {
        return "OPPONENT_COMPETITION@t" + std::to_string(turn) + "(pairs=" + fmt_pairs(cards) + ")";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_suits(out, cards, 4);
    }
};

struct CompetitionChoiceEvent {
    int    turn;
    int8_t chooser;
    int8_t choice_index;
    Suit   chooser_gets[2];
    Suit   offerer_gets[2];

    std::string to_string() const {
        return "COMPETITION_CHOICE@t" + std::to_string(turn)
             + "(chooser="        + std::to_string(static_cast<int>(chooser))
             + ",choice="         + std::to_string(static_cast<int>(choice_index))
             + ",chooser_gets=["  + fmt_suits(chooser_gets, 2)
             + "],offerer_gets=[" + fmt_suits(offerer_gets, 2)
             + "])";
    }

    void append_compact(std::string& out) const {
        put_byte(out, turn);
        put_byte(out, chooser);
        put_byte(out, choice_index);
        put_suits(out, chooser_gets, 2);
        put_suits(out, offerer_gets, 2);
    }
};

// -----------------------------------------------------------------------
// Event variant — the single type used everywhere in the engine.
// -----------------------------------------------------------------------
using Event = std::variant<
    RoundStartEvent,
    DrawCardEvent,
    OwnReserveEvent,
    OpponentReserveEvent,
    OwnSplitEvent,
    OpponentSplitEvent,
    SplitChoiceEvent,
    OwnDiscardEvent,
    OpponentDiscardEvent,
    RoundEndEvent,
    OwnTradeOffEvent,
    OpponentTradeOffEvent,
    OwnGiftEvent,
    OpponentGiftEvent,
    GiftChoiceEvent,
    OwnCompetitionEvent,
    OpponentCompetitionEvent,
    CompetitionChoiceEvent
>;

// Dispatch to each event's to_string().
inline std::string event_to_string(const Event& e) {
    return std::visit([](const auto& ev) { return ev.to_string(); }, e);
}

// Append an event's compact encoding: its code (the variant index, which is
// the order of the Event alternatives above), then its fields.
inline void event_append_compact(const Event& e, std::string& out) {
    put_byte(out, static_cast<int>(e.index()));
    std::visit([&](const auto& ev) { ev.append_compact(out); }, e);
}

// -----------------------------------------------------------------------
// Helper — build a RoundStartEvent.
//
// n_suits must match the config's num_suits so fmt_favors prints only
// the active suits — keeping to_string() byte-identical to Python.
// -----------------------------------------------------------------------
inline RoundStartEvent make_round_start_event(
    int turn,
    const Suit* initial_hand, int hand_count,
    const int8_t favors[],
    int n_suits,
    int starting_player,
    ActionMask actions_bitmask)
{
    RoundStartEvent ev;
    ev.turn = turn;
    ev.initial_hand_count = static_cast<int8_t>(hand_count);
    for (int i = 0; i < hand_count; i++) ev.initial_hand[i] = initial_hand[i];
    ev.n_suits = static_cast<int8_t>(n_suits);
    for (int i = 0; i < n_suits; i++) ev.favors[i] = favors[i];
    ev.starting_player   = static_cast<int8_t>(starting_player);
    ev.actions_bitmask   = actions_bitmask;
    return ev;
}

} // namespace hana
