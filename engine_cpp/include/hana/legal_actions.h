#pragma once

#include "action.h"
#include "enums.h"
#include "suit.h"
#include <initializer_list>

namespace hana {

// -----------------------------------------------------------------------
// Legal-action enumeration shared by GameState and RoundInfoState.
// Mirrors Python's engine/legal_actions.py, including the order:
//
//   RESERVE      one per distinct suit, by suit value
//   DISCARD      by (suit_a, suit_b), then choice 0 before 1
//                (same-suit pair: choice 0 only)
//   SPLIT        by (suit_a, suit_b)
//   TRADE_OFF    by (suit_a, suit_b)
//   GIFT         by (a, b, c), a <= b <= c
//   COMPETITION  by (pair0, pair1), each pair ascending and pair0 <= pair1
//
// Each enumerates distinct suit multisets drawable from counts[] (the hand's
// cards per suit), so the result does not depend on hand order.
// -----------------------------------------------------------------------

template <typename F>
inline void enumerate_player_actions(const int counts[SUIT_CAPACITY], int num_suits, int hand_count,
                                     ActionMask avail, ActionMask used, F&& callback) {
    auto can = [&](ActionType t) {
        return (avail & action_bit(t)) && !(used & action_bit(t));
    };
    auto S = [](int i) { return static_cast<Suit>(i); };

    if (can(ActionType::RESERVE)) {
        for (int s = 0; s < num_suits; s++)
            if (counts[s] > 0) callback(Action::make_reserve(S(s)));
    }

    if (hand_count < 2) return;

    if (can(ActionType::DISCARD)) {
        for (int a = 0; a < num_suits; a++) {
            for (int b = a; b < num_suits; b++) {
                if (a == b) {
                    if (counts[a] >= 2) callback(Action::make_discard(S(a), S(a), 0));
                } else if (counts[a] >= 1 && counts[b] >= 1) {
                    callback(Action::make_discard(S(a), S(b), 0));
                    callback(Action::make_discard(S(a), S(b), 1));
                }
            }
        }
    }

    // Can this multiset of suits be drawn from counts?
    auto drawable = [&](std::initializer_list<int> suits) {
        int need[SUIT_CAPACITY] = {};
        for (int s : suits) need[s]++;
        for (int s : suits)
            if (need[s] > counts[s]) return false;
        return true;
    };
    auto pair_ok = [&](int a, int b) {   // a <= b; the cheap check (tiny/medium hot path)
        return a == b ? counts[a] >= 2 : counts[a] >= 1 && counts[b] >= 1;
    };

    if (can(ActionType::SPLIT)) {
        for (int a = 0; a < num_suits; a++)
            for (int b = a; b < num_suits; b++)
                if (pair_ok(a, b)) callback(Action::make_split(S(a), S(b)));
    }

    if (can(ActionType::TRADE_OFF)) {
        for (int a = 0; a < num_suits; a++)
            for (int b = a; b < num_suits; b++)
                if (pair_ok(a, b)) callback(Action::make_trade_off(S(a), S(b)));
    }

    if (can(ActionType::GIFT) && hand_count >= 3) {
        for (int a = 0; a < num_suits; a++)
            for (int b = a; b < num_suits; b++)
                for (int c = b; c < num_suits; c++)
                    if (drawable({a, b, c})) callback(Action::make_gift(S(a), S(b), S(c)));
    }

    if (can(ActionType::COMPETITION) && hand_count >= 4) {
        int pairs[SUIT_CAPACITY * (SUIT_CAPACITY + 1) / 2][2];
        int n_pairs = 0;
        for (int a = 0; a < num_suits; a++)
            for (int b = a; b < num_suits; b++)
                if (pair_ok(a, b)) { pairs[n_pairs][0] = a; pairs[n_pairs][1] = b; n_pairs++; }
        for (int i = 0; i < n_pairs; i++)
            for (int j = i; j < n_pairs; j++)
                if (drawable({pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]}))
                    callback(Action::make_competition(S(pairs[i][0]), S(pairs[i][1]),
                                                      S(pairs[j][0]), S(pairs[j][1])));
    }
}

// Every response to a pending offer, one per choice index — even when two
// offered cards (or pairs) are identical, as for SPLIT_RESPONSE.
template <typename F>
inline void enumerate_offer_responses(ActionType offer_type, F&& callback) {
    const ActionType response = offer_response_type(offer_type);
    for (int c = 0; c < offer_response_choices(offer_type); c++)
        callback(Action::make_offer_response(response, c));
}

} // namespace hana
