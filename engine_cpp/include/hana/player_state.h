#pragma once

#include "constants.h"
#include "enums.h"
#include "suit.h"
#include <cassert>
#include <cstdint>
#include <cstring>

namespace hana {

// -----------------------------------------------------------------------
// PlayerState — per-player mutable state for one round.
//
// Uses fixed-size arrays throughout, sized for every variant up to full
// Hanamikoji, without heap allocation. All counts start at 0.
//
// Array size rationale (full_hana upper bounds):
//   hand:          6 dealt + 1 drawn = 7 (MAX_HAND_SIZE)
//   reserved:      1 per round (RESERVE used once)
//   collected:     7 from offers + the reserved card moved in at round end
//   discarded_rev: 1 per round (medium DISCARD)
//   discarded_hid: 2 per round (full TRADE_OFF; medium DISCARD hides 1)
// -----------------------------------------------------------------------
struct PlayerState {
    Suit    hand[MAX_HAND_SIZE];       int8_t hand_count            = 0;
    Suit    reserved[2];               int8_t reserved_count        = 0;
    Suit    collected[MAX_COLLECTED];  int8_t collected_count       = 0;
    Suit    discarded_rev[2];          int8_t discarded_rev_count   = 0;
    Suit    discarded_hid[2];          int8_t discarded_hid_count   = 0;
    ActionMask used_actions            = 0;   // bitmask over ActionType

    // ------------------------------------------------------------------
    // Round reset
    // ------------------------------------------------------------------
    void reset_for_round() noexcept {
        hand_count          = 0;
        reserved_count      = 0;
        collected_count     = 0;
        discarded_rev_count = 0;
        discarded_hid_count = 0;
        used_actions        = 0;
    }

    // ------------------------------------------------------------------
    // Hand management
    // ------------------------------------------------------------------

    // Find first card matching `s` in hand, remove it, write its former
    // index to *out_index. Returns false if not found (no mutation).
    // Mirrors Python's _pop_matching_card.
    bool pop_card_from_hand(Suit s, int* out_index) noexcept {
        for (int i = 0; i < hand_count; i++) {
            if (hand[i] == s) {
                *out_index = i;
                // Shift remaining cards left to fill the gap.
                for (int j = i; j < hand_count - 1; j++)
                    hand[j] = hand[j + 1];
                hand_count--;
                return true;
            }
        }
        return false;
    }

    // Insert card `s` at position `index`, shifting later cards right.
    // Used during undo to restore the hand to its exact pre-action state.
    // Mirrors Python's list.insert(hand_index, card).
    void insert_card_to_hand(int index, Suit s) noexcept {
        assert(hand_count < MAX_HAND_SIZE);
        assert(index >= 0 && index <= hand_count);
        for (int i = hand_count; i > index; i--)
            hand[i] = hand[i - 1];
        hand[index] = s;
        hand_count++;
    }

    bool has_card_in_hand(Suit s) const noexcept {
        for (int i = 0; i < hand_count; i++)
            if (hand[i] == s) return true;
        return false;
    }

    // ------------------------------------------------------------------
    // Used actions bitmask
    // ------------------------------------------------------------------
    bool has_used(ActionType t) const noexcept {
        return (used_actions & action_bit(t)) != 0;
    }
    void mark_used(ActionType t) noexcept {
        used_actions |= action_bit(t);
    }
    void unmark_used(ActionType t) noexcept {
        used_actions &= static_cast<ActionMask>(~action_bit(t));
    }

    // ------------------------------------------------------------------
    // Card counts (used by preview_round_resolution)
    // ------------------------------------------------------------------

    // Count cards of suit `s` across collected + optionally reserved.
    // Mirrors Python's PlayerState.get_suit_count().
    int suit_count(Suit s, bool include_reserved) const noexcept {
        int n = 0;
        for (int i = 0; i < collected_count; i++)
            if (collected[i] == s) n++;
        if (include_reserved)
            for (int i = 0; i < reserved_count; i++)
                if (reserved[i] == s) n++;
        return n;
    }
};

} // namespace hana
