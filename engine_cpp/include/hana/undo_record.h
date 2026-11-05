#pragma once

#include "constants.h"
#include "enums.h"
#include "game_config.h"   // MAX_ACTION_SUITS
#include "suit.h"
#include <cstdint>
#include <cstring>

namespace hana {

// -----------------------------------------------------------------------
// RemovedCard — one entry in UndoRecord::removed_cards.
//
// Records which card was popped from which player's hand and at what
// index, so insert_card_to_hand() can restore the exact ordering.
// -----------------------------------------------------------------------
struct RemovedCard {
    int8_t player_id;
    int8_t hand_index;
    Suit   suit;
};

// -----------------------------------------------------------------------
// UndoRecord — compact pre-action snapshot for one execute_action() call.
//
// Mirrors Python's _UndoRecord dataclass. All optional integer fields use
// NO_PLAYER (-2) as sentinel. All arrays are fixed-size; use the
// corresponding *_count field to know the valid prefix.
//
// Array size rationale:
//   removed_cards:  max 4 (COMPETITION removes 4 cards)
//   dealt_cards:    max MAX_ACTION_SUITS (2 * initial hand size; full = 12)
//   pre_hand_p*:    max MAX_HAND_SIZE (hand is empty before first deal in practice)
// -----------------------------------------------------------------------
struct UndoRecord {
    // ---- Always-present scalar snapshot ----
    ActionType action_type;
    Phase      pre_phase;
    int8_t     pre_current_player;         // CHANCE_PLAYER, 0, or 1

    int        pre_turn_count;

    // ---- Optional: pending offer before this action ----
    // NO_PLAYER means "no offer was pending"
    int8_t     pre_pending_offer_player;   // NO_PLAYER | 0 | 1
    ActionType pre_pending_offer_type;     // SPLIT | GIFT | COMPETITION; valid iff a player
    Suit       pre_pending_offer_cards[MAX_OFFER_CARDS];
    int8_t     pre_pending_offer_count;

    // ---- Optional: pending_draw_player before this action ----
    // NO_PLAYER means "pending_draw_player was None"
    int8_t     pre_pending_draw_player;    // NO_PLAYER | 0 | 1

    // ---- Optional: round_starting_player before this action ----
    // NO_PLAYER means "round_starting_player was None"
    int8_t     pre_round_starting_player;  // NO_PLAYER | 0 | 1

    // ---- Cards removed from hand (card-playing actions) ----
    RemovedCard removed_cards[MAX_OFFER_CARDS];
    int8_t      removed_count;

    // ---- Action added to used_actions (RESERVE / DISCARD / SPLIT) ----
    // NO_PLAYER means "no used action was added"
    int8_t      added_used_action_player;  // NO_PLAYER | 0 | 1
    ActionType  added_used_action_type;    // valid iff added_used_action_player != NO_PLAYER

    // ---- Event history counts (how many events were appended per player) ----
    int8_t      events_appended_p0;
    int8_t      events_appended_p1;

    // ---- CHANCE_DEAL fields ----
    // dealt_cards: the sequence of cards dealt (even indices → p0, odd → p1)
    Suit        dealt_cards[MAX_ACTION_SUITS];
    int8_t      dealt_count;
    // Pre-deal hands (needed to restore exactly, even if usually empty)
    Suit        pre_hand_p0[MAX_HAND_SIZE];  int8_t pre_hand_p0_count;
    Suit        pre_hand_p1[MAX_HAND_SIZE];  int8_t pre_hand_p1_count;

    // ---- CHANCE_DRAW field ----
    // NO_PLAYER means "this record is not a CHANCE_DRAW"
    int8_t      draw_player;               // NO_PLAYER | 0 | 1

    // ------------------------------------------------------------------
    // Factory: create a "default" record from the current-action context,
    // then fill action-specific fields after.
    //
    // Mirrors the _make_undo_record() inner function in Python's
    // execute_action(): captures all scalars up front, then each branch
    // overlays its own fields.
    // ------------------------------------------------------------------
    static UndoRecord make(
        ActionType  at,
        Phase       pre_phase_,
        int8_t      pre_current_player_,
        int         pre_turn_count_,
        int8_t      pre_pending_offer_player_,
        ActionType  pre_pending_offer_type_,
        const Suit* pre_pending_offer_cards_,
        int8_t      pre_pending_offer_count_,
        int8_t      pre_pending_draw_player_,
        int8_t      pre_round_starting_player_) noexcept
    {
        UndoRecord r;
        r.init(at, pre_phase_, pre_current_player_, pre_turn_count_,
               pre_pending_offer_player_, pre_pending_offer_type_,
               pre_pending_offer_cards_, pre_pending_offer_count_,
               pre_pending_draw_player_, pre_round_starting_player_);
        return r;
    }

    // make(), filling this record in place (GameState builds records
    // directly in its undo stack).
    void init(
        ActionType  at,
        Phase       pre_phase_,
        int8_t      pre_current_player_,
        int         pre_turn_count_,
        int8_t      pre_pending_offer_player_,
        ActionType  pre_pending_offer_type_,
        const Suit* pre_pending_offer_cards_,
        int8_t      pre_pending_offer_count_,
        int8_t      pre_pending_draw_player_,
        int8_t      pre_round_starting_player_) noexcept
    {
        action_type                   = at;
        pre_phase                     = pre_phase_;
        pre_current_player            = pre_current_player_;
        pre_turn_count                = pre_turn_count_;
        pre_pending_offer_player      = pre_pending_offer_player_;
        pre_pending_offer_type        = pre_pending_offer_type_;
        pre_pending_offer_count       = pre_pending_offer_count_;
        if (pre_pending_offer_count_ > 0)   // fixed-size copy; the source holds MAX_OFFER_CARDS
            std::memcpy(pre_pending_offer_cards, pre_pending_offer_cards_, sizeof(pre_pending_offer_cards));
        pre_pending_draw_player       = pre_pending_draw_player_;
        pre_round_starting_player     = pre_round_starting_player_;
        removed_count                 = 0;
        added_used_action_player      = static_cast<int8_t>(NO_PLAYER);
        events_appended_p0            = 0;
        events_appended_p1            = 0;
        dealt_count                   = 0;
        pre_hand_p0_count             = 0;
        pre_hand_p1_count             = 0;
        draw_player                   = static_cast<int8_t>(NO_PLAYER);
    }

    // ------------------------------------------------------------------
    // Helpers for recording removed cards (push_back analogue).
    // ------------------------------------------------------------------
    void push_removed(int8_t player_id, int8_t hand_index, Suit suit) noexcept {
        removed_cards[removed_count++] = {player_id, hand_index, suit};
    }
};

} // namespace hana
