#pragma once

#include "enums.h"
#include "suit.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace hana {

// Compile-time upper bound on the number of suits a single Action can carry.
// Driven by CHANCE_DEAL = initial_deal_count = 2 * initial_hand_size.
// Must be >= max(initial_deal_count) across all registered variants.
// If a new variant needs a higher value, update this constant AND recompile.
static constexpr int MAX_ACTION_SUITS = 12;  // tiny=2, medium=4, full=12

// -----------------------------------------------------------------------
// RuntimeConfig — runtime game-variant configuration.
//
// Replaces the old compile-time GameConfig struct so that the same
// binary supports tiny_hana, medium_hana, and future variants without
// recompilation.
//
// suit_ranks[i] is both the number of copies of suit i in the initial
// deck AND the favor-point value of controlling that geisha.
// Entries beyond num_suits are unused (zero).
// -----------------------------------------------------------------------
struct RuntimeConfig {
    int     num_suits;
    int     suit_ranks[SUIT_CAPACITY];   // rank/deck-count per suit
    int     initial_hand_size;
    int     turns_per_round;             // 2 * actions_per_player
    int     initial_deal_count;          // = 2 * initial_hand_size
    int     win_geisha_count;
    bool    has_win_points;
    int     win_favor_points;            // ignored when has_win_points == false
    ActionMask available_actions;        // bitmask over ActionType
    // False when the game tree is far too large to enumerate (full
    // Hanamikoji).  Tree-walking tools (treeplex extraction, CFR, the Markov
    // solver) check it and refuse.  Mirrors Python GameConfig.enumerable.
    bool    enumerable;

    bool has_action(ActionType t) const noexcept {
        return (available_actions & action_bit(t)) != 0;
    }

    // ---- Built-in variant factories ----

    static const RuntimeConfig& tiny_hana() noexcept {
        static const RuntimeConfig cfg = []() noexcept {
            RuntimeConfig c;
            memset(&c, 0, sizeof(c));
            c.num_suits           = 3;
            c.suit_ranks[0]       = 2;
            c.suit_ranks[1]       = 2;
            c.suit_ranks[2]       = 3;
            c.initial_hand_size   = 1;
            c.turns_per_round     = 4;   // 2 players × 2 actions
            c.initial_deal_count  = 2;
            c.win_geisha_count    = 2;
            c.has_win_points      = false;
            c.win_favor_points    = 0;
            c.available_actions   = action_bit(ActionType::RESERVE) | action_bit(ActionType::SPLIT);
            c.enumerable          = true;
            static_assert(2 <= MAX_ACTION_SUITS,
                "tiny_hana initial_deal_count exceeds MAX_ACTION_SUITS — increase the constant");
            return c;
        }();
        return cfg;
    }

    static const RuntimeConfig& medium_hana() noexcept {
        static const RuntimeConfig cfg = []() noexcept {
            RuntimeConfig c;
            memset(&c, 0, sizeof(c));
            c.num_suits           = 5;
            c.suit_ranks[0]       = 1;
            c.suit_ranks[1]       = 1;
            c.suit_ranks[2]       = 2;
            c.suit_ranks[3]       = 3;
            c.suit_ranks[4]       = 4;
            c.initial_hand_size   = 2;
            c.turns_per_round     = 6;   // 2 players × 3 actions
            c.initial_deal_count  = 4;
            c.win_geisha_count    = 3;
            c.has_win_points      = true;
            c.win_favor_points    = 6;
            c.available_actions   = action_bit(ActionType::RESERVE) | action_bit(ActionType::DISCARD)
                                  | action_bit(ActionType::SPLIT);
            c.enumerable          = true;
            static_assert(4 <= MAX_ACTION_SUITS,
                "medium_hana initial_deal_count exceeds MAX_ACTION_SUITS — increase the constant");
            return c;
        }();
        return cfg;
    }

    // Full Hanamikoji: 7 geishas, 21 cards, 6-card hands, 4 actions each.
    static const RuntimeConfig& full_hana() noexcept {
        static const RuntimeConfig cfg = []() noexcept {
            RuntimeConfig c;
            memset(&c, 0, sizeof(c));
            c.num_suits           = 7;
            const int ranks[7]    = {2, 2, 2, 3, 3, 4, 5};
            for (int i = 0; i < 7; i++) c.suit_ranks[i] = ranks[i];
            c.initial_hand_size   = 6;
            c.turns_per_round     = 8;   // 2 players × 4 actions
            c.initial_deal_count  = 12;
            c.win_geisha_count    = 4;
            c.has_win_points      = true;
            c.win_favor_points    = 11;
            c.available_actions   = action_bit(ActionType::RESERVE) | action_bit(ActionType::TRADE_OFF)
                                  | action_bit(ActionType::GIFT) | action_bit(ActionType::COMPETITION);
            c.enumerable          = false;
            static_assert(12 <= MAX_ACTION_SUITS,
                "full_hana initial_deal_count exceeds MAX_ACTION_SUITS — increase the constant");
            return c;
        }();
        return cfg;
    }
};

// Throw if cfg's game tree is too large to enumerate (full Hanamikoji).
// Called by tools that walk the whole tree (treeplex extraction, CFR, the
// Markov solver) so they fail at once instead of running out of memory.
// Mirrors Python's engine.models.require_enumerable.
inline void require_enumerable(const RuntimeConfig& cfg, const char* what) {
    if (!cfg.enumerable)
        throw std::invalid_argument(
            std::string("the game tree of this variant is far too large to enumerate, so ") + what
            + " is not supported; use sampling-based methods instead");
}

} // namespace hana
