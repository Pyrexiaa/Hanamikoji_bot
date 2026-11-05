#pragma once

#include "action.h"
#include "constants.h"
#include "events.h"
#include "game_config.h"
#include "legal_actions.h"
#include "suit.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <string>

namespace hana {

// -----------------------------------------------------------------------
// Result structs for RoundInfoState query methods
// -----------------------------------------------------------------------

struct HandAndActions {
    Suit       hand[MAX_HAND_SIZE];  // current hand cards (in event-stream order)
    int8_t     hand_count;
    ActionMask used_actions;         // ActionType bitmask
};

struct CollectedCards {
    Suit   my_cards[MAX_COLLECTED];
    int8_t my_count;
    Suit   opp_cards[MAX_COLLECTED];
    int8_t opp_count;
};

// Discard information (my revealed, my hidden, opponent revealed).
// Both cards of a TRADE_OFF are hidden discards.
struct DiscardedCards {
    Suit   my_revealed[2];
    Suit   my_hidden[2];
    Suit   opp_revealed[2];
    int8_t my_revealed_count;
    int8_t my_hidden_count;
    int8_t opp_revealed_count;
};

struct PendingSplit {
    bool   has_pending;
    Suit   cards[2];  // cards[0].value <= cards[1].value (canonical order)
};

// An opponent's offer awaiting this player's response.
struct PendingOffer {
    bool       has_pending;
    ActionType offer_type;               // SPLIT | GIFT | COMPETITION
    Suit       cards[MAX_OFFER_CARDS];   // canonical; COMPETITION pairs = [0..1], [2..3]
    int8_t     count;                    // 2, 3 or 4
};

// -----------------------------------------------------------------------
// RoundInfoState
//
// Immutable snapshot of what one player has observed during a round.
// Mirrors Python's RoundInfoState dataclass in engine/info_state.py.
//
// Key invariant: to_string() must be byte-identical to Python's
//   str(RoundInfoState) — it is used as the policy table key.
//
// Lifecycle:
//   Built by GameState::get_info_state(player_id).
//   Self-contained: does NOT hold a pointer back to GameState.
//   Safe to use after the originating GameState is mutated or destroyed.
// -----------------------------------------------------------------------
struct RoundInfoState {
    int   player_id;
    int   round_number;
    int   num_suits;     // set by GameState::get_info_state from config_.num_suits
    Event events[MAX_EVENTS_PER_ROUND];
    int   event_count;

    // ==================================================================
    // Core API — to_string / hash / equality
    // ==================================================================

    // The policy table key.
    // Mirrors Python's RoundInfoState.__str__():
    //   "p{player_id}r{round_number}|event0_str|event1_str|..."
    std::string to_string() const {
        std::string s = "p";
        s += std::to_string(player_id);
        s += "r";
        s += std::to_string(round_number);
        for (int i = 0; i < event_count; i++) {
            s += "|";
            s += event_to_string(events[i]);
        }
        return s;
    }

    // The same information as to_string(), as a few bytes: player id, round
    // number, then each event's code and fields (see events.h and
    // engine/compact_key.py). An alternative key for solvers that store many
    // information sets; to_string() stays the policy-table key.
    std::string compact_key() const {
        std::string out;
        out.reserve(2 + 8 * event_count);
        put_byte(out, player_id);
        put_byte(out, round_number);
        for (int i = 0; i < event_count; i++) event_append_compact(events[i], out);
        return out;
    }

    // FNV-1a hash of to_string() — consistent with Python hash for same key.
    std::size_t hash() const {
        return std::hash<std::string>{}(to_string());
    }

    bool operator==(const RoundInfoState& o) const noexcept {
        if (player_id    != o.player_id)    return false;
        if (round_number != o.round_number) return false;
        if (event_count  != o.event_count)  return false;
        // Event variant lacks operator==; compare via canonical string.
        return to_string() == o.to_string();
    }
    bool operator!=(const RoundInfoState& o) const noexcept {
        return !(*this == o);
    }

    // ==================================================================
    // Query methods — all mirror Python's RoundInfoState methods.
    // Each replays the event stream; O(event_count).
    // ==================================================================

    // Returns the variant's available_actions bitmask.
    ActionMask get_available_action_types() const noexcept {
        assert(event_count > 0 && "RoundInfoState has no events");
        const auto* rs = std::get_if<RoundStartEvent>(&events[0]);
        assert(rs && "events[0] must be RoundStartEvent");
        return rs->actions_bitmask;
    }

    // Reconstructs current hand and used-action bitmask by replaying events.
    HandAndActions get_current_hand_and_actions() const noexcept {
        HandAndActions ha{};
        for (int i = 0; i < event_count; i++) {
            if (const auto* rs = std::get_if<RoundStartEvent>(&events[i])) {
                ha.hand_count = rs->initial_hand_count;
                for (int j = 0; j < rs->initial_hand_count; j++)
                    ha.hand[j] = rs->initial_hand[j];
            }
            else if (const auto* dc = std::get_if<DrawCardEvent>(&events[i])) {
                ha.hand[ha.hand_count++] = dc->card;
            }
            else if (const auto* orv = std::get_if<OwnReserveEvent>(&events[i])) {
                remove_first_(ha.hand, ha.hand_count, orv->card);
                ha.used_actions |= action_bit(ActionType::RESERVE);
            }
            else if (const auto* od = std::get_if<OwnDiscardEvent>(&events[i])) {
                remove_first_(ha.hand, ha.hand_count, od->revealed_card);
                remove_first_(ha.hand, ha.hand_count, od->hidden_card);
                ha.used_actions |= action_bit(ActionType::DISCARD);
            }
            else if (const auto* os = std::get_if<OwnSplitEvent>(&events[i])) {
                remove_first_(ha.hand, ha.hand_count, os->cards[0]);
                remove_first_(ha.hand, ha.hand_count, os->cards[1]);
                ha.used_actions |= action_bit(ActionType::SPLIT);
            }
            else if (const auto* ot = std::get_if<OwnTradeOffEvent>(&events[i])) {
                for (Suit c : ot->cards) remove_first_(ha.hand, ha.hand_count, c);
                ha.used_actions |= action_bit(ActionType::TRADE_OFF);
            }
            else if (const auto* og = std::get_if<OwnGiftEvent>(&events[i])) {
                for (Suit c : og->cards) remove_first_(ha.hand, ha.hand_count, c);
                ha.used_actions |= action_bit(ActionType::GIFT);
            }
            else if (const auto* oc = std::get_if<OwnCompetitionEvent>(&events[i])) {
                for (Suit c : oc->cards) remove_first_(ha.hand, ha.hand_count, c);
                ha.used_actions |= action_bit(ActionType::COMPETITION);
            }
        }
        return ha;
    }

    // Returns collected cards split between self and opponent.
    CollectedCards get_collected_cards() const noexcept {
        CollectedCards cc{};
        // Append the chooser's and the offerer's cards to the right sides.
        auto add = [&](int8_t chooser, const Suit* chooser_gets, int nc,
                       const Suit* offerer_gets, int no) {
            const bool mine = chooser == static_cast<int8_t>(player_id);
            for (int k = 0; k < nc; k++)
                (mine ? cc.my_cards[cc.my_count++] : cc.opp_cards[cc.opp_count++]) = chooser_gets[k];
            for (int k = 0; k < no; k++)
                (mine ? cc.opp_cards[cc.opp_count++] : cc.my_cards[cc.my_count++]) = offerer_gets[k];
        };
        for (int i = 0; i < event_count; i++) {
            if (const auto* sc = std::get_if<SplitChoiceEvent>(&events[i]))
                add(sc->chooser, &sc->chooser_gets, 1, &sc->offerer_gets, 1);
            else if (const auto* gc = std::get_if<GiftChoiceEvent>(&events[i]))
                add(gc->chooser, &gc->chooser_gets, 1, gc->offerer_gets, 2);
            else if (const auto* cc2 = std::get_if<CompetitionChoiceEvent>(&events[i]))
                add(cc2->chooser, cc2->chooser_gets, 2, cc2->offerer_gets, 2);
        }
        return cc;
    }

    // Writes current favour array into out[SUIT_CAPACITY].
    void get_favors(int8_t out[SUIT_CAPACITY]) const noexcept {
        for (int i = 0; i < num_suits; i++) out[i] = 0;
        for (int i = 0; i < event_count; i++) {
            if (const auto* rs = std::get_if<RoundStartEvent>(&events[i])) {
                for (int j = 0; j < num_suits; j++) out[j] = rs->favors[j];
            }
            else if (const auto* re = std::get_if<RoundEndEvent>(&events[i])) {
                for (int j = 0; j < num_suits; j++) out[j] = re->final_favors[j];
            }
        }
    }

    // Returns suits this player reserved (from OwnReserveEvents).
    int get_my_reserved(Suit out[2]) const noexcept {
        int count = 0;
        for (int i = 0; i < event_count; i++) {
            if (const auto* orv = std::get_if<OwnReserveEvent>(&events[i]))
                out[count++] = orv->card;
        }
        return count;
    }

    // Count of OpponentReserveEvents.
    int get_opponent_reserved_count() const noexcept {
        int count = 0;
        for (int i = 0; i < event_count; i++) {
            if (std::holds_alternative<OpponentReserveEvent>(events[i]))
                count++;
        }
        return count;
    }

    // Mirrors get_discarded_cards().
    DiscardedCards get_discarded_cards() const noexcept {
        DiscardedCards dc{};
        for (int i = 0; i < event_count; i++) {
            if (const auto* od = std::get_if<OwnDiscardEvent>(&events[i])) {
                dc.my_revealed[dc.my_revealed_count++] = od->revealed_card;
                dc.my_hidden[dc.my_hidden_count++]     = od->hidden_card;
            }
            else if (const auto* opd = std::get_if<OpponentDiscardEvent>(&events[i])) {
                dc.opp_revealed[dc.opp_revealed_count++] = opd->revealed_card;
            }
            else if (const auto* ot = std::get_if<OwnTradeOffEvent>(&events[i])) {
                for (Suit c : ot->cards) dc.my_hidden[dc.my_hidden_count++] = c;
            }
        }
        return dc;
    }

    // Bitmask of action types the opponent has used.
    ActionMask get_opponent_used_actions() const noexcept {
        ActionMask used = 0;
        for (int i = 0; i < event_count; i++) {
            if (std::holds_alternative<OpponentReserveEvent>(events[i]))
                used |= action_bit(ActionType::RESERVE);
            else if (std::holds_alternative<OpponentDiscardEvent>(events[i]))
                used |= action_bit(ActionType::DISCARD);
            else if (std::holds_alternative<OpponentSplitEvent>(events[i]))
                used |= action_bit(ActionType::SPLIT);
            else if (std::holds_alternative<OpponentTradeOffEvent>(events[i]))
                used |= action_bit(ActionType::TRADE_OFF);
            else if (std::holds_alternative<OpponentGiftEvent>(events[i]))
                used |= action_bit(ActionType::GIFT);
            else if (std::holds_alternative<OpponentCompetitionEvent>(events[i]))
                used |= action_bit(ActionType::COMPETITION);
        }
        return used;
    }

    // Turn number of the last event. Returns 0 if no events.
    int get_current_turn() const noexcept {
        int turn = 0;
        for (int i = 0; i < event_count; i++)
            std::visit([&](const auto& ev) { turn = ev.turn; }, events[i]);
        return turn;
    }

    // Returns the pending split the player must respond to (if any).
    PendingSplit get_pending_split() const noexcept {
        PendingSplit ps{false, {}};
        for (int i = 0; i < event_count; i++) {
            if (const auto* os = std::get_if<OpponentSplitEvent>(&events[i])) {
                ps.has_pending = true;
                ps.cards[0]    = os->cards[0];
                ps.cards[1]    = os->cards[1];
            }
            else if (std::holds_alternative<SplitChoiceEvent>(events[i])) {
                ps.has_pending = false;
            }
        }
        return ps;
    }

    // The opponent's offer this player must respond to (if any).
    // Mirrors Python's get_pending_offer().
    PendingOffer get_pending_offer() const noexcept {
        PendingOffer po{};
        po.has_pending = false;
        auto set = [&](ActionType t, const Suit* cards, int n) {
            po.has_pending = true;
            po.offer_type  = t;
            po.count       = static_cast<int8_t>(n);
            for (int k = 0; k < n; k++) po.cards[k] = cards[k];
        };
        for (int i = 0; i < event_count; i++) {
            if (const auto* os = std::get_if<OpponentSplitEvent>(&events[i]))
                set(ActionType::SPLIT, os->cards, 2);
            else if (const auto* og = std::get_if<OpponentGiftEvent>(&events[i]))
                set(ActionType::GIFT, og->cards, 3);
            else if (const auto* oc = std::get_if<OpponentCompetitionEvent>(&events[i]))
                set(ActionType::COMPETITION, oc->cards, 4);
            else if (std::holds_alternative<SplitChoiceEvent>(events[i]) ||
                     std::holds_alternative<GiftChoiceEvent>(events[i]) ||
                     std::holds_alternative<CompetitionChoiceEvent>(events[i]))
                po.has_pending = false;
        }
        return po;
    }

    // ==================================================================
    // Legal action enumeration — mirrors Python's get_legal_actions():
    // the responses to a pending offer, otherwise the player's actions in
    // the order documented in legal_actions.h.  Only valid for player
    // decision nodes (not chance nodes).
    // ==================================================================
    template <typename F>
    void get_legal_actions(F&& callback) const {
        const PendingOffer po = get_pending_offer();
        if (po.has_pending) {
            enumerate_offer_responses(po.offer_type, callback);
            return;
        }

        const HandAndActions ha = get_current_hand_and_actions();
        int counts[SUIT_CAPACITY] = {};
        for (int i = 0; i < ha.hand_count; i++)
            counts[suit_value(ha.hand[i])]++;
        enumerate_player_actions(counts, num_suits, ha.hand_count,
                                 get_available_action_types(), ha.used_actions, callback);
    }

private:

    // Remove the first occurrence of `target` from hand[0..count).
    static void remove_first_(Suit* hand, int8_t& count, Suit target) noexcept {
        for (int j = 0; j < count; j++) {
            if (hand[j] == target) {
                for (int k = j; k < count - 1; k++)
                    hand[k] = hand[k + 1];
                count--;
                return;
            }
        }
    }
};

} // namespace hana
