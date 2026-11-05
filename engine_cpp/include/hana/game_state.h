#pragma once

#include "action.h"
#include "constants.h"
#include "deck.h"
#include "enums.h"
#include "events.h"
#include "game_config.h"
#include "info_state.h"
#include "legal_actions.h"
#include "player_state.h"
#include "suit.h"
#include "undo_record.h"
#include <cassert>
#include <cstring>
#include <variant>

// Hot-path helpers the compiler might otherwise leave out of line.
#if defined(__GNUC__) || defined(__clang__)
#define HANA_ALWAYS_INLINE __attribute__((always_inline))
#else
#define HANA_ALWAYS_INLINE
#endif

namespace hana {

// -----------------------------------------------------------------------
// RoundResolution — pure end-of-round snapshot, no mutation.
// Mirrors Python's RoundResolutionPreview.
// -----------------------------------------------------------------------
struct RoundResolution {
    Suit   p0_reserved[2];  int8_t p0_reserved_count;
    Suit   p1_reserved[2];  int8_t p1_reserved_count;
    int8_t final_favors[SUIT_CAPACITY]; // -1=p0, 0=tie, 1=p1
    int8_t num_suits;
    int8_t winner;          // NO_PLAYER if no winner, else 0 or 1
};

// -----------------------------------------------------------------------
// GameState — mutable game state (header-only for inlining).
//
// Supports multiple game variants via a RuntimeConfig stored at runtime.
//
// Favour encoding:  favors[i] = -1  → player 0 controls suit i
//                               0   → neutral
//                               1   → player 1 controls suit i
// -----------------------------------------------------------------------

// Full Hanamikoji: start + deal + 8 draws + 8 actions + 4 offer responses = 22.
static constexpr int MAX_UNDO_DEPTH = 24;

struct GameState {

    // ---- Configuration ----
    RuntimeConfig config_;

    // ---- Persistent across rounds ----
    int8_t  favors[SUIT_CAPACITY];
    int8_t  winner_;
    int     round_count;

    // ---- Per-round mutable state ----
    Deck        deck;
    PlayerState players[2];
    Phase       phase;
    int8_t      current_player_;
    int         turn_count;
    int8_t      round_starting_player_;
    int8_t      pending_draw_player_;
    // Offer awaiting the opponent's response within the same turn.
    int8_t      pending_offer_player_;                  // NO_PLAYER when none
    ActionType  pending_offer_type_;                    // SPLIT | GIFT | COMPETITION
    Suit        pending_offer_cards_[MAX_OFFER_CARDS];  // canonical; COMPETITION pairs = [0..1], [2..3]
    int8_t      pending_offer_count_;
    bool        favors_updated_this_round_;

    // ---- Undo stack ----
    UndoRecord  undo_stack_[MAX_UNDO_DEPTH];
    int         undo_stack_size_;

    // ---- Per-round event history ----
    Event  info_history_[2][MAX_EVENTS_PER_ROUND];
    int    info_history_count_[2];

    // ==================================================================
    // Construction / reset
    // ==================================================================

    void init(const RuntimeConfig& cfg) noexcept {
        config_ = cfg;
        for (int i = 0; i < SUIT_CAPACITY; i++) favors[i] = 0;
        winner_                = static_cast<int8_t>(NO_PLAYER);
        round_count            = 0;
        round_starting_player_ = static_cast<int8_t>(NO_PLAYER);
        begin_round_();
    }

    GameState() noexcept { init(RuntimeConfig::tiny_hana()); }
    explicit GameState(const RuntimeConfig& cfg) noexcept { init(cfg); }

    // ==================================================================
    // Public accessors
    // ==================================================================

    int current_player() const noexcept {
        return (phase == Phase::ROUND_COMPLETE)
            ? NO_PLAYER
            : static_cast<int>(current_player_);
    }

    bool is_round_complete() const noexcept {
        return turn_count >= config_.turns_per_round;
    }

    const Event* current_round_events(int pid) const noexcept {
        return info_history_[pid];
    }
    int current_round_event_count(int pid) const noexcept {
        return info_history_count_[pid];
    }

    RoundInfoState get_info_state(int player_id) const noexcept {
        assert(player_id == 0 || player_id == 1);
        RoundInfoState ris;
        ris.player_id    = player_id;
        ris.round_number = round_count;
        ris.num_suits    = config_.num_suits;
        ris.event_count  = info_history_count_[player_id];
        for (int i = 0; i < ris.event_count; i++)
            ris.events[i] = info_history_[player_id][i];
        return ris;
    }

    // ==================================================================
    // Legal action enumeration
    // ==================================================================

    template <typename F>
    void get_legal_actions(F&& callback) const {
        if (phase == Phase::ROUND_COMPLETE) return;

        if (current_player_ == static_cast<int8_t>(CHANCE_PLAYER)) {
            if (phase == Phase::CHOOSE_START) {
                callback(Action::make_chance_start(0));
                callback(Action::make_chance_start(1));
            } else if (phase == Phase::DEAL_INITIAL) {
                deck.iter_deal_actions(config_.initial_deal_count,
                    [&](const Action& a) { callback(a); });
            } else if (phase == Phase::DRAW_FOR_TURN) {
                assert(pending_draw_player_ == 0 || pending_draw_player_ == 1);
                deck.iter_draw_actions(static_cast<int>(pending_draw_player_),
                    [&](const Action& a) { callback(a); });
            }
            return;
        }

        if (has_pending_offer()) {
            enumerate_offer_responses(pending_offer_type_, callback);
            return;
        }

        enumerate_player_actions_(static_cast<int>(current_player_), callback);
    }

    template <typename F>
    void get_chance_actions_with_probs(F&& callback) const {
        assert(current_player_ == static_cast<int8_t>(CHANCE_PLAYER));

        if (phase == Phase::CHOOSE_START) {
            callback(Action::make_chance_start(0), 0.5);
            callback(Action::make_chance_start(1), 0.5);
        } else if (phase == Phase::DEAL_INITIAL) {
            deck.iter_deal_actions_with_probs(config_.initial_deal_count,
                [&](const Action& a, double p) { callback(a, p); });
        } else if (phase == Phase::DRAW_FOR_TURN) {
            assert(pending_draw_player_ == 0 || pending_draw_player_ == 1);
            deck.iter_draw_actions_with_probs(static_cast<int>(pending_draw_player_),
                [&](const Action& a, double p) { callback(a, p); });
        }
    }

    Action sample_chance_action(std::mt19937& rng) const {
        assert(current_player_ == static_cast<int8_t>(CHANCE_PLAYER));

        if (phase == Phase::CHOOSE_START) {
            std::uniform_int_distribution<int> coin(0, 1);
            return Action::make_chance_start(coin(rng));
        }
        if (phase == Phase::DEAL_INITIAL) {
            return deck.sample_deal(config_.initial_deal_count, rng);
        }
        assert(pending_draw_player_ == 0 || pending_draw_player_ == 1);
        return deck.sample_draw(static_cast<int>(pending_draw_player_), rng);
    }

    // ==================================================================
    // Core API — execute_action / undo_move
    // ==================================================================

    bool has_pending_offer() const noexcept {
        return pending_offer_player_ != static_cast<int8_t>(NO_PLAYER);
    }

    bool execute_action(const Action& action) noexcept {
        if (phase == Phase::ROUND_COMPLETE) return false;

        // Pre-action scalars for the undo record.  pending_offer_cards_ is
        // only written by a new offer, which requires no offer pending, so
        // the record may read the cards later.
        const Phase      pre_phase = phase;
        const int8_t     pre_cp    = current_player_;
        const int        pre_turn  = turn_count;
        const int8_t     pre_pop   = pending_offer_player_;
        const ActionType pre_pot   = pending_offer_type_;
        const int8_t     pre_poc   = pre_pop == static_cast<int8_t>(NO_PLAYER) ? 0 : pending_offer_count_;
        const int8_t     pre_pdp   = pending_draw_player_;
        const int8_t     pre_rsp   = round_starting_player_;
        // The record is built in place in the next undo-stack slot; it only
        // becomes part of the stack when push_undo_() is called on success.
        auto make_record = [&](ActionType at) HANA_ALWAYS_INLINE -> UndoRecord& {
            assert(undo_stack_size_ < MAX_UNDO_DEPTH);
            UndoRecord& r = undo_stack_[undo_stack_size_];
            r.init(at, pre_phase, pre_cp, pre_turn,
                   pre_pop, pre_pot, pending_offer_cards_, pre_poc, pre_pdp, pre_rsp);
            return r;
        };

        if (action.action_type == ActionType::CHANCE_START) {
            if (!execute_chance_start_(action)) return false;
            make_record(ActionType::CHANCE_START);
            push_undo_();
            return true;
        }

        if (action.action_type == ActionType::CHANCE_DEAL) {
            const int8_t ph0c = players[0].hand_count;
            const int8_t ph1c = players[1].hand_count;
            Suit ph0[MAX_HAND_SIZE], ph1[MAX_HAND_SIZE];
            for (int i = 0; i < ph0c; i++) ph0[i] = players[0].hand[i];
            for (int i = 0; i < ph1c; i++) ph1[i] = players[1].hand[i];

            if (!execute_chance_deal_(action)) return false;

            UndoRecord& r = make_record(ActionType::CHANCE_DEAL);
            r.events_appended_p0 = 1;
            r.events_appended_p1 = 1;
            r.dealt_count        = action.suit_count;
            for (int i = 0; i < action.suit_count; i++)
                r.dealt_cards[i] = action.suits[i];
            r.pre_hand_p0_count = ph0c;
            r.pre_hand_p1_count = ph1c;
            for (int i = 0; i < ph0c; i++) r.pre_hand_p0[i] = ph0[i];
            for (int i = 0; i < ph1c; i++) r.pre_hand_p1[i] = ph1[i];
            push_undo_();
            return true;
        }

        if (action.action_type == ActionType::CHANCE_DRAW) {
            const int8_t draw_player = pending_draw_player_;
            if (!execute_chance_draw_(action)) return false;
            UndoRecord& r = make_record(ActionType::CHANCE_DRAW);
            r.draw_player        = draw_player;
            r.events_appended_p0 = (draw_player == 0) ? 1 : 0;
            r.events_appended_p1 = (draw_player == 1) ? 1 : 0;
            push_undo_();
            return true;
        }

        if (current_player_ == static_cast<int8_t>(CHANCE_PLAYER)) return false;

        const int8_t actor_id = current_player_;
        const int8_t opp_id   = static_cast<int8_t>(1 - actor_id);

        // While an offer is pending, only its matching response is legal.
        if (has_pending_offer()) {
            if (action.action_type != offer_response_type(pending_offer_type_)) return false;
            if (!execute_offer_response_(action.choice)) return false;
            advance_after_offer_response_();
            UndoRecord& r = make_record(action.action_type);
            r.events_appended_p0 = 1;
            r.events_appended_p1 = 1;
            push_undo_();
            return true;
        }

        if (!config_.has_action(action.action_type)) return false;
        if (players[actor_id].has_used(action.action_type)) return false;

        UndoRecord& r = make_record(action.action_type);
        r.added_used_action_player = actor_id;
        r.added_used_action_type   = action.action_type;
        r.events_appended_p0       = 1;
        r.events_appended_p1       = 1;

        if (action.action_type == ActionType::RESERVE) {
            if (action.suit_count != 1) return false;
            int idx;
            if (!players[actor_id].pop_card_from_hand(action.suits[0], &idx))
                return false;
            r.push_removed(actor_id, static_cast<int8_t>(idx), action.suits[0]);
            players[actor_id].reserved[players[actor_id].reserved_count++] = action.suits[0];
            players[actor_id].mark_used(ActionType::RESERVE);
            record_event_(actor_id, OwnReserveEvent{turn_count, action.suits[0]});
            record_event_(opp_id,   OpponentReserveEvent{turn_count});
        }
        else if (action.action_type == ActionType::DISCARD) {
            if (action.suit_count != 2) return false;
            if (action.choice < 0 || action.choice > 1) return false;

            int idx0;
            if (!players[actor_id].pop_card_from_hand(action.suits[0], &idx0))
                return false;
            r.push_removed(actor_id, static_cast<int8_t>(idx0), action.suits[0]);

            int idx1;
            if (!players[actor_id].pop_card_from_hand(action.suits[1], &idx1)) {
                players[actor_id].insert_card_to_hand(idx0, action.suits[0]);
                return false;
            }
            r.push_removed(actor_id, static_cast<int8_t>(idx1), action.suits[1]);

            const Suit revealed = action.suits[action.choice];
            const Suit hidden   = action.suits[1 - action.choice];
            players[actor_id].discarded_rev[players[actor_id].discarded_rev_count++] = revealed;
            players[actor_id].discarded_hid[players[actor_id].discarded_hid_count++] = hidden;
            players[actor_id].mark_used(ActionType::DISCARD);
            record_event_(actor_id, OwnDiscardEvent{turn_count, revealed, hidden});
            record_event_(opp_id,   OpponentDiscardEvent{turn_count, revealed});
        }
        else if (action.action_type == ActionType::TRADE_OFF || is_offer(action.action_type)) {
            if (action.suit_count != action_card_count(action.action_type)) return false;
            if (action.action_type != ActionType::SPLIT && action.choice != -1) return false;
            if (!pop_cards_(actor_id, action, r)) return false;
            players[actor_id].mark_used(action.action_type);
            const Suit* c = action.suits;

            if (action.action_type == ActionType::TRADE_OFF) {
                // Both cards leave the round face down.
                PlayerState& ps = players[actor_id];
                ps.discarded_hid[ps.discarded_hid_count++] = c[0];
                ps.discarded_hid[ps.discarded_hid_count++] = c[1];
                record_event_(actor_id, OwnTradeOffEvent{turn_count, {c[0], c[1]}});
                record_event_(opp_id,   OpponentTradeOffEvent{turn_count});
            } else {
                // An offer: the opponent responds before the turn ends.
                pending_offer_player_ = actor_id;
                pending_offer_type_   = action.action_type;
                pending_offer_count_  = action.suit_count;
                for (int i = 0; i < action.suit_count; i++) pending_offer_cards_[i] = c[i];
                if (action.action_type == ActionType::SPLIT) {
                    record_event_(actor_id, OwnSplitEvent{turn_count, {c[0], c[1]}});
                    record_event_(opp_id,   OpponentSplitEvent{turn_count, {c[0], c[1]}});
                } else if (action.action_type == ActionType::GIFT) {
                    record_event_(actor_id, OwnGiftEvent{turn_count, {c[0], c[1], c[2]}});
                    record_event_(opp_id,   OpponentGiftEvent{turn_count, {c[0], c[1], c[2]}});
                } else {
                    record_event_(actor_id, OwnCompetitionEvent{turn_count, {c[0], c[1], c[2], c[3]}});
                    record_event_(opp_id,   OpponentCompetitionEvent{turn_count, {c[0], c[1], c[2], c[3]}});
                }
            }
        }
        else {
            return false;
        }

        advance_after_action_(action);
        push_undo_();
        return true;
    }

    void undo_move() noexcept {
        assert(!favors_updated_this_round_);
        assert(undo_stack_size_ > 0);

        const UndoRecord& r  = undo_stack_[--undo_stack_size_];
        const ActionType  at = r.action_type;

        if (at == ActionType::CHANCE_START) {
            restore_pre_action_scalars_(r);
            return;
        }

        if (at == ActionType::CHANCE_DEAL) {
            undo_pop_events_(0, r.events_appended_p0);
            undo_pop_events_(1, r.events_appended_p1);
            players[0].hand_count = r.pre_hand_p0_count;
            players[1].hand_count = r.pre_hand_p1_count;
            for (int i = 0; i < r.pre_hand_p0_count; i++) players[0].hand[i] = r.pre_hand_p0[i];
            for (int i = 0; i < r.pre_hand_p1_count; i++) players[1].hand[i] = r.pre_hand_p1[i];
            for (int i = 0; i < r.dealt_count; i++) deck.restore_suit(r.dealt_cards[i]);
            restore_pre_action_scalars_(r);
            return;
        }

        if (at == ActionType::CHANCE_DRAW) {
            assert(r.draw_player == 0 || r.draw_player == 1);
            undo_pop_events_(r.draw_player, 1);
            assert(players[r.draw_player].hand_count > 0);
            const Suit drawn = players[r.draw_player].hand[--players[r.draw_player].hand_count];
            deck.restore_suit(drawn);
            restore_pre_action_scalars_(r);
            return;
        }

        if (at == ActionType::SPLIT_RESPONSE || at == ActionType::GIFT_RESPONSE ||
            at == ActionType::COMPETITION_RESPONSE) {
            assert(r.pre_pending_offer_player != static_cast<int8_t>(NO_PLAYER));
            const int8_t offerer_id = r.pre_pending_offer_player;
            const int8_t chooser_id = r.pre_current_player;
            int chooser_n, offerer_n;
            offer_split_counts_(r.pre_pending_offer_type, &chooser_n, &offerer_n);
            assert(players[chooser_id].collected_count >= chooser_n);
            assert(players[offerer_id].collected_count >= offerer_n);
            players[chooser_id].collected_count -= chooser_n;
            players[offerer_id].collected_count -= offerer_n;
            undo_pop_events_(0, r.events_appended_p0);
            undo_pop_events_(1, r.events_appended_p1);
            restore_pre_action_scalars_(r);
            return;
        }

        if (at == ActionType::RESERVE) {
            const int8_t actor_id = r.pre_current_player;
            assert(players[actor_id].reserved_count > 0);
            players[actor_id].reserved_count--;
            if (r.added_used_action_player != static_cast<int8_t>(NO_PLAYER))
                players[r.added_used_action_player].unmark_used(r.added_used_action_type);
            undo_restore_removed_cards_(r);
            undo_pop_events_(0, r.events_appended_p0);
            undo_pop_events_(1, r.events_appended_p1);
            restore_pre_action_scalars_(r);
            return;
        }

        if (at == ActionType::DISCARD) {
            const int8_t actor_id = r.pre_current_player;
            assert(players[actor_id].discarded_rev_count > 0);
            assert(players[actor_id].discarded_hid_count > 0);
            players[actor_id].discarded_rev_count--;
            players[actor_id].discarded_hid_count--;
            if (r.added_used_action_player != static_cast<int8_t>(NO_PLAYER))
                players[r.added_used_action_player].unmark_used(r.added_used_action_type);
            undo_restore_removed_cards_(r);
            undo_pop_events_(0, r.events_appended_p0);
            undo_pop_events_(1, r.events_appended_p1);
            restore_pre_action_scalars_(r);
            return;
        }

        if (at == ActionType::TRADE_OFF) {
            const int8_t actor_id = r.pre_current_player;
            assert(players[actor_id].discarded_hid_count >= 2);
            players[actor_id].discarded_hid_count -= 2;
            if (r.added_used_action_player != static_cast<int8_t>(NO_PLAYER))
                players[r.added_used_action_player].unmark_used(r.added_used_action_type);
            undo_restore_removed_cards_(r);
            undo_pop_events_(0, r.events_appended_p0);
            undo_pop_events_(1, r.events_appended_p1);
            restore_pre_action_scalars_(r);
            return;
        }

        if (is_offer(at)) {
            if (r.added_used_action_player != static_cast<int8_t>(NO_PLAYER))
                players[r.added_used_action_player].unmark_used(r.added_used_action_type);
            undo_restore_removed_cards_(r);
            undo_pop_events_(0, r.events_appended_p0);
            undo_pop_events_(1, r.events_appended_p1);
            restore_pre_action_scalars_(r);
            return;
        }

        assert(false && "undo_move: unhandled action type");
    }

    // ==================================================================
    // Round resolution
    // ==================================================================

    RoundResolution preview_round_resolution() const noexcept {
        assert(is_round_complete());

        RoundResolution res;
        res.num_suits         = static_cast<int8_t>(config_.num_suits);
        res.p0_reserved_count = players[0].reserved_count;
        res.p1_reserved_count = players[1].reserved_count;
        for (int i = 0; i < players[0].reserved_count; i++)
            res.p0_reserved[i] = players[0].reserved[i];
        for (int i = 0; i < players[1].reserved_count; i++)
            res.p1_reserved[i] = players[1].reserved[i];

        for (int i = 0; i < config_.num_suits; i++) res.final_favors[i] = favors[i];
        for (int s = 0; s < config_.num_suits; s++) {
            const Suit suit = static_cast<Suit>(s);
            const int  c0   = players[0].suit_count(suit, /*include_reserved=*/true);
            const int  c1   = players[1].suit_count(suit, /*include_reserved=*/true);
            if (c0 > c1)      res.final_favors[s] = -1;
            else if (c1 > c0) res.final_favors[s] =  1;
        }

        res.winner = static_cast<int8_t>(winner_from_favors_(res.final_favors));
        return res;
    }

    float terminal_payoff(int player_id = 0) const noexcept {
        assert(player_id == 0 || player_id == 1);
        const RoundResolution res = preview_round_resolution();
        if (res.winner == static_cast<int8_t>(NO_PLAYER)) return 0.0f;
        return (res.winner == static_cast<int8_t>(player_id)) ? 1.0f : -1.0f;
    }

    void update_favors() noexcept {
        assert(is_round_complete());
        assert(!favors_updated_this_round_);

        const RoundResolution res = preview_round_resolution();

        for (int pid = 0; pid < 2; pid++) {
            PlayerState& ps = players[pid];
            for (int i = 0; i < ps.reserved_count; i++)
                ps.collected[ps.collected_count++] = ps.reserved[i];
            ps.reserved_count = 0;
        }

        for (int i = 0; i < config_.num_suits; i++) favors[i] = res.final_favors[i];

        RoundEndEvent ev;
        ev.turn              = turn_count;
        ev.n_suits           = static_cast<int8_t>(config_.num_suits);
        ev.p0_reserved_count = res.p0_reserved_count;
        ev.p1_reserved_count = res.p1_reserved_count;
        for (int i = 0; i < res.p0_reserved_count; i++) ev.p0_reserved[i] = res.p0_reserved[i];
        for (int i = 0; i < res.p1_reserved_count; i++) ev.p1_reserved[i] = res.p1_reserved[i];
        for (int i = 0; i < config_.num_suits; i++) ev.final_favors[i] = res.final_favors[i];
        record_event_(0, ev);
        record_event_(1, ev);

        favors_updated_this_round_ = true;
    }

    int check_winner() const noexcept {
        return winner_from_favors_(favors);
    }

    void begin_next_round() noexcept {
        assert(favors_updated_this_round_);
        assert(phase == Phase::ROUND_COMPLETE);
        begin_round_();
    }

    // ==================================================================
    // GameState::Key — value-based comparable snapshot.
    // ==================================================================
    struct Key {
        Phase   phase;
        int8_t  raw_current_player;
        int     turn_count;
        int     round_count;
        int8_t  round_starting_player;
        int8_t  pending_draw_player;
        int8_t  winner;
        int8_t  num_suits;                   // guards cross-variant key collisions
        int     deck_counts[SUIT_CAPACITY];
        int8_t  favors[SUIT_CAPACITY];
        int8_t     pending_offer_player;
        ActionType pending_offer_type;
        Suit       pending_offer_cards[MAX_OFFER_CARDS];  int8_t pending_offer_count;
        Suit    p0_hand[MAX_HAND_SIZE];       int8_t p0_hand_count;
        Suit    p0_reserved[2];               int8_t p0_reserved_count;
        Suit    p0_collected[MAX_COLLECTED];  int8_t p0_collected_count;
        Suit    p0_discarded_rev[2];          int8_t p0_discarded_rev_count;
        Suit    p0_discarded_hid[2];          int8_t p0_discarded_hid_count;
        ActionMask p0_used_actions;
        Suit    p1_hand[MAX_HAND_SIZE];       int8_t p1_hand_count;
        Suit    p1_reserved[2];               int8_t p1_reserved_count;
        Suit    p1_collected[MAX_COLLECTED];  int8_t p1_collected_count;
        Suit    p1_discarded_rev[2];          int8_t p1_discarded_rev_count;
        Suit    p1_discarded_hid[2];          int8_t p1_discarded_hid_count;
        ActionMask p1_used_actions;

        bool operator==(const Key& o) const noexcept {
            if (phase                != o.phase)                return false;
            if (raw_current_player   != o.raw_current_player)  return false;
            if (turn_count           != o.turn_count)          return false;
            if (round_count          != o.round_count)         return false;
            if (num_suits            != o.num_suits)           return false;
            if (round_starting_player != o.round_starting_player) return false;
            if (pending_draw_player  != o.pending_draw_player) return false;
            if (winner               != o.winner)              return false;
            if (pending_offer_player != o.pending_offer_player) return false;
            for (int i = 0; i < num_suits; i++) {
                if (deck_counts[i] != o.deck_counts[i]) return false;
                if (favors[i]      != o.favors[i])      return false;
            }
            if (pending_offer_player != static_cast<int8_t>(NO_PLAYER)) {
                if (pending_offer_type != o.pending_offer_type) return false;
                if (!suits_eq_(pending_offer_cards, pending_offer_count,
                               o.pending_offer_cards, o.pending_offer_count)) return false;
            }
            if (!suits_eq_(p0_hand, p0_hand_count, o.p0_hand, o.p0_hand_count))                         return false;
            if (!suits_eq_(p0_reserved, p0_reserved_count, o.p0_reserved, o.p0_reserved_count))         return false;
            if (!suits_eq_(p0_collected, p0_collected_count, o.p0_collected, o.p0_collected_count))     return false;
            if (!suits_eq_(p0_discarded_rev, p0_discarded_rev_count, o.p0_discarded_rev, o.p0_discarded_rev_count)) return false;
            if (!suits_eq_(p0_discarded_hid, p0_discarded_hid_count, o.p0_discarded_hid, o.p0_discarded_hid_count)) return false;
            if (p0_used_actions != o.p0_used_actions) return false;
            if (!suits_eq_(p1_hand, p1_hand_count, o.p1_hand, o.p1_hand_count))                         return false;
            if (!suits_eq_(p1_reserved, p1_reserved_count, o.p1_reserved, o.p1_reserved_count))         return false;
            if (!suits_eq_(p1_collected, p1_collected_count, o.p1_collected, o.p1_collected_count))     return false;
            if (!suits_eq_(p1_discarded_rev, p1_discarded_rev_count, o.p1_discarded_rev, o.p1_discarded_rev_count)) return false;
            if (!suits_eq_(p1_discarded_hid, p1_discarded_hid_count, o.p1_discarded_hid, o.p1_discarded_hid_count)) return false;
            if (p1_used_actions != o.p1_used_actions) return false;
            return true;
        }
        bool operator!=(const Key& o) const noexcept { return !(*this == o); }

    private:
        static bool suits_eq_(const Suit* a, int ac,
                               const Suit* b, int bc) noexcept {
            if (ac != bc) return false;
            for (int i = 0; i < ac; i++)
                if (a[i] != b[i]) return false;
            return true;
        }
    };

    Key to_key() const noexcept {
        Key k;
        k.phase                 = phase;
        k.raw_current_player    = current_player_;
        k.turn_count            = turn_count;
        k.round_count           = round_count;
        k.num_suits             = static_cast<int8_t>(config_.num_suits);
        k.round_starting_player = round_starting_player_;
        k.pending_draw_player   = pending_draw_player_;
        k.winner                = winner_;
        k.pending_offer_player  = pending_offer_player_;
        k.pending_offer_type    = pending_offer_type_;
        k.pending_offer_count   = has_pending_offer() ? pending_offer_count_ : 0;
        for (int i = 0; i < k.pending_offer_count; i++)
            k.pending_offer_cards[i] = pending_offer_cards_[i];
        for (int i = 0; i < config_.num_suits; i++) {
            k.deck_counts[i] = deck.counts[i];
            k.favors[i]      = favors[i];
        }
        fill_player_key_(players[0],
            k.p0_hand, k.p0_hand_count,
            k.p0_reserved, k.p0_reserved_count,
            k.p0_collected, k.p0_collected_count,
            k.p0_discarded_rev, k.p0_discarded_rev_count,
            k.p0_discarded_hid, k.p0_discarded_hid_count,
            k.p0_used_actions);
        fill_player_key_(players[1],
            k.p1_hand, k.p1_hand_count,
            k.p1_reserved, k.p1_reserved_count,
            k.p1_collected, k.p1_collected_count,
            k.p1_discarded_rev, k.p1_discarded_rev_count,
            k.p1_discarded_hid, k.p1_discarded_hid_count,
            k.p1_used_actions);
        return k;
    }

    // ==================================================================
    // Private helpers
    // ==================================================================
private:

    void begin_round_() noexcept {
        round_count++;
        for (int pid = 0; pid < 2; pid++) players[pid].reset_for_round();
        deck                       = Deck::full(config_);
        turn_count                 = 0;
        pending_offer_player_      = static_cast<int8_t>(NO_PLAYER);
        pending_offer_type_        = ActionType::SPLIT;
        pending_offer_count_       = 0;
        pending_draw_player_       = static_cast<int8_t>(NO_PLAYER);
        undo_stack_size_           = 0;
        favors_updated_this_round_ = false;
        info_history_count_[0]     = 0;
        info_history_count_[1]     = 0;

        if (round_starting_player_ == static_cast<int8_t>(NO_PLAYER)) {
            phase           = Phase::CHOOSE_START;
            current_player_ = static_cast<int8_t>(CHANCE_PLAYER);
        } else {
            round_starting_player_ =
                static_cast<int8_t>(1 - static_cast<int>(round_starting_player_));
            start_initial_deal_phase_();
        }
    }

    void start_initial_deal_phase_() noexcept {
        phase                = Phase::DEAL_INITIAL;
        current_player_      = static_cast<int8_t>(CHANCE_PLAYER);
        pending_draw_player_ = static_cast<int8_t>(NO_PLAYER);
    }

    void start_turn_draw_phase_(int player_id) noexcept {
        if (deck.total() <= 0) {
            phase                = Phase::PLAY;
            current_player_      = static_cast<int8_t>(player_id);
            pending_draw_player_ = static_cast<int8_t>(NO_PLAYER);
        } else {
            phase                = Phase::DRAW_FOR_TURN;
            current_player_      = static_cast<int8_t>(CHANCE_PLAYER);
            pending_draw_player_ = static_cast<int8_t>(player_id);
        }
    }

    bool execute_chance_start_(const Action& action) noexcept {
        if (phase != Phase::CHOOSE_START) return false;
        if (action.choice < 0 || action.choice > 1) return false;
        round_starting_player_ = static_cast<int8_t>(action.choice);
        start_initial_deal_phase_();
        return true;
    }

    bool execute_chance_deal_(const Action& action) noexcept {
        if (phase != Phase::DEAL_INITIAL) return false;
        if (round_starting_player_ < 0 || round_starting_player_ > 1) return false;
        if (action.suit_count != config_.initial_deal_count) return false;

        if (!deck.remove_suits(action.suits, action.suit_count)) return false;

        players[0].hand_count = 0;
        players[1].hand_count = 0;
        for (int i = 0; i < action.suit_count; i++)
            players[i % 2].hand[players[i % 2].hand_count++] = action.suits[i];

        for (int pid = 0; pid < 2; pid++) {
            record_event_(pid, make_round_start_event(
                0,
                players[pid].hand, players[pid].hand_count,
                favors,
                config_.num_suits,
                static_cast<int>(round_starting_player_),
                config_.available_actions));
        }

        start_turn_draw_phase_(static_cast<int>(round_starting_player_));
        return true;
    }

    bool execute_chance_draw_(const Action& action) noexcept {
        if (phase != Phase::DRAW_FOR_TURN) return false;
        if (pending_draw_player_ < 0 || pending_draw_player_ > 1) return false;
        if (action.choice != static_cast<int>(pending_draw_player_)) return false;
        if (action.suit_count != 1) return false;

        const int pid = static_cast<int>(pending_draw_player_);
        if (!deck.draw_specific(action.suits[0])) return false;

        players[pid].hand[players[pid].hand_count++] = action.suits[0];
        record_event_(pid, DrawCardEvent{turn_count, action.suits[0]});

        current_player_      = static_cast<int8_t>(pid);
        pending_draw_player_ = static_cast<int8_t>(NO_PLAYER);
        phase                = Phase::PLAY;
        return true;
    }

    // Remove action.suits[0..suit_count) from the actor's hand, recording
    // each removal in r.  On a missing card, restores the hand and r.
    bool pop_cards_(int8_t actor_id, const Action& action, UndoRecord& r) noexcept {
        PlayerState& ps = players[actor_id];
        const int8_t start = r.removed_count;
        for (int i = 0; i < action.suit_count; i++) {
            int idx;
            if (!ps.pop_card_from_hand(action.suits[i], &idx)) {
                while (r.removed_count > start) {
                    const RemovedCard& rc = r.removed_cards[--r.removed_count];
                    ps.insert_card_to_hand(rc.hand_index, rc.suit);
                }
                return false;
            }
            r.push_removed(actor_id, static_cast<int8_t>(idx), action.suits[i]);
        }
        return true;
    }

    // How many offered cards the chooser and the offerer each get.
    static void offer_split_counts_(ActionType offer, int* chooser_n, int* offerer_n) noexcept {
        switch (offer) {
            case ActionType::GIFT:        *chooser_n = 1; *offerer_n = 2; break;
            case ActionType::COMPETITION: *chooser_n = 2; *offerer_n = 2; break;
            default:                      *chooser_n = 1; *offerer_n = 1; break;
        }
    }

    // Apply the response `choice` to the pending offer.  Mirrors Python's
    // _execute_offer_response: the chooser takes card `choice` (SPLIT, GIFT)
    // or pair `choice` (COMPETITION); the offerer gets the rest, in order.
    bool execute_offer_response_(int choice) noexcept {
        if (!has_pending_offer()) return false;
        if (choice < 0 || choice >= offer_response_choices(pending_offer_type_)) return false;

        const int8_t offerer_id = pending_offer_player_;
        const int8_t chooser_id = current_player_;
        const Suit*  c          = pending_offer_cards_;

        if (pending_offer_type_ == ActionType::SPLIT) {   // fast path (tiny / medium)
            const Suit chooser_gets = c[choice];
            const Suit offerer_gets = c[1 - choice];
            players[chooser_id].collected[players[chooser_id].collected_count++] = chooser_gets;
            players[offerer_id].collected[players[offerer_id].collected_count++] = offerer_gets;
            const SplitChoiceEvent ev{turn_count, chooser_id, static_cast<int8_t>(choice),
                                      chooser_gets, offerer_gets};
            record_event_(0, ev);
            record_event_(1, ev);
            pending_offer_player_ = static_cast<int8_t>(NO_PLAYER);
            return true;
        }

        // GIFT and COMPETITION.
        Suit chooser_gets[2], offerer_gets[2];
        int  nc = 0, no = 0;
        if (pending_offer_type_ == ActionType::COMPETITION) {
            for (int k = 0; k < 2; k++) {
                chooser_gets[nc++] = c[2 * choice + k];
                offerer_gets[no++] = c[2 * (1 - choice) + k];
            }
        } else {
            chooser_gets[nc++] = c[choice];
            for (int k = 0; k < pending_offer_count_; k++)
                if (k != choice) offerer_gets[no++] = c[k];
        }

        PlayerState& chooser = players[chooser_id];
        PlayerState& offerer = players[offerer_id];
        for (int k = 0; k < nc; k++) chooser.collected[chooser.collected_count++] = chooser_gets[k];
        for (int k = 0; k < no; k++) offerer.collected[offerer.collected_count++] = offerer_gets[k];

        const int8_t ch = static_cast<int8_t>(choice);
        if (pending_offer_type_ == ActionType::GIFT) {
            const GiftChoiceEvent ev{turn_count, chooser_id, ch, chooser_gets[0],
                                     {offerer_gets[0], offerer_gets[1]}};
            record_event_(0, ev);
            record_event_(1, ev);
        } else {
            const CompetitionChoiceEvent ev{turn_count, chooser_id, ch,
                                            {chooser_gets[0], chooser_gets[1]},
                                            {offerer_gets[0], offerer_gets[1]}};
            record_event_(0, ev);
            record_event_(1, ev);
        }

        pending_offer_player_ = static_cast<int8_t>(NO_PLAYER);
        return true;
    }

    void advance_after_action_(const Action& action) noexcept {
        if (is_offer(action.action_type)) {
            current_player_ =
                static_cast<int8_t>(1 - static_cast<int>(current_player_));
            phase = Phase::PLAY;
        } else {
            turn_count++;
            current_player_ =
                static_cast<int8_t>(1 - static_cast<int>(current_player_));
            if (!is_round_complete())
                start_turn_draw_phase_(static_cast<int>(current_player_));
            else {
                phase                = Phase::ROUND_COMPLETE;
                pending_draw_player_ = static_cast<int8_t>(NO_PLAYER);
            }
        }
    }

    void advance_after_offer_response_() noexcept {
        turn_count++;
        if (!is_round_complete())
            start_turn_draw_phase_(static_cast<int>(current_player_));
        else {
            phase                = Phase::ROUND_COMPLETE;
            pending_draw_player_ = static_cast<int8_t>(NO_PLAYER);
        }
    }

    void restore_pre_action_scalars_(const UndoRecord& r) noexcept {
        phase                   = r.pre_phase;
        current_player_         = r.pre_current_player;
        turn_count              = r.pre_turn_count;
        pending_offer_player_   = r.pre_pending_offer_player;
        pending_offer_type_     = r.pre_pending_offer_type;
        pending_offer_count_    = r.pre_pending_offer_count;
        for (int i = 0; i < r.pre_pending_offer_count; i++)
            pending_offer_cards_[i] = r.pre_pending_offer_cards[i];
        pending_draw_player_    = r.pre_pending_draw_player;
        round_starting_player_  = r.pre_round_starting_player;
    }

    void undo_restore_removed_cards_(const UndoRecord& r) noexcept {
        for (int i = r.removed_count - 1; i >= 0; i--) {
            const RemovedCard& rc = r.removed_cards[i];
            players[rc.player_id].insert_card_to_hand(
                static_cast<int>(rc.hand_index), rc.suit);
        }
    }

    void record_event_(int pid, Event ev) noexcept {
        assert(info_history_count_[pid] < MAX_EVENTS_PER_ROUND);
        info_history_[pid][info_history_count_[pid]++] = ev;
    }

    void undo_pop_events_(int pid, int count) noexcept {
        if (count <= 0) return;
        assert(info_history_count_[pid] >= count);
        info_history_count_[pid] -= count;
    }

    // Commit the record make_record() built in the next undo-stack slot.
    void push_undo_() noexcept {
        assert(undo_stack_size_ < MAX_UNDO_DEPTH);
        undo_stack_size_++;
    }

    // Enumerate the player's actions (order documented in legal_actions.h).
    template <typename F>
    void enumerate_player_actions_(int pid, F&& callback) const {
        const PlayerState& ps = players[pid];
        int counts[SUIT_CAPACITY] = {};
        for (int i = 0; i < ps.hand_count; i++)
            counts[suit_value(ps.hand[i])]++;
        enumerate_player_actions(counts, config_.num_suits, ps.hand_count,
                                 config_.available_actions, ps.used_actions, callback);
    }

    int winner_from_favors_(const int8_t fav[]) const noexcept {
        int points[2] = {0, 0};
        int counts[2] = {0, 0};
        for (int i = 0; i < config_.num_suits; i++) {
            if      (fav[i] == -1) { counts[0]++; points[0] += config_.suit_ranks[i]; }
            else if (fav[i] ==  1) { counts[1]++; points[1] += config_.suit_ranks[i]; }
        }
        if (config_.has_win_points) {
            if (points[0] >= config_.win_favor_points) return 0;
            if (points[1] >= config_.win_favor_points) return 1;
        }
        if (counts[0] >= config_.win_geisha_count) return 0;
        if (counts[1] >= config_.win_geisha_count) return 1;
        return NO_PLAYER;
    }

    static void copy_sorted_(const Suit* src, int count, Suit* dst) noexcept {
        for (int i = 0; i < count; i++) dst[i] = src[i];
        for (int i = 1; i < count; i++) {
            const Suit x = dst[i];
            int j = i - 1;
            while (j >= 0 && suit_value(dst[j]) > suit_value(x)) {
                dst[j + 1] = dst[j];
                j--;
            }
            dst[j + 1] = x;
        }
    }

    static void fill_player_key_(
        const PlayerState& ps,
        Suit* hand,  int8_t& hand_count,
        Suit* rsv,   int8_t& rsv_count,
        Suit* col,   int8_t& col_count,
        Suit* drev,  int8_t& drev_count,
        Suit* dhid,  int8_t& dhid_count,
        ActionMask& used) noexcept
    {
        copy_sorted_(ps.hand,          ps.hand_count,          hand);  hand_count  = ps.hand_count;
        copy_sorted_(ps.reserved,      ps.reserved_count,      rsv);   rsv_count   = ps.reserved_count;
        copy_sorted_(ps.collected,     ps.collected_count,     col);   col_count   = ps.collected_count;
        copy_sorted_(ps.discarded_rev, ps.discarded_rev_count, drev);  drev_count  = ps.discarded_rev_count;
        copy_sorted_(ps.discarded_hid, ps.discarded_hid_count, dhid);  dhid_count  = ps.discarded_hid_count;
        used = ps.used_actions;
    }
};

} // namespace hana
