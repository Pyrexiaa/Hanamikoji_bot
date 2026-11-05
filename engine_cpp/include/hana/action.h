#pragma once

#include "card.h"
#include "enums.h"
#include "game_config.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <utility>

namespace hana {

// -----------------------------------------------------------------------
// Action — represents one action in the game tree.
//
// Invariant (must mirror Python Action.__init__ canonicalization):
//   For SPLIT and DISCARD, suits[0].value <= suits[1].value always holds.
//   When cards are swapped to enforce this, `choice` is flipped (1-choice).
//   For CHANCE_DEAL, each player's cards (even / odd positions) are sorted.
//   For TRADE_OFF and GIFT, the cards are sorted.
//   For COMPETITION (pair 0 = suits[0..1], pair 1 = suits[2..3]), each pair
//   is sorted and then the two pairs are ordered.
//
// Use factory methods only — the private constructor cannot be called
// directly, preventing uncanonicalised Actions from being created.
// -----------------------------------------------------------------------
struct Action {
    ActionType action_type;
    Suit       suits[MAX_ACTION_SUITS]; // card suits; only [0..suit_count) are valid
    int8_t     suit_count;              // 0..MAX_ACTION_SUITS
    int8_t     choice;      // -1 means "no choice" (Python: None)

    // ------------------------------------------------------------------
    // Factory methods
    // ------------------------------------------------------------------

    static Action make_reserve(Suit s) noexcept {
        Action a;
        a.action_type = ActionType::RESERVE;
        a.suits[0]    = s;
        a.suit_count  = 1;
        a.choice      = -1;
        return a;
    }

    // SPLIT: canonicalize so suits[0].value <= suits[1].value.
    static Action make_split(Suit a, Suit b) noexcept {
        Action act;
        act.action_type = ActionType::SPLIT;
        act.suit_count  = 2;
        act.choice      = -1;
        if (suit_value(a) > suit_value(b)) std::swap(a, b);
        act.suits[0] = a;
        act.suits[1] = b;
        return act;
    }

    // DISCARD: canonicalize card order; flip choice when swapped.
    static Action make_discard(Suit a, Suit b, int choice) noexcept {
        assert(choice == 0 || choice == 1);
        Action act;
        act.action_type = ActionType::DISCARD;
        act.suit_count  = 2;
        if (suit_value(a) > suit_value(b)) {
            std::swap(a, b);
            choice = 1 - choice;
        }
        act.suits[0] = a;
        act.suits[1] = b;
        act.choice   = static_cast<int8_t>(choice);
        return act;
    }

    static Action make_split_response(int choice) noexcept {
        assert(choice == 0 || choice == 1);
        Action a;
        a.action_type = ActionType::SPLIT_RESPONSE;
        a.suit_count  = 0;
        a.choice      = static_cast<int8_t>(choice);
        return a;
    }

    // TRADE_OFF: two cards discarded face down, in suit order.
    static Action make_trade_off(Suit a, Suit b) noexcept {
        if (suit_value(a) > suit_value(b)) std::swap(a, b);
        const Suit s[2] = {a, b};
        return make_unchecked(ActionType::TRADE_OFF, s, 2, -1);
    }

    // GIFT: three cards offered, in suit order.
    static Action make_gift(Suit a, Suit b, Suit c) noexcept {
        Suit s[3] = {a, b, c};
        sort_suits_(s, 3);
        return make_unchecked(ActionType::GIFT, s, 3, -1);
    }

    // COMPETITION: pairs (a, b) and (c, d); each pair sorted, then the pairs ordered.
    static Action make_competition(Suit a, Suit b, Suit c, Suit d) noexcept {
        Suit s[4] = {a, b, c, d};
        canonicalize_competition_(s);
        return make_unchecked(ActionType::COMPETITION, s, 4, -1);
    }

    // GIFT_RESPONSE: choice = index (0..2) of the offered card taken.
    static Action make_gift_response(int choice) noexcept {
        assert(choice >= 0 && choice <= 2);
        return make_unchecked(ActionType::GIFT_RESPONSE, nullptr, 0, choice);
    }

    // COMPETITION_RESPONSE: choice = index (0..1) of the offered pair taken.
    static Action make_competition_response(int choice) noexcept {
        assert(choice == 0 || choice == 1);
        return make_unchecked(ActionType::COMPETITION_RESPONSE, nullptr, 0, choice);
    }

    // Response of the given type (SPLIT_/GIFT_/COMPETITION_RESPONSE).
    static Action make_offer_response(ActionType response_type, int choice) noexcept {
        return make_unchecked(response_type, nullptr, 0, choice);
    }

    static Action make_chance_start(int choice) noexcept {
        assert(choice == 0 || choice == 1);
        Action a;
        a.action_type = ActionType::CHANCE_START;
        a.suit_count  = 0;
        a.choice      = static_cast<int8_t>(choice);
        return a;
    }

    // CHANCE_DEAL: suits[i] goes to player i % 2.  Canonicalized so each
    // player's cards are in ascending suit order (even positions sorted, odd
    // positions sorted): a hand is a multiset, so the order its cards were
    // dealt in carries no information.  Mirrors Python Action.__init__.
    static Action make_chance_deal(const Suit* suits, int count) noexcept {
        Action a = make_unchecked(ActionType::CHANCE_DEAL, suits, count, -1);
        canonicalize_deal_(a.suits, count);
        return a;
    }

    // Copies the fields as given, without canonicalization or validation.
    // Used by the permissive Python constructor, which canonicalizes itself.
    static Action make_unchecked(ActionType t, const Suit* suits, int count, int choice) noexcept {
        assert(count >= 0 && count <= MAX_ACTION_SUITS);
        Action a;
        a.action_type = t;
        a.suit_count  = static_cast<int8_t>(count);
        a.choice      = static_cast<int8_t>(choice);
        for (int i = 0; i < count; i++) a.suits[i] = suits[i];
        return a;
    }

    static Action make_chance_draw(Suit s, int player) noexcept {
        assert(player == 0 || player == 1);
        Action a;
        a.action_type = ActionType::CHANCE_DRAW;
        a.suits[0]    = s;
        a.suit_count  = 1;
        a.choice      = static_cast<int8_t>(player);
        return a;
    }

    // ------------------------------------------------------------------
    // Equality — two Actions are equal iff all fields match.
    // ------------------------------------------------------------------
    bool operator==(const Action& o) const noexcept {
        if (action_type != o.action_type) return false;
        if (suit_count  != o.suit_count)  return false;
        if (choice      != o.choice)      return false;
        for (int i = 0; i < suit_count; i++)
            if (suits[i] != o.suits[i]) return false;
        return true;
    }
    bool operator!=(const Action& o) const noexcept { return !(*this == o); }

    // ------------------------------------------------------------------
    // repr() — must produce byte-identical output to Python's __repr__.
    //
    // Python format:
    //   Action(RESERVE, (Card(GEISHA_1),), choice=None)
    //   Action(SPLIT, (Card(GEISHA_1), Card(GEISHA_2)), choice=None)
    //   Action(SPLIT_RESPONSE, (), choice=0)
    //   Action(CHANCE_DEAL, (Card(GEISHA_1), Card(GEISHA_1)), choice=None)
    // ------------------------------------------------------------------
    std::string repr() const {
        std::string s = "Action(";
        s += action_type_name(action_type);
        s += ", (";

        for (int i = 0; i < suit_count; i++) {
            if (i > 0) s += ", ";
            s += "Card(";
            s += suit_name(suits[i]);
            s += ")";
        }
        // Python tuple formatting: single-element tuple has trailing comma.
        if (suit_count == 1) s += ",";
        s += "), choice=";

        if (choice == -1) {
            s += "None";
        } else {
            s += std::to_string(static_cast<int>(choice));
        }
        s += ")";
        return s;
    }

private:
    // Prevent zero-initialised construction from outside the factory methods.
    Action() noexcept : suits{}, suit_count(0), choice(-1) {}

public:
    // Canonical card order for an action of type t (Python Action.__init__):
    // sorts TRADE_OFF / GIFT, orders COMPETITION pairs and CHANCE_DEAL hands.
    // SPLIT / DISCARD (which may flip a choice) are handled by their factories.
    static void canonicalize_cards(ActionType t, Suit* suits, int count) noexcept {
        if (t == ActionType::CHANCE_DEAL) canonicalize_deal_(suits, count);
        else if (t == ActionType::TRADE_OFF || t == ActionType::GIFT) sort_suits_(suits, count);
        else if (t == ActionType::COMPETITION && count == 4) canonicalize_competition_(suits);
    }

private:
    static void sort_suits_(Suit* suits, int count) noexcept {
        for (int i = 1; i < count; i++)
            for (int j = i; j > 0 && suits[j - 1] > suits[j]; j--)
                std::swap(suits[j - 1], suits[j]);
    }

    static void canonicalize_competition_(Suit* s) noexcept {
        if (s[0] > s[1]) std::swap(s[0], s[1]);
        if (s[2] > s[3]) std::swap(s[2], s[3]);
        if (s[2] < s[0] || (s[2] == s[0] && s[3] < s[1])) {
            std::swap(s[0], s[2]);
            std::swap(s[1], s[3]);
        }
    }

    // Insertion-sort each player's interleaved subsequence (stride 2).
    static void canonicalize_deal_(Suit* suits, int count) noexcept {
        for (int start = 0; start < 2; start++)
            for (int i = start + 2; i < count; i += 2)
                for (int j = i; j > start && suits[j - 2] > suits[j]; j -= 2)
                    std::swap(suits[j - 2], suits[j]);
    }
};

} // namespace hana

namespace std {
template<> struct hash<hana::Action> {
    std::size_t operator()(const hana::Action& a) const noexcept {
        auto mix = [](std::size_t seed, std::size_t v) noexcept -> std::size_t {
            return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
        };
        std::size_t h = std::hash<int>{}(static_cast<int>(a.action_type));
        h = mix(h, std::hash<int>{}(static_cast<int>(a.suit_count)));
        h = mix(h, std::hash<int>{}(static_cast<int>(a.choice)));
        for (int i = 0; i < a.suit_count; i++)
            h = mix(h, std::hash<int>{}(static_cast<int>(a.suits[i])));
        return h;
    }
};
} // namespace std
