#pragma once

#include "action.h"
#include "game_config.h"
#include "suit.h"
#include <cassert>
#include <functional>
#include <random>

namespace hana {

// -----------------------------------------------------------------------
// Deck — the remaining cards modelled as a multiset (count per suit).
//
// Mirrors Python's Deck class: remaining_card_counts[suit] = count.
// Cards are indistinguishable within a suit.
//
// n_suits stores the active suit count for this variant; entries in
// counts[] beyond n_suits are always zero.
// -----------------------------------------------------------------------
struct Deck {
    int counts[SUIT_CAPACITY] = {};
    int n_suits               = 0;

    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    // Create a full deck for the start of a round from a RuntimeConfig.
    // Matches Python's Deck.full(suits_enum): counts[suit] = suit.rank.
    static Deck full(const RuntimeConfig& cfg) noexcept {
        Deck d;
        d.n_suits = cfg.num_suits;
        for (int i = 0; i < cfg.num_suits; i++)
            d.counts[i] = cfg.suit_ranks[i];
        return d;
    }

    // ------------------------------------------------------------------
    // Basic queries
    // ------------------------------------------------------------------

    int total() const noexcept {
        int t = 0;
        for (int i = 0; i < n_suits; i++) t += counts[i];
        return t;
    }

    bool empty() const noexcept { return total() == 0; }

    // ------------------------------------------------------------------
    // Mutation
    // ------------------------------------------------------------------

    // Remove one card of `suit`. Returns false if none remain.
    bool draw_specific(Suit s) noexcept {
        int idx = static_cast<int>(s);
        if (counts[idx] <= 0) return false;
        counts[idx]--;
        return true;
    }

    // Remove the given multiset of suits. Returns false and does NOT
    // mutate if any removal would underflow.
    bool remove_suits(const Suit* suits, int n) noexcept {
        int needed[SUIT_CAPACITY] = {};
        for (int i = 0; i < n; i++) {
            int idx = static_cast<int>(suits[i]);
            needed[idx]++;
            if (needed[idx] > counts[idx]) return false;
        }
        for (int i = 0; i < n_suits; i++) counts[i] -= needed[i];
        return true;
    }

    // Restore cards to the deck (used by undo).
    void restore_suit(Suit s) noexcept {
        counts[static_cast<int>(s)]++;
    }

    // ------------------------------------------------------------------
    // Chance action enumeration — callbacks to avoid heap allocation.
    //
    // Iteration order: suits in ascending suit_value order (0,1,2,...).
    // This matches Python's _sorted_suits which sorts by suit.value.
    // ------------------------------------------------------------------

    // Enumerate the distinct initial deals of `n` cards (n/2 per player).
    //
    // A deal is identified by the pair of hands as multisets, so each is
    // listed once, as the canonical CHANCE_DEAL (each player's cards in
    // ascending suit order; see Action::make_chance_deal).  Order: hand0 then
    // hand1, each compared as an ascending suit tuple, lexicographically.
    // For one card per player this is exactly the ordered-sequence listing.
    void iter_deal_actions(
        int n,
        const std::function<void(const Action&)>& cb) const
    {
        iter_deal_actions_with_probs(n, [&](const Action& a, double) { cb(a); });
    }

    // Deals with exact probabilities: P(one interleaved sequence), computed
    // card by card as the old sequence listing did, times the number of
    // sequences giving the same pair of hands (h!/prod k_i! per hand).
    void iter_deal_actions_with_probs(
        int n,
        const std::function<void(const Action&, double)>& cb) const
    {
        assert(n % 2 == 0 && n <= MAX_ACTION_SUITS && n <= total());
        const int h = n / 2;
        int  tmp[SUIT_CAPACITY];
        for (int i = 0; i < n_suits; i++) tmp[i] = counts[i];
        Suit hands[2][MAX_ACTION_SUITS / 2];
        enumerate_hands_(tmp, n_suits, h, hands[0], 0, 0, [&] {
            enumerate_hands_(tmp, n_suits, h, hands[1], 0, 0, [&] {
                Suit seq[MAX_ACTION_SUITS];
                for (int i = 0; i < h; i++) {
                    seq[2 * i]     = hands[0][i];
                    seq[2 * i + 1] = hands[1][i];
                }
                int    c[SUIT_CAPACITY];
                for (int i = 0; i < n_suits; i++) c[i] = counts[i];
                int    rem  = total();
                double prob = 1.0;
                for (int i = 0; i < n; i++) {
                    const int s = static_cast<int>(seq[i]);
                    prob = prob * (static_cast<double>(c[s]) / rem);
                    c[s]--;
                    rem--;
                }
                prob = prob * static_cast<double>(arrangements_(hands[0], h) * arrangements_(hands[1], h));
                cb(Action::make_chance_deal(seq, n), prob);
            });
        });
    }

    // Number of distinct initial deals of `n` cards (pairs of hands).
    long count_deals(int n) const {
        long k = 0;
        iter_deal_actions(n, [&](const Action&) { k++; });
        return k;
    }

    // Enumerate all distinct drawable suits (count > 0) for a CHANCE_DRAW.
    void iter_draw_actions(
        int player,
        const std::function<void(const Action&)>& cb) const
    {
        for (int i = 0; i < n_suits; i++) {
            if (counts[i] > 0)
                cb(Action::make_chance_draw(static_cast<Suit>(i), player));
        }
    }

    // Enumerate draw actions with exact probabilities.
    void iter_draw_actions_with_probs(
        int player,
        const std::function<void(const Action&, double)>& cb) const
    {
        int rem = total();
        assert(rem > 0);
        for (int i = 0; i < n_suits; i++) {
            if (counts[i] > 0) {
                double prob = static_cast<double>(counts[i]) / rem;
                cb(Action::make_chance_draw(static_cast<Suit>(i), player), prob);
            }
        }
    }

    // ------------------------------------------------------------------
    // Sampling
    // ------------------------------------------------------------------

    Action sample_deal(int n, std::mt19937& rng) const {
        assert(n <= total());
        int   tmp[SUIT_CAPACITY];
        for (int i = 0; i < n_suits; i++) tmp[i] = counts[i];
        Suit  seq[MAX_ACTION_SUITS];
        int   rem = total();
        for (int d = 0; d < n; d++) {
            std::uniform_int_distribution<int> dist(0, rem - 1);
            int pick = dist(rng);
            int acc  = 0;
            for (int i = 0; i < n_suits; i++) {
                acc += tmp[i];
                if (pick < acc) {
                    seq[d] = static_cast<Suit>(i);
                    tmp[i]--;
                    rem--;
                    break;
                }
            }
        }
        return Action::make_chance_deal(seq, n);
    }

    Action sample_draw(int player, std::mt19937& rng) const {
        int rem = total();
        assert(rem > 0);
        std::uniform_int_distribution<int> dist(0, rem - 1);
        int pick = dist(rng);
        int acc  = 0;
        for (int i = 0; i < n_suits; i++) {
            acc += counts[i];
            if (pick < acc)
                return Action::make_chance_draw(static_cast<Suit>(i), player);
        }
        // Unreachable if counts are non-negative and sum > 0.
        return Action::make_chance_draw(static_cast<Suit>(0), player);
    }

private:
    // Every hand of `size` cards (ascending suits, lexicographic order)
    // drawable from `counts`; counts are reduced while `cb` runs.
    template <typename F>
    static void enumerate_hands_(int counts[SUIT_CAPACITY], int n_suits, int size,
                                 Suit* hand, int depth, int min_suit, F&& cb)
    {
        if (depth == size) { cb(); return; }
        for (int i = min_suit; i < n_suits; i++) {
            if (counts[i] > 0) {
                hand[depth] = static_cast<Suit>(i);
                counts[i]--;
                enumerate_hands_(counts, n_suits, size, hand, depth + 1, i, cb);
                counts[i]++;
            }
        }
    }

    // Distinct orderings of a sorted hand: size! / prod(run length!).
    static long arrangements_(const Suit* hand, int size) noexcept {
        long num = 1, den = 1;
        int  run = 0;
        for (int i = 0; i < size; i++) {
            num *= i + 1;
            run = (i > 0 && hand[i] == hand[i - 1]) ? run + 1 : 1;
            den *= run;
        }
        return num / den;
    }
};

} // namespace hana
