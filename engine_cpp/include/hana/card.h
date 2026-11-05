#pragma once

#include "suit.h"
#include <string>

namespace hana {

// -----------------------------------------------------------------------
// Card — a single card identified solely by its suit.
// Matches Python's Card dataclass (frozen=True).
// Two cards are equal iff their suits are equal.
// -----------------------------------------------------------------------
struct Card {
    Suit suit;

    constexpr bool operator==(const Card& o) const noexcept { return suit == o.suit; }
    constexpr bool operator!=(const Card& o) const noexcept { return suit != o.suit; }
    constexpr bool operator<(const Card& o)  const noexcept { return suit < o.suit; }

    // Python repr: "Card(GEISHA_1)"
    std::string repr() const {
        return std::string("Card(") + suit_name(suit) + ")";
    }
};

} // namespace hana
