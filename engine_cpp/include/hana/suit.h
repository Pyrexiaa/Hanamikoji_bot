#pragma once

#include <cstdint>

namespace hana {

// -----------------------------------------------------------------------
// Suit enum — covers all Hanamikoji variants up to Full Hanamikoji.
// Values must match Python's Suit enum values exactly.
// tiny_hana   uses GEISHA_1..GEISHA_3 (values 0-2).
// medium_hana uses GEISHA_1..GEISHA_5 (values 0-4).
// full_hana   uses GEISHA_1..GEISHA_7 (values 0-6).
// -----------------------------------------------------------------------
enum class Suit : uint8_t {
    GEISHA_1 = 0,
    GEISHA_2 = 1,
    GEISHA_3 = 2,
    GEISHA_4 = 3,
    GEISHA_5 = 4,
    GEISHA_6 = 5,
    GEISHA_7 = 6,
};

// Maximum number of suits across all variants (Full Hanamikoji: 7 geishas).
static constexpr int SUIT_CAPACITY = 7;

// Python-compatible name strings (used in event serialization).
inline const char* suit_name(Suit s) noexcept {
    switch (s) {
        case Suit::GEISHA_1: return "GEISHA_1";
        case Suit::GEISHA_2: return "GEISHA_2";
        case Suit::GEISHA_3: return "GEISHA_3";
        case Suit::GEISHA_4: return "GEISHA_4";
        case Suit::GEISHA_5: return "GEISHA_5";
        case Suit::GEISHA_6: return "GEISHA_6";
        case Suit::GEISHA_7: return "GEISHA_7";
    }
    return "UNKNOWN";
}

// suit.value as used in Python — equals the underlying integer.
inline int suit_value(Suit s) noexcept {
    return static_cast<int>(s);
}

} // namespace hana
