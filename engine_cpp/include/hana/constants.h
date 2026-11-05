#pragma once

namespace hana {

static constexpr int CHANCE_PLAYER = -1;
static constexpr int NO_PLAYER     = -2;   // terminal / round-complete sentinel

// Maximum events stored per player per round.  Full Hanamikoji needs 18:
// round start + 4 draws + 4 own actions + 4 opponent actions + 4 offer
// choices + round end.
static constexpr int MAX_EVENTS_PER_ROUND = 24;

// Per-player card capacities, sized for full Hanamikoji (the largest variant).
static constexpr int MAX_HAND_SIZE   = 8;  // 6 dealt + 1 drawn = 7 at most
static constexpr int MAX_COLLECTED   = 8;  // 7 from offers + the secret card at round end
static constexpr int MAX_OFFER_CARDS = 4;  // COMPETITION offers two pairs

} // namespace hana
