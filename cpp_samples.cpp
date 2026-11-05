/**
 * cpp_samples.cpp — Native C++ equivalent of engine_samples.py.
 *
 * Demonstrates the hana engine C++ API directly, without the Python bindings.
 * Sections mirror engine_samples.py one-to-one:
 *
 *   Section 1 — tiny_hana, GameRunner (high-level; like Python's Game.play_game())
 *   Section 2 — tiny_hana, manual GameState loop (solver / tree-search style)
 *   Section 3 — medium_hana, GameRunner (high-level)
 *   Section 4 — medium_hana, manual GameState loop with all solver extras:
 *                 get_chance_actions_with_probs, to_key + undo_move,
 *                 preview_round_resolution, info state inspection
 *
 * The custom bot (ReserveHighestRankAgent) is defined once and reused across
 * all four sections — exactly as in the Python version.
 *
 * Compile (header-only engine, standalone):
 *   g++ -std=c++17 -I engine_cpp/include -o cpp_samples cpp_samples.cpp
 *
 * Expected output ends with: "All samples completed successfully."
 */

#include <cassert>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "hana/action.h"
#include "hana/agent.h"
#include "hana/constants.h"
#include "hana/game_config.h"
#include "hana/game_runner.h"
#include "hana/game_state.h"
#include "hana/info_state.h"
#include "hana/suit.h"

using namespace hana;

// ============================================================================
// ReserveHighestRankAgent
//
// Custom agent: always plays RESERVE on the highest-value geisha in hand.
// Falls back to uniform random if no RESERVE is available (e.g., already used
// or the variant doesn't allow it).
//
// Direct C++ equivalent of the Python ReserveHighestRankAgent in engine_samples.py.
//
// Note how agent logic is identical across variants (tiny and medium) — the
// RoundInfoState API abstracts away all variant-specific details.
// ============================================================================
class ReserveHighestRankAgent final : public Agent {
public:
    std::unordered_map<Action, double>
    get_action_distribution(const RoundInfoState& info) override {
        // Collect all legal actions via the callback API.
        // The lambda is called once per legal action; no heap allocation needed.
        std::vector<Action> legal;
        info.get_legal_actions([&](const Action& a) { legal.push_back(a); });

        assert(!legal.empty() && "Agent called with no legal actions");

        // Find the best RESERVE action: highest suit value in hand.
        const Action* best_reserve = nullptr;
        int           best_value   = -1;
        for (const Action& a : legal) {
            if (a.action_type == ActionType::RESERVE) {
                int v = suit_value(a.suits[0]);
                if (v > best_value) {
                    best_value   = v;
                    best_reserve = &a;
                }
            }
        }

        std::unordered_map<Action, double> dist;
        if (best_reserve) {
            // Pure strategy: put all weight on reserving the best geisha.
            dist[*best_reserve] = 1.0;
        } else {
            // Fallback: uniform over all legal actions.
            const double p = 1.0 / static_cast<double>(legal.size());
            for (const Action& a : legal) dist[a] = p;
        }
        return dist;
    }
};


// ============================================================================
// Helpers
// ============================================================================

// Print a RoundInfoState: player, round, hand, favors, used actions.
static void print_info_state(const RoundInfoState& info) {
    printf("  RoundInfoState p%d r%d\n", info.player_id, info.round_number);

    // Current hand and used-action bitmask.
    const HandAndActions ha = info.get_current_hand_and_actions();
    printf("    hand (%d cards):", ha.hand_count);
    for (int i = 0; i < ha.hand_count; i++)
        printf(" %s", suit_name(ha.hand[i]));
    printf("\n");

    // Favor array: -1 = player 0 controls, 0 = neutral, 1 = player 1 controls.
    int8_t favors[SUIT_CAPACITY];
    info.get_favors(favors);
    printf("    favors:");
    for (int i = 0; i < info.num_suits; i++)
        printf(" %s=%+d", suit_name(static_cast<Suit>(i)), static_cast<int>(favors[i]));
    printf("\n");

    // Legal actions (count only — full list gets verbose fast).
    int action_count = 0;
    info.get_legal_actions([&](const Action&) { action_count++; });
    printf("    legal_actions: %d\n", action_count);

    // Policy table key — this is the canonical string used by PolicyTableAgent.
    printf("    key_prefix: %.60s...\n", info.to_string().c_str());
}

// Manual game loop: drives a full game using GameState directly.
// Mirrors run_low_level_game() from engine_samples.py.
// Returns the winner (0 or 1).
static int run_manual_game(const RuntimeConfig& cfg,
                           Agent& agent0, Agent& agent1,
                           uint64_t seed,
                           bool inspect_first_decision = false)
{
    std::mt19937 rng(seed);

    GameState gs(cfg);
    Agent*    agents[2] = {&agent0, &agent1};

    bool inspected = false;

    while (true) {
        const int cp = gs.current_player();

        // ---- Chance node ----
        if (cp == CHANCE_PLAYER) {
            const Action a = gs.sample_chance_action(rng);
            gs.execute_action(a);
            continue;
        }

        // ---- Round complete ----
        if (cp == NO_PLAYER) {
            gs.update_favors();
            const int w = gs.check_winner();
            if (w != NO_PLAYER) return w;
            gs.begin_next_round();
            continue;
        }

        // ---- Optional: inspect the info state once ----
        if (inspect_first_decision && !inspected) {
            inspected = true;
            const RoundInfoState info = gs.get_info_state(cp);
            print_info_state(info);
        }

        // ---- Player decision ----
        const RoundInfoState info = gs.get_info_state(cp);
        const auto dist = agents[cp]->get_action_distribution(info);

        // Sample from distribution using std::discrete_distribution.
        std::vector<Action>  acts;
        std::vector<double>  wts;
        acts.reserve(dist.size());
        wts.reserve(dist.size());
        for (const auto& [a, p] : dist) { acts.push_back(a); wts.push_back(p); }
        std::discrete_distribution<int> dd(wts.begin(), wts.end());
        const Action& chosen = acts[static_cast<std::size_t>(dd(rng))];

        gs.execute_action(chosen);
    }
}


// ============================================================================
// Section 1 — tiny_hana, GameRunner (high-level)
// ============================================================================
static void section1_tiny_gamerunner() {
    printf("=== Section 1: tiny_hana via GameRunner ===\n");

    // GameRunner mirrors Python's Game class:
    //   Game(agent0, agent1, verbose=False, seed=42)
    // Agents are held by non-owning pointer; they must outlive the runner.
    ReserveHighestRankAgent bot0, bot1;
    GameRunner runner(bot0, bot1,
                      /*verbose=*/false,
                      /*seed=*/42,
                      RuntimeConfig::tiny_hana());

    const int winner = runner.play_game();
    assert(winner == 0 || winner == 1);
    printf("  Winner: Player %d\n", winner);

    // Seeded games are deterministic.
    ReserveHighestRankAgent bot2, bot3;
    GameRunner runner2(bot2, bot3, false, 42, RuntimeConfig::tiny_hana());
    assert(runner2.play_game() == winner);
    printf("  Seeded reproducibility: OK\n");

    // One random agent vs one strategy agent.
    RandomAgent rand0;
    ReserveHighestRankAgent strat1;
    int wins[2] = {0, 0};
    for (uint64_t s = 0; s < 100; s++) {
        GameRunner r(rand0, strat1, false, s, RuntimeConfig::tiny_hana());
        wins[r.play_game()]++;
    }
    printf("  100 games (Random vs Strategy): P0=%d P1=%d\n", wins[0], wins[1]);
    printf("\n");
}


// ============================================================================
// Section 2 — tiny_hana, manual GameState loop (solver style)
// ============================================================================
static void section2_tiny_manual() {
    printf("=== Section 2: tiny_hana, manual GameState loop ===\n");

    ReserveHighestRankAgent bot0, bot1;

    // Inspect the first player decision node.
    printf("  First decision node inspection:\n");
    run_manual_game(RuntimeConfig::tiny_hana(), bot0, bot1,
                    /*seed=*/7, /*inspect=*/true);

    // Solver extra: undo_move + to_key
    // Demonstrate that executing then undoing an action is an exact identity.
    // This is the core of MCTS tree traversal and counterfactual regret solvers.
    {
        GameState gs(RuntimeConfig::tiny_hana());

        // Advance past chance nodes to the first player decision.
        std::mt19937 rng(99);
        while (gs.current_player() == CHANCE_PLAYER)
            gs.execute_action(gs.sample_chance_action(rng));

        const int cp = gs.current_player();
        assert(cp == 0 || cp == 1);

        // Snapshot the state key before trying each legal action.
        const GameState::Key key_before = gs.to_key();

        int checked = 0;
        gs.get_legal_actions([&](const Action& a) {
            gs.execute_action(a);
            gs.undo_move();
            assert(gs.to_key() == key_before && "undo_move must restore exact state");
            checked++;
        });
        printf("  undo_move roundtrip: %d actions checked, all OK\n", checked);
    }

    // Solver extra: get_chance_actions_with_probs
    // Enumerate chance outcomes with their exact probabilities.
    // Useful for building the full game tree (expectimax / CFR).
    {
        GameState gs(RuntimeConfig::tiny_hana());
        assert(gs.current_player() == CHANCE_PLAYER);

        double prob_sum = 0.0;
        int    count    = 0;
        gs.get_chance_actions_with_probs([&](const Action& a, double p) {
            (void)a;
            prob_sum += p;
            count++;
        });
        // CHOOSE_START: two outcomes (p0 starts, p1 starts) each with prob=0.5.
        assert(count == 2);
        assert(prob_sum > 0.999 && prob_sum < 1.001);
        printf("  get_chance_actions_with_probs: %d outcomes, sum=%.3f\n",
               count, prob_sum);
    }

    // Run 50 games and verify all produce a winner.
    int wins[2] = {0, 0};
    for (uint64_t s = 0; s < 50; s++) {
        int w = run_manual_game(RuntimeConfig::tiny_hana(), bot0, bot1, s);
        assert(w == 0 || w == 1);
        wins[w]++;
    }
    printf("  50 manual games: P0=%d P1=%d\n", wins[0], wins[1]);
    printf("\n");
}


// ============================================================================
// Section 3 — medium_hana, GameRunner (high-level)
// ============================================================================
static void section3_medium_gamerunner() {
    printf("=== Section 3: medium_hana via GameRunner ===\n");

    // medium_hana adds DISCARD (2 cards hidden/revealed) to RESERVE + SPLIT.
    // ReserveHighestRankAgent still works: it prefers reserving the most
    // valuable geisha (rank 4 = GEISHA_5), and falls back to uniform random
    // for DISCARD and SPLIT.
    ReserveHighestRankAgent bot0, bot1;
    GameRunner runner(bot0, bot1,
                      /*verbose=*/false,
                      /*seed=*/42,
                      RuntimeConfig::medium_hana());

    const int winner = runner.play_game();
    assert(winner == 0 || winner == 1);
    printf("  Winner: Player %d\n", winner);

    // Seeded determinism.
    ReserveHighestRankAgent bot2, bot3;
    GameRunner runner2(bot2, bot3, false, 42, RuntimeConfig::medium_hana());
    assert(runner2.play_game() == winner);
    printf("  Seeded reproducibility: OK\n");

    // Sanity: 100 medium_hana games all terminate.
    int wins[2] = {0, 0};
    for (uint64_t s = 0; s < 100; s++) {
        ReserveHighestRankAgent a0, a1;
        GameRunner r(a0, a1, false, s, RuntimeConfig::medium_hana());
        wins[r.play_game()]++;
    }
    printf("  100 medium_hana games: P0=%d P1=%d\n", wins[0], wins[1]);
    printf("\n");
}


// ============================================================================
// Section 4 — medium_hana, manual loop with all solver extras
// ============================================================================
static void section4_medium_manual() {
    printf("=== Section 4: medium_hana, manual loop + solver extras ===\n");

    ReserveHighestRankAgent bot0, bot1;

    // 4a. Info state inspection at first DISCARD opportunity.
    printf("  Inspecting first player decision (medium_hana):\n");
    run_manual_game(RuntimeConfig::medium_hana(), bot0, bot1,
                    /*seed=*/7, /*inspect=*/true);

    // 4b. Direct GameState construction — bypass the Python shim entirely.
    //     RuntimeConfig::medium_hana() selects the 5-suit variant at runtime.
    {
        GameState gs(RuntimeConfig::medium_hana());

        // Drive through chance nodes: CHOOSE_START → DEAL_INITIAL → DRAW_FOR_TURN.
        std::mt19937 rng(0);
        while (gs.current_player() == CHANCE_PLAYER)
            gs.execute_action(gs.sample_chance_action(rng));

        // Snapshot and verify DISCARD is available.
        const int cp = gs.current_player();
        const RoundInfoState info = gs.get_info_state(cp);
        bool found_discard = false;
        info.get_legal_actions([&](const Action& a) {
            if (a.action_type == ActionType::DISCARD) found_discard = true;
        });
        assert(found_discard && "DISCARD must be legal after the initial deal in medium_hana");
        printf("  DISCARD in legal actions after initial deal: OK\n");

        // 4c. undo_move: snapshot → execute DISCARD → undo → compare key.
        const GameState::Key key_before = gs.to_key();
        Action discard_action = Action::make_chance_start(0); // placeholder
        info.get_legal_actions([&](const Action& a) {
            if (a.action_type == ActionType::DISCARD && discard_action.action_type != ActionType::DISCARD)
                discard_action = a;
        });
        gs.execute_action(discard_action);
        gs.undo_move();
        assert(gs.to_key() == key_before);
        printf("  undo_move roundtrip (DISCARD): OK\n");

        // 4d. get_chance_actions_with_probs during DEAL_INITIAL.
        //     Reset to DEAL_INITIAL by constructing a fresh state.
        GameState gs2(RuntimeConfig::medium_hana());
        gs2.execute_action(Action::make_chance_start(0)); // advance to DEAL_INITIAL

        assert(gs2.current_player() == CHANCE_PLAYER);
        double sum = 0.0;
        int    n   = 0;
        gs2.get_chance_actions_with_probs([&](const Action&, double p) {
            sum += p;
            n++;
        });
        assert(n > 0 && sum > 0.999 && sum < 1.001);
        printf("  get_chance_actions_with_probs (DEAL_INITIAL): %d outcomes, sum=%.6f\n",
               n, sum);
    }

    // 4e. preview_round_resolution — inspect the expected end-of-round winner
    //     *before* calling update_favors() (non-destructive).
    //     Verify: favor-points threshold (≥6) takes priority over geisha count (≥3).
    {
        GameState gs(RuntimeConfig::medium_hana());

        // Play a full round with the first legal action at every step.
        std::mt19937 rng(17);
        while (!gs.is_round_complete()) {
            if (gs.current_player() == CHANCE_PLAYER)
                gs.execute_action(gs.sample_chance_action(rng));
            else {
                std::vector<Action> legal;
                gs.get_legal_actions([&](const Action& a) { legal.push_back(a); });
                gs.execute_action(legal[0]);
            }
        }

        const RoundResolution res = gs.preview_round_resolution();
        printf("  preview_round_resolution:\n");
        printf("    winner: ");
        if (res.winner == static_cast<int8_t>(NO_PLAYER))
            printf("none (no winner yet)\n");
        else
            printf("Player %d\n", static_cast<int>(res.winner));

        printf("    final_favors:");
        for (int i = 0; i < res.num_suits; i++)
            printf(" %s=%+d", suit_name(static_cast<Suit>(i)),
                   static_cast<int>(res.final_favors[i]));
        printf("\n");

        // Compute expected winner independently and cross-check.
        // suit_ranks for medium_hana: [1, 1, 2, 3, 4].
        const int suit_ranks[5] = {1, 1, 2, 3, 4};
        int pts[2] = {0, 0};
        int cnt[2] = {0, 0};
        for (int i = 0; i < res.num_suits; i++) {
            if      (res.final_favors[i] == -1) { pts[0] += suit_ranks[i]; cnt[0]++; }
            else if (res.final_favors[i] ==  1) { pts[1] += suit_ranks[i]; cnt[1]++; }
        }
        int expected_winner = NO_PLAYER;
        if      (pts[0] >= 6) expected_winner = 0;
        else if (pts[1] >= 6) expected_winner = 1;
        else if (cnt[0] >= 3) expected_winner = 0;
        else if (cnt[1] >= 3) expected_winner = 1;

        printf("    computed pts=[%d,%d] cnt=[%d,%d] → expected winner: ",
               pts[0], pts[1], cnt[0], cnt[1]);
        if (expected_winner == NO_PLAYER) printf("none\n");
        else printf("Player %d\n", expected_winner);

        assert(static_cast<int>(res.winner) == expected_winner);
        printf("  preview_round_resolution matches independent calculation: OK\n");
    }

    // 4f. Transposition table demo using info state string keys.
    //     The policy table key (info.to_string()) identifies unique information
    //     sets. A simple solver can memoize values in an unordered_map.
    {
        std::unordered_map<std::string, int> visit_count;

        GameState gs(RuntimeConfig::medium_hana());
        std::mt19937 rng(3);

        for (int game = 0; game < 20; game++) {
            gs.init(RuntimeConfig::medium_hana());

            while (true) {
                const int cp = gs.current_player();
                if (cp == CHANCE_PLAYER) {
                    gs.execute_action(gs.sample_chance_action(rng));
                    continue;
                }
                if (cp == NO_PLAYER) {
                    gs.update_favors();
                    if (gs.check_winner() != NO_PLAYER) break;
                    gs.begin_next_round();
                    continue;
                }

                const RoundInfoState info = gs.get_info_state(cp);
                const std::string    key  = info.to_string();
                visit_count[key]++;

                std::vector<Action> legal;
                info.get_legal_actions([&](const Action& a) { legal.push_back(a); });
                std::uniform_int_distribution<int> pick(0, static_cast<int>(legal.size()) - 1);
                gs.execute_action(legal[static_cast<std::size_t>(pick(rng))]);
            }
        }

        printf("  Transposition table after 20 games: %zu unique info states visited\n",
               visit_count.size());
        // Some states are visited more than once across different games.
        int revisits = 0;
        for (const auto& [k, v] : visit_count)
            if (v > 1) revisits++;
        printf("  Info states visited >1 time: %d\n", revisits);
    }

    printf("\n");
}


// ============================================================================
// main
// ============================================================================
int main() {
    printf("hana engine — native C++ samples\n");
    printf("=================================\n\n");

    section1_tiny_gamerunner();
    section2_tiny_manual();
    section3_medium_gamerunner();
    section4_medium_manual();

    printf("All samples completed successfully.\n");
    return 0;
}
