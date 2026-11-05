#pragma once

#include "agent.h"
#include "constants.h"
#include "game_state.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace hana {

// -----------------------------------------------------------------------
// GameRunner — mirrors Python's engine.game.Game.
//
// Drives a full game between two agents by:
//   1. Sampling chance actions via GameState::sample_chance_action.
//   2. Querying agents for action distributions at player nodes.
//   3. Sampling from those distributions via std::discrete_distribution.
//
// Agents are held by raw (non-owning) pointer; the caller keeps them alive.
// -----------------------------------------------------------------------
class GameRunner {
public:
    GameRunner(Agent& agent0, Agent& agent1,
               bool verbose = false, uint64_t seed = 0,
               const RuntimeConfig& cfg = RuntimeConfig::tiny_hana())
        : agents_{&agent0, &agent1}
        , gs_(cfg)
        , verbose_(verbose)
        , rng_(seed)
        , game_winner_(NO_PLAYER)
    {}

    // Public single-step: execute one decision node (chance or player).
    // Used by Python bindings to drive tree traversal incrementally.
    void play_decision() { play_decision_(); }

    // Run a complete game. Returns winner (0 or 1) or NO_PLAYER for draw.
    // Mirrors Python's Game::play_game().
    int play_game(int max_rounds = 10) {
        int rounds = 0;
        while (game_winner_ == NO_PLAYER && rounds < max_rounds) {
            play_round();
            rounds++;
        }
        return game_winner_;
    }

    // Play one round from start to finish, then call update_favors().
    // Mirrors Python's Game::play_round().
    void play_round() {
        if (gs_.phase == Phase::ROUND_COMPLETE)
            gs_.begin_next_round();

        // Initial chance phase: CHOOSE_START → DEAL_INITIAL → DRAW_FOR_TURN.
        while (gs_.current_player() == CHANCE_PLAYER)
            play_decision_();

        // Main turn loop (player actions + interleaved chance draws).
        while (!gs_.is_round_complete())
            play_step_();

        gs_.update_favors();
        game_winner_ = gs_.check_winner();
    }

    int         game_winner()  const noexcept { return game_winner_; }
    const GameState& state()   const noexcept { return gs_; }

private:
    Agent*       agents_[2];
    GameState    gs_;
    bool         verbose_;
    std::mt19937 rng_;
    int          game_winner_;

    void log_(const char* /*msg*/) const noexcept { if (verbose_) { /* placeholder for future logging */ } }

    // Execute exactly one decision node (chance or player).
    void play_decision_() {
        if (gs_.current_player() == CHANCE_PLAYER) {
            const Action a = gs_.sample_chance_action(rng_);
            if (!gs_.execute_action(a))
                throw std::runtime_error("GameRunner: invalid chance action sampled");
            return;
        }

        const int pid = gs_.current_player();
        const RoundInfoState info = gs_.get_info_state(pid);
        const auto dist = agents_[pid]->get_action_distribution(info);
        const Action a  = sample_from_dist_(dist);
        if (!gs_.execute_action(a))
            throw std::runtime_error("GameRunner: agent returned action rejected by execute_action");
    }

    // One gameplay step: one player action, plus the opponent's response if
    // the action was an offer (SPLIT / GIFT / COMPETITION).  Mirrors Python's
    // Game.play_step().
    void play_step_() {
        play_decision_();
        // If an offer was just played, the opponent must immediately respond.
        if (gs_.has_pending_offer())
            play_decision_();
    }

    // Weighted sampling from an action distribution.
    // Mirrors Python's Game::sample_action_from_distribution().
    Action sample_from_dist_(const std::unordered_map<Action, double>& dist) {
        if (dist.empty())
            throw std::runtime_error("GameRunner: empty action distribution from agent");

        std::vector<Action>  actions;
        std::vector<double>  weights;
        actions.reserve(dist.size());
        weights.reserve(dist.size());
        double total = 0.0;
        for (const auto& [a, p] : dist) {
            if (p < 0.0)
                throw std::runtime_error("GameRunner: negative probability in action distribution");
            actions.push_back(a);
            weights.push_back(p);
            total += p;
        }
        if (total <= 0.0)
            throw std::runtime_error("GameRunner: action distribution has non-positive total probability");
        if (std::abs(total - 1.0) > 1e-6)
            throw std::runtime_error("GameRunner: action probabilities do not sum to 1.0");

        std::discrete_distribution<int> dd(weights.begin(), weights.end());
        return actions[static_cast<std::size_t>(dd(rng_))];
    }
};

} // namespace hana
