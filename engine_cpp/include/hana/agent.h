#pragma once

#include "action.h"
#include "info_state.h"
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace hana {

// -----------------------------------------------------------------------
// Agent ABC — mirrors Python's engine.agents.Agent.
//
// Agents receive a RoundInfoState (their private view of the game) and
// return a mixed strategy: a map from legal Action to probability.
//
// Constraints the game runner enforces at runtime:
//   - Distribution must be non-empty.
//   - All probabilities must be non-negative.
//   - Probabilities must sum to 1.0 within 1e-6 tolerance.
// -----------------------------------------------------------------------
class Agent {
public:
    virtual ~Agent() = default;

    virtual std::unordered_map<Action, double>
    get_action_distribution(const RoundInfoState& info_state) = 0;
};

// -----------------------------------------------------------------------
// RandomAgent — uniform distribution over legal actions.
// Mirrors Python's engine.agents.RandomAgent.
// -----------------------------------------------------------------------
class RandomAgent final : public Agent {
public:
    std::unordered_map<Action, double>
    get_action_distribution(const RoundInfoState& info_state) override {
        std::vector<Action> actions;
        info_state.get_legal_actions([&](const Action& a) { actions.push_back(a); });

        if (actions.empty())
            throw std::runtime_error("RandomAgent: no legal actions in info state");

        const double p = 1.0 / static_cast<double>(actions.size());
        std::unordered_map<Action, double> dist;
        dist.reserve(actions.size());
        for (const Action& a : actions) dist[a] = p;
        return dist;
    }
};

} // namespace hana
