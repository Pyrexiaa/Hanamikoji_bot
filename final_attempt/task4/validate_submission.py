"""Independently exercise the actual serialized Task 3 strategies.

Run from the repository root with ``python final_attempt/task4/validate_submission.py``.
The checks use the supplied unmodified treeplex, exhaustive Python-engine payoff
enumeration, exact best responses, and 100 reproducible single-round samples.
"""

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Keep this attempt's implementations ahead of the shared game engine.
REPO = next((path for path in ROOT.parents if (path / "engine").is_dir()), ROOT.parent)
sys.path[:0] = [str(ROOT / "student_attempts"), str(ROOT), str(REPO)]

# Serialization is graded with the Python engine, regardless of shell settings.
os.environ["HANA_ENGINE"] = "python"

from env_config import env, PLAYER_1, PLAYER_2
from solver import Solver
from treeplex_representation import (
    TreeplexGame,
    convert_behavioral_to_sequence_form_strategy,
)
from utility import compute_agent_payoffs
from tiny_hana.example_agents import GreedyAgent, CardCountingAgent


TREEPLEX_SHA256 = "6b59cacf00e51e5b46f63995b139f910837f371b52f0740fb6836f49f652c626"


def flow_error(sequence, treeplex):
    errors = [abs(float(sequence[0]) - 1.0)]
    for start, end, parent in zip(treeplex.info_state_start_seq,
                                  treeplex.info_state_end_seq,
                                  treeplex.info_state_parent_seq):
        errors.append(abs(float(sequence[start:end].sum() - sequence[parent])))
    return max(errors)


def validate_loaded_agent(game, path, player):
    treeplex = game.treeplex_p1 if player == PLAYER_1 else game.treeplex_p2
    agent = game.load_agent(str(path))
    with np.load(path, allow_pickle=False) as data:
        raw = data["strategy"].copy()
        saved_player = int(data["player_id"][0])
    assert saved_player == player, (path, "wrong player id")
    assert raw.ndim == 1 and raw.size == treeplex.num_sequences
    assert np.all(np.isfinite(raw)) and np.all(raw >= 0.0)
    assert np.all(raw <= 1.0 + 1e-12)
    normalization_errors = [
        abs(float(raw[start:end].sum()) - 1.0)
        for start, end in zip(treeplex.info_state_start_seq, treeplex.info_state_end_seq)
    ]
    assert max(normalization_errors) < 1e-10
    behavioral = game.agent_to_behavioral_strategy(agent, player)
    # Entry zero is a dummy root; only actual action entries are serialized.
    assert np.array_equal(behavioral[1:], raw[1:])
    sequence = convert_behavioral_to_sequence_form_strategy(
        behavioral, treeplex, in_place=False)
    assert np.all(np.isfinite(sequence)) and np.all(sequence >= 0.0)
    error = flow_error(sequence, treeplex)
    assert error < 1e-10
    metadata = {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "player_id": player,
        "infosets_checked": treeplex.get_num_infosets(),
        "num_sequences": treeplex.num_sequences,
        "minimum_action_probability": float(raw[1:].min()),
        "maximum_action_probability": float(raw[1:].max()),
        "maximum_behavioral_normalization_error": max(normalization_errors),
        "maximum_sequence_flow_error": error,
    }
    return agent, behavioral, sequence, metadata


def main():
    original_hash = hashlib.sha256(
        (ROOT / "student_attempts" / "treeplex_representation.py").read_bytes()).hexdigest()
    assert original_hash == TREEPLEX_SHA256, "Provided treeplex representation changed"
    game = TreeplexGame()
    solver = Solver(game)
    a1, b1, x, m1 = validate_loaded_agent(game, ROOT / "tasks" / "nash_p1.npz", PLAYER_1)
    a2, b2, y, m2 = validate_loaded_agent(game, ROOT / "tasks" / "nash_p2.npz", PLAYER_2)
    comparisons = []

    def compare(label, p1, p2, seq1=None, seq2=None):
        if seq1 is None:
            seq1 = convert_behavioral_to_sequence_form_strategy(
                game.agent_to_behavioral_strategy(p1, PLAYER_1), game.treeplex_p1, in_place=False)
        if seq2 is None:
            seq2 = convert_behavioral_to_sequence_form_strategy(
                game.agent_to_behavioral_strategy(p2, PLAYER_2), game.treeplex_p2, in_place=False)
        represented = solver.compute_payoffs_from_seq_form_strategies(seq1, seq2)
        enumerated = compute_agent_payoffs(env.GameState(), p1, p2)
        error = abs(represented - enumerated)
        assert error < 1e-10, (label, represented, enumerated)
        comparisons.append({
            "profile": label,
            "sequence_form_payoff_p1": represented,
            "exhaustive_engine_payoff_p1": enumerated,
            "absolute_difference": error,
        })
        return represented

    profile_value = compare("loaded P1 vs loaded P2", a1, a2, x, y)
    br1_behavioral = solver.compute_best_response(PLAYER_1, y)
    br2_behavioral = solver.compute_best_response(PLAYER_2, x)
    br1_sequence = convert_behavioral_to_sequence_form_strategy(
        br1_behavioral, game.treeplex_p1, in_place=False)
    br2_sequence = convert_behavioral_to_sequence_form_strategy(
        br2_behavioral, game.treeplex_p2, in_place=False)
    upper = compare("best response P1 vs loaded P2",
                    game.behavioral_strategy_to_agent(br1_behavioral, PLAYER_1), a2,
                    br1_sequence, y)
    lower = compare("loaded P1 vs best response P2", a1,
                    game.behavioral_strategy_to_agent(br2_behavioral, PLAYER_2),
                    x, br2_sequence)
    spr = solver.saddle_point_residual(x, y)
    assert lower <= profile_value + 1e-12 <= upper + 2e-12
    assert abs(spr - (upper - lower)) < 1e-12

    heuristic_results = []
    for agent_class in (env.RandomAgent, GreedyAgent, CardCountingAgent):
        name = agent_class.__name__
        as_p1 = compare(f"loaded P1 vs {name} P2", a1, agent_class(), seq1=x)
        as_p2 = compare(f"{name} P1 vs loaded P2", agent_class(), a2, seq2=y)
        heuristic_results.append({
            "opponent": name,
            "loaded_player1_expected_payoff": as_p1,
            "loaded_player2_expected_payoff": -as_p2,
            "mean_payoff_across_seats": (as_p1 - as_p2) / 2.0,
        })

    outcomes = {"player1_wins": 0, "player2_wins": 0, "ties": 0}
    for seed in range(100):
        runner = env.Game(a1, a2, verbose=False, seed=seed)
        runner.play_round()
        assert runner.state.is_round_complete()
        outcome = runner.state.winner
        assert outcome in (PLAYER_1, PLAYER_2, None)
        outcomes[{PLAYER_1: "player1_wins", PLAYER_2: "player2_wins", None: "ties"}[outcome]] += 1

    result = {
        "engine": "python",
        "python": sys.version,
        "numpy": np.__version__,
        "treeplex_original_sha256": original_hash,
        "treeplex_unchanged": True,
        "strategies": [m1, m2],
        "loaded_profile_payoff_p1": profile_value,
        "exact_best_response_lower_bound": lower,
        "exact_best_response_upper_bound": upper,
        "saddle_point_residual": spr,
        "p1_unilateral_improvement": upper - profile_value,
        "p2_unilateral_improvement": profile_value - lower,
        "engine_payoff_comparisons": comparisons,
        "maximum_engine_representation_difference": max(r["absolute_difference"] for r in comparisons),
        "heuristic_expected_payoffs": heuristic_results,
        "seeded_single_rounds": {"count": 100, "seeds": "0 through 99", **outcomes},
        "all_checks_passed": True,
    }
    output = ROOT / "task4" / "results" / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "spr": spr,
        "bounds": [lower, upper],
        "profile_payoff": profile_value,
        "maximum_engine_representation_difference": result["maximum_engine_representation_difference"],
        "seeded_rounds": outcomes,
        "all_checks_passed": True,
    }, indent=2))


if __name__ == "__main__":
    main()
