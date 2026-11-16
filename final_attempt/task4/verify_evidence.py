"""Check preserved experiments and replay short prefixes without replacing measurements.

Run from the repository root. Recorded paths are normalized across operating
systems. Results are printed to the console; all saved evidence is read-only.
"""
import ast
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import run_cfr_ablation as ablation
import run_experiments as experiment
from lp_solver import solve_sequence_form_lp
from solver import Solver
from validate_submission import validate_loaded_agent

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "task4" / "results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_path(base, name):
    relative = Path(name.replace("\\", "/"))
    assert not relative.is_absolute() and ".." not in relative.parts, name
    return base / relative


def check_sources(sources):
    for name, expected in sources.items():
        assert digest(recorded_path(ROOT, name)) == expected, name
    return len(sources)


def lp_certificate(game, reference_value):
    lp = solve_sequence_form_lp(game)
    x, y = lp["sol_seq_form_p1"], lp["sol_seq_form_p2"]
    spr = Solver(game).saddle_point_residual(x, y)
    assert lp["status"] == "optimal" and abs(spr) < 1e-8
    assert abs(lp["value"] - reference_value) < 1e-9
    for field in ("feasibility_p1", "feasibility_p2", "dual_feasibility_p1",
                  "dual_feasibility_p2", "optimality_p1", "optimality_p2"):
        assert lp[field] < 1e-8, (field, lp[field])
    return {k: v for k, v in lp.items() if not isinstance(v, np.ndarray)} | {"spr": spr}


def check_interfaces():
    names = {
        "utility.py": {"compute_posterior_given_opponent_agent", "compute_agent_payoffs"},
        "solver.py": {"compute_payoffs_from_seq_form_strategies", "compute_reward_vector",
                      "compute_best_response", "saddle_point_residual"},
    }
    for filename, required in names.items():
        def interfaces(base):
            tree = ast.parse((base / "student_attempts" / filename).read_text(encoding="utf-8"))
            return {node.name: ast.dump(node.args) for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name in required}
        assert interfaces(ROOT) == interfaces(ROOT.parent), filename
    return sorted(set.union(*names.values()))


def check_experiment_a():
    data = json.loads((RESULTS / "experiments.json").read_text(encoding="utf-8"))
    config = data["configuration"]
    assert (config["iterations"], config["repeats"], config["warmup_iterations"]) == (20000, 3, 2)
    assert config["full_budget"] and config["spr_threshold"] == 0.002
    assert set(data["scenarios"]) == {"neutral", "first_player", "second_player", "favored_g3"}
    count, total = 0, 0
    report = {"source_hashes": check_sources(data["environment"]["source_sha256"]), "scenarios": {}}
    for name, scenario in data["scenarios"].items():
        game = experiment.scenario(name)
        assert scenario["sequences"] == [game.treeplex_p1.num_sequences, game.treeplex_p2.num_sequences]
        assert scenario["infosets"] == [game.treeplex_p1.get_num_infosets(), game.treeplex_p2.get_num_infosets()]
        assert scenario["payoff_entries"] == len(game.seq_pair_to_leaves)
        assert scenario["nonzero_payoffs"] == sum(v != 0 for v in game.seq_pair_to_leaves.values())
        details = {k: scenario[k] for k in ("sequences", "infosets", "payoff_entries")}
        details["lp"] = lp_certificate(game, scenario["lp"][0]["value"])
        details["methods"] = {}
        assert set(scenario["methods"]) == set(experiment.METHODS)
        for method, runs in scenario["methods"].items():
            assert [run["repeat"] for run in runs] == [0, 1, 2]
            for run in runs:
                path = recorded_path(RESULTS, run["history_file"])
                assert path.name == f"{name}_{experiment.METHOD_SLUGS[method]}_r{run['repeat'] + 1}.jsonl.gz"
                assert digest(path) == run["history_sha256"], path
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    history = [json.loads(line) for line in handle]
                cadence = 25 if method == "CFR" else 1
                assert run["history_cadence"] == cadence
                assert [h["iteration"] for h in history] == list(range(0, 20001, cadence))
                assert len(history) == run["history_records"]
                assert run["iterations"] == run["requested_iterations"] == 20000
                assert run["full_budget"] and run["completed_full_budget"]
                assert history[-1]["spr"] == run["spr"]
                # Oracle certificates and separately summed payoffs can differ by roundoff.
                assert abs(run["spr"] - run["independent_spr"]) < 1e-10
                assert history[-1]["elapsed_seconds"] <= run["wall_seconds"]
                assert all(a["elapsed_seconds"] <= b["elapsed_seconds"] for a, b in zip(history, history[1:]))
                for h in history:
                    assert all(np.isfinite(h[k]) for k in ("spr", "value", "elapsed_seconds"))
                    assert h["spr"] >= -1e-10
                    assert abs(h["elapsed_seconds"] - h["solve_elapsed_seconds"] - run["constructor_seconds"]) < 1e-10
                first = next((h for h in history if h["spr"] <= config["spr_threshold"]), None)
                assert first == run["first_target"]
                by_iteration = {h["iteration"]: h for h in history}
                for h in run["history"] + run["summary_checkpoints"]:
                    assert h == by_iteration[h["iteration"]]
                count += 1
                total += len(history)
            # A short full-budget replay verifies the executable path, without new timing claims.
            learner = experiment.METHODS[method](game, check_every=experiment.history_cadence(method, config))
            x, y = learner.solve(max_iter=200, spr_threshold=None)
            target = next(h for h in runs[0]["history"] if h["iteration"] == 200)
            assert learner.iterations == 200
            assert abs(Solver(game).saddle_point_residual(x, y) - target["spr"]) < 1e-9
            details["methods"][method] = {
                "final_spr": runs[0]["spr"], "first_target_iteration": runs[0]["first_target"]["iteration"],
                "median_wall_seconds": float(np.median([r["wall_seconds"] for r in runs])),
                "replay_iterations": 200, "replay_spr": float(learner.spr),
            }
            if method == "Double oracle":
                first_precise = next(h for h in learner.history if h["spr"] <= 1e-9)
                exact = data["analysis"]["double_oracle_exact"][name]
                recorded_first = next(h for h in exact["history"] if h["spr"] <= 1e-9)
                assert recorded_first["iteration"] == exact["iterations"]
                assert recorded_first["spr"] == exact["spr"]
                assert abs(first_precise["spr"]) < 1e-9
                assert abs(sum(row[2] for row in exact["restricted_lp_trace"])
                           - exact["restricted_lp_seconds"]) < 1e-10
                details["methods"][method].update(
                    recorded_first_strict_target_iteration=exact["iterations"],
                    fresh_first_strict_target_iteration=first_precise["iteration"],
                    fresh_first_strict_target_spr=first_precise["spr"],
                    fresh_first_target_iteration=next(h["iteration"] for h in learner.history if h["spr"] <= 0.002),
                )
            if name == "neutral" and method in ("CFR", "Fictitious play"):
                # The subsidiary rate study logs both methods every 25 iterations,
                # although main FP histories retain every iteration.
                samples = [h for h in history if h["iteration"] >= 100 and h["spr"] > 0
                           and h["iteration"] % config["check_every"] == 0]
                slope = float(np.polyfit(np.log([h["iteration"] for h in samples]),
                                        np.log([h["spr"] for h in samples]), 1)[0])
                assert abs(slope - data["analysis"]["rates"][method]["slope_100_to_end"]) < 1e-10
        if name in data["analysis"]["lp_versus_cfr"]:
            fresh = experiment.lp_versus_cfr(game, config)
            recorded = data["analysis"]["lp_versus_cfr"][name]
            assert fresh["cfr_iterations"] == recorded["cfr_iterations"]
            for metric in ("cfr_spr", "cross_spr_lpP1_cfrP2", "cross_spr_cfrP1_lpP2"):
                assert abs(fresh[metric] - recorded[metric]) < 1e-9
            for player in ("p1", "p2"):
                assert fresh[player] == recorded[player]
            for metric in ("optimal_min", "optimal_max"):
                assert abs(fresh["nonunique"][metric] - recorded["nonunique"][metric]) < 1e-8
            details["lp_versus_cfr"] = fresh
        report["scenarios"][name] = details
    assert count == 36
    report.update(histories=count, observations=total)
    return report


def check_experiment_b():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["solver_source"] == "student_attempts/solver.py"
    assert summary["lp_source"] == "student_attempts/lp_solver.py"
    assert (summary["iterations"], summary["repeats"], summary["warmup_iterations"]) == (20000, 3, 2)
    assert summary["methods"] == ablation.METHODS
    assert summary["checkpoints"] == ablation.DEFAULT_CHECKPOINTS
    assert len(summary["checkpoints"]) == 14
    with (RESULTS / "convergence.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 630
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["method"], int(row["repeat"]))].append(row)
        lower, upper, spr, value = (float(row[k]) for k in ("lower", "upper", "spr", "value"))
        assert all(np.isfinite(float(v)) for k, v in row.items() if k not in ("case", "method"))
        assert abs(spr - (upper - lower)) < 1e-12 and lower - 1e-12 <= value <= upper + 1e-12
        assert lower <= summary["games"][row["case"]]["lp_upper"] + 1e-8
        assert upper >= summary["games"][row["case"]]["lp_lower"] - 1e-8
    assert len(groups) == 45
    for group in groups.values():
        assert [int(row["iteration"]) for row in group] == summary["checkpoints"]
        assert all(float(a["elapsed_solve_seconds"]) <= float(b["elapsed_solve_seconds"])
                   for a, b in zip(group, group[1:]))
    report = {"source_hashes": check_sources(summary["source_sha256"]), "observations": len(rows), "scenarios": {}}
    for case in ablation.CASES:
        game = ablation.make_game(case)
        reference = summary["games"][case]
        assert reference["sequences"] == [game.treeplex_p1.num_sequences, game.treeplex_p2.num_sequences]
        details = {"lp": lp_certificate(game, reference["lp"]["value"]), "methods": {}}
        for method, flags in ablation.METHODS.items():
            learner = ablation.CFRSolver(game, **flags, check_every=20001)
            for step, previous in zip(summary["checkpoints"], [0] + summary["checkpoints"]):
                if step > 200:
                    break
                x, y = learner.solve(max_iter=step - previous, spr_threshold=None)
                expected = next(r for r in groups[(case, method, 0)] if int(r["iteration"]) == step)
                assert abs(Solver(game).saddle_point_residual(x, y) - float(expected["spr"])) < 1e-10
                assert max(ablation.flow_error(x, game.treeplex_p1), ablation.flow_error(y, game.treeplex_p2)) < 1e-10
            finals = [groups[(case, method, repeat)][-1] for repeat in range(3)]
            for step in range(14):
                for metric in ("spr", "lower", "upper", "value", "current_spr"):
                    assert len({groups[(case, method, repeat)][step][metric] for repeat in range(3)}) == 1
            details["methods"][method] = {
                "final_spr": float(finals[0]["spr"]),
                "median_solve_seconds": float(np.median([float(r["elapsed_solve_seconds"]) for r in finals])),
                "replay_iterations": learner.iterations,
            }
        if case == "neutral":
            plans, metadata = [], []
            for player in (0, 1):
                _, _, plan, meta = validate_loaded_agent(game, RESULTS / f"cfr_plus_p{player + 1}.npz", player)
                plans.append(plan)
                metadata.append(meta)
            spr = Solver(game).saddle_point_residual(*plans)
            assert abs(spr - reference["experimental_export_spr"]) < 1e-12
            details["experimental_exports"] = {"spr": spr, "strategies": metadata}
        report["scenarios"][case] = details
    return report


def main():
    checks = {"interfaces": check_interfaces(), "experiment_a": check_experiment_a(),
              "experiment_b": check_experiment_b()}
    print(f"PASS: {len(checks['interfaces'])} required interfaces.")
    a, b = checks["experiment_a"], checks["experiment_b"]
    print(f"PASS: Experiment A: {a['histories']} histories, {a['observations']} observations, "
          f"{a['source_hashes']} matching source hashes.")
    print(f"PASS: Experiment B: {b['observations']} observations, "
          f"{b['source_hashes']} matching source hashes.")
    print("PASS: seven scenario LP checks; 200-iteration replays; experimental exports.")


if __name__ == "__main__":
    main()
