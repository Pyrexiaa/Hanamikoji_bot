"""Deterministic full-tree CFR ablation using the canonical student solver.

Compares five CFR variants built cleanly on ``student_attempts/solver.py``
(``CFRSolver``, ``Solver``) and referenced against ``student_attempts/lp_solver.py``
(``solve_sequence_form_lp``).

Optional experiment dependencies: scipy and matplotlib. The graded solver only
needs NumPy. The output CSV contains every timing repetition, not just medians.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Keep this attempt's canonical implementations ahead of the shared game engine.
REPO = next((path for path in ROOT.parents if (path / "engine").is_dir()), ROOT.parent)
sys.path[:0] = [str(ROOT / "student_attempts"), str(ROOT), str(REPO)]

from env_config import PLAYER_1, PLAYER_2, env
from lp_solver import solve_sequence_form_lp
from solver import CFRSolver, Solver
from treeplex_representation import (
    TreeplexGame,
    convert_behavioral_to_sequence_form_strategy,
    convert_sequence_to_behavioral_form_strategy,
)

METHODS = {
    "CFR": {},
    "CFR-linear": {"linear_averaging": True},
    "RM+": {"regret_matching_plus": True},
    "RM+-linear": {"regret_matching_plus": True, "linear_averaging": True},
    "CFR+": {
        "regret_matching_plus": True,
        "linear_averaging": True,
        "alternating": True,
    },
}

METHOD_NOTES = {
    "CFR": "Simultaneous vanilla CFR with unit-weight regret updates and uniform strategy averaging.",
    "CFR-linear": "Simultaneous CFR with unit-weight regrets and linear strategy averaging (w_t = t; not full LCFR).",
    "RM+": "Simultaneous CFR with regret matching+ clipping R <- max(R + r, 0) and uniform strategy averaging.",
    "RM+-linear": "Simultaneous RM+ clipping with linear strategy averaging (w_t = t; unit-weight regrets).",
    "CFR+": "Alternating updates (P2 responds to P1's refreshed strategy), RM+ clipping, and linear pre-update strategy averaging.",
}

COLORS = ["#52677d", "#c58220", "#218c79", "#b15176", "#3864d9"]
DEFAULT_CHECKPOINTS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
CASES = ("neutral", "advantage", "contested")


def make_game(case):
    if case == "neutral":
        return TreeplexGame()
    state = env.GameState()
    if case == "advantage":
        state.favors[env.Suit.GEISHA_1] = -1
    elif case == "contested":
        state.favors[env.Suit.GEISHA_1] = -1
        state.favors[env.Suit.GEISHA_3] = 1
    else:
        raise ValueError(f"Unknown ablation case: {case}")
    return TreeplexGame(state)


def flow_error(x, treeplex):
    errors = [abs(float(x[0]) - 1.0), max(0.0, float(-np.min(x)))]
    for s, e, p in zip(
        treeplex.info_state_start_seq,
        treeplex.info_state_end_seq,
        treeplex.info_state_parent_seq,
    ):
        errors.append(abs(float(np.sum(x[s:e]) - x[p])))
    return float(max(errors))


def bounds(solver, x, y):
    upper = solver.compute_payoffs_from_seq_form_strategies(
        solver.compute_best_response(PLAYER_1, y, True), y
    )
    lower = solver.compute_payoffs_from_seq_form_strategies(
        x, solver.compute_best_response(PLAYER_2, x, True)
    )
    return float(lower), float(upper)


def save_experimental_cfr_plus(game, x, y, out_dir):
    """Save experimental CFR+ policies under results/ without touching tasks/nash_p*.npz."""
    loaded_seq = []
    for player, seq, tp in [(PLAYER_1, x, game.treeplex_p1), (PLAYER_2, y, game.treeplex_p2)]:
        beh = convert_sequence_to_behavioral_form_strategy(
            seq, tp, in_place=False, eps=np.finfo(float).tiny
        )
        agent = game.behavioral_strategy_to_agent(beh, player)
        path = out_dir / f"cfr_plus_p{player + 1}.npz"
        game.save_agent(agent, player, str(path))
        restored = game.load_agent(str(path))
        restored_seq = convert_behavioral_to_sequence_form_strategy(
            game.agent_to_behavioral_strategy(restored, player), tp, in_place=False
        )
        assert flow_error(restored_seq, tp) < 1e-10
        loaded_seq.append(restored_seq)
    return float(Solver(game).saddle_point_residual(*loaded_seq))


def plot_results(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )
    for horizontal, filename, label in [
        ("iteration", "convergence_iterations.png", "Iterations"),
        (
            "elapsed_solve_seconds",
            "convergence_time.png",
            "Solve wall seconds (median)",
        ),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        for ax, case in zip(axes, CASES):
            for (method, _), color in zip(METHODS.items(), COLORS):
                records = [r for r in rows if r["case"] == case and r["method"] == method]
                its = sorted({r["iteration"] for r in records})
                xx = [float(np.median([r[horizontal] for r in records if r["iteration"] == t])) for t in its]
                yy = [float(np.median([r["spr"] for r in records if r["iteration"] == t])) for t in its]
                ax.loglog(xx, np.maximum(yy, 1e-15), label=method, color=color, lw=1.7)
            ax.axhline(0.002, color="#88929a", ls=":", lw=1)
            ax.set_title(case.capitalize())
            ax.set_xlabel(label)
            ax.grid(alpha=0.18, which="major")
        axes[0].set_ylabel("Exact saddle-point residual")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
        fig.savefig(out / filename, dpi=200)
        plt.close(fig)


def run_ablation(iterations=20000, repeats=3, out_dir=None, warmup_iterations=2):
    import scipy

    if iterations < 1 or repeats < 1 or warmup_iterations < 0:
        raise ValueError("iterations and repeats must be positive; warmup_iterations must be >= 0")
    if out_dir is None:
        out_dir = ROOT / "task4" / "results"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [t for t in DEFAULT_CHECKPOINTS if t < iterations] + [iterations]
    sources = [
        ROOT / "student_attempts" / "solver.py",
        ROOT / "student_attempts" / "lp_solver.py",
        ROOT / "student_attempts" / "treeplex_representation.py",
        ROOT / "student_attempts" / "env_config.py",
        Path(__file__).resolve(),
    ]
    source_sha256 = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }

    rows, games = [], {}
    for case in CASES:
        begin = perf_counter()
        game = make_game(case)
        build_seconds = perf_counter() - begin
        evaluate = Solver(game)

        lp_started = perf_counter()
        lp_res = solve_sequence_form_lp(game, tolerance=1e-9)
        lp_total_seconds = perf_counter() - lp_started
        lp_x, lp_y = lp_res["sol_seq_form_p1"], lp_res["sol_seq_form_p2"]
        lp_lower, lp_upper = bounds(evaluate, lp_x, lp_y)
        lp_flow_err = max(
            flow_error(lp_x, game.treeplex_p1),
            flow_error(lp_y, game.treeplex_p2),
        )
        assert lp_res["status"] == "optimal"
        assert lp_upper - lp_lower < 1e-7
        assert lp_flow_err < 1e-9
        if case == "neutral":
            assert abs(lp_upper) < 1e-7 and abs(lp_lower) < 1e-7

        nonzero_pairs = sum(v != 0 for v in game.seq_pair_to_leaves.values())
        lp_summary = {
            k: v
            for k, v in lp_res.items()
            if k not in ("sol_seq_form_p1", "sol_seq_form_p2")
        }
        lp_summary["nonzero_payoff_pairs"] = nonzero_pairs
        lp_summary["total_seconds"] = lp_total_seconds

        games[case] = {
            "build_seconds": build_seconds,
            "sequences": [game.treeplex_p1.num_sequences, game.treeplex_p2.num_sequences],
            "infosets": [game.treeplex_p1.get_num_infosets(), game.treeplex_p2.get_num_infosets()],
            "stored_pairs": len(game.seq_pair_to_leaves),
            "nonzero_pairs": nonzero_pairs,
            "lp_source": "student_attempts/lp_solver.py",
            "lp": lp_summary,
            "lp_lower": lp_lower,
            "lp_upper": lp_upper,
            "lp_flow_error": lp_flow_err,
        }

        if warmup_iterations > 0:
            for kw in METHODS.values():
                CFRSolver(game, **kw, check_every=warmup_iterations + 1).solve(
                    max_iter=warmup_iterations, spr_threshold=None
                )

        for repetition in range(repeats):
            order = list(METHODS)
            shift = repetition % len(order)
            for method in order[shift:] + order[:shift]:
                cfr = CFRSolver(game, **METHODS[method], check_every=iterations + 1)
                previous = 0
                for t in checkpoints:
                    call_start = perf_counter()
                    x, y = cfr.solve(max_iter=t - previous, spr_threshold=None)
                    seg_wall = perf_counter() - call_start
                    assert cfr.iterations == t, (cfr.iterations, t)
                    lower, upper = bounds(evaluate, x, y)
                    spr = upper - lower
                    assert abs(spr - float(cfr.spr)) < 1e-10
                    assert max(
                        flow_error(x, game.treeplex_p1),
                        flow_error(y, game.treeplex_p2),
                    ) < 1e-9
                    assert lower <= lp_upper + 1e-8 and upper >= lp_lower - 1e-8
                    current_spr = evaluate.saddle_point_residual(
                        cfr.current_seq_form_p1, cfr.current_seq_form_p2
                    )
                    rows.append(
                        dict(
                            case=case,
                            repeat=repetition,
                            method=method,
                            iteration=t,
                            spr=spr,
                            lower=lower,
                            upper=upper,
                            value=float(cfr.val),
                            current_spr=current_spr,
                            elapsed_solve_seconds=float(cfr.elapsed_seconds),
                            segment_wall_seconds=seg_wall,
                        )
                    )
                    previous = t
                print(
                    f"{case:10} rep={repetition} {method:12} SPR={cfr.spr:.8g} "
                    f"solve={cfr.elapsed_seconds:.3f}s",
                    flush=True,
                )
                if case == "neutral" and repetition == 0 and method == "CFR+":
                    export_spr = save_experimental_cfr_plus(game, x, y, out_dir)
                    assert abs(export_spr - (upper - lower)) < 1e-8
                    games[case]["experimental_export_method"] = method
                    games[case]["experimental_export_iterations"] = t
                    games[case]["experimental_export_spr"] = export_spr

    summary = {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "engine": "Python",
        "solver_source": "student_attempts/solver.py",
        "lp_source": "student_attempts/lp_solver.py",
        "iterations": iterations,
        "repeats": repeats,
        "warmup_iterations": warmup_iterations,
        "checkpoints": checkpoints,
        "methods": METHODS,
        "method_notes": METHOD_NOTES,
        "averaging_convention": (
            "Canonical CFRSolver averages both players' pre-update realization plans at every iteration "
            "(with w_t = t when linear_averaging=True and unit-weight regret accumulation). In alternating mode, "
            "Player 1's regrets update first and Player 2 updates regrets against Player 1's updated behavioral/sequence "
            "strategy, while both pre-update plans enter the strategy average."
        ),
        "determinism_note": (
            "All five CFR variants are deterministic; repetitions provide timing variance across rotated execution "
            "orders rather than stochastic learning trajectories."
        ),
        "timing_definition": (
            "elapsed_solve_seconds records cumulative wall-clock time spent inside CFRSolver.solve() across the "
            "segmented checkpoint calls, including entry and exit exact SPR certificates per segment, after a "
            f"{warmup_iterations}-iteration warm-up. Excludes game construction, LP reference solves, CSV/JSON writes, "
            "current_spr diagnostics, and plotting."
        ),
        "source_sha256": source_sha256,
        "games": games,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "convergence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_results(rows, out_dir)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "task4" / "results")
    args = parser.parse_args()
    run_ablation(
        iterations=args.iterations,
        repeats=args.repeats,
        out_dir=args.out,
        warmup_iterations=args.warmup,
    )


if __name__ == "__main__":
    main()
