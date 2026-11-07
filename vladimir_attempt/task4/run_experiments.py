"""Deterministic full-tree CFR ablation; run from the project root.

Optional experiment dependencies: scipy and matplotlib. The graded solver only
needs NumPy. CSV contains every timing repetition, not just selected results.
"""
import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Keep this attempt's implementations ahead of the shared game engine.
# The ancestor fallback also supports a locally staged copy before upload.
REPO = next((path for path in ROOT.parents if (path / "engine").is_dir()), ROOT.parent)
sys.path[:0] = [str(ROOT / "student_attempts"), str(ROOT / "task4"), str(ROOT), str(REPO)]
from env_config import env
from solver import CFRSolver, Solver
from treeplex_representation import (
    TreeplexGame, convert_sequence_to_behavioral_form_strategy,
)
from linear_program import solve_sequence_form_lp

METHODS = {
    'CFR': {},
    'CFR-linear': {'averaging': 'linear'},
    'RM+': {'regret_matching_plus': True},
    'RM+-linear': {'regret_matching_plus': True, 'averaging': 'linear'},
    'CFR+': {'regret_matching_plus': True, 'averaging': 'linear', 'alternating': True},
}
COLORS = ['#52677d', '#c58220', '#218c79', '#b15176', '#3864d9']


def make_game(case):
    state = env.GameState()
    if case == 'advantage':
        state.favors[env.Suit.GEISHA_1] = -1
    elif case == 'contested':
        state.favors[env.Suit.GEISHA_1] = -1
        state.favors[env.Suit.GEISHA_3] = 1
    return TreeplexGame(state)


def flow_error(x, treeplex):
    errors = [abs(x[0] - 1), max(0, float(-x.min()))]
    for s, e, p in zip(treeplex.info_state_start_seq,
                       treeplex.info_state_end_seq, treeplex.info_state_parent_seq):
        errors.append(abs(x[s:e].sum() - x[p]))
    return float(max(errors))


def bounds(solver, x, y):
    upper = solver.compute_payoffs_from_seq_form_strategies(
        solver.compute_best_response(0, y, True), y)
    lower = solver.compute_payoffs_from_seq_form_strategies(
        x, solver.compute_best_response(1, x, True))
    return float(lower), float(upper)


def save_submission(game, x, y):
    # eps=0 preserves all positive-reach distributions in the saved artifact.
    for player, seq, tp in [(0, x, game.treeplex_p1), (1, y, game.treeplex_p2)]:
        beh = convert_sequence_to_behavioral_form_strategy(seq, tp, in_place=False,
                                                          eps=np.finfo(float).tiny)
        agent = game.behavioral_strategy_to_agent(beh, player)
        game.save_agent(agent, player, str(ROOT / 'tasks' / f'nash_p{player+1}.npz'))


def plot_results(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
    cases = ['neutral', 'advantage', 'contested']
    for horizontal, filename, label in [('iteration', 'convergence_iterations.png', 'Iterations'),
                                         ('training_seconds', 'convergence_time.png', 'Training seconds (median across repeats)')]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        for ax, case in zip(axes, cases):
            for (method, _), color in zip(METHODS.items(), COLORS):
                records = [r for r in rows if r['case'] == case and r['method'] == method]
                its = sorted(set(r['iteration'] for r in records))
                xx = [np.median([r[horizontal] for r in records if r['iteration'] == t]) for t in its]
                yy = [np.median([r['spr'] for r in records if r['iteration'] == t]) for t in its]
                ax.loglog(xx, np.maximum(yy, 1e-15), label=method, color=color, lw=1.7)
            ax.axhline(0.002, color='#88929a', ls=':', lw=1)
            ax.set_title(case.capitalize())
            ax.set_xlabel(label)
            ax.grid(alpha=.18, which='major')
        axes[0].set_ylabel('Exact saddle-point residual')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False)
        fig.savefig(out / filename, dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out', type=Path, default=ROOT / 'task4' / 'results')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints = [t for t in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
                               2000, 5000, 10000, 20000] if t < args.iterations] + [args.iterations]
    rows, games = [], {}
    for case in ['neutral', 'advantage', 'contested']:
        begin = perf_counter()
        game = make_game(case)
        build_seconds = perf_counter() - begin
        evaluate = Solver(game)
        lp_x, lp_y, lp_meta = solve_sequence_form_lp(game)
        lp_lower, lp_upper = bounds(evaluate, lp_x, lp_y)
        assert lp_upper - lp_lower < 1e-7
        games[case] = {
            'build_seconds': build_seconds,
            'sequences': [game.treeplex_p1.num_sequences, game.treeplex_p2.num_sequences],
            'infosets': [game.treeplex_p1.get_num_infosets(), game.treeplex_p2.get_num_infosets()],
            'stored_pairs': len(game.seq_pair_to_leaves),
            'nonzero_pairs': sum(v != 0 for v in game.seq_pair_to_leaves.values()),
            'lp': lp_meta, 'lp_lower': lp_lower, 'lp_upper': lp_upper,
            'lp_flow_error': max(flow_error(lp_x, game.treeplex_p1), flow_error(lp_y, game.treeplex_p2)),
        }
        if case == 'neutral':
            assert abs(lp_upper) < 1e-7 and abs(lp_lower) < 1e-7
        for repetition in range(args.repeats):
            # Rotate method order to reduce systematic warm-up/time drift effects.
            order = list(METHODS)
            shift = repetition % len(order)
            for method in order[shift:] + order[:shift]:
                cfr = CFRSolver(game, **METHODS[method])
                previous = 0
                for t in checkpoints:
                    start = perf_counter()
                    x, y = cfr.solve(max_iter=t - previous, spr_threshold=0,
                                     check_every=args.iterations + 1)
                    wall = perf_counter() - start
                    lower, upper = bounds(evaluate, x, y)
                    assert max(flow_error(x, game.treeplex_p1), flow_error(y, game.treeplex_p2)) < 1e-9
                    assert lower <= lp_upper + 1e-8 and upper >= lp_lower - 1e-8
                    current_spr = evaluate.saddle_point_residual(
                        cfr.current_seq_form_p1, cfr.current_seq_form_p2)
                    rows.append(dict(case=case, repeat=repetition, method=method,
                                     iteration=t, spr=upper-lower, lower=lower, upper=upper,
                                     value=float(cfr.val), current_spr=current_spr,
                                     training_seconds=cfr.training_seconds,
                                     checkpoint_wall_seconds=wall))
                    previous = t
                print(f'{case:10} rep={repetition} {method:12} SPR={cfr.spr:.8g} '
                      f'train={cfr.training_seconds:.3f}s', flush=True)
                if case == 'neutral' and repetition == 0 and method == 'CFR+':
                    save_submission(game, x, y)
                    games[case]['submitted_method'] = method
                    games[case]['submitted_iterations'] = t
                    games[case]['submitted_spr'] = upper - lower
        with (args.out / 'summary.json').open('w') as f:
            json.dump({'python': sys.version, 'numpy': np.__version__,
                       'platform': platform.platform(), 'machine': platform.machine(),
                       'iterations': args.iterations, 'repeats': args.repeats,
                       'checkpoints': checkpoints, 'methods': METHODS, 'games': games}, f, indent=2)
        with (args.out / 'convergence.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_results(rows, args.out)


if __name__ == '__main__':
    main()
