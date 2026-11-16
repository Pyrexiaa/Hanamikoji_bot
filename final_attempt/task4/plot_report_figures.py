"""Render the CFR ablation CSV as report Figure 2, at its printed size with 12 pt labels.

The figures are drawn at exactly the size they occupy in the report (the 16 cm text
width by 5.2 cm), so the report embeds them unscaled and the labels stay 12 pt.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullFormatter, NullLocator
import numpy as np
from run_cfr_ablation import CASES, METHODS, COLORS
from source_digest import matches_recorded_digest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'task4/results'
REPORT_FIGURE_SIZE = (16 / 2.54, 5.2 / 2.54)   # inches: 16 cm text width by 5.2 cm


def main():
    path = RESULTS / 'convergence.csv'
    summary = json.loads((RESULTS / 'summary.json').read_text(encoding='utf-8'))
    for name, expected in summary['source_sha256'].items():
        assert matches_recorded_digest(ROOT / name, expected), name
    with path.open(newline='', encoding='utf-8') as handle:
        rows = [{k: v if k in ('case', 'method') else float(v) for k, v in r.items()} for r in csv.DictReader(handle)]
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
                         'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    for horizontal, filename, label in [('iteration', 'convergence_iterations.png', 'Iterations'),
            ('elapsed_solve_seconds', 'convergence_time.png', 'Median solve time (s)')]:
        fig, axes = plt.subplots(1, 3, figsize=REPORT_FIGURE_SIZE, sharey=True, layout='constrained')
        for ax, case in zip(axes, CASES):
            for method, color in zip(METHODS, COLORS):
                records = [r for r in rows if r['case'] == case and r['method'] == method]
                checkpoints = sorted({r['iteration'] for r in records})
                xx = [float(np.median([r[horizontal] for r in records if r['iteration'] == t])) for t in checkpoints]
                yy = [float(np.median([r['spr'] for r in records if r['iteration'] == t])) for t in checkpoints]
                assert all(5e-7 < value < 1.0 for value in yy), (case, method, yy)
                ax.loglog(xx, yy, label=method, color=color, linewidth=1.5)
            ax.axhline(0.002, color='#88929a', linestyle=':', linewidth=1)
            # Every curve falls from the top left, so the top right corner is free for the name.
            ax.text(.97, .95, case.capitalize(), transform=ax.transAxes, ha='right', va='top')
            ax.set_ylim(5e-7, 1.0)
            ax.yaxis.set_major_locator(FixedLocator([1e-1, 1e-3, 1e-5]))
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            ax.yaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_locator(FixedLocator([1, 100, 10000] if horizontal == 'iteration' else [0.01, 1]))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=.18, which='major')
        axes[0].set_ylabel('SPR')
        fig.supxlabel(label, fontsize=12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=5, frameon=False,
                   handlelength=1.2, columnspacing=.9, handletextpad=.4, borderaxespad=0)
        fig.savefig(RESULTS / filename, dpi=300)
        plt.close(fig)
    print('Generated both CFR figures from the CSV at 16 cm x 5.2 cm, 12 pt labels')


if __name__ == '__main__':
    main()
