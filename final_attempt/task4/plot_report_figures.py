"""Render the fresh CFR CSV at the final 16 cm report width with 12 pt labels."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullFormatter
import numpy as np
from run_cfr_ablation import CASES, METHODS, COLORS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'task4/results'


def main():
    path = RESULTS / 'convergence.csv'
    summary = json.loads((RESULTS / 'summary.json').read_text(encoding='utf-8'))
    for name, expected in summary['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    with path.open(newline='', encoding='utf-8') as handle:
        rows = [{k: v if k in ('case', 'method') else float(v) for k, v in r.items()} for r in csv.DictReader(handle)]
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
                         'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    for horizontal, filename, label in [('iteration', 'convergence_iterations.png', 'Iterations'),
            ('elapsed_solve_seconds', 'convergence_time.png', 'Median solve time (s)')]:
        fig, axes = plt.subplots(1, 3, figsize=(16/2.54, 7.2/2.54), sharey=True, layout='constrained')
        for ax, case in zip(axes, CASES):
            for method, color in zip(METHODS, COLORS):
                records = [r for r in rows if r['case'] == case and r['method'] == method]
                checkpoints = sorted({r['iteration'] for r in records})
                xx = [float(np.median([r[horizontal] for r in records if r['iteration'] == t])) for t in checkpoints]
                yy = [float(np.median([r['spr'] for r in records if r['iteration'] == t])) for t in checkpoints]
                assert all(5e-7 < value < 1.0 for value in yy), (case, method, yy)
                ax.loglog(xx, yy, label=method, color=color, linewidth=1.5)
            ax.axhline(0.002, color='#88929a', linestyle=':', linewidth=1)
            ax.set_title(case.capitalize())
            ax.set_ylim(5e-7, 1.0)
            ax.yaxis.set_major_locator(FixedLocator([1e-1, 1e-3, 1e-5]))
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            ax.xaxis.set_major_locator(FixedLocator([1, 100, 10000] if horizontal == 'iteration' else [0.01, 1]))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=.18, which='major')
        axes[0].set_ylabel('SPR')
        fig.supxlabel(label, fontsize=12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False,
                   handlelength=1.4, columnspacing=.8, handletextpad=.4)
        fig.savefig(RESULTS / filename, dpi=300)
        plt.close(fig)
    print('Generated both CFR figures from fresh CSV at 16 cm width, 12 pt labels')


if __name__ == '__main__':
    main()
