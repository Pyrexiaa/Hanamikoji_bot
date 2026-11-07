"""Build the five-page report from the recorded experiment and validation data."""
import csv
import json
from pathlib import Path
from statistics import median

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'task4' / 'results'
METHODS = ['CFR', 'CFR-linear', 'RM+', 'RM+-linear', 'CFR+']
PALETTE = ['#52677d', '#c58220', '#218c79', '#b15176', '#3864d9']
# SimpleDocTemplate's frame has six points of inner padding on each side.
WIDTH = A4[0] - 5 * cm - 12
BODY = ParagraphStyle('body', fontName='Times-Roman', fontSize=12, leading=15,
                      spaceAfter=7, textColor=colors.HexColor('#172638'))
HEAD = ParagraphStyle('heading', parent=BODY, fontName='Times-Bold', spaceBefore=5,
                      spaceAfter=8)
TITLE = ParagraphStyle('title', parent=HEAD, alignment=TA_CENTER, spaceAfter=12)
CELL = ParagraphStyle('cell', parent=BODY, leading=14, spaceAfter=0)


def p(text, style=BODY):
    return Paragraph(text, style)


def table(rows, widths):
    data = [[p(str(value), CELL) for value in row] for row in rows]
    result = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1)
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8edf4')),
        ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#7e94ad')),
        ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#dce2e8')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    return result


def neutral_plot(rows, horizontal, filename, xlabel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11.5,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.3, 2.85), constrained_layout=True)
    for method, color in zip(METHODS, PALETTE):
        subset = [r for r in rows if r['case'] == 'neutral' and r['method'] == method]
        iterations = sorted(set(int(r['iteration']) for r in subset))
        xx = [median(float(r[horizontal]) for r in subset if int(r['iteration']) == t) for t in iterations]
        yy = [median(float(r['spr']) for r in subset if int(r['iteration']) == t) for t in iterations]
        ax.loglog(xx, yy, color=color, label=method, lw=1.6)
    ax.axhline(.002, color='#8592a1', ls=':', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Exact SPR')
    ax.grid(alpha=.2)
    ax.legend(loc='lower left', ncol=2, fontsize=10, framealpha=.9)
    fig.savefig(OUT / filename, dpi=220)
    plt.close(fig)


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    validation = json.loads((OUT / 'validation.json').read_text())
    with (OUT / 'convergence.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    n = summary['iterations']
    final = {(case, method): [r for r in rows if r['case'] == case and r['method'] == method
                              and int(r['iteration']) == n]
             for case in summary['games'] for method in METHODS}
    value = lambda case, method, key='spr': median(float(r[key]) for r in final[case, method])
    neutral_plot(rows, 'iteration', 'report_iterations.png', 'Iterations')
    neutral_plot(rows, 'training_seconds', 'report_time.png', 'Training seconds (median of 3 runs)')
    story = []
    add = story.append
    add(p('CS6208: Solving tiny_hana and testing CFR improvements', TITLE))
    add(p('1. Problem, representation, and required implementation', HEAD))
    add(p('The objective is an approximate Nash equilibrium of one round of tiny_hana, '
          'a two-player zero-sum game with hidden cards. The required game starts with '
          'neutral favors and chooses the starting player uniformly. A win pays +1, '
          'a loss -1, and an unresolved round 0. The supplied engine and treeplex '
          'representation were retained unchanged.'))
    add(p('<b>Task 1: beliefs.</b> A depth-first traversal collects concrete states '
          'consistent with the target observation history. Each state receives the '
          'product of chance probabilities and opponent action probabilities, followed '
          'by normalization. The target player\'s own reach factor cancels under perfect '
          'recall. Inconsistent observation prefixes and zero-probability branches are '
          'pruned. Every executed action is undone in a finally block, including when '
          'an opponent raises an exception. An unreachable target returns an empty dictionary.'))
    add(p('<b>Task 2: best responses.</b> Let x and y be the two realization plans and A '
          'the chance-weighted payoff matrix. Fixing y gives Player 1 reward Ay; fixing x '
          'gives Player 2 reward -A<super>T</super>x. A backward treeplex pass selects '
          'the largest action value at each information set and adds it to its parent '
          'sequence. Values from multiple child information sets are added. The returned '
          'deterministic policy can be converted to sequence form.'))
    add(p('upper = u(BR<sub>1</sub>(y), y), &nbsp; lower = u(x, BR<sub>2</sub>(x))<br/>'
          'SPR(x, y) = upper - lower.'))
    add(p('This is the sum of the two unilateral improvement opportunities, without '
          'the factor 1/2 sometimes used for exploitability. For feasible strategies, '
          'lower bounds the game value from below and upper bounds it from above.'))
    add(p('<b>Task 3: CFR.</b> At each information set, positive cumulative regrets '
          'are normalized into an action distribution, with uniform fallback. A '
          'backward pass computes continuation values, adds action value minus expected '
          'value to local regret, and propagates the expected value to the parent. '
          'Averages are accumulated in sequence form, ensuring correct own-reach '
          'weighting when converted back to behavioral strategies [1].'))
    add(p('The implementation batches independent information sets by own-action depth '
          'and action count, using NumPy for regret matching and sparse reward products. '
          'It removes zero payoff entries from training. These changes preserve the '
          'backward-pass equations; they introduce no information abstraction.'))
    add(PageBreak())

    add(p('2. Extension: a controlled ablation of CFR improvements', HEAD))
    add(p('The research question is whether retaining only nonnegative regret, '
          'changing the average, or changing the update order explains faster convergence '
          'on tiny_hana. The first four configurations form a 2 x 2 ablation. They use '
          'the same simultaneous update schedule, initial zero regrets, full tree, '
          'iteration budget, and diagnostics.'))
    add(table([
        ['Configuration', 'Regret update / average / schedule'],
        ['CFR', 'Signed cumulative / uniform / simultaneous'],
        ['CFR-linear', 'Signed cumulative / linear / simultaneous'],
        ['RM+', 'Nonnegative cumulative / uniform / simultaneous'],
        ['RM+-linear', 'Nonnegative cumulative / linear / simultaneous'],
        ['CFR+', 'Nonnegative cumulative / linear / alternating'],
    ], [100, WIDTH - 100]))
    add(Spacer(1, 8))
    add(p('The nonnegative update is R &larr; max(R + instantaneous regret, 0), '
          'applied after adding regret. Linear strategy averaging uses weight t at '
          'iteration t, with no averaging delay. It does not weight the regret updates '
          'and therefore is not LCFR. In the fifth configuration, Player 2 uses '
          'Player 1\'s newly updated strategy. This follows the alternating, '
          'regret-matching-plus design of CFR+ [2]. Its improvement over the fourth '
          'configuration tests update order conditional on those other choices.'))
    add(p('<b>Game conditions.</b> All use a uniformly random starter. Neutral has '
          'no initial favors. Advantage gives Player 1 the favor of character 1. '
          'Contested gives Player 1 character 1 and Player 2 character 3. These are '
          'three single-round payoff conditions, not a solution of the multi-round game.'))
    add(p(f'<b>Protocol.</b> Each configuration runs for {n:,} iterations in each '
          f'condition, with {summary["repeats"]} timing repeats and rotated method order. '
          'There are 14 predetermined checkpoints from 1 to 20,000 iterations. '
          'The reported residual is evaluated exactly through best responses to '
          'the average strategy; training and evaluation use no sampled trajectories. '
          'The repeated residuals coincide, so repetitions characterize timing, not '
          'statistical uncertainty in solution quality.'))
    add(p('<b>Verification.</b> Thirteen tests cover analytic posterior probabilities '
          'for both players, independently enumerated conditional histories, state '
          'restoration, exhaustive pure-strategy responses on a branching toy game, '
          'a known matrix equilibrium, and exact continuation/reset behavior. All '
          'checkpoint strategies satisfy the sequence-flow constraints within 1e-9. '
          'Separate maximin and minimax linear programs provide independent value '
          'certificates for every game condition.'))
    add(PageBreak())

    add(p('3. Convergence results and interaction between improvements', HEAD))
    add(Image(str(OUT / 'report_iterations.png'), width=WIDTH, height=WIDTH * 2.85 / 6.3))
    add(p('Figure 1. Neutral-game average-strategy SPR. The dotted line is the '
          'notebook example target, 0.002. Curves are exact deterministic evaluations.'))
    add(table([['Final SPR', 'Neutral', 'Advantage', 'Contested']] + [
        [method] + [f'{value(case, method):.3g}' for case in ['neutral', 'advantage', 'contested']]
        for method in METHODS
    ], [118, (WIDTH - 118) / 3, (WIDTH - 118) / 3, (WIDTH - 118) / 3]))
    add(Spacer(1, 8))
    ratio_plus = value('neutral', 'CFR') / value('neutral', 'RM+')
    ratio_weight = value('neutral', 'RM+') / value('neutral', 'RM+-linear')
    ratio_order = value('neutral', 'RM+-linear') / value('neutral', 'CFR+')
    add(p(f'At 20,000 iterations on Neutral, nonnegative regrets with uniform '
          f'averaging reduce SPR by {ratio_plus:.1f} times relative to vanilla CFR. '
          f'Adding linear averaging to RM+ reduces it by another {ratio_weight:.1f} '
          f'times. Alternating updates then reduce it by {ratio_order:.1f} times. '
          'The same ranking holds in all three conditions at the final checkpoint.'))
    add(p('Linear averaging alone is worse than uniform averaging for signed '
          'cumulative regret in every condition here. Weighting recent iterates '
          'more heavily is not sufficient when those iterates continue to oscillate. '
          'The results are consistent with an interaction between regret flooring '
          'and averaging; they do not establish a universal advantage for either '
          'choice. Intermediate curves also need not decrease monotonically.'))
    add(PageBreak())

    add(p('4. Runtime, averaging, and computational cost', HEAD))
    add(Image(str(OUT / 'report_time.png'), width=WIDTH, height=WIDTH * 2.85 / 6.3))
    add(p('Figure 2. Neutral-game SPR against training-only time. Medians use '
          'three repetitions; exact diagnostic work is excluded from the horizontal axis.'))
    timing_rows = [['Neutral method', 'Seconds: median [range]', 'First checked T at SPR &le; .002']]
    for method in METHODS:
        times = [float(r['training_seconds']) for r in final['neutral', method]]
        eligible = [int(r['iteration']) for r in rows if r['case'] == 'neutral'
                    and r['method'] == method and float(r['spr']) <= .002]
        timing_rows.append([method, f'{median(times):.2f} [{min(times):.2f}, {max(times):.2f}]',
                            str(min(eligible)) if eligible else 'Not reached'])
    add(table(timing_rows, [100, 165, WIDTH - 265]))
    add(Spacer(1, 8))
    factor = value('neutral', 'CFR+', 'training_seconds') / value('neutral', 'CFR', 'training_seconds')
    add(p(f'Alternating CFR+ costs {factor:.2f} times as much training time as vanilla '
          'CFR at the same iteration budget in the Neutral median. It performs an '
          'additional Player 1 strategy computation per iteration, but its lower '
          'residual also persists when compared against elapsed training time.'))
    add(p(f'For vanilla CFR at the final Neutral checkpoint, the current policy pair '
          f'has SPR {value("neutral", "CFR", "current_spr"):.4g}, while its time average '
          f'has SPR {value("neutral", "CFR"):.4g}. This illustrates why the submitted '
          'artifact uses the average rather than the latest policy.'))
    add(p('Each player has 3,672 sequences and 2,757 information sets. Neutral stores '
          '12,832 payoff pairs, but only 2,816 are nonzero. Tree extraction is measured '
          'separately. Training time excludes payoff preparation, best-response checks, '
          'serialization, and plotting; it is not total end-to-end time. The reference '
          'LP is very fast on this small game, so these results do not argue that CFR '
          'is the fastest solver for this instance.'))
    add(PageBreak())

    add(p('5. Independent certificates, limitations, and reproducibility', HEAD))
    add(p('The independent LP uses nonnegative realization plans satisfying E x = e '
          'and F y = f. Player 1 maximizes f<super>T</super>q subject to '
          'F<super>T</super>q &le; A<super>T</super>x, with unrestricted q. A separate '
          'Player 2 program minimizes e<super>T</super>p subject to '
          'A y &le; E<super>T</super>p. Agreement of both optima and best-response '
          'bounds checks the value and the signs in the implementation.'))
    add(table([['LP reference', 'Game value (P1)', 'Exact BR gap']] + [
        [case.capitalize(), f'{(summary["games"][case]["lp"]["maximin_value"] or 0.):.8f}',
         f'{max(0., summary["games"][case]["lp_upper"]-summary["games"][case]["lp_lower"]):.2g}']
        for case in ['neutral', 'advantage', 'contested']
    ], [130, 150, WIDTH - 280]))
    add(Spacer(1, 8))
    add(p('The neutral value is also zero by player-swap symmetry with a uniformly '
          'random starter. The asymmetric condition checks that the LP and payoff '
          'calculations also agree when the value is nonzero.'))
    add(p(f'<b>Submitted strategy.</b> The saved policies are the Neutral CFR+ linear '
          f'time average after {n:,} iterations, with measured SPR '
          f'{summary["games"]["neutral"]["submitted_spr"]:.8g}. They were saved and '
          'reloaded through the provided API. Validation independently checks '
          'all behavioral distributions, sequence flows, expected payoff through '
          'full engine traversal, and 100 seeded single-round games. The supplied '
          'treeplex file retains its original SHA-256.'))
    add(p(f'After reloading, exact response bounds are '
          f'[{validation["exact_best_response_lower_bound"]:.3g}, '
          f'{validation["exact_best_response_upper_bound"]:.3g}]. Nine payoff '
          f'comparisons against exhaustive engine traversal agree within '
          f'{validation["maximum_engine_representation_difference"]:.2g}.'))
    add(p('<b>Limits.</b> These are full-tree experiments on tiny_hana only. They '
          'do not establish scalability to medium_hana or optimality over repeated '
          'rounds. Three timing repetitions on one machine do not support precise '
          'claims about small speed differences. The final ranking is empirical, '
          'and the intermediate curves are not monotone. A useful next experiment '
          'would vary the averaging delay under a fixed time budget.'))
    add(p('<b>Reproduce.</b> Run the tests, run_experiments.py, validate_submission.py, '
          'and build_report.py as documented in task4/README.md. Raw checkpoint '
          'data, all timing repeats, LP certificates, and validation results are '
          'included. The experiment environment is Python 3.13, NumPy 2.5.4, '
          'macOS 15.6.1 on arm64. Core graded code depends only on NumPy; '
          'SciPy and plotting/report libraries are optional experiment tools.'))
    add(p('<b>References</b>', HEAD))
    add(p('[1] Zinkevich, Johanson, Bowling, and Piccione (2007). '
          '<i>Regret Minimization in Games with Incomplete Information.</i> NIPS 20. '
          '<link href="https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html" color="#28518a">Original paper</link>.'))
    add(p('[2] Tammelin (2014). <i>Solving Large Imperfect Information Games Using '
          'CFR+.</i> arXiv:1407.5042. '
          '<link href="https://arxiv.org/abs/1407.5042" color="#28518a">Original report</link>.'))

    def footer(canvas, doc):
        canvas.setFont('Times-Roman', 12)
        canvas.setFillColor(colors.HexColor('#5f6c79'))
        canvas.drawRightString(A4[0] - 2.5 * cm, 1.45 * cm, str(doc.page))

    doc = SimpleDocTemplate(str(ROOT / 'report.pdf'), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm,
                            title='CS6208: Solving tiny_hana and testing CFR improvements',
                            author='CS6208 project')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader
    reader = PdfReader(ROOT / 'report.pdf')
    if len(reader.pages) > 5:
        raise RuntimeError(f'Report exceeds five pages: {len(reader.pages)}')
    print(f'Created report.pdf ({len(reader.pages)} A4 pages).')


if __name__ == '__main__':
    main()
