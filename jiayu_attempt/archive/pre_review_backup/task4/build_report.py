"""Build my five-page A4 report from recorded three-algorithm measurements."""
from pathlib import Path
from statistics import median
import json
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ATTEMPT = Path(__file__).resolve().parents[1]
RESULTS = ATTEMPT / 'task4' / 'results'
r = json.loads((RESULTS / 'experiments.json').read_text())
c = r['configuration']
n = r['scenarios']['neutral']
labels = [('CFR', 'CFR'), ('Fictitious play', 'FP'), ('Double oracle', 'DO')]
scenarios = ['neutral', 'first_player', 'favored_g3']
assert c['iterations'] == 2000 and c['spr_threshold'] == .002
assert not c['full_budget'] and c['check_every'] == 25 and c['repeats'] == 3
assert set(n['methods']) == {name for name, _ in labels}
match = re.search(r'Ran (\d+) tests', (RESULTS / 'tests.txt').read_text())
assert match and '\nOK' in (RESULTS / 'tests.txt').read_text()
test_count = int(match.group(1))
W = A4[0] - 5 * cm
ink = colors.HexColor('#203346')
body = ParagraphStyle('body', fontName='Times-Roman', fontSize=12, leading=14.4,
                      spaceAfter=7, alignment=TA_LEFT)
heading = ParagraphStyle('heading', parent=body, fontName='Times-Bold', spaceBefore=9,
                         spaceAfter=7, textColor=ink)
title = ParagraphStyle('title', parent=heading, spaceBefore=0, spaceAfter=10)
caption = ParagraphStyle('caption', parent=body, leading=13.6)
cell = ParagraphStyle('cell', parent=body, spaceAfter=0)
story = []
def p(text, style=body): story.append(Paragraph(text, style))
def h(text): p(text, heading)
def page(): story.append(PageBreak())
def f(v): return f'{v:.3g}'
def med(runs): return median(v['wall_seconds'] for v in runs)
def table(rows, widths):
    t = Table([[Paragraph(str(v), cell) for v in row] for row in rows],
              colWidths=widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf1f4')),
        ('LINEBELOW', (0, 0), (-1, 0), .6, ink),
        ('LINEBELOW', (0, -1), (-1, -1), .4, colors.HexColor('#9ba8b4')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    story.extend([t, Spacer(1, 7)])

p('Tiny Hanamikoji: three approaches to equilibrium', title)
p('CS6208 Homework 2 | 5 November 2026')
h('1. Problem, beliefs, and the equilibrium certificate')
p('I solve one round of Tiny Hanamikoji: winning pays +1, losing -1, and an unresolved round 0. The default game has neutral favors and a uniformly chosen starting player. Its tree contains 2,757 information sets and 3,672 sequences per player; 2,816 of the 12,832 stored payoff entries are nonzero. I compare counterfactual regret minimization (CFR), fictitious play (FP), and double oracle (DO), with a separate sequence-form LP reference.')
p('For the guided posterior task, I weight each history consistent with my information set by chance reach times opponent reach, combine identical state keys, then normalize. My own reach cancels under perfect recall. I prune incompatible observation prefixes and preserve the caller\'s state. Impossible conditioning events return an empty posterior.')
p('Let x and y be realization plans and A the chance-weighted payoff matrix for P1. Each plan records own-action reach, with every information set\'s outgoing weights summing to its parent weight. A backward best-response pass adds all child information-set values to their parent sequence. The reward vectors are Ay for P1 and -A<super>T</super>x for P2.')
p('<b>SPR(x,y) = max<sub>x\'</sub> x\'<super>T</super>Ay - min<sub>y\'</sub> x<super>T</super>Ay\'.</b> The saddle-point residual is the sum of the two unilateral improvement opportunities; I do not divide it by two. SPR &le; 0.002 therefore bounds each player\'s possible improvement by 0.002 expected payoff. It is not a win-rate error. Near-zero self-play payoff alone does not certify equilibrium.')
h('2. Three distinct algorithms')
p('<b>CFR [1].</b> At each information set I accumulate counterfactual regret q(I,a)-v(I), then normalize positive cumulative regrets, using uniform play when none is positive. Each iteration updates both players simultaneously. I return the uniform time average of realization plans; regret clipping, alternating updates, and linear averaging are disabled in this comparison.')
p('<b>Fictitious play [2].</b> Both players compute a full best response to the opponent\'s previous empirical mixture, then add their responses to their own averages. A uniform initial plan has weight one: after t updates, the mixture averages t+1 observations. This is simultaneous FP; it uses neither regret updates nor LPs.')
p('<b>Double oracle [3].</b> I start each candidate pool with a uniform plan, add global best responses to the current restricted equilibrium, and solve the expanded payoff matrix with two LPs. Duplicate plans are skipped. I return the current equilibrium mixture in realization form. Convergence uses full-game SPR, since a small restricted-game gap can hide profitable omitted strategies.')

page()
h('3. Common settings and timing protocol')
p('All three methods use a <b>2,000-iteration maximum</b>, a <b>0.002 full-SPR target</b>, and checks at entry, every 25 iterations, and exit. I stop at the first successful scheduled check. The cap is a practical computational budget, not a theoretical requirement or a guarantee of convergence. It suffices for every reported run; a different game or stricter target may need more iterations. The 25-step grid can delay the observed stopping time.')
p('I reuse the same extracted game, payoff scale, uniform starting profile, exact chance enumeration, and deterministic best-response tie rule. Algorithm-specific state still differs: CFR has zero regrets, FP has one prior observation, and DO has one seed per pool. Before timing each scenario, I warm every iterative method for two discarded iterations and warm the reference LP once. Three timing repetitions rotate algorithm order. No training seed or sampled deals are involved.')
p('Elapsed time uses a monotonic wall clock around solver construction and solve(), including updates, averaging, and exact SPR checks. It excludes tree extraction, warm-up, post-run independent validation, plotting, and export. Plot and table clocks have the same scope. LP timings include matrix assembly, both optimizations, and its certificate calculations.')
story.append(Image(str(RESULTS / 'convergence.png'), width=W, height=W * 2.9 / 6.3))
p('Figure 1. Default-game full SPR versus iteration and elapsed seconds, using each method\'s median-time repetition. The dotted line is 0.002. Curves end when the common target is met.', caption)
rows = [['Method', 'Stop iteration', 'Final SPR', 'Median (s)']]
for name, short in labels:
    runs = n['methods'][name]
    rows.append([short, runs[0]['iterations'], f(runs[0]['spr']), f'{med(runs):.3f}'])
table(rows, [W*.22, W*.26, W*.26, W*.26])
p('Table 1. All runs meet the target before the common cap. Their different stopping gaps reflect checkpoint spacing and convergence paths; equal iteration counts would not imply equal computation.', caption)

page()
h('4. Findings: iteration count and work are different')
a, b, d = [n['methods'][name] for name, _ in labels]
p(f'DO needs {a[0]["iterations"]/d[0]["iterations"]:.1f} times fewer iterations than CFR on the default game, yet takes {med(d)/med(a):.2f} times as long. Each expansion solves two restricted LPs. FP has a slightly lower observed median than CFR ({med(b):.3f} versus {med(a):.3f} s), but their timing ranges overlap ({min(v["wall_seconds"] for v in b):.3f}-{max(v["wall_seconds"] for v in b):.3f} and {min(v["wall_seconds"] for v in a):.3f}-{max(v["wall_seconds"] for v in a):.3f} s). Three repetitions do not establish a reliable speed advantage between them.')
rows = [['Method', 'Training work', 'SPR BR calls', 'Restricted LPs']]
for name, short in labels:
    q = n['methods'][name][0]
    work = f'{q["regret_sweeps"]:,} regret sweeps' if name == 'CFR' else f'{q["training_br_calls"]:,} BR calls'
    rows.append([short, work, q['diagnostic_br_calls'], q['restricted_lp_solves']])
table(rows, [W*.15, W*.37, W*.24, W*.24])
p('Table 2. Default-game work per repetition. Each SPR evaluation uses two global best responses (BRs); the exit evaluation repeats the successful checkpoint. CFR uses two regret sweeps per iteration. DO also solves its initial 1-by-1 game, explaining 252 LPs for 125 expansions.', caption)
p('DO stores 126 candidate plans per player at termination, while only 23 per player have mixture weight above 10<super>-12</super>. Candidate count and positive-weight equilibrium support are different quantities. LP and mixture costs depend on the growing candidate matrix, so an oracle iteration is not a constant-cost substitute for a CFR iteration.')
h('5. Same protocol on changed starting conditions')
rows = [['Method', 'Random start', 'P1 starts', 'P1 holds G3']]
for name, short in labels:
    rows.append([short] + [f'{r["scenarios"][s]["methods"][name][0]["iterations"]} / {med(r["scenarios"][s]["methods"][name]):.3f}' for s in scenarios])
rows.append(['LP value'] + [f'{0 if abs(r["scenarios"][s]["lp"][0]["value"]) < 1e-12 else r["scenarios"][s]["lp"][0]["value"]:.6f}' for s in scenarios])
table(rows, [W*.19, W*.27, W*.27, W*.27])
p('Table 3. Entries give stopping iteration / median seconds, except the LP-value row. The last case also fixes P1 as starter; G1 and G2 remain neutral. Every algorithm reaches SPR &le; 0.002 in all three cases.', caption)
p('The fixed-start trees have 1,297/1,460 information sets. FP has the lowest observed median with neutral fixed start; CFR is fastest when P1 holds G3. Thus the runtime ranking is not universal. DO reaches near-zero SPR in the fixed-start case but oversolves the target there. The positive LP values measure P(win)-P(loss) in one round, not win probability or multi-round match value.')

page()
h('6. Independent LP reference and submitted strategies')
p('With realization-flow constraints Ex=e and Fy=f, I solve the sequence-form primal-dual pair [4]:')
p('<b>P1:</b> maximize f<super>T</super>q subject to Ex=e, x &ge; 0, F<super>T</super>q &le; A<super>T</super>x.<br/><b>P2:</b> minimize e<super>T</super>p subject to Fy=f, y &ge; 0, E<super>T</super>p &ge; Ay.')
p('The dual vectors p and q are unrestricted, including in negative-value games. Sparse HiGHS programs use 10<super>-9</super> primal and dual feasibility tolerances, matching DO\'s restricted LPs. These tolerances control numerical feasibility, whereas 0.002 controls full-game strategic exploitability. I independently recompute SPR and check realization flows for every output.')
lp = n['lp'][0]
p(f'The default reference value is numerically zero. Its maximum flow residual is {max(lp["feasibility_p1"], lp["feasibility_p2"]):.2e}, maximum dual residual {max(lp["dual_feasibility_p1"], lp["dual_feasibility_p2"]):.2e}, and independent SPR {lp["independent_spr"]:.2e}. I submit these LP policies using the supplied save_agent API. Reloading preserves SPR {n["submitted"]["spr"]:.2e}, with maximum realization-weight change {n["submitted"]["max_realization_change"]:.2e}. This is a floating-point numerical certificate.')
p(f'The warmed full LP takes {med(n["lp"]):.3f} s at the median; the other two scenarios take {med(r["scenarios"]["first_player"]["lp"]):.3f} and {med(r["scenarios"]["favored_g3"]["lp"]):.3f} s. It is faster and more accurate than all three iterative methods on these small games. It is a reference calculation, not a fourth method with an artificial 2,000 LP-pivot limit. Default tree extraction takes {n["build_seconds"]:.3f} s separately; sharing this preprocessing keeps solver timing comparable.')
h('7. Exact evaluation against fixed opponents')
rows = [['Opponent', 'Nash as P1', 'Nash as P2']]
for label, values in n['baseline_payoffs'].items():
    rows.append([label.replace('Agent', ''), f'{values["nash_as_p1"]:+.4f}', f'{values["nash_as_p2"]:+.4f}'])
table(rows, [W*.46, W*.27, W*.27])
p('Table 4. Expected payoff to the submitted policy, computed by complete chance-weighted evaluation. There is no Monte Carlo sampling error. Equilibrium protects against arbitrary opponents but need not maximize exploitation of a particular heuristic. Seats may select different members of a nonunique equilibrium set.', caption)
p('The policy-table export covers both seats and both starting players with neutral first-round favors. A fallback used in later rounds is not part of this solved game and has no equilibrium guarantee. The report\'s claims concern the single-round policies only.')

page()
h('8. Verification and reproducibility')
p(f'All {test_count} tests pass. Posterior checks include analytic hidden-card probabilities, private-reserve signals, a separate joint-law oracle, impossible events, and state restoration. Best-response checks enumerate a branching treeplex and compare full Tiny responses with independent LPs. A live-tree traversal independently verifies sequence-form payoffs.')
p('Known matrix equilibria test reward signs and negative game values. Additional tests cover flow conservation, averaging on shared-parent branches, simultaneous FP updates and its initial prior, DO mixture weights and full-game stopping, continuation/reset, changed payoff refresh, and lazy SciPy imports. All four solution notebooks execute. Saved policies pass shape, player-ID, probability, and reload checks.')
p(f'I used Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1, the Python engine, and an Apple M4 with 16 GiB RAM on macOS 15.6. Repetitions measure timing variation; exact enumeration yields deterministic numerical trajectories. Peak process RSS is {r["peak_process_rss_mib"]:.1f} MiB across the whole benchmark, not a per-method memory result.')
p('All solutions, executed notebooks, scripts, measurements, this report, and the submission archive are isolated in jiayu_attempt. The shared starter\'s 80 files remain byte-identical, including the protected treeplex representation. The attempt README gives ordered reproduction commands; task4/experiment_config.json records the common controls. Results retain source hashes, full histories, work counts, and certificates. The archive wrapper preserves the grader\'s original top-level layout.')
h('9. Limits and interpretation')
p('The evidence supports different practical tradeoffs among regret updates, empirical best responses, and restricted-game optimization. DO saves iterations but its LP overhead matters here; no method dominates iterative runtime across all scenarios. The 2,000 cap and 0.002 target are stated experimental choices, not convergence theorems. More repetitions would sharpen timing comparisons, and larger games would be needed to assess scaling. All three implementations still enumerate the full game; this DO implementation does not remove that bottleneck. No result establishes performance on Medium/Full Hanamikoji or the multi-round match.')
h('References')
p('[1] M. Zinkevich, M. Johanson, M. Bowling, and C. Piccione. <i>Regret Minimization in Games with Incomplete Information.</i> NeurIPS 20, 2007. <link href="https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html" color="#203346">Proceedings</link>.')
p('[2] G. W. Brown. <i>Iterative Solution of Games by Fictitious Play.</i> Activity Analysis of Production and Allocation, chapter XXIV, 1951. <link href="https://cowles.yale.edu/sites/default/files/2022-09/m13-all.pdf" color="#203346">Cowles monograph</link>.')
p('[3] H. B. McMahan, G. J. Gordon, and A. Blum. <i>Planning in the Presence of Cost Functions Controlled by an Adversary.</i> ICML, 2003. <link href="https://www.cs.cmu.edu/~ggordon/mcmahan-ggordon-blum.icml2003.pdf" color="#203346">Paper</link>.')
p('[4] B. von Stengel. <i>Efficient Computation of Behavior Strategies.</i> Games and Economic Behavior 14(2), 220-246, 1996. <link href="https://doi.org/10.1006/game.1996.0050" color="#203346">doi:10.1006/game.1996.0050</link>.')

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Roman', 12)
    canvas.setFillColor(colors.HexColor('#526170'))
    canvas.drawRightString(A4[0] - 2.5*cm, 2.5*cm + 1, str(doc.page))
    canvas.restoreState()

doc = SimpleDocTemplate(str(ATTEMPT / 'report.pdf'), pagesize=A4,
                        leftMargin=2.5*cm, rightMargin=2.5*cm, topMargin=2.5*cm,
                        bottomMargin=2.5*cm+19, title='Tiny Hanamikoji: Three Approaches to Equilibrium',
                        author='', pageCompression=1)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
from pypdf import PdfReader
pdf = PdfReader(ATTEMPT / 'report.pdf')
assert len(pdf.pages) == 5, f'Report has {len(pdf.pages)} pages instead of five'
for pg in pdf.pages:
    assert abs(float(pg.mediabox.width) - A4[0]) < .1
    assert abs(float(pg.mediabox.height) - A4[1]) < .1
print(f'{ATTEMPT / "report.pdf"}: {len(pdf.pages)} A4 pages')
