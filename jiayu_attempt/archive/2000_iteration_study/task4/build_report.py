"""Build the Task 4 report (Direction D, at most five A4 pages) from the recorded measurements."""
from datetime import date
from fractions import Fraction
from pathlib import Path
from statistics import median
import json
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

ATTEMPT = Path(__file__).resolve().parents[1]
RESULTS = ATTEMPT / 'task4' / 'results'
r = json.loads((RESULTS / 'experiments.json').read_text())
c, S, an, env = r['configuration'], r['scenarios'], r['analysis'], r['environment']
n = S['neutral']
labels = [('CFR', 'CFR'), ('Fictitious play', 'FP'), ('Double oracle', 'DO')]
scenarios = ['neutral', 'first_player', 'second_player', 'favored_g3']
assert c['iterations'] == 2000 and c['spr_threshold'] == .002
assert not c['full_budget'] and c['check_every'] == 25 and c['repeats'] == 3
assert set(n['methods']) == {name for name, _ in labels} and list(S) == scenarios
tests_text = (RESULTS / 'tests.txt').read_text()
match = re.search(r'Ran (\d+) tests', tests_text)
assert match and '\nOK' in tests_text
test_count = int(match.group(1))

W = A4[0] - 5 * cm
ink = colors.HexColor('#203346')
body = ParagraphStyle('body', fontName='Times-Roman', fontSize=12, leading=14.4,
                      spaceAfter=6, alignment=TA_LEFT)
heading = ParagraphStyle('heading', parent=body, fontName='Times-Bold', spaceBefore=6,
                         spaceAfter=4, textColor=ink, keepWithNext=1)
# Used where the next paragraph may split across pages, so the heading never sits alone.
heading_free = ParagraphStyle('heading_free', parent=heading, keepWithNext=0)
title = ParagraphStyle('title', parent=heading, spaceBefore=0, spaceAfter=2)
caption = ParagraphStyle('caption', parent=body, leading=13.6)
cell = ParagraphStyle('cell', parent=body, leading=13.6, spaceAfter=0)
ref = ParagraphStyle('ref', parent=body, leading=13.6, spaceAfter=1)
story = []
def p(text, style=body): story.append(Paragraph(text, style))
def h(text, keep=True): p(text, heading if keep else heading_free)
def med(runs): return median(v['wall_seconds'] for v in runs)
def runs(s, m): return S[s]['methods'][m]
def it(s, m): return runs(s, m)[0]['iterations']
def sci(v, digits=2):
    if v == 0:
        return '0'
    mantissa, exponent = f'{v:.{digits - 1}e}'.split('e')
    return f'{mantissa}&times;10<super>{int(exponent)}</super>'
def table(rows, widths):
    t = Table([[Paragraph(str(v), cell) for v in row] for row in rows],
              colWidths=widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf1f4')),
        ('LINEBELOW', (0, 0), (-1, 0), .6, ink),
        ('LINEBELOW', (0, -1), (-1, -1), .4, colors.HexColor('#9ba8b4')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    story.extend([t, Spacer(1, 5)])
def figure(path, width):
    story.append(Image(str(path), width=W * width, height=W * width * 2.9 / 6.3))

# ---------------------------------------------------------------- derived numbers
f_inf = S['first_player']['infosets']
v1 = S['first_player']['lp'][0]['value']
v1_frac = Fraction(v1).limit_denominator(1000)
assert abs(S['second_player']['lp'][0]['value'] + v1) < 1e-12 and abs(n['lp'][0]['value']) < 1e-12
cfr_spr = {s: runs(s, 'CFR')[0]['spr'] for s in scenarios}
fp_spr = {s: runs(s, 'Fictitious play')[0]['spr'] for s in scenarios}
assert all(abs(d[s] - d['neutral']) <= 1e-14 * d['neutral'] and it(s, m) == it('neutral', m)
           for d, m in ((cfr_spr, 'CFR'), (fp_spr, 'Fictitious play'))
           for s in ('first_player', 'second_player')), 'Section 5.2 assumes identical CFR/FP results on the halves'
rates, lvc, dox = an['rates'], an['lp_versus_cfr'], an['double_oracle_exact']
do_n, do_f, do_s = it('neutral', 'Double oracle'), it('first_player', 'Double oracle'), it('second_player', 'Double oracle')
t = {(s, m): med(runs(s, m)) for s in scenarios for m, _ in labels}
lp_t = {s: med(S[s]['lp']) for s in scenarios}
lp0 = n['lp'][0]
lp_residual = max(lp0['feasibility_p1'], lp0['feasibility_p2'], lp0['dual_feasibility_p1'], lp0['dual_feasibility_p2'])
fastest_iterative = min(t[('neutral', m)] for m, _ in labels)
do_exact = dox['neutral']
lp_share = 100 * do_exact['restricted_lp_seconds'] / do_exact['wall_seconds']
lp_trace = do_exact['restricted_lp_trace']
lp_small = median(row[2] for row in lp_trace if 2 <= row[0] <= 10) * 1000
lp_large = median(row[2] for row in lp_trace[-10:]) * 1000
ratio_t = t[('neutral', 'Double oracle')] / t[('neutral', 'CFR')]
gb = lambda bytes_: f'{bytes_ / 2**30:.0f} GiB RAM' if bytes_ else 'unknown RAM'
machine = (f"{env['cpu']} ({env.get('cpu_count', '?')} cores, {gb(env['ram_bytes'])}) running "
           f"{env['platform'].split('-')[0]}, Python {env['python'].split()[0]}, NumPy {env['numpy']} and SciPy {env['scipy']}")
today = date.today()

# ---------------------------------------------------------------- title and 1
p('Alternative Nash Solvers for Tiny Hanamikoji', title)
p(f'CS6208 Homework 2, Task 4 (Direction D) | {today.day} {today:%B %Y}')
h('1. Introduction')
p(f'Task 4 follows Direction D: the Nash equilibrium of one round of Tiny Hanamikoji is computed with solvers other than counterfactual regret minimization (CFR) and compared with it. CFR only guarantees that the saddle-point residual decays as O(1/&radic;T), so it approaches equilibrium slowly and never reaches it. A round is small enough ({n["infosets"][0]:,} information sets per player) to be solved exactly as well, so every method can be judged against the true answer rather than another approximation. We study two families built on best responses rather than regrets, fictitious play (FP) and the double oracle (DO), use the sequence-form linear program (LP) as the exact reference, and ask why each method behaves as it does and where it would break down on larger games.')
p('The study extends Tasks 2 and 3: all methods share the Task 2 treeplex, sparse payoff matrix A and best response (BR); the saddle-point residual (SPR) is the accuracy measure; and the Task 3 CFR solver is the baseline. The exact LP equilibrium is also our Task 3 submission.')
p(f"With realization plans x and y, P1 earns x<super>T</super>Ay (+1 for a win, -1 for a loss, 0 for an unresolved round). Quality is measured by <b>SPR(x,y) = max<sub>x'</sub> x'<super>T</super>Ay - min<sub>y'</sub> x<super>T</super>Ay'</b> (not halved), the sum of both players' best unilateral gains, so SPR &le; 0.002 bounds each player's gain by 0.002. Because the starting player is the first chance event and is observed by both players, the default game (neutral favors, random starter) is a 50/50 mixture of two disjoint mirror-image subgames, \"P1 starts\" and \"P2 starts\" ({f_inf[0]:,}/{f_inf[1]:,} information sets for the starter/responder). The LP values the first mover's advantage at v = {v1_frac} &asymp; {v1:.3f}, so the default value is v/2 - v/2 = 0.")

# ---------------------------------------------------------------- 2
h('2. How each solver works and why it converges')
p(f"<b>CFR, the baseline [1].</b> The reward vector Ay weights each leaf by chance and opponent reach but not by own reach, so a backward pass over the treeplex gives each sequence its counterfactual value q(I,a). The regret R(I,a) accumulates q(I,a) - &Sigma;<sub>b</sub>&sigma;(I,b)q(I,b), and regret matching plays &sigma;(I,a) &prop; max(R(I,a),0). Total regret is bounded by the sum of these local O(&radic;T) regrets, so the average strategy, formed over realization plans so that each choice is weighted by its own reach, has SPR O(1/&radic;T). The current iterate need not converge: after {rates['CFR']['final_iteration']:,} iterations its SPR is still {rates['CFR']['last_iterate_spr']:.3f}, against {rates['CFR']['final_spr']:.4f} for the average.")
p("<b>Fictitious play [2].</b> Each player best-responds to the opponent's empirical average, which is then updated with weight 1/(t+1) (a uniform plan is the first observation). Averaging realization plans is exactly the reach-weighted update of extensive-form FP [5]. FP converges in two-player zero-sum games [6], but without a useful rate guarantee [7]. Since the BRs of iteration t answer the current averages, u(BR<sub>1</sub>,y<sub>t</sub>) - u(x<sub>t</sub>,BR<sub>2</sub>) is exactly SPR(x<sub>t</sub>,y<sub>t</sub>), so the stopping test is free.")
p("<b>Double oracle [3].</b> Each player keeps a restricted set of pure plans, initially the uniform plan. Each iteration solves the restricted matrix game exactly with two small LPs, mixes the plans in realization space into (x<sub>t</sub>,y<sub>t</sub>), and calls both BR oracles against it. The oracle values give the exact full-game certificate SPR(x<sub>t</sub>,y<sub>t</sub>) = u(BR<sub>1</sub>(y<sub>t</sub>),y<sub>t</sub>) - u(x<sub>t</sub>,BR<sub>2</sub>(x<sub>t</sub>)), which is the termination criterion; otherwise the responses join the sets. The restricted game's own gap is zero by construction and certifies nothing. If neither response is new, (x<sub>t</sub>,y<sub>t</sub>) is an exact equilibrium, and finitely many pure plans guarantee termination. Ties (interchangeable suits G1/G2, identical split cards) make the oracle return new but equally good plans, however, so a profile is called exact once its certified SPR is at most 10<super>-9</super>, the LP accuracy.")
p("<b>Sequence-form LP [4, 8].</b> In max<sub>x</sub> min<sub>y</sub> x<super>T</super>Ay over the treeplex polytopes Ex=e, Fy=f, x,y &ge; 0, the inner BR problem min<sub>y</sub> {(A<super>T</super>x)<super>T</super>y : Fy=f, y &ge; 0} is replaced by its dual max<sub>q</sub> {f<super>T</super>q : F<super>T</super>q &le; A<super>T</super>x}. P1's problem becomes one LP, maximize f<super>T</super>q subject to Ex=e, x &ge; 0 and F<super>T</super>q &le; A<super>T</super>x, and P2's is symmetric. Both are linear in the number of sequences, their objectives bound the value from both sides, and HiGHS solves them at 10<super>-9</super> tolerances.")

# ---------------------------------------------------------------- 3
h('3. How correctness is verified')
p(f'Correctness is checked at three levels, always against an independent computation (Table 1): the building blocks against brute force on problems with known answers; each solver against known equilibria, with its own certificates against an independent evaluator; and the results on the real game against each other, through independent SPR recomputation, agreement with the LP value and the symmetries of the game. Further tests cover continuation, reset and payoff changes. All {test_count} unit tests in task4/ pass.')
rows = [['Check', 'Independent reference', 'Outcome'],
        ['Best responses', 'All pure plans of a test treeplex; a separate LP', 'Equal (10<super>-11</super>; 10<super>-9</super>)'],
        ['Payoff x<super>T</super>Ay', 'Traversal of the live game tree', 'Equal (10<super>-11</super>)'],
        ['CFR, FP, DO', '2&times;2 games with known equilibria', 'All converge; DO exact'],
        ['FP, DO certificates', 'Independent SPR at every iteration', 'Equal (10<super>-12</super>)'],
        ['Sequence-form LP', 'Primal and dual residuals, duality gap', f'&le; {sci(lp_residual, 1)}'],
        ['Benchmark outputs', 'Flow constraints; independent SPR', 'Errors &lt; 10<super>-8</super>; &lt; 10<super>-10</super>'],
        ['Game structure', 'Mirror symmetry; decomposition (5.2)', f'Values &plusmn;{v1_frac} and 0']]
table(rows, [W * .26, W * .46, W * .28])
p('Table 1. Correctness checks: the first five are unit tests, the last two check the benchmark results.', caption)

# ---------------------------------------------------------------- 4
h('4. Experimental setup')
p(f'All iterative methods share the game, a {c["iterations"]:,}-iteration cap, a {c["spr_threshold"]} SPR target, exact chance enumeration, first-maximizer tie-breaking, warm-up and {c["repeats"]} timed repetitions; the runs are deterministic, so repetitions only measure timing noise. Timed work includes solver construction and certificates but not tree extraction ({n["build_seconds"]:.2f} s). FP and DO certify every iteration for free, whereas the CFR average needs two extra BRs per test and is therefore tested every {c["check_every"]} iterations, overshooting its first crossing by at most {c["check_every"] - 1}. The scenarios are the default game, its two halves, and "G3 favors P1" (P1 starts holding the G3 favor). The solvers are in student_attempts/ (solver.py, fictitious_play.py, double_oracle.py, lp_solver.py); the benchmark, tests, figures and report generator are in task4/. Timings come from {machine}.')

# ---------------------------------------------------------------- 5
h('5. Results')
p(f"<b>5.1 Iterations are not work.</b> Figure 1 and Table 2 summarize the default game. DO needs {it('neutral', 'CFR') / do_n:.0f}&times; fewer iterations than CFR, but each iteration solves two LPs that grow with the restricted sets: {lp_share:.0f}% of DO's time is LP solving, and DO takes {ratio_t:.2f}&times; as long as CFR (reading P2's mixture from the duals of P1's restricted LP would roughly halve the LP cost). CFR and FP cost about the same per iteration; between iterations 100 and {rates['CFR']['final_iteration']:,} their SPR decays like T<super>{rates['CFR']['slope_100_to_end']:.2f}</super> and T<super>{rates['Fictitious play']['slope_100_to_end']:.2f}</super>, faster than the worst-case T<super>-1/2</super> of CFR, a guarantee FP lacks. The exact LP is {fastest_iterative / lp_t['neutral']:.0f}&times; faster than the fastest iterative method.")
figure(RESULTS / 'convergence.png', .8)
p('Figure 1. Default game: certified SPR against iterations and elapsed time for the median-time run of each method; the dotted line is the 0.002 target.', caption)
rows = [['Method', 'Stop iteration', 'SPR at stop', 'Median time (s)', 'ms per iteration', 'Training work']]
for name, short in labels:
    q = runs('neutral', name)[0]
    work = (f'{q["regret_sweeps"]:,} regret sweeps' if name == 'CFR' else
            f'{q["training_br_calls"]:,} BRs' + (f', {q["restricted_lp_solves"]} LPs' if name == 'Double oracle' else ''))
    rows.append([short, f'{q["iterations"]:,}', sci(q['spr']), f'{t[("neutral", name)]:.3f}',
                 f'{1000 * t[("neutral", name)] / q["iterations"]:.2f}', work])
rows.append(['LP', '-', sci(lp0['independent_spr']), f'{lp_t["neutral"]:.3f}', '-', '2 sparse LPs'])
table(rows, [W * .12, W * .14, W * .15, W * .15, W * .15, W * .29])
cert = runs('neutral', 'CFR')[0]['diagnostic_br_calls']
p(f'Table 2. Default game. Training work excludes certificate BRs (CFR {cert}; FP and DO 2).', caption)
rows = [['', 'Default game', 'P1 starts', 'P2 starts', 'G3 favors P1']]
for name, short in labels:
    rows.append([short] + [f'{it(s, name):,} / {t[(s, name)]:.3f}' for s in scenarios])
rows.append(['DO exact'] + [f'{dox[s]["iterations"]} ({dox[s]["active_support_counts"][0]}/{dox[s]["active_support_counts"][1]})' for s in scenarios])
rows.append(['LP'] + [f'{S[s]["lp"][0]["value"] + 0:+.4f} / {lp_t[s]:.3f}' for s in scenarios])
table(rows, [W * .16, W * .21, W * .21, W * .21, W * .21])
p('Table 3. Iterations to SPR &le; 0.002 / median seconds. "DO exact" continues to a certified SPR &le; 10<super>-9</super>; in brackets, the plans with positive weight (P1/P2). The LP row gives the game value / median seconds.', caption)
p(f"<b>5.2 Decomposition exposes a structural weakness of DO.</b> CFR and FP stop at the same iteration with the same SPR (to 15 digits) on the default game and on each half (Table 3, Figure 2a): regret matching and BRs act separately on disjoint subtrees, ignore the 1/2 payoff scaling, and the halves are mirror images. DO instead needs {do_n} iterations on the default game but only {do_f} and {do_s} on the halves ({dox['neutral']['iterations']} against {dox['first_player']['iterations']} and {dox['second_player']['iterations']} for exactness), nearer their sum than their maximum. Its restricted strategies are whole-game pure plans, so a mixture must put the same weights on both halves and the opponent's columns are paired likewise; the restricted game is far poorer than the product of the restricted games of the halves. DO on each half separately would need only max({do_f}, {do_s}) rounds, and a sequence-form DO [9], whose restricted game is a sub-treeplex, avoids the coupling by construction. In time, DO beats CFR on each half ({t[('first_player', 'Double oracle')]:.2f} vs {t[('first_player', 'CFR')]:.2f} s) but {'loses' if ratio_t > 1 else 'wins'} on the union ({t[('neutral', 'Double oracle')]:.2f} vs {t[('neutral', 'CFR')]:.2f} s), also because each restricted solve grows from about {lp_small:.0f} ms with a few plans to {lp_large:.0f} ms with over 100 (Figure 2b).")
figure(RESULTS / 'decomposition.png', .8)
p('Figure 2. (a) Certified SPR of CFR and DO on the default game and its halves (the CFR curves coincide; exact DO certificates are drawn at the lower edge). (b) Time of one restricted solve (two LPs) against the number of stored plans, in the runs to exactness.', caption)
p(f"<b>5.3 DO is exact, but its path is fragile.</b> DO certifies 0.002 at iteration {do_exact['target_iteration']} and exactness at {do_exact['iterations']}, with {do_exact['support_counts'][0]} plans per player of which only {do_exact['active_support_counts'][0]}/{do_exact['active_support_counts'][1]} carry weight; CFR and FP get nowhere near exactness within the budget. The small active support explains the few iterations; equilibria that mix over many pure plans would need many more. The certified SPR of DO is not monotone (Figure 1): a new plan can make the next restricted equilibrium more exploitable. Two caveats affect any iteration count. DO follows whichever restricted equilibrium the LP solver returns: an earlier version of our DO, with identical package versions, had SPR 1.1&times;10<super>-3</super> at iteration 125 on macOS but 1.3&times;10<super>-14</super> on Linux. FP depends on how exact ties are broken: changing only the floating-point summation order inside the BR moved its first certified iteration from 1,493 to {it('neutral', 'Fictitious play'):,}, while CFR, which never takes an argmax, was unchanged.")
fp1 = lvc['first_player']
nu = fp1['nonunique']
p(f"<b>5.4 Does the LP solution match the CFR solution?</b> The values agree within the certificate: with P1 starting, the LP value is {fp1['lp_value']:.6f} and the CFR average earns {fp1['cfr_value']:.6f}, a difference of {sci(abs(fp1['cfr_value'] - fp1['lp_value']))}, below its SPR of {sci(fp1['cfr_spr'])}. The strategies differ, and need not agree: the LP returns a vertex mixing at only {fp1['p1']['mixed_infosets_lp']} and {fp1['p2']['mixed_infosets_lp']} reachable information sets (P1/P2), the CFR average at {fp1['p1']['mixed_infosets_cfr']} and {fp1['p2']['mixed_infosets_cfr']}, and at some commonly reached information sets their action distributions are almost disjoint. An LP over the optimal face of P1 proves that the equilibrium is not unique: at the first decision of P1 when {nu['situation']}, the probability of {nu['description']} can be anything in [{nu['optimal_min']:.3f}, {nu['optimal_max']:.3f}] (LP {nu['lp']:.3f}, CFR {nu['cfr']:.3f}). What should match is the quality of each strategy: SPR splits into the separate exploitabilities of the two players (hence zero-sum equilibrium strategies are interchangeable), so pairing the LP strategy of P1 with the CFR strategy of P2 isolates the P2 part ({sci(fp1['cross_spr_lpP1_cfrP2'])}) and the reverse pairing isolates the P1 part ({sci(fp1['cross_spr_cfrP1_lpP2'])}). Both CFR strategies are near-optimal, that of P2 less so.")
r_cfr = t[('neutral', 'CFR')] / t[('first_player', 'CFR')]
r_fp = t[('neutral', 'Fictitious play')] / t[('first_player', 'Fictitious play')]
r_lp = lp_t['neutral'] / lp_t['first_player']
r_do_it = do_n / do_f
r_do_t = t[('neutral', 'Double oracle')] / t[('first_player', 'Double oracle')]
p(f"<b>5.5 Where each method breaks down as the game grows.</b> Doubling the game, from one half to the default game, leaves the CFR and FP iteration counts unchanged and multiplies their time by {f'about {r_cfr:.1f}' if f'{r_cfr:.1f}' == f'{r_fp:.1f}' else f'{r_cfr:.1f} and {r_fp:.1f}'}, slightly below the 2&times; growth of A; the LP time grows {r_lp:.1f}&times;, the DO iterations {r_do_it:.1f}&times; and the DO time {r_do_t:.1f}&times;. A CFR or FP iteration costs O(nnz(A)) time and O(#sequences) memory, so both scale to any tree that can be enumerated and then need sampling (MCCFR) or abstraction; the weak rate guarantee of FP makes it the riskier one. The LP is fastest here, but its solver time and factorization memory grow superlinearly, so memory fails first on large trees. DO pays two BRs and a k&times;k LP per iteration and needs at least about as many iterations as the support of its equilibrium (exponentially many in the worst case), so it breaks down when equilibria use many pure plans or the game has many weakly coupled parts. Medium Hanamikoji was not attempted; full-tree enumeration would be the first bottleneck.")

# ---------------------------------------------------------------- 6
h('6. Conclusions')
p('At this size the exact LP dominates: it is faster than every iterative method and exact. DO wins on iterations and exactness when equilibria have small support, but its normal-form restricted game ignores structure that CFR and FP exploit for free, and the iteration counts of FP and DO depend on tie-breaking and LP details. The cap and target are experimental choices, not theorems. Natural next steps are LP duals and warm starts for DO, a sequence-form DO, and, since the LP solves a round in a few hundredths of a second, using it inside value iteration over favor states for the multi-round game.')

h('References')
refs = [
    '[1] M. Zinkevich et al. Regret minimization in games with incomplete information. NeurIPS, 2007.',
    '[2] G. W. Brown. Iterative solution of games by fictitious play. Activity Analysis of Production and Allocation, 1951.',
    '[3] H. B. McMahan, G. J. Gordon, A. Blum. Planning in the presence of cost functions controlled by an adversary. ICML, 2003.',
    '[4] B. von Stengel. Efficient computation of behavior strategies. Games Econ. Behav. 14(2):220-246, 1996.',
    '[5] J. Heinrich, M. Lanctot, D. Silver. Fictitious self-play in extensive-form games. ICML, 2015.',
    '[6] J. Robinson. An iterative method of solving a game. Annals of Mathematics 54(2):296-301, 1951.',
    '[7] C. Daskalakis, Q. Pan. A counter-example to Karlin\'s strong conjecture for fictitious play. FOCS, 2014.',
    '[8] D. Koller, N. Megiddo, B. von Stengel. Efficient computation of equilibria for extensive two-person games. Games Econ. Behav. 14(2):247-259, 1996.',
    '[9] B. Bosansky et al. An exact double-oracle algorithm for zero-sum extensive-form games with imperfect information. JAIR 51:829-866, 2014.',
]
for text in refs:
    p(text, ref)

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Roman', 12)
    canvas.setFillColor(colors.HexColor('#526170'))
    canvas.drawRightString(A4[0] - 2.5*cm, 2.5*cm + 1, str(doc.page))
    canvas.restoreState()

doc = SimpleDocTemplate(str(ATTEMPT / 'report.pdf'), pagesize=A4,
                        leftMargin=2.5*cm, rightMargin=2.5*cm, topMargin=2.5*cm,
                        bottomMargin=2.5*cm+19, title='Alternative Nash Solvers for Tiny Hanamikoji',
                        author='', pageCompression=1)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
from pypdf import PdfReader
pdf = PdfReader(ATTEMPT / 'report.pdf')
assert len(pdf.pages) <= 5, f'Report has {len(pdf.pages)} pages; the limit is five'
for pg in pdf.pages:
    assert abs(float(pg.mediabox.width) - A4[0]) < .1
    assert abs(float(pg.mediabox.height) - A4[1]) < .1
print(f'{ATTEMPT / "report.pdf"}: {len(pdf.pages)} A4 pages')
