"""Build my five-page Direction D report from the full 20,000-iteration study."""
from datetime import date
from fractions import Fraction
from pathlib import Path
from statistics import median
import json
import gzip
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
c, S, an, env = r['configuration'], r['scenarios'], r['analysis'], r['environment']
n = S['neutral']
labels = [('CFR', 'CFR'), ('Fictitious play', 'FP'), ('Double oracle', 'DO')]
scenarios = ['neutral', 'first_player', 'second_player', 'favored_g3']
assert c['iterations'] == 20000 and c['spr_threshold'] == .002 and c['full_budget']
assert c['check_every'] == 25 and c['repeats'] == 3
assert set(n['methods']) == {name for name, _ in labels} and list(S) == scenarios
for s in S.values():
    for records in s['methods'].values():
        assert len(records) == 3 and all(q['iterations'] == 20000 for q in records)
        assert all(q['first_target'] is not None for q in records)
        assert len({q['first_target']['iteration'] for q in records}) == 1
        assert max(q['first_target']['spr'] for q in records) - min(q['first_target']['spr'] for q in records) < 1e-12
tests_text = (RESULTS / 'tests.txt').read_text()
match = re.search(r'Ran (\d+) tests', tests_text)
assert match and '\nOK' in tests_text
test_count = int(match.group(1))

W = A4[0] - 5 * cm
ink = colors.HexColor('#203346')
body = ParagraphStyle('body', fontName='Times-Roman', fontSize=12, leading=14.1, spaceAfter=6, alignment=TA_LEFT)
heading = ParagraphStyle('heading', parent=body, fontName='Times-Bold', spaceBefore=7, spaceAfter=5,
                         textColor=ink, keepWithNext=1)
title = ParagraphStyle('title', parent=heading, spaceBefore=0, spaceAfter=7)
caption = ParagraphStyle('caption', parent=body, leading=13.6)
cell = ParagraphStyle('cell', parent=body, leading=13.4, spaceAfter=0)
ref = ParagraphStyle('ref', parent=body, leading=13.6, spaceAfter=5)
story = []
def p(text, style=body): story.append(Paragraph(text, style))
def h(text): p(text, heading)
def page(): story.append(PageBreak())
def runs(s, m): return S[s]['methods'][m]
def med(records): return median(q['wall_seconds'] for q in records)
def hit(s, m): return runs(s, m)[0]['first_target']
def hit_time(s, m): return median(q['first_target']['elapsed_seconds'] for q in runs(s, m))
def checkpoint(q, t): return next(h for h in q['history'] if h['iteration'] == t)
def sci(v, digits=2):
    if abs(v) < 1e-19: return '0'
    mantissa, exponent = f'{v:.{digits - 1}e}'.split('e')
    return f'{mantissa}&times;10<super>{int(exponent)}</super>'
def table(rows, widths):
    t = Table([[Paragraph(str(v), cell) for v in row] for row in rows], colWidths=widths,
              hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf1f4')),
        ('LINEBELOW', (0, 0), (-1, 0), .6, ink),
        ('LINEBELOW', (0, -1), (-1, -1), .4, colors.HexColor('#9ba8b4')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    story.extend([t, Spacer(1, 6)])
def figure(name): story.append(Image(str(RESULTS/name), width=W, height=W*2.9/6.3))

lp0 = n['lp'][0]
dox, lvc, rates = an['double_oracle_exact'], an['lp_versus_cfr'], an['rates']
assert all(q['spr'] <= 1e-9 for q in dox.values())
do_final = n['methods']['Double oracle'][0]
with gzip.open(RESULTS / do_final['history_file'], 'rt') as stream:
    do_history = [json.loads(line) for line in stream]
do_stable_from = next(q['iteration'] for q in do_history if q['restricted_lp_solves'] == do_final['restricted_lp_solves'])
assert all(abs(q['spr'] - do_final['spr']) < 1e-12 for q in do_history if q['iteration'] >= do_stable_from)
ram_label = f"{env['ram_bytes']//2**30} GiB RAM" if env['ram_bytes'] else 'RAM unavailable'
first_value = S['first_player']['lp'][0]['value']
first_fraction = Fraction(first_value).limit_denominator(1000)
today = date.today()

p('Alternative Nash Solvers for Tiny Hanamikoji', title)
p(f'CS6208 Homework 2, Task 4 (Direction D) | {today.day} {today:%B %Y}')
h('1. Question and equilibrium certificate')
p('I compare counterfactual regret minimization (CFR), fictitious play (FP), and double oracle (DO) on the same single-round game. The question is how their convergence paths and computational work differ. All main runs now complete 20,000 iterations, retaining intermediate measurements; a sequence-form linear program (LP) supplies an independent numerical equilibrium reference and the submitted Task 3 policies.')
p(f'The default game has neutral favors, a random starting player, {n["infosets"][0]:,} information sets and {n["sequences"][0]:,} sequences per player. Payoffs are +1 for a win, -1 for a loss, and 0 for an unresolved round. Let x,y be realization plans and A the chance-weighted payoff matrix for P1. Each plan conserves own-action reach at every information set.')
p('<b>SPR(x,y) = max<sub>x\'</sub> x\'<super>T</super>Ay - min<sub>y\'</sub> x<super>T</super>Ay\'.</b> This full saddle-point residual is the sum of the two unilateral improvement opportunities, without division by two. SPR &le; 0.002 bounds each player\'s possible expected-payoff gain by 0.002; it is not a win-rate error. Self-play payoff close to the game value alone does not establish equilibrium.')
h('2. Methods, update conventions, and convergence')
p('<b>CFR [1].</b> Counterfactual values weight chance and opponent reach, without multiplying by own prefix reach. I accumulate local regret q(I,a)-v(I), then normalize positive regrets, using uniform play when all are nonpositive. Both players update simultaneously. The returned strategy averages pre-update realization plans uniformly. The average has an O(1/&radic;T) worst-case SPR bound; finite-time exact equilibrium is not guaranteed. CFR+, alternation, and linear averaging are disabled here.')
p("<b>Fictitious play [2].</b> Both players best-respond to the opponent's previous empirical mixture and update simultaneously. A uniform prior has weight one, so the average after t updates contains t+1 observations. Realization-plan mixtures preserve own reach and implement normal-form FP without enumerating pure plans. Its empirical gap tends to zero in finite two-player zero-sum games. The two update best responses also certify SPR before the next update.")
p('<b>Double oracle [3].</b> Pools start with one mixed uniform plan and expand with pure global best responses. Two LPs solve each changed restricted game; mixtures are formed in realization space. If neither oracle finds a new plan, the profile is a full-game equilibrium under exact solves. Otherwise a new pure plan is added; finiteness guarantees eventual certification. In the full-budget experiment, real oracle rounds continue after certification, reusing unchanged restricted equilibria without redundant LPs.')
p('<b>LP [4].</b> With flow constraints Ex=e and Fy=f, dualizing the opponent\'s response gives P1: maximize f<super>T</super>q subject to Ex=e, x &ge; 0, F<super>T</super>q &le; A<super>T</super>x. P2\'s symmetric LP minimizes e<super>T</super>p with Fy=f, y &ge; 0 and E<super>T</super>p &ge; Ay. The dual vectors are unrestricted. Sparse HiGHS solves both with 10<super>-9</super> feasibility tolerances.')

page()
h('3. Experimental protocol and intermediate evidence')
p('<b>Full-budget comparison.</b> Each of the three methods completes 20,000 outer iterations in four scenarios, with three fresh timing repetitions: 36 runs in total. Early stopping is disabled. The 0.002 target is retained as a retrospective first observed crossing, separately from final SPR and total runtime. Equal iteration counts measure equal outer-iteration budgets, not equal amounts of computation.')
p('The scenarios are the default neutral game, neutral games with P1 or P2 fixed as starter, and P1 starting with the G3 favor. Every method uses exact weighted chance enumeration, the same payoff scale and deterministic tie handling. Uniform initial profiles are shared, although CFR starts with zero regrets, FP with one prior observation, and DO with one candidate per player.')
p('Before timing each scenario I discard a two-iteration warm-up of every iterative method and one full LP solve. Three repetitions rotate method order. Wall time includes construction, updates, averaging, logging, and exact certificates; it excludes tree extraction, warm-up, file export, and subsequent independent validation. Repetitions measure timing noise, not sampling uncertainty. The reference LP is timed separately without an outer-iteration budget.')
p('CFR records SPR every 25 iterations plus entry/exit. FP and DO save the certificate from every oracle iteration. Full measured histories are stored as compressed JSON Lines in task4/results/histories/; experiments.json retains the common 25-step grid, the first crossing, and endpoint. Explicit checkpoints include 2,000, 5,000, 10,000, and 20,000. No intermediate curve is reconstructed from final values.')
p('CFR\'s first observed crossing may miss an earlier transient dip between checks. FP/DO crossings use their per-iteration certificates. Auxiliary analyses are labelled separately: DO stops at numerical SPR &le; 10<super>-9</super>, and LP-versus-CFR comparisons stop CFR at 0.002. They do not replace the full-budget results.')
h('4. How correctness is verified')
rows = [['Check', 'Independent evidence'],
        ['Posterior (6 tests)', 'Analytic hidden-card probabilities, full joint histories, unreachable events and state restoration'],
        ['Best response / payoff', 'Exhaustive pure plans and independent response LPs; live game-tree payoff traversal'],
        ['CFR, FP and DO', 'Known matrix equilibria, realization flows, continuation/reset and changed payoffs'],
        ['Full-budget execution', 'Real oracle calls after DO stabilizes; actual iteration totals, complete histories and saved checkpoints'],
        ['Final strategies / LP', 'Reloaded probabilities and SPR; primal/dual residuals and gap']]
table(rows, [W*.32, W*.68])
p(f'All {test_count} tests pass. Every benchmark output is checked independently for flow error below 10<super>-8</super> and SPR agreement within 10<super>-10</super>. Full-budget runs must report exactly 20,000 iterations. The protected treeplex and all 80 shared starter files remain unchanged. Source hashes and the environment are recorded with the results.', caption)

page()
h('5. Default-game results: full budget and first crossing')
figure('convergence.png')
p('Figure 1. Recorded full-budget SPR versus iterations and constructor-plus-solve wall time, using each method\'s median-total-time repetition. Values below 10<super>-8</super> are drawn at that floor for readability; raw certificates are preserved. The dotted line is the 0.002 target.', caption)
rows = [['Method', 'SPR at 2,000', 'SPR at 20,000', 'Full time (s)', 'First target iteration', 'Time there (s)']]
for name, short in labels:
    q = runs('neutral', name)[0]
    first = hit('neutral', name)
    rows.append([short, sci(checkpoint(q,2000)['spr']), sci(q['spr']),
                 f'{med(runs("neutral",name)):.3f}', f'{first["iteration"]:,}',
                 f'{hit_time("neutral",name):.3f}'])
rows.append(['LP', '-', sci(lp0['independent_spr']), f'{med(n["lp"]):.3f}', '-', '-'])
table(rows, [W*.11, W*.18, W*.18, W*.17, W*.19, W*.17])
p('Table 1. Endpoint accuracy, median full runtime, and first recorded SPR &le; 0.002 with its median timestamp. Crossings are historical observations from uninterrupted runs. LP is a separate reference with no outer-iteration budget.', caption)
cr=runs('neutral','CFR')[0]
fp=runs('neutral','Fictitious play')[0]
d=runs('neutral','Double oracle')[0]
p(f'From 2,000 to 20,000 iterations, CFR\'s final SPR decreases by a factor of {checkpoint(cr,2000)["spr"]/cr["spr"]:.2f}; FP\'s decreases by {checkpoint(fp,2000)["spr"]/fp["spr"]:.2f}. Neither endpoint is numerically exact. DO already meets the 10<super>-9</super> certificate in its separate precision run at iteration {dox["neutral"]["iterations"]}; the full run performs no further restricted LPs after iteration {do_stable_from}, repeatedly certifying its unchanged mixture. Its 20,000-step time is therefore not the cost required to obtain that equilibrium.')
p(f'The full runs perform {cr["regret_sweeps"]:,} CFR regret sweeps, {fp["training_br_calls"]:,} FP training best responses, and {d["training_br_calls"]:,} DO training best responses plus {d["restricted_lp_solves"]:,} individual restricted LPs. DO does not solve two fresh LPs on every late iteration. The finite-range fitted SPR slopes from iteration 100 are {rates["CFR"]["slope_100_to_end"]:.2f} for CFR and {rates["Fictitious play"]["slope_100_to_end"]:.2f} for FP; these observations are not asymptotic-rate guarantees.')

page()
h('6. Starting conditions, structure, and oracle cost')
rows=[['Scenario','CFR','FP','DO']]
for s,label in [('neutral','Random start'),('first_player','P1 starts'),('second_player','P2 starts'),('favored_g3','P1 holds G3')]:
    rows.append([label]+[f'{hit(s,m)["iteration"]:,} / {hit_time(s,m):.3f}' for m,_ in labels])
table(rows,[W*.25,W*.25,W*.25,W*.25])
p('Table 2. First observed target iteration / median seconds, extracted from full histories. Every method reaches the target in every scenario; the final iteration remains 20,000 in all cases.', caption)
figure('decomposition.png')
p('Figure 2. (a) Early certificates on the default game and its two fixed-starter halves; values below 10<super>-8</super> share the lower edge. (b) Cost of a restricted solve, comprising two LPs, in separate DO runs to SPR &le; 10<super>-9</super>.', caption)
p(f'The random-starter game is a 50/50 mixture of disjoint, observed starting-player subgames. Their LP values are +{first_fraction} and -{first_fraction}, giving default value zero. At the target, CFR needs {hit("neutral","CFR")["iteration"]:,} iterations on the default game and {hit("first_player","CFR")["iteration"]:,}/{hit("second_player","CFR")["iteration"]:,} on its halves; FP needs {hit("neutral","Fictitious play")["iteration"]:,} versus {hit("first_player","Fictitious play")["iteration"]:,}/{hit("second_player","Fictitious play")["iteration"]:,}. Their local updates exploit the disjoint subtrees naturally.')
p(f'DO needs {hit("neutral","Double oracle")["iteration"]} iterations to the target on the default game versus {hit("first_player","Double oracle")["iteration"]}/{hit("second_player","Double oracle")["iteration"]} on the halves. Its candidates are whole-game plans: the restricted mixtures couple decisions in both halves until enough candidates are added. This helps explain why oracle iteration counts change even when the component games remain simple. It is an empirical structural diagnosis, not a general scaling theorem.')
dex=dox['neutral']
p(f'The default precision run spends {100*dex["restricted_lp_seconds"]/dex["wall_seconds"]:.1f}% of elapsed time in restricted LP solves and stores {dex["support_counts"][0]}/{dex["support_counts"][1]} candidate plans, with {dex["active_support_counts"][0]}/{dex["active_support_counts"][1]} weights above 10<super>-12</super>. Stored candidates and positive-weight support are different quantities. Full-budget DO can continue discovering tied plans after first becoming numerically unexploitable. Its path depends on oracle tie-breaking and the restricted equilibrium returned by the numerical LP solver.')

page()
h('7. LP comparison, submitted policies, and limitations')
fp1=lvc['first_player']
nu=fp1['nonunique']
p(f'In the separate target-accuracy comparison with P1 starting, the LP value is {fp1["lp_value"]:.6f}, while CFR earns {fp1["cfr_value"]:.6f}. Their difference, {sci(abs(fp1["cfr_value"]-fp1["lp_value"]))}, lies below CFR\'s SPR of {sci(fp1["cfr_spr"])}. Their action probabilities need not coincide: an LP over P1\'s optimal face lets the probability of {nu["description"]}, when {nu["situation"]}, range from {nu["optimal_min"]:.3f} to {nu["optimal_max"]:.3f}. This explicitly demonstrates equilibrium nonuniqueness.')
p(f'For the default game, the LP\'s maximum flow residual is {sci(max(lp0["feasibility_p1"],lp0["feasibility_p2"]))} and independent SPR is {sci(lp0["independent_spr"])}. The submitted NPZ files still contain the LP policies, saved and reloaded through the supplied API; their reloaded SPR is {sci(n["submitted"]["spr"])}. Increasing the experimental budget does not change their solver provenance. The certificate is numerical, not symbolic exact arithmetic.')
p('The main comparison answers accuracy after a fixed number of outer updates. Historical first-target timestamps answer a different question: effort to reach useful precision. In particular, post-convergence DO certification is deliberate experimental work, not work needed for deployment. A finite update budget and a common target do not make per-iteration work equal; both accuracy and elapsed time must be reported.')
p('All methods still use a fully enumerated Tiny game. Larger games would stress tree memory, LP factorization, or DO candidate growth; these experiments do not establish which resource fails first. No Medium/Full or multi-round claim is made. A neutral first-round policy table requires fallback play elsewhere, without an equilibrium guarantee. Three timing repetitions are insufficient for strong claims about small speed differences.')
p(f'Measurements use {env["cpu"]}, {ram_label}, {env["platform"]}, Python {env["python"].split()[0]}, NumPy {env["numpy"]}, and SciPy {env["scipy"]}. Peak process RSS is {r["peak_process_rss_mib"]:.1f} MiB, recorded during the main benchmark, including its histories; it is not a per-solver memory comparison. The attempt README records the reproduction sequence. Source code, compressed intermediate histories, checkpoint summaries, tests, and figures accompany this report; the earlier 2,000-iteration study is archived separately.')
h('References')
p('[1] M. Zinkevich, M. Johanson, M. Bowling, and C. Piccione. <i>Regret Minimization in Games with Incomplete Information.</i> NeurIPS 20, 2007. <link href="https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html" color="#203346">Proceedings</link>.',ref)
p('[2] J. Robinson. <i>An Iterative Method of Solving a Game.</i> Annals of Mathematics 54(2), 296-301, 1951. <link href="https://cs.brown.edu/courses/csci1440/labs/2023/lab01/robinson.pdf" color="#203346">Paper</link>.',ref)
p('[3] H. B. McMahan, G. J. Gordon, and A. Blum. <i>Planning in the Presence of Cost Functions Controlled by an Adversary.</i> ICML, 2003. <link href="https://www.cs.cmu.edu/~ggordon/mcmahan-ggordon-blum.icml2003.pdf" color="#203346">Paper</link>.',ref)
p('[4] B. von Stengel. <i>Efficient Computation of Behavior Strategies.</i> Games and Economic Behavior 14(2), 220-246, 1996. <link href="https://doi.org/10.1006/game.1996.0050" color="#203346">doi:10.1006/game.1996.0050</link>.',ref)

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Roman',12)
    canvas.setFillColor(colors.HexColor('#526170'))
    canvas.drawRightString(A4[0]-2.5*cm,2.5*cm+1,str(doc.page))
    canvas.restoreState()

doc=SimpleDocTemplate(str(ATTEMPT/'report.pdf'),pagesize=A4,leftMargin=2.5*cm,rightMargin=2.5*cm,
                      topMargin=2.5*cm,bottomMargin=2.5*cm+19,
                      title='Alternative Nash Solvers: 20,000-Iteration Study',author='',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
pdf=PdfReader(ATTEMPT/'report.pdf')
assert len(pdf.pages)==5, f'Report has {len(pdf.pages)} pages instead of five'
for pg in pdf.pages:
    assert abs(float(pg.mediabox.width)-A4[0])<.1
    assert abs(float(pg.mediabox.height)-A4[1])<.1
print(f'{ATTEMPT/"report.pdf"}: {len(pdf.pages)} A4 pages')
