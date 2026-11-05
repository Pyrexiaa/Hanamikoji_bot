"""Build the five-page A4 report from the measured results (12 pt, 2.5 cm margins)."""
from pathlib import Path
from statistics import median
import json
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak, KeepTogether)
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'task4'/'results'
r=json.loads((RESULTS/'experiments.json').read_text())
n=r['scenarios']['neutral']
W=A4[0]-5*cm
ink=colors.HexColor('#203346')
style=ParagraphStyle('body',fontName='Times-Roman',fontSize=12,leading=14.4,spaceAfter=7,alignment=TA_LEFT)
heading=ParagraphStyle('heading',parent=style,fontName='Times-Bold',fontSize=12,leading=15,spaceBefore=9,spaceAfter=7,textColor=ink)
title=ParagraphStyle('title',parent=heading,spaceBefore=0,spaceAfter=10)
caption=ParagraphStyle('caption',parent=style,leading=13.6,spaceAfter=7)
mono=ParagraphStyle('mono',parent=style,fontName='Courier',fontSize=12,leading=14.4)
story=[]
def p(text,s=style):story.append(Paragraph(text,s))
def h(text):p(text,heading)
def page():story.append(PageBreak())
def table(rows,widths):
    content=[[Paragraph(str(v),style) for v in row] for row in rows]
    t=Table(content,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf1f4')),
                          ('LINEBELOW',(0,0),(-1,0),.6,ink),
                          ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#9ba8b4')),
                          ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
                          ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),
                          ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.append(t);story.append(Spacer(1,7))
def f(v):return f'{v:.3g}'

p('Solving Tiny Hanamikoji: beliefs, regret, and equilibrium',title)
p('CS6208 Homework 2 | 28 October 2026')
h('1. Problem and design')
p('I solve one round of Tiny Hanamikoji with neutral initial favors and a uniformly chosen starting player. Payoff is +1 for winning, -1 for losing, and 0 for an unresolved round. Hidden hands and reserves make decisions information-set dependent. I implemented the guided posterior, best-response, and CFR tasks, then investigated which CFR modifications improve convergence. A sequence-form LP supplies an independent reference and the submitted strategies.')
p('The default tree has 2,757 information sets and 3,672 sequences per player. The supplied payoff map contains 12,832 sequence pairs, of which 2,816 have nonzero payoff. I keep the supplied treeplex file unchanged. Its sequence ordering determines compatibility with the strategy loader.')
h('2. Beliefs and best responses')
p('For target information set I, my posterior weight at a consistent history h is chance reach times opponent reach. I sum weights for identical state keys, then normalize. My own reach cancels under perfect recall: histories in I share my previous information sets and actions. A mismatched observation prefix cannot recover later, so I prune it. Zero-probability branches are omitted; impossible conditioning events return an empty dictionary. A copied initial state and finally/undo blocks preserve the caller\'s state.')
p('Let x and y be realization plans and A the chance-weighted payoff matrix. Rewards are Ay for Player 1 and -A<super>T</super>x for Player 2. My best-response dynamic program traverses information sets from children to parents, maximizes each local action value, and adds it to the parent sequence. All child information sets sharing a sequence contribute; selecting only one would lose possible future observations.')
p('<b>SPR(x,y) = max<sub>x\'</sub> x\'<super>T</super>Ay - min<sub>y\'</sub> x<super>T</super>Ay\'.</b> I use the full gap, without division by two. It equals the sum of unilateral deviation gains and bounds each player\'s incentive to deviate.')
h('3. CFR implementation')
p('I use full-tree counterfactual regret minimization [1]. Backward propagation computes continuation action values q(I,a), the behavioral expectation v(I), and regret increment q(I,a)-v(I). Regret matching normalizes positive cumulative regrets, using a uniform distribution when none is positive. Opponent and chance reach are already in the reward vector; my prefix reach must not multiply these regrets.')
p('I average realization plans, not raw behavioral probabilities. This weights each information-set strategy by its own reach. Independent information sets at the same own-action depth are processed in NumPy batches; repeated parent indices use accumulation. Payoffs are refreshed between solve calls, and continuation and reset are supported.')

page()
h('4. Controlled convergence experiment')
p('My hypothesis is that alternating updates, regret clipping, and reduced weight on early strategies explain different parts of the speedup. I therefore compare four configurations: CFR-S (simultaneous, uniform average), CFR-A (alternating, uniform), CFR+-U (alternating RM+, uniform), and CFR+-L (alternating RM+, linear average). RM+ updates stored regret as max(R+r,0), following [2]. Linear weights t affect only the strategy average; this is not linear weighting of regret increments.')
p('All configurations start from zero regrets and use complete weighted chance enumeration for 2,000 iterations. Alternation updates P1 first, then P2 against refreshed P1; both pre-update plans enter the average. This explicit averaging convention is shared across the ablations. SPR is checked every 25 iterations, with no early stopping during measurement.')
story.append(Image(str(RESULTS/'convergence.png'),width=W,height=W*2.9/6.3))
p('Figure 1. Default-game SPR against iterations and solve time for the first timing repetition. The dotted line is the assignment\'s illustrative target, 0.002. Time includes exact periodic SPR checks; tree construction and solver construction are excluded from the plotted solve clock.',caption)
rows=[['Method','SPR at 2,000','First target check','Median time (s)']]
for name,short in [('CFR simultaneous','CFR-S'),('CFR alternating','CFR-A'),('CFR+ uniform','CFR+-U'),('CFR+ linear','CFR+-L')]:
 q=n['methods'][name]
 rows.append([short,f(q[0]['spr']),q[0]['first_target']['iteration'],f'{median(v["wall_seconds"] for v in q):.3f}'])
table(rows,[W*.21,W*.26,W*.26,W*.27])
p('Table 1. Three timing repetitions with rotated method order. Table time includes solver construction, excludes the shared tree build, and covers all 2,000 iterations. A first checkpoint crossing need not be the earliest exact crossing or remain below the target thereafter.',caption)

page()
h('5. What accounts for the improvement?')
p('At 2,000 iterations, alternation reduces the default gap by about 5.95 times relative to simultaneous CFR. Clipping adds only a 2.8% reduction when both use uniform averaging. Linear averaging then reduces the clipped solver\'s gap by about 29.1 times, yielding a total 178-fold reduction relative to CFR-S. The two clipped variants follow identical regret trajectories; the final-average difference isolates weighting, rather than a different self-play path.')
p(f'The target is first observed at iteration 75 for CFR+-L versus 1,550 for CFR-S. Their first-repetition solve clocks are {n["methods"]["CFR+ linear"][0]["first_target"]["elapsed_seconds"]:.4f} s and {n["methods"]["CFR simultaneous"][0]["first_target"]["elapsed_seconds"]:.3f} s. These are checkpoint measurements, not precise hitting times. Alternation costs slightly more per iteration but improves accuracy substantially. Full timing ranges are retained in the raw results.')
h('6. Independent sequence-form LP')
p('I also solve the compact zero-sum LP [3]. Let Ex=e and Fy=f encode flow conservation: the empty sequence has weight one and each information set\'s action weights sum to its parent weight. Dualizing the opponent\'s optimization gives the following pair:')
p('<b>P1:</b> maximize f<super>T</super>q subject to Ex=e, x &ge; 0, F<super>T</super>q &le; A<super>T</super>x.<br/><b>P2:</b> minimize e<super>T</super>p subject to Fy=f, y &ge; 0, E<super>T</super>p &ge; Ay.')
p('The dual vectors p and q are unrestricted. I build sparse matrices and solve both programs with SciPy/HiGHS, checking success, primal and dual residuals, and the independent best-response gap. SciPy is imported inside the optional LP helper, so it is not required for the grader-facing functions.')
p(f'The default LP value is numerically zero. Maximum primal residual is {max(n["lp"][0]["feasibility_p1"],n["lp"][0]["feasibility_p2"]):.2e}; maximum dual residual is {max(n["lp"][0]["dual_feasibility_p1"],n["lp"][0]["dual_feasibility_p2"]):.2e}. The LP objective gap is zero to floating-point precision. Its independent SPR is {n["lp"][0]["independent_spr"]:.2e}. I submit these LP plans through the prescribed save_agent API; reloaded SPR remains {n["submitted"]["spr"]:.2e}. Conversion changes realization weights by at most {n["submitted"]["max_realization_change"]:.2e}. This is a numerical certificate, not symbolic exact arithmetic.')
p(f'LP assembly plus both solves takes {median(v["wall_seconds"] for v in n["lp"]):.4f} s at the median. The first LP run takes {n["lp"][0]["wall_seconds"]:.3f} s because it includes lazy optimizer initialization; the next two take {n["lp"][1]["wall_seconds"]:.4f} and {n["lp"][2]["wall_seconds"]:.4f} s. Only CFR receives an explicit two-iteration warm-up. Tree construction takes {n["build_seconds"]:.3f} s and is excluded from both comparisons. At this scale, preprocessing dominates an accurate LP solve; the experiment does not establish a large-game scaling advantage.')

page()
h('7. Robustness and interpretation')
p('I repeat the same four configurations without tuning on two single-round variants: P1 starts from neutral favors; and P1 starts already controlling G3, with G1 and G2 neutral. Their treeplexes have 1,297/1,460 information sets and 1,902/1,771 sequences. These are changes in information and terminal rewards, not tests of a larger card game.')
rows=[['Method','Neutral, random start','Neutral, P1 start','P1 start, G3 held']]
for label,short in [('CFR simultaneous','CFR-S'),('CFR alternating','CFR-A'),('CFR+ uniform','CFR+-U'),('CFR+ linear','CFR+-L')]:
 rows.append([short]+[f(r['scenarios'][k]['methods'][label][0]['spr']) for k in ['neutral','first_player','favored_g3']])
rows.append(['LP value']+[f'{0.0 if abs(r["scenarios"][k]["lp"][0]["value"]) < 1e-12 else r["scenarios"][k]["lp"][0]["value"]:.6f}' for k in ['neutral','first_player','favored_g3']])
table(rows,[W*.2,W*.27,W*.265,W*.265])
p('Table 2. Final average SPR at 2,000 iterations and independent LP values. CFR+-L is best among the four averages in all three cases. Positive values measure P(win)-P(loss) in this round, not win probability or the value of a multi-round match.',caption)
p('The default value is zero because the chance choice of starting player restores player symmetry. With P1 fixed to start, the LP value is about 0.01905; retaining G3 increases it to 0.05714. Equality of self-play payoff with the value does not certify an equilibrium: default CFR-S has near-zero self-play payoff even while exploitable. I therefore measure SPR directly.')
p('Averaging is not uniformly better at every finite time. Default CFR-S has current-iterate SPR 0.11265 versus average SPR 0.001665. Default RM+ has current SPR 0.001656 versus linear-average SPR 0.00000934. In the G3-held game, alternating current strategies reach numerical equilibrium while averages still retain earlier mistakes. This explains why current payoff or a single current strategy is an unreliable stopping rule across games.')
h('Exact evaluation against fixed opponents')
rows=[['Opponent','Nash as P1','Nash as P2']]
for label,vals in n['baseline_payoffs'].items():
 rows.append([label.replace('Agent',''),f'{vals["nash_as_p1"]:+.4f}',f'{vals["nash_as_p2"]:+.4f}'])
table(rows,[W*.46,W*.27,W*.27])
p('Table 3. Expected payoff to the submitted policy, computed by exact chance-weighted evaluation. No Monte Carlo sampling error is involved. Equilibrium protects against all opponents, but need not maximize exploitation of a particular heuristic. The two seats may use different members of a nonunique equilibrium set.',caption)

page()
h('8. Verification and reproducibility')
p('All 21 tests pass. Six posterior tests include analytic hidden-card probabilities 1/5, 1/5, 3/5, a reserve-signal posterior of 1/2, 1/2, both seats, late own actions, impossible events, and state restoration after exceptions. A separate joint-law oracle includes both players\' reach, independently checking the cancellation argument.')
p('Fifteen solver tests include enumeration of all 32 pure plans of a branching treeplex, full Tiny best responses against separate LPs, and sequence payoff against live-tree traversal. The matrix game [[2,-1],[-1,1]] checks equilibrium (0.4,0.6), value 0.2, and reward signs; positive and negative payoff shifts check free dual variables.')
p('Further checks cover flow conservation, RM+ nonnegativity, branching-game convergence, continuation/reset in all four configurations, zero iterations, changed-payoff cache refresh, and core imports with SciPy blocked. All four notebooks execute. The protected treeplex hash matches the starter. Export checks cover array shape, player ID, finite/nonnegative normalized probabilities, and reloaded SPR.')
p(f'Measurements use Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1, the Python engine, an Apple M4 with 16 GiB RAM, and macOS 15.6. Three repetitions measure timing only: full enumeration gives identical numerical trajectories. Peak process RSS is {r["peak_process_rss_mib"]:.1f} MiB, not a per-solver comparison. The task4 folder retains source hashes, histories, certificates, tests, and a dependency lock.')
p('I reproduce all results with the command sequence in task4/README.md. The experiment runner regenerates measurements and submitted strategies; the report builder and supplied packager produce the final PDF and ZIP.')
h('9. Limits and next steps')
p('This study supports a practical benefit from alternation and strategy averaging on small, fully enumerated games. It does not establish a better asymptotic convergence rate, scalability to Medium/Full Hanamikoji, or a solution to the multi-round game. My policy table covers only neutral first rounds; any fallback in a later round has no equilibrium guarantee. The next useful experiment would measure enumeration growth and sampled counterfactual updates under a fixed memory budget, rather than tune more parameters on Tiny.')
h('References')
p('[1] M. Zinkevich, M. Johanson, M. Bowling, and C. Piccione. <i>Regret Minimization in Games with Incomplete Information.</i> NeurIPS 20, 2007. <link href="https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html" color="#203346">Proceedings paper</link>.')
p('[2] O. Tammelin. <i>Solving Large Imperfect Information Games Using CFR+.</i> 2014. <link href="https://arxiv.org/abs/1407.5042" color="#203346">arXiv:1407.5042</link>.')
p('[3] B. von Stengel. <i>Efficient Computation of Behavior Strategies.</i> Games and Economic Behavior 14(2), 220-246, 1996. <link href="https://doi.org/10.1006/game.1996.0050" color="#203346">doi:10.1006/game.1996.0050</link>.')

def footer(canvas,doc):
    canvas.saveState()
    canvas.setFont('Times-Roman',12)
    canvas.setFillColor(colors.HexColor('#526170'))
    # Keep the complete footer within the required margin box.
    canvas.drawRightString(A4[0]-2.5*cm,2.5*cm+1,str(doc.page))
    canvas.restoreState()

doc=SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=A4,leftMargin=2.5*cm,rightMargin=2.5*cm,
                      topMargin=2.5*cm,bottomMargin=2.5*cm+19,title='Solving Tiny Hanamikoji',
                      author='',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
pdf=PdfReader(ROOT/'report.pdf')
assert len(pdf.pages)<=5, f'Report has {len(pdf.pages)} pages'
for pg in pdf.pages:
    assert abs(float(pg.mediabox.width)-A4[0])<.1
    assert abs(float(pg.mediabox.height)-A4[1])<.1
print(f'{ROOT/"report.pdf"}: {len(pdf.pages)} A4 pages')
