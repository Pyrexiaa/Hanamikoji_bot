"""Compare three solver families through a common full iteration budget."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ATTEMPT = Path(__file__).resolve().parents[1]
REPOSITORY = ATTEMPT.parent
sys.path[:0] = [str(ATTEMPT / 'student_attempts'), str(REPOSITORY), str(ATTEMPT)]
import numpy as np
from env_config import env, make_round_start_state
from solver import Solver, CFRSolver
from fictitious_play import FictitiousPlaySolver
from double_oracle import DoubleOracleSolver
from lp_solver import solve_sequence_form_lp
from treeplex_representation import (TreeplexGame,
    convert_sequence_to_behavioral_form_strategy as to_behavior,
    convert_behavioral_to_sequence_form_strategy as to_sequence)

METHODS = {'CFR': CFRSolver, 'Fictitious play': FictitiousPlaySolver,
           'Double oracle': DoubleOracleSolver}
CONFIG_PATH = Path(__file__).with_name('experiment_config.json')
SUMMARY_CHECKPOINTS = (25, 100, 500, 1000, 2000, 5000, 10000, 20000)
METHOD_SLUGS = {'CFR': 'cfr', 'Fictitious play': 'fictitious_play',
                'Double oracle': 'double_oracle'}


def history_cadence(label, config):
    """CFR pays for sparse certificates; the other families already certify every step."""
    return config['check_every'] if label == 'CFR' else 1


def validate_main_run(learner, label, config):
    """Require actual updates and complete measured observations, without padding."""
    if config['full_budget'] and learner.iterations != config['iterations']:
        raise RuntimeError(f'{label} completed {learner.iterations} of '
                           f'{config["iterations"]} requested full-budget iterations')
    cadence = history_cadence(label, config)
    expected = list(range(0, learner.iterations + 1, cadence))
    if expected[-1] != learner.iterations:
        expected.append(learner.iterations)
    observed = [entry['iteration'] for entry in learner.history]
    if observed != expected:
        raise RuntimeError(f'{label} history does not contain every expected certificate '
                           f'(cadence {cadence}, final iteration {learner.iterations})')


def save_history(raw_history, constructor_seconds, label, scenario_name, repeat,
                 config, results_directory):
    """Archive every measured certificate and retain exact report checkpoints."""
    history = []
    for observation in raw_history:
        entry = dict(observation)
        entry['solve_elapsed_seconds'] = entry['elapsed_seconds']
        entry['elapsed_seconds'] += constructor_seconds
        history.append(entry)
    first_target = next((entry for entry in history
                         if entry['spr'] <= config['spr_threshold']), None)
    final_iteration = history[-1]['iteration']
    summary_iterations = sorted(set(config.get('summary_checkpoints', SUMMARY_CHECKPOINTS))
                                | {config['iterations']})
    by_iteration = {entry['iteration']: entry for entry in history}
    # No interpolation: a nonstandard CFR cadence may omit a requested checkpoint.
    summary = [dict(by_iteration[iteration]) for iteration in summary_iterations
               if iteration in by_iteration]
    compact_every = config.get('history_output_every', 25)
    retained = {0, final_iteration}
    if first_target is not None:
        retained.add(first_target['iteration'])
    retained.update(entry['iteration'] for entry in summary)
    compact = [entry for entry in history
               if entry['iteration'] % compact_every == 0 or entry['iteration'] in retained]
    relative_path = Path('histories') / f'{scenario_name}_{METHOD_SLUGS[label]}_r{repeat + 1}.jsonl.gz'
    path = Path(results_directory) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compression and serialization occur after the measured constructor + solve.
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for entry in history:
            stream.write(json.dumps(entry, separators=(',', ':'), allow_nan=False) + '\n')
    return {'history': compact, 'history_file': relative_path.as_posix(),
            'history_records': len(history), 'history_cadence': history_cadence(label, config),
            'history_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'history_output_every': compact_every, 'first_target': first_target,
            'ever_reached_target': first_target is not None, 'summary_checkpoints': summary,
            'unrecorded_summary_checkpoints': [iteration for iteration in summary_iterations
                if iteration <= final_iteration and iteration not in by_iteration]}


def check_plan(plan, tree):
    assert plan.shape == (tree.num_sequences,)
    assert np.isfinite(plan).all() and plan.min() >= -1e-12
    residual = max([abs(plan[0] - 1)] + [abs(plan[a:b].sum()-plan[p]) for a,b,p in zip(
        tree.info_state_start_seq, tree.info_state_end_seq, tree.info_state_parent_seq)])
    assert residual < 1e-8, residual
    return float(residual)


def save_and_check(game, x, y, destination, target):
    destination.mkdir(parents=True, exist_ok=True)
    restored=[]
    for player, plan, tree in [(0,x,game.treeplex_p1),(1,y,game.treeplex_p2)]:
        check_plan(plan,tree)
        behavior=to_behavior(plan,tree,in_place=False)
        agent=game.behavioral_strategy_to_agent(behavior,player)
        path=destination/f'nash_p{player+1}.npz'
        game.save_agent(agent,player,str(path))
        with np.load(path) as saved:
            assert saved['strategy'].shape==(tree.num_sequences,)
            assert saved['player_id'].tolist()==[player]
            assert np.isfinite(saved['strategy']).all() and (saved['strategy']>=0).all()
        loaded=game.load_agent(str(path))
        restored.append(to_sequence(game.agent_to_behavioral_strategy(loaded,player),tree,in_place=False))
        check_plan(restored[-1],tree)
    evaluator=Solver(game)
    gap=float(evaluator.saddle_point_residual(*restored))
    assert gap <= target
    return {'source':'sequence-form LP reference','spr':gap,
            'value':float(evaluator.compute_payoffs_from_seq_form_strategies(*restored)),
            'max_realization_change':float(max(np.max(np.abs(restored[0]-x)),np.max(np.abs(restored[1]-y))))}


def scenario(name):
    if name=='neutral':return TreeplexGame()
    if name=='first_player':return TreeplexGame(make_round_start_state(0))
    # The neutral game is a 50/50 chance mixture of first_player and this mirror image.
    if name=='second_player':return TreeplexGame(make_round_start_state(1))
    if name=='favored_g3':
        return TreeplexGame(make_round_start_state(0,{env.Suit.GEISHA_3:-1}))
    raise ValueError(f'Unknown scenario {name}')


def benchmark(name, config, results_directory):
    started=time.perf_counter()
    game=scenario(name)
    build_seconds=time.perf_counter()-started
    evaluator=Solver(game)
    # All methods get a discarded warm-up; optimizer imports are outside timed runs.
    for label, cls in METHODS.items():
        cls(game,check_every=history_cadence(label,config)).solve(
            max_iter=config['warmup_iterations'],spr_threshold=None)
    solve_sequence_form_lp(game,tolerance=config['lp_tolerance'])
    methods={label:[] for label in METHODS}
    for repeat in range(config['repeats']):
        labels=list(METHODS)
        offset=repeat % len(labels)
        for label in labels[offset:]+labels[:offset]:
            started=time.perf_counter()
            learner=METHODS[label](game,check_every=history_cadence(label,config))
            constructor_seconds=time.perf_counter()-started
            total_br_calls=[0]
            original=learner.compute_best_response
            def counted(*args,**kwargs):
                total_br_calls[0]+=1
                return original(*args,**kwargs)
            learner.compute_best_response=counted
            threshold=None if config['full_budget'] else config['spr_threshold']
            x,y=learner.solve(max_iter=config['iterations'],spr_threshold=threshold)
            wall_seconds=time.perf_counter()-started
            validate_main_run(learner,label,config)
            feasibility=[check_plan(plan,tree) for plan,tree in [(x,game.treeplex_p1),(y,game.treeplex_p2)]]
            independent_spr=float(evaluator.saddle_point_residual(x,y))
            assert abs(independent_spr-learner.spr)<1e-10
            history_record=save_history(learner.history,constructor_seconds,label,name,repeat,
                                        config,results_directory)
            training_br_calls=int(getattr(learner,'training_br_calls',getattr(learner,'br_calls',0)))
            record={'repeat':repeat,'iterations':learner.iterations,
                    'requested_iterations':config['iterations'],
                    'full_budget':config['full_budget'],
                    'completed_full_budget':learner.iterations==config['iterations'],
                    'reached_target':bool(learner.spr<=config['spr_threshold']),
                    'wall_seconds':wall_seconds,'constructor_seconds':constructor_seconds,
                    'spr':float(learner.spr),'independent_spr':independent_spr,'value':float(learner.val),
                    'feasibility':feasibility,'training_br_calls':training_br_calls,
                    'diagnostic_br_calls':total_br_calls[0]-training_br_calls,
                    'regret_sweeps':2*learner.iterations if label=='CFR' else 0,
                    'restricted_lp_solves':int(getattr(learner,'restricted_lp_solves',0)),
                    'support_counts':list(getattr(learner,'support_counts',())),
                    'active_support_counts':[int(np.count_nonzero(w>1e-12)) for w in
                        (learner.weights_p1,learner.weights_p2)] if hasattr(learner,'weights_p1') else [],
                    **history_record}
            if hasattr(learner,'stop_reason'):
                record['stop_reason']=learner.stop_reason
            if hasattr(learner,'current_seq_form_p1'):
                record['current_spr']=float(evaluator.saddle_point_residual(learner.current_seq_form_p1,learner.current_seq_form_p2))
            methods[label].append(record)
            print(f'{name}: {label}, repeat {repeat+1}: iterations={learner.iterations}, SPR={learner.spr:.8g}, target={record["reached_target"]}, {wall_seconds:.3f}s',flush=True)
    reference=[]
    for repeat in range(config['repeats']):
        started=time.perf_counter()
        lp=solve_sequence_form_lp(game,tolerance=config['lp_tolerance'])
        wall=time.perf_counter()-started
        for plan,tree in [(lp['sol_seq_form_p1'],game.treeplex_p1),(lp['sol_seq_form_p2'],game.treeplex_p2)]:check_plan(plan,tree)
        independent_spr=float(evaluator.saddle_point_residual(lp['sol_seq_form_p1'],lp['sol_seq_form_p2']))
        assert independent_spr<1e-7
        row={k:v for k,v in lp.items() if not isinstance(v,np.ndarray)}
        row.update({'wall_seconds':wall,'independent_spr':independent_spr})
        reference.append(row)
    result={'build_seconds':build_seconds,
            'infosets':[game.treeplex_p1.get_num_infosets(),game.treeplex_p2.get_num_infosets()],
            'sequences':[game.treeplex_p1.num_sequences,game.treeplex_p2.num_sequences],
            'payoff_entries':len(game.seq_pair_to_leaves),'nonzero_payoffs':sum(v!=0 for v in game.seq_pair_to_leaves.values()),
            'methods':methods,'lp':reference}
    if name=='neutral':
        x,y=lp['sol_seq_form_p1'],lp['sol_seq_form_p2']
        result['submitted']=save_and_check(game,x,y,ATTEMPT/'tasks',config['spr_threshold'])
        table={}
        for player,infos in [(0,game.info_states_id_to_key_p1),(1,game.info_states_id_to_key_p2)]:
            agent=game.load_agent(str(ATTEMPT/'tasks'/f'nash_p{player+1}.npz'))
            table.update({info:agent.get_action_distribution(info) for info in infos})
        env.PolicyTableAgent(table).dump_artifact(str(ATTEMPT/'tasks'/'my_nash_tiny.json'),variant='tiny_hana')
        result['policy_information_sets']=len(table)
        from tiny_hana.example_agents import GreedyAgent,CardCountingAgent,BalancedAgent,AdaptiveAgent
        result['baseline_payoffs']={}
        for cls in [env.RandomAgent,GreedyAgent,BalancedAgent,CardCountingAgent,AdaptiveAgent]:
            a,b=[to_sequence(game.agent_to_behavioral_strategy(cls(),player),tree,in_place=False)
                 for player,tree in [(0,game.treeplex_p1),(1,game.treeplex_p2)]]
            result['baseline_payoffs'][cls.__name__]={'nash_as_p1':evaluator.compute_payoffs_from_seq_form_strategies(x,b),
                'nash_as_p2':-evaluator.compute_payoffs_from_seq_form_strategies(a,y),
                'baseline_profile_spr':evaluator.saddle_point_residual(a,b)}
    print(f'{name}: LP independent SPR={independent_spr:.3g}, value={lp["value"]:.8g}',flush=True)
    return result, game


def behavioral(plan, tree, eps=1e-9):
    """Conditional action probabilities where own reach exceeds eps (else NaN)."""
    out = np.full(tree.num_sequences, np.nan)
    for a, b, p in zip(tree.info_state_start_seq, tree.info_state_end_seq, tree.info_state_parent_seq):
        if plan[p] > eps:
            out[a:b] = plan[a:b] / plan[p]
    return out


def lp_versus_cfr(game, config):
    """Compare the LP equilibrium with CFR's average at the common target.

    Values must agree within SPR; strategies need not, because the equilibrium
    set is a polytope. Non-uniqueness is shown by ranging one first-decision
    probability over ALL optimal P1 plans (an LP on P1's optimal face).
    """
    from scipy import sparse
    from scipy.optimize import linprog
    from lp_solver import payoff_matrix, sequence_form_constraints
    evaluator = Solver(game)
    lp = solve_sequence_form_lp(game, tolerance=config['lp_tolerance'])
    xl, yl, v = lp['sol_seq_form_p1'], lp['sol_seq_form_p2'], lp['value']
    cfr = CFRSolver(game, check_every=config['check_every'])
    xc, yc = cfr.solve(max_iter=config['iterations'], spr_threshold=config['spr_threshold'])
    def split(x, y):
        up = evaluator.compute_payoffs_from_seq_form_strategies(evaluator.compute_best_response(0, y, True), y)
        lo = evaluator.compute_payoffs_from_seq_form_strategies(x, evaluator.compute_best_response(1, x, True))
        return float(v - lo), float(up - v)
    result = {'protocol': {'max_iterations': config['iterations'],
                          'spr_threshold': config['spr_threshold'], 'full_budget': False},
              'lp_value': v, 'cfr_iterations': cfr.iterations, 'cfr_value': float(cfr.val),
              'cfr_spr': float(cfr.spr), 'cfr_exploitability_p1_p2': split(xc, yc),
              'lp_exploitability_p1_p2': split(xl, yl),
              'cross_spr_lpP1_cfrP2': float(evaluator.saddle_point_residual(xl, yc)),
              'cross_spr_cfrP1_lpP2': float(evaluator.saddle_point_residual(xc, yl))}
    for player, tree, a, b in ((0, game.treeplex_p1, xl, xc), (1, game.treeplex_p2, yl, yc)):
        ba, bb = behavioral(a, tree), behavioral(b, tree)
        mixed = lambda plan, beh: sum(1 for s, e, p in zip(tree.info_state_start_seq, tree.info_state_end_seq,
            tree.info_state_parent_seq) if plan[p] > 1e-9 and np.sum(beh[s:e] > 1e-3) > 1)
        common = [0.5 * float(np.abs(ba[s:e] - bb[s:e]).sum()) for s, e, p in zip(tree.info_state_start_seq,
                  tree.info_state_end_seq, tree.info_state_parent_seq) if a[p] > 1e-3 and b[p] > 1e-3]
        result[f'p{player+1}'] = {'mixed_infosets_lp': mixed(a, ba), 'mixed_infosets_cfr': mixed(b, bb),
            'own_reachable_infosets_lp': int(sum(a[p] > 1e-9 for p in tree.info_state_parent_seq)),
            'max_tv_common': max(common), 'common_infosets': len(common)}
    # Range the most different first-decision probability over P1's optimal face.
    tree = game.treeplex_p1
    gap, info, seq = max((float(abs(xl[s] - xc[s])), i, s) for i, (a, b, p) in enumerate(zip(
        tree.info_state_start_seq, tree.info_state_end_seq, tree.info_state_parent_seq)) if p == 0
        for s in range(a, b))
    A = payoff_matrix(game)
    E, e = sequence_form_constraints(game.treeplex_p1)
    F, f = sequence_form_constraints(game.treeplex_p2)
    nx, ny = A.shape
    ne, nf = E.shape[0], F.shape[0]
    a_ub = sparse.vstack([sparse.hstack([-A.T, F.T]),
                          sparse.hstack([sparse.csr_matrix((1, nx)), -sparse.csr_matrix(f)])], format='csr')
    b_ub = np.r_[np.zeros(ny), -(v - 1e-9)]
    a_eq = sparse.hstack([E, sparse.csr_matrix((ne, nf))], format='csr')
    bounds = []
    for sign in (1.0, -1.0):
        c = np.zeros(nx + nf)
        c[seq] = sign
        r = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=e,
                    bounds=[(0, None)] * nx + [(None, None)] * nf, method='highs')
        assert r.success, r.message
        bounds.append(float(sign * r.fun))
    key = game.info_states_id_to_key_p1[info]
    action = list(key.get_legal_actions())[seq - tree.info_state_start_seq[info]]
    hand, _ = key.get_current_hand_and_actions()
    short = lambda card: card.suit.name.replace('GEISHA_', 'G')
    pending = key.get_pending_split() or []
    kind = action.action_type.name
    if kind == 'RESERVE':
        description = f'reserving {short(action.cards[0])}'
    elif kind == 'SPLIT':
        description = 'splitting ' + '/'.join(short(card) for card in action.cards)
    else:
        description = f'taking {short(pending[action.choice])}'
    situation = f"P1 holds {'+'.join(short(card) for card in hand)}"
    if pending:
        situation += f" and P2 has offered {'/'.join(short(card) for card in pending)}"
    result['nonunique'] = {'infoset': str(key), 'action': str(action), 'description': description,
                           'situation': situation, 'lp': float(xl[seq]), 'cfr': float(xc[seq]),
                           'optimal_min': bounds[0], 'optimal_max': bounds[1]}
    return result


def analysis(config, games):
    """Untimed diagnostics behind the report's discussion (one run each)."""
    out = {'double_oracle_exact': {}, 'rates': {}, 'protocols': {
        'double_oracle_exact': {'max_iterations': 1000, 'spr_threshold': 1e-9,
                                'full_budget': False, 'repeats': 1},
        'rates': {'max_iterations': config['iterations'], 'spr_threshold': None,
                  'full_budget': True, 'check_every': config['check_every'], 'repeats': 1},
        'lp_versus_cfr': {'max_iterations': config['iterations'],
                         'spr_threshold': config['spr_threshold'],
                         'full_budget': False, 'repeats': 1}}}
    for name, game in games.items():
        # DO to an exact equilibrium: stop at certified SPR <= 1e-9 (LP accuracy).
        learner = DoubleOracleSolver(game)
        lp_seconds = [0.0]
        lp_trace = []
        restricted = learner._solve_restricted
        def timed():
            started = time.perf_counter()
            restricted()
            elapsed = time.perf_counter() - started
            lp_seconds[0] += elapsed
            # Restricted-game size (stored plans per player) and the time to solve it.
            lp_trace.append([*learner.restricted_payoffs.shape, elapsed])
        learner._solve_restricted = timed
        started = time.perf_counter()
        learner.solve(max_iter=1000, spr_threshold=1e-9)
        wall = time.perf_counter() - started
        h = learner.history
        target = next((q for q in h if q['spr'] <= config['spr_threshold']), None)
        out['double_oracle_exact'][name] = {
            'stop_reason': learner.stop_reason, 'iterations': learner.iterations,
            'spr': float(learner.spr), 'wall_seconds': wall, 'restricted_lp_seconds': lp_seconds[0],
            'restricted_lp_trace': lp_trace,
            'target_iteration': target['iteration'] if target is not None else None,
            'target_seconds': target['elapsed_seconds'] if target is not None else None,
            'support_counts': list(learner.support_counts),
            'active_support_counts': [int(np.count_nonzero(w > 1e-12)) for w in (learner.weights_p1, learner.weights_p2)],
            'history': h}
    # Empirical rates over the full budget (no early stopping); CFR's last iterate for contrast.
    game = games['neutral']
    evaluator = Solver(game)
    for label, cls in (('CFR', CFRSolver), ('Fictitious play', FictitiousPlaySolver)):
        learner = cls(game, check_every=config['check_every'])
        learner.solve(max_iter=config['iterations'], spr_threshold=None)
        h = [q for q in learner.history if q['iteration'] >= 100 and q['spr'] > 0]
        slope = (float(np.polyfit(np.log([q['iteration'] for q in h]),
                                 np.log([q['spr'] for q in h]), 1)[0]) if len(h) >= 2 else None)
        entry = {'slope_100_to_end': slope, 'final_iteration': learner.iterations, 'final_spr': float(learner.spr)}
        if label == 'CFR':
            entry['last_iterate_spr'] = float(evaluator.saddle_point_residual(
                learner.current_seq_form_p1, learner.current_seq_form_p2))
        out['rates'][label] = entry
    out['lp_versus_cfr'] = {name: lp_versus_cfr(games[name], config) for name in ('neutral', 'first_player')}
    return out


def hardware_info():
    cpu=platform.processor() or platform.machine()
    ram=None
    if sys.platform=='darwin':
        try:
            cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
            ram=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
        except (OSError,subprocess.CalledProcessError,ValueError):pass
    elif hasattr(os,'sysconf'):
        try:ram=int(os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'))
        except (OSError,ValueError):pass
    return cpu,ram


def parse_arguments(argv=None):
    config=json.loads(CONFIG_PATH.read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=config['iterations'])
    parser.add_argument('--spr-threshold',type=float,default=config['spr_threshold'])
    parser.add_argument('--repeats',type=int,default=config['repeats'])
    parser.add_argument('--check-every',type=int,default=config['check_every'])
    stopping=parser.add_mutually_exclusive_group()
    stopping.add_argument('--full-budget',dest='full_budget',action='store_true',
                          help='Complete all requested iterations; retain the SPR target for evaluation (default).')
    stopping.add_argument('--early-stop',dest='full_budget',action='store_false',
                          help='Stop each main run at its first certified SPR target crossing.')
    parser.set_defaults(full_budget=config.get('full_budget',True))
    parser.add_argument('--output',type=Path,default=ATTEMPT/'task4'/'results'/'experiments.json')
    args=parser.parse_args(argv)
    config.update({'iterations':args.iterations,'spr_threshold':args.spr_threshold,
                   'repeats':args.repeats,'check_every':args.check_every,'full_budget':args.full_budget})
    if config['iterations']<1 or config['repeats']<1 or config['check_every']<1:parser.error('Iteration, repetition, and checkpoint counts must be positive.')
    if not np.isfinite(config['spr_threshold']) or config['spr_threshold']<=0:parser.error('SPR target must be finite and positive.')
    return args,config


def main():
    args,config=parse_arguments()
    import scipy
    cpu,ram=hardware_info()
    sources=[ATTEMPT/'student_attempts'/name for name in ['solver.py','fictitious_play.py','double_oracle.py','lp_solver.py','treeplex_representation.py']]+[Path(__file__),CONFIG_PATH]
    output={'schema_version':3,'configuration':config,
        'main_protocol': {'max_iterations':config['iterations'],
            'spr_target':config['spr_threshold'],'full_budget':config['full_budget'],
            'stopping_enabled':not config['full_budget'],
            'history_cadence_by_method':{label:history_cadence(label,config) for label in METHODS},
            'first_target':'First recorded certificate at or below the target: CFR checks every check_every iterations, FP and DO every iteration. A later rise does not erase this observation.',
            'history_storage':'Each history_file is relative to this JSON file and contains all measured observations as gzip JSONL. Main history contains exact regular history_output_every checkpoints, the first target certificate, summary_checkpoints, and the final observation; no interpolation.',
            'scope':'Full-budget or early-stop mode applies to main scenario/method repetitions. Subsidiary analyses have their own explicit protocols.'},
        'environment':{
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'platform':platform.platform(),'cpu':cpu,'ram_bytes':ram,'cpu_count':os.cpu_count(),'engine':'Python','seed':None,
        'timing':'Warmed runs, constructor + solve including exact SPR certificates (CFR: two extra BRs every check_every iterations; FP and DO: every iteration, reusing their own oracle values) and in-memory history logging (CFR every check_every, FP and DO every iteration). Excludes tree construction, post-run independent validation, history compression and serialization, policy export, and the analysis section. History elapsed_seconds includes constructor; solve_elapsed_seconds excludes it. All repetitions rotate method order.',
        'source_sha256':{str(p.relative_to(ATTEMPT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},'scenarios':{}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    games={}
    for name in config['scenarios']:
        output['scenarios'][name],games[name]=benchmark(name,config,args.output.parent)
        output['peak_process_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
        args.output.write_text(json.dumps(output,indent=2)+'\n')
    output['analysis']=analysis(config,games)
    for name,q in output['analysis']['double_oracle_exact'].items():
        print(f"{name}: DO target iteration={q['target_iteration']}; exact-check run ends at {q['iterations']} with SPR={q['spr']:.8g} ({q['stop_reason']})",flush=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(args.output)

if __name__=='__main__':main()
