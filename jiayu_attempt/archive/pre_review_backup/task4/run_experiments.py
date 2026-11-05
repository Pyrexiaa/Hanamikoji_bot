"""Compare three solver families under one iteration cap and SPR target."""
from __future__ import annotations
import argparse
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
    if name=='favored_g3':
        return TreeplexGame(make_round_start_state(0,{env.Suit.GEISHA_3:-1}))
    raise ValueError(f'Unknown scenario {name}')


def benchmark(name, config):
    started=time.perf_counter()
    game=scenario(name)
    build_seconds=time.perf_counter()-started
    evaluator=Solver(game)
    # All methods get a discarded warm-up; optimizer imports are outside timed runs.
    for cls in METHODS.values():
        cls(game,check_every=config['check_every']).solve(max_iter=config['warmup_iterations'],spr_threshold=-1)
    solve_sequence_form_lp(game,tolerance=config['lp_tolerance'])
    methods={label:[] for label in METHODS}
    for repeat in range(config['repeats']):
        labels=list(METHODS)
        offset=repeat % len(labels)
        for label in labels[offset:]+labels[:offset]:
            started=time.perf_counter()
            learner=METHODS[label](game,check_every=config['check_every'])
            constructor_seconds=time.perf_counter()-started
            total_br_calls=[0]
            original=learner.compute_best_response
            def counted(*args,**kwargs):
                total_br_calls[0]+=1
                return original(*args,**kwargs)
            learner.compute_best_response=counted
            threshold=-1 if config['full_budget'] else config['spr_threshold']
            x,y=learner.solve(max_iter=config['iterations'],spr_threshold=threshold)
            wall_seconds=time.perf_counter()-started
            feasibility=[check_plan(plan,tree) for plan,tree in [(x,game.treeplex_p1),(y,game.treeplex_p2)]]
            independent_spr=float(evaluator.saddle_point_residual(x,y))
            assert abs(independent_spr-learner.spr)<1e-10
            histories=[]
            for h in learner.history:
                entry=dict(h)
                entry['solve_elapsed_seconds']=entry['elapsed_seconds']
                entry['elapsed_seconds']+=constructor_seconds
                histories.append(entry)
            training_br_calls=int(getattr(learner,'training_br_calls',getattr(learner,'br_calls',0)))
            record={'repeat':repeat,'iterations':learner.iterations,
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
                    'history':histories,
                    'first_target':next((h for h in histories if h['spr']<=config['spr_threshold']),None)}
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
    return result


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


def main():
    config=json.loads(CONFIG_PATH.read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=config['iterations'])
    parser.add_argument('--spr-threshold',type=float,default=config['spr_threshold'])
    parser.add_argument('--repeats',type=int,default=config['repeats'])
    parser.add_argument('--check-every',type=int,default=config['check_every'])
    parser.add_argument('--full-budget',action='store_true',help='Measure all iterations without early stopping; retain the SPR target for evaluation.')
    parser.add_argument('--output',type=Path,default=ATTEMPT/'task4'/'results'/'experiments.json')
    args=parser.parse_args()
    config.update({'iterations':args.iterations,'spr_threshold':args.spr_threshold,
                   'repeats':args.repeats,'check_every':args.check_every,'full_budget':args.full_budget})
    if config['iterations']<1 or config['repeats']<1 or config['check_every']<1:parser.error('Iteration, repetition, and checkpoint counts must be positive.')
    if not np.isfinite(config['spr_threshold']) or config['spr_threshold']<=0:parser.error('SPR target must be finite and positive.')
    import scipy
    cpu,ram=hardware_info()
    sources=[ATTEMPT/'student_attempts'/name for name in ['solver.py','fictitious_play.py','double_oracle.py','lp_solver.py','treeplex_representation.py']]+[Path(__file__),CONFIG_PATH]
    output={'schema_version':2,'configuration':config,'environment':{
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'platform':platform.platform(),'cpu':cpu,'ram_bytes':ram,'engine':'Python','seed':None,
        'timing':'Warmed runs, constructor + solve including exact SPR checkpoints; excludes tree construction, post-run independent validation, policy export. History times include constructor. All repetitions rotate method order.',
        'source_sha256':{str(p.relative_to(ATTEMPT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},'scenarios':{}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for name in config['scenarios']:
        output['scenarios'][name]=benchmark(name,config)
        output['peak_process_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
        args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(args.output)

if __name__=='__main__':main()
