"""Deterministic CFR ablations and an independent sequence-form LP reference."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'student_attempts')]
import numpy as np
from env_config import env, make_round_start_state
from solver import Solver, CFRSolver
from lp_solver import solve_sequence_form_lp
from treeplex_representation import (TreeplexGame,
    convert_sequence_to_behavioral_form_strategy as to_behavior,
    convert_behavioral_to_sequence_form_strategy as to_sequence)

CONFIGS = {
    'CFR simultaneous': {},
    'CFR alternating': {'alternating': True},
    'CFR+ uniform': {'alternating': True, 'regret_matching_plus': True},
    'CFR+ linear': {'alternating': True, 'regret_matching_plus': True, 'linear_averaging': True},
}


def check_plan(x, tree):
    assert x.shape == (tree.num_sequences,)
    assert np.all(np.isfinite(x)) and np.min(x) >= -1e-12
    err = max([abs(x[0] - 1.0)] + [abs(x[a:b].sum() - x[p]) for a,b,p in zip(
        tree.info_state_start_seq, tree.info_state_end_seq, tree.info_state_parent_seq)])
    assert err < 1e-8, err
    return float(err)


def save_and_check(game, x, y, destination):
    destination.mkdir(parents=True, exist_ok=True)
    restored = []
    for player, plan, tree in [(0,x,game.treeplex_p1),(1,y,game.treeplex_p2)]:
        check_plan(plan, tree)
        behavior = to_behavior(plan, tree, in_place=False)
        agent = game.behavioral_strategy_to_agent(behavior, player)
        path = destination / f'nash_p{player+1}.npz'
        game.save_agent(agent, player, str(path))
        with np.load(path) as saved:
            assert saved['strategy'].shape == (tree.num_sequences,)
            assert saved['player_id'].tolist() == [player]
            assert np.isfinite(saved['strategy']).all()
            assert (saved['strategy'] >= 0).all()
        loaded = game.load_agent(str(path))
        restored.append(to_sequence(game.agent_to_behavioral_strategy(loaded,player),tree,in_place=False))
        check_plan(restored[-1], tree)
    evaluator = Solver(game)
    gap = float(evaluator.saddle_point_residual(*restored))
    assert gap <= 2e-3, gap
    return {'spr': gap, 'value': float(evaluator.compute_payoffs_from_seq_form_strategies(*restored)),
            'max_realization_change': float(max(np.max(np.abs(restored[0]-x)),np.max(np.abs(restored[1]-y))))}


def scenario(name):
    if name == 'neutral':
        return TreeplexGame()
    if name == 'first_player':
        return TreeplexGame(make_round_start_state(0))
    suits = list(env.Suit)
    return TreeplexGame(make_round_start_state(0, {suits[-1]: -1}))


def benchmark(name, iterations, repeats):
    started = time.perf_counter()
    game = scenario(name)
    build_seconds = time.perf_counter() - started
    evaluator = Solver(game)
    # Warm numerical kernels before timing; initialization remains timed per solver.
    CFRSolver(game).solve(max_iter=2, spr_threshold=-1)
    methods = {key: [] for key in CONFIGS}
    for repeat in range(repeats):
        labels = list(CONFIGS)
        labels = labels[repeat % len(labels):] + labels[:repeat % len(labels)]
        for label in labels:
            started = time.perf_counter()
            cfr = CFRSolver(game, check_every=25, **CONFIGS[label])
            x,y = cfr.solve(max_iter=iterations, spr_threshold=-1)
            elapsed = time.perf_counter() - started
            check_plan(x,game.treeplex_p1)
            check_plan(y,game.treeplex_p2)
            current_gap = float(evaluator.saddle_point_residual(cfr.current_seq_form_p1,cfr.current_seq_form_p2))
            record = {'repeat':repeat, 'wall_seconds':elapsed, 'spr':float(cfr.spr),
                      'value':float(cfr.val), 'current_spr':current_gap, 'history':cfr.history,
                      'first_target':next((h for h in cfr.history if h['spr'] <= 2e-3),None)}
            methods[label].append(record)
            print(f'{name}: {label}, repeat {repeat+1}: SPR={cfr.spr:.8g}, {elapsed:.3f}s',flush=True)
    lp_runs=[]
    for repeat in range(repeats):
        started=time.perf_counter()
        lp=solve_sequence_form_lp(game)
        elapsed=time.perf_counter()-started
        check_plan(lp['sol_seq_form_p1'],game.treeplex_p1)
        check_plan(lp['sol_seq_form_p2'],game.treeplex_p2)
        independent_gap=float(evaluator.saddle_point_residual(lp['sol_seq_form_p1'],lp['sol_seq_form_p2']))
        assert independent_gap < 1e-7
        row={k:v for k,v in lp.items() if not isinstance(v,np.ndarray)}
        row.update({'wall_seconds':elapsed,'independent_spr':independent_gap})
        lp_runs.append(row)
    output={'build_seconds':build_seconds, 'infosets':[game.treeplex_p1.get_num_infosets(),game.treeplex_p2.get_num_infosets()],
            'sequences':[game.treeplex_p1.num_sequences,game.treeplex_p2.num_sequences],
            'payoff_entries':len(game.seq_pair_to_leaves),'nonzero_payoffs':sum(v!=0 for v in game.seq_pair_to_leaves.values()),
            'methods':methods,'lp':lp_runs}
    if name=='neutral':
        output['submitted']=save_and_check(game,lp['sol_seq_form_p1'],lp['sol_seq_form_p2'],ROOT/'tasks')
        table={}
        for player,infos in [(0,game.info_states_id_to_key_p1),(1,game.info_states_id_to_key_p2)]:
            agent=game.load_agent(str(ROOT/'tasks'/f'nash_p{player+1}.npz'))
            for info in infos:
                table[info]=agent.get_action_distribution(info)
        env.PolicyTableAgent(table).dump_artifact(str(ROOT/'tasks'/'my_nash_tiny.json'),variant='tiny_hana')
        output['policy_information_sets']=len(table)
        from tiny_hana.example_agents import GreedyAgent, CardCountingAgent, BalancedAgent, AdaptiveAgent
        x,y=lp['sol_seq_form_p1'],lp['sol_seq_form_p2']
        output['baseline_payoffs']={}
        for cls in [env.RandomAgent, GreedyAgent, BalancedAgent, CardCountingAgent, AdaptiveAgent]:
            a,b=[to_sequence(game.agent_to_behavioral_strategy(cls(),player),tree,in_place=False)
                 for player,tree in [(0,game.treeplex_p1),(1,game.treeplex_p2)]]
            output['baseline_payoffs'][cls.__name__]={
                'nash_as_p1':evaluator.compute_payoffs_from_seq_form_strategies(x,b),
                'nash_as_p2':-evaluator.compute_payoffs_from_seq_form_strategies(a,y),
                'baseline_profile_spr':evaluator.saddle_point_residual(a,b)}
    print(f'{name}: LP SPR={independent_gap:.3g}, value={lp["value"]:.8g}',flush=True)
    return output


def hardware_info():
    cpu=platform.processor() or platform.machine()
    ram=None
    if sys.platform=='darwin':
        try:
            cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
            ram=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
        except (OSError,subprocess.CalledProcessError,ValueError):
            pass
    elif hasattr(os,'sysconf'):
        try:
            ram=int(os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
        except (OSError,ValueError):
            pass
    return cpu,ram


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=2000)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'task4'/'results'/'experiments.json')
    args=parser.parse_args()
    import scipy
    cpu,ram=hardware_info()
    info={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
          'platform':platform.platform(),'machine':platform.machine(),
          'cpu':cpu, 'ram_bytes':ram,
          'iterations':args.iterations,'repeats':args.repeats,'check_every':25,'engine':'Python',
          'seed':'No stochastic sampling: complete weighted chance enumeration; zero initial regrets.',
          'timing':'perf_counter; CFR wall includes solver initialization and periodic exact SPR; excludes tree build; order rotated across repeats; LP includes assembly and both solves.',
          'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'student_attempts'/'solver.py',ROOT/'student_attempts'/'lp_solver.py',ROOT/'student_attempts'/'treeplex_representation.py',Path(__file__)]}}
    output={'environment':info,'scenarios':{}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for name in ['neutral','first_player','favored_g3']:
        output['scenarios'][name]=benchmark(name,args.iterations,args.repeats)
        output['peak_process_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
        args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(args.output)

if __name__=='__main__':
    main()
