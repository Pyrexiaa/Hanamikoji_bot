"""Print the report's numerical tables and comparisons from primary experiment data."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'task4/results'
METHODS_A = ['CFR', 'Fictitious play', 'Double oracle']
METHODS_B = ['CFR', 'CFR-linear', 'RM+', 'RM+-linear', 'CFR+']
CASES_A = ['neutral', 'first_player', 'second_player', 'favored_g3']
CASES_B = ['neutral', 'advantage', 'contested']
LABELS = {'neutral': 'Random start', 'first_player': 'P1 starts',
          'second_player': 'P2 starts', 'favored_g3': 'P1 + G3'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def scientific(value):
    return f'{value:.2e}'


def extract_data():
    a, b = read('experiments.json'), read('summary.json')
    for sources in (a['environment']['source_sha256'], b['source_sha256']):
        for name, expected in sources.items():
            assert digest(ROOT / name.replace('\\', '/')) == expected, name
    rows_a, lp_a, compact_a = [], [], []
    for case in CASES_A:
        scenario = a['scenarios'][case]
        compact = [LABELS[case]]
        for method in METHODS_A:
            runs = scenario['methods'][method]
            assert len(runs) == 3 and all(r['iterations'] == 20000 for r in runs)
            targets = [r['first_target'] for r in runs]
            assert len({r['iteration'] for r in targets}) == 1
            row = dict(case=case, method=method, final_spr=median(r['spr'] for r in runs),
                       target_iteration=targets[0]['iteration'],
                       target_seconds=median(r['elapsed_seconds'] for r in targets),
                       full_seconds=median(r['wall_seconds'] for r in runs),
                       full_seconds_min=min(r['wall_seconds'] for r in runs),
                       full_seconds_max=max(r['wall_seconds'] for r in runs),
                       final_spr_min=min(r['spr'] for r in runs),
                       final_spr_max=max(r['spr'] for r in runs),
                       maximum_flow_error=max(max(r['feasibility']) for r in runs),
                       final_iterations=20000)
            rows_a.append(row)
            compact.append(f"{row['target_iteration']:,} / {row['target_seconds']:.3f}")
        compact_a.append(compact)
        lp_a.append({'case': case, 'value': scenario['lp'][0]['value'],
                     'spr': max(r['independent_spr'] for r in scenario['lp']),
                     'median_seconds': median(r['wall_seconds'] for r in scenario['lp']),
                     'maximum_feasibility_error': max(max(r['feasibility_p1'], r['feasibility_p2'],
                         r['dual_feasibility_p1'], r['dual_feasibility_p2']) for r in scenario['lp'])})
    with (RESULTS / 'convergence.csv').open(newline='', encoding='utf-8') as handle:
        raw = [{k: v if k in ('case', 'method') else float(v) for k, v in row.items()}
               for row in csv.DictReader(handle)]
    assert len(raw) == 630
    grouped = defaultdict(list)
    for row in raw:
        grouped[row['case'], row['method']].append(row)
    rows_b, by_b, ratios = [], {}, {}
    for case in CASES_B:
        by_b[case] = {}
        for method in METHODS_B:
            records = grouped[case, method]
            finals = [r for r in records if r['iteration'] == 20000]
            assert len(finals) == 3
            first = [min((r for r in records if r['repeat'] == repeat and r['spr'] <= 0.002),
                         key=lambda r: r['iteration']) for repeat in range(3)]
            row = dict(case=case, method=method, final_spr=median(r['spr'] for r in finals),
                       current_spr=median(r['current_spr'] for r in finals),
                       solve_seconds=median(r['elapsed_solve_seconds'] for r in finals),
                       solve_seconds_min=min(r['elapsed_solve_seconds'] for r in finals),
                       solve_seconds_max=max(r['elapsed_solve_seconds'] for r in finals),
                       first_checkpoint=int(first[0]['iteration']),
                       first_checkpoint_seconds=median(r['elapsed_solve_seconds'] for r in first))
            rows_b.append(row)
            by_b[case][method] = row
        q = by_b[case]
        ratios[case] = {
            'linear_over_cfr_spr': q['CFR-linear']['final_spr'] / q['CFR']['final_spr'],
            'clipping_improvement': q['CFR']['final_spr'] / q['RM+']['final_spr'],
            'linear_with_clipping_improvement': q['RM+']['final_spr'] / q['RM+-linear']['final_spr'],
            'alternation_conditional_improvement': q['RM+-linear']['final_spr'] / q['CFR+']['final_spr'],
            'cfr_plus_overall_improvement': q['CFR']['final_spr'] / q['CFR+']['final_spr'],
            'cfr_plus_runtime_ratio': q['CFR+']['solve_seconds'] / q['CFR']['solve_seconds'],
            'cfr_current_over_average_spr': q['CFR']['current_spr'] / q['CFR']['final_spr'],
        }
    compact_b = [[method, *[scientific(by_b[c][method]['final_spr']) for c in CASES_B],
                  f"{by_b['neutral'][method]['solve_seconds']:.3f}"] for method in METHODS_B]
    tables = {
        'configurations': {'headers': ['Method', 'Regrets', 'Average', 'Updates'], 'rows': [
            ['CFR', 'Signed', 'Uniform', 'Simultaneous'],
            ['CFR-linear', 'Signed', 'Linear', 'Simultaneous'],
            ['RM+', 'Clipped', 'Uniform', 'Simultaneous'],
            ['RM+-linear', 'Clipped', 'Linear', 'Simultaneous'],
            ['CFR+', 'Clipped', 'Linear', 'Alternating']]},
        'direction_d': {'headers': ['Scenario', 'CFR', 'FP', 'DO'], 'rows': compact_a},
        'direction_c': {'headers': ['Method', 'Neutral', 'Advantage', 'Contested', 'Time (s)'], 'rows': compact_b},
    }
    result = {'tables': tables,
              'experiment_a': {'rows': rows_a, 'lp': lp_a, 'analysis': a['analysis'],
                               'dimensions': {c: {k: a['scenarios'][c][k] for k in ('sequences','infosets','payoff_entries')} for c in CASES_A}},
              'experiment_b': {'rows': rows_b, 'by_case': by_b, 'ratios': ratios,
                               'lp': {c: b['games'][c]['lp'] for c in CASES_B}}}
    return result


if __name__ == '__main__':
    data = extract_data()
    print(json.dumps({'tables': data['tables'],
                      'comparisons': data['experiment_b']['ratios']}, indent=2))
