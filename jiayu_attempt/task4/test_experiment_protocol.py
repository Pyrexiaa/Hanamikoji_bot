"""Keep full-budget runs and their archived certificates reproducible."""
from contextlib import redirect_stderr
import copy
import gzip
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as runner


class ExperimentProtocolTests(unittest.TestCase):
    def test_default_budget_and_explicit_stopping_modes(self):
        _, config = runner.parse_arguments([])
        self.assertEqual(config['iterations'], 20000)
        self.assertTrue(config['full_budget'])
        self.assertFalse(runner.parse_arguments(['--early-stop'])[1]['full_budget'])
        self.assertTrue(runner.parse_arguments(['--full-budget'])[1]['full_budget'])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            runner.parse_arguments(['--full-budget', '--early-stop'])

    def test_budget_and_history_guards_reject_missing_work(self):
        config = {'iterations': 5, 'full_budget': True, 'check_every': 2}
        learner = SimpleNamespace(iterations=4, history=[{'iteration': i} for i in range(5)])
        with self.assertRaisesRegex(RuntimeError, 'completed 4 of 5'):
            runner.validate_main_run(learner, 'Fictitious play', config)
        learner.iterations = 5
        with self.assertRaisesRegex(RuntimeError, 'every expected certificate'):
            runner.validate_main_run(learner, 'Fictitious play', config)
        learner.history.append({'iteration': 5})
        runner.validate_main_run(learner, 'Fictitious play', config)
        learner.history = [{'iteration': i} for i in (0, 2, 4, 5)]
        runner.validate_main_run(learner, 'CFR', config)
        config['full_budget'] = False
        learner.iterations = 2
        learner.history = [{'iteration': i} for i in (0, 2)]
        runner.validate_main_run(learner, 'CFR', config)

    def test_gzip_preserves_all_observations_and_first_crossing(self):
        config = {'iterations': 60, 'spr_threshold': .002, 'check_every': 25,
                  'history_output_every': 25, 'summary_checkpoints': [25, 50, 60]}
        # A later rise must not hide an earlier crossing between report checkpoints.
        original = [{'iteration': i, 'spr': .001 if i == 33 else .006,
                     'value': .1, 'elapsed_seconds': i / 100} for i in range(61)]
        untouched = copy.deepcopy(original)
        with TemporaryDirectory() as folder:
            record = runner.save_history(original, .125, 'Fictitious play', 'neutral', 0,
                                         config, folder)
            path = Path(folder) / record['history_file']
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                archived = [json.loads(line) for line in stream]
            self.assertEqual(len(archived), record['history_records'])
            self.assertEqual(len(archived), 61)
            self.assertEqual(record['history_cadence'], 1)
            self.assertEqual(record['history_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record['first_target'], archived[33])
            self.assertEqual([q['iteration'] for q in record['history']], [0, 25, 33, 50, 60])
            self.assertEqual(record['summary_checkpoints'], [archived[i] for i in (25, 50, 60)])
            self.assertTrue(record['ever_reached_target'])
            for source, restored in zip(original, archived):
                self.assertEqual(restored['solve_elapsed_seconds'], source['elapsed_seconds'])
                self.assertEqual(restored['elapsed_seconds'], source['elapsed_seconds'] + .125)
            self.assertEqual(original, untouched)

    def test_nonstandard_cfr_cadence_does_not_interpolate(self):
        config = {'iterations': 90, 'spr_threshold': .002, 'check_every': 40,
                  'history_output_every': 25, 'summary_checkpoints': [25, 80, 90]}
        original = [{'iteration': i, 'spr': .1, 'value': 0., 'elapsed_seconds': float(i)}
                    for i in (0, 40, 80, 90)]
        with TemporaryDirectory() as folder:
            record = runner.save_history(original, 0., 'CFR', 'first_player', 2, config, folder)
            self.assertEqual(record['history_records'], 4)
            self.assertEqual(record['history_cadence'], 40)
            self.assertEqual(record['unrecorded_summary_checkpoints'], [25])
            self.assertEqual([q['iteration'] for q in record['summary_checkpoints']], [80, 90])
            self.assertIsNone(record['first_target'])


if __name__ == '__main__':
    unittest.main()
