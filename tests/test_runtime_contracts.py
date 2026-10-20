"""Offline regression tests for completion, shard artifacts, and configuration.

Run: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
Only model calls/process creation are stubbed; scoring, wrappers, and finalizers run.
"""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'evaluation'), str(ROOT / 'evaluation/sspro')]
from common.grounding_completion import completion_metadata, result_key, validate_completion
from perceptionbench.artifacts import discover_predictions, prediction_basename
import eval_screenspot_pro_agentic as ss


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = load_module(ROOT / 'evaluation/scripts/merge_osworld_results.py', 'osworld_merge')
BASE_ENV = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG') if k in os.environ}
BASE_ENV['PYTHONDONTWRITEBYTECODE'] = '1'


def run(command, env=None):
    return subprocess.run(command, env=dict(BASE_ENV, **(env or {})),
                          capture_output=True, text=True, timeout=30)


def task(i, negative=False):
    return dict(img_filename=f'image_{i}.png', instruction_style='instruction',
                gt_type='negative' if negative else 'positive', language='en',
                prompt_to_evaluate=f'Target {i}', platform='Linux', application='demo',
                ui_type='text', task_filename='demo', img_size=[100, 100], bbox=[0, 0, 100, 100])


def record(t, args):
    row = dict(t, img_path=os.path.join(args.screenspot_imgs, t['img_filename']), lang=t['language'],
               correctness='correct', samples=[{'result': 'negative' if t['gt_type'] == 'negative' else 'positive'}])
    return row


def spawned_worker(tasks, args, proc_id, num_procs):
    """Real spawned process running the worker with a local model stub."""
    def inference(*unused, **kwargs):
        if args.debug and kwargs['sample']['img_filename'] == 'image_1.png':
            raise RuntimeError('Injected child inference failure')
        return ([{'point': [0.5, 0.5], 'result': 'positive'}], [{'final_content': '(500, 500)'}])
    with patch.object(ss, 'build_model', return_value=Mock()), patch.object(ss, 'run_inference', inference), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ss._run_single_process(tasks, args, proc_id, num_procs)


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='lbs-contract-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        with patch.object(sys, 'argv', ['eval', '--model', 'test-model', '--log_path', str(self.root / 'sspro.json')]):
            self.args = ss.parse_args()
        self.args.screenspot_imgs = str(self.root / 'images')
        self.args.checkpoint_path = str(self.root / 'sspro.ckpt.json')
        self.args.num_workers = 1
        self.tasks = [task(i) for i in range(4)]
        self.rows = [record(t, self.args) for t in self.tasks]

    def report(self, rows=None, limit=None):
        rows = self.rows if rows is None else rows
        self.args.max_samples = limit
        report = ss.evaluate(copy.deepcopy(rows))
        report['run_config'] = ss._run_config_dict(self.args)
        report['completion'] = completion_metadata(rows, [result_key(r) for r in rows], limit)
        return report

    def guard(self, report):
        path = self.root / 'guard.json'
        path.write_text(json.dumps(report))
        return run(['bash', '-c', 'source "$1"; stage_output_ready "$2" test-model completion.scope=full',
                    'test', str(ROOT / 'evaluation/scripts/config.sh'), str(path)]).returncode

    def test_full_report_is_reusable(self):
        self.assertEqual(self.guard(self.report()), 0)

    def test_empty_legacy_or_missing_model_is_not_reusable(self):
        for report in ({}, {'run_config': {'model': 'test-model'}},
                       {'metrics': {'overall': {'num_total': 1}}}):
            with self.subTest(report=report):
                self.assertNotEqual(self.guard(report), 0)
        legacy = self.report(self.rows[:1])
        legacy.pop('completion')
        legacy['run_config'].pop('max_samples')
        path = self.root / 'legacy.json'
        path.write_text(json.dumps(legacy))
        result = run(['bash', '-c', 'source "$1"; stage_output_ready "$2" test-model',
                      'test', str(ROOT / 'evaluation/scripts/config.sh'), str(path)])
        self.assertNotEqual(result.returncode, 0)

    def test_smoke_is_never_full_even_if_limit_covers_dataset(self):
        for limit in (1, 100):
            with self.subTest(limit=limit):
                self.assertNotEqual(self.guard(self.report(self.rows[:1], limit)), 0)

    def test_truncated_duplicate_foreign_and_wrong_model_rejected(self):
        original = self.report()
        for case in ('truncate', 'duplicate', 'foreign', 'model', 'count'):
            report = copy.deepcopy(original)
            if case == 'truncate': report['details'].pop()
            if case == 'duplicate': report['details'][0] = report['details'][1]
            if case == 'foreign': report['details'][0]['prompt_to_evaluate'] = 'unrelated'
            if case == 'model': report['run_config']['model'] = 'different'
            if case == 'count': report['metrics']['overall']['num_total'] = 0
            with self.subTest(case=case):
                self.assertNotEqual(self.guard(report), 0)

    def worker(self, failing=False):
        def inference(*args, **kwargs):
            if failing and kwargs['sample']['img_filename'] == 'image_1.png':
                raise RuntimeError('Injected inference failure')
            return ([{'point': [0.5, 0.5], 'result': 'positive'}], [{'final_content': '(500, 500)'}])
        with patch.object(ss, 'build_model', return_value=Mock()), patch.object(ss, 'run_inference', inference), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return ss._run_single_process(self.tasks, self.args, 0, 1)

    def test_worker_task_failure_preserves_checkpoint_without_report(self):
        for workers in (1, 2):
            self.args.num_workers = workers
            with self.subTest(workers=workers), self.assertRaisesRegex(RuntimeError, 'coverage'):
                self.worker(failing=True)
            self.assertFalse(Path(self.args.log_path).exists())
            self.assertEqual(len(json.loads(Path(self.args.checkpoint_path).read_text())['results']), 3)

    def test_resume_after_failure_finishes_all_tasks(self):
        with self.assertRaises(RuntimeError): self.worker(failing=True)
        self.worker()
        report = json.loads(Path(self.args.log_path).read_text())
        self.assertEqual(validate_completion(report, require_full=True)['completed_tasks'], 4)
        self.assertEqual(report['metrics']['overall']['action_acc'], 1.0)

    def test_stale_checkpoint_cannot_publish_foreign_records(self):
        foreign = record(task(99), self.args)
        Path(self.args.checkpoint_path).write_text(json.dumps({'results': [foreign]}))
        with self.assertRaisesRegex(RuntimeError, 'out-of-shard'): self.worker()
        self.assertFalse(Path(self.args.log_path).exists())
        self.assertEqual(json.loads(Path(self.args.checkpoint_path).read_text())['results'], [foreign])

    def test_real_spawn_failure_then_resume(self):
        self.args.num_processes = 2
        self.args.debug = True
        with patch.object(ss, 'build_tasks_to_run', lambda _: copy.deepcopy(self.tasks)), \
                patch.object(ss, '_run_single_process', spawned_worker), \
                patch.object(ss.random, 'shuffle'), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Worker processes failed'): ss.main(self.args)
            self.assertFalse(Path(self.args.log_path).exists())
            self.assertEqual(len(list(self.root.glob('sspro.ckpt_p*.json'))), 2)
            self.args.debug = False
            ss.main(self.args)
        report = json.loads(Path(self.args.log_path).read_text())
        self.assertEqual(validate_completion(report)['completed_tasks'], 4)

    def coordinator(self, mode):
        args = self.args
        args.num_processes = 2
        class Manager:
            def list(self, values): return values
            def shutdown(self): pass
        class Process:
            def __init__(self, target, args):
                self.params = args
                self.pid = 100 + args[2]
                self.exitcode = None
            def start(self):
                tasks, options, index, count = self.params
                self.exitcode = 2 if mode == 'exit' and index == 1 else 0
                size = (len(tasks) + count - 1) // count
                rows = [record(t, options) for t in tasks[index * size:(index + 1) * size]]
                if index == 1:
                    if mode == 'missing': return
                    if mode == 'partial': rows.pop()
                    if mode == 'duplicate': rows[-1] = rows[0]
                p = Path(options.checkpoint_path)
                p.with_name(p.stem + f'_p{index}' + p.suffix).write_text(json.dumps({'results': rows}))
            def join(self): pass
        ctx = types.SimpleNamespace(Manager=Manager, Process=Process)
        with patch.object(ss, 'build_tasks_to_run', lambda _: copy.deepcopy(self.tasks)), \
                patch.object(multiprocessing, 'get_context', return_value=ctx), \
                contextlib.redirect_stdout(io.StringIO()):
            ss.main(args)

    def test_coordinator_blocks_failed_missing_partial_duplicate_workers(self):
        for mode in ('exit', 'missing', 'partial', 'duplicate'):
            # Independent names avoid a previous case masking a missing checkpoint.
            self.args.checkpoint_path = str(self.root / f'{mode}.ckpt.json')
            with self.subTest(mode=mode), self.assertRaises((RuntimeError, FileNotFoundError)):
                self.coordinator(mode)
            self.assertFalse(Path(self.args.log_path).exists())
            self.assertTrue(list(self.root.glob(f'{mode}.ckpt_p*.json')))

    def test_coordinator_success_and_derived_checkpoint(self):
        self.args.checkpoint_path = None
        self.coordinator('ok')
        self.assertTrue(Path(self.args.checkpoint_path).exists())
        report = json.loads(Path(self.args.log_path).read_text())
        self.assertEqual(validate_completion(report)['completed_tasks'], 4)
        self.assertEqual(report['metrics'], ss.evaluate(self.rows)['metrics'])

    def test_coordinator_smoke_metadata(self):
        self.args.max_samples = 1
        self.coordinator('ok')
        report = json.loads(Path(self.args.log_path).read_text())
        self.assertEqual(report['completion']['scope'], 'smoke')
        self.assertEqual(report['run_config']['max_samples'], 1)
        self.assertNotEqual(self.guard(report), 0)

    def test_invalid_limit_fails_before_launch(self):
        for limit in (0, -1):
            self.args.max_samples = limit
            with self.subTest(limit=limit), self.assertRaises(ValueError): self.coordinator('ok')

    def test_osworld_completion_propagation_and_legacy_coverage(self):
        pos = self.report(self.rows[:1])
        neg_row = record(task(9, negative=True), self.args)
        neg = self.report([neg_row])
        ann = [{'img_filename': 'image_0.png', 'instruction': 'Target 0', 'data_type': 'bbox'},
               {'img_filename': 'image_9.png', 'instruction': 'Target 9', 'data_type': 'refusal'}]
        self.assertEqual(merge.merge_completion(pos, neg, ann)['scope'], 'full')
        for r in (pos, neg): r.pop('completion')
        self.assertEqual(merge.merge_completion(pos, neg, ann)['scope'], 'full')
        with self.assertRaises(ValueError): merge.merge_completion(pos, {'details': []}, ann)
        pos = self.report(self.rows[:1], 1)
        neg = self.report([neg_row], 1)
        self.assertEqual(merge.merge_completion(pos, neg, ann)['scope'], 'smoke')
        neg['completion']['scope'] = 'full'
        with self.assertRaises(ValueError): merge.merge_completion(pos, neg, ann)

    def test_wrapper_failure_keeps_checkpoints_and_does_not_merge(self):
        bindir = self.root / 'bin'; bindir.mkdir()
        stub = bindir / 'python3'
        stub.write_text('#!' + sys.executable + '\nimport sys\nfrom pathlib import Path\na=sys.argv\n'
                        'if "--checkpoint_path" in a:\n p=Path(a[a.index("--checkpoint_path")+1]);p.write_text("checkpoint")\n'
                        'sys.exit(7)\n')
        stub.chmod(0o755)
        env = self.root / 'run.env'
        env.write_text('MODEL=test-model\nOPENAI_API_KEY=test-only\nOPENAI_BASE_URL=http://localhost:9/v1\n')
        for name, checkpoint in (('sspro', 'sspro.ckpt.json'), ('osworld', 'osworld_pos.ckpt.json')):
            result = self.root / name
            process = run(['bash', str(ROOT / f'evaluation/scripts/run_{name}.sh'), str(env), str(result)],
                          {'PATH': str(bindir) + os.pathsep + BASE_ENV['PATH']})
            with self.subTest(name=name):
                self.assertNotEqual(process.returncode, 0)
                self.assertTrue((result / checkpoint).exists())
                self.assertFalse((result / f'{name}.json').exists())


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='lbs-pb-contract-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = 'test/model'

    def fixture(self, directory, n, suffix=''):
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n):
            p = directory / (prediction_basename(self.model, n, i, suffix) + '.jsonl')
            row = dict(index=i, judge_result=i % 2, judge_status='ok', final_answer='A', error_category='test')
            p.write_text(json.dumps(row) + '\n')
            paths.append(p)
        return paths

    def finalize(self, directory, n, suffix='', keep=True, env=None):
        command = [sys.executable, str(ROOT / 'evaluation/scripts/finalize_perceptionbench.py'),
                   '--results', str(directory), '--model', self.model, '--num-shards', str(n)]
        if suffix is not None: command += ['--output-suffix=' + suffix]
        if keep: command += ['--keep-shards']
        return run(command, env)

    def test_finalizer_single_multi_suffix_and_legacy_alias(self):
        for n in (1, 2, 8):
            for suffix in ('', '_custom', '-custom'):
                d = self.root / f'{n}{suffix}'; paths = self.fixture(d, n, suffix)
                r = self.finalize(d, n, suffix)
                with self.subTest(n=n, suffix=suffix):
                    self.assertEqual(r.returncode, 0, r.stderr)
                    scores = json.loads((d / 'perceptionbench_scores.json').read_text())
                    self.assertEqual(scores['counts']['total'], n)
                    self.assertEqual(scores['counts']['correct'], n // 2)
                    self.assertEqual(scores['config']['output_suffix'], suffix)
                    self.assertTrue(all(p.exists() for p in paths))
        d = self.root / 'legacy'; p = self.fixture(d, 1)[0]
        p.rename(p.with_name(p.stem + '_shard0of1.jsonl'))
        self.assertEqual(self.finalize(d, 1).returncode, 0)

    def test_finalizer_reads_suffix_from_environment(self):
        self.fixture(self.root, 2, '_env')
        r = self.finalize(self.root, 2, None, env={'PB_OUTPUT_SUFFIX': '_env'})
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_cleanup_removes_only_consumed_artifacts(self):
        for n in (1, 2):
            d = self.root / str(n); used = self.fixture(d, n, '_used')
            unrelated = self.fixture(d, 1, '_unrelated')
            for p in used: p.with_name(p.stem + '_scores.json').write_text('{}')
            r = self.finalize(d, n, '_used', keep=False)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(all(p.exists() for p in unrelated))
            self.assertTrue(all(not p.exists() for p in used))
            self.assertEqual(sorted(p.name for p in d.glob('*_scores.json')), ['perceptionbench_scores.json'])

    def test_ambiguous_single_inputs_rejected(self):
        p = self.fixture(self.root, 1)[0]
        p.with_name(p.stem + '_shard0of1.jsonl').write_text(p.read_text())
        r = self.finalize(self.root, 1)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Ambiguous', r.stderr)
        self.assertFalse((self.root / 'perceptionbench_scores.json').exists())

    def test_partial_set_cannot_fall_back_to_stale_complete_set(self):
        prefix = prediction_basename(self.model, 1, 0)
        first, fallback = self.root / 'run', self.root / 'fallback'
        self.fixture(first, 2)[1].unlink()
        self.fixture(fallback, 2)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            discover_predictions([first, fallback], prefix, 2)

    def test_invalid_shard_count_index_and_suffix_rejected(self):
        for n, i, suffix in ((0, 0, ''), (-1, 0, ''), (1, 1, ''), (2, -1, ''), (1, 0, '/bad')):
            with self.subTest(n=n, i=i, suffix=suffix), self.assertRaises(ValueError):
                prediction_basename(self.model, n, i, suffix)

    def test_standalone_env_and_explicit_shard_precedence(self):
        bindir = self.root / 'bin'; bindir.mkdir()
        stub = bindir / 'python3'
        stub.write_text('#!' + sys.executable + '\nimport os,json\nfrom pathlib import Path\n'
                        'keys=["NUM_SHARDS","THINKING_BUDGET","AGENTIC_THINKING_BUDGET"]\n'
                        '(Path(os.environ["CAPTURE"])/(os.environ["SHARD_INDEX"]+".json")).write_text('
                        'json.dumps({k:os.getenv(k) for k in keys}))\n')
        stub.chmod(0o755)
        env = self.root / 'run.env'
        env.write_text('MODEL=test-model\nOPENAI_API_KEY=test-only\nOPENAI_BASE_URL=http://localhost:9/v1\n'
                       'PB_NUM_SHARDS=2\nTHINKING_BUDGET=1234\nFINALIZE=0\n')
        for explicit, count in ((None, 2), ('3', 3), ('0', 0), ('-1', 0), ('abc', 0)):
            d = self.root / f'capture_{explicit}'; d.mkdir()
            command = ['bash', str(ROOT / 'evaluation/scripts/run_perceptionbench.sh'), str(env), str(d)]
            if explicit is not None: command.append(explicit)
            r = run(command, {'PATH': str(bindir) + os.pathsep + BASE_ENV['PATH'], 'CAPTURE': str(d)})
            calls = [json.loads(p.read_text()) for p in d.glob('*.json')]
            with self.subTest(explicit=explicit):
                self.assertEqual(len(calls), count, r.stderr)
                self.assertEqual(r.returncode == 0, count > 0)
                self.assertTrue(all(c['NUM_SHARDS'] == str(count) and c['THINKING_BUDGET'] == '1234' for c in calls))

    def test_actual_pb_import_budget_and_pipeline_default(self):
        code = '''import importlib.util,sys,json
s=importlib.util.spec_from_file_location('pb',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
import inspect
print(json.dumps([m.THINKING_BUDGET,inspect.signature(m.AgenticQAPipeline).parameters['thinking_budget'].default]))
'''
        cases = [({}, 4096), ({'THINKING_BUDGET': '1234'}, 1234),
                 ({'THINKING_BUDGET': '1234', 'AGENTIC_THINKING_BUDGET': '2345'}, 2345),
                 ({'AGENTIC_THINKING_BUDGET': '3456'}, 3456)]
        for env, expected in cases:
            r = run([sys.executable, '-c', code, str(ROOT / 'evaluation/perceptionbench/eval_perceptionbench_agentic.py')], env)
            with self.subTest(env=env):
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(json.loads(r.stdout), [expected, expected])

    def test_wrapper_auto_finalizes_suffix_and_skips_matching_run(self):
        bindir = self.root / 'bin'; bindir.mkdir()
        stub = bindir / 'python3'
        stub.write_text('#!' + sys.executable + '\n' + '''import os,sys,json
from pathlib import Path
if Path(sys.argv[1]).name != 'eval_perceptionbench_agentic.py':
    os.execv(sys.executable, [sys.executable] + sys.argv[1:])
sys.path.insert(0, os.getcwd())
from artifacts import prediction_basename
name=prediction_basename(os.environ['MODEL'],int(os.environ['NUM_SHARDS']),int(os.environ['SHARD_INDEX']),os.environ['PB_OUTPUT_SUFFIX'])
path=Path(os.environ['PB_RESULTS_DIR'])/(name+'.jsonl')
path.write_text(json.dumps(dict(index=int(os.environ['SHARD_INDEX']),judge_result=1,judge_status='ok',final_answer='A',error_category='test'))+'\\n')
''')
        stub.chmod(0o755)
        envfile = self.root / 'run.env'
        envfile.write_text('MODEL=test-model\nOPENAI_API_KEY=test-only\nOPENAI_BASE_URL=http://localhost:9/v1\n'
                           'PB_NUM_SHARDS=2\nPB_OUTPUT_SUFFIX=_custom\nCLEAN_INTERMEDIATES=0\n')
        output = self.root / 'results'
        env = {'PATH': str(bindir) + os.pathsep + BASE_ENV['PATH']}
        command = ['bash', str(ROOT / 'evaluation/scripts/run_perceptionbench.sh'), str(envfile), str(output)]
        r = run(command, env)
        self.assertEqual(r.returncode, 0, r.stderr)
        scores = json.loads((output / 'perceptionbench_scores.json').read_text())
        self.assertEqual(scores['counts']['total'], 2)
        self.assertEqual(scores['config']['output_suffix'], '_custom')
        r = run(command, env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('skipping', r.stdout)
        envfile.write_text(envfile.read_text().replace('_custom', '_different'))
        r = run(command, env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads((output / 'perceptionbench_scores.json').read_text())['config']['output_suffix'], '_different')
        # Default-suffix summaries without output_suffix remain reusable.
        envfile.write_text(envfile.read_text().replace('PB_OUTPUT_SUFFIX=_different', 'PB_OUTPUT_SUFFIX='))
        report_path = output / 'perceptionbench_scores.json'
        legacy = json.loads(report_path.read_text())
        legacy['config'].pop('output_suffix')
        report_path.write_text(json.dumps(legacy))
        r = run(command, env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('skipping', r.stdout)

    def test_producer_expressions_match_finalizer_contract(self):
        path = ROOT / 'evaluation/perceptionbench/eval_perceptionbench_agentic.py'
        tree = ast.parse(path.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        assignment = next(n for n in ast.walk(main) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'out_base' for t in n.targets))
        for n in (1, 2):
            namespace = dict(os=os, out_dir=str(self.root), model=self.model, num_shards=n,
                             shard_index=0, prediction_basename=prediction_basename)
            with patch.dict(os.environ, {'PB_OUTPUT_SUFFIX': '_custom'}):
                exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<producer>', 'exec'), namespace)
            self.assertEqual(Path(namespace['out_base']).name, prediction_basename(self.model, n, 0, '_custom'))


if __name__ == '__main__':
    unittest.main()
