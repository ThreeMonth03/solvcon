# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Test the benchmark process boundary and view models."""

import json
import pathlib
import unittest
import unittest.mock

import solvcon

try:
    from PySide6 import QtCore, QtTest, QtWidgets
    from solvcon.pilot.panel import _matmul_benchmark
except ImportError:
    QtCore = QtTest = QtWidgets = None
    _matmul_benchmark = None


if QtCore is not None:
    class _FakeProcess(QtCore.QObject):
        started = QtCore.Signal()
        readyReadStandardOutput = QtCore.Signal()
        readyReadStandardError = QtCore.Signal()
        finished = QtCore.Signal(int, object)
        errorOccurred = QtCore.Signal(object)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.environment = None
            self.program = None
            self.arguments = None
            self.written = bytearray()
            self.stdout = bytearray()
            self.stderr = bytearray()
            self.terminated = False
            self.killed = False

        def setProcessEnvironment(self, environment):
            self.environment = environment

        def start(self, program, arguments):
            self.program = program
            self.arguments = list(arguments)
            self.started.emit()

        def write(self, data):
            self.written.extend(bytes(data))

        def closeWriteChannel(self):
            pass

        def readAllStandardOutput(self):
            data = bytes(self.stdout)
            self.stdout.clear()
            return data

        def readAllStandardError(self):
            data = bytes(self.stderr)
            self.stderr.clear()
            return data

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def waitForFinished(self, _milliseconds):
            return True

        def emit_json(self, event):
            self.emit_stdout(json.dumps(event).encode('utf8') + b'\n')

        def emit_stdout(self, payload):
            self.stdout.extend(payload)
            self.readyReadStandardOutput.emit()

        def exit(self, code=0):
            self.finished.emit(code, None)


class _FeatureRegistry:
    def __init__(self):
        self._features = {
            'K': lambda observation: observation['K'],
            'M': lambda observation: observation['M'],
            'N': lambda observation: observation['N'],
            'layout': lambda observation: observation['layout'],
        }

    def names(self):
        return list(self._features)

    def evaluate(self, name, observation):
        return self._features[name](observation)

    def register_expression(self, name, expression):
        if expression != 'M * K':
            raise ValueError('test registry accepts only M * K')
        self._features[name] = (
            lambda observation: observation['M'] * observation['K'])


def _artifact():
    return {
        'schema_version': 1,
        'schema_kind': 'solvcon.matmul_benchmark',
        'artifact_id': 'artifact-id',
        'created_at': '2026-08-15T00:00:00+00:00',
        'request': {
            'lhs': {'shape': [16, 32], 'strides': [32, 1]},
            'rhs': {'shape': [32, 16], 'strides': [16, 1]},
        },
        'metadata': {},
        'panels': [],
        'observations': [
            {
                'M': 16,
                'K': 32,
                'N': 16,
                'layout': 'contiguous',
                'selected': 'generic',
                'winner': 'blas_gemm',
                'winner_margin': 0.25,
                'noise': 0.03,
                'candidates': [
                    {
                        'name': 'generic',
                        'eligible': True,
                        'packing': {
                            'eager_lhs': False,
                            'eager_rhs': False,
                            'scratch_lhs': False,
                            'scratch_rhs': False,
                        },
                        'correctness': {'passed': True},
                        'timing': {
                            'median_ns': 520.0,
                            'p95_ns': 550.0,
                            'mad_ns': 10.4,
                        },
                        'python_timing': {'median_ns': 800.0},
                        'numpy_ratio': 0.8,
                    },
                    {
                        'name': 'blas_gemm',
                        'eligible': True,
                        'packing': {
                            'eager_lhs': True,
                            'eager_rhs': False,
                            'scratch_lhs': False,
                            'scratch_rhs': True,
                        },
                        'correctness': {
                            'passed': True,
                            'max_error': 1e-7,
                        },
                        'timing': {
                            'median_ns': 400.0,
                            'p95_ns': 430.0,
                            'mad_ns': 12.0,
                        },
                        'python_timing': {'median_ns': 620.0},
                        'numpy_ratio': 0.62,
                    },
                ],
            },
            {
                'M': 32,
                'K': 32,
                'N': 16,
                'layout': 'lhs_packed',
                'winner': 'generic',
                'winner_margin': 0.02,
                'noise': 0.15,
                'ambiguous': True,
                'candidates': [],
            },
            {
                'M': 64,
                'K': 64,
                'N': 64,
                'layout': 'contiguous',
                'winner': '',
                'invalid': True,
                'candidates': [],
            },
        ],
    }


def _collection():
    return {
        'schema_version': 1,
        'schema_kind': 'solvcon.matmul_benchmark_collection',
        'collection_id': 'collection-id',
        'created_at': '2026-08-15T00:00:00+00:00',
        'sources': [],
        'panels': [],
        'observations': [],
        'artifact_count': 0,
    }


def _activity(**updates):
    event = {
        'type': 'progress',
        'phase': 'native_batch',
        'state': 'started',
        'route': 'auto',
        'resolved_route': 'winograd',
        'cell_id': None,
        'panel': 1,
        'panels': 2,
        'chunk': 1,
        'completed_calls': 0,
        'total_calls': 5,
        'chunk_calls': 5,
        'event_at_ns': 123,
        'message': 'timing auto with winograd',
        'scope': 'native_batch',
    }
    event.update(updates)
    return event


@unittest.skipUnless(solvcon.HAS_PILOT, 'Qt pilot is not built')
class BenchmarkProcessTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def setUp(self):
        self.processes = []

        def factory(parent):
            process = _FakeProcess(parent)
            self.processes.append(process)
            return process

        self.runner = _matmul_benchmark._process.BenchmarkProcess(factory)

    def test_duration_collection_prefers_logical_aggregates(self):
        raw = [{'source_id': 'raw-1'}, {'source_id': 'raw-2'}]
        aggregate = [{'cell_id': 'cell-1', 'source_ids': ['raw-1']}]

        observations = _matmul_benchmark._process._artifact_observations({
            'observations': raw,
            'aggregate_observations': aggregate,
        })

        self.assertIs(observations, aggregate)

    def test_worker_uses_json_lines_and_thread_environment(self):
        completed = []
        progress = []
        self.runner.completed.connect(completed.append)
        self.runner.progress.connect(
            lambda done, total, message: progress.append(
                (done, total, message)))
        self.runner.start({'schema_version': 1, 'threads': 3})
        process = self.processes[0]

        for name in _matmul_benchmark._process._THREAD_VARIABLES:
            self.assertEqual(process.environment.value(name), '3')
        request = json.loads(process.written)
        self.assertTrue(request['id'])
        self.assertTrue(request['output_path'].endswith('artifact.json'))
        self.assertEqual(
            process.arguments[-3:],
            ['-m', 'solvcon_matmul_benchmark', '--json-lines'])

        process.emit_json({
            'type': 'progress',
            'completed': 1,
            'total': 4,
            'message': 'Measured panel 1',
        })
        pathlib.Path(request['output_path']).write_text(
            json.dumps(_collection()), encoding='utf8')
        process.emit_json({
            'type': 'result',
            'artifact_path': request['output_path'],
        })
        process.exit()

        self.assertEqual(progress, [(1, 4, 'Measured panel 1')])
        self.assertEqual(completed, [_collection()])
        self.assertFalse(self.runner.running)

    def test_activity_is_validated_and_forwarded_without_rewriting(self):
        activities = []
        self.runner.activity.connect(activities.append)
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]
        event = _activity()

        process.emit_json(event)

        self.assertEqual(activities, [event])
        self.runner.cancel()
        process.exit(1)

    def test_terminal_status_does_not_leak_activity_into_the_next_error(self):
        finishers = (
            ('complete', lambda status: status.complete()),
            ('fail', lambda status: status.fail('worker failed')),
            ('cancel', lambda status: status.cancel()),
        )
        for name, finish in finishers:
            with self.subTest(name=name):
                messages = []
                status = _matmul_benchmark._process.RunStatus()
                status.changed.connect(messages.append)
                status.show_activity(_activity())

                finish(status)
                status.fail('invalid next request')

                self.assertEqual(messages[-1], 'Error: invalid next request')

    def test_malformed_activity_fails_closed(self):
        failures = []
        self.runner.failed.connect(failures.append)
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        process.emit_json(_activity(chunk=0))
        process.exit(1)

        self.assertEqual(
            failures,
            ['Invalid worker output: malformed activity event'])
        self.assertTrue(process.terminated)

    def test_unknown_activity_scope_fails_closed(self):
        failures = []
        self.runner.failed.connect(failures.append)
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        process.emit_json(_activity(scope='mixed_timing'))
        process.exit(1)

        self.assertEqual(
            failures,
            ['Invalid worker output: malformed activity event'])
        self.assertTrue(process.terminated)

    def test_embedded_pilot_adds_python_mode(self):
        with unittest.mock.patch.object(
                _matmul_benchmark._process.sys, 'executable',
                '/opt/solvcon/bin/pilot'):
            program, arguments = \
                _matmul_benchmark._process.worker_command()
        self.assertEqual(program, '/opt/solvcon/bin/pilot')
        self.assertEqual(arguments[:3],
                         ['--mode=python', '-m',
                          'solvcon_matmul_benchmark'])

    def test_python_worker_ignores_pilot_process_metadata(self):
        with unittest.mock.patch.object(
                _matmul_benchmark._process.sys, 'executable',
                '/opt/python/bin/python3'):
            program, arguments = \
                _matmul_benchmark._process.worker_command()
        self.assertEqual(program, '/opt/python/bin/python3')
        self.assertEqual(
            arguments,
            ['-m', 'solvcon_matmul_benchmark', '--json-lines'])

    def test_chunked_json_line_is_buffered_until_complete(self):
        completed = []
        self.runner.completed.connect(completed.append)
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]
        request = json.loads(process.written)
        pathlib.Path(request['output_path']).write_text(
            json.dumps(_collection()), encoding='utf8')
        payload = json.dumps({
            'type': 'result',
            'artifact_path': request['output_path'],
        }).encode('utf8') + b'\n'
        split = len(payload) // 2

        process.emit_stdout(payload[:split])
        self.assertEqual(completed, [])
        process.emit_stdout(payload[split:])
        process.exit()

        self.assertEqual(completed, [_collection()])
        self.assertFalse(self.runner.running)

    def test_valid_json_with_an_invalid_event_shape_fails_closed(self):
        failures = []
        self.runner.failed.connect(failures.append)
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        process.emit_json([])
        process.exit(1)

        self.assertEqual(
            failures,
            ['Invalid worker output: event must be an object'])
        self.assertTrue(process.terminated)

    def test_malformed_progress_fails_closed(self):
        failures = []
        self.runner.failed.connect(failures.append)
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        process.emit_json({
            'type': 'progress',
            'completed': 'one',
            'total': 2,
            'message': 'bad',
        })
        process.exit(1)

        self.assertEqual(
            failures,
            ['Invalid worker output: malformed progress event'])
        self.assertTrue(process.terminated)

    def test_checkpoint_reports_progress_without_loading_partial_json(self):
        checkpoints = []
        self.runner.checkpoint.connect(
            lambda *values: checkpoints.append(values))
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        with unittest.mock.patch.object(
                solvcon.matmul_benchmark.artifact,
                'load_artifact') as load_artifact:
            process.emit_json({
                'type': 'checkpoint',
                'artifact_path': '/tmp/atlas.checkpoint.json',
                'completed_shards': 2,
                'total_shards': 5,
                'completed_panels': 20,
                'total_panels': 50,
            })

        self.assertEqual(checkpoints, [(
            '/tmp/atlas.checkpoint.json', 2, 5, 20, 50)])
        load_artifact.assert_not_called()
        self.assertTrue(self.runner.running)
        self.runner.cancel()
        process.exit(1)

    def test_malformed_checkpoint_fails_closed(self):
        failures = []
        self.runner.failed.connect(failures.append)
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        process.emit_json({
            'type': 'checkpoint',
            'artifact_path': '/tmp/atlas.checkpoint.json',
            'completed_shards': 3,
            'total_shards': 2,
            'completed_panels': 20,
            'total_panels': 50,
        })
        process.exit(1)

        self.assertEqual(
            failures,
            ['Invalid worker output: malformed checkpoint event'])
        self.assertTrue(process.terminated)

    def test_malformed_output_reports_one_failure(self):
        failures = []
        self.runner.failed.connect(failures.append)
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        process.emit_stdout(b'not-json\n')
        process.exit(1)

        self.assertEqual(len(failures), 1)
        self.assertIn('Invalid worker output', failures[0])
        self.assertFalse(self.runner.running)

    def test_nonzero_exit_reports_stderr(self):
        failures = []
        self.runner.failed.connect(failures.append)
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]
        process.stderr.extend(b'backend crashed')
        process.readyReadStandardError.emit()

        process.exit(7)

        self.assertEqual(failures, ['backend crashed'])
        self.assertFalse(self.runner.running)

    def test_cancel_kills_worker_without_failure(self):
        cancelled = []
        failures = []
        self.runner.cancelled.connect(lambda: cancelled.append(True))
        self.runner.failed.connect(failures.append)
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        self.runner.cancel()
        process.exit(1)

        self.assertTrue(process.killed)
        self.assertEqual(cancelled, [True])
        self.assertEqual(failures, [])
        self.assertFalse(self.runner.running)

    def test_cancel_ignores_a_late_result_and_emits_one_terminal_signal(self):
        completed = []
        cancelled = []
        self.runner.completed.connect(completed.append)
        self.runner.cancelled.connect(lambda: cancelled.append(True))
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]
        request = json.loads(process.written)
        pathlib.Path(request['output_path']).write_text(
            json.dumps(_collection()), encoding='utf8')

        self.runner.cancel()
        process.emit_json({
            'type': 'result',
            'artifact_path': request['output_path'],
        })
        process.exit()

        self.assertEqual(completed, [])
        self.assertEqual(cancelled, [True])

    def test_cancel_ignores_a_late_worker_error(self):
        failures = []
        cancelled = []
        self.runner.failed.connect(failures.append)
        self.runner.cancelled.connect(lambda: cancelled.append(True))
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        self.runner.cancel()
        process.emit_json({
            'type': 'error',
            'error_type': 'RuntimeError',
            'message': 'late failure',
        })
        process.exit(1)

        self.assertEqual(failures, [])
        self.assertEqual(cancelled, [True])

    def test_cancel_ignores_a_malformed_late_checkpoint(self):
        failures = []
        cancelled = []
        self.runner.failed.connect(failures.append)
        self.runner.cancelled.connect(lambda: cancelled.append(True))
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        self.runner.cancel()
        process.emit_json({
            'type': 'checkpoint',
            'artifact_path': '',
            'completed_shards': 2,
            'total_shards': 1,
            'completed_panels': 0,
            'total_panels': 1,
        })
        process.exit(1)

        self.assertEqual(failures, [])
        self.assertEqual(cancelled, [True])

    def test_result_remains_the_only_terminal_outcome(self):
        completed = []
        failures = []
        self.runner.completed.connect(completed.append)
        self.runner.failed.connect(failures.append)
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]
        request = json.loads(process.written)
        pathlib.Path(request['output_path']).write_text(
            json.dumps(_collection()), encoding='utf8')

        process.emit_json({
            'type': 'result',
            'artifact_path': request['output_path'],
        })
        process.errorOccurred.emit(
            QtCore.QProcess.ProcessError.Crashed)
        process.emit_stdout(b'late malformed output\n')
        process.exit(1)

        self.assertEqual(completed, [_collection()])
        self.assertEqual(failures, [])

    def test_worker_error_remains_the_only_terminal_outcome(self):
        completed = []
        failures = []
        self.runner.completed.connect(completed.append)
        self.runner.failed.connect(failures.append)
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        process.emit_json({
            'type': 'error',
            'error_type': 'RuntimeError',
            'message': 'first failure',
        })
        process.emit_json({
            'type': 'error',
            'error_type': 'RuntimeError',
            'message': 'late failure',
        })
        process.emit_stdout(b'late malformed output\n')
        process.exit(1)

        self.assertEqual(completed, [])
        self.assertEqual(failures, ['first failure'])

    def test_protocol_failure_kills_a_worker_that_does_not_terminate(self):
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        process.emit_stdout(b'not-json\n')
        QtTest.QTest.qWait(1100)

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        process.exit(1)

    def test_cancel_does_not_depend_on_the_terminate_timer(self):
        self.runner.start({'schema_version': 1, 'threads': 1})
        process = self.processes[0]

        self.runner.cancel()

        self.assertTrue(process.killed)
        self.assertFalse(self.runner._kill_timer.isActive())
        process.exit(1)


@unittest.skipUnless(solvcon.HAS_PILOT, 'Qt pilot is not built')
class RouteInspectorTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def setUp(self):
        dispatches = frozenset(_matmul_benchmark._DISPATCH_NAMES)
        self.dispatch_patch = unittest.mock.patch.object(
            _matmul_benchmark.RouteInspectorWidget,
            '_available_dispatches', return_value=dispatches)
        self.dispatch_probe = self.dispatch_patch.start()
        self.addCleanup(self.dispatch_patch.stop)
        self.widget = _matmul_benchmark.RouteInspectorWidget(_FakeProcess)

    def test_request_preserves_arbitrary_shapes_and_element_strides(self):
        self.widget._lhs_shape.setText('2, 1, 7, 11')
        self.widget._lhs_strides.setText('0, 77, 11, 1')
        self.widget._rhs_shape.setText('2 x 1 x 11 x 5')
        self.widget._rhs_strides.setText('0 55 5 1')
        self.widget._threads.setValue(3)
        self.widget._sampling.quality.setCurrentText('Stable')
        self.widget._dispatches.boxes['blas_gemm'].setChecked(True)
        self.widget._dispatches.boxes['generic'].setChecked(True)

        request = self.widget.make_request()

        self.assertEqual(request['lhs']['shape'], [2, 1, 7, 11])
        self.assertEqual(request['lhs']['strides'], [0, 77, 11, 1])
        self.assertEqual(request['rhs']['shape'], [2, 1, 11, 5])
        self.assertEqual(request['rhs']['strides'], [0, 55, 5, 1])
        self.assertEqual(request['routes'], ['generic', 'blas_gemm'])
        self.assertEqual(request['mode']['name'], 'stable')
        self.assertEqual(request['mode']['panels'], 8)

    def test_dispatch_checkboxes_default_to_every_eligible_dispatch(self):
        self.assertEqual(tuple(self.widget._dispatches.boxes), (
            'generic', 'blas_dot', 'blas_gevm', 'blas_gemv',
            'blas_gemm', 'winograd',
        ))
        self.assertEqual(self.widget._dispatches.selected(), (
            'generic', 'blas_gemm', 'winograd',
        ))
        self.assertTrue(self.widget._dispatches.boxes['generic'].isEnabled())
        self.assertFalse(
            self.widget._dispatches.boxes['blas_dot'].isEnabled())
        self.assertIn(
            '3 dispatches apply', self.widget._dispatch_helper.text())
        self.assertEqual(self.widget.make_request()['routes'], [
            'generic', 'blas_gemm', 'winograd',
        ])

    def test_large_shape_keeps_structurally_eligible_dispatches(self):
        self.widget._lhs_shape.setText('2048, 2048')
        self.widget._rhs_shape.setText('2048, 2048')

        generic = self.widget._dispatches.boxes['generic']
        self.assertTrue(generic.isEnabled())
        self.assertTrue(generic.isChecked())
        self.assertTrue(
            self.widget._dispatches.boxes['blas_gemm'].isEnabled())
        self.assertTrue(
            self.widget._dispatches.boxes['winograd'].isEnabled())
        self.assertEqual(self.widget.make_request()['routes'], [
            'generic', 'blas_gemm', 'winograd',
        ])

    def test_sampling_does_not_change_dispatch_eligibility(self):
        self.widget._lhs_shape.setText('1024, 1024')
        self.widget._rhs_shape.setText('1024, 1024')
        generic = self.widget._dispatches.boxes['generic']
        generic.setChecked(False)

        self.widget._sampling.quality.setCurrentText('Stable')

        self.assertTrue(generic.isEnabled())
        self.assertFalse(generic.isChecked())

        self.widget._sampling.quality.setCurrentText('Preview')

        self.assertTrue(generic.isEnabled())
        self.assertFalse(generic.isChecked())

    def test_custom_schedule_keeps_generic_for_2048(self):
        self.widget._lhs_shape.setText('2048, 2048')
        self.widget._rhs_shape.setText('2048, 2048')
        generic = self.widget._dispatches.boxes['generic']
        self.assertTrue(generic.isEnabled())

        quality = self.widget._sampling.quality
        quality.setCurrentIndex(quality.findData('custom'))
        self.widget._sampling.warmups.setValue(0)
        self.widget._sampling.repetitions.setValue(1)
        self.widget._sampling.rounds.setValue(1)

        self.assertTrue(generic.isEnabled())
        self.assertTrue(generic.isChecked())
        self.assertIn('3 dispatches apply',
                      self.widget._dispatch_helper.text())

    def test_stable_2048_keeps_all_eligible_dispatches_runnable(self):
        self.widget._lhs_shape.setText('2048, 2048')
        self.widget._rhs_shape.setText('2048, 2048')
        self.widget._sampling.quality.setCurrentText('Stable')

        self.assertEqual(
            self.widget._dispatches.selected(),
            ('generic', 'blas_gemm', 'winograd'))
        self.assertTrue(self.widget._run.isEnabled())
        self.assertNotIn(
            'Current selection cannot run',
            self.widget._dispatch_helper.text())
        self.assertEqual(
            self.widget.make_request()['mode']['name'], 'stable')

    def test_huge_accelerated_schedule_has_no_time_based_ui_guard(self):
        gibibyte = 1024 ** 3
        budget = solvcon.matmul_benchmark.arrays.ResourceBudget(
            available_bytes=60 * gibibyte,
            peak_bytes=30 * gibibyte,
            single_allocation_bytes=4 * gibibyte)
        with unittest.mock.patch.object(
                solvcon.matmul_benchmark.arrays,
                'resolve_resource_budget', return_value=budget):
            self.widget._lhs_shape.setText('16384, 16384')
            self.widget._rhs_shape.setText('16384, 16384')

            self.assertTrue(self.widget._run.isEnabled())
            self.assertEqual(self.widget.make_request()['routes'], [
                'generic', 'blas_gemm', 'winograd',
            ])

    def test_run_requires_a_selected_runnable_dispatch(self):
        for box in self.widget._dispatches.boxes.values():
            if box.isEnabled():
                box.setChecked(False)

        self.assertFalse(self.widget._run.isEnabled())
        with self.assertRaisesRegex(
                ValueError, 'Select at least one runnable dispatch'):
            self.widget.make_request()

    def test_progress_uses_plain_sampling_terms(self):
        self.widget._show_progress(
            1, 2, 'completed panel 1/2')

        self.assertEqual(self.widget._progress.maximum(), 2)
        self.assertEqual(self.widget._progress.value(), 1)
        self.assertEqual(
            self.widget._status.text(),
            'completed measurement round 1/2')

    def test_request_accepts_a_slow_programmatic_selection(self):
        self.widget._lhs_shape.setText('2048, 2048')
        self.widget._rhs_shape.setText('2048, 2048')
        generic = self.widget._dispatches.boxes['generic']
        generic.setEnabled(True)
        generic.setChecked(True)

        self.assertIn('generic', self.widget.make_request()['routes'])

    def test_stop_reports_current_activity_and_resets_controls(self):
        self.widget.start_benchmark()
        process = self.widget._runner._process
        process.emit_json(_activity())
        QtTest.QTest.qWait(120)

        self.assertIn('Running: Auto -> Winograd',
                      self.widget._status.text())
        self.assertIn('phase: native timing', self.widget._status.text())
        self.assertIn('call 1-5/5', self.widget._status.text())
        self.widget.stop()

        self.assertTrue(process.killed)
        self.assertIn('Stopping: Auto -> Winograd',
                      self.widget._status.text())
        process.emit_json(_activity(state='completed', completed_calls=5,
                                    elapsed_ns=130_000_000))
        process.exit(1)

        self.assertIn('Cancelled while Auto -> Winograd',
                      self.widget._status.text())
        self.assertIn('elapsed:', self.widget._status.text())
        self.assertTrue(self.widget._run.isEnabled())

    def test_preparation_activity_uses_plain_status_labels(self):
        self.widget.start_benchmark()
        process = self.widget._runner._process

        process.emit_json(_activity(
            phase='preparation', route='input', resolved_route=None,
            panel=None, panels=None, total_calls=1, chunk_calls=1,
            scope=None))

        self.assertIn('Running: Inputs', self.widget._status.text())
        self.assertIn('phase: input preparation',
                      self.widget._status.text())
        self.widget.stop()
        process.exit(1)

    def test_provenance_activity_uses_plain_status_labels(self):
        self.widget.start_benchmark()
        process = self.widget._runner._process

        process.emit_json(_activity(
            phase='provenance', route='input', resolved_route=None,
            panel=None, panels=None, total_calls=1, chunk_calls=1,
            scope=None))

        self.assertIn('Running: Environment', self.widget._status.text())
        self.assertIn('phase: recording machine and build details',
                      self.widget._status.text())
        self.widget.stop()
        process.exit(1)

    def test_setup_activity_explains_its_timing_scope(self):
        self.widget.start_benchmark()
        process = self.widget._runner._process

        process.emit_json(_activity(
            phase='warmup', route='winograd', resolved_route=None,
            panel=None, panels=None, scope='python_end_to_end'))

        self.assertIn('phase: Python end-to-end setup run',
                      self.widget._status.text())
        self.widget.stop()
        process.exit(1)

    def test_result_and_checkpoint_io_never_show_a_stale_dispatch(self):
        self.widget.start_benchmark()
        process = self.widget._runner._process
        cases = (
            ('finalization', 'artifact', 'Result',
             'building and validating result'),
            ('checkpoint_load', 'checkpoint', 'Checkpoint',
             'loading saved progress'),
            ('checkpoint_write', 'checkpoint', 'Checkpoint',
             'saving progress'),
            ('artifact_write', 'artifact', 'Result',
             'saving final result'),
        )

        for phase, route, route_label, phase_label in cases:
            with self.subTest(phase=phase):
                process.emit_json(_activity(
                    phase=phase, route=route, resolved_route=None,
                    panel=None, panels=None, total_calls=1,
                    chunk_calls=1, scope=None))
                self.assertIn(f'Running: {route_label}',
                              self.widget._status.text())
                self.assertIn(f'phase: {phase_label}',
                              self.widget._status.text())

        self.widget.stop()
        process.exit(1)

    def test_worker_error_keeps_activity_context_and_allows_restart(self):
        self.widget.start_benchmark()
        process = self.widget._runner._process
        process.emit_json(_activity(route='generic', resolved_route=None))
        process.emit_json(_activity(
            state='failed', route='generic', resolved_route=None,
            elapsed_ns=20_000_000))

        self.assertIn('Failed: Generic', self.widget._status.text())
        self.assertIn('call 1-5/5', self.widget._status.text())

        process.emit_json({
            'type': 'error',
            'error_type': 'RuntimeError',
            'message': 'kernel failed',
        })
        process.exit(1)

        self.assertTrue(process.terminated)
        self.assertIn('Error while Generic', self.widget._status.text())
        self.assertIn('kernel failed', self.widget._status.text())
        self.assertTrue(self.widget._run.isEnabled())

        self.widget.start_benchmark()
        self.assertEqual(self.widget._status.text(), 'Starting worker...')
        QtTest.QTest.qWait(150)
        self.assertEqual(self.widget._status.text(), 'Starting worker...')
        self.widget.stop()
        process.exit(1)

    def test_dispatch_checkboxes_use_stable_request_order(self):
        self.widget._dispatches.boxes['blas_gemm'].setChecked(False)
        self.widget._dispatches.boxes['generic'].setChecked(False)
        self.widget._dispatches.boxes['winograd'].setChecked(False)
        self.widget._dispatches.boxes['winograd'].setChecked(True)
        self.widget._dispatches.boxes['generic'].setChecked(True)

        self.assertEqual(
            self.widget.make_request()['routes'], ['generic', 'winograd'])

    def test_dispatch_checkboxes_cover_vector_roles(self):
        cases = (
            ('16', '16', 'blas_dot'),
            ('16', '16, 16', 'blas_gevm'),
            ('16, 16', '16', 'blas_gemv'),
        )
        for lhs, rhs, expected in cases:
            with self.subTest(lhs=lhs, rhs=rhs):
                self.widget._lhs_shape.setText(lhs)
                self.widget._rhs_shape.setText(rhs)
                enabled = tuple(
                    name for name, box in
                    self.widget._dispatches.boxes.items()
                    if box.isEnabled())
                self.assertEqual(enabled, ('generic', expected))
                self.assertEqual(
                    self.widget.make_request()['routes'],
                    ['generic', expected])

    def test_dispatch_checkboxes_follow_shape_roles_and_dimensions(self):
        self.widget._lhs_shape.setText('15, 16')
        self.widget._rhs_shape.setText('16, 17')

        self.assertTrue(
            self.widget._dispatches.boxes['blas_gemm'].isEnabled())
        self.assertFalse(
            self.widget._dispatches.boxes['winograd'].isEnabled())
        self.assertEqual(self.widget.make_request()['routes'], [
            'generic', 'blas_gemm',
        ])

        self.widget._lhs_shape.setText('2, 16, 16')
        self.widget._rhs_shape.setText('2, 16, 18')

        self.assertFalse(
            self.widget._dispatches.boxes['winograd'].isEnabled())
        self.assertEqual(self.widget.make_request()['routes'], [
            'generic', 'blas_gemm',
        ])

    def test_dispatch_checkboxes_preserve_preferences_through_invalid_input(
            self):
        gemm = self.widget._dispatches.boxes['blas_gemm']
        gemm.setChecked(False)
        self.widget._lhs_strides.setText('1')

        self.assertFalse(self.widget._run.isEnabled())
        self.assertFalse(any(
            box.isEnabled() for box in self.widget._dispatches.boxes.values()))
        self.assertIn(
            'selections are preserved', self.widget._dispatch_helper.text())

        self.widget._lhs_strides.clear()

        self.assertTrue(self.widget._run.isEnabled())
        self.assertTrue(gemm.isEnabled())
        self.assertFalse(gemm.isChecked())
        self.assertTrue(
            self.widget._dispatches.boxes['winograd'].isChecked())

    def test_dispatch_checkboxes_disable_routes_missing_from_build(self):
        dispatches = frozenset(_matmul_benchmark._DISPATCH_NAMES)
        self.dispatch_probe.side_effect = lambda dtype: (
            frozenset({'generic'}) if dtype == 'float64' else dispatches)
        self.widget._dtype.setCurrentText('float64')

        generic = self.widget._dispatches.boxes['generic']
        gemm = self.widget._dispatches.boxes['blas_gemm']
        self.assertTrue(generic.isEnabled())
        self.assertTrue(generic.isChecked())
        self.assertFalse(gemm.isEnabled())
        self.assertFalse(gemm.isChecked())
        self.assertEqual(self.widget.make_request()['routes'], ['generic'])

    def test_sampling_help_and_custom_fixed_schedule_match_atlas(self):
        self.assertIn('2 setup runs', self.widget._sampling.helper.text())
        self.assertIn('5 back-to-back calls',
                      self.widget._sampling.helper.text())

        quality = self.widget._sampling.quality
        quality.setCurrentIndex(quality.findData('custom'))
        self.widget._sampling.warmups.setValue(3)
        self.widget._sampling.repetitions.setValue(7)
        self.widget._sampling.rounds.setValue(4)

        request = self.widget.make_request()
        self.assertEqual(request['mode'], {
            'name': 'preview',
            'warmups': 3,
            'repetitions': 7,
            'panels': 4,
        })
        self.assertIn('4 measurement rounds',
                      self.widget._sampling.helper.text())
        self.assertTrue(self.widget._run.isEnabled())

        self.widget._sampling.repetitions.setValue(10_000)
        self.assertFalse(self.widget._run.isEnabled())
        self.assertIn('too many calls',
                      self.widget._sampling.helper.text())

    def test_result_exposes_selection_packing_and_correctness(self):
        self.widget.show_artifact(_artifact())

        self.assertEqual(self.widget._table.rowCount(), 2)
        blas_row = 1
        columns = {
            name: index for index, name in enumerate(self.widget.HEADERS)}
        self.assertEqual(
            self.widget._table.item(
                blas_row, columns['Winner']).text(), 'yes')
        self.assertEqual(
            self.widget._table.item(
                blas_row, columns['Eager pack']).text(), 'lhs')
        self.assertEqual(
            self.widget._table.item(
                blas_row, columns['Scratch pack']).text(), 'rhs')
        self.assertTrue(
            self.widget._table.item(
                blas_row, columns['Correctness']).text().startswith('pass'))
        self.assertEqual(
            self.widget._table.item(
                blas_row, columns['Native median']).text(), '400.0 ns')
        self.assertEqual(
            self.widget._table.item(
                blas_row, columns['Python E2E median']).text(), '620.0 ns')
        self.assertEqual(
            self.widget._table.item(
                blas_row, columns['vs NumPy']).text(), '0.620x')

    def test_closing_window_stops_an_active_worker(self):
        window = _matmul_benchmark.MatmulBenchmarkWindow(_FakeProcess)
        window.show()
        window.route_inspector.start_benchmark()
        process = window.route_inspector._runner._process

        window.close()

        self.assertTrue(process.killed)
        self.assertFalse(window.route_inspector._runner.running)

    def test_global_stop_follows_the_active_worker_across_tabs(self):
        window = _matmul_benchmark.MatmulBenchmarkWindow(_FakeProcess)
        inspector = window.route_inspector
        atlas = window.dispatch_atlas

        self.assertFalse(window._stop.isEnabled())
        inspector.start_benchmark()
        inspector_process = inspector._runner._process
        window._tabs.setCurrentWidget(atlas)

        self.assertTrue(window._stop.isEnabled())
        self.assertEqual(window._stop.text(), 'Stop Route Inspector')
        window._stop.click()

        self.assertTrue(inspector_process.killed)
        self.assertFalse(window._stop.isEnabled())
        self.assertEqual(window._stop.text(),
                         'Stopping Route Inspector...')
        inspector_process.exit(1)
        self.assertEqual(window._stop.text(), 'Stop')

        atlas.start_starter_collection()
        atlas_process = atlas._runner._process
        window._tabs.setCurrentWidget(inspector)

        self.assertTrue(window._stop.isEnabled())
        self.assertEqual(window._stop.text(), 'Stop Dispatch Atlas')
        window._stop.click()

        self.assertTrue(atlas_process.killed)
        self.assertFalse(window._stop.isEnabled())
        atlas_process.exit(1)
        self.assertEqual(window._stop.text(), 'Stop')
        window.close()

    def test_window_exposes_current_operation_without_genericizing_data(self):
        window = _matmul_benchmark.MatmulBenchmarkWindow(_FakeProcess)

        self.assertEqual(
            _matmul_benchmark.MatmulBenchmark.TITLE,
            'Benchmark Visualizer',
        )
        self.assertEqual(window._operation.count(), 1)
        self.assertEqual(window._operation.currentText(), 'Matmul')
        self.assertIn('matmul benchmarks only', window._operation.toolTip())
        self.assertEqual(window._tabs.count(), 2)

        window.close()


@unittest.skipUnless(solvcon.HAS_PILOT, 'Qt pilot is not built')
class AtlasTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def setUp(self):
        self.loads = []

        def load(path):
            self.loads.append(path)
            return _artifact()

        self.widget = _matmul_benchmark._atlas.AtlasWidget(
            _FeatureRegistry(), load)

    def test_axes_and_slices_only_reproject_loaded_observations(self):
        self.widget.load_artifacts(('first.json', 'second.json'))
        self.assertEqual(self.loads, ['first.json', 'second.json'])
        self.widget._x_axis.setCurrentText('M')
        self.widget._y_axis.setCurrentText('K')
        self.widget._z_axis.setCurrentText('N')
        self.widget._slice._feature.setCurrentText('layout')
        self.widget._slice._enabled.setChecked(True)
        self.widget._slice._categorical.setCurrentText('contiguous')
        self.widget.render()

        self.assertEqual(self.loads, ['first.json', 'second.json'])
        self.assertEqual(len(self.widget._cloud_canvas._points), 2)
        self.assertEqual(
            sum(point['sample_count']
                for point in self.widget._cloud_canvas._points),
            4)
        self.assertEqual(
            sum(point['invalid_count']
                for point in self.widget._cloud_canvas._points),
            2)

    def test_numeric_and_categorical_constraints_use_distinct_controls(self):
        self.widget.add_artifacts([_artifact()])
        self.widget._slice._feature.setCurrentText('layout')
        self.widget._slice._enabled.setChecked(True)
        self.assertEqual(self.widget._slice._values_stack.currentIndex(), 0)

        self.widget._filter._feature.setCurrentText('M')
        self.widget._filter._enabled.setChecked(True)
        self.assertEqual(self.widget._filter._values_stack.currentIndex(), 1)

    def test_layout_detects_implicit_batch_broadcasting(self):
        observation = {
            'lhs': {'shape': [2, 3], 'strides': [3, 1]},
            'rhs': {'shape': [4, 3, 2], 'strides': [6, 2, 1]},
            'contraction': {'batch_shape': [4]},
        }
        adapter = _matmul_benchmark._atlas.FeatureRegistryAdapter(
            _FeatureRegistry())

        self.assertEqual(
            adapter.evaluate('lhs_layout', observation), 'row-major')
        self.assertEqual(
            adapter.evaluate('rhs_layout', observation), 'row-major')
        self.assertEqual(
            adapter.evaluate('lhs_broadcast', observation), 'yes')
        self.assertEqual(
            adapter.evaluate('rhs_broadcast', observation), 'no')
        self.assertEqual(
            adapter.evaluate('broadcast_profile', observation), 'yes / no')

        batched_column_major = {
            'shape': [4, 2, 3],
            'strides': [6, 1, 2],
        }
        self.assertEqual(
            adapter._layout_kind(batched_column_major), 'column-major')
        self.assertEqual(
            adapter._layout_kind({
                'shape': [4, 2, 3],
                'strides': [16, 4, 1],
            }),
            'row-major-padded')

    def test_projection_groups_conflicting_winners(self):
        artifact = _artifact()
        first = artifact['observations'][0]
        second = dict(first)
        second['id'] = 'same-coordinate'
        second['winner'] = 'generic'
        artifact['observations'] = [first, second]

        self.widget.add_artifacts([artifact])
        self.widget._x_axis.setCurrentText('M')
        self.widget._y_axis.setCurrentText('K')
        self.widget._z_axis.setCurrentText('N')
        self.widget.render()

        self.assertEqual(len(self.widget._cloud_canvas._points), 1)
        point = self.widget._cloud_canvas._points[0]
        self.assertEqual(point['sample_count'], 2)
        self.assertEqual(point['hidden_sample_count'], 1)
        self.assertEqual(point['routes'], ('blas_gemm', 'generic'))
        self.assertEqual(
            point['winner_counts'], {'blas_gemm': 1, 'generic': 1})
        self.assertTrue(point['conflicting'])
        self.assertTrue(point['ambiguous'])

        canvas = self.widget._cloud_canvas
        canvas.resize(640, 400)
        image = canvas.grab().toImage()

        self.assertFalse(image.isNull())
        self.assertEqual(len(canvas._projected), 1)
        tooltip = canvas.point_tooltip(point)
        self.assertIn('M: 16', tooltip)
        self.assertIn('K: 32', tooltip)
        self.assertIn('blas_gemm: 1', tooltip)
        self.assertIn('generic: 1', tooltip)
        self.assertIn('Mixed or near-tie results: yes', tooltip)
        self.assertIn('same-coordinate', tooltip)

    def test_runner_up_noise_marks_a_boundary_ambiguous(self):
        observation = _artifact()['observations'][0]
        observation.pop('noise')
        observation['runner_up'] = 'generic'
        observation['winner_margin'] = 0.1
        routes = {
            candidate['name']: candidate
            for candidate in observation['candidates']
        }
        routes['blas_gemm']['timing']['mad_ns'] = 4.0
        routes['generic']['timing']['mad_ns'] = 156.0

        point = self.widget._make_point(observation, 16.0, 32.0)

        self.assertAlmostEqual(point['noise'], 0.3)
        self.assertTrue(point['ambiguous'])

    def test_collection_observations_keep_their_source_context(self):
        artifact = _artifact()
        requests = []
        for threads, mode in ((1, 'preview'), (8, 'stable')):
            request = dict(artifact['request'])
            request['threads'] = threads
            request['mode'] = {'name': mode}
            requests.append(request)
        sources = [
            {
                'source_id': f'source-{index}',
                'artifact_id': f'artifact-{index}',
                'path': None,
                'created_at': artifact['created_at'],
                'request': requests[index],
                'metadata': {
                    'machine': {
                        'machine': 'arm64',
                        'node': f'host-{index}',
                        'system': 'Darwin',
                    },
                },
            }
            for index in range(2)
        ]
        collection = {
            'schema_version': 1,
            'schema_kind': 'solvcon.matmul_benchmark_collection',
            'collection_id': 'collection-id',
            'created_at': artifact['created_at'],
            'sources': sources,
            'panels': [],
            'observations': [
                {
                    'source_id': f'source-{index}',
                    'source_observation_index': 0,
                    'observation': artifact['observations'][index],
                }
                for index in range(2)
            ],
            'artifact_count': 2,
        }

        self.widget.add_artifacts([collection])

        contexts = [observation['_artifact_context']
                    for observation in self.widget._observations]
        self.assertEqual(
            [context['machine'] for context in contexts],
            ['host-0', 'host-1'])
        self.assertEqual(
            [context['threads'] for context in contexts], [1, 8])
        self.assertEqual(
            [context['mode'] for context in contexts],
            [('preview', 2, 5, 2), ('stable', 4, 20, 8)])
        self.assertEqual(
            [observation['M'] for observation in self.widget._observations],
            [16, 32])

    def test_mode_context_distinguishes_custom_sampling_counts(self):
        first = {
            'request': {
                'mode': {
                    'name': 'preview', 'warmups': 2,
                    'repetitions': 5, 'panels': 2,
                },
            },
            'metadata': {},
        }
        second = {
            'request': {
                'mode': {
                    'name': 'preview', 'warmups': 3,
                    'repetitions': 7, 'panels': 4,
                },
            },
            'metadata': {},
        }

        first_context = self.widget._artifact_context(first)
        second_context = self.widget._artifact_context(second)

        self.assertEqual(first_context['mode'], ('preview', 2, 5, 2))
        self.assertEqual(second_context['mode'], ('preview', 3, 7, 4))
        self.assertNotEqual(first_context['mode'], second_context['mode'])
        observation = {'_artifact_context': second_context}
        self.assertEqual(
            self.widget._registry.evaluate('mode', observation),
            'Custom: 3 setup runs, 7 calls per result, '
            '4 measurement rounds')

        duration_document = {
            'duration_run': {
                'template_plan': {'mode': {
                    'name': 'stable', 'warmups': 4,
                    'repetitions': 20, 'panels': 8,
                }},
                'schedule': {'repetitions': 40, 'panels': 120},
            },
        }
        duration_signature = \
            _matmul_benchmark._atlas._duration_mode_signature(
                duration_document)
        self.assertEqual(duration_signature, ('stable', 4, 40, 120))
        self.assertEqual(
            _matmul_benchmark._atlas._mode_description(
                duration_signature, target_duration=True),
            'Stable calibrated: 4 setup runs, 40 calls per result, '
            '120 measurement rounds')


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
