# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Test starter Atlas collection controls without mapping a window."""

import json
import pathlib
import unittest
import unittest.mock

import solvcon

try:
    from PySide6 import QtCore, QtTest, QtWidgets
    from solvcon.pilot.panel import _matmul_benchmark
    from solvcon.pilot.panel import _matmul_benchmark_collection
except ImportError:
    QtCore = QtTest = QtWidgets = None
    _matmul_benchmark = _matmul_benchmark_collection = None


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
            self.written = bytearray()
            self.stdout = bytearray()
            self.stderr = bytearray()
            self.terminated = False
            self.killed = False

        def setProcessEnvironment(self, environment):
            self.environment = environment

        def start(self, _program, _arguments):
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
            self.stdout.extend(
                json.dumps(event).encode('utf8') + b'\n')
            self.readyReadStandardOutput.emit()

        def exit(self, code=0):
            self.finished.emit(code, None)


class _FeatureRegistry:
    def names(self):
        return ('M', 'K', 'N')

    def evaluate(self, name, observation):
        return observation[name]

    def register_expression(self, _name, _expression):
        raise AssertionError('not used in starter collection tests')


def _collection_document(plan):
    observations = []
    sources = []
    for index, cell in enumerate(plan['cells'][:2]):
        source_id = f'source-{index}'
        sources.append({
            'source_id': source_id,
            'request': {
                'threads': plan['threads'],
                'mode': plan['mode'],
            },
            'metadata': {},
        })
        observations.append({
            'source_id': source_id,
            'source_observation_index': 0,
            'observation': {
                'id': cell['id'],
                'M': cell['lhs']['shape'][-2],
                'K': cell['lhs']['shape'][-1],
                'N': cell['rhs']['shape'][-1],
                'winner': 'generic',
                'winner_margin': 0.2,
                'candidates': [],
            },
        })
    return {
        'schema_kind': 'solvcon.matmul_benchmark_collection',
        'plan': plan,
        'sources': sources,
        'panels': [],
        'observations': observations,
    }


def _activity(**updates):
    event = {
        'type': 'progress',
        'phase': 'native_batch',
        'state': 'started',
        'route': 'auto',
        'resolved_route': 'blas_gemm',
        'cell_id': 'm2048-k2048-n2048',
        'panel': 1,
        'panels': 2,
        'chunk': 1,
        'completed_calls': 0,
        'total_calls': 5,
        'chunk_calls': 5,
        'event_at_ns': 123,
        'message': 'timing auto with blas_gemm',
        'scope': 'native_batch',
    }
    event.update(updates)
    return event


@unittest.skipUnless(solvcon.HAS_PILOT, 'Qt pilot is not built')
class StarterPlanDialogTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    @staticmethod
    def _set_linear(editor, start, end, step):
        editor._spacing.setCurrentIndex(
            editor._spacing.findData('linear'))
        editor._start.setValue(start)
        editor._end.setValue(end)
        editor._step.setValue(step)

    def test_ranges_support_linear_and_power_of_two_spacing(self):
        linear = {
            'start': 4, 'end': 10, 'spacing': 'linear', 'step': 3,
        }
        powers = {
            'start': 8, 'end': 1024,
            'spacing': 'powers_of_two', 'step': 1,
        }

        self.assertEqual(
            _matmul_benchmark_collection.dimension_values(linear),
            (4, 7, 10))
        self.assertEqual(
            _matmul_benchmark_collection.dimension_values(powers),
            (8, 16, 32, 64, 128, 256, 512, 1024))
        too_dense = {
            'start': 1, 'end': 16_385,
            'spacing': 'linear', 'step': 1,
        }
        with self.assertRaisesRegex(ValueError, '16,385 values'):
            _matmul_benchmark_collection.dimension_values(too_dense)
        self.assertEqual(
            len(_matmul_benchmark_collection.make_plan(
                _matmul_benchmark_collection.default_options()).cells),
            36)

    def test_configured_grid_updates_exact_preflight_estimate(self):
        options = _matmul_benchmark_collection.default_options(
            dtype='float64', threads=3)
        dialog = _matmul_benchmark_collection.StarterPlanDialog(options)
        self._set_linear(dialog._m_range, 4, 8, 4)
        self._set_linear(dialog._k_range, 16, 16, 1)
        self._set_linear(dialog._n_range, 2, 6, 4)

        plan = dialog.plan()

        self.assertEqual(len(plan.cells), 4)
        self.assertEqual(plan.dtype, 'float64')
        self.assertEqual(plan.threads, 3)
        self.assertIn('4 input points', dialog._estimate.text())
        self.assertIn('matmul calls', dialog._estimate.text())
        self.assertIn('estimated work', dialog._estimate.text())
        self.assertIn('conservative peak memory', dialog._estimate.text())
        self.assertIn('current worker-safe budget', dialog._estimate.text())
        self.assertIn('artifact JSON', dialog._estimate.text())
        self.assertIn('Fixed schedule', dialog._estimate.text())

    def test_current_host_memory_disables_execution_not_plan_parsing(self):
        dialog = _matmul_benchmark_collection.StarterPlanDialog(
            _matmul_benchmark_collection.default_options())

        with unittest.mock.patch.object(
                solvcon.matmul_benchmark.collection,
                'validate_plan_resources',
                side_effect=MemoryError('current worker-safe limit')):
            dialog._update_estimate()

        ok = dialog._buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.assertFalse(ok.isEnabled())
        self.assertIn('current worker-safe limit', dialog._estimate.text())

    def test_target_duration_requires_durable_output_and_calibration(self):
        options = _matmul_benchmark_collection.default_options()
        options.update({
            'budget': 'target_duration',
            'target_duration_seconds': 3600.0,
            'checkpoint_seconds': 90.0,
        })
        dialog = _matmul_benchmark_collection.StarterPlanDialog(options)

        ok = dialog._buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.assertFalse(ok.isEnabled())
        self.assertIn('require an Output JSON path', dialog._estimate.text())
        self.assertIn('required for target duration',
                      dialog._output_path.placeholderText())

        dialog._output_path.setText('one-hour-atlas.json')
        dialog._sampling.quality.setCurrentText('Stable')
        plan = dialog.plan()

        self.assertTrue(ok.isEnabled())
        self.assertEqual(plan.target_duration.seconds, 3600.0)
        self.assertEqual(plan.target_duration.mode, 'stable')
        self.assertEqual(plan.target_duration.checkpoint_seconds, 90.0)
        self.assertIn(
            'Target wall-time budget: 1 hour', dialog._estimate.text())
        self.assertIn('Calibration is required', dialog._estimate.text())
        self.assertIn('not elapsed time', dialog._estimate.text())
        self.assertIn('complete balanced measurement rounds',
                      dialog._budget_helper.text())
        self.assertIn('at least 8 measurement rounds',
                      dialog._sampling.helper.text())

    def test_quality_help_and_custom_fixed_schedule_are_explicit(self):
        options = _matmul_benchmark_collection.default_options()
        dialog = _matmul_benchmark_collection.StarterPlanDialog(options)

        self.assertIn('2 setup runs', dialog._sampling.helper.text())
        self.assertIn('5 back-to-back calls',
                      dialog._sampling.helper.text())
        self.assertIn('2 measurement rounds',
                      dialog._sampling.helper.text())

        quality = dialog._sampling.quality
        quality.setCurrentIndex(quality.findData('custom'))
        dialog._sampling.warmups.setValue(3)
        dialog._sampling.repetitions.setValue(7)
        dialog._sampling.rounds.setValue(4)

        self.assertFalse(dialog._sampling.custom_schedule.isHidden())
        self.assertIn('3 setup runs', dialog._sampling.helper.text())
        plan = dialog.plan()
        self.assertEqual(plan.mode.name, 'preview')
        self.assertEqual(plan.mode.warmups, 3)
        self.assertEqual(plan.mode.repetitions, 7)
        self.assertEqual(plan.mode.panels, 4)
        self.assertEqual(dialog.options()['mode'], 'custom')

        dialog._sampling.repetitions.setValue(10_000)
        ok = dialog._buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.assertFalse(ok.isEnabled())
        self.assertIn('too many calls', dialog._sampling.helper.text())
        self.assertIn('Invalid plan', dialog._estimate.text())

        dialog._budget.setCurrentIndex(
            dialog._budget.findData('target_duration'))
        custom_index = dialog._sampling.quality.findData('custom')
        self.assertEqual(dialog._sampling.quality_name(), 'preview')
        self.assertTrue(
            dialog._sampling.quality.view().isRowHidden(custom_index))
        self.assertTrue(dialog._sampling.custom_schedule.isHidden())

    def test_target_duration_rejects_custom_sampling_options(self):
        options = _matmul_benchmark_collection.default_options()
        options.update({
            'mode': 'custom',
            'budget': 'target_duration',
            'output_path': 'target.json',
        })

        with self.assertRaisesRegex(
                ValueError, 'only with a fixed budget'):
            _matmul_benchmark_collection.make_plan(options)

    def test_target_duration_presets_include_custom_and_one_hour(self):
        dialog = _matmul_benchmark_collection.StarterPlanDialog(
            _matmul_benchmark_collection.default_options())
        names = tuple(
            dialog._duration_preset.itemText(index)
            for index in range(dialog._duration_preset.count()))

        self.assertIn('1 hour', names)
        self.assertIn('Custom', names)
        dialog._duration_preset.setCurrentText('Custom')
        dialog._custom_duration.setValue(75)

        self.assertEqual(dialog.options()['target_duration_seconds'], 75)

    def test_dispatch_checkboxes_require_at_least_one_route(self):
        options = _matmul_benchmark_collection.default_options()
        dialog = _matmul_benchmark_collection.StarterPlanDialog(options)
        for box in dialog._routes.boxes.values():
            box.setChecked(False)

        ok = dialog._buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok)

        self.assertFalse(ok.isEnabled())
        self.assertIn('must not be empty', dialog._estimate.text())
        self.assertTrue(all(
            box.toolTip() for box in dialog._routes.boxes.values()))

        options['routes'] = ('generic', 'imaginary_kernel')
        with self.assertRaisesRegex(ValueError, 'Unknown starter routes'):
            _matmul_benchmark_collection.make_plan(options)

    def test_exact_fixed_plan_keeps_slow_dispatch_coverage(self):
        dialog = _matmul_benchmark_collection.StarterPlanDialog(
            _matmul_benchmark_collection.default_options())
        for editor in (
                dialog._m_range, dialog._k_range, dialog._n_range):
            self._set_linear(editor, 2048, 2048, 1)

        plan = dialog.plan()
        generic = dialog._routes.boxes['generic']

        self.assertEqual(plan.cells[0].routes, (
            'generic', 'blas_gemm', 'winograd',
        ))
        self.assertTrue(generic.isChecked())
        self.assertTrue(generic.isEnabled())
        self.assertIn(
            'Generic: 1/1 input points',
            dialog._route_helper.text())
        self.assertIn('BLAS GEMM: 1/1 input points',
                      dialog._route_helper.text())
        self.assertIn('3 dispatches', dialog._estimate.text())
        self.assertIn('generic', dialog.options()['routes'])

    def test_mixed_grid_reports_partial_dispatch_coverage(self):
        dialog = _matmul_benchmark_collection.StarterPlanDialog(
            _matmul_benchmark_collection.default_options())
        self._set_linear(dialog._m_range, 256, 2048, 1792)
        self._set_linear(dialog._k_range, 2048, 2048, 1)
        self._set_linear(dialog._n_range, 2048, 2048, 1)

        generic = dialog._routes.boxes['generic']

        self.assertTrue(generic.isEnabled())
        self.assertTrue(generic.isChecked())
        self.assertIn('Generic: 2/2 input points',
                      dialog._route_helper.text())
        self.assertIn('BLAS GEMM: 2/2 input points',
                      dialog._route_helper.text())
        self.assertIn('3 dispatches', dialog._estimate.text())

    def test_target_duration_does_not_cap_generic_coverage(self):
        options = _matmul_benchmark_collection.default_options()
        options.update({
            'budget': 'target_duration',
            'output_path': 'target.json',
        })
        dialog = _matmul_benchmark_collection.StarterPlanDialog(options)
        for editor in (
                dialog._m_range, dialog._k_range, dialog._n_range):
            self._set_linear(editor, 2048, 2048, 1)

        generic = dialog._routes.boxes['generic']

        self.assertTrue(generic.isEnabled())
        self.assertTrue(generic.isChecked())
        self.assertIn('Generic: 1/1 input points',
                      dialog._route_helper.text())
        self.assertEqual(dialog.plan().cells[0].routes, (
            'generic', 'blas_gemm', 'winograd',
        ))

    def test_stable_2048_target_remains_a_valid_plan(self):
        options = _matmul_benchmark_collection.default_options()
        options.update({
            'budget': 'target_duration',
            'mode': 'stable',
            'output_path': 'stable-2048.json',
        })
        dialog = _matmul_benchmark_collection.StarterPlanDialog(options)
        for editor in (
                dialog._m_range, dialog._k_range, dialog._n_range):
            self._set_linear(editor, 2048, 2048, 1)

        ok = dialog._buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok)

        self.assertTrue(ok.isEnabled())
        self.assertIsNotNone(dialog.plan().target_duration)
        self.assertNotIn('confirmation required', dialog._estimate.text())

    def test_exact_input_profile_expands_into_one_resolved_cell(self):
        dialog = _matmul_benchmark_collection.StarterPlanDialog(
            _matmul_benchmark_collection.default_options())
        self._set_linear(dialog._m_range, 4, 4, 1)
        self._set_linear(dialog._k_range, 4, 4, 1)
        self._set_linear(dialog._n_range, 4, 4, 1)
        profile = solvcon.matmul_benchmark.profiles.InputProfile(
            profile_id='mixed-input',
            name='Mixed input',
            lhs_core=solvcon.matmul_benchmark.profiles.CoreStorage.c_gap(2),
            rhs_core=solvcon.matmul_benchmark.profiles.CoreStorage.f_compact(),
            batch_axes=(solvcon.matmul_benchmark.profiles.BatchAxis(
                output_extent=3,
                lhs_extent='one',
                lhs_stride=(
                    solvcon.matmul_benchmark.profiles.BatchStride.zero()),
            ),),
        )
        dialog._profiles.set_profiles((profile,))

        plan = dialog.plan()

        self.assertEqual(1, len(plan.cells))
        cell = plan.cells[0]
        self.assertEqual((1, 4, 4), cell.lhs.shape)
        self.assertEqual((0, 6, 1), cell.lhs.strides)
        self.assertEqual((3, 4, 4), cell.rhs.shape)
        self.assertEqual((16, 1, 4), cell.rhs.strides)
        self.assertEqual('broadcast_lhs', cell.broadcast)
        self.assertEqual(
            'Mixed input', dialog._profiles._table.item(0, 0).text())

    def test_plan_scale_is_visible_without_stale_confirmation(self):
        with unittest.mock.patch.object(
                solvcon.matmul_benchmark.collection,
                'RECOMMENDED_COLLECTION_CELLS', 1):
            dialog = _matmul_benchmark_collection.StarterPlanDialog(
                _matmul_benchmark_collection.default_options())

        self.assertIn('Plan scale', dialog._estimate.text())
        with unittest.mock.patch.object(
                QtWidgets.QMessageBox, 'warning') as warning:
            dialog.accept()

        warning.assert_not_called()
        self.assertEqual(
            dialog.result(), QtWidgets.QDialog.DialogCode.Accepted)

    def test_output_path_and_reserved_cpu_limit_are_explicit(self):
        options = _matmul_benchmark_collection.default_options(threads=4)
        with unittest.mock.patch.object(
                _matmul_benchmark_collection.os, 'cpu_count',
                return_value=8):
            dialog = _matmul_benchmark_collection.StarterPlanDialog(options)
            inspector = _matmul_benchmark.RouteInspectorWidget()

        output_path = str(pathlib.Path('starter-atlas.json'))
        dialog._output_path.setText(output_path)

        self.assertEqual(dialog._threads.maximum(), 7)
        self.assertEqual(inspector._threads.maximum(), 7)
        self.assertIn('reserving one CPU', dialog._threads.toolTip())
        self.assertIn('reserving one CPU', inspector._threads.toolTip())
        self.assertEqual(dialog.plan().output_path, output_path)

        with unittest.mock.patch.object(
                _matmul_benchmark_collection.os, 'cpu_count',
                return_value=1):
            self.assertEqual(
                _matmul_benchmark_collection.recommended_max_threads(), 1)


@unittest.skipUnless(solvcon.HAS_PILOT, 'Qt pilot is not built')
class StarterAtlasCollectionTC(unittest.TestCase):
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

        self.factory = factory

    def _complete_collection(self, widget):
        process = self.processes[-1]
        request = json.loads(process.written)
        document = _collection_document(request)
        with unittest.mock.patch.object(
                solvcon.matmul_benchmark.artifact, 'load_artifact',
                return_value=document):
            process.emit_json({
                'type': 'result',
                'artifact_path': request['output_path'],
            })
        process.exit()
        return request, document

    def test_one_click_collects_profile_and_loads_mnk_cloud(self):
        widget = _matmul_benchmark._atlas.AtlasWidget(
            _FeatureRegistry(), process_factory=self.factory,
            starter_profile={'dtype': 'float64', 'threads': 3})

        widget.start_starter_collection()
        process = self.processes[0]
        request = json.loads(process.written)

        self.assertEqual(request['dtype'], 'float64')
        self.assertEqual(request['threads'], 3)
        self.assertEqual(len(request['cells']), 36)
        self.assertEqual(request['mode']['name'], 'preview')
        self.assertIsNone(request['target_duration'])
        self.assertIsInstance(request['output_path'], str)
        self.assertTrue(request['output_path'].endswith('artifact.json'))
        self.assertTrue(widget.running)

        process.emit_json({
            'type': 'progress',
            'completed': 7,
            'total': 72,
            'message': 'panel 1/2, cell 7/36',
        })
        self.assertEqual(widget._collection_progress.value(), 7)
        self.assertIn('cell 7/36', widget._collection_status.text())

        _request, document = self._complete_collection(widget)

        self.assertEqual(widget._artifacts, [document])
        self.assertEqual(widget._x_axis.currentText(), 'M')
        self.assertEqual(widget._y_axis.currentText(), 'N')
        self.assertEqual(widget._z_axis.currentText(), 'K')
        self.assertFalse(widget._slice._enabled.isChecked())
        self.assertEqual(len(widget._cloud_canvas._points), 2)
        self.assertIn('Complete: 2 cells', widget._collection_status.text())
        self.assertIn('unsaved', widget._collection_status.text())
        self.assertIs(widget._last_collection, document)
        self.assertFalse(widget._last_collection_saved)
        self.assertTrue(widget._save_collection.isEnabled())
        self.assertFalse(widget._runner.running)

    def test_large_work_starts_without_a_confirmation_dialog(self):
        widget = _matmul_benchmark._atlas.AtlasWidget(
            _FeatureRegistry(), process_factory=self.factory)
        large = solvcon.matmul_benchmark.collection.CollectionEstimate(
            cell_count=36, route_count=3, panel_count=2,
            preflight_calls=1, matmul_calls=1,
            scalar_contractions=1,
            measurement_work=10 ** 30,
            peak_bytes=1,
        )
        with unittest.mock.patch.object(
                solvcon.matmul_benchmark.collection, 'estimate_plan',
                return_value=large):
            with unittest.mock.patch.object(
                    QtWidgets.QMessageBox, 'warning') as warning:
                widget.start_starter_collection()

        request = json.loads(self.processes[0].written)
        warning.assert_not_called()
        self.assertFalse(request['allow_large_work'])
        widget.stop()
        self.processes[0].exit(1)

    def test_target_duration_progress_and_completion_are_explicit(self):
        widget = _matmul_benchmark._atlas.AtlasWidget(
            _FeatureRegistry(), process_factory=self.factory)
        widget._starter_options.update({
            'budget': 'target_duration',
            'target_duration_seconds': 3600.0,
            'checkpoint_seconds': 60.0,
            'output_path': 'one-hour-atlas.json',
        })

        widget.start_starter_collection()
        process = self.processes[0]
        request = json.loads(process.written)

        self.assertEqual(request['target_duration']['seconds'], 3600.0)
        self.assertEqual(widget._collection_progress.minimum(), 0)
        self.assertEqual(widget._collection_progress.maximum(), 0)
        self.assertIn('calibration', widget._collection_status.text())

        process.emit_json({
            'type': 'progress',
            'phase': 'calibration',
            'completed': 3,
            'total': 12,
            'message': 'Calibration 3/12',
        })
        self.assertEqual(widget._collection_progress.maximum(), 12)
        self.assertEqual(widget._collection_progress.value(), 3)

        process.emit_json({
            'type': 'progress',
            'phase': 'measurement',
            'completed': 7,
            'total': 720,
            'message': 'Shard 1/3, panel 1',
        })
        self.assertEqual(widget._collection_progress.maximum(), 720)
        self.assertEqual(widget._collection_progress.value(), 7)
        self.assertIn('Saved chunk 1/3, measurement round 1',
                      widget._collection_status.text())

        document = _collection_document(request)
        raw = document['observations'][0]
        document['aggregate_observations'] = [{
            'cell_id': request['cells'][0]['id'],
            'source_ids': [raw['source_id']],
            'observation': raw['observation'],
        }]
        document['duration_run'] = {
            'requested': {'seconds': 3600.0},
            'actual_elapsed_ns': 3_590_000_000_000,
            'schedule': {
                'predicted': {'central_seconds': 3575.0},
                'panels': 120,
                'repetitions': 40,
                'shard_count': 3,
            },
        }
        with unittest.mock.patch.object(
                solvcon.matmul_benchmark.artifact, 'load_artifact',
                return_value=document):
            process.emit_json({
                'type': 'result',
                'artifact_path': request['output_path'],
            })
        process.exit()

        self.assertEqual(len(widget._observations), 1)
        self.assertIn('Complete: 1 cells', widget._collection_status.text())
        self.assertIn('target 1 hour', widget._collection_status.text())
        self.assertIn('active', widget._collection_status.text())
        self.assertIn('predicted', widget._collection_status.text())
        self.assertIn(
            '120 measurement rounds, 40 calls per result, 3 saved chunks',
            widget._collection_status.text())

    def test_cancelled_duration_run_reports_durable_checkpoint(self):
        widget = _matmul_benchmark._atlas.AtlasWidget(
            _FeatureRegistry(), process_factory=self.factory)
        widget._starter_options.update({
            'budget': 'target_duration',
            'target_duration_seconds': 600.0,
            'output_path': 'resumable-atlas.json',
        })
        widget.start_starter_collection()
        process = self.processes[0]

        process.emit_json({
            'type': 'checkpoint',
            'artifact_path': 'resumable-atlas.checkpoint.json',
            'completed_shards': 1,
            'total_shards': 4,
            'completed_panels': 10,
            'total_panels': 40,
        })
        self.assertIn('Checkpoint 1/4 saved',
                      widget._collection_status.text())
        widget._runner.cancel()
        process.exit(1)

        self.assertIn('rerun the same plan to resume',
                      widget._collection_status.text())
        self.assertIn('resumable-atlas.checkpoint.json',
                      widget._collection_status.text())

        widget.start_starter_collection()

        self.assertIn('Resuming from checkpoint',
                      widget._collection_status.text())
        widget._runner.cancel()
        process.exit(1)

    def test_unsaved_collection_can_be_saved_atomically(self):
        widget = _matmul_benchmark._atlas.AtlasWidget(
            _FeatureRegistry(), process_factory=self.factory)
        widget.start_starter_collection()
        _request, document = self._complete_collection(widget)
        destination = pathlib.Path('exported-atlas.json')

        with unittest.mock.patch.object(
                QtWidgets.QFileDialog, 'getSaveFileName',
                return_value=(str(destination), '')):
            with unittest.mock.patch.object(
                    solvcon.matmul_benchmark.artifact, 'write_artifact',
                    return_value=destination) as write_artifact:
                saved = widget.save_collection()

        self.assertTrue(saved)
        write_artifact.assert_called_once_with(document, str(destination))
        self.assertTrue(widget._last_collection_saved)
        self.assertEqual(widget._last_collection_path, str(destination))
        self.assertIn(str(destination), widget._collection_status.text())

    def test_configured_output_is_durable_without_save_prompt(self):
        widget = _matmul_benchmark._atlas.AtlasWidget(
            _FeatureRegistry(), process_factory=self.factory)
        destination = str(pathlib.Path('configured-atlas.json'))
        widget._starter_options['output_path'] = destination

        widget.start_starter_collection()
        request, _document = self._complete_collection(widget)

        self.assertEqual(request['output_path'], destination)
        self.assertTrue(widget._last_collection_saved)
        self.assertEqual(widget._last_collection_path, destination)
        self.assertIn(f'saved to {destination}',
                      widget._collection_status.text())

    def test_unsaved_collection_blocks_close_until_user_decides(self):
        widget = _matmul_benchmark._atlas.AtlasWidget(
            _FeatureRegistry(), process_factory=self.factory)
        widget.start_starter_collection()
        self._complete_collection(widget)

        with unittest.mock.patch.object(
                QtWidgets.QMessageBox, 'question',
                return_value=QtWidgets.QMessageBox.StandardButton.Cancel):
            self.assertFalse(widget.confirm_close())
        with unittest.mock.patch.object(
                QtWidgets.QMessageBox, 'question',
                return_value=QtWidgets.QMessageBox.StandardButton.Discard):
            self.assertTrue(widget.confirm_close())

        self.assertTrue(widget._last_collection_saved)

    def test_estimate_refreshes_after_profile_changes_during_collection(self):
        widget = _matmul_benchmark._atlas.AtlasWidget(
            _FeatureRegistry(), process_factory=self.factory)
        initial = widget._collection_estimate.text()
        widget.start_starter_collection()

        widget.set_starter_profile('float64', 2)

        self.assertEqual(widget._collection_estimate.text(), initial)
        widget._runner.cancel()
        self.processes[-1].exit(1)
        self.assertNotEqual(widget._collection_estimate.text(), initial)
        self.assertEqual(widget._starter_options['dtype'], 'float64')
        self.assertEqual(widget._starter_options['threads'], 2)

    def test_cancel_terminates_collection_without_publishing(self):
        widget = _matmul_benchmark._atlas.AtlasWidget(
            _FeatureRegistry(), process_factory=self.factory)
        widget.start_starter_collection()
        process = self.processes[0]

        widget._runner.cancel()
        process.exit(1)

        self.assertTrue(process.killed)
        self.assertEqual(widget._artifacts, [])
        self.assertEqual(widget._collection_status.text(), 'Cancelled')

    def test_stop_reports_current_cell_and_preserves_checkpoint(self):
        widget = _matmul_benchmark._atlas.AtlasWidget(
            _FeatureRegistry(), process_factory=self.factory)
        widget.start_starter_collection()
        process = self.processes[0]
        process.emit_json(_activity())
        QtTest.QTest.qWait(120)

        self.assertIn('Running: Auto -> BLAS GEMM',
                      widget._collection_status.text())
        self.assertIn('cell m2048-k2048-n2048',
                      widget._collection_status.text())
        process.emit_json({
            'type': 'checkpoint',
            'artifact_path': 'safe.checkpoint.json',
            'completed_shards': 1,
            'total_shards': 4,
            'completed_panels': 2,
            'total_panels': 8,
        })
        widget.stop()

        self.assertTrue(process.killed)
        self.assertIn('Stopping: Auto -> BLAS GEMM',
                      widget._collection_status.text())
        process.emit_json({
            'type': 'error',
            'error_type': 'RuntimeError',
            'message': 'late failure',
        })
        process.exit(1)

        self.assertIn('Cancelled while Auto -> BLAS GEMM',
                      widget._collection_status.text())
        self.assertIn('safe.checkpoint.json',
                      widget._collection_status.text())
        self.assertTrue(widget._collect.isEnabled())

    def test_worker_error_keeps_context_and_clears_stale_timer(self):
        widget = _matmul_benchmark._atlas.AtlasWidget(
            _FeatureRegistry(), process_factory=self.factory)
        widget.start_starter_collection()
        process = self.processes[0]
        process.emit_json(_activity(route='winograd', resolved_route=None))

        process.emit_json({
            'type': 'error',
            'error_type': 'RuntimeError',
            'message': 'route failed',
        })
        process.exit(1)

        self.assertIn('Error while Winograd',
                      widget._collection_status.text())
        self.assertIn('route failed', widget._collection_status.text())
        self.assertTrue(widget._collect.isEnabled())

        widget.start_starter_collection()
        self.assertIn('Starting fixed schedule',
                      widget._collection_status.text())
        QtTest.QTest.qWait(150)
        self.assertIn('Starting fixed schedule',
                      widget._collection_status.text())
        widget.stop()
        process.exit(1)

    def test_inspector_and_atlas_never_enable_both_workers(self):
        window = _matmul_benchmark.MatmulBenchmarkWindow(
            self.factory, _FeatureRegistry())
        inspector = window.route_inspector
        atlas = window.dispatch_atlas

        inspector._dtype.setCurrentText('complex64')
        inspector._threads.setValue(2)

        self.assertEqual(atlas._starter_options['dtype'], 'complex64')
        self.assertEqual(atlas._starter_options['threads'], 2)

        inspector.start_benchmark()

        self.assertTrue(inspector._runner.running)
        self.assertFalse(atlas._collect.isEnabled())
        self.assertFalse(atlas._configure_collection.isEnabled())

        inspector._runner.cancel()
        self.processes[0].exit(1)

        self.assertTrue(atlas._collect.isEnabled())
        atlas.start_starter_collection()

        self.assertTrue(atlas._runner.running)
        self.assertFalse(inspector._run.isEnabled())

        atlas._runner.cancel()
        self.processes[1].exit(1)

        self.assertTrue(inspector._run.isEnabled())
        window.close()

    def test_closing_window_stops_an_active_atlas_worker(self):
        window = _matmul_benchmark.MatmulBenchmarkWindow(
            self.factory, _FeatureRegistry())
        window.dispatch_atlas.start_starter_collection()
        process = self.processes[1]

        window.close()

        self.assertTrue(process.killed)
        self.assertFalse(window.dispatch_atlas._runner.running)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
