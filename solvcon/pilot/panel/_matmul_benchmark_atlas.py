# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Offline feature projection and painting for matmul artifacts."""

import math

from PySide6 import QtCore, QtWidgets

from ... import matmul_benchmark
from . import _matmul_benchmark_cloud as _cloud
from . import _matmul_benchmark_collection as _collection
from . import _matmul_benchmark_process as _process

__all__ = [
    'AtlasWidget',
    'FeatureRegistryAdapter',
]


class FeatureRegistryAdapter:
    """Normalize the engine feature registry behind a small GUI interface."""

    CATEGORICAL_NAMES = (
        'architecture',
        'broadcast_profile',
        'dtype',
        'lhs_broadcast',
        'lhs_layout',
        'layout_profile',
        'machine',
        'mode',
        'rhs_broadcast',
        'rhs_layout',
        'system',
        'threads',
    )

    def __init__(self, registry=None):
        self._registry = (
            registry or matmul_benchmark.features.FeatureRegistry())

    def names(self):
        return list(self._registry.names())

    def constraint_names(self):
        return sorted(set(self.names()) | set(self.CATEGORICAL_NAMES))

    def evaluate(self, name, observation):
        if name in self.CATEGORICAL_NAMES:
            return self._categorical_value(name, observation)
        return self._registry.evaluate(name, observation)

    def add_expression(self, name, expression):
        self._registry.register_expression(name, expression)

    @staticmethod
    def _categorical_value(name, observation):
        context = observation.get('_artifact_context', {})
        batch_shape = observation.get(
            'contraction', {}).get('batch_shape')
        if name == 'dtype':
            return observation.get('dtype')
        if name == 'lhs_broadcast':
            return FeatureRegistryAdapter._broadcast_kind(
                observation.get('lhs', {}), batch_shape)
        if name == 'rhs_broadcast':
            return FeatureRegistryAdapter._broadcast_kind(
                observation.get('rhs', {}), batch_shape)
        if name == 'broadcast_profile':
            lhs = FeatureRegistryAdapter._broadcast_kind(
                observation.get('lhs', {}), batch_shape)
            rhs = FeatureRegistryAdapter._broadcast_kind(
                observation.get('rhs', {}), batch_shape)
            return f'{lhs} / {rhs}'
        if name == 'lhs_layout':
            return FeatureRegistryAdapter._layout_kind(
                observation.get('lhs', {}))
        if name == 'rhs_layout':
            return FeatureRegistryAdapter._layout_kind(
                observation.get('rhs', {}))
        if name == 'layout_profile':
            lhs = FeatureRegistryAdapter._layout_kind(
                observation.get('lhs', {}))
            rhs = FeatureRegistryAdapter._layout_kind(
                observation.get('rhs', {}))
            return f'{lhs} / {rhs}'
        if name == 'mode':
            return _mode_description(
                context.get('mode'), context.get('target_duration', False))
        return context.get(name)

    @staticmethod
    def _broadcast_kind(operand, batch_shape):
        shape = operand.get('shape') or []
        strides = operand.get('strides') or []
        if len(shape) != len(strides) or batch_shape is None:
            return 'unknown'
        operand_shape = [] if len(shape) == 1 else shape[:-2]
        operand_strides = [] if len(shape) == 1 else strides[:-2]
        missing = len(batch_shape) - len(operand_shape)
        if missing < 0:
            return 'unknown'
        aligned_shape = [1] * missing + list(operand_shape)
        aligned_strides = [0] * missing + list(operand_strides)
        reused = any(
            (source == 1 and target > 1)
            or (source > 1 and stride == 0)
            for source, target, stride in zip(
                aligned_shape, batch_shape, aligned_strides))
        return 'yes' if reused else 'no'

    @staticmethod
    def _layout_kind(operand):
        shape = operand.get('shape') or []
        strides = operand.get('strides') or []
        if len(shape) != len(strides):
            return 'unknown'
        core_rank = min(2, len(shape))
        core_shape = shape[-core_rank:]
        core_strides = strides[-core_rank:]
        if any(size > 1 and stride == 0
               for size, stride in zip(core_shape, core_strides)):
            return 'zero-stride'
        if any(stride < 0 for stride in core_strides):
            return 'negative-stride'
        if core_rank == 1:
            if core_strides[0] == 1:
                return 'row-major'
            return 'strided'
        rows, columns = core_shape
        row_stride, column_stride = core_strides
        if column_stride == 1 and row_stride >= columns:
            if row_stride == max(1, columns):
                return 'row-major'
            return 'row-major-padded'
        if row_stride == 1 and column_stride >= rows:
            if column_stride == max(1, rows):
                return 'column-major'
            return 'column-major-padded'
        if list(core_strides) == _process.contiguous_strides(core_shape):
            return 'row-major'
        return 'strided'


def _duration_run_summary(document):
    run = document.get('duration_run')
    if not isinstance(run, dict):
        return ''
    requested = run.get('requested', {})
    schedule = run.get('schedule', {})
    parts = []
    requested_seconds = requested.get('seconds') \
        if isinstance(requested, dict) else None
    if isinstance(requested_seconds, (int, float)):
        parts.append(
            f'target {_collection._format_duration(requested_seconds)}')
    actual_ns = run.get('actual_elapsed_ns')
    if isinstance(actual_ns, (int, float)):
        parts.append(
            f'active {_collection._format_duration(actual_ns / 1e9)}')
    if isinstance(schedule, dict):
        predicted = schedule.get('predicted')
        central = predicted.get('central_seconds') \
            if isinstance(predicted, dict) else None
        if isinstance(central, (int, float)):
            parts.append(
                f'predicted {_collection._format_duration(central)}')
        panels = schedule.get('panels')
        repetitions = schedule.get('repetitions')
        shards = schedule.get('shard_count')
        if all(isinstance(value, int)
               for value in (panels, repetitions, shards)):
            shard_suffix = '' if shards == 1 else 's'
            parts.append(
                f'{panels} measurement rounds, {repetitions} calls per '
                f'result, {shards} saved chunk{shard_suffix}')
    return '; '.join(parts)


def _mode_signature(mode):
    """Return the exact sampling identity used by one request."""
    if not mode:
        return None
    try:
        spec = matmul_benchmark.schema.ModeSpec.from_dict(mode)
    except (TypeError, ValueError):
        return None
    return spec.name, spec.warmups, spec.repetitions, spec.panels


def _mode_description(signature, target_duration=False):
    if signature is None:
        return None
    name, warmups, repetitions, rounds = signature
    try:
        preset = matmul_benchmark.schema.ModeSpec.preset(name)
    except ValueError:
        label = str(name).title()
    else:
        preset_signature = (
            preset.name, preset.warmups, preset.repetitions, preset.panels)
        if target_duration:
            label = f'{name.title()} calibrated'
        else:
            label = name.title() \
                if signature == preset_signature else 'Custom'
    return (
        f'{label}: {warmups} setup runs, {repetitions} calls per result, '
        f'{rounds} measurement rounds')


def _duration_mode_signature(document):
    run = document.get('duration_run')
    if not isinstance(run, dict):
        return None
    template = run.get('template_plan', {})
    schedule = run.get('schedule', {})
    mode = template.get('mode', {}) if isinstance(template, dict) else {}
    signature = _mode_signature(mode)
    repetitions = schedule.get('repetitions') \
        if isinstance(schedule, dict) else None
    rounds = schedule.get('panels') if isinstance(schedule, dict) else None
    if signature is None or not all(
            isinstance(value, int) for value in (repetitions, rounds)):
        return None
    return signature[0], signature[1], repetitions, rounds


class _FeatureConstraint(QtWidgets.QWidget):
    """Select one categorical value or one numeric slice coordinate."""

    changed = QtCore.Signal()

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self._values = []
        self._enabled = QtWidgets.QCheckBox(label)
        self._feature = QtWidgets.QComboBox()
        self._categorical = QtWidgets.QComboBox()
        self._numeric = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._numeric.setRange(0, 0)
        self._numeric_label = QtWidgets.QLabel('')

        numeric = QtWidgets.QWidget()
        numeric_layout = QtWidgets.QHBoxLayout(numeric)
        numeric_layout.setContentsMargins(0, 0, 0, 0)
        numeric_layout.addWidget(self._numeric, 1)
        numeric_layout.addWidget(self._numeric_label)

        self._values_stack = QtWidgets.QStackedWidget()
        self._values_stack.addWidget(self._categorical)
        self._values_stack.addWidget(numeric)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._enabled)
        layout.addWidget(self._feature, 1)
        layout.addWidget(self._values_stack, 2)

        self._feature.setEnabled(False)
        self._values_stack.setEnabled(False)
        self._enabled.toggled.connect(self._feature.setEnabled)
        self._enabled.toggled.connect(self._values_stack.setEnabled)
        self._enabled.toggled.connect(self.changed)
        self._feature.currentTextChanged.connect(self.changed)
        self._categorical.currentTextChanged.connect(self.changed)
        self._numeric.valueChanged.connect(self._update_numeric_label)
        self._numeric.valueChanged.connect(self.changed)

    def set_features(self, names):
        current = self._feature.currentText()
        self._feature.blockSignals(True)
        self._feature.clear()
        self._feature.addItems(names)
        index = self._feature.findText(current)
        self._feature.setCurrentIndex(max(0, index))
        self._feature.blockSignals(False)

    def update_values(self, values):
        clean = [value for value in values if value is not None]
        numeric = clean and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) for value in clean)
        self._values = sorted(set(clean), key=float if numeric else str)
        if numeric:
            self._numeric.setRange(0, max(0, len(self._values) - 1))
            self._values_stack.setCurrentIndex(1)
            self._update_numeric_label(self._numeric.value())
        else:
            current = self._categorical.currentText()
            self._categorical.blockSignals(True)
            self._categorical.clear()
            self._categorical.addItems(str(value) for value in self._values)
            index = self._categorical.findText(current)
            self._categorical.setCurrentIndex(max(0, index))
            self._categorical.blockSignals(False)
            self._values_stack.setCurrentIndex(0)

    def feature_name(self):
        return self._feature.currentText() if self._enabled.isChecked() else ''

    def configured_feature_name(self):
        return self._feature.currentText()

    def selected_value(self):
        if not self._enabled.isChecked() or not self._values:
            return None
        if self._values_stack.currentIndex() == 1:
            return self._values[self._numeric.value()]
        index = self._categorical.currentIndex()
        return self._values[index] if index >= 0 else None

    def accepts(self, value):
        if not self._enabled.isChecked():
            return True
        selected = self.selected_value()
        if (isinstance(selected, (int, float))
                and isinstance(value, (int, float))):
            return math.isclose(float(value), float(selected), rel_tol=1e-12)
        return value == selected

    def _update_numeric_label(self, index):
        if 0 <= index < len(self._values):
            self._numeric_label.setText(f'{self._values[index]:.6g}')
        else:
            self._numeric_label.clear()


class AtlasWidget(QtWidgets.QWidget):
    """Load offline artifacts and project observations onto feature axes."""

    collection_running_changed = QtCore.Signal(bool)

    def __init__(self, registry=None, artifact_loader=None,
                 process_factory=None, starter_profile=None, parent=None):
        super().__init__(parent)
        self._registry = FeatureRegistryAdapter(registry)
        self._artifact_loader = (
            artifact_loader or matmul_benchmark.artifact.load_artifact)
        self._runner = _process.BenchmarkProcess(process_factory, self)
        self._live_status = _process.RunStatus(self)
        self._artifacts = []
        self._observations = []
        self._peer_running = False
        self._last_collection = None
        self._last_collection_path = None
        self._last_collection_saved = True
        self._active_output_path = None
        self._active_plan_sha256 = None
        self._latest_checkpoint_path = None
        self._latest_checkpoint_plan_sha256 = None
        profile = starter_profile or {}
        self._starter_options = _collection.default_options(
            profile.get('dtype', 'float32'), profile.get('threads', 1))
        self._build_ui()
        self._connect_runner()
        self._refresh_starter_estimate()

    @property
    def running(self):
        return self._runner.running

    def _build_ui(self):
        self._load = QtWidgets.QPushButton('Load / merge artifacts...')
        self._save_collection = QtWidgets.QPushButton('Save collection...')
        self._save_collection.setEnabled(False)
        self._clear = QtWidgets.QPushButton('Clear')
        self._add_feature = QtWidgets.QPushButton('Add feature expression...')
        self._artifact_status = QtWidgets.QLabel('No artifacts loaded')
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self._load)
        top.addWidget(self._save_collection)
        top.addWidget(self._clear)
        top.addWidget(self._add_feature)
        top.addWidget(self._artifact_status, 1)

        self._collect = QtWidgets.QPushButton('Collect starter Atlas')
        self._configure_collection = QtWidgets.QPushButton('Configure...')
        self._collection_progress = QtWidgets.QProgressBar()
        self._collection_progress.setRange(0, 1)
        self._collection_progress.setValue(0)
        self._collection_status = QtWidgets.QLabel('Ready')
        collect = QtWidgets.QHBoxLayout()
        collect.addWidget(self._collect)
        collect.addWidget(self._configure_collection)
        collect.addWidget(self._collection_progress, 1)
        collect.addWidget(self._collection_status)
        self._collection_estimate = QtWidgets.QLabel()
        self._collection_estimate.setWordWrap(True)

        self._x_axis = QtWidgets.QComboBox()
        self._y_axis = QtWidgets.QComboBox()
        self._z_axis = QtWidgets.QComboBox()
        axes = QtWidgets.QHBoxLayout()
        axes.addWidget(QtWidgets.QLabel('X axis'))
        axes.addWidget(self._x_axis, 1)
        axes.addWidget(QtWidgets.QLabel('Y axis'))
        axes.addWidget(self._y_axis, 1)
        axes.addWidget(QtWidgets.QLabel('Z axis'))
        axes.addWidget(self._z_axis, 1)

        self._slice = _FeatureConstraint('Slice')
        self._filter = _FeatureConstraint('Filter')
        self._cloud_canvas = _cloud.CloudCanvas()
        self._cloud_controls = self._build_cloud_controls()
        self._point_status = QtWidgets.QLabel('0 observations shown')

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(collect)
        layout.addWidget(self._collection_estimate)
        layout.addLayout(axes)
        layout.addWidget(self._slice)
        layout.addWidget(self._filter)
        layout.addWidget(self._cloud_controls)
        layout.addWidget(self._cloud_canvas, 1)
        layout.addWidget(self._point_status)

        self._load.clicked.connect(self.choose_artifacts)
        self._save_collection.clicked.connect(self.save_collection)
        self._clear.clicked.connect(self.clear_artifacts)
        self._add_feature.clicked.connect(self.add_feature_expression)
        self._collect.clicked.connect(self.start_starter_collection)
        self._configure_collection.clicked.connect(
            self.configure_starter_collection)
        self._x_axis.currentTextChanged.connect(self.render)
        self._y_axis.currentTextChanged.connect(self.render)
        self._z_axis.currentTextChanged.connect(self.render)
        self._slice.changed.connect(self._constraint_changed)
        self._filter.changed.connect(self._constraint_changed)

    def _connect_runner(self):
        self._live_status.changed.connect(self._collection_status.setText)
        self._runner.activity.connect(self._live_status.show_activity)
        self._runner.progress.connect(self._show_collection_progress)
        self._runner.checkpoint.connect(self._show_collection_checkpoint)
        self._runner.completed.connect(self._collection_completed)
        self._runner.failed.connect(self._collection_failed)
        self._runner.cancelled.connect(self._collection_cancelled)
        self._runner.running_changed.connect(self._set_collection_running)

    def set_starter_profile(self, dtype, threads):
        """Inherit production dtype and thread settings from the inspector."""
        if dtype not in matmul_benchmark.schema.SUPPORTED_DTYPES:
            raise ValueError(f'Unsupported starter dtype: {dtype!r}')
        if isinstance(threads, bool) or not isinstance(threads, int) \
                or threads < 1:
            raise ValueError('Starter threads must be a positive integer')
        self._starter_options['dtype'] = dtype
        self._starter_options['threads'] = threads
        if not self._runner.running:
            self._refresh_starter_estimate()

    def set_peer_running(self, running):
        """Disable collection while Route Inspector owns the benchmark slot."""
        self._peer_running = running
        self._update_collection_controls()

    @QtCore.Slot()
    def configure_starter_collection(self):
        dialog = _collection.StarterPlanDialog(
            self._starter_options, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._starter_options = dialog.options()
        self._refresh_starter_estimate()

    @QtCore.Slot()
    def start_starter_collection(self):
        if self._peer_running:
            self._collection_failed('Route Inspector is benchmarking')
            return
        if not self._confirm_unsaved('start another collection'):
            return
        try:
            plan = _collection.make_plan(self._starter_options)
            estimate = matmul_benchmark.collection.estimate_plan(plan)
            matmul_benchmark.collection.validate_execution_plan(plan)
            self._live_status.begin()
            self._runner.start(plan.to_dict())
        except (MemoryError, RuntimeError, TypeError, ValueError) as exc:
            self._collection_failed(str(exc))
            return
        plan_sha256 = plan.sha256()
        resume_path = self._latest_checkpoint_path
        can_resume = (
            plan.target_duration is not None
            and plan.output_path == self._active_output_path
            and plan_sha256 == self._latest_checkpoint_plan_sha256)
        if not can_resume:
            resume_path = None
        if not resume_path:
            self._latest_checkpoint_path = None
            self._latest_checkpoint_plan_sha256 = None
        self._active_output_path = plan.output_path
        self._active_plan_sha256 = plan_sha256
        if plan.target_duration is None:
            self._collection_progress.setRange(
                0, estimate.panel_count * estimate.cell_count)
            status = 'Starting fixed schedule...'
        else:
            self._collection_progress.setRange(0, 0)
            if resume_path:
                status = f'Resuming from checkpoint: {resume_path}'
            else:
                status = 'Starting target-duration calibration...'
        self._collection_progress.setValue(0)
        self._live_status.show_summary(status)

    @QtCore.Slot()
    def stop(self):
        """Show the interrupted operation and stop its worker."""
        if not self._runner.running:
            return
        self._live_status.stopping()
        self._runner.cancel()

    def shutdown(self):
        """Stop an Atlas collector before closing its owning window."""
        self._runner.shutdown()

    def confirm_close(self):
        """Return whether the explorer may close without losing a result."""
        return self._confirm_unsaved('close the benchmark explorer')

    @QtCore.Slot()
    def save_collection(self):
        """Atomically export the most recently collected Atlas document."""
        if self._last_collection is None:
            return False
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Atlas collection',
            self._last_collection_path or '', 'JSON files (*.json)')
        if not path:
            return False
        try:
            saved_path = matmul_benchmark.artifact.write_artifact(
                self._last_collection, path)
        except (OSError, TypeError, ValueError) as exc:
            self._collection_failed(str(exc))
            return False
        self._last_collection_path = str(saved_path)
        self._last_collection_saved = True
        self._collection_status.setText(f'Saved: {saved_path}')
        return True

    def _confirm_unsaved(self, action):
        if self._last_collection is None or self._last_collection_saved:
            return True
        buttons = (
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel)
        choice = QtWidgets.QMessageBox.question(
            self, 'Unsaved Atlas collection',
            f'Save the latest collection before you {action}?',
            buttons, QtWidgets.QMessageBox.StandardButton.Save)
        if choice == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save_collection()
        if choice == QtWidgets.QMessageBox.StandardButton.Discard:
            self._last_collection_saved = True
            return True
        return False

    @QtCore.Slot(int, int, str)
    def _show_collection_progress(self, completed, total, message):
        if total:
            self._collection_progress.setRange(0, total)
            self._collection_progress.setValue(min(completed, total))
            fallback = f'{completed}/{total}'
        else:
            self._collection_progress.setRange(0, 0)
            fallback = 'Calibrating...'
        text = _collection.humanize_sampling_text(message)
        self._live_status.show_summary(text or fallback)

    @QtCore.Slot(str, int, int, int, int)
    def _show_collection_checkpoint(
            self, path, completed_shards, total_shards,
            completed_panels, total_panels):
        self._latest_checkpoint_path = path
        self._latest_checkpoint_plan_sha256 = self._active_plan_sha256
        if not self._runner.cancelling:
            self._live_status.show_summary(
                f'Checkpoint {completed_shards}/{total_shards} saved '
                f'after measurement round '
                f'{completed_panels}/{total_panels}: {path}')

    @QtCore.Slot(object)
    def _collection_completed(self, document):
        try:
            self.add_artifacts([document])
            self._show_starter_projection()
        except (KeyError, TypeError, ValueError) as exc:
            self._collection_failed(str(exc))
            return
        self._last_collection = document
        self._last_collection_path = self._active_output_path
        self._last_collection_saved = bool(self._active_output_path)
        self._save_collection.setEnabled(True)
        aggregate = document.get('aggregate_observations')
        observations = (
            aggregate if isinstance(aggregate, list)
            else _process._artifact_observations(document))
        count = len(observations)
        self._collection_progress.setRange(0, 1)
        self._collection_progress.setValue(1)
        status = f'Complete: {count} cells'
        duration_summary = _duration_run_summary(document)
        if duration_summary:
            status += f'; {duration_summary}'
        if self._last_collection_saved:
            status += f'; saved to {self._last_collection_path}'
        else:
            status += '; unsaved, use Save collection'
        self._live_status.complete(status)
        self._latest_checkpoint_path = None
        self._latest_checkpoint_plan_sha256 = None

    @QtCore.Slot(str)
    def _collection_failed(self, message):
        self._collection_progress.setRange(0, 1)
        self._collection_progress.setValue(0)
        self._live_status.fail(message + self._checkpoint_suffix())

    @QtCore.Slot()
    def _collection_cancelled(self):
        self._collection_progress.setRange(0, 1)
        self._collection_progress.setValue(0)
        self._live_status.cancel(self._checkpoint_suffix())

    def _checkpoint_suffix(self):
        if not self._latest_checkpoint_path:
            return ''
        return (
            f'; rerun the same plan to resume from '
            f'{self._latest_checkpoint_path}')

    @QtCore.Slot(bool)
    def _set_collection_running(self, running):
        self._update_collection_controls()
        if not running:
            self._refresh_starter_estimate()
        self.collection_running_changed.emit(running)

    def _update_collection_controls(self):
        enabled = not self._runner.running and not self._peer_running
        self._collect.setEnabled(enabled)
        self._configure_collection.setEnabled(enabled)

    def _refresh_starter_estimate(self):
        try:
            plan = _collection.make_plan(self._starter_options)
            estimate = matmul_benchmark.collection.estimate_plan(plan)
            matmul_benchmark.collection.validate_plan_resources(
                plan, estimate)
        except (MemoryError, TypeError, ValueError) as exc:
            self._collection_estimate.setText(
                f'Starter plan is invalid: {exc}')
            self._collect.setEnabled(False)
            return
        self._collection_estimate.setText(
            f'Starter plan: '
            f'{_collection.plan_estimate_text(estimate, plan)}')
        self._update_collection_controls()

    def _show_starter_projection(self):
        self._x_axis.setCurrentText('M')
        self._y_axis.setCurrentText('N')
        self._z_axis.setCurrentText('K')
        self._slice._enabled.setChecked(False)
        self.render()

    def _build_cloud_controls(self):
        controls = QtWidgets.QWidget()
        self._projection = QtWidgets.QComboBox()
        self._projection.addItems(('Orthographic', 'Perspective'))
        self._point_size = QtWidgets.QSpinBox()
        self._point_size.setRange(4, 36)
        self._point_size.setValue(12)
        self._opacity = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal)
        self._opacity.setRange(10, 100)
        self._opacity.setValue(90)
        self._opacity.setFixedWidth(100)
        reset = QtWidgets.QPushButton('Reset')
        top = QtWidgets.QPushButton('Top')
        front = QtWidgets.QPushButton('Front')
        side = QtWidgets.QPushButton('Side')

        layout = QtWidgets.QHBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel('Projection'))
        layout.addWidget(self._projection)
        layout.addWidget(QtWidgets.QLabel('Point size'))
        layout.addWidget(self._point_size)
        layout.addWidget(QtWidgets.QLabel('Opacity'))
        layout.addWidget(self._opacity)
        layout.addStretch(1)
        layout.addWidget(reset)
        layout.addWidget(top)
        layout.addWidget(front)
        layout.addWidget(side)

        self._projection.currentTextChanged.connect(
            self._projection_changed)
        self._point_size.valueChanged.connect(
            self._cloud_canvas.set_point_size)
        self._opacity.valueChanged.connect(self._opacity_changed)
        reset.clicked.connect(self._cloud_canvas.reset_view)
        top.clicked.connect(
            lambda _checked=False: self._cloud_canvas.set_view('top'))
        front.clicked.connect(
            lambda _checked=False: self._cloud_canvas.set_view('front'))
        side.clicked.connect(
            lambda _checked=False: self._cloud_canvas.set_view('side'))
        return controls

    @QtCore.Slot(str)
    def _projection_changed(self, name):
        self._cloud_canvas.set_perspective(name == 'Perspective')

    @QtCore.Slot(int)
    def _opacity_changed(self, value):
        self._cloud_canvas.set_opacity(value / 100.0)

    @QtCore.Slot()
    def choose_artifacts(self):
        paths, _selected_filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'Load matmul benchmark artifacts', '',
            'JSON files (*.json);;All files (*)')
        if not paths:
            return
        try:
            self.load_artifacts(paths)
        except (OSError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self, 'Could not load artifacts', str(exc))

    def load_artifacts(self, paths):
        """Load and merge observations from every selected artifact path."""
        artifacts = [self._artifact_loader(path) for path in paths]
        self.add_artifacts(artifacts)

    def add_artifacts(self, artifacts):
        """Merge in-memory artifacts without starting a benchmark worker."""
        for artifact in artifacts:
            self._artifacts.append(artifact)
            for observation, context in self._contextual_observations(
                    artifact):
                contextual = dict(observation)
                contextual['_artifact_context'] = context
                self._observations.append(contextual)
        self._artifact_status.setText(
            f'{len(self._artifacts)} artifacts, '
            f'{len(self._observations)} observations')
        self._refresh_features()

    @classmethod
    def _contextual_observations(cls, artifact):
        aggregate = artifact.get('aggregate_observations')
        using_aggregate = isinstance(aggregate, list)
        observations = (aggregate if using_aggregate
                        else _process._artifact_observations(artifact))
        if artifact.get('schema_kind') \
                != matmul_benchmark.schema.COLLECTION_KIND:
            context = cls._artifact_context(artifact)
            for observation in observations:
                yield observation, context
            return

        sources = {
            source['source_id']: source for source in artifact['sources']
        }
        aggregate_mode = _duration_mode_signature(artifact) \
            if using_aggregate else None
        for wrapper in observations:
            if using_aggregate:
                source_ids = wrapper.get('source_ids')
                if not isinstance(source_ids, list) or not source_ids:
                    raise ValueError(
                        'Aggregate observation has no source IDs')
                source_id = source_ids[0]
            else:
                source_id = wrapper.get('source_id')
            if source_id not in sources:
                raise ValueError(
                    'Collection observation has unknown source: '
                    f'{source_id!r}')
            observation = wrapper.get('observation')
            if not isinstance(observation, dict):
                raise ValueError(
                    'Collection observation payload must be an object')
            context = cls._artifact_context(sources[source_id])
            if aggregate_mode is not None:
                context['mode'] = aggregate_mode
                context['target_duration'] = True
            yield observation, context

    @staticmethod
    def _artifact_context(artifact):
        request = artifact.get('request', {})
        metadata = artifact.get('metadata', {})
        machine = metadata.get('machine', {})
        mode = request.get('mode', {})
        return {
            'architecture': machine.get('machine'),
            'machine': machine.get('node'),
            'mode': _mode_signature(mode),
            'system': machine.get('system'),
            'threads': request.get('threads'),
        }

    @QtCore.Slot()
    def clear_artifacts(self):
        self._artifacts.clear()
        self._observations.clear()
        self._artifact_status.setText('No artifacts loaded')
        self._refresh_features()

    @QtCore.Slot()
    def add_feature_expression(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Add feature expression')
        name = QtWidgets.QLineEdit()
        expression = QtWidgets.QLineEdit()
        expression.setPlaceholderText('log2(M * K * N)')
        form = QtWidgets.QFormLayout()
        form.addRow('Feature name', name)
        form.addRow('Expression', expression)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addLayout(form)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        try:
            self._registry.add_expression(
                name.text().strip(), expression.text().strip())
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid feature', str(exc))
            return
        self._refresh_features(preferred=name.text().strip())

    def _refresh_features(self, preferred=''):
        names = sorted(self._registry.names())
        constraint_names = self._registry.constraint_names()
        current_x = preferred or self._x_axis.currentText()
        current_y = self._y_axis.currentText()
        current_z = self._z_axis.currentText()
        for combo, current in ((self._x_axis, current_x),
                               (self._y_axis, current_y),
                               (self._z_axis, current_z)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            index = combo.findText(current)
            combo.setCurrentIndex(max(0, index))
            combo.blockSignals(False)
        if len(names) > 1 and not current_y:
            self._y_axis.setCurrentIndex(1)
        if len(names) > 2 and not current_z:
            self._z_axis.setCurrentIndex(2)
        self._slice.set_features(constraint_names)
        self._filter.set_features(constraint_names)
        self._update_constraint_values()
        self.render()

    @QtCore.Slot()
    def _constraint_changed(self):
        self._update_constraint_values()
        self.render()

    def _update_constraint_values(self):
        for control in (self._slice, self._filter):
            name = control.configured_feature_name()
            values = [self._feature_value(name, observation)
                      for observation in self._observations]
            control.update_values(values)

    def _feature_value(self, name, observation):
        if not name:
            return None
        try:
            return self._registry.evaluate(name, observation)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None

    @QtCore.Slot()
    def render(self):
        """Render the selected projection using existing observations only."""
        x_name = self._x_axis.currentText()
        y_name = self._y_axis.currentText()
        z_name = self._z_axis.currentText()
        ignored_slice = self._slice.feature_name() == z_name
        self._slice.setToolTip(
            'This slice is ignored because the same feature is the Z axis.'
            if ignored_slice else '')
        points = []
        for observation in self._observations:
            if not self._accepts_constraints(observation, z_name):
                continue
            x_value = self._feature_value(x_name, observation)
            y_value = self._feature_value(y_name, observation)
            z_value = self._feature_value(z_name, observation)
            if not self._is_finite_number(x_value):
                continue
            if not self._is_finite_number(y_value):
                continue
            if not self._is_finite_number(z_value):
                continue
            points.append(self._make_point(
                observation, x_value, y_value, z_value))
        points = self._group_points(points)
        self._cloud_canvas.set_points(points, x_name, y_name, z_name)
        shown_count = sum(point['sample_count'] for point in points)
        self._point_status.setText(
            f'{shown_count} of '
            f'{len(self._observations)} observations shown in '
            f'{len(points)} points')

    def _accepts_constraints(self, observation, z_name=''):
        for control in (self._slice, self._filter):
            name = control.feature_name()
            if control is self._slice and name and name == z_name:
                continue
            if name and not control.accepts(
                    self._feature_value(name, observation)):
                return False
        return True

    @staticmethod
    def _is_finite_number(value):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        try:
            return math.isfinite(float(value))
        except OverflowError:
            return False

    @staticmethod
    def _make_point(observation, x_value, y_value, z_value=None):
        winner = _process._winner_name(observation)
        runner_up = _process._nested_value(
            observation, 'runner_up', 'summary.runner_up', default='')
        margin = _process._nested_value(
            observation, 'winner_margin', 'summary.winner_margin',
            default=0.0)
        reported_noise = _process._nested_value(
            observation, 'noise', 'summary.noise', 'summary.rmad',
            default=None)
        candidates = {
            _process._candidate_name(candidate): candidate
            for candidate in _process._observation_candidates(observation)
        }
        noise_values = [reported_noise]
        for route_name in (winner, runner_up):
            if route_name:
                noise_values.append(_process._summary_value(
                    candidates.get(str(route_name), {}), 'noise'))
        noise = max(
            (float(value) for value in noise_values if value is not None),
            default=0.0)
        invalid = bool(_process._nested_value(
            observation, 'invalid', 'quality.invalid', default=False))
        ambiguous = bool(_process._nested_value(
            observation, 'ambiguous', 'quality.ambiguous', default=False))
        if not winner:
            invalid = True
        if margin is not None and float(margin) <= max(0.03, 2 * noise):
            ambiguous = True
        layout = AtlasWidget._layout_summary(observation)
        packing = AtlasWidget._packing_summary(candidates.get(winner, {}))
        return {
            'x': x_value,
            'y': y_value,
            'z': z_value,
            'route': winner,
            'observation_id': observation.get('id'),
            'margin': float(margin or 0.0),
            'noise': noise,
            'invalid': invalid,
            'ambiguous': ambiguous,
            'layout': layout,
            'packing': packing,
        }

    @staticmethod
    def _layout_summary(observation):
        explicit = observation.get('layout_profile')
        if explicit is None:
            explicit = observation.get('layout')
        lhs_operand = observation.get('lhs')
        rhs_operand = observation.get('rhs')
        lhs = (FeatureRegistryAdapter._layout_kind(lhs_operand)
               if lhs_operand else 'unknown')
        rhs = (FeatureRegistryAdapter._layout_kind(rhs_operand)
               if rhs_operand else 'unknown')
        if lhs != 'unknown' or rhs != 'unknown':
            return f'{lhs} / {rhs}'
        return str(explicit or 'unknown')

    @staticmethod
    def _packing_summary(candidate):
        summaries = []
        for kind in ('eager', 'scratch'):
            value = _process._packing_text(candidate, kind)
            if value and value != 'none':
                summaries.append(f'{kind}: {value}')
        return '; '.join(summaries) or 'none'

    @staticmethod
    def _group_points(points):
        groups = {}
        for point in points:
            key = point['x'], point['y'], point['z']
            groups.setdefault(key, []).append(point)

        grouped = []
        for (x_value, y_value, z_value), samples in groups.items():
            route_counts = {}
            for sample in samples:
                route = sample['route']
                if route and not sample['invalid']:
                    route_counts[route] = route_counts.get(route, 0) + 1
            routes = tuple(sorted(route_counts))
            invalid_count = sum(sample['invalid'] for sample in samples)
            valid_samples = [sample for sample in samples
                             if not sample['invalid']]
            grouped.append({
                'x': x_value,
                'y': y_value,
                'z': z_value,
                'route': routes[0] if len(routes) == 1 else '',
                'routes': routes,
                'winner_counts': route_counts,
                'observation_ids': tuple(
                    sample['observation_id'] for sample in samples),
                'sample_count': len(samples),
                'hidden_sample_count': len(samples) - 1,
                'invalid_count': invalid_count,
                'ambiguous_count': sum(
                    sample['ambiguous'] for sample in samples),
                'margin': min(
                    (sample['margin'] for sample in valid_samples),
                    default=0.0),
                'noise': max(
                    (sample['noise'] for sample in samples), default=0.0),
                'layouts': tuple(sorted(set(
                    sample['layout'] for sample in samples))),
                'packing': tuple(sorted(set(
                    sample['packing'] for sample in samples))),
                'invalid': not routes,
                'has_invalid': invalid_count > 0,
                'conflicting': len(routes) > 1,
                'ambiguous': (
                    len(samples) > 1
                    or any(sample['ambiguous'] for sample in samples)),
            })
        return grouped


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
