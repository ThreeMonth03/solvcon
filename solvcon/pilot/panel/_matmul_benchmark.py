# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Pilot window for interactive matmul route inspection."""

import json

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ... import matmul_benchmark
from ..base import _gui_common
from . import _matmul_benchmark_atlas as _atlas
from . import _matmul_benchmark_chart as _chart
from . import _matmul_benchmark_collection as _collection
from . import _matmul_benchmark_process as _process

__all__ = [
    'MatmulBenchmark',
    'MatmulBenchmarkWindow',
    'RouteInspectorWidget',
]

_DISPATCH_CHOICES = (
    ('generic', 'Generic',
     'Direct native loop. It is available for every valid input.'),
    ('blas_dot', 'BLAS DOT',
     'BLAS vector dot product for A vector @ B vector.'),
    ('blas_gevm', 'BLAS GEVM',
     'BLAS vector-matrix product for A vector @ B matrix.'),
    ('blas_gemv', 'BLAS GEMV',
     'BLAS matrix-vector product for A matrix @ B vector.'),
    ('blas_gemm', 'BLAS GEMM',
     'BLAS matrix product for A matrix @ B matrix, including batches.'),
    ('winograd', 'Winograd',
     'Winograd matrix product for unbatched, positive, even M, K, and N.'),
)
_DISPATCH_NAMES = tuple(choice[0] for choice in _DISPATCH_CHOICES)
_DISPATCH_HELP = {
    name: helper for name, _label, helper in _DISPATCH_CHOICES
}


class RouteInspectorWidget(QtWidgets.QWidget):
    """Collect one shape and compare every requested execution route."""

    artifact_ready = QtCore.Signal(object)
    benchmark_running_changed = QtCore.Signal(bool)
    profile_changed = QtCore.Signal(str, int)

    HEADERS = (
        'Dispatch',
        'Eligible',
        'Auto',
        'Winner',
        'Eager pack',
        'Scratch pack',
        'Correctness',
        'Native median',
        'Native p95',
        'Native MAD/median',
        'Python E2E median',
        'vs NumPy',
    )

    def __init__(self, process_factory=None, parent=None):
        super().__init__(parent)
        self._runner = _process.BenchmarkProcess(process_factory, self)
        self._live_status = _process.RunStatus(self)
        self._artifact = None
        self._peer_running = False
        self._dispatch_preferences = {name: True for name in _DISPATCH_NAMES}
        self._available_dispatch_cache = {}
        self._runnable_dispatches = frozenset()
        self._dispatch_summary = ''
        self._build_ui()
        self._connect_runner()

    @property
    def running(self):
        return self._runner.running

    def _build_ui(self):
        self._dtype = QtWidgets.QComboBox()
        self._dtype.addItems(('float32', 'float64', 'complex64', 'complex128'))
        self._lhs_shape = QtWidgets.QLineEdit('64, 64')
        self._rhs_shape = QtWidgets.QLineEdit('64, 64')
        self._lhs_strides = QtWidgets.QLineEdit('')
        self._rhs_strides = QtWidgets.QLineEdit('')
        self._lhs_strides.setPlaceholderText('blank = contiguous')
        self._rhs_strides.setPlaceholderText('blank = contiguous')
        self._threads = QtWidgets.QSpinBox()
        thread_limit = _collection.recommended_max_threads()
        self._threads.setRange(1, thread_limit)
        self._threads.setValue(min(4, thread_limit))
        self._threads.setToolTip(_collection.THREAD_LIMIT_TOOLTIP)
        self._sampling = _collection.SamplingControls()
        self._dispatches = _collection._CheckGrid(
            _DISPATCH_CHOICES, selected=_DISPATCH_NAMES, columns=3)
        self._dispatch_helper = QtWidgets.QLabel(
            'Checked dispatches are benchmarked in addition to Auto and '
            'NumPy. Unavailable choices are disabled for the current input.')
        self._dispatch_helper.setWordWrap(True)
        dispatch_control = QtWidgets.QWidget()
        dispatch_layout = QtWidgets.QVBoxLayout(dispatch_control)
        dispatch_layout.setContentsMargins(0, 0, 0, 0)
        dispatch_layout.addWidget(self._dispatches)
        dispatch_layout.addWidget(self._dispatch_helper)

        form = QtWidgets.QFormLayout()
        form.addRow('Data type', self._dtype)
        form.addRow('A shape', self._lhs_shape)
        form.addRow('A element strides', self._lhs_strides)
        form.addRow('B shape', self._rhs_shape)
        form.addRow('B element strides', self._rhs_strides)
        form.addRow('BLAS threads', self._threads)
        form.addRow('Measurement quality', self._sampling)
        form.addRow('Dispatches', dispatch_control)

        self._run = QtWidgets.QPushButton('Run benchmark')
        self._save = QtWidgets.QPushButton('Save result...')
        self._save.setEnabled(False)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._run)
        buttons.addWidget(self._save)
        buttons.addStretch(1)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._status = QtWidgets.QLabel('Ready')
        progress = QtWidgets.QHBoxLayout()
        progress.addWidget(self._progress, 1)
        progress.addWidget(self._status)

        self._table = QtWidgets.QTableWidget(0, len(self.HEADERS))
        self._table.setHorizontalHeaderLabels(self.HEADERS)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)

        self._chart = _chart.RouteTimingChart()

        self._details = QtWidgets.QPlainTextEdit()
        self._details.setReadOnly(True)
        self._details.setPlaceholderText(
            'Select a dispatch to inspect its timed samples and summary.')
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self._chart)
        splitter.addWidget(self._table)
        splitter.addWidget(self._details)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addLayout(progress)
        layout.addWidget(splitter, 1)

        self._run.clicked.connect(self.start_benchmark)
        self._save.clicked.connect(self.save_result)
        self._chart.row_selected.connect(self._select_candidate_row)
        self._table.currentCellChanged.connect(self._show_candidate_details)
        self._dtype.currentTextChanged.connect(self._emit_profile)
        self._threads.valueChanged.connect(self._emit_profile)
        self._sampling.changed.connect(self._refresh_dispatches)
        self._dispatches.changed.connect(self._remember_dispatch_preferences)
        for control in (
                self._dtype, self._lhs_shape, self._rhs_shape,
                self._lhs_strides, self._rhs_strides):
            signal = (control.currentTextChanged
                      if isinstance(control, QtWidgets.QComboBox)
                      else control.textChanged)
            signal.connect(self._refresh_dispatches)
        self._refresh_dispatches()

    def _connect_runner(self):
        self._live_status.changed.connect(self._status.setText)
        self._runner.activity.connect(self._live_status.show_activity)
        self._runner.progress.connect(self._show_progress)
        self._runner.completed.connect(self.show_artifact)
        self._runner.completed.connect(self.artifact_ready)
        self._runner.failed.connect(self._show_error)
        self._runner.cancelled.connect(self._show_cancelled)
        self._runner.running_changed.connect(self._set_running)

    def make_request(self):
        """Return a validated worker request from the current controls."""
        dispatches = self._dispatches.selected()
        if not dispatches:
            raise ValueError('Select at least one runnable dispatch')
        request = self._request_data(dispatches)
        parsed = matmul_benchmark.schema.BenchmarkRequest.from_dict(request)
        eligible = self._eligible_dispatches(parsed)
        inapplicable = set(dispatches) - eligible
        if inapplicable:
            names = ', '.join(
                name for name in dispatches if name in inapplicable)
            raise ValueError(
                f'Selected dispatches do not apply to these shapes: {names}')
        matmul_benchmark.collector.validate_request_resources(parsed)
        return request

    def _request_data(self, dispatches):
        lhs_shape = _process.parse_integer_tuple(
            self._lhs_shape.text(), minimum=0)
        rhs_shape = _process.parse_integer_tuple(
            self._rhs_shape.text(), minimum=0)
        lhs_strides = self._parse_strides(self._lhs_strides.text(), lhs_shape)
        rhs_strides = self._parse_strides(self._rhs_strides.text(), rhs_shape)
        return {
            'schema_version': 1,
            'dtype': self._dtype.currentText(),
            'lhs': {'shape': list(lhs_shape), 'strides': lhs_strides},
            'rhs': {'shape': list(rhs_shape), 'strides': rhs_strides},
            'threads': self._threads.value(),
            'mode': self._sampling.mode_spec().to_dict(),
            'routes': list(dispatches),
            'numpy_baseline': True,
            'seed': 0,
        }

    def benchmark_profile(self):
        """Return the production settings shared with starter collections."""
        return {
            'dtype': self._dtype.currentText(),
            'threads': self._threads.value(),
        }

    def set_peer_running(self, running):
        """Disable Route Inspector while Atlas owns the benchmark slot."""
        self._peer_running = running
        self._update_run_enabled()

    @QtCore.Slot()
    def _update_run_enabled(self):
        selected = set(self._dispatches.selected())
        request_error = None
        if self._sampling.is_valid() and selected:
            try:
                request = matmul_benchmark.schema.BenchmarkRequest.from_dict(
                    self._request_data(self._dispatches.selected()))
                matmul_benchmark.collector.validate_request_resources(
                    request)
            except (MemoryError, RuntimeError, ValueError) as exc:
                request_error = str(exc)
        helper = self._dispatch_summary
        if request_error:
            helper += f' Current selection cannot run: {request_error}.'
        self._dispatch_helper.setText(helper)
        self._run.setToolTip(request_error or '')
        self._run.setEnabled(
            self._sampling.is_valid()
            and bool(selected & self._runnable_dispatches)
            and request_error is None
            and not self._peer_running and not self._runner.running)

    @QtCore.Slot()
    def _emit_profile(self):
        profile = self.benchmark_profile()
        self.profile_changed.emit(profile['dtype'], profile['threads'])

    @staticmethod
    def _parse_strides(text, shape):
        if not text.strip():
            return _process.contiguous_strides(shape)
        strides = _process.parse_integer_tuple(text)
        if len(strides) != len(shape):
            raise ValueError('A stride is required for every shape dimension')
        return list(strides)

    def _available_dispatches(self, dtype):
        cached = self._available_dispatch_cache.get(dtype)
        if cached is not None:
            return cached
        shapes = (
            ((2,), (2,)),
            ((2,), (2, 2)),
            ((2, 2), (2,)),
            ((2, 2), (2, 2)),
        )
        dispatches = set()
        engine = matmul_benchmark.engine.SolvconRouteEngine()
        try:
            for lhs_shape, rhs_shape in shapes:
                lhs = np.ones(lhs_shape, dtype=dtype)
                rhs = np.ones(rhs_shape, dtype=dtype)
                case = engine.prepare(lhs, rhs, dtype)
                dispatches.update(route.name for route in case.routes)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            dispatches.clear()
        cached = frozenset(dispatches)
        self._available_dispatch_cache[dtype] = cached
        return cached

    @QtCore.Slot()
    def _remember_dispatch_preferences(self):
        for name, box in self._dispatches.boxes.items():
            if box.isEnabled():
                self._dispatch_preferences[name] = box.isChecked()
        self._update_run_enabled()

    def _eligible_dispatches(self, request):
        shape_dispatches = self._structurally_eligible_dispatches(
            request.lhs.shape, request.rhs.shape)
        build_dispatches = self._available_dispatches(request.dtype)
        return set(shape_dispatches) & set(build_dispatches)

    @QtCore.Slot()
    def _refresh_dispatches(self):
        eligible = set()
        try:
            request = matmul_benchmark.schema.BenchmarkRequest.from_dict(
                self._request_data(()))
            build_dispatches = self._available_dispatches(request.dtype)
            eligible = self._eligible_dispatches(request)
        except (RuntimeError, ValueError) as exc:
            helper = (
                f'Fix the benchmark settings to choose dispatches: {exc}. '
                'Checkbox selections are preserved.')
        else:
            if build_dispatches:
                helper = (
                    f'{len(eligible)} dispatches apply to this input. '
                    'Checked dispatches are benchmarked in addition to '
                    'Auto and NumPy.')
            else:
                helper = (
                    'This build does not expose matmul benchmark '
                    f'dispatches for {request.dtype}.')

        for name, box in self._dispatches.boxes.items():
            available = name in eligible
            blocker = QtCore.QSignalBlocker(box)
            box.setChecked(available and self._dispatch_preferences[name])
            box.setEnabled(available)
            del blocker
            tooltip = _DISPATCH_HELP[name]
            if name not in eligible:
                tooltip += ' Not available for the current input or build.'
            box.setToolTip(tooltip)
        self._runnable_dispatches = frozenset(eligible)
        self._dispatch_summary = helper
        self._update_run_enabled()

    @staticmethod
    def _structurally_eligible_dispatches(lhs_shape, rhs_shape):
        dispatches = ['generic']
        contraction = rhs_shape[-1] if len(rhs_shape) == 1 else rhs_shape[-2]
        lhs_vector = len(lhs_shape) == 1
        rhs_vector = len(rhs_shape) == 1
        dimensions = (
            1 if lhs_vector else lhs_shape[-2],
            contraction,
            1 if rhs_vector else rhs_shape[-1],
        )
        if not all(dimensions):
            return dispatches
        if lhs_vector and rhs_vector:
            dispatches.append('blas_dot')
        elif lhs_vector:
            dispatches.append('blas_gevm')
        elif rhs_vector:
            dispatches.append('blas_gemv')
        else:
            dispatches.append('blas_gemm')
            unbatched = len(lhs_shape) == len(rhs_shape) == 2
            if unbatched and all(value % 2 == 0 for value in dimensions):
                dispatches.append('winograd')
        return dispatches

    @QtCore.Slot()
    def start_benchmark(self):
        if self._peer_running:
            self._show_error('Dispatch Atlas is collecting')
            return
        try:
            request = self.make_request()
            self._live_status.begin()
            self._runner.start(request)
        except (MemoryError, ValueError, RuntimeError) as exc:
            self._show_error(str(exc))
            return
        self._artifact = None
        self._chart.set_candidates([], '', '')
        self._table.setRowCount(0)
        self._details.clear()
        self._save.setEnabled(False)
        self._progress.setRange(0, 0)

    @QtCore.Slot()
    def stop(self):
        """Show the interrupted operation and stop its worker."""
        if not self._runner.running:
            return
        self._live_status.stopping()
        self._runner.cancel()

    def shutdown(self):
        """Stop the worker before closing the inspector."""
        self._runner.shutdown()

    @QtCore.Slot(int, int, str)
    def _show_progress(self, completed, total, message):
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(min(completed, total))
        else:
            self._progress.setRange(0, 0)
        text = _collection.humanize_sampling_text(message)
        self._live_status.show_summary(text or f'{completed}/{total}')

    @QtCore.Slot(bool)
    def _set_running(self, running):
        self._update_run_enabled()
        self.benchmark_running_changed.emit(running)

    @QtCore.Slot(object)
    def show_artifact(self, artifact):
        """Render one completed artifact into the route comparison table."""
        try:
            observations = _process._artifact_observations(artifact)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        if not observations:
            self._show_error('Benchmark result contains no observations')
            return
        observation = observations[0]
        candidates = _process._observation_candidates(observation)
        selected = _process._selected_name(observation)
        winner = _process._winner_name(observation)

        self._artifact = artifact
        self._chart.set_candidates(candidates, selected, winner)
        self._table.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            name = _process._candidate_name(candidate)
            eligible = candidate.get('eligible', True)
            median = _process._summary_value(candidate, 'median_ns')
            numpy_ratio = _process._summary_value(candidate, 'numpy_ratio')
            values = (
                name,
                'yes' if eligible else 'no',
                'yes' if name == selected else '',
                'yes' if name == winner else '',
                _process._packing_text(candidate, 'eager'),
                _process._packing_text(candidate, 'scratch'),
                _process._correctness_text(candidate),
                _process._format_time(median),
                _process._format_time(
                    _process._summary_value(candidate, 'p95_ns')),
                self._format_ratio(
                    _process._summary_value(candidate, 'noise')),
                _process._format_time(
                    _process._summary_value(
                        candidate, 'python_median_ns')),
                self._format_ratio(numpy_ratio, suffix='x'),
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                if not eligible:
                    item.setForeground(QtGui.QColor(130, 130, 130))
                elif name == winner:
                    item.setBackground(QtGui.QColor(80, 150, 90, 80))
                self._table.setItem(row, column, item)
            self._table.item(row, 0).setData(
                QtCore.Qt.ItemDataRole.UserRole, candidate)
        self._table.resizeColumnsToContents()
        if candidates:
            self._table.selectRow(0)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._live_status.complete()
        self._save.setEnabled(True)

    @staticmethod
    def _format_ratio(value, suffix='%'):
        if value is None:
            return ''
        value = float(value)
        if suffix == '%':
            return f'{100 * value:.1f}%'
        return f'{value:.3f}{suffix}'

    @QtCore.Slot(int, int, int, int)
    def _show_candidate_details(self, row, _column, _old_row, _old_column):
        self._chart.set_current_row(row)
        if row < 0:
            self._details.clear()
            return
        item = self._table.item(row, 0)
        candidate = (item.data(QtCore.Qt.ItemDataRole.UserRole)
                     if item else None)
        route = _process._candidate_name(candidate or {})
        samples = []
        if isinstance(self._artifact, dict):
            for panel in self._artifact.get('panels', []):
                samples.extend(
                    sample for sample in panel.get('samples', [])
                    if sample.get('route') == route)
        details = {'candidate': candidate or {}, 'panel_samples': samples}
        self._details.setPlainText(json.dumps(
            details, indent=2, sort_keys=True))

    @QtCore.Slot(int)
    def _select_candidate_row(self, row):
        if 0 <= row < self._table.rowCount():
            self._table.setCurrentCell(row, 0)
            self._table.selectRow(row)

    @QtCore.Slot(str)
    def _show_error(self, message):
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._live_status.fail(message)

    @QtCore.Slot()
    def _show_cancelled(self):
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._live_status.cancel()

    @QtCore.Slot()
    def save_result(self):
        if self._artifact is None:
            return
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save matmul benchmark', '', 'JSON files (*.json)')
        if not path:
            return
        matmul_benchmark.artifact.write_artifact(self._artifact, path)


class MatmulBenchmarkWindow(QtWidgets.QWidget):
    """Combine route inspection and offline atlas views in one window."""

    def __init__(self, process_factory=None, registry=None, parent=None):
        super().__init__(parent)
        self._operation = QtWidgets.QComboBox()
        self._operation.addItem('Matmul')
        self._operation.setToolTip(
            'This prototype currently provides matmul benchmarks only.')
        self._stop = QtWidgets.QPushButton('Stop')
        self._stop.setEnabled(False)
        self._stop.setToolTip(
            'Immediately stop the active worker from either page.')
        self._tabs = QtWidgets.QTabWidget()
        self.route_inspector = RouteInspectorWidget(process_factory)
        self.dispatch_atlas = _atlas.AtlasWidget(
            registry, process_factory=process_factory,
            starter_profile=self.route_inspector.benchmark_profile())
        self._tabs.addTab(self.route_inspector, 'Route Inspector')
        self._tabs.addTab(self.dispatch_atlas, 'Dispatch Atlas')

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        operation = QtWidgets.QHBoxLayout()
        operation.addWidget(QtWidgets.QLabel('Operation'))
        operation.addWidget(self._operation)
        operation.addStretch(1)
        operation.addWidget(self._stop)
        layout.addLayout(operation)
        layout.addWidget(self._tabs)

        self._stop.clicked.connect(self.stop_active_run)
        self.route_inspector.artifact_ready.connect(
            lambda artifact: self.dispatch_atlas.add_artifacts([artifact]))
        self.route_inspector.profile_changed.connect(
            self.dispatch_atlas.set_starter_profile)
        self.route_inspector.benchmark_running_changed.connect(
            self.dispatch_atlas.set_peer_running)
        self.route_inspector.benchmark_running_changed.connect(
            self._update_stop)
        self.dispatch_atlas.collection_running_changed.connect(
            self.route_inspector.set_peer_running)
        self.dispatch_atlas.collection_running_changed.connect(
            self._update_stop)

    def _active_page(self):
        pages = (
            ('Route Inspector', self.route_inspector),
            ('Dispatch Atlas', self.dispatch_atlas),
        )
        for name, page in pages:
            if page.running:
                return name, page
        return None, None

    @QtCore.Slot()
    def stop_active_run(self):
        """Stop the one active worker regardless of the selected page."""
        name, page = self._active_page()
        if page is None:
            return
        self._stop.setText(f'Stopping {name}...')
        self._stop.setEnabled(False)
        page.stop()

    @QtCore.Slot(bool)
    def _update_stop(self, _running):
        name, page = self._active_page()
        self._stop.setText('Stop' if page is None else f'Stop {name}')
        self._stop.setEnabled(page is not None)

    def closeEvent(self, event):
        if not self.dispatch_atlas.confirm_close():
            event.ignore()
            return
        self.route_inspector.shutdown()
        self.dispatch_atlas.shutdown()
        super().closeEvent(event)


class MatmulBenchmark(_gui_common.PilotFeature):
    """Open the benchmark visualizer as an MDI feature window."""

    TITLE = 'Benchmark Visualizer'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subwindow = None
        self.window = None

    def populate_menu(self):
        self._action = self.add_action(
            'Profiling',
            text='Benchmark visualizer',
            tip='Compare operation routes and inspect benchmark maps',
            func=self.open_window,
            id='profiling.matmul_benchmark',
            weight=30,
        )

    @QtCore.Slot()
    def open_window(self):
        if self._subwindow is not None and self._subwindow.isVisible():
            self._mgr.mdiArea.setActiveSubWindow(self._subwindow)
            return
        self.window = MatmulBenchmarkWindow()
        self._subwindow = self._mgr.addSubWindow(QtWidgets.QWidget())
        self._subwindow.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._subwindow.destroyed.connect(self._window_destroyed)
        self._subwindow.setWindowTitle(self.TITLE)
        self._subwindow.setWidget(self.window)
        self._subwindow.resize(1000, 720)
        self._subwindow.show()

    @QtCore.Slot()
    def _window_destroyed(self):
        self._subwindow = None
        self.window = None


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
