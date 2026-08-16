# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Configuration helpers for Atlas starter collections."""

import os

from PySide6 import QtCore, QtWidgets

from ... import matmul_benchmark
from . import _matmul_benchmark_profiles as _profile_ui

__all__ = [
    'DimensionRangeEditor',
    'SamplingControls',
    'StarterPlanDialog',
    'default_options',
    'dimension_values',
    'humanize_sampling_text',
    'make_plan',
    'plan_estimate_text',
    'recommended_max_threads',
]


THREAD_LIMIT_TOOLTIP = (
    'Uses at most detected logical CPUs minus one (minimum 1), reserving '
    'one CPU for Qt and system work where possible.')
DIMENSION_MAXIMUM = 1_000_000
DEFAULT_DIMENSION_RANGES = {
    'm_range': {
        'start': 8, 'end': 256, 'spacing': 'powers_of_two', 'step': 1,
    },
    'k_range': {
        'start': 64, 'end': 64, 'spacing': 'powers_of_two', 'step': 1,
    },
    'n_range': {
        'start': 8, 'end': 256, 'spacing': 'powers_of_two', 'step': 1,
    },
}
ROUTE_CHOICES = (
    ('generic', 'Generic',
     'Direct native loop. It is the baseline and is often useful for '
     'small matrices.'),
    ('blas_gemm', 'BLAS GEMM',
     'Calls the configured BLAS backend. The selected BLAS thread count '
     'applies and some layouts may need packing.'),
    ('winograd', 'Winograd',
     'Tests the Winograd route. It is available only for unbatched '
     'matrices with even M, K, and N.'),
)
_ROUTE_LABELS = {
    name: label for name, label, _helper in ROUTE_CHOICES
}
QUALITY_CHOICES = (
    ('Preview', 'preview'),
    ('Stable', 'stable'),
    ('Custom fixed schedule', 'custom'),
)
BUDGET_HELP = {
    'Fixed sample count': (
        'Run the exact setup, averaging, and measurement-round counts shown '
        'by the selected quality setting.'),
    'Target duration': (
        'Calibrate every selected input and dispatch on this machine, then '
        'choose calls per result and complete balanced measurement rounds '
        'near the requested wall-time budget. The conservative uncertainty '
        'bound and safety margin may finish early. Long runs use bounded '
        'saved chunks.'),
}
DURATION_PRESETS = (
    ('1 minute', 60.0),
    ('10 minutes', 600.0),
    ('1 hour', 3600.0),
    ('Custom', None),
)


def recommended_max_threads():
    """Reserve one detected logical CPU for UI and system work."""
    return max(1, (os.cpu_count() or 1) - 1)


def _quality_description(mode, target_duration=False):
    """Describe one sampling schedule without backend terminology."""
    setup_runs = mode.warmups
    rounds = mode.panels
    if target_duration:
        return (
            f'Discard {setup_runs} setup runs before calibration and each '
            f'saved chunk. Collect at least {rounds} measurement rounds; '
            f'calibration chooses calls averaged per result and the final '
            f'number of rounds.')
    return (
        f'Discard {setup_runs} setup runs, then record {rounds} measurement '
        f'rounds. Each result averages {mode.repetitions} back-to-back calls.')


def humanize_sampling_text(text):
    """Translate worker sampling terms for the interactive UI."""
    for backend, plain in (
            ('Repetitions', 'Calls per result'),
            ('repetitions', 'calls per result'),
            ('Repetition', 'Call per result'),
            ('repetition', 'call per result'),
            ('Panels', 'Measurement rounds'),
            ('panels', 'measurement rounds'),
            ('Panel', 'Measurement round'),
            ('panel', 'measurement round'),
            ('Shards', 'Saved chunks'),
            ('shards', 'saved chunks'),
            ('Shard', 'Saved chunk'),
            ('shard', 'saved chunk')):
        text = text.replace(backend, plain)
    return text


class SamplingControls(QtWidgets.QWidget):
    """Select a preset or one bounded custom fixed sampling schedule."""

    changed = QtCore.Signal()

    def __init__(self, quality='preview', warmups=2, repetitions=5,
                 rounds=2, parent=None):
        super().__init__(parent)
        self._target_duration = False
        self.quality = QtWidgets.QComboBox()
        for label, name in QUALITY_CHOICES:
            self.quality.addItem(label, name)
        self.helper = QtWidgets.QLabel()
        self.helper.setWordWrap(True)
        self.warmups = self._spin_box(
            0, matmul_benchmark.schema.MAX_MODE_WARMUPS, warmups,
            'Calls discarded before measurements begin.')
        self.repetitions = self._spin_box(
            1, matmul_benchmark.schema.MAX_MODE_REPETITIONS, repetitions,
            'Back-to-back calls combined into each reported timing result.')
        self.rounds = self._spin_box(
            1, matmul_benchmark.schema.MAX_MODE_PANELS, rounds,
            'Balanced measurement rounds retained as independent samples.')
        self.custom_schedule = QtWidgets.QWidget()
        custom_layout = QtWidgets.QGridLayout(self.custom_schedule)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        for column, (label, control) in enumerate((
                ('Discarded setup runs', self.warmups),
                ('Calls averaged per result', self.repetitions),
                ('Measurement rounds', self.rounds))):
            custom_layout.addWidget(QtWidgets.QLabel(label), 0, column)
            custom_layout.addWidget(control, 1, column)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.quality)
        layout.addWidget(self.helper)
        layout.addWidget(self.custom_schedule)

        index = self.quality.findData(quality)
        if index < 0:
            raise ValueError(f'unsupported measurement quality: {quality!r}')
        self.quality.setCurrentIndex(index)
        self.quality.currentIndexChanged.connect(self._controls_changed)
        for control in (self.warmups, self.repetitions, self.rounds):
            control.valueChanged.connect(self._controls_changed)
        self._refresh()

    @staticmethod
    def _spin_box(minimum, maximum, value, tooltip):
        control = QtWidgets.QSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        control.setToolTip(tooltip)
        return control

    def quality_name(self):
        return self.quality.currentData()

    def mode_spec(self):
        name = self.quality_name()
        if name != 'custom':
            return matmul_benchmark.schema.ModeSpec.preset(name)
        return matmul_benchmark.schema.ModeSpec(
            name='preview', warmups=self.warmups.value(),
            repetitions=self.repetitions.value(), panels=self.rounds.value())

    def is_valid(self):
        try:
            self.mode_spec()
        except ValueError:
            return False
        return True

    def set_target_duration(self, enabled):
        self._target_duration = bool(enabled)
        custom_index = self.quality.findData('custom')
        if enabled and self.quality_name() == 'custom':
            self.quality.setCurrentIndex(self.quality.findData('preview'))
        self.quality.view().setRowHidden(custom_index, enabled)
        self._refresh()

    @QtCore.Slot()
    def _controls_changed(self):
        self._refresh()
        self.changed.emit()

    def _refresh(self):
        custom = self.quality_name() == 'custom'
        show_custom = custom and not self._target_duration
        self.custom_schedule.setVisible(show_custom)
        self.custom_schedule.setEnabled(show_custom)
        try:
            mode = self.mode_spec()
        except ValueError as exc:
            text = f'This custom schedule is invalid: {exc}.'
        else:
            text = _quality_description(mode, self._target_duration)
        self.helper.setText(text)
        self.quality.setToolTip(text)
        for index in range(self.quality.count()):
            name = self.quality.itemData(index)
            if name == 'custom':
                item_text = (
                    'Choose exact setup runs, calls averaged per result, and '
                    'measurement rounds. Available only with a fixed budget.')
            else:
                item_text = _quality_description(
                    matmul_benchmark.schema.ModeSpec.preset(name),
                    self._target_duration)
            self.quality.setItemData(
                index, item_text, QtCore.Qt.ToolTipRole)


def dimension_values(spec, name='dimension'):
    """Expand an inclusive linear or power-of-two dimension range."""

    if not isinstance(spec, dict):
        raise ValueError(f'{name} range must be an object')
    required = {'start', 'end', 'spacing', 'step'}
    if set(spec) != required:
        raise ValueError(
            f'{name} range fields must be {sorted(required)}')
    start, end, step = spec['start'], spec['end'], spec['step']
    for label, value in (
            ('start', start), ('end', end), ('step', step)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{name} {label} must be an integer')
    if start < 1:
        raise ValueError(f'{name} start must be at least 1')
    if end < start:
        raise ValueError(f'{name} end must not be smaller than start')
    if step < 1:
        raise ValueError(f'{name} step must be at least 1')
    if spec['spacing'] == 'linear':
        count = (end - start) // step + 1
        if count > matmul_benchmark.collection.MAX_COLLECTION_CELLS:
            raise ValueError(
                f'{name} range contains {count:,} values, limit is '
                f'{matmul_benchmark.collection.MAX_COLLECTION_CELLS:,}')
        return tuple(range(start, end + 1, step))
    if spec['spacing'] != 'powers_of_two':
        raise ValueError(f'{name} spacing is not supported')
    values = []
    value = start
    while value <= end:
        values.append(value)
        value *= 2
    return tuple(values)


def default_options(dtype='float32', threads=1):
    """Return editable defaults for the first shape-boundary Atlas."""
    return {
        **{
            name: dict(spec)
            for name, spec in DEFAULT_DIMENSION_RANGES.items()
        },
        'dtype': dtype,
        'threads': threads,
        'input_profiles': tuple(
            profile.to_dict()
            for profile in _profile_ui.default_input_profiles()),
        'mode': 'preview',
        'custom_warmups': 2,
        'custom_repetitions': 5,
        'custom_rounds': 2,
        'budget': 'fixed',
        'target_duration_seconds': 60.0,
        'checkpoint_seconds': 60.0,
        'routes': matmul_benchmark.collection.DEFAULT_ROUTES,
        'numpy_baseline': True,
        'seed': 20260815,
        'output_path': None,
    }


def make_plan(options):
    """Validate editable options and build one explicit backend plan."""
    plan_options = dict(options)
    unknown_routes = sorted(
        set(plan_options.get('routes', ()))
        - set(matmul_benchmark.collection.DEFAULT_ROUTES))
    if unknown_routes:
        raise ValueError(f'Unknown starter routes: {unknown_routes}')
    for axis in ('m', 'k', 'n'):
        plan_options[f'{axis}_values'] = dimension_values(
            plan_options.pop(f'{axis}_range'), axis.upper())
    raw_profiles = plan_options.pop('input_profiles')
    input_profiles = tuple(
        matmul_benchmark.profiles.InputProfile.from_dict(profile)
        for profile in raw_profiles)
    budget = plan_options.pop('budget')
    target_seconds = plan_options.pop('target_duration_seconds')
    checkpoint_seconds = plan_options.pop('checkpoint_seconds')
    quality = plan_options.pop('mode')
    custom_values = {
        'warmups': plan_options.pop('custom_warmups', 2),
        'repetitions': plan_options.pop('custom_repetitions', 5),
        'panels': plan_options.pop('custom_rounds', 2),
    }
    if budget == 'fixed':
        target_duration = None
        mode = (
            matmul_benchmark.schema.ModeSpec(
                name='preview', **custom_values)
            if quality == 'custom' else
            matmul_benchmark.schema.ModeSpec.preset(quality))
    elif budget == 'target_duration':
        if quality == 'custom':
            raise ValueError(
                'Custom sampling is available only with a fixed budget')
        if not plan_options.get('output_path'):
            raise ValueError(
                'Target-duration runs require an Output JSON path so '
                'checkpoints survive worker restarts')
        target_duration = matmul_benchmark.duration.TargetDurationSpec(
            seconds=target_seconds, mode=quality,
            checkpoint_seconds=checkpoint_seconds)
        mode = matmul_benchmark.schema.ModeSpec.preset(quality)
    else:
        raise ValueError(f'Unsupported run budget: {budget!r}')
    return matmul_benchmark.collection.input_profile_plan(
        input_profiles=input_profiles, target_duration=target_duration,
        mode=mode, **plan_options)


def format_estimate(estimate, plan=None):
    resource_budget = matmul_benchmark.arrays.resolve_resource_budget()
    text = (
        f'{estimate.cell_count:,} input points, '
        f'{estimate.route_count} dispatches, '
        f'{estimate.panel_count} measurement rounds, '
        f'{estimate.preflight_calls:,} input-check calls, '
        f'{estimate.matmul_calls:,} matmul calls, '
        f'{estimate.scalar_contractions:,} multiply-add terms, '
        f'{estimate.measurement_work:,} estimated work, '
        f'{_format_bytes(estimate.peak_bytes)} conservative peak memory, '
        f'{_format_bytes(resource_budget.peak_bytes)} current worker-safe '
        f'budget from {_format_bytes(resource_budget.available_bytes)} '
        'available')
    if plan is not None:
        artifact_bytes = \
            matmul_benchmark.collection.estimate_artifact_bytes(plan)
        text += f', about {_format_bytes(artifact_bytes)} artifact JSON'
    return text


def plan_estimate_text(estimate, plan):
    """Describe either an exact fixed run or a pre-calibration floor."""
    details = format_estimate(estimate, plan)
    target = plan.target_duration
    if target is None:
        return f'Fixed schedule: {details}'
    return (
        f'Target wall-time budget: {_format_duration(target.seconds)}. '
        f'Calibration is required before the worker can predict calls per '
        f'result, measurement rounds, saved chunks, or active time. The '
        f'conservative upper estimate '
        f'must fit a {target.safety_fraction:.0%} safety budget, so the run '
        f'may finish early. Minimum {target.mode} quality schedule: '
        f'{details}. '
        f'Work units are a hardware-neutral size estimate, not elapsed time '
        f'or an execution limit.')


def _format_bytes(value):
    value = float(value)
    for suffix in ('B', 'KiB', 'MiB', 'GiB'):
        if value < 1024 or suffix == 'GiB':
            return f'{value:.0f} {suffix}' if suffix == 'B' \
                else f'{value:.2f} {suffix}'
        value /= 1024


def _format_duration(seconds):
    seconds = float(seconds)
    if seconds >= 3600:
        count = seconds / 3600
        unit = 'hour' if count == 1 else 'hours'
    elif seconds >= 60:
        count = seconds / 60
        unit = 'minute' if count == 1 else 'minutes'
    else:
        count = seconds
        unit = 'second' if count == 1 else 'seconds'
    value = f'{count:.3g}'
    return f'{value} {unit}'


class DimensionRangeEditor(QtWidgets.QWidget):
    """Edit a bounded integer sequence without comma-separated input."""

    changed = QtCore.Signal()

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self._start = self._spin_box()
        self._end = self._spin_box()
        self._spacing = QtWidgets.QComboBox()
        self._spacing.addItem('Linear', 'linear')
        self._spacing.addItem('Powers of two', 'powers_of_two')
        self._spacing.setItemData(
            0, 'Add Step until reaching End.', QtCore.Qt.ToolTipRole)
        self._spacing.setItemData(
            1, 'Multiply by 2 until reaching End.',
            QtCore.Qt.ToolTipRole)
        self._step = self._spin_box()
        self._count = QtWidgets.QLabel()
        self.set_spec(spec)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for label, control in (
                ('Start', self._start), ('End', self._end),
                ('Spacing', self._spacing), ('Step', self._step)):
            layout.addWidget(QtWidgets.QLabel(label))
            layout.addWidget(control)
        layout.addWidget(self._count)
        layout.addStretch(1)

        for control in (self._start, self._end, self._step):
            control.valueChanged.connect(self._value_changed)
        self._spacing.currentIndexChanged.connect(
            self._spacing_changed)
        self._spacing_changed()

    @staticmethod
    def _spin_box():
        control = QtWidgets.QSpinBox()
        control.setRange(1, DIMENSION_MAXIMUM)
        return control

    def set_spec(self, spec):
        for control, value in (
                (self._start, spec['start']),
                (self._end, spec['end']),
                (self._step, spec['step'])):
            control.setValue(value)
        index = self._spacing.findData(spec['spacing'])
        if index < 0:
            raise ValueError('unsupported dimension spacing')
        self._spacing.setCurrentIndex(index)
        self._refresh_count()

    def spec(self):
        return {
            'start': self._start.value(),
            'end': self._end.value(),
            'spacing': self._spacing.currentData(),
            'step': self._step.value(),
        }

    def values(self):
        return dimension_values(self.spec())

    @QtCore.Slot()
    def _value_changed(self):
        self._refresh_count()
        self.changed.emit()

    @QtCore.Slot()
    def _spacing_changed(self):
        linear = self._spacing.currentData() == 'linear'
        self._step.setEnabled(linear)
        self._spacing.setToolTip(
            self._spacing.currentData(QtCore.Qt.ToolTipRole) or '')
        self._refresh_count()
        self.changed.emit()

    def _refresh_count(self):
        try:
            values = self.values()
        except ValueError:
            self._count.setText('invalid')
        else:
            suffix = '' if len(values) == 1 else 's'
            self._count.setText(
                f'{len(values)} value{suffix}, last {values[-1]}')


class _CheckGrid(QtWidgets.QWidget):
    """Expose a fixed set of named choices as multi-select checkboxes."""

    changed = QtCore.Signal()

    def __init__(self, choices, selected, columns=3, parent=None):
        super().__init__(parent)
        selected = set(selected)
        self.boxes = {}
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for index, (name, label, helper) in enumerate(choices):
            box = QtWidgets.QCheckBox(label)
            box.setChecked(name in selected)
            box.setToolTip(helper)
            box.toggled.connect(
                lambda _checked: self.changed.emit())
            self.boxes[name] = box
            layout.addWidget(box, index // columns, index % columns)

    def selected(self):
        return tuple(
            name for name, box in self.boxes.items()
            if box.isChecked())


class StarterPlanDialog(QtWidgets.QDialog):
    """Edit a primitive input grid while showing its exact schedule size."""

    def __init__(self, options, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Configure starter Atlas collection')
        self.setMinimumWidth(1100)
        self._plan = None
        self._build_ui(options)
        self._connect_controls()
        self._update_estimate()

    def _build_ui(self, options):
        self._m_range = DimensionRangeEditor(options['m_range'])
        self._k_range = DimensionRangeEditor(options['k_range'])
        self._n_range = DimensionRangeEditor(options['n_range'])
        self._dtype = QtWidgets.QComboBox()
        self._dtype.addItems(matmul_benchmark.schema.SUPPORTED_DTYPES)
        self._dtype.setCurrentText(options['dtype'])
        self._dtype.setToolTip(
            'Element type used by every generated input and dispatch.')
        self._threads = QtWidgets.QSpinBox()
        self._threads.setRange(1, recommended_max_threads())
        self._threads.setValue(options['threads'])
        self._threads.setToolTip(THREAD_LIMIT_TOOLTIP)
        self._sampling = SamplingControls(
            quality=options['mode'],
            warmups=options.get('custom_warmups', 2),
            repetitions=options.get('custom_repetitions', 5),
            rounds=options.get('custom_rounds', 2))
        self._budget = QtWidgets.QComboBox()
        for name, value in (
                ('Fixed sample count', 'fixed'),
                ('Target duration', 'target_duration')):
            self._budget.addItem(name, value)
            self._budget.setItemData(
                self._budget.count() - 1, BUDGET_HELP[name],
                QtCore.Qt.ToolTipRole)
        budget_index = self._budget.findData(options['budget'])
        if budget_index < 0:
            raise ValueError('unsupported run budget')
        self._budget.setCurrentIndex(budget_index)
        self._duration_preset = QtWidgets.QComboBox()
        for name, seconds in DURATION_PRESETS:
            self._duration_preset.addItem(name, seconds)
        self._custom_duration = QtWidgets.QDoubleSpinBox()
        self._custom_duration.setDecimals(0)
        self._custom_duration.setRange(1, 604_800)
        self._custom_duration.setSuffix(' s')
        self._custom_duration.setValue(options['target_duration_seconds'])
        self._set_duration_preset(options['target_duration_seconds'])
        duration_control = QtWidgets.QWidget()
        duration_layout = QtWidgets.QHBoxLayout(duration_control)
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.addWidget(self._duration_preset)
        duration_layout.addWidget(self._custom_duration)
        duration_layout.addStretch(1)
        self._duration_control = duration_control
        self._checkpoint = QtWidgets.QDoubleSpinBox()
        self._checkpoint.setDecimals(0)
        self._checkpoint.setRange(1, 86_400)
        self._checkpoint.setSuffix(' s')
        self._checkpoint.setValue(options['checkpoint_seconds'])
        self._checkpoint.setToolTip(
            'Maximum requested duration of each saved chunk. A checkpoint is '
            'published only after its complete measurement rounds finish. One '
            'timed block may make the practical interval longer.')
        self._budget_helper = QtWidgets.QLabel()
        self._budget_helper.setWordWrap(True)
        self._routes = _CheckGrid(ROUTE_CHOICES, options['routes'])
        self._route_helper = QtWidgets.QLabel()
        self._route_helper.setWordWrap(True)
        route_control = QtWidgets.QWidget()
        route_layout = QtWidgets.QVBoxLayout(route_control)
        route_layout.setContentsMargins(0, 0, 0, 0)
        route_layout.addWidget(self._routes)
        route_layout.addWidget(self._route_helper)
        input_profiles = tuple(
            matmul_benchmark.profiles.InputProfile.from_dict(profile)
            for profile in options['input_profiles'])
        self._profiles = _profile_ui.InputProfilesWidget(
            profiles=input_profiles,
            preview_shape=self._preview_shape(),
            routes=self._routes.selected())
        self._numpy = QtWidgets.QCheckBox(
            'Include NumPy end-to-end baseline')
        self._numpy.setChecked(options['numpy_baseline'])
        self._numpy.setToolTip(
            'Times np.matmul through the Python call boundary for reference.')
        self._seed = QtWidgets.QSpinBox()
        self._seed.setRange(0, 2_147_483_647)
        self._seed.setValue(options['seed'])
        self._seed.setToolTip(
            'Makes generated values and balanced cell order reproducible.')
        output = self._build_output_control(options)

        form = QtWidgets.QFormLayout()
        form.addRow('M range (inclusive)', self._m_range)
        form.addRow('K range (inclusive)', self._k_range)
        form.addRow('N range (inclusive)', self._n_range)
        form.addRow('Data type', self._dtype)
        form.addRow('BLAS threads', self._threads)
        form.addRow('Input cases', self._profiles)
        form.addRow('Measurement quality', self._sampling)
        form.addRow('Run budget', self._budget)
        form.addRow('', self._budget_helper)
        form.addRow('Target duration', self._duration_control)
        form.addRow('Checkpoint interval', self._checkpoint)
        form.addRow('Dispatches', route_control)
        form.addRow('', self._numpy)
        form.addRow('Shuffle seed', self._seed)
        form.addRow('Output JSON', output)

        self._estimate = QtWidgets.QLabel()
        self._estimate.setWordWrap(True)
        self._estimate.setToolTip(
            'Input-check calls verify results before timing. Estimated work '
            'is a hardware-neutral comparison metric, not a time prediction. '
            'Peak memory includes '
            'inputs, native copies, outputs, and comparison temporaries.')
        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._estimate)
        layout.addWidget(self._buttons)
        self._update_budget_controls()

    def _build_output_control(self, options):
        self._output_path = QtWidgets.QLineEdit(
            options.get('output_path') or '')
        self._output_path.setPlaceholderText(
            'blank = keep in memory, then use Save collection')
        self._choose_output = QtWidgets.QPushButton('Browse...')
        output = QtWidgets.QWidget()
        output_layout = QtWidgets.QHBoxLayout(output)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self._output_path, 1)
        output_layout.addWidget(self._choose_output)
        return output

    def _connect_controls(self):
        for control in (self._m_range, self._k_range, self._n_range):
            control.changed.connect(self._update_estimate)
        self._profiles.profiles_changed.connect(self._update_estimate)
        self._routes.changed.connect(self._profiles_changed)
        self._dtype.currentTextChanged.connect(self._update_estimate)
        self._sampling.changed.connect(self._update_estimate)
        self._budget.currentIndexChanged.connect(
            self._budget_changed)
        self._duration_preset.currentIndexChanged.connect(
            self._duration_preset_changed)
        self._custom_duration.valueChanged.connect(self._update_estimate)
        self._checkpoint.valueChanged.connect(self._update_estimate)
        for control in (self._threads, self._seed):
            control.valueChanged.connect(self._update_estimate)
        self._numpy.toggled.connect(self._update_estimate)
        self._output_path.textChanged.connect(self._update_estimate)
        self._choose_output.clicked.connect(self._choose_output_path)

    def options(self):
        return {
            'm_range': self._m_range.spec(),
            'k_range': self._k_range.spec(),
            'n_range': self._n_range.spec(),
            'dtype': self._dtype.currentText(),
            'threads': self._threads.value(),
            'input_profiles': tuple(
                profile.to_dict() for profile in self._profiles.profiles()),
            'mode': self._sampling.quality_name(),
            'custom_warmups': self._sampling.warmups.value(),
            'custom_repetitions': self._sampling.repetitions.value(),
            'custom_rounds': self._sampling.rounds.value(),
            'budget': self._budget.currentData(),
            'target_duration_seconds': self._target_duration_seconds(),
            'checkpoint_seconds': self._checkpoint.value(),
            'routes': self._routes.selected(),
            'numpy_baseline': self._numpy.isChecked(),
            'seed': self._seed.value(),
            'output_path': self._output_path.text().strip() or None,
        }

    def plan(self):
        if self._plan is None:
            raise ValueError('Collection plan is not valid')
        return self._plan

    @QtCore.Slot()
    def _profiles_changed(self):
        self._profiles.set_context(
            preview_shape=self._preview_shape(),
            routes=self._routes.selected())
        self._update_estimate()

    @QtCore.Slot()
    def _budget_changed(self):
        self._update_budget_controls()
        self._update_estimate()

    @QtCore.Slot()
    def _duration_preset_changed(self):
        seconds = self._duration_preset.currentData()
        custom = seconds is None
        self._custom_duration.setEnabled(custom)
        if not custom:
            self._custom_duration.setValue(seconds)
        self._update_estimate()

    def _set_duration_preset(self, seconds):
        index = next((
            index for index in range(self._duration_preset.count())
            if self._duration_preset.itemData(index) == seconds
        ), self._duration_preset.count() - 1)
        self._duration_preset.setCurrentIndex(index)
        self._custom_duration.setEnabled(
            self._duration_preset.currentData() is None)

    def _target_duration_seconds(self):
        seconds = self._duration_preset.currentData()
        return self._custom_duration.value() if seconds is None else seconds

    def _update_budget_controls(self):
        target = self._budget.currentData() == 'target_duration'
        self._sampling.set_target_duration(target)
        self._duration_control.setEnabled(target)
        self._checkpoint.setEnabled(target)
        self._budget.setToolTip(BUDGET_HELP[self._budget.currentText()])
        self._budget_helper.setText(BUDGET_HELP[self._budget.currentText()])
        placeholder = (
            'required for target duration; checkpoints are stored beside it'
            if target else
            'blank = keep in memory, then use Save collection')
        self._output_path.setPlaceholderText(placeholder)

    def _preview_shape(self):
        values = tuple(
            control.values()
            for control in (self._m_range, self._k_range, self._n_range))
        return tuple(axis[len(axis) // 2] for axis in values)

    def _update_route_coverage(self, plan):
        if plan is None:
            for box in self._routes.boxes.values():
                box.setEnabled(True)
            self._route_helper.setText(
                'Dispatch coverage is available when the plan is valid.')
            return

        total = len(plan.cells)
        counts = {name: 0 for name in plan.routes}
        for cell in plan.cells:
            for name in cell.routes:
                counts[name] += 1

        descriptions = []
        for name in plan.routes:
            count = counts[name]
            description = (
                f'{_ROUTE_LABELS[name]}: {count:,}/{total:,} input points')
            if count == 0:
                description += ', unavailable for these inputs'
            descriptions.append(description)
        self._route_helper.setText('; '.join(descriptions) + '.')

        single_cell = total == 1
        for name, box in self._routes.boxes.items():
            unavailable = (
                single_cell and box.isChecked()
                and counts.get(name, 0) == 0)
            box.setEnabled(not unavailable)

    @QtCore.Slot()
    def _update_estimate(self):
        if hasattr(self, '_profiles'):
            try:
                self._profiles.set_context(
                    preview_shape=self._preview_shape(),
                    routes=self._routes.selected())
            except ValueError:
                pass
        plan = None
        try:
            plan = make_plan(self.options())
            estimate = matmul_benchmark.collection.estimate_plan(plan)
            matmul_benchmark.collection.validate_plan_resources(
                plan, estimate)
        except (MemoryError, TypeError, ValueError) as exc:
            self._plan = None
            self._estimate.setText(f'Invalid plan: {exc}')
            valid = False
        else:
            self._plan = plan
            text = plan_estimate_text(estimate, plan)
            warnings = \
                matmul_benchmark.collection.recommended_budget_warnings(
                    estimate, plan)
            if warnings:
                text += '\nPlan scale: ' + '; '.join(warnings)
            self._estimate.setText(text)
            valid = True
        self._update_route_coverage(plan)
        self._buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(valid)

    @QtCore.Slot()
    def _choose_output_path(self):
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save starter Atlas collection',
            self._output_path.text(), 'JSON files (*.json)')
        if path:
            self._output_path.setText(path)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
