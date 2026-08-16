# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Compose explicit matmul input profiles for Atlas collections."""

import dataclasses
import re
import uuid

from PySide6 import QtCore, QtWidgets

from ...matmul_benchmark import collection as _collection
from ...matmul_benchmark import profiles as _profiles
from ...matmul_benchmark import schema as _schema

__all__ = [
    'BatchAxesEditor',
    'InputProfileEditor',
    'InputProfilesWidget',
    'default_input_profiles',
]


CORE_STORAGE_CHOICES = (
    (
        'c_compact', 'Rows adjacent', 'c', False,
        'Elements within each row are adjacent. Element strides resolve '
        'to (columns, 1).',
    ),
    (
        'c_gap', 'Rows adjacent with padding', 'c', True,
        'Adds unused elements after each row. The gap is measured in '
        'elements.',
    ),
    (
        'f_compact', 'Columns adjacent', 'f', False,
        'Elements within each column are adjacent. Element strides resolve '
        'to (1, rows).',
    ),
    (
        'f_gap', 'Columns adjacent with padding',
        'f', True,
        'Adds unused elements after each column. The gap is measured in '
        'elements.',
    ),
    (
        'custom', 'Fixed row and column strides', 'custom', False,
        'Enter the element offset between adjacent rows and columns. '
        'Negative and zero values are supported.',
    ),
)
BATCH_MAPPING_CHOICES = (
    (
        'unbatched', 'No batch axes',
        'Uses A(M,K) and B(K,N) without batch axes.',
    ),
    (
        'matched', 'Independent A and B matrices per batch',
        'Both inputs have the same batch axis and advance to independent '
        'matrices.',
    ),
    (
        'reuse_lhs', 'Reuse A across B batches',
        'A has size 1 and reuses the same data; B advances through the batch.',
    ),
    (
        'reuse_rhs', 'Reuse B across A batches',
        'B has size 1 and reuses the same data; A advances through the batch.',
    ),
)
EXTENT_CHOICES = (
    ('same', 'Output size'),
    ('one', '1 (broadcast)'),
)
STRIDE_CHOICES = (
    ('auto', 'Auto (next matrix)'),
    ('zero', 'Zero (reuse same data)'),
    ('custom', 'Fixed integer'),
)
DEFAULT_ROUTES = _collection.DEFAULT_ROUTES
ROUTE_LABELS = {
    'naive': 'Naive',
    'blas_dot': 'BLAS DOT',
    'blas_gevm': 'BLAS GEVM',
    'blas_gemv': 'BLAS GEMV',
    'blas_gemm': 'BLAS GEMM',
    'winograd': 'Winograd',
    'numpy': 'NumPy',
}


def _profile_id():
    return f'profile-{uuid.uuid4().hex}'


def _integer_tuple(text, name):
    text = text.strip()
    delimiters = {'(': ')', '[': ']'}
    if text[:1] in delimiters:
        if not text.endswith(delimiters[text[0]]):
            raise ValueError(f'{name} has mismatched brackets')
        text = text[1:-1].strip()
    elif text[-1:] in delimiters.values():
        raise ValueError(f'{name} has mismatched brackets')
    tokens = [token for token in re.split(r'[\s,]+', text) if token]
    if not tokens:
        raise ValueError(f'{name} must contain integers')
    try:
        return tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(
            f'{name} must contain comma-separated integers') from exc


def _core_choice(name):
    return next(choice for choice in CORE_STORAGE_CHOICES
                if choice[0] == name)


def _core_storage(name, gap, row_stride, column_stride):
    _key, _label, order, uses_gap, _helper = _core_choice(name)
    if order == 'custom':
        return _profiles.CoreStorage.custom(row_stride, column_stride)
    return _profiles.CoreStorage(
        order=order, leading_dimension_gap=gap if uses_gap else 0)


def _core_choice_name(core):
    if core.order == 'custom':
        return 'custom'
    suffix = 'gap' if core.leading_dimension_gap else 'compact'
    return f'{core.order}_{suffix}'


def _stride(mode, value=None):
    return _profiles.BatchStride(mode=mode, value=value)


def _batch_axis(output_extent, lhs_extent, lhs_mode,
                rhs_extent, rhs_mode, lhs_value=None, rhs_value=None):
    return _profiles.BatchAxis(
        output_extent=output_extent,
        lhs_extent=lhs_extent,
        rhs_extent=rhs_extent,
        lhs_stride=_stride(lhs_mode, lhs_value),
        rhs_stride=_stride(rhs_mode, rhs_value),
    )


def _basic_batch_axes(mapping, extent):
    if mapping == 'unbatched':
        return ()
    if mapping == 'matched':
        values = ('same', 'auto', 'same', 'auto')
    elif mapping == 'reuse_lhs':
        values = ('one', 'zero', 'same', 'auto')
    elif mapping == 'reuse_rhs':
        values = ('same', 'auto', 'one', 'zero')
    else:
        raise ValueError(f'Unknown basic batch mapping: {mapping!r}')
    return (_batch_axis(extent, *values),)


def _basic_mapping(profile):
    axes = profile.batch_axes
    if not axes:
        return 'unbatched', 8
    if len(axes) != 1:
        return None, axes[0].output_extent
    axis = axes[0]
    signature = (
        axis.lhs_extent, axis.lhs_stride.mode,
        axis.rhs_extent, axis.rhs_stride.mode,
    )
    mappings = {
        ('same', 'auto', 'same', 'auto'): 'matched',
        ('one', 'zero', 'same', 'auto'): 'reuse_lhs',
        ('same', 'auto', 'one', 'zero'): 'reuse_rhs',
    }
    mapping = mappings.get(signature)
    if mapping is None:
        return None, axis.output_extent
    if axis.lhs_stride.value is not None \
            or axis.rhs_stride.value is not None:
        return None, axis.output_extent
    return mapping, axis.output_extent


def _storage_text(core):
    if core.order == 'custom':
        return (
            f'fixed row step {core.row_stride}, '
            f'column step {core.column_stride}')
    order = 'rows' if core.order == 'c' else 'columns'
    gap = core.leading_dimension_gap
    return f'{order} padded by {gap}' if gap else f'{order} adjacent'


def _batch_text(profile):
    mapping, extent = _basic_mapping(profile)
    labels = {
        choice[0]: choice[1] for choice in BATCH_MAPPING_CHOICES
    }
    if mapping == 'unbatched':
        return labels[mapping]
    if mapping is not None:
        return f'{labels[mapping]} (b={extent})'
    count = len(profile.batch_axes)
    suffix = '' if count == 1 else 's'
    return f'Custom ({count} batch axis{suffix})'


def _eligible_routes(profile, m_value, k_value, n_value, routes):
    resolved = profile.resolve(m_value, k_value, n_value)
    return tuple(
        route for route in routes
        if route != 'winograd'
        or (resolved.broadcast == 'matrix'
            and resolved.m % 2 == 0 and resolved.k % 2 == 0
            and resolved.n % 2 == 0)
    )


def _route_text(routes):
    return ', '.join(ROUTE_LABELS.get(route, route) for route in routes)


def _row_major_compatible(spec):
    rows, columns = spec.shape[-2:]
    row_stride, column_stride = spec.strides[-2:]
    return column_stride == 1 and row_stride >= columns


def _blas_compatible(spec):
    rows, columns = spec.shape[-2:]
    row_stride, column_stride = spec.strides[-2:]
    return (
        column_stride == 1 and row_stride >= columns
        or row_stride == 1 and column_stride >= rows
    )


def _packed_operands(resolved, predicate):
    names = tuple(
        name for name, operand in (
            ('A', resolved.lhs), ('B', resolved.rhs))
        if not predicate(operand)
    )
    return 'none' if not names else 'pack ' + ' + '.join(names)


def _matrix_pack_kind(operand, output_batch):
    batch_shape = operand.shape[:-2]
    batch_strides = operand.strides[:-2]
    missing = len(output_batch) - len(batch_shape)
    shape = (1,) * missing + batch_shape
    strides = (0,) * missing + batch_strides
    broadcast = any(
        input_extent == 1 and output_extent > 1
        for input_extent, output_extent in zip(shape, output_batch))
    zero_step = any(
        input_extent > 1 and stride == 0
        for input_extent, stride in zip(shape, strides))
    return 'eager' if broadcast and not zero_step else 'scratch'


def _blas_packing_text(resolved):
    output_batch = resolved.output_shape[:-2]
    packing = tuple(
        f'{name} {_matrix_pack_kind(operand, output_batch)} pack'
        for name, operand in (
            ('A', resolved.lhs), ('B', resolved.rhs))
        if not _blas_compatible(operand)
    )
    return 'none' if not packing else ' + '.join(packing)


def _packing_text(resolved, routes):
    details = []
    if 'naive' in routes:
        details.append('Naive: none')
    if 'blas_gemm' in routes:
        details.append('BLAS GEMM: ' + _blas_packing_text(resolved))
    if 'winograd' in routes:
        details.append(
            'Winograd: ' + _packed_operands(
                resolved, _row_major_compatible))
    return '; '.join(details) or 'not applicable'


def _batch_behavior_text(value):
    return {
        'matrix': 'No batch axes',
        'matched_batch': 'A and B both advance across batches',
        'broadcast_lhs': 'A is reused across one or more batch axes',
        'broadcast_rhs': 'B is reused across one or more batch axes',
        'broadcast_both': 'A and B each reuse data on batch axes',
    }.get(value, value)


def _resolved_text(profile, m_value, k_value, n_value, routes):
    resolved = profile.resolve(m_value, k_value, n_value)
    eligible = _eligible_routes(
        profile, m_value, k_value, n_value, routes)
    return '\n'.join((
        f'A shape {resolved.lhs.shape}, element strides '
        f'{resolved.lhs.strides}, backing storage '
        f'{resolved.lhs_storage_span} elements',
        f'B shape {resolved.rhs.shape}, element strides '
        f'{resolved.rhs.strides}, backing storage '
        f'{resolved.rhs_storage_span} elements',
        f'Output shape {resolved.output_shape}',
        f'Derived M={resolved.m}, K={resolved.k}, N={resolved.n}',
        f'Batch behavior: {_batch_behavior_text(resolved.broadcast)}',
        f'Packing expected by kernel: '
        f'{_packing_text(resolved, eligible)}',
        f'Kernels available for this input: '
        f'{_route_text(eligible) or "none"}',
        'Installed BLAS support is checked before timing.',
    ))


def default_input_profiles():
    """Return one explicit compact, unbatched starter profile."""
    return (_profiles.InputProfile(
        profile_id=_profile_id(),
        name='Rows adjacent, no batch',
        lhs_core=_profiles.CoreStorage('c', 0),
        rhs_core=_profiles.CoreStorage('c', 0),
        batch_axes=(),
    ),)


class BatchAxesEditor(QtWidgets.QWidget):
    """Edit output batch axes and each input's reuse behavior."""

    changed = QtCore.Signal()

    HEADERS = (
        'Axis', 'Output size', 'A size', 'A step', 'A fixed step',
        'B size', 'B step', 'B fixed step',
    )

    def __init__(self, axes=(), parent=None):
        super().__init__(parent)
        self._updating = False
        self._table = QtWidgets.QTableWidget(0, len(self.HEADERS))
        self._table.setHorizontalHeaderLabels(self.HEADERS)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._add = QtWidgets.QPushButton('Add batch axis')
        self._remove = QtWidgets.QPushButton('Remove selected axis')
        helper = QtWidgets.QLabel(
            'One row describes one output batch axis, outermost first. '
            'Choose size 1 to broadcast an input. Auto moves to the next '
            'matrix; Zero reuses the same data; Fixed uses an element step.')
        helper.setWordWrap(True)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._add)
        buttons.addWidget(self._remove)
        buttons.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(helper)
        layout.addWidget(self._table)
        layout.addLayout(buttons)

        self._add.clicked.connect(self.add_axis)
        self._remove.clicked.connect(self.remove_selected_axis)
        self._table.itemSelectionChanged.connect(
            self._update_remove_button)
        self.set_axes(axes)

    def set_axes(self, axes):
        self._updating = True
        self._table.setRowCount(0)
        for axis in axes:
            self._append_axis(axis)
        self._updating = False
        self._update_axis_names()
        self._update_remove_button()
        self.changed.emit()

    def axes(self):
        return tuple(self._axis(row)
                     for row in range(self._table.rowCount()))

    def axis_count(self):
        return self._table.rowCount()

    @QtCore.Slot()
    def add_axis(self):
        axis = _batch_axis(
            8, 'same', 'auto', 'same', 'auto')
        self._append_axis(axis)
        self._update_axis_names()
        self._table.selectRow(self._table.rowCount() - 1)
        self.changed.emit()

    @QtCore.Slot()
    def remove_selected_axis(self):
        row = self._table.currentRow()
        if row < 0:
            return
        self._table.removeRow(row)
        self._update_axis_names()
        self._update_remove_button()
        self.changed.emit()

    def _append_axis(self, axis):
        row = self._table.rowCount()
        self._table.insertRow(row)
        axis_item = QtWidgets.QTableWidgetItem(f'b{row}')
        axis_item.setFlags(axis_item.flags()
                           & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 0, axis_item)
        output = QtWidgets.QSpinBox()
        output.setRange(1, 1_000_000)
        output.setValue(axis.output_extent)
        output.setToolTip('Output size for this batch axis.')
        self._table.setCellWidget(row, 1, output)
        lhs_extent = self._extent_combo(axis.lhs_extent)
        lhs_mode = self._stride_combo(axis.lhs_stride.mode)
        lhs_custom = self._custom_stride(axis.lhs_stride.value)
        rhs_extent = self._extent_combo(axis.rhs_extent)
        rhs_mode = self._stride_combo(axis.rhs_stride.mode)
        rhs_custom = self._custom_stride(axis.rhs_stride.value)
        for column, control in enumerate((
                lhs_extent, lhs_mode, lhs_custom,
                rhs_extent, rhs_mode, rhs_custom), start=2):
            self._table.setCellWidget(row, column, control)
        self._connect_axis_control(output)
        self._connect_extent_controls(
            lhs_extent, lhs_mode, lhs_custom)
        self._connect_extent_controls(
            rhs_extent, rhs_mode, rhs_custom)
        self._connect_stride_controls(
            lhs_extent, lhs_mode, lhs_custom)
        self._connect_stride_controls(
            rhs_extent, rhs_mode, rhs_custom)
        self._update_stride_controls(
            lhs_extent, lhs_mode, lhs_custom)
        self._update_stride_controls(
            rhs_extent, rhs_mode, rhs_custom)

    def _extent_combo(self, value):
        combo = QtWidgets.QComboBox()
        for name, label in EXTENT_CHOICES:
            combo.addItem(label, name)
            helper = (
                'Use the output batch size for this input.'
                if name == 'same' else
                'Use size 1 and broadcast this input.')
            combo.setItemData(
                combo.count() - 1, helper,
                QtCore.Qt.ItemDataRole.ToolTipRole)
        combo.setCurrentIndex(combo.findData(value))
        combo.setToolTip(
            'Choose the output size or size 1 to broadcast this input.')
        return combo

    def _stride_combo(self, value):
        combo = QtWidgets.QComboBox()
        for name, label in STRIDE_CHOICES:
            combo.addItem(label, name)
            helpers = {
                'auto': 'Move to the next non-overlapping inner input view.',
                'zero': 'Reuse the same data for every index.',
                'custom': 'Use the signed element step entered beside it.',
            }
            combo.setItemData(
                combo.count() - 1, helpers[name],
                QtCore.Qt.ItemDataRole.ToolTipRole)
        combo.setCurrentIndex(combo.findData(value))
        combo.setToolTip(
            'Auto moves to the next matrix; Zero reuses data; Fixed uses '
            'the integer element step beside it.')
        return combo

    @staticmethod
    def _custom_stride(value):
        control = QtWidgets.QSpinBox()
        control.setRange(-1_000_000_000, 1_000_000_000)
        control.setValue(value or 0)
        control.setToolTip('Signed element step for Fixed integer mode.')
        return control

    def _connect_axis_control(self, control):
        signal = (control.valueChanged if isinstance(
            control, QtWidgets.QSpinBox) else control.currentIndexChanged)
        signal.connect(self._axis_changed)

    def _connect_extent_controls(self, extent, mode, custom):
        extent.currentIndexChanged.connect(
            lambda _index: self._extent_changed(extent, mode, custom))

    def _connect_stride_controls(self, extent, mode, custom):
        mode.currentIndexChanged.connect(
            lambda _index: self._stride_mode_changed(
                extent, mode, custom))
        custom.valueChanged.connect(self._axis_changed)

    def _extent_changed(self, extent, mode, custom):
        if extent.currentData() == 'one':
            mode.setCurrentIndex(mode.findData('zero'))
        self._update_stride_controls(extent, mode, custom)
        self._axis_changed()

    def _stride_mode_changed(self, extent, mode, custom):
        self._update_stride_controls(extent, mode, custom)
        self._axis_changed()

    @staticmethod
    def _update_stride_controls(extent, mode, custom):
        broadcast = extent.currentData() == 'one'
        mode.setEnabled(not broadcast)
        custom.setEnabled(
            not broadcast and mode.currentData() == 'custom')
        if broadcast:
            mode.setToolTip(
                'Size 1 broadcasts this input with effective step 0.')
        else:
            mode.setToolTip(
                'Auto moves to the next matrix; Zero reuses data; Fixed '
                'uses the integer element step beside it.')

    @QtCore.Slot()
    def _axis_changed(self):
        if not self._updating:
            self.changed.emit()

    def _axis(self, row):
        lhs_mode = self._table.cellWidget(row, 3).currentData()
        rhs_mode = self._table.cellWidget(row, 6).currentData()
        lhs_value = (self._table.cellWidget(row, 4).value()
                     if lhs_mode == 'custom' else None)
        rhs_value = (self._table.cellWidget(row, 7).value()
                     if rhs_mode == 'custom' else None)
        return _batch_axis(
            self._table.cellWidget(row, 1).value(),
            self._table.cellWidget(row, 2).currentData(),
            lhs_mode,
            self._table.cellWidget(row, 5).currentData(),
            rhs_mode,
            lhs_value, rhs_value,
        )

    def _update_axis_names(self):
        for row in range(self._table.rowCount()):
            self._table.item(row, 0).setText(f'b{row}')

    def _update_remove_button(self):
        self._remove.setEnabled(self._table.currentRow() >= 0)


class InputProfileEditor(QtWidgets.QDialog):
    """Edit one M/K/N sweep recipe or one exact A/B case."""

    def __init__(self, profile=None, preview_shape=(64, 64, 64),
                 routes=DEFAULT_ROUTES, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Edit input case')
        self.setMinimumWidth(960)
        profile = profile or default_input_profiles()[0]
        self._profile_id = profile.profile_id
        self._routes = tuple(routes)
        self._build_ui()
        self._set_preview_shape(preview_shape)
        self._load_profile(profile)
        self._connect_controls()
        self._update_dependent_controls()
        self._update_preview()

    def _build_ui(self):
        self._name = QtWidgets.QLineEdit()
        self._tabs = QtWidgets.QTabWidget()
        self._sweep_page = self._build_sweep_page()
        self._exact_page = self._build_exact_page()
        self._tabs.addTab(self._sweep_page, 'Preset / sweep')
        self._tabs.addTab(self._exact_page, 'Exact case')

        self._preview = QtWidgets.QLabel()
        self._preview.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self._preview.setWordWrap(True)
        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        form = QtWidgets.QFormLayout()
        form.addRow('Case name', self._name)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._tabs)
        layout.addWidget(QtWidgets.QLabel('Resolved input'))
        layout.addWidget(self._preview)
        layout.addWidget(self._buttons)

    def _build_sweep_page(self):
        page = QtWidgets.QWidget()
        helper = QtWidgets.QLabel(
            'This recipe is expanded across every M, K, and N value in the '
            'collection. Presets only fill the editable fields below.')
        helper.setWordWrap(True)

        self._preset = QtWidgets.QComboBox()
        for name, label, description in BATCH_MAPPING_CHOICES:
            self._preset.addItem(label, name)
            self._preset.setItemData(
                self._preset.count() - 1, description,
                QtCore.Qt.ItemDataRole.ToolTipRole)
        self._preset_extent = QtWidgets.QSpinBox()
        self._preset_extent.setRange(2, 1_000_000)
        self._preset_extent.setValue(8)
        self._preset_extent.setToolTip(
            'Batch size filled by a batched preset.')
        self._apply_preset = QtWidgets.QPushButton('Fill fields')
        preset_control = QtWidgets.QWidget()
        preset_layout = QtWidgets.QHBoxLayout(preset_control)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.addWidget(self._preset, 1)
        preset_layout.addWidget(QtWidgets.QLabel('Batch size'))
        preset_layout.addWidget(self._preset_extent)
        preset_layout.addWidget(self._apply_preset)

        self._lhs_storage = self._storage_combo()
        self._lhs_gap = self._gap_control()
        self._lhs_row_stride = self._core_stride_control()
        self._lhs_column_stride = self._core_stride_control()
        self._rhs_storage = self._storage_combo()
        self._rhs_gap = self._gap_control()
        self._rhs_row_stride = self._core_stride_control()
        self._rhs_column_stride = self._core_stride_control()

        form = QtWidgets.QFormLayout()
        form.addRow('Fill from preset', preset_control)
        form.addRow('A matrix storage', self._lhs_storage)
        form.addRow('A padding elements', self._lhs_gap)
        form.addRow('A fixed row stride', self._lhs_row_stride)
        form.addRow('A fixed column stride', self._lhs_column_stride)
        form.addRow('B matrix storage', self._rhs_storage)
        form.addRow('B padding elements', self._rhs_gap)
        form.addRow('B fixed row stride', self._rhs_row_stride)
        form.addRow('B fixed column stride', self._rhs_column_stride)

        self._axes = BatchAxesEditor()

        self._preview_m = self._preview_dimension()
        self._preview_k = self._preview_dimension()
        self._preview_n = self._preview_dimension()
        preview_controls = QtWidgets.QHBoxLayout()
        preview_controls.addWidget(QtWidgets.QLabel('Example from range'))
        for label, control in (
                ('M', self._preview_m), ('K', self._preview_k),
                ('N', self._preview_n)):
            preview_controls.addWidget(QtWidgets.QLabel(label))
            preview_controls.addWidget(control)
        preview_controls.addStretch(1)

        equation = QtWidgets.QLabel('A[..., M, K]  @  B[..., K, N]')
        equation.setToolTip(
            'M, K, and N come from the collection ranges.')
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(equation)
        layout.addWidget(helper)
        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel('Batch shape and reuse'))
        layout.addWidget(self._axes)
        layout.addLayout(preview_controls)
        return page

    def _build_exact_page(self):
        page = QtWidgets.QWidget()
        helper = QtWidgets.QLabel(
            'Enter full shapes and element strides. This mode contributes '
            'exactly one collection cell; the collection M/K/N ranges do '
            'not duplicate it.')
        helper.setWordWrap(True)
        self._lhs_shape = QtWidgets.QLineEdit()
        self._lhs_strides = QtWidgets.QLineEdit()
        self._rhs_shape = QtWidgets.QLineEdit()
        self._rhs_strides = QtWidgets.QLineEdit()
        for control in (
                self._lhs_shape, self._lhs_strides,
                self._rhs_shape, self._rhs_strides):
            control.setPlaceholderText('comma-separated integers')
        self._copy_sweep = QtWidgets.QPushButton(
            'Copy the current sweep example')

        form = QtWidgets.QFormLayout()
        form.addRow('A full shape', self._lhs_shape)
        form.addRow('A element strides', self._lhs_strides)
        form.addRow('B full shape', self._rhs_shape)
        form.addRow('B element strides', self._rhs_strides)
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(QtWidgets.QLabel('A[..., M, K]  @  B[..., K, N]'))
        layout.addWidget(helper)
        layout.addLayout(form)
        layout.addWidget(self._copy_sweep)
        layout.addStretch(1)
        return page

    @staticmethod
    def _storage_combo():
        combo = QtWidgets.QComboBox()
        for name, label, _order, _uses_gap, helper \
                in CORE_STORAGE_CHOICES:
            combo.addItem(label, name)
            combo.setItemData(
                combo.count() - 1, helper,
                QtCore.Qt.ItemDataRole.ToolTipRole)
        return combo

    @staticmethod
    def _gap_control():
        control = QtWidgets.QSpinBox()
        control.setRange(1, 1_000_000)
        control.setValue(1)
        control.setToolTip(
            'Unused elements after each logical row or column selected above.')
        return control

    @staticmethod
    def _core_stride_control():
        control = QtWidgets.QSpinBox()
        control.setRange(-1_000_000_000, 1_000_000_000)
        control.setToolTip(
            'Signed element offset between adjacent matrix rows or columns.')
        return control

    @staticmethod
    def _preview_dimension():
        control = QtWidgets.QSpinBox()
        control.setRange(1, 1_000_000)
        return control

    def _load_profile(self, profile):
        self._name.setText(profile.name)
        if profile.is_exact:
            self._set_exact_operands(profile.exact_lhs, profile.exact_rhs)
            self._tabs.setCurrentWidget(self._exact_page)
            return
        m_value = self._preview_m.value()
        k_value = self._preview_k.value()
        n_value = self._preview_n.value()
        self._set_storage(
            self._lhs_storage, self._lhs_gap,
            self._lhs_row_stride, self._lhs_column_stride,
            profile.lhs_core, m_value, k_value)
        self._set_storage(
            self._rhs_storage, self._rhs_gap,
            self._rhs_row_stride, self._rhs_column_stride,
            profile.rhs_core, k_value, n_value)
        mapping, extent = _basic_mapping(profile)
        self._preset_extent.setValue(extent)
        if mapping is not None:
            self._preset.setCurrentIndex(self._preset.findData(mapping))
        self._axes.set_axes(profile.batch_axes)
        resolved = profile.resolve(m_value, k_value, n_value)
        self._set_exact_operands(resolved.lhs, resolved.rhs)
        self._tabs.setCurrentWidget(self._sweep_page)

    def _set_exact_operands(self, lhs, rhs):
        for control, values in (
                (self._lhs_shape, lhs.shape),
                (self._lhs_strides, lhs.strides),
                (self._rhs_shape, rhs.shape),
                (self._rhs_strides, rhs.strides)):
            control.setText(', '.join(str(value) for value in values))

    @staticmethod
    def _set_storage(combo, gap, row_stride, column_stride,
                     core, rows, columns):
        combo.setCurrentIndex(combo.findData(_core_choice_name(core)))
        gap.setValue(max(1, core.leading_dimension_gap))
        _shape, strides, _span = core.resolve(rows, columns)
        row_stride.setValue(strides[0])
        column_stride.setValue(strides[1])

    def _set_preview_shape(self, shape):
        if len(shape) != 3:
            raise ValueError('preview shape must contain M, K, and N')
        for control, value in zip(
                (self._preview_m, self._preview_k, self._preview_n), shape):
            control.setValue(value)

    def _connect_controls(self):
        self._name.textChanged.connect(self._update_preview)
        for combo in (self._lhs_storage, self._rhs_storage):
            combo.currentIndexChanged.connect(self._selection_changed)
        for control in (
                self._lhs_gap, self._rhs_gap,
                self._lhs_row_stride, self._lhs_column_stride,
                self._rhs_row_stride, self._rhs_column_stride,
                self._preview_m, self._preview_k, self._preview_n):
            control.valueChanged.connect(self._value_changed)
        self._axes.changed.connect(self._update_preview)
        self._preset.currentIndexChanged.connect(
            self._preset_changed)
        self._apply_preset.clicked.connect(self._fill_from_preset)
        self._copy_sweep.clicked.connect(self._copy_sweep_example)
        self._tabs.currentChanged.connect(self._update_preview)
        for control in (
                self._lhs_shape, self._lhs_strides,
                self._rhs_shape, self._rhs_strides):
            control.textChanged.connect(self._update_preview)
        self._preset_changed()

    @QtCore.Slot()
    def _selection_changed(self):
        self._update_dependent_controls()
        self._update_preview()

    @QtCore.Slot()
    def _value_changed(self):
        self._update_preview()

    @QtCore.Slot()
    def _preset_changed(self):
        batched = self._preset.currentData() != 'unbatched'
        self._preset_extent.setEnabled(batched)
        self._preset.setToolTip(
            self._preset.currentData(
                QtCore.Qt.ItemDataRole.ToolTipRole) or '')

    @QtCore.Slot()
    def _fill_from_preset(self):
        m_value = self._preview_m.value()
        k_value = self._preview_k.value()
        n_value = self._preview_n.value()
        compact = _profiles.CoreStorage.c_compact()
        self._set_storage(
            self._lhs_storage, self._lhs_gap,
            self._lhs_row_stride, self._lhs_column_stride,
            compact, m_value, k_value)
        self._set_storage(
            self._rhs_storage, self._rhs_gap,
            self._rhs_row_stride, self._rhs_column_stride,
            compact, k_value, n_value)
        self._axes.set_axes(_basic_batch_axes(
            self._preset.currentData(), self._preset_extent.value()))
        self._update_dependent_controls()
        self._update_preview()

    @QtCore.Slot()
    def _copy_sweep_example(self):
        try:
            resolved = self._sweep_profile().resolve(
                self._preview_m.value(), self._preview_k.value(),
                self._preview_n.value())
        except (TypeError, ValueError):
            return
        self._set_exact_operands(resolved.lhs, resolved.rhs)

    def _update_dependent_controls(self):
        lhs_choice = _core_choice(self._lhs_storage.currentData())
        rhs_choice = _core_choice(self._rhs_storage.currentData())
        lhs_gap = lhs_choice[3]
        rhs_gap = rhs_choice[3]
        self._lhs_gap.setEnabled(lhs_gap)
        self._rhs_gap.setEnabled(rhs_gap)
        lhs_custom = lhs_choice[2] == 'custom'
        rhs_custom = rhs_choice[2] == 'custom'
        self._lhs_row_stride.setEnabled(lhs_custom)
        self._lhs_column_stride.setEnabled(lhs_custom)
        self._rhs_row_stride.setEnabled(rhs_custom)
        self._rhs_column_stride.setEnabled(rhs_custom)
        self._lhs_storage.setToolTip(lhs_choice[4])
        self._rhs_storage.setToolTip(rhs_choice[4])

    def profile(self):
        if self._tabs.currentWidget() is self._exact_page:
            return self._exact_profile()
        return self._sweep_profile()

    def _profile_name(self):
        name = self._name.text().strip()
        if not name:
            raise ValueError('Case name must not be empty')
        return name

    def _sweep_profile(self):
        return _profiles.InputProfile(
            profile_id=self._profile_id,
            name=self._profile_name(),
            lhs_core=_core_storage(
                self._lhs_storage.currentData(), self._lhs_gap.value(),
                self._lhs_row_stride.value(),
                self._lhs_column_stride.value()),
            rhs_core=_core_storage(
                self._rhs_storage.currentData(), self._rhs_gap.value(),
                self._rhs_row_stride.value(),
                self._rhs_column_stride.value()),
            batch_axes=self._axes.axes(),
        )

    def _exact_profile(self):
        lhs = _schema.OperandSpec(
            shape=_integer_tuple(self._lhs_shape.text(), 'A shape'),
            strides=_integer_tuple(
                self._lhs_strides.text(), 'A strides'))
        rhs = _schema.OperandSpec(
            shape=_integer_tuple(self._rhs_shape.text(), 'B shape'),
            strides=_integer_tuple(
                self._rhs_strides.text(), 'B strides'))
        return _profiles.InputProfile.exact(
            profile_id=self._profile_id, name=self._profile_name(),
            lhs=lhs, rhs=rhs)

    @QtCore.Slot()
    def _update_preview(self):
        try:
            profile = self.profile()
            text = _resolved_text(
                profile,
                self._preview_m.value(), self._preview_k.value(),
                self._preview_n.value(), self._routes)
        except (TypeError, ValueError) as exc:
            self._preview.setText(f'Invalid input case: {exc}')
            valid = False
        else:
            mode = (
                'Exact case: one collection cell; M/K/N ranges are ignored.'
                if profile.is_exact else
                'Sweep: repeated for every collection M/K/N combination.')
            self._preview.setText(f'{mode}\n{text}')
            valid = True
        self._buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(valid)


class InputProfilesWidget(QtWidgets.QWidget):
    """Manage complete input cases and show one resolved example."""

    profiles_changed = QtCore.Signal()

    HEADERS = (
        'Input case', 'Mode', 'A input', 'B input', 'Batch / reuse',
        'Resolved example', 'Available kernels',
    )

    def __init__(self, profiles=None, preview_shape=(64, 64, 64),
                 routes=DEFAULT_ROUTES, editor_factory=None, parent=None):
        super().__init__(parent)
        if profiles is None:
            profiles = default_input_profiles()
        self._profiles = list(profiles)
        if not self._profiles:
            raise ValueError('At least one input case is required')
        self._preview_shape = tuple(preview_shape)
        self._routes = tuple(routes)
        self._editor_factory = editor_factory or InputProfileEditor
        self._build_ui()
        self._refresh_table()

    def _build_ui(self):
        equation = QtWidgets.QLabel('A[..., M, K]  @  B[..., K, N]')
        equation.setToolTip(
            'Each row below is one complete shape and element-stride recipe.')
        self._table = QtWidgets.QTableWidget(0, len(self.HEADERS))
        self._table.setHorizontalHeaderLabels(self.HEADERS)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._add = QtWidgets.QPushButton('Add input case...')
        self._edit = QtWidgets.QPushButton('Edit...')
        self._duplicate = QtWidgets.QPushButton('Duplicate')
        self._remove = QtWidgets.QPushButton('Remove')
        for control, helper in (
                (self._add, 'Create one complete A/B input recipe.'),
                (self._edit, 'Edit the selected complete input recipe.'),
                (self._duplicate, 'Copy the selected recipe for variation.'),
                (self._remove, 'Remove the selected recipe.')):
            control.setToolTip(helper)
        buttons = QtWidgets.QHBoxLayout()
        for control in (
                self._add, self._edit, self._duplicate, self._remove):
            buttons.addWidget(control)
        buttons.addStretch(1)

        self._preview = QtWidgets.QLabel()
        self._preview.setWordWrap(True)
        self._preview.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(equation)
        layout.addWidget(self._table)
        layout.addLayout(buttons)
        layout.addWidget(self._preview)

        self._table.itemSelectionChanged.connect(
            self._selection_changed)
        self._table.itemDoubleClicked.connect(
            lambda _item, _column: self.edit_selected())
        self._add.clicked.connect(self.add_with_editor)
        self._edit.clicked.connect(self.edit_selected)
        self._duplicate.clicked.connect(self.duplicate_selected)
        self._remove.clicked.connect(self.remove_selected)

    def profiles(self):
        return tuple(self._profiles)

    def set_context(self, preview_shape=None, routes=None):
        """Update the resolved example without changing any profile."""
        if preview_shape is not None:
            preview_shape = tuple(preview_shape)
            if len(preview_shape) != 3 or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    or value < 1 for value in preview_shape):
                raise ValueError(
                    'Input case preview must contain positive M, K, N')
            self._preview_shape = preview_shape
        if routes is not None:
            routes = tuple(routes)
            if any(not isinstance(route, str) or not route
                   for route in routes):
                raise ValueError(
                    'Input case preview dispatches must be names')
            self._routes = routes
        self._refresh_table(self._table.currentRow())

    def set_profiles(self, profiles):
        profiles = list(profiles)
        if not profiles:
            raise ValueError('At least one input case is required')
        self._profiles = profiles
        self._refresh_table()
        self.profiles_changed.emit()

    def add_profile(self, profile):
        self._profiles.append(self._with_unique_name(profile))
        self._refresh_table(len(self._profiles) - 1)
        self.profiles_changed.emit()

    def replace_profile(self, row, profile):
        if not 0 <= row < len(self._profiles):
            raise IndexError('Input case row is out of range')
        self._profiles[row] = self._with_unique_name(profile, exclude=row)
        self._refresh_table(row)
        self.profiles_changed.emit()

    @QtCore.Slot()
    def add_with_editor(self):
        editor = self._editor_factory(
            profile=None, preview_shape=self._preview_shape,
            routes=self._routes, parent=self)
        if editor.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.add_profile(editor.profile())

    @QtCore.Slot()
    def edit_selected(self):
        row = self._table.currentRow()
        if row < 0:
            return
        editor = self._editor_factory(
            profile=self._profiles[row], preview_shape=self._preview_shape,
            routes=self._routes, parent=self)
        if editor.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.replace_profile(row, editor.profile())

    @QtCore.Slot()
    def duplicate_selected(self):
        row = self._table.currentRow()
        if row < 0:
            return
        source = self._profiles[row]
        duplicate = dataclasses.replace(
            source, profile_id=_profile_id(),
            name=f'{source.name} copy')
        self.add_profile(duplicate)

    @QtCore.Slot()
    def remove_selected(self):
        row = self._table.currentRow()
        if row < 0 or len(self._profiles) == 1:
            return
        del self._profiles[row]
        self._refresh_table(min(row, len(self._profiles) - 1))
        self.profiles_changed.emit()

    def _with_unique_name(self, profile, exclude=None):
        names = {
            item.name for index, item in enumerate(self._profiles)
            if index != exclude
        }
        name = profile.name
        suffix = 2
        while name in names:
            name = f'{profile.name} {suffix}'
            suffix += 1
        if name == profile.name:
            return profile
        return dataclasses.replace(profile, name=name)

    def _refresh_table(self, selected=0):
        self._table.setRowCount(len(self._profiles))
        m_value, k_value, n_value = self._preview_shape
        for row, profile in enumerate(self._profiles):
            resolved = profile.resolve(m_value, k_value, n_value)
            routes = _eligible_routes(
                profile, m_value, k_value, n_value, self._routes)
            if profile.is_exact:
                mode = 'Exact case (one cell)'
                lhs = f'{resolved.lhs.shape}:{resolved.lhs.strides}'
                rhs = f'{resolved.rhs.shape}:{resolved.rhs.strides}'
                batch = _batch_behavior_text(resolved.broadcast)
            else:
                mode = 'Preset / sweep'
                lhs = _storage_text(profile.lhs_core)
                rhs = _storage_text(profile.rhs_core)
                batch = _batch_text(profile)
            values = (
                profile.name,
                mode,
                lhs,
                rhs,
                batch,
                f'M={resolved.m}, K={resolved.k}, N={resolved.n}; '
                f'output {resolved.output_shape}',
                _route_text(routes) or 'none',
            )
            tooltip = _resolved_text(
                profile, m_value, k_value, n_value, self._routes)
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(tooltip)
                self._table.setItem(row, column, item)
        if self._profiles:
            self._table.selectRow(max(0, min(selected,
                                             len(self._profiles) - 1)))
        self._selection_changed()

    @QtCore.Slot()
    def _selection_changed(self):
        row = self._table.currentRow()
        selected = 0 <= row < len(self._profiles)
        self._edit.setEnabled(selected)
        self._duplicate.setEnabled(selected)
        self._remove.setEnabled(selected and len(self._profiles) > 1)
        if not selected:
            self._preview.setText('Select an input case to inspect it.')
            return
        m_value, k_value, n_value = self._preview_shape
        profile = self._profiles[row]
        mode = (
            'Exact case: contributes one collection cell.'
            if profile.is_exact else
            'Sweep: repeated for every collection M/K/N combination.')
        self._preview.setText(
            mode + '\n' + _resolved_text(
                profile, m_value, k_value, n_value, self._routes))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
