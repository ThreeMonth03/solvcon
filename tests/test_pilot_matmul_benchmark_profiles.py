# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Test explicit matmul input-profile controls without mapping a window."""

import unittest

import solvcon
from solvcon.matmul_benchmark import profiles
from solvcon.matmul_benchmark import schema

try:
    from PySide6 import QtCore, QtWidgets
    from solvcon.pilot.panel import _matmul_benchmark_profiles as _profile_ui
except ImportError:
    QtCore = QtWidgets = None
    _profile_ui = None


def _set_combo(combo, value):
    index = combo.findData(value)
    if index < 0:
        raise AssertionError(f'combo has no value {value!r}')
    combo.setCurrentIndex(index)


@unittest.skipUnless(solvcon.HAS_PILOT, 'Qt pilot is not built')
class InputProfileEditorTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def test_sweep_preset_fills_editable_storage_and_batch_fields(self):
        editor = _profile_ui.InputProfileEditor(
            preview_shape=(8, 16, 32))
        editor._name.setText('Reuse gapped A')
        _set_combo(editor._preset, 'reuse_lhs')
        editor._preset_extent.setValue(4)
        editor._apply_preset.click()
        _set_combo(editor._lhs_storage, 'c_gap')
        editor._lhs_gap.setValue(3)
        _set_combo(editor._rhs_storage, 'f_gap')
        editor._rhs_gap.setValue(5)

        profile = editor.profile()
        resolved = profile.resolve(8, 16, 32)

        self.assertEqual((1, 8, 16), resolved.lhs.shape)
        self.assertEqual((0, 19, 1), resolved.lhs.strides)
        self.assertEqual((4, 16, 32), resolved.rhs.shape)
        self.assertEqual((667, 1, 21), resolved.rhs.strides)
        self.assertEqual((4, 8, 32), resolved.output_shape)
        self.assertIn(
            'A shape (1, 8, 16), element strides (0, 19, 1), '
            'backing storage 149 elements',
            editor._preview.text())
        self.assertIn('Kernels available for this input: Naive, BLAS GEMM',
                      editor._preview.text())
        self.assertIn('Sweep: repeated for every collection M/K/N',
                      editor._preview.text())
        self.assertNotIn('Winograd', editor._preview.text())

    def test_batch_table_resolves_inside_out_and_rejects_invalid_rows(self):
        editor = _profile_ui.InputProfileEditor(
            preview_shape=(8, 16, 32))
        editor._axes.set_axes((
            profiles.BatchAxis(
                output_extent=2,
                lhs_extent='same',
                rhs_extent='one',
                lhs_stride=profiles.BatchStride.custom(1000),
                rhs_stride=profiles.BatchStride.zero(),
            ),
            profiles.BatchAxis(
                output_extent=3,
                lhs_extent='one',
                rhs_extent='same',
                lhs_stride=profiles.BatchStride.zero(),
                rhs_stride=profiles.BatchStride.auto(),
            ),
        ))

        resolved = editor.profile().resolve(8, 16, 32)

        self.assertEqual((2, 1, 8, 16), resolved.lhs.shape)
        self.assertEqual((1000, 0, 16, 1), resolved.lhs.strides)
        self.assertEqual((1, 3, 16, 32), resolved.rhs.shape)
        self.assertEqual((0, 512, 32, 1), resolved.rhs.strides)
        self.assertEqual((2, 3, 8, 32), resolved.output_shape)
        ok = editor._buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.assertTrue(ok.isEnabled())

        _set_combo(editor._axes._table.cellWidget(0, 2), 'one')

        self.assertFalse(ok.isEnabled())
        self.assertIn('at least one operand extent same',
                      editor._preview.text())

    def test_custom_core_strides_are_signed_and_exact(self):
        editor = _profile_ui.InputProfileEditor(
            preview_shape=(3, 5, 4))
        _set_combo(editor._lhs_storage, 'custom')
        editor._lhs_row_stride.setValue(11)
        editor._lhs_column_stride.setValue(-2)
        _set_combo(editor._rhs_storage, 'custom')
        editor._rhs_row_stride.setValue(0)
        editor._rhs_column_stride.setValue(3)

        resolved = editor.profile().resolve(3, 5, 4)

        self.assertEqual((11, -2), resolved.lhs.strides)
        self.assertEqual(31, resolved.lhs_storage_span)
        self.assertEqual((0, 3), resolved.rhs.strides)
        self.assertEqual(10, resolved.rhs_storage_span)
        self.assertIn('backing storage 31 elements', editor._preview.text())
        self.assertIn('backing storage 10 elements', editor._preview.text())

    def test_extent_one_disables_and_normalizes_stride(self):
        editor = _profile_ui.InputProfileEditor()
        editor._axes.set_axes((profiles.BatchAxis(output_extent=4),))
        extent = editor._axes._table.cellWidget(0, 2)
        mode = editor._axes._table.cellWidget(0, 3)
        custom = editor._axes._table.cellWidget(0, 4)

        _set_combo(mode, 'custom')
        custom.setValue(-123)
        _set_combo(extent, 'one')

        self.assertFalse(mode.isEnabled())
        self.assertFalse(custom.isEnabled())
        self.assertEqual('zero', mode.currentData())
        self.assertEqual(
            profiles.BatchStride.zero(),
            editor.profile().batch_axes[0].lhs_stride,
        )
        self.assertIn('effective step 0', mode.toolTip())

    def test_controls_explain_storage_preset_and_batch_step_choices(self):
        editor = _profile_ui.InputProfileEditor()
        tooltip_role = QtCore.Qt.ItemDataRole.ToolTipRole

        self.assertTrue(all(
            editor._lhs_storage.itemData(index, tooltip_role)
            for index in range(editor._lhs_storage.count())))
        self.assertTrue(all(
            editor._preset.itemData(index, tooltip_role)
            for index in range(editor._preset.count())))
        editor._axes.add_axis()
        extent = editor._axes._table.cellWidget(0, 2)
        stride = editor._axes._table.cellWidget(0, 3)
        self.assertTrue(all(
            extent.itemData(index, tooltip_role)
            for index in range(extent.count())))
        self.assertTrue(all(
            stride.itemData(index, tooltip_role)
            for index in range(stride.count())))

    def test_exact_case_derives_one_cell_and_blocks_invalid_input(self):
        editor = _profile_ui.InputProfileEditor(
            preview_shape=(8, 16, 32))
        editor._tabs.setCurrentWidget(editor._exact_page)
        editor._lhs_shape.setText('2, 1, 3, 5')
        editor._lhs_strides.setText('33, 0, 11, -2')
        editor._rhs_shape.setText('1, 4, 5, 7')
        editor._rhs_strides.setText('0, 35, 7, 1')

        profile = editor.profile()
        resolved = profile.resolve(8, 16, 32)

        self.assertTrue(profile.is_exact)
        self.assertEqual((2, 4, 3, 7), resolved.output_shape)
        self.assertEqual('broadcast_both', resolved.broadcast)
        self.assertIn('Exact case: one collection cell',
                      editor._preview.text())
        self.assertIn('Derived M=3, K=5, N=7', editor._preview.text())
        self.assertIn('Batch behavior: A and B each reuse data',
                      editor._preview.text())
        self.assertNotIn('broadcast_both', editor._preview.text())
        self.assertIn('BLAS GEMM: A eager pack', editor._preview.text())
        ok = editor._buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.assertTrue(ok.isEnabled())

        editor._rhs_shape.setText('1, 4, 6, 7')

        self.assertFalse(ok.isEnabled())
        self.assertIn('dimensions K do not match', editor._preview.text())

        editor._rhs_shape.setText('1, 4, 5, 7')
        editor._lhs_strides.setText('33, 0, 11')

        self.assertFalse(ok.isEnabled())
        self.assertIn('same length', editor._preview.text())

    def test_exact_profile_reopens_without_becoming_a_sweep(self):
        exact = profiles.InputProfile.exact(
            profile_id='exact', name='Exact',
            lhs=schema.OperandSpec(
                shape=(3, 5), strides=(11, -2)),
            rhs=schema.OperandSpec(
                shape=(5, 7), strides=(7, 1)))

        editor = _profile_ui.InputProfileEditor(
            profile=exact, preview_shape=(8, 16, 32))

        self.assertIs(editor._tabs.currentWidget(), editor._exact_page)
        self.assertEqual(exact, editor.profile())


@unittest.skipUnless(solvcon.HAS_PILOT, 'Qt pilot is not built')
class InputProfilesWidgetTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def setUp(self):
        self.widget = _profile_ui.InputProfilesWidget(profiles=(
            profiles.unbatched_profile(
                profile_id='compact', name='Compact'),
            profiles.reuse_lhs_profile(
                3, profile_id='reuse-a', name='Reuse A'),
        ), preview_shape=(8, 16, 32))

    def test_table_shows_complete_recipes_and_exact_resolved_example(self):
        self.assertEqual(2, self.widget._table.rowCount())
        self.assertEqual(
            _profile_ui.InputProfilesWidget.HEADERS,
            tuple(
                self.widget._table.horizontalHeaderItem(index).text()
                for index in range(self.widget._table.columnCount())))
        resolved_item = self.widget._table.item(1, 5)
        self.assertIn('M=8, K=16, N=32', resolved_item.text())
        self.assertIn('Output shape (3, 8, 32)',
                      resolved_item.toolTip())
        self.assertIn('Naive, BLAS GEMM',
                      self.widget._table.item(1, 6).text())
        self.assertEqual('Preset / sweep',
                         self.widget._table.item(1, 1).text())

    def test_duplicate_remove_and_replace_preserve_complete_profiles(self):
        self.widget._table.selectRow(0)
        self.widget.duplicate_selected()

        duplicated = self.widget.profiles()[-1]
        self.assertEqual('Compact copy', duplicated.name)
        self.assertNotEqual('compact', duplicated.profile_id)
        self.assertEqual(3, self.widget._table.rowCount())

        replacement = profiles.InputProfile(
            profile_id=duplicated.profile_id,
            name='Edited profile',
            lhs_core=profiles.CoreStorage.f_gap(2),
            rhs_core=profiles.CoreStorage.c_gap(4),
            batch_axes=duplicated.batch_axes,
        )
        self.widget.replace_profile(2, replacement)
        self.assertEqual('Edited profile', self.widget.profiles()[2].name)
        self.assertEqual(
            'columns padded by 2', self.widget._table.item(2, 2).text())
        self.assertEqual(
            'rows padded by 4', self.widget._table.item(2, 3).text())

        while len(self.widget.profiles()) > 1:
            self.widget._table.selectRow(0)
            self.widget.remove_selected()
        self.widget._table.selectRow(0)
        self.widget.remove_selected()
        self.assertEqual(1, len(self.widget.profiles()))
        self.assertFalse(self.widget._remove.isEnabled())

    def test_add_and_edit_actions_use_complete_profile_dialog_results(self):
        created = profiles.unbatched_profile(
            profile_id='created', name='Created')
        original = self.widget.profiles()[0]
        edited = profiles.InputProfile(
            profile_id=original.profile_id,
            name='Edited compact',
            lhs_core=profiles.CoreStorage.c_gap(2),
            rhs_core=original.rhs_core,
            batch_axes=original.batch_axes,
        )
        results = [created, edited]
        opened_profiles = []

        class AcceptedEditor:
            def __init__(self, profile, **_kwargs):
                opened_profiles.append(profile)
                self._result = results.pop(0)

            @staticmethod
            def exec():
                return QtWidgets.QDialog.DialogCode.Accepted

            def profile(self):
                return self._result

        self.widget._editor_factory = AcceptedEditor
        self.widget.add_with_editor()
        self.widget._table.selectRow(0)
        self.widget.edit_selected()

        self.assertIsNone(opened_profiles[0])
        self.assertIs(original, opened_profiles[1])
        self.assertEqual('Edited compact', self.widget.profiles()[0].name)
        self.assertEqual('Created', self.widget.profiles()[-1].name)
        self.assertEqual(
            'rows padded by 2', self.widget._table.item(0, 2).text())

    def test_exact_row_and_duplicate_keep_one_cell_recipe(self):
        exact = profiles.InputProfile.exact(
            profile_id='exact', name='Exact',
            lhs=schema.OperandSpec(
                shape=(1, 3, 5), strides=(0, 11, -2)),
            rhs=schema.OperandSpec(
                shape=(4, 5, 7), strides=(35, 7, 1)))
        self.widget.add_profile(exact)

        row = self.widget._table.rowCount() - 1
        self.assertEqual('Exact case (one cell)',
                         self.widget._table.item(row, 1).text())
        self.assertEqual('(1, 3, 5):(0, 11, -2)',
                         self.widget._table.item(row, 2).text())
        self.assertEqual(
            'A is reused across one or more batch axes',
            self.widget._table.item(row, 4).text())
        self.assertIn('output (4, 3, 7)',
                      self.widget._table.item(row, 5).text())

        self.widget._table.selectRow(row)
        self.widget.duplicate_selected()

        duplicate = self.widget.profiles()[-1]
        self.assertTrue(duplicate.is_exact)
        self.assertEqual(exact.exact_lhs, duplicate.exact_lhs)
        self.assertNotEqual(exact.profile_id, duplicate.profile_id)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
