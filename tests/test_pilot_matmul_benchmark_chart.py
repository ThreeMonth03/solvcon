# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Test completed-artifact timing charts in the Pilot explorer."""

import unittest

import solvcon

try:
    from PySide6 import QtCore, QtGui, QtTest, QtWidgets
    from solvcon.pilot.panel import _matmul_benchmark
    from solvcon.pilot.panel import _matmul_benchmark_chart
except ImportError:
    QtCore = QtGui = QtTest = QtWidgets = None
    _matmul_benchmark = _matmul_benchmark_chart = None


def _summary(median, p95, deviation):
    return {
        'median_ns': median,
        'p95_ns': p95,
        'mad_ns': deviation,
    }


def _candidates():
    return [
        {
            'name': 'auto',
            'kind': 'solvcon_auto',
            'selected_route': 'naive',
            'packing': {},
            'correctness': {'correct': True},
            'timing': _summary(420.0, 460.0, 8.4),
            'python_timing': _summary(660.0, 700.0, 19.8),
        },
        {
            'name': 'naive',
            'kind': 'solvcon_route',
            'selected_by_auto': True,
            'packing': {
                'eager_lhs': True,
                'eager_rhs': False,
                'scratch_lhs': False,
                'scratch_rhs': True,
            },
            'correctness': {'correct': True},
            'timing': _summary(400.0, 430.0, 12.0),
            'python_timing': _summary(620.0, 680.0, 31.0),
        },
        {
            'name': 'blas_gemm',
            'kind': 'solvcon_route',
            'packing': {},
            'correctness': {'correct': True},
            'timing': _summary(300.0, 350.0, 15.0),
            'python_timing': _summary(500.0, 560.0, 20.0),
        },
        {
            'name': 'broken',
            'kind': 'solvcon_route',
            'eligible': False,
            'packing': {},
            'correctness': {
                'correct': False,
                'reason': 'values differ',
            },
            'timing': _summary(250.0, 280.0, 10.0),
            'python_timing': None,
        },
        {
            'name': 'numpy',
            'kind': 'numpy',
            'packing': None,
            'correctness': {'correct': True},
            'timing': None,
            'python_timing': _summary(400.0, 420.0, 8.0),
        },
    ]


def _artifact():
    candidates = _candidates()
    return {
        'observations': [{
            'auto_route': 'naive',
            'winner': 'blas_gemm',
            'routes': {
                candidate['name']: candidate for candidate in candidates
            },
        }],
        'panels': [],
    }


@unittest.skipUnless(solvcon.HAS_PILOT, 'Qt pilot is not built')
class RouteTimingChartTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def setUp(self):
        self.chart = _matmul_benchmark_chart.RouteTimingChart()
        self.chart.set_candidates(
            _candidates(), 'naive', 'blas_gemm')

    def test_scope_switches_measurements_and_only_e2e_shows_numpy(self):
        native = self.chart._canvas.entries()

        self.assertNotIn('numpy', [entry['name'] for entry in native])
        self.assertEqual(native[1]['median'], 400.0)
        self.assertEqual(native[1]['p95'], 430.0)
        self.assertTrue(native[1]['auto'])
        self.assertTrue(native[2]['winner'])

        self.chart._scope.setCurrentIndex(1)
        end_to_end = self.chart._canvas.entries()

        self.assertIn('numpy', [entry['name'] for entry in end_to_end])
        self.assertEqual(end_to_end[1]['median'], 620.0)
        self.assertEqual(end_to_end[1]['p95'], 680.0)

    def test_scope_options_explain_what_each_measurement_includes(self):
        role = QtCore.Qt.ItemDataRole.ToolTipRole
        native_help = self.chart._scope.itemData(0, role)
        end_to_end_help = self.chart._scope.itemData(1, role)

        self.assertIn('Python call overhead is left out', native_help)
        self.assertIn('NumPy is not shown', native_help)
        self.assertIn('complete call from Python', end_to_end_help)
        self.assertIn('NumPy is included', end_to_end_help)
        self.assertEqual(self.chart._scope.toolTip(), native_help)
        self.assertEqual(
            self.chart._scope.accessibleDescription(), native_help)
        self.assertEqual(self.chart._scope_help.text(), native_help)

        self.chart._scope.setCurrentIndex(1)
        self.assertEqual(self.chart._scope_help.text(), end_to_end_help)
        self.assertEqual(self.chart._scope.toolTip(), end_to_end_help)

    def test_tooltip_reports_measurement_and_route_context(self):
        tooltip = self.chart._canvas.tooltip_for_row(1)

        self.assertIn('Median: 400.0 ns', tooltip)
        self.assertIn('P95: 430.0 ns', tooltip)
        self.assertIn('Noise (MAD / median): 3.000%', tooltip)
        self.assertIn('Speedup vs auto: 1.0500x', tooltip)
        self.assertIn('Eager packing: lhs', tooltip)
        self.assertIn('Scratch packing: rhs', tooltip)
        self.assertIn('Correctness: pass', tooltip)

        self.chart._scope.setCurrentIndex(1)
        tooltip = self.chart._canvas.tooltip_for_row(1)

        self.assertIn('Scope: Python end-to-end', tooltip)
        self.assertIn('Speedup vs NumPy: 0.6452x', tooltip)

    def test_incorrect_or_unmeasured_routes_are_gray(self):
        native = self.chart._canvas.entries()
        broken = next(entry for entry in native
                      if entry['name'] == 'broken')

        self.assertFalse(broken['available'])
        self.assertEqual(
            self.chart._canvas._bar_color(broken),
            QtGui.QColor('#8a8a8a'))

        self.chart._scope.setCurrentIndex(1)
        tooltip = self.chart._canvas.tooltip_for_row(3)

        self.assertIn('Median: unavailable', tooltip)
        self.assertIn('Correctness: FAIL: values differ', tooltip)

    def test_chart_click_and_table_selection_stay_synchronized(self):
        inspector = _matmul_benchmark.RouteInspectorWidget()
        inspector.show_artifact(_artifact())
        canvas = inspector._chart._canvas
        canvas.resize(800, 240)
        image = QtGui.QImage(
            canvas.size(), QtGui.QImage.Format.Format_ARGB32)
        canvas.render(image)
        hit_rect = next(
            rect for rect, entry in canvas._hit_regions
            if entry['row'] == 2)

        QtTest.QTest.mouseClick(
            canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=hit_rect.center().toPoint())

        self.assertEqual(inspector._table.currentRow(), 2)
        self.assertEqual(inspector._chart._canvas._current_row, 2)

        inspector._table.setCurrentCell(1, 0)

        self.assertEqual(inspector._chart._canvas._current_row, 1)
        inspector.close()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
