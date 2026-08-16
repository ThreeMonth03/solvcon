# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Exercise the registered matmul benchmark MDI window."""

import os
import unittest

import solvcon

try:
    from PySide6 import QtCore, QtTest, QtWidgets
    from solvcon.pilot.base import _gui
except ImportError:
    QtCore = QtTest = QtWidgets = None
    _gui = None


NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 'pilot windows need a real window surface')
class MatmulBenchmarkWindowTC(unittest.TestCase):
    def setUp(self):
        self.manager = _gui.controller.build()
        self.manager.show()
        self.manager.mdiArea.closeAllSubWindows()
        QtWidgets.QApplication.processEvents()

    def tearDown(self):
        self.manager.mdiArea.closeAllSubWindows()
        QtWidgets.QApplication.processEvents()

    def _open_visualizer(self):
        action = self.manager.menu_model.action(
            'profiling.matmul_benchmark')
        action.trigger()
        QtWidgets.QApplication.processEvents()
        return _gui.controller.matmul_benchmark.window

    def _wait_until(self, predicate, timeout_ms):
        timer = QtCore.QElapsedTimer()
        timer.start()
        while not predicate() and timer.elapsed() < timeout_ms:
            QtWidgets.QApplication.processEvents()
            QtTest.QTest.qWait(10)
        return predicate()

    def test_registered_feature_opens_both_views_and_closes_cleanly(self):
        action = self.manager.menu_model.action(
            'profiling.matmul_benchmark')
        self.assertIsNotNone(action)
        self.assertEqual(action.text(), 'Benchmark visualizer')

        action.trigger()
        QtWidgets.QApplication.processEvents()

        subwindow = self.manager.mdiArea.activeSubWindow()
        feature = _gui.controller.matmul_benchmark
        self.assertEqual(feature.TITLE, 'Benchmark Visualizer')
        self.assertEqual(subwindow.windowTitle(), feature.TITLE)
        self.assertEqual(feature.window._tabs.count(), 2)
        self.assertEqual(feature.window._tabs.tabText(0), 'Route Inspector')
        self.assertEqual(feature.window._tabs.tabText(1), 'Dispatch Atlas')

        subwindow.close()
        QtWidgets.QApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete)
        QtWidgets.QApplication.processEvents()
        self.assertIsNone(self.manager.mdiArea.activeSubWindow())

    def test_global_stop_kills_a_native_call_from_the_other_page(self):
        window = self._open_visualizer()
        inspector = window.route_inspector
        inspector._lhs_shape.setText('2048, 2048')
        inspector._rhs_shape.setText('2048, 2048')
        inspector._threads.setValue(1)
        for name, box in inspector._dispatches.boxes.items():
            box.setChecked(name == 'generic')
        sampling = inspector._sampling
        sampling.quality.setCurrentText('Custom fixed schedule')
        sampling.warmups.setValue(0)
        sampling.repetitions.setValue(1)
        sampling.rounds.setValue(1)
        activities = []
        inspector._runner.activity.connect(activities.append)

        inspector.start_benchmark()
        self.assertTrue(self._wait_until(
            lambda: any(
                event['route'] == 'generic' and event['state'] == 'started'
                for event in activities),
            30_000))
        window._tabs.setCurrentWidget(window.dispatch_atlas)

        self.assertEqual(window._stop.text(), 'Stop Route Inspector')
        QtTest.QTest.mouseClick(
            window._stop, QtCore.Qt.MouseButton.LeftButton)

        self.assertTrue(self._wait_until(
            lambda: not inspector.running, 5_000))
        self.assertIn('Cancelled while Generic',
                      inspector._status.text())
        self.assertFalse(window._stop.isEnabled())
        self.assertTrue(inspector._run.isEnabled())
        self.assertTrue(window.dispatch_atlas._collect.isEnabled())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
