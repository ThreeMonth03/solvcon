# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Route timing comparison chart for completed matmul artifacts."""

import math

from PySide6 import QtCore, QtGui, QtWidgets

from . import _matmul_benchmark_cloud as _cloud
from . import _matmul_benchmark_process as _process

__all__ = ['RouteTimingChart']


class _RouteTimingCanvas(QtWidgets.QWidget):
    """Paint route latency bars without collecting new measurements."""

    row_selected = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._candidates = []
        self._selected_name = ''
        self._winner_name = ''
        self._current_row = -1
        self._hit_regions = []
        self.setMouseTracking(True)
        self.setMinimumSize(420, 210)

    def set_candidates(self, candidates, selected_name, winner_name):
        self._candidates = list(candidates)
        self._selected_name = selected_name
        self._winner_name = winner_name
        self._current_row = -1
        self.update()

    def set_current_row(self, row):
        if self._current_row == row:
            return
        self._current_row = row
        self.update()

    def entries(self):
        entries = []
        for row, candidate in enumerate(self._candidates):
            name = _process._candidate_name(candidate)
            summary = self._timing_summary(candidate)
            median = self._number(summary.get('median_ns'))
            p95 = self._number(summary.get('p95_ns'))
            noise = self._summary_noise(summary, median)
            correctness = _process._correctness_text(candidate)
            eligible = candidate.get('eligible', True)
            correct = self._correctness_passed(candidate)
            available = (eligible and correct is not False
                         and median is not None)
            entries.append({
                'row': row,
                'candidate': candidate,
                'name': name,
                'median': median,
                'p95': p95,
                'noise': noise,
                'eligible': eligible,
                'correctness': correctness,
                'available': available,
                'auto': (
                    name == self._selected_name
                    or candidate.get('selected_by_auto') is True),
                'winner': name == self._winner_name,
            })
        baseline = self._baseline(entries)
        for entry in entries:
            median = entry['median']
            entry['speedup'] = (
                baseline['median'] / median
                if baseline is not None and median is not None and median > 0
                else None)
            entry['speedup_baseline'] = (
                baseline['name'] if baseline is not None else '')
        return entries

    def paintEvent(self, _event):
        self._hit_regions = []
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        entries = self.entries()
        if not entries:
            painter.setPen(self.palette().text().color())
            painter.drawText(
                self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                'Run a benchmark to compare route timings')
            return

        labels = [self._entry_label(entry) for entry in entries]
        label_width = max(
            painter.fontMetrics().horizontalAdvance(label)
            for label in labels)
        label_width = min(max(105, label_width + 12), self.width() * 0.42)
        plot = QtCore.QRectF(
            label_width, 12, max(1, self.width() - label_width - 96),
            max(1, self.height() - 40))
        maximum = self._maximum_timing(entries)
        row_height = plot.height() / len(entries)
        painter.setPen(self.palette().mid().color())
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawText(
            QtCore.QRectF(plot.left(), plot.bottom() + 4, 45, 20),
            QtCore.Qt.AlignmentFlag.AlignLeft, '0')
        painter.drawText(
            QtCore.QRectF(plot.right() - 80, plot.bottom() + 4, 80, 20),
            QtCore.Qt.AlignmentFlag.AlignRight,
            _process._format_time(maximum))

        for index, (entry, label) in enumerate(zip(entries, labels)):
            row_rect = QtCore.QRectF(
                0, plot.top() + index * row_height,
                self.width(), row_height)
            self._draw_entry(
                painter, plot, row_rect, entry, label, maximum)
            self._hit_regions.append((row_rect, entry))

    def _draw_entry(self, painter, plot, row_rect, entry, label, maximum):
        if entry['row'] == self._current_row:
            highlight = self.palette().highlight().color()
            highlight.setAlpha(38)
            painter.fillRect(row_rect, highlight)

        painter.setPen(self.palette().text().color())
        painter.drawText(
            QtCore.QRectF(
                6, row_rect.top(), plot.left() - 12, row_rect.height()),
            QtCore.Qt.AlignmentFlag.AlignVCenter
            | QtCore.Qt.AlignmentFlag.AlignRight,
            label)
        center_y = row_rect.center().y()
        bar_height = max(5.0, min(22.0, row_rect.height() * 0.52))
        median = entry['median']
        if median is None:
            painter.setPen(QtGui.QColor('#8a8a8a'))
            painter.drawText(
                QtCore.QRectF(
                    plot.left() + 6, row_rect.top(), plot.width(),
                    row_rect.height()),
                QtCore.Qt.AlignmentFlag.AlignVCenter, 'No timing')
            return

        median_x = plot.left() + plot.width() * median / maximum
        bar = QtCore.QRectF(
            plot.left(), center_y - bar_height / 2,
            max(1.0, median_x - plot.left()), bar_height)
        painter.fillRect(bar, self._bar_color(entry))
        p95 = entry['p95']
        if p95 is not None:
            p95_x = plot.left() + plot.width() * p95 / maximum
            whisker = QtGui.QPen(self.palette().text().color())
            whisker.setWidthF(1.2)
            painter.setPen(whisker)
            painter.drawLine(
                QtCore.QPointF(median_x, center_y),
                QtCore.QPointF(p95_x, center_y))
            painter.drawLine(
                QtCore.QPointF(p95_x, center_y - bar_height * 0.34),
                QtCore.QPointF(p95_x, center_y + bar_height * 0.34))

        painter.setPen(self.palette().text().color())
        painter.drawText(
            QtCore.QRectF(
                plot.right() + 6, row_rect.top(), 86, row_rect.height()),
            QtCore.Qt.AlignmentFlag.AlignVCenter
            | QtCore.Qt.AlignmentFlag.AlignLeft,
            _process._format_time(median))

    def _bar_color(self, entry):
        if not entry['available']:
            return QtGui.QColor('#8a8a8a')
        candidate = entry['candidate']
        color_name = entry['name']
        if color_name == 'auto' and candidate.get('selected_route'):
            color_name = candidate['selected_route']
        return _cloud.route_color(color_name)

    @staticmethod
    def _entry_label(entry):
        badges = []
        if entry['auto']:
            badges.append('Auto')
        if entry['winner']:
            badges.append('Winner')
        suffix = f' [{" / ".join(badges)}]' if badges else ''
        return f'{entry["name"]}{suffix}'

    @staticmethod
    def _maximum_timing(entries):
        values = [
            value for entry in entries
            for value in (entry['median'], entry['p95'])
            if value is not None
        ]
        return max(values, default=1.0) * 1.04 or 1.0

    def _timing_summary(self, candidate):
        summary = candidate.get('timing')
        return summary if isinstance(summary, dict) else {}

    @staticmethod
    def _baseline(entries):
        return next(
            (entry for entry in entries
             if entry['name'].lower() == 'numpy'
             and entry['median'] is not None),
            None,
        )

    @staticmethod
    def _number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None

    @staticmethod
    def _summary_noise(summary, median):
        for name in ('rmad', 'noise'):
            value = _RouteTimingCanvas._number(summary.get(name))
            if value is not None:
                return value
        deviation = _RouteTimingCanvas._number(summary.get('mad_ns'))
        if deviation is None or not median:
            return None
        return deviation / median

    @staticmethod
    def _correctness_passed(candidate):
        correctness = candidate.get('correctness')
        if isinstance(correctness, dict):
            for name in ('passed', 'ok', 'correct'):
                value = correctness.get(name)
                if isinstance(value, bool):
                    return value
        correctness = candidate.get('correct')
        return correctness if isinstance(correctness, bool) else None

    def mouseMoveEvent(self, event):
        for rect, entry in reversed(self._hit_regions):
            if rect.contains(event.position()):
                tooltip = self._entry_tooltip(entry)
                self.setToolTip(tooltip)
                QtWidgets.QToolTip.showText(
                    event.globalPosition().toPoint(), tooltip, self)
                return
        self.setToolTip('')
        QtWidgets.QToolTip.hideText()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            for rect, entry in reversed(self._hit_regions):
                if rect.contains(event.position()):
                    self.set_current_row(entry['row'])
                    self.row_selected.emit(entry['row'])
                    return
        super().mousePressEvent(event)

    def leaveEvent(self, event):
        self.setToolTip('')
        QtWidgets.QToolTip.hideText()
        super().leaveEvent(event)

    def tooltip_for_row(self, row):
        entry = next(
            (entry for entry in self.entries() if entry['row'] == row),
            None)
        return self._entry_tooltip(entry) if entry is not None else ''

    def _entry_tooltip(self, entry):
        candidate = entry['candidate']
        correctness = entry['correctness'] or 'unknown'
        reason = candidate.get('correctness', {})
        reason = reason.get('reason') if isinstance(reason, dict) else None
        if reason:
            correctness = f'{correctness}: {reason}'
        lines = (
            f'Route: {entry["name"]}',
            f'Median: {self._exact_time(entry["median"])}',
            f'P95: {self._exact_time(entry["p95"])}',
            f'Noise (MAD / median): {self._format_noise(entry["noise"])}',
            self._speedup_text(entry),
            'Eager packing: '
            f'{_process._packing_text(candidate, "eager") or "none"}',
            'Scratch packing: '
            f'{_process._packing_text(candidate, "scratch") or "none"}',
            f'Correctness: {correctness}',
            f'Eligible: {"yes" if entry["eligible"] else "no"}',
        )
        return '\n'.join(lines)

    @staticmethod
    def _exact_time(value):
        if value is None:
            return 'unavailable'
        exact = f'{value!r} ns'
        formatted = _process._format_time(value)
        return exact if formatted == exact else f'{exact} ({formatted})'

    @staticmethod
    def _format_noise(value):
        return 'unavailable' if value is None else f'{100 * value:.3f}%'

    @staticmethod
    def _speedup_text(entry):
        speedup = entry['speedup']
        baseline = entry['speedup_baseline']
        if speedup is None:
            return 'Speedup: unavailable'
        shown_baseline = 'NumPy' if baseline == 'numpy' else baseline
        return f'Speedup vs {shown_baseline}: {speedup:.4f}x'


class RouteTimingChart(QtWidgets.QWidget):
    """Compare route timings from a completed benchmark."""

    row_selected = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvas = _RouteTimingCanvas()

        controls = QtWidgets.QHBoxLayout()
        controls.addStretch(1)
        controls.addWidget(QtWidgets.QLabel('Lower is better; whisker = p95'))

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self._canvas, 1)

        self._canvas.row_selected.connect(self.row_selected)

    def set_candidates(self, candidates, selected_name, winner_name):
        self._canvas.set_candidates(
            candidates, selected_name, winner_name)

    def set_current_row(self, row):
        self._canvas.set_current_row(row)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
