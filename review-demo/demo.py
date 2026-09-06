# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Run PR #1497's unchanged control with a fixed real benchmark."""

import argparse
import os
from pathlib import Path
import statistics
import sys
import time

from PySide6 import QtCore, QtWidgets


class Demo(QtWidgets.QWidget):
    def __init__(self, rounds, repetitions, output_dir):
        from solvcon.benchmark.matmul import MatmulSpec
        from solvcon.pilot._benchmark import BenchmarkControl

        super().__init__()
        self.setWindowTitle('PR #1497 - standalone worker control')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spec = MatmulSpec.from_dict({
            'operation': 'matmul',
            'lhs': {'shape': [256, 256], 'strides': [256, 1]},
            'rhs': {'shape': [256, 256], 'strides': [256, 1]},
            'dtype': 'float64',
            'sampling': {'warmups': 1, 'repetitions': repetitions,
                         'rounds': rounds},
            'kernels': ['naive'],
        })
        self._closing = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            'Task 6: worker control', self))
        layout.addWidget(QtWidgets.QLabel(
            'Fixed demo: float64 | 256 x 256 | Naive + NumPy', self))
        layout.addWidget(QtWidgets.QLabel(
            f'Warmups: 1 | Repetitions: {repetitions} | Rounds: {rounds}',
            self))
        self.control = BenchmarkControl(self)
        layout.addWidget(self.control)
        self.run_button = QtWidgets.QPushButton('Run', self)
        self.run_button.clicked.connect(self.start)
        layout.addWidget(self.run_button)
        self.results = QtWidgets.QTableWidget(0, 5, self)
        self.results.setHorizontalHeaderLabels([
            'Kernel', 'Status', 'Median ms/call', 'Max abs diff',
            'Relative diff'])
        self.results.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.results.verticalHeader().hide()
        layout.addWidget(self.results)
        self.message = QtWidgets.QLabel('Press Run to collect results.', self)
        self.message.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.artifact_path = QtWidgets.QLineEdit(self)
        self.artifact_path.setReadOnly(True)
        self.artifact_path.setPlaceholderText('Completed artifact path')
        layout.addWidget(self.artifact_path)
        self.control.completed.connect(self.show_results)
        self.control.failed.connect(self.failed)
        self.control.stopped.connect(self.stopped)
        self.resize(820, 360)

    def start(self):
        self.results.setRowCount(0)
        self.artifact_path.clear()
        self.message.setText('Collecting results...')
        path = self.output_dir / f'benchmark-{time.time_ns()}.json'
        try:
            self.control.start(self.spec, path)
        except (OSError, ValueError, RuntimeError) as exc:
            self.failed(str(exc))
            return
        self.run_button.setEnabled(False)

    def show_results(self, path):
        from solvcon.benchmark.artifact import load_artifact

        self.artifact_path.setText(str(Path(path).resolve()))
        try:
            document = load_artifact(path)
            if document['spec'] != self.spec.to_dict():
                raise ValueError('artifact does not match the active spec')
        except (OSError, ValueError) as exc:
            self.finished(f'Could not display results: {exc}')
            return

        repetitions = document['spec']['sampling']['repetitions']
        self.results.setRowCount(len(document['results']))
        for row, result in enumerate(document['results']):
            elapsed = result['round_elapsed_ns']
            median = (statistics.median(elapsed) / repetitions / 1e6
                      if elapsed else None)
            values = [result['name'], result['status'],
                      '-' if median is None else f'{median:.6f}']
            for key in ('max_abs_diff', 'relative_diff'):
                value = result[key]
                values.append('-' if value is None else f'{value:.6g}')
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(result['reason'] or '')
                self.results.setItem(row, column, item)
        self.finished('Results loaded. Times are medians per call; '
                      'differences are relative to the NumPy result. '
                      'Hover over a row for an unavailable-result reason.')
        print(f'Artifact: {path}', flush=True)

    def failed(self, message):
        self.finished(f'Failed: {message}')

    def stopped(self):
        self.finished('Stopped. No completed results for this run.')

    def finished(self, message):
        self.message.setText(message)
        print(message, flush=True)
        self.run_button.setEnabled(True)
        if self._closing:
            self.close()

    def closeEvent(self, event):
        if self.control.running:
            self._closing = True
            self.control.stop()
            event.ignore()
        else:
            super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd(),
                        help='built PR #1497 checkout (default: current dir)')
    parser.add_argument('--rounds', type=int, default=400)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('pr1497-demo-output'))
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    if not (repo / 'solvcon/pilot/_benchmark.py').is_file():
        parser.error('--repo must point to a built PR #1497 checkout')
    output_dir = args.output_dir.expanduser().resolve()
    sys.path.insert(0, str(repo))
    os.environ['PYTHONPATH'] = str(repo)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    os.chdir(repo)
    app = QtWidgets.QApplication(sys.argv[:1])
    try:
        window = Demo(args.rounds, args.repetitions, output_dir)
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
