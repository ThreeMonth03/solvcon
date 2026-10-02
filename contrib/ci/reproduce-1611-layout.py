#!/usr/bin/env python3
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import signal
import subprocess
import sys


def reproduce(mode):
    from PySide6 import QtWidgets
    import shiboken6

    app = QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(parent)
    table = QtWidgets.QTableWidget(1, 1)
    widgets = [QtWidgets.QWidget() for _ in range(100)]
    for widget in widgets:
        layout.addWidget(widget)
    addresses = set()
    if mode == 'baseline':
        for index in range(layout.count()):
            receiver = layout.itemAt(index)
            addresses.add(shiboken6.getCppPointer(receiver)[0])
            assert receiver.widget() is widgets[index]
    else:
        for index, widget in enumerate(widgets):
            assert layout.indexOf(widget) == index
        assert parent.findChild(QtWidgets.QWidget) is widgets[0]

    for widget in widgets:
        shiboken6.delete(widget)
    items = [QtWidgets.QTableWidgetItem('a') for _ in range(1000)]
    for index in range(len(items)):
        item = items[index]
        if shiboken6.getCppPointer(item)[0] in addresses:
            print('Reused layout item address', flush=True)
        table.setItem(0, 0, item)
        # enumerate(items) would retain the item while Qt destroys it.
        items[index] = None
        del item
        table.setItem(0, 0, QtWidgets.QTableWidgetItem('b'))
    print('Completed 1000 table replacements', flush=True)
    shiboken6.delete(parent)
    shiboken6.delete(table)
    app.processEvents()


def compare():
    for mode in ('baseline', 'fixed'):
        result = subprocess.run(
            [sys.executable, __file__, mode], capture_output=True, text=True)
        print(f'{mode}: returncode={result.returncode}', flush=True)
        print(result.stdout, end='', flush=True)
        print(result.stderr, end='', flush=True)
        if mode == 'baseline':
            assert 'Reused layout item address' in result.stdout
            assert result.returncode in (-signal.SIGSEGV, -signal.SIGABRT)
        else:
            result.check_returncode()
            assert 'Completed 1000 table replacements' in result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('baseline', 'fixed', 'compare'))
    args = parser.parse_args()
    if args.mode == 'compare':
        compare()
    else:
        reproduce(args.mode)


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
