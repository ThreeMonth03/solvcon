# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Bootstrap the worker before a Qt-enabled solvcon extension is imported."""

import sys


try:
    import PySide6.QtWidgets  # noqa: F401
except ImportError:
    pass

from solvcon.matmul_benchmark import worker  # noqa: E402


if __name__ == '__main__':
    sys.exit(worker.main())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
