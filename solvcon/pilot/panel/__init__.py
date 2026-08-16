# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Pilot panel and inspection-window features."""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _profiling
    from . import _matmul_benchmark
    from . import _tree_panel
    from . import _window_manager

    EntityTreeWidget = _tree_panel.EntityTreeWidget
    MeshInfoTree = _tree_panel.MeshInfoTree
    MatmulBenchmark = _matmul_benchmark.MatmulBenchmark
    MatmulBenchmarkWindow = _matmul_benchmark.MatmulBenchmarkWindow
    TreePanel = _tree_panel.TreePanel
    TreePanelBase = _tree_panel.TreePanelBase
    WindowManager = _window_manager.WindowManager
    Profiling = _profiling.Profiling
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    EntityTreeWidget = None
    MeshInfoTree = None
    MatmulBenchmark = None
    MatmulBenchmarkWindow = None
    TreePanel = None
    TreePanelBase = None
    WindowManager = None
    Profiling = None

__all__ = [
    'EntityTreeWidget',
    'MeshInfoTree',
    'MatmulBenchmark',
    'MatmulBenchmarkWindow',
    'Profiling',
    'TreePanel',
    'TreePanelBase',
    'WindowManager',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
