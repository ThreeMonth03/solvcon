# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import gc
import json
import pathlib
import platform
import statistics
import subprocess
import time

import profile_benchmark_environment  # noqa: F401

import numpy as np
import solvcon

from profile_matmul_cartesian import matrix_axis_layouts
from profile_matmul_cartesian import vector_layouts


SEED = 20260728
CORE_LAYOUTS = (
    ('c', 'c', 'c'),
    ('f', 'f-matrix', 'f-matrix'),
    (
        'negative-core',
        'axis-minus-1-negative',
        'axis-minus-2-negative',
    ),
    (
        'step2-core',
        'axis-minus-1-step-2',
        'axis-minus-2-step-2',
    ),
)
PACK_LAYOUTS = (
    ('dense', 'stride-1', 'c'),
    ('positive-step2-vector', 'stride-2', 'c'),
    ('fortran-matrix', 'stride-1', 'f-matrix'),
    ('negative-vector', 'stride-minus-1', 'c'),
    ('negative-step2-vector', 'stride-minus-2', 'c'),
    ('zero-vector', 'stride-0', 'c'),
    ('matrix-row-step2', 'stride-1', 'axis-minus-2-step-2'),
    ('matrix-column-step2', 'stride-1', 'axis-minus-1-step-2'),
    (
        'matrix-both-step2',
        'stride-1',
        'both-matrix-axes-step-2',
    ),
    ('matrix-row-negative', 'stride-1', 'axis-minus-2-negative'),
    (
        'matrix-column-negative',
        'stride-1',
        'axis-minus-1-negative',
    ),
    (
        'matrix-both-negative',
        'stride-1',
        'both-matrix-axes-negative',
    ),
    ('matrix-row-zero', 'stride-1', 'axis-minus-2-zero'),
    ('matrix-column-zero', 'stride-1', 'axis-minus-1-zero'),
    (
        'both-operands-unsupported',
        'stride-minus-1',
        'both-matrix-axes-step-2',
    ),
)
RECTANGULAR_PAIRS = (
    (8, 72),
    (24, 24),
    (72, 8),
    (8, 128),
    (32, 32),
    (128, 8),
    (16, 256),
    (64, 64),
    (256, 16),
)
RECTANGULAR_LAYOUTS = (
    ('negative-vector', 'stride-minus-1'),
    ('negative-step2-vector', 'stride-minus-2'),
    ('zero-vector', 'stride-0'),
)


def make_array(values):
    if values.dtype == np.dtype('float32'):
        return solvcon.SimpleArrayFloat32(array=values)
    return solvcon.SimpleArrayFloat64(array=values)


def element_strides(values):
    return tuple(
        stride // values.itemsize for stride in values.strides)


def check_result(actual, expected):
    if hasattr(actual, 'ndarray'):
        actual = actual.ndarray
    if expected.ndim == 0:
        expected = expected.reshape((1,))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-5,
        atol=2e-6,
    )


def calibrate(function, target_ns):
    loops = 1
    while True:
        start = time.perf_counter_ns()
        for _ in range(loops):
            function()
        if time.perf_counter_ns() - start >= target_ns:
            return loops
        loops *= 2


def measure_calls(calls, samples, warmups, target_ms):
    target_ns = int(target_ms * 1_000_000)
    loops = {
        name: calibrate(function, target_ns)
        for name, function in calls.items()
    }
    names = list(calls)
    raw = {name: [] for name in names}
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for round_number in range(warmups + samples):
            order = names if round_number % 2 == 0 else names[::-1]
            for name in order:
                function = calls[name]
                start = time.perf_counter_ns()
                for _ in range(loops[name]):
                    function()
                elapsed = time.perf_counter_ns() - start
                if round_number >= warmups:
                    raw[name].append(elapsed / loops[name] / 1e9)
    finally:
        if gc_enabled:
            gc.enable()

    return {
        name: {
            'loops': loops[name],
            'median_seconds': statistics.median(raw[name]),
            'samples_seconds': raw[name],
        }
        for name in names
    }


def benchmark_case(group, topology, layout, lhs, rhs, args):
    expected = np.matmul(lhs, rhs)
    lhs_array = make_array(lhs)
    rhs_array = make_array(rhs)
    calls = {
        'numpy': lambda: np.matmul(lhs, rhs),
        'planned': lambda: lhs_array.matmul_planned(rhs_array),
    }
    if group == 'pack-crossover':
        packed_lhs = np.ascontiguousarray(lhs)
        packed_rhs = np.ascontiguousarray(rhs)
        packed_lhs_array = make_array(packed_lhs)
        packed_rhs_array = make_array(packed_rhs)
        calls.update({
            'numpy_prepacked': (
                lambda: np.matmul(packed_lhs, packed_rhs)),
            'planned_prepacked': (
                lambda: packed_lhs_array.matmul_planned(
                    packed_rhs_array)),
        })

    for function in calls.values():
        check_result(function(), expected)
    timings = measure_calls(
        calls,
        args.samples,
        args.warmups,
        args.target_ms,
    )
    numpy_time = timings['numpy']['median_seconds']
    planned_time = timings['planned']['median_seconds']
    return {
        'group': group,
        'topology': topology,
        'layout': layout,
        'dtype': lhs.dtype.name,
        'lhs_shape': list(lhs.shape),
        'rhs_shape': list(rhs.shape),
        'lhs_element_strides': list(element_strides(lhs)),
        'rhs_element_strides': list(element_strides(rhs)),
        'output_shape': list(expected.shape),
        'numpy_over_planned': numpy_time / planned_time,
        'timings': timings,
    }


def matrix_operands(rng, dtype, side, batch, lhs_layout, rhs_layout):
    prefix = () if batch is None else (batch,)
    lhs_values = rng.random(prefix + (side, side)).astype(dtype)
    rhs_values = rng.random(prefix + (side, side)).astype(dtype)
    lhs = matrix_axis_layouts(lhs_values)[lhs_layout]
    rhs = matrix_axis_layouts(rhs_values)[rhs_layout]
    return lhs, rhs


def core_cases(rng):
    for dtype in ('float32', 'float64'):
        for topology, batch in (('2d-2d', None), ('nd-nd', 8)):
            for side in (8, 16, 32, 64, 128, 256):
                for layout, lhs_layout, rhs_layout in CORE_LAYOUTS:
                    lhs, rhs = matrix_operands(
                        rng,
                        dtype,
                        side,
                        batch,
                        lhs_layout,
                        rhs_layout,
                    )
                    yield topology, layout, lhs, rhs


def make_core_layout(values, name, lhs):
    layouts = matrix_axis_layouts(values)
    if name == 'c':
        return layouts['c']
    if name == 'negative-core':
        axis = 'axis-minus-1' if lhs else 'axis-minus-2'
        return layouts[f'{axis}-negative']
    axis = 'axis-minus-1' if lhs else 'axis-minus-2'
    return layouts[f'{axis}-step-2']


def broadcast_operands(rng, dtype, topology, side, batch, layout):
    matrix = rng.random((side, side)).astype(dtype)
    singleton = rng.random((1, side, side)).astype(dtype)
    batched = rng.random((batch, side, side)).astype(dtype)
    cross_lhs = rng.random((batch, 1, side, side)).astype(dtype)
    cross_rhs = rng.random((1, batch, side, side)).astype(dtype)
    bases = {
        '2d-nd': (matrix, batched),
        'nd-2d': (batched, matrix),
        'singleton-nd': (singleton, batched),
        'nd-singleton': (batched, singleton),
        'cross-broadcast': (cross_lhs, cross_rhs),
    }
    lhs, rhs = bases[topology]
    return (
        make_core_layout(lhs, layout, True),
        make_core_layout(rhs, layout, False),
    )


def broadcast_cases(rng):
    topologies = (
        '2d-nd',
        'nd-2d',
        'singleton-nd',
        'nd-singleton',
        'cross-broadcast',
    )
    for dtype in ('float32', 'float64'):
        for batch in (1, 4, 16, 64):
            for topology in topologies:
                for layout in ('c', 'negative-core', 'step2-core'):
                    lhs, rhs = broadcast_operands(
                        rng, dtype, topology, 64, batch, layout)
                    case_layout = f'{layout}/batch={batch}'
                    yield topology, case_layout, lhs, rhs


def vector_operands(rng, topology, inner_size, output_extent,
                    batch, vector_layout, matrix_layout):
    vector_values = rng.random((inner_size,))
    vector = vector_layouts(vector_values)[vector_layout]
    if topology == '1d-nd':
        matrix_shape = (batch, inner_size, output_extent)
    else:
        matrix_shape = (batch, output_extent, inner_size)
    matrix_values = rng.random(matrix_shape)
    matrix = matrix_axis_layouts(matrix_values)[matrix_layout]
    if topology == '1d-nd':
        return vector, matrix
    return matrix, vector


def pack_cases(rng):
    for topology in ('1d-nd', 'nd-1d'):
        for layout, vector_layout, matrix_layout in PACK_LAYOUTS:
            for side in (8, 16, 32, 64, 128, 256):
                for batch in (1, 2, 4, 8, 16, 64):
                    lhs, rhs = vector_operands(
                        rng,
                        topology,
                        side,
                        side,
                        batch,
                        vector_layout,
                        matrix_layout,
                    )
                    case_layout = (
                        f'{layout}/side={side}/batch={batch}')
                    yield topology, case_layout, lhs, rhs


def vector_threshold_cases(rng):
    for topology in ('1d-nd', 'nd-1d'):
        for side in (2, 4, 8, 16, 24, 32, 48, 64, 96, 128):
            for batch in (1, 4, 16, 64):
                lhs, rhs = vector_operands(
                    rng,
                    topology,
                    side,
                    side,
                    batch,
                    'stride-1',
                    'c',
                )
                layout = f'dense/side={side}/batch={batch}'
                yield topology, layout, lhs, rhs


def rectangular_cases(rng):
    for topology in ('1d-nd', 'nd-1d'):
        for layout, vector_layout in RECTANGULAR_LAYOUTS:
            for inner_size, output_extent in RECTANGULAR_PAIRS:
                for batch in (1, 2, 4, 8, 16):
                    lhs, rhs = vector_operands(
                        rng,
                        topology,
                        inner_size,
                        output_extent,
                        batch,
                        vector_layout,
                        'c',
                    )
                    case_layout = (
                        f'{layout}/k={inner_size}/'
                        f'o={output_extent}/batch={batch}')
                    yield topology, case_layout, lhs, rhs


def group_cases(group, rng):
    generators = {
        'core': core_cases,
        'broadcast-scaling': broadcast_cases,
        'pack-crossover': pack_cases,
        'vector-batch-threshold': vector_threshold_cases,
        'vector-pack-rectangular': rectangular_cases,
    }
    return generators[group](rng)


def git_revision():
    return subprocess.check_output(
        ('git', 'rev-parse', 'HEAD'),
        text=True,
    ).strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark planned matmul against NumPy.')
    parser.add_argument(
        '--group',
        action='append',
        choices=(
            'core',
            'broadcast-scaling',
            'pack-crossover',
            'vector-batch-threshold',
            'vector-pack-rectangular',
        ),
    )
    parser.add_argument('--samples', type=int, default=15)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--target-ms', type=float, default=20.0)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    groups = args.group or [
        'core',
        'broadcast-scaling',
        'pack-crossover',
        'vector-batch-threshold',
        'vector-pack-rectangular',
    ]
    rng = np.random.default_rng(SEED)
    rows = []
    for group in groups:
        group_count = 0
        for topology, layout, lhs, rhs in group_cases(group, rng):
            rows.append(benchmark_case(
                group, topology, layout, lhs, rhs, args))
            group_count += 1
            if group_count % 25 == 0:
                print(
                    f'Benchmarked {group}: {group_count} cases',
                    flush=True,
                )
        print(f'Benchmarked {group}: {group_count} cases')

    payload = {
        'metadata': {
            'revision': git_revision(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'numpy_version': np.__version__,
            'solvcon_binary': solvcon.core._impl.__file__,
            'samples': args.samples,
            'warmups': args.warmups,
            'target_ms': args.target_ms,
            'groups': groups,
            'seed': SEED,
        },
        'results': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + '\n',
        encoding='utf-8',
    )
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
