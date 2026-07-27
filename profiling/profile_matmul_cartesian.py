# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import itertools
import json
import pathlib
import platform
import subprocess

import profile_benchmark_environment  # noqa: F401

import numpy as np
import solvcon


SEED = 20260728
VECTOR_LAYOUT_NAMES = (
    'stride-1',
    'stride-minus-1',
    'stride-2',
    'stride-minus-2',
    'stride-0',
)
BATCH_AXIS_LAYOUT_NAMES = ('c', 'negative', 'step-2', 'zero')
MATRIX_LAYOUT_SPECS = {
    'c': ('c', 'c', 'c'),
    'f-matrix': ('f', 'c', 'c'),
    'axis-minus-2-step-2': ('c', 'step-2', 'c'),
    'axis-minus-1-step-2': ('c', 'c', 'step-2'),
    'both-matrix-axes-step-2': ('c', 'step-2', 'step-2'),
    'axis-minus-2-negative': ('c', 'negative', 'c'),
    'axis-minus-1-negative': ('c', 'c', 'negative'),
    'both-matrix-axes-negative': ('c', 'negative', 'negative'),
    'axis-minus-2-zero': ('c', 'zero', 'c'),
    'axis-minus-1-zero': ('c', 'c', 'zero'),
}


def make_zero_stride(values, axis):
    strides = list(values.strides)
    strides[axis] = 0
    return np.lib.stride_tricks.as_strided(
        values,
        shape=values.shape,
        strides=tuple(strides),
        writeable=True,
    )


def make_stepped(values):
    storage = np.empty(values.size * 2, dtype=values.dtype)
    result = storage[::2]
    result[...] = values
    return result


def make_combined_layout(values, modes, matrix_order='c'):
    shape = values.shape
    if matrix_order == 'f':
        fastest_axes = [
            values.ndim - 2,
            values.ndim - 1,
            *range(values.ndim - 3, -1, -1),
        ]
    else:
        fastest_axes = list(range(values.ndim - 1, -1, -1))

    strides = [0] * values.ndim
    storage_size = 1
    offset = 0
    for axis in fastest_axes:
        mode = modes[axis]
        if mode == 'zero':
            strides[axis] = 0
            continue
        step = 2 if mode == 'step-2' else 1
        stride = storage_size * step
        if mode == 'negative':
            strides[axis] = -stride
            offset += (shape[axis] - 1) * stride
        else:
            strides[axis] = stride
        storage_size *= max(shape[axis], 1) * step

    storage = np.empty(storage_size, dtype=values.dtype)
    itemsize = storage.dtype.itemsize
    view = np.ndarray(
        shape=shape,
        dtype=values.dtype,
        buffer=storage,
        offset=offset * itemsize,
        strides=tuple(stride * itemsize for stride in strides),
    )
    view[...] = values
    return view


def vector_layouts(values):
    contiguous = np.ascontiguousarray(values)
    layouts = {
        'stride-1': contiguous,
        'stride-minus-1': contiguous[::-1],
        'stride-2': make_stepped(contiguous),
        'stride-minus-2': make_stepped(contiguous)[::-1],
        'stride-0': make_zero_stride(contiguous, 0),
    }
    return {name: layouts[name] for name in VECTOR_LAYOUT_NAMES}


def matrix_axis_layouts(values):
    batch_modes = ('c',) * (values.ndim - 2)
    return {
        name: make_combined_layout(
            values,
            batch_modes + layout_spec[1:],
            layout_spec[0],
        )
        for name, layout_spec in MATRIX_LAYOUT_SPECS.items()
    }


def matrix_layouts(values):
    if values.ndim == 2:
        return matrix_axis_layouts(values)

    result = {}
    for name, layout_spec in MATRIX_LAYOUT_SPECS.items():
        matrix_order = layout_spec[0]
        core_modes = layout_spec[1:]
        for batch_cases in itertools.product(
                BATCH_AXIS_LAYOUT_NAMES,
                repeat=values.ndim - 2):
            modes = batch_cases + core_modes
            batch_names = [
                f'axis-{axis}-{mode}'
                for axis, mode in enumerate(batch_cases)
            ]
            layout = f"{name}__{'__'.join(batch_names)}"
            result[layout] = make_combined_layout(
                values, modes, matrix_order)
    return result


def topology_bases(side, batch):
    rng = np.random.default_rng(SEED)
    vector_lhs = rng.random((side,))
    vector_rhs = rng.random((side,))
    matrix_lhs = rng.random((side, side))
    matrix_rhs = rng.random((side, side))
    batch_lhs = rng.random((batch, side, side))
    batch_rhs = rng.random((batch, side, side))
    lhs_one = rng.random((1, side, side))
    rhs_one = rng.random((1, side, side))
    cross_lhs = rng.random((batch, 1, side, side))
    cross_rhs = rng.random((1, batch, side, side))
    return (
        ('1d-1d', vector_lhs, vector_rhs),
        ('1d-2d', vector_lhs, matrix_rhs),
        ('2d-1d', matrix_lhs, vector_rhs),
        ('2d-2d', matrix_lhs, matrix_rhs),
        ('1d-nd', vector_lhs, batch_rhs),
        ('nd-1d', batch_lhs, vector_rhs),
        ('2d-nd', matrix_lhs, batch_rhs),
        ('nd-2d', batch_lhs, matrix_rhs),
        ('nd-nd-same-batch', batch_lhs, batch_rhs),
        ('nd-nd-lhs-broadcast', lhs_one, batch_rhs),
        ('nd-nd-rhs-broadcast', batch_lhs, rhs_one),
        ('nd-nd-cross-broadcast', cross_lhs, cross_rhs),
    )


def operand_layouts(values):
    if values.ndim == 1:
        return vector_layouts(values)
    return matrix_layouts(values)


def make_array(values):
    return solvcon.SimpleArrayFloat64(array=values)


def check_case(lhs, rhs):
    expected = np.matmul(lhs, rhs)
    actual = make_array(lhs).matmul_planned(make_array(rhs)).ndarray
    if expected.ndim == 0:
        expected = expected.reshape((1,))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def git_revision():
    return subprocess.check_output(
        ('git', 'rev-parse', 'HEAD'),
        text=True,
    ).strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate every planned matmul layout pair.')
    parser.add_argument('--side', type=int, default=4)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--output', type=pathlib.Path)
    return parser.parse_args()


def main():
    args = parse_args()
    case_count = 0
    topology_counts = {}
    for topology, lhs_base, rhs_base in topology_bases(
            args.side, args.batch):
        count = 0
        layouts = itertools.product(
            operand_layouts(lhs_base).values(),
            operand_layouts(rhs_base).values(),
        )
        for lhs, rhs in layouts:
            check_case(lhs, rhs)
            case_count += 1
            count += 1
            if case_count % 250 == 0:
                print(
                    f'Validated {case_count} Cartesian cases...',
                    flush=True,
                )
        topology_counts[topology] = count

    payload = {
        'revision': git_revision(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'solvcon_binary': solvcon.core._impl.__file__,
        'side': args.side,
        'batch': args.batch,
        'case_count': case_count,
        'topology_counts': topology_counts,
        'vector_layouts': list(VECTOR_LAYOUT_NAMES),
        'matrix_core_layouts': list(MATRIX_LAYOUT_SPECS),
        'batch_axis_layouts': list(BATCH_AXIS_LAYOUT_NAMES),
    }
    print(f'Validated Cartesian matmul cases: {case_count}')
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2) + '\n',
            encoding='utf-8',
        )
        print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
