# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
HEAD = 'a1494b454017d4b6dd7a83cbe85e15746ac78e98'
SIDES = (8, 16, 32, 64, 128, 256, 512, 1024)
LARGE_SIDES = (8192, 16384, 20480)
FAMILIES = ('single', 'batch', 'broadcast', 'large')
DTYPES = ('float32', 'float64')


def write_record(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def blas_runtime():
    paths = sorted({
        line.split()[-1]
        for line in Path('/proc/self/maps').read_text().splitlines()
        if 'libopenblas' in line
    })
    assert len(paths) == 1, paths
    library = ctypes.CDLL(paths[0])
    library.openblas_get_config.restype = ctypes.c_char_p
    library.openblas_get_corename.restype = ctypes.c_char_p
    library.openblas_get_num_threads.restype = ctypes.c_int
    return {
        'path': paths[0],
        'config': library.openblas_get_config().decode(),
        'core': library.openblas_get_corename().decode(),
        'threads': library.openblas_get_num_threads(),
    }


def make_operand(np, rng, shape, dtype, step):
    storage = np.empty((*shape[:-1], step * shape[-1]), dtype=dtype)
    view = storage[..., ::step]
    for index in np.ndindex(shape[:-2]):
        matrix = view[index]
        for start in range(0, shape[-2], 256):
            block = matrix[start:start + 256]
            values = rng.random(block.shape, dtype=dtype)
            values *= 2
            values -= 1
            block[...] = values
    return view


def array_metadata(array):
    return {
        'shape': list(array.shape),
        'element_strides': [
            stride // array.itemsize for stride in array.strides
        ],
        'dtype': array.dtype.name,
    }


def check_result(np, actual, reference, dtype):
    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype
    error_sq = 0.0
    reference_sq = 0.0
    max_abs = 0.0
    actual_rows = actual.reshape(-1, actual.shape[-1])
    reference_rows = reference.reshape(-1, reference.shape[-1])
    for start in range(0, len(actual_rows), 128):
        lhs = actual_rows[start:start + 128].astype('float64')
        rhs = reference_rows[start:start + 128].astype('float64')
        assert np.isfinite(lhs).all()
        difference = lhs - rhs
        max_abs = max(max_abs, float(np.max(np.abs(difference))))
        error_sq += float(np.sum(difference * difference))
        reference_sq += float(np.sum(rhs * rhs))
    relative = math.sqrt(error_sq / reference_sq)
    tolerance = 5e-5 if dtype == 'float32' else 5e-13
    assert relative <= tolerance, (relative, tolerance)
    return {
        'relative_frobenius_error': relative,
        'relative_frobenius_tolerance': tolerance,
        'maximum_absolute_error': max_abs,
        'output': array_metadata(actual),
    }


def timed_batch(function, loops):
    begin = time.perf_counter_ns()
    for _ in range(loops):
        result = function()
        del result
    return time.perf_counter_ns() - begin


def run_case(args):
    large = args.family == 'large'
    affinity = sorted(os.sched_getaffinity(0)) if large else [2]
    os.sched_setaffinity(0, affinity)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import solvcon as sc

    assert np.__version__ == '2.5.1', np.__version__
    assert subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT,
        text=True).strip() == HEAD
    extension = Path(sc.core._impl.__file__).resolve()
    assert extension.is_relative_to(ROOT), extension
    blas = blas_runtime()
    expected_threads = 24 if large else 1
    assert blas['threads'] == expected_threads, blas
    side = args.side
    if args.family in ('single', 'large'):
        shapes = ((side, side), (side, side))
    else:
        count = 1 if args.family == 'broadcast' else 10
        shapes = ((count, side, side), (10, side, side))
    steps = (1, 1) if large else (
        2, 1 if args.family == 'single' else 2)
    seed = (20261004 + FAMILIES.index(args.family) * 100000
            + side * 2 + DTYPES.index(args.dtype))
    rng = np.random.default_rng(seed)
    lhs, rhs = [
        make_operand(np, rng, shape, args.dtype, step)
        for shape, step in zip(shapes, steps)
    ]
    array_class = getattr(sc, 'SimpleArray' + args.dtype.capitalize())
    lhs_sa, rhs_sa = array_class(array=lhs), array_class(array=rhs)
    for source, wrapped in ((lhs, lhs_sa), (rhs, rhs_sa)):
        view = wrapped.ndarray
        assert np.shares_memory(source, view)
        assert array_metadata(source) == array_metadata(view)
    methods = {
        'numpy': lambda: np.matmul(lhs, rhs),
        'solvcon': lambda: lhs_sa.matmul(rhs_sa),
    }
    reference = methods['numpy']()
    actual = methods['solvcon']()
    validation = check_result(np, actual.ndarray, reference, args.dtype)
    assert array_metadata(reference) == validation['output']
    del reference, actual
    gc.collect()
    gc.disable()
    loops = {}
    for name, function in methods.items():
        warmup_ns = timed_batch(function, 1)
        loops[name] = (1 if large else min(
            100000, max(1, math.ceil(20000000 / warmup_ns))))
        if not large:
            for _ in range(4):
                elapsed = timed_batch(function, loops[name])
                if elapsed >= 16000000:
                    break
                loops[name] = min(100000, max(
                    1, math.ceil(loops[name] * 20000000 / elapsed)))
            for _ in range(2):
                timed_batch(function, loops[name])
    case_id = f'{args.family}-{args.dtype}-{side}'
    common = {
        'case': case_id,
        'family': args.family,
        'dtype': args.dtype,
        'side': side,
        'seed': seed,
        'sha': HEAD,
        'numpy_version': np.__version__,
        'python': sys.version,
        'python_executable': sys.executable,
        'extension': str(extension),
        'platform': platform.platform(),
        'affinity': affinity,
        'blas': blas,
        'lhs': array_metadata(lhs),
        'rhs': array_metadata(rhs),
        'validation': validation,
        'expected_route': (
            'winograd' if large and side >= 16384
            else 'direct_blas' if large
            else 'generic' if side < 16 else 'packed_blas'),
        'target_batch_ms': 20 if not large else None,
        'samples_per_round': 1 if large else 15,
        'rounds': 6,
        'includes': ['public_api', 'output_allocation',
                     'internal_packing_or_scratch', 'output_release'],
        'cache_policy': 'warm repeated operands within each process',
    }
    print(json.dumps({'event': 'validated', 'case': case_id,
                      'error': validation['relative_frobenius_error']}),
          flush=True)
    for round_index in range(6):
        order = ['numpy', 'solvcon']
        if (round_index + side + DTYPES.index(args.dtype)) % 2:
            order.reverse()
        for name in order:
            durations = [
                timed_batch(methods[name], loops[name])
                for _ in range(common['samples_per_round'])
            ]
            record = {
                **common,
                'round': round_index,
                'method': name,
                'method_order': order,
                'inner_loops': loops[name],
                'batch_elapsed_ns': durations,
                'samples_ns': [value / loops[name] for value in durations],
                'peak_rss_kib': resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss,
                'major_faults': resource.getrusage(
                    resource.RUSAGE_SELF).ru_majflt,
                'unix_time': time.time(),
            }
            write_record(DEST / f'{case_id}.jsonl', record)
        print(json.dumps({'event': 'round', 'case': case_id,
                          'round': round_index + 1}), flush=True)
    gc.enable()


def run_suite(args):
    cases = [
        (family, dtype, side)
        for family in FAMILIES
        for dtype in DTYPES
        for side in (LARGE_SIDES if family == 'large' else SIDES)
        if args.suite == 'all' or (family == 'large') == (
            args.suite == 'large')
    ]
    manifest = {
        'sha': HEAD,
        'families': FAMILIES,
        'small_sides': SIDES,
        'large_sides': LARGE_SIDES,
        'dtypes': DTYPES,
        'small_threads': 1,
        'large_threads': 24,
        'rounds': 6,
        'small_batches_per_round': 15,
        'large_calls_per_round': 1,
        'relative_frobenius_tolerances': {
            'float32': 5e-5, 'float64': 5e-13},
        'script_sha256': hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
    }
    (DEST / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n')
    for index, (family, dtype, side) in enumerate(cases):
        path = DEST / f'{family}-{dtype}-{side}.jsonl'
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) == 12:
                continue
            raise RuntimeError(f'Incomplete case: {path}')
        env = dict(os.environ)
        threads = '24' if family == 'large' else '1'
        env.update(OPENBLAS_NUM_THREADS=threads, OMP_NUM_THREADS=threads,
                   MKL_NUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads)
        print(f'CASE {index + 1}/{len(cases)} {family} {dtype} {side}',
              flush=True)
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            '--family', family, '--dtype', dtype, '--side', str(side),
        ], cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--dtype', choices=DTYPES, default='float32')
    parser.add_argument('--side', type=int, default=32)
    parser.add_argument('--suite', choices=('all', 'small', 'large'),
                        default='all')
    args = parser.parse_args()
    run_case(args) if args.family else run_suite(args)


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
