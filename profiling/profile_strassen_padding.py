#!/usr/bin/env python3

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import functools
import gc
import itertools
import json
import os
import platform
import statistics
import time

import numpy as np

import solvcon


ARRAY_TYPES = {
    "float32": solvcon.SimpleArrayFloat32,
    "float64": solvcon.SimpleArrayFloat64,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure full-call padded Strassen matmul")
    parser.add_argument("--sides", nargs="+", type=int, default=(512,))
    parser.add_argument(
        "--cases", nargs="+",
        choices=("divisible", "m", "n", "k", "all"),
        default=("divisible", "all"))
    parser.add_argument(
        "--dtypes", nargs="+", choices=tuple(ARRAY_TYPES),
        default=tuple(ARRAY_TYPES))
    parser.add_argument("--depths", nargs="+", type=int, default=(1, 2))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def shape_for(side, case):
    offsets = {
        "divisible": (0, 0, 0),
        "m": (1, 0, 0),
        "n": (0, 0, 1),
        "k": (0, 1, 0),
        "all": (1, 1, 1),
    }
    dm, dk, dn = offsets[case]
    return side + dm, side + dk, side + dn


def round_up(value, divisor):
    return (value + divisor - 1) // divisor * divisor


def workspace_elements(rows, columns, inner_size, depth):
    if depth == 0:
        return 0
    half_rows = rows // 2
    half_columns = columns // 2
    half_inner = inner_size // 2
    current = (half_rows * half_inner +
               half_inner * half_columns +
               half_rows * half_columns)
    return current + workspace_elements(
        half_rows, half_columns, half_inner, depth - 1)


def memory_metadata(shape, depth, itemsize):
    rows, inner_size, columns = shape
    divisor = 1 << depth
    padded = tuple(round_up(value, divisor) for value in shape)
    prows, pinner, pcolumns = padded
    needs_padding = padded != shape
    temporary_elements = 0
    copied_elements = 0
    if needs_padding:
        temporary_elements = (prows * pinner + pinner * pcolumns +
                              prows * pcolumns)
        copied_elements = (prows * pinner + pinner * pcolumns +
                           2 * (rows * inner_size +
                                inner_size * columns +
                                rows * columns))
    workspace = workspace_elements(
        prows, pcolumns, pinner, depth)
    return {
        "padded_shape": [prows, pinner, pcolumns],
        "padding_ratio": (prows * pcolumns * pinner) /
                         (rows * columns * inner_size),
        "temporary_bytes": temporary_elements * itemsize,
        "minimum_copy_bytes": copied_elements * itemsize,
        "workspace_bytes": workspace * itemsize,
    }


def relative_frobenius(actual, expected):
    numerator = np.linalg.norm(actual - expected)
    denominator = np.linalg.norm(expected)
    return float(numerator / denominator)


def numpy_matmul(lhs, rhs):
    return np.matmul(lhs, rhs)


def blas_matmul(lhs, rhs):
    return lhs.matmul_blas(rhs).ndarray


def strassen_matmul(lhs, rhs, depth):
    return lhs._matmul_strassen_control(rhs, depth, True).ndarray


def time_method(method, warmups, samples):
    for _ in range(warmups):
        result = method()
        del result

    timings = []
    result = None
    for _ in range(samples):
        result = None
        start = time.perf_counter_ns()
        result = method()
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return timings, result


def benchmark_case(dtype_name, shape, depths, warmups, samples, rounds):
    array_type = ARRAY_TYPES[dtype_name]
    rows, inner_size, columns = shape
    seed = rows * 1_000_003 + inner_size * 1_009 + columns
    rng = np.random.default_rng(seed)
    lhs_np = rng.standard_normal((rows, inner_size), dtype=dtype_name)
    rhs_np = rng.standard_normal((inner_size, columns), dtype=dtype_name)
    lhs = array_type(array=lhs_np)
    rhs = array_type(array=rhs_np)

    methods = {
        "numpy": functools.partial(numpy_matmul, lhs_np, rhs_np),
        "blas": functools.partial(blas_matmul, lhs, rhs),
    }
    for depth in depths:
        methods[f"strassen_{depth}"] = functools.partial(
            strassen_matmul, lhs, rhs, depth)

    expected = methods["numpy"]()
    residuals = {}
    for name, method in methods.items():
        residuals[name] = relative_frobenius(method(), expected)

    timings = {name: [] for name in methods}
    names = tuple(methods)
    for round_index in range(rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            values, result = time_method(
                methods[name], warmups, samples)
            timings[name].extend(values)
            del result

    records = []
    for name in names:
        records.append({
            "method": name,
            "median_ms": statistics.median(timings[name]),
            "samples_ms": timings[name],
            "relative_frobenius": residuals[name],
        })
    del methods, expected, lhs, rhs, lhs_np, rhs_np
    gc.collect()
    return records


def main():
    args = parse_args()
    print(json.dumps({
        "kind": "environment",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "openblas_threads": os.getenv("OPENBLAS_NUM_THREADS"),
        "omp_threads": os.getenv("OMP_NUM_THREADS"),
        "warmups": args.warmups,
        "samples": args.samples,
        "rounds": args.rounds,
    }), flush=True)

    combinations = itertools.product(
        args.dtypes, args.sides, args.cases)
    for dtype_name, side, case in combinations:
        shape = shape_for(side, case)
        timings = benchmark_case(
            dtype_name, shape, tuple(args.depths), args.warmups,
            args.samples, args.rounds)
        memory = {
            str(depth): memory_metadata(
                shape, depth, np.dtype(dtype_name).itemsize)
            for depth in args.depths
        }
        print(json.dumps({
            "kind": "case",
            "dtype": dtype_name,
            "case": case,
            "shape_mkn": list(shape),
            "memory": memory,
            "timings": timings,
        }), flush=True)


if __name__ == "__main__":
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
