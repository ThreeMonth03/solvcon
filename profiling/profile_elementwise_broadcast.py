# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import collections
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time


THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"

import numpy as np  # noqa: E402
import solvcon  # noqa: E402

from profiling import (  # noqa: E402
    elementwise_benchmark_cases as benchmark_cases,
)


ARRAY_TYPES = {
    "bool": solvcon.SimpleArrayBool,
    "int8": solvcon.SimpleArrayInt8,
    "int16": solvcon.SimpleArrayInt16,
    "int32": solvcon.SimpleArrayInt32,
    "int64": solvcon.SimpleArrayInt64,
    "uint8": solvcon.SimpleArrayUint8,
    "uint16": solvcon.SimpleArrayUint16,
    "uint32": solvcon.SimpleArrayUint32,
    "uint64": solvcon.SimpleArrayUint64,
    "float32": solvcon.SimpleArrayFloat32,
    "float64": solvcon.SimpleArrayFloat64,
    "complex64": solvcon.SimpleArrayComplex64,
    "complex128": solvcon.SimpleArrayComplex128,
}

IMPLEMENTATIONS = ("legacy", "legacy_simd", "planned")

NUMPY_OPERATIONS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "eq": np.equal,
    "ne": np.not_equal,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
}

BUG_STATUSES = {
    "unexpected-error",
    "unexpected-success",
    "wrong-shape",
    "wrong-dtype",
    "wrong-value",
    "input-mutation",
    "rhs-mutation",
    "out-of-view-write",
    "construction-error",
    "process-crash",
    "unsafe-memory-access",
    "unsafe-shape-validation",
    "benchmark-error",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Audit and profile SimpleArray elementwise broadcasting."
        )
    )
    parser.add_argument(
        "--catalog",
        choices=("smoke", "correctness", "performance", "exhaustive"),
        default="smoke",
    )
    parser.add_argument("--operation", action="append")
    parser.add_argument("--dtype", action="append")
    parser.add_argument("--mode", action="append")
    parser.add_argument("--topology", action="append")
    parser.add_argument("--lhs-layout", action="append")
    parser.add_argument("--rhs-layout", action="append")
    parser.add_argument("--value-pattern", action="append")
    parser.add_argument("--size", type=int, action="append")
    parser.add_argument(
        "--implementation",
        choices=IMPLEMENTATIONS,
        action="append",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--timing",
        choices=("none", "matched", "stable"),
        default="none",
    )
    parser.add_argument(
        "--record",
        choices=("all", "findings", "summary"),
        default="all",
        help=(
            "Store all cases, findings, or only aggregate counts in JSON."
        ),
    )
    parser.add_argument(
        "--record-status",
        action="append",
        help="With --record findings, keep only these statuses.",
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--target-ms", type=float, default=20.0)
    parser.add_argument("--stable-processes", type=int, default=3)
    parser.add_argument("--stable-rounds", type=int, default=5)
    parser.add_argument("--warmup-ms", type=float, default=20.0)
    parser.add_argument(
        "--preallocated-output",
        action="store_true",
        help=(
            "Also validate and time NumPy and planned execution into a "
            "reused C-contiguous output."
        ),
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--fail-on-bug", action="store_true")
    parser.add_argument(
        "--fail-on-benchmark-error",
        action="store_true",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--child-spec",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stable-child-spec",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stable-process-index",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def stable_seed(identifier, operand):
    digest = hashlib.sha256(
        f"{identifier}/{operand}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def make_case_data(spec):
    if spec.alias != "none":
        return make_alias_case_data(spec)
    lhs_values = benchmark_cases.make_values(
        spec.topology.lhs_shape,
        spec.dtype,
        stable_seed(spec.identifier, "lhs"),
        spec.value_pattern,
    )
    lhs = benchmark_cases.make_layout(lhs_values, spec.lhs_layout)
    if spec.topology.rhs_scalar:
        rhs = np.dtype(spec.dtype).type(2)
    else:
        rhs_values = benchmark_cases.make_values(
            spec.topology.rhs_shape,
            spec.dtype,
            stable_seed(spec.identifier, "rhs"),
            spec.value_pattern,
        )
        rhs = benchmark_cases.make_layout(
            rhs_values, spec.rhs_layout
        )
    return lhs, rhs


def make_alias_case_data(spec):
    seed = stable_seed(spec.identifier, "shared")
    if spec.alias == "broadcast-row":
        storage = benchmark_cases.make_values(
            spec.topology.lhs_shape,
            spec.dtype,
            seed,
            spec.value_pattern,
        )
        lhs_view = storage
        rhs_view = storage[:1, :]
    else:
        length = spec.topology.lhs_shape[0]
        guard = length if spec.alias == "reversed" else 0
        storage_length = length + 2 * guard
        if spec.alias in ("shift-forward", "shift-backward"):
            storage_length += 1
        storage = benchmark_cases.make_values(
            (storage_length,),
            spec.dtype,
            seed,
            spec.value_pattern,
        )
        if (
            spec.alias == "shift-forward"
            and spec.mode == "in"
            and spec.operation == "div"
            and np.issubdtype(storage.dtype, np.integer)
        ):
            storage[:2] = (2, 1)
        logical = storage[guard:guard + length]
        if spec.alias == "self":
            lhs_view = logical
            rhs_view = logical
        elif spec.alias == "shift-forward":
            lhs_view = storage[1:]
            rhs_view = storage[:-1]
        elif spec.alias == "shift-backward":
            lhs_view = storage[:-1]
            rhs_view = storage[1:]
        elif spec.alias == "reversed":
            lhs_view = logical
            rhs_view = logical[::-1]
        else:
            raise ValueError(f"unsupported alias case: {spec.alias}")
    lhs = benchmark_cases.LayoutData(
        spec.lhs_layout, lhs_view, storage
    )
    rhs = benchmark_cases.LayoutData(
        spec.rhs_layout, rhs_view, storage
    )
    return lhs, rhs


def make_simple_array(layout):
    return ARRAY_TYPES[layout.view.dtype.name](array=layout.view)


def make_simple_scalar(value):
    dtype = np.asarray(value).dtype
    if dtype == np.dtype("complex64"):
        return solvcon.complex64(
            float(np.real(value)),
            float(np.imag(value)),
        )
    if dtype == np.dtype("complex128"):
        return solvcon.complex128(
            float(np.real(value)),
            float(np.imag(value)),
        )
    return value


def compare_arrays(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return "wrong-shape"
    if actual.dtype != expected.dtype:
        return "wrong-dtype"
    if (
        np.issubdtype(expected.dtype, np.floating)
        or np.issubdtype(expected.dtype, np.complexfloating)
    ):
        tolerance = 2e-5 if expected.dtype.itemsize <= 8 else 1e-12
        matches = np.allclose(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
        )
    else:
        matches = np.array_equal(actual, expected)
    return "match" if matches else "wrong-value"


def changed_storage_offsets(before, after):
    before = np.asarray(before).reshape(-1)
    after = np.asarray(after).reshape(-1)
    if (
        np.issubdtype(before.dtype, np.floating)
        or np.issubdtype(before.dtype, np.complexfloating)
    ):
        equal = (before == after) | (np.isnan(before) & np.isnan(after))
    else:
        equal = before == after
    return set(np.flatnonzero(~equal).tolist())


def linear_traversal_exceeds_span(layout):
    if layout.view.size == 0:
        return False
    itemsize = layout.view.dtype.itemsize
    strides = (
        stride // itemsize for stride in layout.view.strides
    )
    maximum = sum(
        (length - 1) * max(0, stride)
        for length, stride in zip(layout.view.shape, strides)
    )
    return layout.view.size - 1 > maximum


def legacy_shape_validation_hazard(spec):
    if spec.topology.rhs_scalar:
        return None
    lhs_shape = spec.topology.lhs_shape
    rhs_shape = spec.topology.rhs_shape
    if lhs_shape == rhs_shape:
        return None
    if len(lhs_shape) > len(rhs_shape):
        return "lhs rank exceeds rhs rank"
    if lhs_shape == rhs_shape[:len(lhs_shape)]:
        return "lhs shape is a prefix of rhs shape"
    return None


def expected_case_result(spec, lhs, rhs):
    expected_valid = spec.topology.numpy_valid
    if spec.mode == "in":
        expected_valid = expected_valid and spec.topology.inplace_valid
    if not expected_valid:
        try:
            result = benchmark_cases.operation_result(
                spec.operation,
                lhs.view,
                rhs if np.isscalar(rhs) else rhs.view,
                spec.dtype,
            )
        except (TypeError, ValueError):
            return None, False
        if spec.mode == "in" and result.shape != lhs.view.shape:
            return None, False
        raise RuntimeError(
            "case is marked invalid but NumPy accepts it: "
            f"{spec.identifier}"
        )
    expected = benchmark_cases.operation_result(
        spec.operation,
        lhs.view,
        rhs if np.isscalar(rhs) else rhs.view,
        spec.dtype,
    )
    if tuple(expected.shape) != spec.topology.result_shape:
        raise RuntimeError(
            "topology result shape does not match NumPy: "
            f"{spec.identifier}"
        )
    return expected, True


def array_preview(array):
    array = np.asarray(array)
    flat = array.reshape(-1)
    values = []
    for value in flat[:16]:
        if np.issubdtype(array.dtype, np.complexfloating):
            values.append({
                "real": float(np.real(value)),
                "imag": float(np.imag(value)),
            })
        elif np.issubdtype(array.dtype, np.bool_):
            values.append(bool(value))
        elif np.issubdtype(array.dtype, np.integer):
            values.append(int(value))
        else:
            values.append(float(value))
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.name,
        "values": values,
        "truncated": array.size > len(values),
    }


def implementation_method(spec, implementation):
    method = spec.operation
    if spec.mode == "in":
        method = f"i{method}"
    if implementation == "legacy_simd":
        method = f"{method}_simd"
    elif implementation == "planned":
        method = f"_planned_{method}"
    return method


def invoke_simple_array(spec, implementation, lhs_array, rhs_operand):
    method = implementation_method(spec, implementation)
    function = getattr(lhs_array, method)
    result = function(rhs_operand)
    return lhs_array if spec.mode == "in" else result


def audit_implementation(
    spec,
    implementation,
    lhs,
    rhs,
    expected,
    expected_valid,
):
    if (
        implementation == "legacy_simd"
        and (
            spec.topology.rhs_scalar
            or spec.operation not in (
                "add",
                "sub",
                "mul",
                "div",
            )
        )
    ):
        return {
            "status": "unavailable",
            "error_type": "",
            "error": "",
        }
    method = implementation_method(spec, implementation)
    if not hasattr(ARRAY_TYPES[spec.dtype], method):
        return {
            "status": "unavailable",
            "error_type": "",
            "error": "",
        }
    lhs_before = lhs.storage.copy()
    rhs_before = None
    if isinstance(rhs, benchmark_cases.LayoutData):
        rhs_before = rhs.storage.copy()
    try:
        lhs_array = make_simple_array(lhs)
    except Exception as error:
        return {
            "status": "construction-error",
            "error_type": type(error).__name__,
            "error": str(error),
            "operand": "lhs",
        }
    rhs_operand = rhs
    if isinstance(rhs, benchmark_cases.LayoutData):
        try:
            rhs_operand = make_simple_array(rhs)
        except Exception as error:
            return {
                "status": "construction-error",
                "error_type": type(error).__name__,
                "error": str(error),
                "operand": "rhs",
            }
    else:
        rhs_operand = make_simple_scalar(rhs)
    if implementation in ("legacy", "legacy_simd"):
        shape_hazard = legacy_shape_validation_hazard(spec)
        if shape_hazard:
            return {
                "status": "unsafe-shape-validation",
                "error_type": "",
                "error": (
                    "legacy shape equality does not compare both ranks"
                ),
                "details": {"reason": shape_hazard},
            }
        if expected_valid and spec.topology.legacy_shape_supported:
            unsafe_operands = []
            if linear_traversal_exceeds_span(lhs):
                unsafe_operands.append("lhs")
            if (
                isinstance(rhs, benchmark_cases.LayoutData)
                and linear_traversal_exceeds_span(rhs)
            ):
                unsafe_operands.append("rhs")
            if unsafe_operands:
                return {
                    "status": "unsafe-memory-access",
                    "error_type": "",
                    "error": (
                        "legacy linear traversal exceeds the supplied "
                        "layout span"
                    ),
                    "details": {"operands": unsafe_operands},
                }
    try:
        result = invoke_simple_array(
            spec,
            implementation,
            lhs_array,
            rhs_operand,
        )
    except Exception as error:
        if not expected_valid:
            status = "expected-error"
        elif (
            spec.operation == "div"
            and np.issubdtype(
                np.dtype(spec.dtype),
                np.complexfloating,
            )
            and isinstance(error, RuntimeError)
            and str(error) == "Division by zero in complex number"
            and np.any(
                (
                    rhs
                    if np.isscalar(rhs)
                    else rhs.view
                )
                == 0
            )
        ):
            status = "unsupported-ieee-division"
        elif (
            implementation in ("legacy", "legacy_simd")
            and not spec.topology.legacy_shape_supported
        ):
            status = "unsupported-broadcast"
        else:
            status = "unexpected-error"
        return {
            "status": status,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    if not expected_valid:
        return {
            "status": "unexpected-success",
            "error_type": "",
            "error": "",
        }

    actual = np.asarray(result.ndarray)
    status = compare_arrays(actual, expected)
    details = {}
    if status != "match":
        details = {
            "expected": array_preview(expected),
            "actual": array_preview(actual),
        }
    lhs_changes = changed_storage_offsets(lhs_before, lhs.storage)
    if spec.mode == "out" and lhs_changes:
        status = "input-mutation"
        details["changed_lhs_storage_offsets"] = sorted(lhs_changes)[:32]
    elif spec.mode == "in":
        outside = lhs_changes - lhs.touched_storage_offsets()
        if outside:
            status = "out-of-view-write"
            details["changed_outside_offsets"] = sorted(outside)[:32]
    if isinstance(rhs, benchmark_cases.LayoutData):
        rhs_changes = changed_storage_offsets(rhs_before, rhs.storage)
        shares_storage = np.shares_memory(lhs.storage, rhs.storage)
        if rhs_changes and not shares_storage:
            status = "rhs-mutation"
            details["changed_rhs_storage_offsets"] = sorted(
                rhs_changes
            )[:32]
    return {
        "status": status,
        "error_type": "",
        "error": "",
        "details": details,
    }


def numpy_call(spec, lhs, rhs):
    rhs_operand = rhs if np.isscalar(rhs) else rhs.view
    if spec.mode == "in":
        with np.errstate(all="ignore"):
            NUMPY_OPERATIONS[spec.operation](
                lhs.view,
                rhs_operand,
                out=lhs.view,
                casting="unsafe",
            )
        return lhs.view
    return benchmark_cases.operation_result(
        spec.operation,
        lhs.view,
        rhs_operand,
        spec.dtype,
    )


def simple_array_call(spec, implementation, lhs_array, rhs_operand):
    result = invoke_simple_array(
        spec,
        implementation,
        lhs_array,
        rhs_operand,
    )
    return result.ndarray


def supports_preallocated_output(spec):
    return (
        spec.mode == "out"
        and spec.alias == "none"
        and spec.operation in ("add", "sub", "mul", "div")
        and spec.topology.result_shape is not None
    )


def audit_preallocated_output(spec, lhs, rhs, expected):
    destination = np.empty(
        spec.topology.result_shape, dtype=spec.dtype
    )
    lhs_array = make_simple_array(lhs)
    rhs_operand = rhs
    if isinstance(rhs, benchmark_cases.LayoutData):
        rhs_operand = make_simple_array(rhs)
    else:
        rhs_operand = make_simple_scalar(rhs)
    destination_array = ARRAY_TYPES[spec.dtype](
        array=destination
    )
    try:
        getattr(
            lhs_array, f"_planned_{spec.operation}_to"
        )(rhs_operand, destination_array)
    except Exception as error:
        return {
            "status": "unexpected-error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    status = compare_arrays(destination, expected)
    details = {}
    if status != "match":
        details = {
            "expected": array_preview(expected),
            "actual": array_preview(destination),
        }
    return {
        "status": status,
        "error_type": "",
        "error": "",
        "details": details,
    }


def make_preallocated_timed_method(spec, method, lhs, rhs):
    destination = np.empty(
        spec.topology.result_shape, dtype=spec.dtype
    )
    if method == "numpy":
        operation = NUMPY_OPERATIONS[spec.operation]
        rhs_operand = rhs if np.isscalar(rhs) else rhs.view

        def function():
            return operation(
                lhs.view,
                rhs_operand,
                out=destination,
                casting="unsafe",
            )
    else:
        lhs_array = make_simple_array(lhs)
        rhs_operand = rhs
        if isinstance(rhs, benchmark_cases.LayoutData):
            rhs_operand = make_simple_array(rhs)
        else:
            rhs_operand = make_simple_scalar(rhs)
        destination_array = ARRAY_TYPES[spec.dtype](
            array=destination
        )
        method_function = getattr(
            lhs_array, f"_planned_{spec.operation}_to"
        )

        def planned_function():
            return method_function(
                rhs_operand, destination_array
            )

        function = planned_function
    return function, None, "out"


def time_preallocated_output(
    spec,
    method,
    lhs,
    rhs,
    args,
):
    function, reset, mode = make_preallocated_timed_method(
        spec, method, lhs, rhs
    )
    return timed_samples(
        function,
        reset,
        mode,
        args.samples,
        args.warmup,
        args.target_ms,
    )


def timed_samples(
    function,
    reset,
    mode,
    samples,
    warmup,
    target_ms,
):
    for _ in range(warmup):
        if reset:
            reset()
        function()
    if mode == "in":
        repeat = 1
    else:
        start = time.perf_counter_ns()
        function()
        elapsed = max(1, time.perf_counter_ns() - start)
        target_ns = int(target_ms * 1e6)
        repeat = max(1, min(10000, target_ns // elapsed))
    durations = []
    for _ in range(samples):
        if reset:
            reset()
        start = time.perf_counter_ns()
        for _ in range(repeat):
            function()
        elapsed = time.perf_counter_ns() - start
        durations.append(elapsed / repeat / 1e6)
    return {
        "repeat": repeat,
        "samples_ms": durations,
        "median_ms": statistics.median(durations),
        "minimum_ms": min(durations),
    }


def make_timed_case_method(spec, method, lhs, rhs):
    lhs_initial = lhs.storage.copy()
    if method == "numpy":
        operation = NUMPY_OPERATIONS[spec.operation]
        rhs_operand = rhs if np.isscalar(rhs) else rhs.view

        def function():
            if spec.mode == "in":
                operation(
                    lhs.view,
                    rhs_operand,
                    out=lhs.view,
                    casting="unsafe",
                )
                return lhs.view
            result = operation(lhs.view, rhs_operand)
            if result.dtype != np.dtype(spec.dtype):
                result = result.astype(spec.dtype)
            return result
    else:
        lhs_array = make_simple_array(lhs)
        rhs_operand = rhs
        if isinstance(rhs, benchmark_cases.LayoutData):
            rhs_operand = make_simple_array(rhs)
        else:
            rhs_operand = make_simple_scalar(rhs)
        function_method = getattr(
            lhs_array,
            implementation_method(spec, method),
        )

        def function():
            return function_method(rhs_operand)

    reset = None
    if spec.mode == "in":
        def reset_destination():
            lhs.storage[...] = lhs_initial
        reset = reset_destination
    return function, reset, spec.mode


def time_case_method(
    spec,
    method,
    lhs,
    rhs,
    args,
):
    function, reset, mode = make_timed_case_method(
        spec, method, lhs, rhs
    )
    return timed_samples(
        function,
        reset,
        mode,
        args.samples,
        args.warmup,
        args.target_ms,
    )


def balanced_method_order(methods, sequence):
    methods = tuple(methods)
    if not methods:
        return ()
    rotation = sequence % len(methods)
    ordered = methods[rotation:] + methods[:rotation]
    if (sequence // len(methods)) % 2:
        ordered = tuple(reversed(ordered))
    return ordered


def run_timed_block(timed_method, repeat):
    function = timed_method[0]
    reset = timed_method[1]
    if reset:
        reset()
    start = time.perf_counter_ns()
    for _ in range(repeat):
        function()
    return max(1, time.perf_counter_ns() - start)


def calibrate_timed_method(timed_method, target_ms):
    mode = timed_method[2]
    elapsed = run_timed_block(timed_method, 1)
    if mode == "in":
        repeat = 1
    else:
        target_ns = max(1, int(target_ms * 1e6))
        repeat = max(1, min(10000, target_ns // elapsed))
    return repeat, elapsed


def warm_stable_methods(methods, repeats, warmup_ms, sequence):
    target_ns = max(0, int(warmup_ms * 1e6))
    elapsed_by_method = {name: 0 for name in methods}
    blocks_by_method = {name: 0 for name in methods}
    while any(value < target_ns for value in elapsed_by_method.values()):
        order = balanced_method_order(methods, sequence)
        sequence += 1
        for name in order:
            if elapsed_by_method[name] >= target_ns:
                continue
            elapsed = run_timed_block(methods[name], repeats[name])
            elapsed_by_method[name] += elapsed
            blocks_by_method[name] += 1
    return elapsed_by_method, blocks_by_method


def make_stable_timed_methods(spec, preallocated_output):
    methods = {}
    for name in ("numpy", "planned"):
        lhs, rhs = make_case_data(spec)
        methods[name] = make_timed_case_method(
            spec, name, lhs, rhs
        )
    if preallocated_output and supports_preallocated_output(spec):
        for name in ("numpy", "planned"):
            lhs, rhs = make_case_data(spec)
            methods[f"{name}_to"] = make_preallocated_timed_method(
                spec, name, lhs, rhs
            )
    return methods


def run_stable_process(spec, args, process_index):
    observations = []
    calibration = []
    for round_index in range(args.stable_rounds):
        round_sequence = (
            process_index * args.stable_rounds * args.samples
            + round_index * args.samples
        )
        methods = make_stable_timed_methods(
            spec, args.preallocated_output
        )
        repeats = {}
        calibration_elapsed = {}
        calibration_order = balanced_method_order(
            methods, round_sequence
        )
        for order_index, name in enumerate(calibration_order):
            repeats[name], calibration_elapsed[name] = (
                calibrate_timed_method(methods[name], args.target_ms)
            )
            calibration.append({
                "process": process_index,
                "round": round_index,
                "method": name,
                "order": order_index,
                "repeat": repeats[name],
                "elapsed_ns": calibration_elapsed[name],
            })
        warmup_elapsed, warmup_blocks = warm_stable_methods(
            methods,
            repeats,
            args.warmup_ms,
            round_sequence,
        )
        for item in calibration[-len(methods):]:
            name = item["method"]
            item["warmup_elapsed_ns"] = warmup_elapsed[name]
            item["warmup_blocks"] = warmup_blocks[name]
        for sample_index in range(args.samples):
            sample_sequence = round_sequence + sample_index
            order = balanced_method_order(methods, sample_sequence)
            for order_index, name in enumerate(order):
                elapsed = run_timed_block(methods[name], repeats[name])
                observations.append({
                    "process": process_index,
                    "round": round_index,
                    "sample": sample_index,
                    "sequence": sample_sequence,
                    "order": order_index,
                    "method": name,
                    "repeat": repeats[name],
                    "elapsed_ns": elapsed,
                    "per_call_ns": elapsed / repeats[name],
                })
    return {
        "process": process_index,
        "pid": os.getpid(),
        "calibration": calibration,
        "observations": observations,
    }


def summarize_ratio(rounds, numerator, denominator):
    values = []
    for item in rounds:
        methods = item["methods"]
        if numerator in methods and denominator in methods:
            ratio = (
                methods[numerator]["median_ms"]
                / methods[denominator]["median_ms"]
            )
            item[f"{numerator}_over_{denominator}"] = ratio
            values.append(ratio)
    if not values:
        return None
    return {
        "round_count": len(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_stable_observations(observations):
    seen = set()
    by_method = collections.defaultdict(list)
    order_counts = collections.defaultdict(collections.Counter)
    by_round = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for observation in observations:
        key = (
            observation["process"],
            observation["round"],
            observation["sample"],
            observation["method"],
        )
        if key in seen:
            raise ValueError(f"duplicate stable observation: {key}")
        seen.add(key)
        value_ms = observation["per_call_ns"] / 1e6
        by_method[observation["method"]].append(value_ms)
        order_counts[observation["method"]][observation["order"]] += 1
        by_round[
            (observation["process"], observation["round"])
        ][observation["method"]].append(value_ms)
    methods = {
        name: {
            "sample_count": len(values),
            "median_ms": statistics.median(values),
            "minimum_ms": min(values),
            "maximum_ms": max(values),
        }
        for name, values in sorted(by_method.items())
    }
    rounds = []
    for (process_index, round_index), grouped in sorted(by_round.items()):
        rounds.append({
            "process": process_index,
            "round": round_index,
            "methods": {
                name: {
                    "sample_count": len(values),
                    "median_ms": statistics.median(values),
                }
                for name, values in sorted(grouped.items())
            },
        })
    ratios = {
        "normal": summarize_ratio(rounds, "numpy", "planned"),
        "reused": summarize_ratio(
            rounds, "numpy_to", "planned_to"
        ),
    }
    return {
        "methods": methods,
        "order_counts": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(order_counts.items())
        },
        "rounds": rounds,
        "ratios": ratios,
    }


def stable_child_command(spec, args, process_index):
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--timing", "stable",
        "--samples", str(args.samples),
        "--target-ms", str(args.target_ms),
        "--stable-rounds", str(args.stable_rounds),
        "--warmup-ms", str(args.warmup_ms),
        "--stable-process-index", str(process_index),
        "--stable-child-spec", serialize_case_spec(spec),
    ]
    if args.preallocated_output:
        command.append("--preallocated-output")
    return command


def run_stable_timing(spec, args):
    processes = []
    observations = []
    calibration = []
    for process_index in range(args.stable_processes):
        process = subprocess.run(
            stable_child_command(spec, args, process_index),
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode:
            raise RuntimeError(
                "stable timing child failed with return code "
                f"{process.returncode}: {process.stderr.strip()}"
            )
        result = json.loads(process.stdout)
        processes.append({
            "process": result["process"],
            "pid": result["pid"],
        })
        calibration.extend(result["calibration"])
        observations.extend(result["observations"])
    return {
        "policy": "balanced-interleaved-v1",
        "process_count": args.stable_processes,
        "round_count": args.stable_rounds,
        "samples_per_round": args.samples,
        "target_ms": args.target_ms,
        "warmup_ms": args.warmup_ms,
        "processes": processes,
        "calibration": calibration,
        "observations": observations,
        "summary": summarize_stable_observations(observations),
    }


def case_matches_filters(spec, args):
    filters = (
        ("operation", spec.operation),
        ("dtype", spec.dtype),
        ("mode", spec.mode),
        ("topology", spec.topology.name),
        ("lhs_layout", spec.lhs_layout),
        ("rhs_layout", spec.rhs_layout),
        ("value_pattern", spec.value_pattern),
        ("size", spec.size),
    )
    return all(
        getattr(args, name) is None or value in getattr(args, name)
        for name, value in filters
    )


def run_case(spec, args):
    try:
        lhs, rhs = make_case_data(spec)
        expected, expected_valid = expected_case_result(spec, lhs, rhs)
    except Exception as error:
        return {
            "id": spec.identifier,
            "status": "benchmark-error",
            "error_type": type(error).__name__,
            "error": str(error),
        }

    row = {
        "id": spec.identifier,
        "catalog": spec.catalog,
        "size": spec.size,
        "mode": spec.mode,
        "operation": spec.operation,
        "dtype": spec.dtype,
        "topology": spec.topology.name,
        "value_pattern": spec.value_pattern,
        "numpy_valid": expected_valid,
        "legacy_shape_supported": (
            spec.topology.legacy_shape_supported
        ),
        "result_shape": (
            list(spec.topology.result_shape)
            if spec.topology.result_shape is not None
            else None
        ),
        "lhs": lhs.descriptor(),
        "rhs": (
            {"layout": "scalar", "scalar": float(np.real(rhs))}
            if np.isscalar(rhs)
            else rhs.descriptor()
        ),
        "implementations": {},
        "timing": {},
    }
    for implementation in IMPLEMENTATIONS:
        if (
            args.implementation is not None
            and implementation not in args.implementation
        ):
            row["implementations"][implementation] = {
                "status": "unavailable",
                "error_type": "",
                "error": "not selected",
            }
            continue
        audit_lhs, audit_rhs = make_case_data(spec)
        audit = audit_implementation(
            spec,
            implementation,
            audit_lhs,
            audit_rhs,
            expected,
            expected_valid,
        )
        row["implementations"][implementation] = audit

    if (
        args.preallocated_output
        and expected_valid
        and supports_preallocated_output(spec)
        and row["implementations"]["planned"]["status"] == "match"
    ):
        audit_lhs, audit_rhs = make_case_data(spec)
        row["preallocated_output"] = audit_preallocated_output(
            spec, audit_lhs, audit_rhs, expected
        )

    if args.timing == "matched" and expected_valid:
        row["timing"]["numpy"] = time_case_method(
            spec,
            "numpy",
            lhs,
            rhs,
            args,
        )
        for implementation in IMPLEMENTATIONS:
            status = row["implementations"][implementation]["status"]
            if status != "match":
                continue
            timed_lhs, timed_rhs = make_case_data(spec)
            row["timing"][implementation] = time_case_method(
                spec,
                implementation,
                timed_lhs,
                timed_rhs,
                args,
            )
        if (
            row.get("preallocated_output", {}).get("status")
            == "match"
        ):
            for method in ("numpy", "planned"):
                timed_lhs, timed_rhs = make_case_data(spec)
                row["timing"][f"{method}_to"] = (
                    time_preallocated_output(
                        spec,
                        method,
                        timed_lhs,
                        timed_rhs,
                        args,
                    )
                )
    elif (
        args.timing == "stable"
        and expected_valid
        and row["implementations"]["planned"]["status"] == "match"
    ):
        try:
            row["timing"]["stable"] = run_stable_timing(spec, args)
        except Exception as error:
            row["timing"]["stable"] = {
                "status": "benchmark-error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    statuses = [
        result["status"]
        for result in row["implementations"].values()
    ]
    if "preallocated_output" in row:
        statuses.append(row["preallocated_output"]["status"])
    if row.get("timing", {}).get("stable", {}).get("status"):
        statuses.append(row["timing"]["stable"]["status"])
    row["status"] = (
        "bug" if any(status in BUG_STATUSES for status in statuses)
        else "ok"
    )
    return row


def requires_process_isolation(spec):
    dtype = np.dtype(spec.dtype)
    return (
        spec.alias != "none"
        and spec.operation == "div"
        and np.issubdtype(dtype, np.integer)
    )


def crash_row(spec, process, implementation):
    lhs, rhs = make_case_data(spec)
    implementations = {
        name: {
            "status": "unavailable",
            "error_type": "",
            "error": "not selected",
        }
        for name in IMPLEMENTATIONS
    }
    implementations[implementation] = {
        "status": "process-crash",
        "error_type": "ProcessExit",
        "error": f"return code {process.returncode}",
    }
    return {
        "id": spec.identifier,
        "catalog": spec.catalog,
        "size": spec.size,
        "mode": spec.mode,
        "operation": spec.operation,
        "dtype": spec.dtype,
        "topology": spec.topology.name,
        "value_pattern": spec.value_pattern,
        "numpy_valid": True,
        "legacy_shape_supported": (
            spec.topology.legacy_shape_supported
        ),
        "result_shape": list(spec.topology.result_shape),
        "lhs": lhs.descriptor(),
        "rhs": rhs.descriptor(),
        "implementations": implementations,
        "timing": {},
        "status": "bug",
    }


def serialize_case_spec(spec):
    topology = spec.topology
    return json.dumps({
        "catalog": spec.catalog,
        "size": spec.size,
        "topology": {
            "name": topology.name,
            "lhs_shape": list(topology.lhs_shape),
            "rhs_shape": (
                None
                if topology.rhs_shape is None
                else list(topology.rhs_shape)
            ),
            "result_shape": (
                None
                if topology.result_shape is None
                else list(topology.result_shape)
            ),
            "numpy_valid": topology.numpy_valid,
            "rhs_scalar": topology.rhs_scalar,
        },
        "operation": spec.operation,
        "dtype": spec.dtype,
        "mode": spec.mode,
        "lhs_layout": spec.lhs_layout,
        "rhs_layout": spec.rhs_layout,
        "alias": spec.alias,
        "value_pattern": spec.value_pattern,
    })


def deserialize_case_spec(payload):
    values = json.loads(payload)
    topology_values = values.pop("topology")
    for field in ("lhs_shape", "rhs_shape", "result_shape"):
        if topology_values[field] is not None:
            topology_values[field] = tuple(topology_values[field])
    values["topology"] = benchmark_cases.Topology(**topology_values)
    return benchmark_cases.CaseSpec(**values)


def run_isolated_case(spec, args):
    merged = None
    implementations = args.implementation or IMPLEMENTATIONS
    for implementation in implementations:
        command = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--timing",
            "none",
            "--implementation",
            implementation,
            "--child",
            "--child-spec",
            serialize_case_spec(spec),
        ]
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode:
            row = crash_row(spec, process, implementation)
        else:
            report = json.loads(process.stdout)
            row = report["cases"][0]
        if merged is None:
            merged = row
        else:
            merged["implementations"][implementation] = (
                row["implementations"][implementation]
            )
    statuses = [
        result["status"]
        for result in merged["implementations"].values()
    ]
    merged["status"] = (
        "bug" if any(status in BUG_STATUSES for status in statuses)
        else "ok"
    )
    return merged


def git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_dirty():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def build_metadata(args):
    return {
        "revision": git_revision(),
        "git_dirty": git_dirty(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "thread_variables": {
            name: os.environ[name] for name in THREAD_VARIABLES
        },
        "catalog": args.catalog,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "timing": args.timing,
        "record": args.record,
        "record_status": args.record_status,
        "max_cases": args.max_cases,
        "samples": args.samples,
        "warmup": args.warmup,
        "target_ms": args.target_ms,
        "stable_processes": args.stable_processes,
        "stable_rounds": args.stable_rounds,
        "warmup_ms": args.warmup_ms,
        "preallocated_output": args.preallocated_output,
        "filters": {
            name: getattr(args, name)
            for name in (
                "operation",
                "dtype",
                "mode",
                "topology",
                "lhs_layout",
                "rhs_layout",
                "value_pattern",
                "size",
            )
        },
    }


def empty_coverage():
    fields = (
        "topology",
        "size",
        "operation",
        "dtype",
        "mode",
        "value_pattern",
        "lhs_layout",
        "rhs_layout",
    )
    return {field: set() for field in fields}


def update_coverage(coverage, row):
    for field in (
        "topology",
        "size",
        "operation",
        "dtype",
        "mode",
        "value_pattern",
    ):
        if row.get(field) is not None:
            coverage[field].add(row[field])
    if row.get("lhs"):
        coverage["lhs_layout"].add(row["lhs"]["layout"])
    if row.get("rhs"):
        coverage["rhs_layout"].add(row["rhs"]["layout"])


def sorted_coverage(coverage):
    return {
        field: sorted(values)
        for field, values in coverage.items()
    }


def should_record(row, selected_statuses=None):
    if selected_statuses:
        statuses = {
            row["status"],
            *(
                result["status"]
                for result in row.get(
                    "implementations", {}
                ).values()
            ),
        }
        if "preallocated_output" in row:
            statuses.add(row["preallocated_output"]["status"])
        return bool(statuses.intersection(selected_statuses))
    if row["status"] == "bug":
        return True
    statuses = (
        result["status"]
        for result in row.get("implementations", {}).values()
    )
    if any(
        status.startswith("unsupported-") for status in statuses
    ):
        return True
    return (
        row.get("preallocated_output", {})
        .get("status", "")
        .startswith("unsupported-")
    )


def main():
    args = parse_arguments()
    if args.stable_child_spec is not None:
        spec = deserialize_case_spec(args.stable_child_spec)
        result = run_stable_process(
            spec, args, args.stable_process_index
        )
        print(json.dumps(result))
        return 0
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be smaller than shard count")
    if min(args.samples, args.stable_processes, args.stable_rounds) < 1:
        raise ValueError("timing counts must be positive")
    if min(args.target_ms, args.warmup_ms) < 0:
        raise ValueError("timing durations must be non-negative")
    if args.child_spec is not None:
        cases = (deserialize_case_spec(args.child_spec),)
    else:
        cases = (
            case
            for case in benchmark_cases.iter_case_specs(args.catalog)
            if case_matches_filters(case, args)
        )
        cases = benchmark_cases.shard_cases(
            cases, args.shard_index, args.shard_count
        )

    case_count = 0
    recorded_rows = []
    statuses = collections.Counter()
    implementation_statuses = {
        implementation: collections.Counter()
        for implementation in IMPLEMENTATIONS
    }
    preallocated_output_statuses = collections.Counter()
    coverage = empty_coverage()
    benchmark_error_count = 0
    for index, case in enumerate(cases):
        if args.max_cases is not None and index >= args.max_cases:
            break
        if (
            not args.child
            and args.progress_every
            and index
            and index % args.progress_every == 0
        ):
            print(
                f"completed {index} cases",
                file=sys.stderr,
                flush=True,
            )
        if not args.child and requires_process_isolation(case):
            row = run_isolated_case(case, args)
        else:
            row = run_case(case, args)
        case_count += 1
        statuses[row["status"]] += 1
        benchmark_error_count += row["status"] == "benchmark-error"
        update_coverage(coverage, row)
        for implementation in IMPLEMENTATIONS:
            status = (
                row.get("implementations", {})
                .get(implementation, {})
                .get("status", "missing")
            )
            implementation_statuses[implementation][status] += 1
        if "preallocated_output" in row:
            preallocated_output_statuses[
                row["preallocated_output"]["status"]
            ] += 1
        if (
            args.record == "all"
            or (
                args.record == "findings"
                and should_record(row, args.record_status)
            )
        ):
            recorded_rows.append(row)
    report = {
        "metadata": build_metadata(args),
        "summary": {
            "case_count": case_count,
            "recorded_case_count": len(recorded_rows),
            "statuses": dict(statuses),
            "coverage": sorted_coverage(coverage),
            "implementation_statuses": {
                name: dict(counts)
                for name, counts in implementation_statuses.items()
            },
            "preallocated_output_statuses": dict(
                preallocated_output_statuses
            ),
        },
        "cases": recorded_rows,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
        print(f"wrote {args.output}")
    else:
        print(rendered)
    bug_count = statuses.get("bug", 0)
    if args.fail_on_benchmark_error and benchmark_error_count:
        return 2
    if args.fail_on_bug and bug_count:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
