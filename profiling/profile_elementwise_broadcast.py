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
        choices=("none", "matched"),
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


def time_case_method(
    spec,
    method,
    lhs,
    rhs,
    args,
):
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
    return timed_samples(
        function,
        reset,
        spec.mode,
        args.samples,
        args.warmup,
        args.target_ms,
    )


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
    statuses = [
        result["status"]
        for result in row["implementations"].values()
    ]
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


def build_metadata(args):
    return {
        "revision": git_revision(),
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
        return bool(statuses.intersection(selected_statuses))
    if row["status"] == "bug":
        return True
    statuses = (
        result["status"]
        for result in row.get("implementations", {}).values()
    )
    return any(status.startswith("unsupported-") for status in statuses)


def main():
    args = parse_arguments()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be smaller than shard count")
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
