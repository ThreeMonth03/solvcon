# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import dataclasses
import hashlib
import math
import random
import re
import statistics
import time
from collections.abc import Mapping

from .data import SCHEMA_VERSION, _snake_case, _stable_order
from .environment import environment_with_backend


AUTO_LANE = "__auto__"
PROFILE_FIELDS = (
    "operation",
    "dtype",
    "backend",
    "rows",
    "columns",
    "inner_size",
    "batch_size",
    "has_batch_axes",
    "lhs_layout",
    "rhs_layout",
    "lhs_row_stride",
    "lhs_inner_stride",
    "rhs_inner_stride",
    "rhs_column_stride",
    "lhs_reused",
    "rhs_reused",
    "lhs_zero_batch_stride",
    "rhs_zero_batch_stride",
)


@dataclasses.dataclass(frozen=True)
class MeasurementConfig:
    seed: int
    warmups: int
    minimum_samples: int
    maximum_samples: int
    tie_gap: float
    loop_work_limit: int
    target_sample_ns: int
    maximum_inner_repetitions: int


def kernel_name(kernel):
    name = getattr(kernel, "name", None)
    if name is None:
        name = str(kernel).rsplit(".", 1)[-1]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"invalid kernel name: {name!r}")
    return name


def _profile_value(profile, name):
    if isinstance(profile, Mapping):
        return profile[name]
    return getattr(profile, name)


def normalize_profile(profile):
    missing = []
    for name in (*PROFILE_FIELDS, "current_kernel", "eligible_kernels"):
        try:
            _profile_value(profile, name)
        except (AttributeError, KeyError):
            missing.append(name)
    if missing:
        fields = ", ".join(missing)
        raise RuntimeError(f"matmul profile is missing fields: {fields}")

    facts = {name: _profile_value(profile, name)
             for name in PROFILE_FIELDS}
    if str(facts["operation"]).lower() != "gemm":
        raise RuntimeError("the dispatch tuner currently accepts GEMM only")
    kernels = tuple(_profile_value(profile, "eligible_kernels"))
    kernel_objects = {kernel_name(kernel): kernel for kernel in kernels}
    if len(kernel_objects) != len(kernels):
        raise RuntimeError("eligible kernel names are not unique")
    current = kernel_name(_profile_value(profile, "current_kernel"))
    if current not in kernel_objects:
        raise RuntimeError("the current kernel is not eligible")
    return facts, kernel_objects, current


def _make_operand(np, rng, shape, dtype, order):
    data = rng.random(shape, dtype=dtype)
    return np.array(data, dtype=dtype, order=order, copy=True)


def _make_arrays(case):
    import numpy as np
    import solvcon

    seed_bytes = hashlib.sha256(case.identifier.encode("ascii")).digest()
    data_seed = int.from_bytes(seed_bytes[:8], "little")
    rng = np.random.default_rng(data_seed)
    lhs = _make_operand(
        np, rng, (case.rows, case.inner_size), case.dtype,
        case.lhs_order)
    rhs = _make_operand(
        np, rng, (case.inner_size, case.columns), case.dtype,
        case.rhs_order)
    array_type = {
        "float32": solvcon.SimpleArrayFloat32,
        "float64": solvcon.SimpleArrayFloat64,
    }[case.dtype]
    return lhs, rhs, array_type(array=lhs), array_type(array=rhs)


def _assert_correct(np, actual, expected, case, route):
    epsilon = np.finfo(case.dtype).eps
    try:
        np.testing.assert_allclose(
            actual.ndarray,
            expected,
            rtol=64 * epsilon,
            atol=64 * epsilon,
        )
    except AssertionError as exc:
        raise RuntimeError(
            f"{case.identifier}: {route} failed correctness") from exc


def _need_more_samples(timings, minimum, maximum, tie_gap):
    if any(len(values) < minimum for values in timings.values()):
        return True
    if all(len(values) >= maximum for values in timings.values()):
        return False
    medians = sorted(statistics.median(values)
                     for values in timings.values())
    if len(medians) < 2 or medians[0] == 0:
        return False
    return medians[1] / medians[0] - 1 <= tie_gap


def _run_lane(lhs, rhs, kernel, repetitions):
    result = None
    for _ in range(repetitions):
        if kernel is None:
            result = lhs.matmul_planned(rhs)
        else:
            result = lhs._matmul_planned_forced(rhs, kernel)
    return result


def _calibrate_repetitions(
        lhs, rhs, kernel, target_ns, maximum_repetitions):
    started = time.perf_counter_ns()
    _run_lane(lhs, rhs, kernel, 1)
    elapsed = max(1, time.perf_counter_ns() - started)
    repetitions = math.ceil(target_ns / elapsed)
    return min(maximum_repetitions, max(1, repetitions))


def measure_case(case, deadline, config, environment):
    import numpy as np

    lhs, rhs, lhs_array, rhs_array = _make_arrays(case)
    profile = lhs_array._matmul_planned_profile(rhs_array)
    facts, kernel_objects, current = normalize_profile(profile)
    environment = environment_with_backend(
        environment, str(facts["backend"]))
    eligible_names = list(kernel_objects)
    work = case.rows * case.inner_size * case.columns
    if work > config.loop_work_limit:
        for route in ("GenericIjk", "DynamicIkj"):
            kernel_objects.pop(route, None)
    if current not in kernel_objects:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped",
            "skip_reason": (
                f"current_{_snake_case(current)}_above_work_limit"),
            "sample_id": case.identifier,
            "group": case.group,
            "family": case.family,
            "facts": facts,
            "current_kernel": current,
            "eligible_kernels": eligible_names,
            "loop_work_limit": config.loop_work_limit,
            "environment": environment,
        }
    expected_shape = case.rows, case.inner_size, case.columns
    actual_shape = (facts["rows"], facts["inner_size"],
                    facts["columns"])
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"profile shape {actual_shape} != requested {expected_shape}")

    expected = np.matmul(lhs, rhs)
    route_names = list(kernel_objects)
    lane_names = [*route_names, AUTO_LANE]
    lane_kernels = {**kernel_objects, AUTO_LANE: None}
    order_rng = random.Random(
        int.from_bytes(
            _stable_order(case.identifier, config.seed)[:8], "little"))
    for _ in range(config.warmups):
        order_rng.shuffle(lane_names)
        for lane in lane_names:
            if time.monotonic() >= deadline:
                return None
            result = _run_lane(
                lhs_array, rhs_array, lane_kernels[lane], 1)
            label = "Auto" if lane == AUTO_LANE else lane
            _assert_correct(np, result, expected, case, label)

    lane_repetitions = {}
    for lane in lane_names:
        if time.monotonic() >= deadline:
            return None
        lane_repetitions[lane] = _calibrate_repetitions(
            lhs_array,
            rhs_array,
            lane_kernels[lane],
            config.target_sample_ns,
            config.maximum_inner_repetitions,
        )

    timings = {route: [] for route in kernel_objects}
    timing_batches = {route: [] for route in kernel_objects}
    auto_timings = []
    auto_timing_batches = []
    while _need_more_samples(
            timings, config.minimum_samples, config.maximum_samples,
            config.tie_gap):
        if time.monotonic() >= deadline:
            return None
        order_rng.shuffle(lane_names)
        for lane in lane_names:
            lane_timings = (auto_timings if lane == AUTO_LANE
                            else timings[lane])
            if len(lane_timings) >= config.maximum_samples:
                continue
            if time.monotonic() >= deadline:
                return None
            started = time.perf_counter_ns()
            _run_lane(
                lhs_array,
                rhs_array,
                lane_kernels[lane],
                lane_repetitions[lane],
            )
            elapsed = time.perf_counter_ns() - started
            per_call = elapsed / lane_repetitions[lane]
            if lane == AUTO_LANE:
                auto_timing_batches.append(elapsed)
                auto_timings.append(per_call)
            else:
                timing_batches[lane].append(elapsed)
                timings[lane].append(per_call)

    medians = {
        route: statistics.median(values)
        for route, values in timings.items()
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "measured",
        "sample_id": case.identifier,
        "group": case.group,
        "family": case.family,
        "requested_layout": (
            case.lhs_order.lower() + case.rhs_order.lower()),
        "facts": facts,
        "current_kernel": current,
        "eligible_kernels": eligible_names,
        "measured_kernels": list(kernel_objects),
        "loop_work_limit": config.loop_work_limit,
        "timings_ns": timings,
        "timing_batches_ns": timing_batches,
        "inner_repetitions": {
            route: lane_repetitions[route]
            for route in route_names
        },
        "median_ns": medians,
        "auto_timings_ns": auto_timings,
        "auto_timing_batches_ns": auto_timing_batches,
        "auto_inner_repetitions": lane_repetitions[AUTO_LANE],
        "auto_median_ns": statistics.median(auto_timings),
        "correct": True,
        "environment": environment,
    }
    record["measurement_blocks"] = [
        _timing_block(record, "coarse")]
    return record


def _timing_block(record, phase, route_offsets=None, auto_offset=0):
    route_offsets = route_offsets or {}
    route_ranges = {
        route: [route_offsets.get(route, 0),
                route_offsets.get(route, 0) + len(values)]
        for route, values in record["timings_ns"].items()
    }
    return {
        "phase": phase,
        "route_ranges": route_ranges,
        "auto_range": [
            auto_offset,
            auto_offset + len(record["auto_timings_ns"]),
        ],
        "inner_repetitions": record["inner_repetitions"],
        "auto_inner_repetitions": record["auto_inner_repetitions"],
    }


def _first_pass_gap(record):
    medians = sorted(record["median_ns"].values())
    if len(medians) < 2 or medians[0] <= 0:
        return None
    return medians[1] / medians[0] - 1


def _annotate_refinement(record, threshold):
    gap = _first_pass_gap(record)
    candidate = gap is not None and gap < threshold
    record["refinement"] = {
        "first_pass_top_two_gap": gap,
        "threshold": threshold,
        "status": "pending" if candidate else "not_selected",
        "completed_blocks": 0,
        "added_route_samples": {},
        "added_auto_samples": 0,
    }
    return candidate


def _merge_refinement(record, refined):
    fields = ("sample_id", "current_kernel", "measured_kernels")
    if any(record[name] != refined[name] for name in fields):
        raise RuntimeError("refinement changed the measured route set")

    if "measurement_blocks" not in record:
        record["measurement_blocks"] = [
            _timing_block(record, "coarse")]
    route_offsets = {
        route: len(values)
        for route, values in record["timings_ns"].items()
    }
    auto_offset = len(record["auto_timings_ns"])
    block = _timing_block(
        refined, "refinement", route_offsets, auto_offset)
    added_route_samples = {}
    for route, values in refined["timings_ns"].items():
        record["timings_ns"][route].extend(values)
        record["timing_batches_ns"][route].extend(
            refined["timing_batches_ns"][route])
        added_route_samples[route] = len(values)
    record["auto_timings_ns"].extend(refined["auto_timings_ns"])
    record["auto_timing_batches_ns"].extend(
        refined["auto_timing_batches_ns"])
    record["median_ns"] = {
        route: statistics.median(values)
        for route, values in record["timings_ns"].items()
    }
    record["auto_median_ns"] = statistics.median(
        record["auto_timings_ns"])
    record["measurement_blocks"].append(block)

    refinement = record["refinement"]
    refinement["status"] = "completed"
    refinement["completed_blocks"] += 1
    refinement["added_route_samples"] = added_route_samples
    refinement["added_auto_samples"] = len(
        refined["auto_timings_ns"])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
