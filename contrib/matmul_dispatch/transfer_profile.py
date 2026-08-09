# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import dataclasses
import json
import math
from collections.abc import Mapping

from .data import _stable_order, calibration_target


LANDMARK_FACTS = (
    "operation",
    "dtype",
    "rows",
    "columns",
    "inner_size",
    "batch_size",
    "lhs_layout",
    "rhs_layout",
    "has_batch_axes",
    "lhs_reused",
    "rhs_reused",
    "lhs_zero_batch_stride",
    "rhs_zero_batch_stride",
)
MEASUREMENT_SOURCES = (
    "contrib/matmul_dispatch/collect.py",
    "contrib/matmul_dispatch/measurement.py",
    "cpp/solvcon/buffer/matmul.cpp",
    "cpp/solvcon/buffer/matmul.hpp",
    "cpp/solvcon/buffer/pymod/wrap_SimpleArray.cpp",
    "cpp/solvcon/buffer/pymod/wrap_SimpleArray.hpp",
)
DEFAULT_LANDMARK_MAX_WORK = 256 * 1024 * 1024
DEFAULT_LANDMARK_MAX_BYTES = 64 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class LandmarkManifest:
    keys: tuple[tuple, ...]
    routes: tuple[str, ...]


def calibration_key(record):
    facts = record["facts"]
    return tuple(facts.get(name) for name in LANDMARK_FACTS)


def _shape_vector(record):
    facts = record["facts"]
    rows = int(facts["rows"])
    columns = int(facts["columns"])
    inner_size = int(facts["inner_size"])
    batch_size = int(facts["batch_size"])
    lhs_elements = rows * inner_size
    rhs_elements = inner_size * columns
    output_elements = rows * columns
    work = max(1, batch_size * output_elements * inner_size)
    bytes_moved = max(
        1, batch_size * (lhs_elements + rhs_elements + output_elements))
    return (
        math.log2(rows + 1),
        math.log2(columns + 1),
        math.log2(inner_size + 1),
        math.log2(batch_size + 1),
        math.log2(lhs_elements + 1),
        math.log2(rhs_elements + 1),
        math.log2(output_elements + 1),
        math.log2(work + 1),
        math.log2(bytes_moved + 1),
        math.log2((rows + 1) / (inner_size + 1)),
        math.log2((columns + 1) / (inner_size + 1)),
        int(bool(facts.get("has_batch_axes"))),
        int(bool(facts.get("lhs_reused"))),
        int(bool(facts.get("rhs_reused"))),
        int(bool(facts.get("lhs_zero_batch_stride"))),
        int(bool(facts.get("rhs_zero_batch_stride"))),
    )


def _landmark_cost(record):
    facts = record["facts"]
    rows = int(facts["rows"])
    columns = int(facts["columns"])
    inner_size = int(facts["inner_size"])
    batch_size = int(facts["batch_size"])
    item_size = 4 if facts["dtype"] == "float32" else 8
    work = batch_size * rows * columns * inner_size
    elements = batch_size * (
        rows * inner_size + inner_size * columns + rows * columns)
    return work, elements * item_size


def _record_map(records):
    result = {}
    for record in records:
        key = calibration_key(record)
        if key in result:
            raise ValueError("a device profile repeats a calibration key")
        result[key] = record
    return result


def _measurement_identity(records):
    identities = set()
    missing = False
    for record in records:
        environment = record.get("environment")
        hashes = (environment.get("tuning_source_sha256")
                  if isinstance(environment, Mapping) else None)
        if not isinstance(hashes, Mapping):
            missing = True
            continue
        identity = tuple(
            (path, hashes.get(path)) for path in MEASUREMENT_SOURCES)
        if any(value is None for _, value in identity):
            raise ValueError(
                "a device profile is missing measurement source hashes")
        identities.add(identity)
    if missing and identities:
        raise ValueError("a device profile mixes measurement provenance")
    if len(identities) > 1:
        raise ValueError("a device profile mixes measurement sources")
    return next(iter(identities), None)


def _hardware_identity(records):
    identities = set()
    for record in records:
        environment = record.get("environment")
        if not isinstance(environment, Mapping):
            continue
        identity = {
            name: environment.get(name)
            for name in ("cpu", "machine", "platform", "blas",
                         "thread_environment", "threadpools")
        }
        if any(value is not None for value in identity.values()):
            identities.add(json.dumps(identity, sort_keys=True))
    if len(identities) > 1:
        raise ValueError("a device profile mixes hardware identities")
    return next(iter(identities), None)


def _validate_profiles(profiles, minimum_devices=2):
    if len(profiles) < minimum_devices:
        raise ValueError(
            f"transfer modeling needs at least {minimum_devices} devices")
    targets = {
        tuple(sorted(calibration_target(records).items()))
        for records in profiles.values()
    }
    if len(targets) != 1:
        raise ValueError("device profiles mix calibration targets")
    maps = {name: _record_map(records)
            for name, records in profiles.items()}
    key_sets = {frozenset(records) for records in maps.values()}
    if len(key_sets) != 1:
        raise ValueError(
            "device profiles must use the same calibration manifest")
    route_schemas = {
        tuple(
            (key, tuple(sorted(record["median_ns"])))
            for key, record in sorted(
                records.items(), key=lambda item: repr(item[0]))
        )
        for records in maps.values()
    }
    if len(route_schemas) != 1:
        raise ValueError(
            "device profiles must measure the same route schema")
    identities = {_measurement_identity(records)
                  for records in profiles.values()}
    if len(identities) != 1:
        raise ValueError(
            "device profiles must use the same measurement sources")
    hardware = [_hardware_identity(records)
                for records in profiles.values()]
    known_hardware = [identity for identity in hardware
                      if identity is not None]
    if known_hardware and len(known_hardware) != len(profiles):
        raise ValueError("device profiles mix missing hardware identities")
    if len(set(known_hardware)) != len(known_hardware):
        raise ValueError("device profiles must identify distinct hardware")
    return maps


def make_landmark_manifest(
        profiles, count, seed=1208,
        max_work=DEFAULT_LANDMARK_MAX_WORK,
        max_bytes=DEFAULT_LANDMARK_MAX_BYTES):
    maps = _validate_profiles(profiles, minimum_devices=1)
    common = set.intersection(*(set(records) for records in maps.values()))
    reference = next(iter(maps.values()))

    def within_budget(key):
        work, byte_count = _landmark_cost(reference[key])
        return work <= max_work and byte_count <= max_bytes

    common = {key for key in common if within_budget(key)}
    if count < 1 or count > len(common):
        raise ValueError("landmark count exceeds common device records")

    ordered = sorted(
        common, key=lambda key: _stable_order(repr(key), seed))
    routes = sorted({
        route
        for key in common
        for route in reference[key]["median_ns"]
    })

    import numpy as np

    matrix = np.asarray(
        [_shape_vector(reference[key]) for key in ordered],
        dtype="float64",
    )
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1
    normalized = (matrix - matrix.mean(axis=0)) / scale
    selected = []
    for route in routes:
        candidates = [
            index for index, key in enumerate(ordered)
            if route in reference[key]["median_ns"]
        ]
        if candidates:
            cheapest = min(
                candidates,
                key=lambda index: (
                    _landmark_cost(reference[ordered[index]]),
                    _stable_order(repr(ordered[index]), seed),
                ),
            )
            if cheapest not in selected:
                selected.append(cheapest)
    if len(selected) > count:
        raise ValueError("landmark count cannot cover every measured route")
    if not selected:
        selected.append(
            int(np.argmin(np.square(normalized).sum(axis=1))))
    while len(selected) < count:
        chosen = normalized[np.asarray(selected, dtype="int64")]
        distances = np.square(
            normalized[:, None, :] - chosen[None, :, :]).sum(axis=2)
        nearest = distances.min(axis=1)
        nearest[np.asarray(selected, dtype="int64")] = -1
        selected.append(int(np.argmax(nearest)))
    return LandmarkManifest(
        keys=tuple(ordered[index] for index in selected),
        routes=tuple(routes),
    )


def make_device_signature(records, manifest):
    record_map = _record_map(records)
    values = []
    for key in manifest.keys:
        if key not in record_map:
            raise ValueError("device profile is missing a landmark")
        record = record_map[key]
        timings = record["median_ns"]
        current = timings[record["current_kernel"]]
        facts = record["facts"]
        work = max(
            1,
            int(facts["rows"]) * int(facts["columns"]) *
            int(facts["inner_size"]) * int(facts["batch_size"]),
        )
        values.append(math.log(current / work))
        for route in manifest.routes:
            available = route in timings
            values.append(math.log(timings[route] / current)
                          if available else 0.0)
            values.append(float(available))

    import numpy as np

    return np.asarray(values, dtype="float64")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
