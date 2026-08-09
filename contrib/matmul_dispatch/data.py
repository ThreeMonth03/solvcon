# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


SCHEMA_VERSION = 1


def _stable_order(value, seed):
    payload = f"{seed}:{value}".encode("ascii")
    return hashlib.sha256(payload).digest()


def _snake_case(value):
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def load_dataset(path):
    records = []
    skipped = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"line {line_number}: unsupported schema version")
            if record.get("status") == "skipped":
                skipped.append(record)
                continue
            if not record.get("correct"):
                raise ValueError(f"line {line_number}: incorrect result")
            timings = record.get("median_ns", {})
            current = record.get("current_kernel")
            if current not in timings or not timings:
                raise ValueError(
                    f"line {line_number}: incomplete route timings")
            records.append(record)
    if not records:
        raise ValueError("the input contains no records")
    return records, skipped


def dataset_loop_work_limit(records, skipped):
    limits = {
        record.get("loop_work_limit")
        for record in (*records, *skipped)
    }
    if None in limits:
        raise ValueError("input is missing loop_work_limit")
    if len(limits) != 1:
        raise ValueError("input mixes loop_work_limit values")
    limit = next(iter(limits))
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("loop_work_limit must be a positive integer")
    return limit


def has_work_limit_censoring(skipped):
    return any(
        re.fullmatch(r"current_.+_above_work_limit",
                     record.get("skip_reason", ""))
        for record in skipped
    )


def make_grouped_folds(records, fold_count, seed):
    grouped = {}
    for record in records:
        grouped.setdefault(record["group"], []).append(record)
    if len(grouped) < fold_count:
        raise ValueError(
            f"grouped {fold_count}-fold validation needs at least "
            f"{fold_count} shape groups")

    group_specs = []
    for group, members in grouped.items():
        families = {
            record.get("family") or
            str(record["group"]).partition(":")[0]
            for record in members
        }
        group_specs.append((group, members, families))
    group_specs.sort(
        key=lambda item: (-len(item[1]),
                          _stable_order(str(item[0]), seed)))

    fold_groups = [set() for _ in range(fold_count)]
    fold_sizes = [0] * fold_count
    fold_family_sizes = [{} for _ in range(fold_count)]
    for group, members, families in group_specs:
        size = len(members)
        fold_index = min(
            range(fold_count),
            key=lambda index: (
                fold_sizes[index],
                sum(fold_family_sizes[index].get(family, 0)
                    for family in families),
                len(fold_groups[index]),
                _stable_order(f"{group}:{index}", seed),
            ),
        )
        fold_groups[fold_index].add(group)
        fold_sizes[fold_index] += size
        for family in families:
            family_sizes = fold_family_sizes[fold_index]
            family_sizes[family] = family_sizes.get(family, 0) + size

    folds = []
    for validation_groups in fold_groups:
        train = [
            record for record in records
            if record["group"] not in validation_groups
        ]
        validation = [
            record for record in records
            if record["group"] in validation_groups
        ]
        folds.append((train, validation))
    return tuple(folds)


def calibration_target(records):
    targets = {
        (
            str(record["facts"]["dtype"]),
            str(record["facts"]["lhs_layout"]),
            str(record["facts"]["rhs_layout"]),
        )
        for record in records
    }
    if len(targets) != 1:
        raise ValueError(
            "input mixes dtype/layout calibration targets; fit each "
            "target separately")
    dtype, lhs_layout, rhs_layout = next(iter(targets))
    return {
        "dtype": dtype,
        "lhs_layout": lhs_layout,
        "rhs_layout": rhs_layout,
    }


def _write_text_atomic(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent,
                prefix=f".{output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
