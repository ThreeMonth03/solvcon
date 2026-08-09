# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import dataclasses
import json
import sys
import time
from pathlib import Path

from .data import _write_text_atomic
from .environment import make_environment
from .measurement import (
    MeasurementConfig,
    _annotate_refinement,
    _merge_refinement,
    measure_case,
)
from .sampling import make_gemm_cases


REFINEMENT_GAP = 0.15
REFINEMENT_BUDGET_SHARE = 0.25
DEFAULT_RESULTS = Path("profiling/results/matmul_dispatch.jsonl")


def _measurement_config(args):
    return MeasurementConfig(
        seed=args.seed,
        warmups=args.warmups,
        minimum_samples=args.min_samples,
        maximum_samples=args.max_samples,
        tie_gap=args.tie_gap,
        loop_work_limit=args.loop_work_limit,
        target_sample_ns=int(args.target_sample_ms * 1_000_000),
        maximum_inner_repetitions=args.max_inner_repetitions,
    )


def _rewrite_jsonl(path, records):
    text = "".join(
        json.dumps(record, sort_keys=True) + "\n"
        for record in records
    )
    _write_text_atomic(path, text)


def _validate_collection_scope(args):
    if len(args.dtypes) != 1 or len(args.layouts) != 1:
        raise ValueError(
            "collect one dtype and one layout per calibration dataset")


def _collect_coarse(cases, output, mode, deadline, config,
                    environment, refinement_gap):
    completed = 0
    skip_reasons = {}
    output_records = []
    candidates = []
    with output.open(mode, encoding="utf-8") as stream:
        for index, case in enumerate(cases, 1):
            if time.monotonic() >= deadline:
                break
            record = measure_case(case, deadline, config, environment)
            if record is None:
                break
            output_records.append(record)
            if record["status"] == "skipped":
                reason = record["skip_reason"]
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                print(
                    f"[{index}/{len(cases)}] skipped {case.identifier}: "
                    f"{reason}",
                    file=sys.stderr,
                )
            else:
                if _annotate_refinement(record, refinement_gap):
                    candidates.append((case, record))
                completed += 1
                print(
                    f"[{index}/{len(cases)}] {case.identifier}",
                    file=sys.stderr,
                )
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
    return output_records, candidates, completed, skip_reasons


def _refine_candidates(candidates, deadline, config, environment):
    refinement_config = dataclasses.replace(
        config, seed=config.seed + 1)
    ordered = sorted(
        candidates,
        key=lambda item: item[1]["refinement"][
            "first_pass_top_two_gap"],
    )
    for index, (case, record) in enumerate(ordered, 1):
        if time.monotonic() >= deadline:
            break
        refined = measure_case(
            case, deadline, refinement_config, environment)
        if refined is None:
            break
        if refined["status"] != "measured":
            record["refinement"]["status"] = "unavailable"
            continue
        _merge_refinement(record, refined)
        print(
            f"[refine {index}/{len(ordered)}] {case.identifier}",
            file=sys.stderr,
        )
    for _, record in ordered:
        if record["refinement"]["status"] == "pending":
            record["refinement"]["status"] = "budget_exhausted"


def _refinement_status_counts(records):
    counts = {}
    for record in records:
        refinement = record.get("refinement")
        if refinement is None:
            continue
        status = refinement["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def collect_records(args):
    _validate_collection_scope(args)
    cases = make_gemm_cases(
        args.dtypes,
        args.layouts,
        args.max_dimension,
        args.max_bytes,
        args.max_cases,
        args.seed,
    )
    if not cases:
        raise RuntimeError("sampling constraints produced no GEMM cases")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "x"
    started = time.monotonic()
    deadline = started + args.budget_seconds
    coarse_deadline = deadline - (
        args.budget_seconds * REFINEMENT_BUDGET_SHARE)
    environment = make_environment()
    config = _measurement_config(args)
    refinement_gap = getattr(
        args, "refinement_gap", REFINEMENT_GAP)
    result = _collect_coarse(
        cases, output, mode, coarse_deadline, config,
        environment, refinement_gap)
    output_records, candidates, completed, skip_reasons = result
    _refine_candidates(candidates, deadline, config, environment)
    _rewrite_jsonl(output, output_records)
    if completed == 0:
        raise RuntimeError("no measured case completed within the budget")
    return {
        "output": str(output),
        "completed_cases": completed,
        "scheduled_cases": len(cases),
        "skipped_cases": sum(skip_reasons.values()),
        "skip_reasons": skip_reasons,
        "budget_seconds": args.budget_seconds,
        "refinement": {
            "threshold": refinement_gap,
            "budget_share": REFINEMENT_BUDGET_SHARE,
            "candidates": len(candidates),
            "status_counts": _refinement_status_counts(output_records),
        },
    }

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
