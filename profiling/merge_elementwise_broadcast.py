# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import collections
import copy
import json
import pathlib


COMPATIBLE_METADATA = (
    "revision",
    "platform",
    "machine",
    "python",
    "numpy",
    "thread_variables",
    "catalog",
    "shard_count",
    "timing",
    "record",
    "record_status",
    "max_cases",
    "samples",
    "warmup",
    "target_ms",
    "filters",
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Merge a complete elementwise benchmark shard set."
    )
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args()


def validate_reports(reports):
    if not reports:
        raise ValueError("at least one report is required")
    reference = reports[0]["metadata"]
    if reference["max_cases"] is not None:
        raise ValueError("cannot merge truncated shards")
    for report in reports[1:]:
        metadata = report["metadata"]
        for field in COMPATIBLE_METADATA:
            if metadata[field] != reference[field]:
                raise ValueError(f"incompatible shard metadata: {field}")
    shard_count = reference["shard_count"]
    shard_indices = {
        report["metadata"]["shard_index"] for report in reports
    }
    expected = set(range(shard_count))
    if len(reports) != shard_count or shard_indices != expected:
        raise ValueError(
            f"shard indices {sorted(shard_indices)} != "
            f"{sorted(expected)}"
        )


def merge_reports(reports):
    validate_reports(reports)
    metadata = copy.deepcopy(reports[0]["metadata"])
    metadata["shard_index"] = "merged"

    statuses = collections.Counter()
    implementation_statuses = collections.defaultdict(
        collections.Counter
    )
    coverage = collections.defaultdict(set)
    cases = []
    case_count = 0
    for report in reports:
        summary = report["summary"]
        case_count += summary["case_count"]
        statuses.update(summary["statuses"])
        for implementation, counts in (
            summary["implementation_statuses"].items()
        ):
            implementation_statuses[implementation].update(counts)
        for field, values in summary["coverage"].items():
            coverage[field].update(values)
        cases.extend(report["cases"])

    cases.sort(key=lambda case: case["id"])
    identifiers = [case["id"] for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("recorded case identifiers overlap between shards")
    return {
        "metadata": metadata,
        "summary": {
            "case_count": case_count,
            "recorded_case_count": len(cases),
            "statuses": dict(statuses),
            "coverage": {
                field: sorted(values)
                for field, values in coverage.items()
            },
            "implementation_statuses": {
                implementation: dict(counts)
                for implementation, counts in (
                    implementation_statuses.items()
                )
            },
        },
        "cases": cases,
    }


def main():
    args = parse_arguments()
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.inputs
    ]
    merged = merge_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
