# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import collections
import json
import pathlib


BUG_STATUSES = (
    "construction-error",
    "unexpected-error",
    "unexpected-success",
    "wrong-shape",
    "wrong-dtype",
    "wrong-value",
    "input-mutation",
    "rhs-mutation",
    "out-of-view-write",
    "process-crash",
    "unsafe-memory-access",
    "unsafe-shape-validation",
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Render an elementwise benchmark JSON report."
    )
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--representatives", type=int, default=20)
    return parser.parse_args()


def format_shape(shape):
    if shape is None:
        return "invalid"
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(value) for value in shape) + ")"


def format_operand(operand):
    if operand["layout"] == "scalar":
        return "scalar"
    shape = format_shape(operand["shape"])
    strides = format_shape(operand["strides"])
    return f"`{operand['layout']}` {shape}, strides {strides}"


def reproduction_command(row, catalog):
    arguments = (
        ("operation", row["operation"]),
        ("dtype", row["dtype"]),
        ("mode", row["mode"]),
        ("topology", row["topology"]),
        ("lhs-layout", row["lhs"]["layout"]),
        ("rhs-layout", row["rhs"]["layout"]),
        ("value-pattern", row["value_pattern"]),
        ("size", row["size"]),
    )
    command = [
        "PYTHONPATH=.:profiling python3",
        "profiling/profile_elementwise_broadcast.py",
        f"--catalog {catalog}",
    ]
    command.extend(
        f"--{name} {value}" for name, value in arguments
    )
    command.append("--max-cases 1 --output /tmp/elementwise-case.json")
    return " \\\n    ".join(command)


def representative_score(family, row):
    score = 100 + min(row["size"], 20) if row["size"] else 0
    if row["lhs"]["layout"] == "c":
        score += 10
    if row["rhs"]["layout"] == "c":
        score += 10
    if family == "shape-equality-ignores-rank":
        reason = (
            row["implementations"]["legacy"]
            .get("details", {})
            .get("reason", "")
        )
        if "prefix" in reason:
            score += 100
    if (
        family == "empty-array-construction"
        and row["topology"] in ("empty-leading", "empty-inner")
    ):
        score += 100
    return score


def bug_family(row):
    legacy = row["implementations"]["legacy"]
    status = legacy["status"]
    shapes = (
        row["lhs"].get("shape", []),
        row["rhs"].get("shape", []),
    )
    if status == "construction-error" and any(
        0 in shape for shape in shapes
    ):
        return "empty-array-construction"
    if status == "wrong-value" and row["topology"].startswith("alias-"):
        return "overlap-wrong-value"
    if status == "process-crash":
        return "overlap-process-crash"
    if status == "unsafe-memory-access":
        return "linear-traversal-exceeds-layout-span"
    if status == "unsafe-shape-validation":
        return "shape-equality-ignores-rank"
    if status == "wrong-value":
        mode = "in-place" if row["mode"] == "in" else "out-of-place"
        return f"{mode}-stride-traversal"
    if status == "out-of-view-write":
        return "in-place-out-of-view-write"
    return status


def append_summary(lines, report):
    summary = report["summary"]
    metadata = report["metadata"]
    lines.extend([
        "# Elementwise broadcasting audit",
        "",
        "## Environment",
        "",
        f"- Revision: `{metadata['revision']}`",
        f"- Platform: `{metadata['platform']}`",
        f"- Python: `{metadata['python']}`",
        f"- NumPy: `{metadata['numpy']}`",
        f"- Catalog: `{metadata['catalog']}`",
        (
            f"- Shard: `{metadata['shard_index']}` of "
            f"`{metadata['shard_count']}`"
        ),
        "",
        "## Result summary",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Cases | {summary['case_count']} |",
        f"| Recorded cases | {summary['recorded_case_count']} |",
    ])
    for status, count in sorted(summary["statuses"].items()):
        lines.append(f"| Overall `{status}` | {count} |")
    for implementation, statuses in (
        summary["implementation_statuses"].items()
    ):
        for status, count in sorted(statuses.items()):
            lines.append(
                f"| {implementation} `{status}` | {count} |"
            )
    lines.append("")


def append_coverage(lines, summary):
    coverage = summary["coverage"]
    fields = {
        "Topologies": coverage["topology"],
        "Sizes": coverage["size"],
        "Operations": coverage["operation"],
        "Dtypes": coverage["dtype"],
        "Modes": coverage["mode"],
        "Value patterns": coverage["value_pattern"],
        "Lhs layouts": coverage["lhs_layout"],
        "Rhs layouts": coverage["rhs_layout"],
    }
    lines.extend([
        "## Catalog coverage",
        "",
        "| Axis | Distinct values | Values |",
        "| --- | ---: | --- |",
    ])
    for label, values in fields.items():
        rendered = ", ".join(f"`{value}`" for value in sorted(values))
        lines.append(f"| {label} | {len(values)} | {rendered} |")
    lines.append("")


def append_bug_groups(lines, rows):
    bugs = [
        row for row in rows
        if row.get("implementations", {})
        .get("legacy", {})
        .get("status") in BUG_STATUSES
    ]
    groups = collections.defaultdict(list)
    for row in bugs:
        groups[bug_family(row)].append(row)
    lines.extend([
        "## Legacy bug groups",
        "",
        "| Family | Cases | Topologies | Modes | Layout pairs |",
        "| --- | ---: | --- | --- | ---: |",
    ])
    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for family, grouped_rows in ordered:
        topologies = sorted({row["topology"] for row in grouped_rows})
        modes = sorted({row["mode"] for row in grouped_rows})
        layout_pairs = {
            (row["lhs"]["layout"], row["rhs"]["layout"])
            for row in grouped_rows
        }
        lines.append(
            f"| `{family}` | {len(grouped_rows)} | "
            f"{', '.join(f'`{value}`' for value in topologies)} | "
            f"{', '.join(f'`{value}`' for value in modes)} | "
            f"{len(layout_pairs)} |"
        )
    if not groups:
        lines.append("| None | 0 | | | 0 |")
    lines.append("")


def append_benchmark_errors(lines, rows):
    errors = [
        row for row in rows
        if row.get("status") == "benchmark-error"
    ]
    if not errors:
        return
    lines.extend([
        "## Benchmark errors",
        "",
        "These rows indicate a catalog or audit failure, not a SimpleArray "
        "finding. Fix them before interpreting the library results.",
        "",
        "| Case | Error |",
        "| --- | --- |",
    ])
    for row in errors:
        lines.append(
            f"| `{row['id']}` | `{row['error_type']}: "
            f"{row['error']}` |"
        )
    lines.append("")


def append_feature_gaps(lines, rows):
    gaps = collections.Counter(
        (row["topology"], row["mode"])
        for row in rows
        if row.get("implementations", {})
        .get("legacy", {})
        .get("status") == "unsupported-broadcast"
    )
    lines.extend([
        "## Expected legacy broadcasting gaps",
        "",
        "These rows are valid NumPy broadcasting but are rejected by the "
        "current same-shape-only legacy API. They are tracked separately "
        "from wrong results and storage corruption.",
        "",
        "| Topology | Mode | Cases |",
        "| --- | --- | ---: |",
    ])
    for signature, count in gaps.most_common():
        topology, mode = signature
        lines.append(f"| `{topology}` | `{mode}` | {count} |")
    if not gaps:
        lines.append("| None | | 0 |")
    lines.append("")


def append_representatives(lines, rows, catalog, limit):
    representatives = {}
    for row in rows:
        legacy = (
            row.get("implementations", {})
            .get("legacy", {})
        )
        if legacy.get("status") not in BUG_STATUSES:
            continue
        family = bug_family(row)
        current = representatives.get(family)
        if (
            current is None
            or representative_score(family, row)
            > representative_score(family, current)
        ):
            representatives[family] = row
    lines.extend([
        "## Representative reproductions",
        "",
    ])
    if not representatives:
        lines.extend(["None.", ""])
        return
    for index, (family, row) in enumerate(representatives.items()):
        if index >= limit:
            break
        legacy = row["implementations"]["legacy"]
        lines.extend([
            f"### `{family}`",
            "",
            f"- Case: `{row['id']}`",
            f"- Lhs: {format_operand(row['lhs'])}",
            f"- Rhs: {format_operand(row['rhs'])}",
            f"- Result shape: `{format_shape(row['result_shape'])}`",
        ])
        if legacy.get("error"):
            prefix = legacy.get("error_type")
            error = legacy["error"]
            if prefix:
                error = f"{prefix}: {error}"
            lines.append(
                f"- Error: `{error}`"
            )
        details = legacy.get("details", {})
        if details:
            lines.extend([
                "",
                "```json",
                json.dumps(details, indent=2),
                "```",
            ])
        lines.extend([
            "",
            "```console",
            reproduction_command(row, catalog),
            "```",
            "",
        ])


def append_timing(lines, rows):
    timed = [row for row in rows if row.get("timing")]
    if not timed:
        return
    lines.extend([
        "## Timing",
        "",
        (
            "| Case | NumPy (ms) | Legacy (ms) | SIMD (ms) | "
            "Planned (ms) | NumPy / SIMD | NumPy / planned |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in timed:
        numpy = row["timing"].get("numpy", {}).get("median_ms")
        legacy = row["timing"].get("legacy", {}).get("median_ms")
        simd = row["timing"].get(
            "legacy_simd", {}
        ).get("median_ms")
        planned = row["timing"].get("planned", {}).get("median_ms")
        if numpy is None:
            continue
        values = (legacy, simd, planned)
        rendered = [
            "" if value is None else f"{value:.6g}"
            for value in values
        ]
        simd_speedup = "" if simd is None else f"{numpy / simd:.3f}"
        planned_speedup = (
            "" if planned is None else f"{numpy / planned:.3f}"
        )
        lines.append(
            f"| `{row['id']}` | {numpy:.6g} | "
            f"{rendered[0]} | {rendered[1]} | {rendered[2]} | "
            f"{simd_speedup} | {planned_speedup} |"
        )
    lines.append("")


def main():
    args = parse_arguments()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    rows = report["cases"]
    lines = []
    append_summary(lines, report)
    append_coverage(lines, report["summary"])
    append_benchmark_errors(lines, rows)
    append_bug_groups(lines, rows)
    append_feature_gaps(lines, rows)
    append_representatives(
        lines,
        rows,
        report["metadata"]["catalog"],
        args.representatives,
    )
    append_timing(lines, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
