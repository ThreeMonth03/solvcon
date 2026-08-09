# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import dataclasses
import math
import random

from .data import _stable_order


BOUNDARIES = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072,
              4096, 6144)
GEOMETRIC_MIDPOINTS = (12, 24, 48, 96, 192, 384, 768, 1536)
SHAPE_ANCHORS = tuple(sorted((*BOUNDARIES, *GEOMETRIC_MIDPOINTS)))
ASPECTS = {
    "square": (1, 1, 1),
    "low_inner": (8, 1, 8),
    "high_inner": (1, 4, 1),
    "tall": (4, 2, 1),
    "wide": (1, 2, 4),
}
LOW_INNER_SIZES = (1, 2, 4, 7, 8, 9, 16)
LOW_INNER_OUTER_RATIOS = ((1, 1), (4, 1), (1, 4))
SKINNY_OUTPUT_SMALL_AXES = (8, 16, 32)
SKINNY_OUTPUT_ANCHORS = (
    (128, 512),
    (512, 128),
    (512, 2048),
    (2048, 512),
)
EXTREME_LOW_INNER_SIZES = (1, 2, 4)
EXTREME_LOW_INNER_OUTER_AXES = (128, 512, 2048)
JITTER_GROUP_COUNT = 48
JITTER_OFFSETS = (-5, -3, -2, 2, 3, 5)


@dataclasses.dataclass(frozen=True)
class GemmCase:
    family: str
    group: str
    rows: int
    inner_size: int
    columns: int
    dtype: str
    lhs_order: str
    rhs_order: str

    @property
    def identifier(self):
        shape = f"{self.rows}x{self.inner_size}x{self.columns}"
        layout = f"{self.lhs_order.lower()}{self.rhs_order.lower()}"
        return f"{self.dtype}:{self.family}:{shape}:{layout}"


def _scaled_dimension(anchor, numerator, denominator=1):
    return max(2, anchor * numerator // denominator)


def make_shape_specs(max_dimension):
    specs = {}
    for boundary in SHAPE_ANCHORS:
        if boundary > max_dimension:
            continue
        for value in (boundary - 1, boundary, boundary + 1):
            if 2 <= value <= max_dimension:
                key = (value, value, value)
                specs[key] = ("square_boundary", f"square:{boundary}")

        for family, ratios in ASPECTS.items():
            scale = max(ratios)
            unit = max(2, boundary // scale)
            rows = _scaled_dimension(unit, ratios[0])
            inner_size = _scaled_dimension(unit, ratios[1])
            columns = _scaled_dimension(unit, ratios[2])
            if max(rows, inner_size, columns) > max_dimension:
                continue
            key = (rows, inner_size, columns)
            specs.setdefault(key, (family, f"{family}:{boundary}"))

        for lhs_ratio, rhs_ratio in LOW_INNER_OUTER_RATIOS:
            unit = max(2, boundary // max(lhs_ratio, rhs_ratio))
            rows = unit * lhs_ratio
            columns = unit * rhs_ratio
            for inner_size in LOW_INNER_SIZES:
                if max(rows, inner_size, columns) > max_dimension:
                    continue
                key = rows, inner_size, columns
                group = f"low_inner_grid:{boundary}"
                specs.setdefault(key, ("low_inner_grid", group))

    for small_axis in SKINNY_OUTPUT_SMALL_AXES:
        for inner_size, other_axis in SKINNY_OUTPUT_ANCHORS:
            if max(small_axis, inner_size, other_axis) > max_dimension:
                continue
            group = (
                f"skinny_output:{small_axis}:{inner_size}:"
                f"{other_axis}")
            specs.setdefault(
                (small_axis, inner_size, other_axis),
                ("skinny_output", group),
            )
            specs.setdefault(
                (other_axis, inner_size, small_axis),
                ("skinny_output", group),
            )

    for outer_axis in EXTREME_LOW_INNER_OUTER_AXES:
        if outer_axis > max_dimension:
            continue
        small_axis = outer_axis // 16
        for inner_size in EXTREME_LOW_INNER_SIZES:
            group = f"extreme_low_inner:{outer_axis}:{inner_size}"
            specs.setdefault(
                (small_axis, inner_size, outer_axis),
                ("extreme_low_inner", group),
            )
            specs.setdefault(
                (outer_axis, inner_size, small_axis),
                ("extreme_low_inner", group),
            )
    return tuple(
        (family, group, *shape)
        for shape, (family, group) in specs.items()
    )


def make_jitter_shape_specs(max_dimension, seed,
                            group_count=JITTER_GROUP_COUNT):
    rng = random.Random(seed)
    anchors = [value for value in SHAPE_ANCHORS
               if value <= max_dimension]
    shapes = []
    seen = set()

    def logarithmic_dimension():
        if max_dimension <= 2:
            return 2
        value = math.exp(rng.uniform(math.log(2),
                                     math.log(max_dimension)))
        return min(max_dimension, max(2, round(value)))

    def threshold_neighbor():
        if not anchors:
            return logarithmic_dimension()
        anchor = rng.choice(anchors)
        offset = rng.choice(JITTER_OFFSETS)
        return min(max_dimension, max(2, anchor + offset))

    attempts = 0
    while len(shapes) < group_count and attempts < group_count * 100:
        attempts += 1
        dimensions = [logarithmic_dimension() for _ in range(3)]
        dimensions[len(shapes) % 3] = threshold_neighbor()
        shape = tuple(dimensions)
        if shape in seen:
            continue
        seen.add(shape)
        index = len(shapes)
        shapes.append((
            "jitter",
            f"jitter:{index:02d}",
            *shape,
        ))
    return tuple(shapes)


def _estimated_bytes(shape, dtype):
    rows, inner_size, columns = shape
    item_size = 4 if dtype == "float32" else 8
    elements = (rows * inner_size + inner_size * columns +
                rows * columns)
    return elements * item_size


def make_gemm_cases(dtypes, layouts, max_dimension, max_bytes,
                    max_cases, seed):
    buckets = {}
    shape_specs = (
        make_shape_specs(max_dimension) +
        make_jitter_shape_specs(max_dimension, seed)
    )
    for family, group, rows, inner_size, columns in shape_specs:
        shape = rows, inner_size, columns
        for dtype in dtypes:
            if _estimated_bytes(shape, dtype) > max_bytes:
                continue
            for layout in layouts:
                case = GemmCase(
                    family=family,
                    group=group,
                    rows=rows,
                    inner_size=inner_size,
                    columns=columns,
                    dtype=dtype,
                    lhs_order=layout[0].upper(),
                    rhs_order=layout[1].upper(),
                )
                key = family, dtype, layout
                buckets.setdefault(key, []).append(case)
    for cases in buckets.values():
        cases.sort(key=lambda case: _stable_order(case.identifier, seed))
    keys = sorted(buckets, key=lambda key: _stable_order(str(key), seed))
    ordered = []
    position = 0
    while len(ordered) < max_cases:
        added = False
        for key in keys:
            cases = buckets[key]
            if position < len(cases):
                ordered.append(cases[position])
                added = True
                if len(ordered) == max_cases:
                    break
        if not added:
            break
        position += 1
    return tuple(ordered)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
