# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

from .data import SCHEMA_VERSION
from .model import (
    BOOLEAN_FEATURES,
    CPP_ENUM_TYPES,
    Leaf,
    _category_expression,
)


def _render_leaf(node, indent):
    lines = []
    for route in node.routes:
        lines.extend((
            f"{indent}if (eligible(MatmulKernel::{route}))",
            f"{indent}{{",
            f"{indent}    return MatmulKernel::{route};",
            f"{indent}}}",
        ))
    lines.append(f"{indent}return std::nullopt;")
    return lines


def _render_tree(node, indent):
    if isinstance(node, Leaf):
        return _render_leaf(node, indent)
    expression = node.feature.expression
    if node.feature.boolean and 0 <= node.threshold < 1:
        condition = f"!({expression})"
    else:
        condition = f"({expression}) <= {node.threshold:.17g}"
    lines = [f"{indent}if ({condition})", f"{indent}{{"]
    lines.extend(_render_tree(node.left, indent + "    "))
    lines.extend((f"{indent}}}", f"{indent}else", f"{indent}{{"))
    lines.extend(_render_tree(node.right, indent + "    "))
    lines.append(f"{indent}}}")
    return lines


def make_codegen_scope(records, loop_work_limit):
    facts = [record["facts"] for record in records]
    dimensions = {
        name: [
            min(int(value[name]) for value in facts),
            max(int(value[name]) for value in facts),
        ]
        for name in ("rows", "columns", "inner_size", "batch_size")
    }
    categories = {
        name: sorted({str(value[name]) for value in facts})
        for name in CPP_ENUM_TYPES
    }
    booleans = {
        name: sorted({bool(value[name]) for value in facts})
        for name in BOOLEAN_FEATURES
    }
    return {
        "dimensions": dimensions,
        "categories": categories,
        "booleans": booleans,
        "loop_work_limit": loop_work_limit,
    }


def _render_scope(scope, indent):
    lines = [
        f"{indent}if (facts.operation != MatmulOperation::Gemm)",
        f"{indent}{{",
        f"{indent}    return std::nullopt;",
        f"{indent}}}",
    ]
    for source, values in scope["categories"].items():
        conditions = [
            _category_expression(source, value)
            for value in values
        ]
        lines.append("")
        lines.append(f"{indent}bool const {source}_supported =")
        for index, condition in enumerate(conditions):
            ending = ";" if index == len(conditions) - 1 else " ||"
            lines.append(f"{indent}    {condition}{ending}")
        lines.extend((
            f"{indent}if (!{source}_supported)",
            f"{indent}{{",
            f"{indent}    return std::nullopt;",
            f"{indent}}}",
        ))
    lines.append("")
    for name, (minimum, maximum) in scope["dimensions"].items():
        lines.extend((
            f"{indent}if (facts.{name} < {minimum} || "
            f"facts.{name} > {maximum})",
            f"{indent}{{",
            f"{indent}    return std::nullopt;",
            f"{indent}}}",
        ))
    for name, values in scope["booleans"].items():
        if len(values) != 1:
            continue
        condition = f"!facts.{name}" if values[0] else f"facts.{name}"
        lines.extend((
            f"{indent}if ({condition})",
            f"{indent}{{",
            f"{indent}    return std::nullopt;",
            f"{indent}}}",
        ))
    lines.extend((
        "",
        f"{indent}auto const eligible =",
        f"{indent}    [&facts, &eligible_kernel](MatmulKernel kernel)",
        f"{indent}{{",
        f"{indent}    bool const calibrated_loop =",
        f"{indent}        kernel == MatmulKernel::GenericIjk ||",
        f"{indent}        kernel == MatmulKernel::DynamicIkj;",
        f"{indent}    std::size_t const contraction_work =",
        f"{indent}        static_cast<std::size_t>(facts.rows) *",
        f"{indent}        static_cast<std::size_t>(facts.columns) *",
        f"{indent}        static_cast<std::size_t>(facts.inner_size);",
        f"{indent}    if (calibrated_loop && contraction_work >",
        f"{indent}        {scope['loop_work_limit']})",
        f"{indent}    {{",
        f"{indent}        return false;",
        f"{indent}    }}",
        f"{indent}    return eligible_kernel(kernel);",
        f"{indent}}};",
        "",
    ))
    return lines


def render_include(node, scope):
    lines = [
        "static_assert(MATMUL_POLICY_SCHEMA_VERSION == "
        f"{SCHEMA_VERSION});",
        "",
        "template <typename Eligible>",
        "inline std::optional<MatmulKernel> select_calibrated_gemm(",
        "    MatmulFacts const & facts,",
        "    Eligible && eligible_kernel)",
        "{",
    ]
    lines.extend(_render_scope(scope, "    "))
    lines.extend(_render_tree(node, "    "))
    lines.extend((
        "}",
        "",
        "// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:",
        "",
    ))
    return "\n".join(lines)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
