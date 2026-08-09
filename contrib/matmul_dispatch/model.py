# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import dataclasses
import math
import statistics

from .data import _snake_case, make_grouped_folds


OOF_FOLD_COUNT = 5
NUMERIC_FEATURES = (
    "rows",
    "columns",
    "inner_size",
    "batch_size",
    "minimum_dimension",
    "maximum_dimension",
    "lhs_elements",
    "rhs_elements",
    "output_elements",
    "contraction_work",
)
BOOLEAN_FEATURES = (
    "has_batch_axes",
    "lhs_reused",
    "rhs_reused",
    "lhs_zero_batch_stride",
    "rhs_zero_batch_stride",
)
CATEGORICAL_FEATURES = (
    "dtype",
    "backend",
    "lhs_layout",
    "rhs_layout",
    "eligible_kernels",
)
CPP_NUMERIC_EXPRESSIONS = {
    "rows": "facts.rows",
    "columns": "facts.columns",
    "inner_size": "facts.inner_size",
    "batch_size": "facts.batch_size",
    "minimum_dimension": (
        "std::min({facts.rows, facts.columns, facts.inner_size})"),
    "maximum_dimension": (
        "std::max({facts.rows, facts.columns, facts.inner_size})"),
    "lhs_elements": (
        "static_cast<std::size_t>(facts.rows) * facts.inner_size"),
    "rhs_elements": (
        "static_cast<std::size_t>(facts.inner_size) * facts.columns"),
    "output_elements": (
        "static_cast<std::size_t>(facts.rows) * facts.columns"),
    "contraction_work": (
        "static_cast<std::size_t>(facts.rows) * facts.columns * "
        "facts.inner_size"),
}
CPP_ENUM_TYPES = {
    "dtype": "MatmulDataType",
    "backend": "MatmulBackend",
    "lhs_layout": "MatmulLayout",
    "rhs_layout": "MatmulLayout",
}


@dataclasses.dataclass(frozen=True)
class Feature:
    name: str
    expression: str
    source: str
    category: str | None = None
    boolean: bool = False


@dataclasses.dataclass(frozen=True)
class Leaf:
    routes: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Branch:
    feature: Feature
    threshold: float
    left: "Branch | Leaf"
    right: "Branch | Leaf"


@dataclasses.dataclass(frozen=True)
class FittedTrees:
    features: tuple[Feature, ...]
    model: object
    stump_model: object
    tree: Branch | Leaf
    stump_tree: Branch | Leaf


def _cpp_enum_member(value):
    return "".join(word.capitalize() for word in value.split("_"))


def _category_expression(source, value):
    if source == "eligible_kernels":
        return f"eligible_kernel(MatmulKernel::{value})"
    enum_type = CPP_ENUM_TYPES[source]
    member = _cpp_enum_member(value)
    return f"facts.{source} == {enum_type}::{member}"


def make_features(records):
    features = [
        Feature(
            name=name,
            expression=CPP_NUMERIC_EXPRESSIONS[name],
            source=name,
        )
        for name in NUMERIC_FEATURES
    ]
    features.extend(
        Feature(
            name=name,
            expression=f"facts.{name}",
            source=name,
            boolean=True,
        )
        for name in BOOLEAN_FEATURES
    )
    for source in CATEGORICAL_FEATURES:
        if source == "eligible_kernels":
            values = sorted({value for record in records
                             for value in record[source]})
            prefix = "eligible"
        else:
            values = sorted({str(record["facts"][source])
                             for record in records})
            prefix = source
        for value in values:
            name = f"{prefix}_{_snake_case(value)}"
            features.append(Feature(
                name=name,
                expression=_category_expression(source, value),
                source=source,
                category=value,
                boolean=True,
            ))
    return tuple(features)


def _derived_fact(facts, name):
    derived = {
        "minimum_dimension",
        "maximum_dimension",
        "lhs_elements",
        "rhs_elements",
        "output_elements",
        "contraction_work",
    }
    if name not in derived:
        return facts[name]
    rows = int(facts["rows"])
    columns = int(facts["columns"])
    inner_size = int(facts["inner_size"])
    if name == "minimum_dimension":
        return min(rows, columns, inner_size)
    if name == "maximum_dimension":
        return max(rows, columns, inner_size)
    if name == "lhs_elements":
        return rows * inner_size
    if name == "rhs_elements":
        return inner_size * columns
    if name == "output_elements":
        return rows * columns
    if name == "contraction_work":
        return rows * columns * inner_size
    raise KeyError(name)


def feature_value(record, feature):
    if feature.source == "eligible_kernels":
        return int(feature.category in record[feature.source])
    facts = record["facts"]
    if feature.category is not None:
        return int(str(facts[feature.source]) == feature.category)
    value = _derived_fact(facts, feature.source)
    return int(value) if feature.boolean else value


def make_feature_matrix(records, features):
    return [[feature_value(record, feature) for feature in features]
            for record in records]


def fastest_route(record):
    return min(record["median_ns"],
               key=lambda route: (record["median_ns"][route], route))


def training_route(record, min_speedup):
    oracle = fastest_route(record)
    current = record["current_kernel"]
    timings = record["median_ns"]
    if timings[current] / timings[oracle] < min_speedup:
        return current
    return oracle


def training_weight(record):
    values = sorted(record["median_ns"].values())
    if len(values) < 2:
        return 1.0
    return min(20.0, max(1.0, values[1] / values[0]))


def geometric_mean(values):
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean needs positive values")
    return math.exp(math.fsum(math.log(value) for value in values) /
                    len(values))


def percentile(values, quantile):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return (ordered[lower] * (1 - weight) +
            ordered[upper] * weight)


def select_tree_route(record, node):
    while isinstance(node, Branch):
        value = feature_value(record, node.feature)
        node = node.left if value <= node.threshold else node.right
    timings = record["median_ns"]
    return next((route for route in node.routes if route in timings), None)


def evaluate_predictions(records, predictions):
    current_regrets = []
    auto_current_regrets = []
    policy_regrets = []
    speedups = []
    auto_speedup_proxies = []
    slowdowns = []
    fallback_count = 0
    for record, prediction in zip(records, predictions, strict=True):
        timings = record["median_ns"]
        oracle = min(timings.values())
        current_time = timings[record["current_kernel"]]
        auto_time = record.get("auto_median_ns", current_time)
        if prediction not in timings:
            policy_time = current_time
            fallback_count += 1
        else:
            policy_time = timings[prediction]
        current_regrets.append(current_time / oracle)
        auto_current_regrets.append(auto_time / oracle)
        policy_regrets.append(policy_time / oracle)
        speedups.append(current_time / policy_time)
        auto_speedup_proxies.append(auto_time / policy_time)
        slowdowns.append(policy_time / current_time)

    current_gap = statistics.fmean(
        value - 1 for value in current_regrets)
    policy_gap = statistics.fmean(
        value - 1 for value in policy_regrets)
    captured_gap = (1 - policy_gap / current_gap
                    if current_gap > 0 else 0.0)
    return {
        "samples": len(records),
        "oracle_geomean_regret": 1.0,
        "current_geomean_regret": geometric_mean(current_regrets),
        "auto_current_geomean_regret": geometric_mean(
            auto_current_regrets),
        "policy_geomean_regret": geometric_mean(policy_regrets),
        "policy_p95_regret": percentile(policy_regrets, 0.95),
        "policy_worst_regret": max(policy_regrets),
        "current_over_policy_speedup": geometric_mean(speedups),
        "auto_over_policy_forced_speedup_proxy": geometric_mean(
            auto_speedup_proxies),
        "policy_worst_slowdown_vs_current": max(slowdowns),
        "captured_oracle_gap": captured_gap,
        "ineligible_fallbacks": fallback_count,
    }


def model_tree(model, features):
    tree = model.tree_
    classes = tuple(str(value) for value in model.classes_)

    def build(node):
        if tree.children_left[node] == tree.children_right[node]:
            counts = tree.value[node].ravel().tolist()
            ranked = sorted(zip(counts, classes),
                            key=lambda item: (-item[0], item[1]))
            routes = tuple(route for count, route in ranked if count > 0)
            return Leaf(routes)
        feature = features[tree.feature[node]]
        return Branch(
            feature=feature,
            threshold=float(tree.threshold[node]),
            left=build(tree.children_left[node]),
            right=build(tree.children_right[node]),
        )

    return build(0)


def tree_as_json(node):
    if isinstance(node, Leaf):
        return {"routes": list(node.routes)}
    return {
        "feature": node.feature.name,
        "threshold": node.threshold,
        "left": tree_as_json(node.left),
        "right": tree_as_json(node.right),
    }


def _fit_trees(records, args, classifier):
    features = make_features(records)
    matrix = make_feature_matrix(records, features)
    labels = [
        training_route(record, args.min_speedup)
        for record in records
    ]
    weights = [training_weight(record) for record in records]
    common = {
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.seed,
    }
    model = classifier(max_depth=args.max_depth, **common)
    stump_model = classifier(max_depth=1, **common)
    model.fit(matrix, labels, sample_weight=weights)
    stump_model.fit(matrix, labels, sample_weight=weights)
    return FittedTrees(
        features=features,
        model=model,
        stump_model=stump_model,
        tree=model_tree(model, features),
        stump_tree=model_tree(stump_model, features),
    )


def _tree_predictions(fitted, records):
    predictions = [
        select_tree_route(record, fitted.tree)
        for record in records
    ]
    stump_predictions = [
        select_tree_route(record, fitted.stump_tree)
        for record in records
    ]
    return predictions, stump_predictions


def evaluate_grouped_oof(records, args, classifier):
    evaluated_records = []
    predictions = []
    stump_predictions = []
    fold_reports = []
    folds = make_grouped_folds(
        records, OOF_FOLD_COUNT, args.seed)
    for index, (train, validation) in enumerate(folds, 1):
        fitted = _fit_trees(train, args, classifier)
        fold_predictions, fold_stump_predictions = _tree_predictions(
            fitted, validation)
        evaluated_records.extend(validation)
        predictions.extend(fold_predictions)
        stump_predictions.extend(fold_stump_predictions)
        fold_reports.append({
            "fold": index,
            "train_records": len(train),
            "validation_records": len(validation),
            "train_groups": sorted({
                record["group"] for record in train}),
            "validation_groups": sorted({
                record["group"] for record in validation}),
            "decision_tree": evaluate_predictions(
                validation, fold_predictions),
            "single_threshold_stump": evaluate_predictions(
                validation, fold_stump_predictions),
        })
    return {
        "method": "grouped_5_fold_oof",
        "fold_count": OOF_FOLD_COUNT,
        "folds": fold_reports,
        "aggregate": {
            "decision_tree": evaluate_predictions(
                evaluated_records, predictions),
            "single_threshold_stump": evaluate_predictions(
                evaluated_records, stump_predictions),
        },
    }

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
