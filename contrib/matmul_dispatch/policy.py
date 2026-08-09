# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import dataclasses
import json
from pathlib import Path

from .codegen import make_codegen_scope, render_include
from .data import (
    SCHEMA_VERSION,
    _write_text_atomic,
    calibration_target,
    dataset_loop_work_limit,
    has_work_limit_censoring,
    load_dataset,
)
from .environment import validate_tuning_sources
from .model import (
    _fit_trees,
    evaluate_grouped_oof,
    tree_as_json,
)


DEFAULT_REPORT = Path("profiling/results/matmul_dispatch_report.json")
DEFAULT_INCLUDE = Path(
    "profiling/results/matmul_dispatch.generated.inc")


def _write_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _policy_checks(metrics, args, work_limit_censored=False):
    return {
        "uncensored_work_limit": not work_limit_censored,
        "minimum_holdout_samples": (
            metrics["samples"] >= args.min_holdout_samples),
        "minimum_oracle_headroom": (
            metrics["current_geomean_regret"] >=
            args.min_oracle_headroom),
        "speedup_over_current": (
            metrics["current_over_policy_speedup"] >= args.min_speedup),
        "geomean_regret": (
            metrics["policy_geomean_regret"] <=
            args.max_geomean_regret),
        "p95_regret": (
            metrics["policy_p95_regret"] <= args.max_p95_regret),
        "worst_slowdown_vs_current": (
            metrics["policy_worst_slowdown_vs_current"] <=
            args.max_regression),
        "captured_oracle_gap": (
            metrics["captured_oracle_gap"] >= args.min_captured_gap),
    }


def _select_policy(metrics, stump_metrics, args,
                   work_limit_censored=False):
    tree_checks = _policy_checks(
        metrics, args, work_limit_censored)
    stump_checks = _policy_checks(
        stump_metrics, args, work_limit_censored)
    tree_passes = all(tree_checks.values())
    stump_passes = all(stump_checks.values())
    tree_over_stump = (
        metrics["current_over_policy_speedup"] /
        stump_metrics["current_over_policy_speedup"])
    if tree_passes and (
            not stump_passes or tree_over_stump >= args.min_stump_speedup):
        selected = "decision_tree"
    elif stump_passes:
        selected = "single_threshold_stump"
    else:
        selected = None
    return selected, tree_checks, stump_checks, tree_over_stump


def _environment_map(records):
    environments = {
        record["environment"]["fingerprint"]:
        record["environment"]
        for record in records
        if "environment" in record
    }
    if len(environments) > 1:
        raise ValueError(
            "input mixes environment fingerprints; fit each target "
            "separately")
    return environments


def _skip_reason_counts(skipped):
    counts = {}
    for record in skipped:
        reason = record["skip_reason"]
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _model_report(args, scope, fitted=None, selected=None):
    report = {
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "calibration_scope": scope,
        "final_refit": None,
    }
    if fitted is None:
        return report
    model = (fitted.model if selected == "decision_tree"
             else fitted.stump_model)
    tree = (fitted.tree if selected == "decision_tree"
            else fitted.stump_tree)
    report["final_refit"] = {
        "policy": selected,
        "trained_records": None,
        "actual_depth": int(model.get_depth()),
        "leaves": int(model.get_n_leaves()),
        "features": [
            dataclasses.asdict(feature)
            for feature in fitted.features
        ],
        "tree": tree_as_json(tree),
    }
    return report


def _refit_policy(records, args, classifier, selected, scope, include):
    if selected is None:
        return None
    fitted = _fit_trees(records, args, classifier)
    tree = (fitted.tree if selected == "decision_tree"
            else fitted.stump_tree)
    _write_text_atomic(include, render_include(tree, scope))
    return fitted


def fit_policy(args):
    include = Path(args.generated_include)
    include.unlink(missing_ok=True)

    records, skipped = load_dataset(args.input)
    validate_tuning_sources((*records, *skipped))
    loop_work_limit = dataset_loop_work_limit(records, skipped)
    work_limit_censored = has_work_limit_censoring(skipped)
    target = calibration_target((*records, *skipped))
    environments = _environment_map(records)

    from sklearn.tree import DecisionTreeClassifier

    validation = evaluate_grouped_oof(
        records, args, DecisionTreeClassifier)
    aggregate = validation["aggregate"]
    metrics = aggregate["decision_tree"]
    stump_metrics = aggregate["single_threshold_stump"]
    selected, tree_checks, stump_checks, tree_over_stump = _select_policy(
        metrics, stump_metrics, args, work_limit_censored)
    aggregate["tree_over_stump_speedup"] = tree_over_stump
    scope = make_codegen_scope(records, loop_work_limit)
    fitted = _refit_policy(
        records, args, DecisionTreeClassifier,
        selected, scope, include)

    model_report = _model_report(args, scope, fitted, selected)
    if fitted is not None:
        model_report["final_refit"]["trained_records"] = len(records)
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "records": len(records),
            "groups": len({record["group"] for record in records}),
            "skipped_records": len(skipped),
            "skip_reasons": _skip_reason_counts(skipped),
            "loop_work_limit": loop_work_limit,
            "work_limit_censored": work_limit_censored,
            "calibration_target": target,
        },
        "model": model_report,
        "environments": environments,
        "validation": validation,
        "go": selected is not None,
        "selected_policy": selected,
        "go_checks": {
            "decision_tree": tree_checks,
            "single_threshold_stump": stump_checks,
        },
        "generated_include": (
            str(include) if selected is not None else None),
    }
    _write_json(args.report, report)
    return report

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
