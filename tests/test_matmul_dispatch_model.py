# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import math
import types
import unittest
from unittest import mock

from contrib import matmul_dispatch as tune
from contrib.matmul_dispatch import model


def make_record(group, current, timings, family=None):
    return {
        "family": family or group.partition(":")[0],
        "group": group,
        "current_kernel": current,
        "median_ns": timings,
    }


class MatmulDispatchModelTC(unittest.TestCase):

    def test_regret_uses_current_route_for_ineligible_prediction(self):
        records = [
            make_record(
                "a", "GenericIjk",
                {"GenericIjk": 10, "BlasGemm": 5}),
            make_record(
                "b", "FixedIkj",
                {"GenericIjk": 5, "FixedIkj": 6}),
        ]
        metrics = tune.evaluate_predictions(
            records, ("BlasGemm", "MissingKernel"))
        self.assertEqual(1, metrics["ineligible_fallbacks"])
        self.assertAlmostEqual(
            math.sqrt(2), metrics["current_over_policy_speedup"])
        self.assertAlmostEqual(1.2, metrics["policy_worst_regret"])

    def test_route_metrics_use_forced_current_and_report_auto_proxy(self):
        record = make_record(
            "a", "GenericIjk",
            {"GenericIjk": 10, "BlasGemm": 5})
        record["auto_median_ns"] = 20
        metrics = tune.evaluate_predictions((record,), ("BlasGemm",))
        self.assertAlmostEqual(
            2.0, metrics["current_over_policy_speedup"])
        self.assertAlmostEqual(
            2.0, metrics["current_geomean_regret"])
        self.assertAlmostEqual(
            4.0, metrics["auto_current_geomean_regret"])
        self.assertAlmostEqual(
            4.0, metrics["auto_over_policy_forced_speedup_proxy"])

    def test_training_weight_uses_runner_up_not_worst_route(self):
        record = make_record(
            "a", "GenericIjk",
            {"GenericIjk": 5, "FixedIkj": 6, "BlasGemm": 1000})
        self.assertAlmostEqual(1.2, tune.training_weight(record))
        record["median_ns"] = {"GenericIjk": 5}
        self.assertEqual(1.0, tune.training_weight(record))

    def test_tree_training_keeps_current_route_for_small_gain(self):
        records = (
            make_record(
                "near", "GenericIjk",
                {"GenericIjk": 100, "BlasGemm": 98}),
            make_record(
                "clear", "GenericIjk",
                {"GenericIjk": 100, "BlasGemm": 90}),
        )

        class RecordingClassifier:
            labels = []

            def __init__(self, **kwargs):
                pass

            def fit(self, matrix, labels, sample_weight):
                self.labels.append(tuple(labels))
                return self

        args = types.SimpleNamespace(
            max_depth=5,
            min_samples_leaf=1,
            min_speedup=1.03,
            seed=9,
        )
        with (
            mock.patch.object(model, "make_features", return_value=()),
            mock.patch.object(
                model, "model_tree",
                return_value=tune.Leaf(("GenericIjk",))),
        ):
            tune._fit_trees(records, args, RecordingClassifier)
        self.assertEqual(
            [("GenericIjk", "BlasGemm")] * 2,
            RecordingClassifier.labels,
        )

    def test_tree_uses_first_measured_route_from_reached_leaf(self):
        rows = tune.Feature(
            name="rows", expression="facts.rows", source="rows")
        tree = tune.Branch(
            feature=rows,
            threshold=16.0,
            left=tune.Leaf(("FixedIkj", "BlasGemm")),
            right=tune.Leaf(("GenericIjk",)),
        )
        record = make_record("a", "BlasGemm", {"BlasGemm": 5})
        record["facts"] = {"rows": 8}
        self.assertEqual(
            "BlasGemm", tune.select_tree_route(record, tree))

    def test_dataset_rejects_mixed_loop_work_limits(self):
        measured = [{"loop_work_limit": 256}]
        skipped = [{"loop_work_limit": 512}]
        with self.assertRaisesRegex(ValueError, "mixes"):
            tune.dataset_loop_work_limit(measured, skipped)

    def test_dataset_rejects_mixed_dtype_or_layout_targets(self):
        first = {"facts": {
            "dtype": "float32",
            "lhs_layout": "row_major",
            "rhs_layout": "row_major",
        }}
        second = {"facts": {
            "dtype": "float64",
            "lhs_layout": "row_major",
            "rhs_layout": "row_major",
        }}
        with self.assertRaisesRegex(ValueError, "dtype/layout"):
            tune.calibration_target((first, second))


if __name__ == "__main__":
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
