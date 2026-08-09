# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from contrib import matmul_dispatch as tune
from contrib.matmul_dispatch import environment
from contrib.matmul_dispatch import policy


def make_record(group, current, timings, family=None):
    return {
        "family": family or group.partition(":")[0],
        "group": group,
        "current_kernel": current,
        "median_ns": timings,
    }


class MatmulDispatchPolicyTC(unittest.TestCase):

    def test_package_tuning_sources_are_complete_and_sorted(self):
        package = Path(environment.__file__).resolve().parent
        expected = tuple(sorted(
            f"contrib/matmul_dispatch/{path.name}"
            for path in package.glob("*.py")
        ))
        self.assertEqual(expected, environment.PACKAGE_TUNING_SOURCES)

    def test_tuning_sources_match_current_checkout(self):
        sources = {"contrib/tune_matmul_dispatch.py": "current"}
        records = ({
            "sample_id": "matching",
            "environment": {"tuning_source_sha256": sources},
        },)
        with mock.patch.object(
                environment, "_source_hashes", return_value=sources):
            environment.validate_tuning_sources(records)

    def test_fit_rejects_stale_tuning_sources_before_validation(self):
        record = {
            "sample_id": "stale",
            "environment": {
                "tuning_source_sha256": {"removed.hpp": "old"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(
                generated_include=Path(directory) / "policy.inc",
                input=Path(directory) / "records.jsonl",
            )
            with (
                mock.patch.object(
                    policy, "load_dataset", return_value=([record], [])),
                mock.patch.object(
                    environment, "_source_hashes",
                    return_value={"current.hpp": "new"}),
                mock.patch.object(
                    policy, "evaluate_grouped_oof") as evaluate,
            ):
                with self.assertRaisesRegex(
                        ValueError, "stale: tuning sources"):
                    policy.fit_policy(args)
        evaluate.assert_not_called()

    def test_refit_happens_only_for_a_selected_oof_policy(self):
        records = [make_record("a", "GenericIjk", {"GenericIjk": 1})]
        fitted = tune.FittedTrees(
            features=(),
            model=None,
            stump_model=None,
            tree=tune.Leaf(("GenericIjk",)),
            stump_tree=tune.Leaf(("GenericIjk",)),
        )
        args = types.SimpleNamespace()
        with mock.patch.object(policy, "_fit_trees") as fit:
            result = policy._refit_policy(
                records, args, object, None, {}, Path("unused"))
        self.assertIsNone(result)
        fit.assert_not_called()

        with (
            mock.patch.object(
                policy, "_fit_trees", return_value=fitted) as fit,
            mock.patch.object(
                policy, "render_include", return_value="policy"),
            mock.patch.object(policy, "_write_text_atomic"),
        ):
            result = policy._refit_policy(
                records, args, object, "decision_tree", {},
                Path("policy.inc"),
            )
        self.assertIs(fitted, result)
        fit.assert_called_once_with(records, args, object)

    def test_codegen_uses_lazy_nested_eligibility_guards(self):
        rows = tune.Feature(
            name="rows", expression="facts.rows", source="rows")
        fixed_eligible = tune.Feature(
            name="eligible_fixed_ikj",
            expression=tune._category_expression(
                "eligible_kernels", "FixedIkj"),
            source="eligible_kernels",
            category="FixedIkj",
            boolean=True,
        )
        tree = tune.Branch(
            feature=rows,
            threshold=16.0,
            left=tune.Branch(
                feature=fixed_eligible,
                threshold=0.5,
                left=tune.Leaf(("GenericIjk",)),
                right=tune.Leaf(("FixedIkj", "GenericIjk")),
            ),
            right=tune.Leaf(("BlasGemm", "GenericIjk")),
        )
        scope = {
            "dimensions": {
                "rows": [8, 32],
                "columns": [8, 32],
                "inner_size": [8, 32],
                "batch_size": [1, 1],
            },
            "categories": {
                "dtype": ["float32"],
                "backend": ["cblas"],
                "lhs_layout": ["row_major"],
                "rhs_layout": ["row_major"],
            },
            "booleans": {
                name: [False]
                for name in tune.BOOLEAN_FEATURES
            },
            "loop_work_limit": 256,
        }
        generated = tune.render_include(tree, scope)
        self.assertIn(
            "static_assert(MATMUL_POLICY_SCHEMA_VERSION == 1);",
            generated,
        )
        self.assertIn("template <typename Eligible>", generated)
        self.assertIn("std::optional<MatmulKernel>", generated)
        self.assertIn("Eligible && eligible_kernel", generated)
        self.assertIn("facts.dtype == MatmulDataType::Float32", generated)
        self.assertIn("if ((facts.rows) <= 16)", generated)
        self.assertIn(
            "!(eligible_kernel(MatmulKernel::FixedIkj))", generated)
        self.assertIn("eligible(MatmulKernel::FixedIkj)", generated)
        self.assertIn("eligible(MatmulKernel::BlasGemm)", generated)
        self.assertIn("kernel == MatmulKernel::DynamicIkj", generated)
        self.assertIn("return eligible_kernel(kernel);", generated)
        self.assertNotIn("matmul_kernel_mask_type", generated)
        self.assertIn("contraction_work >\n            256", generated)
        fixed = generated.index("eligible(MatmulKernel::FixedIkj)")
        generic = generated.index(
            "eligible(MatmulKernel::GenericIjk)", fixed)
        self.assertLess(fixed, generic)
        work_limit = generated.index("contraction_work >")
        callback = generated.index("return eligible_kernel(kernel);")
        self.assertLess(work_limit, callback)
        self.assertTrue(all(len(line) <= 79
                            for line in generated.splitlines()))

    def test_fit_removes_stale_include_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            include = Path(directory) / "policy.inc"
            include.write_text("stale", encoding="utf-8")
            args = types.SimpleNamespace(
                generated_include=include,
                input=Path(directory) / "missing.jsonl",
            )
            with mock.patch.object(
                    policy, "load_dataset", side_effect=RuntimeError):
                with self.assertRaises(RuntimeError):
                    policy.fit_policy(args)
            self.assertFalse(include.exists())

    def test_passing_stump_is_selected_when_tree_adds_no_value(self):
        args = types.SimpleNamespace(
            min_holdout_samples=20,
            min_oracle_headroom=1.05,
            min_speedup=1.03,
            min_stump_speedup=1.005,
            max_geomean_regret=1.03,
            max_p95_regret=1.05,
            max_regression=1.10,
            min_captured_gap=0.70,
        )
        stump = {
            "samples": 20,
            "current_geomean_regret": 1.05,
            "current_over_policy_speedup": 1.030,
            "policy_geomean_regret": 1.02,
            "policy_p95_regret": 1.04,
            "policy_worst_slowdown_vs_current": 1.05,
            "captured_oracle_gap": 0.75,
        }
        tree = dict(stump)
        tree["current_over_policy_speedup"] = 1.031
        selected, _, _, _ = policy._select_policy(tree, stump, args)
        self.assertEqual("single_threshold_stump", selected)

        tree["policy_worst_slowdown_vs_current"] = 1.11
        stump["policy_worst_slowdown_vs_current"] = 1.11
        selected, tree_checks, stump_checks, _ = policy._select_policy(
            tree, stump, args)
        self.assertIsNone(selected)
        self.assertFalse(tree_checks["worst_slowdown_vs_current"])
        self.assertFalse(stump_checks["worst_slowdown_vs_current"])

        selected, tree_checks, stump_checks, _ = policy._select_policy(
            tree, stump, args, work_limit_censored=True)
        self.assertIsNone(selected)
        self.assertFalse(tree_checks["uncensored_work_limit"])
        self.assertFalse(stump_checks["uncensored_work_limit"])


if __name__ == "__main__":
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
