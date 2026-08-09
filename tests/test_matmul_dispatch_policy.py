# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest
from unittest import mock

from contrib import matmul_dispatch as tune
from contrib.matmul_dispatch import environment


class MatmulDispatchPolicyTC(unittest.TestCase):

    def test_tuning_sources_match_current_checkout(self):
        sources = {"contrib/tune_matmul_dispatch.py": "current"}
        records = ({
            "sample_id": "matching",
            "environment": {"tuning_source_sha256": sources},
        },)
        with mock.patch.object(
                environment, "_source_hashes", return_value=sources):
            environment.validate_tuning_sources(records)

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


if __name__ == "__main__":
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
