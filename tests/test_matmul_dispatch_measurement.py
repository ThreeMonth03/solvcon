# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import enum
import unittest

from contrib.matmul_dispatch import measurement


class Kernel(enum.Enum):
    GenericIjk = 0
    FixedIkj = 1
    BlasGemm = 2


def make_profile():
    return {
        "operation": "gemm",
        "dtype": "float32",
        "backend": "cblas",
        "rows": 8,
        "columns": 8,
        "inner_size": 8,
        "batch_size": 1,
        "has_batch_axes": False,
        "lhs_layout": "row_major",
        "rhs_layout": "row_major",
        "lhs_row_stride": 8,
        "lhs_inner_stride": 1,
        "rhs_inner_stride": 8,
        "rhs_column_stride": 1,
        "lhs_reused": False,
        "rhs_reused": False,
        "lhs_zero_batch_stride": False,
        "rhs_zero_batch_stride": False,
        "current_kernel": Kernel.FixedIkj,
        "eligible_kernels": (
            Kernel.GenericIjk,
            Kernel.FixedIkj,
            Kernel.BlasGemm,
        ),
    }


class MatmulDispatchMeasurementTC(unittest.TestCase):

    def test_profile_keeps_enum_objects_for_forced_calls(self):
        facts, kernels, current = measurement.normalize_profile(make_profile())
        self.assertEqual("gemm", facts["operation"])
        self.assertIs(Kernel.BlasGemm, kernels["BlasGemm"])
        self.assertEqual("FixedIkj", current)


if __name__ == "__main__":
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
