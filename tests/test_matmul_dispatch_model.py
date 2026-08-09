# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

from contrib import matmul_dispatch as tune


class MatmulDispatchModelTC(unittest.TestCase):

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
