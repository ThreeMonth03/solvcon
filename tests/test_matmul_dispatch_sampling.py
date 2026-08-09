# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

from contrib import matmul_dispatch as tune


class MatmulDispatchSamplingTC(unittest.TestCase):

    def test_sampling_covers_boundary_neighbors_and_aspects(self):
        cases = tune.make_gemm_cases(
            ("float32",), ("cc",), 33, 1_000_000, 1000, 17)
        shapes = {
            (case.rows, case.inner_size, case.columns)
            for case in cases
        }
        self.assertTrue({(7, 7, 7), (8, 8, 8), (9, 9, 9)} <= shapes)
        self.assertTrue(any(case.family == "low_inner" for case in cases))
        low_inner = {
            case.inner_size
            for case in cases
            if case.family == "low_inner_grid"
        }
        self.assertEqual(set(tune.LOW_INNER_SIZES), low_inner)
        self.assertTrue(any(
            case.rows != case.columns
            for case in cases
            if case.family == "low_inner_grid"
        ))
        self.assertEqual(
            cases,
            tune.make_gemm_cases(
                ("float32",), ("cc",), 33, 1_000_000, 1000, 17),
        )

    def test_jitter_shapes_are_deterministic_and_irregular(self):
        specs = tune.make_jitter_shape_specs(257, 17)
        boundary_values = {
            boundary + offset
            for boundary in tune.BOUNDARIES
            for offset in (-1, 0, 1)
        }
        self.assertEqual(tune.JITTER_GROUP_COUNT, len(specs))
        self.assertEqual(specs, tune.make_jitter_shape_specs(257, 17))
        self.assertEqual(len(specs), len({spec[1] for spec in specs}))
        self.assertTrue(any(
            any(value not in boundary_values for value in spec[2:])
            for spec in specs
        ))

    def test_sampling_covers_midpoints_and_skinny_siblings(self):
        specs = tune.make_shape_specs(2048)
        shapes = {
            (rows, inner_size, columns): (family, group)
            for family, group, rows, inner_size, columns in specs
        }
        for anchor in tune.GEOMETRIC_MIDPOINTS:
            self.assertIn((anchor, anchor, anchor), shapes)
        self.assertIn((47, 47, 47), shapes)
        self.assertIn((49, 49, 49), shapes)

        for shape, sibling in (
            ((8, 512, 128), (128, 512, 8)),
            ((32, 2048, 512), (512, 2048, 32)),
            ((32, 2, 512), (512, 2, 32)),
        ):
            self.assertIn(shape, shapes)
            self.assertIn(sibling, shapes)
            self.assertEqual(shapes[shape], shapes[sibling])
        capped = tune.make_gemm_cases(
            ("float32",), ("cc",), 2048, 256 * 1024 * 1024,
            25, 17)
        self.assertEqual(25, len(capped))


if __name__ == "__main__":
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
