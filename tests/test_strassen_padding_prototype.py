# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import itertools
import unittest

import numpy as np

import solvcon


class StrassenPaddingTestBase:
    dtype = None
    array_type = None

    def test_all_residues(self):
        """Padding preserves every M, K, and N residue for both depths."""
        rng = np.random.default_rng(20260809)
        for depth in (1, 2):
            divisor = 1 << depth
            residues = itertools.product(range(divisor), repeat=3)
            for m_residue, k_residue, n_residue in residues:
                shape = tuple(
                    3 * divisor + residue
                    for residue in (m_residue, k_residue, n_residue))
                rows, inner_size, columns = shape
                lhs_data = rng.standard_normal(
                    (rows, inner_size), dtype=self.dtype)
                rhs_data = rng.standard_normal(
                    (inner_size, columns), dtype=self.dtype)
                lhs = self.array_type(array=lhs_data)
                rhs = self.array_type(array=rhs_data)

                with self.subTest(depth=depth, shape=shape):
                    result = lhs._matmul_strassen_control(
                        rhs, depth, True).ndarray
                    tolerance = 256 * np.finfo(self.dtype).eps
                    np.testing.assert_allclose(
                        result, lhs_data @ rhs_data,
                        rtol=tolerance, atol=tolerance)


class StrassenPaddingFloat32TC(
        StrassenPaddingTestBase, unittest.TestCase):
    dtype = 'float32'
    array_type = solvcon.SimpleArrayFloat32


class StrassenPaddingFloat64TC(
        StrassenPaddingTestBase, unittest.TestCase):
    dtype = 'float64'
    array_type = solvcon.SimpleArrayFloat64


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
