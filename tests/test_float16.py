# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

import solvcon as sc


class Float16ArrayTC(unittest.TestCase):

    def test_numpy_storage_is_zero_copy(self):
        source = np.arange(12, dtype="float16").reshape(3, 4)
        array = sc.SimpleArrayFloat16(array=source)

        self.assertEqual(2, array.nbytes // array.size)
        self.assertEqual(np.dtype("float16"), array.ndarray.dtype)
        view = memoryview(array)
        self.assertEqual("e", view.format)
        self.assertEqual(2, view.itemsize)
        self.assertTrue(np.shares_memory(source, np.asarray(view)))
        array[1, 2] = np.float16(17.5)
        self.assertEqual(np.float16(17.5), source[1, 2])

        source[2, 3] = np.float16(-4.25)
        self.assertEqual(-4.25, array[2, 3])

    def test_cpu_arithmetic_and_reductions(self):
        lhs_np = np.array([-3, -1.5, 2, 4.5], dtype="float16")
        rhs_np = np.array([1, 2, 3, 4], dtype="float16")
        lhs = sc.SimpleArrayFloat16(array=lhs_np.copy())
        rhs = sc.SimpleArrayFloat16(array=rhs_np.copy())

        np.testing.assert_array_equal((lhs_np + rhs_np),
                                      lhs.add(rhs).ndarray)
        np.testing.assert_array_equal((lhs_np - rhs_np),
                                      lhs.sub(rhs).ndarray)
        np.testing.assert_array_equal((lhs_np * rhs_np),
                                      lhs.mul(rhs).ndarray)
        np.testing.assert_array_equal((lhs_np / rhs_np),
                                      lhs.div(rhs).ndarray)
        np.testing.assert_array_equal(np.abs(lhs_np), lhs.abs().ndarray)
        self.assertEqual(float(lhs_np.sum(dtype="float16")), lhs.sum())
        self.assertEqual(float(lhs_np.mean(dtype="float16")), lhs.mean())
        self.assertEqual(float(lhs_np.var(dtype="float16")), lhs.var())
        self.assertEqual(float(lhs_np.std(dtype="float16")), lhs.std())
        self.assertEqual(float(np.median(lhs_np)), lhs.median())

    def test_cpu_matrix_helpers(self):
        np.testing.assert_array_equal(
            np.eye(3, dtype="float16"),
            sc.SimpleArrayFloat16.eye(3).ndarray)

        source = np.array([[1, 3], [5, 7]], dtype="float16")
        array = sc.SimpleArrayFloat16(array=source)
        np.testing.assert_array_equal(
            np.array([[1, 4], [4, 7]], dtype="float16"),
            array.symmetrize().ndarray)
        self.assertEqual(8.0, array.trace())

    def test_cpu_matmul_uses_float16_storage(self):
        lhs_np = np.arange(15, dtype="float16").reshape(3, 5) / 8
        rhs_np = np.arange(20, dtype="float16").reshape(5, 4) / 7
        lhs = sc.SimpleArrayFloat16(array=lhs_np)
        rhs = sc.SimpleArrayFloat16(array=rhs_np)

        result = lhs.matmul(rhs)

        self.assertEqual(np.dtype("float16"), result.ndarray.dtype)
        np.testing.assert_allclose(lhs_np @ rhs_np, result.ndarray,
                                   rtol=2e-3, atol=2e-3)

    def test_plex_collector_and_broadcast(self):
        plex = sc.SimpleArray((2, 3), "float16")
        self.assertIsInstance(plex.typed, sc.SimpleArrayFloat16)
        self.assertEqual(np.dtype("float16"), plex.typed.ndarray.dtype)

        collector = sc.SimpleCollectorFloat16()
        collector.push_back(1.25)
        collector.push_back(np.float16(-2.5))
        np.testing.assert_array_equal(
            np.array([1.25, -2.5], dtype="float16"),
            collector.as_array().ndarray)

        target = sc.SimpleArrayFloat16((2, 3))
        source = np.arange(6, dtype="float64").reshape(2, 3) / 3
        target[...] = source
        np.testing.assert_array_equal(source.astype("float16"),
                                      target.ndarray)

        target[0, 0] = 7.25
        self.assertEqual(np.float16(7.25), target.ndarray[0, 0])

        wider = sc.SimpleArrayFloat64((2, 3))
        wider[...] = target.ndarray
        np.testing.assert_array_equal(target.ndarray.astype("float64"),
                                      wider.ndarray)

        with self.assertRaisesRegex(
                RuntimeError,
                "Cannot convert between complex and non-complex types"):
            target[...] = np.ones((2, 3), dtype="complex64")

    def test_nan_ordering_and_reductions(self):
        source = np.array([2, np.nan, -1, 4], dtype="float16")
        array = sc.SimpleArrayFloat16(array=source.copy())

        self.assertEqual(np.argmin(source), array.argmin())
        self.assertEqual(np.argmax(source), array.argmax())
        array.sort()
        np.testing.assert_array_equal(np.sort(source), array.ndarray)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
