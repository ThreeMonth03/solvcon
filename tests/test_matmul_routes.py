# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

import solvcon as sc


class MatmulRouteTestBase(sc.testing.TestBase):

    def setUp(self):
        probe_data = self.make_data(4).reshape(2, 2)
        lhs = self.SimpleArray(array=probe_data)
        rhs = self.SimpleArray(array=probe_data.copy())
        self.has_blas_backend = any(
            route.kernel == 'blas_gemm'
            for route in lhs.matmul_routes(rhs))

    def make_data(self, size):
        dtype = np.dtype(self.dtype)
        real_dtype = np.empty(0, dtype=self.dtype).real.dtype.name
        values = np.arange(size, dtype=real_dtype)
        if np.issubdtype(dtype, np.complexfloating):
            values = values + 1j * values[::-1]
        return values.astype(self.dtype)

    def assert_route_results(
            self, lhs_data, rhs_data, expected_kernel=None):
        lhs = self.SimpleArray(array=lhs_data)
        rhs = self.SimpleArray(array=rhs_data)
        routes = lhs.matmul_routes(rhs)
        expected = np.atleast_1d(np.matmul(lhs_data, rhs_data))

        kernels = {route.kernel for route in routes}
        self.assertIn('generic', kernels)
        if expected_kernel is not None:
            if self.has_blas_backend:
                self.assertIn(expected_kernel, kernels)
            else:
                self.assertEqual({'generic'}, kernels)
        self.assertEqual(1, sum(route.selected_by_auto for route in routes))
        tol = 128 * np.finfo(lhs_data.real.dtype).eps * max(
            lhs_data.shape[-1], 1)
        for route in routes:
            with self.subTest(kernel=route.kernel):
                result = lhs.matmul_with_route(rhs, route)
                np.testing.assert_allclose(
                    result.ndarray, expected, rtol=tol, atol=tol)
        return lhs, rhs, routes

    def test_routes_force_kernels_below_thresholds(self):
        cases = (
            (
                'blas_dot',
                self.make_data(6)[::-1],
                self.make_data(6)[::-1],
            ),
            (
                'blas_gevm',
                self.make_data(4)[::-1],
                self.make_data(48).reshape(8, 6)[::2, ::2],
            ),
            (
                'blas_gemv',
                self.make_data(48).reshape(8, 6)[::2, ::2],
                self.make_data(6)[::2],
            ),
            (
                'blas_gemm',
                self.make_data(24).reshape(4, 6),
                self.make_data(48).reshape(6, 8),
            ),
        )
        for kernel, lhs_data, rhs_data in cases:
            with self.subTest(kernel=kernel):
                _, _, routes = self.assert_route_results(
                    lhs_data, rhs_data, kernel)
                selected = next(
                    route for route in routes if route.selected_by_auto)
                self.assertEqual('generic', selected.kernel)

    def test_winograd_route_is_structural(self):
        lhs_data = np.asfortranarray(
            self.make_data(24).reshape(4, 6), dtype=self.dtype)
        rhs_data = self.make_data(48).reshape(6, 8)
        _, _, routes = self.assert_route_results(
            lhs_data, rhs_data, 'winograd')

        if not self.has_blas_backend:
            return
        winograd = next(
            route for route in routes if route.kernel == 'winograd')
        self.assertTrue(winograd.eager_pack_lhs)
        self.assertFalse(winograd.eager_pack_rhs)
        self.assertFalse(winograd.selected_by_auto)

    def test_batched_route_reports_packing(self):
        lhs_data = self.make_data(
            2 * 4 * 6).reshape(2, 4, 6)[:, :, ::2]
        rhs_data = self.make_data(
            6 * 4).reshape(1, 6, 4)[:, ::2, ::-1]
        _, _, routes = self.assert_route_results(
            lhs_data, rhs_data, 'blas_gemm')

        if not self.has_blas_backend:
            return
        gemm = next(route for route in routes
                    if route.kernel == 'blas_gemm')
        self.assertTrue(gemm.scratch_pack_lhs)
        self.assertTrue(gemm.eager_pack_rhs)
        self.assertFalse(gemm.eager_pack_lhs)
        self.assertFalse(gemm.scratch_pack_rhs)

    def test_route_is_read_only_and_input_scoped(self):
        lhs_data = self.make_data(4).reshape(2, 2)
        rhs_data = self.make_data(4).reshape(2, 2)
        lhs, rhs, routes = self.assert_route_results(
            lhs_data, rhs_data, 'blas_gemm')
        route = routes[0]

        with self.assertRaises(TypeError):
            type(route)()
        with self.assertRaises(AttributeError):
            route.kernel = 'blas_gemm'

        other_lhs = self.SimpleArray(array=lhs_data.copy())
        with self.assertRaisesRegex(ValueError, 'route is not eligible'):
            other_lhs.matmul_with_route(rhs, route)

        reshaped_lhs = lhs.reshape((1, 2, 2))
        reshaped_rhs = rhs.reshape((1, 2, 2))
        with self.assertRaisesRegex(ValueError, 'route is not eligible'):
            reshaped_lhs.matmul_with_route(reshaped_rhs, route)
        with self.assertRaisesRegex(ValueError, 'route is not eligible'):
            reshaped_lhs.benchmark_matmul_route(
                reshaped_rhs, route, 2)

        result, elapsed_ns = lhs.benchmark_matmul(rhs, 2)
        expected = np.matmul(lhs_data, rhs_data)
        np.testing.assert_allclose(result.ndarray, expected)
        self.assertGreater(elapsed_ns, 0)
        for benchmark_route in routes:
            with self.subTest(benchmark_kernel=benchmark_route.kernel):
                route_result, route_ns = lhs.benchmark_matmul_route(
                    rhs, benchmark_route, 2)
                np.testing.assert_allclose(
                    route_result.ndarray, expected)
                self.assertGreater(route_ns, 0)

        with self.assertRaisesRegex(
                ValueError, 'repetitions must be positive'):
            lhs.benchmark_matmul(rhs, 0)
        with self.assertRaisesRegex(
                ValueError, 'repetitions must be positive'):
            lhs.benchmark_matmul_route(rhs, route, 0)

    def test_empty_contraction_and_batch_domains(self):
        dtype = np.dtype(self.dtype).name
        cases = (
            (np.empty((0,), dtype=dtype),
             np.empty((0,), dtype=dtype)),
            (np.empty((3, 0), dtype=dtype),
             np.empty((0, 2), dtype=dtype)),
            (np.empty((0, 3, 4), dtype=dtype),
             np.empty((0, 4, 2), dtype=dtype)),
        )
        for lhs_data, rhs_data in cases:
            with self.subTest(lhs=lhs_data.shape, rhs=rhs_data.shape):
                self.assert_route_results(lhs_data, rhs_data)

    def test_explicit_zero_batch_stride_uses_scratch(self):
        lhs_source = self.make_data(4 * 12).reshape(1, 4, 12)[
            :, :, ::2]
        lhs_data = np.lib.stride_tricks.as_strided(
            lhs_source,
            shape=(3, 4, 6),
            strides=(0, lhs_source.strides[1], lhs_source.strides[2]),
        )
        rhs_data = self.make_data(3 * 6 * 8).reshape(3, 6, 8)
        _, _, routes = self.assert_route_results(
            lhs_data, rhs_data, 'blas_gemm')

        if not self.has_blas_backend:
            return
        gemm = next(route for route in routes
                    if route.kernel == 'blas_gemm')
        self.assertFalse(gemm.eager_pack_lhs)
        self.assertTrue(gemm.scratch_pack_lhs)


class MatmulRouteFloat32TC(MatmulRouteTestBase, unittest.TestCase):
    dtype = 'float32'
    SimpleArray = sc.SimpleArrayFloat32


class MatmulRouteFloat64TC(MatmulRouteTestBase, unittest.TestCase):
    dtype = 'float64'
    SimpleArray = sc.SimpleArrayFloat64


class MatmulRouteComplex64TC(MatmulRouteTestBase, unittest.TestCase):
    dtype = 'complex64'
    SimpleArray = sc.SimpleArrayComplex64


class MatmulRouteComplex128TC(MatmulRouteTestBase, unittest.TestCase):
    dtype = 'complex128'
    SimpleArray = sc.SimpleArrayComplex128


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
