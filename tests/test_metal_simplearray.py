# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

import solvcon as sc


_METAL_RUNTIME = sc.METAL_BUILT and sc.metal_running()


class MetalBuildContractTC(unittest.TestCase):

    def test_cpu_device_is_the_default(self):
        array = sc.SimpleArrayFloat32([2, 3])
        self.assertEqual("cpu", array.device)
        self.assertTrue(array.ready)

    @unittest.skipIf(sc.METAL_BUILT, "Metal is built")
    def test_unbuilt_metal_allocation_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError,
                                    "Metal support is not built"):
            sc.SimpleArrayFloat32([2, 3], device="metal")


@unittest.skipUnless(_METAL_RUNTIME, "Metal runtime is unavailable")
class MetalStorageTC(unittest.TestCase):

    _DTYPES = (
        (sc.SimpleArrayBool, np.bool_),
        (sc.SimpleArrayInt8, np.int8),
        (sc.SimpleArrayInt16, np.int16),
        (sc.SimpleArrayInt32, np.int32),
        (sc.SimpleArrayInt64, np.int64),
        (sc.SimpleArrayUint8, np.uint8),
        (sc.SimpleArrayUint16, np.uint16),
        (sc.SimpleArrayUint32, np.uint32),
        (sc.SimpleArrayUint64, np.uint64),
        (sc.SimpleArrayFloat32, np.float32),
        (sc.SimpleArrayFloat64, np.float64),
        (sc.SimpleArrayComplex64, np.complex64),
        (sc.SimpleArrayComplex128, np.complex128),
    )

    def test_round_trip_all_simplearray_dtypes(self):
        for array_type, dtype in self._DTYPES:
            with self.subTest(dtype=np.dtype(dtype).name):
                source = np.arange(12, dtype="int64").reshape(3, 4)
                source = source.astype(np.dtype(dtype).name)
                if np.issubdtype(dtype, np.complexfloating):
                    source += 1j * (source + 1)
                metal = array_type(array=source, device="metal")
                self.assertEqual("metal", metal.device)
                self.assertFalse(metal.host_exported)

                host = metal.cpu()
                self.assertEqual("cpu", host.device)
                self.assertFalse(metal.host_exported)
                np.testing.assert_array_equal(source, host.ndarray)

    def test_clone_and_view_keep_metal_storage(self):
        source = np.arange(20, dtype="float32").reshape(4, 5)
        array = sc.SimpleArrayFloat32(array=source, device="metal")
        clone = array.clone()
        view = array.reshape([2, 10])

        self.assertEqual("metal", clone.device)
        self.assertEqual("metal", view.device)
        self.assertFalse(array.host_exported)
        self.assertFalse(clone.host_exported)

        np.testing.assert_array_equal(source, clone.cpu().ndarray)
        np.testing.assert_array_equal(source.reshape(2, 10),
                                      view.cpu().ndarray)

    def test_strided_numpy_storage_round_trip(self):
        source = np.arange(24, dtype="float32").reshape(4, 6)
        for view in (np.asfortranarray(source), source[::-1, ::-2]):
            with self.subTest(strides=view.strides):
                metal = sc.SimpleArrayFloat32(
                    array=view, device="metal")
                self.assertEqual(tuple(s // view.itemsize
                                       for s in view.strides),
                                 metal.stride)
                np.testing.assert_array_equal(view, metal.cpu().ndarray)

    def test_numpy_export_is_zero_copy_and_sticky(self):
        source = np.arange(6, dtype="float32").reshape(2, 3)
        array = sc.SimpleArrayFloat32(array=source, device="metal")
        view = array.ndarray
        self.assertTrue(array.host_exported)

        view[1, 2] = 123.0
        self.assertEqual(123.0, array.ndarray[1, 2])

        other = sc.SimpleArrayFloat32(
            array=np.eye(3, dtype="float32"), device="metal")
        with self.assertRaisesRegex(RuntimeError, "host pointer or view"):
            array.matmul_metal(other)

    def test_buffer_protocol_is_a_sticky_host_export(self):
        array = sc.SimpleArrayInt32([4], value=3, device="metal")
        view = memoryview(array)
        self.assertEqual([3, 3, 3, 3], list(view))
        self.assertTrue(array.host_exported)

    def test_alignment_and_lifetime(self):
        array = sc.SimpleArrayFloat32([16], alignment=64,
                                      device="metal")
        self.assertEqual(64, array.alignment)
        self.assertEqual(0, array.ndarray.ctypes.data % 64)

        with self.assertRaisesRegex(ValueError, "multiple of alignment"):
            sc.SimpleArrayFloat32([3], alignment=16, device="metal")

    def test_scalar_metal_storage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scalar Metal storage"):
            sc.SimpleArrayFloat32([], device="metal")

        scalar = sc.SimpleArrayFloat32([])
        with self.assertRaisesRegex(ValueError, "scalar Metal storage"):
            scalar.to("metal")


@unittest.skipUnless(_METAL_RUNTIME, "Metal runtime is unavailable")
class MetalMatmulTC(unittest.TestCase):

    @staticmethod
    def _array(values):
        return sc.SimpleArrayFloat32(
            array=np.ascontiguousarray(values, dtype="float32"),
            device="metal")

    def test_rectangular_matmul_is_resident_and_async(self):
        rng = np.random.default_rng(20260816)
        lhs_np = rng.standard_normal((15, 17), dtype="float32")
        rhs_np = rng.standard_normal((17, 13), dtype="float32")
        tail_np = rng.standard_normal((13, 11), dtype="float32")
        lhs = self._array(lhs_np)
        rhs = self._array(rhs_np)
        tail = self._array(tail_np)

        sc.reset_metal_statistics()
        middle = lhs.matmul_metal(rhs)
        result = middle.matmul_metal(tail)
        statistics = sc.metal_statistics()
        self.assertEqual(2, statistics["submitted_commands"])
        self.assertEqual(0, statistics["host_waits"])
        self.assertEqual("metal", middle.device)
        self.assertEqual("metal", result.device)
        self.assertFalse(middle.host_exported)

        result.wait()
        self.assertFalse(result.host_exported)
        self.assertEqual(1, sc.metal_statistics()["host_waits"])

        identity = self._array(np.eye(11, dtype="float32"))
        after_wait = result.matmul_metal(identity)
        self.assertEqual(3,
                         sc.metal_statistics()["submitted_commands"])
        got = after_wait.ndarray
        want = (lhs_np @ rhs_np) @ tail_np
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
        self.assertTrue(after_wait.host_exported)

    def test_padded_row_major_inputs(self):
        lhs_base = np.arange(24, dtype="float32").reshape(4, 6)
        rhs_base = np.arange(28, dtype="float32").reshape(4, 7)
        lhs_view = lhs_base[:, :4]
        rhs_view = rhs_base[:, :5]
        lhs = sc.SimpleArrayFloat32(
            array=lhs_view, device="metal")
        rhs = sc.SimpleArrayFloat32(
            array=rhs_view, device="metal")

        result = lhs.matmul_metal(rhs)
        np.testing.assert_allclose(
            lhs_view @ rhs_view, result.cpu().ndarray,
            rtol=2e-5, atol=2e-5)

    def test_pending_view_shares_access_state(self):
        lhs = self._array(np.arange(16, dtype="float32").reshape(4, 4))
        rhs = self._array(np.eye(4, dtype="float32"))
        result = lhs.matmul_metal(rhs)
        view = result.reshape([2, 8])

        np.testing.assert_array_equal(
            np.arange(16, dtype="float32").reshape(2, 8),
            view.ndarray)
        self.assertTrue(result.host_exported)
        with self.assertRaisesRegex(RuntimeError, "host pointer or view"):
            result.matmul_metal(rhs)

    def test_inputs_may_die_before_completion(self):
        def submit():
            lhs = self._array(
                np.arange(256, dtype="float32").reshape(16, 16))
            rhs = self._array(np.eye(16, dtype="float32"))
            return lhs.matmul_metal(rhs)

        result = submit()
        result.wait()
        np.testing.assert_array_equal(
            np.arange(256, dtype="float32").reshape(16, 16),
            result.ndarray)

    def test_pending_clone_waits_without_exporting_source(self):
        lhs = self._array(
            np.arange(64, dtype="float32").reshape(8, 8))
        identity = self._array(np.eye(8, dtype="float32"))
        sc.reset_metal_statistics()
        result = lhs.matmul_metal(identity)
        clone = result.clone()

        self.assertEqual(1, sc.metal_statistics()["host_waits"])
        self.assertFalse(result.host_exported)
        self.assertFalse(clone.host_exported)
        chained = result.matmul_metal(identity)
        np.testing.assert_array_equal(clone.cpu().ndarray,
                                      chained.cpu().ndarray)

    def test_completed_task_history_is_compacted(self):
        value = self._array(np.ones((1, 1), dtype="float32"))
        weight = self._array(np.ones((1, 1), dtype="float32"))
        sc.reset_metal_statistics()

        for _ in range(300):
            result = value.matmul_metal(weight)
            result.wait()

        self.assertEqual(300,
                         sc.metal_statistics()["submitted_commands"])
        self.assertEqual(300, sc.metal_statistics()["host_waits"])
        result.wait()
        self.assertEqual(300, sc.metal_statistics()["host_waits"])

    def test_host_export_can_be_copied_back_to_metal(self):
        array = self._array(np.eye(4, dtype="float32"))
        array.ndarray[0, 0] = 2.0
        recovered = array.to("metal")

        self.assertTrue(array.host_exported)
        self.assertFalse(recovered.host_exported)
        result = recovered.matmul_metal(
            self._array(np.eye(4, dtype="float32")))
        np.testing.assert_array_equal(array.ndarray,
                                      result.cpu().ndarray)

    def test_mixed_device_contract(self):
        values = np.arange(16, dtype="float32").reshape(4, 4)
        metal = self._array(values)
        cpu = sc.SimpleArrayFloat32(array=values)
        with self.assertRaisesRegex(ValueError, "Metal-backed operands"):
            metal.matmul_metal(cpu)

    def test_empty_contractions(self):
        lhs = self._array(np.empty((3, 0), dtype="float32"))
        rhs = self._array(np.empty((0, 4), dtype="float32"))
        result = lhs.matmul_metal(rhs)
        self.assertEqual("metal", result.device)
        np.testing.assert_array_equal(
            np.zeros((3, 4), dtype="float32"), result.ndarray)

        empty_rows = self._array(
            np.empty((0, 3), dtype="float32"))
        matrix = self._array(np.ones((3, 4), dtype="float32"))
        outer_empty = empty_rows.matmul_metal(matrix)
        self.assertEqual((0, 4), outer_empty.shape)
        self.assertEqual("metal", outer_empty.device)

    def test_rejects_unsupported_shape_and_layout(self):
        vector = self._array(np.arange(4, dtype="float32"))
        matrix = self._array(np.eye(4, dtype="float32"))
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            vector.matmul_metal(matrix)

        transposed = self._array(
            np.arange(16, dtype="float32").reshape(4, 4))
        transposed.transpose()
        with self.assertRaisesRegex(ValueError, "row-major"):
            transposed.matmul_metal(matrix)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
