# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

import solvcon


def make_array(data):
    return solvcon.SimpleArrayFloat64(array=data)


def make_stepped(data, axis=-1):
    storage_shape = list(data.shape)
    storage_shape[axis] *= 2
    storage = np.empty(storage_shape, dtype="float64")
    selection = [slice(None)] * data.ndim
    selection[axis] = slice(None, None, 2)
    view = storage[tuple(selection)]
    view[...] = data
    return view


class PlannedElementwiseTC(unittest.TestCase):

    def test_binary_layouts_follow_logical_coordinates(self):
        lhs_base = np.arange(
            2 * 3 * 4, dtype="float64"
        ).reshape(2, 3, 4)
        rhs_base = np.arange(
            2 * 3 * 4, dtype="float64"
        ).reshape(2, 3, 4)
        rhs_base += 1.5
        layouts = {
            "contiguous": lambda values: values,
            "fortran": lambda values: np.asfortranarray(
                values, dtype="float64"
            ),
            "transposed": lambda values: values.transpose(2, 0, 1),
            "reversed": lambda values: values[::-1, :, ::-1],
            "stepped": make_stepped,
        }
        operations = {
            "add": np.add,
            "sub": np.subtract,
            "mul": np.multiply,
            "div": np.divide,
        }

        for layout_name, layout in layouts.items():
            lhs = layout(lhs_base)
            rhs = layout(rhs_base)
            for operation, reference in operations.items():
                with self.subTest(
                    layout=layout_name,
                    operation=operation,
                ):
                    function = getattr(
                        make_array(lhs), f"_planned_{operation}"
                    )
                    result = function(make_array(rhs))
                    np.testing.assert_allclose(
                        result.ndarray, reference(lhs, rhs)
                    )

    def test_mixed_rank_broadcast_uses_zero_stride_mapping(self):
        lhs = np.arange(
            2 * 3, dtype="float64"
        ).reshape(2, 3, 1)
        lhs = make_stepped(lhs)
        rhs = np.arange(
            4, dtype="float64"
        ).reshape(1, 4) + 1

        result = make_array(lhs)._planned_mul(make_array(rhs))

        np.testing.assert_array_equal(result.ndarray, lhs * rhs)

    def test_scalar_and_inplace_broadcast_keep_destination_shape(self):
        source = np.arange(
            3 * 4, dtype="float64"
        ).reshape(3, 4)
        destinations = (
            source[:, ::-1],
            make_stepped(source),
            np.asfortranarray(source),
        )
        for destination in destinations:
            with self.subTest(strides=destination.strides):
                expected_scalar = destination + 2.5
                sarr = make_array(destination)
                scalar_result = sarr._planned_add(2.5)
                np.testing.assert_allclose(
                    scalar_result.ndarray, expected_scalar
                )

                rhs = np.arange(
                    4, dtype="float64"
                ).reshape(1, 4)
                expected_inplace = destination + rhs
                sarr._planned_iadd(make_array(rhs))
                np.testing.assert_allclose(
                    sarr.ndarray, expected_inplace
                )

    def test_inplace_rejects_result_that_expands_destination(self):
        destination = make_array(
            np.zeros((1, 3), dtype="float64")
        )
        rhs = make_array(np.zeros((2, 3), dtype="float64"))

        with self.assertRaisesRegex(ValueError, "output shape"):
            destination._planned_iadd(rhs)

    def test_fortran_inplace_leading_broadcast(self):
        destination = np.asfortranarray(
            np.arange(
                2 * 17 * 32, dtype="float64"
            ).reshape(2, 17, 32)
        )
        rhs = np.asfortranarray(
            np.arange(
                17 * 32, dtype="float64"
            ).reshape(1, 17, 32)
        )
        expected = destination + rhs
        result = make_array(destination)

        result._planned_iadd(make_array(rhs))

        np.testing.assert_array_equal(result.ndarray, expected)
        self.assertTrue(result.ndarray.flags.f_contiguous)

    def test_invalid_trailing_dimensions_are_rejected(self):
        lhs = make_array(np.zeros((2, 3), dtype="float64"))
        rhs = make_array(np.zeros((4,), dtype="float64"))

        with self.assertRaisesRegex(ValueError, "broadcast"):
            lhs._planned_add(rhs)

    def test_array_dispatch_selects_matching_inherited_holder(self):
        float32_array_type = solvcon.SimpleArrayFloat32
        float64_array_type = solvcon.SimpleArrayFloat64

        class MixedArray(float32_array_type, float64_array_type):

            def __init__(self, shape):
                float32_array_type.__init__(self, shape)
                float64_array_type.__init__(self, shape)

        operand = MixedArray((2,))
        float32_values = float32_array_type.ndarray.__get__(
            operand, MixedArray
        )
        float32_values.fill(0)
        operand_values = float64_array_type.ndarray.__get__(
            operand, MixedArray
        )
        operand_values[:] = np.array([10.0, 20.0], dtype="float64")
        lhs_values = np.array([1.0, 2.0], dtype="float64")
        expected = lhs_values + operand_values

        lhs = make_array(lhs_values)
        result = float64_array_type._planned_add(lhs, operand)
        destination = make_array(np.empty((2,), dtype="float64"))
        float64_array_type._planned_add_to(lhs, operand, destination)
        inplace = make_array(lhs_values.copy())
        float64_array_type._planned_iadd(inplace, operand)

        np.testing.assert_array_equal(result.ndarray, expected)
        np.testing.assert_array_equal(destination.ndarray, expected)
        np.testing.assert_array_equal(inplace.ndarray, expected)

    def test_partial_overlap_reads_from_a_snapshot(self):
        cases = (
            (slice(1, None), slice(None, -1)),
            (slice(None, -1), slice(1, None)),
        )
        for destination_slice, source_slice in cases:
            with self.subTest(destination=destination_slice):
                storage = np.arange(1, 9, dtype="float64")
                destination_values = storage[destination_slice]
                source_values = storage[source_slice]
                expected = (
                    destination_values.copy() +
                    source_values.copy()
                )
                destination = make_array(destination_values)
                source = make_array(source_values)

                destination._planned_iadd(source)

                np.testing.assert_array_equal(
                    destination.ndarray, expected
                )

    def test_sparse_input_produces_compact_broadcast_output(self):
        values = np.arange(
            3 * 8, dtype="float64"
        ).reshape(3, 8)[:, ::2]

        result = make_array(values)._planned_mul(2.0)

        np.testing.assert_array_equal(result.ndarray, values * 2.0)
        self.assertTrue(result.ndarray.flags.c_contiguous)
        self.assertEqual(
            result.size * result.ndarray.itemsize, result.nbytes
        )

    def test_singleton_broadcast_preserves_dense_expanded_layout(self):
        values = np.asfortranarray(
            np.arange(
                3 * 4, dtype="float64"
            ).reshape(3, 4)
        )
        rhs = np.array([2.0], dtype="float64")

        result = make_array(values)._planned_add(make_array(rhs))

        np.testing.assert_array_equal(result.ndarray, values + rhs)
        self.assertTrue(result.ndarray.flags.f_contiguous)

    def test_conflicting_broadcast_layout_uses_compact_output(self):
        lhs = np.arange(
            3 * 4, dtype="float64"
        ).reshape(1, 3, 4)
        rhs = np.asfortranarray(
            np.arange(
                2 * 3 * 4, dtype="float64"
            ).reshape(2, 3, 4)
        )

        result = make_array(lhs)._planned_add(make_array(rhs))

        np.testing.assert_array_equal(result.ndarray, lhs + rhs)
        self.assertTrue(result.ndarray.flags.c_contiguous)

    def test_nonconstant_broadcast_uses_compact_output(self):
        lhs = np.asfortranarray(
            np.arange(
                1 * 3 * 8, dtype="float64"
            ).reshape(1, 3, 8)
        )
        rhs = np.asfortranarray(
            np.arange(
                2 * 3 * 8, dtype="float64"
            ).reshape(2, 3, 8)
        )

        result = make_array(lhs)._planned_add(make_array(rhs))

        np.testing.assert_array_equal(result.ndarray, lhs + rhs)
        self.assertTrue(result.ndarray.flags.c_contiguous)

    def test_preallocated_output_covers_broadcast_and_scalar(self):
        lhs = np.arange(
            1, 4, dtype="float64"
        ).reshape(3, 1)
        rhs = np.arange(
            2, 6, dtype="float64"
        ).reshape(1, 4)
        operations = {
            "add": np.add,
            "sub": np.subtract,
            "mul": np.multiply,
            "div": np.divide,
        }

        lhs_array = make_array(lhs)
        rhs_array = make_array(rhs)
        for operation, reference in operations.items():
            with self.subTest(operation=operation):
                destination = np.empty(
                    (3, 4), dtype="float64", order="F"
                )
                result = getattr(
                    lhs_array, f"_planned_{operation}_to"
                )(
                    rhs_array, make_array(destination)
                )

                self.assertIsNone(result)
                np.testing.assert_allclose(
                    destination, reference(lhs, rhs)
                )

        scalar_destination = make_stepped(
            np.empty_like(lhs, dtype="float64")
        )
        lhs_array._planned_mul_to(
            2.5, make_array(scalar_destination)
        )
        np.testing.assert_allclose(
            scalar_destination, lhs * 2.5
        )

    def test_preallocated_output_validates_shape(self):
        lhs = make_array(
            np.ones((2, 1), dtype="float64")
        )
        rhs = make_array(
            np.ones((1, 3), dtype="float64")
        )
        destination = make_array(
            np.empty((2, 1), dtype="float64")
        )

        with self.assertRaisesRegex(ValueError, "output shape"):
            lhs._planned_add_to(rhs, destination)

    def test_preallocated_output_snapshots_overlapping_input(self):
        storage = np.arange(1, 7, dtype="float64")
        lhs_values = storage[:-1]
        destination_values = storage[1:]
        rhs_values = np.arange(10, 15, dtype="float64")
        expected = lhs_values.copy() + rhs_values

        make_array(lhs_values)._planned_add_to(
            make_array(rhs_values),
            make_array(destination_values),
        )

        np.testing.assert_array_equal(
            destination_values, expected
        )


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
