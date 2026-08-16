# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import dataclasses
import sys
import unittest
from unittest import mock

import numpy as np

from solvcon.matmul_benchmark import arrays
from solvcon.matmul_benchmark import collection
from solvcon.matmul_benchmark import engine
from solvcon.matmul_benchmark import profiles
from solvcon.matmul_benchmark import schema


class CoreStorageTC(unittest.TestCase):
    def test_storage_orders_and_leading_dimension_gaps(self):
        cases = (
            (profiles.CoreStorage.c_compact(), (5, 1), 15),
            (profiles.CoreStorage.c_gap(2), (7, 1), 19),
            (profiles.CoreStorage.f_compact(), (1, 3), 15),
            (profiles.CoreStorage.f_gap(2), (1, 5), 23),
            (profiles.CoreStorage.custom(-11, 2), (-11, 2), 31),
            (profiles.CoreStorage.custom(0, -2), (0, -2), 9),
        )
        for storage, expected_strides, expected_span in cases:
            with self.subTest(storage=storage):
                shape, strides, span = storage.resolve(3, 5)
                self.assertEqual((3, 5), shape)
                self.assertEqual(expected_strides, strides)
                self.assertEqual(expected_span, span)

    def test_recipe_objects_round_trip_and_are_immutable(self):
        axis = profiles.BatchAxis(
            output_extent=7,
            lhs_extent='one',
            rhs_extent='same',
            lhs_stride=profiles.BatchStride.zero(),
            rhs_stride=profiles.BatchStride.custom(-31),
        )
        profile = profiles.InputProfile(
            profile_id='mixed-layout',
            name='Mixed layout',
            lhs_core=profiles.CoreStorage.c_gap(2),
            rhs_core=profiles.CoreStorage.f_gap(3),
            batch_axes=(axis,),
        )

        self.assertEqual(
            profiles.CoreStorage.c_gap(2),
            profiles.CoreStorage.from_dict(
                profiles.CoreStorage.c_gap(2).to_dict()),
        )
        custom_core = profiles.CoreStorage.custom(-11, 0)
        self.assertEqual(
            custom_core,
            profiles.CoreStorage.from_dict(custom_core.to_dict()),
        )
        self.assertEqual(
            profiles.BatchStride.custom(-31),
            profiles.BatchStride.from_dict(
                profiles.BatchStride.custom(-31).to_dict()),
        )
        self.assertEqual(
            axis, profiles.BatchAxis.from_dict(axis.to_dict()))
        self.assertEqual(
            profile, profiles.InputProfile.from_dict(profile.to_dict()))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            profile.name = 'Changed'

    def test_invalid_recipe_is_rejected_before_resolution(self):
        with self.assertRaisesRegex(schema.SchemaError, 'order'):
            profiles.CoreStorage(order='row-major')
        with self.assertRaisesRegex(schema.SchemaError, 'at least 0'):
            profiles.CoreStorage.c_gap(-1)
        with self.assertRaisesRegex(schema.SchemaError, 'row_stride'):
            profiles.CoreStorage(order='custom')
        with self.assertRaisesRegex(schema.SchemaError, 'custom strides'):
            profiles.CoreStorage(order='c', row_stride=4,
                                 column_stride=1)
        with self.assertRaisesRegex(schema.SchemaError, 'at least one'):
            profiles.BatchAxis(
                output_extent=4, lhs_extent='one', rhs_extent='one')
        with self.assertRaisesRegex(schema.SchemaError, 'profile_id'):
            profiles.InputProfile(profile_id='Display name', name='Name')


class InputProfileResolutionTC(unittest.TestCase):
    def test_independent_core_storage_resolves_exact_specs(self):
        profile = profiles.InputProfile(
            profile_id='independent-cores',
            name='Independent cores',
            lhs_core=profiles.CoreStorage.c_gap(2),
            rhs_core=profiles.CoreStorage.f_gap(3),
        )
        resolved = profile.resolve(3, 5, 4)

        self.assertEqual((3, 5), resolved.lhs.shape)
        self.assertEqual((7, 1), resolved.lhs.strides)
        self.assertEqual(19, resolved.lhs_storage_span)
        self.assertEqual((5, 4), resolved.rhs.shape)
        self.assertEqual((1, 8), resolved.rhs.strides)
        self.assertEqual(29, resolved.rhs_storage_span)
        self.assertEqual((3, 4), resolved.output_shape)
        self.assertEqual(
            'lhs_c_gap2-rhs_f_gap3', resolved.layout)
        self.assertEqual('matrix', resolved.broadcast)
        self.assertIn(
            'LHS shape=(3, 5), strides=(7, 1), storage span=19',
            resolved.facts)

    def test_multiple_batch_axes_resolve_inside_out(self):
        profile = profiles.InputProfile(
            profile_id='multi-axis',
            name='Multi-axis custom strides',
            lhs_core=profiles.CoreStorage.c_gap(1),
            rhs_core=profiles.CoreStorage.f_gap(2),
            batch_axes=(
                profiles.BatchAxis(output_extent=2),
                profiles.BatchAxis(
                    output_extent=3,
                    lhs_extent='one',
                    rhs_extent='same',
                    lhs_stride=profiles.BatchStride.zero(),
                    rhs_stride=profiles.BatchStride.custom(-20),
                ),
            ),
        )
        resolved = profile.resolve(2, 3, 4)

        self.assertEqual((2, 1, 2, 3), resolved.lhs.shape)
        self.assertEqual((7, 0, 4, 1), resolved.lhs.strides)
        self.assertEqual(14, resolved.lhs_storage_span)
        self.assertEqual((2, 3, 3, 4), resolved.rhs.shape)
        self.assertEqual((58, -20, 1, 5), resolved.rhs.strides)
        self.assertEqual(116, resolved.rhs_storage_span)
        self.assertEqual((2, 3, 2, 4), resolved.output_shape)
        self.assertEqual('broadcast_lhs', resolved.broadcast)
        self.assertIn(
            'batch axis 1: output extent=3, LHS extent=1, stride=0 '
            '(zero), RHS extent=3, stride=-20 (custom)',
            resolved.facts)

        lhs = arrays.make_strided_array(resolved.lhs, 'float32')
        rhs = arrays.make_strided_array(resolved.rhs, 'float32')
        self.assertEqual(resolved.output_shape, np.matmul(lhs, rhs).shape)

    def test_zero_stride_reuse_uses_canonical_broadcast_category(self):
        profile = profiles.InputProfile(
            profile_id='zero-reuse',
            name='Zero-stride reuse',
            batch_axes=(profiles.BatchAxis(
                output_extent=4,
                lhs_stride=profiles.BatchStride.zero(),
            ),),
        )

        resolved = profile.resolve(2, 3, 5)

        self.assertEqual('broadcast_lhs', resolved.broadcast)
        self.assertEqual(
            collection._infer_broadcast_profile(
                resolved.lhs, resolved.rhs),
            resolved.broadcast,
        )

    def test_cross_axis_reuse_is_broadcast_both(self):
        profile = profiles.InputProfile(
            profile_id='cross-reuse',
            name='Cross-axis reuse',
            batch_axes=(
                profiles.BatchAxis(
                    output_extent=2, rhs_extent='one'),
                profiles.BatchAxis(
                    output_extent=3, lhs_extent='one'),
            ),
        )

        resolved = profile.resolve(2, 3, 5)

        self.assertEqual('broadcast_both', resolved.broadcast)
        self.assertEqual(
            collection._infer_broadcast_profile(
                resolved.lhs, resolved.rhs),
            resolved.broadcast,
        )

    def test_extent_one_normalizes_ignored_stride_to_zero(self):
        axis = profiles.BatchAxis(
            output_extent=7,
            lhs_extent='one',
            lhs_stride=profiles.BatchStride.custom(-999),
        )

        self.assertEqual(profiles.BatchStride.zero(), axis.lhs_stride)

    def test_canonical_presets_match_existing_operand_conventions(self):
        expected = (
            (
                profiles.unbatched_profile(),
                (2, 3), (3, 1), (3, 4), (4, 1), 'matrix',
            ),
            (
                profiles.matched_batch_profile(5),
                (5, 2, 3), (6, 3, 1),
                (5, 3, 4), (12, 4, 1), 'matched_batch',
            ),
            (
                profiles.reuse_lhs_profile(5),
                (1, 2, 3), (0, 3, 1),
                (5, 3, 4), (12, 4, 1), 'broadcast_lhs',
            ),
            (
                profiles.reuse_rhs_profile(5),
                (5, 2, 3), (6, 3, 1),
                (1, 3, 4), (0, 4, 1), 'broadcast_rhs',
            ),
        )
        for (profile, lhs_shape, lhs_strides, rhs_shape, rhs_strides,
             broadcast) in expected:
            with self.subTest(profile=profile.profile_id):
                resolved = profile.resolve(2, 3, 4)
                self.assertEqual(lhs_shape, resolved.lhs.shape)
                self.assertEqual(lhs_strides, resolved.lhs.strides)
                self.assertEqual(rhs_shape, resolved.rhs.shape)
                self.assertEqual(rhs_strides, resolved.rhs.strides)
                self.assertEqual(broadcast, resolved.broadcast)

    def test_exact_case_derives_cross_broadcast_and_round_trips(self):
        lhs = schema.OperandSpec(
            shape=(2, 1, 3, 5), strides=(33, 0, 11, -2))
        rhs = schema.OperandSpec(
            shape=(1, 4, 5, 7), strides=(0, 35, 7, 1))
        profile = profiles.InputProfile.exact(
            profile_id='exact-cross', name='Exact cross',
            lhs=lhs, rhs=rhs)

        resolved = profile.resolve(999, 999, 999)

        self.assertEqual((3, 5, 7),
                         (resolved.m, resolved.k, resolved.n))
        self.assertEqual((2, 4, 3, 7), resolved.output_shape)
        self.assertEqual('broadcast_both', resolved.broadcast)
        self.assertEqual(64, resolved.lhs_storage_span)
        self.assertEqual(140, resolved.rhs_storage_span)
        self.assertEqual(
            collection._infer_broadcast_profile(lhs, rhs),
            resolved.broadcast)
        self.assertEqual(
            profile, profiles.InputProfile.from_dict(profile.to_dict()))

        zero_reuse = profiles.InputProfile.exact(
            profile_id='exact-zero', name='Exact zero reuse',
            lhs=schema.OperandSpec(
                shape=(4, 3, 5), strides=(0, 5, 1)),
            rhs=schema.OperandSpec(
                shape=(4, 5, 7), strides=(35, 7, 1)))
        zero_resolved = zero_reuse.resolve(1, 1, 1)
        self.assertEqual('broadcast_lhs', zero_resolved.broadcast)
        self.assertEqual(
            collection._infer_broadcast_profile(
                zero_resolved.lhs, zero_resolved.rhs),
            zero_resolved.broadcast)

    def test_exact_case_rejects_invalid_matmul_and_storage(self):
        matrix = schema.OperandSpec(shape=(3, 5), strides=(5, 1))
        with self.assertRaisesRegex(schema.SchemaError, 'at least two axes'):
            profiles.InputProfile.exact(
                'rank', 'Rank',
                schema.OperandSpec(shape=(5,), strides=(1,)), matrix)
        with self.assertRaisesRegex(schema.SchemaError, 'dimensions K'):
            profiles.InputProfile.exact(
                'inner', 'Inner', matrix,
                schema.OperandSpec(shape=(4, 7), strides=(7, 1)))
        with self.assertRaisesRegex(schema.SchemaError, 'do not broadcast'):
            profiles.InputProfile.exact(
                'batch', 'Batch',
                schema.OperandSpec(
                    shape=(2, 3, 5), strides=(15, 5, 1)),
                schema.OperandSpec(
                    shape=(4, 5, 7), strides=(35, 7, 1)))
        with self.assertRaisesRegex(schema.SchemaError, 'positive ssize_t'):
            profiles.InputProfile.exact(
                'empty', 'Empty',
                schema.OperandSpec(shape=(0, 5), strides=(5, 1)),
                schema.OperandSpec(shape=(5, 7), strides=(7, 1)))
        with self.assertRaisesRegex(schema.SchemaError, 'storage span'):
            profiles.InputProfile.exact(
                'span', 'Span',
                schema.OperandSpec(
                    shape=(2, 5), strides=(sys.maxsize, 1)),
                schema.OperandSpec(shape=(5, 7), strides=(7, 1)))


class ProfileExpansionTC(unittest.TestCase):
    def test_cartesian_order_and_collection_cell_compatibility(self):
        profile_set = (
            profiles.unbatched_profile(),
            profiles.reuse_lhs_profile(3),
        )
        expanded = profiles.expand_profiles(
            profile_set, m_values=(2, 4), k_values=(3,),
            n_values=(5, 7))

        self.assertEqual(8, len(expanded))
        self.assertEqual(
            (
                'unbatched-m2-k3-n5',
                'reuse_lhs-m2-k3-n5',
                'unbatched-m2-k3-n7',
                'reuse_lhs-m2-k3-n7',
                'unbatched-m4-k3-n5',
                'reuse_lhs-m4-k3-n5',
                'unbatched-m4-k3-n7',
                'reuse_lhs-m4-k3-n7',
            ),
            tuple(item.cell_id for item in expanded),
        )
        cell = collection.CollectionCell(
            **expanded[1].collection_cell_kwargs())
        self.assertEqual(expanded[1].lhs, cell.lhs)
        self.assertEqual('broadcast_lhs', cell.broadcast)

    def test_duplicate_profile_identity_is_rejected(self):
        first = profiles.unbatched_profile(name='First')
        duplicate_id = profiles.unbatched_profile(name='Second')
        with self.assertRaisesRegex(schema.SchemaError, 'profile_id'):
            profiles.expand_profiles(
                (first, duplicate_id), (2,), (3,), (4,))

        duplicate_name = profiles.unbatched_profile(
            profile_id='another', name='First')
        with self.assertRaisesRegex(schema.SchemaError, 'name'):
            profiles.expand_profiles(
                (first, duplicate_name), (2,), (3,), (4,))

    def test_semantically_duplicate_profile_is_rejected(self):
        first = profiles.unbatched_profile(
            profile_id='first', name='First')
        duplicate = profiles.unbatched_profile(
            profile_id='second', name='Second')

        with self.assertRaisesRegex(schema.SchemaError, 'same input recipe'):
            collection.input_profile_plan(
                (first, duplicate), m_values=(2,), k_values=(3,),
                n_values=(4,), routes=('naive',),
                numpy_baseline=False)

        exact_duplicate = profiles.InputProfile.exact(
            profile_id='exact', name='Exact duplicate',
            lhs=schema.OperandSpec(shape=(2, 3), strides=(3, 1)),
            rhs=schema.OperandSpec(shape=(3, 4), strides=(4, 1)))
        with self.assertRaisesRegex(schema.SchemaError, 'same exact A/B'):
            collection.input_profile_plan(
                (first, exact_duplicate), m_values=(2,), k_values=(3,),
                n_values=(4,), routes=('naive',),
                numpy_baseline=False)

    def test_duplicate_or_invalid_dimensions_are_rejected(self):
        profile_set = (profiles.unbatched_profile(),)
        with self.assertRaisesRegex(schema.SchemaError, 'M values'):
            profiles.expand_profiles(
                profile_set, (2, 2), (3,), (4,))
        with self.assertRaisesRegex(schema.SchemaError, 'K values'):
            profiles.expand_profiles(
                profile_set, (2,), (0,), (4,))

    def test_collection_plan_uses_each_exact_composite_profile(self):
        profile = profiles.InputProfile(
            profile_id='mixed-batch',
            name='Mixed batch',
            lhs_core=profiles.CoreStorage.c_gap(2),
            rhs_core=profiles.CoreStorage.f_gap(3),
            batch_axes=(profiles.BatchAxis(
                output_extent=5,
                lhs_extent='one',
                lhs_stride=profiles.BatchStride.zero(),
            ),),
        )

        plan = collection.input_profile_plan(
            (profile,), m_values=(3,), k_values=(5,), n_values=(4,),
            routes=('naive', 'blas_gemm', 'winograd'),
            numpy_baseline=False, plan_id='profile-plan')

        self.assertEqual(1, len(plan.cells))
        cell = plan.cells[0]
        self.assertEqual((1, 3, 5), cell.lhs.shape)
        self.assertEqual((0, 7, 1), cell.lhs.strides)
        self.assertEqual((5, 5, 4), cell.rhs.shape)
        self.assertEqual((29, 1, 8), cell.rhs.strides)
        self.assertEqual(('naive', 'blas_gemm'), cell.routes)
        self.assertEqual('broadcast_lhs', cell.broadcast)

    def test_exact_case_contributes_one_cell_outside_mkn_product(self):
        exact = profiles.InputProfile.exact(
            profile_id='one-case', name='One exact case',
            lhs=schema.OperandSpec(
                shape=(1, 3, 5), strides=(0, 11, -2)),
            rhs=schema.OperandSpec(
                shape=(4, 5, 7), strides=(35, 7, 1)))
        sweep = profiles.unbatched_profile(
            profile_id='sweep', name='Sweep')

        expanded = profiles.expand_profiles(
            (exact, sweep), m_values=(2, 4), k_values=(3,),
            n_values=(6, 8))

        self.assertEqual(5, len(expanded))
        self.assertEqual('one-case-exact', expanded[0].cell_id)
        self.assertEqual((3, 5, 7),
                         (expanded[0].m, expanded[0].k, expanded[0].n))
        self.assertEqual(1, sum(
            item.profile_id == 'one-case' for item in expanded))

        plan = collection.input_profile_plan(
            (exact,), m_values=(2, 4), k_values=(3, 5),
            n_values=(6, 8),
            routes=('naive', 'blas_gemm', 'winograd'),
            numpy_baseline=False)
        self.assertEqual(1, len(plan.cells))
        self.assertEqual(('naive', 'blas_gemm'), plan.cells[0].routes)
        self.assertEqual('broadcast_lhs', plan.cells[0].broadcast)

    def test_exact_only_plan_does_not_charge_or_traverse_mkn_grid(self):
        exact = profiles.InputProfile.exact(
            profile_id='one-case', name='One exact case',
            lhs=schema.OperandSpec(shape=(3, 5), strides=(5, 1)),
            rhs=schema.OperandSpec(shape=(5, 7), strides=(7, 1)))
        m_values = tuple(range(1, 130))
        k_values = tuple(range(1, 130))
        calls = []
        product_steps = []
        original_resolve = profiles.InputProfile.resolve
        original_product = profiles.itertools.product

        def counted_resolve(profile, m_value, k_value, n_value):
            calls.append((m_value, k_value, n_value))
            return original_resolve(profile, m_value, k_value, n_value)

        def counted_product(*dimensions):
            for combination in original_product(*dimensions):
                product_steps.append(combination)
                yield combination

        with mock.patch.object(
                profiles.InputProfile, 'resolve', new=counted_resolve), \
                mock.patch.object(
                    profiles.itertools, 'product', new=counted_product):
            plan = collection.input_profile_plan(
                (exact,), m_values=m_values, k_values=k_values,
                n_values=(1,), routes=('naive',),
                numpy_baseline=False)

        self.assertEqual(1, len(plan.cells))
        self.assertEqual([(1, 1, 1)], calls)
        self.assertEqual([(1, 1, 1)], product_steps)


class NativeCustomCoreTC(unittest.TestCase):
    def _case(self, profile):
        resolved = profile.resolve(3, 5, 4)
        lhs = arrays.make_strided_array(
            resolved.lhs, 'float32', seed=17)
        rhs = arrays.make_strided_array(
            resolved.rhs, 'float32', seed=19)
        case = engine.SolvconRouteEngine().prepare(lhs, rhs, 'float32')
        routes = {route.name: route for route in case.routes}
        if 'blas_gemm' not in routes:
            self.skipTest('the build has no BLAS backend')
        expected = np.matmul(lhs, rhs)
        actual = case.execute_route('blas_gemm')
        tolerance = 128 * np.finfo('float32').eps * 5
        np.testing.assert_allclose(
            actual, expected, rtol=tolerance, atol=tolerance)
        return routes['blas_gemm']

    def test_custom_core_uses_scratch_packing_when_not_reused(self):
        route = self._case(profiles.InputProfile(
            profile_id='scratch-core',
            name='Scratch custom core',
            lhs_core=profiles.CoreStorage.custom(11, -2),
        ))

        self.assertFalse(route.eager_pack_lhs)
        self.assertTrue(route.scratch_pack_lhs)

    def test_custom_broadcast_core_uses_one_eager_pack(self):
        route = self._case(profiles.InputProfile(
            profile_id='eager-core',
            name='Eager custom core',
            lhs_core=profiles.CoreStorage.custom(11, -2),
            batch_axes=(profiles.BatchAxis(
                output_extent=3, lhs_extent='one'),),
        ))

        self.assertTrue(route.eager_pack_lhs)
        self.assertFalse(route.scratch_pack_lhs)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
