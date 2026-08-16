# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import copy
import dataclasses
import io
import json
import os
import pathlib
import tempfile
import unittest
import unittest.mock

import numpy as np

from solvcon import matmul_benchmark


class FakeRoute:
    def __init__(self, name, selected_by_auto=False):
        self.name = name
        self.selected_by_auto = selected_by_auto

    def packing_dict(self):
        return {
            'eager_lhs': False,
            'eager_rhs': False,
            'scratch_lhs': False,
            'scratch_rhs': False,
        }


class FakeCase:
    def __init__(self, lhs, rhs, case_index, events):
        self.lhs = lhs
        self.rhs = rhs
        self.case_index = case_index
        self.events = events
        self.routes = (
            FakeRoute('generic', selected_by_auto=True),
            FakeRoute('blas_gemm'),
        )

    def _result(self):
        return np.atleast_1d(np.matmul(self.lhs, self.rhs))

    def execute_auto(self):
        self.events.append(('execute', self.case_index, 'auto', 1))
        return self._result()

    def execute_route(self, name):
        self.events.append(('execute', self.case_index, name, 1))
        return self._result()

    def benchmark_auto(self, repetitions):
        self.events.append(
            ('benchmark', self.case_index, 'auto', repetitions))
        return self._result(), 20 * repetitions

    def benchmark_route(self, name, repetitions):
        self.events.append(
            ('benchmark', self.case_index, name, repetitions))
        duration = 15 if name == 'generic' else 5
        return self._result(), duration * repetitions


class FakeEngine:
    def __init__(self):
        self.cases = []
        self.events = []

    def prepare(self, lhs, rhs, _dtype):
        case = FakeCase(lhs, rhs, len(self.cases), self.events)
        self.cases.append(case)
        return case


class StepClock:
    def __init__(self, step=100):
        self.value = 0
        self.step = step

    def __call__(self):
        result = self.value
        self.value += self.step
        return result


def make_plan(**overrides):
    options = {
        'm_values': (2, 4, 6),
        'k_values': (4,),
        'n_values': (2,),
        'dtype': 'float32',
        'threads': 1,
        'mode': {
            'name': 'stable',
            'warmups': 1,
            'repetitions': 2,
            'panels': 3,
        },
        'routes': ('generic', 'blas_gemm'),
        'numpy_baseline': False,
        'seed': 91,
        'plan_id': 'collection-test',
    }
    options.update(overrides)
    return matmul_benchmark.collection.default_shape_boundary_plan(
        **options)


def make_target_plan(**overrides):
    target = matmul_benchmark.duration.TargetDurationSpec(
        seconds=0.05,
        mode='preview',
        safety_fraction=0.9,
        calibration_block_seconds=1e-6,
        checkpoint_seconds=0.005,
    )
    options = {
        'm_values': (2,),
        'k_values': (3,),
        'n_values': (2,),
        'dtype': 'float32',
        'threads': 1,
        'routes': ('generic', 'blas_gemm'),
        'numpy_baseline': False,
        'seed': 91,
        'plan_id': 'target-collection-test',
        'target_duration': target,
    }
    options.update(overrides)
    return matmul_benchmark.collection.default_shape_boundary_plan(
        **options)


def fake_metadata(threads=1):
    environment = {
        name: str(threads)
        for name in matmul_benchmark.collector.THREAD_ENVIRONMENT
    }
    return {
        'process': {
            'pid': 1,
            'executable': '/usr/bin/python3',
            'python': '3.12.0',
            'affinity': None,
        },
        'machine': {
            'node': 'test-machine',
            'system': 'TestOS',
            'release': '1',
            'machine': 'test-arch',
            'processor': '',
            'logical_cpu_count': 4,
        },
        'build': {
            'git_commit': None,
            'git_dirty': None,
            'dirty_diff_sha256': None,
            'dirty_source_complete': False,
            'solvcon_extension': None,
            'extension_mtime_ns': None,
            'extension_sha256': None,
            'native_loader': {
                'command': None,
                'dependencies': [],
                'returncode': None,
            },
            'solvcon_profile': None,
        },
        'backend': {
            'numpy_version': np.__version__,
            'numpy_configuration': '',
        },
        'threading': {
            'requested_threads': threads,
            'environment': environment,
        },
    }


def thread_environment(threads=1):
    return {
        name: str(threads)
        for name in matmul_benchmark.collector.THREAD_ENVIRONMENT
    }


def resource_budget(peak_bytes, single_allocation_bytes=None):
    if single_allocation_bytes is None:
        single_allocation_bytes = min(peak_bytes, 4 * 1024 ** 3)
    return matmul_benchmark.arrays.ResourceBudget(
        available_bytes=max(peak_bytes, single_allocation_bytes),
        peak_bytes=peak_bytes,
        single_allocation_bytes=single_allocation_bytes,
    )


class CollectionPlanTC(unittest.TestCase):
    def test_target_duration_is_hashed_but_transport_identity_is_not(self):
        plan = make_target_plan(output_path='/tmp/first.json')
        rebuilt = matmul_benchmark.collection.CollectionPlan.from_dict(
            plan.to_dict())
        changed_target = copy.deepcopy(plan.to_dict())
        changed_target['target_duration']['seconds'] += 1
        changed_transport = copy.deepcopy(plan.to_dict())
        changed_transport['id'] = 'fresh-ui-id'
        changed_transport['output_path'] = '/tmp/second.json'

        self.assertEqual(rebuilt, plan)
        self.assertEqual(
            matmul_benchmark.collection.CollectionPlan.from_dict(
                changed_transport).sha256(),
            plan.sha256())
        self.assertNotEqual(
            matmul_benchmark.collection.CollectionPlan.from_dict(
                changed_target).sha256(),
            plan.sha256())

    def test_large_work_permission_round_trips_but_does_not_change_hash(self):
        plan = make_plan(allow_large_work=True)
        rebuilt = matmul_benchmark.collection.CollectionPlan.from_dict(
            plan.to_dict())
        version_three = plan.to_dict()
        version_three['schema_version'] = 3
        version_three.pop('allow_large_work')

        self.assertTrue(rebuilt.allow_large_work)
        self.assertEqual(plan.sha256(), make_plan().sha256())
        self.assertFalse(
            matmul_benchmark.collection.CollectionPlan.from_dict(
                version_three).allow_large_work)

    def test_target_duration_cannot_receive_large_work_permission(self):
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError,
                'only for fixed schedules'):
            make_target_plan(allow_large_work=True)

    def test_fixed_and_target_plans_ignore_abstract_work_estimates(self):
        target = make_target_plan()
        fixed = make_plan(
            m_values=(2,), k_values=(3,), n_values=(2,))

        self.assertEqual(
            matmul_benchmark.collection.validate_execution_plan(fixed),
            matmul_benchmark.collection.estimate_plan(fixed))
        self.assertEqual(
            matmul_benchmark.collection.validate_execution_plan(target),
            matmul_benchmark.collection.estimate_plan(target))

        self.assertIsNotNone(target.target_duration)
        self.assertEqual(target.mode.name, 'preview')

    def test_stable_2048_target_uses_duration_planner(self):
        target = matmul_benchmark.duration.TargetDurationSpec(
            seconds=60, mode='stable')
        plan = make_target_plan(
            m_values=(2048,), k_values=(2048,), n_values=(2048,),
            routes=('generic', 'blas_gemm', 'winograd'),
            numpy_baseline=True, target_duration=target)
        estimate = matmul_benchmark.collection.estimate_plan(plan)

        self.assertGreater(estimate.measurement_work, 0)
        self.assertEqual(
            matmul_benchmark.collection.validate_execution_plan(plan),
            estimate)
        self.assertIsNone(
            matmul_benchmark.collection.duration_shard_guard(
                plan).maximum_work)

    def test_checkpoint_resume_keeps_its_recorded_shard_guard(self):
        plan = make_target_plan()
        saved = dataclasses.asdict(
            matmul_benchmark.collection.duration_shard_guard(plan))
        saved['maximum_work'] = 123_456

        guard = matmul_benchmark.collector._duration_run_guard(
            plan, {'shard_guard': saved})

        self.assertEqual(guard.maximum_work, 123_456)
        self.assertEqual(guard.maximum_calls, saved['maximum_calls'])

    def test_huge_target_has_no_abstract_work_ceiling(self):
        target = matmul_benchmark.duration.TargetDurationSpec(
            seconds=60, mode='preview')
        plan = make_target_plan(
            m_values=(16_384,), k_values=(16_384,), n_values=(16_384,),
            routes=('generic', 'blas_gemm', 'winograd'),
            numpy_baseline=True, target_duration=target)
        gibibyte = 1024 ** 3
        budget = matmul_benchmark.arrays.ResourceBudget(
            available_bytes=60 * gibibyte,
            peak_bytes=30 * gibibyte,
            single_allocation_bytes=4 * gibibyte)

        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=budget):
            estimate = \
                matmul_benchmark.collection.validate_execution_plan(plan)

        self.assertEqual(
            estimate, matmul_benchmark.collection.estimate_plan(plan))
        self.assertIsNone(
            matmul_benchmark.collection.duration_shard_guard(
                plan).maximum_work)

    def test_default_plan_is_an_explicit_configurable_shape_grid(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            layout='lhs_padded', padding=3, plan_id='starter')

        self.assertEqual(len(plan.cells), 36)
        self.assertEqual(
            {cell.lhs.shape[-1] for cell in plan.cells}, {64})
        self.assertEqual(
            {cell.lhs.shape[-2] for cell in plan.cells},
            {8, 16, 32, 64, 128, 256})
        self.assertEqual(
            {cell.rhs.shape[-1] for cell in plan.cells},
            {8, 16, 32, 64, 128, 256})
        self.assertTrue(all(
            cell.lhs.strides[0] == 67 for cell in plan.cells))
        self.assertEqual({cell.broadcast for cell in plan.cells}, {'matrix'})
        self.assertEqual(
            matmul_benchmark.collection.CollectionPlan.from_dict(
                plan.to_dict()),
            plan,
        )

    def test_fixed_grid_keeps_slow_routes_for_manual_cancellation(self):
        preview = \
            matmul_benchmark.collection.default_shape_boundary_plan(
                m_values=(1024, 2048), k_values=(1024, 2048),
                n_values=(1024, 2048), routes=(
                    'generic', 'blas_gemm', 'winograd'),
                numpy_baseline=False)
        routes_by_shape = {
            (cell.lhs.shape[-2], cell.lhs.shape[-1],
             cell.rhs.shape[-1]): cell.routes
            for cell in preview.cells
        }

        self.assertIn('generic', routes_by_shape[(1024, 1024, 1024)])
        self.assertEqual(
            routes_by_shape[(2048, 2048, 2048)],
            ('generic', 'blas_gemm', 'winograd'))

    def test_short_fixed_schedule_can_keep_generic_for_2048(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            m_values=(2048,), k_values=(2048,), n_values=(2048,),
            routes=('generic', 'blas_gemm'), numpy_baseline=False,
            mode={
                'name': 'preview',
                'warmups': 0,
                'repetitions': 1,
                'panels': 1,
            })

        self.assertEqual(
            plan.cells[0].routes, ('generic', 'blas_gemm'))

    def test_fixed_2048_preview_keeps_generic_despite_large_work(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            m_values=(2048,), k_values=(2048,), n_values=(2048,),
            routes=('generic', 'blas_gemm', 'winograd'))

        estimate = \
            matmul_benchmark.collection.validate_execution_plan(plan)

        self.assertEqual(
            plan.cells[0].routes,
            ('generic', 'blas_gemm', 'winograd'))
        self.assertEqual(estimate.route_count, 3)
        self.assertGreater(estimate.measurement_work, 900_000_000_000)

    def test_estimate_counts_actual_routes(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            m_values=(3,), k_values=(3,), n_values=(3,),
            routes=('generic', 'blas_gemm', 'winograd'),
            numpy_baseline=False)

        estimate = matmul_benchmark.collection.estimate_plan(plan)
        self.assertEqual(plan.cells[0].routes, ('generic', 'blas_gemm'))
        self.assertEqual(estimate.route_count, 2)

    def test_fixed_grid_accepts_a_large_generic_only_shape(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            m_values=(2048,), k_values=(2048,), n_values=(2048,),
            routes=('generic',), numpy_baseline=False)

        self.assertEqual(plan.cells[0].routes, ('generic',))

    def test_target_duration_does_not_apply_the_fixed_schedule_cap(self):
        plan = make_target_plan(
            m_values=(2048,), k_values=(2048,), n_values=(2048,),
            routes=('generic',), numpy_baseline=False)

        self.assertEqual(plan.cells[0].routes, ('generic',))
        self.assertEqual(
            matmul_benchmark.collection.validate_execution_plan(plan),
            matmul_benchmark.collection.estimate_plan(plan))

    def test_hash_covers_measurements_but_not_transport_identity(self):
        plan = make_plan(output_path='/tmp/first.json')
        changed_transport = plan.to_dict()
        changed_transport['id'] = 'another-run'
        changed_transport['output_path'] = '/tmp/second.json'
        changed_measurement = copy.deepcopy(changed_transport)
        changed_measurement['cells'][0]['lhs']['strides'][0] += 1

        self.assertEqual(
            plan.sha256(),
            matmul_benchmark.collection.CollectionPlan.from_dict(
                changed_transport).sha256(),
        )
        self.assertNotEqual(
            plan.sha256(),
            matmul_benchmark.collection.CollectionPlan.from_dict(
                changed_measurement).sha256(),
        )

    def test_version_one_plan_round_trip_preserves_its_hash_contract(self):
        data = make_plan().to_dict()
        data['schema_version'] = 1
        data.pop('target_duration')
        data.pop('allow_large_work')
        for cell in data['cells']:
            cell.pop('broadcast')

        plan = matmul_benchmark.collection.CollectionPlan.from_dict(data)

        self.assertEqual(plan.to_dict(), data)
        self.assertEqual(
            plan.sha256(),
            matmul_benchmark.collection.CollectionPlan.from_dict(
                plan.to_dict()).sha256())
        self.assertEqual(
            {cell.broadcast for cell in plan.cells}, {'matrix'})

    def test_grid_records_routes_that_are_eligible_for_each_input(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            m_values=(3, 4), k_values=(4,), n_values=(4,),
            plan_id='mixed-eligibility')

        self.assertEqual(
            plan.cells[0].routes, ('generic', 'blas_gemm'))
        self.assertEqual(
            plan.cells[1].routes,
            ('generic', 'blas_gemm', 'winograd'))
        self.assertLess(
            matmul_benchmark.collection.estimate_plan(plan).matmul_calls,
            2 * 108,
        )

    def test_grid_expands_layout_and_broadcast_profiles_explicitly(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            m_values=(2,), k_values=(4,), n_values=(6,),
            layouts=('contiguous', 'lhs_padded'), padding=3,
            broadcasts=(
                'matrix', 'matched_batch', 'broadcast_lhs',
                'broadcast_rhs'),
            batch_size=5, plan_id='profile-grid')
        cells = {
            (cell.layout, cell.broadcast): cell for cell in plan.cells
        }

        self.assertEqual(len(cells), 8)
        self.assertEqual(len({cell.cell_id for cell in plan.cells}), 8)
        self.assertEqual(
            cells['contiguous', 'matched_batch'].lhs.shape,
            (5, 2, 4))
        self.assertEqual(
            cells['contiguous', 'matched_batch'].rhs.shape,
            (5, 4, 6))
        self.assertEqual(
            cells['contiguous', 'broadcast_lhs'].lhs.shape,
            (1, 2, 4))
        self.assertEqual(
            cells['contiguous', 'broadcast_lhs'].lhs.strides[0], 0)
        self.assertEqual(
            cells['contiguous', 'broadcast_lhs'].rhs.shape,
            (5, 4, 6))
        self.assertEqual(
            cells['contiguous', 'broadcast_rhs'].rhs.shape,
            (1, 4, 6))
        self.assertEqual(
            cells['contiguous', 'broadcast_rhs'].rhs.strides[0], 0)
        self.assertEqual(
            cells['lhs_padded', 'matrix'].lhs.strides, (7, 1))
        self.assertNotIn(
            'winograd', cells['contiguous', 'matched_batch'].routes)
        self.assertEqual(
            matmul_benchmark.collection.CollectionPlan.from_dict(
                plan.to_dict()),
            plan,
        )

    def test_grid_rejects_profiles_without_an_eligible_dispatch(self):
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError,
                'no selected dispatch is eligible'):
            matmul_benchmark.collection.default_shape_boundary_plan(
                m_values=(2,), k_values=(4,), n_values=(6,),
                broadcasts=('broadcast_lhs',), routes=('winograd',))

    def test_schema_rejects_derived_sampling_and_aggregate_overruns(self):
        plan = make_plan()
        with_feature = plan.to_dict()
        with_feature['feature_expression'] = 'log2(M * K * N)'
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'unknown fields'):
            matmul_benchmark.collection.CollectionPlan.from_dict(
                with_feature)

        with unittest.mock.patch.object(
                matmul_benchmark.collection,
                'MAX_COLLECTION_CALLS', 1):
            with self.assertRaisesRegex(MemoryError, 'matmul calls'):
                matmul_benchmark.collection.CollectionPlan.from_dict(
                    plan.to_dict())

    def test_grid_rejects_cell_product_before_materializing_cells(self):
        with unittest.mock.patch.object(
                matmul_benchmark.collection, 'CollectionCell') as cell_type:
            with self.assertRaisesRegex(
                    matmul_benchmark.schema.SchemaError,
                    '16385 cells, limit is 16384'):
                matmul_benchmark.collection.default_shape_boundary_plan(
                    m_values=tuple(range(1, 16386)),
                    k_values=(1,), n_values=(1,))

        cell_type.assert_not_called()

    def test_estimate_accounts_for_every_planned_call_and_byte(self):
        plan = make_plan(
            m_values=(2,), k_values=(3,), n_values=(4,),
            mode={
                'name': 'stable',
                'warmups': 1,
                'repetitions': 2,
                'panels': 3,
            },
            numpy_baseline=True,
        )

        estimate = matmul_benchmark.collection.estimate_plan(plan)

        self.assertEqual(estimate.cell_count, 1)
        self.assertEqual(estimate.route_count, 2)
        self.assertEqual(estimate.panel_count, 3)
        self.assertEqual(estimate.preflight_calls, 4)
        self.assertEqual(estimate.matmul_calls, 53)
        self.assertEqual(estimate.scalar_contractions, 1272)
        self.assertEqual(estimate.measurement_work, 2650)
        self.assertEqual(estimate.peak_bytes, 400)
        self.assertEqual(
            matmul_benchmark.collection.estimate_artifact_bytes(plan),
            32_000)
        with unittest.mock.patch.object(
                matmul_benchmark.collection,
                'MAX_COLLECTION_ARTIFACT_BYTES', 31_999):
            with self.assertRaisesRegex(
                    MemoryError, 'artifact estimate needs 32000'):
                matmul_benchmark.collection.estimate_plan(plan)

    def test_estimate_includes_logical_operand_materialization(self):
        cell = matmul_benchmark.collection.CollectionCell(
            cell_id='broadcast',
            lhs=matmul_benchmark.schema.OperandSpec(
                shape=(4, 3), strides=(0, 0)),
            rhs=matmul_benchmark.schema.OperandSpec(
                shape=(3, 2), strides=(0, 0)),
            routes=('generic',),
        )
        plan = matmul_benchmark.collection.CollectionPlan(
            cells=(cell,), routes=('generic',),
            numpy_baseline=False, plan_id='logical-materialization')

        estimate = matmul_benchmark.collection.estimate_plan(plan)

        self.assertEqual(estimate.peak_bytes, 336)

    def test_collection_peak_retains_operands_but_not_all_references(self):
        cells = tuple(
            matmul_benchmark.collection.CollectionCell(
                cell_id=f'broadcast-{index}',
                lhs=matmul_benchmark.schema.OperandSpec(
                    shape=(4, 3), strides=(0, 0)),
                rhs=matmul_benchmark.schema.OperandSpec(
                    shape=(3, 2), strides=(0, 0)),
                routes=('generic',))
            for index in range(2)
        )
        plan = matmul_benchmark.collection.CollectionPlan(
            cells=cells, routes=('generic',), numpy_baseline=False,
            plan_id='streamed-references')

        estimate = matmul_benchmark.collection.estimate_plan(plan)

        legacy_data = plan.to_dict()
        legacy_data['schema_version'] = 1
        legacy_data.pop('target_duration')
        legacy_data.pop('allow_large_work')
        for cell in legacy_data['cells']:
            cell.pop('broadcast')
        legacy = matmul_benchmark.collection.CollectionPlan.from_dict(
            legacy_data)

        self.assertEqual(estimate.peak_bytes, 416)
        self.assertEqual(
            matmul_benchmark.collection.estimate_plan(
                legacy).peak_bytes,
            352)

    def test_large_collection_has_a_soft_budget_below_the_hard_cap(self):
        estimate = matmul_benchmark.collection.CollectionEstimate(
            cell_count=4097, route_count=1, panel_count=2,
            preflight_calls=1, matmul_calls=1,
            scalar_contractions=1,
            measurement_work=200_000_000_001,
            peak_bytes=1)

        warnings = \
            matmul_benchmark.collection.recommended_budget_warnings(
                estimate)

        self.assertEqual(len(warnings), 1)
        self.assertIn('4,097 cells', warnings[0])
        self.assertEqual(
            matmul_benchmark.collection.MAX_COLLECTION_CELLS, 16_384)

    def test_host_budget_allows_accelerated_16384_square(self):
        gibibyte = 1024 ** 3
        host_budget = resource_budget(
            30 * gibibyte, single_allocation_bytes=4 * gibibyte)
        mode = {
            'name': 'preview', 'warmups': 0,
            'repetitions': 1, 'panels': 1,
        }
        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=host_budget):
            with unittest.mock.patch.object(
                    matmul_benchmark.arrays,
                    'make_strided_array') as make_array:
                float32 = make_plan(
                    m_values=(16_384,), k_values=(16_384,),
                    n_values=(16_384,), dtype='float32', mode=mode,
                    routes=('blas_gemm',), numpy_baseline=False)
                float64 = make_plan(
                    m_values=(16_384,), k_values=(16_384,),
                    n_values=(16_384,), dtype='float64', mode=mode,
                    routes=('blas_gemm',), numpy_baseline=False)
                float32_estimate = \
                    matmul_benchmark.collection.estimate_plan(float32)
                float64_estimate = \
                    matmul_benchmark.collection.estimate_plan(float64)

                self.assertEqual(
                    float32_estimate.peak_bytes, 25 * gibibyte // 2)
                self.assertEqual(
                    float64_estimate.peak_bytes, 25 * gibibyte)
                self.assertFalse(
                    matmul_benchmark.collection
                    .requires_large_work_approval(
                        float32, float32_estimate))
                self.assertEqual(
                    matmul_benchmark.collection.validate_execution_plan(
                        float32),
                    float32_estimate)
                self.assertEqual(
                    matmul_benchmark.collection.validate_execution_plan(
                        float64),
                    float64_estimate)
        make_array.assert_not_called()

    def test_hand_built_large_generic_route_is_not_work_capped(self):
        gibibyte = 1024 ** 3
        host_budget = resource_budget(
            30 * gibibyte, single_allocation_bytes=4 * gibibyte)
        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=host_budget):
            operand = matmul_benchmark.schema.OperandSpec(
                shape=(16_384, 16_384), strides=(16_384, 1))
            cell = matmul_benchmark.collection.CollectionCell(
                cell_id='unsafe-generic', lhs=operand, rhs=operand,
                routes=('generic',))
            plan = matmul_benchmark.collection.CollectionPlan(
                cells=(cell,), dtype='float32', routes=('generic',),
                numpy_baseline=False,
                mode=matmul_benchmark.schema.ModeSpec(
                    name='preview', warmups=0,
                    repetitions=1, panels=1),
                allow_large_work=True)

            self.assertEqual(
                matmul_benchmark.collection.validate_execution_plan(plan),
                matmul_benchmark.collection.estimate_plan(plan))

    def test_worker_memory_snapshot_can_reject_a_previously_valid_plan(self):
        gibibyte = 1024 ** 3
        generous = resource_budget(
            30 * gibibyte, single_allocation_bytes=4 * gibibyte)
        constrained = resource_budget(
            20 * gibibyte, single_allocation_bytes=4 * gibibyte)
        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=generous):
            plan = make_plan(
                m_values=(16_384,), k_values=(16_384,),
                n_values=(16_384,), dtype='float64',
                routes=('blas_gemm',), numpy_baseline=False,
                mode={
                    'name': 'preview', 'warmups': 0,
                    'repetitions': 1, 'panels': 1,
                })
            payload = plan.to_dict()

        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=constrained):
            with unittest.mock.patch.object(
                    matmul_benchmark.arrays,
                    'make_strided_array') as make_array:
                rebuilt = \
                    matmul_benchmark.collection.CollectionPlan.from_dict(
                        payload)
                self.assertEqual(
                    matmul_benchmark.collection.estimate_plan(rebuilt),
                    matmul_benchmark.collection.estimate_plan(plan))
                with self.assertRaisesRegex(
                        MemoryError, 'current worker-safe limit'):
                    matmul_benchmark.collection.validate_execution_plan(
                        rebuilt)
        make_array.assert_not_called()

    def test_sample_heavy_plan_warns_before_the_artifact_hard_cap(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            m_values=tuple(range(1, 105)), k_values=(1,), n_values=(1,),
            routes=('generic',), numpy_baseline=False,
            mode={
                'name': 'stable', 'warmups': 0,
                'repetitions': 1, 'panels': 1000,
            })
        estimate = matmul_benchmark.collection.estimate_plan(plan)

        warnings = \
            matmul_benchmark.collection.recommended_budget_warnings(
                estimate, plan)

        self.assertEqual(estimate.matmul_calls, 416_312)
        self.assertEqual(
            matmul_benchmark.collection.estimate_artifact_bytes(plan),
            533_012_480)
        self.assertEqual(len(warnings), 2)
        self.assertIn('416,312 calls', warnings[0])
        self.assertIn('533,012,480 projected artifact bytes', warnings[1])

    def test_collection_rejects_correctness_temporary_peak(self):
        constrained = resource_budget(512 * 1024 ** 2)
        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=constrained):
            plan = \
                matmul_benchmark.collection.default_shape_boundary_plan(
                    m_values=(6000,), k_values=(1,), n_values=(6000,),
                    routes=('generic',), numpy_baseline=False)
            with self.assertRaisesRegex(
                    MemoryError,
                    'collection peak estimate needs 1152096000'):
                matmul_benchmark.collection.validate_execution_plan(plan)

        version_one = {
            'schema_version': 1,
            'schema_kind': matmul_benchmark.collection.PLAN_KIND,
            'id': 'unsafe-version-one',
            'cells': [{
                'id': 'large-output',
                'lhs': {'shape': [6000, 1], 'strides': [1, 1]},
                'rhs': {'shape': [1, 6000], 'strides': [6000, 1]},
                'layout': 'contiguous',
                'routes': ['generic'],
            }],
            'dtype': 'float32',
            'mode': matmul_benchmark.schema.ModeSpec().to_dict(),
            'routes': ['generic'],
            'numpy_baseline': False,
            'seed': 1,
            'threads': 1,
            'output_path': None,
        }
        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=constrained):
            legacy = \
                matmul_benchmark.collection.CollectionPlan.from_dict(
                    version_one)
            with self.assertRaisesRegex(
                    MemoryError,
                    'collection peak estimate needs 1152096000'):
                matmul_benchmark.collection.validate_execution_plan(legacy)

    def test_logical_materialization_contributes_to_peak_cap(self):
        cell = matmul_benchmark.collection.CollectionCell(
            cell_id='broadcast',
            lhs=matmul_benchmark.schema.OperandSpec(
                shape=(4, 3), strides=(0, 0)),
            rhs=matmul_benchmark.schema.OperandSpec(
                shape=(3, 2), strides=(0, 0)),
            routes=('generic',),
        )

        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=resource_budget(150)):
            plan = matmul_benchmark.collection.CollectionPlan(
                cells=(cell,), routes=('generic',),
                numpy_baseline=False, plan_id='peak-cap')
            with self.assertRaisesRegex(
                    MemoryError, 'collection peak estimate needs 336'):
                matmul_benchmark.collection.validate_execution_plan(plan)

    def test_cell_order_is_seeded_and_position_balanced(self):
        plan = make_plan()
        repeated = matmul_benchmark.collection.CollectionPlan.from_dict(
            plan.to_dict())
        orders = matmul_benchmark.collection.panel_cell_orders(plan)

        self.assertEqual(
            orders,
            matmul_benchmark.collection.panel_cell_orders(repeated),
        )
        self.assertEqual(len(set(orders)), plan.mode.panels)
        for order in orders:
            self.assertEqual(sorted(order), list(range(len(plan.cells))))


class PlanCollectorTC(unittest.TestCase):
    def _collect(self, plan, engine, progress=None, cancelled=None):
        with unittest.mock.patch.dict(
                os.environ, thread_environment(), clear=False):
            with unittest.mock.patch(
                    'solvcon.matmul_benchmark.collector._metadata',
                    return_value=fake_metadata()):
                return matmul_benchmark.collector.collect_plan(
                    plan, engine=engine, clock=StepClock(),
                    progress=progress, cancelled=cancelled)

    def _collect_target(self, plan, engine, progress=None,
                        cancelled=None, checkpoint_path=None):
        with unittest.mock.patch.dict(
                os.environ, thread_environment(), clear=False):
            with unittest.mock.patch(
                    'solvcon.matmul_benchmark.collector._metadata',
                    return_value=fake_metadata()):
                return matmul_benchmark.collector.collect_plan(
                    plan, engine=engine, clock=StepClock(100_000),
                    wall_clock=StepClock(1_000_000),
                    progress=progress, cancelled=cancelled,
                    checkpoint_path=checkpoint_path)

    def test_legacy_requested_route_count_remains_loadable(self):
        plan = make_plan(
            m_values=(3,), k_values=(3,), n_values=(3,),
            routes=('generic', 'blas_gemm', 'winograd'))
        document = self._collect(plan, FakeEngine())
        self.assertEqual(document['estimate']['route_count'], 2)

        document['estimate']['route_count'] = len(plan.routes)

        matmul_benchmark.schema.validate_document(document)

    def test_large_work_allows_auto_generic_preparation(self):
        gibibyte = 1024 ** 3
        budget = resource_budget(
            30 * gibibyte, single_allocation_bytes=4 * gibibyte)
        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=budget):
            plan = make_plan(
                m_values=(16_384,), k_values=(16_384,),
                n_values=(16_384,), dtype='float32',
                routes=('blas_gemm',), numpy_baseline=False,
                mode={
                    'name': 'preview', 'warmups': 0,
                    'repetitions': 1, 'panels': 1,
                }, allow_large_work=True)
            small = np.ones((2, 2), dtype='float32')
            with unittest.mock.patch.object(
                    matmul_benchmark.arrays, 'make_strided_array',
                    side_effect=(small, small)):
                with unittest.mock.patch.object(
                        matmul_benchmark.collector, '_metadata',
                        return_value=fake_metadata()):
                    prepared = matmul_benchmark.collector._prepare_case(
                        plan.request_at(0), FakeEngine())

        self.assertEqual(prepared.native_names, ('auto', 'blas_gemm'))

    def test_target_collection_calibrates_then_runs_bounded_shards(self):
        plan = make_target_plan()
        progress = []

        document = self._collect_target(
            plan, FakeEngine(), progress=progress.append)

        run = document['duration_run']
        schedule_data = run['schedule']
        self.assertEqual(run['status'], 'complete')
        self.assertGreater(schedule_data['shard_count'], 1)
        self.assertEqual(
            len(document['sources']),
            len(plan.cells) * schedule_data['shard_count'])
        self.assertEqual(len(document['aggregate_observations']), 1)
        aggregate = document['aggregate_observations'][0]
        self.assertEqual(
            len(aggregate['source_ids']), schedule_data['shard_count'])
        self.assertEqual(
            aggregate['observation']['routes']['generic']['timing'][
                'sample_count'],
            schedule_data['panels'])
        self.assertIn(
            len(run['calibration_measurements']),
            (6 * len(plan.cells), 2 * 6 * len(plan.cells)))
        phases = [
            event['phase'] for event in progress
            if event['type'] == 'progress'
        ]
        self.assertLess(
            phases.index('preparation'), phases.index('calibration'))
        self.assertLess(
            phases.index('calibration'), phases.index('measurement'))
        self.assertEqual(phases[-1], 'finalization')
        checkpoint_load = [
            event for event in progress
            if event.get('phase') == 'checkpoint_load'
        ]
        self.assertEqual(
            [event['state'] for event in checkpoint_load],
            ['started', 'completed'])
        for shard in run['shards']:
            mode = matmul_benchmark.schema.ModeSpec.from_dict(
                shard['mode'])
            self.assertLessEqual(
                mode.warmups + mode.repetitions * mode.panels,
                matmul_benchmark.schema.MAX_MODE_CALLS_PER_ROUTE)
            self.assertLessEqual(
                mode.panels,
                matmul_benchmark.schema.MAX_MODE_PANELS)

        second = run['shards'][1]
        source_id = second['source_ids'][0]
        native_panel = next(
            item['panel'] for item in document['panels']
            if item['source_id'] == source_id
            and item['panel']['scope'] == 'native_batch'
            and item['panel']['index'] == 0)
        self.assertEqual(
            native_panel['order'],
            list(matmul_benchmark.collection.balanced_order_at(
                ('auto', 'generic', 'blas_gemm'),
                second['panel_offset'])))
        matmul_benchmark.schema.validate_document(document)

    def test_target_checkpoint_resumes_by_measurement_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / 'target.json'
            plan = make_target_plan(
                m_values=(2, 4), output_path=str(output))
            checkpoint = \
                matmul_benchmark.collector.duration_checkpoint_path(
                    output, plan.sha256())
            cancel = [False]
            progress_events = []

            def progress(event):
                progress_events.append(event)
                if event['type'] == 'checkpoint':
                    cancel[0] = True

            with self.assertRaises(
                    matmul_benchmark.collector.CollectionCancelled):
                self._collect_target(
                    plan, FakeEngine(), progress=progress,
                    cancelled=lambda: cancel[0],
                    checkpoint_path=checkpoint)

            partial = matmul_benchmark.artifact.load_artifact(checkpoint)
            self.assertEqual(
                partial['duration_run']['status'], 'checkpoint')
            self.assertEqual(len(partial['duration_run']['shards']), 1)
            checkpoint_writes = [
                event for event in progress_events
                if event.get('phase') == 'checkpoint_write'
            ]
            self.assertEqual(
                [event['state'] for event in checkpoint_writes],
                ['started', 'completed'])
            partial['duration_run']['shard_guard']['maximum_work'] = (
                matmul_benchmark.collection.
                LEGACY_MAX_COLLECTION_MEASUREMENT_WORK)
            partial['sources'].reverse()
            matmul_benchmark.artifact.write_artifact(
                partial, checkpoint)
            fresh_plan = dataclasses.replace(
                plan, plan_id='fresh-ui-generated-id')

            completed = self._collect_target(
                fresh_plan, FakeEngine(), checkpoint_path=checkpoint)

            self.assertEqual(
                completed['duration_run']['status'], 'complete')
            self.assertEqual(
                completed['duration_run']['shard_guard']['maximum_work'],
                matmul_benchmark.collection.
                LEGACY_MAX_COLLECTION_MEASUREMENT_WORK)
            self.assertTrue(
                completed['duration_run']['resumed_from_checkpoint'])
            self.assertEqual(completed['duration_run']['resume_count'], 1)
            self.assertEqual(
                completed['aggregate_observations'][0]['observation']['id'],
                f'{plan.plan_id}:{plan.cells[0].cell_id}')

    def test_duration_provenance_rejects_raw_and_status_mutations(self):
        document = self._collect_target(
            make_target_plan(), FakeEngine())
        wrong_input = copy.deepcopy(document)
        wrong_input['observations'][0]['observation']['lhs'][
            'strides'][0] += 1
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'wrong lhs'):
            matmul_benchmark.schema.validate_document(wrong_input)

        duplicate = copy.deepcopy(document)
        duplicate['observations'].append(
            copy.deepcopy(duplicate['observations'][0]))
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError,
                'contiguous and unique|one observation'):
            matmul_benchmark.schema.validate_document(duplicate)

        wrong_status = copy.deepcopy(document)
        wrong_status['duration_run']['status'] = 'checkpoint'
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError,
                'unfinished shards'):
            matmul_benchmark.schema.validate_document(wrong_status)

        wrong_guard = copy.deepcopy(document)
        wrong_guard['duration_run']['shard_guard']['maximum_work'] = 123
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'shard guard'):
            matmul_benchmark.schema.validate_document(wrong_guard)

        wrong_latency = copy.deepcopy(document)
        wrong_latency['panels'][0]['panel']['samples'][0][
            'latency_ns'] += 1
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError,
                'elapsed_ns/repetitions'):
            matmul_benchmark.schema.validate_document(wrong_latency)

    def test_artifact_projection_fails_before_formal_measurement(self):
        plan = make_target_plan()
        progress = []
        with unittest.mock.patch(
                'solvcon.matmul_benchmark.collector.'
                '_duration_artifact_projection',
                return_value=(
                    matmul_benchmark.collection.
                    MAX_COLLECTION_ARTIFACT_BYTES + 1)):
            with self.assertRaisesRegex(
                    MemoryError, 'artifact projection'):
                self._collect_target(
                    plan, FakeEngine(), progress=progress.append)

        self.assertNotIn(
            'measurement', {
                event.get('phase') for event in progress
            })

    def test_large_minimum_calibration_never_becomes_the_pilot(self):
        target = matmul_benchmark.duration.TargetDurationSpec(
            seconds=0.05,
            mode='preview',
            safety_fraction=0.9,
            calibration_block_seconds=1e-6,
            minimum_calibration_repetitions=10_000,
            maximum_calibration_repetitions=10_000,
            checkpoint_seconds=0.005,
        )
        plan = make_target_plan(target_duration=target)
        engine = FakeEngine()

        with self.assertRaises(
                matmul_benchmark.duration.DurationModelError):
            self._collect_target(plan, engine)

        repetitions = [
            event[3] for event in engine.events
            if event[0] == 'benchmark'
        ]
        self.assertTrue(repetitions)
        self.assertLessEqual(max(repetitions), plan.mode.warmups)

    def test_prepared_case_keeps_output_shape_not_numpy_reference(self):
        plan = make_plan(
            m_values=(2,), k_values=(3,), n_values=(4,))

        prepared = matmul_benchmark.collector._prepare_case(
            plan.request_at(0), FakeEngine())

        self.assertEqual(prepared.output_shape, (2, 4))
        self.assertFalse(hasattr(prepared, 'reference'))

    def test_one_engine_interleaves_cells_across_seeded_panels(self):
        plan = make_plan()
        engine = FakeEngine()
        progress = []

        document = self._collect(
            plan, engine, progress=progress.append)

        self.assertEqual(len(engine.cases), len(plan.cells))
        panel_progress = [
            event for event in progress if 'state' not in event]
        activity = [event for event in progress if 'state' in event]
        self.assertEqual(len(panel_progress), 9)
        self.assertEqual(panel_progress[-1]['completed'], 9)
        self.assertEqual(panel_progress[-1]['total'], 9)
        self.assertEqual(panel_progress[0]['panel'], 1)
        self.assertEqual(panel_progress[0]['cells'], 3)
        self.assertEqual(set(panel_progress[0]['shape']), {
            'm', 'k', 'n', 'lhs', 'rhs'})
        self.assertEqual(
            {event['cell_id'] for event in activity
             if event['cell_id'] is not None},
            {cell.cell_id for cell in plan.cells})
        self.assertEqual(activity[0]['phase'], 'provenance')
        self.assertIsNone(activity[0]['cell_id'])

        measured = [
            event for event in engine.events
            if event[0] == 'benchmark' and event[3] == 2
        ]
        measured_cells = [
            measured[index][1]
            for index in range(0, len(measured), 3)
        ]
        expected_cells = [
            cell_index
            for order in matmul_benchmark.collection.panel_cell_orders(plan)
            for cell_index in order
        ]
        self.assertEqual(measured_cells, expected_cells)

        self.assertEqual(
            document['schema_kind'],
            matmul_benchmark.schema.COLLECTION_KIND,
        )
        self.assertEqual(document['plan_sha256'], plan.sha256())
        self.assertEqual(document['artifact_count'], len(plan.cells))
        self.assertEqual(len(document['observations']), len(plan.cells))
        self.assertEqual(
            len(document['panels']),
            2 * plan.mode.panels * len(plan.cells))
        self.assertEqual(document['cell_orders'], [
            [plan.cells[index].cell_id for index in order]
            for order in matmul_benchmark.collection.panel_cell_orders(plan)
        ])
        first_source = document['sources'][0]['source_id']
        first_panels = [
            item['panel'] for item in document['panels']
            if item['source_id'] == first_source
            and item['panel']['scope'] == 'native_batch'
        ]
        generic_samples = [
            sample['latency_ns'] for panel in first_panels
            for sample in panel['samples']
            if sample['route'] == 'generic'
        ]
        first_observation = document['observations'][0]['observation']
        self.assertEqual(
            matmul_benchmark.collector._summarize(generic_samples),
            first_observation['routes']['generic']['timing'])
        matmul_benchmark.schema.validate_document(document)
        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=resource_budget(1)):
            matmul_benchmark.schema.validate_document(document)

        missing_observation = copy.deepcopy(document)
        missing_observation['observations'].pop()
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError,
                'observation count'):
            matmul_benchmark.schema.validate_document(missing_observation)

        changed_plan = copy.deepcopy(document)
        changed_plan['plan']['cells'][0]['lhs']['strides'][0] += 1
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'hash'):
            matmul_benchmark.schema.validate_document(changed_plan)

        swapped_observations = copy.deepcopy(document)
        first = swapped_observations['observations'][0]
        second = swapped_observations['observations'][1]
        first['observation'], second['observation'] = (
            second['observation'], first['observation'])
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError,
                'observation .* wrong'):
            matmul_benchmark.schema.validate_document(swapped_observations)

        inconsistent_selection = copy.deepcopy(document)
        observation = inconsistent_selection['observations'][0][
            'observation']
        selected = observation['auto_route']
        observation['routes'][selected]['selected_by_auto'] = False
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError,
                'inconsistent auto selection'):
            matmul_benchmark.schema.validate_document(
                inconsistent_selection)

        mislabeled_contraction = copy.deepcopy(document)
        mislabeled_contraction['observations'][0]['observation'][
            'contraction']['m'] += 1
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'contraction'):
            matmul_benchmark.schema.validate_document(
                mislabeled_contraction)

        missing_panel = copy.deepcopy(document)
        missing_panel['panels'].pop(0)
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'panel indexes'):
            matmul_benchmark.schema.validate_document(missing_panel)

        duplicate_panel = copy.deepcopy(document)
        duplicate_panel['panels'].append(
            copy.deepcopy(duplicate_panel['panels'][0]))
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'panel indexes'):
            matmul_benchmark.schema.validate_document(duplicate_panel)

        mislabeled_panel = copy.deepcopy(document)
        mislabeled_panel['panels'][0]['source_id'] = (
            mislabeled_panel['sources'][1]['source_id'])
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'artifact_id'):
            matmul_benchmark.schema.validate_document(mislabeled_panel)

        swapped_panels = copy.deepcopy(document)
        first_panel = swapped_panels['panels'][0]
        second_panel = swapped_panels['panels'][1]
        first_panel['panel'], second_panel['panel'] = (
            second_panel['panel'], first_panel['panel'])
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'source digest'):
            matmul_benchmark.schema.validate_document(swapped_panels)

        changed_order = copy.deepcopy(document)
        changed_order['cell_orders'][0].reverse()
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'cell orders'):
            matmul_benchmark.schema.validate_document(changed_order)

    def test_cancellation_after_progress_never_returns_a_collection(self):
        plan = make_plan()
        cancelled = [False]

        def progress(_event):
            cancelled[0] = True

        with self.assertRaises(
                matmul_benchmark.collector.CollectionCancelled):
            self._collect(
                plan, FakeEngine(), progress=progress,
                cancelled=lambda: cancelled[0])

    def test_worker_error_does_not_publish_the_output_path(self):
        plan = make_plan()
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / 'collection.json'
            request = plan.to_dict()
            request['output_path'] = str(output)
            protocol = io.StringIO()
            with unittest.mock.patch(
                    'solvcon.matmul_benchmark.worker._read_request',
                    return_value=request):
                with unittest.mock.patch(
                        'solvcon.matmul_benchmark.collector.collect_plan',
                        side_effect=RuntimeError('collector failed')):
                    with unittest.mock.patch('sys.stdout', protocol):
                        status = matmul_benchmark.worker.main([])

            self.assertEqual(status, 1)
            self.assertFalse(output.exists())
            event = json.loads(protocol.getvalue())
            self.assertEqual(event['type'], 'error')
            self.assertEqual(event['message'], 'collector failed')

    def test_worker_publish_failure_has_one_terminal_error(self):
        plan = make_plan()
        protocol = io.StringIO()
        with unittest.mock.patch(
                'solvcon.matmul_benchmark.worker._read_request',
                return_value=plan.to_dict()):
            with unittest.mock.patch(
                    'solvcon.matmul_benchmark.collector.collect_plan',
                    return_value={'collection_id': 'unfinished'}):
                with unittest.mock.patch(
                        'solvcon.matmul_benchmark.artifact.write_artifact',
                        side_effect=OSError('publish failed')):
                    with unittest.mock.patch('sys.stdout', protocol):
                        status = matmul_benchmark.worker.main([])

        events = [
            json.loads(line) for line in protocol.getvalue().splitlines()
        ]
        self.assertEqual(status, 1)
        self.assertEqual(
            [event['state'] for event in events[:-1]],
            ['started', 'failed'])
        self.assertEqual(events[-1]['type'], 'error')
        self.assertEqual(events[-1]['message'], 'publish failed')

    def test_worker_deletes_only_hash_qualified_checkpoint_after_publish(self):
        plan = make_target_plan()
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / 'final.json'
            events = []

            def collect(request, progress, checkpoint_path):
                self.assertEqual(request.output_path, str(output))
                self.assertEqual(
                    checkpoint_path,
                    matmul_benchmark.collector.duration_checkpoint_path(
                        output, request.sha256()))
                checkpoint_path.write_text('checkpoint', encoding='utf8')
                events.append('checkpoint')
                return {'collection_id': 'finished'}

            def publish(_document, path):
                self.assertTrue(
                    matmul_benchmark.collector.duration_checkpoint_path(
                        output, plan.sha256()).exists())
                events.append('publish')
                return pathlib.Path(path)

            protocol = io.StringIO()
            with unittest.mock.patch(
                    'solvcon.matmul_benchmark.worker._read_request',
                    return_value=plan.to_dict()):
                with unittest.mock.patch(
                        'solvcon.matmul_benchmark.collector.collect_plan',
                        side_effect=collect):
                    with unittest.mock.patch(
                            'solvcon.matmul_benchmark.artifact.write_artifact',
                            side_effect=publish):
                        with unittest.mock.patch('sys.stdout', protocol):
                            status = matmul_benchmark.worker.main(
                                ['--output', str(output)])

            checkpoint = \
                matmul_benchmark.collector.duration_checkpoint_path(
                    output, plan.sha256())
            self.assertEqual(status, 0)
            self.assertEqual(events, ['checkpoint', 'publish'])
            self.assertFalse(checkpoint.exists())

    def test_worker_reports_success_when_checkpoint_cleanup_fails(self):
        plan = make_target_plan()
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / 'final.json'
            checkpoint = \
                matmul_benchmark.collector.duration_checkpoint_path(
                    output, plan.sha256())

            def collect(_request, progress, checkpoint_path):
                checkpoint_path.write_text('checkpoint', encoding='utf8')
                return {'collection_id': 'finished'}

            protocol = io.StringIO()
            with unittest.mock.patch(
                    'solvcon.matmul_benchmark.worker._read_request',
                    return_value=plan.to_dict()):
                with unittest.mock.patch(
                        'solvcon.matmul_benchmark.collector.collect_plan',
                        side_effect=collect):
                    with unittest.mock.patch(
                            'solvcon.matmul_benchmark.artifact.write_artifact',
                            return_value=output):
                        with unittest.mock.patch.object(
                                pathlib.Path, 'unlink',
                                side_effect=OSError('busy')):
                            with unittest.mock.patch(
                                    'sys.stdout', protocol):
                                status = matmul_benchmark.worker.main(
                                    ['--output', str(output)])

            self.assertEqual(status, 0)
            self.assertTrue(checkpoint.exists())
            events = [
                json.loads(line)
                for line in protocol.getvalue().splitlines()
            ]
            self.assertEqual(events[-1]['type'], 'result')
            self.assertEqual(
                [event['state'] for event in events
                 if event.get('phase') == 'artifact_write'],
                ['started', 'completed'])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
