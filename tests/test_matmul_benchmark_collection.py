# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import copy
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
            FakeRoute('naive', selected_by_auto=True),
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
        'routes': ('naive', 'blas_gemm'),
        'numpy_baseline': False,
        'seed': 91,
        'plan_id': 'collection-test',
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
                    'naive', 'blas_gemm', 'winograd'),
                numpy_baseline=False)
        routes_by_shape = {
            (cell.lhs.shape[-2], cell.lhs.shape[-1],
             cell.rhs.shape[-1]): cell.routes
            for cell in preview.cells
        }

        self.assertIn('naive', routes_by_shape[(1024, 1024, 1024)])
        self.assertEqual(
            routes_by_shape[(2048, 2048, 2048)],
            ('naive', 'blas_gemm', 'winograd'))

    def test_short_fixed_schedule_can_keep_naive_for_2048(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            m_values=(2048,), k_values=(2048,), n_values=(2048,),
            routes=('naive', 'blas_gemm'), numpy_baseline=False,
            mode={
                'name': 'preview',
                'warmups': 0,
                'repetitions': 1,
                'panels': 1,
            })

        self.assertEqual(
            plan.cells[0].routes, ('naive', 'blas_gemm'))

    def test_fixed_2048_preview_keeps_naive_despite_large_work(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            m_values=(2048,), k_values=(2048,), n_values=(2048,),
            routes=('naive', 'blas_gemm', 'winograd'))

        estimate = \
            matmul_benchmark.collection.validate_execution_plan(plan)

        self.assertEqual(
            plan.cells[0].routes,
            ('naive', 'blas_gemm', 'winograd'))
        self.assertEqual(estimate.route_count, 3)
        self.assertGreater(estimate.measurement_work, 500_000_000_000)

    def test_estimate_counts_actual_routes(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            m_values=(3,), k_values=(3,), n_values=(3,),
            routes=('naive', 'blas_gemm', 'winograd'),
            numpy_baseline=False)

        estimate = matmul_benchmark.collection.estimate_plan(plan)
        self.assertEqual(plan.cells[0].routes, ('naive', 'blas_gemm'))
        self.assertEqual(estimate.route_count, 2)

    def test_fixed_grid_accepts_a_large_naive_only_shape(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            m_values=(2048,), k_values=(2048,), n_values=(2048,),
            routes=('naive',), numpy_baseline=False)

        self.assertEqual(plan.cells[0].routes, ('naive',))

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

    def test_grid_records_routes_that_are_eligible_for_each_input(self):
        plan = matmul_benchmark.collection.default_shape_boundary_plan(
            m_values=(3, 4), k_values=(4,), n_values=(4,),
            plan_id='mixed-eligibility')

        self.assertEqual(
            plan.cells[0].routes, ('naive', 'blas_gemm'))
        self.assertEqual(
            plan.cells[1].routes,
            ('naive', 'blas_gemm', 'winograd'))
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
        self.assertEqual(estimate.matmul_calls, 32)
        self.assertEqual(estimate.scalar_contractions, 768)
        self.assertEqual(estimate.measurement_work, 1600)
        self.assertEqual(estimate.peak_bytes, 400)
        self.assertEqual(
            matmul_benchmark.collection.estimate_artifact_bytes(plan),
            20_480)
        with unittest.mock.patch.object(
                matmul_benchmark.collection,
                'MAX_COLLECTION_ARTIFACT_BYTES', 20_479):
            with self.assertRaisesRegex(
                    MemoryError, 'artifact estimate needs 20480'):
                matmul_benchmark.collection.estimate_plan(plan)

    def test_estimate_includes_logical_operand_materialization(self):
        cell = matmul_benchmark.collection.CollectionCell(
            cell_id='broadcast',
            lhs=matmul_benchmark.schema.OperandSpec(
                shape=(4, 3), strides=(0, 0)),
            rhs=matmul_benchmark.schema.OperandSpec(
                shape=(3, 2), strides=(0, 0)),
            routes=('naive',),
        )
        plan = matmul_benchmark.collection.CollectionPlan(
            cells=(cell,), routes=('naive',),
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
                routes=('naive',))
            for index in range(2)
        )
        plan = matmul_benchmark.collection.CollectionPlan(
            cells=cells, routes=('naive',), numpy_baseline=False,
            plan_id='streamed-references')

        estimate = matmul_benchmark.collection.estimate_plan(plan)

        self.assertEqual(estimate.peak_bytes, 416)

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
                self.assertEqual(
                    matmul_benchmark.collection.validate_execution_plan(
                        float32),
                    float32_estimate)
                self.assertEqual(
                    matmul_benchmark.collection.validate_execution_plan(
                        float64),
                    float64_estimate)
        make_array.assert_not_called()

    def test_hand_built_large_naive_route_is_not_work_capped(self):
        gibibyte = 1024 ** 3
        host_budget = resource_budget(
            30 * gibibyte, single_allocation_bytes=4 * gibibyte)
        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=host_budget):
            operand = matmul_benchmark.schema.OperandSpec(
                shape=(16_384, 16_384), strides=(16_384, 1))
            cell = matmul_benchmark.collection.CollectionCell(
                cell_id='unsafe-naive', lhs=operand, rhs=operand,
                routes=('naive',))
            plan = matmul_benchmark.collection.CollectionPlan(
                cells=(cell,), dtype='float32', routes=('naive',),
                numpy_baseline=False,
                mode=matmul_benchmark.schema.ModeSpec(
                    name='preview', warmups=0,
                    repetitions=1, panels=1))

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

    def test_collection_rejects_correctness_temporary_peak(self):
        constrained = resource_budget(512 * 1024 ** 2)
        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=constrained):
            plan = \
                matmul_benchmark.collection.default_shape_boundary_plan(
                    m_values=(6000,), k_values=(1,), n_values=(6000,),
                    routes=('naive',), numpy_baseline=False)
            with self.assertRaisesRegex(
                    MemoryError,
                    'collection peak estimate needs 1152096000'):
                matmul_benchmark.collection.validate_execution_plan(plan)

    def test_logical_materialization_contributes_to_peak_cap(self):
        cell = matmul_benchmark.collection.CollectionCell(
            cell_id='broadcast',
            lhs=matmul_benchmark.schema.OperandSpec(
                shape=(4, 3), strides=(0, 0)),
            rhs=matmul_benchmark.schema.OperandSpec(
                shape=(3, 2), strides=(0, 0)),
            routes=('naive',),
        )

        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=resource_budget(150)):
            plan = matmul_benchmark.collection.CollectionPlan(
                cells=(cell,), routes=('naive',),
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
    def _collect(self, plan, engine, progress=None, cancelled=None,
                 partial_path=None):
        with unittest.mock.patch.dict(
                os.environ, thread_environment(), clear=False):
            with unittest.mock.patch(
                    'solvcon.matmul_benchmark.collector._metadata',
                    return_value=fake_metadata()):
                return matmul_benchmark.collector.collect_plan(
                    plan, engine=engine, clock=StepClock(),
                    progress=progress, cancelled=cancelled,
                    partial_path=partial_path)

    def test_large_work_allows_auto_naive_preparation(self):
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
                })
            small = np.ones((2, 2), dtype='float32')
            with unittest.mock.patch.object(
                    matmul_benchmark.arrays, 'make_strided_array',
                    side_effect=(small, small)):
                with unittest.mock.patch.object(
                        matmul_benchmark.collector, '_metadata',
                        return_value=fake_metadata()):
                    prepared = matmul_benchmark.collector._prepare_case(
                        plan.request_at(0), FakeEngine())

        self.assertEqual(prepared.names, ('auto', 'blas_gemm'))

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
            event for event in activity
            if event['phase'] == 'timing'
            and event['state'] == 'started'
        ]
        measured_cells = [
            measured[index]['cell_id']
            for index in range(0, len(measured), 3)
        ]
        expected_cells = [
            plan.cells[cell_index].cell_id
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
            plan.mode.panels * len(plan.cells))
        self.assertEqual(document['cell_orders'], [
            [plan.cells[index].cell_id for index in order]
            for order in matmul_benchmark.collection.panel_cell_orders(plan)
        ])
        first_source = document['sources'][0]['source_id']
        first_panels = [
            item['panel'] for item in document['panels']
            if item['source_id'] == first_source
        ]
        naive_samples = [
            sample['latency_ns'] for panel in first_panels
            for sample in panel['samples']
            if sample['route'] == 'naive'
        ]
        first_observation = document['observations'][0]['observation']
        self.assertEqual(
            matmul_benchmark.collector._summarize(naive_samples),
            first_observation['routes']['naive']['timing'])
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

    def test_cancellation_keeps_the_last_complete_measurement_round(self):
        plan = make_plan(
            m_values=(2,), k_values=(3,), n_values=(2,),
            mode={
                'name': 'preview', 'warmups': 0,
                'repetitions': 1, 'panels': 2,
            })
        cancelled = [False]
        events = []

        def progress(event):
            events.append(event)
            if event['type'] == 'partial':
                cancelled[0] = True

        with tempfile.TemporaryDirectory() as directory:
            partial_path = pathlib.Path(directory) / 'partial.json'
            with self.assertRaises(
                    matmul_benchmark.collector.CollectionCancelled):
                self._collect(
                    plan, FakeEngine(), progress=progress,
                    cancelled=lambda: cancelled[0],
                    partial_path=partial_path)
            partial = matmul_benchmark.artifact.load_artifact(
                partial_path)

        self.assertEqual(partial['plan']['mode']['panels'], 1)
        self.assertEqual(len(partial['cell_orders']), 1)
        self.assertEqual(
            [event['type'] for event in events if 'state' not in event][-1],
            'partial')

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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
