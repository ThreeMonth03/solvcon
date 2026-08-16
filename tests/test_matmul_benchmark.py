# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import copy
import io
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np

import solvcon as sc
from solvcon import matmul_benchmark


class FakeRoute:
    def __init__(self, name, selected_by_auto=False, packing=None):
        packing = packing or {}
        self.name = name
        self.selected_by_auto = selected_by_auto
        self.eager_pack_lhs = packing.get('eager_lhs', False)
        self.eager_pack_rhs = packing.get('eager_rhs', False)
        self.scratch_pack_lhs = packing.get('scratch_lhs', False)
        self.scratch_pack_rhs = packing.get('scratch_rhs', False)

    def packing_dict(self):
        return {
            'eager_lhs': self.eager_pack_lhs,
            'eager_rhs': self.eager_pack_rhs,
            'scratch_lhs': self.scratch_pack_lhs,
            'scratch_rhs': self.scratch_pack_rhs,
        }


class FakeCase:
    def __init__(self, lhs, rhs, wrong_route=None, nonfinite_route=None,
                 auto_route='naive'):
        self.lhs = lhs
        self.rhs = rhs
        self.wrong_route = wrong_route
        self.nonfinite_route = nonfinite_route
        self.events = []
        self.routes = (
            FakeRoute(
                'naive', selected_by_auto=auto_route == 'naive'),
            FakeRoute(
                'blas_gemm', selected_by_auto=auto_route == 'blas_gemm',
                packing={'scratch_rhs': True}),
        )
        self.durations = {
            'auto': 20,
            'naive': 15,
            'blas_gemm': 5,
        }

    def _result(self, name):
        result = np.atleast_1d(np.matmul(self.lhs, self.rhs))
        if name == self.wrong_route:
            result = result.copy()
            result.flat[0] += 1
        if name == self.nonfinite_route:
            result = result.copy()
            result.flat[0] = np.nan
        return result

    def execute_auto(self):
        self.events.append(('correctness', 'auto'))
        return self._result('auto')

    def execute_route(self, name):
        self.events.append(('correctness', name))
        return self._result(name)


class FakeEngine:
    def __init__(self, wrong_route=None, nonfinite_route=None,
                 auto_route='naive'):
        self.wrong_route = wrong_route
        self.nonfinite_route = nonfinite_route
        self.auto_route = auto_route
        self.case = None

    def prepare(self, lhs, rhs, dtype):
        self.case = FakeCase(
            lhs, rhs, self.wrong_route, self.nonfinite_route,
            self.auto_route)
        return self.case


class StepClock:
    def __init__(self, step):
        self.value = 0
        self.step = step

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


def make_request(**overrides):
    data = {
        'schema_version': 1,
        'id': 'test-case',
        'lhs': {'shape': [2, 3], 'strides': [3, 1]},
        'rhs': {'shape': [3, 2], 'strides': [2, 1]},
        'dtype': 'float64',
        'mode': {
            'name': 'stable',
            'warmups': 1,
            'repetitions': 4,
            'panels': 6,
        },
        'numpy_baseline': False,
        'seed': 17,
    }
    data.update(overrides)
    return matmul_benchmark.schema.BenchmarkRequest.from_dict(data)


def resource_budget(peak_bytes, single_allocation_bytes=None):
    if single_allocation_bytes is None:
        single_allocation_bytes = peak_bytes
    return matmul_benchmark.arrays.ResourceBudget(
        available_bytes=max(peak_bytes, single_allocation_bytes),
        peak_bytes=peak_bytes,
        single_allocation_bytes=single_allocation_bytes,
    )


class MatmulBenchmarkSchemaTC(unittest.TestCase):
    def test_request_round_trip_preserves_element_strides(self):
        request = make_request(
            lhs={'shape': [2, 3], 'strides': [-5, 1]},
            rhs={'shape': [3, 2], 'strides': [0, 1]},
        )

        rebuilt = matmul_benchmark.schema.BenchmarkRequest.from_dict(
            request.to_dict())

        self.assertEqual(rebuilt, request)
        self.assertEqual(rebuilt.lhs.strides, (-5, 1))
        self.assertEqual(rebuilt.rhs.strides, (0, 1))

    def test_rejects_invalid_matmul_and_schema_version(self):
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'contraction'):
            make_request(rhs={
                'shape': [4, 2],
                'strides': [2, 1],
            })
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'schema version'):
            make_request(schema_version=2)
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'routes'):
            make_request(routes='naive')
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'unknown fields'):
            make_request(thread=1)
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'unknown fields'):
            make_request(mode={'name': 'preview', 'panel': 1})

    def test_bounds_operand_rank_and_custom_measurement_counts(self):
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'at most'):
            make_request(lhs={
                'shape': [1] * (
                    matmul_benchmark.schema.MAX_OPERAND_RANK - 1) + [2, 3],
                'strides': [0] * (
                    matmul_benchmark.schema.MAX_OPERAND_RANK - 1) + [3, 1],
            })
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'repetitions'):
            make_request(mode={
                'name': 'preview',
                'warmups': 0,
                'repetitions': 10 ** 18,
                'panels': 1,
            })
        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'too many calls'):
            make_request(mode={
                'name': 'preview',
                'warmups': 1000,
                'repetitions': 1000,
                'panels': 10,
            })

    def test_accepts_vector_and_broadcast_batch_roles(self):
        vector = make_request(
            lhs={'shape': [3], 'strides': [-1]},
            rhs={'shape': [3], 'strides': [0]},
        )
        batched = make_request(
            lhs={'shape': [2, 1, 3, 4],
                 'strides': [12, 0, 4, 1]},
            rhs={'shape': [1, 5, 4, 6],
                 'strides': [0, 24, 6, 1]},
        )

        self.assertEqual(vector.lhs.shape, (3,))
        self.assertEqual(batched.rhs.shape, (1, 5, 4, 6))


class MatmulBenchmarkArrayTC(unittest.TestCase):
    def test_builds_positive_negative_and_broadcast_strides(self):
        positive = matmul_benchmark.arrays.make_strided_array(
            {'shape': [2, 3], 'strides': [5, 1]}, 'float64')
        negative = matmul_benchmark.arrays.make_strided_array(
            {'shape': [2, 3], 'strides': [-5, 1]}, 'float64')
        broadcast = matmul_benchmark.arrays.make_strided_array(
            {'shape': [2, 3], 'strides': [0, 1]}, 'float64')

        self.assertEqual(positive.strides, (40, 8))
        self.assertEqual(negative.strides, (-40, 8))
        self.assertEqual(broadcast.strides, (0, 8))
        np.testing.assert_array_equal(broadcast[0], broadcast[1])

    def test_storage_limit_is_checked_before_allocation(self):
        with self.assertRaises(MemoryError):
            matmul_benchmark.arrays.make_strided_array(
                {'shape': [2, 2], 'strides': [1_000_000, 1]},
                'float64', max_storage_bytes=1024)

    def test_complex_generation_is_direct_and_deterministic(self):
        spec = {'shape': [3, 4], 'strides': [4, 1]}

        first = matmul_benchmark.arrays.make_strided_array(
            spec, 'complex64', seed=91)
        second = matmul_benchmark.arrays.make_strided_array(
            spec, 'complex64', seed=91)

        self.assertEqual(first.dtype, np.dtype('complex64'))
        np.testing.assert_array_equal(first, second)


class MatmulBenchmarkResourceBudgetTC(unittest.TestCase):
    GIBIBYTE = 1024 * 1024 * 1024

    def test_resolves_half_of_the_tightest_memory_reading(self):
        budget = matmul_benchmark.arrays.resolve_resource_budget(
            os_available_bytes=64 * self.GIBIBYTE,
            cgroup_remaining_bytes=60 * self.GIBIBYTE,
            commit_headroom_bytes=59 * self.GIBIBYTE,
        )

        self.assertEqual(budget.available_bytes, 59 * self.GIBIBYTE)
        self.assertEqual(budget.peak_bytes, 59 * self.GIBIBYTE // 2)
        self.assertEqual(
            budget.single_allocation_bytes, 4 * self.GIBIBYTE)
        with self.assertRaises(AttributeError):
            budget.peak_bytes = 1

    def test_falls_back_to_the_previous_memory_limit(self):
        budget = matmul_benchmark.arrays.resolve_resource_budget(
            os_available_bytes=None,
            cgroup_remaining_bytes=None,
            commit_headroom_bytes=None,
        )

        fallback = matmul_benchmark.arrays.DEFAULT_MAX_PEAK_BYTES
        self.assertEqual(budget.available_bytes, fallback)
        self.assertEqual(budget.peak_bytes, fallback)
        self.assertEqual(budget.single_allocation_bytes, fallback)

    def test_reads_only_finite_cgroup_remaining_memory(self):
        with tempfile.TemporaryDirectory() as dirname:
            directory = pathlib.Path(dirname)
            (directory / 'memory.max').write_text(
                str(10 * self.GIBIBYTE), encoding='ascii')
            (directory / 'memory.current').write_text(
                str(3 * self.GIBIBYTE), encoding='ascii')
            with unittest.mock.patch.object(
                    matmul_benchmark.arrays,
                    '_current_cgroup_directories',
                    return_value=[directory]):
                remaining = (
                    matmul_benchmark.arrays._read_cgroup_remaining_bytes())

            self.assertEqual(remaining, 7 * self.GIBIBYTE)

            (directory / 'memory.max').write_text(
                'max', encoding='ascii')
            with unittest.mock.patch.object(
                    matmul_benchmark.arrays,
                    '_current_cgroup_directories',
                    return_value=[directory]):
                remaining = (
                    matmul_benchmark.arrays._read_cgroup_remaining_bytes())

            self.assertIsNone(remaining)

    def test_uses_commit_headroom_only_in_strict_mode(self):
        meminfo = {
            'CommitLimit': 20 * self.GIBIBYTE,
            'Committed_AS': 7 * self.GIBIBYTE,
        }
        with unittest.mock.patch.object(
                matmul_benchmark.arrays,
                '_read_nonnegative_integer', return_value=0):
            self.assertIsNone(
                matmul_benchmark.arrays._read_linux_commit_headroom_bytes())

        with unittest.mock.patch.object(
                matmul_benchmark.arrays,
                '_read_nonnegative_integer', return_value=2), \
                unittest.mock.patch.object(
                    matmul_benchmark.arrays, '_read_linux_meminfo',
                    return_value=meminfo):
            headroom = (
                matmul_benchmark.arrays._read_linux_commit_headroom_bytes())

        self.assertEqual(headroom, 13 * self.GIBIBYTE)

    def test_default_array_limits_use_the_resolved_budget(self):
        budget = matmul_benchmark.arrays.ResourceBudget(
            available_bytes=16,
            peak_bytes=16,
            single_allocation_bytes=16,
        )
        with unittest.mock.patch(
                'solvcon.matmul_benchmark.arrays.resolve_resource_budget',
                return_value=budget):
            with self.assertRaisesRegex(MemoryError, 'operand storage'):
                matmul_benchmark.arrays.make_strided_array(
                    {'shape': [2, 2], 'strides': [2, 1]}, 'float64')

    def test_explicit_array_limits_skip_budget_resolution(self):
        with unittest.mock.patch(
                'solvcon.matmul_benchmark.arrays.resolve_resource_budget',
                side_effect=AssertionError('must not resolve')):
            array = matmul_benchmark.arrays.make_strided_array(
                {'shape': [2, 2], 'strides': [2, 1]}, 'float64',
                max_storage_bytes=32, max_logical_bytes=32)

        self.assertEqual(array.shape, (2, 2))

    def test_single_allocation_cap_rejects_bad_stride(self):
        budget = matmul_benchmark.arrays.resolve_resource_budget(
            os_available_bytes=64 * self.GIBIBYTE,
            cgroup_remaining_bytes=None,
            commit_headroom_bytes=None,
        )
        matrix = {'shape': [16384, 16384], 'strides': [16384, 1]}
        bad_stride = {'shape': [2, 3], 'strides': [1_000_000_000, 1]}

        matrix_bytes = matmul_benchmark.arrays.operand_storage_bytes(
            matrix, 'float64')
        bad_stride_bytes = matmul_benchmark.arrays.operand_storage_bytes(
            bad_stride, 'float64')

        self.assertEqual(matrix_bytes, 2 * self.GIBIBYTE)
        self.assertLessEqual(matrix_bytes, budget.single_allocation_bytes)
        self.assertGreater(bad_stride_bytes, budget.single_allocation_bytes)

    def test_calculates_winograd_scratch_without_allocation(self):
        scratch_bytes = matmul_benchmark.arrays.winograd_scratch_bytes(
            rows=8, inner_size=12, columns=16, itemsize=8)

        self.assertEqual(scratch_bytes, (4 * 6 + 6 * 8) * 8)


class MatmulBenchmarkBudgetTC(unittest.TestCase):
    def _assert_rejected_before_allocation(self, message, request):
        with unittest.mock.patch(
                'solvcon.matmul_benchmark.arrays.make_strided_array') \
                as make_array:
            with self.assertRaisesRegex(MemoryError, message):
                matmul_benchmark.collector.collect(
                    request, engine=FakeEngine())
        make_array.assert_not_called()

    def test_rejects_huge_physical_stride_span(self):
        request = make_request(lhs={
            'shape': [2, 3], 'strides': [1_000_000_000, 1],
        })

        self._assert_rejected_before_allocation('lhs storage', request)

    def test_rejects_small_broadcast_inputs_with_huge_output(self):
        request = make_request(
            lhs={'shape': [1_000_000, 1], 'strides': [0, 0]},
            rhs={'shape': [1, 1_000_000], 'strides': [0, 0]},
        )

        self._assert_rejected_before_allocation('matmul output', request)

    def test_rejects_huge_logical_inner_axis_with_zero_strides(self):
        request = make_request(
            lhs={'shape': [600_000_000], 'strides': [0]},
            rhs={'shape': [600_000_000], 'strides': [0]},
        )

        self._assert_rejected_before_allocation('lhs logical size', request)

    def test_naive_work_does_not_make_a_request_ineligible(self):
        request = make_request(
            lhs={'shape': [2000, 26000], 'strides': [0, 0]},
            rhs={'shape': [26000, 2000], 'strides': [0, 0]},
        )
        budget = resource_budget(8 * 1024 ** 3)

        work = matmul_benchmark.collector.validate_request_resources(
            request, resource_budget=budget)

        self.assertGreater(work, 100_000_000_000)

    def test_stable_naive_schedule_has_no_runtime_work_cap(self):
        request = make_request(
            lhs={'shape': [2200, 2200], 'strides': [2200, 1]},
            rhs={'shape': [2200, 2200], 'strides': [2200, 1]},
        )
        budget = resource_budget(2 * 1024 ** 3)

        work = matmul_benchmark.collector.validate_request_resources(
            request, resource_budget=budget)

        self.assertGreater(work, 10_000_000_000)

    def test_accelerated_2048_preview_is_not_charged_as_naive(self):
        request = make_request(
            lhs={'shape': [2048, 2048], 'strides': [2048, 1]},
            rhs={'shape': [2048, 2048], 'strides': [2048, 1]},
            dtype='float32',
            mode='preview',
            routes=('blas_gemm',),
            numpy_baseline=False,
        )
        engine = FakeEngine(auto_route='blas_gemm')
        small = np.ones((2, 2), dtype='float32')
        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=resource_budget(1024 ** 3)):
            with unittest.mock.patch.object(
                    matmul_benchmark.arrays, 'make_strided_array',
                    side_effect=(small, small)):
                prepared = matmul_benchmark.collector._prepare_case(
                    request, engine)

        self.assertEqual(prepared.names, ('auto', 'blas_gemm'))

    def test_accelerated_schedule_is_not_charged_as_naive(self):
        operands = {
            'lhs': {'shape': [2048, 2048], 'strides': [2048, 1]},
            'rhs': {'shape': [2048, 2048], 'strides': [2048, 1]},
            'dtype': 'float32',
            'routes': ('blas_gemm',),
            'numpy_baseline': False,
        }
        stable = make_request(mode='stable', **operands)
        budget = resource_budget(2 * 1024 ** 3)

        matmul_benchmark.collector.validate_request_resources(
            stable, resource_budget=budget)

    def test_large_accelerated_request_has_no_runtime_work_cap(self):
        request = make_request(
            lhs={'shape': [16_384, 16_384],
                 'strides': [16_384, 1]},
            rhs={'shape': [16_384, 16_384],
                 'strides': [16_384, 1]},
            dtype='float32',
            mode='preview',
            routes=('blas_gemm', 'winograd'),
            numpy_baseline=True,
        )
        budget = resource_budget(
            30 * 1024 ** 3,
            single_allocation_bytes=4 * 1024 ** 3)

        work = matmul_benchmark.collector.validate_request_resources(
            request, resource_budget=budget)

        self.assertGreater(work, 4_000_000_000_000)

    def test_zero_contraction_schedule_is_not_runtime_capped(self):
        request = make_request(
            lhs={'shape': [2048, 0], 'strides': [0, 1]},
            rhs={'shape': [0, 2048], 'strides': [2048, 1]},
            mode={
                'name': 'stable',
                'warmups': 0,
                'repetitions': 10000,
                'panels': 1,
            },
        )

        work = matmul_benchmark.collector.validate_request_resources(
            request, resource_budget=resource_budget(512 * 1024 ** 2))

        self.assertEqual(work, 2048 ** 2)

    def test_rejects_aggregate_peak_memory_before_allocation(self):
        request = make_request(
            lhs={'shape': [8192, 1], 'strides': [1, 1]},
            rhs={'shape': [1, 8192], 'strides': [8192, 1]},
        )

        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=resource_budget(1024 ** 3)):
            self._assert_rejected_before_allocation(
                'benchmark peak estimate', request)

    def test_peak_memory_includes_zero_stride_materialization(self):
        request = make_request(
            lhs={'shape': [4, 3], 'strides': [0, 0]},
            rhs={'shape': [3, 2], 'strides': [0, 0]},
            dtype='float32',
        )

        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=resource_budget(150)):
            self._assert_rejected_before_allocation(
                'benchmark peak estimate needs 336', request)

    def test_correctness_temporaries_cannot_bypass_peak_budget(self):
        request = make_request(
            lhs={'shape': [6000, 1], 'strides': [1, 1]},
            rhs={'shape': [1, 6000], 'strides': [6000, 1]},
            dtype='float32', routes=('naive',),
            numpy_baseline=False)

        with unittest.mock.patch.object(
                matmul_benchmark.arrays, 'resolve_resource_budget',
                return_value=resource_budget(512 * 1024 ** 2)):
            self._assert_rejected_before_allocation(
                'benchmark peak estimate needs 1152096000', request)


class MatmulBenchmarkScheduleTC(unittest.TestCase):
    def test_balances_positions_across_a_complete_schedule(self):
        orders = matmul_benchmark.schedule.balanced_orders(('a', 'b', 'c'), 6)

        self.assertEqual(orders[0], ('a', 'b', 'c'))
        self.assertEqual(orders[1], ('b', 'c', 'a'))
        self.assertEqual(orders[3], ('c', 'b', 'a'))
        for route in ('a', 'b', 'c'):
            positions = [order.index(route) for order in orders]
            self.assertEqual(positions.count(0), 2)
            self.assertEqual(positions.count(1), 2)
            self.assertEqual(positions.count(2), 2)


class MatmulBenchmarkCollectorTC(unittest.TestCase):
    def test_checks_every_route_before_interleaved_timing(self):
        engine = FakeEngine()
        progress = []

        artifact = matmul_benchmark.collector.collect(
            make_request(), engine=engine, clock=StepClock(100),
            progress=progress.append)

        observation = artifact['observations'][0]
        candidates = observation['routes']
        self.assertEqual(set(candidates), {
            'auto', 'naive', 'blas_gemm'})
        first_timing = next(
            index for index, event in enumerate(progress)
            if event.get('phase') == 'timing')
        correctness = [
            event for event in progress[:first_timing]
            if event.get('phase') == 'correctness']
        self.assertEqual(len(correctness), 6)
        self.assertEqual(len(artifact['panels']), 6)
        panel_progress = [
            event for event in progress if 'state' not in event]
        activity = [event for event in progress if 'state' in event]
        self.assertEqual(len(panel_progress), 6)
        self.assertTrue(all(event['type'] == 'progress'
                            for event in progress))
        self.assertEqual(activity[0]['phase'], 'provenance')
        self.assertEqual(activity[0]['state'], 'started')
        self.assertEqual(activity[2]['phase'], 'preparation')
        self.assertEqual(activity[4]['phase'], 'reference')
        self.assertEqual(activity[4]['route'], 'numpy')
        self.assertEqual(
            {event['phase'] for event in activity}, {
                'provenance', 'preparation', 'reference', 'correctness',
                'warmup', 'timing', 'finalization'})
        required = {
            'type', 'phase', 'state', 'route', 'resolved_route',
            'cell_id', 'panel', 'panels', 'chunk',
            'completed_calls', 'total_calls', 'chunk_calls',
            'event_at_ns', 'message',
        }
        self.assertTrue(all(required <= set(event) for event in activity))
        self.assertTrue(all(
            start['state'] == 'started'
            and finish['state'] == 'completed'
            and start['phase'] == finish['phase']
            and start['route'] == finish['route']
            for start, finish in zip(activity[::2], activity[1::2])))
        auto_started = next(
            event for event in activity
            if event['phase'] == 'correctness'
            and event['route'] == 'auto'
            and event['state'] == 'started')
        self.assertEqual(auto_started['resolved_route'], 'naive')
        self.assertEqual(
            candidates['blas_gemm']['timing']['median_ns'], 25)
        self.assertEqual(observation['winner'], 'naive')
        self.assertEqual(
            observation['routes']['blas_gemm']['packing'], {
                'eager_lhs': False,
                'eager_rhs': False,
                'scratch_lhs': False,
                'scratch_rhs': True,
            })

    def test_cancel_after_started_event_skips_the_timing_block(self):
        request = make_request(mode={
            'name': 'preview',
            'warmups': 0,
            'repetitions': 4,
            'panels': 1,
        })
        engine = FakeEngine()
        events = []
        cancel = [False]

        def progress(event):
            events.append(event)
            if (event.get('phase') == 'timing'
                    and event.get('state') == 'started'):
                cancel[0] = True

        with self.assertRaises(
                matmul_benchmark.collector.CollectionCancelled):
            matmul_benchmark.collector.collect(
                request, engine=engine, progress=progress,
                cancelled=lambda: cancel[0])

        self.assertEqual(events[-1]['state'], 'started')
        self.assertEqual(events[-1]['total_calls'], 4)

    def test_failed_activity_is_reported_before_the_error_escapes(self):
        original = FakeCase.execute_route
        events = []

        def execute_route(case, name):
            if name == 'naive':
                raise RuntimeError('forced correctness failure')
            return original(case, name)

        with unittest.mock.patch.object(
                FakeCase, 'execute_route', execute_route):
            with self.assertRaisesRegex(
                    RuntimeError, 'forced correctness failure'):
                matmul_benchmark.collector.collect(
                    make_request(), engine=FakeEngine(),
                    progress=events.append)

        naive = [
            event for event in events
            if event.get('phase') == 'correctness'
            and event.get('route') == 'naive'
        ]
        self.assertEqual(
            [event['state'] for event in naive], ['started', 'failed'])
        self.assertEqual(naive[-1]['error_type'], 'RuntimeError')
        self.assertGreaterEqual(naive[-1]['elapsed_ns'], 0)

    def test_excludes_an_incorrect_route_from_timing_and_winner(self):
        engine = FakeEngine(wrong_route='blas_gemm')

        artifact = matmul_benchmark.collector.collect(
            make_request(), engine=engine)

        observation = artifact['observations'][0]
        self.assertIsNone(observation['routes']['blas_gemm']['timing'])
        self.assertEqual(observation['winner'], 'naive')
        executed = [name for _phase, name in engine.case.events]
        self.assertEqual(executed.count('blas_gemm'), 1)

    def test_timing_uses_the_same_boundary_for_numpy_and_routes(self):
        request = make_request(
            numpy_baseline=True,
            mode={
                'name': 'preview',
                'warmups': 0,
                'repetitions': 5,
                'panels': 2,
            },
        )

        artifact = matmul_benchmark.collector.collect(
            request, engine=FakeEngine(), clock=StepClock(100))

        samples = [
            sample for panel in artifact['panels']
            for sample in panel['samples']
            if sample['route'] == 'numpy'
        ]
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(sample['elapsed_ns'] == 100
                            for sample in samples))
        self.assertTrue(all(sample['latency_ns'] == 20
                            for sample in samples))
        observation = artifact['observations'][0]
        self.assertEqual(
            observation['routes']['numpy']['timing']['median_ns'], 20)
        self.assertEqual(set(observation['routes']), {
            'auto', 'naive', 'blas_gemm', 'numpy'})
        self.assertEqual(
            observation['routes']['blas_gemm']['numpy_ratio'], 1)
        self.assertEqual(
            observation['routes']['numpy']['numpy_ratio'], 1)

    def test_nonfinite_failure_remains_serializable(self):
        artifact = matmul_benchmark.collector.collect(
            make_request(), engine=FakeEngine(
                nonfinite_route='blas_gemm'))
        route = artifact['observations'][0]['routes']['blas_gemm']

        self.assertFalse(route['correctness']['correct'])
        self.assertTrue(route['correctness']['nonfinite_result'])
        self.assertIsNone(
            route['correctness']['max_absolute_error'])
        self.assertIsNone(route['timing'])
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'nonfinite.json'
            matmul_benchmark.artifact.write_artifact(artifact, path)
            self.assertEqual(
                matmul_benchmark.artifact.load_artifact(path)['artifact_id'],
                artifact['artifact_id'])

    def test_correctness_tolerance_stays_bounded_for_large_inner_axes(self):
        comparison = matmul_benchmark.collector._comparison(
            np.array([1], dtype='float32'),
            np.array([0], dtype='float32'),
            'float32',
            131072,
        )

        self.assertFalse(comparison['correct'])
        self.assertLessEqual(comparison['rtol'], 1e-3)
        self.assertLessEqual(comparison['atol'], 1e-3)

    def test_overflowing_relative_error_remains_serializable(self):
        comparison = matmul_benchmark.collector._comparison(
            np.array([0.0], dtype='float32'),
            np.array([10.0], dtype='float32'),
            'float32', 1)

        self.assertFalse(comparison['correct'])
        self.assertIsNone(comparison['max_relative_error'])
        json.dumps(comparison, allow_nan=False)

    def test_thread_request_requires_a_preconfigured_environment(self):
        environment = {
            name: '2' for name in (
                'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'BLIS_NUM_THREADS')
        }
        request = make_request(threads=2)
        with unittest.mock.patch.dict(os.environ, environment, clear=False):
            artifact = matmul_benchmark.collector.collect(
                request, engine=FakeEngine())
        self.assertEqual(
            artifact['metadata']['threading']['environment'], environment)

        invalid_environment = dict(environment)
        invalid_environment['OMP_NUM_THREADS'] = '4'
        with unittest.mock.patch.dict(
                os.environ, invalid_environment, clear=False):
            with self.assertRaisesRegex(
                    matmul_benchmark.schema.SchemaError, 'threads=2'):
                matmul_benchmark.collector.collect(
                    request, engine=FakeEngine())

    def test_empty_optional_thread_environment_is_not_serialized(self):
        with unittest.mock.patch.dict(
                os.environ, {'OMP_NUM_THREADS': ''}, clear=False):
            artifact = matmul_benchmark.collector.collect(
                make_request(), engine=FakeEngine())

        self.assertNotIn(
            'OMP_NUM_THREADS',
            artifact['metadata']['threading']['environment'])
        matmul_benchmark.schema.validate_artifact(artifact)


class MatmulBenchmarkArtifactTC(unittest.TestCase):
    def test_atomic_round_trip_and_merge_only_reads_artifacts(self):
        artifact = matmul_benchmark.collector.collect(
            make_request(), engine=FakeEngine())
        with tempfile.TemporaryDirectory() as directory:
            first = pathlib.Path(directory) / 'first.json'
            second = pathlib.Path(directory) / 'second.json'
            collection_path = pathlib.Path(directory) / 'collection.json'
            matmul_benchmark.artifact.write_artifact(artifact, first)
            matmul_benchmark.artifact.write_artifact(artifact, second)

            loaded = matmul_benchmark.artifact.load_artifact(first)
            with unittest.mock.patch(
                    'solvcon.matmul_benchmark.collector.collect',
                    side_effect=AssertionError('collector started')):
                merged = matmul_benchmark.artifact.merge_artifacts(
                    (first, second))

            self.assertEqual(loaded['artifact_id'], artifact['artifact_id'])
            self.assertEqual(merged['artifact_count'], 2)
            self.assertEqual(len(merged['observations']), 2)
            self.assertEqual(
                len(merged['panels']), 2 * len(artifact['panels']))
            self.assertEqual(
                merged['observations'][0]['source_id'], 'source-0')
            self.assertEqual(
                merged['observations'][1]['source_id'], 'source-1')
            self.assertEqual(
                merged['sources'][0]['request'], artifact['request'])
            self.assertEqual(
                merged['sources'][0]['metadata'], artifact['metadata'])
            self.assertEqual(
                merged['sources'][0]['panel_count'],
                len(artifact['panels']))
            self.assertEqual(
                merged['sources'][0]['panels_sha256'],
                matmul_benchmark.schema.panels_sha256(artifact['panels']))
            self.assertEqual(
                [item['panel'] for item in merged['panels']
                 if item['source_id'] == 'source-0'],
                artifact['panels'])
            matmul_benchmark.artifact.write_artifact(merged, collection_path)
            loaded_collection = matmul_benchmark.artifact.load_artifact(
                collection_path)
            self.assertEqual(
                loaded_collection['collection_id'], merged['collection_id'])
            self.assertEqual(
                loaded_collection['panels'], merged['panels'])
            self.assertFalse(any(
                path.suffix == '.tmp'
                for path in pathlib.Path(directory).iterdir()))

    def test_rejects_malformed_nested_artifact_data(self):
        artifact = matmul_benchmark.collector.collect(
            make_request(), engine=FakeEngine())
        malformed = copy.deepcopy(artifact)
        malformed['observations'] = [None]

        with self.assertRaisesRegex(
                matmul_benchmark.schema.SchemaError, 'must be an object'):
            matmul_benchmark.schema.validate_artifact(malformed)

    def test_rejects_inconsistent_route_references(self):
        artifact = matmul_benchmark.collector.collect(
            make_request(), engine=FakeEngine())
        mutations = (
            ('routes', lambda item: item['observations'][0]['routes']
             ['naive'].__setitem__('name', 'typo')),
            ('winner', lambda item: item['observations'][0].__setitem__(
                'winner', 'typo')),
            ('runner_up', lambda item: item['observations'][0].__setitem__(
                'runner_up', 'typo')),
            ('auto_route', lambda item: item['observations'][0].__setitem__(
                'auto_route', 'typo')),
        )

        for message, mutate in mutations:
            with self.subTest(message=message):
                malformed = copy.deepcopy(artifact)
                mutate(malformed)
                with self.assertRaisesRegex(
                        matmul_benchmark.schema.SchemaError, message):
                    matmul_benchmark.schema.validate_artifact(malformed)

    def test_records_build_and_native_dependency_identity(self):
        artifact = matmul_benchmark.collector.collect(
            make_request(), engine=FakeEngine())
        build = artifact['metadata']['build']

        self.assertEqual(len(build['extension_sha256']), 64)
        self.assertIsInstance(build['git_dirty'], bool)
        self.assertIsInstance(build['dirty_source_complete'], bool)
        if build['git_dirty']:
            self.assertEqual(len(build['dirty_diff_sha256']), 64)
        self.assertEqual(set(build['native_loader']), {
            'command', 'dependencies', 'returncode'})


class MatmulBenchmarkFeatureTC(unittest.TestCase):
    def setUp(self):
        artifact = matmul_benchmark.collector.collect(
            make_request(), engine=FakeEngine())
        self.observation = artifact['observations'][0]

    def test_registers_projects_and_slices_dynamic_features(self):
        registry = matmul_benchmark.features.FeatureRegistry()
        registry.register_expression('linear_axis', '2 * M + K - N')

        value = registry.evaluate('linear_axis', self.observation)
        projected = matmul_benchmark.features.project_observations(
            [self.observation], registry,
            ('linear_axis', 'work'),
            constraints={'dtype': 'float64', 'K': (3, 3)},
        )

        self.assertEqual(value, 5)
        self.assertEqual(projected[0]['coordinates'], [5, 12])
        self.assertEqual(
            projected[0]['winner'], self.observation['winner'])

    def test_rejects_code_execution_and_cyclic_definitions(self):
        registry = matmul_benchmark.features.FeatureRegistry()
        with self.assertRaises(matmul_benchmark.features.FeatureError):
            registry.register_expression(
                'bad', '__import__("os").system("true")')
        registry.register_expression('first', 'second + 1')
        registry.register_expression('second', 'first + 1')

        with self.assertRaisesRegex(
                matmul_benchmark.features.FeatureError, 'cyclic'):
            registry.evaluate('first', self.observation)

    def test_bounds_numeric_resources_and_nonfinite_expressions(self):
        registry = matmul_benchmark.features.FeatureRegistry()
        expression = '2'
        for _ in range(9):
            expression = f'({expression}) ** 16'
        registry.register_expression('large_integer', expression)

        with self.assertRaisesRegex(
                matmul_benchmark.features.FeatureError, 'size limit'):
            registry.evaluate('large_integer', self.observation)
        with self.assertRaisesRegex(
                matmul_benchmark.features.FeatureError, 'not finite'):
            registry.register_expression('infinity', '1e309')

    def test_undefined_winner_margin_is_a_clean_nan_feature(self):
        registry = matmul_benchmark.features.FeatureRegistry()
        observation = copy.deepcopy(self.observation)
        observation['winner_margin'] = None

        self.assertTrue(math.isnan(
            registry.evaluate('winner_margin', observation)))
        registry.register_expression('shifted_margin', 'winner_margin + 1')
        with self.assertRaisesRegex(
                matmul_benchmark.features.FeatureError, 'not finite'):
            registry.evaluate('shifted_margin', observation)

    def test_scratch_packing_counts_source_pointer_transitions(self):
        observation = copy.deepcopy(self.observation)
        observation['lhs'] = {
            'shape': [1, 2, 3],
            'strides': [6, 3, 1],
        }
        observation['rhs'] = {
            'shape': [4, 3, 2],
            'strides': [6, 2, 1],
        }
        observation['contraction'].update({
            'batch_shape': [4],
            'batch_count': 4,
            'm': 2,
            'k': 3,
            'n': 2,
            'output_shape': [4, 2, 2],
        })
        winner = observation['routes'][observation['winner']]
        winner['packing'] = {
            'eager_lhs': False,
            'eager_rhs': False,
            'scratch_lhs': True,
            'scratch_rhs': False,
        }

        packing_bytes = matmul_benchmark.features.FeatureRegistry().evaluate(
            'packing_bytes', observation)

        self.assertEqual(packing_bytes, 2 * 3 * 8)

    def test_exposes_core_and_aligned_batch_strides_as_axes(self):
        observation = copy.deepcopy(self.observation)
        observation['lhs'] = {
            'shape': [2, 1, 4, 5],
            'strides': [20, 20, 5, 1],
        }
        observation['rhs'] = {
            'shape': [1, 3, 5, 6],
            'strides': [90, 30, 6, 1],
        }
        observation['contraction'].update({
            'batch_shape': [2, 3],
            'batch_count': 6,
            'm': 4,
            'k': 5,
            'n': 6,
            'output_shape': [2, 3, 4, 6],
        })
        registry = matmul_benchmark.features.FeatureRegistry()

        self.assertEqual(registry.evaluate(
            'lhs_row_stride', observation), 5)
        self.assertEqual(registry.evaluate(
            'rhs_inner_stride', observation), 6)
        self.assertEqual(registry.evaluate(
            'batch_extent_1', observation), 3)
        self.assertEqual(registry.evaluate(
            'lhs_batch_stride_0', observation), 20)
        self.assertEqual(registry.evaluate(
            'lhs_batch_stride_1', observation), 0)
        self.assertEqual(registry.evaluate(
            'rhs_batch_stride_0', observation), 0)
        self.assertEqual(registry.evaluate(
            'rhs_batch_stride_1', observation), 30)


class MatmulBenchmarkWorkerTC(unittest.TestCase):
    def test_parent_death_signal_is_a_portable_no_op(self):
        with unittest.mock.patch.object(sys, 'platform', 'darwin'):
            with unittest.mock.patch.object(
                    matmul_benchmark.worker.ctypes, 'CDLL') as load:
                installed = \
                    matmul_benchmark.worker._install_parent_death_signal()

        self.assertTrue(installed)
        load.assert_not_called()

    def test_parent_death_signal_detects_the_startup_race(self):
        library = unittest.mock.Mock()
        library.prctl.return_value = 0
        with unittest.mock.patch.object(sys, 'platform', 'linux'):
            with unittest.mock.patch.object(
                    matmul_benchmark.worker.os, 'getppid',
                    side_effect=(41, 1)):
                with unittest.mock.patch.object(
                        matmul_benchmark.worker.ctypes, 'CDLL',
                        return_value=library):
                    installed = matmul_benchmark.worker \
                        ._install_parent_death_signal()

        self.assertFalse(installed)
        library.prctl.assert_called_once_with(
            matmul_benchmark.worker.PR_SET_PDEATHSIG,
            matmul_benchmark.worker.signal.SIGTERM, 0, 0, 0)

    def test_json_worker_reports_a_missing_controller_once(self):
        stdout = io.StringIO()
        with unittest.mock.patch.object(
                matmul_benchmark.worker,
                '_install_parent_death_signal', return_value=False):
            with unittest.mock.patch.object(sys, 'stdout', stdout):
                return_code = matmul_benchmark.worker.main(
                    ('--json-lines',))

        events = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(return_code, 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['type'], 'error')
        self.assertIn('controller exited', events[0]['message'])

    def test_schema_error_is_a_single_json_error_event(self):
        stdin = io.StringIO('{"schema_version": 99}')
        stdout = io.StringIO()
        with unittest.mock.patch.object(sys, 'stdin', stdin), \
                unittest.mock.patch.object(sys, 'stdout', stdout):
            return_code = matmul_benchmark.worker.main(())

        events = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(return_code, 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['type'], 'error')

    def test_tiny_real_worker_writes_a_valid_artifact(self):
        lhs = sc.SimpleArrayFloat64(
            array=np.ones((2, 2), dtype='float64'))
        if not hasattr(lhs, 'matmul_routes'):
            self.skipTest('native matmul benchmark binding is not built')
        with tempfile.TemporaryDirectory() as directory:
            request_path = pathlib.Path(directory) / 'request.json'
            output_path = pathlib.Path(directory) / 'artifact.json'
            request = make_request(
                lhs={'shape': [2, 2], 'strides': [2, 1]},
                rhs={'shape': [2, 2], 'strides': [2, 1]},
                mode={
                    'name': 'preview',
                    'warmups': 0,
                    'repetitions': 1,
                    'panels': 1,
                },
                output_path=str(output_path),
            )
            request_path.write_text(
                json.dumps(request.to_dict()), encoding='utf8')
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(pathlib.Path.cwd())

            process = subprocess.run(
                [sys.executable, '-m',
                 'solvcon_matmul_benchmark',
                 '--request', str(request_path), '--json-lines'],
                cwd=pathlib.Path.cwd(), env=environment,
                capture_output=True, text=True, check=False)

            self.assertEqual(process.returncode, 0, process.stdout)
            events = [json.loads(line)
                      for line in process.stdout.splitlines()]
            self.assertEqual(events[-1]['type'], 'result')
            self.assertEqual(
                [event['state'] for event in events
                 if event.get('phase') == 'artifact_write'],
                ['started', 'completed'])
            artifact = matmul_benchmark.artifact.load_artifact(output_path)
            self.assertEqual(len(artifact['panels']), 1)
            self.assertIsNotNone(artifact['observations'][0]['winner'])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
