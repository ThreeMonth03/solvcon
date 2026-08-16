# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import math
import unittest

from solvcon.matmul_benchmark import duration
from solvcon.matmul_benchmark import schema


def measurement(cell_id, route, scope, latency_ns, repetitions=1):
    return duration.CalibrationMeasurement(
        cell_id=cell_id,
        route=route,
        scope=scope,
        elapsed_ns=latency_ns * repetitions,
        repetitions=repetitions,
    )


def simple_measurements(repetitions=3):
    values = (
        ('auto', duration.NATIVE_SCOPE, 100_000),
        ('generic', duration.NATIVE_SCOPE, 80_000),
        ('auto', duration.PYTHON_SCOPE, 120_000),
        ('generic', duration.PYTHON_SCOPE, 100_000),
        ('numpy', duration.PYTHON_SCOPE, 70_000),
    )
    return tuple(
        measurement('cell-0', route, scope, latency, repetitions)
        for route, scope, latency in values
    )


class TargetDurationSpecTC(unittest.TestCase):
    def test_quality_presets_supply_minimum_panels_and_warmups(self):
        preview = duration.TargetDurationSpec(seconds=1, mode='preview')
        stable = duration.TargetDurationSpec(seconds=1, mode='stable')

        self.assertEqual(preview.minimum_panels, 2)
        self.assertEqual(preview.warmups, 2)
        self.assertEqual(stable.minimum_panels, 8)
        self.assertEqual(stable.warmups, 4)

    def test_rejects_invalid_duration_constraints(self):
        cases = (
            ({'seconds': 0}, 'seconds'),
            ({'seconds': True}, 'seconds'),
            ({'seconds': 1, 'mode': 'long'}, 'mode'),
            ({'seconds': 1, 'safety_fraction': 1.1}, 'safety'),
            ({'seconds': 1, 'uncertainty_fraction': 1}, 'uncertainty'),
            ({'seconds': 1,
              'minimum_calibration_repetitions': 3,
              'maximum_calibration_repetitions': 2}, 'minimum'),
            ({'seconds': 1,
              'maximum_calibration_repetitions':
                  schema.MAX_MODE_REPETITIONS + 1}, 'schema'),
        )
        for options, message in cases:
            with self.subTest(options=options):
                with self.assertRaisesRegex(
                        duration.DurationModelError, message):
                    duration.TargetDurationSpec(**options)


class BalancedPanelEstimateTC(unittest.TestCase):
    def test_normalizes_blocks_and_resists_one_large_outlier(self):
        values = []
        for latency in (100_000, 102_000, 10_000_000):
            values.extend((
                measurement(
                    'cell', 'auto', duration.NATIVE_SCOPE,
                    latency, repetitions=4),
                measurement(
                    'cell', 'auto', duration.PYTHON_SCOPE,
                    latency * 2, repetitions=4),
            ))
        estimate = duration.estimate_balanced_panel(
            values,
            (duration.ControllerOverhead(300_000, panels=3),),
            uncertainty_fraction=0.05,
        )

        self.assertEqual(estimate.stream_count, 2)
        self.assertAlmostEqual(
            estimate.per_repetition.central_seconds, 306e-6)
        self.assertLess(
            estimate.per_repetition.upper_seconds, 400e-6)
        self.assertAlmostEqual(
            estimate.controller_per_panel.central_seconds, 100e-6)
        self.assertAlmostEqual(
            estimate.calibration_seconds, 0.122724)

    def test_requires_native_python_coverage_for_each_cell(self):
        missing_route = (
            measurement('cell', 'auto', duration.NATIVE_SCOPE, 10),
            measurement('cell', 'other', duration.PYTHON_SCOPE, 10),
        )
        missing_scope = (
            measurement('cell', 'numpy', duration.PYTHON_SCOPE, 10),
        )

        with self.assertRaisesRegex(
                duration.DurationModelError, 'lack Python'):
            duration.estimate_balanced_panel(missing_route)
        with self.assertRaisesRegex(
                duration.DurationModelError, 'both timing scopes'):
            duration.estimate_balanced_panel(missing_scope)

    def test_rejects_invalid_measurement_values(self):
        with self.assertRaisesRegex(
                duration.DurationModelError, 'elapsed_ns'):
            duration.CalibrationMeasurement(
                'cell', 'auto', duration.NATIVE_SCOPE, 0, 1)
        with self.assertRaisesRegex(
                duration.DurationModelError, 'scope'):
            duration.CalibrationMeasurement(
                'cell', 'auto', 'unknown', 1, 1)
        with self.assertRaisesRegex(
                duration.DurationModelError, 'ControllerOverhead'):
            duration.estimate_balanced_panel(
                simple_measurements(), controller_overheads=(object(),))

    def test_calibration_block_repetitions_obey_explicit_caps(self):
        estimate = duration.estimate_balanced_panel(
            simple_measurements(repetitions=1))
        spec = duration.TargetDurationSpec(
            seconds=10,
            calibration_block_seconds=10,
            maximum_calibration_repetitions=7,
        )

        self.assertEqual(
            duration.choose_calibration_repetitions(spec, estimate), 7)


class TargetDurationScheduleTC(unittest.TestCase):
    def test_builds_complete_preview_panels_inside_safety_budget(self):
        spec = duration.TargetDurationSpec(
            seconds=10,
            mode='preview',
            checkpoint_seconds=1,
        )
        schedule = duration.plan_target_duration(
            spec,
            simple_measurements(),
            (duration.ControllerOverhead(100_000),),
            preflight_elapsed_ns=500_000,
        )

        self.assertTrue(schedule.feasible)
        self.assertGreaterEqual(schedule.panels, spec.minimum_panels)
        self.assertEqual(
            schedule.panels,
            schedule.full_shard_count
            * schedule.panels_per_full_shard
            + schedule.final_shard_panels,
        )
        self.assertLessEqual(
            schedule.predicted.upper_seconds,
            schedule.safety_budget_seconds,
        )
        self.assertLessEqual(
            schedule.maximum_timed_block.upper_seconds,
            spec.checkpoint_seconds,
        )
        self.assertLessEqual(
            schedule.maximum_calls_per_route_per_shard,
            schema.MAX_MODE_CALLS_PER_ROUTE,
        )
        self.assertLessEqual(
            schedule.panels_per_full_shard,
            schema.MAX_MODE_PANELS,
        )

    def test_short_target_returns_quality_infeasibility(self):
        spec = duration.TargetDurationSpec(
            seconds=0.0025,
            safety_fraction=1,
            calibration_block_seconds=1e-6,
        )
        schedule = duration.plan_target_duration(
            spec, simple_measurements(repetitions=1))

        self.assertFalse(schedule.feasible)
        self.assertEqual(schedule.limiter, 'target_duration')
        self.assertIn('complete preview panels', schedule.reason)
        self.assertIsNone(schedule.predicted)

    def test_preflight_can_consume_the_safety_budget(self):
        spec = duration.TargetDurationSpec(seconds=1)
        schedule = duration.plan_target_duration(
            spec, simple_measurements(),
            preflight_elapsed_ns=1_000_000_000)

        self.assertFalse(schedule.feasible)
        self.assertEqual(schedule.limiter, 'calibration_budget')
        self.assertIn('preflight', schedule.reason)

    def test_checkpoint_reports_an_unresponsive_minimum_block(self):
        values = (
            measurement(
                'slow', 'auto', duration.NATIVE_SCOPE,
                1_000_000_000),
            measurement(
                'slow', 'auto', duration.PYTHON_SCOPE,
                1_000_000_000),
        )
        spec = duration.TargetDurationSpec(
            seconds=100,
            calibration_block_seconds=0.01,
            checkpoint_seconds=1,
        )
        schedule = duration.plan_target_duration(spec, values)

        self.assertFalse(schedule.feasible)
        self.assertEqual(schedule.limiter, 'timed_block_checkpoint')
        self.assertEqual(
            schedule.shard_limiter, 'timed_block_checkpoint')

    def test_call_and_work_limits_only_bound_each_shard(self):
        spec = duration.TargetDurationSpec(
            seconds=5,
            safety_fraction=0.95,
            calibration_block_seconds=0.001,
            checkpoint_seconds=10,
        )
        guard = duration.ShardGuard(
            maximum_calls=180,
            maximum_work=22_000,
            fixed_calls=10,
            work_per_balanced_repetition=1_000,
        )
        schedule = duration.plan_target_duration(
            spec, simple_measurements(repetitions=1),
            shard_guard=guard)

        self.assertTrue(schedule.feasible)
        self.assertEqual(schedule.shard_limiter, 'work_guard')
        self.assertGreater(schedule.shard_count, 1)
        self.assertLessEqual(schedule.maximum_calls_per_shard, 180)
        self.assertLessEqual(schedule.maximum_work_per_shard, 22_000)
        self.assertGreater(schedule.predicted.central_seconds, 4)
        self.assertLessEqual(
            schedule.predicted.upper_seconds,
            schedule.safety_budget_seconds)

    def test_cool_like_calibration_scales_to_one_hour(self):
        latency_ns = round(
            4.967 * 1_000_000_000 / 14_464)
        values = []
        native_routes = ('auto', 'generic', 'blas', 'strassen')
        python_routes = native_routes + ('numpy',)
        for cell_index in range(128):
            cell_id = f'cell-{cell_index:03d}'
            values.extend(
                measurement(
                    cell_id, route, duration.NATIVE_SCOPE, latency_ns)
                for route in native_routes)
            values.extend(
                measurement(
                    cell_id, route, duration.PYTHON_SCOPE, latency_ns)
                for route in python_routes)
        spec = duration.TargetDurationSpec(
            seconds=3600,
            mode='stable',
            safety_fraction=0.95,
            calibration_block_seconds=0.02,
            checkpoint_seconds=60,
        )
        schedule = duration.plan_target_duration(
            spec, values,
            (duration.ControllerOverhead(5_000_000),))

        self.assertTrue(schedule.feasible)
        self.assertEqual(schedule.repetitions, 59)
        self.assertGreaterEqual(schedule.panels, 120)
        self.assertLessEqual(schedule.panels, 140)
        self.assertGreater(schedule.shard_count, 50)
        self.assertLessEqual(
            schedule.predicted.upper_seconds, 3600 * 0.95)
        self.assertGreater(schedule.predicted.central_seconds, 3000)
        self.assertLessEqual(
            schedule.maximum_timed_block.upper_seconds, 0.023)
        self.assertLessEqual(
            schedule.maximum_calls_per_route_per_shard,
            schema.MAX_MODE_CALLS_PER_ROUTE)
        self.assertTrue(math.isfinite(
            schedule.predicted.central_seconds))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
