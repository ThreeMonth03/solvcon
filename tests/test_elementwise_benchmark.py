# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import types
import unittest
import unittest.mock

import numpy as np

from profiling import elementwise_benchmark_cases as benchmark_cases
from profiling import merge_elementwise_broadcast as benchmark_merge
from profiling import profile_elementwise_broadcast as benchmark_profile


class ElementwiseBenchmarkCatalogTC(unittest.TestCase):

    def test_topology_contract_matches_numpy(self):
        """Verify every declared result and invalid broadcast with NumPy."""
        for size in (0, 1, 2, 3, 7):
            for topology in benchmark_cases.make_topologies(size):
                lhs = np.empty(topology.lhs_shape, dtype="float64")
                rhs_shape = () if topology.rhs_scalar else topology.rhs_shape
                rhs = np.empty(rhs_shape, dtype="float64")
                with self.subTest(size=size, topology=topology.name):
                    if topology.numpy_valid:
                        shape = np.broadcast_shapes(lhs.shape, rhs.shape)
                        self.assertEqual(topology.result_shape, shape)
                        self.assertEqual(
                            topology.inplace_valid,
                            topology.lhs_shape == shape,
                        )
                    else:
                        with self.assertRaises(ValueError):
                            np.broadcast_shapes(lhs.shape, rhs.shape)

    def test_layout_metadata_matches_physical_view(self):
        """Verify descriptors and touched offsets for every layout rule."""
        values = np.arange(1, 13, dtype="float64").reshape(3, 4)
        for layout_name in benchmark_cases.INPUT_LAYOUTS:
            with self.subTest(layout=layout_name):
                layout = benchmark_cases.make_layout(
                    values, layout_name
                )
                descriptor = layout.descriptor()
                offsets = layout.touched_storage_offsets()

                self.assertEqual(list(values.shape), descriptor["shape"])
                self.assertEqual(layout_name, descriptor["layout"])
                self.assertTrue(all(
                    0 <= offset < layout.storage.size
                    for offset in offsets
                ))
                if not layout_name.startswith("zero-"):
                    np.testing.assert_array_equal(values, layout.view)
                else:
                    self.assertIn(0, descriptor["strides"])

                before = layout.storage.copy()
                layout.view[...] = 23
                changed = set(np.flatnonzero(
                    before.reshape(-1) != layout.storage.reshape(-1)
                ))
                self.assertEqual(offsets, changed)

    def test_unsafe_linear_layouts_are_detected(self):
        """Prevent the legacy audit from executing out-of-span traversal."""
        values = np.arange(1, 13, dtype="float64").reshape(3, 4)
        unsafe = {
            "negative-inner",
            "negative-outer",
            "zero-inner",
            "zero-outer",
        }
        for layout_name in benchmark_cases.INPUT_LAYOUTS:
            layout = benchmark_cases.make_layout(
                values, layout_name
            )
            with self.subTest(layout=layout_name):
                self.assertEqual(
                    layout_name in unsafe,
                    benchmark_profile.linear_traversal_exceeds_span(
                        layout
                    ),
                )

    def test_unsafe_shape_comparisons_are_detected(self):
        """Detect prefix and shorter-rhs comparisons before legacy calls."""
        topologies = {
            topology.name: topology
            for topology in benchmark_cases.make_topologies(2)
        }
        self.assertEqual(
            "lhs shape is a prefix of rhs shape",
            benchmark_profile.legacy_shape_validation_hazard(
                _case_for_topology(
                    topologies["mixed-rank-reversed"]
                )
            ),
        )
        self.assertEqual(
            "lhs rank exceeds rhs rank",
            benchmark_profile.legacy_shape_validation_hazard(
                _case_for_topology(topologies["mixed-rank"])
            ),
        )
        self.assertIsNone(
            benchmark_profile.legacy_shape_validation_hazard(
                _case_for_topology(topologies["same-2d"])
            )
        )

    def test_supported_operation_reference_dtypes(self):
        """Verify the NumPy oracle returns the promised result dtype."""
        for dtype in benchmark_cases.ALL_DTYPES:
            lhs = benchmark_cases.make_values(
                (2, 3), dtype, seed=1
            )
            rhs = benchmark_cases.make_values(
                (1, 3), dtype, seed=2
            )
            for operation in benchmark_cases.ALL_OPERATIONS:
                if not benchmark_cases.operation_supported(
                    operation, dtype
                ):
                    continue
                with self.subTest(dtype=dtype, operation=operation):
                    result = benchmark_cases.operation_result(
                        operation,
                        lhs,
                        rhs,
                        dtype,
                    )
                    expected_dtype = (
                        np.dtype(dtype)
                        if operation in ("add", "sub", "mul", "div")
                        else np.dtype("bool")
                    )
                    self.assertEqual(expected_dtype, result.dtype)
                    self.assertEqual((2, 3), result.shape)

    def test_edge_value_patterns_include_numeric_boundaries(self):
        """Exercise IEEE specials and integer limits outside timing runs."""
        ieee = benchmark_cases.make_values(
            (8,),
            "float64",
            seed=0,
            pattern="ieee",
        )
        self.assertTrue(np.isnan(ieee).any())
        self.assertTrue(np.isposinf(ieee).any())
        self.assertTrue(np.isneginf(ieee).any())
        self.assertTrue(np.signbit(ieee[1]))

        boundary = benchmark_cases.make_values(
            (5,),
            "int16",
            seed=0,
            pattern="integer-boundary",
        )
        information = np.iinfo("int16")
        self.assertIn(information.min, boundary)
        self.assertIn(information.max, boundary)

    def test_integer_overlap_division_crash_case_is_deterministic(self):
        """Make the isolated legacy SIGFPE reproduction independent of RNG."""
        case = next(
            case
            for case in benchmark_cases.iter_case_specs("smoke")
            if (
                case.alias == "shift-forward"
                and case.operation == "div"
                and case.dtype == "int32"
                and case.mode == "in"
            )
        )
        lhs, rhs = benchmark_profile.make_alias_case_data(case)
        self.assertEqual([2, 1], lhs.storage[:2].tolist())
        self.assertTrue(np.shares_memory(lhs.view, rhs.view))

    def test_isolated_case_spec_round_trips_without_catalog_scan(self):
        """Let crash isolation execute the selected case directly."""
        case = benchmark_cases.CaseSpec(
            catalog="correctness",
            size=4,
            topology=benchmark_cases.Topology(
                "alias-reversed",
                (4,),
                (4,),
                (4,),
            ),
            operation="mul",
            dtype="int64",
            mode="in",
            lhs_layout="alias-reversed-lhs",
            rhs_layout="alias-reversed-rhs",
            alias="reversed",
        )

        restored = benchmark_profile.deserialize_case_spec(
            benchmark_profile.serialize_case_spec(case)
        )

        self.assertEqual(case, restored)

    def test_numpy_inplace_timing_uses_ufunc_output(self):
        """Keep the NumPy baseline allocation-free for in-place timing."""
        topology = next(
            topology
            for topology in benchmark_cases.make_topologies(4)
            if topology.name == "same-1d"
        )
        case = benchmark_cases.CaseSpec(
            catalog="performance",
            size=4,
            topology=topology,
            operation="add",
            dtype="float64",
            mode="in",
            lhs_layout="c",
            rhs_layout="c",
        )
        lhs = benchmark_cases.make_layout(
            np.arange(4, dtype="float64"), "c"
        )
        rhs = benchmark_cases.make_layout(
            np.arange(4, dtype="float64") + 1, "c"
        )
        expected = lhs.view.copy() + rhs.view

        result = benchmark_profile.numpy_call(case, lhs, rhs)

        self.assertTrue(np.shares_memory(result, lhs.view))
        np.testing.assert_array_equal(lhs.view, expected)

    def test_planned_timing_does_not_extract_numpy_view(self):
        """Time the operation without an extra ndarray property access."""
        topology = next(
            topology
            for topology in benchmark_cases.make_topologies(4)
            if topology.name == "same-1d"
        )
        case = benchmark_cases.CaseSpec(
            catalog="performance",
            size=4,
            topology=topology,
            operation="add",
            dtype="float64",
            mode="out",
            lhs_layout="c",
            rhs_layout="c",
        )
        lhs = benchmark_cases.make_layout(
            np.arange(4, dtype="float64"), "c"
        )
        rhs = benchmark_cases.make_layout(
            np.arange(4, dtype="float64") + 1, "c"
        )
        result = None

        def capture(function, *args):
            nonlocal result
            result = function()
            return {}

        arguments = types.SimpleNamespace(
            samples=1, warmup=0, target_ms=1
        )
        with unittest.mock.patch.object(
            benchmark_profile,
            "timed_samples",
            side_effect=capture,
        ):
            benchmark_profile.time_case_method(
                case, "planned", lhs, rhs, arguments
            )

        self.assertFalse(isinstance(result, np.ndarray))
        self.assertTrue(hasattr(result, "ndarray"))

    def test_preallocated_timing_reuses_destination(self):
        """Exercise the allocation-free planned diagnostic path."""
        topology = benchmark_cases.Topology(
            "outer",
            (4, 1),
            (1, 4),
            (4, 4),
        )
        case = benchmark_cases.CaseSpec(
            catalog="performance",
            size=4,
            topology=topology,
            operation="add",
            dtype="float64",
            mode="out",
            lhs_layout="c",
            rhs_layout="c",
        )
        lhs = benchmark_cases.make_layout(
            np.arange(4, dtype="float64").reshape(4, 1),
            "c",
        )
        rhs = benchmark_cases.make_layout(
            np.arange(4, dtype="float64").reshape(1, 4),
            "c",
        )
        expected = lhs.view + rhs.view
        audit = benchmark_profile.audit_preallocated_output(
            case, lhs, rhs, expected
        )
        result = object()

        def capture(function, *args):
            nonlocal result
            result = function()
            return {}

        arguments = types.SimpleNamespace(
            samples=1, warmup=0, target_ms=1
        )
        with unittest.mock.patch.object(
            benchmark_profile,
            "timed_samples",
            side_effect=capture,
        ):
            benchmark_profile.time_preallocated_output(
                case, "planned", lhs, rhs, arguments
            )

        self.assertEqual("match", audit["status"])
        self.assertIsNone(result)

    def test_stable_timing_order_balances_every_position(self):
        """Cancel timing-position bias across a complete schedule cycle."""
        methods = ("numpy", "planned", "numpy_to", "planned_to")
        orders = [
            benchmark_profile.balanced_method_order(methods, sequence)
            for sequence in range(8)
        ]

        for position in range(len(methods)):
            counts = {
                method: sum(order[position] == method for order in orders)
                for method in methods
            }
            self.assertEqual(dict.fromkeys(methods, 2), counts)

    def test_stable_summary_reports_independent_round_ratios(self):
        """Derive parity claims from rounds while retaining every sample."""
        observations = []
        values = {
            "numpy": (20, 24),
            "planned": (10, 12),
            "numpy_to": (15, 18),
            "planned_to": (10, 12),
        }
        for process_index in range(2):
            for round_index in range(2):
                for sample_index in range(2):
                    for order, (method, pair) in enumerate(values.items()):
                        per_call_ns = pair[round_index] + process_index
                        observations.append({
                            "process": process_index,
                            "round": round_index,
                            "sample": sample_index,
                            "sequence": round_index * 2 + sample_index,
                            "order": order,
                            "method": method,
                            "repeat": 3,
                            "elapsed_ns": per_call_ns * 3,
                            "per_call_ns": per_call_ns,
                        })

        summary = benchmark_profile.summarize_stable_observations(
            observations
        )

        self.assertEqual(8, summary["methods"]["numpy"]["sample_count"])
        self.assertEqual(
            {0: 8}, summary["order_counts"]["numpy"]
        )
        self.assertEqual(4, len(summary["rounds"]))
        self.assertEqual(4, summary["ratios"]["normal"]["round_count"])
        self.assertEqual(4, summary["ratios"]["reused"]["round_count"])
        self.assertGreater(summary["ratios"]["normal"]["minimum"], 1.8)
        self.assertGreater(summary["ratios"]["reused"]["minimum"], 1.4)

    def test_stable_summary_rejects_duplicate_samples(self):
        """Prevent retries from silently weighting a stable timing result."""
        observation = {
            "process": 0,
            "round": 0,
            "sample": 0,
            "sequence": 0,
            "order": 0,
            "method": "numpy",
            "repeat": 1,
            "elapsed_ns": 10,
            "per_call_ns": 10,
        }

        with self.assertRaisesRegex(ValueError, "duplicate stable"):
            benchmark_profile.summarize_stable_observations(
                [observation, dict(observation)]
            )

    def test_stable_reused_timing_requires_correct_destination(self):
        """Never benchmark a reused-output implementation that failed audit."""
        topology = benchmark_cases.Topology(
            "same-1d",
            (4,),
            (4,),
            (4,),
        )
        case = benchmark_cases.CaseSpec(
            catalog="performance",
            size=4,
            topology=topology,
            operation="add",
            dtype="float64",
            mode="out",
            lhs_layout="c",
            rhs_layout="c",
        )
        arguments = types.SimpleNamespace(
            implementation=None,
            preallocated_output=True,
            timing="stable",
        )
        for audit_status in ("match", "wrong-value"):
            with self.subTest(audit_status=audit_status):
                audit = {
                    "status": audit_status,
                    "error_type": "",
                    "error": "",
                }
                with (
                    unittest.mock.patch.object(
                        benchmark_profile,
                        "audit_implementation",
                        return_value={"status": "match"},
                    ),
                    unittest.mock.patch.object(
                        benchmark_profile,
                        "audit_preallocated_output",
                        return_value=audit,
                    ),
                    unittest.mock.patch.object(
                        benchmark_profile,
                        "run_stable_timing",
                        return_value={},
                    ) as run_stable_timing,
                ):
                    benchmark_profile.run_case(case, arguments)

                run_stable_timing.assert_called_once_with(
                    case, arguments, audit_status == "match"
                )

    def test_stable_benchmark_error_honors_failure_flag(self):
        """Return failure when an isolated stable timing process fails."""
        row = {
            "status": "bug",
            "implementations": {},
            "timing": {
                "stable": {"status": "benchmark-error"},
            },
        }
        arguments = [
            "profile_elementwise_broadcast.py",
            "--max-cases", "1",
            "--record", "summary",
            "--timing", "stable",
            "--fail-on-benchmark-error",
        ]
        self.assertTrue(
            benchmark_profile.should_record(row, ["benchmark-error"])
        )
        with (
            unittest.mock.patch.object(
                benchmark_profile.sys, "argv", arguments
            ),
            unittest.mock.patch.object(
                benchmark_profile, "run_case", return_value=row
            ),
            unittest.mock.patch("builtins.print"),
        ):
            self.assertEqual(2, benchmark_profile.main())

    def test_smoke_case_identifiers_are_unique(self):
        """Keep command-line filtering unambiguous for reproductions."""
        identifiers = [
            case.identifier
            for case in benchmark_cases.iter_case_specs("smoke")
        ]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_shards_partition_the_catalog(self):
        """Verify sharding neither drops nor duplicates benchmark cases."""
        cases = list(benchmark_cases.iter_case_specs("smoke"))
        shards = [
            list(
                benchmark_cases.shard_cases(
                    iter(cases), index, 7
                )
            )
            for index in range(7)
        ]
        identifiers = [
            case.identifier
            for shard in shards
            for case in shard
        ]
        expected = [case.identifier for case in cases]
        self.assertCountEqual(expected, identifiers)
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_reports_merge_only_as_a_complete_shard_set(self):
        """Preserve counts and reject partial benchmark shard results."""
        reports = [
            _shard_report(0, "case-0", "match"),
            _shard_report(1, "case-1", "wrong-value"),
        ]
        merged = benchmark_merge.merge_reports(reports)
        self.assertEqual(2, merged["summary"]["case_count"])
        self.assertEqual(2, merged["summary"]["recorded_case_count"])
        self.assertEqual(
            {"match": 1, "wrong-value": 1},
            merged["summary"]["implementation_statuses"]["legacy"],
        )
        self.assertEqual(
            {"match": 2},
            merged["summary"]["preallocated_output_statuses"],
        )
        self.assertEqual(
            ["case-0", "case-1"],
            [case["id"] for case in merged["cases"]],
        )
        with self.assertRaisesRegex(ValueError, "shard indices"):
            benchmark_merge.merge_reports(reports[:1])


def _case_for_topology(topology):
    return benchmark_cases.CaseSpec(
        catalog="correctness",
        size=2,
        topology=topology,
        operation="add",
        dtype="float64",
        mode="out",
        lhs_layout="c",
        rhs_layout="c",
    )


def _shard_report(index, identifier, legacy_status):
    return {
        "metadata": {
            "revision": "revision",
            "git_dirty": False,
            "platform": "platform",
            "machine": "machine",
            "python": "python",
            "numpy": "numpy",
            "thread_variables": {},
            "catalog": "smoke",
            "shard_index": index,
            "shard_count": 2,
            "timing": "none",
            "record": "findings",
            "record_status": None,
            "max_cases": None,
            "samples": 7,
            "warmup": 2,
            "target_ms": 20.0,
            "preallocated_output": True,
            "filters": {},
        },
        "summary": {
            "case_count": 1,
            "recorded_case_count": 1,
            "statuses": {"bug": 1},
            "coverage": {
                "topology": ["same-1d"],
                "size": [1],
                "operation": ["add"],
                "dtype": ["float64"],
                "mode": ["out"],
                "value_pattern": ["finite"],
                "lhs_layout": ["c"],
                "rhs_layout": ["c"],
            },
            "implementation_statuses": {
                "legacy": {legacy_status: 1},
                "legacy_simd": {"match": 1},
                "planned": {"unavailable": 1},
            },
            "preallocated_output_statuses": {"match": 1},
        },
        "cases": [{"id": identifier}],
    }


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
