# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import enum
import json
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from contrib import matmul_dispatch as tune
from contrib.matmul_dispatch import collect
from contrib.matmul_dispatch import measurement


class Kernel(enum.Enum):
    GenericIjk = 0
    FixedIkj = 1
    BlasGemm = 2


def make_profile():
    return {
        "operation": "gemm",
        "dtype": "float32",
        "backend": "cblas",
        "rows": 8,
        "columns": 8,
        "inner_size": 8,
        "batch_size": 1,
        "has_batch_axes": False,
        "lhs_layout": "row_major",
        "rhs_layout": "row_major",
        "lhs_row_stride": 8,
        "lhs_inner_stride": 1,
        "rhs_inner_stride": 8,
        "rhs_column_stride": 1,
        "lhs_reused": False,
        "rhs_reused": False,
        "lhs_zero_batch_stride": False,
        "rhs_zero_batch_stride": False,
        "current_kernel": Kernel.FixedIkj,
        "eligible_kernels": (
            Kernel.GenericIjk,
            Kernel.FixedIkj,
            Kernel.BlasGemm,
        ),
    }


def make_measured_record(
        timings, auto_timings=(12, 12), sample_id="sample"):
    current = next(iter(timings))
    return {
        "schema_version": tune.SCHEMA_VERSION,
        "status": "measured",
        "sample_id": sample_id,
        "current_kernel": current,
        "measured_kernels": list(timings),
        "timings_ns": {
            route: list(values)
            for route, values in timings.items()
        },
        "timing_batches_ns": {
            route: list(values)
            for route, values in timings.items()
        },
        "inner_repetitions": {
            route: 1 for route in timings
        },
        "median_ns": {
            route: statistics.median(values)
            for route, values in timings.items()
        },
        "auto_timings_ns": list(auto_timings),
        "auto_timing_batches_ns": list(auto_timings),
        "auto_inner_repetitions": 1,
        "auto_median_ns": statistics.median(auto_timings),
    }


class MatmulDispatchMeasurementTC(unittest.TestCase):

    def test_profile_keeps_enum_objects_for_forced_calls(self):
        facts, kernels, current = measurement.normalize_profile(make_profile())
        self.assertEqual("gemm", facts["operation"])
        self.assertIs(Kernel.BlasGemm, kernels["BlasGemm"])
        self.assertEqual("FixedIkj", current)

    def test_collection_refines_only_after_the_coarse_sweep(self):
        first = tune.GemmCase(
            "square", "a", 8, 8, 8, "float32", "C", "C")
        second = tune.GemmCase(
            "square", "b", 16, 16, 16, "float32", "C", "C")
        coarse_first = make_measured_record(
            {"GenericIjk": (10, 10), "BlasGemm": (11, 11)},
            sample_id=first.identifier,
        )
        coarse_second = make_measured_record(
            {"GenericIjk": (10, 10), "BlasGemm": (20, 20)},
            sample_id=second.identifier,
        )
        refined_first = make_measured_record(
            {"GenericIjk": (12, 12), "BlasGemm": (9, 9)},
            sample_id=first.identifier,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.jsonl"
            args = tune.make_parser().parse_args((
                "collect",
                "--output", str(output),
                "--budget-seconds", "100",
                "--max-cases", "2",
            ))
            with (
                mock.patch.object(
                    collect, "make_gemm_cases",
                    return_value=(first, second),
                ),
                mock.patch.object(
                    collect, "make_environment", return_value={}),
                mock.patch.object(
                    collect, "measure_case",
                    side_effect=(coarse_first, coarse_second,
                                 refined_first),
                ) as measure,
            ):
                result = collect.collect_records(args)
            records = [
                json.loads(line)
                for line in output.read_text(
                    encoding="utf-8").splitlines()
            ]
        measured_cases = [call.args[0] for call in measure.call_args_list]
        self.assertEqual([first, second, first], measured_cases)
        coarse_deadline = measure.call_args_list[0].args[1]
        self.assertEqual(
            coarse_deadline, measure.call_args_list[1].args[1])
        self.assertLess(
            coarse_deadline, measure.call_args_list[2].args[1])
        self.assertEqual(
            {"completed": 1, "not_selected": 1},
            result["refinement"]["status_counts"],
        )
        self.assertEqual(
            "completed", records[0]["refinement"]["status"])

    def test_collection_scope_accepts_only_one_dtype_and_layout(self):
        args = tune.make_parser().parse_args((
            "collect", "--dtypes", "float32", "float64"))
        with self.assertRaisesRegex(ValueError, "one dtype"):
            collect._validate_collection_scope(args)

    def test_refinement_appends_a_raw_block_and_updates_medians(self):
        record = make_measured_record({
            "GenericIjk": (10, 10),
            "BlasGemm": (11, 11),
        })
        refined = make_measured_record({
            "GenericIjk": (12, 12, 12),
            "BlasGemm": (9, 9, 9),
        })
        self.assertTrue(measurement._annotate_refinement(record, 0.15))
        measurement._merge_refinement(record, refined)
        self.assertEqual("completed", record["refinement"]["status"])
        self.assertEqual(1, record["refinement"]["completed_blocks"])
        self.assertEqual(2, len(record["measurement_blocks"]))
        block = record["measurement_blocks"][1]
        self.assertEqual("refinement", block["phase"])
        self.assertEqual([2, 5], block["route_ranges"]["BlasGemm"])
        self.assertEqual(12, record["median_ns"]["GenericIjk"])
        self.assertEqual(9, record["median_ns"]["BlasGemm"])


if __name__ == "__main__":
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
