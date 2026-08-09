# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import math
import unittest

from contrib import matmul_dispatch as tune


def make_record(device, side, inner_size, index, family="shape"):
    work = side * side * inner_size
    latent = {
        "slow_blas": -0.8,
        "balanced": 0.0,
        "fast_blas": 0.8,
        "faster_blas": 1.2,
    }[device]
    shape = math.log2(work + 1) / 20
    blas_ratio = math.exp(0.22 - shape - latent * (0.4 + shape))
    current = 1000.0 + work / 100
    return {
        "family": family,
        "group": f"{family}:{index // 2}",
        "sample_id": f"{device}:{index}",
        "facts": {
            "operation": "gemm",
            "dtype": "float64",
            "backend": device,
            "rows": side,
            "columns": side,
            "inner_size": inner_size,
            "batch_size": 1,
            "lhs_layout": "row_major",
            "rhs_layout": "row_major",
            "has_batch_axes": False,
            "lhs_reused": False,
            "rhs_reused": False,
            "lhs_zero_batch_stride": False,
            "rhs_zero_batch_stride": False,
        },
        "current_kernel": "GenericIjk",
        "eligible_kernels": ("GenericIjk", "BlasGemm"),
        "median_ns": {
            "GenericIjk": current,
            "BlasGemm": current * blas_ratio,
        },
    }


def make_profiles():
    shapes = (
        (4, 2), (6, 3), (8, 4), (12, 4),
        (16, 6), (24, 8), (32, 12), (48, 16),
        (64, 24), (96, 32), (128, 48), (192, 64),
    )
    return {
        device: tuple(
            make_record(device, side, inner_size, index)
            for index, (side, inner_size) in enumerate(shapes)
        )
        for device in (
            "slow_blas", "balanced", "fast_blas", "faster_blas")
    }


def add_measurement_source(records, source_hash):
    hashes = {
        path: source_hash for path in tune.MEASUREMENT_SOURCES
    }
    return tuple({
        **record,
        "environment": {"tuning_source_sha256": hashes},
    } for record in records)


def add_hardware(records, hardware):
    return tuple({
        **record,
        "environment": {
            **record.get("environment", {}),
            "cpu": hardware,
            "machine": hardware,
        },
    } for record in records)


class MatmulDispatchTransferTC(unittest.TestCase):

    def test_calibration_key_excludes_machine_identity(self):
        profiles = make_profiles()
        slow = profiles["slow_blas"][0]
        fast = profiles["fast_blas"][0]
        self.assertNotEqual(
            slow["facts"]["backend"], fast["facts"]["backend"])
        self.assertEqual(
            tune.calibration_key(slow), tune.calibration_key(fast))

    def test_landmarks_are_timing_independent_and_space_filling(self):
        profiles = make_profiles()
        manifest = tune.make_landmark_manifest(profiles, 4, seed=23)
        changed = {
            device: tuple({
                **record,
                "median_ns": {
                    route: value * (index + 2)
                    for route, value in record["median_ns"].items()
                },
            } for index, record in enumerate(records))
            for device, records in profiles.items()
        }
        repeated = tune.make_landmark_manifest(changed, 4, seed=23)
        self.assertEqual(manifest, repeated)
        self.assertEqual(4, len(set(manifest.keys)))
        self.assertEqual(
            ("BlasGemm", "GenericIjk"), manifest.routes)
        records = {
            tune.calibration_key(record): record
            for record in profiles["balanced"]
        }
        for route in manifest.routes:
            self.assertTrue(any(
                route in records[key]["median_ns"]
                for key in manifest.keys
            ))
        selected_sides = [
            records[key]["facts"]["rows"] for key in manifest.keys
        ]
        self.assertEqual(4, min(selected_sides))
        self.assertEqual(192, max(selected_sides))

    def test_one_source_can_freeze_a_target_landmark_manifest(self):
        profiles = make_profiles()
        manifest = tune.make_landmark_manifest(
            {"source": profiles["balanced"]}, 4)
        self.assertEqual(4, len(manifest.keys))

    def test_signature_requires_every_frozen_landmark(self):
        profiles = make_profiles()
        manifest = tune.make_landmark_manifest(profiles, 4)
        missing = manifest.keys[0]
        incomplete = tuple(
            record for record in profiles["balanced"]
            if tune.calibration_key(record) != missing
        )
        with self.assertRaisesRegex(ValueError, "missing a landmark"):
            tune.make_device_signature(incomplete, manifest)

    def test_signature_changes_with_route_costs(self):
        profiles = make_profiles()
        manifest = tune.make_landmark_manifest(profiles, 4)
        slow = tune.make_device_signature(
            profiles["slow_blas"], manifest)
        fast = tune.make_device_signature(
            profiles["fast_blas"], manifest)
        self.assertFalse((slow == fast).all())

    def test_profiles_require_the_same_shape_manifest(self):
        profiles = make_profiles()
        mismatched = dict(profiles)
        mismatched["balanced"] = mismatched["balanced"][:-1]
        with self.assertRaisesRegex(ValueError, "same calibration manifest"):
            tune.make_landmark_manifest(mismatched, 4)

    def test_profiles_require_the_same_route_schema(self):
        profiles = make_profiles()
        changed = dict(profiles)
        first = changed["balanced"][0]
        changed["balanced"] = (
            {**first, "median_ns": {"GenericIjk": 1000.0}},
            *changed["balanced"][1:],
        )
        with self.assertRaisesRegex(ValueError, "same route schema"):
            tune.make_landmark_manifest(changed, 4)

    def test_profiles_require_matching_measurement_sources(self):
        profiles = make_profiles()
        changed = {
            name: add_measurement_source(records, name)
            for name, records in profiles.items()
        }
        with self.assertRaisesRegex(ValueError, "measurement sources"):
            tune.make_landmark_manifest(changed, 4)

    def test_profiles_require_complete_measurement_sources(self):
        profiles = make_profiles()
        first = profiles["balanced"][0]
        incomplete = ({
            **first,
            "environment": {
                "tuning_source_sha256": {"unknown.cpp": "hash"},
            },
        }, *profiles["balanced"][1:])
        with self.assertRaisesRegex(ValueError, "source hashes"):
            tune.make_landmark_manifest({"source": incomplete}, 4)

    def test_profiles_reject_duplicate_hardware_identity(self):
        profiles = make_profiles()
        changed = {
            name: add_hardware(records, "same-cpu")
            for name, records in profiles.items()
        }
        with self.assertRaisesRegex(ValueError, "distinct hardware"):
            tune.make_landmark_manifest(changed, 4)

    def test_leave_one_device_out_excludes_landmark_groups(self):
        profiles = make_profiles()
        config = tune.TransferConfig(
            landmark_count=2,
            signature_rank=2,
            ridge_alpha=2.0,
            min_speedup=1.03,
            seed=17,
        )
        report = tune.evaluate_leave_one_device_out(profiles, config)
        self.assertEqual("leave_one_device_out", report["method"])
        self.assertEqual(2, report["landmarks"])
        for metrics in report["devices"].values():
            self.assertEqual(4, metrics["validation_groups"])
            self.assertEqual(8, metrics["samples"])

    def test_transfer_model_changes_same_shape_decision(self):
        profiles = make_profiles()
        sources = {
            name: records for name, records in profiles.items()
            if name != "balanced"
        }
        manifest = tune.make_landmark_manifest(sources, 4, seed=31)
        config = tune.TransferConfig(
            landmark_count=4,
            signature_rank=2,
            ridge_alpha=1.0,
            min_speedup=1.03,
            seed=31,
        )
        model = tune.fit_transfer_model(sources, manifest, config)
        model.route_margins = {
            route: 0.0 for route in model.route_margins
        }
        slow = profiles["slow_blas"][-1]
        fast = profiles["faster_blas"][-1]
        slow_signature = tune.make_device_signature(
            profiles["slow_blas"], manifest)
        fast_signature = tune.make_device_signature(
            profiles["faster_blas"], manifest)
        self.assertEqual(
            "GenericIjk",
            tune.select_transfer_route(model, slow_signature, slow),
        )
        self.assertEqual(
            "BlasGemm",
            tune.select_transfer_route(model, fast_signature, fast),
        )
        model.route_margins = {
            route: math.inf for route in model.route_margins
        }
        self.assertEqual(
            "GenericIjk",
            tune.select_transfer_route(model, fast_signature, fast),
        )

    def test_leave_one_device_out_requires_three_devices(self):
        profiles = make_profiles()
        with self.assertRaisesRegex(ValueError, "at least three"):
            tune.evaluate_leave_one_device_out({
                name: profiles[name]
                for name in ("slow_blas", "fast_blas")
            })


if __name__ == "__main__":
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
