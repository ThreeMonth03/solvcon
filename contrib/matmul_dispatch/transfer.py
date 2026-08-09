# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import dataclasses
import math

from .model import evaluate_predictions, percentile
from .transfer_profile import (
    LandmarkManifest,
    _shape_vector,
    _validate_profiles,
    calibration_key,
    make_device_signature,
    make_landmark_manifest,
)


@dataclasses.dataclass(frozen=True)
class TransferConfig:
    landmark_count: int = 16
    signature_rank: int = 4
    ridge_alpha: float = 4.0
    min_speedup: float = 1.03
    seed: int = 1208


@dataclasses.dataclass
class _TransferCore:
    manifest: LandmarkManifest
    shape_scaler: object
    signature_scaler: object
    signature_projection: object | None
    signature_mean: object
    route_models: dict


@dataclasses.dataclass
class TransferCostModel:
    core: _TransferCore
    route_margins: dict[str, float]
    config: TransferConfig


def _design_matrix(shape_matrix, device_vector):
    import numpy as np

    if not len(device_vector):
        return shape_matrix
    devices = np.repeat(
        device_vector.reshape(1, -1), len(shape_matrix), axis=0)
    interactions = (
        shape_matrix[:, :, None] * devices[:, None, :]
    ).reshape(len(shape_matrix), -1)
    return np.concatenate((shape_matrix, devices, interactions), axis=1)


def _fit_core(profiles, manifest, config):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    _validate_profiles(profiles, minimum_devices=1)
    names = tuple(sorted(profiles))
    signatures = np.asarray(
        [make_device_signature(profiles[name], manifest) for name in names],
        dtype="float64",
    )
    signature_scaler = StandardScaler().fit(signatures)
    normalized_signatures = signature_scaler.transform(signatures)
    rank = min(
        config.signature_rank,
        len(names) - 1,
        normalized_signatures.shape[1],
    )
    if rank:
        projection = PCA(n_components=rank).fit(normalized_signatures)
        device_vectors = projection.transform(normalized_signatures)
    else:
        projection = None
        device_vectors = np.empty((len(names), 0), dtype="float64")

    all_records = [record for name in names for record in profiles[name]]
    shape_scaler = StandardScaler().fit(np.asarray(
        [_shape_vector(record) for record in all_records],
        dtype="float64",
    ))
    route_models = {}
    for route in manifest.routes:
        rows = []
        targets = []
        for device_index, name in enumerate(names):
            records = profiles[name]
            selected = [record for record in records
                        if route in record["median_ns"]]
            if not selected:
                continue
            shapes = shape_scaler.transform(np.asarray(
                [_shape_vector(record) for record in selected],
                dtype="float64",
            ))
            rows.extend(_design_matrix(
                shapes, device_vectors[device_index]))
            targets.extend(
                math.log(
                    record["median_ns"][route] /
                    record["median_ns"][record["current_kernel"]])
                for record in selected
            )
        if rows:
            route_models[route] = Ridge(
                alpha=config.ridge_alpha).fit(
                    np.asarray(rows, dtype="float64"),
                    np.asarray(targets, dtype="float64"),
                )
    return _TransferCore(
        manifest=manifest,
        shape_scaler=shape_scaler,
        signature_scaler=signature_scaler,
        signature_projection=projection,
        signature_mean=signatures.mean(axis=0),
        route_models=route_models,
    )


def _device_vector(core, signature):
    import numpy as np

    if core.signature_projection is None:
        return np.empty(0, dtype="float64")
    raw = core.signature_mean if signature is None else signature
    normalized = core.signature_scaler.transform(raw.reshape(1, -1))
    return core.signature_projection.transform(normalized)[0]


def predict_log_ratios(core, signature, record):
    import numpy as np

    shape = core.shape_scaler.transform(np.asarray(
        [_shape_vector(record)], dtype="float64"))
    design = _design_matrix(shape, _device_vector(core, signature))
    return {
        route: float(model.predict(design)[0])
        for route, model in core.route_models.items()
    }


def _landmark_groups(records, manifest):
    keys = set(manifest.keys)
    return {
        record["group"] for record in records
        if calibration_key(record) in keys
    }


def _estimate_route_margins(profiles, manifest, config):
    errors = {route: [] for route in manifest.routes}
    for held_name, held_records in profiles.items():
        training = {
            name: records for name, records in profiles.items()
            if name != held_name
        }
        core = _fit_core(training, manifest, config)
        signature = make_device_signature(held_records, manifest)
        excluded = _landmark_groups(held_records, manifest)
        for record in held_records:
            if record["group"] in excluded:
                continue
            predictions = predict_log_ratios(core, signature, record)
            current = record["median_ns"][record["current_kernel"]]
            for route, prediction in predictions.items():
                if route in record["median_ns"]:
                    actual = math.log(record["median_ns"][route] / current)
                    errors[route].append(actual - prediction)
    return {
        route: max(0.0, percentile(values, 0.95))
        if values else math.inf
        for route, values in errors.items()
    }


def fit_transfer_model(profiles, manifest, config=TransferConfig()):
    _validate_profiles(profiles)
    core = _fit_core(profiles, manifest, config)
    margins = _estimate_route_margins(profiles, manifest, config)
    return TransferCostModel(
        core=core, route_margins=margins, config=config)


def select_transfer_route(model, signature, record):
    predictions = predict_log_ratios(model.core, signature, record)
    eligible = set(record.get("eligible_kernels", predictions))
    candidates = {
        route: prediction + model.route_margins.get(route, math.inf)
        for route, prediction in predictions.items()
        if route in eligible
    }
    if not candidates:
        return record["current_kernel"]
    route, upper_ratio = min(candidates.items(), key=lambda item: item[1])
    if upper_ratio > -math.log(model.config.min_speedup):
        return record["current_kernel"]
    return route


def evaluate_transfer_target(model, records, use_landmarks=True):
    signature = (make_device_signature(records, model.core.manifest)
                 if use_landmarks else None)
    excluded = (_landmark_groups(records, model.core.manifest)
                if use_landmarks else set())
    validation = [record for record in records
                  if record["group"] not in excluded]
    predictions = [
        select_transfer_route(model, signature, record)
        for record in validation
    ]
    metrics = evaluate_predictions(validation, predictions)
    metrics["landmark_records"] = len(model.core.manifest.keys)
    metrics["validation_groups"] = len({
        record["group"] for record in validation
    })
    return metrics


def evaluate_leave_one_device_out(profiles, config=TransferConfig()):
    if len(profiles) < 3:
        raise ValueError(
            "leave-one-device-out validation needs at least three devices")
    manifest = make_landmark_manifest(
        profiles, config.landmark_count, config.seed)
    results = {}
    for held_name, held_records in profiles.items():
        training = {
            name: records for name, records in profiles.items()
            if name != held_name
        }
        model = fit_transfer_model(training, manifest, config)
        results[held_name] = evaluate_transfer_target(model, held_records)
    return {
        "method": "leave_one_device_out",
        "landmarks": len(manifest.keys),
        "devices": results,
    }


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
