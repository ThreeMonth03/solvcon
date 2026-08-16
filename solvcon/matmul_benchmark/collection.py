# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Versioned collection plans for reproducible matmul atlas sampling."""

import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import uuid

import numpy as np

from . import arrays
from . import duration
from . import profiles as profiles_module
from . import schedule
from . import schema


PLAN_SCHEMA_VERSION = 4
SUPPORTED_PLAN_SCHEMA_VERSIONS = (1, 2, 3, PLAN_SCHEMA_VERSION)
PLAN_KIND = 'solvcon.matmul_benchmark_plan'
DEFAULT_SHAPE_VALUES = (8, 16, 32, 64, 128, 256)
DEFAULT_ROUTES = ('generic', 'blas_gemm', 'winograd')
LAYOUT_PROFILES = (
    'contiguous',
    'lhs_padded',
    'rhs_padded',
    'both_padded',
    'lhs_column_major',
    'rhs_column_major',
    'both_column_major',
)
BROADCAST_PROFILES = (
    'matrix',
    'matched_batch',
    'broadcast_lhs',
    'broadcast_rhs',
)
DEFAULT_BATCH_SIZE = 8
RECOMMENDED_COLLECTION_CELLS = 4096
RECOMMENDED_COLLECTION_CALLS = 250_000
RECOMMENDED_COLLECTION_ARTIFACT_BYTES = 128 * 1024 * 1024
MAX_COLLECTION_CELLS = 16_384
MAX_COLLECTION_CALLS = 1_000_000
LEGACY_MAX_COLLECTION_MEASUREMENT_WORK = 1_000_000_000_000
MAX_COLLECTION_ARTIFACT_BYTES = 512 * 1024 * 1024
ARTIFACT_FIXED_BYTES_PER_CELL = 4096
ARTIFACT_BYTES_PER_PANEL_SAMPLE = 1024
TARGET_DURATION_FIELDS = (
    'seconds',
    'mode',
    'safety_fraction',
    'calibration_block_seconds',
    'minimum_calibration_repetitions',
    'maximum_calibration_repetitions',
    'checkpoint_seconds',
    'uncertainty_fraction',
)


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise schema.SchemaError(f'{name} must be an object')
    return value


def _reject_unknown(data, name, allowed):
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise schema.SchemaError(f'{name} has unknown fields: {unknown}')


def _require_fields(data, name, required):
    missing = sorted(set(required) - set(data))
    if missing:
        raise schema.SchemaError(f'{name} is missing fields: {missing}')


def _require_int(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise schema.SchemaError(f'{name} must be an integer')
    if minimum is not None and value < minimum:
        raise schema.SchemaError(f'{name} must be at least {minimum}')
    return value


def _require_str(value, name):
    if not isinstance(value, str) or not value:
        raise schema.SchemaError(f'{name} must be a non-empty string')
    return value


def _string_tuple(value, name):
    if not isinstance(value, (list, tuple)):
        raise schema.SchemaError(f'{name} must be an array')
    result = tuple(
        _require_str(item, f'{name}[{index}]')
        for index, item in enumerate(value)
    )
    if not result:
        raise schema.SchemaError(f'{name} must not be empty')
    if len(result) != len(set(result)):
        raise schema.SchemaError(f'{name} must not contain duplicates')
    return result


def target_duration_from_dict(data):
    """Build a target-duration contract from its JSON representation."""

    if data is None:
        return None
    data = _require_mapping(data, 'target_duration')
    _reject_unknown(data, 'target_duration', TARGET_DURATION_FIELDS)
    _require_fields(data, 'target_duration', TARGET_DURATION_FIELDS)
    try:
        return duration.TargetDurationSpec(**data)
    except duration.DurationModelError as exc:
        raise schema.SchemaError(str(exc)) from exc


def target_duration_to_dict(spec):
    if spec is None:
        return None
    if not isinstance(spec, duration.TargetDurationSpec):
        raise schema.SchemaError(
            'target_duration must be a TargetDurationSpec or null')
    return dataclasses.asdict(spec)


def _infer_broadcast_profile(lhs, rhs):
    lhs_batch = lhs.shape[:-2] if len(lhs.shape) > 1 else ()
    rhs_batch = rhs.shape[:-2] if len(rhs.shape) > 1 else ()
    batch_shape = np.broadcast_shapes(lhs_batch, rhs_batch)
    if not batch_shape:
        return 'matrix'

    def reused(operand):
        operand_shape = operand.shape[:-2] if len(operand.shape) > 1 \
            else ()
        operand_strides = operand.strides[:-2] \
            if len(operand.shape) > 1 else ()
        missing = len(batch_shape) - len(operand_shape)
        shapes = (1,) * missing + operand_shape
        strides = (0,) * missing + operand_strides
        return any(
            (source == 1 and target > 1)
            or (source > 1 and stride == 0)
            for source, target, stride in zip(
                shapes, batch_shape, strides))

    lhs_reused = reused(lhs)
    rhs_reused = reused(rhs)
    if lhs_reused and not rhs_reused:
        return 'broadcast_lhs'
    if rhs_reused and not lhs_reused:
        return 'broadcast_rhs'
    if not lhs_reused and not rhs_reused:
        return 'matched_batch'
    return 'broadcast_both'


@dataclasses.dataclass(frozen=True)
class CollectionCell:
    """Describe one explicit primitive input point in a collection plan."""

    cell_id: str
    lhs: schema.OperandSpec
    rhs: schema.OperandSpec
    layout: str = 'contiguous'
    broadcast: str = 'matrix'
    routes: tuple | None = None

    def __post_init__(self):
        _require_str(self.cell_id, 'cell.id')
        _require_str(self.layout, 'cell.layout')
        _require_str(self.broadcast, 'cell.broadcast')
        if not isinstance(self.lhs, schema.OperandSpec):
            raise schema.SchemaError('cell.lhs must be an OperandSpec')
        if not isinstance(self.rhs, schema.OperandSpec):
            raise schema.SchemaError('cell.rhs must be an OperandSpec')
        if self.routes is not None:
            if not isinstance(self.routes, (list, tuple)):
                raise schema.SchemaError(
                    'cell.routes must be an array or null')
            routes = tuple(
                _require_str(route, f'cell.routes[{index}]')
                for index, route in enumerate(self.routes)
            )
            if len(routes) != len(set(routes)):
                raise schema.SchemaError(
                    'cell.routes must not contain duplicates')
            object.__setattr__(self, 'routes', routes)

    @classmethod
    def from_dict(cls, data):
        data = _require_mapping(data, 'cell')
        fields = ('id', 'lhs', 'rhs', 'layout', 'broadcast', 'routes')
        _reject_unknown(data, 'cell', fields)
        _require_fields(
            data, 'cell', ('id', 'lhs', 'rhs', 'layout', 'routes'))
        lhs = schema.OperandSpec.from_dict(data['lhs'])
        rhs = schema.OperandSpec.from_dict(data['rhs'])
        return cls(
            cell_id=data['id'],
            lhs=lhs,
            rhs=rhs,
            layout=data['layout'],
            broadcast=data.get(
                'broadcast', _infer_broadcast_profile(lhs, rhs)),
            routes=data['routes'],
        )

    def to_dict(self, include_broadcast=True):
        result = {
            'id': self.cell_id,
            'lhs': self.lhs.to_dict(),
            'rhs': self.rhs.to_dict(),
            'layout': self.layout,
            'routes': (None if self.routes is None
                       else list(self.routes)),
        }
        if include_broadcast:
            result['broadcast'] = self.broadcast
        return result


@dataclasses.dataclass(frozen=True)
class CollectionPlan:
    """Freeze every input and measurement control for one worker run."""

    cells: tuple
    dtype: str = 'float32'
    mode: schema.ModeSpec = dataclasses.field(
        default_factory=schema.ModeSpec)
    target_duration: duration.TargetDurationSpec | None = None
    allow_large_work: bool = False
    routes: tuple = DEFAULT_ROUTES
    numpy_baseline: bool = True
    seed: int = 20260815
    threads: int = 1
    output_path: str | None = None
    plan_id: str = dataclasses.field(
        default_factory=lambda: uuid.uuid4().hex)
    schema_version: int = PLAN_SCHEMA_VERSION
    schema_kind: str = PLAN_KIND

    def __post_init__(self):
        if self.schema_version not in SUPPORTED_PLAN_SCHEMA_VERSIONS:
            raise schema.SchemaError(
                f'unsupported collection plan version: '
                f'{self.schema_version}')
        if self.schema_kind != PLAN_KIND:
            raise schema.SchemaError('not a matmul benchmark plan')
        if not isinstance(self.cells, (list, tuple)):
            raise schema.SchemaError('plan.cells must be an array')
        cells = tuple(self.cells)
        if not cells:
            raise schema.SchemaError('plan.cells must not be empty')
        if len(cells) > MAX_COLLECTION_CELLS:
            raise schema.SchemaError(
                f'plan has {len(cells)} cells, limit is '
                f'{MAX_COLLECTION_CELLS}')
        if not all(isinstance(cell, CollectionCell) for cell in cells):
            raise schema.SchemaError(
                'plan.cells must contain CollectionCell objects')
        cell_ids = [cell.cell_id for cell in cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise schema.SchemaError('plan cell IDs must be unique')
        object.__setattr__(self, 'cells', cells)
        if self.dtype not in schema.SUPPORTED_DTYPES:
            raise schema.SchemaError(f'unsupported dtype: {self.dtype!r}')
        if not isinstance(self.mode, schema.ModeSpec):
            raise schema.SchemaError('plan.mode must be a ModeSpec')
        if self.target_duration is not None:
            if self.schema_version < 3:
                raise schema.SchemaError(
                    'target_duration requires collection plan version 3 '
                    'or newer')
            if not isinstance(
                    self.target_duration, duration.TargetDurationSpec):
                raise schema.SchemaError(
                    'plan.target_duration must be a '
                    'TargetDurationSpec or null')
            expected_mode = schema.ModeSpec.preset(
                self.target_duration.mode)
            if self.mode != expected_mode:
                raise schema.SchemaError(
                    'a target-duration plan mode must match its '
                    'quality preset')
        if not isinstance(self.allow_large_work, bool):
            raise schema.SchemaError(
                'plan.allow_large_work must be a boolean')
        if self.allow_large_work and self.schema_version < 4:
            raise schema.SchemaError(
                'allow_large_work requires collection plan version 4')
        if self.allow_large_work and self.target_duration is not None:
            raise schema.SchemaError(
                'plan.allow_large_work compatibility field is available '
                'only for fixed schedules')
        routes = _string_tuple(self.routes, 'plan.routes')
        reserved_routes = sorted(set(routes) & {'auto', 'numpy'})
        if reserved_routes:
            raise schema.SchemaError(
                f'plan.routes contains reserved names: {reserved_routes}')
        object.__setattr__(self, 'routes', routes)
        for cell in cells:
            unknown_routes = (
                set() if cell.routes is None
                else set(cell.routes) - set(routes))
            if unknown_routes:
                raise schema.SchemaError(
                    f'cell {cell.cell_id!r} has routes outside plan.routes: '
                    f'{sorted(unknown_routes)}')
        if not isinstance(self.numpy_baseline, bool):
            raise schema.SchemaError(
                'plan.numpy_baseline must be a boolean')
        _require_int(self.seed, 'plan.seed', 0)
        _require_int(self.threads, 'plan.threads', 1)
        if self.output_path is not None:
            path = os.fspath(self.output_path)
            if not path:
                raise schema.SchemaError(
                    'plan.output_path must not be empty')
            object.__setattr__(self, 'output_path', path)
        _require_str(self.plan_id, 'plan.id')
        for index in range(len(cells)):
            self.request_at(index)
        estimate = _calculate_estimate(self, legacy_peak=False)
        _validate_schedule_caps(estimate)
        _validate_artifact_cap(self)

    @classmethod
    def from_dict(cls, data):
        data = _require_mapping(data, 'plan')
        common_fields = (
            'schema_version', 'schema_kind', 'id', 'cells', 'dtype',
            'mode', 'routes', 'numpy_baseline', 'seed', 'threads',
            'output_path',
        )
        version = data.get('schema_version')
        fields = common_fields
        if isinstance(version, int) and version >= 3:
            fields += ('target_duration',)
        if version == 4:
            fields += ('allow_large_work',)
        _reject_unknown(data, 'plan', fields)
        _require_fields(data, 'plan', fields)
        raw_cells = data['cells']
        if not isinstance(raw_cells, list):
            raise schema.SchemaError('plan.cells must be an array')
        if len(raw_cells) > MAX_COLLECTION_CELLS:
            raise schema.SchemaError(
                f'plan has {len(raw_cells)} cells, limit is '
                f'{MAX_COLLECTION_CELLS}')
        return cls(
            schema_version=data['schema_version'],
            schema_kind=data['schema_kind'],
            plan_id=data['id'],
            cells=tuple(CollectionCell.from_dict(item)
                        for item in raw_cells),
            dtype=data['dtype'],
            mode=schema.ModeSpec.from_dict(data['mode']),
            target_duration=target_duration_from_dict(
                data.get('target_duration')),
            allow_large_work=data.get('allow_large_work', False),
            routes=data['routes'],
            numpy_baseline=data['numpy_baseline'],
            seed=data['seed'],
            threads=data['threads'],
            output_path=data['output_path'],
        )

    def request_at(self, index):
        cell = self.cells[index]
        return schema.BenchmarkRequest(
            lhs=cell.lhs,
            rhs=cell.rhs,
            dtype=self.dtype,
            mode=self.mode,
            routes=(self.routes if cell.routes is None else cell.routes),
            numpy_baseline=self.numpy_baseline,
            seed=self.seed + index * 2,
            threads=self.threads,
            request_id=f'{self.plan_id}:{cell.cell_id}',
        )

    def requests(self):
        return tuple(self.request_at(index)
                     for index in range(len(self.cells)))

    def measurement_dict(self):
        data = self.to_dict()
        data.pop('id')
        data.pop('output_path')
        data.pop('allow_large_work', None)
        return data

    def sha256(self):
        payload = json.dumps(
            self.measurement_dict(), sort_keys=True,
            separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(payload.encode('utf8')).hexdigest()

    def to_dict(self):
        result = {
            'schema_version': self.schema_version,
            'schema_kind': self.schema_kind,
            'id': self.plan_id,
            'cells': [
                cell.to_dict(
                    include_broadcast=self.schema_version >= 2)
                for cell in self.cells
            ],
            'dtype': self.dtype,
            'mode': self.mode.to_dict(),
            'routes': list(self.routes),
            'numpy_baseline': self.numpy_baseline,
            'seed': self.seed,
            'threads': self.threads,
            'output_path': self.output_path,
        }
        if self.schema_version >= 3:
            result['target_duration'] = target_duration_to_dict(
                self.target_duration)
        if self.schema_version >= 4:
            result['allow_large_work'] = self.allow_large_work
        return result


@dataclasses.dataclass(frozen=True)
class CollectionEstimate:
    """Report exact schedule size under the collection accounting model."""

    cell_count: int
    route_count: int
    panel_count: int
    preflight_calls: int
    matmul_calls: int
    scalar_contractions: int
    measurement_work: int
    peak_bytes: int

    @classmethod
    def from_dict(cls, data):
        data = _require_mapping(data, 'collection estimate')
        fields = (
            'cell_count', 'route_count', 'panel_count', 'preflight_calls',
            'matmul_calls', 'scalar_contractions', 'measurement_work',
            'peak_bytes',
        )
        _reject_unknown(data, 'collection estimate', fields)
        _require_fields(data, 'collection estimate', fields)
        return cls(**{
            field: _require_int(data[field], f'estimate.{field}', 0)
            for field in fields
        })

    def to_dict(self):
        return dataclasses.asdict(self)


def _cell_metrics(cell, dtype, routes, resource_budget=None):
    request = schema.BenchmarkRequest(
        lhs=cell.lhs,
        rhs=cell.rhs,
        dtype=np.dtype(dtype).name,
    )
    lhs_shape = request.lhs.shape
    rhs_shape = request.rhs.shape
    lhs_batch = () if len(lhs_shape) == 1 else lhs_shape[:-2]
    rhs_batch = () if len(rhs_shape) == 1 else rhs_shape[:-2]
    batch_shape = np.broadcast_shapes(lhs_batch, rhs_batch)
    rows = 1 if len(lhs_shape) == 1 else lhs_shape[-2]
    inner_size = lhs_shape[-1]
    columns = 1 if len(rhs_shape) == 1 else rhs_shape[-1]
    output_shape = tuple(batch_shape)
    if len(lhs_shape) != 1:
        output_shape += (rows,)
    if len(rhs_shape) != 1:
        output_shape += (columns,)
    dtype = np.dtype(dtype)
    lhs_storage = arrays.operand_storage_bytes(cell.lhs, dtype)
    rhs_storage = arrays.operand_storage_bytes(cell.rhs, dtype)
    operand_peak = 0
    for name, operand, storage_bytes in (
            ('lhs', cell.lhs, lhs_storage),
            ('rhs', cell.rhs, rhs_storage)):
        if (resource_budget is not None
                and storage_bytes > resource_budget.single_allocation_bytes):
            raise MemoryError(
                f'{name} storage needs {storage_bytes} bytes, limit is '
                f'{resource_budget.single_allocation_bytes}')
        logical_bytes = arrays.operand_logical_bytes(operand, dtype)
        if (resource_budget is not None
                and logical_bytes > resource_budget.single_allocation_bytes):
            raise MemoryError(
                f'{name} logical size needs {logical_bytes} bytes, '
                f'limit is {resource_budget.single_allocation_bytes}')
        operand_peak += storage_bytes + max(storage_bytes, logical_bytes)
    output_elements = math.prod(output_shape)
    output_bytes = output_elements * dtype.itemsize
    if (resource_budget is not None
            and output_bytes > resource_budget.single_allocation_bytes):
        raise MemoryError(
            f'matmul output needs {output_bytes} bytes, limit is '
            f'{resource_budget.single_allocation_bytes}')
    scalar_contractions = (
        math.prod(batch_shape) * rows * inner_size * columns)
    operand_elements = math.prod(lhs_shape) + math.prod(rhs_shape)
    per_call_work = (
        scalar_contractions + output_elements + operand_elements)
    forced_winograd = 'winograd' in routes
    automatic_winograd = (
        len(lhs_shape) == len(rhs_shape) == 2
        and rows == inner_size == columns
        and rows % 2 == 0
        and cell.lhs.strides == (inner_size, 1)
        and cell.rhs.strides == (columns, 1)
        and rows >= 16_384)
    winograd_scratch = 0
    if forced_winograd or automatic_winograd:
        winograd_scratch = arrays.winograd_scratch_bytes(
            rows, inner_size, columns, dtype.itemsize)
    return (
        scalar_contractions, per_call_work, operand_peak, output_bytes,
        winograd_scratch)


def _calculate_estimate(plan, legacy_peak=None):
    calls_per_stream = (
        plan.mode.warmups
        + plan.mode.repetitions * plan.mode.panels)
    matmul_calls = 0
    preflight_calls = 0
    scalar_contractions = 0
    measurement_work = 0
    persistent_bytes = 0
    maximum_output_bytes = 0
    maximum_native_scratch_bytes = 0
    legacy_peak_bytes = 0
    actual_routes = set()
    for cell in plan.cells:
        routes = plan.routes if cell.routes is None else cell.routes
        actual_routes.update(routes)
        native_streams = 1 + len(routes)
        python_streams = native_streams + int(plan.numpy_baseline)
        cell_preflight_calls = 1 + native_streams
        calls_per_cell = (
            cell_preflight_calls
            + (native_streams + python_streams) * calls_per_stream)
        (contractions, per_call_work, operand_peak, output_bytes,
         native_scratch_bytes) = \
            _cell_metrics(cell, plan.dtype, routes)
        matmul_calls += calls_per_cell
        preflight_calls += cell_preflight_calls
        scalar_contractions += contractions * calls_per_cell
        measurement_work += per_call_work * calls_per_cell
        persistent_bytes += operand_peak
        maximum_output_bytes = max(maximum_output_bytes, output_bytes)
        maximum_native_scratch_bytes = max(
            maximum_native_scratch_bytes, native_scratch_bytes)
        legacy_peak_bytes += operand_peak + 3 * output_bytes
    if legacy_peak is None:
        legacy_peak = plan.schema_version == 1
    peak_bytes = (
        legacy_peak_bytes if legacy_peak
        else arrays.correctness_peak_bytes(
            persistent_bytes, maximum_output_bytes)
        + maximum_native_scratch_bytes)
    return CollectionEstimate(
        cell_count=len(plan.cells),
        route_count=len(actual_routes),
        panel_count=plan.mode.panels,
        preflight_calls=preflight_calls,
        matmul_calls=matmul_calls,
        scalar_contractions=scalar_contractions,
        measurement_work=measurement_work,
        peak_bytes=peak_bytes,
    )


def _validate_schedule_caps(estimate):
    if estimate.matmul_calls > MAX_COLLECTION_CALLS:
        raise MemoryError(
            f'collection needs {estimate.matmul_calls} matmul calls, '
            f'limit is {MAX_COLLECTION_CALLS}')


def _validate_resource_caps(plan, estimate, resource_budget):
    for cell in plan.cells:
        routes = plan.routes if cell.routes is None else cell.routes
        _cell_metrics(cell, plan.dtype, routes, resource_budget)
    if estimate.peak_bytes > resource_budget.peak_bytes:
        raise MemoryError(
            f'collection peak estimate needs {estimate.peak_bytes} bytes, '
            f'current worker-safe limit is {resource_budget.peak_bytes}')


def estimate_plan(plan):
    if not isinstance(plan, CollectionPlan):
        plan = CollectionPlan.from_dict(plan)
    estimate = _calculate_estimate(plan)
    safe_estimate = _calculate_estimate(plan, legacy_peak=False)
    _validate_schedule_caps(safe_estimate)
    _validate_artifact_cap(plan)
    return estimate


def _estimate_matches_plan(estimate, plan):
    expected = estimate_plan(plan)
    if estimate == expected:
        return True
    legacy = dataclasses.replace(
        expected, route_count=len(plan.routes))
    return estimate == legacy


def validate_plan_resources(plan, estimate=None, resource_budget=None):
    """Validate current-host memory without affecting plan portability."""

    if not isinstance(plan, CollectionPlan):
        plan = CollectionPlan.from_dict(plan)
    estimate = estimate or estimate_plan(plan)
    safe_estimate = _calculate_estimate(plan, legacy_peak=False)
    resource_budget = (
        resource_budget or arrays.resolve_resource_budget())
    _validate_resource_caps(plan, safe_estimate, resource_budget)
    return estimate


def requires_large_work_approval(plan, estimate=None):
    """Compatibility shim for version-4 plans; always return False."""

    if not isinstance(plan, CollectionPlan):
        plan = CollectionPlan.from_dict(plan)
    return False


def validate_execution_plan(plan):
    """Apply execution permissions without making estimation impure."""

    if not isinstance(plan, CollectionPlan):
        plan = CollectionPlan.from_dict(plan)
    estimate = estimate_plan(plan)
    validate_plan_resources(plan, estimate)
    if plan.target_duration is not None:
        _validate_target_startup_guard(plan)
    return estimate


def recommended_budget_warnings(estimate, plan=None):
    """Describe schedules that exceed the conservative starter budget."""

    if not isinstance(estimate, CollectionEstimate):
        estimate = CollectionEstimate.from_dict(estimate)
    warnings = []
    if estimate.cell_count > RECOMMENDED_COLLECTION_CELLS:
        warnings.append(
            f'{estimate.cell_count:,} cells exceed the recommended '
            f'{RECOMMENDED_COLLECTION_CELLS:,}')
    if estimate.matmul_calls >= RECOMMENDED_COLLECTION_CALLS:
        warnings.append(
            f'{estimate.matmul_calls:,} calls reach the recommended '
            f'{RECOMMENDED_COLLECTION_CALLS:,}')
    if plan is not None:
        artifact_bytes = estimate_artifact_bytes(plan)
        if artifact_bytes >= RECOMMENDED_COLLECTION_ARTIFACT_BYTES:
            warnings.append(
                f'{artifact_bytes:,} projected artifact bytes reach the '
                f'recommended '
                f'{RECOMMENDED_COLLECTION_ARTIFACT_BYTES:,}')
    return tuple(warnings)


def estimate_artifact_bytes(plan, panel_count=None):
    """Estimate formatted JSON size from the bounded panel structure."""

    if not isinstance(plan, CollectionPlan):
        plan = CollectionPlan.from_dict(plan)
    if panel_count is None:
        panel_count = plan.mode.panels
    _require_int(panel_count, 'panel_count', 0)
    total = 0
    for cell in plan.cells:
        route_count = len(
            plan.routes if cell.routes is None else cell.routes)
        samples_per_panel = (
            2 + 2 * route_count + int(plan.numpy_baseline))
        total += (
            ARTIFACT_FIXED_BYTES_PER_CELL
            + ARTIFACT_BYTES_PER_PANEL_SAMPLE
            * panel_count * samples_per_panel)
    return total * 5 // 4


def duration_shard_guard(plan):
    """Build the call guard and descriptive work totals for one shard."""

    if not isinstance(plan, CollectionPlan):
        plan = CollectionPlan.from_dict(plan)
    fixed_calls = 0
    fixed_work = 0
    work_per_repetition = 0
    for cell in plan.cells:
        routes = plan.routes if cell.routes is None else cell.routes
        native_streams = 1 + len(routes)
        timing_streams = (
            2 * native_streams + int(plan.numpy_baseline))
        preflight_calls = 1 + native_streams
        _contractions, per_call_work, _peak, _output, _scratch = \
            _cell_metrics(cell, plan.dtype, routes)
        fixed_calls += preflight_calls
        fixed_work += per_call_work * preflight_calls
        work_per_repetition += per_call_work * timing_streams
    return duration.ShardGuard(
        maximum_calls=MAX_COLLECTION_CALLS,
        maximum_work=None,
        fixed_calls=fixed_calls,
        fixed_work=fixed_work,
        work_per_balanced_repetition=work_per_repetition,
    )


def _duration_shard_guard_matches_plan(guard, plan):
    current = duration_shard_guard(plan)
    if guard == current:
        return True
    legacy = dataclasses.replace(
        current, maximum_work=LEGACY_MAX_COLLECTION_MEASUREMENT_WORK)
    return guard == legacy


def _validate_target_startup_guard(plan):
    guard = duration_shard_guard(plan)
    stream_count = 0
    for cell in plan.cells:
        routes = plan.routes if cell.routes is None else cell.routes
        native_streams = 1 + len(routes)
        stream_count += 2 * native_streams + int(plan.numpy_baseline)
    repetitions = plan.target_duration.warmups + 1
    startup_calls = guard.fixed_calls + stream_count * repetitions
    if startup_calls > guard.maximum_calls:
        raise MemoryError(
            f'target-duration startup needs {startup_calls} calls, limit is '
            f'{guard.maximum_calls}')


def _validate_artifact_cap(plan):
    artifact_bytes = estimate_artifact_bytes(plan)
    if artifact_bytes > MAX_COLLECTION_ARTIFACT_BYTES:
        raise MemoryError(
            f'collection artifact estimate needs {artifact_bytes} bytes, '
            f'limit is {MAX_COLLECTION_ARTIFACT_BYTES}')


def panel_cell_orders(plan):
    if not isinstance(plan, CollectionPlan):
        plan = CollectionPlan.from_dict(plan)
    indices = list(range(len(plan.cells)))
    random.Random(plan.seed).shuffle(indices)
    return schedule.balanced_orders(indices, plan.mode.panels)


def balanced_order_at(values, panel_index):
    """Return one deterministic globally balanced order without a prefix."""

    values = tuple(values)
    _require_int(panel_index, 'panel_index', 0)
    if not values:
        return ()
    if len(values) != len(set(values)):
        raise ValueError('balanced values must be unique')
    if len(values) == 1:
        return values
    indices = [0]
    for index in range(1, len(values)):
        offset = (index + 1) // 2
        indices.append(offset if index % 2 else -offset)
    offset = panel_index % len(values)
    order = tuple(
        values[(index + offset) % len(values)] for index in indices)
    if len(values) % 2 and panel_index // len(values) % 2:
        return tuple(reversed(order))
    return order


def panel_cell_order_at(plan, panel_index):
    if not isinstance(plan, CollectionPlan):
        plan = CollectionPlan.from_dict(plan)
    indices = list(range(len(plan.cells)))
    random.Random(plan.seed).shuffle(indices)
    return balanced_order_at(indices, panel_index)


def _matrix_strides(rows, columns, profile, side, padding):
    padded = profile in (f'{side}_padded', 'both_padded')
    column_major = profile in (
        f'{side}_column_major', 'both_column_major')
    if column_major:
        return 1, max(1, rows)
    return max(1, columns) + (padding if padded else 0), 1


def _matrix_span(rows, columns, strides):
    if rows == 0 or columns == 0:
        return 0
    return ((rows - 1) * strides[0]
            + (columns - 1) * strides[1] + 1)


def _matrix_operand(rows, columns, profile, side, padding):
    strides = _matrix_strides(
        rows, columns, profile, side, padding)
    return (rows, columns), strides, _matrix_span(
        rows, columns, strides)


def _profile_operands(rows, inner_size, columns, layout, padding,
                      broadcast, batch_size):
    lhs_shape, lhs_strides, lhs_span = _matrix_operand(
        rows, inner_size, layout, 'lhs', padding)
    rhs_shape, rhs_strides, rhs_span = _matrix_operand(
        inner_size, columns, layout, 'rhs', padding)
    if broadcast == 'matrix':
        return lhs_shape, lhs_strides, rhs_shape, rhs_strides
    if broadcast == 'matched_batch':
        return (
            (batch_size, *lhs_shape), (lhs_span, *lhs_strides),
            (batch_size, *rhs_shape), (rhs_span, *rhs_strides),
        )
    if broadcast == 'broadcast_lhs':
        return (
            (1, *lhs_shape), (0, *lhs_strides),
            (batch_size, *rhs_shape), (rhs_span, *rhs_strides),
        )
    if broadcast == 'broadcast_rhs':
        return (
            (batch_size, *lhs_shape), (lhs_span, *lhs_strides),
            (1, *rhs_shape), (0, *rhs_strides),
        )
    raise schema.SchemaError(
        f'unknown broadcast profile: {broadcast!r}')


def _matrix_routes(routes, rows, inner_size, columns, broadcast):
    return tuple(
        route for route in routes
        if route != 'winograd'
        or (broadcast == 'matrix'
            and rows % 2 == 0 and inner_size % 2 == 0
            and columns % 2 == 0)
    )


def default_shape_boundary_plan(
        m_values=DEFAULT_SHAPE_VALUES, k_values=(64,),
        n_values=DEFAULT_SHAPE_VALUES, dtype='float32', threads=1,
        layout='contiguous', layouts=None, padding=1, mode='preview',
        broadcast='matrix', broadcasts=None,
        batch_size=DEFAULT_BATCH_SIZE,
        routes=DEFAULT_ROUTES, numpy_baseline=True, seed=20260815,
        output_path=None, plan_id=None, target_duration=None,
        allow_large_work=False):
    """Build the configurable matrix grid used by the starter Atlas."""

    layouts = (layout,) if layouts is None else _string_tuple(
        layouts, 'layouts')
    broadcasts = (broadcast,) if broadcasts is None else _string_tuple(
        broadcasts, 'broadcasts')
    routes = _string_tuple(routes, 'routes')
    unknown_layouts = sorted(set(layouts) - set(LAYOUT_PROFILES))
    if unknown_layouts:
        raise schema.SchemaError(
            f'unknown layout profiles: {unknown_layouts}')
    unknown_broadcasts = sorted(
        set(broadcasts) - set(BROADCAST_PROFILES))
    if unknown_broadcasts:
        raise schema.SchemaError(
            f'unknown broadcast profiles: {unknown_broadcasts}')
    _require_int(padding, 'padding', 0)
    if any('padded' in item for item in layouts) and padding == 0:
        raise schema.SchemaError('a padded layout needs positive padding')
    _require_int(batch_size, 'batch_size', 2)
    values = []
    for name, raw_values in (
            ('m_values', m_values), ('k_values', k_values),
            ('n_values', n_values)):
        if not isinstance(raw_values, (list, tuple)) or not raw_values:
            raise schema.SchemaError(f'{name} must be a non-empty array')
        clean = tuple(
            _require_int(value, f'{name}[{index}]', 1)
            for index, value in enumerate(raw_values)
        )
        if len(clean) != len(set(clean)):
            raise schema.SchemaError(f'{name} must not contain duplicates')
        values.append(clean)
    m_values, k_values, n_values = values
    cell_count = (len(m_values) * len(k_values) * len(n_values)
                  * len(layouts) * len(broadcasts))
    if cell_count > MAX_COLLECTION_CELLS:
        raise schema.SchemaError(
            f'plan has {cell_count} cells, limit is '
            f'{MAX_COLLECTION_CELLS}')
    target_duration = (
        target_duration_from_dict(target_duration)
        if not isinstance(target_duration, duration.TargetDurationSpec)
        else target_duration)
    if target_duration is not None:
        mode = schema.ModeSpec.preset(target_duration.mode)
    else:
        mode = (schema.ModeSpec.from_dict(mode)
                if not isinstance(mode, schema.ModeSpec) else mode)
    cells = []
    for m_value in m_values:
        for k_value in k_values:
            for n_value in n_values:
                for layout_profile in layouts:
                    for broadcast_profile in broadcasts:
                        lhs_shape, lhs_strides, rhs_shape, rhs_strides = \
                            _profile_operands(
                                m_value, k_value, n_value,
                                layout_profile, padding,
                                broadcast_profile, batch_size)
                        cell_routes = _matrix_routes(
                            routes, m_value, k_value, n_value,
                            broadcast_profile)
                        if not cell_routes:
                            raise schema.SchemaError(
                                'no selected dispatch is eligible for '
                                f'M={m_value}, K={k_value}, N={n_value}, '
                                f'broadcast={broadcast_profile}')
                        lhs = schema.OperandSpec(
                            shape=lhs_shape, strides=lhs_strides)
                        rhs = schema.OperandSpec(
                            shape=rhs_shape, strides=rhs_strides)
                        cells.append(CollectionCell(
                            cell_id=(
                                f'm{m_value}-k{k_value}-n{n_value}-'
                                f'{layout_profile}-{broadcast_profile}'),
                            lhs=lhs,
                            rhs=rhs,
                            layout=layout_profile,
                            broadcast=broadcast_profile,
                            routes=cell_routes,
                        ))
    return CollectionPlan(
        cells=tuple(cells),
        dtype=dtype,
        mode=mode,
        target_duration=target_duration,
        allow_large_work=allow_large_work,
        routes=routes,
        numpy_baseline=numpy_baseline,
        seed=seed,
        threads=threads,
        output_path=output_path,
        plan_id=plan_id or uuid.uuid4().hex,
    )


def input_profile_plan(
        input_profiles, m_values=DEFAULT_SHAPE_VALUES, k_values=(64,),
        n_values=DEFAULT_SHAPE_VALUES, dtype='float32', threads=1,
        mode='preview', routes=DEFAULT_ROUTES, numpy_baseline=True,
        seed=20260815, output_path=None, plan_id=None,
        target_duration=None, allow_large_work=False):
    """Build an exact grid from independently composed operand profiles."""

    routes = _string_tuple(routes, 'routes')
    if not isinstance(input_profiles, (list, tuple)) or not input_profiles:
        raise schema.SchemaError(
            'input_profiles must be a non-empty array')
    if not all(isinstance(profile, profiles_module.InputProfile)
               for profile in input_profiles):
        raise schema.SchemaError(
            'profiles must contain InputProfile objects')
    sweep_count = sum(not profile.is_exact for profile in input_profiles)
    exact_count = len(input_profiles) - sweep_count
    cell_count = (
        len(m_values) * len(k_values) * len(n_values) * sweep_count
        + exact_count)
    if cell_count > MAX_COLLECTION_CELLS:
        raise schema.SchemaError(
            f'plan has {cell_count} cells, limit is '
            f'{MAX_COLLECTION_CELLS}')

    target_duration = (
        target_duration_from_dict(target_duration)
        if not isinstance(target_duration, duration.TargetDurationSpec)
        else target_duration)
    mode = (
        schema.ModeSpec.preset(target_duration.mode)
        if target_duration is not None else
        schema.ModeSpec.from_dict(mode)
        if not isinstance(mode, schema.ModeSpec) else mode)
    cells = []
    for expanded in profiles_module.expand_profiles(
            input_profiles, m_values, k_values, n_values):
        cell_routes = _matrix_routes(
            routes, expanded.m, expanded.k, expanded.n,
            expanded.broadcast)
        if not cell_routes:
            raise schema.SchemaError(
                'no selected dispatch is eligible for '
                f'M={expanded.m}, K={expanded.k}, N={expanded.n}, '
                f'profile={expanded.name!r}')
        cell_fields = expanded.collection_cell_kwargs()
        cell_fields['routes'] = cell_routes
        cells.append(CollectionCell(**cell_fields))

    return CollectionPlan(
        cells=tuple(cells),
        dtype=dtype,
        mode=mode,
        target_duration=target_duration,
        allow_large_work=allow_large_work,
        routes=routes,
        numpy_baseline=numpy_baseline,
        seed=seed,
        threads=threads,
        output_path=output_path,
        plan_id=plan_id or uuid.uuid4().hex,
    )


def _validate_observation_request(observation, request, index):
    expected_identity = {
        'id': request.request_id,
        'dtype': request.dtype,
        'lhs': request.lhs.to_dict(),
        'rhs': request.rhs.to_dict(),
    }
    for field, expected in expected_identity.items():
        if observation[field] != expected:
            raise schema.SchemaError(
                f'collection observation {index} has the wrong {field}')
    if observation['contraction'] != _request_contraction(request):
        raise schema.SchemaError(
            f'collection observation {index} has the wrong contraction')

    expected_routes = {'auto', *request.routes}
    if request.numpy_baseline:
        expected_routes.add('numpy')
    if set(observation['routes']) != expected_routes:
        raise schema.SchemaError(
            f'collection observation {index} routes do not match '
            'its request')
    auto_route = observation['auto_route']
    if auto_route in ('auto', 'numpy'):
        raise schema.SchemaError(
            f'collection observation {index} has an invalid auto route')
    expected_kinds = {
        'auto': 'solvcon_auto',
        **{route: 'solvcon_route' for route in request.routes},
    }
    if request.numpy_baseline:
        expected_kinds['numpy'] = 'numpy'
    for route, kind in expected_kinds.items():
        candidate = observation['routes'][route]
        if candidate['kind'] != kind:
            raise schema.SchemaError(
                f'collection observation {index} route {route!r} '
                'has the wrong kind')
        if kind == 'solvcon_route' and (
                candidate.get('selected_by_auto')
                != (route == auto_route)):
            raise schema.SchemaError(
                f'collection observation {index} route {route!r} '
                'has an inconsistent auto selection')


def _request_contraction(request):
    lhs_shape = request.lhs.shape
    rhs_shape = request.rhs.shape
    lhs_vector = len(lhs_shape) == 1
    rhs_vector = len(rhs_shape) == 1
    lhs_batch = () if lhs_vector else lhs_shape[:-2]
    rhs_batch = () if rhs_vector else rhs_shape[:-2]
    batch_shape = np.broadcast_shapes(lhs_batch, rhs_batch)
    rows = 1 if lhs_vector else lhs_shape[-2]
    inner_size = lhs_shape[-1]
    columns = 1 if rhs_vector else rhs_shape[-1]
    output_shape = list(batch_shape)
    if not lhs_vector:
        output_shape.append(rows)
    if not rhs_vector:
        output_shape.append(columns)
    if not output_shape:
        output_shape.append(1)
    return {
        'batch_shape': list(batch_shape),
        'batch_count': math.prod(batch_shape),
        'm': rows,
        'k': inner_size,
        'n': columns,
        'output_shape': output_shape,
        'lhs_vector': lhs_vector,
        'rhs_vector': rhs_vector,
    }


def validate_collection_provenance(document):
    plan = CollectionPlan.from_dict(document['plan'])
    if document['plan_sha256'] != plan.sha256():
        raise schema.SchemaError('collection plan hash does not match')
    estimate = CollectionEstimate.from_dict(document['estimate'])
    if not _estimate_matches_plan(estimate, plan):
        raise schema.SchemaError('collection estimate does not match plan')
    if document['artifact_count'] != len(plan.cells):
        raise schema.SchemaError('collection plan cell count is inconsistent')
    if len(document['observations']) != len(plan.cells):
        raise schema.SchemaError(
            'collection plan observation count is inconsistent')
    expected_orders = [
        [plan.cells[index].cell_id for index in order]
        for order in panel_cell_orders(plan)
    ]
    if document['cell_orders'] != expected_orders:
        raise schema.SchemaError(
            'collection cell orders do not match the seeded plan')
    requests = plan.requests()
    observations_by_source = {
        source['source_id']: [] for source in document['sources']
    }
    for item in document['observations']:
        observations_by_source[item['source_id']].append(item)
    panels_by_source = {
        source['source_id']: [] for source in document['sources']
    }
    for item in document['panels']:
        panels_by_source[item['source_id']].append(item['panel'])
    if any(len(items) != 1
           for items in observations_by_source.values()):
        raise schema.SchemaError(
            'each collection plan cell must have one observation')
    for index, (source, request) in enumerate(
            zip(document['sources'], requests)):
        if source['request'] != request.to_dict():
            raise schema.SchemaError(
                f'collection source {index} does not match its plan cell')
        wrapper = observations_by_source[source['source_id']][0]
        _validate_observation_request(
            wrapper['observation'], request, index)
        _validate_source_panels(
            panels_by_source[source['source_id']], request, index)


def _validate_source_panels(panels, request, source_index):
    by_scope = {
        (panel['index'], panel['scope']): panel for panel in panels
    }
    native_names = ('auto', *request.routes)
    python_names = native_names + (
        ('numpy',) if request.numpy_baseline else ())
    native_orders = schedule.balanced_orders(
        native_names, request.mode.panels)
    python_orders = schedule.balanced_orders(
        python_names, request.mode.panels)
    for panel_index in range(request.mode.panels):
        expected = {
            'native_batch': native_orders[panel_index],
            'python_end_to_end': python_orders[panel_index],
        }
        for scope, order in expected.items():
            panel = by_scope.get((panel_index, scope))
            if panel is None or panel['order'] != list(order):
                raise schema.SchemaError(
                    f'collection source {source_index} panels do not '
                    'match its request schedule')


def _percentile(values, percentile):
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] * (1 - fraction)
                 + values[upper] * fraction)


def _summarize(values):
    median = float(statistics.median(values))
    deviations = [abs(value - median) for value in values]
    return {
        'sample_count': len(values),
        'median_ns': median,
        'mad_ns': float(statistics.median(deviations)),
        'p95_ns': _percentile(values, 0.95),
        'minimum_ns': float(min(values)),
        'maximum_ns': float(max(values)),
    }


def _aggregate_observation(request, observations, panels):
    first = observations[0]
    identity_fields = (
        'dtype', 'lhs', 'rhs', 'contraction', 'auto_route')
    for observation in observations[1:]:
        if any(observation[field] != first[field]
               for field in identity_fields):
            raise schema.SchemaError(
                'duration shard observations have incompatible inputs')
        for route_name, route in observation['routes'].items():
            first_route = first['routes'].get(route_name)
            if first_route is None:
                raise schema.SchemaError(
                    'duration shard observations have incompatible routes')
            static_fields = (
                'name', 'kind', 'selected_route', 'selected_by_auto',
                'packing', 'correctness')
            if any(route.get(field) != first_route.get(field)
                   for field in static_fields):
                raise schema.SchemaError(
                    'duration shard observations have incompatible routes')

    native_samples = {}
    python_samples = {}
    for panel in panels:
        destination = (
            native_samples if panel['scope'] == duration.NATIVE_SCOPE
            else python_samples)
        for sample in panel['samples']:
            destination.setdefault(sample['route'], []).append(
                sample['latency_ns'])
    native_summaries = {
        name: _summarize(samples)
        for name, samples in native_samples.items()
    }
    python_summaries = {
        name: _summarize(samples)
        for name, samples in python_samples.items()
    }
    numpy_summary = python_summaries.get('numpy')
    numpy_median = (None if numpy_summary is None
                    else numpy_summary['median_ns'])
    routes = {}
    for name, first_route in first['routes'].items():
        route = dict(first_route)
        route['timing'] = native_summaries.get(name)
        route['python_timing'] = python_summaries.get(name)
        python_timing = route['python_timing']
        route['numpy_ratio'] = (
            python_timing['median_ns'] / numpy_median
            if python_timing is not None and numpy_median else None)
        routes[name] = route
    ranked = sorted(
        (route for route in routes.values()
         if route['kind'] == 'solvcon_route'
         and route['correctness']['correct']
         and route['timing'] is not None),
        key=lambda route: route['timing']['median_ns'],
    )
    winner = ranked[0]['name'] if ranked else None
    runner_up = ranked[1]['name'] if len(ranked) > 1 else None
    margin = None
    if runner_up is not None:
        winner_time = routes[winner]['timing']['median_ns']
        runner_time = routes[runner_up]['timing']['median_ns']
        margin = ((runner_time - winner_time) / winner_time
                  if winner_time else None)
    return {
        'id': request.request_id,
        'dtype': first['dtype'],
        'lhs': first['lhs'],
        'rhs': first['rhs'],
        'contraction': first['contraction'],
        'routes': routes,
        'auto_route': first['auto_route'],
        'winner': winner,
        'runner_up': runner_up,
        'winner_margin': margin,
    }


def aggregate_duration_observations(document):
    """Recompute one cumulative logical observation for every plan cell."""

    run = document['duration_run']
    plan = CollectionPlan.from_dict(run['template_plan'])
    observations = {
        item['source_id']: item['observation']
        for item in document['observations']
    }
    panels = {source['source_id']: [] for source in document['sources']}
    for item in document['panels']:
        panels[item['source_id']].append(item['panel'])
    result = []
    for cell_index, cell in enumerate(plan.cells):
        source_ids = [
            shard['source_ids'][cell_index]
            for shard in run['shards']
        ]
        source_observations = [observations[item] for item in source_ids]
        source_panels = [
            panel for source_id in source_ids for panel in panels[source_id]
        ]
        result.append({
            'cell_id': cell.cell_id,
            'source_ids': source_ids,
            'observation': _aggregate_observation(
                plan.request_at(cell_index), source_observations,
                source_panels),
        })
    return result


def metadata_identity(metadata):
    """Return the machine and executable identity required for resume."""

    build = metadata['build']
    return {
        'process': {
            key: value for key, value in metadata['process'].items()
            if key != 'pid'
        },
        'machine': metadata['machine'],
        'backend': metadata['backend'],
        'threading': metadata['threading'],
        'build': build,
    }


def _duration_dataclass(data, cls, name):
    data = _require_mapping(data, name)
    fields = tuple(field.name for field in dataclasses.fields(cls))
    _reject_unknown(data, name, fields)
    _require_fields(data, name, fields)
    try:
        return cls(**data)
    except (TypeError, duration.DurationModelError) as exc:
        raise schema.SchemaError(f'invalid {name}: {exc}') from exc


def duration_shard_sizes(schedule):
    sizes = [schedule.panels_per_full_shard] * \
        schedule.full_shard_count
    if schedule.final_shard_panels:
        sizes.append(schedule.final_shard_panels)
    return sizes


def cell_orders_sha256(plan, panel_offset, panel_count):
    orders = [
        [plan.cells[index].cell_id for index in
         panel_cell_order_at(plan, panel_offset + local_index)]
        for local_index in range(panel_count)
    ]
    payload = json.dumps(
        orders, sort_keys=True, separators=(',', ':'),
        allow_nan=False)
    return hashlib.sha256(payload.encode('utf8')).hexdigest()


def _validate_duration_source_panels(panels, request, panel_offset,
                                     source_index):
    by_scope = {
        (panel['index'], panel['scope']): panel for panel in panels
    }
    native_names = ('auto', *request.routes)
    python_names = native_names + (
        ('numpy',) if request.numpy_baseline else ())
    for local_index in range(request.mode.panels):
        global_index = panel_offset + local_index
        expected = {
            duration.NATIVE_SCOPE:
                balanced_order_at(native_names, global_index),
            duration.PYTHON_SCOPE:
                balanced_order_at(python_names, global_index),
        }
        for scope, order in expected.items():
            panel = by_scope.get((local_index, scope))
            if panel is None or panel['order'] != list(order):
                raise schema.SchemaError(
                    f'duration source {source_index} panels do not '
                    'match the global balanced schedule')


def validate_duration_collection(document):
    """Validate target-duration provenance and cumulative observations."""

    run = _require_mapping(document['duration_run'], 'duration_run')
    fields = (
        'version', 'run_id', 'status', 'started_at', 'requested',
        'template_plan', 'template_plan_sha256', 'metadata_identity',
        'preflight_elapsed_ns', 'calibration_measurements',
        'controller_overheads', 'shard_guard', 'schedule',
        'completed_panels', 'measurement_elapsed_ns',
        'actual_elapsed_ns', 'resumed_from_checkpoint', 'resume_count',
        'shards',
    )
    _reject_unknown(run, 'duration_run', fields)
    _require_fields(run, 'duration_run', fields)
    if run['version'] != 1:
        raise schema.SchemaError('unsupported duration_run version')
    _require_str(run['run_id'], 'duration_run.run_id')
    if run['run_id'] != document['collection_id']:
        raise schema.SchemaError(
            'duration_run.run_id must match collection_id')
    if run['status'] not in ('checkpoint', 'complete'):
        raise schema.SchemaError('duration_run.status is unsupported')
    _require_str(run['started_at'], 'duration_run.started_at')
    plan = CollectionPlan.from_dict(run['template_plan'])
    if plan.target_duration is None:
        raise schema.SchemaError(
            'duration_run template plan lacks target_duration')
    if run['template_plan_sha256'] != plan.sha256():
        raise schema.SchemaError(
            'duration_run template plan hash does not match')
    requested = target_duration_from_dict(run['requested'])
    if requested != plan.target_duration:
        raise schema.SchemaError(
            'duration_run requested target does not match its plan')
    identity = _require_mapping(
        run['metadata_identity'], 'duration_run.metadata_identity')
    measurements = tuple(
        _duration_dataclass(
            item, duration.CalibrationMeasurement,
            f'duration_run.calibration_measurements[{index}]')
        for index, item in enumerate(run['calibration_measurements'])
    )
    overheads = tuple(
        _duration_dataclass(
            item, duration.ControllerOverhead,
            f'duration_run.controller_overheads[{index}]')
        for index, item in enumerate(run['controller_overheads'])
    )
    guard = _duration_dataclass(
        run['shard_guard'], duration.ShardGuard,
        'duration_run.shard_guard')
    if not _duration_shard_guard_matches_plan(guard, plan):
        raise schema.SchemaError(
            'duration_run shard guard does not match its plan')
    expected_keys = set()
    for cell, request in zip(plan.cells, plan.requests()):
        native_names = ('auto', *request.routes)
        python_names = native_names + (
            ('numpy',) if request.numpy_baseline else ())
        expected_keys.update(
            (cell.cell_id, route, duration.NATIVE_SCOPE)
            for route in native_names)
        expected_keys.update(
            (cell.cell_id, route, duration.PYTHON_SCOPE)
            for route in python_names)
    grouped_measurements = {}
    for item in measurements:
        key = item.cell_id, item.route, item.scope
        grouped_measurements.setdefault(key, []).append(item)
    if set(grouped_measurements) != expected_keys:
        raise schema.SchemaError(
            'duration calibration streams do not match the template plan')
    pass_counts = {len(items) for items in grouped_measurements.values()}
    if not pass_counts <= {1, 2} or len(pass_counts) != 1:
        raise schema.SchemaError(
            'duration calibration passes are inconsistent')
    calibration_repetitions = set()
    for items in grouped_measurements.values():
        if items[0].repetitions != 1:
            raise schema.SchemaError(
                'duration pilot calibration has the wrong repetitions')
        if len(items) == 2:
            calibration_repetitions.add(items[1].repetitions)
    if calibration_repetitions and len(calibration_repetitions) != 1:
        raise schema.SchemaError(
            'duration timed calibration repetitions are inconsistent')
    if calibration_repetitions:
        timed_repetitions = next(iter(calibration_repetitions))
        if not (requested.minimum_calibration_repetitions
                <= timed_repetitions
                <= requested.maximum_calibration_repetitions):
            raise schema.SchemaError(
                'duration timed calibration repetitions are out of range')
    if (len(overheads) != next(iter(pass_counts))
            or any(item.panels != 1 for item in overheads)):
        raise schema.SchemaError(
            'duration controller overheads do not match calibration passes')
    preflight_elapsed_ns = _require_int(
        run['preflight_elapsed_ns'],
        'duration_run.preflight_elapsed_ns', 0)
    expected_schedule = duration.plan_target_duration(
        requested, measurements, overheads,
        preflight_elapsed_ns=preflight_elapsed_ns,
        shard_guard=guard)
    if not expected_schedule.feasible:
        raise schema.SchemaError(
            'duration_run records an infeasible schedule')
    if run['schedule'] != dataclasses.asdict(expected_schedule):
        raise schema.SchemaError('duration_run schedule does not match')
    expected_sizes = duration_shard_sizes(expected_schedule)
    shards = run['shards']
    if not isinstance(shards, list) or not shards:
        raise schema.SchemaError('duration_run.shards must be an array')
    if len(shards) > len(expected_sizes):
        raise schema.SchemaError('duration_run has too many shards')
    if run['status'] == 'checkpoint' \
            and len(shards) >= len(expected_sizes):
        raise schema.SchemaError(
            'checkpoint duration_run must have unfinished shards')
    if run['status'] == 'complete' and len(shards) != len(expected_sizes):
        raise schema.SchemaError(
            'complete duration_run is missing shards')
    completed_panels = _require_int(
        run['completed_panels'], 'duration_run.completed_panels', 1)
    for field in ('measurement_elapsed_ns', 'actual_elapsed_ns'):
        _require_int(run[field], f'duration_run.{field}', 0)
    if not isinstance(run['resumed_from_checkpoint'], bool):
        raise schema.SchemaError(
            'duration_run.resumed_from_checkpoint must be a boolean')
    _require_int(run['resume_count'], 'duration_run.resume_count', 0)
    if run['resumed_from_checkpoint'] != (run['resume_count'] > 0):
        raise schema.SchemaError(
            'duration_run resume fields are inconsistent')

    sources = {
        source['source_id']: source for source in document['sources']
    }
    panels_by_source = {source_id: [] for source_id in sources}
    for item in document['panels']:
        panels_by_source[item['source_id']].append(item['panel'])
    observations_by_source = {source_id: [] for source_id in sources}
    for item in document['observations']:
        observations_by_source[item['source_id']].append(
            item['observation'])
    source_ids = []
    panel_offset = 0
    shard_fields = (
        'id', 'index', 'panel_offset', 'panel_count', 'started_at',
        'completed_at', 'elapsed_ns', 'mode', 'plan_id', 'plan_sha256',
        'estimate', 'cell_orders_sha256', 'source_ids',
    )
    shard_ids = set()
    for shard_index, shard in enumerate(shards):
        name = f'duration_run.shards[{shard_index}]'
        shard = _require_mapping(shard, name)
        _reject_unknown(shard, name, shard_fields)
        _require_fields(shard, name, shard_fields)
        shard_id = _require_str(shard['id'], f'{name}.id')
        if shard_id in shard_ids:
            raise schema.SchemaError(
                'duration_run shard IDs must be unique')
        shard_ids.add(shard_id)
        if shard_id != f'{run["run_id"]}:shard-{shard_index:04d}':
            raise schema.SchemaError(
                'duration_run shard ID does not match its index')
        index = _require_int(shard['index'], f'{name}.index', 0)
        offset = _require_int(
            shard['panel_offset'], f'{name}.panel_offset', 0)
        panel_count = _require_int(
            shard['panel_count'], f'{name}.panel_count', 1)
        if index != shard_index:
            raise schema.SchemaError(
                'duration_run shard indexes must be contiguous')
        if offset != panel_offset:
            raise schema.SchemaError(
                'duration_run shard panel offsets must be contiguous')
        if panel_count != expected_sizes[shard_index]:
            raise schema.SchemaError(
                'duration_run shard panel count does not match schedule')
        _require_str(shard['started_at'], f'{name}.started_at')
        _require_str(shard['completed_at'], f'{name}.completed_at')
        _require_int(shard['elapsed_ns'], f'{name}.elapsed_ns', 0)
        expected_mode = schema.ModeSpec(
            name=requested.mode,
            warmups=requested.warmups,
            repetitions=expected_schedule.repetitions,
            panels=panel_count,
        )
        shard_mode = schema.ModeSpec.from_dict(shard['mode'])
        if shard_mode != expected_mode:
            raise schema.SchemaError(
                'duration shard plan has the wrong mode')
        plan_id = _require_str(
            shard['plan_id'], f'{name}.plan_id')
        if plan_id != f'{run["run_id"]}:shard-{shard_index:04d}':
            raise schema.SchemaError(
                'duration shard plan ID does not match its index')
        shard_plan = dataclasses.replace(
            plan, mode=shard_mode, target_duration=None,
            output_path=None,
            plan_id=plan_id)
        if shard['plan_sha256'] != shard_plan.sha256():
            raise schema.SchemaError(
                'duration shard plan hash does not match')
        shard_estimate = CollectionEstimate.from_dict(shard['estimate'])
        if not _estimate_matches_plan(shard_estimate, shard_plan):
            raise schema.SchemaError(
                'duration shard estimate does not match')
        if shard['cell_orders_sha256'] != cell_orders_sha256(
                plan, panel_offset, panel_count):
            raise schema.SchemaError(
                'duration shard cell order digest does not match')
        shard_source_ids = shard['source_ids']
        if not isinstance(shard_source_ids, list) \
                or len(shard_source_ids) != len(plan.cells):
            raise schema.SchemaError(
                'duration shard source_ids do not match its cells')
        if any(source_id not in sources
               for source_id in shard_source_ids):
            raise schema.SchemaError(
                'duration shard references an unknown source')
        for cell_index, (source_id, request) in enumerate(
                zip(shard_source_ids, shard_plan.requests())):
            source = sources[source_id]
            if source['request'] != request.to_dict():
                raise schema.SchemaError(
                    'duration source does not match its shard plan')
            if metadata_identity(source['metadata']) != identity:
                raise schema.SchemaError(
                    'duration source metadata is incompatible')
            _validate_duration_source_panels(
                panels_by_source[source_id], request, panel_offset,
                cell_index)
            source_observations = observations_by_source[source_id]
            if len(source_observations) != 1:
                raise schema.SchemaError(
                    'each duration source must have one observation')
            _validate_observation_request(
                source_observations[0], request, cell_index)
            if any(
                    not route['correctness']['correct']
                    for route in source_observations[0][
                        'routes'].values()):
                raise schema.SchemaError(
                    'duration source contains an incorrect route')
        source_ids.extend(shard_source_ids)
        panel_offset += panel_count
    if len(source_ids) != len(set(source_ids)):
        raise schema.SchemaError(
            'duration shard source IDs must be unique')
    if set(source_ids) != set(sources):
        raise schema.SchemaError(
            'duration collection has unclaimed sources')
    if completed_panels != panel_offset:
        raise schema.SchemaError(
            'duration_run completed panel count does not match shards')
    if document['artifact_count'] != len(plan.cells) * len(shards):
        raise schema.SchemaError(
            'duration collection source count does not match shards')
    measurement_elapsed_ns = sum(
        item['elapsed_ns'] for item in shards)
    if run['measurement_elapsed_ns'] != measurement_elapsed_ns:
        raise schema.SchemaError(
            'duration_run measurement elapsed time does not match shards')
    calibration_elapsed_ns = sum(
        item.elapsed_ns for item in measurements)
    calibration_elapsed_ns += sum(
        item.elapsed_ns for item in overheads)
    minimum_actual_ns = (
        preflight_elapsed_ns + calibration_elapsed_ns
        + measurement_elapsed_ns)
    if run['actual_elapsed_ns'] < minimum_actual_ns:
        raise schema.SchemaError(
            'duration_run actual elapsed time is inconsistent')
    expected_aggregates = aggregate_duration_observations(document)
    if document['aggregate_observations'] != expected_aggregates:
        raise schema.SchemaError(
            'duration aggregate observations do not match raw samples')


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
