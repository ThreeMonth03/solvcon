# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Versioned data contracts for matmul benchmark collection."""

import dataclasses
import hashlib
import json
import math
import os
import re
import uuid

import numpy as np


SCHEMA_VERSION = 1
ARTIFACT_KIND = 'solvcon.matmul_benchmark'
COLLECTION_KIND = 'solvcon.matmul_benchmark_collection'
SUPPORTED_DTYPES = (
    'float32',
    'float64',
    'complex64',
    'complex128',
)
MAX_OPERAND_RANK = 8
MAX_MODE_WARMUPS = 1000
MAX_MODE_REPETITIONS = 10000
MAX_MODE_PANELS = 1000
MAX_MODE_CALLS_PER_ROUTE = 10000


class SchemaError(ValueError):
    """Report an invalid request or artifact contract."""


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise SchemaError(f'{name} must be an object')
    return value


def _reject_unknown(data, name, allowed):
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise SchemaError(f'{name} has unknown fields: {unknown}')


def _require_fields(data, name, required):
    missing = sorted(set(required) - set(data))
    if missing:
        raise SchemaError(f'{name} is missing fields: {missing}')


def _require_int(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f'{name} must be an integer')
    if minimum is not None and value < minimum:
        raise SchemaError(f'{name} must be at least {minimum}')
    return value


def _require_str(value, name):
    if not isinstance(value, str) or not value:
        raise SchemaError(f'{name} must be a non-empty string')
    return value


def _integer_tuple(value, name, minimum=None):
    if not isinstance(value, (list, tuple)):
        raise SchemaError(f'{name} must be an array')
    return tuple(
        _require_int(item, f'{name}[{index}]', minimum)
        for index, item in enumerate(value)
    )


@dataclasses.dataclass(frozen=True)
class OperandSpec:
    """Describe one ndarray by shape and element strides."""

    shape: tuple
    strides: tuple

    def __post_init__(self):
        shape = _integer_tuple(self.shape, 'shape', 0)
        strides = _integer_tuple(self.strides, 'strides')
        if not shape:
            raise SchemaError('an operand must have at least one axis')
        if len(shape) > MAX_OPERAND_RANK:
            raise SchemaError(
                f'an operand may have at most {MAX_OPERAND_RANK} axes')
        if len(shape) != len(strides):
            raise SchemaError('shape and strides must have the same length')
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'strides', strides)

    @classmethod
    def from_dict(cls, data):
        data = _require_mapping(data, 'operand')
        _reject_unknown(data, 'operand', ('shape', 'strides'))
        return cls(shape=data.get('shape'), strides=data.get('strides'))

    def to_dict(self):
        return {
            'shape': list(self.shape),
            'strides': list(self.strides),
        }


@dataclasses.dataclass(frozen=True)
class ModeSpec:
    """Control the duration and panel count of one benchmark."""

    name: str = 'preview'
    warmups: int = 2
    repetitions: int = 5
    panels: int = 2

    def __post_init__(self):
        if self.name not in ('preview', 'stable'):
            raise SchemaError("mode name must be 'preview' or 'stable'")
        _require_int(self.warmups, 'mode.warmups', 0)
        _require_int(self.repetitions, 'mode.repetitions', 1)
        _require_int(self.panels, 'mode.panels', 1)
        if self.warmups > MAX_MODE_WARMUPS:
            raise SchemaError('mode.warmups exceeds the supported limit')
        if self.repetitions > MAX_MODE_REPETITIONS:
            raise SchemaError(
                'mode.repetitions exceeds the supported limit')
        if self.panels > MAX_MODE_PANELS:
            raise SchemaError('mode.panels exceeds the supported limit')
        calls = self.warmups + self.repetitions * self.panels
        if calls > MAX_MODE_CALLS_PER_ROUTE:
            raise SchemaError(
                'mode requests too many calls per measured route')

    @classmethod
    def preset(cls, name):
        if name == 'preview':
            return cls(name='preview', warmups=2, repetitions=5, panels=2)
        if name == 'stable':
            return cls(name='stable', warmups=4,
                       repetitions=20, panels=8)
        raise SchemaError(f'unknown benchmark mode: {name!r}')

    @classmethod
    def from_dict(cls, data):
        if isinstance(data, str):
            return cls.preset(data)
        data = _require_mapping(data, 'mode')
        _reject_unknown(
            data, 'mode', ('name', 'warmups', 'repetitions', 'panels'))
        preset = cls.preset(data.get('name', 'preview'))
        return cls(
            name=data.get('name', preset.name),
            warmups=data.get('warmups', preset.warmups),
            repetitions=data.get('repetitions', preset.repetitions),
            panels=data.get('panels', preset.panels),
        )

    def to_dict(self):
        return {
            'name': self.name,
            'warmups': self.warmups,
            'repetitions': self.repetitions,
            'panels': self.panels,
        }


@dataclasses.dataclass(frozen=True)
class BenchmarkRequest:
    """Describe one reproducible route benchmark job."""

    lhs: OperandSpec
    rhs: OperandSpec
    dtype: str = 'float64'
    mode: ModeSpec = dataclasses.field(default_factory=ModeSpec)
    routes: tuple | None = None
    numpy_baseline: bool = True
    seed: int = 20260815
    threads: int | None = None
    output_path: str | None = None
    request_id: str = dataclasses.field(
        default_factory=lambda: uuid.uuid4().hex)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f'unsupported schema version: {self.schema_version}')
        if self.dtype not in SUPPORTED_DTYPES:
            raise SchemaError(f'unsupported dtype: {self.dtype!r}')
        if not isinstance(self.lhs, OperandSpec):
            raise SchemaError('lhs must be an OperandSpec')
        if not isinstance(self.rhs, OperandSpec):
            raise SchemaError('rhs must be an OperandSpec')
        if not isinstance(self.mode, ModeSpec):
            raise SchemaError('mode must be a ModeSpec')
        if self.routes is not None:
            if not isinstance(self.routes, (list, tuple)):
                raise SchemaError('routes must be an array or null')
            routes = tuple(_require_str(item, 'route')
                           for item in self.routes)
            if len(routes) != len(set(routes)):
                raise SchemaError('routes must not contain duplicates')
            object.__setattr__(self, 'routes', routes)
        if not isinstance(self.numpy_baseline, bool):
            raise SchemaError('numpy_baseline must be a boolean')
        _require_int(self.seed, 'seed', 0)
        if self.threads is not None:
            _require_int(self.threads, 'threads', 1)
        if self.output_path is not None:
            path = os.fspath(self.output_path)
            if not path:
                raise SchemaError('output_path must not be empty')
            object.__setattr__(self, 'output_path', path)
        _require_str(self.request_id, 'request_id')
        self._validate_matmul_shapes()

    def _validate_matmul_shapes(self):
        lhs = self.lhs.shape
        rhs = self.rhs.shape
        contraction = rhs[-1] if len(rhs) == 1 else rhs[-2]
        if lhs[-1] != contraction:
            raise SchemaError('matmul contraction dimensions do not match')
        lhs_batch = () if len(lhs) == 1 else lhs[:-2]
        rhs_batch = () if len(rhs) == 1 else rhs[:-2]
        try:
            np.broadcast_shapes(lhs_batch, rhs_batch)
        except ValueError as exc:
            raise SchemaError('matmul batch dimensions do not match') \
                from exc

    @classmethod
    def from_dict(cls, data):
        data = _require_mapping(data, 'request')
        _reject_unknown(data, 'request', (
            'schema_version', 'id', 'lhs', 'rhs', 'dtype', 'mode',
            'routes', 'numpy_baseline', 'seed', 'threads', 'output_path',
        ))
        return cls(
            schema_version=data.get('schema_version', SCHEMA_VERSION),
            request_id=data.get('id', uuid.uuid4().hex),
            lhs=OperandSpec.from_dict(data.get('lhs')),
            rhs=OperandSpec.from_dict(data.get('rhs')),
            dtype=data.get('dtype', 'float64'),
            mode=ModeSpec.from_dict(data.get('mode', 'preview')),
            routes=data.get('routes'),
            numpy_baseline=data.get('numpy_baseline', True),
            seed=data.get('seed', 20260815),
            threads=data.get('threads'),
            output_path=data.get('output_path'),
        )

    def to_dict(self):
        return {
            'schema_version': self.schema_version,
            'id': self.request_id,
            'lhs': self.lhs.to_dict(),
            'rhs': self.rhs.to_dict(),
            'dtype': self.dtype,
            'mode': self.mode.to_dict(),
            'routes': (None if self.routes is None
                       else list(self.routes)),
            'numpy_baseline': self.numpy_baseline,
            'seed': self.seed,
            'threads': self.threads,
            'output_path': self.output_path,
        }


def _require_bool(value, name):
    if not isinstance(value, bool):
        raise SchemaError(f'{name} must be a boolean')
    return value


def _require_number(value, name, minimum=None, optional=False):
    if value is None and optional:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f'{name} must be a number')
    if not np.isfinite(value):
        raise SchemaError(f'{name} must be finite')
    if minimum is not None and value < minimum:
        raise SchemaError(f'{name} must be at least {minimum}')
    return value


def _require_optional_str(value, name):
    if value is not None:
        _require_str(value, name)


def _validate_serialized_request(data):
    request = BenchmarkRequest.from_dict(data)
    _require_fields(data, 'artifact request', request.to_dict())
    return request


def _validate_packing(data, name):
    if data is None:
        return
    data = _require_mapping(data, name)
    fields = ('eager_lhs', 'eager_rhs', 'scratch_lhs', 'scratch_rhs')
    _reject_unknown(data, name, fields)
    _require_fields(data, name, fields)
    for field_name in fields:
        _require_bool(data[field_name], f'{name}.{field_name}')


def _validate_correctness(data, name):
    data = _require_mapping(data, name)
    allowed = (
        'correct', 'reason', 'rtol', 'atol', 'max_absolute_error',
        'max_relative_error', 'nonfinite_result',
    )
    _reject_unknown(data, name, allowed)
    _require_fields(data, name, (
        'correct', 'reason', 'max_absolute_error', 'max_relative_error',
        'nonfinite_result',
    ))
    _require_bool(data['correct'], f'{name}.correct')
    _require_optional_str(data['reason'], f'{name}.reason')
    for field_name in ('rtol', 'atol'):
        if field_name in data:
            _require_number(data[field_name], f'{name}.{field_name}', 0)
    for field_name in ('max_absolute_error', 'max_relative_error'):
        _require_number(
            data[field_name], f'{name}.{field_name}', 0, optional=True)
    _require_bool(data['nonfinite_result'], f'{name}.nonfinite_result')


def _validate_summary(data, name):
    data = _require_mapping(data, name)
    fields = (
        'sample_count', 'median_ns', 'mad_ns', 'p95_ns', 'minimum_ns',
        'maximum_ns',
    )
    _reject_unknown(data, name, fields)
    _require_fields(data, name, fields)
    _require_int(data['sample_count'], f'{name}.sample_count', 1)
    for field_name in fields[1:]:
        _require_number(data[field_name], f'{name}.{field_name}', 0)


def _validate_candidate(data, name, with_timing=False):
    data = _require_mapping(data, name)
    allowed = {
        'name', 'kind', 'selected_route', 'selected_by_auto', 'packing',
        'correctness',
    }
    if with_timing:
        allowed.update(('timing', 'numpy_ratio'))
    _reject_unknown(data, name, allowed)
    _require_fields(data, name, ('name', 'kind', 'packing', 'correctness'))
    _require_str(data['name'], f'{name}.name')
    _require_str(data['kind'], f'{name}.kind')
    if data['kind'] not in ('solvcon_auto', 'solvcon_route', 'numpy'):
        raise SchemaError(f'{name}.kind is unsupported')
    if 'selected_route' in data:
        _require_optional_str(data['selected_route'],
                              f'{name}.selected_route')
    if 'selected_by_auto' in data:
        _require_bool(data['selected_by_auto'],
                      f'{name}.selected_by_auto')
    _validate_packing(data['packing'], f'{name}.packing')
    _validate_correctness(data['correctness'], f'{name}.correctness')
    if with_timing:
        _require_fields(data, name, ('timing', 'numpy_ratio'))
        if data.get('timing') is not None:
            _validate_summary(data['timing'], f'{name}.timing')
        _require_number(data.get('numpy_ratio'), f'{name}.numpy_ratio',
                        0, optional=True)


def _validate_sample(data, name):
    data = _require_mapping(data, name)
    fields = ('route', 'repetitions', 'elapsed_ns', 'latency_ns')
    _reject_unknown(data, name, fields)
    _require_fields(data, name, fields)
    _require_str(data['route'], f'{name}.route')
    _require_int(data['repetitions'], f'{name}.repetitions', 1)
    _require_int(data['elapsed_ns'], f'{name}.elapsed_ns', 0)
    _require_number(data['latency_ns'], f'{name}.latency_ns', 0)
    expected_latency = data['elapsed_ns'] / data['repetitions']
    if not math.isclose(
            data['latency_ns'], expected_latency,
            rel_tol=1e-12, abs_tol=1e-9):
        raise SchemaError(
            f'{name}.latency_ns does not match elapsed_ns/repetitions')


def _validate_panel(data, name):
    data = _require_mapping(data, name)
    fields = ('index', 'order', 'samples')
    _reject_unknown(data, name, fields)
    _require_fields(data, name, fields)
    _require_int(data['index'], f'{name}.index', 0)
    if not isinstance(data['order'], list):
        raise SchemaError(f'{name}.order must be an array')
    for index, route in enumerate(data['order']):
        _require_str(route, f'{name}.order[{index}]')
    if not isinstance(data['samples'], list):
        raise SchemaError(f'{name}.samples must be an array')
    for index, sample in enumerate(data['samples']):
        _validate_sample(sample, f'{name}.samples[{index}]')
    if [sample['route'] for sample in data['samples']] != data['order']:
        raise SchemaError(f'{name}.order does not match its samples')


def panels_sha256(panels):
    """Return a stable digest for one source artifact's raw panels."""
    payload = json.dumps(
        panels, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf8')).hexdigest()


def _validate_contraction(data, name):
    data = _require_mapping(data, name)
    fields = (
        'batch_shape', 'batch_count', 'm', 'k', 'n', 'output_shape',
        'lhs_vector', 'rhs_vector',
    )
    _reject_unknown(data, name, fields)
    _require_fields(data, name, fields)
    _integer_tuple(data['batch_shape'], f'{name}.batch_shape', 0)
    _integer_tuple(data['output_shape'], f'{name}.output_shape', 0)
    for field_name in ('batch_count', 'm', 'k', 'n'):
        _require_int(data[field_name], f'{name}.{field_name}', 0)
    _require_bool(data['lhs_vector'], f'{name}.lhs_vector')
    _require_bool(data['rhs_vector'], f'{name}.rhs_vector')


def _validate_observation(data, name):
    data = _require_mapping(data, name)
    fields = (
        'id', 'dtype', 'lhs', 'rhs', 'contraction', 'routes',
        'auto_route', 'winner', 'runner_up', 'winner_margin',
    )
    _reject_unknown(data, name, fields)
    _require_fields(data, name, fields)
    _require_str(data['id'], f'{name}.id')
    if data['dtype'] not in SUPPORTED_DTYPES:
        raise SchemaError(f'{name}.dtype is unsupported')
    OperandSpec.from_dict(data['lhs'])
    OperandSpec.from_dict(data['rhs'])
    _validate_contraction(data['contraction'], f'{name}.contraction')
    routes = _require_mapping(data['routes'], f'{name}.routes')
    for route_name, route in routes.items():
        _require_str(route_name, f'{name}.route name')
        _validate_candidate(route, f'{name}.routes[{route_name!r}]', True)
        if route['name'] != route_name:
            raise SchemaError(
                f'{name}.routes[{route_name!r}].name does not match its key')
    for field_name in ('auto_route', 'winner', 'runner_up'):
        _require_optional_str(data[field_name], f'{name}.{field_name}')
    auto = routes.get('auto')
    if data['auto_route'] is not None and (
            auto is None
            or auto['kind'] != 'solvcon_auto'
            or auto.get('selected_route') != data['auto_route']):
        raise SchemaError(f'{name}.auto_route does not match auto selection')
    for field_name in ('winner', 'runner_up'):
        route_name = data[field_name]
        if route_name is None:
            continue
        route = routes.get(route_name)
        if (route is None
                or route['kind'] != 'solvcon_route'
                or not route['correctness']['correct']
                or route['timing'] is None):
            raise SchemaError(
                f'{name}.{field_name} does not name a timed correct route')
    _require_number(data['winner_margin'], f'{name}.winner_margin',
                    0, optional=True)


def _validate_metadata(data, name):
    data = _require_mapping(data, name)
    fields = ('process', 'machine', 'build', 'backend', 'threading')
    _reject_unknown(data, name, fields)
    _require_fields(data, name, fields)
    for field_name in fields:
        _require_mapping(data[field_name], f'{name}.{field_name}')
    process = data['process']
    process_fields = ('pid', 'executable', 'python', 'affinity')
    _reject_unknown(process, f'{name}.process', process_fields)
    _require_fields(process, f'{name}.process', process_fields)
    _require_int(process['pid'], f'{name}.process.pid', 1)
    _require_str(process['executable'], f'{name}.process.executable')
    _require_str(process['python'], f'{name}.process.python')
    affinity = process['affinity']
    if affinity is not None:
        _integer_tuple(affinity, f'{name}.process.affinity', 0)
    machine = data['machine']
    machine_fields = (
        'node', 'system', 'release', 'machine', 'processor',
        'logical_cpu_count',
    )
    _reject_unknown(machine, f'{name}.machine', machine_fields)
    _require_fields(machine, f'{name}.machine', machine_fields)
    for field_name in ('node', 'system', 'release', 'machine', 'processor'):
        if field_name not in machine or not isinstance(
                machine[field_name], str):
            raise SchemaError(f'{name}.machine.{field_name} must be a string')
    if machine['logical_cpu_count'] is not None:
        _require_int(machine['logical_cpu_count'],
                     f'{name}.machine.logical_cpu_count', 1)
    build = data['build']
    build_fields = (
        'git_commit', 'git_dirty', 'dirty_diff_sha256',
        'dirty_source_complete',
        'solvcon_extension', 'extension_mtime_ns', 'extension_sha256',
        'native_loader', 'solvcon_profile',
    )
    _reject_unknown(build, f'{name}.build', build_fields)
    _require_fields(build, f'{name}.build', build_fields)
    for field_name in (
            'git_commit', 'dirty_diff_sha256', 'solvcon_extension',
            'extension_sha256', 'solvcon_profile'):
        _require_optional_str(build[field_name],
                              f'{name}.build.{field_name}')
    if build['git_dirty'] is not None:
        _require_bool(build['git_dirty'], f'{name}.build.git_dirty')
    _require_bool(build['dirty_source_complete'],
                  f'{name}.build.dirty_source_complete')
    if build['extension_mtime_ns'] is not None:
        _require_int(build['extension_mtime_ns'],
                     f'{name}.build.extension_mtime_ns', 0)
    loader = _require_mapping(
        build['native_loader'], f'{name}.build.native_loader')
    loader_fields = ('command', 'dependencies', 'returncode')
    _reject_unknown(loader, f'{name}.build.native_loader', loader_fields)
    _require_fields(loader, f'{name}.build.native_loader', loader_fields)
    if loader['command'] is not None:
        if not isinstance(loader['command'], list):
            raise SchemaError(
                f'{name}.build.native_loader.command must be an array')
        for index, argument in enumerate(loader['command']):
            _require_str(
                argument,
                f'{name}.build.native_loader.command[{index}]')
    if not isinstance(loader['dependencies'], list):
        raise SchemaError(
            f'{name}.build.native_loader.dependencies must be an array')
    for index, dependency in enumerate(loader['dependencies']):
        _require_str(
            dependency,
            f'{name}.build.native_loader.dependencies[{index}]')
    if loader['returncode'] is not None:
        _require_int(loader['returncode'],
                     f'{name}.build.native_loader.returncode')
    backend = data['backend']
    backend_fields = ('numpy_version', 'numpy_configuration')
    _reject_unknown(backend, f'{name}.backend', backend_fields)
    _require_fields(backend, f'{name}.backend', backend_fields)
    _require_str(backend['numpy_version'], f'{name}.backend.numpy_version')
    if not isinstance(backend['numpy_configuration'], str):
        raise SchemaError(
            f'{name}.backend.numpy_configuration must be a string')
    threading = data['threading']
    threading_fields = ('requested_threads', 'environment')
    _reject_unknown(threading, f'{name}.threading', threading_fields)
    _require_fields(threading, f'{name}.threading',
                    threading_fields)
    if threading['requested_threads'] is not None:
        _require_int(threading['requested_threads'],
                     f'{name}.threading.requested_threads', 1)
    environment = _require_mapping(
        threading['environment'], f'{name}.threading.environment')
    for key, value in environment.items():
        _require_str(key, f'{name}.threading environment name')
        _require_str(value, f'{name}.threading.environment[{key!r}]')


def _validate_artifact_version(data, kind, name):
    if data.get('schema_version') != SCHEMA_VERSION:
        raise SchemaError(f'unsupported {name} schema version')
    if data.get('schema_kind') != kind:
        raise SchemaError(f'not a matmul benchmark {name}')


def validate_artifact(artifact):
    artifact = _require_mapping(artifact, 'artifact')
    fields = (
        'schema_version', 'schema_kind', 'artifact_id', 'created_at',
        'request', 'metadata', 'panels', 'observations',
    )
    _reject_unknown(artifact, 'artifact', fields)
    _require_fields(artifact, 'artifact', fields)
    _validate_artifact_version(artifact, ARTIFACT_KIND, 'artifact')
    _require_str(artifact['artifact_id'], 'artifact.artifact_id')
    _require_str(artifact['created_at'], 'artifact.created_at')
    _validate_serialized_request(artifact['request'])
    _validate_metadata(artifact['metadata'], 'artifact.metadata')
    if not isinstance(artifact['panels'], list):
        raise SchemaError('artifact.panels must be an array')
    for index, panel in enumerate(artifact['panels']):
        _validate_panel(panel, f'artifact.panels[{index}]')
    if not isinstance(artifact['observations'], list):
        raise SchemaError('artifact.observations must be an array')
    for index, observation in enumerate(artifact['observations']):
        _validate_observation(
            observation, f'artifact.observations[{index}]')
    return artifact


def validate_collection(collection):
    collection = _require_mapping(collection, 'collection')
    required_fields = (
        'schema_version', 'schema_kind', 'collection_id', 'created_at',
        'sources', 'panels', 'observations', 'artifact_count',
    )
    fields = required_fields + (
        'started_at', 'plan', 'plan_sha256', 'estimate', 'cell_orders',
    )
    _reject_unknown(collection, 'collection', fields)
    _require_fields(collection, 'collection', required_fields)
    _validate_artifact_version(collection, COLLECTION_KIND, 'collection')
    _require_str(collection['collection_id'], 'collection.collection_id')
    _require_str(collection['created_at'], 'collection.created_at')
    if not isinstance(collection['sources'], list):
        raise SchemaError('collection.sources must be an array')
    source_ids = set()
    sources_by_id = {}
    for index, source in enumerate(collection['sources']):
        name = f'collection.sources[{index}]'
        source = _require_mapping(source, name)
        source_fields = (
            'source_id', 'artifact_id', 'path', 'created_at', 'request',
            'metadata', 'panel_count', 'panels_sha256',
        )
        _reject_unknown(source, name, source_fields)
        _require_fields(source, name, source_fields)
        source_id = _require_str(source['source_id'], f'{name}.source_id')
        if source_id in source_ids:
            raise SchemaError('collection source IDs must be unique')
        source_ids.add(source_id)
        sources_by_id[source_id] = source
        _require_str(source['artifact_id'], f'{name}.artifact_id')
        _require_optional_str(source['path'], f'{name}.path')
        _require_str(source['created_at'], f'{name}.created_at')
        request = _validate_serialized_request(source['request'])
        _validate_metadata(source['metadata'], f'{name}.metadata')
        panel_count = _require_int(
            source['panel_count'], f'{name}.panel_count', 0)
        if panel_count != request.mode.panels:
            raise SchemaError(f'{name}.panel_count does not match its request')
        digest = _require_str(
            source['panels_sha256'], f'{name}.panels_sha256')
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise SchemaError(
                f'{name}.panels_sha256 must be a SHA-256 digest')
    if not isinstance(collection['panels'], list):
        raise SchemaError('collection.panels must be an array')
    panels_by_source = {source_id: [] for source_id in source_ids}
    for index, item in enumerate(collection['panels']):
        name = f'collection.panels[{index}]'
        item = _require_mapping(item, name)
        panel_fields = (
            'source_id', 'source_artifact_id', 'source_panel_index', 'panel')
        _reject_unknown(item, name, panel_fields)
        _require_fields(item, name, panel_fields)
        source_id = item['source_id']
        if source_id not in source_ids:
            raise SchemaError(f'{name}.source_id is unknown')
        source = sources_by_id[source_id]
        if item['source_artifact_id'] != source['artifact_id']:
            raise SchemaError(
                f'{name}.source_artifact_id does not match its source')
        panel_index = _require_int(
            item['source_panel_index'], f'{name}.source_panel_index', 0)
        _validate_panel(item['panel'], f'{name}.panel')
        panels_by_source[source_id].append((panel_index, item['panel']))
    for source_id, indexed_panels in panels_by_source.items():
        indexed_panels.sort(key=lambda item: item[0])
        indexes = [index for index, _panel in indexed_panels]
        source = sources_by_id[source_id]
        if indexes != list(range(source['panel_count'])):
            raise SchemaError(
                f'collection panel indexes for {source_id!r} '
                'must be complete, contiguous, and unique')
        panels = [panel for _index, panel in indexed_panels]
        request = BenchmarkRequest.from_dict(source['request'])
        if {panel['index'] for panel in panels} != set(
                range(request.mode.panels)):
            raise SchemaError(
                f'collection panels for {source_id!r} do not cover '
                'every request panel')
        if any(
                sample['repetitions'] != request.mode.repetitions
                for panel in panels for sample in panel['samples']):
            raise SchemaError(
                f'collection panels for {source_id!r} use the wrong '
                'repetition count')
        if panels_sha256(panels) != source['panels_sha256']:
            raise SchemaError(
                f'collection panels for {source_id!r} do not match '
                'their source digest')
    if not isinstance(collection['observations'], list):
        raise SchemaError('collection.observations must be an array')
    observation_indexes = {source_id: [] for source_id in source_ids}
    for index, item in enumerate(collection['observations']):
        name = f'collection.observations[{index}]'
        item = _require_mapping(item, name)
        fields = ('source_id', 'source_observation_index', 'observation')
        _reject_unknown(item, name, fields)
        _require_fields(item, name, fields)
        if item['source_id'] not in source_ids:
            raise SchemaError(f'{name}.source_id is unknown')
        _require_int(item['source_observation_index'],
                     f'{name}.source_observation_index', 0)
        observation_indexes[item['source_id']].append(
            item['source_observation_index'])
        _validate_observation(item['observation'], f'{name}.observation')
    for source_id, indexes in observation_indexes.items():
        if sorted(indexes) != list(range(len(indexes))):
            raise SchemaError(
                f'collection observation indexes for {source_id!r} '
                'must be contiguous and unique')
    _require_int(collection['artifact_count'],
                 'collection.artifact_count', 0)
    if collection['artifact_count'] != len(collection['sources']):
        raise SchemaError('collection artifact_count is inconsistent')
    provenance_fields = (
        'started_at', 'plan', 'plan_sha256', 'estimate', 'cell_orders')
    present = [field in collection for field in provenance_fields]
    if any(present) and not all(present):
        raise SchemaError(
            'collection plan provenance must be complete')
    if all(present):
        _require_str(collection['started_at'], 'collection.started_at')
        digest = _require_str(
            collection['plan_sha256'], 'collection.plan_sha256')
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise SchemaError(
                'collection.plan_sha256 must be a SHA-256 digest')
        from . import collection as collection_module
        collection_module.validate_collection_provenance(collection)
    return collection


def validate_document(document):
    document = _require_mapping(document, 'document')
    if document.get('schema_kind') == ARTIFACT_KIND:
        return validate_artifact(document)
    if document.get('schema_kind') == COLLECTION_KIND:
        return validate_collection(document)
    raise SchemaError('unknown matmul benchmark document kind')


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
