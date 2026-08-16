# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Collect correctness-first, interleaved matmul route measurements."""

import contextlib
import dataclasses
import datetime
import hashlib
import io
import math
import os
import pathlib
import platform
import re
import shutil
import stat
import statistics
import subprocess
import sys
import time
import uuid

import numpy as np

from . import arrays
from . import artifact as artifact_module
from . import collection as collection_module
from . import engine as engine_module
from . import schedule
from . import schema


THREAD_ENVIRONMENT = (
    'OPENBLAS_NUM_THREADS',
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'BLIS_NUM_THREADS',
)
MAX_UNTRACKED_FILE_HASH_BYTES = 4 * 1024 * 1024
MAX_TOTAL_UNTRACKED_HASH_BYTES = 32 * 1024 * 1024


def _now_iso():
    return datetime.datetime.now(datetime.UTC).isoformat()


def _numpy_configuration():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_config()
    return output.getvalue().strip()


def _run_command(arguments, cwd=None):
    try:
        return subprocess.run(
            arguments, cwd=cwd, capture_output=True, check=False,
            timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git_state(ignored_paths=()):
    repository = pathlib.Path(__file__).resolve().parents[2]
    ignored_paths = {
        pathlib.Path(path).expanduser().resolve()
        for path in ignored_paths if path is not None
    }

    def ignored_untracked(path):
        return path.resolve() in ignored_paths

    commit_process = _run_command(
        ['git', 'rev-parse', 'HEAD'], cwd=repository)
    commit = None
    if commit_process is not None and commit_process.returncode == 0:
        commit = commit_process.stdout.decode('ascii').strip()

    status_process = _run_command(
        ['git', 'status', '--porcelain=v1', '-z'], cwd=repository)
    if status_process is None or status_process.returncode != 0:
        return commit, None, None, False
    digest = hashlib.sha256()
    complete = True
    dirty = False
    diff_process = _run_command(
        ['git', 'diff', '--binary', 'HEAD'], cwd=repository)
    if diff_process is not None and diff_process.returncode == 0:
        if diff_process.stdout:
            dirty = True
            digest.update(diff_process.stdout)
    else:
        complete = False
        digest.update(b'tracked-diff-unavailable\0')
    untracked_process = _run_command(
        ['git', 'ls-files', '--others', '--exclude-standard', '-z'],
        cwd=repository)
    hashed_bytes = 0
    if untracked_process is not None and untracked_process.returncode == 0:
        for encoded_path in untracked_process.stdout.split(b'\0'):
            if not encoded_path:
                continue
            path = repository / encoded_path.decode(
                'utf8', errors='surrogateescape')
            if ignored_untracked(path):
                continue
            dirty = True
            digest.update(encoded_path + b'\0')
            try:
                metadata = path.lstat()
            except OSError:
                complete = False
                digest.update(b'unavailable\0')
                continue
            digest.update(
                f'{metadata.st_mode}:{metadata.st_size}:'
                f'{metadata.st_mtime_ns}'.encode('ascii'))
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    digest.update(os.readlink(path).encode(
                        'utf8', errors='surrogateescape'))
                except OSError:
                    complete = False
                    digest.update(b'symlink-unavailable\0')
                continue
            if not stat.S_ISREG(metadata.st_mode):
                continue
            if (metadata.st_size > MAX_UNTRACKED_FILE_HASH_BYTES
                    or hashed_bytes + metadata.st_size
                    > MAX_TOTAL_UNTRACKED_HASH_BYTES):
                complete = False
                digest.update(b'content-omitted\0')
                continue
            try:
                flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
                descriptor = os.open(path, flags)
                opened = os.fstat(descriptor)
                if (not stat.S_ISREG(opened.st_mode)
                        or opened.st_size != metadata.st_size
                        or opened.st_ino != metadata.st_ino):
                    os.close(descriptor)
                    complete = False
                    digest.update(b'content-changed\0')
                    continue
                with os.fdopen(descriptor, 'rb') as stream:
                    for chunk in iter(
                            lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                hashed_bytes += metadata.st_size
            except OSError:
                complete = False
                digest.update(b'content-unavailable\0')
    else:
        complete = False
        digest.update(b'untracked-list-unavailable\0')
    return (
        (commit, True, digest.hexdigest(), complete)
        if dirty else (commit, False, None, complete))


def _file_sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _native_loader_identity(extension_path):
    if not extension_path:
        return {'command': None, 'dependencies': [], 'returncode': None}
    if platform.system() == 'Darwin':
        executable = shutil.which('otool')
        arguments = [executable, '-L', extension_path] if executable else None
    else:
        executable = shutil.which('ldd')
        arguments = [executable, extension_path] if executable else None
    if arguments is None:
        return {'command': None, 'dependencies': [], 'returncode': None}
    process = _run_command(arguments)
    if process is None:
        return {
            'command': arguments, 'dependencies': [], 'returncode': None,
        }
    dependencies = []
    if process.returncode == 0:
        output = process.stdout.decode('utf8', errors='replace')
        dependencies = [
            re.sub(r'\s+\(0x[0-9a-fA-F]+\)$', '', line.strip())
            for line in output.splitlines() if line.strip()
        ]
    return {
        'command': arguments,
        'dependencies': dependencies,
        'returncode': process.returncode,
    }


def _metadata(request, ignored_paths=()):
    affinity = None
    if hasattr(os, 'sched_getaffinity'):
        affinity = sorted(os.sched_getaffinity(0))
    thread_environment = {
        name: os.environ.get(name)
        for name in THREAD_ENVIRONMENT
        if os.environ.get(name)
    }
    extension_path = None
    try:
        import _solvcon
        extension_path = _solvcon.__file__
    except ImportError:
        pass
    extension_mtime_ns = None
    extension_sha256 = None
    if extension_path and pathlib.Path(extension_path).exists():
        extension_mtime_ns = pathlib.Path(extension_path).stat().st_mtime_ns
        extension_sha256 = _file_sha256(extension_path)
    git_commit, git_dirty, dirty_diff_sha256, dirty_source_complete = \
        _git_state(ignored_paths)
    return {
        'process': {
            'pid': os.getpid(),
            'executable': sys.executable,
            'python': platform.python_version(),
            'affinity': affinity,
        },
        'machine': {
            'node': platform.node(),
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'logical_cpu_count': os.cpu_count(),
        },
        'build': {
            'git_commit': git_commit,
            'git_dirty': git_dirty,
            'dirty_diff_sha256': dirty_diff_sha256,
            'dirty_source_complete': dirty_source_complete,
            'solvcon_extension': extension_path,
            'extension_mtime_ns': extension_mtime_ns,
            'extension_sha256': extension_sha256,
            'native_loader': _native_loader_identity(extension_path),
            'solvcon_profile': os.environ.get('SOLVCON_PROFILE') or None,
        },
        'backend': {
            'numpy_version': np.__version__,
            'numpy_configuration': _numpy_configuration(),
        },
        'threading': {
            'requested_threads': request.threads,
            'environment': thread_environment,
        },
    }


def _problem_shape(request):
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
    output_shape = tuple(batch_shape)
    if not lhs_vector:
        output_shape += (rows,)
    if not rhs_vector:
        output_shape += (columns,)
    return batch_shape, output_shape, rows, inner_size, columns


def _request_winograd_scratch(request, dtype):
    lhs_shape = request.lhs.shape
    rhs_shape = request.rhs.shape
    if len(lhs_shape) != 2 or len(rhs_shape) != 2:
        return 0
    rows, inner_size = lhs_shape
    columns = rhs_shape[-1]
    forced = request.routes is None or 'winograd' in request.routes
    automatic = (
        rows == inner_size == columns
        and rows >= 16_384
        and rows % 2 == 0
        and request.lhs.strides == (inner_size, 1)
        and request.rhs.strides == (columns, 1))
    if not forced and not automatic:
        return 0
    if rows % 2 or inner_size % 2 or columns % 2:
        return 0
    return arrays.winograd_scratch_bytes(
        rows, inner_size, columns, dtype.itemsize)


def _validate_resource_budget(request, resource_budget=None):
    resource_budget = (
        resource_budget or arrays.resolve_resource_budget())
    dtype = np.dtype(request.dtype)
    operand_peak = 0
    for name, operand in (('lhs', request.lhs), ('rhs', request.rhs)):
        storage_bytes = arrays.operand_storage_bytes(operand, dtype)
        if storage_bytes > resource_budget.single_allocation_bytes:
            raise MemoryError(
                f'{name} storage needs {storage_bytes} bytes, limit is '
                f'{resource_budget.single_allocation_bytes}')
        logical_bytes = arrays.operand_logical_bytes(operand, dtype)
        if logical_bytes > resource_budget.single_allocation_bytes:
            raise MemoryError(
                f'{name} logical size needs {logical_bytes} bytes, limit is '
                f'{resource_budget.single_allocation_bytes}')
        operand_peak += storage_bytes + max(storage_bytes, logical_bytes)
    batch_shape, output_shape, rows, inner_size, columns = \
        _problem_shape(request)
    output_elements = math.prod(output_shape)
    output_bytes = output_elements * dtype.itemsize
    if output_bytes > resource_budget.single_allocation_bytes:
        raise MemoryError(
            f'matmul output needs {output_bytes} bytes, limit is '
            f'{resource_budget.single_allocation_bytes}')
    scalar_work = math.prod(batch_shape) * rows * inner_size * columns
    peak_bytes = arrays.correctness_peak_bytes(
        operand_peak, output_bytes)
    peak_bytes += _request_winograd_scratch(request, dtype)
    if peak_bytes > resource_budget.peak_bytes:
        raise MemoryError(
            f'benchmark peak estimate needs {peak_bytes} bytes, limit is '
            f'{resource_budget.peak_bytes}')
    operand_elements = (
        math.prod(request.lhs.shape) + math.prod(request.rhs.shape))
    per_call_io_work = output_elements + operand_elements
    per_call_work = scalar_work + per_call_io_work
    return per_call_work


def validate_request_resources(request, resource_budget=None):
    """Validate one fixed request before allocating its operands."""

    if not isinstance(request, schema.BenchmarkRequest):
        request = schema.BenchmarkRequest.from_dict(request)
    return _validate_resource_budget(
        request, resource_budget=resource_budget)


def _validate_thread_environment(request):
    if request.threads is None:
        return
    expected = str(request.threads)
    mismatches = {
        name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise schema.SchemaError(
            f'threads={request.threads} requires every BLAS thread '
            f'environment variable to equal {expected}; got {mismatches}')


def _comparison(reference, result, dtype, inner_size):
    reference = np.atleast_1d(reference)
    result = np.atleast_1d(result)
    nonfinite_result = bool(
        not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(result)))
    if result.shape != reference.shape:
        return {
            'correct': False,
            'reason': 'shape mismatch',
            'max_absolute_error': None,
            'max_relative_error': None,
            'nonfinite_result': nonfinite_result,
        }
    real_dtype = np.empty(0, dtype=dtype).real.dtype
    tolerance_cap = 1e-3 if real_dtype == np.dtype('float32') else 1e-11
    tolerance = min(
        64 * np.finfo(real_dtype).eps * max(inner_size, 1),
        tolerance_cap,
    )
    maximum_absolute = None
    maximum_relative = None
    if not nonfinite_result:
        difference = np.abs(result - reference)
        denominator = np.maximum(
            np.abs(reference), np.finfo(real_dtype).tiny)
        maximum_absolute = (
            float(np.max(difference)) if difference.size else 0.0)
        with np.errstate(over='ignore', invalid='ignore'):
            maximum_relative = (
                float(np.max(difference / denominator))
                if difference.size else 0.0)
        if not math.isfinite(maximum_absolute):
            maximum_absolute = None
        if not math.isfinite(maximum_relative):
            maximum_relative = None
    correct = bool(
        not nonfinite_result
        and np.allclose(result, reference, rtol=tolerance,
                        atol=tolerance, equal_nan=False))
    return {
        'correct': correct,
        'reason': (None if correct else
                   'non-finite values' if nonfinite_result
                   else 'values differ'),
        'rtol': float(tolerance),
        'atol': float(tolerance),
        'max_absolute_error': maximum_absolute,
        'max_relative_error': maximum_relative,
        'nonfinite_result': nonfinite_result,
    }


def _execute_python(case, name, lhs, rhs):
    if name == 'auto':
        return case.execute_auto()
    if name == 'numpy':
        return np.atleast_1d(np.matmul(lhs, rhs))
    return case.execute_route(name)


def _benchmark_python(case, name, repetitions, lhs, rhs, clock):
    start = clock()
    result = None
    for _ in range(repetitions):
        result = _execute_python(case, name, lhs, rhs)
    elapsed_ns = clock() - start
    return np.atleast_1d(result), int(elapsed_ns)


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


def _summarize(samples):
    median = float(statistics.median(samples))
    deviations = [abs(value - median) for value in samples]
    return {
        'sample_count': len(samples),
        'median_ns': median,
        'mad_ns': float(statistics.median(deviations)),
        'p95_ns': _percentile(samples, 0.95),
        'minimum_ns': float(min(samples)),
        'maximum_ns': float(max(samples)),
    }


def _contraction(lhs_shape, rhs_shape, output_shape):
    lhs_vector = len(lhs_shape) == 1
    rhs_vector = len(rhs_shape) == 1
    lhs_batch = () if lhs_vector else lhs_shape[:-2]
    rhs_batch = () if rhs_vector else rhs_shape[:-2]
    batch_shape = np.broadcast_shapes(lhs_batch, rhs_batch)
    return {
        'batch_shape': list(batch_shape),
        'batch_count': math.prod(batch_shape),
        'm': 1 if lhs_vector else lhs_shape[-2],
        'k': lhs_shape[-1],
        'n': 1 if rhs_vector else rhs_shape[-1],
        'output_shape': list(output_shape),
        'lhs_vector': lhs_vector,
        'rhs_vector': rhs_vector,
    }


def _activity_event(phase, state, route, resolved_route,
                    cell_id, panel, panels, total_calls,
                    elapsed_ns=None, error=None):
    event = {
        'type': 'progress',
        'phase': phase,
        'state': state,
        'route': route,
        'resolved_route': resolved_route,
        'cell_id': cell_id,
        'panel': panel,
        'panels': panels,
        'chunk': 1,
        'completed_calls': total_calls if state == 'completed' else 0,
        'total_calls': total_calls,
        'chunk_calls': total_calls,
        'event_at_ns': time.perf_counter_ns(),
    }
    route_label = (
        f'{route} -> {resolved_route}'
        if resolved_route is not None else route)
    event['message'] = f'{state} {phase}: {route_label}'
    if elapsed_ns is not None:
        event['elapsed_ns'] = elapsed_ns
    if error is not None:
        event['error_type'] = type(error).__name__
        event['message'] = str(error)
    return event


def _run_activity(operation, phase, route, total_calls, progress,
                  cancelled, resolved_route=None, cell_id=None,
                  panel=None, panels=None):
    def report(state, elapsed_ns=None, error=None):
        if progress is not None:
            progress(_activity_event(
                phase, state, route, resolved_route, cell_id,
                panel, panels, total_calls, elapsed_ns, error))

    _check_cancelled(cancelled)
    report('started')
    _check_cancelled(cancelled)
    started_ns = time.perf_counter_ns()
    try:
        result = operation()
    except Exception as exc:
        elapsed_ns = max(0, time.perf_counter_ns() - started_ns)
        report('failed', elapsed_ns, exc)
        raise
    elapsed_ns = max(0, time.perf_counter_ns() - started_ns)
    report('completed', elapsed_ns)
    _check_cancelled(cancelled)
    return result


def _route_candidates(case, request, reference, inner_size,
                      progress=None, cancelled=None, cell_id=None):
    descriptors = {route.name: route for route in case.routes}
    if request.routes is None:
        selected_names = tuple(descriptors)
    else:
        unknown = sorted(set(request.routes) - set(descriptors))
        if unknown:
            raise schema.SchemaError(
                f'ineligible or unknown requested routes: {unknown}')
        selected_names = request.routes

    candidates = []
    auto_descriptor = next(
        (route for route in case.routes if route.selected_by_auto), None)
    resolved_route = (
        None if auto_descriptor is None else auto_descriptor.name)
    auto_correctness = _run_activity(
        lambda: _comparison(
            reference, case.execute_auto(), request.dtype, inner_size),
        'correctness', 'auto', 1, progress, cancelled,
        resolved_route=resolved_route, cell_id=cell_id)
    candidates.append({
        'name': 'auto',
        'kind': 'solvcon_auto',
        'selected_route': (None if auto_descriptor is None
                           else auto_descriptor.name),
        'packing': (None if auto_descriptor is None
                    else auto_descriptor.packing_dict()),
        'correctness': auto_correctness,
    })
    for name in selected_names:
        descriptor = descriptors[name]
        correctness = _run_activity(
            lambda name=name: _comparison(
                reference, case.execute_route(name),
                request.dtype, inner_size),
            'correctness', name, 1, progress, cancelled,
            cell_id=cell_id)
        candidates.append({
            'name': name,
            'kind': 'solvcon_route',
            'selected_by_auto': descriptor.selected_by_auto,
            'packing': descriptor.packing_dict(),
            'correctness': correctness,
        })
    if request.numpy_baseline:
        correctness = _run_activity(
            lambda: _comparison(
                reference, reference, request.dtype, inner_size),
            'correctness', 'numpy', 1, progress, cancelled,
            cell_id=cell_id)
        candidates.append({
            'name': 'numpy',
            'kind': 'numpy',
            'packing': None,
            'correctness': correctness,
        })
    return candidates


def _make_observation(request, output_shape, candidates, summaries):
    solvcon_routes = [
        candidate for candidate in candidates
        if candidate['kind'] == 'solvcon_route'
        and candidate['correctness']['correct']
        and candidate['name'] in summaries
    ]
    ranked = sorted(
        solvcon_routes,
        key=lambda item: summaries[item['name']]['median_ns'],
    )
    winner = ranked[0]['name'] if ranked else None
    runner_up = ranked[1]['name'] if len(ranked) > 1 else None
    margin = None
    if runner_up is not None:
        winner_time = summaries[winner]['median_ns']
        runner_time = summaries[runner_up]['median_ns']
        margin = ((runner_time - winner_time) / winner_time
                  if winner_time else None)
    auto = next(item for item in candidates if item['name'] == 'auto')
    routes = {}
    numpy_summary = summaries.get('numpy')
    numpy_median = (None if numpy_summary is None
                    else numpy_summary['median_ns'])
    for candidate in candidates:
        route = dict(candidate)
        route['timing'] = summaries.get(candidate['name'])
        timing = route['timing']
        route['numpy_ratio'] = (
            timing['median_ns'] / numpy_median
            if timing is not None and numpy_median else None)
        routes[candidate['name']] = route
    return {
        'id': request.request_id,
        'dtype': request.dtype,
        'lhs': request.lhs.to_dict(),
        'rhs': request.rhs.to_dict(),
        'contraction': _contraction(
            request.lhs.shape, request.rhs.shape, output_shape),
        'routes': routes,
        'auto_route': auto['selected_route'],
        'winner': winner,
        'runner_up': runner_up,
        'winner_margin': margin,
    }


@dataclasses.dataclass
class _PreparedCase:
    request: schema.BenchmarkRequest
    lhs: object
    rhs: object
    output_shape: tuple
    case: object
    candidates: list
    names: tuple
    per_call_work: int
    cell_id: str | None = None
    panels: list = dataclasses.field(default_factory=list)
    samples_by_route: dict = dataclasses.field(default_factory=dict)


def _prepare_case(request, engine, cell_id=None, progress=None,
                  cancelled=None):
    resource_budget = arrays.resolve_resource_budget()
    per_call_work = validate_request_resources(
        request, resource_budget=resource_budget)

    def prepare_inputs():
        lhs = arrays.make_strided_array(
            request.lhs, request.dtype, request.seed,
            max_storage_bytes=resource_budget.single_allocation_bytes,
            max_logical_bytes=resource_budget.single_allocation_bytes)
        rhs = arrays.make_strided_array(
            request.rhs, request.dtype, request.seed + 1,
            max_storage_bytes=resource_budget.single_allocation_bytes,
            max_logical_bytes=resource_budget.single_allocation_bytes)
        return lhs, rhs, engine.prepare(lhs, rhs, request.dtype)

    lhs, rhs, case = _run_activity(
        prepare_inputs, 'preparation', 'input', 1, progress,
        cancelled, cell_id=cell_id)
    reference = _run_activity(
        lambda: np.atleast_1d(np.matmul(lhs, rhs)),
        'reference', 'numpy', 1, progress, cancelled, cell_id=cell_id)
    output_shape = tuple(reference.shape)
    candidates = _route_candidates(
        case, request, reference, request.lhs.shape[-1],
        progress=progress, cancelled=cancelled, cell_id=cell_id)
    names = tuple(
        candidate['name'] for candidate in candidates
        if candidate['correctness']['correct']
    )
    return _PreparedCase(
        request=request,
        lhs=lhs,
        rhs=rhs,
        output_shape=output_shape,
        case=case,
        candidates=candidates,
        names=names,
        per_call_work=per_call_work,
        cell_id=cell_id,
        samples_by_route={name: [] for name in names},
    )


def _warm_up(prepared, warmups=None, progress=None, cancelled=None):
    request = prepared.request
    if warmups is None:
        warmups = request.mode.warmups
    for order in schedule.balanced_orders(prepared.names, 1):
        for name in order:
            if warmups:
                _run_activity(
                    lambda name=name: _execute_python_repetitions(
                        prepared, name, warmups),
                    'warmup', name, warmups, progress, cancelled,
                    resolved_route=_resolved_route(prepared, name),
                    cell_id=prepared.cell_id)


def _resolved_route(prepared, name):
    if name != 'auto':
        return None
    auto = next(
        candidate for candidate in prepared.candidates
        if candidate['name'] == 'auto')
    return auto['selected_route']


def _execute_python_repetitions(prepared, name, repetitions):
    result = None
    for _ in range(repetitions):
        result = _execute_python(
            prepared.case, name, prepared.lhs, prepared.rhs)
    return result


def _measure_panel(prepared, panel_index, clock, order_index=None,
                   progress=None, cancelled=None, panel=None,
                   panels=None):
    request = prepared.request
    if order_index is None:
        order_index = panel_index
    order = collection_module.balanced_order_at(
        prepared.names, order_index)
    samples = []
    for name in order:
        _, elapsed_ns = _run_activity(
            lambda name=name: _benchmark_python(
                prepared.case, name, request.mode.repetitions,
                prepared.lhs, prepared.rhs, clock),
            'timing', name, request.mode.repetitions,
            progress, cancelled,
            resolved_route=_resolved_route(prepared, name),
            cell_id=prepared.cell_id, panel=panel, panels=panels)
        latency_ns = elapsed_ns / request.mode.repetitions
        prepared.samples_by_route[name].append(latency_ns)
        samples.append({
            'route': name,
            'repetitions': request.mode.repetitions,
            'elapsed_ns': elapsed_ns,
            'latency_ns': latency_ns,
        })
    prepared.panels.append({
        'index': panel_index,
        'order': list(order),
        'samples': samples,
    })


def _finish_artifact(prepared, metadata):
    summaries = {
        name: _summarize(samples)
        for name, samples in prepared.samples_by_route.items()
    }
    observation = _make_observation(
        prepared.request, prepared.output_shape, prepared.candidates,
        summaries)
    return {
        'schema_version': schema.SCHEMA_VERSION,
        'schema_kind': schema.ARTIFACT_KIND,
        'artifact_id': uuid.uuid4().hex,
        'created_at': _now_iso(),
        'request': prepared.request.to_dict(),
        'metadata': metadata,
        'panels': prepared.panels,
        'observations': [observation],
    }


def collect(request, engine=None, clock=time.perf_counter_ns,
            progress=None, cancelled=None):
    """Collect one artifact without involving the Qt event loop."""

    if not isinstance(request, schema.BenchmarkRequest):
        request = schema.BenchmarkRequest.from_dict(request)
    _validate_thread_environment(request)
    metadata = _run_activity(
        lambda: _metadata(request), 'provenance', 'input', 1,
        progress, cancelled)
    engine = engine or engine_module.SolvconRouteEngine()
    prepared = _prepare_case(
        request, engine, progress=progress, cancelled=cancelled)
    _warm_up(prepared, progress=progress, cancelled=cancelled)
    for panel_index in range(request.mode.panels):
        _measure_panel(
            prepared, panel_index, clock, progress=progress,
            cancelled=cancelled, panel=panel_index + 1,
            panels=request.mode.panels)
        if progress is not None:
            progress({
                'type': 'progress',
                'phase': 'measurement',
                'completed': panel_index + 1,
                'total': request.mode.panels,
                'message': (
                    f'completed panel {panel_index + 1}/'
                    f'{request.mode.panels}'),
            })
    return _run_activity(
        lambda: _finish_artifact(prepared, metadata),
        'finalization', 'artifact', 1, progress, cancelled)


class CollectionCancelled(RuntimeError):
    """Stop an in-process collection without publishing a document."""


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise CollectionCancelled('matmul benchmark collection cancelled')


def _require_correct_plan_case(prepared):
    incorrect = [
        candidate['name'] for candidate in prepared.candidates
        if not candidate['correctness']['correct']
    ]
    if incorrect:
        raise RuntimeError(
            f'collection correctness failed for routes: {incorrect}')


def _collection_progress(plan, panel_index, cell_position, cell_index):
    cell = plan.cells[cell_index]
    request = plan.request_at(cell_index)
    _batch_shape, _output_shape, rows, inner_size, columns = \
        _problem_shape(request)
    completed = panel_index * len(plan.cells) + cell_position + 1
    total = plan.mode.panels * len(plan.cells)
    return {
        'type': 'progress',
        'phase': 'measurement',
        'completed': completed,
        'total': total,
        'panel': panel_index + 1,
        'panels': plan.mode.panels,
        'cell': cell_position + 1,
        'cells': len(plan.cells),
        'cell_id': cell.cell_id,
        'shape': {
            'm': rows,
            'k': inner_size,
            'n': columns,
            'lhs': list(cell.lhs.shape),
            'rhs': list(cell.rhs.shape),
        },
        'message': (
            f'panel {panel_index + 1}/{plan.mode.panels}, '
            f'cell {cell_position + 1}/{len(plan.cells)}: '
            f'M={rows}, K={inner_size}, N={columns}'),
    }


def _finish_fixed_collection(plan, plan_sha256, started_at, estimate,
                             orders, prepared_cases, metadata):
    artifacts = [
        _finish_artifact(prepared, metadata)
        for prepared in prepared_cases
    ]
    result = artifact_module.merge_artifacts(artifacts)
    result.update({
        'started_at': started_at,
        'plan': plan.to_dict(),
        'plan_sha256': plan_sha256,
        'estimate': estimate.to_dict(),
        'cell_orders': [
            [plan.cells[index].cell_id for index in order]
            for order in orders
        ],
    })
    return schema.validate_document(result)


def collection_partial_path(output_path):
    if output_path is None:
        return None
    return pathlib.Path(f'{os.fspath(output_path)}.partial.json')


def _finish_partial_collection(plan, started_at, orders, prepared_cases,
                               metadata, completed_panels):
    mode = dataclasses.replace(plan.mode, panels=completed_panels)
    partial_plan = dataclasses.replace(plan, mode=mode, output_path=None)
    partial_cases = [
        dataclasses.replace(
            prepared, request=partial_plan.request_at(index))
        for index, prepared in enumerate(prepared_cases)
    ]
    return _finish_fixed_collection(
        partial_plan, partial_plan.sha256(), started_at,
        collection_module.estimate_plan(partial_plan),
        orders[:completed_panels], partial_cases, metadata)


def _write_partial_collection(plan, started_at, orders, prepared_cases,
                              metadata, completed_panels, partial_path):
    document = _finish_partial_collection(
        plan, started_at, orders, prepared_cases, metadata,
        completed_panels)
    artifact_module.write_artifact(document, partial_path)
    return document


def _collect_fixed_plan(plan, engine, clock, progress, cancelled,
                        partial_path):
    estimate = collection_module.validate_execution_plan(plan)
    plan_sha256 = plan.sha256()
    started_at = _now_iso()
    requests = plan.requests()
    _validate_thread_environment(requests[0])
    metadata = _run_activity(
        lambda: _metadata(requests[0]),
        'provenance', 'input', 1, progress, cancelled)
    engine = engine or engine_module.SolvconRouteEngine()

    prepared_cases = []
    for index, request in enumerate(requests):
        _check_cancelled(cancelled)
        prepared = _prepare_case(
            request, engine, cell_id=plan.cells[index].cell_id,
            progress=progress, cancelled=cancelled)
        _require_correct_plan_case(prepared)
        prepared_cases.append(prepared)

    orders = collection_module.panel_cell_orders(plan)
    for cell_index in orders[0]:
        _check_cancelled(cancelled)
        _warm_up(
            prepared_cases[cell_index], progress=progress,
            cancelled=cancelled)

    for panel_index, order in enumerate(orders):
        for cell_position, cell_index in enumerate(order):
            _check_cancelled(cancelled)
            _measure_panel(
                prepared_cases[cell_index], panel_index, clock,
                progress=progress, cancelled=cancelled,
                panel=panel_index + 1, panels=plan.mode.panels)
            if progress is not None:
                progress(_collection_progress(
                    plan, panel_index, cell_position, cell_index))
        if partial_path is not None:
            completed_panels = panel_index + 1
            _run_activity(
                lambda: _write_partial_collection(
                    plan, started_at, orders, prepared_cases, metadata,
                    completed_panels, partial_path),
                'partial_write', 'partial', 1, progress, cancelled)
            if progress is not None:
                progress({
                    'type': 'partial',
                    'artifact_path': str(partial_path.resolve()),
                    'completed_panels': completed_panels,
                    'total_panels': plan.mode.panels,
                    'message': (
                        f'saved measurement round {completed_panels}/'
                        f'{plan.mode.panels}'),
                })

    return _run_activity(
        lambda: _finish_fixed_collection(
            plan, plan_sha256, started_at, estimate, orders,
            prepared_cases, metadata),
        'finalization', 'artifact', 1, progress, cancelled)


def collect_plan(plan, engine=None, clock=time.perf_counter_ns,
                 progress=None, cancelled=None, partial_path=None):
    """Collect one frozen plan in this process and publish on success."""

    if not isinstance(plan, collection_module.CollectionPlan):
        plan = collection_module.CollectionPlan.from_dict(plan)
    plan = collection_module.CollectionPlan.from_dict(plan.to_dict())
    engine = engine or engine_module.SolvconRouteEngine()
    if partial_path is not None:
        partial_path = pathlib.Path(partial_path)
    return _collect_fixed_plan(
        plan, engine, clock, progress, cancelled, partial_path)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
