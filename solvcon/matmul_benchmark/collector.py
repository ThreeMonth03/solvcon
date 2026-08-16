# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Collect correctness-first, interleaved matmul route measurements."""

import contextlib
import dataclasses
import datetime
import hashlib
import io
import json
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
from . import duration
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


def _git_state(ignored_paths=(), ignored_checkpoint_outputs=()):
    repository = pathlib.Path(__file__).resolve().parents[2]
    ignored_paths = {
        pathlib.Path(path).expanduser().resolve()
        for path in ignored_paths if path is not None
    }
    ignored_checkpoint_outputs = tuple(
        pathlib.Path(path).expanduser().resolve()
        for path in ignored_checkpoint_outputs if path is not None)

    def ignored_untracked(path):
        path = path.resolve()
        if path in ignored_paths:
            return True
        return any(
            path.parent == output.parent
            and re.fullmatch(
                re.escape(output.name)
                + r'\.checkpoint\.[0-9a-f]{64}\.json',
                path.name)
            for output in ignored_checkpoint_outputs)
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


def _metadata(request, ignored_paths=(), ignored_checkpoint_outputs=()):
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
        _git_state(ignored_paths, ignored_checkpoint_outputs)
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


def _activity_event(phase, state, route, resolved_route, scope,
                    cell_id, panel, panels, total_calls,
                    elapsed_ns=None, error=None):
    event = {
        'type': 'progress',
        'phase': phase,
        'state': state,
        'route': route,
        'resolved_route': resolved_route,
        'scope': scope,
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
                  panel=None, panels=None, scope=None):
    if scope is None and phase in ('native_batch', 'python_end_to_end'):
        scope = phase

    def report(state, elapsed_ns=None, error=None):
        if progress is not None:
            progress(_activity_event(
                phase, state, route, resolved_route, scope, cell_id,
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


def _run_native_benchmark(case, name, repetitions):
    if name == 'auto':
        return case.benchmark_auto(repetitions)
    return case.benchmark_route(name, repetitions)


def _make_observation(request, output_shape, candidates, summaries,
                      python_summaries):
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
    numpy_summary = python_summaries.get('numpy')
    numpy_median = (None if numpy_summary is None
                    else numpy_summary['median_ns'])
    for candidate in candidates:
        route = dict(candidate)
        route['timing'] = summaries.get(candidate['name'])
        route['python_timing'] = python_summaries.get(candidate['name'])
        python_timing = route['python_timing']
        route['numpy_ratio'] = (
            python_timing['median_ns'] / numpy_median
            if python_timing is not None and numpy_median else None)
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
    native_names: tuple
    python_names: tuple
    per_call_work: int
    cell_id: str | None = None
    panels: list = dataclasses.field(default_factory=list)
    samples_by_route: dict = dataclasses.field(default_factory=dict)
    python_samples_by_route: dict = dataclasses.field(default_factory=dict)


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
    native_names = tuple(
        candidate['name'] for candidate in candidates
        if candidate['kind'] != 'numpy'
        and candidate['correctness']['correct']
    )
    python_names = tuple(
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
        native_names=native_names,
        python_names=python_names,
        per_call_work=per_call_work,
        cell_id=cell_id,
        samples_by_route={name: [] for name in native_names},
        python_samples_by_route={name: [] for name in python_names},
    )


def _warm_up(prepared, warmups=None, progress=None, cancelled=None):
    request = prepared.request
    if warmups is None:
        warmups = request.mode.warmups
    for order in schedule.balanced_orders(prepared.native_names, 1):
        for name in order:
            if warmups:
                _run_activity(
                    lambda name=name: _run_native_benchmark(
                        prepared.case, name, warmups),
                    'warmup', name, warmups, progress, cancelled,
                    resolved_route=_resolved_route(prepared, name),
                    cell_id=prepared.cell_id, scope='native_batch')
    for order in schedule.balanced_orders(prepared.python_names, 1):
        for name in order:
            if warmups:
                _run_activity(
                    lambda name=name: _execute_python_repetitions(
                        prepared, name, warmups),
                    'warmup', name, warmups, progress, cancelled,
                    resolved_route=_resolved_route(prepared, name),
                    cell_id=prepared.cell_id,
                    scope='python_end_to_end')


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
    native_order = collection_module.balanced_order_at(
        prepared.native_names, order_index)
    native_samples = []
    for name in native_order:
        _, elapsed_ns = _run_activity(
            lambda name=name: _run_native_benchmark(
                prepared.case, name, request.mode.repetitions),
            'native_batch', name, request.mode.repetitions,
            progress, cancelled,
            resolved_route=_resolved_route(prepared, name),
            cell_id=prepared.cell_id, panel=panel, panels=panels)
        latency_ns = elapsed_ns / request.mode.repetitions
        prepared.samples_by_route[name].append(latency_ns)
        native_samples.append({
            'route': name,
            'scope': 'native_batch',
            'repetitions': request.mode.repetitions,
            'elapsed_ns': elapsed_ns,
            'latency_ns': latency_ns,
        })
    prepared.panels.append({
        'index': panel_index,
        'scope': 'native_batch',
        'order': list(native_order),
        'samples': native_samples,
    })

    python_order = collection_module.balanced_order_at(
        prepared.python_names, order_index)
    python_samples = []
    for name in python_order:
        _, elapsed_ns = _run_activity(
            lambda name=name: _benchmark_python(
                prepared.case, name, request.mode.repetitions,
                prepared.lhs, prepared.rhs, clock),
            'python_end_to_end', name, request.mode.repetitions,
            progress, cancelled,
            resolved_route=_resolved_route(prepared, name),
            cell_id=prepared.cell_id, panel=panel, panels=panels)
        latency_ns = elapsed_ns / request.mode.repetitions
        prepared.python_samples_by_route[name].append(latency_ns)
        python_samples.append({
            'route': name,
            'scope': 'python_end_to_end',
            'repetitions': request.mode.repetitions,
            'elapsed_ns': elapsed_ns,
            'latency_ns': latency_ns,
        })
    prepared.panels.append({
        'index': panel_index,
        'scope': 'python_end_to_end',
        'order': list(python_order),
        'samples': python_samples,
    })


def _finish_artifact(prepared, metadata):
    summaries = {
        name: _summarize(samples)
        for name, samples in prepared.samples_by_route.items()
    }
    python_summaries = {
        name: _summarize(samples)
        for name, samples in prepared.python_samples_by_route.items()
    }
    observation = _make_observation(
        prepared.request, prepared.output_shape, prepared.candidates,
        summaries, python_summaries)
    return {
        'schema_version': schema.SCHEMA_VERSION,
        'schema_kind': schema.ARTIFACT_KIND,
        'artifact_id': uuid.uuid4().hex,
        'created_at': _now_iso(),
        'request': prepared.request.to_dict(),
        'metadata': metadata,
        'candidates': prepared.candidates,
        'panels': prepared.panels,
        'summaries': summaries,
        'python_summaries': python_summaries,
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


def duration_checkpoint_path(output_path, plan_sha256):
    if output_path is None:
        return None
    if not isinstance(plan_sha256, str) or not re.fullmatch(
            r'[0-9a-f]{64}', plan_sha256):
        raise schema.SchemaError(
            'checkpoint plan hash must be a SHA-256 digest')
    return pathlib.Path(
        f'{os.fspath(output_path)}.checkpoint.{plan_sha256}.json')


def _duration_stream_count(prepared_cases):
    return sum(
        len(prepared.native_names) + len(prepared.python_names)
        for prepared in prepared_cases)


def _calibration_progress(prepared, route, scope, completed, total,
                          pass_name):
    cell_id = prepared.cell_id or prepared.request.request_id
    return {
        'type': 'progress',
        'phase': 'calibration',
        'completed': completed,
        'total': total,
        'cell_id': cell_id,
        'route': route,
        'scope': scope,
        'message': (
            f'{pass_name} calibration {completed}/{total}: '
            f'{cell_id} / '
            f'{route} / {scope}'),
    }


def _calibration_pass(prepared_cases, repetitions, clock, wall_clock,
                      progress, completed, total, pass_name,
                      cancelled):
    measurements = []
    timed_ns = 0
    started_ns = wall_clock()
    for prepared in prepared_cases:
        cell_id = prepared.cell_id or prepared.request.request_id
        for route in collection_module.balanced_order_at(
                prepared.native_names, 0):
            _, elapsed_ns = _run_activity(
                lambda route=route: _run_native_benchmark(
                    prepared.case, route, repetitions),
                'native_batch', route, repetitions, progress,
                cancelled,
                resolved_route=_resolved_route(prepared, route),
                cell_id=cell_id)
            measurement = duration.CalibrationMeasurement(
                cell_id=cell_id, route=route,
                scope=duration.NATIVE_SCOPE,
                elapsed_ns=max(1, int(elapsed_ns)),
                repetitions=repetitions)
            measurements.append(measurement)
            timed_ns += measurement.elapsed_ns
            completed += 1
            if progress is not None:
                progress(_calibration_progress(
                    prepared, route, duration.NATIVE_SCOPE,
                    completed, total, pass_name))
        for route in collection_module.balanced_order_at(
                prepared.python_names, 0):
            _, elapsed_ns = _run_activity(
                lambda route=route: _benchmark_python(
                    prepared.case, route, repetitions,
                    prepared.lhs, prepared.rhs, clock),
                'python_end_to_end', route, repetitions, progress,
                cancelled,
                resolved_route=_resolved_route(prepared, route),
                cell_id=cell_id)
            measurement = duration.CalibrationMeasurement(
                cell_id=cell_id, route=route,
                scope=duration.PYTHON_SCOPE,
                elapsed_ns=max(1, int(elapsed_ns)),
                repetitions=repetitions)
            measurements.append(measurement)
            timed_ns += measurement.elapsed_ns
            completed += 1
            if progress is not None:
                progress(_calibration_progress(
                    prepared, route, duration.PYTHON_SCOPE,
                    completed, total, pass_name))
    wall_elapsed_ns = max(0, wall_clock() - started_ns)
    overhead = duration.ControllerOverhead(
        elapsed_ns=max(0, wall_elapsed_ns - timed_ns))
    return tuple(measurements), overhead, completed


def _calibration_repetitions(spec, pilot_measurements, pilot_overhead,
                             preflight_elapsed_ns, guard, stream_count):
    estimate = duration.estimate_balanced_panel(
        pilot_measurements, (pilot_overhead,),
        uncertainty_fraction=spec.uncertainty_fraction)
    repetitions = duration.choose_calibration_repetitions(spec, estimate)
    used_repetitions = spec.warmups + 1
    call_repetitions = (
        (collection_module.MAX_COLLECTION_CALLS - guard.fixed_calls)
        // stream_count - used_repetitions)
    route_repetitions = (
        schema.MAX_MODE_CALLS_PER_ROUTE - used_repetitions)
    minimum_formal_seconds = (
        estimate.per_repetition.upper_seconds
        * (spec.warmups
           + spec.minimum_calibration_repetitions
           * spec.minimum_panels)
        + estimate.controller_per_panel.upper_seconds
        * spec.minimum_panels)
    remaining_seconds = (
        spec.seconds * spec.safety_fraction
        - preflight_elapsed_ns / 1_000_000_000
        - estimate.calibration_seconds
        - minimum_formal_seconds)
    duration_repetitions = (
        math.floor(
            remaining_seconds
            / estimate.per_repetition.upper_seconds)
        if remaining_seconds > 0 else 0)
    limits = [
        repetitions, call_repetitions, route_repetitions,
        duration_repetitions,
    ]
    if guard.maximum_work is not None:
        limits.append(
            (guard.maximum_work - guard.fixed_work)
            // guard.work_per_balanced_repetition - used_repetitions)
    repetitions = min(limits)
    maximum_latency = estimate.maximum_timed_block.upper_seconds
    if maximum_latency:
        repetitions = min(
            repetitions,
            math.floor(spec.checkpoint_seconds / maximum_latency))
    return (
        repetitions
        if repetitions >= spec.minimum_calibration_repetitions
        else None)


def _duration_shard_plan(plan, run_id, shard_index, panel_count,
                         repetitions):
    return dataclasses.replace(
        plan,
        mode=schema.ModeSpec(
            name=plan.target_duration.mode,
            warmups=plan.target_duration.warmups,
            repetitions=repetitions,
            panels=panel_count),
        target_duration=None,
        output_path=None,
        plan_id=f'{run_id}:shard-{shard_index:04d}',
        schema_version=collection_module.PLAN_SCHEMA_VERSION,
    )


def _shard_prepared_case(prepared, request):
    return dataclasses.replace(
        prepared,
        request=request,
        panels=[],
        samples_by_route={name: [] for name in prepared.native_names},
        python_samples_by_route={
            name: [] for name in prepared.python_names},
    )


def _checkpoint_artifact(source, observation, panels):
    candidates = []
    summaries = {}
    python_summaries = {}
    for route in observation['routes'].values():
        candidates.append({
            key: value for key, value in route.items()
            if key not in ('timing', 'python_timing', 'numpy_ratio')
        })
        if route['timing'] is not None:
            summaries[route['name']] = route['timing']
        if route['python_timing'] is not None:
            python_summaries[route['name']] = route['python_timing']
    return {
        'schema_version': schema.SCHEMA_VERSION,
        'schema_kind': schema.ARTIFACT_KIND,
        'artifact_id': source['artifact_id'],
        'created_at': source['created_at'],
        'request': source['request'],
        'metadata': source['metadata'],
        'candidates': candidates,
        'panels': panels,
        'summaries': summaries,
        'python_summaries': python_summaries,
        'observations': [observation],
    }


def _checkpoint_artifacts(document):
    source_ids = [
        source_id
        for shard in document['duration_run']['shards']
        for source_id in shard['source_ids']
    ]
    sources = {
        source['source_id']: source for source in document['sources']
    }
    observations = {}
    for wrapper in document['observations']:
        observations.setdefault(wrapper['source_id'], []).append(
            wrapper['observation'])
    panels = {source_id: [] for source_id in sources}
    for item in document['panels']:
        panels[item['source_id']].append(item)
    artifacts = []
    for source_id in source_ids:
        source_observations = observations.get(source_id, ())
        if len(source_observations) != 1:
            raise schema.SchemaError(
                'duration checkpoint source must have one observation')
        source_panels = sorted(
            panels[source_id],
            key=lambda item: item['source_panel_index'])
        artifacts.append(_checkpoint_artifact(
            sources[source_id], source_observations[0],
            [item['panel'] for item in source_panels]))
    return artifacts


def _duration_artifact_projection(plan, target_schedule, metadata,
                                  measurements, overheads):
    cell_count = len(plan.cells)
    shard_count = target_schedule.shard_count
    raw_bytes = collection_module.estimate_artifact_bytes(
        plan, panel_count=target_schedule.panels)
    raw_bytes += (
        shard_count * cell_count
        * collection_module.ARTIFACT_FIXED_BYTES_PER_CELL)
    metadata_bytes = len(json.dumps(
        metadata, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf8'))
    raw_bytes += metadata_bytes * cell_count * shard_count
    raw_bytes += (
        collection_module.ARTIFACT_FIXED_BYTES_PER_CELL * cell_count)
    evidence_bytes = len(json.dumps({
        'measurements': [dataclasses.asdict(item)
                         for item in measurements],
        'overheads': [dataclasses.asdict(item) for item in overheads],
        'schedule': dataclasses.asdict(target_schedule),
    }, sort_keys=True, separators=(',', ':'), allow_nan=False).encode(
        'utf8'))
    provenance_bytes = shard_count * (4096 + 64 * cell_count)
    return (raw_bytes + evidence_bytes + provenance_bytes) * 5 // 4


def _build_duration_collection(artifacts, run):
    document = artifact_module.merge_artifacts(artifacts)
    document['collection_id'] = run['run_id']
    document['duration_run'] = run
    document['aggregate_observations'] = \
        collection_module.aggregate_duration_observations(document)
    return schema.validate_document(document)


def _load_duration_checkpoint(plan, checkpoint_path, identity):
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    document = artifact_module.load_artifact(checkpoint_path)
    run = document.get('duration_run')
    if run is None:
        raise schema.SchemaError(
            'checkpoint is not a target-duration collection')
    if run['template_plan_sha256'] != plan.sha256():
        raise schema.SchemaError(
            'checkpoint target does not match the requested plan')
    if run['metadata_identity'] != identity:
        raise schema.SchemaError(
            'checkpoint machine, build, or threading identity changed')
    return document


def _duration_run_guard(plan, run):
    if run is None:
        return collection_module.duration_shard_guard(plan)
    return duration.ShardGuard(**run['shard_guard'])


def _target_measurement_progress(plan, target_schedule, shard_index,
                                 panel_index, cell_position, cell_index):
    event = _collection_progress(
        plan, panel_index, cell_position, cell_index)
    event.update({
        'completed': panel_index * len(plan.cells) + cell_position + 1,
        'total': target_schedule.panels * len(plan.cells),
        'panel': panel_index + 1,
        'panels': target_schedule.panels,
        'shard': shard_index + 1,
        'shards': target_schedule.shard_count,
    })
    event['message'] = (
        f'shard {shard_index + 1}/{target_schedule.shard_count}, '
        f'panel {panel_index + 1}/{target_schedule.panels}, '
        f'cell {cell_position + 1}/{len(plan.cells)}')
    return event


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


def _collect_fixed_plan(plan, engine, clock, progress, cancelled):
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

    return _run_activity(
        lambda: _finish_fixed_collection(
            plan, plan_sha256, started_at, estimate, orders,
            prepared_cases, metadata),
        'finalization', 'artifact', 1, progress, cancelled)


def _collect_target_plan(plan, engine, clock, wall_clock, progress,
                         cancelled, checkpoint_path):
    collection_module.validate_execution_plan(plan)
    target = plan.target_duration
    run_started_at = _now_iso()
    output_path = plan.output_path
    checkpoint_path = (
        duration_checkpoint_path(output_path, plan.sha256())
        if checkpoint_path is None else pathlib.Path(checkpoint_path))
    ignored_paths = (output_path, checkpoint_path)
    segment_started_ns = wall_clock()
    _validate_thread_environment(plan.request_at(0))
    metadata = _run_activity(
        lambda: _metadata(
            plan.request_at(0), ignored_paths=ignored_paths,
            ignored_checkpoint_outputs=(output_path,)),
        'provenance', 'input', 1, progress, cancelled)
    identity = collection_module.metadata_identity(metadata)
    checkpoint = _run_activity(
        lambda: _load_duration_checkpoint(
            plan, checkpoint_path, identity),
        'checkpoint_load', 'checkpoint', 1, progress, cancelled)
    if checkpoint is not None:
        canonical = collection_module.CollectionPlan.from_dict(
            checkpoint['duration_run']['template_plan'])
        run = dict(checkpoint['duration_run'])
        run['shards'] = list(run['shards'])
        run['resume_count'] += 1
        run['resumed_from_checkpoint'] = True
        all_artifacts = _run_activity(
            lambda: _checkpoint_artifacts(checkpoint),
            'finalization', 'artifact', 1, progress, cancelled)
    else:
        canonical = dataclasses.replace(plan, output_path=None)
        run = None
        all_artifacts = []

    prepared_cases = []
    for index, request in enumerate(canonical.requests()):
        _check_cancelled(cancelled)
        prepared = _prepare_case(
            request, engine, cell_id=canonical.cells[index].cell_id,
            progress=progress, cancelled=cancelled)
        _require_correct_plan_case(prepared)
        prepared_cases.append(prepared)
        if progress is not None:
            progress({
                'type': 'progress',
                'phase': 'preparation',
                'completed': index + 1,
                'total': len(canonical.cells),
                'message': (
                    f'validated input {index + 1}/'
                    f'{len(canonical.cells)}'),
            })
    guard = _duration_run_guard(canonical, run)
    stream_count = _duration_stream_count(prepared_cases)

    if run is None:
        used_repetitions = target.warmups + 1
        if (guard.fixed_calls + stream_count * used_repetitions
                > collection_module.MAX_COLLECTION_CALLS):
            raise duration.DurationModelError(
                'one-repetition pilot exceeds the collection guards')
        for prepared in prepared_cases:
            _check_cancelled(cancelled)
            _warm_up(
                prepared, target.warmups, progress=progress,
                cancelled=cancelled)
        preflight_elapsed_ns = max(
            0, wall_clock() - segment_started_ns)
        total_calibrations = stream_count
        pilot, pilot_overhead, completed = _calibration_pass(
            prepared_cases, 1,
            clock, wall_clock, progress, 0, total_calibrations,
            'pilot', cancelled)
        calibration_repetitions = _calibration_repetitions(
            target, pilot, pilot_overhead, preflight_elapsed_ns,
            guard, stream_count)
        if calibration_repetitions is None:
            measurements = pilot
            overheads = (pilot_overhead,)
        else:
            total_calibrations = 2 * stream_count
            calibrated, calibrated_overhead, _completed = \
                _calibration_pass(
                    prepared_cases, calibration_repetitions,
                    clock, wall_clock, progress, completed,
                    total_calibrations, 'timed', cancelled)
            measurements = pilot + calibrated
            overheads = (pilot_overhead, calibrated_overhead)
        if progress is not None:
            progress({
                'type': 'progress',
                'phase': 'planning',
                'completed': 0,
                'total': 0,
                'message': 'planning target-duration shards',
            })
        target_schedule = duration.plan_target_duration(
            target, measurements, overheads,
            preflight_elapsed_ns=preflight_elapsed_ns,
            shard_guard=guard)
        if not target_schedule.feasible:
            raise duration.DurationModelError(target_schedule.reason)
        projected_bytes = _duration_artifact_projection(
            canonical, target_schedule, metadata,
            measurements, overheads)
        if projected_bytes \
                > collection_module.MAX_COLLECTION_ARTIFACT_BYTES:
            raise MemoryError(
                'target-duration artifact projection needs '
                f'{projected_bytes} bytes, limit is '
                f'{collection_module.MAX_COLLECTION_ARTIFACT_BYTES}')
        run_id = uuid.uuid4().hex
        run = {
            'version': 1,
            'run_id': run_id,
            'status': 'checkpoint',
            'started_at': run_started_at,
            'requested': collection_module.target_duration_to_dict(target),
            'template_plan': canonical.to_dict(),
            'template_plan_sha256': canonical.sha256(),
            'metadata_identity': identity,
            'preflight_elapsed_ns': preflight_elapsed_ns,
            'calibration_measurements': [
                dataclasses.asdict(item) for item in measurements],
            'controller_overheads': [
                dataclasses.asdict(item) for item in overheads],
            'shard_guard': dataclasses.asdict(guard),
            'schedule': dataclasses.asdict(target_schedule),
            'completed_panels': 0,
            'measurement_elapsed_ns': 0,
            'actual_elapsed_ns': 0,
            'resumed_from_checkpoint': False,
            'resume_count': 0,
            'shards': [],
        }
        base_actual_elapsed_ns = 0
    else:
        measurements = tuple(
            duration.CalibrationMeasurement(**item)
            for item in run['calibration_measurements'])
        overheads = tuple(
            duration.ControllerOverhead(**item)
            for item in run['controller_overheads'])
        target_schedule = duration.plan_target_duration(
            target, measurements, overheads,
            preflight_elapsed_ns=run['preflight_elapsed_ns'],
            shard_guard=guard)
        if dataclasses.asdict(target_schedule) != run['schedule']:
            raise schema.SchemaError(
                'checkpoint schedule no longer matches its target')
        base_actual_elapsed_ns = run['actual_elapsed_ns']
        passes = len(measurements) // stream_count
        total_calibrations = passes * stream_count
        if progress is not None:
            progress({
                'type': 'progress',
                'phase': 'calibration',
                'completed': total_calibrations,
                'total': total_calibrations,
                'cell_id': None,
                'route': None,
                'scope': None,
                'message': 'reused calibration from checkpoint',
            })
        if run['status'] == 'complete':
            _check_cancelled(cancelled)
            run['actual_elapsed_ns'] = (
                base_actual_elapsed_ns
                + max(0, wall_clock() - segment_started_ns))
            document = _run_activity(
                lambda: _build_duration_collection(all_artifacts, run),
                'finalization', 'artifact', 1, progress, cancelled)
            if progress is not None:
                progress({
                    'type': 'progress',
                    'phase': 'finalization',
                    'completed': 1,
                    'total': 1,
                    'message': 'validated completed checkpoint',
                })
            return document

    sizes = collection_module.duration_shard_sizes(target_schedule)
    first_shard = len(run['shards'])
    panel_offset = run['completed_panels']
    for shard_index in range(first_shard, len(sizes)):
        _check_cancelled(cancelled)
        panel_count = sizes[shard_index]
        shard_plan = _duration_shard_plan(
            canonical, run['run_id'], shard_index,
            panel_count, target_schedule.repetitions)
        shard_cases = [
            _shard_prepared_case(prepared, request)
            for prepared, request in zip(
                prepared_cases, shard_plan.requests())
        ]
        shard_started_at = _now_iso()
        shard_started_ns = wall_clock()
        for cell_index in collection_module.panel_cell_order_at(
                canonical, panel_offset):
            _check_cancelled(cancelled)
            _warm_up(
                shard_cases[cell_index], progress=progress,
                cancelled=cancelled)
        for local_index in range(panel_count):
            global_index = panel_offset + local_index
            order = collection_module.panel_cell_order_at(
                canonical, global_index)
            for cell_position, cell_index in enumerate(order):
                _check_cancelled(cancelled)
                _measure_panel(
                    shard_cases[cell_index], local_index, clock,
                    order_index=global_index, progress=progress,
                    cancelled=cancelled, panel=global_index + 1,
                    panels=target_schedule.panels)
                if progress is not None:
                    progress(_target_measurement_progress(
                        canonical, target_schedule, shard_index,
                        global_index, cell_position, cell_index))
        shard_elapsed_ns = max(0, wall_clock() - shard_started_ns)
        shard_artifacts = _run_activity(
            lambda: [
                _finish_artifact(prepared, metadata)
                for prepared in shard_cases
            ],
            'finalization', 'artifact', 1, progress, cancelled)
        source_offset = len(all_artifacts)
        all_artifacts.extend(shard_artifacts)
        source_ids = [
            f'source-{source_offset + index}'
            for index in range(len(canonical.cells))
        ]
        run['shards'].append({
            'id': f'{run["run_id"]}:shard-{shard_index:04d}',
            'index': shard_index,
            'panel_offset': panel_offset,
            'panel_count': panel_count,
            'started_at': shard_started_at,
            'completed_at': _now_iso(),
            'elapsed_ns': shard_elapsed_ns,
            'mode': shard_plan.mode.to_dict(),
            'plan_id': shard_plan.plan_id,
            'plan_sha256': shard_plan.sha256(),
            'estimate': collection_module.estimate_plan(
                shard_plan).to_dict(),
            'cell_orders_sha256': collection_module.cell_orders_sha256(
                canonical, panel_offset, panel_count),
            'source_ids': source_ids,
        })
        panel_offset += panel_count
        run['completed_panels'] = panel_offset
        run['measurement_elapsed_ns'] = sum(
            item['elapsed_ns'] for item in run['shards'])
        run['actual_elapsed_ns'] = (
            base_actual_elapsed_ns
            + max(0, wall_clock() - segment_started_ns))
        run['status'] = (
            'complete' if len(run['shards']) == len(sizes)
            else 'checkpoint')
        document = _run_activity(
            lambda: _build_duration_collection(all_artifacts, run),
            'finalization', 'artifact', 1, progress, cancelled)
        if checkpoint_path is not None:
            _run_activity(
                lambda: artifact_module.write_artifact(
                    document, checkpoint_path),
                'checkpoint_write', 'checkpoint', 1,
                progress, cancelled)
            if progress is not None:
                progress({
                    'type': 'checkpoint',
                    'phase': 'measurement',
                    'artifact_path': str(checkpoint_path.resolve()),
                    'completed_shards': len(run['shards']),
                    'total_shards': target_schedule.shard_count,
                    'completed_panels': panel_offset,
                    'total_panels': target_schedule.panels,
                    'message': (
                        f'saved shard {shard_index + 1}/'
                        f'{target_schedule.shard_count} checkpoint'),
                })
        run['actual_elapsed_ns'] = (
            base_actual_elapsed_ns
            + max(0, wall_clock() - segment_started_ns))
        if run['status'] == 'complete':
            document = _run_activity(
                lambda: _build_duration_collection(all_artifacts, run),
                'finalization', 'artifact', 1, progress, cancelled)
    if progress is not None:
        progress({
            'type': 'progress',
            'phase': 'finalization',
            'completed': 1,
            'total': 1,
            'message': 'validated target-duration collection',
        })
    return document


def collect_plan(plan, engine=None, clock=time.perf_counter_ns,
                 progress=None, cancelled=None,
                 wall_clock=time.perf_counter_ns,
                 checkpoint_path=None):
    """Collect one frozen plan in this process and publish on success."""

    if not isinstance(plan, collection_module.CollectionPlan):
        plan = collection_module.CollectionPlan.from_dict(plan)
    plan = collection_module.CollectionPlan.from_dict(plan.to_dict())
    engine = engine or engine_module.SolvconRouteEngine()
    if plan.target_duration is not None:
        return _collect_target_plan(
            plan, engine, clock, wall_clock, progress, cancelled,
            checkpoint_path)
    return _collect_fixed_plan(
        plan, engine, clock, progress, cancelled)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
