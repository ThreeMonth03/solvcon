# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""QProcess protocol and artifact helpers for matmul benchmarks."""

import json
import os
import re
import sys
import tempfile
import uuid

from PySide6 import QtCore

from ... import matmul_benchmark

__all__ = ['BenchmarkProcess', 'RunStatus']


_WORKER_MODULE = 'solvcon_matmul_benchmark'
_THREAD_VARIABLES = (
    'OPENBLAS_NUM_THREADS',
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'BLIS_NUM_THREADS',
)
_TOKEN_RE = re.compile(r'[xX,\s]+')
_ACTIVITY_PHASES = frozenset((
    'provenance',
    'preparation',
    'reference',
    'correctness',
    'warmup',
    'timing',
    'finalization',
    'partial_write',
    'artifact_write',
))
_ACTIVITY_STATES = frozenset(('started', 'completed', 'failed'))
_ROUTE_LABELS = {
    'input': 'Inputs',
    'artifact': 'Result',
    'partial': 'Partial result',
    'auto': 'Auto',
    'numpy': 'NumPy',
    'naive': 'Naive',
    'blas_dot': 'BLAS DOT',
    'blas_gevm': 'BLAS GEVM',
    'blas_gemv': 'BLAS GEMV',
    'blas_gemm': 'BLAS GEMM',
    'winograd': 'Winograd',
}
_PHASE_LABELS = {
    'provenance': 'recording machine and build details',
    'preparation': 'input preparation',
    'reference': 'NumPy reference',
    'correctness': 'correctness check',
    'warmup': 'setup run',
    'timing': 'measurement',
    'finalization': 'building and validating result',
    'partial_write': 'saving completed measurement rounds',
    'artifact_write': 'saving final result',
}


def parse_integer_tuple(text, *, minimum=None):
    """Return the integer tuple encoded by a comma, space, or ``x`` list."""
    tokens = [token for token in _TOKEN_RE.split(text.strip()) if token]
    if not tokens:
        raise ValueError('Enter at least one integer')
    try:
        values = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError('Every value must be an integer') from exc
    if minimum is not None and any(value < minimum for value in values):
        raise ValueError(f'Every value must be at least {minimum}')
    return values


def contiguous_strides(shape):
    """Return element strides for a row-major contiguous shape."""
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * max(1, shape[index + 1])
    return strides


def worker_command():
    """Return the executable and arguments for the benchmark worker."""
    program = sys.executable
    arguments = []
    basename = os.path.basename(program)
    if 'pilot' in basename.lower():
        arguments.append('--mode=python')
    arguments.extend(('-m', _WORKER_MODULE, '--json-lines'))
    return program, arguments


def _format_time(value):
    if value is None:
        return ''
    value = float(value)
    if value < 1000:
        return f'{value:.1f} ns'
    if value < 1000000:
        return f'{value / 1000:.2f} us'
    return f'{value / 1000000:.2f} ms'


def _candidate_name(candidate):
    return str(candidate.get('name') or '')


def _artifact_observations(artifact):
    if not isinstance(artifact, dict):
        raise ValueError('Benchmark artifact must be a JSON object')
    observations = artifact.get('observations')
    if not isinstance(observations, list):
        raise ValueError('Artifact observations must be a list')
    return observations


def _observation_candidates(observation):
    routes = observation.get('routes')
    if not isinstance(routes, dict):
        return []
    return list(routes.values())


def _winner_name(observation):
    return str(observation.get('winner') or '')


def _selected_name(observation):
    return str(observation.get('auto_route') or '')


def _packing_text(candidate, kind):
    packing = candidate.get('packing', {})
    if not isinstance(packing, dict):
        return ''
    names = [
        side for side in ('lhs', 'rhs')
        if packing.get(f'{kind}_{side}')
    ]
    return '+'.join(names) if names else 'none'


def _correctness_text(candidate):
    correctness = candidate.get('correctness')
    if not isinstance(correctness, dict):
        return ''
    text = 'pass' if correctness.get('correct') else 'FAIL'
    error = correctness.get('max_absolute_error')
    return f'{text} ({error:.3g})' if error is not None else text


def _summary_value(candidate, name):
    timing = candidate.get('timing')
    if name == 'median_ns':
        return timing.get('median_ns') if isinstance(timing, dict) else None
    if name == 'p95_ns':
        return timing.get('p95_ns') if isinstance(timing, dict) else None
    if name == 'numpy_ratio':
        return candidate.get('numpy_ratio')
    if name == 'noise' and isinstance(timing, dict):
        median = timing.get('median_ns')
        deviation = timing.get('mad_ns')
        if deviation is not None and median:
            return float(deviation) / float(median)
    return None


def _is_integer(value, minimum=0):
    return (not isinstance(value, bool)
            and isinstance(value, int) and value >= minimum)


def _valid_activity(event):
    required = {
        'phase', 'state', 'route', 'resolved_route', 'cell_id',
        'panel', 'panels', 'chunk', 'completed_calls', 'total_calls',
        'chunk_calls', 'event_at_ns', 'message',
    }
    if not required <= set(event):
        return False
    state = event.get('state')
    route = event.get('route')
    resolved_route = event.get('resolved_route')
    cell_id = event.get('cell_id')
    panel = event.get('panel')
    panels = event.get('panels')
    elapsed_ns = event.get('elapsed_ns')
    valid_panel = (
        panel is None and panels is None
        or _is_integer(panel, 1) and _is_integer(panels, 1)
        and panel <= panels)
    valid_elapsed = (
        state == 'started' and elapsed_ns is None
        or state != 'started' and _is_integer(elapsed_ns))
    completed_calls = event.get('completed_calls')
    total_calls = event.get('total_calls')
    chunk_calls = event.get('chunk_calls')
    valid_calls = (
        _is_integer(completed_calls)
        and _is_integer(total_calls, 1)
        and _is_integer(chunk_calls, 1)
        and (completed_calls >= chunk_calls
             if state == 'completed'
             else completed_calls + chunk_calls <= total_calls))
    return (
        event.get('phase') in _ACTIVITY_PHASES
        and state in _ACTIVITY_STATES
        and isinstance(route, str) and bool(route)
        and (resolved_route is None
             or isinstance(resolved_route, str) and bool(resolved_route))
        and (cell_id is None
             or isinstance(cell_id, str) and bool(cell_id))
        and valid_panel
        and _is_integer(event.get('chunk'), 1)
        and valid_calls
        and completed_calls <= total_calls
        and _is_integer(event.get('event_at_ns'))
        and valid_elapsed
        and isinstance(event.get('message', ''), str))


def _route_label(route, resolved_route=None):
    route = _ROUTE_LABELS.get(route, route)
    if resolved_route:
        resolved_route = _ROUTE_LABELS.get(
            resolved_route, resolved_route)
        return f'{route} -> {resolved_route}'
    return route


class RunStatus(QtCore.QObject):
    """Render worker activity with an elapsed clock owned by Qt."""

    changed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._activity = None
        self._elapsed_ns = None
        self._clock = QtCore.QElapsedTimer()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)

    def begin(self):
        """Reset stale activity before starting another worker."""
        self._timer.stop()
        self._clock.invalidate()
        self._activity = None
        self._elapsed_ns = None
        self.changed.emit('Starting worker...')

    @QtCore.Slot(object)
    def show_activity(self, activity):
        self._activity = dict(activity)
        state = activity['state']
        if state == 'started':
            self._elapsed_ns = None
            self._clock.start()
            self._timer.start()
            prefix = 'Running'
        else:
            self._timer.stop()
            self._clock.invalidate()
            self._elapsed_ns = activity['elapsed_ns']
            prefix = 'Completed' if state == 'completed' else 'Failed'
        self.changed.emit(self._format(prefix))

    def show_summary(self, text):
        if text:
            self.changed.emit(text)

    def stopping(self):
        """Freeze and report the operation interrupted by the user."""
        if self._clock.isValid():
            self._elapsed_ns = self._clock.nsecsElapsed()
        self._timer.stop()
        self._clock.invalidate()
        self.changed.emit(self._format('Stopping'))

    def complete(self, text='Complete'):
        self._finish_clock()
        self.changed.emit(text)
        self._clear_activity()

    def fail(self, message):
        self._finish_clock()
        if self._activity is None:
            text = f'Error: {message}'
        else:
            text = f'Error while {self._context()}: {message}'
        self.changed.emit(text)
        self._clear_activity()

    def cancel(self, suffix=''):
        self._finish_clock()
        text = ('Cancelled' if self._activity is None
                else f'Cancelled while {self._context()}')
        self.changed.emit(text + suffix)
        self._clear_activity()

    def _finish_clock(self):
        if self._clock.isValid():
            self._elapsed_ns = self._clock.nsecsElapsed()
        self._timer.stop()
        self._clock.invalidate()

    def _clear_activity(self):
        self._activity = None
        self._elapsed_ns = None

    @QtCore.Slot()
    def _tick(self):
        self.changed.emit(self._format('Running'))

    def _format(self, prefix):
        if self._activity is None:
            return prefix
        return f'{prefix}: {self._context()}'

    def _context(self):
        activity = self._activity
        route = _route_label(
            activity['route'], activity.get('resolved_route'))
        if activity['phase'] == 'provenance':
            route = 'Environment'
        phase = _PHASE_LABELS[activity['phase']]
        parts = [route, f'phase: {phase}']
        if activity['panel'] is not None:
            parts.append(
                f'round {activity["panel"]}/{activity["panels"]}')
        completed = activity['completed_calls']
        if activity['state'] != 'completed':
            first = completed + 1
            last = min(
                activity['total_calls'], completed + activity['chunk_calls'])
        else:
            last = completed
            first = max(1, last - activity['chunk_calls'] + 1)
        calls = str(first) if first == last else f'{first}-{last}'
        parts.append(f'call {calls}/{activity["total_calls"]}')
        if activity['cell_id']:
            parts.append(f'cell {activity["cell_id"]}')
        parts.append(f'elapsed: {self._elapsed_seconds():.1f} s')
        return ' | '.join(parts)

    def _elapsed_seconds(self):
        if self._clock.isValid():
            return self._clock.nsecsElapsed() / 1e9
        return (self._elapsed_ns or 0) / 1e9


class BenchmarkProcess(QtCore.QObject):
    """Run a benchmark worker and translate its JSON-lines protocol."""

    progress = QtCore.Signal(int, int, str)
    activity = QtCore.Signal(object)
    partial = QtCore.Signal(str, int, int)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()
    running_changed = QtCore.Signal(bool)

    def __init__(self, process_factory=None, parent=None):
        super().__init__(parent)
        factory = process_factory or QtCore.QProcess
        self._process = factory(self)
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._request = None
        self._result_received = False
        self._cancel_requested = False
        self._error_received = False
        self._running = False
        self._temporary_directory = None
        self._kill_timer = QtCore.QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill_if_running)

        self._process.started.connect(self._send_request)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.finished.connect(self._finished)
        self._process.errorOccurred.connect(self._process_error)

    @property
    def running(self):
        return self._running

    @property
    def cancelling(self):
        return self._cancel_requested

    def start(self, request):
        """Start one worker after applying its thread environment."""
        if self._running:
            raise RuntimeError('A benchmark is already running')
        self._kill_timer.stop()
        self._request = dict(request)
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix='solvcon-matmul-benchmark-')
        self._request.setdefault('id', uuid.uuid4().hex)
        if not self._request.get('output_path'):
            self._request['output_path'] = os.path.join(
                self._temporary_directory.name, 'artifact.json')
        self._stdout.clear()
        self._stderr.clear()
        self._result_received = False
        self._cancel_requested = False
        self._error_received = False

        environment = QtCore.QProcessEnvironment.systemEnvironment()
        threads = str(request['threads'])
        for name in _THREAD_VARIABLES:
            environment.insert(name, threads)
        self._process.setProcessEnvironment(environment)

        program, arguments = worker_command()
        self._set_running(True)
        self._process.start(program, arguments)

    def cancel(self):
        """Kill the current isolated worker without blocking the GUI thread."""
        if not self._running:
            return
        if not self._result_received and not self._error_received:
            self._cancel_requested = True
        self._kill_timer.stop()
        self._process.kill()

    def shutdown(self):
        """Stop a worker before its owning window is destroyed."""
        self._kill_timer.stop()
        if not self._running:
            return
        if not self._result_received and not self._error_received:
            self._cancel_requested = True
        self._process.kill()
        wait = getattr(self._process, 'waitForFinished', None)
        if wait is not None:
            wait(1000)
        self._set_running(False)

    def _set_running(self, running):
        if self._running != running:
            self._running = running
            self.running_changed.emit(running)

    def _send_request(self):
        payload = json.dumps(self._request, separators=(',', ':')) + '\n'
        self._process.write(payload.encode('utf8'))
        self._process.closeWriteChannel()

    def _read_stdout(self):
        self._stdout.extend(bytes(self._process.readAllStandardOutput()))
        self._drain_stdout()

    def _read_stderr(self):
        self._stderr.extend(bytes(self._process.readAllStandardError()))

    def _drain_stdout(self, final=False):
        while b'\n' in self._stdout:
            line, _, rest = self._stdout.partition(b'\n')
            self._stdout = bytearray(rest)
            self._handle_line(line)
        if final and self._stdout:
            line = bytes(self._stdout)
            self._stdout.clear()
            self._handle_line(line)

    def _handle_line(self, line):
        if self._error_received or self._result_received:
            return
        if not line.strip():
            return
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if self._cancel_requested:
                return
            self._protocol_failure(f'Invalid worker output: {exc}')
            return
        if not isinstance(event, dict):
            if self._cancel_requested:
                return
            self._protocol_failure(
                'Invalid worker output: event must be an object')
            return
        event_type = event.get('type')
        if self._cancel_requested and event_type != 'partial':
            return
        if event_type == 'progress':
            if 'state' in event:
                if not _valid_activity(event):
                    self._protocol_failure(
                        'Invalid worker output: malformed activity event')
                    return
                self.activity.emit(event)
                return
            completed = event.get('completed')
            total = event.get('total')
            message = event.get('message', '')
            if (isinstance(completed, bool)
                    or not isinstance(completed, int)
                    or isinstance(total, bool)
                    or not isinstance(total, int)
                    or completed < 0 or total < 0 or completed > total
                    or not isinstance(message, str)):
                self._protocol_failure(
                    'Invalid worker output: malformed progress event')
                return
            self.progress.emit(
                completed, total, message)
        elif event_type == 'partial':
            path = event.get('artifact_path')
            values = tuple(event.get(name) for name in (
                'completed_panels', 'total_panels'))
            valid_values = all(
                not isinstance(value, bool) and isinstance(value, int)
                for value in values)
            completed_panels, total_panels = values
            if (not isinstance(path, str) or not path
                    or not valid_values
                    or completed_panels < 1
                    or total_panels < completed_panels):
                if self._cancel_requested:
                    return
                self._protocol_failure(
                    'Invalid worker output: malformed partial event')
                return
            self.partial.emit(path, completed_panels, total_panels)
        elif event_type == 'result':
            artifact_path = event.get('artifact_path')
            if not isinstance(artifact_path, str) or not artifact_path:
                self._protocol_failure(
                    'Invalid worker output: result path is missing')
                return
            try:
                artifact = matmul_benchmark.artifact.load_artifact(
                    artifact_path)
            except (MemoryError, OSError, ValueError,
                    json.JSONDecodeError) as exc:
                self._protocol_failure(
                    f'Could not read benchmark result: {exc}')
                return
            self._result_received = True
            self.completed.emit(artifact)
        elif event_type == 'error':
            message = event.get('message')
            if not isinstance(message, str) or not message:
                self._protocol_failure(
                    'Invalid worker output: malformed error event')
                return
            self._error_received = True
            self.failed.emit(message)
            self._terminate_worker()
        else:
            self._protocol_failure(
                f'Invalid worker output: unknown event type {event_type!r}')

    def _protocol_failure(self, message):
        if self._error_received:
            return
        self._error_received = True
        self.failed.emit(message)
        self._terminate_worker()

    def _finished(self, exit_code, _exit_status):
        self._kill_timer.stop()
        self._read_stdout()
        self._read_stderr()
        self._drain_stdout(final=True)
        self._set_running(False)
        if self._cancel_requested:
            self.cancelled.emit()
        elif exit_code and not self._result_received \
                and not self._error_received:
            message = self._stderr.decode('utf8', errors='replace').strip()
            self.failed.emit(message or f'Worker exited with code {exit_code}')
        elif not self._result_received and not self._error_received:
            self.failed.emit('Worker exited without a benchmark result')

    def _process_error(self, error):
        if not self._cancel_requested and not self._error_received \
                and not self._result_received:
            self._error_received = True
            if error == QtCore.QProcess.ProcessError.FailedToStart:
                self._kill_timer.stop()
                self._set_running(False)
            else:
                self._terminate_worker()
            error_code = getattr(error, 'value', error)
            self.failed.emit(
                f'Could not run benchmark worker ({error_code})')

    def _terminate_worker(self):
        if self._running:
            self._process.terminate()
            self._kill_timer.start(1000)

    def _kill_if_running(self):
        if self._running:
            self._process.kill()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
