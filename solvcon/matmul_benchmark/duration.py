# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Calibrate balanced matmul schedules against a wall-time target."""

import collections
import dataclasses
import math
import statistics

from . import schema


NATIVE_SCOPE = 'native_batch'
PYTHON_SCOPE = 'python_end_to_end'
_SCOPES = (NATIVE_SCOPE, PYTHON_SCOPE)
_NANOSECONDS_PER_SECOND = 1_000_000_000


class DurationModelError(ValueError):
    """Report invalid calibration evidence or duration settings."""


def _require_integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise DurationModelError(f'{name} must be an integer')
    if value < minimum:
        raise DurationModelError(f'{name} must be at least {minimum}')
    return value


def _require_seconds(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DurationModelError(f'{name} must be a number')
    value = float(value)
    minimum = 0.0 if allow_zero else 1.0 / _NANOSECONDS_PER_SECOND
    if not math.isfinite(value) or value < minimum:
        qualifier = 'non-negative' if allow_zero else 'positive'
        raise DurationModelError(f'{name} must be a finite {qualifier} value')
    if not math.isfinite(value * _NANOSECONDS_PER_SECOND):
        raise DurationModelError(f'{name} is too large')
    return value


@dataclasses.dataclass(frozen=True)
class TargetDurationSpec:
    """Configure calibration, quality, and checkpoint constraints."""

    seconds: float
    mode: str = 'preview'
    safety_fraction: float = 0.95
    calibration_block_seconds: float = 0.02
    minimum_calibration_repetitions: int = 1
    maximum_calibration_repetitions: int = 10_000
    checkpoint_seconds: float = 60.0
    uncertainty_fraction: float = 0.10

    def __post_init__(self):
        seconds = _require_seconds(self.seconds, 'seconds')
        if self.mode not in ('preview', 'stable'):
            raise DurationModelError(
                "mode must be 'preview' or 'stable'")
        safety_fraction = _require_seconds(
            self.safety_fraction, 'safety_fraction')
        if safety_fraction > 1.0:
            raise DurationModelError(
                'safety_fraction must not exceed 1')
        calibration_seconds = _require_seconds(
            self.calibration_block_seconds,
            'calibration_block_seconds')
        minimum_repetitions = _require_integer(
            self.minimum_calibration_repetitions,
            'minimum_calibration_repetitions', 1)
        maximum_repetitions = _require_integer(
            self.maximum_calibration_repetitions,
            'maximum_calibration_repetitions', 1)
        if minimum_repetitions > maximum_repetitions:
            raise DurationModelError(
                'minimum calibration repetitions exceed the maximum')
        if maximum_repetitions > schema.MAX_MODE_REPETITIONS:
            raise DurationModelError(
                'maximum calibration repetitions exceed the schema limit')
        checkpoint_seconds = _require_seconds(
            self.checkpoint_seconds, 'checkpoint_seconds')
        uncertainty_fraction = _require_seconds(
            self.uncertainty_fraction, 'uncertainty_fraction',
            allow_zero=True)
        if uncertainty_fraction >= 1.0:
            raise DurationModelError(
                'uncertainty_fraction must be less than 1')

        object.__setattr__(self, 'seconds', seconds)
        object.__setattr__(self, 'safety_fraction', safety_fraction)
        object.__setattr__(
            self, 'calibration_block_seconds', calibration_seconds)
        object.__setattr__(
            self, 'minimum_calibration_repetitions', minimum_repetitions)
        object.__setattr__(
            self, 'maximum_calibration_repetitions', maximum_repetitions)
        object.__setattr__(
            self, 'checkpoint_seconds', checkpoint_seconds)
        object.__setattr__(
            self, 'uncertainty_fraction', uncertainty_fraction)

    @property
    def minimum_panels(self):
        return schema.ModeSpec.preset(self.mode).panels

    @property
    def warmups(self):
        return schema.ModeSpec.preset(self.mode).warmups


@dataclasses.dataclass(frozen=True)
class CalibrationMeasurement:
    """Record one timed calibration block for one benchmark stream."""

    cell_id: str
    route: str
    scope: str
    elapsed_ns: int
    repetitions: int

    def __post_init__(self):
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise DurationModelError('cell_id must be a non-empty string')
        if not isinstance(self.route, str) or not self.route:
            raise DurationModelError('route must be a non-empty string')
        if self.scope not in _SCOPES:
            raise DurationModelError(
                f'scope must be one of {_SCOPES!r}')
        _require_integer(self.elapsed_ns, 'elapsed_ns', 1)
        _require_integer(self.repetitions, 'repetitions', 1)


@dataclasses.dataclass(frozen=True)
class ControllerOverhead:
    """Record wall time outside timed calls across calibration panels."""

    elapsed_ns: int
    panels: int = 1

    def __post_init__(self):
        _require_integer(self.elapsed_ns, 'elapsed_ns', 0)
        _require_integer(self.panels, 'panels', 1)


@dataclasses.dataclass(frozen=True)
class ShardGuard:
    """Bound calls and abstract work independently for every shard."""

    maximum_calls: int | None = None
    maximum_work: int | None = None
    fixed_calls: int = 0
    fixed_work: int = 0
    work_per_balanced_repetition: int = 0

    def __post_init__(self):
        for name in ('maximum_calls', 'maximum_work'):
            value = getattr(self, name)
            if value is not None:
                _require_integer(value, name, 1)
        for name in (
                'fixed_calls', 'fixed_work',
                'work_per_balanced_repetition'):
            _require_integer(getattr(self, name), name, 0)


@dataclasses.dataclass(frozen=True)
class DurationRange:
    """Hold a robust lower, central, and upper wall-time estimate."""

    lower_seconds: float
    central_seconds: float
    upper_seconds: float


@dataclasses.dataclass(frozen=True)
class BalancedPanelEstimate:
    """Describe all calibrated streams in one balanced panel."""

    stream_count: int
    per_repetition: DurationRange
    controller_per_panel: DurationRange
    typical_timed_block: DurationRange
    maximum_timed_block: DurationRange
    calibration_seconds: float

    def panel_duration(self, repetitions):
        _require_integer(repetitions, 'repetitions', 1)
        return _add_ranges(
            _scale_range(self.per_repetition, repetitions),
            self.controller_per_panel)


@dataclasses.dataclass(frozen=True)
class TargetDurationSchedule:
    """Describe a feasible schedule or a precise infeasibility reason."""

    feasible: bool
    limiter: str
    reason: str | None
    repetitions: int
    panels: int
    shard_count: int
    panels_per_full_shard: int
    full_shard_count: int
    final_shard_panels: int
    repetition_limiter: str
    shard_limiter: str
    predicted: DurationRange | None
    balanced_panel: DurationRange | None
    maximum_timed_block: DurationRange | None
    calibration_seconds: float
    safety_budget_seconds: float
    maximum_calls_per_route_per_shard: int
    maximum_calls_per_shard: int | None
    maximum_work_per_shard: int | None


def _duration_range(values, uncertainty_fraction):
    central = float(statistics.median(values))
    deviations = [abs(value - central) for value in values]
    spread = max(
        3.0 * float(statistics.median(deviations)),
        central * uncertainty_fraction,
    )
    return DurationRange(
        lower_seconds=max(0.0, central - spread),
        central_seconds=central,
        upper_seconds=central + spread,
    )


def _sum_ranges(ranges):
    return DurationRange(
        lower_seconds=sum(item.lower_seconds for item in ranges),
        central_seconds=sum(item.central_seconds for item in ranges),
        upper_seconds=sum(item.upper_seconds for item in ranges),
    )


def _add_ranges(lhs, rhs):
    return DurationRange(
        lower_seconds=lhs.lower_seconds + rhs.lower_seconds,
        central_seconds=lhs.central_seconds + rhs.central_seconds,
        upper_seconds=lhs.upper_seconds + rhs.upper_seconds,
    )


def _scale_range(value, factor):
    return DurationRange(
        lower_seconds=value.lower_seconds * factor,
        central_seconds=value.central_seconds * factor,
        upper_seconds=value.upper_seconds * factor,
    )


def _validate_scope_coverage(groups):
    native = {
        (cell_id, route)
        for cell_id, route, scope in groups if scope == NATIVE_SCOPE
    }
    python = {
        (cell_id, route)
        for cell_id, route, scope in groups if scope == PYTHON_SCOPE
    }
    missing = sorted(native - python)
    if missing:
        raise DurationModelError(
            f'native streams lack Python calibration: {missing!r}')
    cells = {cell_id for cell_id, _route, _scope in groups}
    for cell_id in sorted(cells):
        scopes = {
            scope for item_cell, _route, scope in groups
            if item_cell == cell_id
        }
        if scopes != set(_SCOPES):
            raise DurationModelError(
                f'cell {cell_id!r} must calibrate both timing scopes')


def estimate_balanced_panel(measurements, controller_overheads=(),
                            uncertainty_fraction=0.10):
    """Robustly estimate one complete pass over every timing stream."""

    uncertainty_fraction = _require_seconds(
        uncertainty_fraction, 'uncertainty_fraction', allow_zero=True)
    if uncertainty_fraction >= 1.0:
        raise DurationModelError(
            'uncertainty_fraction must be less than 1')
    measurements = tuple(measurements)
    controller_overheads = tuple(controller_overheads)
    if not measurements:
        raise DurationModelError(
            'at least one calibration measurement is required')
    if not all(
            isinstance(item, CalibrationMeasurement)
            for item in measurements):
        raise DurationModelError(
            'measurements must contain CalibrationMeasurement values')
    if not all(
            isinstance(item, ControllerOverhead)
            for item in controller_overheads):
        raise DurationModelError(
            'controller_overheads must contain ControllerOverhead values')

    groups = collections.defaultdict(list)
    for item in measurements:
        key = item.cell_id, item.route, item.scope
        groups[key].append(
            item.elapsed_ns / item.repetitions
            / _NANOSECONDS_PER_SECOND)
    _validate_scope_coverage(groups)
    stream_ranges = [
        _duration_range(groups[key], uncertainty_fraction)
        for key in sorted(groups)
    ]
    central_values = [item.central_seconds for item in stream_ranges]
    typical = _duration_range(central_values, uncertainty_fraction)
    maximum = DurationRange(
        lower_seconds=max(item.lower_seconds for item in stream_ranges),
        central_seconds=max(central_values),
        upper_seconds=max(item.upper_seconds for item in stream_ranges),
    )

    if controller_overheads:
        overhead_values = [
            item.elapsed_ns / item.panels / _NANOSECONDS_PER_SECOND
            for item in controller_overheads
        ]
        controller = _duration_range(
            overhead_values, uncertainty_fraction)
    else:
        controller = DurationRange(0.0, 0.0, 0.0)
    calibration_seconds = sum(
        item.elapsed_ns for item in measurements)
    calibration_seconds += sum(
        item.elapsed_ns for item in controller_overheads)
    calibration_seconds /= _NANOSECONDS_PER_SECOND
    return BalancedPanelEstimate(
        stream_count=len(stream_ranges),
        per_repetition=_sum_ranges(stream_ranges),
        controller_per_panel=controller,
        typical_timed_block=typical,
        maximum_timed_block=maximum,
        calibration_seconds=calibration_seconds,
    )


def choose_calibration_repetitions(spec, estimate):
    """Choose a global integer block size from typical stream latency."""

    if not isinstance(spec, TargetDurationSpec):
        raise DurationModelError('spec must be a TargetDurationSpec')
    if not isinstance(estimate, BalancedPanelEstimate):
        raise DurationModelError(
            'estimate must be a BalancedPanelEstimate')
    latency = estimate.typical_timed_block.central_seconds
    if latency <= 0.0:
        raise DurationModelError(
            'calibration produced a zero-duration timed block')
    repetitions = math.ceil(spec.calibration_block_seconds / latency)
    return min(max(
        repetitions, spec.minimum_calibration_repetitions),
        spec.maximum_calibration_repetitions,
    )


def _maximum_panels_per_shard(spec, estimate, repetitions, guard):
    panel = estimate.panel_duration(repetitions)
    warmup = _scale_range(estimate.per_repetition, spec.warmups)
    timed_block = _scale_range(
        estimate.maximum_timed_block, repetitions)
    if timed_block.upper_seconds > spec.checkpoint_seconds:
        return 0, 'timed_block_checkpoint'

    remaining = spec.checkpoint_seconds - warmup.upper_seconds
    checkpoint_panels = (
        math.floor(remaining / panel.upper_seconds)
        if remaining >= panel.upper_seconds else 0)
    route_panels = (
        schema.MAX_MODE_CALLS_PER_ROUTE - spec.warmups) // repetitions
    caps = [
        (checkpoint_panels, 'checkpoint_duration'),
        (route_panels, 'calls_per_route'),
        (schema.MAX_MODE_PANELS, 'panel_count'),
    ]
    if guard is not None and guard.maximum_calls is not None:
        fixed_calls = (
            guard.fixed_calls + estimate.stream_count * spec.warmups)
        remaining_calls = guard.maximum_calls - fixed_calls
        guarded_panels = (
            remaining_calls // (estimate.stream_count * repetitions)
            if remaining_calls >= 0 else 0)
        caps.append((guarded_panels, 'call_guard'))
    if guard is not None and guard.maximum_work is not None:
        fixed_work = (
            guard.fixed_work
            + guard.work_per_balanced_repetition * spec.warmups)
        remaining_work = guard.maximum_work - fixed_work
        panel_work = (
            guard.work_per_balanced_repetition * repetitions)
        guarded_panels = (
            remaining_work // panel_work
            if remaining_work >= 0 and panel_work else
            schema.MAX_MODE_PANELS if remaining_work >= 0 else 0)
        caps.append((guarded_panels, 'work_guard'))
    return min(caps, key=lambda item: item[0])


def _total_panels_for_budget(remaining, panel, warmup, shard_panels):
    if remaining <= 0.0 or shard_panels <= 0:
        return 0
    upper = math.floor(remaining / panel.upper_seconds)
    lower = 0
    while lower < upper:
        middle = (lower + upper + 1) // 2
        shards = math.ceil(middle / shard_panels)
        duration = (
            middle * panel.upper_seconds
            + shards * warmup.upper_seconds)
        if duration <= remaining:
            lower = middle
        else:
            upper = middle - 1
    return lower


def _infeasible(spec, estimate, limiter, reason, repetitions=0,
                repetition_limiter='infeasible',
                shard_limiter='infeasible'):
    return TargetDurationSchedule(
        feasible=False,
        limiter=limiter,
        reason=reason,
        repetitions=repetitions,
        panels=0,
        shard_count=0,
        panels_per_full_shard=0,
        full_shard_count=0,
        final_shard_panels=0,
        repetition_limiter=repetition_limiter,
        shard_limiter=shard_limiter,
        predicted=None,
        balanced_panel=None,
        maximum_timed_block=None,
        calibration_seconds=estimate.calibration_seconds,
        safety_budget_seconds=spec.seconds * spec.safety_fraction,
        maximum_calls_per_route_per_shard=0,
        maximum_calls_per_shard=None,
        maximum_work_per_shard=None,
    )


def plan_target_duration(spec, measurements, controller_overheads=(),
                         preflight_elapsed_ns=0, shard_guard=None):
    """Build complete balanced panels within an approximate wall budget."""

    if not isinstance(spec, TargetDurationSpec):
        raise DurationModelError('spec must be a TargetDurationSpec')
    _require_integer(
        preflight_elapsed_ns, 'preflight_elapsed_ns', 0)
    if shard_guard is not None and not isinstance(shard_guard, ShardGuard):
        raise DurationModelError('shard_guard must be a ShardGuard')
    estimate = estimate_balanced_panel(
        measurements, controller_overheads,
        uncertainty_fraction=spec.uncertainty_fraction)
    calibration_seconds = (
        estimate.calibration_seconds
        + preflight_elapsed_ns / _NANOSECONDS_PER_SECOND)
    estimate = dataclasses.replace(
        estimate, calibration_seconds=calibration_seconds)
    safety_budget = spec.seconds * spec.safety_fraction
    remaining = safety_budget - calibration_seconds
    if remaining <= 0.0:
        return _infeasible(
            spec, estimate, 'calibration_budget',
            'calibration and preflight consume the safety budget')

    requested_repetitions = choose_calibration_repetitions(spec, estimate)
    selected = None
    last_shard_limiter = 'infeasible'
    last_shard_panels = 0
    for repetitions in range(
            requested_repetitions,
            spec.minimum_calibration_repetitions - 1, -1):
        shard_panels, shard_limiter = _maximum_panels_per_shard(
            spec, estimate, repetitions, shard_guard)
        last_shard_limiter = shard_limiter
        last_shard_panels = shard_panels
        panel = estimate.panel_duration(repetitions)
        warmup = _scale_range(estimate.per_repetition, spec.warmups)
        panels = _total_panels_for_budget(
            remaining, panel, warmup, shard_panels)
        if panels >= spec.minimum_panels:
            selected = (
                repetitions, panels, shard_panels, shard_limiter,
                panel, warmup)
            break
    if selected is None:
        reason = (
            f'target cannot fit {spec.minimum_panels} complete '
            f'{spec.mode} panels within the safety budget and shard '
            'constraints')
        return _infeasible(
            spec, estimate,
            (last_shard_limiter if last_shard_panels == 0
             else 'target_duration'),
            reason,
            repetitions=spec.minimum_calibration_repetitions,
            shard_limiter=last_shard_limiter,
        )

    repetitions, panels, shard_panels, shard_limiter, panel, warmup = \
        selected
    shard_count = math.ceil(panels / shard_panels)
    full_shard_count, final_shard_panels = divmod(
        panels, shard_panels)
    predicted = DurationRange(
        lower_seconds=(
            calibration_seconds + panels * panel.lower_seconds
            + shard_count * warmup.lower_seconds),
        central_seconds=(
            calibration_seconds + panels * panel.central_seconds
            + shard_count * warmup.central_seconds),
        upper_seconds=(
            calibration_seconds + panels * panel.upper_seconds
            + shard_count * warmup.upper_seconds),
    )
    maximum_timed_block = _scale_range(
        estimate.maximum_timed_block, repetitions)
    actual_full_panels = min(shard_panels, panels)
    calls_per_route = (
        spec.warmups + repetitions * actual_full_panels)
    maximum_calls = None
    maximum_work = None
    if shard_guard is not None:
        maximum_calls = (
            shard_guard.fixed_calls
            + estimate.stream_count * calls_per_route)
        maximum_work = (
            shard_guard.fixed_work
            + shard_guard.work_per_balanced_repetition
            * calls_per_route)
    if repetitions < requested_repetitions:
        repetition_limiter = (
            shard_limiter
            if _maximum_panels_per_shard(
                spec, estimate, repetitions + 1, shard_guard)[0] == 0
            else 'target_duration')
    elif requested_repetitions \
            == spec.maximum_calibration_repetitions:
        raw_repetitions = math.ceil(
            spec.calibration_block_seconds
            / estimate.typical_timed_block.central_seconds)
        repetition_limiter = (
            'calibration_repetition_cap'
            if raw_repetitions > requested_repetitions
            else 'calibration_block_target')
    else:
        repetition_limiter = 'calibration_block_target'
    return TargetDurationSchedule(
        feasible=True,
        limiter='target_duration',
        reason=None,
        repetitions=repetitions,
        panels=panels,
        shard_count=shard_count,
        panels_per_full_shard=shard_panels,
        full_shard_count=full_shard_count,
        final_shard_panels=final_shard_panels,
        repetition_limiter=repetition_limiter,
        shard_limiter=shard_limiter,
        predicted=predicted,
        balanced_panel=panel,
        maximum_timed_block=maximum_timed_block,
        calibration_seconds=calibration_seconds,
        safety_budget_seconds=safety_budget,
        maximum_calls_per_route_per_shard=calls_per_route,
        maximum_calls_per_shard=maximum_calls,
        maximum_work_per_shard=maximum_work,
    )


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
