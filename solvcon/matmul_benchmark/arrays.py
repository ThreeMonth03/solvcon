# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Construct bounded ndarray views from element-stride descriptions."""

import ctypes
import dataclasses
import math
import os
import pathlib
import sys

import numpy as np

from . import schema


DEFAULT_MAX_STORAGE_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_LOGICAL_OPERAND_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_PEAK_BYTES = 512 * 1024 * 1024
CORRECTNESS_OUTPUT_MEMORY_MULTIPLIER = 8
RANDOM_CHUNK_BYTES = 1024 * 1024
MAX_SINGLE_ALLOCATION_BYTES = 4 * 1024 * 1024 * 1024
RESOURCE_BUDGET_DIVISOR = 2
_UNSET = object()


@dataclasses.dataclass(frozen=True)
class ResourceBudget:
    available_bytes: int
    peak_bytes: int
    single_allocation_bytes: int

    def __post_init__(self):
        values = (
            self.available_bytes,
            self.peak_bytes,
            self.single_allocation_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values):
            raise TypeError('resource budget values must be integers')
        if any(value < 0 for value in values):
            raise ValueError('resource budget values must be non-negative')
        if self.peak_bytes > self.available_bytes:
            raise ValueError('peak budget exceeds available memory')
        if self.single_allocation_bytes > self.peak_bytes:
            raise ValueError('single allocation exceeds peak budget')


def _read_text_file(path):
    try:
        return pathlib.Path(path).read_text(encoding='ascii').strip()
    except (OSError, UnicodeError):
        return None


def _read_linux_meminfo():
    if not sys.platform.startswith('linux'):
        return {}
    text = _read_text_file('/proc/meminfo')
    if text is None:
        return {}

    values = {}
    for line in text.splitlines():
        name, separator, remainder = line.partition(':')
        fields = remainder.split()
        if separator and fields and fields[0].isdigit():
            multiplier = 1024 if len(fields) > 1 and fields[1] == 'kB' else 1
            values[name] = int(fields[0]) * multiplier
    return values


def _read_windows_available_bytes():
    if sys.platform != 'win32':
        return None

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = (
            ('length', ctypes.c_ulong),
            ('memory_load', ctypes.c_ulong),
            ('total_physical', ctypes.c_ulonglong),
            ('available_physical', ctypes.c_ulonglong),
            ('total_page_file', ctypes.c_ulonglong),
            ('available_page_file', ctypes.c_ulonglong),
            ('total_virtual', ctypes.c_ulonglong),
            ('available_virtual', ctypes.c_ulonglong),
            ('available_extended_virtual', ctypes.c_ulonglong),
        )

    status = _MemoryStatusEx()
    status.length = ctypes.sizeof(status)
    try:
        succeeded = ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status))
    except (AttributeError, OSError):
        return None
    return int(status.available_physical) if succeeded else None


def _read_posix_available_bytes():
    try:
        pages = os.sysconf('SC_AVPHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if pages < 0 or page_size <= 0:
        return None
    return pages * page_size


def _read_os_available_bytes():
    mem_available = _read_linux_meminfo().get('MemAvailable')
    if mem_available is not None:
        return mem_available
    windows_available = _read_windows_available_bytes()
    if windows_available is not None:
        return windows_available
    return _read_posix_available_bytes()


def _read_nonnegative_integer(path):
    text = _read_text_file(path)
    if text is None or not text.isdigit():
        return None
    return int(text)


def _current_cgroup_directories():
    root = pathlib.Path('/sys/fs/cgroup')
    directories = [root]
    text = _read_text_file('/proc/self/cgroup')
    if text is None:
        return directories

    for line in text.splitlines():
        hierarchy, separator, relative = line.partition('::')
        if separator and hierarchy == '0':
            directories.insert(0, root / relative.lstrip('/'))
            continue
        fields = line.split(':', 2)
        if len(fields) == 3 and 'memory' in fields[1].split(','):
            directories.insert(
                0, root / 'memory' / fields[2].lstrip('/'))
    return directories


def _finite_cgroup_limit(path):
    text = _read_text_file(path)
    if text is None or text == 'max' or not text.isdigit():
        return None
    value = int(text)
    return value if value < (1 << 60) else None


def _read_cgroup_remaining_bytes():
    remaining = []
    for directory in _current_cgroup_directories():
        for limit_name, usage_name in (
                ('memory.max', 'memory.current'),
                ('memory.limit_in_bytes', 'memory.usage_in_bytes')):
            limit = _finite_cgroup_limit(directory / limit_name)
            usage = _read_nonnegative_integer(directory / usage_name)
            if limit is not None and usage is not None:
                remaining.append(max(0, limit - usage))
    return min(remaining) if remaining else None


def _read_linux_commit_headroom_bytes():
    if _read_nonnegative_integer(
            '/proc/sys/vm/overcommit_memory') != 2:
        return None
    meminfo = _read_linux_meminfo()
    limit = meminfo.get('CommitLimit')
    committed = meminfo.get('Committed_AS')
    if limit is None or committed is None:
        return None
    return max(0, limit - committed)


def _valid_memory_reading(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def resolve_resource_budget(
        *, os_available_bytes=_UNSET, cgroup_remaining_bytes=_UNSET,
        commit_headroom_bytes=_UNSET):
    """Resolve a conservative process memory budget."""
    if os_available_bytes is _UNSET:
        os_available_bytes = _read_os_available_bytes()
    if cgroup_remaining_bytes is _UNSET:
        cgroup_remaining_bytes = _read_cgroup_remaining_bytes()
    if commit_headroom_bytes is _UNSET:
        commit_headroom_bytes = _read_linux_commit_headroom_bytes()

    readings = (
        os_available_bytes,
        cgroup_remaining_bytes,
        commit_headroom_bytes,
    )
    available = [value for value in readings
                 if _valid_memory_reading(value)]
    if not available:
        return ResourceBudget(
            available_bytes=DEFAULT_MAX_PEAK_BYTES,
            peak_bytes=DEFAULT_MAX_PEAK_BYTES,
            single_allocation_bytes=DEFAULT_MAX_STORAGE_BYTES,
        )

    available_bytes = min(available)
    peak_bytes = available_bytes // RESOURCE_BUDGET_DIVISOR
    single_allocation_bytes = min(
        peak_bytes, MAX_SINGLE_ALLOCATION_BYTES)
    return ResourceBudget(
        available_bytes=available_bytes,
        peak_bytes=peak_bytes,
        single_allocation_bytes=single_allocation_bytes,
    )


def correctness_peak_bytes(retained_operand_bytes, output_bytes):
    """Bound live outputs and NumPy comparison temporaries."""
    return (
        retained_operand_bytes
        + CORRECTNESS_OUTPUT_MEMORY_MULTIPLIER * output_bytes)


def winograd_scratch_bytes(rows, inner_size, columns, itemsize):
    return (
        (rows // 2) * (inner_size // 2)
        + (inner_size // 2) * (columns // 2)
    ) * itemsize


def _storage_bounds(spec):
    if any(length == 0 for length in spec.shape):
        return 0, 0
    minimum = 0
    maximum = 0
    for length, stride in zip(spec.shape, spec.strides):
        displacement = (length - 1) * stride
        minimum += min(0, displacement)
        maximum += max(0, displacement)
    return minimum, maximum


def operand_logical_bytes(spec, dtype):
    if not isinstance(spec, schema.OperandSpec):
        spec = schema.OperandSpec.from_dict(spec)
    dtype = np.dtype(dtype)
    return math.prod(spec.shape) * dtype.itemsize


def operand_storage_bytes(spec, dtype):
    if not isinstance(spec, schema.OperandSpec):
        spec = schema.OperandSpec.from_dict(spec)
    dtype = np.dtype(dtype)
    minimum, maximum = _storage_bounds(spec)
    elements = 1 if any(size == 0 for size in spec.shape) \
        else maximum - minimum + 1
    return elements * dtype.itemsize


def _make_random_storage(elements, dtype, rng):
    if not np.issubdtype(dtype, np.complexfloating):
        return rng.standard_normal(elements, dtype=dtype.name)

    component_dtype = np.empty(0, dtype=dtype.name).real.dtype
    storage = np.empty(elements, dtype=dtype.name)
    chunk_elements = max(1, RANDOM_CHUNK_BYTES // component_dtype.itemsize)
    for begin in range(0, elements, chunk_elements):
        end = min(begin + chunk_elements, elements)
        storage.real[begin:end] = rng.standard_normal(
            end - begin, dtype=component_dtype.name)
        storage.imag[begin:end] = rng.standard_normal(
            end - begin, dtype=component_dtype.name)
    return storage


def make_strided_array(spec, dtype, seed=0, max_storage_bytes=None,
                       max_logical_bytes=None):
    """Return a safe ndarray with positive, negative, or zero strides."""

    if not isinstance(spec, schema.OperandSpec):
        spec = schema.OperandSpec.from_dict(spec)
    dtype = np.dtype(dtype)
    if max_storage_bytes is None or max_logical_bytes is None:
        single_allocation_bytes = (
            resolve_resource_budget().single_allocation_bytes)
        if max_storage_bytes is None:
            max_storage_bytes = single_allocation_bytes
        if max_logical_bytes is None:
            max_logical_bytes = single_allocation_bytes
    minimum, maximum = _storage_bounds(spec)
    elements = 1 if any(size == 0 for size in spec.shape) \
        else maximum - minimum + 1
    storage_bytes = operand_storage_bytes(spec, dtype)
    if storage_bytes > max_storage_bytes:
        raise MemoryError(
            f'operand storage needs {storage_bytes} bytes, limit is '
            f'{max_storage_bytes}')
    logical_bytes = operand_logical_bytes(spec, dtype)
    if logical_bytes > max_logical_bytes:
        raise MemoryError(
            f'logical operand needs {logical_bytes} bytes, limit is '
            f'{max_logical_bytes}')

    rng = np.random.default_rng(seed)
    storage = _make_random_storage(elements, dtype, rng)
    strides = tuple(stride * dtype.itemsize for stride in spec.strides)
    offset = -minimum * dtype.itemsize
    view = np.ndarray(
        shape=spec.shape,
        dtype=dtype.name,
        buffer=storage,
        offset=offset,
        strides=strides,
    )
    return view


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
