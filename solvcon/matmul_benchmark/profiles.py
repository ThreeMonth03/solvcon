# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Compose exact matmul operand shapes and element strides."""

import dataclasses
import itertools
import re
import sys

from . import schema


_PROFILE_ID_PATTERN = re.compile(r'^[a-z0-9][a-z0-9_-]*$')
_ORDERS = ('c', 'f', 'custom')
_EXTENTS = ('same', 'one')
_STRIDE_MODES = ('auto', 'zero', 'custom')


def _require_mapping(value, name, fields):
    if not isinstance(value, dict):
        raise schema.SchemaError(f'{name} must be an object')
    unknown = sorted(set(value) - set(fields))
    missing = sorted(set(fields) - set(value))
    if unknown:
        raise schema.SchemaError(f'{name} has unknown fields: {unknown}')
    if missing:
        raise schema.SchemaError(f'{name} is missing fields: {missing}')
    return value


def _require_int(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise schema.SchemaError(f'{name} must be an integer')
    if minimum is not None and value < minimum:
        raise schema.SchemaError(f'{name} must be at least {minimum}')
    return value


def _require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise schema.SchemaError(f'{name} must be a non-empty string')
    if value != value.strip():
        raise schema.SchemaError(
            f'{name} must not have leading or trailing whitespace')
    return value


def _storage_span(shape, strides):
    return 1 + sum(
        (extent - 1) * abs(stride)
        for extent, stride in zip(shape, strides)
    )


def _exact_matmul_shape(lhs, rhs):
    for name, operand in (('A', lhs), ('B', rhs)):
        if not isinstance(operand, schema.OperandSpec):
            raise schema.SchemaError(
                f'exact {name} must be an OperandSpec')
        if len(operand.shape) < 2:
            raise schema.SchemaError(
                f'exact {name} must have at least two axes')
        if any(extent < 1 or extent > sys.maxsize
               for extent in operand.shape):
            raise schema.SchemaError(
                f'exact {name} shape extents must fit a positive ssize_t')
        if any(stride < -sys.maxsize - 1 or stride > sys.maxsize
               for stride in operand.strides):
            raise schema.SchemaError(
                f'exact {name} strides must fit an ssize_t')
        if _storage_span(operand.shape, operand.strides) > sys.maxsize:
            raise schema.SchemaError(
                f'exact {name} storage span exceeds ssize_t')

    m_value, k_value = lhs.shape[-2:]
    rhs_k, n_value = rhs.shape[-2:]
    if k_value != rhs_k:
        raise schema.SchemaError(
            'exact A and B contraction dimensions K do not match')
    output_batch = _broadcast_shape(lhs.shape[:-2], rhs.shape[:-2])
    return m_value, k_value, n_value, (*output_batch, m_value, n_value)


def _broadcast_shape(lhs_shape, rhs_shape):
    rank = max(len(lhs_shape), len(rhs_shape))
    lhs_shape = (1,) * (rank - len(lhs_shape)) + tuple(lhs_shape)
    rhs_shape = (1,) * (rank - len(rhs_shape)) + tuple(rhs_shape)
    output = []
    for lhs_extent, rhs_extent in zip(lhs_shape, rhs_shape):
        if lhs_extent != rhs_extent \
                and lhs_extent != 1 and rhs_extent != 1:
            raise schema.SchemaError(
                'exact A and B batch dimensions do not broadcast')
        output.append(max(lhs_extent, rhs_extent))
    return tuple(output)


def _exact_broadcast_id(lhs, rhs, output_batch):
    if not output_batch:
        return 'matrix'

    def reused(operand):
        batch_shape = operand.shape[:-2]
        batch_strides = operand.strides[:-2]
        missing = len(output_batch) - len(batch_shape)
        shape = (1,) * missing + batch_shape
        strides = (0,) * missing + batch_strides
        return any(
            output_extent > 1
            and (input_extent == 1
                 or (input_extent > 1 and stride == 0))
            for input_extent, output_extent, stride in zip(
                shape, output_batch, strides)
        )

    lhs_reused = reused(lhs)
    rhs_reused = reused(rhs)
    if lhs_reused and rhs_reused:
        return 'broadcast_both'
    if lhs_reused:
        return 'broadcast_lhs'
    if rhs_reused:
        return 'broadcast_rhs'
    return 'matched_batch'


@dataclasses.dataclass(frozen=True)
class CoreStorage:
    """Describe a preset or exact signed-stride matrix core."""

    order: str = 'c'
    leading_dimension_gap: int = 0
    row_stride: int | None = None
    column_stride: int | None = None

    def __post_init__(self):
        if self.order not in _ORDERS:
            raise schema.SchemaError(
                f'core storage order must be one of {_ORDERS}')
        _require_int(
            self.leading_dimension_gap,
            'core storage leading_dimension_gap', 0)
        if self.order == 'custom':
            if self.leading_dimension_gap:
                raise schema.SchemaError(
                    'custom core storage must not have a leading-dimension '
                    'gap')
            _require_int(self.row_stride, 'custom core row_stride')
            _require_int(self.column_stride, 'custom core column_stride')
        elif self.row_stride is not None or self.column_stride is not None:
            raise schema.SchemaError(
                'preset core storage must not have custom strides')

    @classmethod
    def c_compact(cls):
        return cls(order='c')

    @classmethod
    def c_gap(cls, gap):
        return cls(order='c', leading_dimension_gap=gap)

    @classmethod
    def f_compact(cls):
        return cls(order='f')

    @classmethod
    def f_gap(cls, gap):
        return cls(order='f', leading_dimension_gap=gap)

    @classmethod
    def custom(cls, row_stride, column_stride):
        return cls(
            order='custom', row_stride=row_stride,
            column_stride=column_stride)

    @property
    def label(self):
        if self.order == 'custom':
            return (
                f'custom row stride {self.row_stride}, '
                f'column stride {self.column_stride}')
        order = self.order.upper()
        if self.leading_dimension_gap:
            return (
                f'{order} leading-dimension gap '
                f'{self.leading_dimension_gap}')
        return f'{order} compact'

    def resolve(self, rows, columns):
        rows = _require_int(rows, 'matrix rows', 1)
        columns = _require_int(columns, 'matrix columns', 1)
        if self.order == 'custom':
            strides = (self.row_stride, self.column_stride)
        else:
            gap = self.leading_dimension_gap
            strides = (columns + gap, 1) if self.order == 'c' \
                else (1, rows + gap)
        shape = (rows, columns)
        return shape, strides, _storage_span(shape, strides)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise schema.SchemaError('core storage must be an object')
        fields = ('order', 'leading_dimension_gap')
        if data.get('order') == 'custom':
            fields += ('row_stride', 'column_stride')
        data = _require_mapping(data, 'core storage', fields)
        return cls(
            order=data['order'],
            leading_dimension_gap=data['leading_dimension_gap'],
            row_stride=data.get('row_stride'),
            column_stride=data.get('column_stride'),
        )

    def to_dict(self):
        result = {
            'order': self.order,
            'leading_dimension_gap': self.leading_dimension_gap,
        }
        if self.order == 'custom':
            result.update({
                'row_stride': self.row_stride,
                'column_stride': self.column_stride,
            })
        return result


@dataclasses.dataclass(frozen=True)
class BatchStride:
    """Advance by an inner physical span, zero, or an exact element stride."""

    mode: str = 'auto'
    value: int | None = None

    def __post_init__(self):
        if self.mode not in _STRIDE_MODES:
            raise schema.SchemaError(
                f'batch stride mode must be one of {_STRIDE_MODES}')
        if self.mode == 'custom':
            _require_int(self.value, 'custom batch stride')
        elif self.value is not None:
            raise schema.SchemaError(
                f'{self.mode} batch stride must not have a value')

    @classmethod
    def auto(cls):
        return cls(mode='auto')

    @classmethod
    def zero(cls):
        return cls(mode='zero')

    @classmethod
    def custom(cls, value):
        return cls(mode='custom', value=value)

    def resolve(self, inner_span):
        if self.mode == 'auto':
            return inner_span
        if self.mode == 'zero':
            return 0
        return self.value

    @classmethod
    def from_dict(cls, data):
        data = _require_mapping(data, 'batch stride', ('mode', 'value'))
        return cls(mode=data['mode'], value=data['value'])

    def to_dict(self):
        return {'mode': self.mode, 'value': self.value}


@dataclasses.dataclass(frozen=True)
class BatchAxis:
    """Map one output batch extent onto the LHS and RHS operands.

    An extent-one operand is broadcast and therefore normalizes to stride 0.
    """

    output_extent: int
    lhs_extent: str = 'same'
    rhs_extent: str = 'same'
    lhs_stride: BatchStride = dataclasses.field(
        default_factory=BatchStride.auto)
    rhs_stride: BatchStride = dataclasses.field(
        default_factory=BatchStride.auto)

    def __post_init__(self):
        _require_int(self.output_extent, 'batch output_extent', 1)
        for name, extent in (
                ('lhs_extent', self.lhs_extent),
                ('rhs_extent', self.rhs_extent)):
            if extent not in _EXTENTS:
                raise schema.SchemaError(
                    f'batch {name} must be one of {_EXTENTS}')
        if self.lhs_extent == self.rhs_extent == 'one':
            raise schema.SchemaError(
                'a batch axis needs at least one operand extent same')
        for name, stride in (
                ('lhs_stride', self.lhs_stride),
                ('rhs_stride', self.rhs_stride)):
            if not isinstance(stride, BatchStride):
                raise schema.SchemaError(
                    f'batch {name} must be a BatchStride')
        if self.lhs_extent == 'one':
            object.__setattr__(self, 'lhs_stride', BatchStride.zero())
        if self.rhs_extent == 'one':
            object.__setattr__(self, 'rhs_stride', BatchStride.zero())

    def _operand_extent(self, side):
        extent = self.lhs_extent if side == 'lhs' else self.rhs_extent
        return self.output_extent if extent == 'same' else 1

    def _operand_stride(self, side):
        return self.lhs_stride if side == 'lhs' else self.rhs_stride

    @classmethod
    def from_dict(cls, data):
        fields = (
            'output_extent', 'lhs_extent', 'rhs_extent',
            'lhs_stride', 'rhs_stride',
        )
        data = _require_mapping(data, 'batch axis', fields)
        return cls(
            output_extent=data['output_extent'],
            lhs_extent=data['lhs_extent'],
            rhs_extent=data['rhs_extent'],
            lhs_stride=BatchStride.from_dict(data['lhs_stride']),
            rhs_stride=BatchStride.from_dict(data['rhs_stride']),
        )

    def to_dict(self):
        return {
            'output_extent': self.output_extent,
            'lhs_extent': self.lhs_extent,
            'rhs_extent': self.rhs_extent,
            'lhs_stride': self.lhs_stride.to_dict(),
            'rhs_stride': self.rhs_stride.to_dict(),
        }


@dataclasses.dataclass(frozen=True)
class ResolvedInputProfile:
    """Hold exact operand specs and readable facts for one M, K, N."""

    profile_id: str
    name: str
    m: int
    k: int
    n: int
    lhs: schema.OperandSpec
    rhs: schema.OperandSpec
    output_shape: tuple
    lhs_storage_span: int
    rhs_storage_span: int
    layout: str
    broadcast: str
    facts: tuple


@dataclasses.dataclass(frozen=True)
class ExpandedInput:
    """Expose one resolved profile as CollectionCell-compatible facts."""

    cell_id: str
    profile_id: str
    name: str
    m: int
    k: int
    n: int
    lhs: schema.OperandSpec
    rhs: schema.OperandSpec
    output_shape: tuple
    lhs_storage_span: int
    rhs_storage_span: int
    layout: str
    broadcast: str
    facts: tuple
    routes: tuple | None = None

    def collection_cell_kwargs(self):
        return {
            'cell_id': self.cell_id,
            'lhs': self.lhs,
            'rhs': self.rhs,
            'layout': self.layout,
            'broadcast': self.broadcast,
            'routes': self.routes,
        }


@dataclasses.dataclass(frozen=True)
class InputProfile:
    """Describe one M/K/N sweep recipe or one exact operand pair."""

    profile_id: str
    name: str
    lhs_core: CoreStorage = dataclasses.field(
        default_factory=CoreStorage.c_compact)
    rhs_core: CoreStorage = dataclasses.field(
        default_factory=CoreStorage.c_compact)
    batch_axes: tuple = ()
    exact_lhs: schema.OperandSpec | None = None
    exact_rhs: schema.OperandSpec | None = None

    def __post_init__(self):
        profile_id = _require_text(self.profile_id, 'profile_id')
        if not _PROFILE_ID_PATTERN.fullmatch(profile_id):
            raise schema.SchemaError(
                'profile_id must contain only lowercase letters, digits, '
                'underscores, or hyphens')
        _require_text(self.name, 'profile name')
        for name, core in (
                ('lhs_core', self.lhs_core),
                ('rhs_core', self.rhs_core)):
            if not isinstance(core, CoreStorage):
                raise schema.SchemaError(
                    f'profile {name} must be a CoreStorage')
        if not isinstance(self.batch_axes, (list, tuple)):
            raise schema.SchemaError('profile batch_axes must be an array')
        batch_axes = tuple(self.batch_axes)
        if len(batch_axes) > schema.MAX_OPERAND_RANK - 2:
            raise schema.SchemaError(
                'profile has too many batch axes for OperandSpec')
        if not all(isinstance(axis, BatchAxis) for axis in batch_axes):
            raise schema.SchemaError(
                'profile batch_axes must contain BatchAxis objects')
        object.__setattr__(self, 'batch_axes', batch_axes)
        if (self.exact_lhs is None) != (self.exact_rhs is None):
            raise schema.SchemaError(
                'exact input profile needs both A and B operands')
        if self.is_exact:
            if self.lhs_core != CoreStorage.c_compact() \
                    or self.rhs_core != CoreStorage.c_compact() \
                    or self.batch_axes:
                raise schema.SchemaError(
                    'exact input profile must not contain sweep storage')
            _exact_matmul_shape(self.exact_lhs, self.exact_rhs)

    @property
    def is_exact(self):
        return self.exact_lhs is not None

    @classmethod
    def exact(cls, profile_id, name, lhs, rhs):
        return cls(
            profile_id=profile_id, name=name,
            exact_lhs=lhs, exact_rhs=rhs)

    @classmethod
    def from_dict(cls, data):
        if isinstance(data, dict) and data.get('kind') == 'exact':
            fields = ('kind', 'profile_id', 'name', 'lhs', 'rhs')
            data = _require_mapping(data, 'exact input profile', fields)
            return cls.exact(
                profile_id=data['profile_id'], name=data['name'],
                lhs=schema.OperandSpec.from_dict(data['lhs']),
                rhs=schema.OperandSpec.from_dict(data['rhs']))
        fields = (
            'profile_id', 'name', 'lhs_core', 'rhs_core', 'batch_axes',
        )
        data = _require_mapping(data, 'input profile', fields)
        if not isinstance(data['batch_axes'], list):
            raise schema.SchemaError(
                'input profile batch_axes must be an array')
        return cls(
            profile_id=data['profile_id'],
            name=data['name'],
            lhs_core=CoreStorage.from_dict(data['lhs_core']),
            rhs_core=CoreStorage.from_dict(data['rhs_core']),
            batch_axes=tuple(
                BatchAxis.from_dict(item) for item in data['batch_axes']),
        )

    def to_dict(self):
        if self.is_exact:
            return {
                'kind': 'exact',
                'profile_id': self.profile_id,
                'name': self.name,
                'lhs': self.exact_lhs.to_dict(),
                'rhs': self.exact_rhs.to_dict(),
            }
        return {
            'profile_id': self.profile_id,
            'name': self.name,
            'lhs_core': self.lhs_core.to_dict(),
            'rhs_core': self.rhs_core.to_dict(),
            'batch_axes': [axis.to_dict() for axis in self.batch_axes],
        }

    def resolve(self, m, k, n):
        if self.is_exact:
            return self._resolve_exact()
        m = _require_int(m, 'M', 1)
        k = _require_int(k, 'K', 1)
        n = _require_int(n, 'N', 1)
        lhs, lhs_span, lhs_batch = self._resolve_operand(
            self.lhs_core, m, k, 'lhs')
        rhs, rhs_span, rhs_batch = self._resolve_operand(
            self.rhs_core, k, n, 'rhs')
        output_shape = tuple(
            axis.output_extent for axis in self.batch_axes) + (m, n)
        layout = (
            f'lhs_{_storage_id(self.lhs_core)}-'
            f'rhs_{_storage_id(self.rhs_core)}')
        broadcast = _broadcast_id(
            self.batch_axes, lhs_batch, rhs_batch)
        facts = (
            f'profile {self.name} ({self.profile_id})',
            f'M={m}, K={k}, N={n}',
            f'LHS core: {self.lhs_core.label}',
            f'RHS core: {self.rhs_core.label}',
            *(
                _batch_fact(index, axis, lhs_batch[index],
                            rhs_batch[index])
                for index, axis in enumerate(self.batch_axes)
            ),
            f'LHS shape={lhs.shape}, strides={lhs.strides}, '
            f'storage span={lhs_span}',
            f'RHS shape={rhs.shape}, strides={rhs.strides}, '
            f'storage span={rhs_span}',
            f'output shape={output_shape}',
        )
        return ResolvedInputProfile(
            profile_id=self.profile_id,
            name=self.name,
            m=m,
            k=k,
            n=n,
            lhs=lhs,
            rhs=rhs,
            output_shape=output_shape,
            lhs_storage_span=lhs_span,
            rhs_storage_span=rhs_span,
            layout=layout,
            broadcast=broadcast,
            facts=facts,
        )

    def _resolve_exact(self):
        lhs = self.exact_lhs
        rhs = self.exact_rhs
        m_value, k_value, n_value, output_shape = \
            _exact_matmul_shape(lhs, rhs)
        output_batch = output_shape[:-2]
        lhs_span = _storage_span(lhs.shape, lhs.strides)
        rhs_span = _storage_span(rhs.shape, rhs.strides)
        broadcast = _exact_broadcast_id(lhs, rhs, output_batch)
        facts = (
            f'profile {self.name} ({self.profile_id})',
            'exact case: one collection cell',
            f'M={m_value}, K={k_value}, N={n_value}',
            f'LHS shape={lhs.shape}, strides={lhs.strides}, '
            f'storage span={lhs_span}',
            f'RHS shape={rhs.shape}, strides={rhs.strides}, '
            f'storage span={rhs_span}',
            f'output shape={output_shape}',
        )
        return ResolvedInputProfile(
            profile_id=self.profile_id,
            name=self.name,
            m=m_value,
            k=k_value,
            n=n_value,
            lhs=lhs,
            rhs=rhs,
            output_shape=output_shape,
            lhs_storage_span=lhs_span,
            rhs_storage_span=rhs_span,
            layout='lhs_exact-rhs_exact',
            broadcast=broadcast,
            facts=facts,
        )

    def _resolve_operand(self, core, rows, columns, side):
        core_shape, core_strides, span = core.resolve(rows, columns)
        extents = []
        strides = []
        resolved = [None] * len(self.batch_axes)
        for index in range(len(self.batch_axes) - 1, -1, -1):
            axis = self.batch_axes[index]
            extent = axis._operand_extent(side)
            stride = axis._operand_stride(side).resolve(span)
            resolved[index] = (extent, stride)
            span += (extent - 1) * abs(stride)
        for extent, stride in resolved:
            extents.append(extent)
            strides.append(stride)
        spec = schema.OperandSpec(
            shape=(*extents, *core_shape),
            strides=(*strides, *core_strides),
        )
        return spec, span, tuple(resolved)


def _storage_id(storage):
    if storage.order == 'custom':
        return (
            f'custom_r{storage.row_stride}_c{storage.column_stride}')
    if storage.leading_dimension_gap:
        return f'{storage.order}_gap{storage.leading_dimension_gap}'
    return f'{storage.order}_compact'


def _broadcast_id(batch_axes, lhs_batch, rhs_batch):
    if not batch_axes:
        return 'matrix'
    lhs_reused = _batch_operand_reused(batch_axes, lhs_batch)
    rhs_reused = _batch_operand_reused(batch_axes, rhs_batch)
    if lhs_reused and rhs_reused:
        return 'broadcast_both'
    if lhs_reused:
        return 'broadcast_lhs'
    if rhs_reused:
        return 'broadcast_rhs'
    return 'matched_batch'


def _batch_operand_reused(batch_axes, resolved_batch):
    return any(
        axis.output_extent > 1
        and (extent == 1 or (extent > 1 and stride == 0))
        for axis, (extent, stride) in zip(batch_axes, resolved_batch)
    )


def _batch_fact(index, axis, lhs_resolved, rhs_resolved):
    lhs_extent, lhs_stride = lhs_resolved
    rhs_extent, rhs_stride = rhs_resolved
    return (
        f'batch axis {index}: output extent={axis.output_extent}, '
        f'LHS extent={lhs_extent}, stride={lhs_stride} '
        f'({axis.lhs_stride.mode}), RHS extent={rhs_extent}, '
        f'stride={rhs_stride} ({axis.rhs_stride.mode})')


def unbatched_profile(
        profile_id='unbatched', name='Unbatched matrices',
        lhs_core=None, rhs_core=None):
    return InputProfile(
        profile_id=profile_id,
        name=name,
        lhs_core=lhs_core or CoreStorage.c_compact(),
        rhs_core=rhs_core or CoreStorage.c_compact(),
    )


def matched_batch_profile(
        batch_size, profile_id='matched_batch', name='Matched batch',
        lhs_core=None, rhs_core=None):
    return InputProfile(
        profile_id=profile_id,
        name=name,
        lhs_core=lhs_core or CoreStorage.c_compact(),
        rhs_core=rhs_core or CoreStorage.c_compact(),
        batch_axes=(BatchAxis(output_extent=batch_size),),
    )


def reuse_lhs_profile(
        batch_size, profile_id='reuse_lhs', name='Reuse LHS',
        lhs_core=None, rhs_core=None):
    return InputProfile(
        profile_id=profile_id,
        name=name,
        lhs_core=lhs_core or CoreStorage.c_compact(),
        rhs_core=rhs_core or CoreStorage.c_compact(),
        batch_axes=(BatchAxis(
            output_extent=batch_size,
            lhs_extent='one',
            rhs_extent='same',
            lhs_stride=BatchStride.zero(),
        ),),
    )


def reuse_rhs_profile(
        batch_size, profile_id='reuse_rhs', name='Reuse RHS',
        lhs_core=None, rhs_core=None):
    return InputProfile(
        profile_id=profile_id,
        name=name,
        lhs_core=lhs_core or CoreStorage.c_compact(),
        rhs_core=rhs_core or CoreStorage.c_compact(),
        batch_axes=(BatchAxis(
            output_extent=batch_size,
            lhs_extent='same',
            rhs_extent='one',
            rhs_stride=BatchStride.zero(),
        ),),
    )


def expand_profiles(profiles, m_values, k_values, n_values):
    """Expand sweeps across M/K/N and include each exact case once."""

    if not isinstance(profiles, (list, tuple)) or not profiles:
        raise schema.SchemaError('profiles must be a non-empty array')
    profiles = tuple(profiles)
    if not all(isinstance(profile, InputProfile) for profile in profiles):
        raise schema.SchemaError('profiles must contain InputProfile objects')
    for attribute in ('profile_id', 'name'):
        values = tuple(getattr(profile, attribute) for profile in profiles)
        if len(values) != len(set(values)):
            raise schema.SchemaError(
                f'profile {attribute} values must be unique')
    recipes = {}
    for profile in profiles:
        recipe = (
            ('exact', profile.exact_lhs, profile.exact_rhs)
            if profile.is_exact else
            ('sweep', profile.lhs_core, profile.rhs_core,
             profile.batch_axes)
        )
        previous = recipes.get(recipe)
        if previous is not None:
            raise schema.SchemaError(
                f'profiles {previous!r} and {profile.name!r} describe the '
                'same input recipe')
        recipes[recipe] = profile.name
    dimensions = tuple(
        _dimension_values(values, name)
        for values, name in (
            (m_values, 'M'), (k_values, 'K'), (n_values, 'N'))
    )
    expanded = []
    exact_inputs = {}

    def append(profile, m, k, n):
        resolved = profile.resolve(m, k, n)
        input_key = resolved.lhs, resolved.rhs
        previous = exact_inputs.get(input_key)
        if previous is not None:
            raise schema.SchemaError(
                f'profiles {previous!r} and {profile.name!r} '
                'produce the same exact A/B input')
        exact_inputs[input_key] = profile.name
        expanded.append(ExpandedInput(
            cell_id=(
                f'{profile.profile_id}-exact'
                if profile.is_exact else
                f'{profile.profile_id}-m{m}-k{k}-n{n}'),
            profile_id=resolved.profile_id,
            name=resolved.name,
            m=resolved.m,
            k=resolved.k,
            n=resolved.n,
            lhs=resolved.lhs,
            rhs=resolved.rhs,
            output_shape=resolved.output_shape,
            lhs_storage_span=resolved.lhs_storage_span,
            rhs_storage_span=resolved.rhs_storage_span,
            layout=resolved.layout,
            broadcast=resolved.broadcast,
            facts=resolved.facts,
        ))

    combinations = itertools.product(*dimensions)
    first_dimensions = next(combinations)
    for profile in profiles:
        append(profile, *first_dimensions)

    sweep_profiles = tuple(
        profile for profile in profiles if not profile.is_exact)
    if not sweep_profiles:
        return tuple(expanded)
    for m, k, n in combinations:
        for profile in sweep_profiles:
            append(profile, m, k, n)
    return tuple(expanded)


def _dimension_values(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise schema.SchemaError(f'{name} values must be a non-empty array')
    result = tuple(
        _require_int(value, f'{name} values[{index}]', 1)
        for index, value in enumerate(values)
    )
    if len(result) != len(set(result)):
        raise schema.SchemaError(f'{name} values must be unique')
    return result


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
