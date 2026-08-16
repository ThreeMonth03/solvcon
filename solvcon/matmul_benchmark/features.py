# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Safe dynamic features and observation projection helpers."""

import ast
import math
import operator

import numpy as np

from . import schema


class FeatureError(ValueError):
    """Report an invalid feature definition or evaluation."""


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS = {
    'abs': abs,
    'ceil': math.ceil,
    'floor': math.floor,
    'ln': math.log,
    'log2': math.log2,
    'log10': math.log10,
    'max': max,
    'min': min,
    'sqrt': math.sqrt,
}
MAX_BATCH_AXES = schema.MAX_OPERAND_RANK - 2
_BATCH_EXTENT_NAMES = tuple(
    f'batch_extent_{axis}' for axis in range(MAX_BATCH_AXES))
_BATCH_STRIDE_NAMES = tuple(
    f'{operand}_batch_stride_{axis}'
    for operand in ('lhs', 'rhs')
    for axis in range(MAX_BATCH_AXES)
)
_BASE_NAMES = (
    'M',
    'K',
    'N',
    'batch_count',
    'lhs_elements',
    'rhs_elements',
    'output_elements',
    'dtype_bytes',
    'packing_bytes',
    'useful_bytes',
    'winner_latency_ns',
    'winner_margin',
    'batch_rank',
    'lhs_rank',
    'rhs_rank',
    'lhs_row_stride',
    'lhs_inner_stride',
    'rhs_inner_stride',
    'rhs_column_stride',
) + _BATCH_EXTENT_NAMES + _BATCH_STRIDE_NAMES
_DEFAULT_EXPRESSIONS = {
    'work': 'M * K * N * batch_count',
    'output_size': 'M * N * batch_count',
    'lhs_size': 'lhs_elements',
    'rhs_size': 'rhs_elements',
    'aspect_mn': 'log2(M / N)',
    'aspect_mk': 'log2(M / K)',
    'aspect_nk': 'log2(N / K)',
    'packing_ratio': 'packing_bytes / useful_bytes',
}
MAX_FEATURE_INTEGER_BITS = 4096


def _logical_elements(shape):
    return math.prod(shape)


def _scratch_pack_count(operand, batch_shape):
    if any(extent == 0 for extent in batch_shape):
        return 0
    if not batch_shape:
        return 1

    shape = operand['shape']
    strides = operand['strides']
    operand_batch_shape = [] if len(shape) == 1 else shape[:-2]
    operand_batch_strides = [] if len(shape) == 1 else strides[:-2]
    missing = len(batch_shape) - len(operand_batch_shape)
    mapped_strides = [0] * missing
    for extent, target, stride in zip(
            operand_batch_shape, batch_shape[missing:],
            operand_batch_strides):
        mapped_strides.append(stride if extent == target else 0)

    copies = 1
    reset = 0
    for axis in range(len(batch_shape) - 1, -1, -1):
        extent = batch_shape[axis]
        transition = mapped_strides[axis] - reset
        preceding_size = math.prod(batch_shape[:axis])
        if transition != 0:
            copies += preceding_size * max(0, extent - 1)
        reset += mapped_strides[axis] * max(0, extent - 1)
    return copies


def _mapped_batch_strides(operand, batch_shape):
    shape = operand['shape']
    strides = operand['strides']
    operand_shape = [] if len(shape) == 1 else shape[:-2]
    operand_strides = [] if len(shape) == 1 else strides[:-2]
    missing = len(batch_shape) - len(operand_shape)
    mapped = [0] * missing
    for extent, target, stride in zip(
            operand_shape, batch_shape[missing:], operand_strides):
        mapped.append(stride if extent == target else 0)
    return mapped + [0] * (MAX_BATCH_AXES - len(mapped))


def _feature_context(observation):
    contraction = observation['contraction']
    lhs_elements = _logical_elements(observation['lhs']['shape'])
    rhs_elements = _logical_elements(observation['rhs']['shape'])
    output_elements = _logical_elements(contraction['output_shape'])
    dtype_bytes = np.dtype(observation['dtype']).itemsize
    winner = observation.get('winner')
    route = observation.get('routes', {}).get(winner, {})
    packing = route.get('packing') or {}
    batch_count = contraction['batch_count']
    packing_bytes = 0
    if packing.get('eager_lhs'):
        packing_bytes += lhs_elements * dtype_bytes
    if packing.get('eager_rhs'):
        packing_bytes += rhs_elements * dtype_bytes
    if packing.get('scratch_lhs'):
        packing_bytes += (contraction['m'] * contraction['k']
                          * _scratch_pack_count(
                              observation['lhs'],
                              contraction['batch_shape'])
                          * dtype_bytes)
    if packing.get('scratch_rhs'):
        packing_bytes += (contraction['k'] * contraction['n']
                          * _scratch_pack_count(
                              observation['rhs'],
                              contraction['batch_shape'])
                          * dtype_bytes)
    useful_bytes = (
        lhs_elements + rhs_elements + output_elements) * dtype_bytes
    timing = route.get('timing') or {}
    winner_margin = observation.get('winner_margin')
    if winner_margin is None:
        winner_margin = math.nan
    batch_shape = contraction['batch_shape']
    lhs_strides = observation['lhs']['strides']
    rhs_strides = observation['rhs']['strides']
    context = {
        'M': contraction['m'],
        'K': contraction['k'],
        'N': contraction['n'],
        'batch_count': batch_count,
        'lhs_elements': lhs_elements,
        'rhs_elements': rhs_elements,
        'output_elements': output_elements,
        'dtype_bytes': dtype_bytes,
        'packing_bytes': packing_bytes,
        'useful_bytes': useful_bytes,
        'winner_latency_ns': timing.get('median_ns', math.nan),
        'winner_margin': winner_margin,
        'batch_rank': len(batch_shape),
        'lhs_rank': len(observation['lhs']['shape']),
        'rhs_rank': len(observation['rhs']['shape']),
        'lhs_row_stride': lhs_strides[-2]
        if len(lhs_strides) > 1 else 0,
        'lhs_inner_stride': lhs_strides[-1],
        'rhs_inner_stride': rhs_strides[-2]
        if len(rhs_strides) > 1 else rhs_strides[-1],
        'rhs_column_stride': rhs_strides[-1]
        if len(rhs_strides) > 1 else 0,
    }
    padded_batch_shape = list(batch_shape) + [1] * (
        MAX_BATCH_AXES - len(batch_shape))
    lhs_batch_strides = _mapped_batch_strides(
        observation['lhs'], batch_shape)
    rhs_batch_strides = _mapped_batch_strides(
        observation['rhs'], batch_shape)
    for axis in range(MAX_BATCH_AXES):
        context[f'batch_extent_{axis}'] = padded_batch_shape[axis]
        context[f'lhs_batch_stride_{axis}'] = lhs_batch_strides[axis]
        context[f'rhs_batch_stride_{axis}'] = rhs_batch_strides[axis]
    return context


def _checked_number(value, description, allow_nan=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureError(f'{description} is not numeric')
    if isinstance(value, int):
        if value.bit_length() > MAX_FEATURE_INTEGER_BITS:
            raise FeatureError(f'{description} exceeds the integer size limit')
    elif not math.isfinite(value) and not (allow_nan and math.isnan(value)):
        raise FeatureError(f'{description} is not finite')
    return value


class FeatureRegistry:
    """Evaluate named numeric expressions without Python eval."""

    def __init__(self, expressions=None):
        self._expressions = {}
        for name, expression in _DEFAULT_EXPRESSIONS.items():
            self.register_expression(name, expression)
        if expressions:
            for name, expression in expressions.items():
                self.register_expression(name, expression)

    def names(self):
        return tuple(sorted(set(_BASE_NAMES) | set(self._expressions)))

    def definitions(self):
        return {
            name: definition['expression']
            for name, definition in sorted(self._expressions.items())
        }

    def register_expression(self, name, expression):
        if not isinstance(name, str) or not name.isidentifier():
            raise FeatureError('feature name must be a Python identifier')
        if name.startswith('_') or name in _FUNCTIONS or name in _BASE_NAMES:
            raise FeatureError(f'reserved feature name: {name!r}')
        if not isinstance(expression, str) or not expression.strip():
            raise FeatureError('feature expression must be a non-empty string')
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as exc:
            raise FeatureError(f'invalid feature expression: {exc.msg}') \
                from exc
        if sum(1 for _ in ast.walk(tree)) > 128:
            raise FeatureError('feature expression is too complex')
        self._validate_node(tree)
        self._expressions[name] = {
            'expression': expression,
            'tree': tree,
        }

    def _validate_node(self, node):
        if isinstance(node, ast.Expression):
            self._validate_node(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                    node.value, (int, float)):
                raise FeatureError('only numeric constants are allowed')
            _checked_number(node.value, 'feature constant')
        elif isinstance(node, ast.Name):
            if node.id.startswith('_'):
                raise FeatureError('private names are not allowed')
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY_OPERATORS:
                raise FeatureError('unsupported binary operator')
            self._validate_node(node.left)
            self._validate_node(node.right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_OPERATORS:
                raise FeatureError('unsupported unary operator')
            self._validate_node(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) \
                    or node.func.id not in _FUNCTIONS:
                raise FeatureError('unsupported feature function')
            if node.keywords or len(node.args) > 8:
                raise FeatureError('feature calls accept positional arguments')
            for argument in node.args:
                self._validate_node(argument)
        else:
            raise FeatureError(
                f'unsupported feature syntax: {type(node).__name__}')

    def evaluate(self, name, observation):
        context = _feature_context(observation)
        if name in context:
            return _checked_number(
                context[name], f'feature {name!r}', allow_nan=True)
        if name not in self._expressions:
            raise FeatureError(f'unknown feature: {name!r}')
        return self._evaluate_named(name, context, set())

    def _evaluate_named(self, name, context, active):
        if name in active:
            raise FeatureError(f'cyclic feature definition at {name!r}')
        definition = self._expressions.get(name)
        if definition is None:
            if name in context:
                return context[name]
            raise FeatureError(f'unknown name in feature expression: {name!r}')
        active.add(name)
        try:
            value = self._evaluate_node(
                definition['tree'].body, context, active)
        except (ArithmeticError, TypeError, ValueError, OverflowError) as exc:
            raise FeatureError(
                f'cannot evaluate feature {name!r}: {exc}') from exc
        finally:
            active.remove(name)
        return _checked_number(value, f'feature {name!r}')

    def _evaluate_node(self, node, context, active):
        if isinstance(node, ast.Constant):
            return _checked_number(node.value, 'feature constant')
        if isinstance(node, ast.Name):
            if node.id in context:
                return _checked_number(
                    context[node.id], f'feature input {node.id!r}')
            return self._evaluate_named(node.id, context, active)
        if isinstance(node, ast.BinOp):
            left = self._evaluate_node(node.left, context, active)
            right = self._evaluate_node(node.right, context, active)
            if isinstance(node.op, ast.Pow) and abs(right) > 16:
                raise FeatureError('feature exponent magnitude exceeds 16')
            if isinstance(node.op, ast.Pow) \
                    and isinstance(left, int) and isinstance(right, int) \
                    and right >= 0 \
                    and left.bit_length() * right \
                    > MAX_FEATURE_INTEGER_BITS:
                raise FeatureError('feature power exceeds the size limit')
            value = _BINARY_OPERATORS[type(node.op)](left, right)
            return _checked_number(value, 'feature intermediate')
        if isinstance(node, ast.UnaryOp):
            value = self._evaluate_node(node.operand, context, active)
            result = _UNARY_OPERATORS[type(node.op)](value)
            return _checked_number(result, 'feature intermediate')
        if isinstance(node, ast.Call):
            arguments = [
                self._evaluate_node(argument, context, active)
                for argument in node.args
            ]
            value = _FUNCTIONS[node.func.id](*arguments)
            return _checked_number(value, 'feature function result')
        raise FeatureError(f'unsupported feature node: {type(node).__name__}')


def _constraint_value(name, observation, registry):
    if name in observation:
        return observation[name]
    return registry.evaluate(name, observation)


def slice_observations(observations, registry, constraints,
                       relative_tolerance=0.0):
    """Select categorical matches or inclusive numeric feature ranges."""

    selected = []
    for observation in observations:
        matches = True
        for name, expected in constraints.items():
            actual = _constraint_value(name, observation, registry)
            if isinstance(expected, (tuple, list)) and len(expected) == 2:
                matches = expected[0] <= actual <= expected[1]
            elif isinstance(actual, (int, float)) and isinstance(
                    expected, (int, float)):
                matches = math.isclose(
                    actual, expected, rel_tol=relative_tolerance,
                    abs_tol=0.0)
            else:
                matches = actual == expected
            if not matches:
                break
        if matches:
            selected.append(observation)
    return selected


def project_observations(observations, registry, axes, constraints=None):
    """Project observations onto arbitrary registered feature axes."""

    axes = tuple(axes)
    if not 1 <= len(axes) <= 3:
        raise FeatureError('a projection needs one to three axes')
    if constraints:
        observations = slice_observations(
            observations, registry, constraints)
    projected = []
    for observation in observations:
        coordinates = [
            registry.evaluate(name, observation) for name in axes
        ]
        projected.append({
            'id': observation.get('id'),
            'coordinates': coordinates,
            'winner': observation.get('winner'),
            'winner_margin': observation.get('winner_margin'),
            'observation': observation,
        })
    return projected


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
