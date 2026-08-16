# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Route inspection and forced matmul adapters."""

import dataclasses


class EngineUnavailableError(RuntimeError):
    """Report a build without the private benchmark binding."""


@dataclasses.dataclass(frozen=True)
class RouteDescriptor:
    """Describe one input-eligible native route and its packing recipe."""

    name: str
    selected_by_auto: bool
    eager_pack_lhs: bool
    eager_pack_rhs: bool
    scratch_pack_lhs: bool
    scratch_pack_rhs: bool

    def packing_dict(self):
        return {
            'eager_lhs': self.eager_pack_lhs,
            'eager_rhs': self.eager_pack_rhs,
            'scratch_lhs': self.scratch_pack_lhs,
            'scratch_rhs': self.scratch_pack_rhs,
        }


class SolvconCase:
    """Hold native operands and their eligible route descriptions."""

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        route_names = set()
        descriptors = []
        for route in lhs.matmul_routes(rhs):
            descriptor = RouteDescriptor(
                name=route.kernel,
                selected_by_auto=route.selected_by_auto,
                eager_pack_lhs=route.eager_pack_lhs,
                eager_pack_rhs=route.eager_pack_rhs,
                scratch_pack_lhs=route.scratch_pack_lhs,
                scratch_pack_rhs=route.scratch_pack_rhs,
            )
            if descriptor.name in route_names:
                raise RuntimeError(
                    f'duplicate native route: {descriptor.name}')
            route_names.add(descriptor.name)
            descriptors.append(descriptor)
        self.routes = tuple(descriptors)

    def execute_auto(self):
        return self.lhs.matmul(self.rhs).ndarray

    def execute_route(self, name):
        result = self.lhs.matmul(self.rhs, kernel=name)
        return result.ndarray


class SolvconRouteEngine:
    """Adapt selectable matmul kernels without Qt dependencies."""

    _ARRAY_CLASSES = {
        'float32': 'SimpleArrayFloat32',
        'float64': 'SimpleArrayFloat64',
        'complex64': 'SimpleArrayComplex64',
        'complex128': 'SimpleArrayComplex128',
    }

    def prepare(self, lhs, rhs, dtype):
        import solvcon as sc

        array_class = getattr(sc, self._ARRAY_CLASSES[dtype])
        native_lhs = array_class(array=lhs)
        native_rhs = array_class(array=rhs)
        required = ('matmul_routes', 'matmul')
        missing = [name for name in required
                   if not hasattr(native_lhs, name)]
        if missing:
            joined = ', '.join(missing)
            raise EngineUnavailableError(
                f'this build lacks selectable matmul methods: {joined}')
        return SolvconCase(native_lhs, native_rhs)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
