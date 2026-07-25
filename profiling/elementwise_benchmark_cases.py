# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import dataclasses
import itertools

import numpy as np


ARITHMETIC_OPERATIONS = ("add", "sub", "mul", "div")
COMPARISON_OPERATIONS = ("eq", "ne", "lt", "le", "gt", "ge")
ALL_OPERATIONS = ARITHMETIC_OPERATIONS + COMPARISON_OPERATIONS

ALL_DTYPES = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "complex64",
    "complex128",
)

INPUT_LAYOUTS = (
    "c",
    "permuted",
    "negative-inner",
    "negative-outer",
    "step2-inner",
    "step2-outer",
    "offset",
    "zero-inner",
    "zero-outer",
)

DESTINATION_LAYOUTS = tuple(
    layout for layout in INPUT_LAYOUTS
    if not layout.startswith("zero-")
)


@dataclasses.dataclass(frozen=True)
class Topology:
    name: str
    lhs_shape: tuple
    rhs_shape: tuple | None
    result_shape: tuple | None
    numpy_valid: bool = True
    rhs_scalar: bool = False

    @property
    def inplace_valid(self):
        return self.numpy_valid and self.lhs_shape == self.result_shape

    @property
    def legacy_shape_supported(self):
        return self.rhs_scalar or self.lhs_shape == self.rhs_shape


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    catalog: str
    size: int
    topology: Topology
    operation: str
    dtype: str
    mode: str
    lhs_layout: str
    rhs_layout: str
    alias: str = "none"
    value_pattern: str = "finite"

    @property
    def identifier(self):
        fields = (
            self.mode,
            self.operation,
            self.dtype,
            f"n{self.size}",
            self.topology.name,
            self.lhs_layout,
            self.rhs_layout,
            self.alias,
            self.value_pattern,
        )
        return "/".join(fields)


@dataclasses.dataclass
class LayoutData:
    name: str
    view: np.ndarray
    storage: np.ndarray

    def descriptor(self):
        itemsize = self.view.dtype.itemsize
        storage_ptr = self.storage.__array_interface__["data"][0]
        view_ptr = self.view.__array_interface__["data"][0]
        return {
            "layout": self.name,
            "shape": list(self.view.shape),
            "strides": [
                stride // itemsize for stride in self.view.strides
            ],
            "base_offset": (view_ptr - storage_ptr) // itemsize,
            "storage_elements": int(self.storage.size),
            "c_contiguous": bool(self.view.flags.c_contiguous),
            "f_contiguous": bool(self.view.flags.f_contiguous),
            "writeable": bool(self.view.flags.writeable),
        }

    def touched_storage_offsets(self):
        if self.view.size == 0:
            return set()
        itemsize = self.view.dtype.itemsize
        storage_ptr = self.storage.__array_interface__["data"][0]
        view_ptr = self.view.__array_interface__["data"][0]
        origin = (view_ptr - storage_ptr) // itemsize
        strides = tuple(
            stride // itemsize for stride in self.view.strides
        )
        offsets = set()
        for index in np.ndindex(self.view.shape):
            offset = origin + sum(
                coordinate * stride
                for coordinate, stride in zip(index, strides)
            )
            offsets.add(offset)
        return offsets


def dimensions(size):
    return max(1, size // 2 + 1), size, 2, 3


def make_topologies(size, include_invalid=True):
    rows, columns, batch, channels = dimensions(size)
    topologies = [
        Topology(
            "same-1d",
            (columns,),
            (columns,),
            (columns,),
        ),
        Topology(
            "same-2d",
            (rows, columns),
            (rows, columns),
            (rows, columns),
        ),
        Topology(
            "python-scalar",
            (rows, columns),
            None,
            (rows, columns),
            rhs_scalar=True,
        ),
        Topology(
            "rhs-singleton-array",
            (rows, columns),
            (1,),
            (rows, columns),
        ),
        Topology(
            "lhs-singleton-array",
            (1,),
            (rows, columns),
            (rows, columns),
        ),
        Topology(
            "rhs-vector",
            (rows, columns),
            (columns,),
            (rows, columns),
        ),
        Topology(
            "lhs-vector",
            (columns,),
            (rows, columns),
            (rows, columns),
        ),
        Topology(
            "rhs-row",
            (rows, columns),
            (1, columns),
            (rows, columns),
        ),
        Topology(
            "rhs-column",
            (rows, columns),
            (rows, 1),
            (rows, columns),
        ),
        Topology(
            "lhs-row",
            (1, columns),
            (rows, columns),
            (rows, columns),
        ),
        Topology(
            "lhs-column",
            (rows, 1),
            (rows, columns),
            (rows, columns),
        ),
        Topology(
            "outer",
            (rows, 1),
            (1, columns),
            (rows, columns),
        ),
        Topology(
            "outer-reversed",
            (1, columns),
            (rows, 1),
            (rows, columns),
        ),
        Topology(
            "mixed-rank",
            (batch, rows, 1),
            (columns,),
            (batch, rows, columns),
        ),
        Topology(
            "mixed-rank-reversed",
            (columns,),
            (batch, rows, 1),
            (batch, rows, columns),
        ),
        Topology(
            "rhs-leading-batch",
            (batch, rows, columns),
            (1, rows, columns),
            (batch, rows, columns),
        ),
        Topology(
            "lhs-leading-batch",
            (1, rows, columns),
            (batch, rows, columns),
            (batch, rows, columns),
        ),
        Topology(
            "crossed-batch",
            (batch, 1, rows, 1),
            (1, channels, 1, columns),
            (batch, channels, rows, columns),
        ),
        Topology(
            "all-singleton-rhs",
            (batch, rows, columns),
            (1, 1, 1),
            (batch, rows, columns),
        ),
        Topology(
            "all-singleton-lhs",
            (1, 1, 1),
            (batch, rows, columns),
            (batch, rows, columns),
        ),
    ]
    if size == 0:
        topologies.extend([
            Topology(
                "empty-leading",
                (0, 3),
                (1, 3),
                (0, 3),
            ),
            Topology(
                "empty-leading-reversed",
                (1, 3),
                (0, 3),
                (0, 3),
            ),
            Topology(
                "empty-inner",
                (3, 0),
                (3, 1),
                (3, 0),
            ),
            Topology(
                "empty-inner-reversed",
                (3, 1),
                (3, 0),
                (3, 0),
            ),
            Topology(
                "empty-crossed",
                (0, 1, 3),
                (1, 2, 1),
                (0, 2, 3),
            ),
            Topology(
                "empty-crossed-reversed",
                (1, 2, 1),
                (0, 1, 3),
                (0, 2, 3),
            ),
        ])
    if include_invalid:
        invalid_columns = max(2, columns)
        topologies.extend([
            Topology(
                "invalid-trailing",
                (rows, invalid_columns),
                (rows, invalid_columns + 1),
                None,
                numpy_valid=False,
            ),
            Topology(
                "invalid-rank-alignment",
                (batch, 2, invalid_columns),
                (channels, invalid_columns + 1),
                None,
                numpy_valid=False,
            ),
        ])
    return tuple(topologies)


def repeated_values(shape, values, dtype):
    dtype = np.dtype(dtype)
    size = int(np.prod(shape, dtype="int64"))
    if size == 0:
        return np.empty(shape, dtype=dtype.name)
    return np.resize(
        np.asarray(values, dtype=dtype.name), size
    ).reshape(shape)


def make_values(shape, dtype, seed, pattern="finite"):
    dtype = np.dtype(dtype)
    if pattern == "ieee":
        if np.issubdtype(dtype, np.complexfloating):
            values = (
                0,
                -0.0,
                complex(np.inf, 1),
                complex(-np.inf, -1),
                complex(np.nan, 2),
                complex(3, np.nan),
            )
        else:
            values = (
                0.0,
                -0.0,
                np.inf,
                -np.inf,
                np.nan,
                np.finfo(dtype).tiny,
                -np.finfo(dtype).tiny,
            )
        return repeated_values(shape, values, dtype)
    if pattern == "integer-boundary":
        information = np.iinfo(dtype)
        values = [information.min, information.max, 0, 1]
        if np.issubdtype(dtype, np.signedinteger):
            values.append(-1)
        else:
            values.append(information.max - 1)
        return repeated_values(shape, values, dtype)
    if pattern != "finite":
        raise ValueError(f"unsupported value pattern: {pattern}")
    rng = np.random.default_rng(seed)
    if dtype == np.bool_:
        return rng.integers(
            0, 2, size=shape, dtype="int8"
        ).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        if np.issubdtype(dtype, np.signedinteger):
            values = rng.integers(
                -7, 8, size=shape, dtype=dtype.name
            )
            values[values == 0] = 1
            return values
        return rng.integers(
            1, 8, size=shape, dtype=dtype.name
        )
    if np.issubdtype(dtype, np.complexfloating):
        real = rng.random(shape) * 6 - 3
        imaginary = rng.random(shape) * 2 - 1
        return (real + imaginary * 1j).astype(dtype)
    return (rng.random(shape) * 6 - 3).astype(dtype)


def sentinel_for(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return False
    if np.issubdtype(dtype, np.complexfloating):
        return dtype.type(17 + 3j)
    return dtype.type(17)


def reverse_layout(values, axis, name):
    guard = values.size
    storage = np.full(
        values.size + 2 * guard,
        sentinel_for(values.dtype),
        dtype=values.dtype.name,
    )
    forward = storage[guard:guard + values.size].reshape(values.shape)
    forward[...] = np.flip(values, axis=axis)
    return LayoutData(name, np.flip(forward, axis=axis), storage)


def step_layout(values, axis, name):
    storage_shape = list(values.shape)
    storage_shape[axis] *= 2
    storage = np.full(
        storage_shape,
        sentinel_for(values.dtype),
        dtype=values.dtype.name,
    )
    selection = [slice(None)] * values.ndim
    selection[axis] = slice(None, None, 2)
    view = storage[tuple(selection)]
    view[...] = values
    return LayoutData(name, view, storage)


def offset_layout(values):
    storage_shape = tuple(length + 2 for length in values.shape)
    storage = np.full(
        storage_shape,
        sentinel_for(values.dtype),
        dtype=values.dtype.name,
    )
    selection = tuple(slice(1, length + 1) for length in values.shape)
    view = storage[selection]
    view[...] = values
    return LayoutData("offset", view, storage)


def permuted_layout(values):
    if values.ndim < 2:
        storage = values.copy(order="C")
        return LayoutData("permuted", storage, storage)
    axes = tuple(reversed(range(values.ndim)))
    storage = np.transpose(values, axes=axes).copy(order="C")
    view = np.transpose(storage, axes=axes)
    return LayoutData("permuted", view, storage)


def zero_layout(values, axis, name):
    storage = values.copy(order="C")
    strides = list(storage.strides)
    strides[axis] = 0
    view = np.lib.stride_tricks.as_strided(
        storage,
        shape=values.shape,
        strides=tuple(strides),
        writeable=True,
    )
    return LayoutData(name, view, storage)


def make_layout(values, layout):
    storage = values.copy(order="C")
    if layout == "c" or values.ndim == 0 or values.size == 0:
        return LayoutData(layout, storage, storage)
    if layout == "permuted":
        return permuted_layout(values)
    if layout == "negative-inner":
        return reverse_layout(values, -1, layout)
    if layout == "negative-outer":
        return reverse_layout(values, 0, layout)
    if layout == "step2-inner":
        return step_layout(values, -1, layout)
    if layout == "step2-outer":
        return step_layout(values, 0, layout)
    if layout == "offset":
        return offset_layout(values)
    if layout == "zero-inner":
        return zero_layout(values, -1, layout)
    if layout == "zero-outer":
        return zero_layout(values, 0, layout)
    raise ValueError(f"unsupported layout: {layout}")


def operation_supported(operation, dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        return operation not in ("lt", "le", "gt", "ge")
    if dtype == np.bool_:
        return operation not in ("sub", "div")
    return True


def value_pattern_supported(pattern, operation, dtype):
    dtype = np.dtype(dtype)
    if pattern == "finite":
        return True
    if pattern == "ieee":
        return (
            np.issubdtype(dtype, np.floating)
            or np.issubdtype(dtype, np.complexfloating)
        )
    if pattern == "integer-boundary":
        return (
            np.issubdtype(dtype, np.integer)
            and dtype != np.bool_
            and operation in COMPARISON_OPERATIONS
        )
    return False


def operation_result(operation, lhs, rhs, dtype):
    functions = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": np.divide,
        "eq": np.equal,
        "ne": np.not_equal,
        "lt": np.less,
        "le": np.less_equal,
        "gt": np.greater,
        "ge": np.greater_equal,
    }
    with np.errstate(all="ignore"):
        result = functions[operation](lhs, rhs)
    if operation in ARITHMETIC_OPERATIONS:
        result = np.asarray(result).astype(dtype)
    return np.asarray(result)


def catalog_options(catalog):
    if catalog == "smoke":
        return {
            "sizes": (0, 3),
            "topologies": (
                "same-2d",
                "python-scalar",
                "rhs-row",
                "outer",
                "rhs-vector",
                "mixed-rank-reversed",
                "crossed-batch",
                "empty-inner",
                "invalid-trailing",
            ),
            "operations": ("add", "sub", "div", "eq"),
            "dtypes": ("int32", "float64"),
            "lhs_layouts": (
                "c",
                "negative-inner",
                "step2-inner",
                "offset",
            ),
            "rhs_layouts": (
                "c",
                "negative-inner",
                "step2-inner",
                "zero-inner",
            ),
            "modes": ("out", "in"),
            "value_patterns": ("finite",),
        }
    if catalog == "correctness":
        return {
            "sizes": (0, 1, 2, 3, 4, 7, 8),
            "topologies": None,
            "operations": ALL_OPERATIONS,
            "dtypes": ALL_DTYPES,
            "lhs_layouts": INPUT_LAYOUTS,
            "rhs_layouts": INPUT_LAYOUTS,
            "modes": ("out", "in"),
            "value_patterns": (
                "finite",
                "ieee",
                "integer-boundary",
            ),
        }
    if catalog == "performance":
        return {
            "sizes": (
                1, 4, 8, 16, 31, 32, 33, 64, 127, 128, 129,
                256, 512, 1024,
            ),
            "topologies": None,
            "operations": ARITHMETIC_OPERATIONS,
            "dtypes": ("float32", "float64"),
            "lhs_layouts": INPUT_LAYOUTS,
            "rhs_layouts": INPUT_LAYOUTS,
            "modes": ("out", "in"),
            "value_patterns": ("finite",),
        }
    if catalog == "exhaustive":
        return {
            "sizes": (
                0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33,
                63, 64, 65, 127, 128, 129, 256, 512, 1024,
            ),
            "topologies": None,
            "operations": ALL_OPERATIONS,
            "dtypes": ALL_DTYPES,
            "lhs_layouts": INPUT_LAYOUTS,
            "rhs_layouts": INPUT_LAYOUTS,
            "modes": ("out", "in"),
            "value_patterns": (
                "finite",
                "ieee",
                "integer-boundary",
            ),
        }
    raise ValueError(f"unsupported catalog: {catalog}")


def iter_case_specs(catalog):
    options = catalog_options(catalog)
    for size in options["sizes"]:
        for topology in make_topologies(size):
            if (
                options["topologies"] is not None
                and topology.name not in options["topologies"]
            ):
                continue
            rhs_layouts = (
                ("scalar",)
                if topology.rhs_scalar
                else options["rhs_layouts"]
            )
            product = itertools.product(
                options["operations"],
                options["dtypes"],
                options["modes"],
                options["lhs_layouts"],
                rhs_layouts,
                options["value_patterns"],
            )
            for (
                operation,
                dtype,
                mode,
                lhs_layout,
                rhs_layout,
                value_pattern,
            ) in product:
                if not operation_supported(operation, dtype):
                    continue
                if not value_pattern_supported(
                    value_pattern,
                    operation,
                    dtype,
                ):
                    continue
                if mode == "in" and operation not in ARITHMETIC_OPERATIONS:
                    continue
                if mode == "in" and lhs_layout not in DESTINATION_LAYOUTS:
                    continue
                yield CaseSpec(
                    catalog=catalog,
                    size=size,
                    topology=topology,
                    operation=operation,
                    dtype=dtype,
                    mode=mode,
                    lhs_layout=lhs_layout,
                    rhs_layout=rhs_layout,
                    value_pattern=value_pattern,
                )
    if catalog in ("smoke", "correctness", "exhaustive"):
        yield from iter_alias_case_specs(catalog, options)


def iter_alias_case_specs(catalog, options):
    sizes = (4,) if catalog == "smoke" else options["sizes"]
    aliases = (
        "self",
        "shift-forward",
        "shift-backward",
        "reversed",
        "broadcast-row",
    )
    for size, operation, dtype, mode, alias in itertools.product(
        sizes,
        ARITHMETIC_OPERATIONS,
        options["dtypes"],
        ("out", "in"),
        aliases,
    ):
        if not operation_supported(operation, dtype):
            continue
        rows, columns, _, _ = dimensions(max(3, size))
        if alias == "broadcast-row":
            topology = Topology(
                "alias-broadcast-row",
                (rows, columns),
                (1, columns),
                (rows, columns),
            )
        else:
            topology = Topology(
                f"alias-{alias}",
                (columns,),
                (columns,),
                (columns,),
            )
        yield CaseSpec(
            catalog=catalog,
            size=size,
            topology=topology,
            operation=operation,
            dtype=dtype,
            mode=mode,
            lhs_layout=f"alias-{alias}-lhs",
            rhs_layout=f"alias-{alias}-rhs",
            alias=alias,
        )


def shard_cases(cases, shard_index, shard_count):
    for index, case in enumerate(cases):
        if index % shard_count == shard_index:
            yield case


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
