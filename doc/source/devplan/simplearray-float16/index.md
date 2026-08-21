# Add Float16 to SimpleArray

## Motivation

`SimpleArray` supports 32-bit and 64-bit floating-point storage. Adding an
IEEE 754 binary16 dtype lets applications exchange half-precision arrays with
NumPy, reduce array storage, and prepare CPU data for libraries that accept
FP16 input.

## Data path

The dtype uses the existing header-only `half_float::half` implementation.
SOLVCON supplies only the adapter needed by its array and Python type systems.

```mermaid
flowchart LR
    H["half_float::half"] --> F["solvcon::Float16 alias"]
    F --> S["SimpleArrayFloat16"]
    S <--> N["NumPy float16 view"]
```

The same two-byte allocation backs C++ access, the Python buffer protocol,
and a NumPy view. Arithmetic uses the generic CPU `SimpleArray` operations.

## Design

### Portable scalar type

`solvcon::Float16` aliases `half_float::half` version 2.2.1. The dependency is
pinned by version and SHA-256. Small SOLVCON traits classify the alias as a
floating-point, arithmetic, and signed number without specializing standard
library traits.

Scalar conversion first converts the source to `float`, then constructs the
half value. This provides one explicit conversion path on Apple Clang, GCC,
and MSVC.

### Runtime dtype

`DataType::Float16` maps to the `"float16"` name. `SimpleArrayPlex` constructs,
copies, aligns, and destroys `SimpleArrayFloat16` through the same runtime
dispatch used for the existing scalar types.

### Python and NumPy

The Python module exports `SimpleArrayFloat16` and
`SimpleCollectorFloat16`. The pybind11 adapters map the scalar to the PEP 3118
`e` format and NumPy's `float16` dtype. Arrays created from compatible NumPy
storage retain the existing zero-copy behavior.

### CPU operations

The arithmetic concept and floating-point branches recognize FP16. Existing
element-wise operations, reductions, sorting, searching, matrix helpers, and
generic matrix multiplication are reused. Count conversions used by means and
variances use the explicit FP16 conversion helper.

## Code locations

- `cpp/solvcon/math/Float16.hpp`: dependency alias and numeric traits.
- `cpp/solvcon/buffer/SimpleArray.*`: compile-time and runtime array support.
- `cpp/solvcon/buffer/pymod/`: Python scalar, buffer, NumPy, Plex, collector,
  and broadcast integration.
- `tests/test_float16.py`: public Python and NumPy behavior.
- `gtests/test_nopython_buffer.cpp`: C++ storage, dtype, and matmul behavior.
- `contrib/standalone_buffer/`: pinned dependency setup for the standalone
  buffer build.

## Verification

The focused tests cover two-byte storage, zero-copy NumPy views, the Python
buffer format, scalar conversion, element-wise arithmetic, reductions, matrix
helpers, CPU matrix multiplication, runtime Plex dispatch, collectors, and
cross-dtype NumPy assignment. The full C++ and Python suites protect existing
types.

## Scope

This change adds FP16 to the CPU `SimpleArray` API. Accelerator kernels,
automatic mixed precision, BLAS dispatch, and device-backed buffers are
separate work.

## Discussion record

The prototype was separated from the GPU-storage work so its API and
cross-platform behavior can be reviewed independently. The design uses an
existing binary16 implementation and keeps `solvcon::Float16` as a stable
project-facing name because the C++23 extended floating-point typedefs are
optional across supported toolchains.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
