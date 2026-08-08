# Dispatch Strassen through the matmul executor

## Problem

The earlier prototype exposed Strassen as a separate operation.  That made the
caller choose the recursion depth and manage workspace reuse, bypassing the
planned matmul dispatch.  The prototype also mixed the algorithm choice with
the public operation instead of treating it as an execution policy.

This prototype rebases the experiment on the matmul plan, selection, and
executor architecture at commit `7b40cb46`.  A regular planned matmul may use a
calibrated Strassen kernel without changing its logical plan or public API.

## Code analysis

`MatmulPlan` describes shapes, strides, broadcasting, and contraction offsets.
It does not choose an implementation.  `MatmulExecutor` combines that plan with
`MatmulTuning`, creates one `MatmulSelection`, prepares operands when needed,
and executes the selected kernel.

Strassen is therefore an executor policy.  Selection needs to identify the
exact kernel, including recursion depth, while the plan remains independent of
the backend.  The executor owns the temporary workspace because its lifetime
and layout are properties of the chosen implementation.

## Design

```{mermaid}
flowchart LR
    A["MatmulPlan: shape and strides"] --> B["MatmulExecutor::select_gemm"]
    C["MatmulTuning: calibrated routes"] --> B
    B --> D["MatmulSelection: one exact kernel"]
    D --> E["StrassenGemm1"]
    D --> F["StrassenGemm2"]
    D --> G["Existing BLAS, packing, or generic path"]
    E --> H["Thread-local workspace and BLAS leaves"]
    F --> H
```

The selection table contains only shapes supported by the measurements from
the earlier prototype:

| Type | Rows | Columns | Inner | Kernel |
| --- | ---: | ---: | ---: | --- |
| `float32` | 5632 | 5632 | 5632 | depth 1 |
| `float32` | 3072 | 3072 | 24576 | depth 1 |
| `float64` | 3072 | 3072 | 3072 | depth 1 |
| `float64` | 4096 | 4096 | 4096 | depth 2 |
| `float64` | 6144 | 6144 | 6144 | depth 2 |

A route is eligible only on Apple arm64 when both operands are compact
row-major matrices, the operation has no batch axes, and the value type and
shape match the table exactly.  Every other call continues through the
existing selection logic.

`StrassenGemm1` and `StrassenGemm2` are separate `MatmulKernel` values.  This
keeps `MatmulSelection` complete: execution does not make another policy
decision.  Both kernels share one workspace per value type and thread, so a
later call can reuse the largest allocation already made by that thread.

## Implementation

The prototype adds a header-only Strassen kernel with:

- non-owning matrix views for recursive submatrices;
- one arena allocation sized for the selected recursion depth;
- seven recursive products using compact temporary sums and products;
- an injectable leaf operation for platform-independent unit tests;
- BLAS GEMM leaves in production.

The public `matmul_planned()` path is unchanged.  `MatmulExecutor::select_gemm`
selects a Strassen enum value before the existing direct BLAS and packing
checks.  `MatmulExecutor::execute` dispatches that enum to the Strassen kernel.

## Verification

The focused C++ tests cover:

- every calibrated type, shape, and recursion depth;
- known negative controls and unsupported complex values;
- numerical agreement with a direct reference multiplication at depth 1 and
  depth 2, without requiring a platform BLAS.

The full C++ suite, the matrix Python tests, and source checks pass locally.
The documentation build cannot start because Doxygen is not installed in the
current environment.  Performance claims remain based on the earlier Apple
measurements; this rewrite does not introduce new benchmark results.

## Out of scope

This prototype does not add runtime autotuning, fringe handling for odd
dimensions, packing for non-compact matrices, batched Strassen, complex
support, a public depth control, or workspace cache eviction.  A production
change should recalibrate the route table on supported Apple hardware and
decide how long retained thread-local storage may live.

## Delivery status

The implementation updates draft PR #30 on
`codex/strassen-accelerate-prototype`, based on commit `7b40cb46`.  The
extension builds, all 242 C++ tests pass, all 102 matrix Python tests pass, and
lint passes.  Documentation verification is blocked by the missing Doxygen
executable.  The upstream-oriented issue body and its evidence comment are
published for review as
[ThreeMonth03/solvcon#31](https://github.com/ThreeMonth03/solvcon/issues/31).

## Chat history

- The user asked to recover the earlier Strassen draft and rewrite it as a
  clean dispatch prototype.
- The user selected the explicit plan, selection, and executor architecture as
  the only base for this rewrite.
- The user requested an issue draft after reviewing the clean dispatch
  prototype.  The issue keeps implementation tasks in the main body and moves
  measurements from the earlier API prototype to a separate comment.
- The user clarified that the clean rewrite belongs in the existing draft PR
  #30.  The PR branch and description are updated without changing the saved
  Apple measurements.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
