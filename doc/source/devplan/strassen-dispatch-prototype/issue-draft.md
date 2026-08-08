<!-- Published as ThreeMonth03/solvcon#31. Keep the file synchronized with that issue. -->

<!-- PR #30 is the clean dispatch prototype. Its retained measurements predate the dispatch rewrite and still require a clean rerun. -->

## Motivation

Planned matmul can select generic, direct BLAS, and packing routes before traversal, but large dense GEMM always ends at the backend GEMM. For HPC-sized matrices, a shallow Strassen decomposition can reduce the multiplication work while retaining the tuned backend GEMM for each leaf.

The benefit is platform, dtype, shape, and depth dependent. Earlier Apple M1 measurements found a 1.118x speedup over NumPy for float32 `3072 x 24576 x 3072` and a 1.164x speedup for float64 `6144 x 6144 x 6144`. The same experiment found float32 `6144 x 6144 x 6144` slightly slower than NumPy and low-`K` shapes substantially slower. A general size threshold would therefore misroute known cases.

Callers should not choose a Strassen depth or workspace policy. The operation should continue through the planned matmul API, and unsupported or uncalibrated calls should retain the existing execution route.

## Proposed design

This work consumes the explicit execution selection introduced by [solvcon/solvcon#1259](https://github.com/solvcon/solvcon/pull/1259). `MatmulPlan` remains an execution-independent description of shapes and strides. `MatmulTuning` records calibrated Strassen routes, `MatmulSelection` fixes the exact kernel and recursion depth once, and `MatmulExecutor` owns execution and workspace reuse.

```text
MatmulPlan + MatmulTuning
          |
          v
MatmulSelection
    +-- Strassen depth 1
    +-- Strassen depth 2
    `-- existing BLAS, packing, or generic route
          |
          v
MatmulExecutor -> reusable workspace -> backend GEMM leaves
```

The initial policy is limited to non-batched float32 and float64 GEMM on Apple arm64 with compact row-major operands. A Strassen route is selected only for a shape and depth supported by stable measurements. Other dtypes, layouts, batches, shapes, and platforms continue through the existing selection logic.

The workspace is internal and reused by value type and thread. No public `matmul_strassen()`, recursion-depth argument, or workspace control is added.

## Implementation outline

- [ ] **Task 1: Add the internal Strassen kernel and reusable workspace.** Implement depth-1 and depth-2 rectangular GEMM with backend GEMM leaves, shape validation, and bounded temporary storage. Add numerical tests for both dtypes and depths, rectangular dimensions, invalid divisibility, and workspace reuse.
- [ ] **Task 2: Establish the calibration and profiling route.** Compare complete-call NumPy, direct backend GEMM, and Strassen depth 1 and 2 with allocation and reused-workspace controls. Cover square, inner-heavy, and low-`K` shapes, retain raw samples, and reject candidates that do not remain faster or satisfy the numerical error bound.
- [ ] **Task 3: Integrate calibrated selection and executor dispatch.** Extend the [solvcon/solvcon#1259](https://github.com/solvcon/solvcon/pull/1259) kernel and tuning vocabulary with exact Strassen depths, select one route before execution, and dispatch it with the executor-owned workspace. Add tests for every enabled route and for dtype, batch, layout, shape, and platform fallbacks, then run the complete test, lint, and Apple profiling gates.

## Global acceptance

- Planned and eventual public matmul routes select Strassen without exposing a new public API or changing `MatmulPlan` semantics.
- Selection fixes the recursion depth before execution; the executor does not repeat the policy decision.
- Enabled routes remain numerically within the documented scaled residual bound and outperform both NumPy and direct backend GEMM under the recorded Apple sampling protocol.
- Unsupported and uncalibrated calls preserve the existing BLAS, packing, or generic selection.
- Workspace reuse avoids a new temporary allocation after the thread-local capacity has grown for the selected route.
- The final route table, negative controls, raw samples, environment, thread settings, and reproduction commands are published together.

## References

- [Clean dispatch prototype, ThreeMonth03/solvcon#30](https://github.com/ThreeMonth03/solvcon/pull/30)
- [NumPy-compatible batched matmul and layout-aware HPC execution, #1172](https://github.com/solvcon/solvcon/issues/1172)
- [Explicit matmul execution selection, #1259](https://github.com/solvcon/solvcon/pull/1259)

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
