# Add NumPy-compatible elementwise broadcasting with layout-aware execution

<!-- Published as ThreeMonth03/solvcon#23. Keep the file synchronized with that issue. -->

## Motivation

`SimpleArray` arithmetic accepts scalars or arrays with identical shapes but does not provide general NumPy-style broadcasting. Important paths also assume linear storage, so applying them to signed, sparse, or zero-stride layouts can lose logical coordinates.

The public operation should align axes from the right, broadcast axes of length one, preserve the fixed shape of an in-place destination, and reject incompatible shapes:

```python
lhs.shape == (2, 1, 4)
rhs.shape == (1, 3, 1)

result = lhs + rhs
assert result.shape == (2, 3, 4)
```

The result contains 24 values, but the inputs contain only 8 and 3. Broadcasting should represent repeated values with zero strides instead of allocating expanded copies.

Correct traversal is not enough. Common scalar and equal-shape calls should avoid unnecessary runtime-rank planning, while negative, stepped, offset, permuted, and zero-stride layouts must preserve their logical coordinates. Partial aliases must be snapshotted before writes, and a fixed in-place destination must never expand.

[Draft PR #28](https://github.com/ThreeMonth03/solvcon/pull/28) tests these boundaries behind private staging methods. Its NumPy 2.5.1 results show a broad advantage for planned broadcast execution, but not universal dominance. Detailed measurements, revisions, and raw artifacts are kept in the supplement below and the linked fork study.

## Proposed design

Callers continue to use the existing arithmetic methods and operators. Planning and execution remain internal:

```text
SimpleArray arithmetic
        |
        v
ElementwisePlan::make(output, lhs, rhs)
        |
        +-- validate broadcast and destination shapes
        +-- align signed operand strides to the result rank
        `-- select an inner axis and execution route
        |
        v
ElementwiseExecutor::execute(plan, operation)
        +-- shared contiguous traversal
        +-- fixed inner-stride traversal
        `-- general mapped traversal
```

The runtime-rank `LoopDomain`, `OperandMapping`, and `MappedOffsetCursor` types introduced through [PR #1208](https://github.com/solvcon/solvcon/pull/1208) contain only shape and offset facts. They do not own storage, dtype, arithmetic, allocation, or backend policy.

`ElementwisePlan` owns elementwise broadcast validation, aligned input mappings, the fixed output mapping, and route selection. `ElementwiseExecutor` owns output allocation, overlap safety, destination reuse, and dispatch. Typed kernels own scalar, contiguous, and fixed-stride arithmetic.

The fast routes and mapped fallback consume the same validated plan. Broadcast axes use zero strides. Exact aliases reuse the destination mapping, partial overlaps copy the affected input before execution, and sparse broadcast results use compact storage rather than materialized expanded operands.

This boundary does not introduce a universal operation object, virtual per-element dispatch, or platform-specific planner branch. The shared coordinate vocabulary remains operation-independent, while elementwise semantics and tuning remain in the elementwise family.

## Migration path

Development uses private `_planned_add`, `_planned_sub`, `_planned_mul`, and `_planned_div` methods while the existing public operators remain correctness and performance controls. Private `_planned_*_to` methods expose destination reuse only for profiling; they are not a proposed public `out` contract.

The final task moves the existing public arithmetic and in-place operators to planned execution, removes the private staging methods, and removes legacy loops that become exact duplicates. Existing dtype, boolean, division, exception, and result-layout behavior remains explicit during migration.

Every pull request should contain one reviewable architectural step, its immediate elementwise consumer, differential tests, and relevant profiling. New abstractions must not land unused.

## Implementation outline

Dependencies: Task 2 follows Task 1. Tasks 3 and 4 follow Task 2 and may proceed separately. Task 5 requires Tasks 3 and 4. Task 6 follows Task 5.

- [ ] **Task 1: Add generic multiplication broadcasting.** Consume `LoopDomain` and `OperandMapping` from PR #1208, rank-align operands with zero broadcast strides, allocate compact results, and add NumPy differential tests for valid and invalid shapes, empty axes, scalar operands, and signed or zero-stride layouts.
- [ ] **Task 2: Add fixed destinations and alias safety.** Treat in-place output as a fixed shape, distinguish exact, disjoint, and partial overlap, snapshot partial aliases, and add unit tests plus temporary reused-output profiling controls for each case.
- [ ] **Task 3: Lower mapped traversal into inner loops.** Select a profitable inner axis, add contiguous, constant, fixed-stride, reversed, and general mapped routes, and verify every dispatch choice with route tests and topology-level profiles.
- [ ] **Task 4: Preserve low-overhead direct execution.** Bypass redundant planning for scalar and equal-shape dense operations, perform one Python array-or-scalar dispatch, and add regression tests and profiles that include complete public-call overhead.
- [ ] **Task 5: Migrate arithmetic semantics.** Move `add`, `sub`, and `div` onto the executor with differential tests for every supported dtype, value pattern, boolean result, division behavior, and in-place form. Tune only topology families supported by stable cross-platform measurements.
- [ ] **Task 6: Replace legacy public routes.** Move public methods and operators to planned execution, add public-API regression tests, remove staging and duplicate loops, and rerun full correctness, performance, lint, and Apple Accelerate plus WSL2 verification.

## Global acceptance

- Public scalar and array `add`, `sub`, `mul`, and `div`, including in-place forms, match the existing supported semantics and NumPy broadcasting values.
- Broadcasting uses aligned zero strides and never expands an input to the result shape.
- Empty domains, signed and sparse layouts, mixed ranks, fixed destinations, exact aliases, and partial overlaps have differential coverage.
- Common scalar and equal-shape dense calls do not pay for general mapped traversal.
- Unsupported fast-path combinations use the generic signed-stride executor.
- Timed calls include planning, allocation when applicable, alias handling, dispatch, and arithmetic, with thread configuration applied before importing NumPy.
- The final public API contains no `_planned_*` or `_planned_*_to` staging method and exposes no plan, mapping, kernel, or backend selector.
- Final migration includes Apple Accelerate and WSL2 NumPy 2.5.1 correctness and performance evidence.

## References

- [Elementwise broadcast execution: design, methodology, and reproduction guide](https://github.com/ThreeMonth03/solvcon/blob/codex/prototype-elementwise-broadcast/doc/source/devplan/elementwise-broadcast-execution/index.md)
- [Shared runtime-rank loop vocabulary, solvcon/solvcon#1208](https://github.com/solvcon/solvcon/pull/1208)
- [Related batched matmul design, solvcon/solvcon#1172](https://github.com/solvcon/solvcon/issues/1172)

## Appendix: prototype and benchmark evidence

- [Prototype Draft PR #28](https://github.com/ThreeMonth03/solvcon/pull/28)
- [Final Apple audit comment](https://github.com/ThreeMonth03/solvcon/pull/28#issuecomment-5156411746)
- [Final Apple raw reports, stable observations, logs, checksums, and manifest](https://github.com/user-attachments/files/30628902/elementwise-numpy251-macos-stable-20260802-145103.tar.gz), SHA-256 `2592bb0ceed9eeaf9f6c40a92b3c56e9cde6318fc0dc0cafafbbc41b6c7e5cb6`

<!-- Publication gate: confirm that Draft PR #28 still targets ThreeMonth03/solvcon:master and remains open. Do not use a closing keyword when posting. -->

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
