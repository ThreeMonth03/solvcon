<!-- Published as ThreeMonth03/solvcon#23. Keep the file synchronized with that issue. -->

## Motivation

`SimpleArray` arithmetic accepts scalars or arrays with identical shapes but does not provide general NumPy-style broadcasting. Some paths also assume linear storage, so negative, stepped, sparse, or zero-stride layouts may lose their logical coordinates.

The public operation should align axes from the right, broadcast axes of length one, preserve the fixed shape of an in-place destination, and reject incompatible shapes:

```python
lhs.shape == (2, 1, 4)
rhs.shape == (1, 3, 1)

result = lhs + rhs
assert result.shape == (2, 3, 4)
```

The result contains 24 values, but the inputs contain only 8 and 3. Broadcasting should use zero strides rather than allocate expanded operands. Partial aliases must be safe, and common scalar or equal-shape dense calls must retain a low-overhead route.

[Draft PR #28](https://github.com/ThreeMonth03/solvcon/pull/28) validates these boundaries behind private staging methods. NumPy 2.5.1 correctness, performance, revisions, and raw artifacts are recorded in the [prototype evidence comment](https://github.com/ThreeMonth03/solvcon/issues/23#issuecomment-5156722528).

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

The runtime-rank `LoopDomain` and `OperandMapping` types introduced through [PR #1208](https://github.com/solvcon/solvcon/pull/1208) provide shape and offset facts. `ElementwisePlan` owns broadcast validation, aligned mappings, the fixed output mapping, and route selection. `ElementwiseExecutor` owns allocation, overlap safety, destination reuse, and dispatch. Typed kernels own arithmetic.

Broadcast axes use zero strides. Exact aliases reuse the destination mapping, partial overlaps snapshot the affected input, and sparse results use compact storage. Scalar and equal-shape dense operations use a mandatory private direct path; unsupported layouts fall back to the validated signed-stride plan. No plan, route, kernel, or backend selector becomes public.

## Migration path

Development uses private `_planned_*`, `_planned_*_to`, and `_planned_i*` staging methods while the existing public operators remain correctness and performance controls. They are not proposed public APIs.

Each task introduces one reviewable architectural step with its immediate consumer, tests, and profiling. Task 6 moves the existing public arithmetic and in-place operators to planned execution, then removes the staging methods and duplicate legacy loops.

## Implementation outline

Dependencies: Task 2 follows Task 1. Task 3 follows Task 2. Tasks 4 and 5 follow Task 3 and may proceed separately. Task 6 requires Tasks 4 and 5.

In Tasks 1-5, plan API means an internal C++ interface. Private Python methods exist only for testing and profiling.

- [ ] **Task 1: Add generic multiplication broadcasting and create the minimal plan API.** Introduce `ElementwisePlan::make()`, `make_scalar()`, and `ElementwiseExecutor::transform()` using the PR #1208 mappings and one general route. Add private `_planned_mul` staging plus differential tests for shapes, scalar operands, empty axes, and signed layouts.
- [ ] **Task 2: Extend the plan API for fixed destinations and alias safety.** Reuse the Task 1 plan and add `transform_to()` and `transform_into()` with private multiplication staging. Validate fixed output shapes, snapshot partial overlaps, preserve exact aliases, and cover both behavior and reused-output performance.
- [ ] **Task 3: Add route selection to the existing plan API.** Extend `ElementwisePlan` with `ExecutionRoute`, `inner_axis`, and `InnerLoopPlan`. Add contiguous, constant, fixed-stride, reversed, and mapped routes with internal route tests and topology profiles.
- [ ] **Task 4: Add mandatory low-overhead direct execution.** Bypass `ElementwisePlan` for scalar and equal-shape dense operations, perform one Python array-or-scalar dispatch, and fall back to the Task 3 plan for every unsupported layout. Add no public, staging, or plan API. Validate the scope with complete-call profiles on Apple Accelerate and WSL2.
- [ ] **Task 5: Migrate arithmetic semantics without another plan API.** Reuse the same plan and executor for `add`, `sub`, and `div`, adding only typed kernels and private staging methods. Cover supported dtypes, boolean results, division behavior, and in-place forms with differential tests and cross-platform profiles.
- [ ] **Task 6: Replace legacy public routes.** Move public methods and operators to planned execution, add public-API regression tests, remove staging and duplicate loops, and rerun full correctness, performance, lint, Apple Accelerate, and WSL2 verification.

## Global acceptance

- Public scalar and array `add`, `sub`, `mul`, and `div`, including in-place forms, match the existing supported semantics and NumPy broadcasting values.
- Broadcasting uses aligned zero strides and never expands an input to the result shape.
- Empty domains, signed and sparse layouts, mixed ranks, fixed destinations, exact aliases, and partial overlaps have differential coverage.
- Scalar and equal-shape dense calls use the direct route; unsupported combinations use the generic signed-stride executor.
- Timed calls include planning, allocation when applicable, alias handling, dispatch, and arithmetic.
- The final public API exposes no staging method, plan, mapping, kernel, route, or backend selector.
- Final migration includes Apple Accelerate and WSL2 NumPy 2.5.1 correctness and performance evidence.

## References

- [Elementwise broadcast execution: design and reproduction guide](https://github.com/ThreeMonth03/solvcon/blob/codex/prototype-elementwise-broadcast/doc/source/devplan/elementwise-broadcast-execution/index.md)
- [Prototype evidence and artifacts](https://github.com/ThreeMonth03/solvcon/issues/23#issuecomment-5156722528)
- [Shared runtime-rank loop vocabulary, solvcon/solvcon#1208](https://github.com/solvcon/solvcon/pull/1208)
- [Related batched matmul design, solvcon/solvcon#1172](https://github.com/solvcon/solvcon/issues/1172)

<!-- Publication gate: confirm that Draft PR #28 still targets ThreeMonth03/solvcon:master and remains open. Do not use a closing keyword when posting. -->

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
