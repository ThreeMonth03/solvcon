# Add NumPy-compatible elementwise broadcasting with layout-aware execution

<!-- Local upstream issue draft. Keep it unpublished until explicitly approved. -->

## Motivation

`SimpleArray` arithmetic currently combines validation, traversal, and arithmetic in operation-specific paths. Important routes require equal shapes or assume linear storage, so they cannot express general NumPy broadcasting and cannot safely optimize every signed, sparse, or zero-stride layout.

NumPy-style elementwise arithmetic should align ranks from the right, broadcast axes of length one, preserve the fixed shape of an in-place destination, and reject incompatible shapes:

```python
lhs.shape == (2, 1, 4)
rhs.shape == (1, 3, 1)

result = lhs + rhs
assert result.shape == (2, 3, 4)
```

Correct broadcasting is not enough. Execution should represent broadcast axes with zero strides, preserve logical coordinates for negative and sparse layouts, avoid materializing expanded operands, and select optimized inner loops without coupling traversal policy to each arithmetic operation.

## Current prototype status

[Draft PR #28](https://github.com/ThreeMonth03/solvcon/pull/28) evaluates the design behind private `_planned_add`, `_planned_sub`, `_planned_mul`, and `_planned_div` methods, plus private reused-output controls for benchmarking. Existing public arithmetic operators remain unchanged.

The prototype is based on `solvcon/solvcon` revision `8337f48a`, which contains [merged PR #1208](https://github.com/solvcon/solvcon/pull/1208) at merge commit `2405ac2b`. The shared `cpp/solvcon/buffer/loop.hpp` is byte-identical to that base. Elementwise spans, layout classification, planning, alias handling, and route selection remain private to `cpp/solvcon/buffer/elementwise/`.

The current macOS gate is complete:

- Apple M1, macOS 26.5.1, native arm64, and AC power
- Python 3.14.6 and **NumPy 2.5.1** with Accelerate
- all five recorded numerical-library thread variables set to 1 before importing NumPy
- 54,432 raw benchmark rows deduplicated in fixed report order to 49,152 unique identifiers, all with overall status `ok`
- 3,134,108 correctness rows with zero unexpected results, benchmark errors, or process crashes
- 256 C++ tests, 1,532 non-GUI Python tests with 1,100 subtests, 89 focused Python tests with 783 subtests, and 5 SIMD tests with 48 subtests pass
- the complete lint target passes; local clang-format 20 also accepts the changed NEON header

The WSL2 x86-64 NumPy 2.5.1 gate is also complete. Apple and WSL measurements use the same catalog, report order, deduplication rule, and timing policy.

## Correctness evidence

The complete NumPy 2.5.1 catalog contains 3,134,108 rows, all with overall status `ok`. The 1,465,004 planned arithmetic outcomes contain 999,110 value matches, 440,748 expected invalid-broadcast errors, and 25,146 existing complex IEEE division differences. The remaining 1,669,104 comparison rows are unavailable because the prototype implements arithmetic only. Empty domains are included and return before route selection.

## Apple M1 NumPy 2.5.1 evidence

The authoritative Apple run measured clean revision `0c03661fe76ad7e01c0c10ffb0c51843c2bdd7b7`. Each short-sweep case used five samples, two warmups, a one-millisecond timing target, and preallocated-output timing. Ratios are NumPy median time divided by planned median time. A value above one means planned execution is faster.

| Scope | Cases | Win rate | Median ratio | 10th percentile | Minimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| All normal paths | 33,368 | 94.49% | 1.498x | 1.091x | 0.128x |
| Broadcast normal paths | 29,208 | 97.48% | 1.522x | 1.138x | 0.128x |
| All reused outputs | 24,512 | 97.92% | 1.876x | 1.145x | 0.145x |
| Broadcast reused outputs | 22,240 | 98.04% | 1.906x | 1.281x | 0.145x |
| Size-32 normal paths | 16,472 | 95.80% | 1.541x | 1.246x | 0.128x |
| Size-32 reused outputs | 12,384 | 99.52% | 1.902x | 1.429x | 0.145x |

Single-axis and outer-broadcast normal paths win 98.74% and 99.21% of cases, with median ratios of 1.545x and 1.797x. The weaker non-broadcast normal family wins 73.46% with a 1.348x median, so the evidence does not support universal dominance.

The six lowest short-sweep normal cases were selected from this exact run and repeated sequentially with 31 samples, seven warmups, and a 30-millisecond timing target:

| Case ID | Short normal | Long normal | Long reused |
| --- | ---: | ---: | ---: |
| `out/mul/float32/n32/mixed-rank-reversed/negative-inner/offset/none/finite` | 0.128x | 1.388x | 1.466x |
| `out/mul/float32/n32/mixed-rank-reversed/negative-inner/zero-inner/none/finite` | 0.167x | 1.208x | 1.439x |
| `out/mul/float32/n4/rhs-column/c/c/none/finite` | 0.283x | 1.332x | 2.195x |
| `out/mul/float32/n512/all-singleton-rhs/step2-outer/c/none/finite` | 0.319x | 1.095x | 0.977x |
| `out/mul/float32/n32/crossed-batch/zero-inner/zero-inner/none/finite` | 0.358x | 1.508x | 1.738x |
| `out/add/float64/n1024/python-scalar/c/scalar/none/finite` | 0.468x | 1.117x | 1.221x |

All six long-sample normal ratios exceed 1.0. One reused-output rerun remains slightly below NumPy at 0.977x, and the full short sweep still contains losses.

## WSL2 NumPy 2.5.1 evidence

The clean WSL2 x86-64 run measured revision `2310734ca0ce8d7cba226b82c0b6ab03df6094fa` with Python 3.12.7, one pinned CPU, one thread for every recorded numerical library, five samples, two warmups, and a one-millisecond timing target.

| Scope | Cases | Win rate | Median ratio | 10th percentile |
| --- | ---: | ---: | ---: | ---: |
| All normal paths | 33,368 | 98.23% | 1.800x | 1.295x |
| Broadcast normal paths | 29,208 | 98.14% | 1.829x | 1.374x |
| All reused outputs | 24,512 | 98.91% | 2.284x | 1.385x |
| Broadcast reused outputs | 22,240 | 98.86% | 2.321x | 1.582x |
| Size-32 normal paths | 16,472 | 99.81% | 1.817x | 1.451x |
| Size-32 reused outputs | 12,384 | 99.99% | 2.283x | 1.746x |

Long-sample reruns of the six lowest WSL short-sweep normal ratios range from 0.804x to 1.593x, and three remain below 1.0.

## Proposed design

Callers eventually continue to use the existing arithmetic operators. Planning and execution remain internal:

```mermaid
flowchart TD
    A["SimpleArray arithmetic"] --> P["ElementwisePlan"]
    P --> V["Validate result shape and in-place destination"]
    V --> D["LoopDomain and OperandMapping"]
    D --> E["ElementwiseExecutor"]
    E --> C["Contiguous route"]
    E --> S["Fixed inner-stride route"]
    E --> M["General mapped route"]
    C --> K["Typed arithmetic kernel"]
    S --> K
    M --> K
```

`LoopDomain`, `OperandMapping`, and `MappedOffsetCursor` contain only operation-independent runtime-rank traversal. Broadcast axes use zero strides. `ElementwisePlan` owns arithmetic layout facts and route selection. `ElementwiseExecutor` owns compact output allocation, overlap safety, destination reuse, and dispatch. Typed kernels own arithmetic and hot-loop specialization.

The executor recognizes shared contiguous traversal, fixed inner strides, constant operands, signed contiguous traversal, and a fully general mapped fallback. Partial aliases are snapshotted before execution. Sparse broadcast results use compact storage rather than materialized expanded operands.

## Migration path

The prototype remains behind private staging methods while the architecture is reviewed. The final integration task moves existing public arithmetic and in-place operators to planned execution, removes private benchmark entry points, and reruns the complete Apple Accelerate and WSL2 gates.

Every abstraction must be consumed by elementwise arithmetic in the task that introduces it. The plan excludes a universal operation object, virtual per-element dispatch, platform-specific planner branches, and shape-specific benchmark routes.

## Implementation outline

- [x] **Task 1: Establish benchmark and correctness coverage.** Generate arithmetic, dtype, topology, layout, alias, empty-axis, and IEEE cases independently from an implementation.
- [x] **Task 2: Separate traversal from arithmetic policy.** Keep the shared loop vocabulary operation-independent and consume it immediately from the elementwise planner.
- [x] **Task 3: Add NumPy broadcast planning.** Rank-align operands, encode broadcast axes with zero strides, and validate fixed in-place destinations.
- [x] **Task 4: Add layout-aware execution.** Select contiguous, fixed inner-stride, constant-operand, signed, and mapped routes without expanding operands.
- [x] **Task 5: Make allocation and aliasing explicit.** Preserve eligible dense layouts, compact sparse outputs, reuse destinations, and snapshot partial overlaps.
- [x] **Task 6: Tune hot loops without changing the planner.** Dispatch Python operands once, reuse direct paths, resolve AArch64 SIMD at compile time, and unroll the shared NEON scalar transform.
- [ ] **Task 7: Replace the legacy public routes.** Move public arithmetic to planned execution, remove private staging methods, and run final cross-platform validation.

## Performance interpretation

Both platforms show a broad advantage over NumPy 2.5.1, especially for broadcast and reused-output execution. The Apple full sweep wins 97.48% of broadcast normal cases and 98.04% of broadcast reused-output cases. The long Apple reruns show that the six most extreme short normal losses were timing-tail artifacts, but the full sweep and the 0.977x reused rerun still prevent a universal claim.

The architecture should not gain case-specific conditions only to turn every benchmark point into a win. Remaining losses should be evaluated by topology family and stable long-sample evidence.

## Global acceptance

- Planned and eventual public routes match NumPy values for supported arithmetic, dtypes, shapes, broadcasts, and signed or zero-stride layouts, subject to the existing complex IEEE division behavior.
- Empty validated outputs return safely without entering route selection or arithmetic.
- Broadcast mappings use zero strides and do not materialize expanded operands.
- Contiguous and fixed-stride layouts avoid the fully general cursor when their invariants are satisfied.
- Output allocation, exact aliases, partial overlaps, and fixed in-place destination shapes remain correct.
- Timed calls include planning, allocation when applicable, alias handling, dispatch, and arithmetic with thread configuration applied before importing NumPy.
- The final public API contains no `_planned_*` or `_planned_*_to` staging method.
- Final integration requires both Apple Accelerate and WSL2 NumPy 2.5.1 correctness and performance reports.

## Evidence

- [Prototype draft PR #28](https://github.com/ThreeMonth03/solvcon/pull/28)
- [Development plan, detailed results, and reproduction protocol](https://github.com/ThreeMonth03/solvcon/blob/codex/prototype-elementwise-broadcast/doc/source/devplan/elementwise-broadcast-execution/index.md)
- [Apple M1 audit comment](https://github.com/ThreeMonth03/solvcon/pull/28#issuecomment-5152766036)
- [Apple M1 raw reports, samples, logs, and manifest](https://github.com/user-attachments/files/30621311/elementwise-numpy251-macos-current-20260802-013704.tar.gz)
- Archive SHA-256: `1e4cc481bba0eb78d2cb10ba99745d699a44ed3817e569f941fd79f30f1d7456`
- Measured Apple code revision: `0c03661fe76ad7e01c0c10ffb0c51843c2bdd7b7`
- Measured WSL2 code revision: `2310734ca0ce8d7cba226b82c0b6ab03df6094fa`

<!-- Publication gate: confirm that Draft PR #28 still targets ThreeMonth03/solvcon:master and remains open. Do not use a closing keyword when posting upstream. -->

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
