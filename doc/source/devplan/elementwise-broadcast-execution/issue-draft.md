# Evaluate planned elementwise broadcasting for SimpleArray

<!-- Local upstream issue draft. Keep it unpublished until the macOS gate below passes. -->

`SimpleArray` arithmetic currently combines validation, traversal, and arithmetic in operation-specific paths. Important routes assume equal shapes or linear storage, which prevents general NumPy broadcasting and makes signed or sparse layouts difficult to optimize safely.

The proposed direction separates elementwise layout policy from an operation-independent runtime-rank loop layer, then dispatches through contiguous, fixed inner-stride, or general mapped execution. Existing public operators remain unchanged during evaluation.

Prototype draft PR: [ThreeMonth03/solvcon#28](https://github.com/ThreeMonth03/solvcon/pull/28)

The prototype is rebased onto `solvcon/solvcon` revision `8337f48a`, including [merged PR #1208](https://github.com/solvcon/solvcon/pull/1208) at merge commit `2405ac2b`. Its shared `cpp/solvcon/buffer/loop.hpp` is byte-identical to that base, while elementwise spans, layout classification, planning, and route selection remain private to `cpp/solvcon/buffer/elementwise/`.

## Correctness evidence

The clean NumPy 2.5.1 correctness run at code revision `2310734c` processes 3,134,108 catalog rows with every overall status `ok`. The 1,465,004 planned arithmetic outcomes contain 999,110 matches, 440,748 expected invalid-broadcast errors, and 25,146 existing complex IEEE division differences. The remaining 1,669,104 comparison rows are unavailable because the prototype implements arithmetic only. Empty domains are included and no longer raise `SIGFPE`.

## WSL2 performance evidence

The clean WSL2 x86-64 run uses Python 3.12.7, NumPy 2.5.1, CPU 2, one thread for each recorded numerical library, five samples, two warmups, and a one-millisecond timing target. Six reports contain 54,432 raw rows and deduplicate in documented order to 49,152 unique identifiers, all with status `ok`.

| Scope | Cases | Win rate | Median NumPy / planned | 10th percentile |
| --- | ---: | ---: | ---: | ---: |
| All normal paths | 33,368 | 98.23% | 1.800x | 1.295x |
| Broadcast normal paths | 29,208 | 98.14% | 1.829x | 1.374x |
| All reused outputs | 24,512 | 98.91% | 2.284x | 1.385x |
| Broadcast reused outputs | 22,240 | 98.86% | 2.321x | 1.582x |
| Size-32 normal paths | 16,472 | 99.81% | 1.817x | 1.451x |
| Size-32 reused outputs | 12,384 | 99.99% | 2.283x | 1.746x |

This is a broad performance advantage, not a universal one. Long-sample reruns of the six lowest short-sweep normal ratios range from 0.804x to 1.593x, and three remain below 1.0.

## Apple Silicon evidence

The clean native Apple M1 run uses revision `c9752b52`, whose relevant source tree is byte-identical to benchmark anchor `2310734c`. The environment is macOS 26.5.1 arm64, Python 3.14.6, NumPy 2.5.1 with Accelerate, AC power, and one thread for every recorded numerical library variable. The six reports contain 54,432 raw rows and deduplicate in documented order to 49,152 unique identifiers, all with status `ok`.

| Scope | Cases | Win rate | Median NumPy / planned | 10th percentile | Minimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| All normal paths | 33,368 | 90.89% | 1.439x | 1.019x | 0.051x |
| Broadcast normal paths | 29,208 | 93.46% | 1.458x | 1.083x | 0.051x |
| All reused outputs | 24,512 | 96.34% | 1.803x | 1.090x | 0.199x |
| Broadcast reused outputs | 22,240 | 96.55% | 1.822x | 1.228x | 0.199x |
| Size-32 normal paths | 16,472 | 95.64% | 1.487x | 1.190x | 0.439x |
| Size-32 reused outputs | 12,384 | 99.54% | 1.815x | 1.426x | 0.250x |

This is a broad performance advantage, not a universal one. Sequential long-sample reruns of the six lowest short-sweep normal ratios measure 1.243x, 1.822x, 0.910x, 0.760x, 1.318x, and 2.000x. Two in-place singleton-broadcast cases remain below NumPy.

## Questions for review

- Is the plan-and-executor boundary an appropriate basis for future public broadcasting operators?
- Which output-layout guarantees should become part of the public contract?
- Should complex IEEE division semantics be handled separately before public arithmetic adopts the planned path?
- Which remaining slow topology families deserve optimization before public API work begins?

<!-- Publication gate: verify the draft PR URL is exactly https://github.com/ThreeMonth03/solvcon/pull/28, confirm every reported count against the raw JSON, and remove this comment before posting. Do not use a closing keyword. -->

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
