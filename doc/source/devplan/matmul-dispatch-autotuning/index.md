# Calibrate matmul dispatch from measured routes

## Problem

Planned matmul selects generic, fixed-size, BLAS, and packing kernels from a
set of hand-written thresholds.  Those thresholds are easy to review, but a
single minimum-dimension check cannot represent every rectangular crossover.
In particular, a large output with a short contracted dimension may still
amortize a BLAS call even when its minimum dimension is small.

This prototype measures every legal route for the same planned call, learns a
small decision tree in Python, and emits equivalent C++ selection code.  The
experiment is successful only if the compiled selector recovers useful oracle
headroom on held-out shapes without adding a material regression.

## Code analysis

`MatmulPlan` already owns the logical contraction: vector and matrix roles,
`M`, `N`, `K`, signed core strides, batch mappings, and broadcast reuse.
`MatmulExecutor` owns preparation and execution.  Its selection methods
currently combine two separate questions: whether a route is legal and
whether a measured threshold predicts that route will be profitable.

The experiment keeps `MatmulPlan` unchanged.  It extracts pointer-free facts,
enumerates legal kernels from correctness conditions, and lets either the
existing selector or a generated policy choose among them.  The generic
signed-stride kernel remains the unconditional fallback.

## Design

```{mermaid}
flowchart TD
    P["MatmulPlan"] --> F["MatmulFacts"]
    F --> E["Eligible kernel mask"]
    E --> D["Default or generated policy"]
    D --> S["MatmulSelection"]
    S --> X["Prepare once and execute"]
    E --> B["Forced-route benchmark"]
    B --> J["Raw JSONL samples"]
    J --> T["Python training and validation"]
    T --> C["Generated C++ policy"]
    C --> D
```

The benchmark API is private and exposes the exact facts, current selection,
and legal kernels used by C++.  Candidate timings include result allocation,
packing, dispatch, and arithmetic.  Routes are interleaved during timing, and
near ties receive more samples than clear wins.

Training is separated by target, backend, dtype, and operation.  The first
compiled proof covers GEMM only.  A leaf stores a preference order and selects
the first legal kernel, rather than trusting an unguarded predicted label.
The generated policy consists of integer comparisons that the C++ compiler can
inline.  No Python model, JSON parser, or tree interpreter is linked into the
library.

## Sampling and acceptance

The first proof samples compact two-dimensional GEMM.  Seed cases combine
logarithmic sizes with current threshold neighbors, rectangular aspect ratios,
low-`K` grids, and independently jittered shapes.  Validation holds out
complete shape regions instead of individual timing repetitions.  Every
result retains raw round timings and an environment fingerprint.  Padded,
batched, broadcast, and reused-operand policies require separate samples.

A policy group is enabled only when held-out measurements show at least five
percent oracle headroom.  A depth-five-or-smaller compiled tree must recover
at least 70 percent of that headroom, improve the current selector by at least
three percent in geometric mean, keep 95th-percentile oracle regret within
five percent, and introduce no statistically clear regression above ten
percent.  A simpler threshold model is measured alongside the tree; it wins
when it achieves the same result with less policy complexity.

## Implementation

- Add private forced-route facts and execution hooks without changing the
  parameterless planned-matmul path.
- Collect deterministic, interleaved route timings into ignored raw result
  files.
- Fit and validate shallow policies, compare them with simple thresholds, and
  emit a C++ include only after the acceptance gate passes.
- Compile the emitted selector in a separate build and repeat the held-out
  timing through the normal planned-matmul entry point.

## Verification

Forced routes are checked against NumPy before timing.  Focused tests verify
that every advertised route is legal and numerically correct, while invalid
forced routes are rejected.  Before a policy is accepted, its generated C++
prediction is compared with the Python tree for every collected row.  The
final gate includes the focused matrix tests, a generated-policy CMake
fixture, source checks, and a compiled baseline-versus-tuned benchmark.

Fresh WSL2 runs used OpenBLAS 0.3.33 with one thread.  The fixed grouped
five-fold validation produced these out-of-fold results:

| dtype | cases / groups | current / oracle | policy speedup | policy / oracle | p95 regret | worst slowdown | oracle gap recovered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `float32` | 579 / 175 | 1.892 | 1.876 | 1.009 | 1.040 | 1.076 | 99.6% |
| `float64` | 551 / 169 | 1.558 | 1.548 | 1.006 | 1.028 | 1.088 | 99.3% |

Both runs completed without skipped cases and passed the preset acceptance
gate.  A depth-five decision tree was more accurate than the single-threshold
control for both dtypes.

The emitted policies were then compiled into separate modules.  Python and
C++ selected the same route on all 1,130 collected rows.  In an ABBA normal
entry-point check, the seven calls whose route changed had geometric-mean
speedups of 4.08 for `float32` and 2.51 for `float64`.  Two unchanged BLAS
controls measured 0.995 and 0.990, respectively, which did not show material
selector overhead.  The fixed CMake fixture also verifies an in-scope route
selection and an out-of-scope fallback.

## Out of scope

The first proof does not replace every handwritten matmul rule, tune during a
normal build, load runtime policy files, or share one universal fact schema
with elementwise and reduction operations.  Apple and OpenBLAS policies are
separate artifacts.  Each generated file currently covers one dtype and
layout target.  Unknown targets retain the current selector.

## Delivery status

The prototype branch starts from commit `fe4f66e4`, the current head of pull
request 1283.  Dependency setup, forced-route instrumentation, grouped
validation, generated-code parity, lazy eligibility, and compiled A/B checks
are complete.  Fork delivery remains in progress.

## References

- R. C. Whaley, A. Petitet, and J. J. Dongarra, "Automated Empirical
  Optimizations of Software and the ATLAS Project," Parallel Computing 27,
  3-35 (2001), DOI `10.1016/S0167-8191(00)00087-9`.
- M. Frigo and S. G. Johnson, "The Design and Implementation of FFTW3,"
  Proceedings of the IEEE 93(2), 216-231 (2005), DOI
  `10.1109/JPROC.2004.840301`.
- L. Zheng et al., "Ansor: Generating High-Performance Tensor Programs for
  Deep Learning," OSDI 2020, 863-879.

## Chat history

- The user asked how much a Python decision tree could improve dispatch, how
  to sample with a small time budget, and how generated C++ policy could grow
  to other operations.
- The user requested a fork draft based on the current matmul selection work,
  with an evidence-first stop condition and a C++ prototype only when the
  measured opportunity justified it.
<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
