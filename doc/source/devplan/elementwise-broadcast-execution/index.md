# Elementwise Broadcast Execution

## Problem

`SimpleArray` arithmetic currently combines API validation, traversal, and
arithmetic in each operation.  The implementation assumes matching shapes and
linear storage in important paths.  Those assumptions prevent general NumPy
broadcasting and make non-contiguous layouts difficult to optimize safely.

The prototype has two goals:

1. Cover arithmetic behavior broadly enough to separate traversal bugs,
   unsupported semantics, and performance opportunities.
2. Introduce a plan-and-executor architecture that can optimize common layouts
   without giving each operation its own traversal implementation.

The prototype adds private Python methods named `_planned_add`,
`_planned_sub`, `_planned_mul`, and `_planned_div`, with matching in-place
forms.  Existing public operators remain unchanged while the design is
evaluated.

## Code analysis

The existing arithmetic entry points are templates in
`cpp/solvcon/buffer/SimpleArray.hpp`.  Python exposes them from
`cpp/solvcon/buffer/pymod/wrap_SimpleArray.hpp`.  The legacy routes validate
shape equality and then either traverse linearly or call a SIMD helper.

That structure has three limitations:

- Shape equality cannot describe broadcasting.
- Linear traversal does not preserve logical coordinates for every signed or
  sparse stride layout.
- Adding a specialized broadcast loop would duplicate validation and
  traversal across arithmetic operations.

The new implementation is local to
`cpp/solvcon/buffer/elementwise/`.  It does not introduce a shared layer with
matrix multiplication yet.  The two operation families have similar planning
shapes, but their reusable abstractions should be extracted only after both
designs have stable evidence.

## Benchmark coverage

The benchmark generator in `profiling/elementwise_benchmark_cases.py`
describes each case independently of an implementation.  A case selects:

- add, subtract, multiply, or divide;
- out-of-place or in-place execution;
- 13 scalar types;
- valid and invalid broadcast topologies, including mixed ranks, outer
  products, leading batches, crossed batches, scalar operands, singleton
  arrays, and empty axes;
- C-contiguous, permuted, negative-stride, stepped, offset, and zero-stride
  layouts;
- partial aliases and finite or IEEE value patterns.

The runner can audit legacy, legacy SIMD, and planned methods in isolated
processes.  Correctness and timing are separate modes.  NumPy in-place timing
uses a ufunc with `out=`, and out-of-place timing avoids an unnecessary dtype
copy.  Both NumPy and `SimpleArray` callables are bound before the timer
starts.  Large catalogs support deterministic shards, summary-only output,
and merging.

## Design

```mermaid
flowchart LR
  A["SimpleArrayElementwise"] --> B["ElementwisePlan::make"]
  B --> C{"ExecutionRoute"}
  C --> D["contiguous"]
  C --> E["fixed inner-strided"]
  C --> F["mapped cursor"]
  D --> G["ElementwiseExecutor"]
  E --> G
  F --> G
  G --> H["typed arithmetic kernel"]
```

`IterationDomain` owns the broadcast result shape.  `OperandMapping` aligns
each operand to that domain and represents broadcasting with zero strides.
Its signed stride span also describes reversed layouts without changing the
logical origin.

`ElementwisePlan` validates the fixed output shape and selects one of three
routes:

- `contiguous` for a shared dense traversal;
- `inner_strided` when one axis has fixed strides for each outer
  coordinate;
- `mapped` for the fully general signed-stride cursor.

`ElementwiseExecutor` owns output allocation, overlap handling, and route
dispatch.  A partially overlapping in-place source is snapshotted before
execution.  Dense layouts may be preserved, while sparse broadcast results
use compact C-contiguous storage.

The inner-axis selector prefers a unit output stride, then a small output
stride, while also rewarding zero or unit input strides.  This lets
Fortran-contiguous and permuted destinations use their dense direction
without making stepped destinations traverse a distant axis.

The operation kernels own only arithmetic semantics and hot loops.  The
selected inner loop recognizes both common scalar sides:

```text
output stride  lhs stride  rhs stride  specialization
      1             1           1      contiguous vectors
      1             1           0      vector op rhs scalar
      1             0           1      lhs scalar op vector
```

The last route is important for an outer broadcast shaped like `(rows, 1)` op
`(1, columns)`.  It hoists the left value out of the inner loop and lets the
compiler optimize a simple contiguous operation.

A mapping is constant when every non-singleton domain axis has zero stride.
If the other operand already has a dense result layout, out-of-place
execution preserves that layout and uses a full-domain contiguous scalar
kernel.  Singleton-axis strides are ignored when recognizing the constant
mapping.

## Implementation

The prototype adds:

- `plan.hpp` and `plan.cpp` for broadcast domains, mappings, cursors, and
  route selection;
- `kernel.hpp` for operation-specific scalar, vector, and broadcast loops;
- `executor.hpp` for allocation, alias safety, and dispatch;
- `SimpleArrayElementwise.hpp` for the operation-family facade;
- private pybind11 methods for side-by-side measurement;
- focused Python and no-Python C++ tests;
- catalog generation, execution, shard merging, and report rendering tools.

The new code uses `std::ranges::equal` when comparing shape and stride ranges.
This keeps the prototype independent of a known `small_vector` equality
defect in the current base revision.

## Verification

### Correctness catalog

The planned arithmetic path was audited against NumPy across 1,465,004 cases:

| Result | Cases |
| --- | ---: |
| Match | 923,672 |
| Expected invalid-broadcast error | 411,642 |
| Existing array-construction defect | 104,544 |
| Existing complex division semantic gap | 25,146 |
| Unexpected value or exception | 0 |

The construction failures occur before elementwise execution and are retained
as a separate benchmark finding.  Complex division by zero currently raises
from the existing solvcon complex type, while NumPy produces IEEE `inf` or
`nan`; the runner labels that behavior explicitly instead of treating it as a
traversal failure.

Focused verification also covers logical-coordinate traversal for contiguous,
Fortran, transposed, reversed, and stepped arrays; mixed-rank broadcasting;
empty domains; invalid fixed destinations; partial overlap; and compact
result allocation.

### Performance evidence

Measurements used WSL2 on x86-64, Python 3.12.7, NumPy 2.3.0, and one thread
for the common numerical-library environment variables.  All runs were
pinned to the same CPU.  Systematic runs used medians of five or seven samples
after two or three warmups.  The timer repeated each callable long enough to
reach a one- or five-millisecond target.  Suspicious extremes were repeated
with 15 samples, five warmups, and a 20-millisecond target.

The performance catalog contains 340,480 combinations, of which 233,128 are
valid under NumPy.  Stable timing of every valid Cartesian combination would
be needlessly expensive, so the performance audit uses these overlapping
systematic slices:

- every valid size-32 combination across 22 topologies, nine left layouts,
  ten right layouts including Python scalars, four operations, two dtypes,
  and both execution modes;
- all sizes from 1 through 1024 in the catalog for C/C and Python-scalar
  operands;
- sizes 8, 128, and 512 with every left layout and then every right layout;
- every catalog size and layout side for singleton-array broadcasts.

After duplicate case identifiers are removed, these slices contain 33,368
timed cases.  All reports identify revision `0b403d68`.  Overall, the planned
path wins 30,429 cases (91.19%).  The median NumPy/planned ratio is 1.62x,
with p10 at 1.04x and p90 at 2.67x.

| Topology family | Cases | Wins | Median NumPy / planned |
| --- | ---: | ---: | ---: |
| Non-broadcast | 4,160 | 55.87% | 1.13x |
| Python scalar | 672 | 62.05% | 1.11x |
| Singleton broadcast | 9,952 | 95.08% | 1.48x |
| Single-axis broadcast | 9,664 | 98.98% | 1.72x |
| Outer broadcast | 2,288 | 98.56% | 1.73x |
| Mixed-rank broadcast | 6,632 | 96.59% | 1.78x |

The result is not a universal advantage over NumPy.  Small non-broadcast
arrays expose the prototype's fixed planning and binding cost.  Python
scalars and some stepped or reversed division cases also lose.  The strongest
evidence is the broadcast work this prototype targets: every broadcast family
has a median advantage, and the four array-broadcast families win more than
95% of their measured cases.

## Out of scope

- Replacing the existing public arithmetic methods before the prototype is
  reviewed.
- Comparison operators.
- Changing solvcon complex division semantics.
- Fixing the independent `small_vector` construction or equality defects
  globally.
- Extracting a common planning layer with matrix multiplication before the
  two implementations expose a stable shared vocabulary.
- Claiming a universal performance advantage over NumPy.

## Delivery status

- Branch: `codex/prototype-elementwise-broadcast`
- Correctness catalog: complete
- Focused tests: complete
- Performance specialization: complete for layout-selected inner loops and
  dense singleton broadcasts
- Commits: split into benchmark, implementation, and documentation concerns
- CI: pending
- Documentation preview: blocked locally by missing Doxygen and Sphinx

## Chat history

1. The user asked to study the existing elementwise benchmark notes and use a
   broad benchmark, including broadcasting, before optimizing.
2. The user required independent bugs to be separated from the optimization
   work, following earlier shape and equality findings.
3. The user asked for a prototype architecture analogous to the current
   matrix-multiplication planner, followed by evidence before considering a
   shared layer.
4. The user approved implementing the architecture and asked for conditions
   where the planned path beats NumPy.
5. The user required performance evidence for non-broadcast execution and
   different layouts, not only outer broadcasting.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
