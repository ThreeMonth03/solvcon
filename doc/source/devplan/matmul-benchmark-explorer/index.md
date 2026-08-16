# Benchmark Visualizer

## Problem

Matmul can use Naive, BLAS, packed BLAS, or Winograd execution, but Pilot
cannot explain which dispatch is selected or compare eligible alternatives for
one physical input pair. Existing profiling also does not create a reusable
data set for studying dispatch boundaries.

The prototype adds two connected pages:

1. Route Inspector builds one input pair from dtype, complete shapes, and
   element strides. It verifies every selected dispatch against NumPy, records
   native and Python end-to-end timing, and shows a table plus hoverable timing
   chart.
2. Dispatch Atlas collects or loads many completed observations and renders
   exact feature coordinates as a rotatable 3-D point cloud. Rendering,
   filtering, and changing axes never start a benchmark.

The window has an Operation selector with Matmul as its only current option.
It reserves a clear UI extension point without making the current collector or
artifact schema pretend to support operations that do not exist yet.

## Architecture

```text
Route Inspector or Atlas collection dialog
                    |
                    | JSON request
                    v
        isolated headless QProcess
                    |
                    | activity before and after expensive work
                    | validated observations and complete-round partials
                    v
             atomic JSON artifact
                    |
              +-----+-----+
              |           |
              v           v
       timing chart   3-D point cloud
```

Public matmul enters `MatmulPlan` and `MatmulExecutor` in
`cpp/solvcon/buffer/matmul.hpp`. The prototype adds immutable input-scoped
`MatmulRoute` objects, route enumeration, forced execution, and native timed
batches. Forced execution rejects a route created for a different operand
pair.

The headless request, collection, artifact, and feature code lives in
`solvcon/matmul_benchmark/`. Pilot integration lives in adjacent
`solvcon/pilot/panel/_matmul_benchmark*.py` modules. Qt launches the worker
through `solvcon_matmul_benchmark.py`; benchmark code never runs in the GUI
process.

## Dispatch recipes

Automatic selection and structural eligibility are separate. For example, a
BLAS route can be eligible even when Auto selects Naive for a small input. A
route recipe records:

- the dispatch name;
- whether Auto selected it;
- eager whole-operand packing; and
- per-contraction scratch packing.

The UI defaults every eligible and build-supported dispatch to checked. Shape,
stride, and dtype edits immediately disable structurally unavailable choices.
The request carries the checked dispatches explicitly; it never silently runs
an unchecked forced dispatch.

Naive reads signed strides directly. BLAS consumes compatible vector or matrix
views and may pack an operand eagerly or into bounded scratch. Winograd is
available only for its supported matrix shapes. Automatic policy remains
unchanged by the prototype.

## Input cases

Route Inspector accepts complete A and B shapes and signed element strides.
The Atlas collection editor offers the same exact-case entry plus a sweep
helper:

- M, K, and N use start, end, and either linear or powers-of-two spacing;
- common unbatched, matched-batch, reuse-A, and reuse-B presets only fill
  editable fields;
- A and B core storage, batch extents, and batch strides remain editable; and
- an exact case contributes one cell regardless of the sweep grid.

The preview reports resolved shapes, strides, storage span, output shape,
derived M/K/N, packing, and available dispatches. Invalid contraction,
broadcast, rank, or storage combinations cannot be accepted. Layout and
broadcast names are derived facts for filtering, not required user input.

## Measurement

Correctness runs before timing. A wrong or non-finite result is recorded but
excluded from timing and winner selection. Requested routes and cells use a
deterministic balanced order.

Each measurement round has two independent scopes:

- `native_batch` measures repeated native calls with a C++ steady clock. It
  excludes Python call overhead.
- `python_end_to_end` measures complete calls from Python with
  `perf_counter_ns`. NumPy appears only in this scope, so `vs NumPy` compares
  the same call boundary.

Summaries contain median, MAD, p95, minimum, and maximum. Winner and runner-up
use native medians from correct forced routes. Auto and NumPy cannot become
the winner.

The quality selector always shows its schedule in plain language:

- Preview: two discarded setup calls, five calls averaged per result, and two
  measurement rounds.
- Stable: four discarded setup calls, twenty calls averaged per result, and
  eight measurement rounds.
- Custom: the same three counts are directly editable and validated.

There is no machine-independent compute-time cutoff. A large valid dispatch
may run slowly. The worker emits activity before and after provenance,
preparation, reference calculation, each correctness check, every setup or
timed block, partial writing, result validation, and final artifact writing.
Both pages therefore show the current dispatch, phase, round, call count, and
elapsed time while work is running.

Stop terminates the isolated worker and escalates to a forced kill after one
second. Late output cannot replace the cancelled state. Controls recover after
cancellation, protocol errors, worker crashes, or artifact-load failures. A
normal window close stops its child on every platform. Linux additionally
requests a parent-death signal to avoid an orphan after an abnormal Pilot exit.

## Partial results

Atlas writes one ordinary validated collection after every complete
measurement round. The write uses a sibling temporary file, flush, fsync, and
atomic replace. A partial never contains half a round.

On success, the worker publishes the final artifact and removes the partial.
On Stop or worker error, Atlas loads the latest valid partial if one exists.
There is no resume protocol, calibration state, shard model, or alternate
duration schema. Starting again creates a new fixed run.

This fallback is deliberately small. It preserves useful completed data
without making cancellation depend on a long-running recovery subsystem.

## Resource safety

Runtime safety remains fail-closed for inputs and memory. The worker validates
shape, stride span, output size, route eligibility, call count, collection
size, and projected artifact size before allocation. It resolves a
worker-local memory budget from OS and cgroup headroom, reserves half for the
system and native libraries, and rechecks immediately before preparing each
case.

Operation-count estimates are descriptive only. They do not disable a valid
dispatch or open a large-work confirmation dialog. The user sees slow work and
may stop it. Memory limits, malformed inputs, and structurally impossible
dispatches still fail instead of relying on manual cancellation.

The collection memory estimate includes physical stride spans, possible
logical materialization, retained prepared operands, transient correctness
outputs, and Winograd scratch. Plan parsing and artifact loading do not depend
on the current host budget, so a result remains readable on a smaller machine.

## Artifact model

A benchmark artifact records:

- dtype, complete shapes, element strides, and derived contraction facts;
- enumerated dispatches, Auto selection, and packing recipes;
- correctness results and raw timing blocks for both scopes;
- timing summaries, winner, runner-up, margin, and NumPy ratios; and
- process, machine, build, NumPy, loader, threading, and affinity metadata.

A collection preserves every source request, metadata block, observation, and
raw panel with explicit source linkage. The schema validates nested types,
route references, panel order, and collection provenance on write and load.
Offline merging does not assume that different machines or builds are
comparable; Atlas exposes those source facts as filters.

The prototype uses one current schema. Earlier prototype-only schemas and
aliases are intentionally not retained.

## Route Inspector presentation

The timing chart uses horizontal median bars and p95 whiskers. Native and
Python end-to-end scopes are selectable, with visible helper text explaining
what each includes. Exact median, p95, noise, speedup, packing, correctness,
and selection details appear on hover. Chart and table selection stay
synchronized.

Dispatch checkboxes use Naive, BLAS DOT, BLAS GEVM, BLAS GEMV, BLAS GEMM, and
Winograd labels. They are generated in canonical request order and react to
the current input instead of asking users to type route tokens.

## Dispatch Atlas presentation

Atlas exposes numeric contraction, stride, batch, timing, and packing features
plus restricted numeric expressions such as `M*K*N` or `log2(M/K)`. Exact
categorical constraints cover dtype, derived layout, broadcast behavior,
machine, schedule, and thread count.

The renderer plots exact observations as a 3-D point cloud without
interpolation. Samples sharing one coordinate are grouped. Conflicting winners
split the marker, while tooltips retain exact coordinates, counts, dispatches,
packing, layout, margin, noise, and source IDs.

The view supports orbit, zoom, fixed views, point size and opacity controls,
and orthographic or perspective projection. A high-separation palette keeps
dispatch colors distinct. Noise and ambiguity alter the outline instead of
changing route identity.

The renderer stays on QtGui/QPainter. Normalized inputs, projection, depth
order, and screen-space picking buckets are cached. Paint and hit testing share
one projection, and hover checks only nearby buckets. Qt3D is not a prototype
dependency.

## Verification

Native route tests force every enumerated dispatch across float and complex
dtypes, compact and strided storage, broadcast and batch cases, empty domains,
and input identity checks. Backend tests cover correctness gating, balanced
schedules, timing scopes, strict artifacts, merging, resource validation,
complete-round partials, worker failure, and parent-death cleanup.

Pilot tests cover protocol validation, activity rendering, Stop races,
partial-result fallback, schedule controls, exact and swept inputs, timing
charts, point-cloud projection, cache invalidation, indexed picking, and clean
window shutdown. A real worker smoke verifies the complete protocol outside
the fake-process tests.

## Out of scope

- Changing automatic dispatch thresholds.
- Fitting or installing a production cost model.
- Resuming a cancelled collection.
- Predicting portable wall time from operation counts.
- Inspecting an external BLAS thread team.
- Interpolating unsampled feature-space regions.
- Incrementally parsing a near-limit completed JSON artifact in Qt.
- Abnormal parent-death cleanup beyond Linux.

## Delivery

The prototype is developed on `prototype/matmul-benchmark-explorer` in the
personal fork. It remains a product and task-design vehicle. Upstream work is
split only after this implementation is simplified, verified, and measured by
actual changed lines.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
