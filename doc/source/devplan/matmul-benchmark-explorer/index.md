# Benchmark Visualizer

## Problem

The matmul implementation can select generic, BLAS, packed BLAS, and
Winograd execution, but the Pilot cannot explain or compare those routes for
one physical operand pair. Existing profiling also does not provide a
reusable data set for studying dispatch boundaries in transformed feature
spaces.

The prototype provides two connected views in one Pilot MDI window:

1. Route Inspector creates a deterministic operand pair from a dtype, full
   shapes, and element strides. It checks automatic dispatch and every
   requested structurally eligible route against NumPy, then reports route
   selection, packing, correctness, and two timing scopes. A timing chart
   compares medians and p95 values and exposes exact values on hover. The
   default is to benchmark every eligible route. Dispatch checkboxes select a
   subset without requiring route names to be typed. Every currently eligible
   and build-supported choice starts checked. Shape, stride, and dtype edits
   immediately update disabled choices. Sampling changes affect only the
   requested measurement schedule.
2. Dispatch Atlas consumes completed in-memory results, validated artifact
   files, or validated merged collections. A completed Route Inspector run
   is added automatically. A configurable starter action can collect an
   explicit shape grid in one isolated worker, then load its completed
   collection. Changing an axis, filter, slice, camera, or projection never
   starts collection.

The window has a top-level Operation selector whose only current option is
Matmul. It is a visible shell-level extension point for future
operation-specific views. The matmul request, artifact, and collector remain
matmul-specific instead of introducing unused generic wrappers.

## Code layout

Public matmul enters `MatmulPlan` and `MatmulExecutor` in
`cpp/solvcon/buffer/matmul.hpp`. Before this prototype, the Python binding in
`cpp/solvcon/buffer/pymod/wrap_SimpleArray.hpp` exposed only automatic
matmul. The prototype adds immutable, operand-scoped `MatmulRoute` objects,
route enumeration and forced execution, and native timed-batch methods.
Forced execution revalidates that a route belongs to the same operand pair
and still matches its enumerated recipe.

The headless collector, schema, artifact I/O, scheduling, and feature engine
live in `solvcon/matmul_benchmark/`. The JSON-lines worker is launched through
`solvcon_matmul_benchmark.py`. Pilot integration lives in
`solvcon/pilot/panel/_matmul_benchmark.py`, with the QProcess controller and
Atlas in adjacent modules.

The existing call profiler remains a separate diagnostic tool. The
benchmark collector does not invoke it or use its call tree as timing data.
Artifacts record the `SOLVCON_PROFILE` value visible in the worker
environment; the worker does not claim that a profiling-capable extension is
unprofiled.

## Design

```text
Qt Route Inspector
        |
        | versioned request over stdin
        v
headless worker in a separate QProcess
        |
        | enumerate eligible recipes
        | verify auto and requested routes
        | run balanced native and Python panels
        v
versioned observation artifact
        |
        +----------------------+
        |                      |
        v                      v
Route Inspector result     Dispatch Atlas
                           offline 3-D point cloud
```

### Route recipes

Automatic policy and structural eligibility are separate concepts. A BLAS
route below the automatic threshold is still benchmarkable when its operand
roles and layout can support that route. The core enumerates immutable route
recipes and permits forced execution only with one of those input-scoped
objects.

A recipe records:

- the kernel;
- whether automatic policy selected it;
- eager whole-operand packing; and
- per-contraction scratch packing.

Native eligibility is represented by membership in the enumerated route set,
not by an `eligible` boolean on each artifact row. A collection plan resolves
its requested routes per cell and omits only structurally invalid routes. The
artifact records only routes that were requested for that cell. A direct
request for an absent route fails validation rather than producing an
ineligible measurement row.

Eager packing materializes the complete logical operand before batch
traversal. GEMM scratch packing uses bounded matrix storage during traversal
and reuses a packed matrix while its source pointer is unchanged. A broadcast
operand can therefore require one scratch materialization rather than one per
output batch item. The Atlas `packing_bytes` feature counts these source
pointer transitions.

Broadcast behavior is derived from complete operand shapes, element strides,
and the aligned output batch shape. It is not stored as a separate boolean in
the artifact. Missing leading batch axes, size-one expansion, and explicit
zero batch strides all contribute to the derived broadcast profile.

### Measurement process

Qt owns a `QProcess`. Before startup it sets the OpenBLAS, OpenMP, MKL,
vecLib, and BLIS thread environment variables to the requested value. The
collector rejects a request if any of those values differs inside the child.
The controller sends one JSON request on standard input and accepts validated
JSON-lines activity, progress, checkpoint, result, or error events on standard
output. The worker flushes an activity event before and after input
preparation, the NumPy reference, each correctness check, each setup block,
each timed dispatch block, result validation, checkpoint I/O, and final
artifact writing. It identifies the route, the route selected by Auto, the
phase, collection cell, measurement round, and call range. A Qt-side timer
updates elapsed time without adding worker I/O inside a timed block.

One window-level Stop remains visible across both pages and names the active
page. Stop first freezes the visible route and elapsed time, then immediately
kills the isolated worker. A late result or error cannot replace the cancelled
terminal state. Unexpected worker exits retain the last activity context and
restore both pages so another run can start. On Linux the worker also requests
a parent-death signal so an abnormal Pilot exit does not leave an orphan
benchmark process.

The GUI limits the selectable BLAS thread count to one less than the detected
logical CPU count, with a minimum of one. The reserved logical CPU leaves
capacity for Qt and progress handling instead of presenting an all-core run as
uncontaminated. The headless request schema still accepts an explicit all-core
configuration for controlled environments.

Correctness is checked against NumPy before any candidate is timed. An
incorrect or non-finite candidate remains in the artifact with its failure
details, but is excluded from both timing schedules and from winner
selection. Routes are visited in a deterministic balanced order.

Each logical panel records two distinct scopes:

- `native_batch` uses a C++ steady clock around repeated plan construction,
  output allocation, executor construction, packing, and execution. The
  binding releases the GIL. Forced-route validation happens once before this
  timed loop, while automatic selection remains part of each automatic call.
- `python_end_to_end` uses `perf_counter_ns` around repeated Python calls. It
  includes the Python binding boundary and per-call forced-route validation.
  NumPy is measured only in this scope, so `vs NumPy` compares like
  boundaries.

Each candidate contributes one elapsed block and one derived per-call latency
to each applicable scope in every panel. Summaries contain median, MAD, p95,
minimum, and maximum. The winner, runner-up, and winner margin use only the
native medians of correct forced routes; neither the automatic candidate nor
NumPy can become the winner.

No Qt event or artifact serialization is inside either timed block. The GUI
always explains the selected quality in plain language. Preview records two
measurement rounds of five averaged calls after two discarded setup calls.
Stable records eight rounds of twenty averaged calls after four discarded
setup calls. Fixed-budget runs also offer a Custom schedule with those three
counts exposed together and validated against the existing resource limits.
Target-duration runs do not expose the custom counts because calibration owns
their repetitions and final round count.

### Starter collection plan

The Atlas starter action creates a versioned collection plan before starting
its worker. The default plan fixes `K=64` and samples
`M,N={8,16,32,64,128,256}` in Preview mode, for 36 explicit cells. The plan
editor uses start/end controls with either linear spacing or powers of two,
so dense arithmetic grids and `8,16,...,1024` do not require comma-separated
input. Dispatches are explicit checkboxes.

Inputs are configured as complete A/B cases rather than selected by opaque
layout names. Each case chooses one of two modes. `Exact case` accepts the
complete A and B shapes and signed element strides directly, validates the
contraction and broadcast rules, and contributes exactly one collection cell.
`Preset / sweep` describes storage and batch reuse once, then expands it across
the selected M/K/N grid. Presets only fill editable fields for common
unbatched, matched-batch, reuse-A, and reuse-B inputs. The user may still set
row and column strides and every batch-axis extent and stride explicitly.

The resolved preview shows exact A/B shapes, element strides, backing storage,
output shape, derived M/K/N, predicted eager or scratch packing, and available
kernels before a case is accepted. Invalid shapes, contractions, broadcasts,
or storage spans disable acceptance. Layout and broadcast categories are
derived facts used by the artifact and Atlas, not names the user must choose.
Ineligible routes, such as Winograd for a batched input, are omitted before
collection. Eligible routes remain selectable even when a particular shape or
sampling schedule may take a long time. The live activity display and Stop
control make that cost observable instead of predicting it from a machine-
independent work threshold.

The editor separately controls data-quality intent and run budget. Fixed mode
uses Preview, Stable, or the validated Custom fixed schedule exactly.
Target-duration mode treats Preview or Stable only as the minimum
round/setup-call quality intent. It
first calibrates every correct timing stream on the current machine, then
chooses repetitions and complete balanced panels for the requested wall time.
Before allocation, it verifies input, memory, call-count, and artifact bounds.
The built-in choices are one minute, ten minutes, one hour, and a custom
duration. Long schedules are split into independently bounded shards and
checkpointed only after complete balanced-panel boundaries. A durable output
path is required so a worker restart can validate and resume the last complete
shard. Abstract work units remain a hardware-neutral size estimate and are
never presented as seconds or used as an execution limit.

Before launch, the editor shows the exact fixed schedule or the minimum
pre-calibration schedule, plus cell, call, estimated work, prepared-case
memory, current worker-safe memory budget, and conservative artifact-size
estimates. Estimated work is descriptive. It does not disable a dispatch,
reject an otherwise valid run, or open a confirmation dialog. Preview, Stable,
and Custom counts remain unchanged. Target-duration plans use their explicit
wall-time budget and checkpoint interval.

The version-4 collection schema retains `allow_large_work` so existing
prototype plans and artifacts still round-trip, but execution no longer uses
it as permission. Runtime safety checks cover input validity, dispatch
eligibility, bounded allocation and peak memory, collection size, call count,
and projected artifact size. They do not attempt
to translate scalar operation counts into portable wall time. A slow route is
reported as the current activity and remains manually stoppable.

Memory is always fail-closed and cannot be confirmed away. The worker resolves
half of the tightest current OS or cgroup memory headroom, with a 512 MiB
fallback and a 4 GiB single-allocation cap. It checks the complete retained
collection against that budget before allocation. This lets a capable machine
accept a contiguous 16384-square float32 or float64 case while a constrained
worker rejects the same plan. The remaining fixed hard limits are 16,384
cells, one million calls, and a projected 512 MiB artifact. Target-duration
call and artifact limits apply independently to every materialized shard.
Plan parsing and artifact provenance checks do not apply the current host
budget, so a result remains readable on a machine too small to execute it.

Collection keeps each input and native operand prepared so panels can revisit
cells in balanced order. NumPy correctness references are released after
validation; only their output shapes remain. The aggregate memory estimate
therefore sums retained operand storage and adds the largest transient output
allowance, rather than charging every cell for a retained reference. The
transient allowance reserves eight output-sized arrays for the reference,
candidate results, and NumPy correctness-comparison temporaries. Physical
stride span and conservative logical materialization remain accounted for,
so broadcast and zero-stride views cannot hide packing cost.
One-level Winograd adds its two half-size operand scratch blocks explicitly to
the estimate. Opaque BLAS thread workspace is covered by retaining half of the
detected memory headroom for the system and backend.

A plan stores primitive shapes and element strides, not derived feature
coordinates. It has a canonical measurement hash and a seeded balanced cell
order. All cells pass input, route, resource, and correctness preparation
before calibration or timed panels begin. Fixed collections record the exact
plan, estimate, seeded cell order, and source-linked raw panels for every cell.
A target-duration collection additionally records calibration evidence, the
predicted schedule, compact shard descriptors, and cumulative observations
recomputed from every referenced raw shard. The Atlas renders one cumulative
point per input cell while the raw bounded requests remain available for
audit. Route and cell balance use a global panel offset and therefore continue
across checkpoints instead of restarting at each shard.

If no output path was configured for a fixed run, Atlas retains the completed
document in memory, enables Save collection, and warns before replacing it or
closing the window. A target-duration run requires a durable path. An error or
cancellation never publishes a malformed final artifact; the previous
complete-shard checkpoint remains valid and resumable.

### Artifact

The version 1 artifact stores source facts and measured observations:

- dtype, complete shapes, element strides, and the derived contraction;
- the enumerated route set, automatic selection, and packing recipes;
- correctness details and raw blocks for both timing scopes;
- native and Python summaries, the native winner, and the NumPy ratio; and
- process, machine, build, NumPy, native-loader, thread, and affinity
  identities.

The schema validates nested fields and cross-references on write and load.
Offline merging accepts validated benchmark artifacts and preserves each
source request, metadata, panel order, and raw sample block. It does not
impose a machine or build compatibility policy. Atlas exposes source context
such as architecture, machine, system, mode, and thread count as filters so
users can select the comparison domain explicitly.

### Feature projection

Dispatch Atlas renders numeric axes chosen from the feature registry.
The registry exposes contraction sizes, ranks, core strides, aligned batch
extents and strides, winner timing, and packing estimates. Users may add
session-local numeric expressions such as `M*K*N`, `log2(M/K)`, or
`packing_bytes/useful_bytes`. The restricted AST evaluator permits numeric
operators and a small function set; it never evaluates arbitrary Python.
The headless projection helper also supports one to three registered axes.

Categorical values such as dtype, layout, broadcast profile, machine, mode,
and thread count are available to the two exact-value constraint controls.
For a numeric feature, a control selects one coordinate already present in
the loaded observations. For a categorical feature, it selects one exact
category.

The Atlas renders exact observed coordinates as a 3-D point cloud and never
interpolates. Samples sharing a coordinate are grouped; conflicting winners
split the marker, while a count and tooltip preserve multiplicity. The view
supports orbit, zoom, fixed camera views, and orthographic or perspective
projection. It does not create voxels, surfaces, or inferred samples.

The prototype keeps a QtGui/QPainter software renderer instead of adding a
Qt3D dependency. Input normalization is cached until the observation set
changes. Projection and depth sorting are cached until the camera or viewport
changes, and painting and picking share the same projection. A screen-space
uniform grid limits hover and click testing to nearby points. This preserves a
small, deterministic offscreen-test surface while leaving the canvas API open
to a later batched GPU backend if measured point counts require it.

Dispatch identity uses a fixed high-separation categorical palette. Winner
margin makes only a small opacity adjustment so route identity stays clear.
High timing noise changes the outline, and ambiguity adds a dashed outline.
An invalid observation with finite axis coordinates remains visible in gray.
A feature error or non-finite axis coordinate cannot be plotted and is
omitted. Exact feature values, winner counts, layout, packing, margin, noise,
and source IDs remain available in tooltips. The status reports how many
loaded observations are shown.

## Implementation

The prototype is developed on `prototype/matmul-benchmark-explorer` in the
personal fork. Its implementation consists of:

1. input-scoped route inspection, forced execution, and native timed batches
   in the C++ matmul core and Python binding;
2. a headless collector with strict requests, resource checks, atomic
   artifacts, explicit collection plans, offline merging, and dynamic
   features;
3. a QProcess JSON-lines controller plus Route Inspector timing chart and
   Dispatch Atlas 3-D point-cloud Pilot view; and
4. core route, collector, artifact, feature, protocol, rendering, and Pilot
   integration tests.

## Verification

`tests/test_matmul_routes.py` forces every enumerated route across the four
supported dtypes and covers compact, sliced, negative-stride, broadcast,
batched, empty-domain, and input-identity cases. It compares each result with
NumPy and exercises both native benchmark bindings.

`tests/test_matmul_benchmark.py` uses deterministic fake engines for
correctness gating, balanced schedules, both timing scopes, statistics,
resource limits, artifact validation, merging, and feature evaluation. It
also contains a small real child-process smoke test when the native route
binding is available.

`tests/test_pilot_matmul_benchmark.py` uses a fake QProcess to cover protocol,
thread environment, failure, cancellation, result loading, inspector output,
Atlas reprojection, source-context filtering, and shutdown of an active fake
worker. Focused chart, collection-plan, and point-cloud tests cover exact
timing presentation, custom fixed schedules, direct shape/stride cases, plan
estimates and provenance, camera projection, cache invalidation, indexed hit
testing, and the invariant that rendering never starts measurement.
`tests/gui/test_gui_matmul_benchmark.py` checks registration, both tabs, and
clean closure of the real top-level Pilot window.

The final branch is to be checked with the focused Python and GUI suites, the
C++ tests, the Pilot build, project lint, and an offscreen real-application
exercise.

## Out of scope

- Changing dispatch thresholds or automatic policy.
- Fitting or installing a production cost model.
- Treating profiler-derived measurements as winner labels.
- Inspecting the internals of an external BLAS thread team.
- Interpolating unsampled feature-space regions.
- Making a near-limit completed JSON artifact load incrementally in Qt. The
  current synchronous load catches read, validation, and memory errors but is
  not cancellable once parsing begins.
- Providing abnormal parent-death cleanup beyond Linux. Normal window closure
  stops workers on every supported platform.

## Delivery status

- Branch: `prototype/matmul-benchmark-explorer`
- Review vehicle: a fork draft pull request paired with a fork planning
  issue.
- Verification: 155 focused backend/native Python tests, 110 embedded Pilot
  tests, and two real-window Pilot integration tests pass. Project lint also
  passes.
- Live-worker verification: the window-level Stop kills a real Generic native
  call after switching from Route Inspector to Dispatch Atlas and restores
  both pages; a one-cell Atlas completes through artifact writing; a malformed
  request returns to an error state; no worker remains.
- Documentation preview: the focused Sphinx single-page build passes. A full
  documentation build needs the unavailable local `doxygen` and `latex`
  executables.

## Discussion history

The design follows these decisions from the discussion:

- one view generates operands from a supplied dtype, shape, and element
  stride configuration, then compares every eligible route by default;
- another view renders only completed observations in selectable feature
  spaces, including the completed result passed from Route Inspector;
- layout, packing, broadcasting, route eligibility, and timing boundaries
  remain explicit or reproducibly derived from source facts;
- common input presets are optional helpers, while exact A/B shapes and
  element strides remain directly editable;
- Qt stays outside timed regions and a QProcess performs collection;
- the solvcon call profiler is diagnostic only, while winner labels use the
  dedicated native timing scope;
- fork branches use descriptive names without an agent-name prefix; and
- the fork prototype stays on one branch, and task splitting is deferred
  until an upstream proposal is prepared.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
