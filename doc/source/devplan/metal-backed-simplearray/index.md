# Metal-backed SimpleArray prototype

## Goal

This prototype tests whether `SimpleArray` can keep its existing public type
while its owned `ConcreteBuffer` is allocated as a Metal resource. The goal is
not automatic CPU/GPU scheduling. It is to prove the storage and synchronization
foundation needed by a future dependency scheduler.

Apple silicon uses unified memory, but a normal allocation and a shared
`MTLBuffer` are still different allocations. The first GPU prototype copied
between them for every operation:

```text
SimpleArray malloc -> copy -> MTLBuffer -> GPU -> copy -> SimpleArray malloc
```

The new prototype selects the storage when the array is allocated:

```text
                         +-> HostOwned: malloc
SimpleArray -> ConcreteBuffer
                         +-> Metal: shared MTLBuffer
```

`MTLBuffer.contents` remains CPU-addressable. Existing pointer APIs therefore
still work, but they become synchronization boundaries for Metal-backed arrays.
The current runtime deliberately requires a unified-memory Metal device. A
discrete GPU needs managed-memory dirty tracking and is not yet a resident
implementation.

## Portable backend boundary

`ConcreteBuffer` does not select a platform API. Owned allocation crosses one
backend-neutral interface:

```text
SimpleArray
    |
ConcreteBuffer
    |
device::BufferBackend
    +-- CpuBufferBackend          every platform
    +-- MetalManager              macOS build only
    +-- future CUDA/HIP backend   separate task
```

`BufferDevice` identity and string conversion live in
`device/BufferDevice.hpp`. `BufferBackend` separates compile-time support
(`built()`) from runtime availability (`available()`) and owns the allocation
entry point. The single dispatcher lives in `device/BufferBackend.cpp`.
`ConcreteBuffer.hpp` contains neither Metal types nor Metal build guards.

A build without Metal still supplies an unavailable Metal provider. Requesting
Metal storage produces a runtime error, while CPU allocation follows the same
provider contract. `device/CMakeLists.txt` includes generic sources on every
platform and adds `metal.mm` only for a Metal build. Windows and Ubuntu never
compile Objective-C++ or include an Apple framework.

This boundary deliberately covers allocation, not every GPU API. The current
`ConcreteBuffer` contract still requires non-CPU storage to expose a stable
CPU-visible address because existing `SimpleArray` pointer APIs depend on it.
Shared `MTLBuffer` satisfies that contract. A device-only CUDA or HIP backend
needs a separate host-mapping or staging design before registration; it must
not pretend that a device pointer is CPU-addressable.

## Portable float16 contract

`solvcon::Float16` aliases the header-only `half_float::half` 2.2.1 type. The
dependency is pinned by version and SHA-256 during CMake configuration. This
keeps IEEE 754 binary16 storage available to C++23 builds on Clang, GCC, and
MSVC instead of selecting a compiler extension or an Apple-only scalar type.

`DataType::Float16`, `SimpleArray<Float16>`, `SimpleArrayPlex`, and
`SimpleCollector<Float16>` use the same two-byte storage with or without
Metal. The Python binding exposes `SimpleArrayFloat16` and maps it directly to
NumPy `float16`, including zero-copy NumPy views and the buffer protocol.

Metal GEMM carries its element type as operation metadata. The MPS executor
selects `MPSDataTypeFloat16` or `MPSDataTypeFloat32` and validates byte ranges
with the matching item size. The array and synchronization model does not
branch on dtype.

The separation leaves four reviewable follow-up tasks:

1. Add a device identity and provider to the backend dispatcher.
2. Implement storage ownership and allocation-wide access state in the new
   platform directory.
3. Add explicit compute operations without changing buffer allocation.
4. Expose backend capabilities and operations in the Python binding.

## Public experiment

CPU storage remains the default:

```python
cpu = sc.SimpleArrayFloat16([1024, 1024])
gpu = sc.SimpleArrayFloat16([1024, 1024], device="metal")

gpu = sc.SimpleArrayFloat16(array=numpy_array, device="metal")
result = gpu.matmul_metal(weight)
result.wait()
host_copy = result.cpu()

read_view = result.host_view()
write_view = result.host_view(write=True)
```

The initial NumPy-to-Metal construction is one explicit copy into a shared
`MTLBuffer`. Metal results reuse Metal storage. A chain does not copy data back
to CPU between operations:

```text
A, W in MTLBuffer
       |
       +-> matmul_metal -> C0 (pending)
                              |
                              +-> matmul_metal -> C1 (pending)
                                                     |
                                                     +-> wait once
```

## Explicit execution policy

Storage placement and operation execution are separate choices:

```text
SimpleArray(..., device="cpu" | "metal")  -> storage choice

a.matmul(b)                               -> explicit CPU operation
a.matmul_metal(b)                         -> explicit Metal operation
```

The runtime does not compare matrix sizes and does not switch either call to a
different backend. Its scheduler only orders declared CPU/GPU access to shared
storage. Benchmark data describes the tradeoff for users and future explicit
policies; it is not input to an automatic threshold.

Most existing CPU implementations are still conservative host boundaries in
this prototype. `sum()` and `fill()` use scoped internal leases: they wait only
for prior work on the allocation, declare CPU access for the duration of the
operation, and leave the allocation eligible for later Metal work.

## Asynchronous access contract

A mutex alone cannot make an escaped raw pointer safe. The library can wait
when `data()` is called, but it cannot observe when the caller stops using the
returned pointer. This prototype uses a conservative allocation-wide state:

```text
GPU eligible -> pending GPU work -> wait() -> GPU eligible
                         |
                         +-> data()/iterator/.ndarray
                                  -> wait -> host exported
                                             |
                                             +-> reject later GPU submission

GPU eligible -> host_view() -> active host lease -> release last view
                                      |                    |
                                      +-> reject GPU       +-> GPU eligible
```

`wait()` does not export a host pointer, so the array remains GPU-eligible.
`data()`, iterators, references, spans, mdspans, `.ndarray`, and the Python
buffer protocol wait for pending work and mark the shared `ConcreteBuffer` as
host-exported. All reshape and transpose views share this state. Calling
`to("metal")` creates a new GPU-eligible snapshot.

Internal copies, `sum()`, and `fill()` use scoped host-access guards. Public C++
`host_read()` and `host_write()` return the same RAII lease. Python
`host_view(write=False)` returns a read-only NumPy view, while
`host_view(write=True)` returns a writable view. The NumPy base object owns the
lease, so derived NumPy views extend it automatically. A GPU submission while a
lease is active fails instead of blocking; releasing the last view makes the
allocation GPU-eligible again.

Lease acquisition and GPU submission update the access state under the same
short allocation lock. CPU code does not hold that mutex while processing the
payload. Instead, an active-access count prevents a command from being
submitted until the recoverable CPU access ends. This avoids both an unnoticed
race and a same-thread deadlock.

Allocation ownership and access coordination are separate. A
`ConcreteBufferRemover` only controls memory lifetime. An optional
`ConcreteBufferAccess` owns device identity, pending work, host leases, and the
sticky-export state. Ordinary malloc and external NumPy buffers have no access
control block, so `begin()`, `end()`, `data()`, and `logical_data()` take one
inline unlikely null-pointer branch and return the cached CPU pointer. They do
not call a virtual no-op through the remover. Metal buffers enter the slower
access path only at a host boundary.

The first unrestricted Metal host export waits for pending work and sets an
irreversible atomic export flag under the allocation lock. Later raw access
observes that flag and returns without taking the mutex again. Scoped library
operations acquire one lease and use unchecked pointers for their complete
loop, rather than repeating synchronization checks for each element.

The Metal runtime uses one serial command queue. `matmul_metal()` commits a
command buffer and returns immediately. A subsequent operation can consume the
pending output because queue order supplies the GPU-to-GPU dependency. CPU work
on unrelated allocations may run concurrently. A host read of the same
allocation waits for its last submitted task.

Completed commands release their retained buffers and predecessor tasks from a
completion handler. Only a compact success or deferred-error state remains.
This keeps repeated operations on a long-lived input from retaining the whole
command history.

## Prototype scope

The storage experiment covers every `SimpleArray` dtype. GPU computation is
intentionally narrower:

- Metal-backed owned `ConcreteBuffer` storage on unified-memory macOS devices.
- Existing CPU pointer, iterator, span, NumPy, and Python buffer APIs.
- Explicit `device`, `to("metal")`, `cpu()`, `ready`, and `wait()` APIs.
- Native FP16 and FP32 two-dimensional, positive row-major matrix
  multiplication.
- Asynchronous GPU submission and resident matmul chaining on one queue.
- Scoped C++ host read/write access and lifetime-bound Python NumPy views.
- Scoped internal host access for `sum()`, `fill()`, and buffer copies.
- Nullable access control that leaves the CPU/NumPy pointer path free of
  remover dispatch and avoids repeated Metal export locks.
- Allocation-wide dependency tracking shared by all views.
- CPU and non-Metal builds retain their existing default behavior.
- Backend-neutral allocation dispatch keeps Metal types and build guards out
  of `ConcreteBuffer`.

The Python experiment is exposed through the typed classes such as
`SimpleArrayFloat16` and `SimpleArrayFloat32`. The type-erased
`solvcon.SimpleArray` (`SimpleArrayPlex`) is not device-aware in this
prototype.

The prototype does not yet include:

- Automatic CPU/GPU placement. Backend selection remains explicit, so no size
  threshold is planned.
- Batched matmul, broadcasting, vectors, or arbitrary strided GPU inputs.
- GPU elementwise operations or reductions.
- Native complex computation or approximate FP64 routes.
- Buffer pooling, suballocation, byte-range locking, or multiple queues.
- Managed-memory discrete GPUs and their CPU/GPU dirty-state transitions.

Most existing CPU implementations are conservative host boundaries. `sum()`
and `fill()` prove the recoverable model and leave the input eligible for
another Metal operation. Metadata-only views and `clone()` preserve Metal
storage. Legacy operations that materialize a new layout may still return CPU
storage.

The Metal executor already consumes `MatmulPlan` for output shape, dimensions,
and matrix strides. A future executor can extend that use to vector roles,
batch shape, and broadcasting without duplicating the semantic rules.

## Benchmark protocol

The profiling script compares public CPU and Metal matmul chains after both
sets of operands have been allocated. The Metal timed region includes output
allocation, command-buffer creation, encode, commit, and one final wait. It has
no input upload, output download, or result inspection. Both routes include
Python dispatch, output allocation, and result destruction. Every Metal
operation currently has its own command buffer. This is a steady-state resident
public-API comparison, not kernel-only timing or an end-to-end transfer test.

The previous diagnostic table was collected before completed task history was
compacted. It is intentionally omitted because its timing depended on warmup
and sample history. The final committed implementation must be rerun before
performance numbers are published. The explicit API has no dispatch threshold.
These measurements only characterize the two user-selected operations; one
machine and one square sweep could not support a portable policy in any case.

Build a fresh Release extension and reproduce a backend-managed-thread
diagnostic with:

```bash
bench_build=$(mktemp -d /tmp/solvcon-metal-build.XXXXXX)
make buildext BUILD_PATH="$bench_build" CMAKE_BUILD_TYPE=Release \
  BUILD_METAL=ON BUILD_QT=OFF SOLVCON_PROFILE=OFF \
  USE_CLANG_TIDY=OFF USE_CCACHE=OFF

env -u VECLIB_MAXIMUM_THREADS -u OPENBLAS_NUM_THREADS \
  -u OMP_NUM_THREADS -u MKL_NUM_THREADS CMAKE_BUILD_TYPE=Release \
  python3 profiling/profile_metal_simplearray.py \
  --run --dtype float16 --sides 512 1024 2048 4096 --depths 1 4 \
  --warmups 1 --samples 3 --rounds 4 \
  --output /tmp/solvcon-metal-simplearray.jsonl
```

The profiler records the loaded extension path and SHA-256, linked libraries,
Git state, machine and Metal device, power and thermal state, busy processes,
and every raw timing sample. A publishable run should start from a clean commit
on AC power with Low Power Mode disabled. Repeat the sweep in both forward and
exactly reversed size/depth order. Treat repeat drift as instability instead of
pooling it into one headline number. The build's `CMakeCache.txt` should be kept
beside the raw JSONL evidence.

## Clean FP32 benchmark result

The final diagnostic used commit `1f3abff126b7`, a fresh Release build, and
module SHA-256 `fa4a4c3b346c`. It ran on an 8 GB M1 MacBook Air on AC power,
with AC Low Power Mode disabled and no recorded thermal or performance warning.
Accelerate thread-control variables were unset, so the CPU route used its
backend-managed policy.

Three forward and three exactly reversed sweeps produced 12 raw samples per
route and case in each sweep. The table reports the median of the six run-level
medians. The final column is the complete range of run-level Metal/CPU ratios.
Every correctness check reported zero maximum error and zero relative L2 error
for these deterministic inputs.

| Side | Depth | CPU (ms) | Metal (ms) | Metal / CPU | Six-run range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 0.221 | 1.209 | 5.305 | 2.278-7.880 |
| 512 | 4 | 1.195 | 3.586 | 2.672 | 1.728-3.931 |
| 1024 | 1 | 3.217 | 4.009 | 1.251 | 1.131-1.442 |
| 1024 | 4 | 13.034 | 13.334 | 1.018 | 0.906-1.117 |
| 2048 | 1 | 28.591 | 22.084 | 0.753 | 0.693-0.802 |
| 2048 | 4 | 120.512 | 87.233 | 0.712 | 0.678-0.783 |
| 4096 | 1 | 247.373 | 157.400 | 0.633 | 0.601-0.655 |
| 4096 | 4 | 1025.426 | 676.383 | 0.660 | 0.621-0.683 |

Resident Metal lost at side 512 and at side 1024 with depth 1. Side 1024,
depth 4 crossed both sides of parity, so there is no stable crossover there.
Metal won all six runs at sides 2048 and 4096 for both depths. This establishes
that shared-buffer residency can outperform Accelerate for this workload on
this machine. It does not establish a portable threshold.

The process snapshots were not quiet. WindowServer and Parsec remained active,
and one snapshot caught a busy Chrome renderer. Large-case ratios remained
consistent across forward and reverse order, but the results should still be
treated as hardware- and workload-scoped feasibility evidence.

## Verification

- The Metal-enabled Python suite passed 1736 tests and 1731 subtests, with 558
  platform or optional-feature skips and 3 expected failures. The combined
  float16 and Metal subset passed 29 tests and 16 subtests, with 1 skip.
- The same float16 and Metal subset passed 7 tests with 23 expected skips in a
  Metal-disabled build.
- All 247 C++ tests passed independently with `BUILD_METAL=ON` and
  `BUILD_METAL=OFF`. The two backend contract tests cover the CPU provider and
  the build-dependent Metal provider.
- All six `make lint` checks passed. Local clang-format 19 reported only the
  expected version warning against the CI pin of 20.
- An optimized arm64 compiler probe reduced CPU `ConcreteBuffer::data()` to
  one access-state load, one conditional branch, and the cached pointer load;
  the CPU path contains no virtual call or mutex operation.
- Manual concurrency stress covered 500 host-export/submission races and 6000
  submissions with concurrent CPU copies and waits. A 100,000-cycle
  submit/wait run retained bounded resident memory.

## Follow-up experiments

The next useful work is breadth, then dependency scheduling:

1. Extend the current `MatmulPlan` integration to vector roles and batch
   broadcasting. Use native MPS batch when matrix offsets are regular,
   otherwise encode plan contractions on one command buffer.
2. Add a small buffer pool after measuring allocation cost. Pooling removes
   repeated Metal resource creation but does not change dependency semantics.
3. Extend scoped host read/write leases from `sum()` and `fill()` to the
   remaining short CPU operations without changing the external raw-pointer
   contract.
4. Port a small elementwise family and reduction family to demonstrate a
   realistic resident pipeline.
5. Apply one explicit CPU/Metal execution selector consistently across
   operation families. Keep performance measurements diagnostic rather than
   introducing automatic backend selection.

Byte-range tracking is deliberately later. Allocation-wide locking is safe for
all current views, including negative and broadcast strides. Region tracking
only matters after profiling shows independent views are being serialized.

## Development record

The prototype replaced an earlier CPU-in/CPU-out Metal GEMM route. Discussion
first established that Apple unified memory removes a PCIe transfer but does
not make a normal `malloc` allocation an `MTLBuffer`. The next proposal was a
separate `MetalArray`. Review then identified `ConcreteBuffer` as the correct
resource-ownership layer, provided raw host access and asynchronous GPU access
share one allocation-wide synchronization state.

The resulting decisions are:

1. Keep `SimpleArray<T>` as the public typed array and select its owned storage
   with `device="cpu"` or `device="metal"`.
2. Preserve all existing CPU pointer APIs. On Metal storage they wait and create
   a sticky host export, because an escaped pointer has no observable lifetime.
3. Provide recoverable C++ leases and NumPy views whose base object records the
   lifetime that a raw pointer alone cannot express.
4. Define portable binary16 storage and expose the same representation through
   C++, `SimpleArrayPlex`, NumPy, and Python.
5. Submit FP16 or FP32 MPS GEMM asynchronously and keep results in Metal
   storage.
6. Use one serial queue and allocation-wide locks before attempting a general
   dependency scheduler, byte-range tracking, or multiple streams.
7. Restrict the first resident implementation to unified-memory devices rather
   than pretending per-operation managed-memory synchronization is resident.
8. Dispatch owned buffer allocation through `BufferBackend`, keeping platform
   types and conditional compilation outside `ConcreteBuffer`.

The buffer contract lives in `ConcreteBuffer.hpp`; allocation dispatch lives in
`device/BufferBackend.cpp`; and the Apple implementation lives in
`device/metal/metal.mm`. Runtime coverage is in
`tests/test_metal_simplearray.py` and `gtests/test_nopython_buffer.cpp`; the
opt-in benchmark is `profiling/profile_metal_simplearray.py`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
