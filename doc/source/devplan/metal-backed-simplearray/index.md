# Metal-backed SimpleArray prototype

## Goal

This prototype tests whether `SimpleArray` can keep its existing public type
while its owned `ConcreteBuffer` is allocated as a Metal resource. The goal is
not automatic CPU/GPU scheduling. It is to prove the storage and synchronization
foundation needed by a future scheduler.

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

## Public experiment

CPU storage remains the default:

```python
cpu = sc.SimpleArrayFloat32([1024, 1024])
gpu = sc.SimpleArrayFloat32([1024, 1024], device="metal")

gpu = sc.SimpleArrayFloat32(array=numpy_array, device="metal")
result = gpu.matmul_metal(weight)
result.wait()
host_copy = result.cpu()
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
```

`wait()` does not export a host pointer, so the array remains GPU-eligible.
`data()`, iterators, references, spans, mdspans, `.ndarray`, and the Python
buffer protocol wait for pending work and mark the shared `ConcreteBuffer` as
host-exported. All reshape and transpose views share this state. Calling
`to("metal")` creates a new GPU-eligible snapshot.

Internal copies use a scoped host-access guard. The guard holds the same
allocation lock from dependency wait through the final byte copy. A concurrent
GPU submission therefore cannot start in the middle of `clone()` or
`to("cpu")`.

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
- Native FP32, two-dimensional, positive row-major matrix multiplication.
- Asynchronous GPU submission and resident matmul chaining on one queue.
- Allocation-wide dependency tracking shared by all views.
- CPU and non-Metal builds retain their existing default behavior.

The Python experiment is exposed through the typed classes such as
`SimpleArrayFloat32`. The type-erased `solvcon.SimpleArray` (`SimpleArrayPlex`)
is not device-aware in this prototype.

The prototype does not yet include:

- Automatic CPU/GPU placement or a size threshold.
- Batched matmul, broadcasting, vectors, or arbitrary strided GPU inputs.
- GPU elementwise operations or reductions.
- Native complex computation or approximate FP64 routes.
- Buffer pooling, suballocation, byte-range locking, or multiple queues.
- Recoverable host-view leases. A raw host export is sticky in this version.
- Managed-memory discrete GPUs and their CPU/GPU dirty-state transitions.

Existing CPU implementations are conservative host boundaries. For example,
calling `fill()` on a Metal-backed array uses an existing iterator and therefore
marks the buffer host-exported. Metadata-only views and `clone()` preserve Metal
storage. Legacy operations that materialize a new layout may still return CPU
storage. These restrictions avoid hidden races while the operation dispatcher
and scoped host leases are still future work.

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
performance numbers are published. A dispatch threshold cannot be inferred
from one machine or one square-matrix sweep.

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
  --run --sides 512 1024 2048 4096 --depths 1 4 \
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

## Follow-up experiments

The next useful work is breadth, then scheduling:

1. Extend the current `MatmulPlan` integration to vector roles and batch
   broadcasting. Use native MPS batch when matrix offsets are regular,
   otherwise encode plan contractions on one command buffer.
2. Add a small buffer pool after measuring allocation cost. Pooling removes
   repeated Metal resource creation but does not change dependency semantics.
3. Add scoped host read/write leases so short CPU operations can use a Metal
   allocation without permanently disabling GPU execution.
4. Port a small elementwise family and reduction family to demonstrate a
   realistic resident pipeline.
5. Build a cost model from resident state, operation work, layout, queue load,
   and synchronization cost. Do not select a backend from matrix size alone.

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
3. Submit FP32 MPS GEMM asynchronously and keep results in Metal storage.
4. Use one serial queue and allocation-wide locks before attempting a general
   CPU/GPU scheduler, byte-range tracking, or multiple streams.
5. Restrict the first resident implementation to unified-memory devices rather
   than pretending per-operation managed-memory synchronization is resident.

The implementation lives in `ConcreteBuffer.hpp`, `SimpleArray.hpp`, and
`device/metal/metal.mm`. Runtime coverage is in
`tests/test_metal_simplearray.py`; the opt-in benchmark is
`profiling/profile_metal_simplearray.py`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
