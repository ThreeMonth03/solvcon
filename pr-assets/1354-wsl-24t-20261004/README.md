# PR #1354: WSL2 matmul with 24 available BLAS threads

This snapshot replaces the earlier mixed-thread measurements. It compares
public `SimpleArray.matmul()` with NumPy 2.5.1 at the original PR head,
`a1494b454017d4b6dd7a83cbe85e15746ac78e98`.

## Workloads

All four families use FP32 and FP64. Strides are in elements. Both libraries
receive the same input memory and actual strides; outputs are contiguous.

| Family | lhs shape / strides | rhs shape / strides | output shape / strides |
| --- | --- | --- | --- |
| Single | `(S,S)` / `(2S,2)` | `(S,S)` / `(S,1)` | `(S,S)` / `(S,1)` |
| Batch | `(10,S,S)` / `(2S^2,2S,2)` | `(10,S,S)` / `(2S^2,2S,2)` | `(10,S,S)` / `(S^2,S,1)` |
| Broadcast | `(1,S,S)` / `(2S^2,2S,2)` | `(10,S,S)` / `(2S^2,2S,2)` | `(10,S,S)` / `(S^2,S,1)` |
| Large | `(S,S)` / `(S,1)` | `(S,S)` / `(S,1)` | `(S,S)` / `(S,1)` |

Single, Batch, and Broadcast use S = 8, 16, 32, 64, 128, 256, 512, 1024.
Large uses S = 8192, 16384, 20480. Frozen dispatch conditions select direct
BLAS at 8192 and one-level Winograd at 16384 and 20480.

## Environment

- Intel Core i7-13700K, 62 GiB WSL2 memory, 24 available logical CPUs.
- WSL2 Linux 6.6.87.2-microsoft-standard-WSL2, x86_64.
- Existing SCDV: Python 3.14.7, NumPy 2.5.1, GCC 16.0.1.
- Both libraries load the same OpenBLAS 0.3.26 shared library, LP64,
  pthreads, Haswell runtime kernel.
- Release (`-O3 -DNDEBUG`), Qt off, profiling off; CBLAS enabled.
- Every family: 24 BLAS threads and process affinity to WSL CPUs 0-23.
- The runtime thread limit is checked, and process CPU time is recorded.
  OpenBLAS can select fewer active threads for small operations. CPU time
  includes worker spinning, so it is not a measure of useful parallel work.

## Sampling

There are 54 cases and three independent process runs per case. Case order
is shuffled with a fixed seed in each replicate. At the user's request,
the suite paused between processes after the first large case to finish
all non-large cases, then resumed the remaining large cases. No timed
call was paused. The final manifest records actual process order and
timestamps alongside the planned order. Each process performs a
full-output correctness check, calibration where applicable, two discarded
warmup rounds, and ten measured rounds. Inputs remain resident and reused
within a process; all three processes use the same case-specific input seed.

`williams_schedule.py` contains the unmodified `_williams_rows()` function
from the Qt benchmark collector at
`d2d18841774f9e7dc6fa9173a9aadd79ace3e99c`,
`solvcon/benchmark/collector.py`. The two-candidate rows are AB and BA.
Warmups use the cyclic rows immediately preceding the first measured row.
Each method runs first five times per process. The initial candidate order
is reversed according to the replicate and dtype, independently of results.

Single, Batch, and Broadcast calibrate a common repetition count for both
methods, targeting about 50 ms for the faster method. The count is limited
to 200,000 calls. Large uses one call per round. Each timed block is divided
by its repetition count. Calibration and correctness calls are discarded.

Every plotted point has 30 per-call round averages per method across the
three processes. First compute each process's median time for each method.
The plotted time is the geometric mean of those three process medians.
Speedup is the ratio of the plotted times, equivalently the geometric mean
of the three within-process speedups. This gives each independent process
equal weight and preserves the pairing between methods.

Shading is the minimum-to-maximum range of the three process medians or
process speedups, respectively. It is not a confidence interval. All timed
blocks are retained. Pooled medians and round-level p10-p90 values remain
in the data but do not determine the plotted curves.

This aggregation was revised after inspecting the non-large results:
substantial process-to-process shifts caused pooled medians to reverse the
direction of all three within-process comparisons in one case. The same
process-level aggregation is applied to every case, without sample removal.

## Timing scope and correctness

Every call includes the public API, output allocation/release, and internal
operand packing or Winograd scratch. Input generation and zero-copy
SimpleArray wrapping are excluded. No operand is prepacked outside timing.

Before timing, all output elements are checked against NumPy, with relative
Frobenius error limits of 5e-5 for FP32 and 5e-13 for FP64. Shape, dtype,
finite values, output strides, input strides, and zero-copy wrapping are
checked as well. Maximum absolute errors are retained in raw records.

## Files

- `pr1354-*.png`: four figures, each with FP32 and FP64 time/speedup panels.
- `summary.csv`: all 54 points with absolute timings and speedup.
- `summary.json`: complete metadata and process-level results.
- `raw.jsonl`: every measured round and its runtime metadata.
- `manifest.json`: sampling definitions and exact execution order.
- `provenance.json`: measured extension and script hashes.
- `audit.txt`: completed integrity checks.
- `SHA256SUMS`: artifact checksums.

## Reproduce

To regenerate the figures from existing measurements, put `raw.jsonl` and
`plot_pr1354.py` in one directory with NumPy and Matplotlib available:

```bash
python3 plot_pr1354.py --no-checksums
```

To repeat the measurements, use an isolated checkout at the measured SHA.
Place `profile_pr1354.py`,
`plot_pr1354.py`, and `williams_schedule.py` under
`profiling/results/pr1354-wsl-24t-20261004/`. Activate an SCDV with NumPy
exactly 2.5.1 and build through the repository Makefile:

```bash
CC=gcc-16 CXX=g++-16 make CMAKE_BUILD_TYPE=Release BUILD_QT=OFF \
    SOLVCON_PROFILE=OFF USE_GOOGLETEST=OFF NPROC=12
python3 profiling/results/pr1354-wsl-24t-20261004/profile_pr1354.py
python3 profiling/results/pr1354-wsl-24t-20261004/plot_pr1354.py --no-checksums
```

The runner verifies the SHA and NumPy version. It skips complete case files
and refuses to overwrite incomplete ones. Start in a fresh output directory
for new measurements. The CPU affinity assumes this 24-vCPU WSL2 setup;
document any change on another host. No GUI is needed to run the scheduler.
The published provenance, integrity audit, and checksums describe this
specific run; repeating timings on another build requires new provenance.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
