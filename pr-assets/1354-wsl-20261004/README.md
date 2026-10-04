# PR #1354: WSL2 matmul measurements

This snapshot compares public `SimpleArray.matmul()` with NumPy 2.5.1.
The measured solvcon commit is
`a1494b454017d4b6dd7a83cbe85e15746ac78e98`, the original PR head.

## Workloads

FP32 and FP64 use the same four workload families. Strides below are in
elements. Both libraries receive the same input memory and actual strides.
All outputs are contiguous. Array construction and input wrapping happen
outside timing.

| Family | lhs shape / strides | rhs shape / strides | output shape / strides |
| --- | --- | --- | --- |
| Single | `(S,S)` / `(2S,2)` | `(S,S)` / `(S,1)` | `(S,S)` / `(S,1)` |
| Batch | `(10,S,S)` / `(2S^2,2S,2)` | `(10,S,S)` / `(2S^2,2S,2)` | `(10,S,S)` / `(S^2,S,1)` |
| Broadcast | `(1,S,S)` / `(2S^2,2S,2)` | `(10,S,S)` / `(2S^2,2S,2)` | `(10,S,S)` / `(S^2,S,1)` |
| Large | `(S,S)` / `(S,1)` | `(S,S)` / `(S,1)` | `(S,S)` / `(S,1)` |

Single, Batch, and Broadcast use S = 8, 16, 32, 64, 128, 256, 512, 1024.
Large uses S = 8192, 16384, 20480. The frozen dispatch selects direct BLAS
at 8192 and one-level Winograd at 16384 and 20480. These route labels come
from the frozen dispatch conditions and validated input layouts.

## Environment

- Intel Core i7-13700K, 62 GiB WSL2 memory.
- WSL2 Linux 6.6.87.2-microsoft-standard-WSL2, x86_64.
- Existing SCDV: Python 3.14.7, NumPy 2.5.1, GCC 16.0.1.
- Both libraries load the same OpenBLAS 0.3.26 shared library, LP64,
  pthreads, Haswell runtime kernel.
- Release (`-O3 -DNDEBUG`), Qt off, profiling off; CBLAS enabled.
- Single/Batch/Broadcast: one BLAS thread, process affinity to WSL CPU 2.
- Large: 24 BLAS threads, process affinity to WSL CPUs 0-23.
- CPU numbers denote virtual processors; no P-core mapping is assumed.

The different thread configurations are reported separately. There is no
combined geometric mean across single-threaded and 24-thread workloads.

## Timing and correctness

Each of the 54 cases runs in a fresh process. Six rounds alternate the
NumPy/solvcon order, so each method runs first three times. Operands remain
resident and are reused across calls within a process. Input values are
deterministic uniform random numbers in [-1,1), with case seeds in raw data.

For the first three families, calibration targets about 20 ms per timing
batch. Each method receives two discarded warmup batches and 15 timed
batches per round, for 90 per-call estimates. Large cases use one discarded
warmup and one timed call per method per round, for six samples. The initial
full-output correctness calls are also discarded before warmup.

Every timed call includes the public API, output allocation and release,
and internal packing or Winograd scratch allocation. No operand is
prepacked outside timing. Every sample is retained without trimming.

The plotted speedup is median NumPy time divided by median solvcon time.
Shading is the p10-p90 interval of the six paired round speedups, not a
confidence interval. Values below 1 mean solvcon was slower.

Before timing, all output elements are compared with NumPy using relative
Frobenius error, with predeclared limits of 5e-5 for FP32 and 5e-13 for FP64.
Shape, dtype, finite values, output strides, input strides, and zero-copy
input wrapping are checked. Maximum absolute errors are retained as well.

## Files

- `pr1354-*.png`: the four standalone figures.
- `summary.csv`: timings and speedup for each of the 54 points.
- `summary.json`: summary with complete shape/stride metadata.
- `raw.jsonl`: all timing batches, ordering, validation, and runtime data.
- `manifest.json`: fixed workload and sampling definitions.
- `provenance.json`: measured extension and runner hashes.
- `SHA256SUMS`: checksums of the data, figures, and scripts.

## Reproduce

Use an isolated checkout of the measured SHA. Place `profile_pr1354.py`
and `plot_pr1354.py` under
`profiling/results/pr1354-wsl-20261004/` in that checkout. Activate an SCDV
with NumPy exactly 2.5.1 and build through the repository Makefile:

```bash
CC=gcc-16 CXX=g++-16 make CMAKE_BUILD_TYPE=Release BUILD_QT=OFF \
    SOLVCON_PROFILE=OFF USE_GOOGLETEST=OFF NPROC=12
python3 profiling/results/pr1354-wsl-20261004/profile_pr1354.py
python3 profiling/results/pr1354-wsl-20261004/plot_pr1354.py
```

The runner verifies the SHA and NumPy version. It skips complete case
files and refuses to overwrite incomplete cases. Run in a fresh output
directory for independent measurements. The affinity settings assume
this 24-vCPU WSL2 configuration and must be documented if changed.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
