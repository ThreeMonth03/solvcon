# Whole-matrix padding control

## Question

Should Strassen dispatch pad non-divisible matrices before a future decision
tree selects the kernel?

The control rounds `M`, `K`, and `N` up to `2^depth`, materializes zero-filled
compact operands, executes Strassen with BLAS leaves, and crops the output.
The timed call includes output allocation, padding, copies, Strassen workspace
reuse, and cropping.  Input construction and correctness checks remain outside
the timed region.

## Environment and protocol

- Linux 6.6 under WSL2 on an Intel Core i7-13700K
- Python 3.12.7, NumPy 2.5.1, and the prime OpenBLAS build
- `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`
- process pinned to CPU 2
- one warmup before three samples in each of two rotated-order rounds
- table values are medians of the resulting six full-call samples

Reproduce depth 1 with:

```console
$ OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=$PWD taskset -c 2 \
  python3 profiling/profile_strassen_padding.py \
  --sides 3072 4096 --cases divisible all --depths 1 \
  --warmups 1 --samples 3 --rounds 2
```

Replace `--depths 1` with `--depths 2` for the depth-2 control.  The script
prints the raw samples, relative Frobenius error, padded dimensions, temporary
storage, minimum padding/copy traffic, and workspace size as JSON lines.

## Results

`Strassen / BLAS` and `Strassen / NumPy` below are full-call time ratios.  A
ratio below one favors Strassen.

| dtype | depth | `M x K x N` | NumPy (ms) | direct BLAS (ms) | Strassen (ms) | Strassen / BLAS | Strassen / NumPy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| float32 | 1 | `3072 x 3072 x 3072` | 391.66 | 427.39 | 446.99 | 1.046 | 1.141 |
| float32 | 1 | `3073 x 3073 x 3073` | 400.49 | 426.46 | 487.45 | 1.143 | 1.217 |
| float32 | 1 | `4096 x 4096 x 4096` | 913.54 | 979.61 | 1005.92 | 1.027 | 1.101 |
| float32 | 1 | `4097 x 4097 x 4097` | 963.02 | 1035.84 | 1126.70 | 1.088 | 1.170 |
| float64 | 1 | `3072 x 3072 x 3072` | 893.19 | 922.36 | 961.18 | 1.042 | 1.076 |
| float64 | 1 | `3073 x 3073 x 3073` | 897.18 | 903.39 | 1080.97 | 1.197 | 1.205 |
| float64 | 1 | `4096 x 4096 x 4096` | 2163.56 | 2188.78 | 2198.63 | 1.004 | 1.016 |
| float64 | 1 | `4097 x 4097 x 4097` | 2181.39 | 2275.76 | 2555.36 | 1.123 | 1.171 |
| float32 | 2 | `4096 x 4096 x 4096` | 926.41 | 982.99 | 967.50 | 0.984 | 1.044 |
| float32 | 2 | `4097 x 4097 x 4097` | 924.13 | 991.19 | 1110.86 | 1.121 | 1.202 |
| float64 | 2 | `4096 x 4096 x 4096` | 2067.07 | 2109.19 | 2149.47 | 1.019 | 1.040 |
| float64 | 2 | `4097 x 4097 x 4097` | 2093.72 | 2088.64 | 2316.87 | 1.109 | 1.107 |

All tested relative Frobenius errors were at most `1.24e-6` for float32 and
`2.26e-15` for float64.  Exhaustive small controls also passed all 144
dtype, depth, and `(M,K,N) mod 2^depth` combinations.

For float64 depth 1, rounding `4097^3` to `4098^3` adds only 0.073% to the
nominal contraction volume.  The full route nevertheless creates 384.38 MiB
of temporary matrices and incurs at least 1024.63 MiB of zeroing and copy
traffic, in addition to its 96.09 MiB reusable Strassen workspace.  The exact
`4096^3` control is within 0.4% of direct BLAS, while the padded `4097^3`
control is 12.3% slower.

No WSL route in this table beats both NumPy and direct OpenBLAS, so the WSL
selection table should remain empty.  This does not replace the existing Apple
M1 calibration; it isolates the cost of adding whole-matrix padding.

## Depth-2 residue control

A quick three-sample control compares the best, middle, and worst padding
residues near 4096.  These values are exploratory rather than calibration
inputs.

| dtype | Input | Padded | Strassen / BLAS | Strassen / NumPy |
| --- | ---: | ---: | ---: | ---: |
| float32 | 4095 | 4096 | 1.093 | 1.150 |
| float32 | 4097 | 4100 | 1.116 | 1.207 |
| float32 | 4098 | 4100 | 1.187 | 1.180 |
| float64 | 4095 | 4096 | 1.095 | 1.103 |
| float64 | 4097 | 4100 | 1.178 | 1.181 |
| float64 | 4098 | 4100 | 0.987 | 1.067 |

The best residue, `4095 -> 4096`, is faster than the worst-residue padded
route but still loses to both controls.  Float64 `4098 -> 4100` is slightly
faster than direct BLAS but remains slower than NumPy.  Padding profitability
is therefore not monotonic in the number of padded elements; backend blocking,
copy traffic, cache behavior, dtype, and recursion depth all affect the
boundary.

## Decision

Whole-matrix padding should not be part of the initial Strassen dispatch table.
The initial `StrassenGemm1` and `StrassenGemm2` candidates require divisibility
by `2^depth`, and non-divisible calls retain the existing matmul route.

A future generated selector may add a separate padded-Strassen candidate.  Its
features should include padded extents, copy bytes, workspace bytes, dtype, and
backend, rather than treating padding as part of the direct Strassen route.
Fringe handling, selective packing, and reusable buffers remain alternative
candidates.  Each candidate must beat the complete-call baseline before it is
enabled.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
