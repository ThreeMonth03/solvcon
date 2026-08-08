<!-- Published as ThreeMonth03/solvcon#31 issuecomment-5226996668. Keep the file synchronized with that comment. -->

## Supplement: earlier Apple M1 Strassen evidence

The initial revisions of [draft PR #30](https://github.com/ThreeMonth03/solvcon/pull/30) exposed `matmul_strassen()` directly. PR #30 has since been rewritten on the explicit selection and executor architecture. The measurements below come from the earlier revisions and remain useful for identifying candidate routes and negative controls, but the values have not been rerun through the clean dispatch path. A clean Apple rerun is required before finalizing the route table.

Ratios below are Strassen median time divided by NumPy median time, so a value below 1 means Strassen is faster. Each result is the median of 75 calls from five rounds. Method order rotates per call, every round performs two warmups, inputs use a fixed seed, garbage collection is disabled only during timing, and numerical correctness is checked before measurement.

| dtype | `M x K x N` | depth 1 / NumPy | depth 2 / NumPy | Candidate result |
| --- | --- | ---: | ---: | --- |
| float32 | `4096 x 4096 x 4096` | 1.021 | 1.308 | direct backend |
| float32 | `8192 x 1024 x 8192` | 1.283 | 1.863 | direct backend, low `K` |
| float32 | `5632 x 5632 x 5632` | 0.958 | 1.063 | depth 1 |
| float32 | `6144 x 6144 x 6144` | 1.003 | 1.025 | direct backend |
| float32 | `3072 x 24576 x 3072` | 0.895 | 0.963 | depth 1 |
| float64 | `3072 x 3072 x 3072` | 0.960 | 1.012 | depth 1 |
| float64 | `6144 x 768 x 6144` | 1.135 | 1.523 | direct backend, low `K` |
| float64 | `4096 x 4096 x 4096` | 0.933 | 0.915 | depth 2 |
| float64 | `6144 x 6144 x 6144` | 0.927 | 0.859 | depth 2 |

Selected absolute medians show the scale of the candidate wins:

| dtype and shape | NumPy | direct Accelerate | selected Strassen | Speedup over NumPy | Speedup over Accelerate |
| --- | ---: | ---: | ---: | ---: | ---: |
| float32 `3072 x 24576 x 3072`, depth 1 | 775.2 ms | 773.7 ms | 693.5 ms | 1.118x | 1.116x |
| float32 `5632 x 5632 x 5632`, depth 1 | 522.7 ms | 523.4 ms | 500.5 ms | 1.044x | 1.046x |
| float64 `4096 x 4096 x 4096`, depth 2 | 851.2 ms | 851.6 ms | 778.5 ms | 1.093x | 1.094x |
| float64 `6144 x 6144 x 6144`, depth 2 | 3132.8 ms | 3112.6 ms | 2692.0 ms | 1.164x | 1.156x |

The non-monotonic float32 result and the low-`K` controls rule out a single dimension threshold. They support an Apple-specific calibrated table whose default is the existing backend route.

The run used a MacBook Air with Apple M1 and 8 GB memory, macOS 26.5.1 arm64, Python 3.14.6, NumPy 2.5.1 with Accelerate, a native release build, and one thread for the recorded numerical-library variables. The scaled Frobenius residual bound was 64 machine epsilons. All tested results remained finite and within that bound.

The current dispatch prototype keeps `MatmulPlan` unchanged, selects explicit depth-1 or depth-2 kernel values, and stores one reusable workspace per value type and thread. Its platform-independent recursive correctness tests and current Linux regression suite pass, but these results do not substitute for the required clean Apple timing rerun.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
