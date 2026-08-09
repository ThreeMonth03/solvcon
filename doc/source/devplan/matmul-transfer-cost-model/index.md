# Transfer matmul route costs across machines

A dispatch policy learned on one computer does not describe another CPU or
BLAS backend. In particular, CPU metadata does not expose the internal
kernel boundaries used by Accelerate or OpenBLAS. A portable policy therefore
needs a small measurement of the target computer without fitting a complete
target-local model.

This prototype represents a computer with fixed landmark measurements. A
portable model combines that signature with the existing matmul facts and
predicts each route relative to the current selector:

```text
frozen shape and route manifest
              |
      10-20 landmark calls
              |
 normalized device signature ---- shape and route facts
              |                         |
              +---- low-rank model -----+
                            |
                  log(route / current)
                            |
         confident improvement? route : current
```

The target computer does not train a decision tree. It measures the frozen
landmarks and performs one model evaluation. Model training happens offline
from a bank of machine profiles.

## Data contract

Every profile in one transfer experiment must use the same:

- calibration shapes and sibling groups;
- forced-route schema and work limit;
- matmul, binding, collection, and measurement source hashes;
- dtype, layout, and thread policy.

Profiles also record CPU, platform, BLAS, and thread identities. Two repeated
runs on one computer do not count as two transfer devices.

The landmark manifest is selected from shape facts without reading timings.
It uses farthest-first coverage in the normalized shape feature space. Each
landmark contributes current throughput, route-to-current log ratios, and
route availability to the device signature.

## Portable model

The first model is intentionally small. It standardizes analytic shape
features, projects device signatures to a low-rank embedding, and applies
ridge regression to shape, device, and bilinear interaction terms. It predicts
`log(T_route / T_current)` separately for each route.

Safety margins come from leave-one-source-device-out prediction errors. A
route is selected only when its predicted cost plus the margin is at least 3%
below the current route. Missing routes, unsupported records, and uncertain
predictions keep the existing selector.

## Validation

The complete target device is held out from model fitting. Landmark records
and every record in their sibling groups are excluded from target evaluation.
The final experiment needs at least three distinct hardware profiles so that
two source devices predict one unseen device. Four devices are preferred for
nested uncertainty validation.

For each unseen device and dtype, report:

- landmark count and calibration wall time;
- current-over-policy speedup;
- policy-to-oracle geometric-mean and p95 regret;
- worst slowdown relative to current;
- captured oracle gap and fallback count;
- model fit time, model size, and decision overhead.

The few-shot policy passes only if it reaches at least 1.03x speedup, at most
1.03x geometric-mean regret, at most 1.05x p95 regret, at most 1.10x worst
slowdown, and captures at least 70% of the available oracle gap. Zero-shot is
a separate conservative baseline and is not a portability claim.

## Current evidence

Synthetic device-held-out tests verify the data contract, timing-independent
landmarks, signature-dependent decisions, sibling-group exclusion, and
fallback behavior. No existing Apple dataset contains the same forced routes
as the current collector, so the prototype does not yet claim cross-machine
speedup.

Replaying the 16-landmark manifest against an existing 512-case WSL bank gives
1.419 seconds of recorded float32 case wall time and 1.660 seconds for
float64. The combined forced-route timing batches account for 1.755 seconds.
Manifest selection itself takes a median 11.3 milliseconds over 21 runs. These
figures estimate calibration cost from existing records; they are not fresh
Apple measurements.

A preliminary replay of older Apple and Ubuntu data found that direct
cross-machine route reuse was asymmetric and exceeded the 10% slowdown limit.
Those files use an earlier route schema and are evidence for the domain-shift
problem only.

## References

- [HELP](https://papers.neurips.cc/paper_files/paper/2021/hash/e3251075554389fe91d17a794861d47b-Paper.pdf)
  adapts to unseen hardware from ten normalized landmark latencies.
- [Halide autoscheduler](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf)
  combines analytic program features with learned cost coefficients.
- [TenSet](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf)
  studies pretrained tensor-program cost models and target measurements.
- [FFTW wisdom](https://www.fftw.org/doc/Caveats-in-Using-Wisdom.html)
  records why measured plans cannot be assumed portable across systems.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
