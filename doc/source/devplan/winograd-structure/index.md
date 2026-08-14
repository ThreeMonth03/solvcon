# Separate the Winograd Interface and Schedule

## Problem

`Winograd.hpp` served two unrelated users. Planned matrix multiplication only
needed one production entry point, but including that entry point also parsed
the full generic seven-product schedule. The production wrapper then unpacked
a `BlasGemmOperation` at the call site and rebuilt the same object inside the
header.

This made the interface look broader than its actual production use and kept
the BLAS bridge in every translation unit that included `matmul.hpp`.

## Code analysis

The affected code is limited to three boundaries:

- `MatmulExecutor::execute_winograd()` owns the planned-matmul call site.
- `detail::gemm_winograd()` is the production typed BLAS entry point.
- `detail::winograd::multiply()` is the generic one-level schedule used by
  focused C++ tests and the BLAS bridge.

The schedule, scratch allocation, seven products, output accumulation, and
dispatch policy do not need to change.

.. pstake:: schematic/winograd_layers.tex

## Design

`Winograd.hpp` becomes a small production header containing four typed
declarations. `Winograd.cpp` owns the typed BLAS bridge. The generic template
implementation moves to `Winograd_detail.hpp`, where its internal execution
object is named `OneLevelExecutor` rather than the vague `Step`.

Planned matrix multiplication constructs the canonical `BlasGemmOperation`
once and passes it directly to `gemm_winograd()`. The operation is a small
non-owning descriptor, so this does not copy matrix data or create another
allocation.

## Implementation

- Split production declarations, BLAS lowering, and the generic schedule into
  separate files.
- Pass `BlasGemmOperation` directly from planned matrix multiplication.
- Keep generic schedule tests on the detail header and production tests on the
  production entry point.
- Add the new source file to the existing math library target.

## Verification

- Build the C++ and Python extension through the project `Makefile`.
- Run the complete C++ test suite, including the Winograd kernel and dispatch
  coverage.
- Run the matrix and BLAS Python tests.
- Run the project lint targets and the changed-line C++ style review.
- Build this development-plan page with Sphinx.

No benchmark is required because the matrix operations, callback sequence,
and dispatch decisions are unchanged.

## Out of scope

This draft does not change recursive Winograd/Strassen behavior, thresholds,
packing, `MatmulPlan`, backend selection, or autotuning. It also does not
propose a public generic Winograd API.

## Delivery status

- Branch: `refactor/winograd-structure`
- Commits: one implementation commit and one development-plan commit
- CI: pending draft pull request
- Preview: `/devplan/winograd-structure/index.html`

## Chat history

1. The BLAS interface cleanup established `BlasGemmOperation` as the canonical
   GEMM descriptor and prompted a second look at the Winograd code structure.
2. The requested review focused on clean-code boundaries, header/source
   placement, and avoiding a broad planned-matmul rewrite.
3. The requested fork draft turns that discussion into a narrow prototype for
   inspection before any upstream proposal.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
