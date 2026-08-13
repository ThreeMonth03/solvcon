#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Declare one-level Winograd GEMM.
 *
 * @ingroup group_core
 */

#include <solvcon/math/blas_compat.hpp>

namespace solvcon
{

namespace detail
{

void gemm_winograd(BlasGemmOperation<float> const & operation);
void gemm_winograd(BlasGemmOperation<double> const & operation);
void gemm_winograd(BlasGemmOperation<Complex<float>> const & operation);
void gemm_winograd(BlasGemmOperation<Complex<double>> const & operation);

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
