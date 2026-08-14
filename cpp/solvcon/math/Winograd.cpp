/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/math/Winograd.hpp>
#include <solvcon/math/Winograd_detail.hpp>

namespace solvcon
{

namespace detail
{

namespace
{

template <typename T>
void gemm_winograd_impl(BlasGemmOperation<T> const & operation)
{
    auto const execute_product = [](BlasGemmOperation<T> const & product)
    { gemm_blas(product); };
    winograd::multiply(operation, execute_product);
}

} /* end namespace */

void gemm_winograd(BlasGemmOperation<float> const & operation)
{
    gemm_winograd_impl(operation);
}

void gemm_winograd(BlasGemmOperation<double> const & operation)
{
    gemm_winograd_impl(operation);
}

void gemm_winograd(BlasGemmOperation<Complex<float>> const & operation)
{
    gemm_winograd_impl(operation);
}

void gemm_winograd(BlasGemmOperation<Complex<double>> const & operation)
{
    gemm_winograd_impl(operation);
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
