/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/math/blas_compat.hpp>

#if defined(__APPLE__) && defined(__arm64__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#ifndef ACCELERATE_LAPACK_ILP64
#define ACCELERATE_LAPACK_ILP64
#endif
#include <vecLib/cblas_new.h>

#elifdef MM_HAS_MKL
#include <mkl_cblas.h>
#elifdef MM_HAS_CBLAS
#include <cblas.h>
#endif

#include <complex>
#include <format>
#include <limits>
#include <stdexcept>

namespace solvcon
{

#if (defined(__APPLE__) && defined(__arm64__)) || defined(MM_HAS_CBLAS)
#if defined(__APPLE__) && defined(__arm64__)
using blas_int_type = __LAPACK_int;
#else
using blas_int_type = int;
#endif

static blas_int_type to_blas_int(ssize_t value, char const * name)
{
    if (value < 0)
    {
        throw std::out_of_range(
            std::format("solvcon BLAS wrapper: {}={} must be non-negative",
                        name,
                        value));
    }
    if (value <= static_cast<ssize_t>(std::numeric_limits<blas_int_type>::max()))
    {
        return static_cast<blas_int_type>(value);
    }

    throw std::out_of_range(
        std::format("solvcon BLAS wrapper: {}={} exceeds BLAS integer range",
                    name,
                    value));
}

static blas_int_type to_blas_increment(ssize_t value, char const * name)
{
    if (value <= 0)
    {
        throw std::out_of_range(
            std::format("solvcon BLAS wrapper: {}={} must be positive",
                        name,
                        value));
    }
    return to_blas_int(value, name);
}

static CBLAS_TRANSPOSE to_cblas_transpose(bool transpose_matrix)
{
    return transpose_matrix ? CblasTrans : CblasNoTrans;
}

float dot_blas(ssize_t size, float const * lhs, float const * rhs)
{
    return dot_blas(size, lhs, 1, rhs, 1);
}

float dot_blas(ssize_t size,
               float const * lhs,
               ssize_t lhs_increment,
               float const * rhs,
               ssize_t rhs_increment)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    blas_int_type const lhs_inc =
        to_blas_increment(lhs_increment, "lhs_increment");
    blas_int_type const rhs_inc =
        to_blas_increment(rhs_increment, "rhs_increment");
    return cblas_sdot(bsize, lhs, lhs_inc, rhs, rhs_inc);
}

double dot_blas(ssize_t size, double const * lhs, double const * rhs)
{
    return dot_blas(size, lhs, 1, rhs, 1);
}

double dot_blas(ssize_t size,
                double const * lhs,
                ssize_t lhs_increment,
                double const * rhs,
                ssize_t rhs_increment)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    blas_int_type const lhs_inc =
        to_blas_increment(lhs_increment, "lhs_increment");
    blas_int_type const rhs_inc =
        to_blas_increment(rhs_increment, "rhs_increment");
    return cblas_ddot(bsize, lhs, lhs_inc, rhs, rhs_inc);
}

Complex<float> dot_blas(ssize_t size,
                        Complex<float> const * lhs,
                        Complex<float> const * rhs)
{
    return dot_blas(size, lhs, 1, rhs, 1);
}

Complex<float> dot_blas(ssize_t size,
                        Complex<float> const * lhs,
                        ssize_t lhs_increment,
                        Complex<float> const * rhs,
                        ssize_t rhs_increment)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    blas_int_type const lhs_inc =
        to_blas_increment(lhs_increment, "lhs_increment");
    blas_int_type const rhs_inc =
        to_blas_increment(rhs_increment, "rhs_increment");
    std::complex<float> result;
    cblas_cdotu_sub(bsize,
                    as_std_complex_pointer(lhs),
                    lhs_inc,
                    as_std_complex_pointer(rhs),
                    rhs_inc,
                    &result);
    return result;
}

Complex<double> dot_blas(ssize_t size,
                         Complex<double> const * lhs,
                         Complex<double> const * rhs)
{
    return dot_blas(size, lhs, 1, rhs, 1);
}

Complex<double> dot_blas(ssize_t size,
                         Complex<double> const * lhs,
                         ssize_t lhs_increment,
                         Complex<double> const * rhs,
                         ssize_t rhs_increment)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    blas_int_type const lhs_inc =
        to_blas_increment(lhs_increment, "lhs_increment");
    blas_int_type const rhs_inc =
        to_blas_increment(rhs_increment, "rhs_increment");
    std::complex<double> result;
    cblas_zdotu_sub(bsize,
                    as_std_complex_pointer(lhs),
                    lhs_inc,
                    as_std_complex_pointer(rhs),
                    rhs_inc,
                    &result);
    return result;
}

void gemv_blas(ssize_t m,
               ssize_t n,
               float const * matrix,
               float const * vector,
               float * result,
               bool transpose_matrix)
{
    gemv_blas(
        m, n, matrix, vector, result, transpose_matrix, n, 1, 0, 0, 1);
}

void gemv_blas(ssize_t m,
               ssize_t n,
               float const * matrix,
               float const * vector,
               float * result,
               bool transpose_matrix,
               ssize_t leading_dimension,
               ssize_t vector_increment,
               ssize_t matrix_batch_stride,
               ssize_t result_batch_stride,
               size_t batch_size)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bld =
        to_blas_increment(leading_dimension, "leading_dimension");
    blas_int_type const binc =
        to_blas_increment(vector_increment, "vector_increment");
    for (size_t batch = 0; batch < batch_size; ++batch)
    {
        cblas_sgemv(CblasRowMajor,
                    to_cblas_transpose(transpose_matrix),
                    bm,
                    bn,
                    1.0F,
                    matrix + static_cast<ssize_t>(batch) *
                                 matrix_batch_stride,
                    bld,
                    vector,
                    binc,
                    0.0F,
                    result + static_cast<ssize_t>(batch) *
                                 result_batch_stride,
                    1);
    }
}

void gemv_blas(ssize_t m,
               ssize_t n,
               double const * matrix,
               double const * vector,
               double * result,
               bool transpose_matrix)
{
    gemv_blas(
        m, n, matrix, vector, result, transpose_matrix, n, 1, 0, 0, 1);
}

void gemv_blas(ssize_t m,
               ssize_t n,
               double const * matrix,
               double const * vector,
               double * result,
               bool transpose_matrix,
               ssize_t leading_dimension,
               ssize_t vector_increment,
               ssize_t matrix_batch_stride,
               ssize_t result_batch_stride,
               size_t batch_size)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bld =
        to_blas_increment(leading_dimension, "leading_dimension");
    blas_int_type const binc =
        to_blas_increment(vector_increment, "vector_increment");
    for (size_t batch = 0; batch < batch_size; ++batch)
    {
        cblas_dgemv(CblasRowMajor,
                    to_cblas_transpose(transpose_matrix),
                    bm,
                    bn,
                    1.0,
                    matrix + static_cast<ssize_t>(batch) *
                                 matrix_batch_stride,
                    bld,
                    vector,
                    binc,
                    0.0,
                    result + static_cast<ssize_t>(batch) *
                                 result_batch_stride,
                    1);
    }
}

void gemv_blas(ssize_t m,
               ssize_t n,
               Complex<float> const * matrix,
               Complex<float> const * vector,
               Complex<float> * result,
               bool transpose_matrix)
{
    gemv_blas(
        m, n, matrix, vector, result, transpose_matrix, n, 1, 0, 0, 1);
}

void gemv_blas(ssize_t m,
               ssize_t n,
               Complex<float> const * matrix,
               Complex<float> const * vector,
               Complex<float> * result,
               bool transpose_matrix,
               ssize_t leading_dimension,
               ssize_t vector_increment,
               ssize_t matrix_batch_stride,
               ssize_t result_batch_stride,
               size_t batch_size)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bld =
        to_blas_increment(leading_dimension, "leading_dimension");
    blas_int_type const binc =
        to_blas_increment(vector_increment, "vector_increment");
    std::complex<float> const alpha{1.0F, 0.0F};
    std::complex<float> const beta{0.0F, 0.0F};
    for (size_t batch = 0; batch < batch_size; ++batch)
    {
        cblas_cgemv(
            CblasRowMajor,
            to_cblas_transpose(transpose_matrix),
            bm,
            bn,
            &alpha,
            as_std_complex_pointer(
                matrix + static_cast<ssize_t>(batch) *
                             matrix_batch_stride),
            bld,
            as_std_complex_pointer(vector),
            binc,
            &beta,
            as_std_complex_pointer(
                result + static_cast<ssize_t>(batch) *
                             result_batch_stride),
            1);
    }
}

void gemv_blas(ssize_t m,
               ssize_t n,
               Complex<double> const * matrix,
               Complex<double> const * vector,
               Complex<double> * result,
               bool transpose_matrix)
{
    gemv_blas(
        m, n, matrix, vector, result, transpose_matrix, n, 1, 0, 0, 1);
}

void gemv_blas(ssize_t m,
               ssize_t n,
               Complex<double> const * matrix,
               Complex<double> const * vector,
               Complex<double> * result,
               bool transpose_matrix,
               ssize_t leading_dimension,
               ssize_t vector_increment,
               ssize_t matrix_batch_stride,
               ssize_t result_batch_stride,
               size_t batch_size)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bld =
        to_blas_increment(leading_dimension, "leading_dimension");
    blas_int_type const binc =
        to_blas_increment(vector_increment, "vector_increment");
    std::complex<double> const alpha{1.0, 0.0};
    std::complex<double> const beta{0.0, 0.0};
    for (size_t batch = 0; batch < batch_size; ++batch)
    {
        cblas_zgemv(
            CblasRowMajor,
            to_cblas_transpose(transpose_matrix),
            bm,
            bn,
            &alpha,
            as_std_complex_pointer(
                matrix + static_cast<ssize_t>(batch) *
                             matrix_batch_stride),
            bld,
            as_std_complex_pointer(vector),
            binc,
            &beta,
            as_std_complex_pointer(
                result + static_cast<ssize_t>(batch) *
                             result_batch_stride),
            1);
    }
}

void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               float const * lhs,
               float const * rhs,
               float * result)
{
    gemm_blas(
        m, n, k, lhs, rhs, result, false, false, k, n, 0, 0, 0, 1);
}

void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               float const * lhs,
               float const * rhs,
               float * result,
               bool transpose_lhs,
               bool transpose_rhs,
               ssize_t lhs_leading_dimension,
               ssize_t rhs_leading_dimension,
               ssize_t lhs_batch_stride,
               ssize_t rhs_batch_stride,
               ssize_t result_batch_stride,
               size_t batch_size)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bk = to_blas_int(k, "k");
    blas_int_type const blda = to_blas_int(
        lhs_leading_dimension, "lhs_leading_dimension");
    blas_int_type const bldb = to_blas_int(
        rhs_leading_dimension, "rhs_leading_dimension");
    for (size_t batch = 0; batch < batch_size; ++batch)
    {
        cblas_sgemm(CblasRowMajor,
                    to_cblas_transpose(transpose_lhs),
                    to_cblas_transpose(transpose_rhs),
                    bm,
                    bn,
                    bk,
                    1.0F,
                    lhs + static_cast<ssize_t>(batch) *
                              lhs_batch_stride,
                    blda,
                    rhs + static_cast<ssize_t>(batch) *
                              rhs_batch_stride,
                    bldb,
                    0.0F,
                    result + static_cast<ssize_t>(batch) *
                                 result_batch_stride,
                    bn);
    }
}

void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               double const * lhs,
               double const * rhs,
               double * result)
{
    gemm_blas(
        m, n, k, lhs, rhs, result, false, false, k, n, 0, 0, 0, 1);
}

void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               double const * lhs,
               double const * rhs,
               double * result,
               bool transpose_lhs,
               bool transpose_rhs,
               ssize_t lhs_leading_dimension,
               ssize_t rhs_leading_dimension,
               ssize_t lhs_batch_stride,
               ssize_t rhs_batch_stride,
               ssize_t result_batch_stride,
               size_t batch_size)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bk = to_blas_int(k, "k");
    blas_int_type const blda = to_blas_int(
        lhs_leading_dimension, "lhs_leading_dimension");
    blas_int_type const bldb = to_blas_int(
        rhs_leading_dimension, "rhs_leading_dimension");
    for (size_t batch = 0; batch < batch_size; ++batch)
    {
        cblas_dgemm(CblasRowMajor,
                    to_cblas_transpose(transpose_lhs),
                    to_cblas_transpose(transpose_rhs),
                    bm,
                    bn,
                    bk,
                    1.0,
                    lhs + static_cast<ssize_t>(batch) *
                              lhs_batch_stride,
                    blda,
                    rhs + static_cast<ssize_t>(batch) *
                              rhs_batch_stride,
                    bldb,
                    0.0,
                    result + static_cast<ssize_t>(batch) *
                                 result_batch_stride,
                    bn);
    }
}

void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               Complex<float> const * lhs,
               Complex<float> const * rhs,
               Complex<float> * result)
{
    gemm_blas(
        m, n, k, lhs, rhs, result, false, false, k, n, 0, 0, 0, 1);
}

void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               Complex<float> const * lhs,
               Complex<float> const * rhs,
               Complex<float> * result,
               bool transpose_lhs,
               bool transpose_rhs,
               ssize_t lhs_leading_dimension,
               ssize_t rhs_leading_dimension,
               ssize_t lhs_batch_stride,
               ssize_t rhs_batch_stride,
               ssize_t result_batch_stride,
               size_t batch_size)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bk = to_blas_int(k, "k");
    blas_int_type const blda = to_blas_int(
        lhs_leading_dimension, "lhs_leading_dimension");
    blas_int_type const bldb = to_blas_int(
        rhs_leading_dimension, "rhs_leading_dimension");
    std::complex<float> const alpha{1.0F, 0.0F};
    std::complex<float> const beta{0.0F, 0.0F};
    for (size_t batch = 0; batch < batch_size; ++batch)
    {
        cblas_cgemm(
            CblasRowMajor,
            to_cblas_transpose(transpose_lhs),
            to_cblas_transpose(transpose_rhs),
            bm,
            bn,
            bk,
            &alpha,
            as_std_complex_pointer(
                lhs + static_cast<ssize_t>(batch) *
                          lhs_batch_stride),
            blda,
            as_std_complex_pointer(
                rhs + static_cast<ssize_t>(batch) *
                          rhs_batch_stride),
            bldb,
            &beta,
            as_std_complex_pointer(
                result + static_cast<ssize_t>(batch) *
                             result_batch_stride),
            bn);
    }
}

void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               Complex<double> const * lhs,
               Complex<double> const * rhs,
               Complex<double> * result)
{
    gemm_blas(
        m, n, k, lhs, rhs, result, false, false, k, n, 0, 0, 0, 1);
}

void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               Complex<double> const * lhs,
               Complex<double> const * rhs,
               Complex<double> * result,
               bool transpose_lhs,
               bool transpose_rhs,
               ssize_t lhs_leading_dimension,
               ssize_t rhs_leading_dimension,
               ssize_t lhs_batch_stride,
               ssize_t rhs_batch_stride,
               ssize_t result_batch_stride,
               size_t batch_size)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bk = to_blas_int(k, "k");
    blas_int_type const blda = to_blas_int(
        lhs_leading_dimension, "lhs_leading_dimension");
    blas_int_type const bldb = to_blas_int(
        rhs_leading_dimension, "rhs_leading_dimension");
    std::complex<double> const alpha{1.0, 0.0};
    std::complex<double> const beta{0.0, 0.0};
    for (size_t batch = 0; batch < batch_size; ++batch)
    {
        cblas_zgemm(
            CblasRowMajor,
            to_cblas_transpose(transpose_lhs),
            to_cblas_transpose(transpose_rhs),
            bm,
            bn,
            bk,
            &alpha,
            as_std_complex_pointer(
                lhs + static_cast<ssize_t>(batch) *
                          lhs_batch_stride),
            blda,
            as_std_complex_pointer(
                rhs + static_cast<ssize_t>(batch) *
                          rhs_batch_stride),
            bldb,
            &beta,
            as_std_complex_pointer(
                result + static_cast<ssize_t>(batch) *
                             result_batch_stride),
            bn);
    }
}
#else
[[noreturn]] static void throw_blas_unavailable()
{
    throw std::runtime_error(
        "solvcon BLAS wrapper: CBLAS backend is unavailable");
}

float dot_blas(ssize_t, float const *, float const *)
{
    throw_blas_unavailable();
}

double dot_blas(ssize_t, double const *, double const *)
{
    throw_blas_unavailable();
}

Complex<float> dot_blas(ssize_t,
                        Complex<float> const *,
                        Complex<float> const *)
{
    throw_blas_unavailable();
}

Complex<double> dot_blas(ssize_t,
                         Complex<double> const *,
                         Complex<double> const *)
{
    throw_blas_unavailable();
}

void gemv_blas(ssize_t,
               ssize_t,
               float const *,
               float const *,
               float *,
               bool)
{
    throw_blas_unavailable();
}

void gemv_blas(ssize_t,
               ssize_t,
               double const *,
               double const *,
               double *,
               bool)
{
    throw_blas_unavailable();
}

void gemv_blas(ssize_t,
               ssize_t,
               Complex<float> const *,
               Complex<float> const *,
               Complex<float> *,
               bool)
{
    throw_blas_unavailable();
}

void gemv_blas(ssize_t,
               ssize_t,
               Complex<double> const *,
               Complex<double> const *,
               Complex<double> *,
               bool)
{
    throw_blas_unavailable();
}

void gemm_blas(ssize_t, ssize_t, ssize_t, float const *, float const *, float *)
{
    throw_blas_unavailable();
}

void gemm_blas(ssize_t,
               ssize_t,
               ssize_t,
               double const *,
               double const *,
               double *)
{
    throw_blas_unavailable();
}

void gemm_blas(ssize_t,
               ssize_t,
               ssize_t,
               Complex<float> const *,
               Complex<float> const *,
               Complex<float> *)
{
    throw_blas_unavailable();
}

void gemm_blas(ssize_t,
               ssize_t,
               ssize_t,
               Complex<double> const *,
               Complex<double> const *,
               Complex<double> *)
{
    throw_blas_unavailable();
}

float dot_blas(
    ssize_t, float const *, ssize_t, float const *, ssize_t)
{
    throw_blas_unavailable();
}

double dot_blas(
    ssize_t, double const *, ssize_t, double const *, ssize_t)
{
    throw_blas_unavailable();
}

Complex<float> dot_blas(ssize_t,
                        Complex<float> const *,
                        ssize_t,
                        Complex<float> const *,
                        ssize_t)
{
    throw_blas_unavailable();
}

Complex<double> dot_blas(ssize_t,
                         Complex<double> const *,
                         ssize_t,
                         Complex<double> const *,
                         ssize_t)
{
    throw_blas_unavailable();
}

void gemv_blas(ssize_t,
               ssize_t,
               float const *,
               float const *,
               float *,
               bool,
               ssize_t,
               ssize_t,
               ssize_t,
               ssize_t,
               size_t)
{
    throw_blas_unavailable();
}

void gemv_blas(ssize_t,
               ssize_t,
               double const *,
               double const *,
               double *,
               bool,
               ssize_t,
               ssize_t,
               ssize_t,
               ssize_t,
               size_t)
{
    throw_blas_unavailable();
}

void gemv_blas(ssize_t,
               ssize_t,
               Complex<float> const *,
               Complex<float> const *,
               Complex<float> *,
               bool,
               ssize_t,
               ssize_t,
               ssize_t,
               ssize_t,
               size_t)
{
    throw_blas_unavailable();
}

void gemv_blas(ssize_t,
               ssize_t,
               Complex<double> const *,
               Complex<double> const *,
               Complex<double> *,
               bool,
               ssize_t,
               ssize_t,
               ssize_t,
               ssize_t,
               size_t)
{
    throw_blas_unavailable();
}

void gemm_blas(ssize_t,
               ssize_t,
               ssize_t,
               float const *,
               float const *,
               float *,
               bool,
               bool,
               ssize_t,
               ssize_t,
               ssize_t,
               ssize_t,
               ssize_t,
               size_t)
{
    throw_blas_unavailable();
}

void gemm_blas(ssize_t,
               ssize_t,
               ssize_t,
               double const *,
               double const *,
               double *,
               bool,
               bool,
               ssize_t,
               ssize_t,
               ssize_t,
               ssize_t,
               ssize_t,
               size_t)
{
    throw_blas_unavailable();
}

void gemm_blas(ssize_t,
               ssize_t,
               ssize_t,
               Complex<float> const *,
               Complex<float> const *,
               Complex<float> *,
               bool,
               bool,
               ssize_t,
               ssize_t,
               ssize_t,
               ssize_t,
               ssize_t,
               size_t)
{
    throw_blas_unavailable();
}

void gemm_blas(ssize_t,
               ssize_t,
               ssize_t,
               Complex<double> const *,
               Complex<double> const *,
               Complex<double> *,
               bool,
               bool,
               ssize_t,
               ssize_t,
               ssize_t,
               ssize_t,
               ssize_t,
               size_t)
{
    throw_blas_unavailable();
}
#endif

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
