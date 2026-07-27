#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/base.hpp>
#include <solvcon/math/Complex.hpp>

namespace solvcon
{

float dot_blas(ssize_t size, float const * lhs, float const * rhs);
float dot_blas(ssize_t size,
               float const * lhs,
               ssize_t lhs_increment,
               float const * rhs,
               ssize_t rhs_increment);
double dot_blas(ssize_t size, double const * lhs, double const * rhs);
double dot_blas(ssize_t size,
                double const * lhs,
                ssize_t lhs_increment,
                double const * rhs,
                ssize_t rhs_increment);
Complex<float> dot_blas(ssize_t size,
                        Complex<float> const * lhs,
                        Complex<float> const * rhs);
Complex<float> dot_blas(ssize_t size,
                        Complex<float> const * lhs,
                        ssize_t lhs_increment,
                        Complex<float> const * rhs,
                        ssize_t rhs_increment);
Complex<double> dot_blas(ssize_t size,
                         Complex<double> const * lhs,
                         Complex<double> const * rhs);
Complex<double> dot_blas(ssize_t size,
                         Complex<double> const * lhs,
                         ssize_t lhs_increment,
                         Complex<double> const * rhs,
                         ssize_t rhs_increment);
void gemv_blas(ssize_t m,
               ssize_t n,
               float const * matrix,
               float const * vector,
               float * result,
               bool transpose_matrix);
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
               size_t batch_size);
void gemv_blas(ssize_t m,
               ssize_t n,
               double const * matrix,
               double const * vector,
               double * result,
               bool transpose_matrix);
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
               size_t batch_size);
void gemv_blas(ssize_t m,
               ssize_t n,
               Complex<float> const * matrix,
               Complex<float> const * vector,
               Complex<float> * result,
               bool transpose_matrix);
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
               size_t batch_size);
void gemv_blas(ssize_t m,
               ssize_t n,
               Complex<double> const * matrix,
               Complex<double> const * vector,
               Complex<double> * result,
               bool transpose_matrix);
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
               size_t batch_size);
void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               float const * lhs,
               float const * rhs,
               float * result);
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
               size_t batch_size);
void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               double const * lhs,
               double const * rhs,
               double * result);
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
               size_t batch_size);
void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               Complex<float> const * lhs,
               Complex<float> const * rhs,
               Complex<float> * result);
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
               size_t batch_size);
void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               Complex<double> const * lhs,
               Complex<double> const * rhs,
               Complex<double> * result);
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
               size_t batch_size);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
