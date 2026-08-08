/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/matmul.hpp>
#include <solvcon/buffer/SimpleArray.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

TEST(MatmulStrassenDispatch, calibrated_routes)
{
    namespace mm = solvcon;
    using mm::detail::APPLE_ARM64_STRASSEN_TUNING;
    using mm::detail::MatmulKernel;
    using mm::detail::select_strassen_kernel;

    EXPECT_EQ(
        select_strassen_kernel<float>(APPLE_ARM64_STRASSEN_TUNING, 5632, 5632, 5632),
        MatmulKernel::StrassenGemm1);
    EXPECT_EQ(
        select_strassen_kernel<float>(APPLE_ARM64_STRASSEN_TUNING, 3072, 3072, 24576),
        MatmulKernel::StrassenGemm1);
    EXPECT_EQ(
        select_strassen_kernel<double>(APPLE_ARM64_STRASSEN_TUNING, 3072, 3072, 3072),
        MatmulKernel::StrassenGemm1);
    EXPECT_EQ(
        select_strassen_kernel<double>(APPLE_ARM64_STRASSEN_TUNING, 4096, 4096, 4096),
        MatmulKernel::StrassenGemm2);
    EXPECT_EQ(
        select_strassen_kernel<double>(APPLE_ARM64_STRASSEN_TUNING, 6144, 6144, 6144),
        MatmulKernel::StrassenGemm2);
}

TEST(MatmulStrassenDispatch, uncalibrated_routes_remain_generic)
{
    namespace mm = solvcon;
    using mm::detail::APPLE_ARM64_STRASSEN_TUNING;
    using mm::detail::MatmulKernel;
    using mm::detail::select_strassen_kernel;

    EXPECT_EQ(
        select_strassen_kernel<float>(APPLE_ARM64_STRASSEN_TUNING, 4096, 4096, 4096),
        MatmulKernel::Generic);
    EXPECT_EQ(
        select_strassen_kernel<float>(APPLE_ARM64_STRASSEN_TUNING, 6144, 6144, 6144),
        MatmulKernel::Generic);
    EXPECT_EQ(
        select_strassen_kernel<double>(APPLE_ARM64_STRASSEN_TUNING, 6144, 6144, 768),
        MatmulKernel::Generic);
    EXPECT_EQ(
        select_strassen_kernel<mm::Complex<double>>(
            APPLE_ARM64_STRASSEN_TUNING, 4096, 4096, 4096),
        MatmulKernel::Generic);
}

namespace
{

template <typename T>
void check_strassen_result(size_t depth)
{
    namespace mm = solvcon;
    constexpr ssize_t rows = 8;
    constexpr ssize_t columns = 12;
    constexpr ssize_t inner_size = 16;
    mm::SimpleArray<T> lhs(mm::small_vector<ssize_t>{rows, inner_size});
    mm::SimpleArray<T> rhs(mm::small_vector<ssize_t>{inner_size, columns});
    mm::SimpleArray<T> output(mm::small_vector<ssize_t>{rows, columns});
    mm::SimpleArray<T> expected(mm::small_vector<ssize_t>{rows, columns}, T{});

    for (ssize_t row = 0; row < rows; ++row)
    {
        for (ssize_t inner = 0; inner < inner_size; ++inner)
        {
            lhs(row, inner) = static_cast<T>((row * 7 - inner * 3) % 19) / T{8};
        }
    }
    for (ssize_t inner = 0; inner < inner_size; ++inner)
    {
        for (ssize_t column = 0; column < columns; ++column)
        {
            rhs(inner, column) = static_cast<T>((inner * 5 + column * 2) % 23) / T{16};
        }
    }
    for (ssize_t row = 0; row < rows; ++row)
    {
        for (ssize_t column = 0; column < columns; ++column)
        {
            for (ssize_t inner = 0; inner < inner_size; ++inner)
            {
                expected(row, column) += lhs(row, inner) * rhs(inner, column);
            }
        }
    }

    auto const leaf = [](
                          ssize_t leaf_rows,
                          ssize_t leaf_columns,
                          ssize_t leaf_inner_size,
                          mm::detail::StrassenConstMatrixView<T> leaf_lhs,
                          mm::detail::StrassenConstMatrixView<T> leaf_rhs,
                          mm::detail::StrassenMatrixView<T> leaf_output)
    {
        for (ssize_t row = 0; row < leaf_rows; ++row)
        {
            for (ssize_t column = 0; column < leaf_columns; ++column)
            {
                T total{};
                for (ssize_t inner = 0; inner < leaf_inner_size; ++inner)
                {
                    total += leaf_lhs.m_data[row * leaf_lhs.m_row_stride + inner] *
                             leaf_rhs.m_data[inner * leaf_rhs.m_row_stride + column];
                }
                leaf_output.m_data[row * leaf_output.m_row_stride + column] = total;
            }
        }
    };
    mm::detail::StrassenWorkspace<T> workspace;
    mm::detail::gemm_strassen_with_leaf(
        rows,
        columns,
        inner_size,
        lhs.logical_data(),
        rhs.logical_data(),
        output.logical_data(),
        depth,
        workspace,
        leaf);

    T const epsilon = std::numeric_limits<T>::epsilon();
    for (ssize_t row = 0; row < rows; ++row)
    {
        for (ssize_t column = 0; column < columns; ++column)
        {
            T const reference = expected(row, column);
            T const tolerance = T{128} * epsilon * std::max(T{1}, std::abs(reference));
            EXPECT_NEAR(output(row, column), reference, tolerance);
        }
    }
}

} /* end namespace */

TEST(MatmulStrassenKernel, one_level_matches_reference)
{
    check_strassen_result<float>(1);
    check_strassen_result<double>(1);
}

TEST(MatmulStrassenKernel, two_levels_match_reference)
{
    check_strassen_result<float>(2);
    check_strassen_result<double>(2);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
