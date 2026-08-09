/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/matmul.hpp>
#include <solvcon/math/Strassen.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

namespace
{

namespace detail = solvcon::detail;
namespace strassen = solvcon::detail::strassen;
constexpr strassen::TransformSchedule SERIAL_TRANSFORM_SCHEDULE =
    strassen::TransformSchedule::Serial;
constexpr strassen::TransformSchedule PARALLEL_TRANSFORM_SCHEDULE =
    strassen::TransformSchedule::Parallel;

template <typename T>
void fill_operands(ssize_t rows, ssize_t columns, ssize_t inner_size, std::vector<T> & lhs, std::vector<T> & rhs)
{
    lhs.resize(static_cast<size_t>(rows * inner_size));
    rhs.resize(static_cast<size_t>(inner_size * columns));
    for (size_t index = 0; index < lhs.size(); ++index)
    {
        lhs[index] = static_cast<T>((index * 7) % 19) / T{8};
    }
    for (size_t index = 0; index < rhs.size(); ++index)
    {
        rhs[index] = static_cast<T>((index * 5 + 3) % 23) / T{16};
    }
}

template <typename T>
strassen::Gemm<T> make_gemm(ssize_t rows, ssize_t columns, ssize_t inner_size, T const * lhs, T const * rhs, T * output)
{
    solvcon::BlasMatrixView<T> const lhs_view{lhs, inner_size, solvcon::BlasTranspose::None};
    solvcon::BlasMatrixView<T> const rhs_view{rhs, columns, solvcon::BlasTranspose::None};
    return {rows, columns, inner_size, lhs_view, rhs_view, output};
}

template <typename T>
void reference_gemm(strassen::Gemm<T> const & gemm)
{
    for (ssize_t row = 0; row < gemm.rows; ++row)
    {
        for (ssize_t column = 0; column < gemm.columns; ++column)
        {
            T total{};
            for (ssize_t inner = 0; inner < gemm.inner_size; ++inner)
            {
                total += gemm.lhs.m_data[row * gemm.lhs.m_leading_dimension + inner] *
                         gemm.rhs.m_data[inner * gemm.rhs.m_leading_dimension + column];
            }
            gemm.output[row * gemm.columns + column] = total;
        }
    }
}

template <typename T>
void expect_near(std::vector<T> const & output, std::vector<T> const & expected)
{
    ASSERT_EQ(output.size(), expected.size());
    T const epsilon = std::numeric_limits<T>::epsilon();
    for (size_t index = 0; index < output.size(); ++index)
    {
        T const tolerance = T{128} * epsilon * std::max(T{1}, std::abs(expected[index]));
        EXPECT_NEAR(output[index], expected[index], tolerance) << "index " << index;
    }
}

template <typename T, size_t Depth>
size_t run_strassen(
    ssize_t rows,
    ssize_t columns,
    ssize_t inner_size,
    strassen::Workspace<T> & workspace,
    strassen::TransformSchedule transform_schedule)
{
    std::vector<T> lhs;
    std::vector<T> rhs;
    fill_operands(rows, columns, inner_size, lhs, rhs);
    std::vector<T> output(static_cast<size_t>(rows * columns));
    std::vector<T> expected(static_cast<size_t>(rows * columns));
    strassen::Gemm<T> const gemm = make_gemm(rows, columns, inner_size, lhs.data(), rhs.data(), output.data());
    strassen::Gemm<T> expected_gemm = gemm;
    expected_gemm.output = expected.data();
    reference_gemm(expected_gemm);
    size_t leaf_calls = 0;

    auto const leaf = [&leaf_calls](strassen::Gemm<T> const & leaf_gemm)
    {
        ++leaf_calls;
        reference_gemm(leaf_gemm);
    };
    strassen::multiply<Depth>(gemm, workspace, leaf, transform_schedule);

    expect_near(output, expected);
    return leaf_calls;
}

template <typename T, size_t Depth>
void check_depth(size_t expected_leaf_calls)
{
    strassen::Workspace<T> workspace;
    size_t const leaf_calls = run_strassen<T, Depth>(8, 12, 16, workspace, SERIAL_TRANSFORM_SCHEDULE);
    EXPECT_EQ(leaf_calls, expected_leaf_calls);
}

template <typename T>
void check_parallel_transforms()
{
    constexpr ssize_t rows = 8;
    constexpr ssize_t columns = 12;
    constexpr ssize_t inner_size = 16;
    std::vector<T> lhs;
    std::vector<T> rhs;
    fill_operands(rows, columns, inner_size, lhs, rhs);
    std::vector<T> serial_output(static_cast<size_t>(rows * columns));
    std::vector<T> parallel_output(static_cast<size_t>(rows * columns));
    strassen::Gemm<T> const serial_gemm = make_gemm(
        rows, columns, inner_size, lhs.data(), rhs.data(), serial_output.data());
    strassen::Gemm<T> const parallel_gemm = make_gemm(
        rows, columns, inner_size, lhs.data(), rhs.data(), parallel_output.data());
    auto const leaf = [](strassen::Gemm<T> const & product)
    { reference_gemm(product); };

    strassen::Workspace<T> serial_workspace;
    strassen::Workspace<T> parallel_workspace;
    strassen::multiply<1>(serial_gemm, serial_workspace, leaf, SERIAL_TRANSFORM_SCHEDULE);
    strassen::multiply<1>(parallel_gemm, parallel_workspace, leaf, PARALLEL_TRANSFORM_SCHEDULE);

    EXPECT_EQ(serial_output, parallel_output);
}

} /* end namespace */

TEST(StrassenKernel, matches_reference_at_each_depth)
{
    strassen::Workspace<double> workspace;
    size_t const leaf_calls = run_strassen<double, 0>(3, 5, 7, workspace, SERIAL_TRANSFORM_SCHEDULE);
    EXPECT_EQ(leaf_calls, 1);
    check_depth<float, 1>(7);
    check_depth<double, 1>(7);
    check_depth<float, 2>(49);
    check_depth<double, 2>(49);
}

TEST(StrassenKernel, matches_serial_with_parallel_transforms)
{
    check_parallel_transforms<float>();
    check_parallel_transforms<double>();
}

TEST(StrassenKernel, partitions_parallel_transform_rows)
{
    constexpr size_t row_count = 8;
    std::array<std::atomic_size_t, row_count> visits{};
    std::atomic_size_t callback_count = 0;
    auto const visit_rows = [&](ssize_t first, ssize_t last)
    {
        ++callback_count;
        for (ssize_t row = first; row < last; ++row)
        {
            ++visits[static_cast<size_t>(row)];
        }
    };

    strassen::TransformTeam team;
    strassen::run_transform_rows(&team, row_count, visit_rows);

    EXPECT_EQ(callback_count.load(), strassen::TransformTeam::LANE_COUNT);
    for (std::atomic_size_t const & visit_count : visits)
    {
        EXPECT_EQ(visit_count, 1);
    }
}

TEST(StrassenKernel, propagates_parallel_transform_errors_and_recovers)
{
    constexpr ssize_t row_count = 8;
    auto const fail_in_worker = [](ssize_t first, ssize_t)
    {
        if (first == 0)
        {
            throw std::runtime_error("parallel transform failed");
        }
    };

    strassen::TransformTeam team;
    EXPECT_THAT(
        [&]
        { strassen::run_transform_rows(&team, row_count, fail_in_worker); },
        testing::ThrowsMessage<std::runtime_error>("parallel transform failed"));

    constexpr ssize_t caller_first =
        row_count * (strassen::TransformTeam::LANE_COUNT - 1) /
        strassen::TransformTeam::LANE_COUNT;
    auto const fail_in_caller = [=](ssize_t first, ssize_t)
    {
        if (first == caller_first)
        {
            throw std::runtime_error("caller transform failed");
        }
    };
    EXPECT_THAT(
        [&]
        { strassen::run_transform_rows(&team, row_count, fail_in_caller); },
        testing::ThrowsMessage<std::runtime_error>("caller transform failed"));

    std::atomic_size_t visits = 0;
    auto const count_rows = [&](ssize_t first, ssize_t last)
    { visits += static_cast<size_t>(last - first); };
    strassen::run_transform_rows(&team, row_count, count_rows);
    EXPECT_EQ(visits.load(), row_count);
}

TEST(StrassenKernel, rejects_invalid_gemm)
{
    std::vector<double> lhs(128);
    std::vector<double> rhs(192);
    std::vector<double> output(96);
    strassen::Workspace<double> workspace;
    strassen::Gemm<double> gemm = make_gemm(8, 12, 16, lhs.data(), rhs.data(), output.data());
    auto const leaf = [](strassen::Gemm<double> const &) {};

    gemm.rows = 0;
    EXPECT_THAT(
        [&]
        { strassen::multiply<1>(gemm, workspace, leaf, SERIAL_TRANSFORM_SCHEDULE); },
        testing::ThrowsMessage<std::invalid_argument>("Strassen GEMM dimensions must be positive"));
    gemm.rows = 7;
    EXPECT_THAT(
        [&]
        { strassen::multiply<1>(gemm, workspace, leaf, SERIAL_TRANSFORM_SCHEDULE); },
        testing::ThrowsMessage<std::invalid_argument>("Strassen GEMM dimensions must be divisible by 2^depth"));
    gemm.rows = 8;
    gemm.lhs.m_transpose = solvcon::BlasTranspose::Transpose;
    EXPECT_THAT(
        [&]
        { strassen::multiply<1>(gemm, workspace, leaf, SERIAL_TRANSFORM_SCHEDULE); },
        testing::ThrowsMessage<std::invalid_argument>("Strassen GEMM does not support transposed input views"));
    gemm.lhs.m_transpose = solvcon::BlasTranspose::None;
    gemm.lhs.m_leading_dimension = 15;
    EXPECT_THAT(
        [&]
        { strassen::multiply<1>(gemm, workspace, leaf, SERIAL_TRANSFORM_SCHEDULE); },
        testing::ThrowsMessage<std::invalid_argument>("Strassen GEMM input leading dimensions are too small"));
}

TEST(StrassenKernel, reuses_workspace)
{
    strassen::Workspace<double> workspace;
    run_strassen<double, 2>(8, 12, 16, workspace, SERIAL_TRANSFORM_SCHEDULE);
    size_t const capacity = workspace.capacity();
    run_strassen<double, 1>(6, 10, 8, workspace, SERIAL_TRANSFORM_SCHEDULE);
    EXPECT_EQ(workspace.capacity(), capacity);
    run_strassen<double, 2>(8, 12, 16, workspace, SERIAL_TRANSFORM_SCHEDULE);
    EXPECT_EQ(workspace.capacity(), capacity);
}

TEST(StrassenKernel, blas_leaf)
{
    double lhs = 2;
    double rhs = 3;
    double output = 0;
    strassen::Workspace<double> workspace;
    strassen::Gemm<double> const gemm = make_gemm(1, 1, 1, &lhs, &rhs, &output);
#if (defined(__APPLE__) && defined(__arm64__)) || defined(SC_HAS_CBLAS)
    detail::gemm_strassen<0>(gemm, workspace, SERIAL_TRANSFORM_SCHEDULE);
    EXPECT_EQ(output, 6);
#else
    EXPECT_THAT(
        [&]
        { detail::gemm_strassen<0>(gemm, workspace, SERIAL_TRANSFORM_SCHEDULE); },
        testing::ThrowsMessage<std::runtime_error>("solvcon BLAS wrapper: CBLAS backend is unavailable"));
#endif
}

TEST(StrassenDispatch, selects_measured_axis_regions)
{
    auto const & tuning = detail::APPLE_ARM64_STRASSEN_TUNING;

    auto kernel = detail::select_strassen_kernel<float>(tuning, 5632, 5632, 5632);
    ASSERT_TRUE(kernel);
    EXPECT_EQ(*kernel, detail::MatmulKernel::StrassenDepth1);

    kernel = detail::select_strassen_kernel<float>(tuning, 3072, 3072, 24576);
    ASSERT_TRUE(kernel);
    EXPECT_EQ(*kernel, detail::MatmulKernel::StrassenDepth1);

    EXPECT_FALSE(detail::select_strassen_kernel<float>(tuning, 18930, 3072, 3072));
    EXPECT_FALSE(detail::select_strassen_kernel<float>(tuning, 3072, 18930, 3072));
    EXPECT_FALSE(detail::select_strassen_kernel<float>(tuning, 3072, 3072, 18930));
    EXPECT_FALSE(detail::select_strassen_kernel<float>(tuning, 6144, 6144, 6144));
    EXPECT_FALSE(detail::select_strassen_kernel<float>(tuning, 3072, 3072, 24578));

    kernel = detail::select_strassen_kernel<double>(tuning, 3072, 6144, 3072);
    ASSERT_TRUE(kernel);
    EXPECT_EQ(*kernel, detail::MatmulKernel::StrassenDepth1);

    kernel = detail::select_strassen_kernel<double>(tuning, 6144, 6144, 6144);
    ASSERT_TRUE(kernel);
    EXPECT_EQ(*kernel, detail::MatmulKernel::StrassenDepth2);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
