// Copyright (c) 2026, solvcon team <contact@solvcon.net>
// BSD 3-Clause License, see COPYING

#include <solvcon/buffer/SimpleArray.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <utility>

#ifdef SC_MATMUL_POLICY_TEST_FIXTURE_REQUIRED

#ifndef SC_MATMUL_POLICY_TEST_FIXTURE
#error "The configured matmul policy fixture was not included."
#endif

namespace
{

using array_type = solvcon::SimpleArray<double>;
using executor_type = solvcon::detail::MatmulExecutor<array_type>;

array_type make_matrix(ssize_t rows, ssize_t columns, double offset)
{
    array_type matrix(solvcon::small_vector<ssize_t>{rows, columns});
    for (ssize_t row = 0; row < rows; ++row)
    {
        for (ssize_t column = 0; column < columns; ++column)
        {
            matrix(row, column) = offset + row * columns + column;
        }
    }
    return matrix;
}

void expect_product(
    array_type const & lhs,
    array_type const & rhs,
    array_type const & output)
{
    for (ssize_t row = 0; row < lhs.shape(0); ++row)
    {
        for (ssize_t column = 0; column < rhs.shape(1); ++column)
        {
            double expected = 0;
            for (ssize_t inner = 0; inner < lhs.shape(1); ++inner)
            {
                expected += lhs(row, inner) * rhs(inner, column);
            }
            EXPECT_DOUBLE_EQ(output(row, column), expected);
        }
    }
}

std::optional<solvcon::detail::MatmulKernel> fixture_selection(
    executor_type const & executor)
{
    solvcon::detail::matmul_kernel_mask_type const mask =
        executor.eligible_kernels();
    auto const eligible = [mask](solvcon::detail::MatmulKernel kernel)
    {
        return (mask & solvcon::detail::matmul_kernel_bit(kernel)) != 0;
    };
    return solvcon::detail::select_calibrated_gemm(
        executor.facts(), eligible);
}

} /* end namespace */

TEST(MatmulPolicyFixture, NormalEntryPointSelectsDynamicInsideScope)
{
    using namespace solvcon;

    array_type const lhs = make_matrix(3, 4, 1);
    array_type const rhs = make_matrix(4, 7, 2);
    detail::MatmulPlan plan = detail::MatmulPlan::make(lhs, rhs);
    array_type output(plan.output_shape());
    executor_type executor(std::move(plan), output, lhs, rhs);

    std::optional<detail::MatmulKernel> const selection =
        fixture_selection(executor);
    ASSERT_TRUE(selection);
    EXPECT_EQ(*selection, detail::MatmulKernel::DynamicIkj);
    EXPECT_EQ(executor.current_kernel(), detail::MatmulKernel::DynamicIkj);
    executor.execute();
    expect_product(lhs, rhs, output);
}

TEST(MatmulPolicyFixture, NormalEntryPointFallsBackOutsideScope)
{
    using namespace solvcon;

    array_type const lhs = make_matrix(3, 4, 1);
    array_type const rhs = make_matrix(4, 8, 2);
    detail::MatmulPlan plan = detail::MatmulPlan::make(lhs, rhs);
    array_type output(plan.output_shape());
    executor_type executor(std::move(plan), output, lhs, rhs);

    EXPECT_EQ(fixture_selection(executor), std::nullopt);
    EXPECT_EQ(executor.current_kernel(), detail::MatmulKernel::GenericIjk);
    executor.execute();
    expect_product(lhs, rhs, output);
}

#endif

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
