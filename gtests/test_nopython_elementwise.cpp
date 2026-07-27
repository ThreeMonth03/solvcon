/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/SimpleArray.hpp>
#include <solvcon/buffer/elementwise/executor.hpp>
#include <solvcon/buffer/elementwise/plan.hpp>

#include <array>

#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

namespace ew = solvcon::detail::elementwise;
using solvcon::detail::LoopDomain;
using solvcon::detail::MappedOffsetCursor;
using solvcon::detail::OperandMapping;

namespace
{

class CountingAddKernel
    : public ew::BinaryKernelBase<CountingAddKernel, double>
{
public:
    double operator()(double lhs, double rhs) const
    {
        ++m_calls;
        return lhs + rhs;
    }

    static void contiguous(double * output,
                           size_t count,
                           double const * lhs,
                           double const * rhs);
    static void contiguous_scalar(double * output,
                                  size_t count,
                                  double const * lhs,
                                  double rhs);

    static void reset() noexcept;
    static size_t calls() noexcept { return m_calls; }
    static size_t contiguous_scalar_calls() noexcept
    {
        return m_contiguous_scalar_calls;
    }

private:
    inline static size_t m_calls = 0;
    inline static size_t m_contiguous_scalar_calls = 0;
}; /* end class CountingAddKernel */

void CountingAddKernel::reset() noexcept
{
    m_calls = 0;
    m_contiguous_scalar_calls = 0;
}

void CountingAddKernel::contiguous(
    double * output,
    size_t count,
    double const * lhs,
    double const * rhs)
{
    CountingAddKernel const kernel;
    for (size_t index = 0; index < count; ++index)
    {
        output[index] = kernel(lhs[index], rhs[index]);
    }
}

void CountingAddKernel::contiguous_scalar(
    double * output,
    size_t count,
    double const * lhs,
    double rhs)
{
    ++m_contiguous_scalar_calls;
    BinaryKernelBase::contiguous_scalar(
        output, count, lhs, rhs);
}

void expect_mapping_strides(
    OperandMapping const & mapping,
    ew::stride_type const & expected)
{
    ASSERT_EQ(mapping.rank(), expected.size());
    for (size_t axis = 0; axis < mapping.rank(); ++axis)
    {
        EXPECT_EQ(mapping.stride(axis), expected[axis]);
    }
}

} /* end namespace */

TEST(ElementwisePlan, BroadcastMappingsAlignTrailingAxes)
{
    ew::shape_type const lhs_shape{2, 1, 4};
    ew::shape_type const rhs_shape{1, 3, 1};
    LoopDomain const domain(
        ew::broadcast_shape(lhs_shape, rhs_shape));
    OperandMapping const lhs = ew::broadcast_mapping(
        lhs_shape, ew::stride_type{4, 4, 1}, domain);
    OperandMapping const rhs = ew::broadcast_mapping(
        rhs_shape, ew::stride_type{3, 1, 1}, domain);

    EXPECT_EQ(domain.shape(), (ew::shape_type{2, 3, 4}));
    expect_mapping_strides(lhs, ew::stride_type{4, 0, 1});
    expect_mapping_strides(rhs, ew::stride_type{0, 1, 0});
}

TEST(LoopTraversal, EmptyDomainHasNoCursorIteration)
{
    LoopDomain const domain(ew::shape_type{3, 0, 4});
    solvcon::small_vector<OperandMapping> const mappings{
        OperandMapping(ew::stride_type{0, 4, 1})};
    MappedOffsetCursor cursor(domain, mappings);

    EXPECT_EQ(domain.size(), 0);
    EXPECT_FALSE(cursor);
}

TEST(LoopTraversal, SignedOffsetsRetainLogicalOrigin)
{
    LoopDomain const domain(ew::shape_type{2, 3});
    OperandMapping const mapping(ew::stride_type{-3, 1});
    solvcon::small_vector<OperandMapping> const mappings{mapping};
    std::array<ssize_t, 6> const expected{0, 1, 2, -3, -2, -1};

    size_t index = 0;
    for (MappedOffsetCursor cursor(domain, mappings);
         cursor;
         cursor.advance(), ++index)
    {
        EXPECT_EQ(cursor.offset(size_t{0}), expected[index]);
    }
    EXPECT_EQ(index, expected.size());
    EXPECT_TRUE(ew::mapping_is_dense(domain, mapping));
}

TEST(LoopTraversal, DenseMappingRejectsOverlappingStride)
{
    ew::shape_type const shape{2, 2};

    EXPECT_FALSE(
        ew::mapping_is_dense(
            shape, ew::stride_type{3, 0}));
    EXPECT_TRUE(
        ew::mapping_is_dense(
            shape, ew::stride_type{-1, 2}));
}

TEST(LoopTraversal, ConstantMappingIgnoresSingletonAxes)
{
    LoopDomain const domain(ew::shape_type{1, 3, 1});

    EXPECT_TRUE(
        ew::mapping_is_constant(
            domain,
            OperandMapping(ew::stride_type{7, 0, 1})));
    EXPECT_FALSE(
        ew::mapping_is_constant(
            domain,
            OperandMapping(ew::stride_type{7, 1, 1})));
}

TEST(ElementwisePlan, OuterBroadcastSelectsFixedInnerLoop)
{
    solvcon::SimpleArray<double> lhs(ew::shape_type{5, 1});
    solvcon::SimpleArray<double> rhs(ew::shape_type{1, 7});
    solvcon::SimpleArray<double> output(ew::shape_type{5, 7});

    ew::ElementwisePlan const plan =
        ew::ElementwisePlan::make(output, lhs, rhs);

    EXPECT_EQ(plan.route(), ew::ExecutionRoute::inner_strided);
    EXPECT_EQ(plan.inner_axis(), 1);
    expect_mapping_strides(
        plan.input(0), ew::stride_type{1, 0});
    expect_mapping_strides(
        plan.input(1), ew::stride_type{0, 1});
}

TEST(ElementwisePlan, InnerAxisFollowsDensePermutedOutput)
{
    ew::shape_type const shape{5, 7};
    ew::stride_type const column_major{1, 5};
    solvcon::SimpleArray<double> lhs =
        ew::allocate_layout<solvcon::SimpleArray<double>>(
            shape, column_major);
    solvcon::SimpleArray<double> rhs(ew::shape_type{1, 7});
    solvcon::SimpleArray<double> output =
        ew::allocate_layout<solvcon::SimpleArray<double>>(
            shape, column_major);

    ew::ElementwisePlan const plan =
        ew::ElementwisePlan::make(output, lhs, rhs);

    EXPECT_EQ(plan.route(), ew::ExecutionRoute::inner_strided);
    EXPECT_EQ(plan.inner_axis(), 0);
}

TEST(ElementwisePlan, InnerAxisKeepsSmallOutputStride)
{
    ew::shape_type const shape{5, 7};
    ew::stride_type const stepped_inner{14, 2};
    solvcon::SimpleArray<double> lhs =
        ew::allocate_layout<solvcon::SimpleArray<double>>(
            shape, stepped_inner);
    solvcon::SimpleArray<double> rhs(ew::shape_type{1, 7});

    ew::ElementwisePlan const plan =
        ew::ElementwisePlan::make(lhs, lhs, rhs);

    EXPECT_EQ(plan.route(), ew::ExecutionRoute::inner_strided);
    EXPECT_EQ(plan.inner_axis(), 1);
}

TEST(ElementwiseExecutor, ReusesConstantInnerPair)
{
    using array_type = solvcon::SimpleArray<double>;
    array_type lhs(ew::shape_type{2, 1, 3, 1});
    array_type rhs = ew::allocate_layout<array_type>(
        ew::shape_type{1, 4, 1, 5},
        ew::stride_type{4, 1, 1, 0});

    CountingAddKernel::reset();
    array_type const output =
        ew::ElementwiseExecutor<
            array_type,
            double,
            CountingAddKernel>::transform(lhs, rhs, CountingAddKernel{});

    EXPECT_EQ(output.shape(), (ew::shape_type{2, 4, 3, 5}));
    EXPECT_EQ(CountingAddKernel::calls(), 24);
}

TEST(ElementwiseExecutor, NormalizesReversedInnerLoop)
{
    using array_type = solvcon::SimpleArray<double>;
    array_type destination = ew::allocate_layout<array_type>(
        ew::shape_type{3, 5}, ew::stride_type{5, -1});
    array_type rhs(ew::shape_type{3, 1});
    destination.fill(1.0);
    rhs.fill(2.0);

    CountingAddKernel::reset();
    ew::ElementwiseExecutor<
        array_type,
        double,
        CountingAddKernel>::transform_into(destination, rhs, CountingAddKernel{});

    EXPECT_EQ(CountingAddKernel::contiguous_scalar_calls(), 3);
}

TEST(ElementwiseExecutor, RankOneConstantInputUsesContiguousScalar)
{
    using array_type = solvcon::SimpleArray<double>;
    array_type lhs(ew::shape_type{8});
    array_type rhs = ew::allocate_layout<array_type>(
        ew::shape_type{8}, ew::stride_type{0});
    array_type destination(ew::shape_type{8});
    lhs.fill(1.0);
    rhs.at(0) = 2.0;

    CountingAddKernel::reset();
    ew::ElementwiseExecutor<
        array_type,
        double,
        CountingAddKernel>::transform_to(destination, lhs, rhs, CountingAddKernel{});

    EXPECT_EQ(CountingAddKernel::contiguous_scalar_calls(), 1);
    EXPECT_DOUBLE_EQ(destination.at(0), 3.0);
    EXPECT_DOUBLE_EQ(destination.at(7), 3.0);
}

TEST(ElementwiseExecutor, PreservesDenseFullShapeLayoutForBroadcast)
{
    using array_type = solvcon::SimpleArray<double>;
    array_type lhs = ew::allocate_layout<array_type>(
        ew::shape_type{3, 5}, ew::stride_type{1, 3});
    array_type rhs(ew::shape_type{3, 1});
    lhs.fill(1.0);
    rhs.fill(2.0);

    CountingAddKernel::reset();
    array_type const result = ew::ElementwiseExecutor<
        array_type,
        double,
        CountingAddKernel>::transform(lhs, rhs, CountingAddKernel{});

    EXPECT_EQ(result.stride(), lhs.stride());
    EXPECT_EQ(CountingAddKernel::calls(), 15);
}

TEST(ElementwiseExecutor, WritesBroadcastToPreallocatedOutput)
{
    using array_type = solvcon::SimpleArray<double>;
    array_type lhs(ew::shape_type{2, 1});
    array_type rhs(ew::shape_type{1, 3});
    array_type destination(ew::shape_type{2, 3});
    lhs.at(ew::shape_type{0, 0}) = 1.0;
    lhs.at(ew::shape_type{1, 0}) = 2.0;
    rhs.at(ew::shape_type{0, 0}) = 10.0;
    rhs.at(ew::shape_type{0, 1}) = 20.0;
    rhs.at(ew::shape_type{0, 2}) = 30.0;

    ew::ElementwiseExecutor<
        array_type,
        double,
        CountingAddKernel>::transform_to(destination, lhs, rhs, CountingAddKernel{});

    EXPECT_DOUBLE_EQ(
        destination.at(ew::shape_type{0, 0}), 11.0);
    EXPECT_DOUBLE_EQ(
        destination.at(ew::shape_type{0, 2}), 31.0);
    EXPECT_DOUBLE_EQ(
        destination.at(ew::shape_type{1, 0}), 12.0);
    EXPECT_DOUBLE_EQ(
        destination.at(ew::shape_type{1, 2}), 32.0);
}

TEST(ElementwiseExecutor, UpdatesStridedAliasWithConstantBroadcast)
{
    using array_type = solvcon::SimpleArray<double>;
    array_type destination = ew::allocate_layout<array_type>(
        ew::shape_type{3, 5}, ew::stride_type{10, 2});
    array_type rhs(ew::shape_type{1});
    rhs.fill(2.0);
    for (ssize_t row = 0; row < 3; ++row)
    {
        for (ssize_t column = 0; column < 5; ++column)
        {
            destination.at(ew::shape_type{row, column}) = 1.0;
        }
    }

    CountingAddKernel::reset();
    ew::ElementwiseExecutor<
        array_type,
        double,
        CountingAddKernel>::transform_into(destination, rhs, CountingAddKernel{});

    for (ssize_t row = 0; row < 3; ++row)
    {
        for (ssize_t column = 0; column < 5; ++column)
        {
            EXPECT_DOUBLE_EQ(
                destination.at(ew::shape_type{row, column}), 3.0);
        }
    }
    EXPECT_EQ(CountingAddKernel::calls(), 15);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
