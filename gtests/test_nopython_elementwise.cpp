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
    EXPECT_EQ(lhs.strides(), (ew::stride_type{4, 0, 1}));
    EXPECT_EQ(rhs.strides(), (ew::stride_type{0, 1, 0}));
}

TEST(LoopTraversal, EmptyDomainHasNoCursorIteration)
{
    LoopDomain const domain(ew::shape_type{3, 0, 4});
    solvcon::small_vector<OperandMapping> const mappings{
        OperandMapping(ew::stride_type{0, 4, 1})};
    MappedOffsetCursor cursor(domain, mappings);

    EXPECT_TRUE(domain.empty());
    EXPECT_EQ(domain.size(), 0);
    EXPECT_FALSE(cursor);
}

TEST(LoopTraversal, SignedOffsetsRetainLogicalOrigin)
{
    LoopDomain const domain(ew::shape_type{2, 3});
    OperandMapping const mapping(ew::stride_type{-3, 1}, 3);
    solvcon::small_vector<OperandMapping> const mappings{mapping};
    std::array<ssize_t, 6> const expected{3, 4, 5, 0, 1, 2};

    size_t index = 0;
    for (MappedOffsetCursor cursor(domain, mappings);
         cursor;
         cursor.advance(), ++index)
    {
        EXPECT_EQ(cursor.offset(0), expected[index]);
    }
    EXPECT_EQ(index, expected.size());
    EXPECT_TRUE(mapping.is_dense(domain));
}

TEST(LoopTraversal, DenseMappingRejectsOverlappingStride)
{
    ew::shape_type const shape{2, 2};

    EXPECT_FALSE(
        OperandMapping::is_dense(
            shape, ew::stride_type{3, 0}));
    EXPECT_TRUE(
        OperandMapping::is_dense(
            shape, ew::stride_type{-1, 2}));
}

TEST(LoopTraversal, ConstantMappingIgnoresSingletonAxes)
{
    LoopDomain const domain(ew::shape_type{1, 3, 1});

    EXPECT_TRUE(
        OperandMapping(ew::stride_type{7, 0, 1})
            .is_constant(domain));
    EXPECT_FALSE(
        OperandMapping(ew::stride_type{7, 1, 1})
            .is_constant(domain));
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
    EXPECT_EQ(plan.input(0).strides(), (ew::stride_type{1, 0}));
    EXPECT_EQ(plan.input(1).strides(), (ew::stride_type{0, 1}));
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

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
