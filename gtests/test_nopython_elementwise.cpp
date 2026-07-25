/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/SimpleArray.hpp>
#include <solvcon/buffer/elementwise/plan.hpp>

#include <array>

#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

namespace ew = solvcon::detail::elementwise;

TEST(ElementwisePlan, BroadcastMappingsAlignTrailingAxes)
{
    ew::shape_type const lhs_shape{2, 1, 4};
    ew::shape_type const rhs_shape{1, 3, 1};
    ew::IterationDomain const domain(
        ew::IterationDomain::broadcast_shape(
            lhs_shape, rhs_shape));
    ew::OperandMapping const lhs = ew::OperandMapping::broadcast(
        lhs_shape, ew::stride_type{4, 4, 1}, domain);
    ew::OperandMapping const rhs = ew::OperandMapping::broadcast(
        rhs_shape, ew::stride_type{3, 1, 1}, domain);

    EXPECT_EQ(domain.shape(), (ew::shape_type{2, 3, 4}));
    EXPECT_EQ(lhs.strides(), (ew::stride_type{4, 0, 1}));
    EXPECT_EQ(rhs.strides(), (ew::stride_type{0, 1, 0}));
}

TEST(ElementwisePlan, EmptyDomainHasNoCursorIteration)
{
    ew::IterationDomain const domain(ew::shape_type{3, 0, 4});
    solvcon::small_vector<ew::OperandMapping> const mappings{
        ew::OperandMapping(ew::stride_type{0, 4, 1})};
    ew::OffsetCursor cursor(domain, mappings);

    EXPECT_TRUE(domain.empty());
    EXPECT_EQ(domain.size(), 0);
    EXPECT_FALSE(cursor);
}

TEST(ElementwisePlan, SignedOffsetsRetainLogicalOrigin)
{
    ew::IterationDomain const domain(ew::shape_type{2, 3});
    ew::OperandMapping const mapping(ew::stride_type{-3, 1}, 3);
    solvcon::small_vector<ew::OperandMapping> const mappings{mapping};
    std::array<ssize_t, 6> const expected{3, 4, 5, 0, 1, 2};

    size_t index = 0;
    for (ew::OffsetCursor cursor(domain, mappings);
         cursor;
         cursor.advance(), ++index)
    {
        EXPECT_EQ(cursor.offset(0), expected[index]);
    }
    EXPECT_EQ(index, expected.size());
    EXPECT_TRUE(mapping.is_dense(domain));
}

TEST(ElementwisePlan, DenseMappingRejectsOverlappingStride)
{
    ew::shape_type const shape{2, 2};

    EXPECT_FALSE(
        ew::OperandMapping::is_dense(
            shape, ew::stride_type{3, 0}));
    EXPECT_TRUE(
        ew::OperandMapping::is_dense(
            shape, ew::stride_type{-1, 2}));
}

TEST(ElementwisePlan, ConstantMappingIgnoresSingletonAxes)
{
    ew::IterationDomain const domain(ew::shape_type{1, 3, 1});

    EXPECT_TRUE(
        ew::OperandMapping(ew::stride_type{7, 0, 1})
            .is_constant(domain));
    EXPECT_FALSE(
        ew::OperandMapping(ew::stride_type{7, 1, 1})
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

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
