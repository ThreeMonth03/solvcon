/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/elementwise/plan.hpp>

#include <algorithm>
#include <format>
#include <stdexcept>
#include <utility>

namespace solvcon
{

namespace detail
{

namespace elementwise
{

shape_type broadcast_shape(
    shape_type const & lhs, shape_type const & rhs)
{
    size_t const rank = std::max(lhs.size(), rhs.size());
    shape_type result(rank, 1);
    for (size_t offset = 0; offset < rank; ++offset)
    {
        ssize_t const lhs_extent = offset < lhs.size()
                                       ? lhs[lhs.size() - 1 - offset]
                                       : 1;
        ssize_t const rhs_extent = offset < rhs.size()
                                       ? rhs[rhs.size() - 1 - offset]
                                       : 1;
        if (lhs_extent != rhs_extent &&
            lhs_extent != 1 &&
            rhs_extent != 1)
        {
            throw std::invalid_argument(std::format(
                "cannot broadcast dimensions {} and {}",
                lhs_extent,
                rhs_extent));
        }
        result[rank - 1 - offset] = lhs_extent == 1
                                        ? rhs_extent
                                        : lhs_extent;
    }
    return result;
}

OperandMapping broadcast_mapping(
    shape_type const & operand_shape,
    stride_type const & operand_strides,
    LoopDomain const & domain)
{
    if (operand_shape.size() > domain.rank())
    {
        throw std::invalid_argument(
            "operand rank exceeds result rank");
    }

    stride_type strides(domain.rank(), 0);
    size_t const rank_delta = domain.rank() - operand_shape.size();
    for (size_t axis = 0; axis < operand_shape.size(); ++axis)
    {
        ssize_t const source_extent = operand_shape[axis];
        ssize_t const result_extent =
            domain.shape()[rank_delta + axis];
        if (source_extent == result_extent)
        {
            strides[rank_delta + axis] = operand_strides[axis];
        }
        else if (source_extent != 1)
        {
            throw std::invalid_argument(std::format(
                "cannot broadcast dimension {} to {}",
                source_extent,
                result_extent));
        }
    }
    return OperandMapping(std::move(strides));
}

size_t select_inner_axis(
    LoopDomain const & domain,
    OperandMapping const & output,
    small_vector<OperandMapping> const & inputs)
{
    if (domain.rank() == 0)
    {
        throw std::invalid_argument(
            "inner axis requires a positive-rank domain");
    }

    size_t selected_axis = domain.rank() - 1;
    size_t selected_score = 0;
    bool const has_nontrivial_axis = std::ranges::any_of(
        domain.shape(),
        [](ssize_t extent)
        { return extent > 1; });
    auto const magnitude = [](ssize_t stride)
    {
        return stride < 0
                   ? static_cast<size_t>(-(stride + 1)) + 1
                   : static_cast<size_t>(stride);
    };
    for (size_t axis = 0; axis < domain.rank(); ++axis)
    {
        if (has_nontrivial_axis && domain.shape()[axis] <= 1)
        {
            continue;
        }
        ssize_t const output_stride = output.stride(axis);
        size_t score = 0;
        if (output_stride == 1)
        {
            score = 1000;
        }
        else if (output_stride == -1)
        {
            score = 800;
        }
        else
        {
            score = 400 / std::clamp(
                              magnitude(output_stride),
                              size_t{1},
                              size_t{400});
        }
        for (OperandMapping const & input : inputs)
        {
            ssize_t const stride = input.stride(axis);
            if (stride == 0)
            {
                score += 200;
            }
            else if (stride == 1 || stride == -1)
            {
                score += 100;
            }
        }
        if (domain.shape()[axis] > 1)
        {
            score += std::min(
                static_cast<size_t>(domain.shape()[axis]),
                size_t{99});
        }
        if (score > selected_score)
        {
            selected_axis = axis;
            selected_score = score;
        }
    }
    return selected_axis;
}

shape_type InnerLoopPlan::make_outer_shape(
    LoopDomain const & domain, size_t inner_axis)
{
    if (domain.rank() == 0)
    {
        throw std::invalid_argument(
            "inner loop requires a positive-rank domain");
    }
    if (inner_axis >= domain.rank())
    {
        throw std::out_of_range("inner loop axis out of range");
    }

    shape_type shape;
    for (size_t axis = 0; axis < domain.rank(); ++axis)
    {
        if (axis != inner_axis)
        {
            shape.push_back(domain.extent(axis));
        }
    }
    return shape;
}

InnerLoopPlan::InnerLoopPlan(
    LoopDomain const & domain,
    small_vector<OperandMapping> const & mappings,
    size_t inner_axis)
    : m_outer(make_outer_shape(domain, inner_axis))
    , m_size(static_cast<size_t>(domain.extent(inner_axis)))
    , m_strides(mappings.size())
    , m_outer_mappings(mappings.size())
{
    for (size_t operand = 0; operand < mappings.size(); ++operand)
    {
        if (mappings[operand].rank() != domain.rank())
        {
            throw std::invalid_argument(
                "inner loop mapping rank does not match domain");
        }
        m_strides[operand] = mappings[operand].stride(inner_axis);
        m_outer_mappings[operand] =
            mapping_without_axis(mappings[operand], inner_axis);
    }
}

} /* end namespace elementwise */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
