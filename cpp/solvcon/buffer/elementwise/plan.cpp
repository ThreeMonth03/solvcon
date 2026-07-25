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

IterationDomain::IterationDomain(shape_type shape)
    : m_shape(std::move(shape))
{
}

size_t IterationDomain::size() const noexcept
{
    size_t count = 1;
    for (ssize_t const extent : m_shape)
    {
        count *= static_cast<size_t>(extent);
    }
    return count;
}

stride_type IterationDomain::row_major_strides(
    shape_type const & shape)
{
    stride_type strides(shape.size(), 1);
    for (size_t axis = shape.size(); axis > 1; --axis)
    {
        strides[axis - 2] =
            strides[axis - 1] * shape[axis - 1];
    }
    return strides;
}

shape_type IterationDomain::broadcast_shape(
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

size_t MappingSpan::size() const noexcept
{
    return static_cast<size_t>(m_maximum - m_minimum + 1);
}

OperandMapping::OperandMapping(
    stride_type strides, ssize_t base_offset)
    : m_base_offset(base_offset)
    , m_strides(std::move(strides))
{
}

MappingSpan OperandMapping::span(
    IterationDomain const & domain) const
{
    if (domain.empty())
    {
        return MappingSpan(0, -1);
    }
    return span(domain.shape(), m_strides, m_base_offset);
}

bool OperandMapping::is_row_major(
    IterationDomain const & domain) const
{
    if (domain.rank() != m_strides.size())
    {
        return false;
    }
    stride_type const expected =
        IterationDomain::row_major_strides(domain.shape());
    for (size_t axis = 0; axis < domain.rank(); ++axis)
    {
        if (domain.shape()[axis] > 1 &&
            m_strides[axis] != expected[axis])
        {
            return false;
        }
    }
    return true;
}

bool OperandMapping::is_dense(
    IterationDomain const & domain) const
{
    return is_dense(domain.shape(), m_strides);
}

bool OperandMapping::is_constant(
    IterationDomain const & domain) const
{
    if (domain.rank() != m_strides.size())
    {
        return false;
    }
    for (size_t axis = 0; axis < domain.rank(); ++axis)
    {
        if (domain.shape()[axis] > 1 && m_strides[axis] != 0)
        {
            return false;
        }
    }
    return true;
}

OperandMapping OperandMapping::without_axis(size_t axis) const
{
    if (axis >= m_strides.size())
    {
        throw std::out_of_range("mapping axis out of range");
    }
    stride_type strides;
    for (size_t source_axis = 0;
         source_axis < m_strides.size();
         ++source_axis)
    {
        if (source_axis != axis)
        {
            strides.push_back(m_strides[source_axis]);
        }
    }
    return OperandMapping(std::move(strides), m_base_offset);
}

MappingSpan OperandMapping::span(
    shape_type const & shape,
    stride_type const & strides,
    ssize_t base_offset)
{
    ssize_t minimum = base_offset;
    ssize_t maximum = base_offset;
    for (size_t axis = 0; axis < shape.size(); ++axis)
    {
        ssize_t const end = (shape[axis] - 1) * strides[axis];
        if (end < 0)
        {
            minimum += end;
        }
        else
        {
            maximum += end;
        }
    }
    return MappingSpan(minimum, maximum);
}

bool OperandMapping::is_dense(
    shape_type const & shape, stride_type const & strides)
{
    if (shape.size() != strides.size())
    {
        return false;
    }

    auto const magnitude = [](ssize_t stride)
    {
        return stride < 0
                   ? static_cast<size_t>(-(stride + 1)) + 1
                   : static_cast<size_t>(stride);
    };

    small_vector<size_t> axes;
    for (size_t axis = 0; axis < shape.size(); ++axis)
    {
        if (shape[axis] == 0)
        {
            return true;
        }
        if (shape[axis] > 1)
        {
            axes.push_back(axis);
        }
    }
    std::ranges::sort(
        axes,
        {},
        [&](size_t axis)
        { return magnitude(strides[axis]); });

    size_t expected_stride = 1;
    for (size_t const axis : axes)
    {
        if (magnitude(strides[axis]) != expected_stride)
        {
            return false;
        }
        expected_stride *= static_cast<size_t>(shape[axis]);
    }
    return true;
}

OperandMapping OperandMapping::exact(
    stride_type const & strides)
{
    return OperandMapping(strides);
}

OperandMapping OperandMapping::broadcast(
    shape_type const & operand_shape,
    stride_type const & operand_strides,
    IterationDomain const & domain)
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

OffsetCursor::OffsetCursor(
    IterationDomain const & domain,
    small_vector<OperandMapping> const & mappings)
    : m_domain(&domain)
    , m_mappings(&mappings)
    , m_index(domain.rank(), 0)
    , m_offsets(mappings.size(), 0)
    , m_valid(!domain.empty())
{
    for (size_t operand = 0; operand < mappings.size(); ++operand)
    {
        m_offsets[operand] = mappings[operand].base_offset();
    }
}

void OffsetCursor::advance()
{
    for (size_t axis_plus_one = m_domain->rank();
         axis_plus_one > 0;
         --axis_plus_one)
    {
        size_t const axis = axis_plus_one - 1;
        ++m_index[axis];
        for (size_t operand = 0;
             operand < m_mappings->size();
             ++operand)
        {
            m_offsets[operand] +=
                (*m_mappings)[operand].stride(axis);
        }
        if (m_index[axis] < m_domain->shape()[axis])
        {
            return;
        }
        m_index[axis] = 0;
        for (size_t operand = 0;
             operand < m_mappings->size();
             ++operand)
        {
            m_offsets[operand] -=
                (*m_mappings)[operand].stride(axis) *
                m_domain->shape()[axis];
        }
    }
    m_valid = false;
}

size_t select_inner_axis(
    IterationDomain const & domain,
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
        size_t score = output_stride == 1
                           ? 1000
                       : output_stride == -1
                           ? 800
                           : 400 / std::clamp(
                                       magnitude(output_stride),
                                       size_t{1},
                                       size_t{400});
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

InnerLoopPlan::InnerLoopPlan(
    IterationDomain const & domain,
    small_vector<OperandMapping> const & mappings,
    size_t inner_axis)
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

    shape_type outer_shape;
    for (size_t axis = 0; axis < domain.rank(); ++axis)
    {
        if (axis != inner_axis)
        {
            outer_shape.push_back(domain.shape()[axis]);
        }
    }
    m_outer = IterationDomain(std::move(outer_shape));
    m_size = static_cast<size_t>(domain.shape()[inner_axis]);
    m_strides = stride_type(mappings.size());
    m_outer_mappings =
        small_vector<OperandMapping>(mappings.size());
    for (size_t operand = 0; operand < mappings.size(); ++operand)
    {
        if (mappings[operand].strides().size() != domain.rank())
        {
            throw std::invalid_argument(
                "inner loop mapping rank does not match domain");
        }
        m_strides[operand] = mappings[operand].stride(inner_axis);
        m_outer_mappings[operand] =
            mappings[operand].without_axis(inner_axis);
    }
}

} /* end namespace elementwise */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
