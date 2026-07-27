#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/loop.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace solvcon
{

namespace detail
{

namespace elementwise
{

using shape_type = LoopDomain::shape_type;
using stride_type = OperandMapping::stride_type;

class MappingSpan
{
public:
    MappingSpan(ssize_t minimum, ssize_t maximum)
        : m_minimum(minimum)
        , m_maximum(maximum)
    {
    }

    ssize_t minimum() const noexcept { return m_minimum; }
    ssize_t maximum() const noexcept { return m_maximum; }
    size_t size() const noexcept
    {
        return static_cast<size_t>(m_maximum - m_minimum + 1);
    }

private:
    ssize_t m_minimum;
    ssize_t m_maximum;
}; /* end class MappingSpan */

inline MappingSpan mapping_span(
    shape_type const & shape, stride_type const & strides)
{
    if (shape.size() != strides.size())
    {
        throw std::invalid_argument(
            "mapping rank does not match loop domain");
    }
    if (std::ranges::find(shape, 0) != shape.end())
    {
        return MappingSpan(0, -1);
    }

    ssize_t minimum = 0;
    ssize_t maximum = 0;
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

inline MappingSpan mapping_span(
    LoopDomain const & domain, OperandMapping const & mapping)
{
    if (domain.rank() != mapping.rank())
    {
        throw std::invalid_argument(
            "mapping rank does not match loop domain");
    }
    if (domain.size() == 0)
    {
        return MappingSpan(0, -1);
    }

    ssize_t minimum = 0;
    ssize_t maximum = 0;
    for (size_t axis = 0; axis < domain.rank(); ++axis)
    {
        ssize_t const end =
            (domain.extent(axis) - 1) * mapping.stride(axis);
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

inline bool mapping_is_row_major(
    LoopDomain const & domain, OperandMapping const & mapping)
{
    if (domain.rank() != mapping.rank())
    {
        return false;
    }

    ssize_t expected_stride = 1;
    for (size_t axis_plus_one = domain.rank();
         axis_plus_one > 0;
         --axis_plus_one)
    {
        size_t const axis = axis_plus_one - 1;
        if (domain.extent(axis) > 1 &&
            mapping.stride(axis) != expected_stride)
        {
            return false;
        }
        expected_stride *= domain.extent(axis);
    }
    return true;
}

inline bool mapping_is_dense(
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

inline bool mapping_is_dense(
    LoopDomain const & domain, OperandMapping const & mapping)
{
    if (domain.rank() != mapping.rank())
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
    for (size_t axis = 0; axis < domain.rank(); ++axis)
    {
        if (domain.extent(axis) == 0)
        {
            return true;
        }
        if (domain.extent(axis) > 1)
        {
            axes.push_back(axis);
        }
    }
    std::ranges::sort(
        axes,
        {},
        [&](size_t axis)
        { return magnitude(mapping.stride(axis)); });

    size_t expected_stride = 1;
    for (size_t const axis : axes)
    {
        if (magnitude(mapping.stride(axis)) != expected_stride)
        {
            return false;
        }
        expected_stride *=
            static_cast<size_t>(domain.extent(axis));
    }
    return true;
}

inline bool mapping_is_constant(
    LoopDomain const & domain, OperandMapping const & mapping)
{
    if (domain.rank() != mapping.rank())
    {
        return false;
    }
    for (size_t axis = 0; axis < domain.rank(); ++axis)
    {
        if (domain.extent(axis) > 1 &&
            mapping.stride(axis) != 0)
        {
            return false;
        }
    }
    return true;
}

inline bool mapping_strides_equal(
    OperandMapping const & lhs, OperandMapping const & rhs)
{
    if (lhs.rank() != rhs.rank())
    {
        return false;
    }
    for (size_t axis = 0; axis < lhs.rank(); ++axis)
    {
        if (lhs.stride(axis) != rhs.stride(axis))
        {
            return false;
        }
    }
    return true;
}

inline OperandMapping mapping_without_axis(
    OperandMapping const & mapping, size_t axis)
{
    if (axis >= mapping.rank())
    {
        throw std::out_of_range("mapping axis out of range");
    }

    stride_type strides;
    for (size_t source_axis = 0;
         source_axis < mapping.rank();
         ++source_axis)
    {
        if (source_axis != axis)
        {
            strides.push_back(mapping.stride(source_axis));
        }
    }
    return OperandMapping(std::move(strides));
}

} /* end namespace elementwise */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
