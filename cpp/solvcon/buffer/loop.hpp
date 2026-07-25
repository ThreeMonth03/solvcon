#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/small_vector.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace solvcon
{

namespace detail
{

/**
 * @brief Describe a shared runtime-rank iteration space.
 *
 * A higher-level operation plan creates one LoopDomain for every set of
 * coordinates traversed together. The domain owns only the extents and
 * defines their rank, bounds, and empty-domain behavior. Operand-specific
 * strides remain in OperandMapping, while operation-specific axes such as
 * M, N, K, or kept and reduced axes remain in the higher-level plan.
 *
 * For `(2,1,3,4) @ (1,5,4,6)`, a broadcast-capable MatmulPlan uses
 * `LoopDomain({2,5})` for the ten result-batch coordinates. M=3, N=6, and
 * K=4 remain matrix metadata rather than becoming axes of this domain.
 */
class LoopDomain
{
public:
    using shape_type = small_vector<ssize_t>;
    using stride_type = small_vector<ssize_t>;

    LoopDomain() = default;
    explicit LoopDomain(shape_type shape)
        : m_shape(std::move(shape))
    {
    }

    shape_type const & shape() const noexcept { return m_shape; }
    ssize_t extent(size_t axis) const noexcept { return m_shape[axis]; }
    size_t rank() const noexcept { return m_shape.size(); }
    size_t size() const noexcept;
    bool empty() const noexcept { return size() == 0; }

    static stride_type row_major_strides(
        shape_type const & shape, ssize_t trailing_size = 1);

private:
    shape_type m_shape;
}; /* end class LoopDomain */

class MappingSpan
{
public:
    MappingSpan() = default;
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
    ssize_t m_minimum = 0;
    ssize_t m_maximum = -1;
}; /* end class MappingSpan */

/**
 * @brief Describe how one operand advances over a LoopDomain.
 *
 * Each OperandMapping records one signed stride for every domain axis. This
 * separates the shared coordinate space from each operand layout: positive
 * and negative strides traverse supplied layouts, while a zero stride reuses
 * one operand position along a broadcast axis. contiguous_blocks() constructs
 * the mapping for adjacent fixed-size blocks.
 *
 * For `(2,1,3,4) @ (1,5,4,6)`, the domain is `(2,5)`. The output, lhs, and
 * rhs mappings are `{90,18}`, `{12,0}`, and `{0,24}`. The zero lhs stride
 * reuses it across axis 1, and the zero rhs stride reuses it across axis 0.
 */
class OperandMapping
{
public:
    using stride_type = LoopDomain::stride_type;

    OperandMapping() = default;
    explicit OperandMapping(stride_type strides,
                            ssize_t base_offset = 0)
        : m_base_offset(base_offset)
        , m_strides(std::move(strides))
    {
    }

    static OperandMapping contiguous_blocks(LoopDomain const & domain, ssize_t block_size);

    size_t rank() const noexcept { return m_strides.size(); }
    ssize_t base_offset() const noexcept { return m_base_offset; }
    stride_type const & strides() const noexcept { return m_strides; }
    ssize_t stride(size_t axis) const noexcept { return m_strides[axis]; }

    MappingSpan span(LoopDomain const & domain) const;
    bool is_row_major(LoopDomain const & domain) const;
    bool is_dense(LoopDomain const & domain) const;
    bool is_constant(LoopDomain const & domain) const;
    OperandMapping without_axis(size_t axis) const;

    static MappingSpan span(LoopDomain::shape_type const & shape,
                            stride_type const & strides,
                            ssize_t base_offset = 0);
    static bool is_dense(LoopDomain::shape_type const & shape,
                         stride_type const & strides);

private:
    ssize_t m_base_offset = 0;
    stride_type m_strides;
}; /* end class OperandMapping */

/**
 * @brief Advance all operand offsets with one runtime-rank cursor.
 *
 * MappedOffsetCursor owns one coordinate counter for the LoopDomain and one
 * relative offset for each OperandMapping. advance() updates every offset
 * together, so executors do not need rank-specific nested loops or separate
 * coordinate traversal for each operand. The cursor does not dereference
 * data or execute an operation.
 *
 * With domain `(2,5)` and mappings `{90,18}`, `{12,0}`, and `{0,24}`, the
 * output, lhs, and rhs offsets start at `(0,0,0)`. They advance to
 * `(18,0,24)` for coordinate `(0,1)` and `(90,12,0)` for coordinate `(1,0)`.
 *
 * @note The cursor borrows its domain and mappings, which must outlive it.
 */
class MappedOffsetCursor
{
public:
    using mapping_type = small_vector<OperandMapping>;

    MappedOffsetCursor(LoopDomain const & domain,
                       mapping_type const & mappings);

    explicit operator bool() const noexcept { return m_valid; }
    ssize_t offset(size_t operand) const noexcept
    {
        return m_offsets[operand];
    }

    template <typename Operand>
    ssize_t offset(Operand operand) const noexcept
    {
        static_assert(std::is_enum_v<Operand>, "cursor operand must be an enum");
        return offset(static_cast<size_t>(std::to_underlying(operand)));
    }

    void advance();

private:
    LoopDomain const * m_domain;
    mapping_type const * m_mappings;
    LoopDomain::shape_type m_index;
    LoopDomain::stride_type m_offsets;
    bool m_valid = false;
}; /* end class MappedOffsetCursor */

inline size_t LoopDomain::size() const noexcept
{
    size_t count = 1;
    for (ssize_t const extent : m_shape)
    {
        count *= static_cast<size_t>(extent);
    }
    return count;
}

inline LoopDomain::stride_type LoopDomain::row_major_strides(
    shape_type const & shape, ssize_t trailing_size)
{
    stride_type strides(shape.size(), trailing_size);
    for (size_t axis = shape.size(); axis > 1; --axis)
    {
        strides[axis - 2] =
            strides[axis - 1] * shape[axis - 1];
    }
    return strides;
}

inline OperandMapping OperandMapping::contiguous_blocks(
    LoopDomain const & domain, ssize_t block_size)
{
    return OperandMapping(
        LoopDomain::row_major_strides(
            domain.shape(), block_size));
}

inline MappingSpan OperandMapping::span(
    LoopDomain const & domain) const
{
    if (domain.empty())
    {
        return MappingSpan(0, -1);
    }
    return span(domain.shape(), m_strides, m_base_offset);
}

inline bool OperandMapping::is_row_major(
    LoopDomain const & domain) const
{
    if (domain.rank() != rank())
    {
        return false;
    }
    stride_type const expected =
        LoopDomain::row_major_strides(domain.shape());
    for (size_t axis = 0; axis < domain.rank(); ++axis)
    {
        if (domain.shape()[axis] > 1 &&
            stride(axis) != expected[axis])
        {
            return false;
        }
    }
    return true;
}

inline bool OperandMapping::is_dense(
    LoopDomain const & domain) const
{
    return is_dense(domain.shape(), m_strides);
}

inline bool OperandMapping::is_constant(
    LoopDomain const & domain) const
{
    if (domain.rank() != rank())
    {
        return false;
    }
    for (size_t axis = 0; axis < domain.rank(); ++axis)
    {
        if (domain.shape()[axis] > 1 && stride(axis) != 0)
        {
            return false;
        }
    }
    return true;
}

inline OperandMapping OperandMapping::without_axis(size_t axis) const
{
    if (axis >= rank())
    {
        throw std::out_of_range("mapping axis out of range");
    }
    stride_type strides;
    for (size_t source_axis = 0; source_axis < rank(); ++source_axis)
    {
        if (source_axis != axis)
        {
            strides.push_back(m_strides[source_axis]);
        }
    }
    return OperandMapping(std::move(strides), m_base_offset);
}

inline MappingSpan OperandMapping::span(
    LoopDomain::shape_type const & shape,
    stride_type const & strides,
    ssize_t base_offset)
{
    if (shape.size() != strides.size())
    {
        throw std::invalid_argument(
            "mapping rank does not match loop domain");
    }

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

inline bool OperandMapping::is_dense(
    LoopDomain::shape_type const & shape,
    stride_type const & strides)
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

inline MappedOffsetCursor::MappedOffsetCursor(
    LoopDomain const & domain, mapping_type const & mappings)
    : m_domain(&domain)
    , m_mappings(&mappings)
    , m_index(domain.rank(), 0)
    , m_offsets(mappings.size(), 0)
    , m_valid(!domain.empty())
{
    for (size_t operand = 0; operand < mappings.size(); ++operand)
    {
        if (mappings[operand].rank() != domain.rank())
        {
            throw std::invalid_argument(
                "operand mapping rank does not match its loop domain");
        }
        m_offsets[operand] = mappings[operand].base_offset();
    }
}

inline void MappedOffsetCursor::advance()
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

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
