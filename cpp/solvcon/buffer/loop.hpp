#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/small_vector.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace solvcon
{

namespace detail
{

class LoopDomain
{
public:
    using shape_type = small_vector<ssize_t>;
    using stride_type = small_vector<ssize_t>;

    LoopDomain() = default;
    explicit LoopDomain(shape_type shape);

    shape_type const & shape() const noexcept { return m_shape; }
    size_t rank() const noexcept { return m_shape.size(); }
    size_t size() const noexcept;

    static stride_type row_major_strides(
        shape_type const & shape, ssize_t trailing_size = 1);

private:
    shape_type m_shape;
}; /* end class LoopDomain */

class OperandMapping
{
public:
    using stride_type = small_vector<ssize_t>;

    OperandMapping() = default;
    explicit OperandMapping(stride_type strides);

    size_t rank() const noexcept { return m_strides.size(); }
    ssize_t stride(size_t axis) const noexcept { return m_strides[axis]; }

private:
    stride_type m_strides;
}; /* end class OperandMapping */

class MappedOffsetCursor
{
public:
    using mapping_type = small_vector<OperandMapping>;

    MappedOffsetCursor(
        LoopDomain const & domain, mapping_type const & mappings);

    explicit operator bool() const noexcept { return m_valid; }
    ssize_t offset(size_t operand) const noexcept
    {
        return m_offsets[operand];
    }
    void advance();

private:
    LoopDomain const * m_domain;
    mapping_type const * m_mappings;
    LoopDomain::shape_type m_index;
    LoopDomain::stride_type m_offsets;
    bool m_valid = false;
}; /* end class MappedOffsetCursor */

inline LoopDomain::LoopDomain(shape_type shape)
    : m_shape(std::move(shape))
{
}

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

inline OperandMapping::OperandMapping(stride_type strides)
    : m_strides(std::move(strides))
{
}

inline MappedOffsetCursor::MappedOffsetCursor(
    LoopDomain const & domain, mapping_type const & mappings)
    : m_domain(&domain)
    , m_mappings(&mappings)
    , m_index(domain.rank(), 0)
    , m_offsets(mappings.size(), 0)
    , m_valid(domain.size() != 0)
{
    for (OperandMapping const & mapping : mappings)
    {
        if (mapping.rank() != domain.rank())
        {
            throw std::invalid_argument(
                "operand mapping rank does not match its loop domain");
        }
    }
}

inline void MappedOffsetCursor::advance()
{
    for (size_t axis_plus_one = m_domain->rank(); axis_plus_one > 0;
         --axis_plus_one)
    {
        size_t const axis = axis_plus_one - 1;
        ++m_index[axis];
        for (size_t operand = 0; operand < m_mappings->size(); ++operand)
        {
            m_offsets[operand] += (*m_mappings)[operand].stride(axis);
        }
        if (m_index[axis] < m_domain->shape()[axis])
        {
            return;
        }

        m_index[axis] = 0;
        for (size_t operand = 0; operand < m_mappings->size(); ++operand)
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
