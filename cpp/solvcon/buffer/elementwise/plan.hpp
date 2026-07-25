#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/small_vector.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace solvcon
{

namespace detail
{

namespace elementwise
{

using shape_type = small_vector<ssize_t>;
using stride_type = small_vector<ssize_t>;

class IterationDomain
{
public:
    IterationDomain() = default;
    explicit IterationDomain(shape_type shape);

    shape_type const & shape() const noexcept { return m_shape; }
    size_t rank() const noexcept { return m_shape.size(); }
    size_t size() const noexcept;
    bool empty() const noexcept { return size() == 0; }

    static stride_type row_major_strides(shape_type const & shape);
    static shape_type broadcast_shape(shape_type const & lhs,
                                      shape_type const & rhs);

private:
    shape_type m_shape;
}; /* end class IterationDomain */

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
    size_t size() const noexcept;

private:
    ssize_t m_minimum = 0;
    ssize_t m_maximum = -1;
}; /* end class MappingSpan */

class OperandMapping
{
public:
    OperandMapping() = default;
    explicit OperandMapping(stride_type strides,
                            ssize_t base_offset = 0);

    ssize_t base_offset() const noexcept { return m_base_offset; }
    stride_type const & strides() const noexcept { return m_strides; }
    ssize_t stride(size_t axis) const noexcept { return m_strides[axis]; }

    MappingSpan span(IterationDomain const & domain) const;
    bool is_row_major(IterationDomain const & domain) const;
    bool is_dense(IterationDomain const & domain) const;
    OperandMapping without_last_axis() const;

    static MappingSpan span(shape_type const & shape,
                            stride_type const & strides,
                            ssize_t base_offset = 0);
    static bool is_dense(shape_type const & shape,
                         stride_type const & strides);
    static OperandMapping exact(stride_type const & strides);
    static OperandMapping broadcast(shape_type const & operand_shape,
                                    stride_type const & operand_strides,
                                    IterationDomain const & domain);

private:
    ssize_t m_base_offset = 0;
    stride_type m_strides;
}; /* end class OperandMapping */

class OffsetCursor
{
public:
    OffsetCursor(IterationDomain const & domain,
                 small_vector<OperandMapping> const & mappings);

    explicit operator bool() const noexcept { return m_valid; }
    ssize_t offset(size_t operand) const noexcept
    {
        return m_offsets[operand];
    }

    void advance();

private:
    IterationDomain const * m_domain;
    small_vector<OperandMapping> const * m_mappings;
    shape_type m_index;
    stride_type m_offsets;
    bool m_valid;
}; /* end class OffsetCursor */

class InnerLoopPlan
{
public:
    InnerLoopPlan(IterationDomain const & domain,
                  small_vector<OperandMapping> const & mappings);

    IterationDomain const & outer() const noexcept { return m_outer; }
    size_t size() const noexcept { return m_size; }
    ssize_t stride(size_t operand) const noexcept
    {
        return m_strides[operand];
    }
    small_vector<OperandMapping> const & outer_mappings() const noexcept
    {
        return m_outer_mappings;
    }

private:
    IterationDomain m_outer;
    size_t m_size = 0;
    stride_type m_strides;
    small_vector<OperandMapping> m_outer_mappings;
}; /* end class InnerLoopPlan */

enum class ExecutionRoute : uint8_t
{
    contiguous,
    inner_strided,
    mapped,
}; /* end enum class ExecutionRoute */

class ElementwisePlan
{
public:
    IterationDomain const & domain() const noexcept { return m_domain; }
    OperandMapping const & output() const noexcept { return m_output; }
    OperandMapping const & input(size_t index) const noexcept
    {
        return m_inputs[index];
    }
    ExecutionRoute route() const noexcept { return m_route; }

    template <typename... Inputs>
    static shape_type broadcast_shape(Inputs const &... inputs);

    template <typename Output, typename... Inputs>
    static ElementwisePlan make(Output const & output,
                                Inputs const &... inputs);

    template <typename Output, typename Input>
    static ElementwisePlan make_scalar(Output const & output,
                                       Input const & input);

private:
    IterationDomain m_domain;
    OperandMapping m_output;
    small_vector<OperandMapping> m_inputs;
    ExecutionRoute m_route = ExecutionRoute::mapped;
}; /* end class ElementwisePlan */

template <typename... Inputs>
shape_type ElementwisePlan::broadcast_shape(
    Inputs const &... inputs)
{
    shape_type shape;
    ((shape = IterationDomain::broadcast_shape(
          shape, inputs.shape())),
     ...);
    return shape;
}

template <typename Output, typename... Inputs>
ElementwisePlan ElementwisePlan::make(
    Output const & output, Inputs const &... inputs)
{
    IterationDomain const domain(broadcast_shape(inputs...));
    if (!std::ranges::equal(output.shape(), domain.shape()))
    {
        throw std::invalid_argument(
            "elementwise output shape does not match result shape");
    }

    ElementwisePlan plan;
    plan.m_domain = domain;
    plan.m_output = OperandMapping::exact(output.stride());
    plan.m_inputs = small_vector<OperandMapping>{
        OperandMapping::broadcast(
            inputs.shape(), inputs.stride(), domain)...};

    bool row_major = plan.m_output.is_row_major(domain);
    bool common_dense_layout = plan.m_output.is_dense(domain);
    for (OperandMapping const & input : plan.m_inputs)
    {
        row_major = row_major && input.is_row_major(domain);
        common_dense_layout =
            common_dense_layout &&
            std::ranges::equal(
                plan.m_output.strides(), input.strides());
    }
    if (row_major || common_dense_layout)
    {
        plan.m_route = ExecutionRoute::contiguous;
    }
    else if (domain.rank() != 0)
    {
        plan.m_route = ExecutionRoute::inner_strided;
    }
    return plan;
}

template <typename Output, typename Input>
ElementwisePlan ElementwisePlan::make_scalar(
    Output const & output, Input const & input)
{
    if (!std::ranges::equal(output.shape(), input.shape()))
    {
        throw std::invalid_argument(
            "scalar elementwise output shape mismatch");
    }

    ElementwisePlan plan;
    plan.m_domain = IterationDomain(output.shape());
    plan.m_output = OperandMapping::exact(output.stride());
    plan.m_inputs = small_vector<OperandMapping>{
        OperandMapping::broadcast(
            input.shape(), input.stride(), plan.m_domain)};
    bool const row_major =
        plan.m_output.is_row_major(plan.m_domain) &&
        plan.m_inputs[0].is_row_major(plan.m_domain);
    bool const common_dense_layout =
        std::ranges::equal(
            plan.m_output.strides(),
            plan.m_inputs[0].strides()) &&
        plan.m_output.is_dense(plan.m_domain);
    if (row_major || common_dense_layout)
    {
        plan.m_route = ExecutionRoute::contiguous;
    }
    else if (plan.m_domain.rank() != 0)
    {
        plan.m_route = ExecutionRoute::inner_strided;
    }
    return plan;
}

template <typename Array>
Array allocate_layout(shape_type const & shape,
                      stride_type const & strides)
{
    IterationDomain const domain(shape);
    if (domain.empty())
    {
        return Array(shape);
    }
    MappingSpan const span = OperandMapping(strides).span(domain);
    auto buffer = Array::buffer_type::construct(
        span.size() * Array::ITEMSIZE);
    size_t const data_offset =
        static_cast<size_t>(-span.minimum()) * Array::ITEMSIZE;
    return Array(shape, strides, buffer, data_offset);
}

} /* end namespace elementwise */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
