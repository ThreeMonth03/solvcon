#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/loop.hpp>

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

using shape_type = LoopDomain::shape_type;
using stride_type = LoopDomain::stride_type;

shape_type broadcast_shape(shape_type const & lhs,
                           shape_type const & rhs);
OperandMapping broadcast_mapping(
    shape_type const & operand_shape,
    stride_type const & operand_strides,
    LoopDomain const & domain);

class InnerLoopPlan
{
public:
    InnerLoopPlan(LoopDomain const & domain,
                  small_vector<OperandMapping> const & mappings,
                  size_t inner_axis);

    LoopDomain const & outer() const noexcept { return m_outer; }
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
    LoopDomain m_outer;
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

size_t select_inner_axis(
    LoopDomain const & domain,
    OperandMapping const & output,
    small_vector<OperandMapping> const & inputs);

class ElementwisePlan
{
public:
    LoopDomain const & domain() const noexcept { return m_domain; }
    OperandMapping const & output() const noexcept { return m_output; }
    OperandMapping const & input(size_t index) const noexcept
    {
        return m_inputs[index];
    }
    ExecutionRoute route() const noexcept { return m_route; }
    size_t inner_axis() const noexcept { return m_inner_axis; }

    template <typename... Inputs>
    static shape_type broadcast_shape(Inputs const &... inputs);

    template <typename Output, typename... Inputs>
    static ElementwisePlan make(Output const & output,
                                Inputs const &... inputs);

    template <typename Output, typename Input>
    static ElementwisePlan make_scalar(Output const & output,
                                       Input const & input);

private:
    LoopDomain m_domain;
    OperandMapping m_output;
    small_vector<OperandMapping> m_inputs;
    ExecutionRoute m_route = ExecutionRoute::mapped;
    size_t m_inner_axis = 0;
}; /* end class ElementwisePlan */

template <typename... Inputs>
shape_type ElementwisePlan::broadcast_shape(
    Inputs const &... inputs)
{
    shape_type shape;
    ((shape = elementwise::broadcast_shape(
          shape, inputs.shape())),
     ...);
    return shape;
}

template <typename Output, typename... Inputs>
ElementwisePlan ElementwisePlan::make(
    Output const & output, Inputs const &... inputs)
{
    LoopDomain const domain(broadcast_shape(inputs...));
    if (!std::ranges::equal(output.shape(), domain.shape()))
    {
        throw std::invalid_argument(
            "elementwise output shape does not match result shape");
    }

    ElementwisePlan plan;
    plan.m_domain = domain;
    plan.m_output = OperandMapping(output.stride());
    plan.m_inputs = small_vector<OperandMapping>{
        broadcast_mapping(
            inputs.shape(), inputs.stride(), domain)...};
    if (domain.rank() != 0)
    {
        plan.m_inner_axis = select_inner_axis(
            domain, plan.m_output, plan.m_inputs);
    }

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
    plan.m_domain = LoopDomain(output.shape());
    plan.m_output = OperandMapping(output.stride());
    plan.m_inputs = small_vector<OperandMapping>{
        broadcast_mapping(
            input.shape(), input.stride(), plan.m_domain)};
    if (plan.m_domain.rank() != 0)
    {
        plan.m_inner_axis = select_inner_axis(
            plan.m_domain, plan.m_output, plan.m_inputs);
    }
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
    LoopDomain const domain(shape);
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
