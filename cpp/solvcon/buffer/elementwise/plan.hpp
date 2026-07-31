#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/elementwise/layout.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace solvcon
{

namespace detail
{

namespace elementwise
{

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
    struct AxisPartition
    {
        small_vector<size_t> m_inner;
        small_vector<size_t> m_outer;
    }; /* end struct AxisPartition */

    InnerLoopPlan(
        LoopDomain const & domain,
        small_vector<OperandMapping> const & mappings,
        size_t inner_axis,
        AxisPartition const & axes);
    static shape_type make_outer_shape(
        LoopDomain const & domain,
        small_vector<size_t> const & outer_axes);
    static AxisPartition partition_axes(
        LoopDomain const & domain,
        small_vector<OperandMapping> const & mappings,
        size_t inner_axis);

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
    ElementwisePlan(
        LoopDomain domain,
        OperandMapping output,
        small_vector<OperandMapping> inputs,
        ExecutionRoute route,
        size_t inner_axis);

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
    LoopDomain domain(broadcast_shape(inputs...));
    if (!std::ranges::equal(output.shape(), domain.shape()))
    {
        throw std::invalid_argument(
            "elementwise output shape does not match result shape");
    }

    OperandMapping output_mapping(output.stride());
    small_vector<OperandMapping> input_mappings{
        broadcast_mapping(
            inputs.shape(), inputs.stride(), domain)...};
    size_t inner_axis = 0;
    if (domain.rank() != 0)
    {
        inner_axis = select_inner_axis(
            domain, output_mapping, input_mappings);
    }

    bool row_major =
        mapping_is_row_major(domain, output_mapping);
    bool common_dense_layout =
        mapping_is_dense(domain, output_mapping);
    for (OperandMapping const & input : input_mappings)
    {
        row_major =
            row_major && mapping_is_row_major(domain, input);
        common_dense_layout =
            common_dense_layout &&
            mapping_strides_equal(output_mapping, input);
    }
    ExecutionRoute route = ExecutionRoute::mapped;
    if (row_major || common_dense_layout)
    {
        route = ExecutionRoute::contiguous;
    }
    else if (domain.rank() != 0)
    {
        route = ExecutionRoute::inner_strided;
    }
    return ElementwisePlan(
        std::move(domain),
        std::move(output_mapping),
        std::move(input_mappings),
        route,
        inner_axis);
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

    LoopDomain domain(output.shape());
    OperandMapping output_mapping(output.stride());
    small_vector<OperandMapping> input_mappings{
        broadcast_mapping(
            input.shape(), input.stride(), domain)};
    size_t inner_axis = 0;
    if (domain.rank() != 0)
    {
        inner_axis = select_inner_axis(
            domain, output_mapping, input_mappings);
    }
    bool const row_major =
        mapping_is_row_major(domain, output_mapping) &&
        mapping_is_row_major(domain, input_mappings[0]);
    bool const common_dense_layout =
        mapping_strides_equal(
            output_mapping, input_mappings[0]) &&
        mapping_is_dense(domain, output_mapping);
    ExecutionRoute route = ExecutionRoute::mapped;
    if (row_major || common_dense_layout)
    {
        route = ExecutionRoute::contiguous;
    }
    else if (domain.rank() != 0)
    {
        route = ExecutionRoute::inner_strided;
    }
    return ElementwisePlan(
        std::move(domain),
        std::move(output_mapping),
        std::move(input_mappings),
        route,
        inner_axis);
}

template <typename Array>
Array allocate_layout(shape_type const & shape,
                      stride_type const & strides)
{
    LoopDomain const domain(shape);
    if (domain.size() == 0)
    {
        return Array(shape);
    }

    stride_type f_strides(shape.size());
    ssize_t f_stride = 1;
    for (size_t axis = 0; axis < shape.size(); ++axis)
    {
        f_strides[axis] = f_stride;
        f_stride *= shape[axis];
    }
    if (std::ranges::equal(strides, f_strides))
    {
        shape_type reversed_shape(shape);
        std::reverse(
            reversed_shape.begin(), reversed_shape.end());
        Array output(reversed_shape);
        output.transpose();
        return output;
    }

    MappingSpan const span =
        mapping_span(domain, OperandMapping(strides));
    auto buffer = Array::buffer_type::construct(
        span.size() * Array::ITEMSIZE);
    size_t const data_offset =
        static_cast<size_t>(-span.minimum()) * Array::ITEMSIZE;
    return Array(shape, strides, buffer, data_offset);
}

inline ElementwisePlan::ElementwisePlan(
    LoopDomain domain,
    OperandMapping output,
    small_vector<OperandMapping> inputs,
    ExecutionRoute route,
    size_t inner_axis)
    : m_domain(std::move(domain))
    , m_output(std::move(output))
    , m_inputs(std::move(inputs))
    , m_route(route)
    , m_inner_axis(inner_axis)
{
}

} /* end namespace elementwise */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
