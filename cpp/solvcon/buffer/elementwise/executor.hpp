#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/elementwise/kernel.hpp>
#include <solvcon/buffer/elementwise/plan.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

namespace solvcon
{

namespace detail
{

namespace elementwise
{

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
class ElementwiseExecutor
{
public:
    using value_type = T;
    using kernel_type = Kernel;

    static Array transform(Array const & lhs,
                           Array const & rhs,
                           kernel_type kernel);
    static Array transform(Array const & lhs,
                           value_type scalar,
                           kernel_type kernel);
    static void transform_to(Array & destination,
                             Array const & lhs,
                             Array const & rhs,
                             kernel_type kernel);
    static void transform_to(Array & destination,
                             Array const & lhs,
                             value_type scalar,
                             kernel_type kernel);
    static void transform_into(Array & destination,
                               Array const & rhs,
                               kernel_type kernel);
    static void transform_into(Array & destination,
                               value_type scalar,
                               kernel_type kernel);

private:
    struct InnerLoopState
    {
        std::array<ssize_t, 3> m_stride;
        std::array<ssize_t, 3> m_offset;
    }; /* end struct InnerLoopState */

    static constexpr size_t OUTPUT_INDEX = 0;
    static constexpr size_t LHS_INDEX = 1;
    static constexpr size_t RHS_INDEX = 2;

    static void execute(ElementwisePlan const & plan,
                        Array & output,
                        Array const & lhs,
                        Array const & rhs,
                        kernel_type kernel);
    static void execute_inner_strided(
        ElementwisePlan const & plan,
        Array & output,
        Array const & lhs,
        Array const & rhs,
        kernel_type kernel);
    static void normalize_reversed_inner(
        size_t inner_size,
        InnerLoopState & state);
    static bool try_execute_unit_stride_inner(
        size_t inner_size,
        value_type * output_data,
        value_type const * lhs_data,
        value_type const * rhs_data,
        InnerLoopState const & state,
        kernel_type kernel);
    static bool try_execute_output_contiguous_inner(
        size_t inner_size,
        value_type * output_data,
        value_type const * lhs_data,
        value_type const * rhs_data,
        InnerLoopState const & state,
        kernel_type kernel);
    static bool try_execute_inplace_inner(
        size_t inner_size,
        value_type * output_data,
        value_type const * lhs_data,
        value_type const * rhs_data,
        InnerLoopState const & state,
        kernel_type kernel);
    static void execute_generic_inner(
        size_t inner_size,
        value_type * output_data,
        value_type const * lhs_data,
        value_type const * rhs_data,
        InnerLoopState const & state,
        kernel_type kernel);
    static void execute_scalar(ElementwisePlan const & plan,
                               Array & output,
                               Array const & lhs,
                               value_type scalar,
                               kernel_type kernel);
    static void execute_to(Array & destination,
                           Array const & lhs,
                           Array const & rhs,
                           kernel_type kernel);
    static void execute_to(Array & destination,
                           Array const & lhs,
                           value_type scalar,
                           kernel_type kernel);
    static void execute_into(Array & destination,
                             Array const & rhs,
                             kernel_type kernel);
    static bool storage_overlaps(Array const & lhs,
                                 Array const & rhs);
}; /* end class ElementwiseExecutor */

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::normalize_reversed_inner(
    size_t inner_size,
    InnerLoopState & state)
{
    if (state.m_stride[OUTPUT_INDEX] != -1 ||
        (state.m_stride[LHS_INDEX] != -1 &&
         state.m_stride[LHS_INDEX] != 0) ||
        (state.m_stride[RHS_INDEX] != -1 &&
         state.m_stride[RHS_INDEX] != 0))
    {
        return;
    }

    auto const shift =
        static_cast<ssize_t>(inner_size - 1);
    state.m_offset[OUTPUT_INDEX] -= shift;
    state.m_stride[OUTPUT_INDEX] = 1;
    if (state.m_stride[LHS_INDEX] == -1)
    {
        state.m_offset[LHS_INDEX] -= shift;
        state.m_stride[LHS_INDEX] = 1;
    }
    if (state.m_stride[RHS_INDEX] == -1)
    {
        state.m_offset[RHS_INDEX] -= shift;
        state.m_stride[RHS_INDEX] = 1;
    }
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
bool ElementwiseExecutor<Array, T, Kernel>::
    try_execute_unit_stride_inner(
        size_t inner_size,
        value_type * output_data,
        value_type const * lhs_data,
        value_type const * rhs_data,
        InnerLoopState const & state,
        kernel_type kernel)
{
    if (state.m_stride[OUTPUT_INDEX] == 1 &&
        state.m_stride[LHS_INDEX] == 1 &&
        state.m_stride[RHS_INDEX] == 1)
    {
        kernel_type::contiguous(
            output_data + state.m_offset[OUTPUT_INDEX],
            inner_size,
            lhs_data + state.m_offset[LHS_INDEX],
            rhs_data + state.m_offset[RHS_INDEX]);
        return true;
    }
    if (state.m_stride[OUTPUT_INDEX] == 1 &&
        state.m_stride[LHS_INDEX] == 1 &&
        state.m_stride[RHS_INDEX] == 0)
    {
        kernel_type::contiguous_scalar(
            output_data + state.m_offset[OUTPUT_INDEX],
            inner_size,
            lhs_data + state.m_offset[LHS_INDEX],
            rhs_data[state.m_offset[RHS_INDEX]]);
        return true;
    }
    if (state.m_stride[OUTPUT_INDEX] == 1 &&
        state.m_stride[LHS_INDEX] == 0 &&
        state.m_stride[RHS_INDEX] == 1)
    {
        kernel_type::contiguous_lhs_scalar(
            output_data + state.m_offset[OUTPUT_INDEX],
            inner_size,
            lhs_data[state.m_offset[LHS_INDEX]],
            rhs_data + state.m_offset[RHS_INDEX]);
        return true;
    }
    if (state.m_stride[OUTPUT_INDEX] == 1 &&
        state.m_stride[LHS_INDEX] == 0 &&
        state.m_stride[RHS_INDEX] == 0)
    {
        std::fill_n(
            output_data + state.m_offset[OUTPUT_INDEX],
            inner_size,
            kernel(
                lhs_data[state.m_offset[LHS_INDEX]],
                rhs_data[state.m_offset[RHS_INDEX]]));
        return true;
    }
    return false;
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
bool ElementwiseExecutor<Array, T, Kernel>::
    try_execute_output_contiguous_inner(
        size_t inner_size,
        value_type * output_data,
        value_type const * lhs_data,
        value_type const * rhs_data,
        InnerLoopState const & state,
        kernel_type kernel)
{
    if (state.m_stride[OUTPUT_INDEX] != 1)
    {
        return false;
    }

    ssize_t lhs_offset = state.m_offset[LHS_INDEX];
    ssize_t rhs_offset = state.m_offset[RHS_INDEX];
    value_type * selected_output =
        output_data + state.m_offset[OUTPUT_INDEX];
    if (state.m_stride[RHS_INDEX] == 0)
    {
        value_type const rhs_value = rhs_data[rhs_offset];
        for (size_t index = 0; index < inner_size; ++index)
        {
            selected_output[index] =
                kernel(lhs_data[lhs_offset], rhs_value);
            lhs_offset += state.m_stride[LHS_INDEX];
        }
        return true;
    }
    if (state.m_stride[LHS_INDEX] == 0)
    {
        value_type const lhs_value = lhs_data[lhs_offset];
        for (size_t index = 0; index < inner_size; ++index)
        {
            selected_output[index] =
                kernel(lhs_value, rhs_data[rhs_offset]);
            rhs_offset += state.m_stride[RHS_INDEX];
        }
        return true;
    }
    if (state.m_stride[LHS_INDEX] == 1)
    {
        value_type const * selected_lhs = lhs_data + lhs_offset;
        for (size_t index = 0; index < inner_size; ++index)
        {
            selected_output[index] =
                kernel(selected_lhs[index], rhs_data[rhs_offset]);
            rhs_offset += state.m_stride[RHS_INDEX];
        }
        return true;
    }
    if (state.m_stride[RHS_INDEX] == 1)
    {
        value_type const * selected_rhs = rhs_data + rhs_offset;
        for (size_t index = 0; index < inner_size; ++index)
        {
            selected_output[index] =
                kernel(lhs_data[lhs_offset], selected_rhs[index]);
            lhs_offset += state.m_stride[LHS_INDEX];
        }
        return true;
    }
    return false;
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
bool ElementwiseExecutor<Array, T, Kernel>::
    try_execute_inplace_inner(
        size_t inner_size,
        value_type * output_data,
        value_type const * lhs_data,
        value_type const * rhs_data,
        InnerLoopState const & state,
        kernel_type kernel)
{
    if (state.m_stride[RHS_INDEX] == 0 &&
        output_data == lhs_data &&
        state.m_offset[OUTPUT_INDEX] ==
            state.m_offset[LHS_INDEX] &&
        state.m_stride[OUTPUT_INDEX] ==
            state.m_stride[LHS_INDEX])
    {
        value_type const rhs_value =
            rhs_data[state.m_offset[RHS_INDEX]];
        ssize_t output_offset = state.m_offset[OUTPUT_INDEX];
        for (size_t index = 0; index < inner_size; ++index)
        {
            output_data[output_offset] =
                kernel(output_data[output_offset], rhs_value);
            output_offset += state.m_stride[OUTPUT_INDEX];
        }
        return true;
    }
    if (state.m_stride[LHS_INDEX] == 0 &&
        output_data == rhs_data &&
        state.m_offset[OUTPUT_INDEX] ==
            state.m_offset[RHS_INDEX] &&
        state.m_stride[OUTPUT_INDEX] ==
            state.m_stride[RHS_INDEX])
    {
        value_type const lhs_value =
            lhs_data[state.m_offset[LHS_INDEX]];
        ssize_t output_offset = state.m_offset[OUTPUT_INDEX];
        for (size_t index = 0; index < inner_size; ++index)
        {
            output_data[output_offset] =
                kernel(lhs_value, output_data[output_offset]);
            output_offset += state.m_stride[OUTPUT_INDEX];
        }
        return true;
    }
    return false;
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::execute_generic_inner(
    size_t inner_size,
    value_type * output_data,
    value_type const * lhs_data,
    value_type const * rhs_data,
    InnerLoopState const & state,
    kernel_type kernel)
{
    ssize_t output_offset = state.m_offset[OUTPUT_INDEX];
    ssize_t lhs_offset = state.m_offset[LHS_INDEX];
    ssize_t rhs_offset = state.m_offset[RHS_INDEX];
    for (size_t index = 0; index < inner_size; ++index)
    {
        output_data[output_offset] =
            kernel(lhs_data[lhs_offset], rhs_data[rhs_offset]);
        output_offset += state.m_stride[OUTPUT_INDEX];
        lhs_offset += state.m_stride[LHS_INDEX];
        rhs_offset += state.m_stride[RHS_INDEX];
    }
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::execute_inner_strided(
    ElementwisePlan const & plan,
    Array & output,
    Array const & lhs,
    Array const & rhs,
    kernel_type kernel)
{
    value_type * output_data = output.logical_data();
    value_type const * lhs_data = lhs.logical_data();
    value_type const * rhs_data = rhs.logical_data();
    small_vector<OperandMapping> const mappings{
        plan.output(), plan.input(0), plan.input(1)};
    InnerLoopPlan const inner(
        plan.domain(), mappings, plan.inner_axis());
    MappedOffsetCursor cursor(
        inner.outer(), inner.outer_mappings());
    for (; cursor; cursor.advance())
    {
        InnerLoopState state{
            {inner.stride(OUTPUT_INDEX),
             inner.stride(LHS_INDEX),
             inner.stride(RHS_INDEX)},
            {cursor.offset(OUTPUT_INDEX),
             cursor.offset(LHS_INDEX),
             cursor.offset(RHS_INDEX)}};
        normalize_reversed_inner(inner.size(), state);
        if (try_execute_unit_stride_inner(
                inner.size(),
                output_data,
                lhs_data,
                rhs_data,
                state,
                kernel))
        {
            continue;
        }
        if (try_execute_output_contiguous_inner(
                inner.size(),
                output_data,
                lhs_data,
                rhs_data,
                state,
                kernel))
        {
            continue;
        }
        if (try_execute_inplace_inner(
                inner.size(),
                output_data,
                lhs_data,
                rhs_data,
                state,
                kernel))
        {
            continue;
        }
        execute_generic_inner(
            inner.size(),
            output_data,
            lhs_data,
            rhs_data,
            state,
            kernel);
    }
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::execute(
    ElementwisePlan const & plan,
    Array & output,
    Array const & lhs,
    Array const & rhs,
    kernel_type kernel)
{
    if (plan.domain().empty())
    {
        return;
    }

    value_type * output_data = output.logical_data();
    value_type const * lhs_data = lhs.logical_data();
    value_type const * rhs_data = rhs.logical_data();
    OperandMapping const & lhs_mapping = plan.input(0);
    OperandMapping const & rhs_mapping = plan.input(1);
    bool const output_matches_lhs =
        std::ranges::equal(
            plan.output().strides(), lhs_mapping.strides()) &&
        plan.output().is_dense(plan.domain());
    bool const output_matches_rhs =
        std::ranges::equal(
            plan.output().strides(), rhs_mapping.strides()) &&
        plan.output().is_dense(plan.domain());
    bool const lhs_is_constant =
        lhs_mapping.is_constant(plan.domain());
    bool const rhs_is_constant =
        rhs_mapping.is_constant(plan.domain());
    if (output_matches_lhs && rhs_is_constant)
    {
        kernel_type::contiguous_scalar(
            output_data +
                plan.output().span(plan.domain()).minimum(),
            plan.domain().size(),
            lhs_data +
                lhs_mapping.span(plan.domain()).minimum(),
            rhs_data[rhs_mapping.base_offset()]);
        return;
    }
    if (output_matches_rhs && lhs_is_constant)
    {
        kernel_type::contiguous_lhs_scalar(
            output_data +
                plan.output().span(plan.domain()).minimum(),
            plan.domain().size(),
            lhs_data[lhs_mapping.base_offset()],
            rhs_data +
                rhs_mapping.span(plan.domain()).minimum());
        return;
    }

    if (plan.route() == ExecutionRoute::contiguous)
    {
        ssize_t const output_offset =
            plan.output().span(plan.domain()).minimum();
        ssize_t const lhs_offset =
            lhs_mapping.span(plan.domain()).minimum();
        ssize_t const rhs_offset =
            rhs_mapping.span(plan.domain()).minimum();
        kernel_type::contiguous(
            output_data + output_offset,
            plan.domain().size(),
            lhs_data + lhs_offset,
            rhs_data + rhs_offset);
        return;
    }

    if (plan.route() == ExecutionRoute::inner_strided)
    {
        execute_inner_strided(
            plan, output, lhs, rhs, kernel);
        return;
    }

    small_vector<OperandMapping> const mappings{
        plan.output(), lhs_mapping, rhs_mapping};
    for (MappedOffsetCursor cursor(plan.domain(), mappings);
         cursor;
         cursor.advance())
    {
        output_data[cursor.offset(0)] = kernel(
            lhs_data[cursor.offset(1)],
            rhs_data[cursor.offset(2)]);
    }
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::execute_scalar(
    ElementwisePlan const & plan,
    Array & output,
    Array const & lhs,
    value_type scalar,
    kernel_type kernel)
{
    value_type * output_data = output.logical_data();
    value_type const * lhs_data = lhs.logical_data();
    OperandMapping const & lhs_mapping = plan.input(0);
    if (plan.route() == ExecutionRoute::contiguous)
    {
        output_data +=
            plan.output().span(plan.domain()).minimum();
        lhs_data += lhs_mapping.span(plan.domain()).minimum();
        kernel_type::contiguous_scalar(
            output_data, plan.domain().size(), lhs_data, scalar);
        return;
    }

    small_vector<OperandMapping> const mappings{
        plan.output(), lhs_mapping};
    if (plan.route() == ExecutionRoute::inner_strided)
    {
        InnerLoopPlan const inner(
            plan.domain(), mappings, plan.inner_axis());
        ssize_t const output_stride = inner.stride(0);
        ssize_t const lhs_stride = inner.stride(1);
        MappedOffsetCursor cursor(
            inner.outer(), inner.outer_mappings());
        for (; cursor; cursor.advance())
        {
            ssize_t selected_output_stride = output_stride;
            ssize_t selected_lhs_stride = lhs_stride;
            ssize_t output_offset = cursor.offset(0);
            ssize_t lhs_offset = cursor.offset(1);
            if (selected_output_stride == -1 &&
                (selected_lhs_stride == -1 ||
                 selected_lhs_stride == 0))
            {
                auto const shift =
                    static_cast<ssize_t>(inner.size() - 1);
                output_offset -= shift;
                selected_output_stride = 1;
                if (selected_lhs_stride == -1)
                {
                    lhs_offset -= shift;
                    selected_lhs_stride = 1;
                }
            }

            if (selected_output_stride == 1 &&
                selected_lhs_stride == 1)
            {
                kernel_type::contiguous_scalar(
                    output_data + output_offset,
                    inner.size(),
                    lhs_data + lhs_offset,
                    scalar);
                continue;
            }
            if (selected_output_stride == 1 &&
                selected_lhs_stride == 0)
            {
                std::fill_n(
                    output_data + output_offset,
                    inner.size(),
                    kernel(lhs_data[lhs_offset], scalar));
                continue;
            }

            for (size_t index = 0; index < inner.size(); ++index)
            {
                output_data[output_offset] =
                    kernel(lhs_data[lhs_offset], scalar);
                output_offset += selected_output_stride;
                lhs_offset += selected_lhs_stride;
            }
        }
        return;
    }

    for (MappedOffsetCursor cursor(plan.domain(), mappings);
         cursor;
         cursor.advance())
    {
        output_data[cursor.offset(0)] =
            kernel(lhs_data[cursor.offset(1)], scalar);
    }
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
Array ElementwiseExecutor<Array, T, Kernel>::transform(
    Array const & lhs, Array const & rhs, kernel_type kernel)
{
    bool const shapes_match =
        std::ranges::equal(lhs.shape(), rhs.shape());
    if (shapes_match &&
        lhs.is_c_contiguous() &&
        rhs.is_c_contiguous())
    {
        Array output(lhs.shape());
        kernel_type::contiguous(
            output.logical_data(),
            output.size(),
            lhs.logical_data(),
            rhs.logical_data());
        return output;
    }

    if (shapes_match &&
        std::ranges::equal(lhs.stride(), rhs.stride()))
    {
        LoopDomain const domain(lhs.shape());
        OperandMapping const mapping(lhs.stride());
        if (mapping.is_dense(domain))
        {
            auto output =
                allocate_layout<Array>(lhs.shape(), lhs.stride());
            execute_to(output, lhs, rhs, kernel);
            return output;
        }
    }

    if (shapes_match && lhs.shape().size() == 1)
    {
        Array output(lhs.shape());
        execute_to(output, lhs, rhs, kernel);
        return output;
    }

    LoopDomain const result_domain(
        ElementwisePlan::broadcast_shape(lhs, rhs));
    shape_type const & result_shape = result_domain.shape();
    OperandMapping const lhs_mapping(lhs.stride());
    OperandMapping const rhs_mapping(rhs.stride());
    bool const result_matches_lhs =
        std::ranges::equal(result_shape, lhs.shape());
    bool const result_matches_rhs =
        std::ranges::equal(result_shape, rhs.shape());
    bool const preserve_layout =
        result_matches_lhs &&
        result_matches_rhs &&
        std::ranges::equal(lhs.stride(), rhs.stride()) &&
        lhs_mapping.is_dense(result_domain);
    bool const preserve_lhs_layout =
        result_matches_lhs &&
        !result_matches_rhs &&
        lhs_mapping.is_dense(result_domain);
    bool const preserve_rhs_layout =
        result_matches_rhs &&
        !result_matches_lhs &&
        rhs_mapping.is_dense(result_domain);
    auto output = [&]() -> Array
    {
        if (preserve_layout || preserve_lhs_layout)
        {
            return allocate_layout<Array>(
                result_shape, lhs.stride());
        }
        if (preserve_rhs_layout)
        {
            return allocate_layout<Array>(
                result_shape, rhs.stride());
        }
        return Array(result_shape);
    }();
    execute_to(output, lhs, rhs, kernel);
    return output;
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
Array ElementwiseExecutor<Array, T, Kernel>::transform(
    Array const & lhs, value_type scalar, kernel_type kernel)
{
    if (lhs.is_c_contiguous())
    {
        Array output(lhs.shape());
        execute_to(output, lhs, scalar, kernel);
        return output;
    }

    LoopDomain const domain(lhs.shape());
    OperandMapping const mapping(lhs.stride());
    if (mapping.is_dense(domain))
    {
        auto output =
            allocate_layout<Array>(lhs.shape(), lhs.stride());
        execute_to(output, lhs, scalar, kernel);
        return output;
    }

    Array output(lhs.shape());
    execute_to(output, lhs, scalar, kernel);
    return output;
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::transform_to(
    Array & destination,
    Array const & lhs,
    Array const & rhs,
    kernel_type kernel)
{
    shape_type const result_shape =
        ElementwisePlan::broadcast_shape(lhs, rhs);
    if (!std::ranges::equal(destination.shape(), result_shape))
    {
        throw std::invalid_argument(
            "elementwise output shape does not match result shape");
    }

    bool const destination_overlaps_lhs =
        storage_overlaps(destination, lhs);
    bool const destination_overlaps_rhs =
        storage_overlaps(destination, rhs);
    if (destination_overlaps_lhs && destination_overlaps_rhs)
    {
        Array const safe_lhs(lhs);
        Array const safe_rhs(rhs);
        execute_to(destination, safe_lhs, safe_rhs, kernel);
        return;
    }
    if (destination_overlaps_lhs)
    {
        Array const safe_lhs(lhs);
        execute_to(destination, safe_lhs, rhs, kernel);
        return;
    }
    if (destination_overlaps_rhs)
    {
        Array const safe_rhs(rhs);
        execute_to(destination, lhs, safe_rhs, kernel);
        return;
    }
    execute_to(destination, lhs, rhs, kernel);
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::transform_to(
    Array & destination,
    Array const & lhs,
    value_type scalar,
    kernel_type kernel)
{
    if (!std::ranges::equal(destination.shape(), lhs.shape()))
    {
        throw std::invalid_argument(
            "scalar elementwise output shape mismatch");
    }

    if (storage_overlaps(destination, lhs))
    {
        Array const safe_lhs(lhs);
        execute_to(destination, safe_lhs, scalar, kernel);
        return;
    }
    execute_to(destination, lhs, scalar, kernel);
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::execute_to(
    Array & destination,
    Array const & lhs,
    Array const & rhs,
    kernel_type kernel)
{
    bool const shapes_match =
        std::ranges::equal(lhs.shape(), rhs.shape());
    if (shapes_match &&
        destination.is_c_contiguous() &&
        lhs.is_c_contiguous() &&
        rhs.is_c_contiguous())
    {
        kernel_type::contiguous(
            destination.logical_data(),
            destination.size(),
            lhs.logical_data(),
            rhs.logical_data());
        return;
    }

    if (shapes_match &&
        std::ranges::equal(destination.stride(), lhs.stride()) &&
        std::ranges::equal(lhs.stride(), rhs.stride()))
    {
        LoopDomain const domain(lhs.shape());
        OperandMapping const mapping(lhs.stride());
        if (mapping.is_dense(domain))
        {
            ssize_t const offset = mapping.span(domain).minimum();
            kernel_type::contiguous(
                destination.logical_data() + offset,
                destination.size(),
                lhs.logical_data() + offset,
                rhs.logical_data() + offset);
            return;
        }
    }

    if (shapes_match &&
        lhs.shape().size() == 1)
    {
        value_type * destination_data =
            destination.logical_data();
        value_type const * lhs_data = lhs.logical_data();
        value_type const * rhs_data = rhs.logical_data();
        size_t const inner_size = destination.size();
        InnerLoopState state{
            {destination.stride()[0],
             lhs.stride()[0],
             rhs.stride()[0]},
            {0, 0, 0}};
        normalize_reversed_inner(inner_size, state);
        if (try_execute_unit_stride_inner(
                inner_size,
                destination_data,
                lhs_data,
                rhs_data,
                state,
                kernel))
        {
            return;
        }
        if (try_execute_output_contiguous_inner(
                inner_size,
                destination_data,
                lhs_data,
                rhs_data,
                state,
                kernel))
        {
            return;
        }
        execute_generic_inner(
            inner_size,
            destination_data,
            lhs_data,
            rhs_data,
            state,
            kernel);
        return;
    }

    ElementwisePlan const plan =
        ElementwisePlan::make(destination, lhs, rhs);
    execute(plan, destination, lhs, rhs, kernel);
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::execute_to(
    Array & destination,
    Array const & lhs,
    value_type scalar,
    kernel_type kernel)
{
    if (destination.is_c_contiguous() &&
        lhs.is_c_contiguous())
    {
        kernel_type::contiguous_scalar(
            destination.logical_data(),
            destination.size(),
            lhs.logical_data(),
            scalar);
        return;
    }

    if (std::ranges::equal(
            destination.stride(), lhs.stride()))
    {
        LoopDomain const domain(lhs.shape());
        OperandMapping const mapping(lhs.stride());
        if (mapping.is_dense(domain))
        {
            ssize_t const offset = mapping.span(domain).minimum();
            kernel_type::contiguous_scalar(
                destination.logical_data() + offset,
                destination.size(),
                lhs.logical_data() + offset,
                scalar);
            return;
        }
    }

    ElementwisePlan const plan =
        ElementwisePlan::make_scalar(destination, lhs);
    execute_scalar(
        plan, destination, lhs, scalar, kernel);
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::transform_into(
    Array & destination,
    Array const & rhs,
    kernel_type kernel)
{
    bool const exact_alias =
        destination.logical_data() == rhs.logical_data() &&
        std::ranges::equal(
            destination.shape(), rhs.shape()) &&
        std::ranges::equal(
            destination.stride(), rhs.stride());
    if (storage_overlaps(destination, rhs) && !exact_alias)
    {
        Array const safe_rhs(rhs);
        execute_into(destination, safe_rhs, kernel);
        return;
    }
    execute_into(destination, rhs, kernel);
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
bool ElementwiseExecutor<Array, T, Kernel>::storage_overlaps(
    Array const & lhs, Array const & rhs)
{
    if (lhs.size() == 0 || rhs.size() == 0)
    {
        return false;
    }

    auto const lhs_storage_begin =
        std::bit_cast<std::uintptr_t>(lhs.data());
    auto const lhs_storage_end =
        lhs_storage_begin + lhs.buffer().nbytes();
    auto const rhs_storage_begin =
        std::bit_cast<std::uintptr_t>(rhs.data());
    auto const rhs_storage_end =
        rhs_storage_begin + rhs.buffer().nbytes();
    if (lhs_storage_begin >= rhs_storage_end ||
        rhs_storage_begin >= lhs_storage_end)
    {
        return false;
    }

    MappingSpan const lhs_span =
        OperandMapping::span(lhs.shape(), lhs.stride());
    MappingSpan const rhs_span =
        OperandMapping::span(rhs.shape(), rhs.stride());
    auto const lhs_begin =
        std::bit_cast<std::uintptr_t>(
            lhs.logical_data() + lhs_span.minimum());
    auto const lhs_end =
        std::bit_cast<std::uintptr_t>(
            lhs.logical_data() + lhs_span.maximum() + 1);
    auto const rhs_begin =
        std::bit_cast<std::uintptr_t>(
            rhs.logical_data() + rhs_span.minimum());
    auto const rhs_end =
        std::bit_cast<std::uintptr_t>(
            rhs.logical_data() + rhs_span.maximum() + 1);
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::execute_into(
    Array & destination,
    Array const & rhs,
    kernel_type kernel)
{
    if (std::ranges::equal(
            destination.shape(), rhs.shape()) &&
        std::ranges::equal(
            destination.stride(), rhs.stride()) &&
        ((destination.is_c_contiguous() &&
          rhs.is_c_contiguous()) ||
         (destination.is_f_contiguous() &&
          rhs.is_f_contiguous())))
    {
        kernel_type::inplace(
            destination.logical_data(),
            destination.size(),
            rhs.logical_data());
        return;
    }

    if (std::ranges::equal(
            destination.shape(), rhs.shape()) &&
        std::ranges::equal(
            destination.stride(), rhs.stride()) &&
        OperandMapping::is_dense(
            destination.shape(), destination.stride()))
    {
        ssize_t const offset =
            OperandMapping::span(
                destination.shape(), destination.stride())
                .minimum();
        kernel_type::inplace(
            destination.logical_data() + offset,
            destination.size(),
            rhs.logical_data() + offset);
        return;
    }

    if (std::ranges::equal(
            destination.shape(), rhs.shape()) &&
        destination.shape().size() == 1)
    {
        value_type * destination_data =
            destination.logical_data();
        value_type const * rhs_data = rhs.logical_data();
        ssize_t destination_offset = 0;
        ssize_t rhs_offset = 0;
        ssize_t const destination_stride =
            destination.stride()[0];
        ssize_t const rhs_stride = rhs.stride()[0];
        for (size_t index = 0;
             index < destination.size();
             ++index)
        {
            destination_data[destination_offset] = kernel(
                destination_data[destination_offset],
                rhs_data[rhs_offset]);
            destination_offset += destination_stride;
            rhs_offset += rhs_stride;
        }
        return;
    }

    ElementwisePlan const plan =
        ElementwisePlan::make(destination, destination, rhs);
    execute(plan, destination, destination, rhs, kernel);
}

template <typename Array, typename T, typename Kernel>
requires ArithmeticKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::transform_into(
    Array & destination,
    value_type scalar,
    kernel_type kernel)
{
    if (destination.is_c_contiguous() ||
        destination.is_f_contiguous())
    {
        kernel_type::scalar(
            destination.logical_data(),
            destination.size(),
            scalar);
        return;
    }

    if (OperandMapping::is_dense(
            destination.shape(), destination.stride()))
    {
        value_type * data =
            destination.logical_data() +
            OperandMapping::span(
                destination.shape(), destination.stride())
                .minimum();
        kernel_type::scalar(data, destination.size(), scalar);
        return;
    }

    ElementwisePlan const plan =
        ElementwisePlan::make_scalar(destination, destination);
    execute_scalar(
        plan, destination, destination, scalar, kernel);
}

} /* end namespace elementwise */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
