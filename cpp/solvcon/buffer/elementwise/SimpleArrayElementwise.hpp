#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/elementwise/executor.hpp>

#include <format>
#include <stdexcept>
#include <type_traits>

namespace solvcon
{

namespace detail
{

template <typename Array, typename T>
class SimpleArrayElementwise
{
public:
    using value_type = T;

    static Array planned_add(Array const & self,
                             Array const & other);
    static Array planned_add(Array const & self, value_type scalar);
    static Array planned_sub(Array const & self,
                             Array const & other);
    static Array planned_sub(Array const & self, value_type scalar);
    static Array planned_mul(Array const & self,
                             Array const & other);
    static Array planned_mul(Array const & self, value_type scalar);
    static Array planned_div(Array const & self,
                             Array const & other);
    static Array planned_div(Array const & self, value_type scalar);

    static void planned_add_to(Array const & self,
                               Array const & other,
                               Array & destination);
    static void planned_add_to(Array const & self,
                               value_type scalar,
                               Array & destination);
    static void planned_sub_to(Array const & self,
                               Array const & other,
                               Array & destination);
    static void planned_sub_to(Array const & self,
                               value_type scalar,
                               Array & destination);
    static void planned_mul_to(Array const & self,
                               Array const & other,
                               Array & destination);
    static void planned_mul_to(Array const & self,
                               value_type scalar,
                               Array & destination);
    static void planned_div_to(Array const & self,
                               Array const & other,
                               Array & destination);
    static void planned_div_to(Array const & self,
                               value_type scalar,
                               Array & destination);

    static void planned_iadd(Array & self, Array const & other);
    static void planned_iadd(Array & self, value_type scalar);
    static void planned_isub(Array & self, Array const & other);
    static void planned_isub(Array & self, value_type scalar);
    static void planned_imul(Array & self, Array const & other);
    static void planned_imul(Array & self, value_type scalar);
    static void planned_idiv(Array & self, Array const & other);
    static void planned_idiv(Array & self, value_type scalar);

private:
    using add_executor_type = elementwise::ElementwiseExecutor<
        Array,
        value_type,
        elementwise::AddKernel<value_type>>;
    using subtract_executor_type = elementwise::ElementwiseExecutor<
        Array,
        value_type,
        elementwise::SubtractKernel<value_type>>;
    using multiply_executor_type = elementwise::ElementwiseExecutor<
        Array,
        value_type,
        elementwise::MultiplyKernel<value_type>>;
    using divide_executor_type = elementwise::ElementwiseExecutor<
        Array,
        value_type,
        elementwise::DivideKernel<value_type>>;

    static void reject_boolean(char const * operation);
}; /* end class SimpleArrayElementwise */

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::reject_boolean(
    char const * operation)
{
    if constexpr (
        std::is_same_v<bool, std::remove_const_t<value_type>>)
    {
        throw std::runtime_error(std::format(
            "SimpleArray<bool>::planned_{}(): "
            "unsupported operation",
            operation));
    }
}

template <typename Array, typename T>
Array SimpleArrayElementwise<Array, T>::planned_add(
    Array const & self, Array const & other)
{
    return add_executor_type::transform(
        self, other, elementwise::AddKernel<value_type>{});
}

template <typename Array, typename T>
Array SimpleArrayElementwise<Array, T>::planned_add(
    Array const & self, value_type scalar)
{
    return add_executor_type::transform(
        self, scalar, elementwise::AddKernel<value_type>{});
}

template <typename Array, typename T>
Array SimpleArrayElementwise<Array, T>::planned_sub(
    Array const & self, Array const & other)
{
    reject_boolean("sub");
    return subtract_executor_type::transform(
        self, other, elementwise::SubtractKernel<value_type>{});
}

template <typename Array, typename T>
Array SimpleArrayElementwise<Array, T>::planned_sub(
    Array const & self, value_type scalar)
{
    reject_boolean("sub");
    return subtract_executor_type::transform(
        self, scalar, elementwise::SubtractKernel<value_type>{});
}

template <typename Array, typename T>
Array SimpleArrayElementwise<Array, T>::planned_mul(
    Array const & self, Array const & other)
{
    return multiply_executor_type::transform(
        self, other, elementwise::MultiplyKernel<value_type>{});
}

template <typename Array, typename T>
Array SimpleArrayElementwise<Array, T>::planned_mul(
    Array const & self, value_type scalar)
{
    return multiply_executor_type::transform(
        self, scalar, elementwise::MultiplyKernel<value_type>{});
}

template <typename Array, typename T>
Array SimpleArrayElementwise<Array, T>::planned_div(
    Array const & self, Array const & other)
{
    reject_boolean("div");
    return divide_executor_type::transform(
        self, other, elementwise::DivideKernel<value_type>{});
}

template <typename Array, typename T>
Array SimpleArrayElementwise<Array, T>::planned_div(
    Array const & self, value_type scalar)
{
    reject_boolean("div");
    return divide_executor_type::transform(
        self, scalar, elementwise::DivideKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_add_to(
    Array const & self,
    Array const & other,
    Array & destination)
{
    add_executor_type::transform_to(
        destination,
        self,
        other,
        elementwise::AddKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_add_to(
    Array const & self,
    value_type scalar,
    Array & destination)
{
    add_executor_type::transform_to(
        destination,
        self,
        scalar,
        elementwise::AddKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_sub_to(
    Array const & self,
    Array const & other,
    Array & destination)
{
    reject_boolean("sub_to");
    subtract_executor_type::transform_to(
        destination,
        self,
        other,
        elementwise::SubtractKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_sub_to(
    Array const & self,
    value_type scalar,
    Array & destination)
{
    reject_boolean("sub_to");
    subtract_executor_type::transform_to(
        destination,
        self,
        scalar,
        elementwise::SubtractKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_mul_to(
    Array const & self,
    Array const & other,
    Array & destination)
{
    multiply_executor_type::transform_to(
        destination,
        self,
        other,
        elementwise::MultiplyKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_mul_to(
    Array const & self,
    value_type scalar,
    Array & destination)
{
    multiply_executor_type::transform_to(
        destination,
        self,
        scalar,
        elementwise::MultiplyKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_div_to(
    Array const & self,
    Array const & other,
    Array & destination)
{
    reject_boolean("div_to");
    divide_executor_type::transform_to(
        destination,
        self,
        other,
        elementwise::DivideKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_div_to(
    Array const & self,
    value_type scalar,
    Array & destination)
{
    reject_boolean("div_to");
    divide_executor_type::transform_to(
        destination,
        self,
        scalar,
        elementwise::DivideKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_iadd(
    Array & self, Array const & other)
{
    add_executor_type::transform_into(
        self, other, elementwise::AddKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_iadd(
    Array & self, value_type scalar)
{
    add_executor_type::transform_into(
        self, scalar, elementwise::AddKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_isub(
    Array & self, Array const & other)
{
    reject_boolean("isub");
    subtract_executor_type::transform_into(
        self, other, elementwise::SubtractKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_isub(
    Array & self, value_type scalar)
{
    reject_boolean("isub");
    subtract_executor_type::transform_into(
        self, scalar, elementwise::SubtractKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_imul(
    Array & self, Array const & other)
{
    multiply_executor_type::transform_into(
        self, other, elementwise::MultiplyKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_imul(
    Array & self, value_type scalar)
{
    multiply_executor_type::transform_into(
        self, scalar, elementwise::MultiplyKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_idiv(
    Array & self, Array const & other)
{
    reject_boolean("idiv");
    divide_executor_type::transform_into(
        self, other, elementwise::DivideKernel<value_type>{});
}

template <typename Array, typename T>
void SimpleArrayElementwise<Array, T>::planned_idiv(
    Array & self, value_type scalar)
{
    reject_boolean("idiv");
    divide_executor_type::transform_into(
        self, scalar, elementwise::DivideKernel<value_type>{});
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
