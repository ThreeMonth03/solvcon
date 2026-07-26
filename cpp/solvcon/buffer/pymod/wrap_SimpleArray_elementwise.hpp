#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <type_traits>

namespace solvcon
{

namespace python
{

namespace detail
{

template <typename Value>
bool is_fast_elementwise_scalar(pybind11::handle operand)
{
    if constexpr (std::is_floating_point_v<Value>)
    {
        return PyFloat_Check(operand.ptr());
    }
    return false;
}

template <typename Value>
Value cast_elementwise_scalar(pybind11::handle operand)
{
    if constexpr (std::is_floating_point_v<Value>)
    {
        if (PyFloat_Check(operand.ptr()))
        {
            return static_cast<Value>(
                PyFloat_AS_DOUBLE(operand.ptr()));
        }
    }
    return operand.cast<Value>();
}

template <typename Array,
          typename Value,
          typename Elementwise,
          typename Wrapper>
Wrapper & bind_simple_array_elementwise(Wrapper & wrapper)
{
    namespace py = pybind11;

    auto bind_transform =
        [&wrapper](char const * name,
                   auto array_function,
                   auto scalar_function)
    {
        wrapper.def(
            name,
            [array_function, scalar_function](
                Array const & self,
                py::object const & operand)
            {
                if (is_fast_elementwise_scalar<Value>(operand))
                {
                    return scalar_function(
                        self,
                        cast_elementwise_scalar<Value>(operand));
                }
                if (py::isinstance<Array>(operand))
                {
                    return array_function(
                        self,
                        operand.cast<Array const &>());
                }
                return scalar_function(
                    self,
                    cast_elementwise_scalar<Value>(operand));
            });
    };
    auto bind_transform_to =
        [&wrapper](char const * name,
                   auto array_function,
                   auto scalar_function)
    {
        wrapper.def(
            name,
            [array_function, scalar_function](
                Array const & self,
                py::object const & operand,
                Array & destination)
            {
                if (is_fast_elementwise_scalar<Value>(operand))
                {
                    scalar_function(
                        self,
                        cast_elementwise_scalar<Value>(operand),
                        destination);
                    return;
                }
                if (py::isinstance<Array>(operand))
                {
                    array_function(
                        self,
                        operand.cast<Array const &>(),
                        destination);
                    return;
                }
                scalar_function(
                    self,
                    cast_elementwise_scalar<Value>(operand),
                    destination);
            });
    };
    auto bind_transform_into =
        [&wrapper](char const * name,
                   auto array_function,
                   auto scalar_function)
    {
        wrapper.def(
            name,
            [array_function, scalar_function](
                Array & self,
                py::object const & operand)
            {
                if (is_fast_elementwise_scalar<Value>(operand))
                {
                    scalar_function(
                        self,
                        cast_elementwise_scalar<Value>(operand));
                    return;
                }
                if (py::isinstance<Array>(operand))
                {
                    array_function(
                        self,
                        operand.cast<Array const &>());
                    return;
                }
                scalar_function(
                    self,
                    cast_elementwise_scalar<Value>(operand));
            });
    };

    bind_transform(
        "_planned_add",
        py::overload_cast<
            Array const &,
            Array const &>(&Elementwise::planned_add),
        py::overload_cast<
            Array const &,
            Value>(&Elementwise::planned_add));
    bind_transform(
        "_planned_sub",
        py::overload_cast<
            Array const &,
            Array const &>(&Elementwise::planned_sub),
        py::overload_cast<
            Array const &,
            Value>(&Elementwise::planned_sub));
    bind_transform(
        "_planned_mul",
        py::overload_cast<
            Array const &,
            Array const &>(&Elementwise::planned_mul),
        py::overload_cast<
            Array const &,
            Value>(&Elementwise::planned_mul));
    bind_transform(
        "_planned_div",
        py::overload_cast<
            Array const &,
            Array const &>(&Elementwise::planned_div),
        py::overload_cast<
            Array const &,
            Value>(&Elementwise::planned_div));

    bind_transform_to(
        "_planned_add_to",
        py::overload_cast<
            Array const &,
            Array const &,
            Array &>(&Elementwise::planned_add_to),
        py::overload_cast<
            Array const &,
            Value,
            Array &>(&Elementwise::planned_add_to));
    bind_transform_to(
        "_planned_sub_to",
        py::overload_cast<
            Array const &,
            Array const &,
            Array &>(&Elementwise::planned_sub_to),
        py::overload_cast<
            Array const &,
            Value,
            Array &>(&Elementwise::planned_sub_to));
    bind_transform_to(
        "_planned_mul_to",
        py::overload_cast<
            Array const &,
            Array const &,
            Array &>(&Elementwise::planned_mul_to),
        py::overload_cast<
            Array const &,
            Value,
            Array &>(&Elementwise::planned_mul_to));
    bind_transform_to(
        "_planned_div_to",
        py::overload_cast<
            Array const &,
            Array const &,
            Array &>(&Elementwise::planned_div_to),
        py::overload_cast<
            Array const &,
            Value,
            Array &>(&Elementwise::planned_div_to));

    bind_transform_into(
        "_planned_iadd",
        py::overload_cast<
            Array &,
            Array const &>(&Elementwise::planned_iadd),
        py::overload_cast<
            Array &,
            Value>(&Elementwise::planned_iadd));
    bind_transform_into(
        "_planned_isub",
        py::overload_cast<
            Array &,
            Array const &>(&Elementwise::planned_isub),
        py::overload_cast<
            Array &,
            Value>(&Elementwise::planned_isub));
    bind_transform_into(
        "_planned_imul",
        py::overload_cast<
            Array &,
            Array const &>(&Elementwise::planned_imul),
        py::overload_cast<
            Array &,
            Value>(&Elementwise::planned_imul));
    bind_transform_into(
        "_planned_idiv",
        py::overload_cast<
            Array &,
            Array const &>(&Elementwise::planned_idiv),
        py::overload_cast<
            Array &,
            Value>(&Elementwise::planned_idiv));
    return wrapper;
}

} /* end namespace detail */

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
