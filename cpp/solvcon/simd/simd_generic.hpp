#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <concepts>
#include <functional>

namespace solvcon
{

namespace simd
{

namespace generic
{

template <typename T>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
    for (T const * ptr = start; ptr < end; ++ptr)
    {
        if (*ptr < min_val || *ptr > max_val)
        {
            return ptr;
        }
    }
    return nullptr;
}

template <typename T, std::invocable<T, T> ScalarOp>
inline void transform_binary(T * dest, T const * dest_end, T const * src1, T const * src2, ScalarOp scalar_op)
{
    T * ptr = dest;
    while (ptr < dest_end)
    {
        *ptr = scalar_op(*src1, *src2);
        ++ptr;
        ++src1;
        ++src2;
    }
}

template <typename T, std::invocable<T, T> ScalarOp>
inline void transform_scalar(T * dest,
                             T const * dest_end,
                             T const * src,
                             T scalar,
                             ScalarOp scalar_op)
{
    while (dest < dest_end)
    {
        *dest = scalar_op(*src, scalar);
        ++dest;
        ++src;
    }
}

template <typename T>
inline void add(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    transform_binary<T>(dest, dest_end, src1, src2, std::plus<T>{});
}

template <typename T>
inline void sub(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    transform_binary<T>(dest, dest_end, src1, src2, std::minus<T>{});
}

template <typename T>
inline void mul(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    transform_binary<T>(dest, dest_end, src1, src2, std::multiplies<T>{});
}

template <typename T>
inline void div(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    transform_binary<T>(dest, dest_end, src1, src2, std::divides<T>{});
}

template <typename T>
inline void add_scalar(T * dest, T const * dest_end, T const * src, T rhs)
{
    transform_scalar<T>(
        dest, dest_end, src, rhs, std::plus<T>{});
}

template <typename T>
inline void add_lhs_scalar(T * dest, T const * dest_end, T lhs, T const * src)
{
    transform_scalar<T>(
        dest, dest_end, src, lhs, std::plus<T>{});
}

template <typename T>
inline void sub_scalar(T * dest, T const * dest_end, T const * src, T rhs)
{
    transform_scalar<T>(
        dest, dest_end, src, rhs, std::minus<T>{});
}

template <typename T>
inline void sub_lhs_scalar(T * dest, T const * dest_end, T lhs, T const * src)
{
    transform_scalar<T>(
        dest,
        dest_end,
        src,
        lhs,
        [](T rhs, T scalar)
        {
            return scalar - rhs;
        });
}

template <typename T>
inline void mul_scalar(T * dest, T const * dest_end, T const * src, T rhs)
{
    transform_scalar<T>(
        dest, dest_end, src, rhs, std::multiplies<T>{});
}

template <typename T>
inline void mul_lhs_scalar(T * dest, T const * dest_end, T lhs, T const * src)
{
    transform_scalar<T>(
        dest, dest_end, src, lhs, std::multiplies<T>{});
}

template <typename T>
inline void div_scalar(T * dest, T const * dest_end, T const * src, T rhs)
{
    transform_scalar<T>(
        dest, dest_end, src, rhs, std::divides<T>{});
}

template <typename T>
inline void div_lhs_scalar(T * dest, T const * dest_end, T lhs, T const * src)
{
    transform_scalar<T>(
        dest,
        dest_end,
        src,
        lhs,
        [](T rhs, T scalar)
        {
            return scalar / rhs;
        });
}

template <typename T>
T max(T const * start, T const * end)
{
    T max_val = *start;
    for (T const * ptr = start + 1; ptr < end; ++ptr)
    {
        if (*ptr > max_val)
        {
            max_val = *ptr;
        }
    }
    return max_val;
}

} /* end namespace generic */

} /* end namespace simd */

} /* end namespace solvcon */
