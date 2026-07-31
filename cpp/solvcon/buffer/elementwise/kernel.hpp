#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/simd/simd.hpp>

#include <concepts>
#include <cstddef>

#if defined(__ELF__) && defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
#define SOLVCON_ELEMENTWISE_TARGETS \
    [[gnu::target_clones("default", "avx2")]]
#else
#define SOLVCON_ELEMENTWISE_TARGETS
#endif

namespace solvcon
{

namespace detail
{

namespace elementwise
{

template <typename Derived, typename T>
class BinaryKernelBase
{
public:
    static void scalar(T * output, size_t count, T rhs);
    static void inplace(T * output, size_t count, T const * rhs);
    static void contiguous_scalar(T * output,
                                  size_t count,
                                  T const * lhs,
                                  T rhs);
    static void contiguous_lhs_scalar(T * output,
                                      size_t count,
                                      T lhs,
                                      T const * rhs);
}; /* end class BinaryKernelBase */

template <typename Derived, typename T>
SOLVCON_ELEMENTWISE_TARGETS void BinaryKernelBase<Derived, T>::scalar(
    T * output, size_t count, T rhs)
{
    Derived const kernel;
    T const * const end = output + count;
    while (output < end)
    {
        *output = kernel(*output, rhs);
        ++output;
    }
}

template <typename Derived, typename T>
SOLVCON_ELEMENTWISE_TARGETS void BinaryKernelBase<Derived, T>::inplace(
    T * output, size_t count, T const * rhs)
{
    Derived const kernel;
    T const * const end = output + count;
    while (output < end)
    {
        *output = kernel(*output, *rhs);
        ++output;
        ++rhs;
    }
}

template <typename Derived, typename T>
SOLVCON_ELEMENTWISE_TARGETS void BinaryKernelBase<Derived, T>::contiguous_scalar(
    T * output, size_t count, T const * lhs, T rhs)
{
    Derived const kernel;
    for (size_t index = 0; index < count; ++index)
    {
        output[index] = kernel(lhs[index], rhs);
    }
}

template <typename Derived, typename T>
SOLVCON_ELEMENTWISE_TARGETS void BinaryKernelBase<Derived, T>::contiguous_lhs_scalar(
    T * output, size_t count, T lhs, T const * rhs)
{
    Derived const kernel;
    for (size_t index = 0; index < count; ++index)
    {
        output[index] = kernel(lhs, rhs[index]);
    }
}

template <typename T>
class AddKernel : public BinaryKernelBase<AddKernel<T>, T>
{
public:
    T operator()(T lhs, T rhs) const { return lhs + rhs; }

    static void contiguous(T * output,
                           size_t count,
                           T const * lhs,
                           T const * rhs)
    {
        simd::add<T>(output, output + count, lhs, rhs);
    }
}; /* end class AddKernel */

template <typename T>
class SubtractKernel : public BinaryKernelBase<SubtractKernel<T>, T>
{
public:
    T operator()(T lhs, T rhs) const { return lhs - rhs; }

    static void contiguous(T * output,
                           size_t count,
                           T const * lhs,
                           T const * rhs)
    {
        simd::sub<T>(output, output + count, lhs, rhs);
    }
}; /* end class SubtractKernel */

template <typename T>
class MultiplyKernel : public BinaryKernelBase<MultiplyKernel<T>, T>
{
public:
    T operator()(T lhs, T rhs) const { return lhs * rhs; }

    static void contiguous(T * output,
                           size_t count,
                           T const * lhs,
                           T const * rhs)
    {
        simd::mul<T>(output, output + count, lhs, rhs);
    }
}; /* end class MultiplyKernel */

template <typename T>
class DivideKernel : public BinaryKernelBase<DivideKernel<T>, T>
{
public:
    T operator()(T lhs, T rhs) const { return lhs / rhs; }

    static void contiguous(T * output,
                           size_t count,
                           T const * lhs,
                           T const * rhs)
    {
        simd::div<T>(output, output + count, lhs, rhs);
    }
}; /* end class DivideKernel */

template <typename Kernel, typename T>
concept ArithmeticKernel = requires(Kernel kernel,
                                    T lhs,
                                    T rhs,
                                    T * output,
                                    T const * lhs_data,
                                    T const * rhs_data) {
    {
        kernel(lhs, rhs)
    } -> std::convertible_to<T>;
    Kernel::scalar(output, size_t{}, rhs);
    Kernel::inplace(output, size_t{}, rhs_data);
    Kernel::contiguous(output, size_t{}, lhs_data, rhs_data);
    Kernel::contiguous_scalar(output, size_t{}, lhs_data, rhs);
    Kernel::contiguous_lhs_scalar(
        output, size_t{}, lhs, rhs_data);
};

} /* end namespace elementwise */

} /* end namespace detail */

} /* end namespace solvcon */

#undef SOLVCON_ELEMENTWISE_TARGETS

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
